"""
Phase 2 / Stage A — Company universe (entire Indian listed universe), by ISIN.

Builds  company_repo/_index/company_universe.csv  — the identity map for the
company-repository pipeline. ISIN is the stable key: it survives ticker renames
so a company's history is never orphaned.

Sources (each independent and best-effort — a failure of one never aborts the
run; whatever resolved is still written):
  - NSE mainboard      archives EQUITY_L.csv
  - NSE Emerge (SME)   archives SME equity list
  - BSE (all equity)   BSE "ListOfScripData" API — covers BSE mainboard + SME

NOTE: the document pipeline does NOT depend on this file being complete. It
ingests whatever appears on Screener's feeds; a company missing here simply
keys its folder by its Screener slug instead of ISIN. This file only adds the
ISIN tag where it can.

Output columns: isin, name, nse_symbol, bse_code, board, exchange

Usage:
    python scripts/build_company_universe.py
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

# NSE equity CSVs — (url, board label). Mainboard is reliable; SME is best-effort.
NSE_CSV_SOURCES = [
    ("https://archives.nseindia.com/content/equities/EQUITY_L.csv",
     "NSE Mainboard"),
    ("https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv",
     "NSE Emerge (SME)"),
]
# BSE ListOfScrips API — empty Group returns all equity groups incl. SME.
BSE_SCRIP_API = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
                 "?Group=&Scripcode=&industry=&segment=Equity&status=Active")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

ISIN_RE = r"^IN[A-Z0-9]{10}$"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    # CI path: GDRIVE_OAUTH_TOKEN_JSON holds the token JSON inline
    # Local path: GDRIVE_OAUTH_TOKEN_PATH points to a token file on disk
    token_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON", "").strip()
    token_path_str = os.environ.get("GDRIVE_OAUTH_TOKEN_PATH", "")
    creds = None
    if token_json:
        creds = Credentials.from_authorized_user_info(
            __import__("json").loads(token_json), SCOPES)
    elif token_path_str and Path(token_path_str).exists():
        creds = Credentials.from_authorized_user_file(token_path_str, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        # Interactive OAuth — only works locally, not in CI
        cs_path = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_PATH", "")
        if not cs_path:
            raise RuntimeError(
                "Set GDRIVE_OAUTH_TOKEN_JSON (CI) or "
                "GDRIVE_OAUTH_TOKEN_PATH + GDRIVE_OAUTH_CLIENT_SECRET_PATH (local)")
        flow = InstalledAppFlow.from_client_secrets_file(cs_path, SCOPES)
        creds = flow.run_local_server(port=0)
        if token_path_str:
            Path(token_path_str).parent.mkdir(parents=True, exist_ok=True)
            Path(token_path_str).write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_or_create_subfolder(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def find_file(drive, folder_id, name):
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def upload_csv(drive, folder_id, filename, df, existing_id=None):
    media = MediaIoBaseUpload(io.BytesIO(df.to_csv(index=False).encode()),
                              mimetype="text/csv", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ---------- Source fetchers (all best-effort) ----------

def fetch_nse_csv(url: str, board: str) -> pd.DataFrame:
    """An NSE equity CSV (mainboard or Emerge SME). Empty DF on any failure."""
    cols = ["isin", "nse_symbol", "name", "board"]
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        if r.status_code != 200:
            log(f"  {board}: HTTP {r.status_code} — skipping.")
            return pd.DataFrame(columns=cols)
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        log(f"  {board}: fetch failed ({str(e)[:110]}) — skipping.")
        return pd.DataFrame(columns=cols)

    df.columns = [c.strip() for c in df.columns]

    def col(*names):
        for n in names:
            for c in df.columns:
                # Remove both spaces and underscores for comparison
                c_clean = c.upper().replace(" ", "").replace("_", "")
                n_clean = n.upper().replace(" ", "").replace("_", "")
                if c_clean == n_clean:
                    return c
        return None
    
    c_isin = col("ISIN NUMBER", "ISIN")
    c_sym = col("SYMBOL")
    c_name = col("NAME OF COMPANY", "NAME")
    if not (c_isin and c_sym):
        log(f"  {board}: unexpected columns {list(df.columns)} — skipping.")
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "isin": df[c_isin].astype(str).str.strip(),
        "nse_symbol": df[c_sym].astype(str).str.strip(),
        "name": df[c_name].astype(str).str.strip() if c_name else "",
        "board": board,
    })
    out = out[out["isin"].str.match(ISIN_RE, na=False)].reset_index(drop=True)
    log(f"  {board}: {len(out)} equities")
    return out


def fetch_bse() -> pd.DataFrame:
    """BSE equity scrip master (mainboard + SME). Empty DF on failure."""
    cols = ["isin", "bse_code", "name"]
    headers = {"User-Agent": UA, "Accept": "application/json",
               "Referer": "https://www.bseindia.com/"}
    try:
        r = requests.get(BSE_SCRIP_API, headers=headers, timeout=30)
        if r.status_code != 200:
            log(f"  BSE: HTTP {r.status_code} — skipping.")
            return pd.DataFrame(columns=cols)
        data = r.json()
    except Exception as e:
        log(f"  BSE: fetch failed ({str(e)[:110]}) — skipping.")
        return pd.DataFrame(columns=cols)

    if isinstance(data, dict):
        data = data.get("Table") or data.get("data") or []
    if not data:
        log("  BSE: empty response — skipping.")
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(data)

    def col(*subs):
        for c in df.columns:
            if any(s in c.lower() for s in subs):
                return c
        return None
    c_isin   = col("isin")
    c_code   = col("scrip_cd", "scripcd", "scrip_code", "sc_code")
    c_sym    = col("scrip_id")          # text ticker used by TradingView (e.g. "ROBU")
    c_name   = col("sc_name", "scrip_name", "securityname", "company")
    if not (c_isin and c_code):
        log(f"  BSE: unexpected columns {list(df.columns)} — skipping.")
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "isin":       df[c_isin].astype(str).str.strip(),
        "bse_code":   df[c_code].astype(str).str.strip(),
        "bse_symbol": df[c_sym].astype(str).str.strip() if c_sym else "",
        "name":       df[c_name].astype(str).str.strip() if c_name else "",
    })
    out = out[out["isin"].str.match(ISIN_RE, na=False)].reset_index(drop=True)
    log(f"  BSE: {len(out)} equities")
    return out


# ---------- Main ----------

def main() -> None:
    print("Phase 2 / Stage A — Company universe (entire Indian listed market)")
    print("-" * 60)

    log("Fetching NSE equity lists...")
    nse_frames = [fetch_nse_csv(url, board) for url, board in NSE_CSV_SOURCES]
    nse = pd.concat(nse_frames, ignore_index=True) if nse_frames else pd.DataFrame()
    if not nse.empty:
        nse = nse.drop_duplicates("isin").reset_index(drop=True)

    log("Fetching BSE scrip list...")
    bse = fetch_bse()

    if nse.empty and bse.empty:
        print("\nERROR: no source returned data. Check network / endpoints.")
        return

    # Merge on ISIN — NSE is primary for symbol/name/board.
    if not nse.empty and not bse.empty:
        merged = nse.merge(bse[["isin", "bse_code", "bse_symbol"]].drop_duplicates("isin"),
                           on="isin", how="outer")
        bse_names = bse.drop_duplicates("isin").set_index("isin")["name"].to_dict()
        merged["name"] = merged.apply(
            lambda r: r["name"] if isinstance(r.get("name"), str) and r["name"]
            else bse_names.get(r["isin"], ""), axis=1)
    elif not nse.empty:
        merged = nse.copy()
        merged["bse_code"] = ""
    else:
        merged = bse.copy()
        merged["nse_symbol"] = ""
        merged["board"] = ""

    for c in ["nse_symbol", "bse_code", "bse_symbol", "name", "board"]:
        if c not in merged.columns:
            merged[c] = ""
        merged[c] = merged[c].fillna("").astype(str)

    def board_of(r):
        if r["board"]:
            return r["board"]
        return "BSE" if (r["bse_code"] and r["bse_code"] != "nan") else "Unknown"

    def exch(r):
        has_nse = bool(r["nse_symbol"])
        has_bse = bool(r["bse_code"]) and r["bse_code"] != "nan"
        if has_nse and has_bse:
            return "BOTH"
        return "NSE" if has_nse else "BSE"

    merged["board"] = merged.apply(board_of, axis=1)
    merged["exchange"] = merged.apply(exch, axis=1)
    merged = (merged[["isin", "name", "nse_symbol", "bse_code", "bse_symbol", "board", "exchange"]]
              .drop_duplicates("isin").sort_values("name").reset_index(drop=True))

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    # 1. company_repo/_index/company_universe.csv  (Phase 2 pipeline key)
    repo_id   = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id  = get_or_create_subfolder(drive, repo_id, "_index")
    upload_csv(drive, index_id, "company_universe.csv", merged,
               find_file(drive, index_id, "company_universe.csv"))
    log("Wrote company_repo/_index/company_universe.csv")

    # 2. universe/master_list.csv  (canonical read path for all scripts)
    uni_id = get_or_create_subfolder(drive, folder_id, "universe")
    upload_csv(drive, uni_id, "master_list.csv", merged,
               find_file(drive, uni_id, "master_list.csv"))
    log("Wrote universe/master_list.csv")

    print("-" * 60)
    print(f"Total companies : {len(merged)}")
    print("By board:")
    print(merged["board"].value_counts().to_string())
    if merged["board"].eq("NSE Emerge (SME)").sum() == 0:
        print("\nNOTE: no NSE Emerge (SME) rows — that endpoint may have failed.")
        print("      Not fatal: the document pipeline still covers SME companies")
        print("      (they key by Screener slug). Paste the log if you want the")
        print("      NSE SME URL corrected.")


if __name__ == "__main__":
    main()
