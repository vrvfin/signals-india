"""
Stage 2a — Build the master equity universe (NSE + BSE-only).

Output (in your Google Drive `signals-india` folder):
    universe/master_list.csv             — current universe (overwritten each run)
    universe/history/master_list_YYYY-MM-DD.csv — daily snapshot

UNIFIED (2026-06-13): master_list = NSE mainboard (EQUITY_L) + BSE-EXCLUSIVE
names appended from company_repo/_index/company_universe.csv (dual-listed names
stay NSE-primary — they already have an NSE symbol so they are not re-added).
Every row gains:
    exchange    NSE | BSE
    yf_ticker   <symbol>.NS  for NSE  /  <bse_code>.BO  for BSE-only
`symbol` is the storage key (= NSE symbol, or bse_symbol/BSE<code> for BSE-only)
so it matches the OHLCV parquet filename. NSE rows are UNCHANGED (additive).
OHLCV for NSE comes from ingest_ohlcv.py; for BSE-only from
fetch_bse_only_ohlcv.py --promote. compute_features iterates this unified list.

Run from project root, inside the `signals-india` conda env:
    python scripts/build_universe.py
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
KEEP_SERIES = {"EQ", "BE", "BZ"}  # mainboard equity series; we tag BZ/BE for awareness


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_drive_service():
    """Authenticate and return Drive API service, reusing cached OAuth token."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    token_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def fetch_nse_equity_list() -> pd.DataFrame:
    """Download NSE EQUITY_L.csv and normalize column names."""
    log(f"Fetching NSE equity list from {NSE_EQUITY_LIST_URL}")
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0 Safari/537.36"),
        "Accept": "text/csv,*/*",
    }
    r = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # NSE columns can have stray spaces — normalize.
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {
        "name_of_company": "name",
        "isin_number": "isin",
        "date_of_listing": "listing_date",
    }
    df = df.rename(columns=rename)
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["series"] = df["series"].astype(str).str.strip()
    df["exchange"] = "NSE"
    before = len(df)
    df = df[df["series"].isin(KEEP_SERIES)].copy()
    log(f"Series filter: {before} → {len(df)} symbols "
        f"(kept {sorted(KEEP_SERIES)})")
    df["yf_ticker"] = df["symbol"].astype(str).str.strip() + ".NS"
    cols = ["symbol", "exchange", "name", "isin", "series", "listing_date",
            "yf_ticker"]
    return df[cols].reset_index(drop=True)


def _bse_storage_key(bse_symbol: str, bse_code: str) -> str:
    """MUST match fetch_bse_only_ohlcv._storage_key so the parquet filename
    lines up with the master_list symbol."""
    s = str(bse_symbol).strip()
    if s and s.lower() != "nan":
        return s.upper()
    return f"BSE{str(bse_code).strip()}"


def fetch_bse_only_rows(drive, folder_id: str, nse_isins: set[str]) -> pd.DataFrame:
    """BSE-EXCLUSIVE rows (have bse_code, no NSE symbol, ISIN not already on
    the NSE list) from company_universe.csv, shaped for master_list."""
    cols = ["symbol", "exchange", "name", "isin", "series", "listing_date",
            "yf_ticker"]
    try:
        repo = get_or_create_subfolder(drive, folder_id, "company_repo")
        idx = get_or_create_subfolder(drive, repo, "_index")
        q = (f"name='company_universe.csv' and '{idx}' in parents "
             f"and trashed=false")
        files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not files:
            log("  company_universe.csv not found — BSE-only names skipped.")
            return pd.DataFrame(columns=cols)
        raw = drive.files().get_media(fileId=files[0]["id"]).execute()
        uni = pd.read_csv(io.BytesIO(raw)).fillna("")
    except Exception as e:
        log(f"  BSE-only fetch failed ({str(e)[:80]}) — skipped.")
        return pd.DataFrame(columns=cols)
    nse_sym = uni["nse_symbol"].astype(str).str.strip()
    code = uni["bse_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    isin = uni["isin"].astype(str).str.strip()
    mask = (nse_sym.isin(["", "nan"]) & ~code.isin(["", "nan"])
            & ~isin.isin(nse_isins))
    bse = uni[mask].copy()
    if bse.empty:
        return pd.DataFrame(columns=cols)
    bse["bse_code"] = code[mask].values
    out = pd.DataFrame({
        "symbol": [_bse_storage_key(s, c) for s, c in
                   zip(bse.get("bse_symbol", ""), bse["bse_code"])],
        "exchange": "BSE",
        "name": bse["name"].astype(str).str.strip(),
        "isin": bse["isin"].astype(str).str.strip(),
        "series": "",
        "listing_date": "",
        "yf_ticker": bse["bse_code"].astype(str) + ".BO",
    })
    out = out[out["symbol"].astype(str).str.len() > 0].drop_duplicates("symbol")
    log(f"  BSE-only appended: {len(out)}")
    return out[cols].reset_index(drop=True)


# Non-equity instruments that BSE's scrip list carries alongside companies.
# They have no Screener company page (both URL variants 404), so every sweep
# spent two requests each on them and logged them as failures — 63 of the 66
# weekly "failures" were exactly this, which masked the 3 real ones. They also
# cost OHLCV and feature work downstream, since master_list feeds those too.
_NON_EQUITY_NAME = re.compile(
    r"SEGREGATED\s+PORTFOLIO|MUTUAL\s+FUND|\bIDCW\b|"
    r"(?:DIRECT|REGULAR)\s+PLAN|\bLONG[-\s]?SHORT\s+FUND\b|"
    r"\bFUND\b.*\b(?:GROWTH|PAYOUT|REINVEST)", re.IGNORECASE)

# ETFs are managed by a fund house, so their names trip the markers above — but
# unlike the unlisted fund units they DO have Screener pages, price history, and
# are tradable, so they stay in the universe that OHLCV/features/gallery read.
# They have no quarterly results, which is a coverage concern, not a universe one.
_ETF_NAME = re.compile(r"\bETF\b|\bBeES\b|\bINDEX\s+FUND\b", re.IGNORECASE)


def filter_equity_only(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split master_list rows into (keep, drop).

    Conservative by design: a row is dropped only when its NAME matches a fund
    marker, or its SYMBOL ends in '-RE' (a rights entitlement — a temporary
    tradable form, not the company; the underlying company keeps its own row).
    Anything that looks like an ETF is exempted. Nothing is dropped on
    series/exchange alone, so a genuine company with a sparse row is never lost.
    """
    name = df["name"].astype(str)
    sym = df["symbol"].astype(str).str.upper()
    is_fund = name.str.contains(_NON_EQUITY_NAME) & ~name.str.contains(_ETF_NAME)
    is_rights = sym.str.endswith("-RE")
    bad = is_fund | is_rights
    return (df[~bad].reset_index(drop=True), df[bad].reset_index(drop=True))


def fetch_nse_emerge_rows(drive, folder_id: str, nse_isins: set[str]) -> pd.DataFrame:
    """NSE Emerge (SME) rows from company_universe.csv — names that HAVE an NSE
    symbol but are NOT on the mainboard EQUITY_L (so their ISIN is not in
    `nse_isins`). Mainboard rows already cover the EQ/BE/BZ list; these are the
    SME-platform names that would otherwise be dropped (they are not BSE-only —
    `fetch_bse_only_rows` requires an EMPTY nse_symbol — and they are not on the
    mainboard list). Priced via <nse_symbol>.NS by ingest_ohlcv.py."""
    cols = ["symbol", "exchange", "name", "isin", "series", "listing_date",
            "yf_ticker"]
    try:
        repo = get_or_create_subfolder(drive, folder_id, "company_repo")
        idx = get_or_create_subfolder(drive, repo, "_index")
        q = (f"name='company_universe.csv' and '{idx}' in parents "
             f"and trashed=false")
        files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not files:
            log("  company_universe.csv not found — NSE Emerge names skipped.")
            return pd.DataFrame(columns=cols)
        raw = drive.files().get_media(fileId=files[0]["id"]).execute()
        uni = pd.read_csv(io.BytesIO(raw)).fillna("")
    except Exception as e:
        log(f"  NSE Emerge fetch failed ({str(e)[:80]}) — skipped.")
        return pd.DataFrame(columns=cols)
    nse_sym = uni["nse_symbol"].astype(str).str.strip()
    isin = uni["isin"].astype(str).str.strip()
    mask = (~nse_sym.isin(["", "nan"]) & ~isin.isin(nse_isins))
    eme = uni[mask].copy()
    if eme.empty:
        return pd.DataFrame(columns=cols)
    sym = nse_sym[mask].str.upper()
    out = pd.DataFrame({
        "symbol": sym.values,
        "exchange": "NSE",
        "name": eme["name"].astype(str).str.strip(),
        "isin": eme["isin"].astype(str).str.strip(),
        "series": "SME",
        "listing_date": "",
        "yf_ticker": sym.values + ".NS",
    })
    out = out[out["symbol"].astype(str).str.len() > 0].drop_duplicates("symbol")
    log(f"  NSE Emerge appended: {len(out)}")
    return out[cols].reset_index(drop=True)


def get_or_create_subfolder(drive, parent_id: str, name: str) -> str:
    """Find a child folder by name, or create it."""
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id,name)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def upload_csv(drive, df: pd.DataFrame, folder_id: str, filename: str) -> str:
    """Upload CSV to Drive folder. Overwrites if filename already exists."""
    csv_bytes = df.to_csv(index=False).encode()
    media = MediaInMemoryUpload(csv_bytes, mimetype="text/csv")
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    existing = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if existing:
        drive.files().update(fileId=existing[0]["id"], media_body=media).execute()
        return existing[0]["id"]
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the list but skip Drive uploads; write "
                             "master_list_dryrun.csv locally and print counts.")
    args = parser.parse_args()

    print("Stage 2a — Build universe (NSE mainboard + NSE Emerge + BSE-only)")
    print("-" * 50)
    df = fetch_nse_equity_list()
    log(f"NSE symbols ready: {len(df)}")

    drive = get_drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    nse_isins = set(df["isin"].astype(str).str.strip())
    emerge = fetch_nse_emerge_rows(drive, folder_id, nse_isins)
    bse = fetch_bse_only_rows(drive, folder_id, nse_isins)
    extra = [x for x in (emerge, bse) if not x.empty]
    if extra:
        df = pd.concat([df, *extra], ignore_index=True)
    df = df.drop_duplicates("symbol").reset_index(drop=True)
    df, dropped = filter_equity_only(df)
    if not dropped.empty:
        log(f"Non-equity dropped: {len(dropped)} "
            f"(mutual-fund units, segregated portfolios, rights entitlements)")
        for r in dropped.head(8).itertuples():
            log(f"    {r.symbol:12s} {str(r.name)[:56]}")
        if len(dropped) > 8:
            log(f"    ... and {len(dropped) - 8} more")
    log(f"Unified universe: {len(df)} "
        f"(NSE {int((df['exchange'] == 'NSE').sum())} "
        f"[incl. Emerge {len(emerge)}] + "
        f"BSE-only {int((df['exchange'] == 'BSE').sum())})")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
    print()

    if args.dry_run:
        out_path = Path(__file__).resolve().parent / "master_list_dryrun.csv"
        df.to_csv(out_path, index=False)
        log(f"DRY-RUN: wrote {out_path} ({len(df)} rows); no Drive upload.")
        probe = ["AIMTRON", "ANONDITA", "OBSC", "V-MARC", "VMARC", "Z-TECH", "ZTECH"]
        hits = df[df["name"].astype(str).str.upper().str.contains("|".join(probe))
                  | df["symbol"].astype(str).str.upper().str.contains("|".join(probe))]
        print("\nProbe (target SME names now present):")
        print(hits[["symbol", "exchange", "name", "yf_ticker"]].to_string(index=False)
              if not hits.empty else "  (none matched — check company_universe.csv)")
        return

    universe_id = get_or_create_subfolder(drive, folder_id, "universe")
    history_id = get_or_create_subfolder(drive, universe_id, "history")

    today = datetime.now().strftime("%Y-%m-%d")
    upload_csv(drive, df, universe_id, "master_list.csv")
    log("Uploaded universe/master_list.csv")
    upload_csv(drive, df, history_id, f"master_list_{today}.csv")
    log(f"Uploaded universe/history/master_list_{today}.csv")

    print("-" * 50)
    print(f"Done. {len(df)} symbols in master_list.csv.")


if __name__ == "__main__":
    main()