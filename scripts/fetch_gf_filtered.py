"""
fetch_gf_filtered.py — Filter guidance data and download company intel for matches.

Workflow:
  1. Downloads guidance_tracker.parquet from Drive (cached 1h locally)
  2. Shows what metrics + horizons are available in your data
  3. You pick metric (e.g. PAT), horizon (e.g. FY27), min growth % threshold
  4. Shows matching companies + their guidance values
  5. You confirm → downloads company_page.md for each match
  6. Opens all in Obsidian

Example — "PAT growth guidance > 40% for FY27":
  Metric:   PAT
  Horizon:  FY27
  Min %:    40

Usage (interactive):
    python scripts/fetch_gf_filtered.py

Usage (non-interactive / scriptable):
    python scripts/fetch_gf_filtered.py --metric PAT --horizon FY27 --min-pct 40
    python scripts/fetch_gf_filtered.py --metric Revenue --horizon FY27 --min-pct 20 --no-open
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.parse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── CONFIG ────────────────────────────────────────────────────────────────────
CACHE_DIR   = Path(r"D:\EMA_Screener\Reports\signals-india\.cache")
OUTPUT_DIR  = Path(r"D:\EMA_Screener\Reports\signals-india\company_intel")
CACHE_HOURS = 1    # re-download parquet if older than this
# ─────────────────────────────────────────────────────────────────────────────

SCOPES = ["https://www.googleapis.com/auth/drive"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive auth ----------

def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    tk_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    cs_json = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_JSON")
    if tk_json and cs_json:
        import json
        creds = Credentials.from_authorized_user_info(json.loads(tk_json), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    tk_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if tk_path.exists():
        creds = Credentials.from_authorized_user_file(str(tk_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow
            cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        tk_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_subfolder(drive, parent_id: str, name: str) -> str | None:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None


def download_bytes(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    dl = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return fh.getvalue()


def find_file(drive, folder_id: str, filename: str) -> str | None:
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None


# ---------- Parquet cache ----------

def _cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_h = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
    return age_h < CACHE_HOURS


def load_guidance_tracker(drive=None) -> pd.DataFrame:
    """Load guidance_tracker.parquet — from local cache if fresh, else Drive."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "guidance_tracker.parquet"

    if _cache_fresh(cache_path):
        log(f"Using cached guidance_tracker ({cache_path.stat().st_size // 1024:.0f} KB)")
        return pd.read_parquet(cache_path)

    if drive is None:
        drive = get_drive()

    log("Downloading guidance_tracker.parquet from Drive…")
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id   = find_subfolder(drive, folder_id, "company_repo")
    index_id  = find_subfolder(drive, repo_id,   "_index") if repo_id else None
    if not index_id:
        log("ERROR: company_repo/_index not found.")
        sys.exit(1)

    fid = find_file(drive, index_id, "guidance_tracker.parquet")
    if not fid:
        log("ERROR: guidance_tracker.parquet not found. Run Phase 2 first.")
        sys.exit(1)

    data = download_bytes(drive, fid)
    cache_path.write_bytes(data)
    df = pd.read_parquet(io.BytesIO(data))
    log(f"Downloaded {len(df)} guidance rows ({len(data)//1024:.0f} KB)")
    return df


# ---------- Open in Obsidian ----------

def open_in_obsidian(path: Path) -> None:
    uri = "obsidian://open?path=" + urllib.parse.quote(
        str(path).replace("\\", "/"), safe=":/"
    )
    subprocess.run(["cmd", "/c", "start", "", uri], shell=False)


# ---------- Download company page ----------

from _md_utils import fix_markdown_for_obsidian


def download_company_page(drive, isin: str, symbol: str) -> Path | None:
    """Download company_page.md for a given ISIN, fix, save to OUTPUT_DIR."""
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id   = find_subfolder(drive, folder_id, "company_repo")
    if not repo_id:
        return None

    # Try ISIN folder first, then symbol
    for key in [isin, symbol]:
        comp_id = find_subfolder(drive, repo_id, key)
        if comp_id:
            fid = find_file(drive, comp_id, "company_page.md")
            if fid:
                raw  = download_bytes(drive, fid)
                text = raw.decode("utf-8", errors="replace")
                fixed = fix_markdown_for_obsidian(text)
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                out = OUTPUT_DIR / f"{symbol or isin}_company_page.md"
                out.write_text(fixed, encoding="utf-8")
                return out
    return None


# ---------- Interactive prompt helpers ----------

def _pick(prompt: str, options: list[str], allow_all: bool = False) -> str:
    print(f"\n  {prompt}")
    for i, opt in enumerate(options[:20], 1):
        print(f"    [{i:2d}]  {opt}")
    if len(options) > 20:
        print(f"    ... and {len(options) - 20} more (type value directly)")
    if allow_all:
        print(f"    [ 0]  ALL")
    while True:
        raw = input("  Your choice (number or value): ").strip()
        if allow_all and raw == "0":
            return "ALL"
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        if raw.upper() in [o.upper() for o in options]:
            return raw.upper()
        if raw:   # typed directly
            return raw
        print("  Please enter a number or type the value.")


def _ask_float(prompt: str, default: float | None = None) -> float | None:
    hint = f" (Enter to skip filter)" if default is None else f" (default {default})"
    while True:
        raw = input(f"  {prompt}{hint}: ").strip()
        if raw == "" and default is None:
            return None
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Enter a number (e.g. 40) or press Enter to skip.")


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metric",   help="Metric to filter on (e.g. PAT, Revenue)")
    parser.add_argument("--horizon",  help="FY horizon (e.g. FY27, FY28)")
    parser.add_argument("--min-pct",  type=float, help="Minimum cagr_pct value")
    parser.add_argument("--max-pct",  type=float, help="Maximum cagr_pct value")
    parser.add_argument("--guidance-type", choices=["explicit", "derived", "all"],
                        default="all", help="Filter by guidance type")
    parser.add_argument("--no-open",  action="store_true", help="Don't open in Obsidian")
    parser.add_argument("--refresh",  action="store_true",
                        help="Force re-download parquet even if cached")
    args = parser.parse_args()

    if args.refresh and (CACHE_DIR / "guidance_tracker.parquet").exists():
        (CACHE_DIR / "guidance_tracker.parquet").unlink()

    print("\n" + "="*60)
    print("  GF Guidance Filter — find companies by guidance criteria")
    print("="*60)

    drive = get_drive()
    df = load_guidance_tracker(drive)

    if df.empty:
        log("No guidance data found. Run Phase 2 to process concalls first.")
        sys.exit(0)

    # ── Convert cagr_pct to numeric ──────────────────────────────────────────
    df["cagr_pct"] = pd.to_numeric(df["cagr_pct"], errors="coerce")
    # Also try parsing numeric values from the 'value' column as fallback
    df["_val_num"] = pd.to_numeric(
        df["value"].astype(str).str.replace("%","").str.strip(), errors="coerce"
    )
    df["_score"] = df["cagr_pct"].combine_first(df["_val_num"])

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  Data loaded: {len(df)} guidance rows")
    print(f"  Companies  : {df['symbol'].nunique()}")
    print(f"  Quarters   : {sorted(df['quarter'].dropna().unique().tolist())}")

    metrics  = sorted(df["metric"].dropna().unique().tolist())
    horizons = sorted(df["horizon_fy"].dropna().unique().tolist())

    # ── Pick metric ──────────────────────────────────────────────────────────
    metric = args.metric
    if not metric:
        metric = _pick("Which metric do you want to filter on?", metrics)
    print(f"\n  Metric selected: {metric}")

    # ── Pick horizon ─────────────────────────────────────────────────────────
    # Show only horizons that have data for the chosen metric
    avail_horizons = sorted(
        df[df["metric"].str.upper() == metric.upper()]["horizon_fy"].dropna().unique().tolist()
    )
    horizon = args.horizon
    if not horizon:
        horizon = _pick(
            f"Which FY horizon?  (FY27 = next year, FY28 = 2yr forward…)",
            avail_horizons, allow_all=True
        )
    print(f"  Horizon selected: {horizon}")

    # ── Pick threshold ───────────────────────────────────────────────────────
    min_pct = args.min_pct
    max_pct = args.max_pct
    if min_pct is None and max_pct is None:
        print(f"\n  Growth % threshold  (based on cagr_pct column)")
        min_pct = _ask_float("Minimum growth % (e.g. 40 for >40%)")
        max_pct = _ask_float("Maximum growth % (leave blank for no upper limit)")

    # ── Guidance type ─────────────────────────────────────────────────────────
    gtype = args.guidance_type
    if gtype == "all":
        print("\n  Including both Explicit and Derived guidance.")
        print("  Tip: use --guidance-type explicit to see only management-stated values.")

    # ── Apply filters ─────────────────────────────────────────────────────────
    flt = df.copy()
    flt = flt[flt["metric"].str.upper() == metric.upper()]
    if horizon != "ALL":
        flt = flt[flt["horizon_fy"].str.upper() == horizon.upper()]
    if gtype != "all":
        flt = flt[flt["guidance_type"].str.lower() == gtype.lower()]
    if min_pct is not None:
        flt = flt[flt["_score"] >= min_pct]
    if max_pct is not None:
        flt = flt[flt["_score"] <= max_pct]

    # ── Show results ──────────────────────────────────────────────────────────
    if flt.empty:
        print(f"\n  No companies match your criteria.")
        print(f"  Try a lower threshold or a different metric/horizon.")
        sys.exit(0)

    # Deduplicate to one row per company (keep highest score)
    best = (flt.sort_values("_score", ascending=False)
              .groupby("symbol").first().reset_index()
              .sort_values("_score", ascending=False))

    threshold_str = ""
    if min_pct is not None:
        threshold_str += f" ≥{min_pct:.0f}%"
    if max_pct is not None:
        threshold_str += f" ≤{max_pct:.0f}%"

    print(f"\n{'='*60}")
    print(f"  {len(best)} companies match:  {metric} | {horizon}{threshold_str}")
    print(f"{'='*60}")
    print(f"  {'Symbol':<15} {'Company':<35} {'Guidance %':>10}  Type")
    print(f"  {'-'*15} {'-'*35} {'-'*10}  {'-'*8}")
    for _, row in best.iterrows():
        score_str = f"{row['_score']:.1f}%" if pd.notna(row["_score"]) else str(row["value"])[:10]
        gtype_str = str(row.get("guidance_type", ""))[:8]
        print(f"  {str(row['symbol']):<15} {str(row['company_name']):<35} {score_str:>10}  {gtype_str}")
    print()

    # ── Confirm download ──────────────────────────────────────────────────────
    confirm = input(f"  Download company_page.md for all {len(best)} companies? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("  Aborted.")
        sys.exit(0)

    # ── Download ──────────────────────────────────────────────────────────────
    downloaded: list[Path] = []
    failed: list[str] = []

    for _, row in best.iterrows():
        sym  = str(row.get("symbol", ""))
        isin = str(row.get("isin",   ""))
        log(f"Downloading {sym} ({isin})…")
        path = download_company_page(drive, isin, sym)
        if path:
            downloaded.append(path)
            log(f"  ✓ Saved: {path.name}")
        else:
            failed.append(sym or isin)
            log(f"  ✗ Not found on Drive: {sym}")

    print(f"\n{'='*60}")
    print(f"  Downloaded: {len(downloaded)}   Not found: {len(failed)}")
    if failed:
        print(f"  Missing: {', '.join(failed)}")
    print(f"  Saved to: {OUTPUT_DIR}")
    print('='*60)

    if not args.no_open and downloaded:
        log(f"Opening {len(downloaded)} file(s) in Obsidian…")
        for i, path in enumerate(downloaded):
            open_in_obsidian(path)
            if i < len(downloaded) - 1:
                time.sleep(0.6)


if __name__ == "__main__":
    main()
