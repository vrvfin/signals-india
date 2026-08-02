"""
Stage 11a — Fundamentals ingestion from Screener.in.

For each symbol in the universe, fetches the screener page, extracts key
financial fields (P/E, market cap, RoCE, last 4 quarters of EPS/Sales/NetProfit,
growth rates, promoter holding), writes:

  fundamentals/per_symbol/<SYMBOL>.parquet   — full row per stock (long-lived cache)
  fundamentals/summary.parquet              — all stocks unioned, one row each

The summary file is what CANSLIM and PEAD strategies consume.

Cookie expiry is detected and announced — script halts with refresh instructions.

Usage:
    python scripts/ingest_fundamentals.py             # full universe (~40-50 min)
    python scripts/ingest_fundamentals.py --limit 20  # quick test
"""

from __future__ import annotations

import argparse
import io
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from screener_client import ScreenerClient, CookieExpiredError

SCOPES = ["https://www.googleapis.com/auth/drive"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---- Drive helpers (same pattern as our other scripts) ----

def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    tk_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if tk_path.exists():
        creds = Credentials.from_authorized_user_file(str(tk_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        tk_path.write_text(creds.to_json())
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


def list_files_in_folder(drive, folder_id):
    out, page_token = {}, None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name)", pageSize=1000, pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            out[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def find_file(drive, folder_id, name):
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_csv(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


def download_parquet(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


def upload_parquet(drive, folder_id, filename, df, existing_id=None):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="application/octet-stream", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between screener requests (default 1.0)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip symbols already in per_symbol/")
    parser.add_argument("--recent-results-days", type=int, default=None,
                        help="INCREMENTAL results-season mode: refresh ONLY "
                             "companies whose results first appeared in "
                             "_index/results.parquet within the last N days. "
                             "summary.parquet is upserted (not replaced). "
                             "~90 names/day in peak season, ~0 off-season.")
    parser.add_argument("--symbols", default="",
                        help="Comma-separated symbols to fetch INSTEAD of the whole "
                             "universe. Used for on-demand single-company pulls (the "
                             "narrative report calls this when a company has no "
                             "fundamentals/statements/<SYM>.parquet yet). Purely "
                             "additive — omitting it leaves the nightly/weekly sweep "
                             "behaviour unchanged.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch + log only; no Drive write.")
    args = parser.parse_args()

    print("Stage 11a — Fundamentals ingestion")
    print("-" * 50)

    client = ScreenerClient(rate_limit_sec=args.sleep)
    log("Screener client initialized")

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    universe_id = get_or_create_subfolder(drive, folder_id, "universe")
    master_id = find_file(drive, universe_id, "master_list.csv")
    universe = download_csv(drive, master_id)
    if args.limit:
        universe = universe.head(args.limit)

    # Screener resolution token per row: NSE names resolve by their NSE symbol;
    # BSE-only names resolve by bse_code (Screener serves /company/<bse_code>/).
    # The bse_code is the yf_ticker minus the ".BO" suffix. Storage key stays
    # `symbol` so summary/per_symbol filenames match master_list (NSE unchanged).
    def _token(row) -> str:
        if str(row.get("exchange", "NSE")) == "BSE":
            yt = str(row.get("yf_ticker", ""))
            if yt.endswith(".BO"):
                return yt[:-3]
        return str(row["symbol"])

    work = [(str(r["symbol"]), _token(r)) for _, r in universe.iterrows()]

    # On-demand single-company pull. Matches on `symbol` OR the Screener token (BSE
    # names resolve through yf_ticker), so a caller can pass whatever identifier it
    # holds. Applied before the incremental filter so an explicit request is never
    # narrowed away by results-season logic.
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        work = [(s, t) for (s, t) in work
                if s.upper() in want or str(t).upper() in want]
        log(f"--symbols: {len(work)} of {len(want)} requested symbol(s) matched "
            f"the universe")
        if not work:
            log("None of the requested symbols are in master_list.csv — nothing to do.")
            return

    # Results-season incremental mode: shrink work to companies whose results
    # just hit the scraped feed — so quarterly tables update the NEXT morning
    # instead of waiting for the Monday full run.
    if args.recent_results_days:
        repo_id = get_or_create_subfolder(drive, folder_id, "company_repo")
        idx_id = get_or_create_subfolder(drive, repo_id, "_index")
        res_fid = find_file(drive, idx_id, "results.parquet")
        if not res_fid:
            log("No results.parquet — nothing to refresh incrementally.")
            return
        res = download_parquet(drive, res_fid)
        fs = pd.to_datetime(res.get("first_seen_at"), errors="coerce")
        cutoff = datetime.now() - timedelta(days=args.recent_results_days)
        recent_isins = set(res.loc[fs >= cutoff, "isin"].astype(str))
        want = set(universe.loc[universe["isin"].astype(str).isin(recent_isins),
                                "symbol"].astype(str))
        work = [(s, t) for (s, t) in work if s in want]
        log(f"Incremental mode: {len(work)} companies with results in the "
            f"last {args.recent_results_days}d")
        if not work:
            log("Nothing to refresh — done.")
            return

    fund_id = get_or_create_subfolder(drive, folder_id, "fundamentals")
    per_sym_id = get_or_create_subfolder(drive, fund_id, "per_symbol")
    existing = list_files_in_folder(drive, per_sym_id)
    # Gap-2: full P&L / balance-sheet / cash-flow statements (long format),
    # one parquet per company under fundamentals/statements/.
    stmt_sym_id = get_or_create_subfolder(drive, fund_id, "statements")
    existing_stmt = list_files_in_folder(drive, stmt_sym_id)

    if args.resume:
        work = [(s, t) for (s, t) in work if f"{s}.parquet" not in existing]
        log(f"Resume mode: {len(work)} symbols left after skipping {len(existing)} done.")
    log(f"Symbols to fetch: {len(work)} (full universe NSE+BSE)")

    rows = []
    stmt_total = 0
    t_start = time.time()
    fail_count = 0
    for i, (sym, token) in enumerate(work, 1):
        try:
            soup = client.fetch_company(token)
        except CookieExpiredError:
            log("Stopping run — cookie expired. Refresh instructions printed above.")
            return
        except Exception as e:
            log(f"  {sym}: fetch error — {str(e)[:80]}")
            fail_count += 1
            continue

        if soup is None:
            fail_count += 1
            continue

        try:
            summary = client.extract_summary(sym, soup)
            summary["fetched_at"] = datetime.now().isoformat()
            rows.append(summary)
            # Gap-2: full statements (long format), one parquet per company.
            stmts = client.extract_statements(sym, soup)
            stmt_total += len(stmts)
            if not args.dry_run:
                sym_df = pd.DataFrame([summary])
                upload_parquet(drive, per_sym_id, f"{sym}.parquet", sym_df,
                               existing.get(f"{sym}.parquet"))
                if stmts:
                    stmt_df = pd.DataFrame(stmts)
                    stmt_df["fetched_at"] = summary["fetched_at"]
                    upload_parquet(drive, stmt_sym_id, f"{sym}.parquet", stmt_df,
                                   existing_stmt.get(f"{sym}.parquet"))
        except Exception as e:
            log(f"  {sym}: parse error — {str(e)[:80]}")
            fail_count += 1

        if i % 25 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(work) - i) / rate / 60
            log(f"  [{i}/{len(work)}] ok={len(rows)} fail={fail_count} "
                f"rate {rate:.2f}/s | ETA {eta:.1f}m")

        # Checkpoint: flush summary.parquet periodically so a truncated run (cookie
        # expiry, runner hiccup, local restart) still persists partial progress
        # instead of losing everything — summary is otherwise only written at the end.
        if i % 500 == 0 and rows and not args.dry_run:
            upload_parquet(drive, fund_id, "summary.parquet", pd.DataFrame(rows),
                           find_file(drive, fund_id, "summary.parquet"))
            log(f"  [checkpoint] summary.parquet flushed ({len(rows)} rows)")

    if rows and not args.dry_run:
        summary_df = pd.DataFrame(rows)
        sum_fid = find_file(drive, fund_id, "summary.parquet")
        if args.recent_results_days and sum_fid:
            # Incremental: UPSERT into the full-universe summary (a plain write
            # would shrink 5.5k rows down to today's delta).
            full = download_parquet(drive, sum_fid)
            full = full[~full["symbol"].astype(str).isin(
                summary_df["symbol"].astype(str))]
            summary_df = pd.concat([full, summary_df], ignore_index=True)
        upload_parquet(drive, fund_id, "summary.parquet", summary_df, sum_fid)
        log(f"Wrote fundamentals/summary.parquet ({len(summary_df)} rows)")
    elif rows:
        log(f"[dry-run] would write fundamentals/summary.parquet ({len(rows)} rows) "
            f"+ statements ({stmt_total} rows across statements/) — no write")
    log(f"Done. ok={len(rows)} statements={stmt_total} fail={fail_count} "
        f"elapsed={(time.time()-t_start)/60:.1f}m")


if __name__ == "__main__":
    main()
