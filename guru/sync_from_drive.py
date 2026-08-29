r"""
SYNC FROM DRIVE — sync_from_drive.py  (Project Guru)

LEVERAGES THE EXISTING PIPELINES instead of re-fetching. The main repo already
runs on schedule and writes to Google Drive:
    daily.yml        Mon-Fri 16:00 IST : OHLCV + indices  -> data/
    fundamentals.yml Mon 06:30 IST     : full-universe    -> fundamentals/
    universe_refresh Sun               : master list+mcap -> universe/
So Screener is already being scraped weekly for the whole universe — guru must
NOT scrape it again. This script just copies what is already there into the
guru store.

Pulls:
  * fundamentals/statements/<SYM>.parquet -> guru/data/fundamentals_hist/<key>.parquet
    (quarterly + annual statements; the source guru's quarterly_unified is built from)
  * universe/market_cap.csv               -> guru/data/market_cap_current.csv
    (current market cap — the >=100cr filter, already computed by the repo)
  * fundamentals/summary.parquet          -> guru/data/fundamentals_summary_current.parquet
    (latest-quarter snapshot incl. roce/roe/debt — useful for live screening)

After running, rebuild the unified quarterly store:
    python guru/merge_quarterly_unified.py

Usage:
    python guru/sync_from_drive.py --light     # mcap + summary only (seconds)
    python guru/sync_from_drive.py             # + statements (slower, paginated)
    python guru/sync_from_drive.py --limit 500
"""
from __future__ import annotations
import argparse, io, os, sys
from datetime import datetime
import pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
FUND_DIR = os.path.join(DATA, "fundamentals_hist")
SCRIPTS = os.path.join(os.path.dirname(GURU), "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(GURU).parent / ".env")
from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes)


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def list_folder(d, fid):
    out, tok = [], None
    while True:
        r = d.files().list(q=f"'{fid}' in parents and trashed=false",
                           fields="nextPageToken, files(id,name,modifiedTime)",
                           pageSize=1000, pageToken=tok).execute()
        out += r.get("files", [])
        tok = r.get("nextPageToken")
        if not tok:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true",
                    help="mcap + summary only, skip per-company statements")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    os.makedirs(DATA, exist_ok=True)

    # ---- 1. market cap (the >=100cr filter, already computed by the repo) ----
    uid = get_or_create_subfolder(d, root, "universe")
    mid = find_file(d, uid, "market_cap.csv")
    if mid:
        mc = pd.read_csv(io.BytesIO(download_bytes(d, mid)))
        mc.to_csv(os.path.join(DATA, "market_cap_current.csv"), index=False)
        col = next((c for c in mc.columns if "cap" in c.lower()), None)
        n100 = int((pd.to_numeric(mc[col], errors="coerce") >= 100).sum()) if col else -1
        log(f"market_cap_current.csv: {len(mc):,} stocks | >=100cr: {n100:,}")

    # ---- 2. latest-quarter fundamentals snapshot ----
    fid = get_or_create_subfolder(d, root, "fundamentals")
    sid_sum = find_file(d, fid, "summary.parquet")
    if sid_sum:
        s = pd.read_parquet(io.BytesIO(download_bytes(d, sid_sum)))
        s.to_parquet(os.path.join(DATA, "fundamentals_summary_current.parquet"),
                     index=False)
        lq = (s["latest_quarter_label"].value_counts().head(2).to_dict()
              if "latest_quarter_label" in s else {})
        log(f"fundamentals_summary_current.parquet: {len(s):,} rows | latest quarters {lq}")

    if args.light:
        log("light mode — skipping per-company statements")
        return

    # ---- 3. per-company statements -> guru fundamentals_hist ----
    uni = pd.read_parquet(os.path.join(DATA, "universe_hist.parquet"),
                          columns=["guru_key", "nse_symbol"])
    sym2key = {str(s).strip(): k for k, s in
               zip(uni.guru_key, uni.nse_symbol) if isinstance(s, str) and s.strip()}
    stid = get_or_create_subfolder(d, fid, "statements")
    files = list_folder(d, stid)
    log(f"statements on Drive: {len(files):,}")
    if args.limit:
        files = files[:args.limit]
    os.makedirs(FUND_DIR, exist_ok=True)
    n_new = n_upd = n_skip = 0
    for i, f in enumerate(files, 1):
        sym = f["name"].replace(".parquet", "")
        key = sym2key.get(sym)
        if key is None:
            n_skip += 1
            continue
        dst = os.path.join(FUND_DIR, f"{key}.parquet")
        try:
            new = pd.read_parquet(io.BytesIO(download_bytes(d, f["id"])))
            if new.empty:
                continue
            if "symbol" in new.columns:
                new = new.drop(columns=["symbol"])
            new["fetched_at"] = f["modifiedTime"][:19]
            if os.path.exists(dst):
                old = pd.read_parquet(dst)
                comb = pd.concat([old, new], ignore_index=True)
                keys = [c for c in ("statement", "line_item", "period") if c in comb.columns]
                if keys:
                    comb = comb.drop_duplicates(subset=keys, keep="last")
                comb.to_parquet(dst, index=False)
                n_upd += 1
            else:
                new.to_parquet(dst, index=False)
                n_new += 1
        except Exception:
            continue
        if i % 300 == 0:
            log(f"  {i}/{len(files)} (new={n_new} updated={n_upd} unmapped={n_skip})")
    log(f"STATEMENTS DONE: new={n_new} updated={n_upd} unmapped={n_skip}")
    log("next: python guru/merge_quarterly_unified.py   (rebuild unified quarterly)")


if __name__ == "__main__":
    main()
