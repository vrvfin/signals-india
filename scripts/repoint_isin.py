r"""
repoint_isin.py — rewrite superseded ISINs to their current one (user 2026-08-31).

WHAT THIS IS FOR
    When a company's ISIN changes, every row already filed under the old one is
    orphaned: it no longer joins to anything written afterwards, so a company's
    guidance history, rating history, quarterly numbers and processed-document list
    all silently stop at the change. 8 companies are affected, ~4,639 rows.

    This reads `company_repo/_index/isin_alias.parquet` — built and corroborated
    against the exchange's own list by `isin_registry.py` — and rewrites the `isin`
    column old -> new so the history joins up again.

    THIS IS THE ONLY DESTRUCTIVE STEP in the ISIN work. Everything before it merely
    recorded facts. Accordingly:

      • --dry-run is the DEFAULT. Nothing is written unless --live is passed.
      • Every table is copied to a timestamped backup folder BEFORE it is touched.
      • Row COUNT must be identical before and after; a mismatch aborts that table.
      • processing_queue.parquet goes LAST and under the shared lock, because it is
        the one global document ledger (CLAUDE.md rule 7) and Phase 2 writes to it.
      • Duplicate keys created by the merge are REPORTED, never silently collapsed —
        deciding what to do with two rows for one document is not this script's call.

USAGE
    python scripts/repoint_isin.py                 # dry-run: report only (default)
    python scripts/repoint_isin.py --live          # do it, with backups
    python scripts/repoint_isin.py --live --only ratings.parquet   # one table
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

from _extractor_base import (  # noqa: E402
    log, get_drive, get_or_create_subfolder, find_file, download_bytes,
    load_parquet, save_parquet, acquire_lock, release_lock,
)

REGISTRY = "isin_alias.parquet"
REG_COLS = ["old_isin", "new_isin", "symbol", "exchange", "changed_between",
            "event_type", "ratio", "ex_date", "confirmed", "source", "detected_on"]
QUEUE = "processing_queue.parquet"          # rule 7 — always last, always locked
LOCK_NAME = "_extract.lock"
BACKUP_ROOT = "_backup_isin_repoint"
# What actually identifies one document in the queue. NOT (isin, doc_type, period):
# `period` is null on 41,749 of 43,197 rows, so that key reports thousands of
# "duplicates" that have nothing to do with a repoint. doc_id is the real identity.
QUEUE_KEY = ["doc_id"]


def build_map(reg: pd.DataFrame) -> dict:
    """old -> current isin, following chains (A->B->C all land on C)."""
    if reg.empty:
        return {}
    conf = reg[reg["confirmed"].astype(bool)] if "confirmed" in reg.columns else reg
    if len(conf) < len(reg):
        log(f"  ignoring {len(reg) - len(conf)} UNCONFIRMED registry row(s)")
    direct = dict(zip(conf["old_isin"].astype(str), conf["new_isin"].astype(str)))

    def resolve(i, seen=None):
        seen = seen or set()
        while i in direct and i not in seen:
            seen.add(i)
            i = direct[i]
        return i

    return {o: resolve(o) for o in direct}


def _read(drive, fid, name):
    f = find_file(drive, fid, name)
    if not f:
        return None, None
    raw = download_bytes(drive, f)
    df = (pd.read_csv(io.BytesIO(raw)) if name.lower().endswith(".csv")
          else pd.read_parquet(io.BytesIO(raw)))
    return df, raw


def _backup(drive, index_id, stamp: str, name: str, raw: bytes) -> None:
    """Byte-for-byte copy of the ORIGINAL file, before anything is rewritten."""
    from googleapiclient.http import MediaInMemoryUpload
    bfid = get_or_create_subfolder(drive, index_id, BACKUP_ROOT)
    bfid = get_or_create_subfolder(drive, bfid, stamp)
    drive.files().create(
        body={"name": name, "parents": [bfid]},
        media_body=MediaInMemoryUpload(raw, mimetype="application/octet-stream"),
        fields="id").execute()


def _write(drive, fid, name: str, df: pd.DataFrame) -> None:
    from googleapiclient.http import MediaInMemoryUpload
    if name.lower().endswith(".csv"):
        buf = df.to_csv(index=False).encode()
        mime = "text/csv"
    else:
        b = io.BytesIO()
        df.to_parquet(b, index=False)
        buf, mime = b.getvalue(), "application/octet-stream"
    existing = find_file(drive, fid, name)
    media = MediaInMemoryUpload(buf, mimetype=mime, resumable=False)
    if existing:
        drive.files().update(fileId=existing, media_body=media).execute()
    else:
        drive.files().create(body={"name": name, "parents": [fid]},
                             media_body=media, fields="id").execute()


def repoint_table(drive, index_id, name: str, mapping: dict, stamp: str,
                  live: bool) -> tuple[int, int]:
    """Rewrite one table. Returns (rows_changed, dup_keys_created)."""
    df, raw = _read(drive, index_id, name)
    if df is None or df.empty or "isin" not in df.columns:
        return 0, 0
    col = df["isin"].astype(str)
    hits = int(col.isin(mapping).sum())
    if hits == 0:
        return 0, 0

    before_rows = len(df)
    out = df.copy()
    out["isin"] = col.map(lambda v: mapping.get(v, v))

    # Two rows for one document can now exist — one filed under each ISIN. Report it;
    # collapsing them is a judgement call about which extraction wins, not a rewrite.
    dups = 0
    if name == QUEUE:
        keys = [c for c in QUEUE_KEY if c in out.columns]
        if keys:  # measured delta is 0 — repointing merges no documents
            dups = int(out.duplicated(subset=keys, keep=False).sum()
                       - df.duplicated(subset=keys, keep=False).sum())

    if len(out) != before_rows:
        log(f"    !! {name}: row count changed {before_rows} -> {len(out)} — SKIPPED")
        return 0, 0

    if live:
        _backup(drive, index_id, stamp, name, raw)
        _write(drive, index_id, name, out)
    log(f"    {'repointed' if live else 'would repoint'} {name:<42} {hits:>5} rows"
        + (f"   [+{dups} duplicate key(s) created]" if dups else ""))
    return hits, dups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="Actually write. Without this it is a dry run (the default).")
    ap.add_argument("--dry-run", action="store_true", help="Explicit no-op (default).")
    ap.add_argument("--only", default=None, help="Restrict to one table name.")
    args = ap.parse_args()
    live = args.live and not args.dry_run

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log("=" * 70)
    log(f"repoint_isin — {'LIVE (writes + backups)' if live else 'DRY RUN (no writes)'}")
    log("=" * 70)

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    index_id = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root_id, "company_repo"), "_index")

    reg = load_parquet(drive, index_id, REGISTRY, REG_COLS)
    mapping = build_map(reg)
    if not mapping:
        log("Registry is empty or unconfirmed — nothing to do. "
            "Run isin_registry.py --confirm-known first.")
        sys.exit(1)
    log(f"  {len(mapping)} ISIN mapping(s) from the registry:")
    for o, n in mapping.items():
        sym = reg[reg["old_isin"] == o]["symbol"].iloc[0] if not reg.empty else "?"
        log(f"    {sym:<12} {o} -> {n}")

    files = drive.files().list(q=f"'{index_id}' in parents and trashed=false",
                               fields="files(name)", pageSize=300
                               ).execute().get("files", [])
    names = sorted(f["name"] for f in files
                   if f["name"].endswith((".parquet", ".csv")) and f["name"] != REGISTRY)
    if args.only:
        names = [n for n in names if n == args.only]
    # The global document ledger goes last, and only under the lock.
    names = [n for n in names if n != QUEUE] + ([QUEUE] if QUEUE in names else [])

    log(f"\n  scanning {len(names)} table(s) in company_repo/_index …")
    total = dups_total = touched = 0
    for name in names:
        if name == QUEUE:
            if not live:
                c, d = repoint_table(drive, index_id, name, mapping, stamp, False)
                total += c; dups_total += d; touched += bool(c)
                continue
            log(f"\n  {QUEUE} is the ONE global document ledger — taking the lock.")
            if not acquire_lock(drive, index_id, LOCK_NAME, "repoint_isin",
                                wait_min=5.0):
                log("  !! could not acquire the lock — Phase 2 is busy. "
                    "processing_queue NOT touched; re-run later for it alone:")
                log(f"     python scripts/repoint_isin.py --live --only {QUEUE}")
                break
            try:
                c, d = repoint_table(drive, index_id, name, mapping, stamp, True)
                total += c; dups_total += d; touched += bool(c)
            finally:
                release_lock(drive, index_id, LOCK_NAME)
                log("  lock released.")
        else:
            c, d = repoint_table(drive, index_id, name, mapping, stamp, live)
            total += c; dups_total += d; touched += bool(c)

    log("")
    log(f"  {'REPOINTED' if live else 'WOULD REPOINT'} {total} rows across {touched} table(s)")
    if dups_total:
        log(f"  {dups_total} duplicate document key(s) now exist in {QUEUE} — one row "
            f"from each ISIN. REPORTED ONLY, nothing removed; decide which wins before "
            f"the next Phase 2 run.")
    if live:
        log(f"  backups: company_repo/_index/{BACKUP_ROOT}/{stamp}/")
    else:
        log("  nothing was written. Re-run with --live to apply.")


if __name__ == "__main__":
    main()
