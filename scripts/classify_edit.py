r"""
classify_edit.py — revise a company's classification + LOCK it (classify.bat).

Edits company_repo/_index/company_classification.csv on Drive. A locked row is
never overwritten by the nightly build_classification refresh, so your manual
fixes persist. peers recompute automatically on the next build.

Usage:
    python scripts/classify_edit.py --show TCS
    python scripts/classify_edit.py --set TCS sector="Information Technology" \
        industry="IT Services" subsector="IT Services - Large" \
        peer_group="IT Large Cap"
    python scripts/classify_edit.py --unlock TCS
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)

EDITABLE = {"segment", "macro_sector", "sector", "industry", "subsector",
            "peer_group"}


def _idx(drive):
    root = os.environ["GDRIVE_FOLDER_ID"]
    return get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")


def _load(drive, idx):
    fid = find_file(drive, idx, "company_classification.csv")
    if not fid:
        log("company_classification.csv not found — run build_classification first.")
        return None
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid))).fillna("")


def _save(drive, idx, df):
    upload_bytes(drive, idx, "company_classification.csv",
                 df.to_csv(index=False).encode("utf-8"), "text/csv",
                 existing_id=find_file(drive, idx, "company_classification.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=str, metavar="SYMBOL")
    ap.add_argument("--set", nargs="+", metavar="SYMBOL key=value ...")
    ap.add_argument("--unlock", type=str, metavar="SYMBOL")
    args = ap.parse_args()

    drive = get_drive()
    idx = _idx(drive)
    df = _load(drive, idx)
    if df is None:
        return

    def _row(sym):
        m = df["symbol"].astype(str).str.upper() == sym.upper()
        return m if m.any() else None

    if args.show:
        m = _row(args.show)
        if m is None:
            print(f"  {args.show} not found.")
            return
        print(df[m][["symbol", "name", "segment", "macro_sector", "sector",
                     "industry", "subsector", "peer_group", "locked"]]
              .T.to_string())
    elif args.unlock:
        m = _row(args.unlock)
        if m is None:
            print(f"  {args.unlock} not found.")
            return
        df.loc[m, "locked"] = 0
        _save(drive, idx, df)
        print(f"  {args.unlock} unlocked — nightly refresh may update it again.")
    elif args.set:
        sym = args.set[0]
        m = _row(sym)
        if m is None:
            print(f"  {sym} not found.")
            return
        changed = []
        for kv in args.set[1:]:
            k, _, v = kv.partition("=")
            k = k.strip().lower()
            v = v.strip().strip('"').strip("'")
            if k in EDITABLE:
                df.loc[m, k] = v
                changed.append(f"{k}={v}")
            else:
                print(f"  ignored '{k}' (editable: {', '.join(sorted(EDITABLE))})")
        if changed:
            df.loc[m, "locked"] = 1
            df.loc[m, "source"] = "user"
            df.loc[m, "updated_at"] = datetime.now().strftime("%Y-%m-%d")
            _save(drive, idx, df)
            print(f"  {sym} updated + LOCKED: {'; '.join(changed)}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
