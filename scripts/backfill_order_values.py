r"""
backfill_order_values.py — fill order_value_cr over historical announcements.

WHY
    `ingest_announcements.py` gained an `order_value_cr` column on 2026-08-15, but
    that only applies to filings summarised from then on. Every order_win row already
    in `announcement_ledger.parquet` reads NaN, and those rows are the ONLY actuals
    source for an order-inflow commitment in guidance_progress. Without this, a
    company that guided "Rs 12,000 cr of FY27 order inflow" shows 0% booked purely
    because the amounts were never captured.

    Costs no Gemini quota: the summary text is already stored, so this is a pure
    regex pass over it (order_value.py). The source PDFs are long gone — the 2-day
    retention rule — so the stored headline + summary is all there is.

IDEMPOTENT
    Re-running produces the same result: the parse is a pure function of
    headline + summary. Rows that already carry a value are left alone unless
    --overwrite is passed, so a future LLM-supplied value is never clobbered by the
    weaker regex read.

Usage:
    python scripts/backfill_order_values.py --dry-run    # report only, no write
    python scripts/backfill_order_values.py              # write back to Drive
    python scripts/backfill_order_values.py --overwrite  # re-parse even filled rows
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, load_parquet,  # noqa: E402
                             save_parquet, log)
from ingest_announcements import LEDGER_COLS  # noqa: E402
from order_value import parse_order_value  # noqa: E402

LEDGER = "announcement_ledger.parquet"


def _ascii(s) -> str:
    """The local console is cp1252 and the summaries are full of rupee signs."""
    return str(s).encode("ascii", "replace").decode()


def backfill(df: pd.DataFrame, overwrite: bool = False) -> tuple[pd.DataFrame, dict]:
    """Pure transform — returns (new_df, stats). Testable without Drive."""
    df = df.copy()
    if "order_value_cr" not in df.columns:
        df["order_value_cr"] = None
    if "order_currency" not in df.columns:
        df["order_currency"] = ""

    is_order = df["event_type"].astype(str) == "order_win"
    already = pd.to_numeric(df["order_value_cr"], errors="coerce").notna()
    target = is_order & (~already if not overwrite else True)

    hits = 0
    samples = []
    for i in df.index[target]:
        text = f"{df.at[i, 'headline']} . {df.at[i, 'summary']}"
        p = parse_order_value(text)
        if not p:
            continue
        df.at[i, "order_value_cr"] = p["value_cr"]
        df.at[i, "order_currency"] = p["currency"]
        hits += 1
        samples.append((df.at[i, "symbol"], df.at[i, "ann_date"],
                        p["value_cr"], p["currency"], p["matched"]))

    stats = {
        "rows": len(df),
        "order_win": int(is_order.sum()),
        "considered": int(target.sum()),
        "parsed": hits,
        "unparsed": int(target.sum()) - hits,
        "already_filled": int((is_order & already).sum()),
        "samples": samples,
    }
    return df, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and report, write nothing")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-parse rows that already carry a value")
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    idx = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")

    df = load_parquet(drive, idx, LEDGER, LEDGER_COLS)
    if df.empty:
        log("announcement_ledger is empty — nothing to backfill.")
        return 0

    out, st = backfill(df, overwrite=args.overwrite)
    pct = (st["parsed"] / st["considered"] * 100) if st["considered"] else 0.0
    log(f"ledger rows={st['rows']}  order_win={st['order_win']}  "
        f"already filled={st['already_filled']}")
    log(f"considered={st['considered']}  PARSED={st['parsed']} ({pct:.0f}%)  "
        f"unparsed={st['unparsed']}")

    if st["samples"]:
        log("largest parsed:")
        for sym, dt, val, cur, matched in sorted(st["samples"],
                                                 key=lambda x: -x[2])[:15]:
            log(f"   {_ascii(sym):<12} {str(dt)[:10]}  {val:>12,.2f} cr  {cur}"
                f"   <- {_ascii(matched)[:34]}")

    total = pd.to_numeric(out.loc[out["event_type"].astype(str) == "order_win",
                                  "order_value_cr"], errors="coerce").sum()
    log(f"total order value now on the ledger: Rs {total:,.0f} cr")

    if args.dry_run:
        log("DRY RUN - nothing written.")
        return 0
    if not st["parsed"]:
        log("nothing new parsed - not rewriting the ledger.")
        return 0
    save_parquet(drive, idx, LEDGER, out)
    log(f"{LEDGER}: wrote {st['parsed']} order values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
