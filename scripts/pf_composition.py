r"""
pf_composition.py — what did my portfolio look like on <date>?  (user 2026-08-29)

WHY THIS EXISTS
    Every holdings snapshot is ALREADY stored: `pf_decision_tracker.py` appends one
    row per (date, holding) to `company_repo/_index/pf_tracker/pf_snapshots.parquet`
    on every change. What was missing was a way to ASK it. This is that reader — it
    adds no storage and writes nothing by default.

    Snapshots are change-gated: a new one is only appended when the book actually
    changes, so there is NOT a row for every calendar day. Asking for a date with no
    snapshot returns the latest snapshot ON OR BEFORE it — which is genuinely what
    the portfolio was on that date, since nothing changed in between. The answer
    always states which snapshot it came from and how many days back that was, so a
    stale answer can never be mistaken for a fresh one.

USAGE
    python scripts/pf_composition.py --on 2026-07-28      # a specific day
    python scripts/pf_composition.py --on 2026-08         # first snapshot of a month
    python scripts/pf_composition.py --on "28 Jul 2026"   # free-form dates work
    python scripts/pf_composition.py --list               # every date available
    python scripts/pf_composition.py --compare 2026-07-23 2026-08-28   # what changed
    python scripts/pf_composition.py --on 2026-08-01 --csv out.csv     # export

NOTHING BEFORE 2026-07-23 EXISTS. The ledger begins there, Drive keeps only the
newest holdings file (`sync_pf` deletes the rest), so earlier dates cannot be
reconstructed and are reported as unavailable rather than guessed at.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dateutil import parser as dateparser
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

# Reuse the tracker's Drive helpers and ISIN-alias logic (CLAUDE.md rule 4) so a
# corporate action doesn't show one company under two identities across dates.
from _extractor_base import log, get_drive                      # noqa: E402
from pf_decision_tracker import (                               # noqa: E402
    _folder, _read_drive_parquet, build_isin_alias, apply_isin_alias,
)


def load_snapshots(drive, root_id) -> pd.DataFrame:
    """Every stored snapshot, with ISIN changes collapsed onto one identity."""
    out_id = _folder(drive, root_id, "company_repo", "_index", "pf_tracker")
    snaps = _read_drive_parquet(drive, out_id, "pf_snapshots.parquet")
    if snaps.empty:
        return snaps
    alias = build_isin_alias(snaps)
    if not alias.empty:
        snaps = apply_isin_alias(snaps, alias)
    return snaps.sort_values(["snapshot_date", "weight_pct"], ascending=[True, False])


def resolve_date(token: str, dates: list[str]) -> tuple[str | None, str]:
    """Turn a user token into one stored snapshot date.

    A bare month (`2026-08`) means the FIRST snapshot of that month, matching how
    the monthly cohorts are frozen. A full date means the latest snapshot on or
    before it. Returns (snapshot_date | None, human explanation)."""
    t = str(token).strip()
    if not dates:
        return None, "no snapshots stored at all"

    # Bare month → first snapshot in it.
    if len(t) == 7 and t[4] in "-/" and t[:4].isdigit():
        m = t.replace("/", "-")
        inm = [d for d in dates if d.startswith(m)]
        if inm:
            return inm[0], f"first snapshot of {m}"
        return None, f"no snapshot in {m} (ledger runs {dates[0]} .. {dates[-1]})"

    try:
        want = dateparser.parse(t, dayfirst=False, default=datetime(
            date.today().year, 1, 1)).date().isoformat()
    except (ValueError, OverflowError):
        return None, f"could not read {t!r} as a date"

    on_or_before = [d for d in dates if d <= want]
    if not on_or_before:
        return None, (f"{want} predates the ledger — nothing before {dates[0]} exists "
                      f"(Drive keeps only the newest holdings file, so it cannot be "
                      f"reconstructed)")
    got = on_or_before[-1]
    lag = (date.fromisoformat(want) - date.fromisoformat(got)).days
    if got == want:
        return got, f"exact snapshot for {want}"
    return got, (f"no snapshot on {want}; showing {got} — the latest before it "
                 f"({lag} day{'s' if lag != 1 else ''} earlier). Nothing changed in "
                 f"between, so this IS what the book was on {want}.")


def show(snaps: pd.DataFrame, snap_date: str, note: str, csv: str | None) -> None:
    g = snaps[snaps["snapshot_date"] == snap_date].copy()
    g = g.sort_values("weight_pct", ascending=False).reset_index(drop=True)
    g.insert(0, "rank", range(1, len(g) + 1))
    src = g["source_file"].iloc[0] if "source_file" in g.columns and len(g) else "?"
    tot = g["weight_pct"].sum()

    print()
    print(f"PF composition — {snap_date}")
    print(f"  {note}")
    print(f"  {len(g)} holdings · weights sum to {tot:.2f}% · from {src}")
    print("-" * 62)
    print(f"{'#':>3}  {'SYMBOL':<14} {'WT%':>7}  NAME")
    print("-" * 62)
    for r in g.itertuples():
        w = "" if pd.isna(r.weight_pct) else f"{r.weight_pct:7.2f}"
        print(f"{r.rank:>3}  {str(r.symbol)[:14]:<14} {w}  {str(r.name)[:28]}")
    if csv:
        g.to_csv(csv, index=False)
        print(f"\n  -> written to {csv}")


def compare(snaps: pd.DataFrame, a: str, b: str) -> None:
    """What changed between two snapshots — entries, exits, and weight moves."""
    ga = snaps[snaps["snapshot_date"] == a].set_index("isin")
    gb = snaps[snaps["snapshot_date"] == b].set_index("isin")
    entered = [i for i in gb.index if i not in ga.index]
    exited = [i for i in ga.index if i not in gb.index]
    both = [i for i in gb.index if i in ga.index]

    print(f"\nPF change: {a}  ->  {b}")
    print(f"  {len(ga)} holdings -> {len(gb)}   "
          f"(+{len(entered)} entered, -{len(exited)} exited, {len(both)} held throughout)")

    if entered:
        print(f"\n  ENTERED ({len(entered)})")
        for i in sorted(entered, key=lambda x: -float(gb.loc[x, 'weight_pct'] or 0)):
            print(f"    {str(gb.loc[i,'symbol'])[:14]:<14} {float(gb.loc[i,'weight_pct']):6.2f}%")
    if exited:
        print(f"\n  EXITED ({len(exited)})")
        for i in sorted(exited, key=lambda x: -float(ga.loc[x, 'weight_pct'] or 0)):
            print(f"    {str(ga.loc[i,'symbol'])[:14]:<14} {float(ga.loc[i,'weight_pct']):6.2f}%  (was)")

    moves = []
    for i in both:
        wa, wb = float(ga.loc[i, "weight_pct"]), float(gb.loc[i, "weight_pct"])
        if abs(wb - wa) >= 0.5:            # ignore pure price drift noise
            moves.append((str(gb.loc[i, "symbol"]), wa, wb, wb - wa))
    if moves:
        print(f"\n  WEIGHT MOVED by >=0.5pp ({len(moves)})  — price drift OR trading")
        for s, wa, wb, d in sorted(moves, key=lambda x: -abs(x[3])):
            print(f"    {s[:14]:<14} {wa:6.2f}% -> {wb:6.2f}%   {d:+6.2f}pp")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--on", help="Date (2026-07-28), month (2026-08), or free-form.")
    ap.add_argument("--list", action="store_true", help="List every stored snapshot date.")
    ap.add_argument("--compare", nargs=2, metavar=("FROM", "TO"),
                    help="Show what changed between two dates.")
    ap.add_argument("--csv", default=None, help="Also write the composition to this CSV.")
    args = ap.parse_args()

    if not (args.on or args.list or args.compare):
        ap.error("give --on, --list or --compare")

    snaps = load_snapshots(get_drive(), os.environ["GDRIVE_FOLDER_ID"])
    if snaps.empty:
        log("No snapshots stored yet.")
        sys.exit(1)
    dates = sorted(snaps["snapshot_date"].unique())

    if args.list:
        print(f"\n{len(dates)} stored snapshots · {dates[0]} .. {dates[-1]}")
        print("(change-gated: a snapshot exists only where the book actually changed)\n")
        for d in dates:
            n = (snaps["snapshot_date"] == d).sum()
            print(f"  {d}   {n:>3} holdings")

    if args.compare:
        pair = []
        for tok in args.compare:
            d, note = resolve_date(tok, dates)
            if not d:
                log(f"ERROR: {note}")
                sys.exit(1)
            pair.append(d)
        compare(snaps, pair[0], pair[1])

    if args.on:
        d, note = resolve_date(args.on, dates)
        if not d:
            log(f"ERROR: {note}")
            sys.exit(1)
        show(snaps, d, note, args.csv)


if __name__ == "__main__":
    main()
