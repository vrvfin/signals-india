r"""
fix_deck_units.py — one-off: re-label stored deck figures whose OWN quote prints them in
Rs mn / bn / lakh but whose unit says crore (user 2026-10-15, plan ir-and-teardown-fixes P1).

Measured 2026-10-16 before this fix:
  deck_metrics   20 rows, 6 companies  (NETWEB Rs 25,069 million order book stored as cr,
                                         UNIVASTU 102,098 lakh stored as cr, AIMTRON ...)
  deck_summary    5 rows  (APLAPOLLO Rs 55.1 Bn revenue stored as 55.1 cr — 100x small)
  ppt_highlights / ppt_guidance carry no quote, so a row there is re-labelled only when the
  SAME company has the SAME number in a corrected deck_metrics / deck_summary row.

VALUES ARE NEVER CHANGED — only `unit`, to the scale the deck printed. The mail shows the
crore equivalent beside it. No schema change. Rule: deck_summary.corrected_unit (the word
printed directly after the number; "US$ 32.85 million (~ Rs 281 Cr)" stays crore).

    python scripts/fix_deck_units.py --dry-run      # count + list, no write
    python scripts/fix_deck_units.py                # write, under _extract.lock
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

import deck_summary as DS

QUOTED = ("deck_metrics.parquet", "deck_summary.parquet")
UNQUOTED = ("ppt_highlights.parquet", "ppt_guidance.parquet")
_LOCK_NAME = "_extract.lock"


def _numkey(v):
    try:
        f = float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 4)


def fix_quoted(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fixed df, changed rows) for a table whose rows carry `evidence`. Pure."""
    if df is None or df.empty or "evidence" not in df.columns:
        return df, pd.DataFrame()
    new = [DS.corrected_unit(u, v, e) for u, v, e in
           zip(df["unit"], df["value"], df["evidence"])]
    mask = pd.Series([n is not None for n in new], index=df.index)
    out = df.copy()
    out.loc[mask, "unit"] = [n for n in new if n is not None]
    ch = df[mask].assign(new_unit=out.loc[mask, "unit"])
    return out, ch


def scale_map(df: pd.DataFrame) -> dict:
    """{(isin, number): 'mn'|'bn'|'lakh'} from QUOTED rows whose (corrected) unit is a
    non-crore rupee scale — including rows that were right all along (CYIENTDLM's deck
    summary already said Mn; its unquoted highlight said cr). Pure."""
    out = {}
    if df is None or df.empty:
        return out
    for isin, v, u in zip(df["isin"].astype(str), df["value"], df["unit"]):
        sc = {"Rs mn": "mn", "Rs bn": "bn", "Rs lakh": "lakh",
              "US$ mn": "usd_mn", "US$ bn": "usd_bn"}.get(DS.display_unit(u))
        k = _numkey(v)
        if sc and k is not None:
            out[(isin, k)] = sc
    return out


def fix_unquoted(df: pd.DataFrame, known: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-label a crore row when `known[(isin, number)]` says that company printed that
    exact number in another scale. Pure."""
    if df is None or df.empty:
        return df, pd.DataFrame()
    cr = df["unit"].astype(str).str.strip().str.lower().isin(DS._CR_UNITS)
    new = []
    for isin, v, is_cr in zip(df["isin"].astype(str), df["value"], cr):
        k = _numkey(v)
        new.append(known.get((isin, k)) if is_cr and k is not None else None)
    mask = pd.Series([n is not None for n in new], index=df.index)
    out = df.copy()
    out.loc[mask, "unit"] = [n for n in new if n is not None]
    return out, df[mask].assign(new_unit=out.loc[mask, "unit"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="count and list; no write")
    args = ap.parse_args()

    from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                                 download_bytes, save_parquet, acquire_lock, release_lock)
    import io
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    idx = get_or_create_subfolder(drive, get_or_create_subfolder(drive, root, "company_repo"),
                                  "_index")

    def read(name):
        fid = find_file(drive, idx, name)
        return pd.read_parquet(io.BytesIO(download_bytes(drive, fid))) if fid else None

    if not args.dry_run:
        if not acquire_lock(drive, idx, _LOCK_NAME, "fix_deck_units", wait_min=15):
            print("lock busy — nothing written; run again later")
            return 1
    try:
        fixed, known = {}, {}
        for name in QUOTED:
            df = read(name)
            out, ch = fix_quoted(df)
            print(f"\n{name}: {0 if df is None else len(df)} rows -> {len(ch)} re-labelled"
                  + (f" across {ch['symbol'].nunique()} companies" if len(ch) else ""))
            for _, r in ch.iterrows():
                lab = r.get("metric", r.get("label"))
                print(f"   {r.get('symbol')} | {lab} | {r['value']} {r['unit']} -> {r['new_unit']}"
                      f" | {str(r['evidence'])[:90]!r}")
            known.update(scale_map(out))
            if len(ch):
                fixed[name] = out
        for name in UNQUOTED:
            df = read(name)
            out, ch = fix_unquoted(df, known)
            print(f"\n{name}: {0 if df is None else len(df)} rows -> {len(ch)} re-labelled "
                  f"(same company, same number, scale known from a quoted row)")
            for _, r in ch.iterrows():
                print(f"   {r.get('symbol')} | {str(r.get('statement', r.get('metric')))[:70]} | "
                      f"{r['value']} {r['unit']} -> {r['new_unit']}")
            if len(ch):
                fixed[name] = out
        if args.dry_run:
            print("\n--dry-run: nothing written")
            return 0
        for name, df in fixed.items():
            save_parquet(drive, idx, name, df)
            print(f"wrote {name}")
        return 0
    finally:
        if not args.dry_run:
            release_lock(drive, idx, _LOCK_NAME)


def _self_test() -> int:
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    dm = pd.DataFrame([
        {"isin": "I1", "symbol": "NETWEB", "metric": "Order book", "value": 25069.35,
         "unit": "cr", "evidence": "order book stood at Rs 25,069.35 million"},
        {"isin": "I2", "symbol": "DEEP", "metric": "Contract", "value": 281.0,
         "unit": "Cr", "evidence": "US$ 32.85 million (~ Rs 281 Cr)"},
        {"isin": "I3", "symbol": "X", "metric": "Util", "value": 87.0,
         "unit": "%", "evidence": "87 mn"}])
    out, ch = fix_quoted(dm)
    check("quoted: only the mn-printed crore row changes",
          list(out["unit"]) == ["mn", "Cr", "%"] and len(ch) == 1)
    check("quoted: values untouched", list(out["value"]) == list(dm["value"]))
    hi = pd.DataFrame([
        {"isin": "I1", "symbol": "NETWEB", "statement": "Record order book", "value": "25,069.35", "unit": "cr"},
        {"isin": "I1", "symbol": "NETWEB", "statement": "Revenue", "value": "1000", "unit": "cr"},
        {"isin": "I9", "symbol": "Y", "statement": "Order book", "value": "25069.35", "unit": "cr"}])
    out2, ch2 = fix_unquoted(hi, {("I1", 25069.35): "mn"})
    check("unquoted: same company + same number only",
          list(out2["unit"]) == ["mn", "cr", "cr"] and len(ch2) == 1)
    check("unquoted: empty input", fix_unquoted(pd.DataFrame(), {})[1].empty)
    ds = pd.DataFrame([{"isin": "C1", "value": 28137.0, "unit": "Mn"},
                       {"isin": "C1", "value": 26.5, "unit": "%"},
                       {"isin": "C1", "value": 393.0, "unit": "cr"}])
    check("scale map learns from rows already labelled Mn, not % or cr",
          scale_map(ds) == {("C1", 28137.0): "mn"})
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
