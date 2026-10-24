r"""
fix_results_units.py — one-off: correct stored results_gemini rows whose money figures are
in the filing's own scale (lakh / million) under a crore label (user-approved 2026-10-24).

Two deterministic rules, both read-only until --live:
  1. SCREENER: revenue vs Screener's revenue for the SAME company and quarter
     (results.parquet). A clean ~100x / ~10x / ~0.1x / ~0.01x ratio (within +/-25%) scales
     revenue, EBITDA and PAT together - a filing states all of them in one unit.
     Screener values of exactly 250.0 are skipped (a known results.parquet artefact).
  2. MARGIN: an EBITDA / PAT figure that disagrees with its OWN stated margin by a clean
     10x / 100x / 1000x is divided back (filing_results.margin_check).
Anything else (standalone vs consolidated, ~14x) is listed but NOT changed.

    python scripts/fix_results_units.py              # dry run: list every change
    python scripts/fix_results_units.py --live       # write, under _extract.lock
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import filing_results as FR

NAME = "results_gemini.parquet"
_LOCK = "_extract.lock"
_FACTORS = (100.0, 10.0, 0.1, 0.01)        # stored / screener -> divide by this


def _f(v):
    try:
        x = float(str(v).replace(",", "").replace("%", "").strip())
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def screener_revenue(res: pd.DataFrame) -> dict:
    """{(isin, 'Q2FY27'): revenue_cr} from results.parquet sales rows. Pure."""
    if res is None or res.empty:
        return {}
    s = res[res["metric"].astype(str).str.lower().str.contains("sales|revenue", regex=True)].copy()
    s["q"] = pd.to_datetime(s["latest_q"].astype(str), format="%b %Y", errors="coerce")
    out = {}
    for isin, q, v in zip(s["isin"].astype(str), s["q"], s["latest_val"]):
        if pd.isna(q):
            continue
        fy = q.year + 1 if q.month >= 4 else q.year
        qn = {6: 1, 9: 2, 12: 3, 3: 4}.get(q.month)
        val = _f(v)
        if qn and val and val > 0 and val != 250.0:
            out.setdefault((isin, f"Q{qn}FY{str(fy)[2:]}"), val)
    return out


def plan_fixes(df: pd.DataFrame, scr: dict) -> list[dict]:
    """[{idx, symbol, quarter, rule, field, old, new}] — every change, nothing applied. Pure."""
    out = []
    for i, r in df.iterrows():
        q = str(r.get("quarter") or "").replace(" ", "").upper()
        rev = _f(r.get("revenue_cr"))
        ref = scr.get((str(r.get("isin") or ""), q))
        scale = None
        if rev and ref:
            ratio = rev / ref
            scale = next((k for k in _FACTORS if 0.8 <= ratio / k <= 1.25), None)
        vals = {k: _f(r.get(k)) for k in ("revenue_cr", "ebitda_cr", "pat_cr")}
        if scale:
            new_rev = vals["revenue_cr"] / scale
            out.append({"idx": i, "symbol": r.get("symbol"), "quarter": r.get("quarter"),
                        "rule": f"screener x{1/scale:g}", "field": "revenue_cr",
                        "old": vals["revenue_cr"], "new": round(new_rev, 4)})
            vals["revenue_cr"] = new_rev
            # EBITDA / PAT are decided ON THEIR OWN: filings mix units (BETA: revenue in
            # lakh, PAT already in crore). Stated margin first, plausibility second; an
            # ambiguous figure is left as it is and listed, never guessed.
            for k, pk in (("ebitda_cr", "ebitda_margin_pct"), ("pat_cr", "pat_margin_pct")):
                v = vals[k]
                if v is None:
                    continue
                pct = _f(r.get(pk))
                cands = {"keep": v, "scaled": v / scale}
                if pct is not None and abs(pct) >= 0.5:
                    fit = [n for n, c in cands.items()
                           if 0.8 <= (c / new_rev * 100.0) / pct <= 1.25]
                else:
                    fit = [n for n, c in cands.items() if abs(c) <= new_rev]
                if fit == ["scaled"]:
                    out.append({"idx": i, "symbol": r.get("symbol"), "quarter": r.get("quarter"),
                                "rule": f"screener x{1/scale:g}", "field": k, "old": v,
                                "new": round(v / scale, 4)})
                    vals[k] = v / scale
                elif len(fit) != 1:
                    out.append({"idx": i, "symbol": r.get("symbol"), "quarter": r.get("quarter"),
                                "rule": "AMBIGUOUS-kept", "field": k, "old": v, "new": v})
        f = {"symbol": r.get("symbol"), "revenue_cr": vals["revenue_cr"],
             "pat_cr": vals["pat_cr"], "ebitda_cr": vals["ebitda_cr"],
             "pat_margin_pct": _f(r.get("pat_margin_pct")),
             "ebitda_margin_pct": _f(r.get("ebitda_margin_pct"))}
        before = {k: f[k] for k in ("pat_cr", "ebitda_cr")}
        FR.margin_check(f)
        for k in ("pat_cr", "ebitda_cr"):
            if before[k] is not None and f[k] != before[k]:
                out.append({"idx": i, "symbol": r.get("symbol"), "quarter": r.get("quarter"),
                            "rule": "margin", "field": k, "old": before[k], "new": round(f[k], 4)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="write the corrections")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    from _extractor_base import (get_drive, get_or_create_subfolder, find_file, download_bytes,
                                 save_parquet, acquire_lock, release_lock)
    d = get_drive()
    idx = get_or_create_subfolder(d, get_or_create_subfolder(d, os.environ["GDRIVE_FOLDER_ID"],
                                                             "company_repo"), "_index")
    if args.live and not acquire_lock(d, idx, _LOCK, "fix_results_units", wait_min=15):
        print("lock busy - nothing written")
        return 1
    try:
        g = pd.read_parquet(io.BytesIO(download_bytes(d, find_file(d, idx, NAME))))
        res = pd.read_parquet(io.BytesIO(download_bytes(d, find_file(d, idx, "results.parquet"))))
        fixes = plan_fixes(g, screener_revenue(res))
        print(f"{NAME}: {len(g)} rows -> {len(fixes)} field change(s) on "
              f"{len({x['idx'] for x in fixes})} row(s)")
        for x in fixes:
            print(f"  {str(x['symbol']):<12} {str(x['quarter']):<8} {x['rule']:<15} "
                  f"{x['field']:<10} {x['old']:>12,.2f} -> {x['new']:>12,.2f}")
        if not args.live:
            print("DRY RUN - nothing written. Re-run with --live to apply.")
            return 0
        for x in fixes:
            if x["rule"] == "AMBIGUOUS-kept":
                continue
            g.at[x["idx"], x["field"]] = f"{x['new']:.2f}"
        save_parquet(d, idx, NAME, g)
        print(f"wrote {NAME}")
        return 0
    finally:
        if args.live:
            release_lock(d, idx, _LOCK)


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    df = pd.DataFrame([
        {"isin": "B", "symbol": "BETA", "quarter": "Q1 FY27", "revenue_cr": "12713.21",
         "ebitda_cr": "2000", "pat_cr": "1500", "pat_margin_pct": None, "ebitda_margin_pct": None},
        {"isin": "T", "symbol": "TATVA", "quarter": "Q2 FY27", "revenue_cr": "217.32",
         "ebitda_cr": "39.74", "pat_cr": "153.87", "pat_margin_pct": "7.08%", "ebitda_margin_pct": "18.3%"},
        {"isin": "M", "symbol": "MARKSANS", "quarter": "Q1 FY27", "revenue_cr": "321.18",
         "ebitda_cr": None, "pat_cr": None, "pat_margin_pct": None, "ebitda_margin_pct": None}])
    scr = {("B", "Q1FY27"): 126.0, ("T", "Q2FY27"): 217.0, ("M", "Q1FY27"): 841.0}
    fx = plan_fixes(df, scr)
    beta = {x["field"]: x for x in fx if x["symbol"] == "BETA"}
    check("100x lakh row: revenue scaled", abs(beta["revenue_cr"]["new"] - 127.1321) < 0.01)
    check("BETA: PAT 1500 > revenue 127 unless scaled -> scaled to 15",
          abs(beta["pat_cr"]["new"] - 15.0) < 0.01)
    mixed = pd.DataFrame([{"isin": "B", "symbol": "BETA2", "quarter": "Q1 FY27",
                           "revenue_cr": "12713.21", "ebitda_cr": None, "pat_cr": "12.97",
                           "pat_margin_pct": None, "ebitda_margin_pct": None}])
    fx2 = plan_fixes(mixed, {("B", "Q1FY27"): 126.0})
    check("mixed units: PAT already in crore is NOT scaled (both plausible -> kept, listed)",
          any(x["field"] == "pat_cr" and x["rule"] == "AMBIGUOUS-kept" for x in fx2))
    check("margin rule fixes TATVA-style lone 10x PAT",
          any(x["symbol"] == "TATVA" and x["field"] == "pat_cr" for x in fx))
    check("0.38x (standalone vs consolidated) is NOT changed",
          not any(x["symbol"] == "MARKSANS" for x in fx))
    res = pd.DataFrame([{"isin": "Z", "metric": "Sales", "latest_q": "Sep 2026", "latest_val": 250.0}])
    check("the 250.0 artefact is never a reference", screener_revenue(res) == {})
    print(f"\nfix_results_units self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
