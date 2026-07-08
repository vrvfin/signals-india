r"""
build_pead_flags.py — Phase 3 / T2.5 producer (NO Gemini, NO scraping).

The free-tier "earnings surprise": compare what management GUIDED
(guidance_tracker.parquet from concalls; ppt_guidance.parquet from presentations;
ar_guidance.parquet from annual reports — user 2026-07-08) against the ACTUAL
reported number (financials_3stmt.parquet, kept fresh daily by
backfill_results_3stmt --incremental)
→ a rule-based BEAT / INLINE / MISS flag per company×metric×horizon.

Output company_repo/_index/pead_flags.parquet (frozen schema + additive col):
  isin, symbol, quarter, metric, guided_value, actual_value, delta_pct, verdict,
  as_of, guidance_source (concall|presentation|annual_report)
  (`quarter` holds the compared horizon, e.g. "FY26"; verdict band ±2, like GF_TRACK.)

Also emits the UNIFIED guidance-vs-actual view (user 2026-07-08):
  _index/guidance_vs_actual.parquet — pead_flags rows + GF_TRACK
  (mgmt_credibility) verdicts in one queryable table:
  isin, symbol, period, metric, guided, actual, delta, verdict, source,
  cred_score, cred_pattern, as_of

Pure transform over parquets — feeds strategy_pead / a watchlist / the daily mail.

Usage:
    python scripts/build_pead_flags.py --dry-run
    python scripts/build_pead_flags.py --names TCS,INFY
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder,
                             load_parquet, save_parquet, log)

PEAD_COLS = ["isin", "symbol", "quarter", "metric", "guided_value",
             "actual_value", "delta_pct", "verdict", "as_of",
             "guidance_source"]   # additive col (readers get None on old rows)
OUT_NAME = "pead_flags.parquet"
GVA_NAME = "guidance_vs_actual.parquet"   # unified view (user 2026-07-08)
GUIDANCE_NAME = "guidance_tracker.parquet"
FIN3_NAME = "financials_3stmt.parquet"
BAND = 2.0   # ±2 (% for abs, pp for growth/margin) — matches GF_TRACK

GUIDANCE_COLS = ["isin", "symbol", "company_name", "quarter", "metric",
                 "guidance_type", "horizon_fy", "value", "unit", "cagr_pct",
                 "notes", "processed_at", "source_doc_id"]
PPT_G_COLS = ["isin", "symbol", "company_name", "quarter", "metric",
              "guidance_type", "horizon", "value", "unit", "notes",
              "processed_at", "source_doc_id"]
AR_G_COLS = ["isin", "symbol", "company_name", "fy_year", "metric",
             "guidance_type", "horizon_fy", "value", "unit", "cagr_pct",
             "notes", "processed_at", "source_doc_id"]
MGMT_CRED_COLS = ["isin", "symbol", "company_name", "quarter",
                  "qtr_guided", "metric", "guidance_given", "target_period",
                  "actual_delivered", "delta", "verdict",
                  "cred_score", "pattern", "strongest_area", "recurring_miss",
                  "processed_at", "source_doc_id"]
GVA_COLS = ["isin", "symbol", "period", "metric", "guided", "actual",
            "delta", "verdict", "source", "cred_score", "cred_pattern", "as_of"]
FIN3_COLS = ["isin", "symbol", "statement", "line_item", "period", "period_type",
             "value", "basis", "qoq_pct", "yoy_pct", "scraped_at"]

# guidance metric (lowercased contains) -> (financials line_item, kind)
#   kind: "abs" compares ₹Cr levels; "growth" compares YoY%; "margin" compares %level
METRIC_MAP = [
    ("margin",  ("OPM %", "margin")),
    ("opm",     ("OPM %", "margin")),
    ("revenue", ("Sales", "level")),
    ("sales",   ("Sales", "level")),
    ("ebitda",  ("Operating Profit", "level")),
    ("operating profit", ("Operating Profit", "level")),
    ("pat",     ("Net Profit", "level")),
    ("net profit", ("Net Profit", "level")),
    ("profit",  ("Net Profit", "level")),
    ("eps",     ("EPS", "level")),
]


def _map_metric(metric_raw: str):
    low = str(metric_raw).lower().strip()
    for token, mapped in METRIC_MAP:
        if token in low:
            return mapped
    return (None, None)


def _to_float(s):
    """Single number, or midpoint of a 'a-b' range; strip %/₹/Cr/commas."""
    if s is None:
        return None
    txt = str(s)
    nums = re.findall(r"-?\d+(?:\.\d+)?", txt.replace(",", ""))
    if not nums:
        return None
    vals = [float(n) for n in nums[:2]]
    return round(sum(vals) / len(vals), 4)


def _fy_to_period(horizon: str):
    """'FY26'/'FY2026' -> 'Mar 2026' (India FY ends March)."""
    m = re.search(r"(\d{2,4})", str(horizon))
    if not m:
        return None
    yy = m.group(1)
    year = int(yy) if len(yy) == 4 else 2000 + int(yy)
    return f"Mar {year}"


def _resolve_horizon(horizon, quarter):
    """Resolve a guidance horizon to an FY label (user 2026-07-08).

    'FY26' passes through. RELATIVE horizons ('1Y'/'2Y'/'3Y' — the majority of
    guidance_tracker rows) resolve against the quarter the guidance was given
    in: 'Q3 FY25' + 1Y -> FY26 (base FY + n). NEXT_QTR / '3Y+' / blank -> None
    (not annual-grain comparable). Without this, relative rows never matched an
    actual and pead_flags stayed empty."""
    h = str(horizon or "").strip().upper()
    if re.fullmatch(r"FY\s?\d{2,4}", h):
        return h.replace(" ", "")
    m = re.fullmatch(r"([123])\s?Y", h)
    if not m:
        return None
    qm = re.search(r"FY\s?(\d{2,4})", str(quarter or "").upper())
    if not qm:
        return None
    return f"FY{int(qm.group(1)) + int(m.group(1))}"


def _verdict(delta):
    if delta is None:
        return "NA"
    if delta > BAND:
        return "BEAT"
    if delta < -BAND:
        return "MISS"
    return "INLINE"


def compute_flags(g_df: pd.DataFrame, fin_df: pd.DataFrame, now: str) -> list[dict]:
    """Pure comparator — testable offline. One flag row per matchable guidance row."""
    if g_df.empty or fin_df.empty:
        return []
    ann = fin_df[(fin_df["statement"] == "income")
                 & (fin_df["period_type"] == "annual")].copy()
    out: list[dict] = []
    for _, g in g_df.iterrows():
        isin = str(g.get("isin") or "")
        symbol = str(g.get("symbol") or "")
        li, kind = _map_metric(g.get("metric"))
        if not li:
            continue
        fy = _resolve_horizon(g.get("horizon_fy"), g.get("quarter"))
        period = _fy_to_period(fy) if fy else None
        gval = _to_float(g.get("value"))
        if not period or gval is None:
            continue
        unit = str(g.get("unit") or "").strip()
        # A %-unit guidance on a level metric = a GROWTH guidance.
        if kind == "level" and "%" in unit:
            kind = "growth"
        sub = ann[(ann["isin"].astype(str) == isin)
                  & (ann["line_item"] == li)
                  & (ann["period"].astype(str) == period)]
        if sub.empty:
            continue   # horizon not yet reported / no actual — skip (v1)
        row = sub.iloc[-1]
        if kind == "level":
            actual = row.get("value")
            delta = ((actual - gval) / abs(gval) * 100) if (actual is not None and gval) else None
        elif kind == "growth":
            actual = row.get("yoy_pct")           # actual YoY %
            delta = (actual - gval) if actual is not None else None   # pp gap
        else:                                     # margin
            actual = row.get("value")             # OPM % level
            delta = (actual - gval) if actual is not None else None   # pp gap
        out.append({
            "isin": isin, "symbol": symbol, "quarter": fy,   # resolved FY horizon
            "metric": str(g.get("metric") or ""),
            "guided_value": gval,
            "actual_value": round(float(actual), 4) if actual is not None else None,
            "delta_pct": round(float(delta), 2) if delta is not None else None,
            "verdict": _verdict(delta), "as_of": now,
            "guidance_source": str(g.get("guidance_source") or "concall"),
        })
    return out


def merge_guidance_sources(g_df: pd.DataFrame, ppt_df: pd.DataFrame,
                           ar_df: pd.DataFrame) -> pd.DataFrame:
    """Concat concall + presentation + AR guidance into one frame with a
    guidance_source tag, normalised to GUIDANCE_COLS shape (user 2026-07-08).
    Presentation rows: horizon -> horizon_fy, no cagr_pct. AR rows are already
    concall-shaped at FY grain. Non-FY horizons simply find no actual and skip."""
    frames = []
    if not g_df.empty:
        g = g_df.copy()
        g["guidance_source"] = "concall"
        frames.append(g)
    if not ppt_df.empty:
        p = ppt_df.copy()
        p["horizon_fy"] = p.get("horizon")
        p["cagr_pct"] = None
        p["guidance_source"] = "presentation"
        frames.append(p)
    if not ar_df.empty:
        a = ar_df.copy()
        a["quarter"] = a.get("fy_year")   # resolver base for relative horizons
        a["guidance_source"] = "annual_report"
        frames.append(a)
    if not frames:
        return pd.DataFrame(columns=GUIDANCE_COLS + ["guidance_source"])
    return pd.concat(frames, ignore_index=True, sort=False)


def build_unified_view(flags_df: pd.DataFrame, cred_df: pd.DataFrame,
                       now: str) -> pd.DataFrame:
    """guidance_vs_actual.parquet — ONE queryable table of promised-vs-delivered:
    rule-based pead_flags rows + Gemini GF_TRACK (mgmt_credibility) verdicts."""
    rows = []
    for _, f in flags_df.iterrows():
        rows.append({
            "isin": f.get("isin"), "symbol": f.get("symbol"),
            "period": f.get("quarter"), "metric": f.get("metric"),
            "guided": str(f.get("guided_value")),
            "actual": str(f.get("actual_value")),
            "delta": f.get("delta_pct"), "verdict": f.get("verdict"),
            "source": f"pead_{f.get('guidance_source') or 'concall'}",
            "cred_score": None, "cred_pattern": None, "as_of": now,
        })
    for _, c in cred_df.iterrows():
        rows.append({
            "isin": c.get("isin"), "symbol": c.get("symbol"),
            "period": c.get("target_period"), "metric": c.get("metric"),
            "guided": str(c.get("guidance_given") or ""),
            "actual": str(c.get("actual_delivered") or ""),
            "delta": None, "verdict": c.get("verdict"),
            "source": "gf_track",
            "cred_score": c.get("cred_score"), "cred_pattern": c.get("pattern"),
            "as_of": now,
        })
    return pd.DataFrame(rows, columns=GVA_COLS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="", help="Comma list of symbols to limit to.")
    ap.add_argument("--dry-run", action="store_true", help="Compute + print; no Drive write.")
    args = ap.parse_args()

    print("Phase 3 / T2.5 — PEAD surprise flags (guidance vs actual)")
    print("-" * 60)
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    g_df = load_parquet(drive, index_id, GUIDANCE_NAME, GUIDANCE_COLS)
    fin_df = load_parquet(drive, index_id, FIN3_NAME, FIN3_COLS)
    ppt_df = load_parquet(drive, index_id, "ppt_guidance.parquet", PPT_G_COLS)
    ar_df = load_parquet(drive, index_id, "ar_guidance.parquet", AR_G_COLS)
    cred_df = load_parquet(drive, index_id, "mgmt_credibility.parquet", MGMT_CRED_COLS)
    if g_df.empty:
        sys.exit("guidance_tracker.parquet empty — need concall guidance first.")
    if fin_df.empty:
        sys.exit("financials_3stmt.parquet empty — run backfill_results_3stmt.py first.")
    log(f"guidance rows: concall={len(g_df)} presentation={len(ppt_df)} "
        f"annual_report={len(ar_df)} · gf_track={len(cred_df)}")

    g_all = merge_guidance_sources(g_df, ppt_df, ar_df)

    names = {s.strip().upper() for s in args.names.split(",") if s.strip()}
    if names:
        g_all = g_all[g_all["symbol"].astype(str).str.upper().isin(names)]
        cred_df = cred_df[cred_df["symbol"].astype(str).str.upper().isin(names)]

    now = datetime.now().isoformat(timespec="seconds")
    flags = compute_flags(g_all, fin_df, now)
    out_df = pd.DataFrame(flags, columns=PEAD_COLS)
    # same guidance re-stated across concalls/supersedes -> one flag is enough
    out_df = out_df.drop_duplicates(
        subset=["isin", "quarter", "metric", "guided_value"]).reset_index(drop=True)
    gva_df = build_unified_view(out_df, cred_df, now)

    if args.dry_run:
        print(f"DRY RUN — {len(out_df)} flag(s). Sample:")
        print(out_df.head(30).to_string(index=False))
        if not out_df.empty:
            print("\nverdict counts:\n" + out_df["verdict"].value_counts().to_string())
            print("\nby guidance_source:\n"
                  + out_df["guidance_source"].value_counts().to_string())
        print(f"\nunified guidance_vs_actual rows: {len(gva_df)}")
        if not gva_df.empty:
            print(gva_df["source"].value_counts().to_string())
        return

    save_parquet(drive, index_id, OUT_NAME, out_df)
    print(f"Flags written: {len(out_df)}  →  company_repo/_index/{OUT_NAME}")
    if not out_df.empty:
        print(out_df["verdict"].value_counts().to_string())
    save_parquet(drive, index_id, GVA_NAME, gva_df)
    print(f"Unified view: {len(gva_df)} rows  →  company_repo/_index/{GVA_NAME}")


if __name__ == "__main__":
    main()
