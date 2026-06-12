r"""
run_growth_guidance_mail.py — daily "high-growth guidance" mail (NO Gemini).

Flags companies whose FRESH concall (read by Phase 2 in the last 24h) guides
max percentage > 30% — scanning BOTH guidance sources and merging:

  Table_A (guidance_tracker.parquet) — structured rows; only %-bearing rows
      count (unit '%', '%' in value, or cagr_pct present) so absolute ₹cr
      guidance never reads as a percentage. Metrics: revenue/ebitda/pat/
      volume/capacity (margin excluded by the metric set).
  GF1 (gf1_guidance_statements.parquet) — raw forward statements, catches
      what the Table_A parser missed. margin/utilization metric types
      excluded (%-LEVELS, not growth; user rule 2026-06-11).

Phase 2-only filter: source_doc_id joined to processing_queue.parquet; docs
with source=="backfill" (old-quarter fetches) are EXCLUDED — same convention
as guidance_digest_email.py.

Toggle: 'growth_guidance' in mail_settings.json (app sidebar / toggle_mail.bat).
Runs in pead.yml (20:00 IST) after the PEAD mails; manual: growth_mail.bat.

Usage:
    python scripts/run_growth_guidance_mail.py --dry-run
    python scripts/run_growth_guidance_mail.py --hours 24 --min-growth 30
"""
from __future__ import annotations

import argparse
import html as html_mod
import io
import os
import re
import sys
from datetime import date, datetime, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, log)
from guidance_digest_email import _to_pct          # Table_A % parser (midpoint)
from mailer import send_email, load_mail_settings

# %-levels, not growth — excluded from the >30% rule by design.
EXCLUDED_METRIC_TYPES = ("margin", "utilization")
# Table_A growth metrics (margin excluded by the set itself).
TABLE_A_METRICS = {"revenue", "ebitda", "pat", "volume", "capacity"}

_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _read_parquet(drive, index_id, name) -> pd.DataFrame:
    fid = find_file(drive, index_id, name)
    if not fid:
        return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception as e:
        log(f"could not read {name} ({str(e)[:60]})")
        return pd.DataFrame()


def max_pct(*texts) -> float | None:
    """Largest N from 'N%' patterns across the texts ('25-30%' -> 30.0)."""
    vals = [float(m.group(1)) for t in texts if t is not None
            for m in _PCT_RE.finditer(str(t))]
    return max(vals) if vals else None


def _fresh_phase2(df: pd.DataFrame, queue: pd.DataFrame,
                  hours: int) -> pd.DataFrame:
    """Rows processed in the window, minus docs FETCHED by the backfill
    (source=="backfill" — same convention as guidance_digest_email.py)."""
    if df.empty or "processed_at" not in df.columns:
        return pd.DataFrame()
    g = df.copy()
    g["_ts"] = pd.to_datetime(g["processed_at"], errors="coerce")
    g = g[g["_ts"] >= datetime.now() - timedelta(hours=hours)]
    if g.empty:
        return g
    if not queue.empty and "source" in queue.columns and "doc_id" in queue.columns:
        backfill_ids = set(
            queue.loc[queue["source"].astype(str) == "backfill", "doc_id"]
            .astype(str))
        if backfill_ids:
            g = g[~g["source_doc_id"].astype(str).isin(backfill_ids)]
    return g


def _flag_companies(g: pd.DataFrame, min_growth: float, src: str,
                    trigger_of) -> list[dict]:
    """Group >min_growth rows into one dict per company (shared by both sources)."""
    out = []
    for (isin, sym), comp in g.groupby([g["isin"].astype(str),
                                        g["symbol"].astype(str)]):
        peak = comp["_pct"].max()
        if peak <= min_growth:
            continue
        top = comp.sort_values("_pct", ascending=False).head(3)
        out.append({
            "isin": isin, "symbol": sym,
            "company_name": str(comp["company_name"].iloc[0]),
            "quarter": str(comp["quarter"].iloc[0]),
            "max_pct": peak,
            "triggers": [dict(trigger_of(r), src=src) for _, r in top.iterrows()],
        })
    return out


def find_high_growth(gf1: pd.DataFrame, queue: pd.DataFrame,
                     hours: int, min_growth: float) -> list[dict]:
    """GF1 source: raw forward statements, max % of value/range."""
    g = _fresh_phase2(gf1, queue, hours)
    if g.empty:
        return []
    mt = g["metric_type"].astype(str).str.lower()
    g = g[~mt.str.contains("|".join(EXCLUDED_METRIC_TYPES), na=False)]
    if g.empty:
        return []
    g["_pct"] = [max_pct(nv, rv) for nv, rv in
                 zip(g.get("numeric_value"), g.get("range_val"))]
    g = g.dropna(subset=["_pct"])
    return _flag_companies(g, min_growth, "GF1", lambda r: {
        "metric": str(r.get("metric_type", "")),
        "value": str(r.get("numeric_value", "")),
        "range": str(r.get("range_val", "")),
        "timeframe": str(r.get("timeframe", "")),
        "statement": str(r.get("exact_statement", ""))})


def find_high_growth_table_a(gt: pd.DataFrame, queue: pd.DataFrame,
                             hours: int, min_growth: float) -> list[dict]:
    """Table_A source: structured guidance_tracker rows. Only %-bearing rows
    count — bare numbers are absolute ₹cr guidance, never a growth %."""
    g = _fresh_phase2(gt, queue, hours)
    if g.empty:
        return []
    g = g[g["metric"].astype(str).str.lower().str.strip().isin(TABLE_A_METRICS)]
    if g.empty:
        return []
    has_cagr = pd.to_numeric(g.get("cagr_pct"), errors="coerce").notna()
    pct_like = (g["value"].astype(str).str.contains("%")
                | (g["unit"].astype(str).str.strip() == "%") | has_cagr)
    g = g[pct_like]
    if g.empty:
        return []
    g["_pct"] = [_to_pct(v, c) for v, c in zip(g["value"], g["cagr_pct"])]
    g = g.dropna(subset=["_pct"])
    return _flag_companies(g, min_growth, "Table_A", lambda r: {
        "metric": str(r.get("metric", "")),
        "value": str(r.get("value", "")),
        "range": "",
        "timeframe": str(r.get("horizon_fy", "")),
        "statement": str(r.get("notes", ""))})


def merge_flags(*flag_lists: list[dict]) -> list[dict]:
    """Union by symbol; max of max_pct, triggers concatenated (Table_A first)."""
    by: dict[str, dict] = {}
    for flags in flag_lists:
        for f in flags:
            e = by.get(f["symbol"])
            if e is None:
                by[f["symbol"]] = f
            else:
                e["max_pct"] = max(e["max_pct"], f["max_pct"])
                e["triggers"] = (e["triggers"] + f["triggers"])[:4]
    out = list(by.values())
    out.sort(key=lambda x: -x["max_pct"])
    return out


from mailer import esc as _esc_base


def _esc(s, n=160) -> str:
    return _esc_base(s, n)


def _mail_html(flagged: list[dict], hours: int, min_growth: float) -> str:
    rows = []
    for f in flagged:
        trig = "<br>".join(
            f"<b>{_esc(t['metric'], 30)}</b> "
            f"<span style='color:#999;font-size:11px'>[{_esc(t.get('src', ''), 8)}]"
            f"</span>: "
            f"{_esc(t['value'] if t['value'] not in ('NA', 'nan', '') else t['range'], 30)}"
            f" ({_esc(t['timeframe'], 30)})"
            f"<br><i style='color:#777'>{_esc(t['statement'])}</i>"
            for t in f["triggers"])
        rows.append(f"<tr><td><b>{_esc(f['symbol'])}</b></td>"
                    f"<td>{_esc(f['company_name'], 34)}</td>"
                    f"<td>{_esc(f['quarter'], 12)}</td>"
                    f"<td align=center><b>{f['max_pct']:.0f}%</b></td>"
                    f"<td>{trig}</td></tr>")
    return (f"<p><b>{len(flagged)} company(ies)</b> guided &gt;{min_growth:.0f}% "
            f"growth in concalls read in the last {hours}h (Phase 2 only):</p>"
            f"<table border=1 cellpadding=5 cellspacing=0>"
            f"<tr><th>Symbol</th><th>Company</th><th>Qtr</th><th>Max %</th>"
            f"<th>Guidance (top 3 rows)</th></tr>{''.join(rows)}</table>"
            f"<p style='font-size:11px;color:#999'>Sources: Table_A structured "
            f"guidance (%-bearing rows only) + GF1 forward-looking statements, "
            f"merged. Margin/utilization excluded (%-levels, not growth). "
            f"Backfill-fetched (old) concalls excluded. "
            f"Toggle this mail in the app sidebar (📧 Email toggles).</p>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24,
                    help="Lookback over GF1 processed_at (default 24).")
    ap.add_argument("--min-growth", type=float, default=30.0,
                    help="Flag when max guided %% exceeds this (default 30).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be mailed; do not send.")
    args = ap.parse_args()

    drive = get_drive()
    repo_id = get_or_create_subfolder(drive, os.environ["GDRIVE_FOLDER_ID"],
                                      "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")
    gf1 = _read_parquet(drive, index_id, "gf1_guidance_statements.parquet")
    gt = _read_parquet(drive, index_id, "guidance_tracker.parquet")
    queue = _read_parquet(drive, index_id, "processing_queue.parquet")
    flagged = merge_flags(
        find_high_growth_table_a(gt, queue, args.hours, args.min_growth),
        find_high_growth(gf1, queue, args.hours, args.min_growth))
    log(f"high-growth guidance: {len(flagged)} company(ies) "
        f"(>{args.min_growth:.0f}%, last {args.hours}h, phase 2 only, "
        f"Table_A + GF1 merged)")

    if args.dry_run:
        for f in flagged:
            t0 = f["triggers"][0]
            log(f"  {f['symbol']:12s} {f['quarter']:10s} max={f['max_pct']:.0f}% "
                f"({t0['metric']} [{t0.get('src', '')}])")
        return
    if not flagged:
        log("nothing to mail.")
        return
    if not load_mail_settings(drive, index_id).get("growth_guidance", True):
        log("growth_guidance mail toggled OFF — skipped.")
        return
    send_email(f"🚀 High-growth guidance — {len(flagged)} co(s) — {date.today()}",
               _mail_html(flagged, args.hours, args.min_growth))


if __name__ == "__main__":
    main()
