r"""
run_growth_guidance_mail.py — daily "high-growth guidance" mail (NO Gemini).

Flags companies whose FRESH concall (read by Phase 2 in the last 24h) carries
GF1 forward guidance with max percentage > 30% across the row set — excluding
margin/utilization metric types (those are %-LEVELS, not growth; user rule
2026-06-11).

Phase 2-only filter: GF1.source_doc_id is joined to processing_queue.parquet;
docs with a backfill_process_date were processed by the Phase 3 backfill
(old concalls) and are EXCLUDED — only live Phase 2 reads count as "fresh".

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
from mailer import send_email, load_mail_settings

# %-levels, not growth — excluded from the >30% rule by design.
EXCLUDED_METRIC_TYPES = ("margin", "utilization")

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


def find_high_growth(gf1: pd.DataFrame, queue: pd.DataFrame,
                     hours: int, min_growth: float) -> list[dict]:
    """One dict per company whose fresh Phase 2 concall guides >min_growth%."""
    if gf1.empty or "processed_at" not in gf1.columns:
        return []
    g = gf1.copy()
    g["_ts"] = pd.to_datetime(g["processed_at"], errors="coerce")
    g = g[g["_ts"] >= datetime.now() - timedelta(hours=hours)]
    if g.empty:
        return []

    # Phase 2 only: drop docs FETCHED by the backfill (old quarters) — same
    # source=="backfill" convention as guidance_digest_email.py.
    if not queue.empty and "source" in queue.columns and "doc_id" in queue.columns:
        backfill_ids = set(
            queue.loc[queue["source"].astype(str) == "backfill", "doc_id"]
            .astype(str))
        if backfill_ids:
            g = g[~g["source_doc_id"].astype(str).isin(backfill_ids)]
    if g.empty:
        return []

    mt = g["metric_type"].astype(str).str.lower()
    g = g[~mt.str.contains("|".join(EXCLUDED_METRIC_TYPES), na=False)]
    if g.empty:
        return []
    g["_pct"] = [max_pct(nv, rv) for nv, rv in
                 zip(g.get("numeric_value"), g.get("range_val"))]
    g = g.dropna(subset=["_pct"])

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
            "triggers": [
                {"metric": str(r.get("metric_type", "")),
                 "value": str(r.get("numeric_value", "")),
                 "range": str(r.get("range_val", "")),
                 "timeframe": str(r.get("timeframe", "")),
                 "statement": str(r.get("exact_statement", ""))}
                for _, r in top.iterrows()],
        })
    out.sort(key=lambda x: -x["max_pct"])
    return out


def _esc(s, n=160) -> str:
    return html_mod.escape(str(s)[:n])


def _mail_html(flagged: list[dict], hours: int, min_growth: float) -> str:
    rows = []
    for f in flagged:
        trig = "<br>".join(
            f"<b>{_esc(t['metric'], 30)}</b>: "
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
            f"<p style='font-size:11px;color:#999'>Source: GF1 forward-looking "
            f"statements. Margin/utilization metric types excluded (%-levels, "
            f"not growth). Backfill-processed (old) concalls excluded. "
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
    queue = _read_parquet(drive, index_id, "processing_queue.parquet")
    flagged = find_high_growth(gf1, queue, args.hours, args.min_growth)
    log(f"high-growth guidance: {len(flagged)} company(ies) "
        f"(>{args.min_growth:.0f}%, last {args.hours}h, phase 2 only)")

    if args.dry_run:
        for f in flagged:
            log(f"  {f['symbol']:12s} {f['quarter']:10s} max={f['max_pct']:.0f}% "
                f"({f['triggers'][0]['metric']})")
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
