"""
Daily concall-guidance digest email.

Takes the last 24 hours of LIVE Phase 2 concall extractions (backfill rows are
excluded via the queue's source tag) and emails one table row per company that
gave quantified forward guidance on Revenue / EBITDA / PAT / Volume / Capacity.
Companies with no quantified guidance are skipped entirely.

Table: one column per metric; each cell shows the company's best guidance for
that metric with its horizon tag, colour-coded by growth %:
    > 50%  green  ·  > 20%  orange  ·  > 0%  yellow
Rows are sorted by the row maximum (best guidance anywhere in the row), desc.

Runs as the third email in pead.yml (daily ~20:00 IST cron). The 24-hour
window (not calendar-date) guarantees nothing is missed between sends.

Usage:
    python scripts/guidance_digest_email.py            # send (needs GMAIL_* env)
    python scripts/guidance_digest_email.py --dry-run  # print + save preview html
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _extractor_base import (get_drive, get_or_create_subfolder,  # noqa: E402
                             load_parquet, log)
from mailer import send_email, load_mail_settings  # noqa: E402

GUIDANCE_COLS = ["isin", "symbol", "company_name", "quarter", "metric",
                 "guidance_type", "horizon_fy", "value", "unit", "cagr_pct",
                 "notes", "processed_at", "source_doc_id"]
QUEUE_COLS = ["doc_id", "source"]

# Metrics in the email, in column order (margin deliberately excluded).
METRICS = ["revenue", "ebitda", "pat", "volume", "capacity"]
METRIC_LABEL = {"revenue": "Revenue", "ebitda": "EBITDA", "pat": "PAT",
                "volume": "Volume", "capacity": "Capacity"}
HORIZON_LABEL = {"NEXT_QTR": "next qtr", "1Y": "1yr", "2Y": "2yr",
                 "3Y": "3yr", "3Y+": ">3yr"}

# Colour bands (cell background) by guidance growth %.
GREEN, ORANGE, YELLOW = "#c8f0c8", "#ffd9a8", "#fdf6b2"


def _to_pct(raw, cagr) -> float | None:
    """Best-effort numeric % from a guidance cell. Ranges use the midpoint."""
    if cagr is not None and not pd.isna(cagr):
        return float(cagr)
    s = str(raw or "").strip()
    if not s or s.upper() == "NA":
        return None
    s = re.sub(r"[,%₹$]", "", s).replace("–", "-")
    m = re.match(r"^\+?(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    try:
        return float(s.lstrip("+"))
    except ValueError:
        return None


def _cell_color(pct: float | None) -> str:
    if pct is None:
        return ""
    if pct > 50:
        return GREEN
    if pct > 20:
        return ORANGE
    if pct > 0:
        return YELLOW
    return ""


def build_rows(g: pd.DataFrame) -> list[dict]:
    """One dict per company: best guidance per metric + row-max sort key."""
    rows = []
    for (isin, symbol), grp in g.groupby(["isin", "symbol"]):
        cells: dict[str, dict] = {}
        for metric in METRICS:
            mg = grp[grp["metric"] == metric]
            best_pct, best_disp = None, ""
            for _, r in mg.iterrows():
                pct = _to_pct(r["value"], r["cagr_pct"])
                if pct is None:
                    continue
                if best_pct is None or pct > best_pct:
                    hz = HORIZON_LABEL.get(str(r["horizon_fy"]), str(r["horizon_fy"]))
                    disp = str(r["value"]).strip()
                    if not disp.endswith("%"):
                        disp += "%"
                    best_pct, best_disp = pct, f"{disp} ({hz})"
            if best_pct is not None:
                cells[metric] = {"pct": best_pct, "disp": best_disp}
        if not cells:
            continue  # no quantified guidance on the 5 metrics — skip company
        row_max = max(c["pct"] for c in cells.values())
        rows.append({
            "symbol": symbol,
            "company_name": str(grp["company_name"].iloc[0]),
            "quarter": str(grp["quarter"].iloc[0]),
            "cells": cells,
            "row_max": row_max,
        })
    rows.sort(key=lambda r: r["row_max"], reverse=True)
    return rows


HIGH_GROWTH_PCT = 30.0   # 🚀 marker threshold (user rule 2026-06-11)


def build_html(rows: list[dict], since: datetime,
               gf1_extras: list[dict] | None = None) -> str:
    td = ("padding:6px 10px;border:1px solid #ddd;font-size:13px;"
          "font-family:Arial,sans-serif;")
    th = td + "background:#34495e;color:#fff;text-align:left;"
    out = [
        f"<p style='font-family:Arial,sans-serif;font-size:13px'>"
        f"Concalls processed since {since.strftime('%d %b %Y %H:%M IST')} — "
        f"{len(rows)} company(ies) with quantified guidance. "
        f"Cell colour: <span style='background:{GREEN}'>&gt;50%</span> · "
        f"<span style='background:{ORANGE}'>&gt;20%</span> · "
        f"<span style='background:{YELLOW}'>&gt;0%</span>. "
        f"🚀 = guides &gt;{HIGH_GROWTH_PCT:.0f}% growth. "
        f"Sorted by best guidance in the row.</p>",
        "<table style='border-collapse:collapse'>",
        "<tr>" + "".join(
            f"<th style='{th}'>{c}</th>"
            for c in ["#", "Company", "Qtr"] + [METRIC_LABEL[m] for m in METRICS]
        ) + "</tr>",
    ]
    for i, r in enumerate(rows, 1):
        rocket = "🚀 " if r["row_max"] > HIGH_GROWTH_PCT else ""
        tds = [f"<td style='{td}'>{i}</td>",
               f"<td style='{td}'>{rocket}<b>{r['symbol']}</b> · {r['company_name']}</td>",
               f"<td style='{td}'>{r['quarter']}</td>"]
        for m in METRICS:
            c = r["cells"].get(m)
            bg = f"background:{_cell_color(c['pct'])};" if c else ""
            tds.append(f"<td style='{td}{bg}'>{c['disp'] if c else ''}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    if gf1_extras:
        out.append(
            f"<p style='font-family:Arial,sans-serif;font-size:13px'><b>🚀 GF1-only "
            f"high growth</b> — forward statements &gt;{HIGH_GROWTH_PCT:.0f}% the "
            f"structured table missed:</p><table style='border-collapse:collapse'>"
            f"<tr><th style='{th}'>Company</th><th style='{th}'>Max %</th>"
            f"<th style='{th}'>Statement</th></tr>")
        for f in gf1_extras[:15]:
            t0 = f["triggers"][0] if f["triggers"] else {}
            out.append(
                f"<tr><td style='{td}'><b>{f['symbol']}</b> · "
                f"{f['company_name'][:30]}</td>"
                f"<td style='{td}'><b>{f['max_pct']:.0f}%</b></td>"
                f"<td style='{td}'>{t0.get('metric', '')}: "
                f"<i>{str(t0.get('statement', ''))[:160]}</i></td></tr>")
        out.append("</table>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=24.0,
                    help="Lookback window in hours (default 24).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the table + save preview html; no email.")
    args = ap.parse_args()

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    g = load_parquet(drive, index_id, "guidance_tracker.parquet", GUIDANCE_COLS)
    if g.empty:
        log("guidance_tracker empty — nothing to send.")
        return

    since = datetime.now() - timedelta(hours=args.hours)
    g = g[g["processed_at"].astype(str) >= since.isoformat(timespec="seconds")]
    log(f"Guidance rows in last {args.hours:.0f}h: {len(g)}")

    # Exclude backfill docs — this digest is live Phase 2 only.
    queue = load_parquet(drive, index_id, "processing_queue.parquet", QUEUE_COLS)
    if not queue.empty and "source" in queue.columns:
        backfill_ids = set(
            queue.loc[queue["source"].astype(str) == "backfill", "doc_id"]
            .astype(str))
        before = len(g)
        g = g[~g["source_doc_id"].astype(str).isin(backfill_ids)]
        if before - len(g):
            log(f"Excluded {before - len(g)} backfill rows.")

    g = g[g["metric"].isin(METRICS)]
    rows = build_rows(g)

    # 🚀 GF1 supplement (merged mail, user 2026-06-12): high-growth forward
    # statements the Table_A parser missed. Import inside the function —
    # run_growth_guidance_mail imports _to_pct from THIS module at top level.
    gf1_extras: list[dict] = []
    try:
        from run_growth_guidance_mail import find_high_growth
        gf1 = load_parquet(drive, index_id, "gf1_guidance_statements.parquet",
                           ["isin", "symbol", "company_name", "quarter",
                            "metric_type", "numeric_value", "range_val",
                            "timeframe", "exact_statement", "processed_at",
                            "source_doc_id"])
        flags = find_high_growth(gf1, queue, int(args.hours), HIGH_GROWTH_PCT)
        in_table = {r["symbol"] for r in rows}
        gf1_extras = [f for f in flags if f["symbol"] not in in_table]
    except Exception as e:
        log(f"GF1 supplement failed ({str(e)[:60]}) — table only.")

    if not rows and not gf1_extras:
        log("No companies with quantified guidance in window — no email.")
        return

    n_rocket = sum(1 for r in rows if r["row_max"] > HIGH_GROWTH_PCT) + len(gf1_extras)
    html = build_html(rows, since, gf1_extras)
    subject = (f"🎯 Concall guidance — last {args.hours:.0f}h ({len(rows)} cos"
               + (f", 🚀{n_rocket} high-growth" if n_rocket else "") + ")")

    if args.dry_run:
        for i, r in enumerate(rows, 1):
            parts = [f"{METRIC_LABEL[m]}: {r['cells'][m]['disp']}"
                     for m in METRICS if m in r["cells"]]
            print(f"{i:>2}. {r['symbol']:<12} {r['quarter']:<8} "
                  f"max={r['row_max']:.1f}%  " + " | ".join(parts))
        prev = Path(__file__).resolve().parent.parent / "guidance_digest_preview.html"
        prev.write_text(html, encoding="utf-8")
        print(f"\nDRY RUN — preview saved to {prev}; no email sent.")
        return

    if not load_mail_settings(drive, index_id).get("guidance_digest", True):
        log("guidance_digest mail toggled OFF — skipped.")
        return
    sent = send_email(subject, html)
    log(f"Email {'sent' if sent else 'FAILED'}: {subject}")


if __name__ == "__main__":
    main()
