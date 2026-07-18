r"""
flag_growth_surge.py — results-season >100% PAT/EPS growth flagger (NO Gemini).

Classifies every company whose LATEST quarter PAT YoY >= 100% into:
  CONSISTENT  — >=3 of the last 4 quarters had PAT YoY >= 100%  (sustained compounder)
  EMERGING    — exactly 2 of the last 4 quarters                (streak forming)
  ONE_OFF     — only the latest quarter popped                  (single-quarter spike)
  TURNAROUND  — year-ago base Net Profit was a loss or tiny     (base effect; % misleading)
Companies below 100% are excluded entirely (NONE).

Inputs (all already on Drive, no schema change):
  _index/financials_derived.parquet  -> per-quarter pat_yoy_pct / pat_qoq_pct history
  _index/financials_3stmt.parquet    -> quarterly Net Profit levels (year-ago base guard)
  _index/results.parquet             -> latest-Q EPS YoY (dilution check) + first_seen_at
                                        (= when the result appeared -> 24h mail window)
  _index/company_universe.csv        -> isin -> name / bse_code
  pf_tracking|portfolio holdings     -> in_pf flag (load_portfolio_isins)

Output: _index/growth_surge.parquet (one row per surging company, current quarter)
plus a daily results-season email (toggle 'growth_surge') covering reporters in the
last --hours window. Quarter-wise list = the whole parquet; weekly = 7d window.

Usage:
    python scripts/flag_growth_surge.py --dry-run      # compute + preview html; no write/mail
    python scripts/flag_growth_surge.py                # write parquet + send mail (24h window)
    python scripts/flag_growth_surge.py --hours 168    # weekly roll-up mail
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, load_parquet, save_parquet,
                             load_portfolio_isins, log)
from mailer import send_email, load_mail_settings, esc
import gradation as G

EXPLOSIVE_THR = 100.0   # PAT YoY % that counts as a surge (gradation "exceptional")
LOW_BASE_CR = 2.0       # year-ago Net Profit below this (Cr) = low-base distortion
LOOKBACK_Q = 4          # window for the consistent-vs-one-off count
MAX_QTR_AGE_DAYS = 200  # latest quarter older than this = stale data, not this
                        # results season (drops e.g. a company whose last scrape
                        # ended Mar 2020) — current + previous quarter pass

OUT_NAME = "growth_surge.parquet"
OUT_COLS = ["isin", "symbol", "company_name", "quarter",
            "pat_yoy_pct", "pat_qoq_pct", "eps_yoy_pct",
            "n_surge_4q", "streak_len", "base_yearago_sign", "eps_confirms",
            "classification", "in_pf", "reported_at", "computed_at"]

DERIVED_COLS = ["isin", "symbol", "metric", "period", "period_type",
                "value", "unit", "scraped_at"]
FIN3_COLS = ["isin", "symbol", "statement", "line_item", "period", "period_type",
             "value", "basis", "qoq_pct", "yoy_pct", "scraped_at"]
RESULTS_COLS = ["slug", "isin", "company_name", "metric",
                "latest_q", "latest_val", "prev_q", "prev_val",
                "yearago_q", "yearago_val", "yoy_pct", "qoq_pct",
                "scraped_at", "first_seen_at"]

CLASS_ORDER = ["CONSISTENT", "EMERGING", "ONE_OFF", "TURNAROUND"]
CLASS_CHIP = {   # chip background / text colour per classification
    "CONSISTENT": ("#1a7a3a", "#fff"),
    "EMERGING":   ("#a5d6a7", "#1b3a24"),
    "ONE_OFF":    ("#ffb74d", "#4a2c00"),
    "TURNAROUND": ("#ffe0b2", "#5a4a2f"),
}


def _series_by_isin(df: pd.DataFrame, metric: str) -> dict[str, list[tuple[str, float]]]:
    """{isin: [(period, value), ...]} in emit (chronological) order for one metric."""
    out: dict[str, list] = {}
    sub = df[(df["metric"] == metric) & (df["period_type"] == "quarterly")]
    for _, r in sub.iterrows():
        iso = str(r.get("isin", "")).strip()
        v = r.get("value")
        if iso and v is not None and pd.notna(v):
            out.setdefault(iso, []).append((str(r.get("period", "")), float(v)))
    return out


def _np_series_by_isin(fin3: pd.DataFrame) -> dict[str, list[float]]:
    """{isin: [quarterly Net Profit levels]} in chronological order."""
    out: dict[str, list] = {}
    sub = fin3[(fin3["statement"] == "income")
               & (fin3["line_item"] == "Net Profit")
               & (fin3["period_type"] == "quarterly")]
    for _, r in sub.iterrows():
        iso = str(r.get("isin", "")).strip()
        v = r.get("value")
        if iso and v is not None and pd.notna(v):
            out.setdefault(iso, []).append(float(v))
    return out


def _eps_and_reported(results: pd.DataFrame) -> tuple[dict, dict, dict]:
    """From results.parquet: {isin: eps_yoy}, {isin: reported_at}, {isin: slug}."""
    eps_by, rep_by, slug_by = {}, {}, {}
    if results.empty:
        return eps_by, rep_by, slug_by
    r = results[results["isin"].astype(str).str.strip() != ""]
    for iso, grp in r.groupby(r["isin"].astype(str).str.strip()):
        seen = grp.get("first_seen_at")
        stamp = None
        if seen is not None:
            vals = [str(x) for x in seen.dropna() if str(x).strip()]
            stamp = max(vals) if vals else None
        if not stamp and "scraped_at" in grp:
            vals = [str(x) for x in grp["scraped_at"].dropna() if str(x).strip()]
            stamp = max(vals) if vals else None
        rep_by[iso] = stamp
        slug_by[iso] = str(grp["slug"].iloc[-1]) if "slug" in grp else ""
        eps_rows = grp[grp["metric"].astype(str).str.lower().str.contains("eps")]
        if not eps_rows.empty:
            v = pd.to_numeric(eps_rows["yoy_pct"], errors="coerce").dropna()
            if not v.empty:
                eps_by[iso] = float(v.iloc[-1])
    return eps_by, rep_by, slug_by


def classify(yoy_series: list[float], base_yearago: float | None) -> tuple[str, int, int, str]:
    """(classification, n_surge_4q, streak_len, base_sign) for one company.

    yoy_series is chronological; latest = last element. Caller guarantees
    the latest value >= EXPLOSIVE_THR.
    """
    last4 = yoy_series[-LOOKBACK_Q:]
    n_surge = sum(1 for v in last4 if v >= EXPLOSIVE_THR)
    streak = 0
    for v in reversed(yoy_series):
        if v >= EXPLOSIVE_THR:
            streak += 1
        else:
            break
    if base_yearago is not None and base_yearago <= 0:
        return "TURNAROUND", n_surge, streak, "loss"
    if base_yearago is not None and base_yearago < LOW_BASE_CR:
        return "TURNAROUND", n_surge, streak, "tiny"
    sign = "positive" if base_yearago is not None else "unknown"
    if n_surge >= 3:
        return "CONSISTENT", n_surge, streak, sign
    if n_surge == 2:
        return "EMERGING", n_surge, streak, sign
    return "ONE_OFF", n_surge, streak, sign


def build_rows(derived: pd.DataFrame, fin3: pd.DataFrame, results: pd.DataFrame,
               names: dict, pf: set[str], now: str) -> list[dict]:
    yoy_by = _series_by_isin(derived, "pat_yoy_pct")
    qoq_by = _series_by_isin(derived, "pat_qoq_pct")
    np_by = _np_series_by_isin(fin3)
    eps_by, rep_by, _slug = _eps_and_reported(results)
    sym_by = {}
    if not derived.empty:
        for _, r in derived.drop_duplicates("isin").iterrows():
            sym_by[str(r["isin"]).strip()] = str(r.get("symbol", "") or "").strip()

    qtr_cutoff = datetime.now() - timedelta(days=MAX_QTR_AGE_DAYS)
    rows, n_stale = [], 0
    for iso, series in yoy_by.items():
        q0_period, q0_yoy = series[-1]
        if q0_yoy < EXPLOSIVE_THR:
            continue
        q0_date = pd.to_datetime(q0_period, format="%b %Y", errors="coerce")
        if pd.isna(q0_date) or q0_date < qtr_cutoff:
            n_stale += 1     # data ends in an old quarter — not this results season
            continue
        yoy_vals = [v for _, v in series]
        nps = np_by.get(iso, [])
        base = nps[-5] if len(nps) >= 5 else None   # year-ago quarter level
        cls, n_surge, streak, base_sign = classify(yoy_vals, base)
        eps_yoy = eps_by.get(iso)
        qoq_series = qoq_by.get(iso, [])
        rows.append({
            "isin": iso, "symbol": sym_by.get(iso, ""),
            "company_name": names.get(iso, ""),
            "quarter": q0_period,
            "pat_yoy_pct": round(q0_yoy, 2),
            "pat_qoq_pct": round(qoq_series[-1][1], 2) if qoq_series else None,
            "eps_yoy_pct": round(eps_yoy, 2) if eps_yoy is not None else None,
            "n_surge_4q": n_surge, "streak_len": streak,
            "base_yearago_sign": base_sign,
            "eps_confirms": bool(eps_yoy is not None and eps_yoy >= EXPLOSIVE_THR),
            "classification": cls,
            "in_pf": iso in pf,
            "reported_at": rep_by.get(iso),
            "computed_at": now,
        })
    if n_stale:
        log(f"skipped {n_stale} surging row(s) with stale latest quarter "
            f"(older than {MAX_QTR_AGE_DAYS}d)")
    return rows


def _fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return f"{'+' if v >= 0 else ''}{v:,.0f}%"


def build_html(rows: list[dict], since: datetime, hours: float) -> str:
    td = ("padding:6px 10px;border:1px solid #ddd;font-size:13px;"
          "font-family:Arial,sans-serif;")
    th = td + "background:#34495e;color:#fff;text-align:left;"
    n_pf = sum(1 for r in rows if r["in_pf"])
    by_cls = {c: sum(1 for r in rows if r["classification"] == c) for c in CLASS_ORDER}
    summary = " · ".join(f"{by_cls[c]} {c.lower()}" for c in CLASS_ORDER if by_cls[c])
    out = [
        f"<div style='max-width:700px'>"
        f"<p style='font-family:Arial,sans-serif;font-size:13px'>"
        f"<b>{len(rows)} company(ies)</b> reported <b>&gt;{EXPLOSIVE_THR:.0f}% PAT YoY</b> "
        f"since {since.strftime('%d %b %Y %H:%M')} (last {hours:.0f}h)"
        + (f" — 💼 {n_pf} in PF" if n_pf else "") + f".<br>{summary}. "
        f"EPS✓ = EPS also grew &gt;100% (no dilution distortion). "
        f"TURNAROUND = year-ago base was a loss/tiny — treat the % with care.</p>",
        "<table style='border-collapse:collapse;width:100%'>",
        "<tr>" + "".join(f"<th style='{th}'>{c}</th>" for c in
                         ["#", "Company", "Qtr", "PAT YoY", "PAT QoQ", "EPS YoY",
                          "4Q hits", "Class"]) + "</tr>",
    ]
    order = {c: i for i, c in enumerate(CLASS_ORDER)}
    rows = sorted(rows, key=lambda r: (order.get(r["classification"], 9),
                                       not r["in_pf"],
                                       -(r["pat_yoy_pct"] or 0)))
    for i, r in enumerate(rows, 1):
        chip_bg, chip_fg = CLASS_CHIP.get(r["classification"], ("#eee", "#333"))
        yoy_css = G.cell_css(G.grade_growth(r["pat_yoy_pct"]))
        qoq_css = G.cell_css(G.grade_growth(r["pat_qoq_pct"]))
        sym = esc(r["symbol"] or r["isin"], 14)
        link = f"https://www.screener.in/company/{sym}/" if r["symbol"] else "#"
        pf_badge = "💼 " if r["in_pf"] else ""
        eps_txt = _fmt_pct(r["eps_yoy_pct"]) + (" ✓" if r["eps_confirms"] else "")
        out.append(
            "<tr>"
            f"<td style='{td}'>{i}</td>"
            f"<td style='{td}'>{pf_badge}<a href='{link}' style='color:#1a237e'>"
            f"<b>{sym}</b></a> · {esc(r['company_name'], 34)}</td>"
            f"<td style='{td}'>{esc(r['quarter'], 10)}</td>"
            f"<td style='{td}{yoy_css}'><b>{_fmt_pct(r['pat_yoy_pct'])}</b></td>"
            f"<td style='{td}{qoq_css}'>{_fmt_pct(r['pat_qoq_pct'])}</td>"
            f"<td style='{td}'>{eps_txt}</td>"
            f"<td style='{td}'>{r['n_surge_4q']}/4</td>"
            f"<td style='{td}'><span style='background:{chip_bg};color:{chip_fg};"
            f"padding:2px 8px;border-radius:9px;font-size:11px'>"
            f"{r['classification']}</span></td>"
            "</tr>")
    out.append("</table>"
               "<p style='font-size:11px;color:#999;font-family:Arial,sans-serif'>"
               "Source: Screener quarterly numbers (financials_3stmt / results). "
               "Toggle this mail in the app sidebar (📧 Email toggles).</p></div>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=24.0,
                    help="Mail window: reporters seen in the last N hours (default 24).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute + print + save preview html; no Drive write, no mail.")
    args = ap.parse_args()

    print("Growth-surge flagger — >100% PAT/EPS, consistent vs one-off")
    print("-" * 60)
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    derived = load_parquet(drive, index_id, "financials_derived.parquet", DERIVED_COLS)
    fin3 = load_parquet(drive, index_id, "financials_3stmt.parquet", FIN3_COLS)
    results = load_parquet(drive, index_id, "results.parquet", RESULTS_COLS)
    if derived.empty:
        sys.exit("financials_derived.parquet empty — run build_derived_metrics.py first.")
    log(f"inputs: derived={len(derived)} fin3={len(fin3)} results={len(results)}")

    names: dict[str, str] = {}
    uni_id = find_file(drive, index_id, "company_universe.csv")
    if uni_id:
        import io as _io
        uni = pd.read_csv(_io.BytesIO(download_bytes(drive, uni_id)))
        for _, r in uni.iterrows():
            iso = str(r.get("isin", "")).strip()
            if iso:
                names[iso] = str(r.get("name", "") or "").strip()

    pf = load_portfolio_isins(drive, root_id) or set()
    log(f"PF ISINs loaded: {len(pf)}")

    now = datetime.now().isoformat(timespec="seconds")
    rows = build_rows(derived, fin3, results, names, pf, now)
    out_df = pd.DataFrame(rows, columns=OUT_COLS)
    log(f"surging companies (latest-Q PAT YoY >= {EXPLOSIVE_THR:.0f}%): {len(out_df)}")
    if not out_df.empty:
        log("classification split:\n"
            + out_df["classification"].value_counts().to_string())
        log(f"in PF: {int(out_df['in_pf'].sum())} · "
            f"EPS-confirmed: {int(out_df['eps_confirms'].sum())}")

    # mail window: only companies whose results APPEARED in the last N hours
    since = datetime.now() - timedelta(hours=args.hours)
    fresh = [r for r in rows
             if r["reported_at"] and str(r["reported_at"]) >= since.isoformat(timespec="seconds")]
    log(f"fresh reporters in last {args.hours:.0f}h: {len(fresh)}")

    if args.dry_run:
        show = out_df.sort_values("pat_yoy_pct", ascending=False).head(25)
        cols = ["symbol", "quarter", "pat_yoy_pct", "eps_yoy_pct",
                "n_surge_4q", "streak_len", "base_yearago_sign",
                "classification", "in_pf", "reported_at"]
        print(show[cols].to_string(index=False))
        html = build_html(fresh if fresh else rows, since, args.hours)
        prev = Path(__file__).resolve().parent.parent / "growth_surge_preview.html"
        prev.write_text(html, encoding="utf-8")
        print(f"\nDRY RUN — preview saved to {prev.name} "
              f"({'fresh window' if fresh else 'ALL rows — no fresh reporter'}); "
              f"no Drive write, no mail.")
        return

    save_parquet(drive, index_id, OUT_NAME, out_df)
    log(f"wrote _index/{OUT_NAME} ({len(out_df)} rows)")

    if not fresh:
        log("no surging reporter in window — no mail.")
        return
    if not load_mail_settings(drive, index_id).get("growth_surge", True):
        log("growth_surge mail toggled OFF — skipped.")
        return
    n_pf = sum(1 for r in fresh if r["in_pf"])
    subject = (f"💥 Results surge — {len(fresh)} cos >100% PAT YoY "
               f"(last {args.hours:.0f}h)" + (f" · 💼{n_pf} PF" if n_pf else ""))
    sent = send_email(subject, build_html(fresh, since, args.hours))
    # ascii-only in log lines — local console is cp1252, emoji crash print()
    log(f"Email {'sent' if sent else 'FAILED'}: "
        f"{subject.encode('ascii', 'ignore').decode().strip()}")


if __name__ == "__main__":
    main()
