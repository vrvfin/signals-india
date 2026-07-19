r"""
build_combined_strength.py — the 3-axis shortlist (user 2026-07-08; NO Gemini).

STRONG TECHNICAL  ∩  STRONG RESULTS  ∩  STRONG GUIDANCE — one ranked list.
  TECHNICAL: signals/aggregated/conviction.csv — >=2 strategies agree on a
             buy/add zone (aggregate_signals.py, refreshed daily).
  RESULTS:   growth_surge.parquet classification CONSISTENT/EMERGING (P1 flagger)
             OR screener_grades yoy_tier green (>=30% PAT YoY).
  GUIDANCE:  screener_grades guidance_tier green OR guidance CAGR >= 20%.
             (mgmt cred_score from guidance_vs_actual shown as context.)

Rank score = 0.4*tech composite + 0.3*results tier + 0.3*guidance tier
(tiers via gradation.TIER_RANK scaled to 0-100; CONSISTENT surge = top tier).

Outputs:
  signals/combined_strength/latest.csv     — DAILY list (today's intersection)
  signals/combined_strength/weekly.csv     — union of last 7 days (history)
  signals/combined_strength/quarterly.csv  — union of current Indian-FY quarter
  _index/combined_strength_history.parquet — dated rows powering weekly/quarterly
plus a daily email (toggle 'combined_strength') when the list is non-empty.

Runs nightly in t4_nightly.yml AFTER build_scorecard/build_screener_grades and
BEFORE run_pf_digest. Usage:
    python scripts/build_combined_strength.py --dry-run
    python scripts/build_combined_strength.py
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, load_parquet,
                             save_parquet, load_portfolio_isins, log)
from mailer import send_email, load_mail_settings, esc
from flag_growth_surge import _pat_from_results   # freshest-reported-PAT realign
import gradation as G

MIN_GUIDANCE_CAGR = 20.0        # forward CAGR that counts as strong guidance
MAX_GUIDANCE_CAGR = 200.0       # above this the "CAGR" is a misparsed absolute
                                # (e.g. "1,000 Cr" -> 1000) — not trustworthy
GUID_STALE_MAX_Q = 4            # latest guidance older than this many quarters =
                                # not "current" -> doesn't qualify (user 2026-07-18)
GROWTH_METRICS = ("revenue", "ebitda", "pat")   # % growth metrics only (exclude
                                # margin/capacity/utilisation LEVELS which aren't growth)
TECH_ZONES = ("buy", "add")     # conviction zones that count as strong technical
SURGE_STRONG = ("CONSISTENT", "EMERGING")
MAIL_TOP_N = 30                 # mail shows the top N; full list lives on Drive
_QUARTER_RE = re.compile(r"Q([1-4])\s*FY[\s'\-]*?(\d{2,4})", re.I)

HIST_NAME = "combined_strength_history.parquet"
HIST_COLS = ["isin", "symbol", "company_name", "as_of",
             "n_strategies", "composite_score", "strategies",
             "pat_yoy_pct", "pat_qtr", "surge_class", "yoy_tier",
             "guidance_cagr", "guidance_tier", "guid_from", "cred_score",
             "rank_score", "in_pf",
             # guidance provenance (user 2026-07-18): latest-quarter guidance from
             # GF1/GF2, timestamped + freshness + the source document it was pulled
             # from. Additive -> old rows read as None.
             "guid_metric", "guid_qtrs_old", "guid_stmt", "guid_track",
             "guid_src_date", "guid_src_type", "guid_src_title"]

SURGE_COLS = ["isin", "symbol", "company_name", "quarter",
              "pat_yoy_pct", "pat_qoq_pct", "eps_yoy_pct",
              "n_surge_4q", "streak_len", "base_yearago_sign", "eps_confirms",
              "classification", "in_pf", "reported_at", "computed_at"]
GRADES_COLS = ["isin", "symbol", "company_name",
               "yoy", "yoy_tier", "qoq", "qoq_tier",
               "guidance", "guidance_tier",
               "val_metric", "val_value", "val_tier",
               "cfo_ratio", "cfo_tier", "roe", "roe_tier",
               "green_count", "computed_at"]
GVA_COLS = ["isin", "symbol", "period", "metric", "guided", "actual",
            "delta", "verdict", "source", "cred_score", "cred_pattern", "as_of"]


def _folder(drive, parts: str) -> str:
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _read_csv(drive, folder_id: str, name: str) -> pd.DataFrame:
    fid = find_file(drive, folder_id, name)
    if not fid:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
    except Exception as e:
        log(f"  WARNING: could not read {name} ({str(e)[:60]})")
        return pd.DataFrame()


def _write_csv(drive, folder_id: str, name: str, df: pd.DataFrame) -> None:
    upload_bytes(drive, folder_id, name, df.to_csv(index=False).encode("utf-8"),
                 "text/csv", existing_id=find_file(drive, folder_id, name))


def _fy_quarter_start(d: date) -> date:
    """Start date of the Indian-FY quarter containing d (Apr/Jul/Oct/Jan)."""
    m = ((d.month - 1) // 3) * 3 + 1
    return date(d.year, m, 1)


def _tier_pts(tier: str) -> float:
    """gradation tier -> 0-100 points."""
    return max(0, G.TIER_RANK.get(tier, 0)) * 20.0


def _qkey(q) -> int | None:
    """'Q3 FY25' / 'Q1 FY2026' / "Q2 FY'26" -> sortable int (yy*4 + qtr).
    None when unparseable. String-sorting quarters is wrong ('Q1 FY25' < 'Q4 FY24'),
    so every 'latest quarter' decision goes through this."""
    m = _QUARTER_RE.search(str(q or ""))
    if not m:
        return None
    return (int(m.group(2)) % 100) * 4 + int(m.group(1))


def _cur_fy_qkey(d: date) -> int:
    """Today's Indian-FY quarter as the same yy*4+qtr key (Apr-Jun=Q1 ... Jan-Mar=Q4)."""
    if d.month >= 4:
        q, fy = (d.month - 4) // 3 + 1, d.year + 1
    else:
        q, fy = 4, d.year
    return (fy % 100) * 4 + q


def _fmt_date(s) -> str:
    """'2026-05-23' -> '23 May 2026' (blank when unparseable)."""
    try:
        return pd.to_datetime(s).strftime("%d %b %Y")
    except Exception:
        return str(s or "")[:10]


def _latest_guidance(gt_iso: pd.DataFrame, gf1_iso: pd.DataFrame,
                     cur_key: int, doc_by: dict | None = None) -> dict | None:
    """Most-RECENT quantified growth guidance (revenue/EBITDA/PAT) for one company.
    Replaces the old 'max cagr over ALL history' (which surfaced 5-year-old Q2 FY21
    numbers and single 200% misparses). Representative value = median of the newest
    quarter's revenue rows (else EBITDA, else PAT); attaches the verbatim GF1
    statement, how many quarters old it is, and the SOURCE DOCUMENT (date + type
    + title, resolved source_doc_id -> processing_queue). None if no clean
    growth guidance."""
    doc_by = doc_by or {}
    if gt_iso is None or gt_iso.empty:
        return None
    g = gt_iso.copy()
    g["metric_l"] = g["metric"].astype(str).str.strip().str.lower()
    g["cagr"] = pd.to_numeric(g["cagr_pct"], errors="coerce")
    g["qk"] = g["quarter"].map(_qkey)
    g = g[g["metric_l"].isin(GROWTH_METRICS) & g["cagr"].notna()
          & (g["cagr"] > 0) & (g["cagr"] <= MAX_GUIDANCE_CAGR) & g["qk"].notna()]
    if g.empty:
        return None
    latest_qk = int(g["qk"].max())
    qrows = g[g["qk"] == latest_qk]
    metric = cagr = None
    src_id = ""
    for m in GROWTH_METRICS:                     # prefer revenue, then EBITDA, then PAT
        sub = qrows[qrows["metric_l"] == m]
        if not sub.empty:
            metric, cagr = m, round(float(sub["cagr"].median()), 1)
            if "source_doc_id" in sub.columns:
                src_id = str(sub["source_doc_id"].iloc[0] or "")
            break
    quarter = str(qrows["quarter"].iloc[0])
    stmt = tf = ""
    if gf1_iso is not None and not gf1_iso.empty:
        f = gf1_iso[gf1_iso["quarter"].map(_qkey) == latest_qk]
        if not f.empty:
            fm = f[f["metric_type"].astype(str).str.lower().str.contains(metric, na=False)]
            pick = fm if not fm.empty else f
            pref = pick[pick["quantifiable"].astype(str).str.lower() == "yes"]
            row = (pref if not pref.empty else pick).iloc[0]
            stmt = str(row.get("exact_statement") or "").strip().strip('"')
            tf = str(row.get("timeframe") or "").strip()
            gid = str(row.get("source_doc_id") or "")
            if gid:
                src_id = gid              # the quote's own document wins
    doc = doc_by.get(src_id, {})
    return {"quarter": quarter, "cagr": cagr, "metric": metric,
            "qtrs_old": max(0, cur_key - latest_qk), "statement": stmt, "timeframe": tf,
            "src_date": _fmt_date(doc.get("date")) if doc.get("date") else "",
            "src_type": doc.get("type", ""), "src_title": doc.get("title", "")}


def _gf2_track(gf2_iso: pd.DataFrame) -> str:
    """Track record from GF2 historical guidance-vs-actual self-assessments (prefer
    ACTUAL outcomes, user 2026-07-18). Returns 'delivers'/'has missed'/'' ."""
    if gf2_iso is None or gf2_iso.empty or \
       "management_self_assessment" not in gf2_iso.columns:
        return ""
    txt = " ".join(str(x).lower() for x in
                   gf2_iso["management_self_assessment"].dropna().tolist()[-8:])
    pos = sum(w in txt for w in ("met", "achieved", "exceeded", "delivered",
                                 "in line", "on track", "surpass", "ahead of"))
    neg = sum(w in txt for w in ("missed", "below", "short of", "fell short",
                                 "did not", "lower than", "behind"))
    if pos and pos >= neg + 1:
        return "delivers"
    if neg and neg > pos:
        return "has missed"
    return ""


def build_list(conv: pd.DataFrame, surge: pd.DataFrame, grades: pd.DataFrame,
               gva: pd.DataFrame, sym_to_isin: dict, names: dict,
               pf: set[str], today: str,
               pat_qtr_by: dict | None = None,
               pat_fresh_by: dict | None = None,
               gt_by: dict | None = None, gf1_by: dict | None = None,
               gf2_by: dict | None = None, cur_key: int = 0,
               doc_by: dict | None = None) -> tuple[list[dict], int]:
    """Intersect the three axes. Returns (rows, n_unmapped_tech_symbols).
    pat_qtr_by:   isin -> quarter label the PAT YoY belongs to (financials_derived).
    pat_fresh_by: isin -> (quarter, yoy) from the freshest results.parquet row, used
                  to realign the PAT column when derived lags (user 2026-07-18).
    gt_by/gf1_by/gf2_by: per-isin guidance_tracker / GF1 / GF2 slices; guidance now
                  comes from the LATEST quarter (not max-over-history). cur_key: today's
                  FY-quarter key, for the staleness tag."""
    pat_qtr_by = pat_qtr_by or {}
    pat_fresh_by = pat_fresh_by or {}
    gt_by = gt_by or {}
    gf1_by = gf1_by or {}
    gf2_by = gf2_by or {}
    doc_by = doc_by or {}
    # --- axis 1: technical (symbol-keyed) ---
    tech: dict[str, dict] = {}
    if not conv.empty:
        c = conv[conv["zone_type"].astype(str).str.lower().isin(TECH_ZONES)]
        c = c[pd.to_numeric(c["n_strategies"], errors="coerce") >= 2]
        for _, r in c.iterrows():
            sym = str(r["symbol"]).strip().upper()
            cur = tech.get(sym)
            if cur is None or r["n_strategies"] > cur["n_strategies"]:
                tech[sym] = {"n_strategies": int(r["n_strategies"]),
                             "composite_score": float(r["composite_score"]),
                             "strategies": str(r.get("strategies", ""))}

    # --- axis 2: results (isin-keyed) ---
    surge_by = {}
    if not surge.empty:
        for _, r in surge.iterrows():
            surge_by[str(r["isin"]).strip()] = r
    grades_by = {}
    if not grades.empty:
        for _, r in grades.iterrows():
            grades_by[str(r["isin"]).strip()] = r

    # --- axis 3 context: latest cred_score per isin from GF_TRACK rows ---
    cred_by = {}
    if not gva.empty:
        gt = gva[gva["source"].astype(str) == "gf_track"]
        for iso, grp in gt.groupby(gt["isin"].astype(str).str.strip()):
            vals = pd.to_numeric(grp["cred_score"], errors="coerce").dropna()
            if not vals.empty:      # cred_score can be text ("NA" etc.) — skip
                cred_by[iso] = float(vals.iloc[-1])

    rows, unmapped = [], []
    for sym, t in tech.items():
        iso = sym_to_isin.get(sym, "")
        if not iso:
            unmapped.append(sym)
            continue
        g = grades_by.get(iso)
        s = surge_by.get(iso)

        # strong results?
        surge_class = str(s["classification"]) if s is not None else ""
        yoy_tier = str(g["yoy_tier"]) if g is not None else "na"
        pat_yoy = (float(s["pat_yoy_pct"]) if s is not None
                   else (float(g["yoy"]) if g is not None and pd.notna(g["yoy"]) else None))
        results_strong = (surge_class in SURGE_STRONG) or (yoy_tier in G.GREEN_TIERS)
        if not results_strong:
            continue

        # strong guidance? — from the company's LATEST concall quarter (GF1/tracker),
        # not the biggest number ever parsed. Stale guidance (older than
        # GUID_STALE_MAX_Q) does not qualify.
        gl = _latest_guidance(gt_by.get(iso), gf1_by.get(iso), cur_key, doc_by)
        guid_cagr = gl["cagr"] if gl else None
        guid_tier = G.grade_growth(guid_cagr) if guid_cagr is not None else "na"
        fresh = gl is not None and gl["qtrs_old"] <= GUID_STALE_MAX_Q
        guidance_strong = fresh and (
            guid_tier in G.GREEN_TIERS or (guid_cagr or 0) >= MIN_GUIDANCE_CAGR)
        if not guidance_strong:
            continue

        results_pts = 100.0 if surge_class == "CONSISTENT" else (
            80.0 if surge_class == "EMERGING" else _tier_pts(yoy_tier))
        guid_pts = max(_tier_pts(guid_tier),
                       100.0 if guid_cagr >= 50 else
                       80.0 if guid_cagr >= MIN_GUIDANCE_CAGR else 0.0)
        rank = round(min(100.0, 0.4 * min(100.0, t["composite_score"])
                     + 0.3 * results_pts + 0.3 * guid_pts), 1)
        # PAT quarter + value: growth_surge (already realigned upstream) else derived;
        # realign grades-only rows to the freshest reported quarter so the label and
        # the % never come from different quarters.
        pat_qtr = (str(s["quarter"]) if s is not None else pat_qtr_by.get(iso, ""))
        fq, fyoy = pat_fresh_by.get(iso, (None, None))
        if fq:
            cur_d = pd.to_datetime(pat_qtr, format="%b %Y", errors="coerce")
            f_d = pd.to_datetime(fq, format="%b %Y", errors="coerce")
            if pd.notna(f_d) and (pd.isna(cur_d) or f_d > cur_d):
                pat_qtr = fq
                if fyoy is not None:
                    pat_yoy = fyoy
        rows.append({
            "isin": iso, "symbol": sym, "company_name": names.get(iso, ""),
            "as_of": today,
            "n_strategies": t["n_strategies"],
            "composite_score": round(t["composite_score"], 1),
            "strategies": t["strategies"],
            "pat_yoy_pct": pat_yoy, "pat_qtr": pat_qtr,
            "surge_class": surge_class or None,
            "yoy_tier": yoy_tier,
            "guidance_cagr": guid_cagr, "guidance_tier": guid_tier,
            "guid_from": gl["quarter"], "guid_metric": gl["metric"],
            "guid_qtrs_old": gl["qtrs_old"], "guid_stmt": gl["statement"],
            "guid_src_date": gl["src_date"], "guid_src_type": gl["src_type"],
            "guid_src_title": gl["src_title"],
            "guid_track": _gf2_track(gf2_by.get(iso)),
            "cred_score": cred_by.get(iso),
            "rank_score": rank, "in_pf": iso in pf,
        })
    rows.sort(key=lambda r: (-r["rank_score"], -(r["pat_yoy_pct"] or 0)))
    return rows, unmapped


def build_html(rows: list[dict], today: str) -> str:
    td = ("padding:6px 10px;border:1px solid #ddd;font-size:13px;"
          "font-family:Arial,sans-serif;")
    th = td + "background:#34495e;color:#fff;text-align:left;"
    n_pf = sum(1 for r in rows if r["in_pf"])
    out = [
        f"<div style='max-width:700px'>"
        f"<p style='font-family:Arial,sans-serif;font-size:13px'>"
        f"<b>{len(rows)} stock(s)</b> strong on ALL THREE axes today ({today}): "
        f"technical (≥2 strategies agree) + results (surging/green PAT YoY) + "
        f"guidance (green tier or ≥{MIN_GUIDANCE_CAGR:.0f}% CAGR)"
        + (f" — 💼 {n_pf} in PF" if n_pf else "") + ".</p>",
        "<table style='border-collapse:collapse;width:100%'>",
        "<tr>" + "".join(f"<th style='{th}'>{c}</th>" for c in
                         ["#", "Company", "Score", "Tech", "PAT YoY",
                          "Guidance", "Cred"]) + "</tr>",
    ]
    rows = sorted(rows, key=lambda r: (not r["in_pf"], -r["rank_score"]))
    n_total = len(rows)
    rows = rows[:MAIL_TOP_N]
    for i, r in enumerate(rows, 1):
        sym = esc(r["symbol"], 14)
        link = f"https://www.screener.in/company/{sym}/"
        pf_badge = "💼 " if r["in_pf"] else ""
        yoy_css = G.cell_css(G.grade_growth(r["pat_yoy_pct"]))
        guid_css = G.cell_css(r["guidance_tier"])
        surge_txt = f" · {r['surge_class']}" if r["surge_class"] else ""
        qtr_txt = f" ({esc(r.get('pat_qtr') or '', 10)})" if r.get("pat_qtr") else ""
        # guidance: metric + latest-quarter value, then the concall quarter with a
        # freshness tag (✓ latest / ⚠ N q old), then the verbatim GF1 quote + track.
        gq = str(r.get("guid_from") or "").strip()
        qold = r.get("guid_qtrs_old")
        if qold is None:
            fresh_tag = ""
        elif qold <= 1:
            fresh_tag = " <span style='color:#1a7a3a'>✓ latest</span>"
        else:
            fresh_tag = f" <span style='color:#c0392b'>⚠ {int(qold)}q old</span>"
        gmet = str(r.get("guid_metric") or "").upper()[:3]
        cagr_txt = (f"{gmet} +{r['guidance_cagr']:.0f}%"
                    if r["guidance_cagr"] is not None else r["guidance_tier"])
        sdate = str(r.get("guid_src_date") or "").strip()
        date_txt = f" · {esc(sdate, 12)}" if sdate else ""
        gsub = (f"<br><span style='font-size:11px;color:#555'>{esc(gq, 10)}"
                f"{date_txt}{fresh_tag}</span>") if gq else ""
        stmt = str(r.get("guid_stmt") or "").strip()
        gstmt = (f"<br><span style='font-size:10px;color:#888'>“{esc(stmt, 90)}”</span>"
                 if stmt else "")
        # where it was pulled from: doc type + the filing title
        stype = str(r.get("guid_src_type") or "").strip()
        stitle = str(r.get("guid_src_title") or "").strip()
        src_bits = " · ".join(x for x in (stype, stitle) if x)
        gsrc = (f"<br><span style='font-size:10px;color:#999'>src: {esc(src_bits, 70)}"
                f"</span>" if src_bits else "")
        trk = str(r.get("guid_track") or "").strip()
        gtrk = (f"<br><span style='font-size:10px;color:"
                f"{'#1a7a3a' if trk == 'delivers' else '#c0392b'}'>mgmt {trk}</span>"
                if trk else "")
        cred_txt = f"{r['cred_score']:.1f}/5" if r["cred_score"] is not None else ""
        yoy_txt = (f"+{r['pat_yoy_pct']:,.0f}%{qtr_txt}" if r["pat_yoy_pct"] is not None
                   else r["yoy_tier"])
        out.append(
            "<tr>"
            f"<td style='{td}'>{i}</td>"
            f"<td style='{td}'>{pf_badge}<a href='{link}' style='color:#1a237e'>"
            f"<b>{sym}</b></a> · {esc(r['company_name'], 30)}</td>"
            f"<td style='{td}'><b>{r['rank_score']:.0f}</b></td>"
            f"<td style='{td}'>{r['n_strategies']} strat · "
            f"{r['composite_score']:.0f}</td>"
            f"<td style='{td}{yoy_css}'>{yoy_txt}{surge_txt}</td>"
            f"<td style='{td}{guid_css}'>{cagr_txt}{gsub}{gstmt}{gsrc}{gtrk}</td>"
            f"<td style='{td}'>{cred_txt}</td>"
            "</tr>")
    out.append("</table>"
               + (f"<p style='font-family:Arial,sans-serif;font-size:12px'>"
                  f"…plus {n_total - len(rows)} more — full list on Drive.</p>"
                  if n_total > len(rows) else "")
               + "<p style='font-size:11px;color:#999;font-family:Arial,sans-serif'>"
               "<b>How to read:</b> <b>PAT YoY</b> = that quarter's net-profit growth, "
               "quarter in brackets, realigned to the latest REPORTED quarter "
               "(Screener financials). <b>Guidance</b> = the growth management "
               "committed to in its <b>most recent concall</b> (revenue &gt; EBITDA &gt; "
               "PAT), NOT the biggest number ever said. Under it: the concall "
               "<b>quarter + the filing date</b> it was pulled from, freshness "
               f"(✓ latest / ⚠ N quarters old; older than {GUID_STALE_MAX_Q}q doesn't "
               "qualify), the verbatim GF1 statement, <b>src:</b> the exact source "
               "document, and the GF2 delivery record (‘mgmt delivers/has missed’). "
               "<b>Cred</b> = "
               "GF_TRACK mgmt credibility (5=delivers). <b>Tech</b> = strategies "
               "agreeing · composite. Weekly/quarterly lists live on Drive under "
               "signals/combined_strength/. Toggle this mail in the app sidebar "
               "(📧 Email toggles).</p></div>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute + print + preview html; no Drive write, no mail.")
    args = ap.parse_args()

    print("Combined strength — strong technical + results + guidance")
    print("-" * 60)
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    index_id = _folder(drive, "company_repo/_index")
    agg_id = _folder(drive, "signals/aggregated")

    conv = _read_csv(drive, agg_id, "conviction.csv")
    surge = load_parquet(drive, index_id, "growth_surge.parquet", SURGE_COLS)
    grades = load_parquet(drive, index_id, "screener_grades.parquet", GRADES_COLS)
    gva = load_parquet(drive, index_id, "guidance_vs_actual.parquet", GVA_COLS)
    log(f"inputs: conviction={len(conv)} surge={len(surge)} "
        f"grades={len(grades)} gva={len(gva)}")

    sym_to_isin, names = {}, {}
    uni_id = find_file(drive, index_id, "company_universe.csv")
    if uni_id:
        uni = pd.read_csv(io.BytesIO(download_bytes(drive, uni_id)))
        for _, r in uni.iterrows():
            iso = str(r.get("isin", "")).strip()
            if not iso:
                continue
            names[iso] = str(r.get("name", "") or "").strip()
            sym = str(r.get("nse_symbol", "") or "").strip().upper()
            if sym and sym.lower() != "nan":
                sym_to_isin[sym] = iso
            # BSE-only names signal under their scrip code — map those too
            bse = str(r.get("bse_code", "") or "").strip()
            if bse and bse.lower() != "nan":
                sym_to_isin.setdefault(bse.split(".")[0], iso)

    pf = load_portfolio_isins(drive, root_id) or set()

    # PAT quarter (user 2026-07-12): latest pat_yoy period from financials_derived.
    pat_qtr_by: dict = {}
    derived = load_parquet(drive, index_id, "financials_derived.parquet",
                           ["isin", "metric", "period", "period_type", "value"])
    if not derived.empty:
        d = derived[(derived["metric"].astype(str) == "pat_yoy_pct")
                    & (derived["period_type"].astype(str) == "quarterly")]
        for iso, grp in d.groupby(d["isin"].astype(str).str.strip()):
            pat_qtr_by[iso] = str(grp["period"].iloc[-1])   # emit order = chronological
    # fresh reported PAT per isin — realigns the PAT column when derived lags a
    # quarter (same fix as flag_growth_surge; user 2026-07-18).
    res = load_parquet(drive, index_id, "results.parquet",
                       ["isin", "metric", "latest_q", "latest_val",
                        "yearago_val", "scraped_at"])
    if not res.empty:
        r = res[res["isin"].astype(str).str.strip() != ""]
        for iso, grp in r.groupby(r["isin"].astype(str).str.strip()):
            pat_qtr_by.setdefault(iso, str(grp["latest_q"].iloc[-1]))
    pat_fresh_by = _pat_from_results(res)

    # guidance now comes from the LATEST concall quarter (GF1/tracker), not the
    # max cagr ever parsed — per-isin slices of guidance_tracker / GF1 / GF2.
    gt = load_parquet(drive, index_id, "guidance_tracker.parquet",
                      ["isin", "symbol", "quarter", "metric", "cagr_pct",
                       "source_doc_id"])
    gf1 = load_parquet(drive, index_id, "gf1_guidance_statements.parquet",
                       ["isin", "quarter", "metric_type", "timeframe",
                        "quantifiable", "exact_statement", "source_doc_id"])
    gf2 = load_parquet(drive, index_id, "gf2_historical_guidance.parquet",
                       ["isin", "quarter", "management_self_assessment"])
    # source document behind each guidance row (user 2026-07-18: show the DATE and
    # the SOURCE it was pulled from). source_doc_id -> processing_queue resolves
    # 100% for both tracker and GF1.
    queue = load_parquet(drive, index_id, "processing_queue.parquet",
                         ["doc_id", "doc_type", "announcement_date", "title"])
    doc_by = ({str(d): {"date": a, "type": str(t or ""), "title": str(ti or "")}
               for d, t, a, ti in zip(queue["doc_id"], queue["doc_type"],
                                      queue["announcement_date"], queue["title"])}
              if not queue.empty else {})

    def _by_isin(df):
        if df is None or df.empty:
            return {}
        return {iso: grp for iso, grp in df.groupby(df["isin"].astype(str).str.strip())}

    gt_by, gf1_by, gf2_by = _by_isin(gt), _by_isin(gf1), _by_isin(gf2)
    cur_key = _cur_fy_qkey(date.today())
    log(f"guidance inputs: tracker={len(gt)} gf1={len(gf1)} gf2={len(gf2)} "
        f"queue-docs={len(doc_by)} (cur FY-quarter key {cur_key})")

    today = date.today().isoformat()
    rows, unmapped = build_list(conv, surge, grades, gva,
                                sym_to_isin, names, pf, today,
                                pat_qtr_by, pat_fresh_by,
                                gt_by, gf1_by, gf2_by, cur_key, doc_by)
    if unmapped:
        log(f"tech symbols with no isin mapping (skipped): {len(unmapped)} "
            f"e.g. {', '.join(sorted(unmapped)[:8])}")
    log(f"combined-strength stocks today: {len(rows)} "
        f"({sum(1 for r in rows if r['in_pf'])} in PF)")
    daily_df = pd.DataFrame(rows, columns=HIST_COLS)

    # history -> weekly / quarterly roll-ups (latest row per isin in window)
    hist = load_parquet(drive, index_id, HIST_NAME, HIST_COLS)
    hist = hist[hist["as_of"].astype(str) != today]        # idempotent per day
    if not daily_df.empty:
        hist = daily_df.copy() if hist.empty else pd.concat(
            [hist, daily_df], ignore_index=True)

    def _window(since: date) -> pd.DataFrame:
        w = hist[hist["as_of"].astype(str) >= since.isoformat()]
        if w.empty:
            return w
        n_days = w.groupby("isin")["as_of"].nunique().rename("days_on_list")
        w = (w.sort_values("as_of").groupby("isin").tail(1)
             .merge(n_days, on="isin"))
        return w.sort_values(["rank_score"], ascending=False).reset_index(drop=True)

    weekly = _window(date.today() - timedelta(days=7))
    quarterly = _window(_fy_quarter_start(date.today()))
    log(f"weekly (7d union): {len(weekly)} · quarterly "
        f"(since {_fy_quarter_start(date.today())}): {len(quarterly)}")

    if args.dry_run:
        if not daily_df.empty:
            cols = ["symbol", "rank_score", "n_strategies", "pat_yoy_pct",
                    "surge_class", "yoy_tier", "guidance_cagr", "guidance_tier",
                    "cred_score", "in_pf"]
            print(daily_df[cols].head(25).to_string(index=False))
        html = build_html(rows, today)
        prev = Path(__file__).resolve().parent.parent / "combined_strength_preview.html"
        prev.write_text(html, encoding="utf-8")
        print(f"\nDRY RUN — preview saved to {prev.name}; no Drive write, no mail.")
        return

    cs_id = _folder(drive, "signals/combined_strength")
    _write_csv(drive, cs_id, "latest.csv", daily_df)
    _write_csv(drive, cs_id, "weekly.csv", weekly)
    _write_csv(drive, cs_id, "quarterly.csv", quarterly)
    save_parquet(drive, index_id, HIST_NAME, hist)
    log(f"wrote signals/combined_strength/{{latest,weekly,quarterly}}.csv "
        f"+ _index/{HIST_NAME} ({len(hist)} rows)")

    if daily_df.empty:
        log("no stock strong on all three axes today — no mail.")
        return
    if not load_mail_settings(drive, index_id).get("combined_strength", True):
        log("combined_strength mail toggled OFF — skipped.")
        return
    n_pf = int(daily_df["in_pf"].sum())
    subject = (f"🏆 Combined strength — {len(daily_df)} stocks "
               f"(tech+results+guidance)" + (f" · 💼{n_pf} PF" if n_pf else ""))
    sent = send_email(subject, build_html(rows, today))
    # ascii-only in log lines — local console is cp1252, emoji crash print()
    log(f"Email {'sent' if sent else 'FAILED'}: "
        f"{subject.encode('ascii', 'ignore').decode().strip()}")


if __name__ == "__main__":
    main()
