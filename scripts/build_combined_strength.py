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
import gradation as G

MIN_GUIDANCE_CAGR = 20.0        # forward CAGR that counts as strong guidance
MAX_GUIDANCE_CAGR = 200.0       # above this the "CAGR" is a misparsed absolute
                                # (e.g. "1,000 Cr" -> 1000) — not trustworthy
TECH_ZONES = ("buy", "add")     # conviction zones that count as strong technical
SURGE_STRONG = ("CONSISTENT", "EMERGING")
MAIL_TOP_N = 30                 # mail shows the top N; full list lives on Drive

HIST_NAME = "combined_strength_history.parquet"
HIST_COLS = ["isin", "symbol", "company_name", "as_of",
             "n_strategies", "composite_score", "strategies",
             "pat_yoy_pct", "surge_class", "yoy_tier",
             "guidance_cagr", "guidance_tier", "cred_score",
             "rank_score", "in_pf"]

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


def build_list(conv: pd.DataFrame, surge: pd.DataFrame, grades: pd.DataFrame,
               gva: pd.DataFrame, sym_to_isin: dict, names: dict,
               pf: set[str], today: str) -> tuple[list[dict], int]:
    """Intersect the three axes. Returns (rows, n_unmapped_tech_symbols)."""
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
        gt = gva[gva["source"].astype(str) == "gf_track"].dropna(subset=["cred_score"])
        for iso, grp in gt.groupby(gt["isin"].astype(str).str.strip()):
            cred_by[iso] = float(pd.to_numeric(grp["cred_score"],
                                               errors="coerce").dropna().iloc[-1])

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

        # strong guidance? (tier AND value derive from the same cagr number, so
        # an implausible value poisons both — require the plausible window)
        guid_tier = str(g["guidance_tier"]) if g is not None else "na"
        guid_cagr = (float(g["guidance"]) if g is not None and pd.notna(g["guidance"])
                     else None)
        plausible = guid_cagr is not None and 0 < guid_cagr <= MAX_GUIDANCE_CAGR
        guidance_strong = plausible and (
            guid_tier in G.GREEN_TIERS or guid_cagr >= MIN_GUIDANCE_CAGR)
        if not guidance_strong:
            continue

        results_pts = 100.0 if surge_class == "CONSISTENT" else (
            80.0 if surge_class == "EMERGING" else _tier_pts(yoy_tier))
        guid_pts = max(_tier_pts(guid_tier),
                       100.0 if guid_cagr >= 50 else
                       80.0 if guid_cagr >= MIN_GUIDANCE_CAGR else 0.0)
        rank = round(min(100.0, 0.4 * min(100.0, t["composite_score"])
                     + 0.3 * results_pts + 0.3 * guid_pts), 1)
        rows.append({
            "isin": iso, "symbol": sym, "company_name": names.get(iso, ""),
            "as_of": today,
            "n_strategies": t["n_strategies"],
            "composite_score": round(t["composite_score"], 1),
            "strategies": t["strategies"],
            "pat_yoy_pct": pat_yoy, "surge_class": surge_class or None,
            "yoy_tier": yoy_tier,
            "guidance_cagr": guid_cagr, "guidance_tier": guid_tier,
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
        yoy_bg = G.TIER_COLOR.get(G.grade_growth(r["pat_yoy_pct"]), "")
        guid_bg = G.TIER_COLOR.get(r["guidance_tier"], "")
        surge_txt = f" ({r['surge_class']})" if r["surge_class"] else ""
        cagr_txt = (f"{r['guidance_cagr']:.0f}% CAGR"
                    if r["guidance_cagr"] is not None else r["guidance_tier"])
        cred_txt = f"{r['cred_score']:.1f}/5" if r["cred_score"] is not None else ""
        yoy_txt = (f"+{r['pat_yoy_pct']:,.0f}%" if r["pat_yoy_pct"] is not None
                   else r["yoy_tier"])
        out.append(
            "<tr>"
            f"<td style='{td}'>{i}</td>"
            f"<td style='{td}'>{pf_badge}<a href='{link}' style='color:#1a237e'>"
            f"<b>{sym}</b></a> · {esc(r['company_name'], 30)}</td>"
            f"<td style='{td}'><b>{r['rank_score']:.0f}</b></td>"
            f"<td style='{td}'>{r['n_strategies']} strat · "
            f"{r['composite_score']:.0f}</td>"
            f"<td style='{td}background:{yoy_bg};'>{yoy_txt}{surge_txt}</td>"
            f"<td style='{td}background:{guid_bg};'>{cagr_txt}</td>"
            f"<td style='{td}'>{cred_txt}</td>"
            "</tr>")
    out.append("</table>"
               + (f"<p style='font-family:Arial,sans-serif;font-size:12px'>"
                  f"…plus {n_total - len(rows)} more — full list on Drive.</p>"
                  if n_total > len(rows) else "")
               + "<p style='font-size:11px;color:#999;font-family:Arial,sans-serif'>"
               "Tech = strategies agreeing · composite. Cred = GF_TRACK mgmt "
               "credibility (5=delivers). Weekly/quarterly lists live on Drive "
               "under signals/combined_strength/. Toggle this mail in the app "
               "sidebar (📧 Email toggles).</p></div>")
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
    today = date.today().isoformat()
    rows, unmapped = build_list(conv, surge, grades, gva,
                                sym_to_isin, names, pf, today)
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
    log(f"Email {'sent' if sent else 'FAILED'}: {subject}")


if __name__ == "__main__":
    main()
