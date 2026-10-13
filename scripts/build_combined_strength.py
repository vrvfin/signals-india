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
import mail_chart

MIN_GUIDANCE_CAGR = 20.0        # forward CAGR that counts as strong guidance
MAX_GUIDANCE_CAGR = 200.0       # above this the "CAGR" is a misparsed absolute
                                # (e.g. "1,000 Cr" -> 1000) — not trustworthy
GUID_STALE_MAX_Q = 4            # latest guidance older than this many quarters =
                                # not "current" -> doesn't qualify (user 2026-07-18)
GROWTH_METRICS = ("revenue", "ebitda", "pat",   # % growth metrics only (exclude
                  "loan_aum", "deposits")       # margin/NIM/NPA/capacity LEVELS).
                                # BFSI growth metrics added 2026-07-29 so a bank
                                # guiding 25% AUM growth qualifies like any other.
TECH_ZONES = ("buy", "add")     # conviction zones that count as strong technical
MIN_SETUPS = 2                  # distinct setups that must agree (user 2026-10-12;
                                # was n_strategies >= 2, which momentum variants met alone)


def setup_of(strategy: str) -> str:
    """The IDEA behind a strategy. Variants of one idea collapse: momentum_1m..12m ->
    momentum, ma_respect_20ema_30d/_20ema_60d/_50ema_60d -> ma_respect. volume_breakout
    (a volume surge) and volume_vcp (a volume dry-up) are opposite tells, kept apart."""
    s = str(strategy or "").strip()
    if s.startswith("momentum_"):
        return "momentum"
    if s.startswith("ma_respect_"):
        return "ma_respect"
    return s


def setups_of(strategies) -> list[str]:
    """Distinct setups in a conviction row's comma-joined `strategies`."""
    return sorted({setup_of(x) for x in str(strategies or "").split(",") if x.strip()})
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
        c = conv[conv["zone_type"].astype(str).str.lower().isin(TECH_ZONES)].copy()
        # TWO DIFFERENT SETUPS, not two strategies (user 2026-10-12). momentum_3m +
        # momentum_6m, or two ma_respect variants, are one idea counted twice.
        c["_setups"] = c["strategies"].map(setups_of)
        c = c[c["_setups"].map(len) >= MIN_SETUPS]
        for _, r in c.iterrows():
            sym = str(r["symbol"]).strip().upper()
            cur = tech.get(sym)
            n_set = len(r["_setups"])
            if cur is None or n_set > cur["n_families"]:
                tech[sym] = {"n_strategies": int(r["n_strategies"]),
                             "composite_score": float(r["composite_score"]),
                             "strategies": str(r.get("strategies", "")),
                             # mail-only context (user 2026-10-10: "explain it");
                             # rows are projected onto HIST_COLS before any write
                             "zone": str(r.get("zone_type", "")).lower(),
                             "n_families": n_set,
                             "families": ", ".join(r["_setups"]),
                             "entry": _flt(r.get("entry_median")),
                             "stop": _flt(r.get("stop_median"))}

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
            # mail-only: the score's parts and the technical context
            "tech_pts": round(min(100.0, t["composite_score"]), 1),
            "results_pts": results_pts, "guid_pts": guid_pts,
            "zone": t.get("zone", ""), "n_families": t.get("n_families"),
            "families": t.get("families", ""),
            "entry": t.get("entry"), "stop": t.get("stop"),
        })
    rows.sort(key=lambda r: (-r["rank_score"], -(r["pat_yoy_pct"] or 0)))
    return rows, unmapped


def _flt(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _int(v):
    f = _flt(v)
    return None if f is None else int(f)


# Plain-English one-liners, taken from each strategy file's own docstring.
STRATEGY_PLAIN = {
    "canslim": "CANSLIM: EPS growth ≥25% (qtr and year), near 52-wk high, RS ≥80, ROE ≥15%",
    "darvas": "Darvas box: tight sideways box near the 52-wk high, breaking out on volume",
    "ipo_base": "IPO base: first base breakout in a recent listing",
    "minervini": "Minervini trend template: price above stacked, rising 50/100/200 averages",
    "pead": "Post-earnings drift: gapped up ≥5% on results day on ≥2× volume, still holding",
    "pullback": "Pullback: dip to a rising average inside an intact uptrend (lower-risk entry)",
    "qullamaggie": "Qullamaggie: ≥30% 3-month run, then a tight consolidation",
    "volume_breakout": "Volume breakout: volume surge in an uptrend near the highs",
    "volume_vcp": "Volume dry-up: volume and range contracting in a tight base near highs",
}


def plain_strategy(name: str) -> str:
    n = str(name or "").strip()
    if n in STRATEGY_PLAIN:
        return STRATEGY_PLAIN[n]
    m = re.fullmatch(r"momentum_(\d+m)", n)
    if m:
        return (f"Momentum {m.group(1)}: top-decile relative strength over "
                f"{m.group(1)}, price above the 200-day average")
    m = re.fullmatch(r"ma_respect_(\d+)ema_(\d+)d", n)
    if m:
        return (f"Holds above its {m.group(1)}-day average: no close below it "
                f"for {m.group(2)}+ days")
    return n


def _pct(v, signed=True):
    return "—" if v is None else (f"{v:+,.0f}%" if signed else f"{v:,.0f}%")


def why_line(r: dict) -> str:
    """One sentence: why this stock is on the list, in the order the rule checks it."""
    bits = []
    if r.get("pat_yoy_pct") is not None:
        q = f" in {r['pat_qtr']}" if r.get("pat_qtr") else ""
        cls = {"CONSISTENT": " — 3+ of the last 4 quarters ≥100%",
               "EMERGING": " — 2 of the last 4 quarters ≥100%"}.get(
                   r.get("surge_class") or "", "")
        bits.append(f"profit {_pct(r['pat_yoy_pct'])} YoY{q}{cls}")
    if r.get("guidance_cagr") is not None:
        met = str(r.get("guid_metric") or "growth").upper()
        src = f" ({r['guid_from']} concall)" if r.get("guid_from") else ""
        bits.append(f"management guides {met} {_pct(r['guidance_cagr'])}{src}")
    nf = r.get("n_families")
    fam = (f"{nf} independent setups" if nf else f"{r['n_strategies']} strategies")
    bits.append(f"{fam} say {r.get('zone') or 'buy'}")
    return "; ".join(bits) + "."


def score_math(r: dict) -> str:
    """The rank score, worked: same formula build_list uses."""
    t, re_, g = r.get("tech_pts"), r.get("results_pts"), r.get("guid_pts")
    if None in (t, re_, g):
        return f"Score {r['rank_score']:.0f}"
    return (f"Score <b>{r['rank_score']:.0f}</b> = 0.4×chart {t:.0f} + "
            f"0.3×results {re_:.0f} + 0.3×guidance {g:.0f}")


CARD_N = 15            # stocks shown as full cards (chart + reasons); approved cap


def _card(i: int, r: dict, cid: str | None) -> str:
    sym = esc(r["symbol"], 14)
    pf = "💼 " if r["in_pf"] else ""
    strat = [x.strip() for x in str(r.get("strategies") or "").split(",") if x.strip()]
    strat_li = "".join(f"<li>{esc(plain_strategy(x), 140)}</li>" for x in strat[:6])
    if len(strat) > 6:
        strat_li += f"<li style='color:#999'>+{len(strat) - 6} more</li>"
    gq = esc(str(r.get("guid_stmt") or ""), 160)
    src_bits = " · ".join(x for x in (str(r.get("guid_src_type") or ""),
                                      str(r.get("guid_src_title") or ""),
                                      str(r.get("guid_src_date") or "")) if x)
    qold = r.get("guid_qtrs_old")
    fresh = ("" if qold is None else " · latest concall" if qold <= 1
             else f" · ⚠ {int(qold)} quarters old")
    trk = str(r.get("guid_track") or "")
    cred = (f" · credibility {r['cred_score']:.1f}/5" if r.get("cred_score") is not None
            else "")
    lvl = ""
    if r.get("entry") or r.get("stop"):
        lvl = (f"<div style='font-size:12px;color:#555;margin-top:4px'>Strategy levels: "
               f"entry ~{r['entry']:,.1f} · stop ~{r['stop']:,.1f} (median across the "
               f"agreeing strategies)</div>"
               if r.get("entry") and r.get("stop") else "")
    chart = mail_chart.img_tag(cid, sym) if cid else (
        "<div style='color:#999;font-size:11px'>(no price history for a chart)</div>")
    return (
        f"<div style='border:1px solid #dde3e8;border-radius:6px;padding:10px 12px;"
        f"margin:0 0 14px;font-family:Arial,sans-serif;font-size:13px'>"
        f"<div style='font-size:15px'><b>{i}. {pf}<a href='https://www.screener.in/company/"
        f"{sym}/' style='color:#1a237e'>{sym}</a></b> · {esc(r['company_name'], 50)}"
        f"<span style='float:right'>{score_math(r)}</span></div>"
        f"<div style='margin:6px 0;color:#222'><b>Why here:</b> {esc(why_line(r), 400)}</div>"
        f"{chart}"
        f"<table style='width:100%;border-collapse:collapse;margin-top:6px;font-size:12px'>"
        f"<tr><td style='vertical-align:top;width:34%;padding:4px'><b>📈 Chart</b>"
        f"<ul style='margin:2px 0 0 16px;padding:0'>{strat_li}</ul>"
        f"<div style='color:#999'>src: signals/aggregated/conviction.csv</div></td>"
        f"<td style='vertical-align:top;width:33%;padding:4px'><b>📊 Results</b><br>"
        f"PAT YoY {_pct(r.get('pat_yoy_pct'))}"
        f"{(' (' + esc(r['pat_qtr'], 10) + ')') if r.get('pat_qtr') else ''}"
        f"{('<br>' + esc(r['surge_class'], 12)) if r.get('surge_class') else ''}"
        f"<div style='color:#999'>src: Screener quarterly P&amp;L "
        f"(growth_surge / screener_grades)</div></td>"
        f"<td style='vertical-align:top;width:33%;padding:4px'><b>🎯 Guidance</b><br>"
        f"{esc(str(r.get('guid_metric') or '').upper(), 10)} "
        f"{_pct(r.get('guidance_cagr'))}{fresh}{cred}"
        f"{('<br><i>“' + gq + '”</i>') if gq else ''}"
        f"{('<br>mgmt ' + esc(trk, 20)) if trk else ''}"
        f"<div style='color:#999'>src: {esc(src_bits, 90) or 'concall guidance'}</div>"
        f"</td></tr></table>{lvl}</div>")


def explainer() -> str:
    return (
        "<div style='background:#f6f8fa;border:1px solid #e0e6ea;border-radius:6px;"
        "padding:9px 12px;margin:0 0 12px;font:12px Arial,sans-serif;line-height:1.6'>"
        "<b>How this list is built</b> — a stock must pass ALL three:<br>"
        "1. <b>Chart:</b> ≥2 DIFFERENT setups put it in a buy/add zone today "
        "(conviction.csv) — variants of one idea (momentum 3m and 6m; the three "
        "moving-average-hold rules) count once. Chart points = average percentile of its signal in each "
        "agreeing strategy (90 = top 10% of that strategy's picks).<br>"
        f"2. <b>Results:</b> latest-quarter profit growth ≥30% YoY, or a ≥100% surge in "
        f"2+ of the last 4 quarters. Points: 3 of 4 quarters = 100, 2 of 4 = 80, else "
        f"by growth band (≥100% = 100, ≥50% = 80, ≥30% = 60).<br>"
        f"3. <b>Guidance:</b> management's latest concall guides ≥{MIN_GUIDANCE_CAGR:.0f}% "
        f"growth, said within the last {GUID_STALE_MAX_Q} quarters. Points: ≥50% = 100, "
        f"≥{MIN_GUIDANCE_CAGR:.0f}% = 80 (or the growth band, if higher).<br>"
        "<b>Score</b> = 0.4×chart + 0.3×results + 0.3×guidance (0-100). "
        "</div>")


def build_html(rows: list[dict], today: str, cids: dict | None = None) -> str:
    """Cards (chart + why + maths + sources) for the top CARD_N, then one line each."""
    cids = cids or {}
    n_pf = sum(1 for r in rows if r["in_pf"])
    rows = sorted(rows, key=lambda r: (not r["in_pf"], -r["rank_score"]))
    n_total = len(rows)
    rows = rows[:MAIL_TOP_N]
    out = [f"<div style='max-width:720px;font-family:Arial,sans-serif'>"
           f"<p style='font-size:13px'><b>{n_total} stock(s)</b> strong on chart + results "
           f"+ guidance today ({today})" + (f" — 💼 {n_pf} in PF" if n_pf else "")
           + ". Holdings first, then by score.</p>", explainer()]
    for i, r in enumerate(rows[:CARD_N], 1):
        out.append(_card(i, r, cids.get(r["symbol"])))
    rest = rows[CARD_N:]
    if rest:
        td = "padding:4px 8px;border-bottom:1px solid #eee;font-size:12px"
        out.append("<p style='font-size:13px;margin:10px 0 4px'><b>Also on the list</b> "
                   "(no chart to keep the mail light):</p><table style='width:100%;"
                   "border-collapse:collapse'>")
        for i, r in enumerate(rest, CARD_N + 1):
            out.append(f"<tr><td style='{td}'>{i}</td><td style='{td}'>"
                       f"{'💼 ' if r['in_pf'] else ''}<b>{esc(r['symbol'], 14)}</b></td>"
                       f"<td style='{td}'>{r['rank_score']:.0f}</td>"
                       f"<td style='{td}'>{esc(why_line(r), 220)}</td></tr>")
        out.append("</table>")
    if n_total > len(rows):
        out.append(f"<p style='font-size:12px'>…plus {n_total - len(rows)} more — full "
                   f"list on Drive (signals/combined_strength/latest.csv).</p>")
    out.append("<p style='font-size:11px;color:#999'>Charts: 1 year daily, 50/200-day "
               "averages, dashed line = 52-week high. Signals only — not advice. Toggle "
               "this mail in the app sidebar (📧 Email toggles).</p></div>")
    return "\n".join(out)


def render_charts(drive, rows: list[dict], log=print) -> tuple[dict, list]:
    """{symbol: cid}, [(cid, png)] for the card stocks. A failure costs the chart,
    never the mail."""
    cids, imgs = {}, []
    top = sorted(rows, key=lambda r: (not r["in_pf"], -r["rank_score"]))[:CARD_N]
    syms = [r["symbol"] for r in top]
    try:
        from build_gallery import _bulk_parquet, _folder as _gfolder
        omap = _bulk_parquet(drive, _gfolder(drive, "data/ohlcv"), syms)
    except Exception as e:
        log(f"charts skipped — OHLCV load failed ({str(e)[:70]})")
        return cids, imgs
    for s_ in syms:
        try:
            png = mail_chart.render_png(omap.get(s_), s_)
        except Exception as e:
            log(f"  chart {s_} failed ({str(e)[:50]})")
            png = b""
        if png:
            cid = f"cs_{re.sub(r'[^A-Za-z0-9]', '_', s_)}"
            cids[s_] = cid
            imgs.append((cid, png))
    log(f"charts: {len(imgs)}/{len(syms)} rendered "
        f"({sum(len(b) for _, b in imgs) / 1024:.0f} KB)")
    return cids, imgs


def _preview_with_images(html: str, imgs: list) -> str:
    """Dry-run only: inline the PNGs as data URIs so a browser shows the mail."""
    import base64
    for cid, png in imgs:
        html = html.replace(f"cid:{cid}",
                            "data:image/png;base64," + base64.b64encode(png).decode())
    return html


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
        cids, imgs = render_charts(drive, rows, log)
        html = _preview_with_images(build_html(rows, today, cids), imgs)
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
    cids, imgs = render_charts(drive, rows, log)
    sent = send_email(subject, build_html(rows, today, cids), inline_images=imgs)
    # ascii-only in log lines — local console is cp1252, emoji crash print()
    log(f"Email {'sent' if sent else 'FAILED'}: "
        f"{subject.encode('ascii', 'ignore').decode().strip()}")


def _self_test() -> int:
    """Offline: rule + score maths + mail layout. No Drive, no mail."""
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    conv = pd.DataFrame([
        {"symbol": "AAA", "zone_type": "buy", "n_strategies": 3, "composite_score": 90,
         "strategies": "minervini, momentum_3m, momentum_6m", "n_families": 2,
         "families": "minervini, momentum", "entry_median": 100.0, "stop_median": 92.0},
        {"symbol": "BBB", "zone_type": "add", "n_strategies": 2, "composite_score": 70,
         "strategies": "darvas, volume_breakout", "n_families": 2,
         "families": "darvas, volume_breakout"},
        {"symbol": "CCC", "zone_type": "buy", "n_strategies": 1, "composite_score": 99,
         "strategies": "pead"}])
    surge = pd.DataFrame([{"isin": "I1", "classification": "CONSISTENT",
                           "pat_yoy_pct": 150.0, "quarter": "Jun 2026"}])
    grades = pd.DataFrame([{"isin": "I2", "yoy_tier": "great", "yoy": 60.0},
                           {"isin": "I3", "yoy_tier": "exceptional", "yoy": 200.0}])
    gt = {"I1": None, "I2": None}
    orig = globals()["_latest_guidance"]
    globals()["_latest_guidance"] = lambda g, f, k, d: {
        "cagr": 35.0, "quarter": "Q1 FY27", "metric": "revenue", "qtrs_old": 0,
        "statement": "we expect 35% growth", "src_date": "2026-08-10",
        "src_type": "concall", "src_title": "Transcript"}
    try:
        rows, _ = build_list(conv, surge, grades, pd.DataFrame(),
                             {"AAA": "I1", "BBB": "I2", "CCC": "I3"},
                             {"I1": "Alpha", "I2": "Beta", "I3": "Gamma"}, {"I2"},
                             "2026-10-11", gt_by=gt)
    finally:
        globals()["_latest_guidance"] = orig
    by = {r["symbol"]: r for r in rows}
    check("1-strategy stock excluded", "CCC" not in by)
    check("setups collapse variants",
          setups_of("momentum_3m, momentum_6m, ma_respect_20ema_30d, ma_respect_50ema_60d")
          == ["ma_respect", "momentum"])
    check("volume surge and dry-up stay two setups",
          len(setups_of("volume_breakout, volume_vcp")) == 2)
    conv2 = pd.DataFrame([
        {"symbol": "MOM", "zone_type": "buy", "n_strategies": 3, "composite_score": 95,
         "strategies": "momentum_1m, momentum_3m, momentum_6m"},
        {"symbol": "MAR", "zone_type": "buy", "n_strategies": 2, "composite_score": 95,
         "strategies": "ma_respect_20ema_30d, ma_respect_20ema_60d"}])
    globals()["_latest_guidance"] = lambda g, f, k, d: {
        "cagr": 35.0, "quarter": "Q1 FY27", "metric": "revenue", "qtrs_old": 0,
        "statement": "", "src_date": "", "src_type": "", "src_title": ""}
    try:
        rows2, _ = build_list(conv2, surge, grades, pd.DataFrame(),
                              {"MOM": "I1", "MAR": "I2"}, {}, set(), "2026-10-12",
                              gt_by=gt)
    finally:
        globals()["_latest_guidance"] = orig
    check("one idea counted many times does NOT qualify", rows2 == [])
    a = by.get("AAA", {})
    check("score parts: chart 90, results 100 (CONSISTENT), guidance 80 (>=20%)",
          (a.get("tech_pts"), a.get("results_pts"), a.get("guid_pts")) == (90.0, 100.0, 80.0))
    check("score = 0.4*90+0.3*100+0.3*80 = 90", a.get("rank_score") == 90.0)
    check("score maths shown", "0.4×chart 90" in score_math(a))
    check("why line names the surge and families",
          "3+ of the last 4" in why_line(a) and "2 independent setups" in why_line(a))
    check("plain strategy names",
          plain_strategy("momentum_6m").startswith("Momentum 6m")
          and plain_strategy("ma_respect_20ema_30d").startswith("Holds above its 20-day")
          and plain_strategy("unknown_x") == "unknown_x")
    html = build_html(rows, "2026-10-11", {"AAA": "cs_AAA"})
    check("PF holding listed first", html.index("Beta") < html.index("Alpha"))
    check("chart tag present for AAA", 'src="cid:cs_AAA"' in html)
    check("no-chart note for BBB", "no price history" in html)
    check("explainer present", "How this list is built" in html)
    check("persisted frame unchanged (HIST_COLS only)",
          list(pd.DataFrame(rows, columns=HIST_COLS).columns) == HIST_COLS)
    check("mail html small (<60 KB for 2 cards)", len(html.encode()) < 60_000)
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    main()
