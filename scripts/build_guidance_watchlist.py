r"""
build_guidance_watchlist.py — the running >50%-CAGR guidance watchlist.

Writes ONE accumulating table, company_repo/_index/guidance_watchlist.parquet:
one row per (isin, quarter) for every company whose management guided REVENUE or
PAT to grow faster than --min-cagr per year, validated against the transcript.
The same company reappears each quarter it qualifies, so `n_quarters` and
`in_prev_quarter` build a real history.

    python scripts/build_guidance_watchlist.py --dry-run       # nothing written
    python scripts/build_guidance_watchlist.py --backfill      # seed all history
    python scripts/build_guidance_watchlist.py                 # current quarter
    python scripts/build_guidance_watchlist.py --self-test     # offline fixtures

WHAT IT REUSES
--------------
  guidance_strength   cleaning + annualised CAGR (never build_gallery.guidance_scores,
                      which is LIVE for gallery_guidance.html and stays untouched)
  guidance_validate   CONFIRMED/CONSISTENT/CONTRADICTED/NO_EVIDENCE vs GF1
  quarterly_table     season_quarter / norm_q / q_order  (RESULTS convention)
  guidance_progress   q_idx / parse_q for calendar-quarter arithmetic
  _extractor_base     get_drive / load_parquet / save_parquet / acquire_lock

DESIGN NOTES THAT MATTER
------------------------
* date_added is PINNED per (isin, quarter): seeded from the earliest processed_at
  of that key's qualifying rows and carried forward on every later run, so a
  re-run never bumps a name to the top and --backfill does not stamp all of
  history with one date. Idiom copied from earnings_calendar.persist_calendar.
* in_prev_quarter means the previous CALENDAR quarter (q_idx - 1), not "the
  previous row for this ISIN" -- a name qualifying in Q1FY26 and again in Q1FY27
  has in_prev_quarter=False and n_quarters=2. Reading a gap as continuity is the
  bug pf_decision_tracker.build_spells exists to avoid.
* prev_quarter_had_guidance separates "stopped guiding big" from "no concall was
  extracted" -- without it a coverage hole looks like a company going quiet.
* BLANK-QUARTER REPAIR: 10.9% of guidance_tracker rows (4,211 rows / 513
  companies / 998 documents, measured 2026-08-21) carry no quarter at all, because
  extract_concall derives it by regex from the Table_A header and gives up per
  DOCUMENT when that fails. Every one is recoverable from processing_queue's
  announcement_date, so it is repaired HERE -- the live Phase-2 concall path is
  not touched.
* Grouping is by ISIN, never symbol: blank SME symbols would otherwise collapse
  into one "" group.
"""
from __future__ import annotations

import argparse
import atexit
import io
import os
import sys
from datetime import datetime, timezone

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import guidance_progress as GP
import guidance_strength as GS
import guidance_validate as GVAL
import quarterly_table as QT
from _extractor_base import (acquire_lock, download_bytes, find_file, get_drive,
                             get_or_create_subfolder, isin_symbol_map,
                             load_parquet, load_portfolio_isins, log,
                             release_lock, save_parquet)

GW_NAME = "guidance_watchlist.parquet"
GW_KEY = ["isin", "quarter"]

# Schema-first (CLAUDE.md rule 2). ADDITIVE ONLY from here -- load_parquet
# back-fills missing columns with None, so old rows survive a new column.
GW_COLS = [
    # --- identity + the columns the watchlist was asked for -----------------
    "isin", "quarter", "date_added",
    "nse_symbol", "bse_code", "nse_name",
    "in_prev_quarter", "n_quarters",
    # --- honesty about the two above ---------------------------------------
    "quarter_streak", "prev_quarter_had_guidance",
    # --- the score and how it was reached ----------------------------------
    "cagr_pct", "score_metric", "score_kind", "score_rule",
    "horizon_fy", "guided_value", "value_type", "years_used",
    "base_ttm_cr", "target_cr", "base_source", "base_suspect",
    "base_quarters", "base_scale",
    # --- transcript validation (no capping; evidence decides) --------------
    "validation_verdict", "evidence_stmt", "evidence_num", "evidence_delta_pct",
    # --- provenance ---------------------------------------------------------
    "symbol", "company_name", "guidance_source", "source_doc_id",
    "quarter_source", "cred_score", "n_rows_over_min", "in_pf",
    # --- run bookkeeping ----------------------------------------------------
    "min_cagr_used", "max_stale_q_used", "tracking_from", "date_added_source",
    "first_qualified_at", "last_seen_at", "as_of",
]

GUIDANCE_NAME = "guidance_tracker.parquet"
GUIDANCE_COLS = ["isin", "symbol", "company_name", "quarter", "metric",
                 "guidance_type", "horizon_fy", "value", "unit", "cagr_pct",
                 "notes", "processed_at", "source_doc_id",
                 "value_type", "value_num", "value_unit"]
GF1_NAME = "gf1_guidance_statements.parquet"
FIN3_NAME = "financials_3stmt.parquet"
CRED_NAME = "mgmt_credibility.parquet"
GRADES_NAME = "screener_grades.parquet"
PPT_NAME = "ppt_guidance.parquet"
QUEUE_NAME = "processing_queue.parquet"
UNIVERSE_NAME = "company_universe.csv"

_LOCK_NAME = "_extract.lock"
_LOCK_OWNER = "guidance_watchlist"


def _now() -> str:
    """UTC, per CLAUDE.md rule 8 (CI runners are UTC; reports convert to IST)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _folder(drive, parts: str) -> str:
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _read_any(drive, folder_id: str, name: str) -> pd.DataFrame:
    """Plain reader for tables with no shared *_COLS constant."""
    fid = find_file(drive, folder_id, name)
    if not fid:
        log(f"  {name} missing — continuing without it")
        return pd.DataFrame()
    try:
        raw = download_bytes(drive, fid)
        return (pd.read_csv(io.BytesIO(raw)) if name.endswith(".csv")
                else pd.read_parquet(io.BytesIO(raw)))
    except Exception as e:                                    # noqa: BLE001
        log(f"  could not read {name} ({str(e)[:70]}) — continuing without it")
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
#  quarter resolution (incl. the blank-quarter repair)                          #
# --------------------------------------------------------------------------- #
def resolve_quarters(guid: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Add `_qn` (norm quarter), `_qo` (sortable) and `quarter_source`.

    A blank quarter is recovered from the document's announcement_date in the
    ONE global processing queue -- verified 2026-08-21 to resolve 1,018/1,018
    affected documents.
    """
    g = guid.copy()
    g["quarter_source"] = "table_a"
    blank = g["quarter"].astype(str).str.strip().isin(["", "nan", "None", "NaT"])
    n_blank = int(blank.sum())
    if n_blank and not queue.empty and "doc_id" in queue.columns:
        dates = pd.to_datetime(queue.get("announcement_date"), errors="coerce")
        qmap = {str(d): QT.season_quarter(dt)
                for d, dt in zip(queue["doc_id"], dates) if pd.notna(dt)}
        rec = g.loc[blank, "source_doc_id"].astype(str).map(qmap)
        g.loc[blank, "quarter"] = rec.fillna("")
        g.loc[blank & rec.notna(), "quarter_source"] = "queue_announcement_date"
        log(f"  blank quarters: {n_blank} rows -> {int(rec.notna().sum())} repaired "
            f"from processing_queue.announcement_date")
    elif n_blank:
        log(f"  blank quarters: {n_blank} rows, queue unavailable — left blank")

    g["_qn"] = g["quarter"].map(QT.norm_q)
    g["_qo"] = g["quarter"].map(QT.q_order)
    return g


_RE_CAL_YEAR = __import__("re").compile(r"^\s*(20\d{2})\s*$")


def normalise_ppt(ppt: pd.DataFrame) -> pd.DataFrame:
    """ppt_guidance -> the guidance_tracker shape, so ONE cleaner serves both.

    Investor decks are a different animal from Table_A and the differences are
    all load-bearing (measured 2026-08-21: 408 rows / 47 companies):
      * column is `horizon`, not `horizon_fy`
      * NO cagr_pct and NO value_type -- nothing is pre-typed, so every cell goes
        through full inference
      * `value` is a bare NUMBER with the unit in a separate `unit` column
        ('30.0' + '%'  /  '20000.0' + 'cr'), where Table_A carries free text.
        They are recombined into a text cell so the shared cleaner can read it.
      * horizons include bare calendar years ('2030'), 'null', 'medium-term' and
        '5 years'. A bare 20XX is read as that FY; the vague ones do not resolve
        and their rows are dropped rather than guessed at.
      * `metric` is free text and mixed case ('Revenue', 'revenue growth',
        'Target AUM'), so it is lower-cased and left to canon_metric.
    """
    if ppt is None or ppt.empty:
        return pd.DataFrame()
    p = ppt.copy()
    hz_col = "horizon" if "horizon" in p.columns else "horizon_fy"

    def cell(v, u):
        v = str(v or "").strip()
        u = str(u or "").strip()
        if not v or v.lower() in ("nan", "none", "na"):
            return ""
        if u and u.lower() not in ("nan", "none", "na"):
            return f"{v}%" if u == "%" else f"{v} {u}"
        return v

    def hz(v):
        s = str(v or "").strip()
        m = _RE_CAL_YEAR.match(s)
        return f"FY{int(m.group(1)) % 100:02d}" if m else s

    out = pd.DataFrame({
        "isin": p["isin"], "symbol": p.get("symbol", ""),
        "company_name": p.get("company_name", ""),
        "quarter": p["quarter"],
        "metric": p["metric"].astype(str).str.strip().str.lower(),
        "guidance_type": p.get("guidance_type", ""),
        "horizon_fy": p[hz_col].map(hz),
        "value": [cell(v, u) for v, u in zip(p["value"], p.get("unit", ""))],
        "unit": p.get("unit", ""),
        "cagr_pct": None, "notes": p.get("notes", ""),
        "processed_at": p.get("processed_at", ""),
        "source_doc_id": p.get("source_doc_id", ""),
        "value_type": None, "value_num": None, "value_unit": None,
    })
    out["guidance_source"] = "presentation"
    return out[out["value"].astype(str).str.strip() != ""]


def target_quarters(guid: pd.DataFrame, args) -> list:
    """Which quarters to score, oldest -> newest so the log reads as a timeline."""
    present = (guid.loc[guid["_qo"] > 0, "_qn"].dropna().astype(str).unique().tolist())
    present = sorted(set(present), key=QT.q_order)
    # The floor applies to EVERY mode, not just --backfill: a quarter below it is
    # never scored at all.
    if args.from_quarter:
        floor = QT.q_order(QT.norm_q(args.from_quarter))
        present = [q for q in present if QT.q_order(q) >= floor]
    if not present:
        return []
    if args.quarter and args.quarter.lower() != "auto":
        want = QT.norm_q(args.quarter)
        return [want] if want in present else []
    if args.backfill:
        return present
    # default: the newest quarter actually PRESENT, not season_quarter() --
    # season_quarter flips on 1 Jul/Oct/Jan/Apr but the concalls for that quarter
    # land 3-8 weeks later, so a hard current-quarter filter renders an empty
    # page four times a year.
    newest = present[-1]
    season = QT.norm_q(QT.season_quarter())
    if newest != season:
        log(f"  season_quarter()={season} but newest quarter present={newest} "
            f"— using {newest}")
    return [newest]


# --------------------------------------------------------------------------- #
#  TTM bases                                                                    #
# --------------------------------------------------------------------------- #
def ttm_base_maps(fin3: pd.DataFrame, q_index: int):
    """POINT-IN-TIME annualised base for the window ENDING at `q_index`.

    Returns ({isin: revenue_cr}, {isin: pat_cr}, {isin: (n_quarters, scale)}).

    The window is the four quarters ending at the CONCALL's own quarter, so the
    base is what the company had actually reported when management spoke -- not
    today's trailing twelve months (user decision 2026-08-22).

    PARTIAL WINDOWS ARE SCALED UP rather than dropped (user rule):
        4 of 4 quarters -> sum            (a true TTM)
        3 of 4          -> sum x 4/3
        2 of 4          -> sum x 2
        1 of 4          -> sum x 4
    Measured on the Q1FY27 window: 1,698 companies have all four, and the rule
    recovers a further 102 (51 with three, 41 with two, 10 with one).

    The scale-up ASSUMES the missing quarters resemble the present ones, so it
    ignores seasonality -- a one-quarter base x4 for a monsoon-skewed or
    festive-skewed company can be well off. `base_quarters` and `base_method`
    are carried on every row so a thin base is visible rather than silent.
    """
    empty = ({}, {}, {})
    if fin3 is None or fin3.empty:
        return empty
    need = {"statement", "period_type", "period", "line_item", "value", "isin"}
    if not need <= set(fin3.columns):
        return empty
    q = fin3[(fin3["statement"] == "income")
             & (fin3["period_type"] == "quarterly")].copy()
    if q.empty:
        return empty
    q["_qi"] = [GP.parse_q(QT.qtr_label(p)) for p in q["period"]]
    q = q[q["_qi"].notna()].copy()
    q["_qidx"] = [GP.q_idx(*t) for t in q["_qi"]]
    q = q[q["_qidx"].isin(set(range(q_index - 3, q_index + 1)))]
    out = {"Sales": {}, "Net Profit": {}}
    meta = {}
    for li in out:
        sub = q[q["line_item"].astype(str) == li]
        for isin, s in sub.groupby(sub["isin"].astype(str)):
            s = s.drop_duplicates(subset=["_qidx"])
            vals = pd.to_numeric(s["value"], errors="coerce").dropna()
            n = len(vals)
            if n == 0:
                continue
            scale = 4.0 / n
            out[li][isin] = float(vals.sum()) * scale
            if li == "Sales" or isin not in meta:
                meta[isin] = (n, round(scale, 3))
    return out["Sales"], out["Net Profit"], meta


def symbol_isin_map(uni: pd.DataFrame, guid: pd.DataFrame) -> dict:
    """SYMBOL -> isin, universe FIRST and first-write-wins.

    fundamentals/summary.parquet is keyed by SYMBOL and carries no isin, so it
    has to be joined through a symbol map -- and the map must be authoritative.
    Building it from guidance_tracker instead put a dirty row (symbol "KEL"
    carrying VISDEM's isin) on top of the real one, so KEL's Rs 1.31 cr TTM
    overwrote VISDEM's Rs 204 cr and VISDEM scored 522%/yr instead of ~15%.
    company_universe.csv is 1:1 by construction, so it cannot collide; the
    guidance_tracker fallback only fills symbols the universe has never heard of.
    """
    out: dict = {}
    if uni is not None and not uni.empty and "isin" in uni.columns:
        for col in ("nse_symbol", "bse_symbol"):
            if col not in uni.columns:
                continue
            for i, sym in zip(uni["isin"], uni[col]):
                k = str(sym or "").strip().upper()
                if k and k not in out:
                    out[k] = str(i)
    if guid is not None and not guid.empty:
        for i, sym in zip(guid["isin"], guid["symbol"]):
            k = str(sym or "").strip().upper()
            if k and k not in out:
                out[k] = str(i)
    return out


def summary_base_maps(summ: pd.DataFrame, sym2isin: dict):
    """Today's TTM from fundamentals/summary.parquet, re-keyed by ISIN.

    This is the base build_gallery's --mode guidance uses, kept so watchlist
    numbers reconcile against gallery_guidance.html. It is ANACHRONISTIC for a
    backfilled quarter (an old target divided by a current base), which is why
    --base-source fin3 exists.

    First write wins per ISIN: if two symbols still resolve to one isin, a later
    row must not silently replace an earlier base (see symbol_isin_map).
    """
    rev, pat = {}, {}
    if summ is None or summ.empty or "symbol" not in summ.columns:
        return rev, pat
    for _, r in summ.iterrows():
        isin = sym2isin.get(str(r.get("symbol", "")).upper())
        if not isin:
            continue
        for col, dest in (("q_sales_last_4q", rev), ("q_netprofit_last_4q", pat)):
            if isin in dest:
                continue
            try:
                v = float(pd.Series(r.get(col)).astype(float).sum())
            except Exception:                                 # noqa: BLE001
                continue
            if v == v:
                dest[isin] = v
    return rev, pat


# --------------------------------------------------------------------------- #
#  identity                                                                     #
# --------------------------------------------------------------------------- #
def _bse(v) -> str:
    """BSE scrip code as a clean string -- pandas reads the CSV column as float,
    so a raw cast renders '544613.0'."""
    s = str(v or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    return s[:-2] if s.endswith(".0") else s


def identity_maps(uni: pd.DataFrame, grades: pd.DataFrame,
                  guid: pd.DataFrame) -> dict:
    """isin -> {nse_symbol, bse_code, nse_name, symbol}.

    Exchange identifiers come from company_repo/_index/company_universe.csv (NSE
    main + Emerge SME + BSE, and it HAS bse_code) -- NOT universe/master_list.csv,
    which Phase 1 overwrites daily with an NSE-only list carrying no bse_code
    (see company_deep_report.py:87-91).
    """
    out: dict = {}
    if uni is not None and not uni.empty and "isin" in uni.columns:
        for r in uni.itertuples(index=False):
            out[str(getattr(r, "isin", ""))] = {
                "nse_symbol": str(getattr(r, "nse_symbol", "") or ""),
                "bse_code": _bse(getattr(r, "bse_code", "")),
                "nse_name": str(getattr(r, "name", "") or ""),
                "symbol": "",
            }
    # `symbol` is the repo-canonical chart/OHLCV join key and still resolves for
    # BSE-only names where nse_symbol is blank.
    sym = isin_symbol_map(uni if uni is not None else pd.DataFrame(),
                          grades if grades is not None else pd.DataFrame(),
                          guid if guid is not None else pd.DataFrame())
    for isin, s in (sym or {}).items():
        out.setdefault(str(isin), {"nse_symbol": "", "bse_code": "",
                                   "nse_name": "", "symbol": ""})
        out[str(isin)]["symbol"] = str(s or "")
    return out


# --------------------------------------------------------------------------- #
#  score one quarter                                                            #
# --------------------------------------------------------------------------- #
def score_quarter(guid_q: pd.DataFrame, base_rev: dict, base_pat: dict,
                  gf1_q: pd.DataFrame, args) -> tuple:
    """(winners, rejects) for one target quarter.

    `guid_q` is already limited to the staleness window, so a company's newest
    qualifying statement inside that window wins.
    """
    best, rejects, scored = GS.best_per_key(
        guid_q, base_rev, base_pat, key="isin", min_cagr=args.min_cagr)
    src_by = {}
    if "guidance_source" in guid_q.columns:
        for i, s in zip(guid_q["isin"].astype(str), guid_q["guidance_source"]):
            src_by.setdefault(i, str(s or "concall"))
    winners = {}
    for isin, w in best.items():
        if w["cagr_pct"] < args.min_cagr:
            continue
        # how many SEPARATE statements cleared the bar -- three consistent cells
        # is firmer evidence than one borderline line
        w["n_over"] = len(scored.get(isin, [])) or 1
        w["source"] = src_by.get(str(isin), "concall")
        sl = pd.DataFrame()
        if gf1_q is not None and not gf1_q.empty:
            sl = gf1_q[(gf1_q["isin"].astype(str) == str(isin))
                       & (gf1_q["_qn"] == QT.norm_q(w.get("quarter")))]
        w["validation"] = GVAL.validate(w, sl)
        winners[isin] = w
    return winners, rejects


def build_rows(winners: dict, ident: dict, cred_by: dict, quarter: str,
               proc_by: dict, qsrc_by: dict, args, now: str,
               base_meta: dict | None = None, pf_isins=None) -> list:
    """One GW_COLS dict per qualifying company."""
    rows = []
    for isin, w in winners.items():
        v = w.get("validation") or {}
        idn = ident.get(str(isin), {})
        seed = proc_by.get((str(isin), quarter))
        rows.append({
            "isin": str(isin),
            "quarter": quarter,
            "date_added": (str(seed)[:10] if seed else now[:10]),
            "date_added_source": ("processed_at" if seed else "run_date"),
            "nse_symbol": idn.get("nse_symbol", ""),
            "bse_code": idn.get("bse_code", ""),
            "nse_name": idn.get("nse_name", "") or (w.get("company_name") or ""),
            "in_prev_quarter": False,        # filled by recompute_history()
            "n_quarters": 0,
            "quarter_streak": 0,
            "prev_quarter_had_guidance": False,
            "cagr_pct": round(float(w["cagr_pct"]), 2),
            "score_metric": w.get("metric", ""),
            "score_kind": w.get("kind", ""),
            "score_rule": w.get("rule", ""),
            "horizon_fy": str(w.get("horizon_fy") or ""),
            "guided_value": str(w.get("raw") or "")[:300],
            "value_type": w.get("value_type", ""),
            "years_used": w.get("years"),
            "base_ttm_cr": w.get("base_cr"),
            "target_cr": w.get("target_cr"),
            "base_source": args.base_source,
            "base_quarters": (base_meta or {}).get(str(isin), (None, None))[0],
            "base_scale": (base_meta or {}).get(str(isin), (None, None))[1],
            # No cap (user decision) -- but a Rs-few-crore TTM base makes ANY
            # target look explosive, so the weakness is surfaced as a flag the
            # table and the card can show, not silently filtered.
            "base_suspect": bool(w.get("kind") == "absolute"
                                 and (w.get("base_cr") or 0) < args.min_base_cr),
            "validation_verdict": v.get("verdict", GVAL.NO_EVIDENCE),
            "evidence_stmt": str(v.get("evidence_stmt") or "")[:400],
            "evidence_num": v.get("evidence_num"),
            "evidence_delta_pct": v.get("evidence_delta_pct"),
            "symbol": idn.get("symbol", "") or str(w.get("symbol") or ""),
            "company_name": str(w.get("company_name") or ""),
            "guidance_source": w.get("source", "concall"),
            "source_doc_id": str(w.get("source_doc_id") or ""),
            "quarter_source": qsrc_by.get((str(isin), quarter), "table_a"),
            "cred_score": cred_by.get(str(isin)),
            # do I already own this? load_portfolio_isins returns None when no
            # holdings file is on Drive -- that is "unknown", not "not held",
            # so it must not render as a confident No.
            "in_pf": (None if pf_isins is None
                      else bool(str(isin) in pf_isins)),
            "n_rows_over_min": int(w.get("n_over", 1)),
            "min_cagr_used": args.min_cagr,
            "max_stale_q_used": args.max_stale_q,
            "tracking_from": args.from_quarter or "",
            "first_qualified_at": now,
            "last_seen_at": now,
            "as_of": now,
        })
    return rows


# --------------------------------------------------------------------------- #
#  upsert  (earnings_calendar.persist_calendar idiom, verbatim)                 #
# --------------------------------------------------------------------------- #
def upsert_watchlist(old: pd.DataFrame, fresh: pd.DataFrame, now: str,
                     replace_quarters=None):
    """(combined, n_new, n_updated) keyed on (isin, quarter).

    date_added and first_qualified_at are carried forward for a key that already
    exists, so re-running never bumps a name to the top of a date_added sort.
    An EMPTY `fresh` returns the old table untouched -- a quarter with zero
    qualifiers, or a read failure, must never truncate accumulated history.
    """
    if fresh is None or fresh.empty:
        return (old if old is not None else pd.DataFrame(columns=GW_COLS)), 0, 0
    fresh = fresh.copy()
    n_old = 0
    if old is not None and not old.empty:
        n_old = len(old)
        # prev is built from the FULL old table, BEFORE any quarter is dropped,
        # so a --rebuild still carries date_added forward for names that survive.
        prev = {(str(i), str(q)): (da, ds, fq) for i, q, da, ds, fq in
                zip(old["isin"], old["quarter"], old["date_added"],
                    old["date_added_source"], old["first_qualified_at"])}
        if replace_quarters:
            # A cell that no longer qualifies under the CURRENT rules must be
            # able to leave. Without this the table only ever grows, so a fix to
            # the cleaner never takes effect on already-published rows (the
            # LVL:-share rows survived their own fix this way).
            old = old[~old["quarter"].astype(str).isin(set(replace_quarters))]
        keys = list(zip(fresh["isin"].astype(str), fresh["quarter"].astype(str)))
        fresh["date_added"] = [prev[k][0] if k in prev else d
                               for k, d in zip(keys, fresh["date_added"])]
        fresh["date_added_source"] = [prev[k][1] if k in prev else d
                                      for k, d in zip(keys,
                                                      fresh["date_added_source"])]
        fresh["first_qualified_at"] = [prev[k][2] if k in prev else d
                                       for k, d in zip(keys,
                                                       fresh["first_qualified_at"])]
        n_updated = sum(1 for k in keys if k in prev)
        combined = pd.concat([old, fresh], ignore_index=True)
    else:
        n_updated = 0
        combined = fresh
    combined = (combined.drop_duplicates(subset=GW_KEY, keep="last")
                .reset_index(drop=True))
    for c in GW_COLS:
        if c not in combined.columns:
            combined[c] = None
    combined = combined[GW_COLS]
    return combined, len(combined) - n_old, n_updated


# --------------------------------------------------------------------------- #
#  history  (build_signal_membership full-recompute idiom)                      #
# --------------------------------------------------------------------------- #
def recompute_history(df: pd.DataFrame, guid_all: pd.DataFrame) -> pd.DataFrame:
    """Derive in_prev_quarter / n_quarters / quarter_streak from scratch.

    Never incremented in place, so an interrupted run or a quarter added out of
    order by a later backfill self-heals on the next run.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["_qo"] = out["quarter"].map(QT.q_order)
    out["_qi"] = [GP.q_idx(*GP.parse_q(q)) if GP.parse_q(q) else None
                  for q in out["quarter"]]

    # any guidance at all for (isin, quarter) -- distinguishes "stopped guiding
    # big" from "no concall was extracted"
    had = set()
    if guid_all is not None and not guid_all.empty and "_qn" in guid_all.columns:
        had = {(str(i), str(q)) for i, q in zip(guid_all["isin"], guid_all["_qn"])
               if q}

    n_q, in_prev, streak, prev_had = [], [], [], []
    # NOTE: iterate with zip, not itertuples -- itertuples renames any column
    # starting with an underscore to a positional name (_1, _2 ...).
    by_isin = {i: sorted(v.dropna().unique())
               for i, v in out.groupby(out["isin"].astype(str))["_qi"]}
    for isin, qi in zip(out["isin"].astype(str), out["_qi"]):
        qs = by_isin.get(isin, [])
        if qi is None or (isinstance(qi, float) and qi != qi):
            n_q.append(0); in_prev.append(False); streak.append(0)
            prev_had.append(False)
            continue
        qi = int(qi)
        n_q.append(sum(1 for x in qs if x <= qi))
        in_prev.append((qi - 1) in qs)
        s, cur = 0, qi
        while cur in qs:
            s += 1
            cur -= 1
        streak.append(s)
        prev_had.append((isin, QT.norm_q(GP.q_label(qi - 1))) in had)
    out["n_quarters"] = n_q
    out["in_prev_quarter"] = in_prev
    out["quarter_streak"] = streak
    out["prev_quarter_had_guidance"] = prev_had
    return out.drop(columns=["_qo", "_qi"])[GW_COLS]


# --------------------------------------------------------------------------- #
#  main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:                                            # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-cagr", type=float, default=50.0,
                    help="qualifying annual growth %% (default 50)")
    ap.add_argument("--max-stale-q", type=int, default=2,
                    help="how many quarters back the source concall may be "
                         "(default 2)")
    ap.add_argument("--require-validation", default="confirmed,consistent",
                    help="verdicts allowed to publish; 'all' keeps NO_EVIDENCE too")
    ap.add_argument("--base-source", choices=["fin3", "summary"], default="fin3",
                    help="fin3 (DEFAULT) = financials_3stmt, the four quarters "
                         "ending at the CONCALL's own quarter, partial windows "
                         "scaled up (3q x4/3, 2q x2, 1q x4); summary = today's "
                         "TTM from fundamentals/summary.parquet")
    ap.add_argument("--sources", default="concall,presentation",
                    help="comma list: concall (guidance_tracker) and/or "
                         "presentation (ppt_guidance). Decks carry no pre-typed "
                         "numbers, so every deck cell goes through full inference.")
    ap.add_argument("--base-fallback", choices=["summary", "none"], default="summary")
    ap.add_argument("--min-base-cr", type=float, default=25.0,
                    help="TTM base below this flags base_suspect (NOT a filter; "
                         "no capping, per the 2026-08-21 decision)")
    ap.add_argument("--quarter", default="auto")
    ap.add_argument("--backfill", action="store_true",
                    help="score every quarter present, oldest first")
    ap.add_argument("--from-quarter", dest="from_quarter", default="",
                    help="HARD floor, e.g. Q1FY27. Quarters before it are never "
                         "scored, and with --rebuild any already-published rows "
                         "below it are dropped. Q4FY26 measured 64%% empty cells "
                         "and 30%% horizon-tagged, vs Q1FY27 at 0%% / 100%%.")
    ap.add_argument("--rebuild", action="store_true",
                    help="drop and re-derive every quarter being scored, so rows "
                         "that no longer qualify under the current cleaning rules "
                         "LEAVE the table. date_added is still carried forward "
                         "for names that survive.")
    ap.add_argument("--dump-csv", default="")
    ap.add_argument("--dump-rejects", default="",
                    help="write every cell the cleaner refused to score, with the "
                         "rule that refused it, so nothing is discarded invisibly")
    ap.add_argument("--lock-wait-min", type=float, default=0.0,
                    help="minutes to wait for the shared _extract.lock before "
                         "giving up (default 0 = skip immediately). Phase 2 and "
                         "the backfill hold this lock for long stretches during "
                         "extraction, so a 0-wait run can lose the race for "
                         "hours; 15 is a sensible value for a scheduled run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report; no Drive write, no lock")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()

    allow = {v.strip().upper() for v in args.require_validation.split(",")
             if v.strip()}
    keep_all = "ALL" in allow

    drive = get_drive()
    idx = _folder(drive, "company_repo/_index")

    if not args.dry_run:
        if not acquire_lock(drive, idx, _LOCK_NAME, _LOCK_OWNER,
                            max_age_min=360, wait_min=args.lock_wait_min):
            log("  lock unavailable — exiting cleanly, the next run resumes")
            return 0
        atexit.register(release_lock, drive, idx, _LOCK_NAME)

    log("loading tables…")
    guid = load_parquet(drive, idx, GUIDANCE_NAME, GUIDANCE_COLS)
    if guid.empty:
        log("guidance_tracker.parquet is empty — nothing to do.")
        return 0
    queue = _read_any(drive, idx, QUEUE_NAME)
    gf1 = _read_any(drive, idx, GF1_NAME)
    fin3 = _read_any(drive, idx, FIN3_NAME) if args.base_source == "fin3" \
        else pd.DataFrame()
    cred = _read_any(drive, idx, CRED_NAME)
    grades = _read_any(drive, idx, GRADES_NAME)
    uni = _read_any(drive, idx, UNIVERSE_NAME)
    summ = _read_any(drive, _folder(drive, "fundamentals"), "summary.parquet")
    old = load_parquet(drive, idx, GW_NAME, GW_COLS)
    log(f"  guidance {len(guid)} rows · gf1 {len(gf1)} · universe {len(uni)} · "
        f"existing watchlist {len(old)}")

    guid = resolve_quarters(guid, queue)
    if not gf1.empty and "quarter" in gf1.columns:
        gf1 = gf1.copy()
        gf1["_qn"] = gf1["quarter"].map(QT.norm_q)

    ident = identity_maps(uni, grades, guid)
    sym2isin = symbol_isin_map(uni, guid)
    sum_rev, sum_pat = summary_base_maps(summ, sym2isin)
    log(f"  TTM bases resolved: revenue {len(sum_rev)} · pat {len(sum_pat)} isins")

    pf_isins = load_portfolio_isins(drive, os.environ["GDRIVE_FOLDER_ID"])
    log(f"  portfolio: {len(pf_isins) if pf_isins is not None else 'no holdings file'} "
        f"{'holdings' if pf_isins is not None else '(in_pf left unknown)'}")

    cred_by = {}
    if not cred.empty and {"isin", "cred_score"} <= set(cred.columns):
        c = cred.dropna(subset=["cred_score"])
        cred_by = {str(i): float(v) for i, v in
                   zip(c["isin"], pd.to_numeric(c["cred_score"], errors="coerce"))
                   if v == v}

    guid["guidance_source"] = "concall"
    srcs = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    if "presentation" in srcs:
        ppt = normalise_ppt(_read_any(drive, idx, PPT_NAME))
        if not ppt.empty:
            ppt = resolve_quarters(ppt, queue)
            n_co = ppt["isin"].nunique()
            guid = pd.concat([guid, ppt], ignore_index=True)
            log(f"  + presentations: {len(ppt)} rows / {n_co} companies folded in")
    if "concall" not in srcs:
        guid = guid[guid["guidance_source"] != "concall"]

    rp = guid[guid["metric"].astype(str).str.lower().isin(
        ["revenue", "sales", "pat", "profit", "net profit", "earnings",
         "revenue growth", "sales growth", "pat growth", "profit growth"])].copy()

    quarters = target_quarters(guid, args)
    if not quarters:
        log("No quarter to score.")
        return 0
    log(f"scoring {len(quarters)} quarter(s): {', '.join(quarters)}")

    all_rows, all_rejects = [], []
    for q in quarters:
        qo = QT.q_order(q)
        win = rp[(rp["_qo"] <= qo) & (rp["_qo"] >= qo - args.max_stale_q)]
        if win.empty:
            log(f"  {q}: no rows in the {args.max_stale_q}-quarter window — skipped")
            continue
        if args.base_source == "fin3":
            pq = GP.parse_q(q)
            b_rev, b_pat, b_meta = (ttm_base_maps(fin3, GP.q_idx(*pq)) if pq
                                    else ({}, {}, {}))
            n_full = sum(1 for n, _ in b_meta.values() if n >= 4)
            log(f"  {q}: fin3 bases {len(b_rev)} isins "
                f"({n_full} full 4-quarter, {len(b_rev) - n_full} scaled up)")
            if args.base_fallback == "summary":
                b_rev = {**sum_rev, **b_rev}
                b_pat = {**sum_pat, **b_pat}
            if not b_rev:
                log(f"  {q}: no fin3 base at all — "
                    f"use --base-fallback summary to score it")
        else:
            b_rev, b_pat, b_meta = sum_rev, sum_pat, {}

        gq = gf1[gf1["_qn"] == q] if (not gf1.empty and "_qn" in gf1.columns) \
            else pd.DataFrame()
        winners, rejects = score_quarter(win, b_rev, b_pat, gq, args)
        kept = {i: w for i, w in winners.items()
                if keep_all or (w["validation"]["verdict"] in allow
                                or w["validation"]["verdict"].lower() in allow
                                or w["validation"]["verdict"] in
                                {a.upper() for a in allow})}

        proc_by, qsrc_by = {}, {}
        for r in win.itertuples(index=False):
            k = (str(r.isin), q)
            p = str(getattr(r, "processed_at", "") or "")
            if p and (k not in proc_by or p < proc_by[k]):
                proc_by[k] = p
            qsrc_by.setdefault(k, getattr(r, "quarter_source", "table_a"))

        rows = build_rows(kept, ident, cred_by, q, proc_by, qsrc_by, args,
                          _now(), b_meta, pf_isins)
        all_rows.extend(rows)
        for isin, lst in rejects.items():
            for x in lst:
                all_rejects.append({"quarter": q, "isin": isin,
                                    "symbol": x.get("symbol"),
                                    "reason": x.get("reject"),
                                    "value": str(x.get("raw"))[:120]})
        verdicts = pd.Series([w["validation"]["verdict"]
                              for w in winners.values()]).value_counts().to_dict()
        log(f"  {q}: rows_in={len(win)} scored>={args.min_cagr:.0f}%="
            f"{len(winners)} published={len(rows)}  {verdicts}")

    fresh = pd.DataFrame(all_rows)
    if args.from_quarter and args.rebuild and old is not None and not old.empty:
        floor = QT.q_order(QT.norm_q(args.from_quarter))
        keep = old["quarter"].map(QT.q_order).fillna(-1) >= floor
        if (~keep).any():
            log(f"  floor {QT.norm_q(args.from_quarter)}: dropping "
                f"{int((~keep).sum())} already-published rows from earlier quarters")
        old = old[keep]
    n_before = len(old) if old is not None else 0
    combined, n_new, n_upd = upsert_watchlist(
        old, fresh, _now(), replace_quarters=(quarters if args.rebuild else None))
    if args.rebuild:
        log(f"  rebuild: {n_before} rows -> {len(combined)} "
            f"({max(0, n_before - len(combined))} dropped as no longer qualifying)")
    combined = recompute_history(combined, guid)

    log(f"watchlist: {len(combined)} rows · {combined['isin'].nunique() if not combined.empty else 0} "
        f"companies · {combined['quarter'].nunique() if not combined.empty else 0} quarters "
        f"· new={n_new} updated={n_upd} · rejects logged={len(all_rejects)}")

    if args.dump_csv and not combined.empty:
        combined.to_csv(args.dump_csv, index=False)
        log(f"  wrote {args.dump_csv}")
    if args.dump_rejects and all_rejects:
        rj = pd.DataFrame(all_rejects)
        rj["rule"] = rj["reason"].astype(str).str.split(":").str[0]
        rj.sort_values(["rule", "quarter"]).to_csv(args.dump_rejects, index=False)
        log(f"  wrote {args.dump_rejects} ({len(rj)} rejected cells)")

    if args.dry_run:
        if not combined.empty:
            top = combined.sort_values(["date_added", "cagr_pct"],
                                       ascending=[False, False]).head(20)
            cols = ["date_added", "nse_symbol", "bse_code", "quarter", "cagr_pct",
                    "score_metric", "score_rule", "validation_verdict",
                    "in_prev_quarter", "n_quarters", "base_ttm_cr",
                    "base_suspect"]
            log("DRY-RUN — top 20 by date_added, nothing written:")
            print(top[[c for c in cols if c in top.columns]].to_string(index=False))
        if all_rejects:
            rj = pd.DataFrame(all_rejects)
            log("DRY-RUN — reject reasons:")
            print(rj["reason"].str.split(":").str[0].value_counts()
                  .head(12).to_string())
        log("DRY-RUN — no Drive write.")
        return 0

    if combined.empty:
        log("Nothing to write.")
        return 0
    save_parquet(drive, idx, GW_NAME, combined)
    log(f"wrote {GW_NAME} ({len(combined)} rows)")
    return 0


def _self_test() -> int:
    fails = []

    def chk(label, got, want):
        if got != want:
            fails.append(f"{label}: got {got!r}, want {want!r}")

    now = "2026-08-21T00:00:00+00:00"
    base = {c: None for c in GW_COLS}

    def row(isin, q, added, src="processed_at", cagr=60.0):
        return {**base, "isin": isin, "quarter": q, "date_added": added,
                "date_added_source": src, "first_qualified_at": added,
                "cagr_pct": cagr, "last_seen_at": now, "as_of": now}

    # date_added is PINNED: a re-run must not bump it
    old = pd.DataFrame([row("I1", "Q1FY27", "2026-08-01")])
    fresh = pd.DataFrame([row("I1", "Q1FY27", "2026-08-21", "run_date", 75.0)])
    comb, n_new, n_upd = upsert_watchlist(old, fresh, now)
    chk("date_added pinned", comb.iloc[0]["date_added"], "2026-08-01")
    chk("source pinned", comb.iloc[0]["date_added_source"], "processed_at")
    chk("score refreshed", comb.iloc[0]["cagr_pct"], 75.0)
    chk("no new rows", n_new, 0)
    chk("one updated", n_upd, 1)

    # an empty fresh must NEVER truncate the table
    comb2, n2, u2 = upsert_watchlist(old, pd.DataFrame(), now)
    chk("empty fresh keeps old", len(comb2), 1)
    chk("empty fresh adds none", n2, 0)

    # a new quarter appends rather than replacing
    comb3, n3, _ = upsert_watchlist(old, pd.DataFrame([row("I1", "Q2FY27",
                                                          "2026-11-01")]), now)
    chk("new quarter appended", len(comb3), 2)
    chk("n_new counts it", n3, 1)

    # --rebuild: a row that no longer qualifies must LEAVE, but date_added for a
    # survivor must still be pinned.
    old2 = pd.DataFrame([row("I1", "Q1FY27", "2026-08-01"),
                         row("I2", "Q1FY27", "2026-08-02"),
                         row("I1", "Q4FY26", "2026-05-01")])
    keep = pd.DataFrame([row("I1", "Q1FY27", "2026-08-21", "run_date", 90.0)])
    comb4, _, _ = upsert_watchlist(old2, keep, now, replace_quarters=["Q1FY27"])
    chk("rebuild drops the stale row", len(comb4), 2)
    chk("rebuild kept the survivor",
        set(zip(comb4["isin"], comb4["quarter"])),
        {("I1", "Q1FY27"), ("I1", "Q4FY26")})
    chk("rebuild still pins date_added",
        comb4[comb4["quarter"] == "Q1FY27"].iloc[0]["date_added"], "2026-08-01")
    chk("rebuild leaves other quarters alone",
        comb4[comb4["quarter"] == "Q4FY26"].iloc[0]["date_added"], "2026-05-01")

    # history: consecutive quarters -> streak 2; a GAP is not continuity
    h = pd.DataFrame([row("I1", "Q4FY26", "2026-05-01"),
                      row("I1", "Q1FY27", "2026-08-01"),
                      row("I2", "Q1FY26", "2025-08-01"),
                      row("I2", "Q1FY27", "2026-08-01")])
    out = recompute_history(h, pd.DataFrame())
    cur = out[(out["isin"] == "I1") & (out["quarter"] == "Q1FY27")].iloc[0]
    chk("I1 n_quarters", cur["n_quarters"], 2)
    chk("I1 in_prev_quarter", cur["in_prev_quarter"], True)
    chk("I1 streak", cur["quarter_streak"], 2)
    gap = out[(out["isin"] == "I2") & (out["quarter"] == "Q1FY27")].iloc[0]
    chk("I2 n_quarters counts both", gap["n_quarters"], 2)
    chk("I2 gap is not continuity", gap["in_prev_quarter"], False)
    chk("I2 streak resets", gap["quarter_streak"], 1)
    first = out[(out["isin"] == "I1") & (out["quarter"] == "Q4FY26")].iloc[0]
    chk("cumulative not lifetime", first["n_quarters"], 1)

    # prev_quarter_had_guidance separates a coverage gap from going quiet
    g = pd.DataFrame({"isin": ["I2"], "_qn": ["Q4FY26"]})
    out2 = recompute_history(h, g)
    g2 = out2[(out2["isin"] == "I2") & (out2["quarter"] == "Q1FY27")].iloc[0]
    chk("prev had guidance", g2["prev_quarter_had_guidance"], True)

    chk("schema stable", list(out.columns), GW_COLS)

    for f in fails:
        print("FAIL " + f)
    print(("self-test FAILED (%d)" % len(fails)) if fails else "self-test OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
