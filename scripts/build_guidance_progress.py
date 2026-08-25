r"""
build_guidance_progress.py — weekly snapshot of every OPEN guidance commitment.

WHAT IT PRODUCES  (company_repo/_index/)
  guidance_progress.parquet          current state, one row per open commitment
  guidance_progress_history.parquet  append-only weekly snapshot -> delta_week

WHY (see guidance_progress.py for the full argument)
  build_pead_flags compares ANNUAL rows and skips a commitment while the horizon is
  still open, so guidance_vs_actual carries ~24k TOO_EARLY rows that nothing ever
  revisits. This measures those mid-flight: how much of the guided number has landed
  so far versus how much of the horizon has elapsed.

INPUTS  (all existing tables; nothing new is scraped or called)
  guidance_tracker.parquet     concall Table_A guidance      (36k rows / 1.3k cos)
  ppt_guidance / ar_guidance   presentation + annual report  (merged in)
  mgmt_credibility.parquet     GF_TRACK prose guidance       — the ONLY source that
                               carries Order Book and Capex targets
  financials_3stmt.parquet     quarterly actuals for revenue/ebitda/pat/margin
  announcement_ledger.parquet  order_value_cr -> order-inflow actuals
  company_universe.csv         names; portfolio file -> in_pf

RE-PARSING ON READ
  The guided number is re-parsed from the raw `value` text through
  guidance_value.parse_guidance_value rather than trusting the stored value_type.
  Stored rows predate the 2026-08-15 currency fix, so a USD target still reads as
  INR_cr on Drive; re-parsing means this table is correct whether or not
  sanitize_guidance_tracker has been run.

Usage:
    python scripts/build_guidance_progress.py --dry-run
    python scripts/build_guidance_progress.py --dry-run --names POWERMECH,KLBRENG-B
    python scripts/build_guidance_progress.py
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import guidance_progress as GP  # noqa: E402
import quarterly_table as QT  # noqa: E402
from _extractor_base import (get_drive, get_or_create_subfolder, load_parquet,  # noqa: E402
                             save_parquet, load_portfolio_isins, log)
from build_pead_flags import (AR_G_COLS, FIN3_COLS, GUIDANCE_COLS,  # noqa: E402
                              MGMT_CRED_COLS, PPT_G_COLS)
from guidance_value import parse_guidance_value  # noqa: E402
from ingest_announcements import LEDGER_COLS  # noqa: E402

OUT_NAME = "guidance_progress.parquet"
HIST_NAME = "guidance_progress_history.parquet"

PROGRESS_COLS = [
    "commit_id", "isin", "symbol", "company_name", "metric",
    "guided_text", "guided_value", "guided_unit", "kind",
    "target_period", "horizon_start", "horizon_end",
    "periods_total", "periods_elapsed", "actual_to_date", "actual_source",
    "pct_of_target", "time_pct", "pace_ratio", "status", "delta_week",
    "guid_quarter", "guidance_source", "source_doc_id",
    "univ_rank", "in_pf", "as_of",
]
HIST_COLS = ["week_start", "commit_id", "pct_of_target", "actual_to_date",
             "status", "as_of"]

TOP_N = 200


def _week_start(d: date | None = None) -> str:
    """Monday of the given week — the snapshot key."""
    d = d or date.today()
    return (d - timedelta(days=d.weekday())).isoformat()


def _f(v):
    try:
        x = float(v)
        return None if pd.isna(x) else x
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ #
#  1. commitments                                                      #
# ------------------------------------------------------------------ #

def _commitments_from_tracker(g: pd.DataFrame, source: str) -> list[dict]:
    """guidance_tracker / ppt_guidance / ar_guidance -> commitment dicts."""
    out = []
    for r in g.itertuples(index=False):
        raw = getattr(r, "value", "")
        metric = GP.canon_metric(getattr(r, "metric", ""))
        horizon = str(getattr(r, "horizon_fy", "") or "")
        p = parse_guidance_value(raw, metric=metric, horizon=horizon,
                                 typed_prompt=True)
        out.append({
            "isin": str(getattr(r, "isin", "") or ""),
            "symbol": str(getattr(r, "symbol", "") or ""),
            "company_name": str(getattr(r, "company_name", "") or ""),
            "metric": metric,
            "guided_text": str(raw or ""),
            "value_type": p["value_type"],
            # an amount is always compared in Rs crore; a rate/level in its own
            # unit, after rescuing any margin written as a decimal fraction
            "guided_value": (p["value_num_inr_cr"]
                             if p["value_type"] in ("absolute_inr", "absolute_usd")
                             else GP.normalise_rate(p["value_num"], raw,
                                                    p["value_type"])),
            "guided_unit": ("INR_cr" if p["value_type"] in
                            ("absolute_inr", "absolute_usd") else p["value_unit"]),
            "target_period": horizon,
            "guid_quarter": str(getattr(r, "quarter", "") or ""),
            "guidance_source": source,
            "source_doc_id": str(getattr(r, "source_doc_id", "") or ""),
        })
    return out


def _commitments_from_credibility(c: pd.DataFrame) -> list[dict]:
    """mgmt_credibility (GF_TRACK) -> commitment dicts.

    This is the ONLY table carrying Order Book and Capex targets, and its
    `guidance_given` is prose ("INR 12,000 crores inflow"), so the number has to be
    parsed out. Only rows still open are of interest — a row already scored
    DELIVERED/MISSED has a verdict and belongs to build_pead_flags, not here.
    """
    out = []
    for r in c.itertuples(index=False):
        verdict = str(getattr(r, "verdict", "") or "").upper()
        if verdict in ("DELIVERED", "EXCEEDED", "MISSED", "PARTIAL"):
            continue
        raw = getattr(r, "guidance_given", "")
        metric = GP.canon_metric(getattr(r, "metric", ""))
        target = str(getattr(r, "target_period", "") or "")
        p = parse_guidance_value(raw, metric=metric, horizon=target)
        out.append({
            "isin": str(getattr(r, "isin", "") or ""),
            "symbol": str(getattr(r, "symbol", "") or ""),
            "company_name": str(getattr(r, "company_name", "") or ""),
            "metric": metric,
            "guided_text": str(raw or ""),
            "value_type": p["value_type"],
            "guided_value": (p["value_num_inr_cr"]
                             if p["value_type"] in ("absolute_inr", "absolute_usd")
                             else GP.normalise_rate(p["value_num"], raw,
                                                    p["value_type"])),
            "guided_unit": ("INR_cr" if p["value_type"] in
                            ("absolute_inr", "absolute_usd") else p["value_unit"]),
            "target_period": target,
            "guid_quarter": str(getattr(r, "quarter", "") or ""),
            "guidance_source": "gf_track",
            "source_doc_id": str(getattr(r, "source_doc_id", "") or ""),
        })
    return out


# ------------------------------------------------------------------ #
#  2. actuals                                                          #
# ------------------------------------------------------------------ #

def _financial_series(fin: pd.DataFrame) -> dict:
    """{(isin, line_item): {quarter_idx: value}} from quarterly income rows.

    Screener period columns ('Jun 2026') are mapped through quarterly_table.qtr_label
    -> 'Q1 FY27' -> a quarter index, so this file's convention matches the rest of
    the repo's RESULTS convention exactly.
    """
    q = fin[(fin["statement"] == "income")
            & (fin["period_type"] == "quarterly")].copy()
    if q.empty:
        return {}
    q["qi"] = [GP.parse_q(QT.qtr_label(p)) for p in q["period"]]
    q = q[q["qi"].notna()]
    out: dict = {}
    for r in q.itertuples(index=False):
        v = _f(r.value)
        if v is None:
            continue
        out.setdefault((str(r.isin), str(r.line_item)), {})[GP.q_idx(*r.qi)] = v
    return out


def _order_series(led: pd.DataFrame) -> dict:
    """{isin: {quarter_idx: Rs cr of order wins booked}} from the announcement ledger.

    Only rows carrying a parsed value contribute. A quarter with order wins but no
    parseable amount therefore under-counts — the bias is always DOWNWARD, never up,
    and the coverage numbers are reported so it is visible rather than hidden.
    """
    if led.empty:
        return {}
    o = led[led["event_type"].astype(str) == "order_win"].copy()
    o["val"] = pd.to_numeric(o["order_value_cr"], errors="coerce")
    o = o[o["val"].notna() & (o["val"] > 0)]
    out: dict = {}
    for r in o.itertuples(index=False):
        try:
            d = pd.to_datetime(r.ann_date)
        except (ValueError, TypeError):
            continue
        if pd.isna(d):
            continue
        # q_from_date, NOT qtr_label: an announcement lands on any day of any
        # month, and qtr_label only understands quarter-END months. Routing dates
        # through it silently dropped 52 of 57 parsed order wins — every July and
        # August filing — leaving 5 companies with an order series.
        k = GP.q_from_date(d)
        m = out.setdefault(str(r.isin), {})
        m[k] = m.get(k, 0.0) + float(r.val)
    return out


def _ledger_coverage(led: pd.DataFrame) -> int | None:
    """First quarter the announcement ledger FULLY covers.

    The ledger is a rolling window, not history: it currently starts mid-June 2026.
    A quarter only counts as covered if the ledger opened on or before its first
    day, otherwise "no orders booked" means "we were not watching", not "none won".
    """
    if led.empty:
        return None
    d = pd.to_datetime(led["ann_date"], errors="coerce").min()
    if pd.isna(d):
        return None
    q = GP.q_from_date(d)
    # partially-covered opening quarter does not count as covered
    return q if d.date() <= _quarter_first_day(q) else q + 1


def _quarter_first_day(qi: int) -> date:
    fy, q = GP.idx_q(qi)
    month = {1: 4, 2: 7, 3: 10, 4: 1}[q]
    year = 2000 + fy + (0 if q == 4 else -1)
    return date(year, month, 1)


def _netblock_series(fin: pd.DataFrame) -> dict:
    """{isin: {fy: Net Block}} — the capex proxy, ANNUAL ONLY.

    Screener publishes the balance sheet annually, so there is no quarterly capex
    series to be had: verified, all 11,213 Net Block rows are period_type='annual'.
    A capex commitment therefore reads NO_DATA until its fiscal year closes. That is
    a real limitation of the source, not something to paper over with a fabricated
    quarterly split.
    """
    b = fin[(fin["line_item"] == "Net Block")
            & (fin["period_type"] == "annual")].copy()
    out: dict = {}
    for r in b.itertuples(index=False):
        v = _f(r.value)
        qi = GP.parse_q(QT.qtr_label(str(r.period)))
        if v is None or not qi:
            continue
        out.setdefault(str(r.isin), {})[qi[0]] = v
    return out


# ------------------------------------------------------------------ #
#  3. the build                                                        #
# ------------------------------------------------------------------ #

def _row(c: dict, isin: str, names: dict, start: int, end: int, kind,
         src: str, p: dict, pf: set, now: str) -> dict:
    """One guidance_progress row. Shared so the normal path and the
    insufficient-coverage path cannot drift apart."""
    return {
        "commit_id": GP.commit_id(isin, c["metric"], c["target_period"],
                                  c["guided_text"]),
        "isin": isin, "symbol": c["symbol"],
        "company_name": c["company_name"] or names.get(isin, ""),
        "metric": c["metric"], "guided_text": c["guided_text"][:300],
        "guided_value": c["guided_value"], "guided_unit": c["guided_unit"],
        "kind": kind or "",
        "target_period": c["target_period"],
        "horizon_start": GP.q_label(start), "horizon_end": GP.q_label(end),
        "periods_total": p["periods_total"], "periods_elapsed": p["periods_elapsed"],
        "actual_to_date": p["actual_to_date"], "actual_source": src,
        "pct_of_target": p["pct_of_target"], "time_pct": p["time_pct"],
        "pace_ratio": p["pace_ratio"], "status": p["status"],
        "delta_week": None,
        "guid_quarter": c["guid_quarter"],
        "guidance_source": c["guidance_source"],
        "source_doc_id": c["source_doc_id"],
        "univ_rank": None, "in_pf": isin in pf, "as_of": now,
    }


def build(commits: list[dict], fin_series: dict, orders: dict, netblock: dict,
          now_q: int, cal_q: int, pf: set, names: dict, now: str,
          ledger_from: int | None = None) -> pd.DataFrame:
    """Pure-ish assembly: every input is already a plain dict/list.

    now_q       — last quarter whose results are being reported (season_quarter).
    cal_q       — the quarter in progress right now. These differ by one, and which
                  applies depends on whether the feed has a reporting lag:
                  financials do, announcements do not.
    ledger_from — earliest quarter the announcement ledger covers, so an order
                  commitment older than the ledger is not given a fake verdict.
    """
    rows = []
    for c in commits:
        isin = c["isin"]
        if not isin:
            continue
        win = GP.resolve_window(c["target_period"], c["guid_quarter"])
        if not win:
            continue
        start, end, _basis = win
        if end < now_q:
            continue                      # horizon fully in the past — already judged

        kind = GP.classify_kind(c["metric"], c["value_type"])
        line, _ = GP.METRIC_ACTUAL.get(c["metric"], (None, None))

        actuals: dict = {}
        elapsed_idx = None
        src = "none"
        if kind in ("level", "growth", "margin") and line:
            actuals = fin_series.get((isin, line), {})
            src = f"financials_3stmt:{line}"
            if kind == "level":
                # Guard against a SEGMENT/PROJECT figure filed under the parent
                # metric. Compared against the prior year of the same line, since
                # that is the only company-level scale we can be sure of.
                prior = sum(v for k, v in actuals.items()
                            if start - 4 <= k < start)
                if prior <= 0:
                    # financials_3stmt keeps only 12 quarters, so a horizon that
                    # opened years ago has no true prior year on file. The earliest
                    # four quarters we DO have are still the right order of
                    # magnitude for "is this a company-level number at all?" —
                    # without this, AARTIPHARM's Rs 250 cr project target scored
                    # 2,170% against Rs 5,425 cr of company revenue.
                    ks = sorted(actuals)[:4]
                    prior = sum(actuals[k] for k in ks)
                if GP.is_segment_target(c["guided_value"], prior):
                    kind = None
                    src = (f"not comparable: guided {c['guided_value']:,.0f} cr is "
                           f"{c['guided_value'] / prior * 100:.0f}% of the prior "
                           f"year's {line} ({prior:,.0f} cr) — a segment or project "
                           f"target, not a company-level one")
        elif kind == "orders":
            actuals = orders.get(isin, {})
            # A quarter with no order win is a real zero, so elapsed follows the
            # CALENDAR here, not the data (see compute_progress). The boundary is
            # the quarter IN PROGRESS (cal_q), not the last one reported (now_q):
            # order wins are announced the day they happen, with no reporting lag,
            # so this week's win must count — that is the whole point of a weekly
            # mail. The in-progress quarter counts as a full period, which reads
            # slightly harsh early in a quarter; the bias is deliberately toward
            # BEHIND rather than toward flattering the company.
            elapsed_idx = {i for i in range(start, end + 1) if i <= cal_q}
            src = "announcement_ledger:order_win"
            # COVERAGE GATE. The ledger does not go back forever — it currently
            # begins mid-June 2026, while an FY27 target's horizon opened on
            # 1 Apr 2026. Scoring "0 booked at 50% elapsed -> AT_RISK" off a window
            # we simply cannot see is a fabricated verdict, and it fired on BEL,
            # HCC, KEC, AFCONS and DBL. So: when the horizon opens before the
            # ledger does, report the amount actually observed and refuse the
            # verdict. This resolves itself as the ledger accumulates.
            if ledger_from is not None and start < ledger_from:
                booked = round(sum(v for k, v in actuals.items()
                                   if start <= k <= end), 4)
                blank = GP.compute_progress(None, None, (start, end), {})
                blank["actual_to_date"] = booked
                rows.append(_row(
                    c, isin, names, start, end, None,
                    f"announcement_ledger:order_win - INCOMPLETE: ledger starts "
                    f"{GP.q_label(ledger_from)}, horizon opens {GP.q_label(start)}; "
                    f"Rs {booked:,.0f} cr observed since then, pace not assessable",
                    blank, pf, now))
                continue
        elif kind == "capex":
            nb = netblock.get(isin, {})
            fy = GP.idx_q(end)[0]
            cur, prev = nb.get(fy), nb.get(fy - 1)
            if cur is not None and prev is not None:
                actuals = {end: cur - prev}
            src = "financials_3stmt:Net Block (PROXY, annual only)"

        # A Rs-cr target above every Indian company's scale is a units error in the
        # source cell, not a business. Rejected AFTER the feed choice so the reason
        # is recorded rather than the row silently ranking first.
        if kind in ("level", "orders", "capex") and \
                GP.is_implausible_amount(c["guided_value"]):
            src = (f"not comparable: guided {c['guided_value']:,.0f} cr exceeds any "
                   f"Indian company's scale — a units error in the source cell "
                   f"(a bare rupee amount read as crore)")
            kind = None

        p = GP.compute_progress(c["guided_value"], kind, (start, end), actuals,
                                prior_actuals=actuals, elapsed_idx=elapsed_idx)
        rows.append(_row(c, isin, names, start, end, kind, src, p, pf, now))

    df = pd.DataFrame(rows, columns=PROGRESS_COLS)
    if df.empty:
        return df

    # One row per commitment: when the same promise is repeated across concalls,
    # keep the most recently sourced telling of it.
    df["_qo"] = df["guid_quarter"].map(QT.q_order)
    df = (df.sort_values(["commit_id", "_qo"])
            .drop_duplicates("commit_id", keep="last")
            .drop(columns="_qo").reset_index(drop=True))
    return df


def rank_universe(df: pd.DataFrame, pf: set, top_n: int = TOP_N) -> pd.DataFrame:
    """Top-N companies by GUIDANCE RICHNESS (user 2026-08-15), PF force-included.

    Richness = number of open MEASURABLE commitments. Ranking on market cap was the
    alternative; it was rejected because the large caps guide most vaguely while the
    order-book targets worth tracking sit in mid/small caps (POWERMECH, KLBRENG).
    PF holdings are always in, whatever their rank.
    """
    if df.empty:
        return df
    measurable = df[df["status"] != "NO_DATA"]
    score = (measurable.groupby("isin").size().sort_values(ascending=False)
             if not measurable.empty else pd.Series(dtype=int))
    ranked = {isin: i + 1 for i, isin in enumerate(score.index[:top_n])}
    keep = set(ranked) | {i for i in pf if i in set(df["isin"])}
    out = df[df["isin"].isin(keep)].copy()
    out["univ_rank"] = out["isin"].map(ranked)
    return out.reset_index(drop=True)


def apply_history(df: pd.DataFrame, hist: pd.DataFrame, week: str,
                  now: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill delta_week from the most recent EARLIER snapshot, then add this week's."""
    if df.empty:
        return df, hist
    prior = hist[hist["week_start"].astype(str) < week] if not hist.empty else hist
    if not prior.empty:
        last = (prior.sort_values("week_start")
                     .drop_duplicates("commit_id", keep="last")
                     .set_index("commit_id")["pct_of_target"])
        prev = df["commit_id"].map(last)
        df["delta_week"] = (pd.to_numeric(df["pct_of_target"], errors="coerce")
                            - pd.to_numeric(prev, errors="coerce")).round(2)

    snap = df[["commit_id", "pct_of_target", "actual_to_date", "status"]].copy()
    snap["week_start"] = week
    snap["as_of"] = now
    snap = snap[HIST_COLS]
    # re-running in the same week REPLACES that week's snapshot rather than
    # doubling it, so the producer stays idempotent
    base = (hist[hist["week_start"].astype(str) != week]
            if not hist.empty else pd.DataFrame(columns=HIST_COLS))
    # concat with an all-empty frame warns about dtype inference in pandas 2.x
    out = snap if base.empty else pd.concat([base, snap], ignore_index=True)
    return df, out.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute + report, no write")
    ap.add_argument("--names", default="", help="comma-separated symbols to restrict to")
    ap.add_argument("--top-n", type=int, default=TOP_N)
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")
    now = datetime.now().isoformat(timespec="seconds")
    week = _week_start()
    now_q = GP.q_idx(*GP.parse_q(QT.norm_q(QT.season_quarter())))
    cal_q = GP.q_from_date(date.today())
    log(f"season quarter = {QT.season_quarter()} (idx {now_q})   "
        f"in-progress = {GP.q_label(cal_q)} (idx {cal_q})   week_start={week}")

    g = load_parquet(drive, idx, "guidance_tracker.parquet", GUIDANCE_COLS)
    ppt = load_parquet(drive, idx, "ppt_guidance.parquet", PPT_G_COLS)
    ar = load_parquet(drive, idx, "ar_guidance.parquet", AR_G_COLS)
    cred = load_parquet(drive, idx, "mgmt_credibility.parquet", MGMT_CRED_COLS)
    fin = load_parquet(drive, idx, "financials_3stmt.parquet", FIN3_COLS)
    led = load_parquet(drive, idx, "announcement_ledger.parquet", LEDGER_COLS)
    log(f"loaded  guidance={len(g)} ppt={len(ppt)} ar={len(ar)} cred={len(cred)} "
        f"fin3={len(fin)} ledger={len(led)}")

    # presentation/AR use `horizon`/`fy_year`; normalise onto the tracker's shape
    if not ppt.empty:
        ppt = ppt.rename(columns={"horizon": "horizon_fy"})
    if not ar.empty:
        ar = ar.rename(columns={"fy_year": "quarter"})

    commits = (_commitments_from_tracker(g, "concall")
               + _commitments_from_tracker(ppt, "presentation")
               + _commitments_from_tracker(ar, "annual_report")
               + _commitments_from_credibility(cred))
    log(f"raw commitments: {len(commits)}")

    if args.names:
        want = {s.strip().upper() for s in args.names.split(",") if s.strip()}
        commits = [c for c in commits if c["symbol"].upper() in want]
        log(f"--names filter -> {len(commits)} commitments")

    pf = load_portfolio_isins(drive, root) or set()
    names = {}
    fin_series = _financial_series(fin)
    orders = _order_series(led)
    netblock = _netblock_series(fin)
    ledger_from = _ledger_coverage(led)
    log(f"actuals: financial series={len(fin_series)}  order books={len(orders)}  "
        f"net-block={len(netblock)}  pf={len(pf)}")
    log(f"order-ledger coverage starts {GP.q_label(ledger_from) if ledger_from else 'n/a'}"
        " — order commitments whose horizon opens earlier get no pace verdict")

    df = build(commits, fin_series, orders, netblock, now_q, cal_q, pf, names, now,
               ledger_from=ledger_from)
    log(f"open commitments in window: {len(df)}")
    if df.empty:
        log("nothing open — no write.")
        return 0

    if not args.names:
        df = rank_universe(df, pf, args.top_n)
        log(f"after top-{args.top_n} guidance-richness cut: {len(df)} rows / "
            f"{df['isin'].nunique()} companies")

    hist = load_parquet(drive, idx, HIST_NAME, HIST_COLS)
    df, hist = apply_history(df, hist, week, now)

    log("\nstatus mix:\n" + df["status"].value_counts().to_string())
    meas = df[df["status"] != "NO_DATA"]
    log(f"\nmeasurable {len(meas)} / {len(df)}   metrics: "
        + ", ".join(f"{k}={v}" for k, v in
                    meas["metric"].value_counts().head(8).items()))
    if not meas.empty:
        show = meas.sort_values("pct_of_target", ascending=False).head(12)
        log("\nsample:")
        for r in show.itertuples(index=False):
            log(f"   {str(r.symbol)[:12]:<12} {r.metric:<11} {r.target_period:<9} "
                f"guided={r.guided_value} {r.guided_unit:<7} "
                f"actual={r.actual_to_date}  {r.pct_of_target}% of target at "
                f"{r.time_pct}% elapsed -> {r.status}")

    if args.dry_run:
        log("\nDRY RUN - nothing written.")
        return 0
    save_parquet(drive, idx, OUT_NAME, df[PROGRESS_COLS])
    save_parquet(drive, idx, HIST_NAME, hist[HIST_COLS])
    log(f"\nwrote {OUT_NAME} ({len(df)} rows) + {HIST_NAME} ({len(hist)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
