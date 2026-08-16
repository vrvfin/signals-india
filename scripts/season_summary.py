"""
season_summary.py — the per-company earnings-season summary: everything the company
published ABOUT this quarter, in one block.

A quarter is not just a P&L. A company files a results release, puts out an investor
presentation, holds a concall, and files a stream of other disclosures — and those
land in five different parquets that nothing has ever read together. This module
assembles them per company, per season quarter.

Sections, in the order a reader wants them:

    1. Results release      the filing's own headline numbers   (results_gemini)
    2. Investor deck        highlights + any guidance given      (ppt_highlights,
                                                                  ppt_guidance)
    3. Concall              guidance statements + quality flags  (gf1_*, gf4_*)
    4. Other filings        everything else disclosed, tagged    (announcement_ledger)
    5. Deck teardown        KPIs dropped, targets moved, framing (deck_*)

Measured on live Drive data, 2026-08-14, for Q1 FY27 across 51 PF holdings:
    ppt_highlights   267 rows / 23 companies
    ppt_guidance      59 rows / 12 companies
    gf1 statements   236 rows / 28 companies
    gf4 flags         83 rows / 28 companies
    announcements    462 rows / 43 companies
    deck_metrics/diff/flags   FILE DOES NOT EXIST — never written

Section 5 therefore renders "not covered" rather than an implied all-clear.
`--teardown` was only enabled in phase2.yml on 13 Aug 2026 and has not yet produced a
row; `deck_diff` additionally needs TWO consecutive quarters of deck coverage before
it can be non-empty, because deck_teardown drops `diff_without_prior_deck`. When those
rows start landing this section lights up with no further change here.

Everything is scoped to the season quarter, so a filing from another quarter can never
appear under this quarter's heading — the same rule the rest of the teardown follows.

Self-test:  python scripts/season_summary.py --self-test
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import quarterly_table as QT

# Caps per section. This block renders once per company inside a mail already bounded
# by Gmail's ~102 KB clip, so every section is bounded too.
MAX_HIGHLIGHTS = 6
MAX_GUIDANCE = 5
MAX_CONCALL = 4
MAX_FILINGS = 6
MAX_FLAGS = 3

UP, DOWN, MUTED, AMBER = "#1a7a3a", "#c0392b", "#8a97a0", "#b8860b"

_DIR_COLOUR = {"bull": UP, "bullish": UP, "positive": UP,
               "bear": DOWN, "bearish": DOWN, "negative": DOWN}
_MAT_COLOUR = {"high": DOWN, "med": AMBER, "medium": AMBER, "low": MUTED}

# ---- Relevance filter for "other filings" -------------------------------------------
# Measured over 2,924 PF announcements since 1 Jul 2026: 1,225 are `results`, 898 are
# low materiality, 461 are AGM/EGM, 368 are generic `other`. Listing all of that is not
# a summary, it is a firehose — and the results ones duplicate the teardown the reader
# already has. This section exists for what MOVES A VIEW: an order win, a capex plan,
# a fundraise, a downgrade, litigation.

# Event types that carry an investment signal. `results` is deliberately absent — it has
# its own mail and its own section above; repeating it here is duplication, not summary.
_RELEVANT_EVENTS = {"order_win", "expansion", "capex", "mna", "fundraise", "debt",
                    "litigation", "rating", "buyback", "dividend", "management_change",
                    "regulatory"}

# Routine filings that are legally required and analytically empty.
_ROUTINE_CATEGORIES = {"agm/egm", "insider trading / sast", "corp. action", "corp action"}
_ROUTINE_PATTERNS = ("newspaper publication", "trading window", "duplicate share",
                     "share certificate", "investor meet - intimation",
                     "board meeting intimation", "intimation of board meeting",
                     "loss of share", "transfer of shares", "postal ballot",
                     "scrutinizer", "voting results", "compliance certificate",
                     "reg. 74(5)", "regulation 74", "certificate under regulation")


def _is_relevant(a: dict) -> bool:
    """Does this filing plausibly change how the company is viewed?"""
    ev = str(a.get("event_type") or "").strip().lower()
    mat = str(a.get("materiality") or "").strip().lower()
    cat = str(a.get("category") or "").strip().lower()
    head = f"{a.get('headline','')} {a.get('summary','')}".lower()

    if any(p in head for p in _ROUTINE_PATTERNS):
        return False
    if cat in _ROUTINE_CATEGORIES and mat != "high":
        return False
    if ev == "results":                 # covered by the results teardown itself
        return False
    if ev in _RELEVANT_EVENTS:
        return mat in ("high", "med", "medium") or ev in ("rating", "litigation")
    # Anything else (incl. the generic 'other' bucket) only earns a place when the
    # extractor judged it materially important.
    return mat == "high"


def quarter_end(season: str) -> date | None:
    """Last day of the season quarter. 'Q1FY27' -> 2026-06-30.

    FY27 ends in March 2027, so Q1/Q2/Q3 of FY27 fall in calendar 2026 and only Q4
    falls in 2027. Getting this backwards would window the wrong three months.
    """
    q = QT.norm_q(season)
    if len(q) < 6 or not q.startswith("Q"):
        return None
    try:
        n = int(q[1])
        fy = int(q[4:])
    except (ValueError, IndexError):
        return None
    yy = 2000 + fy
    ends = {1: (yy - 1, 6, 30), 2: (yy - 1, 9, 30), 3: (yy - 1, 12, 31), 4: (yy, 3, 31)}
    if n not in ends:
        return None
    y, m, dd = ends[n]
    return date(y, m, dd)


def season_window(season: str, today: date | None = None) -> tuple[str, str]:
    """(start, end) ISO dates for filings that BELONG to this quarter.

    A quarter's disclosures arrive after it ends — results season runs roughly the
    following three months — so the window opens the day the quarter closes.
    """
    qe = quarter_end(season)
    if qe is None:
        return ("", "")
    end = min(qe + timedelta(days=110), (today or date.today()))
    if end < qe:
        end = qe
    return (qe.isoformat(), end.isoformat())


def _season_rows(df, isin: str, season: str, cols: list[str]) -> pd.DataFrame:
    """Rows for one company and one quarter, keyed on the `quarter` column."""
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=cols)
    if "isin" not in df.columns or "quarter" not in df.columns:
        return pd.DataFrame(columns=cols)
    want = QT.norm_q(season)
    d = df[df["isin"].astype(str).str.strip() == str(isin).strip()]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d = d[d["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]
    for c in cols:
        if c not in d.columns:
            d[c] = None
    return d[cols] if not d.empty else pd.DataFrame(columns=cols)


def build_summary(isin: str, season: str, *, ppt_highlights=None, ppt_guidance=None,
                  gf1=None, gf4=None, announcements=None, filing=None,
                  deck=None, today: date | None = None) -> dict:
    """Assemble one company's season summary. Pure — no Drive, no network."""
    out: dict = {"isin": isin, "season": season}

    h = _season_rows(ppt_highlights, isin, season, ["category", "statement", "value", "unit"])
    out["highlights"] = [
        {"category": str(r["category"] or ""), "statement": str(r["statement"] or ""),
         "value": str(r["value"] or ""), "unit": str(r["unit"] or "")}
        for _, r in h.head(MAX_HIGHLIGHTS).iterrows()
        if str(r["statement"] or "").strip()]
    out["n_highlights"] = len(h)

    g = _season_rows(ppt_guidance, isin, season,
                     ["metric", "guidance_type", "horizon", "value", "unit", "notes"])
    out["guidance"] = [
        {"metric": str(r["metric"] or ""), "type": str(r["guidance_type"] or ""),
         "horizon": str(r["horizon"] or ""), "value": str(r["value"] or ""),
         "unit": str(r["unit"] or "")}
        for _, r in g.head(MAX_GUIDANCE).iterrows() if str(r["metric"] or "").strip()]
    out["n_guidance"] = len(g)

    c = _season_rows(gf1, isin, season,
                     ["exact_statement", "metric_type", "timeframe", "quantifiable"])
    out["concall"] = [
        {"statement": str(r["exact_statement"] or ""),
         "metric": str(r["metric_type"] or ""), "timeframe": str(r["timeframe"] or "")}
        for _, r in c.head(MAX_CONCALL).iterrows()
        if str(r["exact_statement"] or "").strip()]
    out["n_concall"] = len(c)

    f = _season_rows(gf4, isin, season, ["flag_type", "evidence"])
    out["flags"] = [
        {"flag": str(r["flag_type"] or ""), "evidence": str(r["evidence"] or "")}
        for _, r in f.head(MAX_FLAGS).iterrows() if str(r["flag_type"] or "").strip()]
    out["n_flags"] = len(f)

    # Announcements are dated, not quarter-tagged — window them instead.
    start, end = season_window(season, today)
    ann = []
    if (announcements is not None and not getattr(announcements, "empty", True)
            and start and "ann_date" in announcements.columns):
        a = announcements[announcements["isin"].astype(str).str.strip()
                          == str(isin).strip()].copy()
        if not a.empty:
            a["_d"] = a["ann_date"].astype(str).str.slice(0, 10)
            a = a[(a["_d"] >= start) & (a["_d"] <= end)]
            # Most material first, then most recent.
            rank = {"high": 0, "medium": 1, "low": 2}
            a["_r"] = a.get("materiality", pd.Series(dtype=str)).astype(str).str.lower(
            ).map(rank).fillna(3)
            a = a.sort_values(["_r", "_d"], ascending=[True, False])
            cand = [{"date": str(r["_d"]),
                     "headline": str(r.get("headline") or ""),
                     "summary": str(r.get("summary") or ""),
                     "event_type": str(r.get("event_type") or ""),
                     "category": str(r.get("category") or ""),
                     "materiality": str(r.get("materiality") or "").lower(),
                     "direction": str(r.get("direction") or "").lower()}
                    for _, r in a.iterrows()]
            # Relevance, not volume. See _is_relevant: routine compliance filings and
            # anything already covered by the results teardown are dropped.
            keep = [x for x in cand if _is_relevant(x)]
            ann = keep[:MAX_FILINGS]
            out["n_filings"] = len(keep)
            out["n_filings_seen"] = len(cand)
    out["filings"] = ann
    out.setdefault("n_filings", 0)
    out["window"] = (start, end)

    out["filing"] = filing or None

    # Deck teardown: distinguish "no rows for this company" from "table never written".
    deck = deck or {}
    present = {k: (v is not None and not getattr(v, "empty", True))
               for k, v in deck.items()}
    out["deck_built"] = any(present.values())
    out["deck"] = {}
    if out["deck_built"]:
        for key, cols in (("metrics", ["category", "metric", "value", "unit", "slide_ref"]),
                          ("diff", ["change_type", "item", "prior_state", "current_state"]),
                          ("flags", ["flag_type", "slide_ref", "severity"])):
            rows = _season_rows(deck.get(key), isin, season, cols)
            out["deck"][key] = [dict(r) for _, r in rows.head(4).iterrows()]
    return out


# ------------------------------------------------------------------ #
#  Render                                                             #
# ------------------------------------------------------------------ #

_TBL = ("border-collapse:collapse;width:100%;font:11.5px Arial,Helvetica,sans-serif;"
        "color:#222;margin:2px 0 8px")


def _esc(s, n=300) -> str:
    from mailer import esc
    return esc(s, n)


def _sub(title: str, note: str = "") -> str:
    return (f"<div style='font-size:11px;font-weight:700;color:#34495e;"
            f"text-transform:uppercase;letter-spacing:.3px;margin:8px 0 1px'>{title}"
            + (f"<span style='color:{MUTED};font-weight:400;text-transform:none;"
               f"letter-spacing:0'> &middot; {note}</span>" if note else "")
            + "</div>")


def render_summary(s: dict, include_deck: bool = True) -> str:
    """One company's season summary as HTML. Empty string when nothing was published."""
    parts = []

    f = s.get("filing")
    if f:
        cells = []
        for label, key, suffix in (("Revenue", "revenue_cr", " Cr"),
                                   ("EBITDA", "ebitda_cr", " Cr"),
                                   ("PAT", "pat_cr", " Cr"), ("EPS", "eps", "")):
            v = f.get(key)
            if v is not None:
                cells.append(f"<b>{label}</b> {v:,.2f}{suffix}")
        if cells:
            parts.append(_sub("Results release",
                              _esc(str(f.get("title") or "")[:60], 60))
                         + f"<div style='font-size:11.5px;margin:0 0 6px'>"
                         + " &nbsp;&middot;&nbsp; ".join(cells) + "</div>")

    # DECK CONTENT BELONGS TO THE DECK MAIL (user, 2026-08-16). The results teardown
    # answers "how are the FINANCIALS behaving"; the presentation mail answers "how is
    # the BUSINESS behaving". Rendering deck highlights, deck guidance and the deck diff
    # inside the teardown duplicated the deck mail and blurred that split, so the
    # teardown passes include_deck=False and keeps only the results release, the concall
    # and the other filings.
    if s["highlights"] and include_deck:
        rows = "".join(
            f"<tr><td style='color:{MUTED};white-space:nowrap'>{_esc(h['category'], 22)}</td>"
            f"<td>{_esc(h['statement'], 190)}"
            + (f" <b>{_esc(h['value'], 24)}{_esc(h['unit'], 10)}</b>"
               if h["value"] else "") + "</td></tr>"
            for h in s["highlights"])
        more = (f"{s['n_highlights']} in the deck" if s["n_highlights"] > len(s["highlights"])
                else "")
        parts.append(_sub("Investor presentation", more)
                     + f"<table cellpadding='2' cellspacing='0' style='{_TBL}'>{rows}</table>")

    if s["guidance"] and include_deck:
        rows = "".join(
            f"<tr><td><b>{_esc(g['metric'], 40)}</b></td>"
            f"<td>{_esc(g['value'], 30)}{_esc(g['unit'], 10)}</td>"
            f"<td style='color:{MUTED}'>{_esc(g['horizon'], 24)}</td></tr>"
            for g in s["guidance"])
        parts.append(_sub("Guidance given in the deck")
                     + f"<table cellpadding='2' cellspacing='0' style='{_TBL}'>{rows}</table>")

    if s["concall"]:
        rows = "".join(
            f"<tr><td style='color:{MUTED};white-space:nowrap'>{_esc(c['metric'], 20)}</td>"
            f"<td>&ldquo;{_esc(c['statement'], 200)}&rdquo;</td></tr>"
            for c in s["concall"])
        parts.append(_sub("Said on the concall",
                          f"{s['n_concall']} statements captured"
                          if s["n_concall"] > len(s["concall"]) else "")
                     + f"<table cellpadding='2' cellspacing='0' style='{_TBL}'>{rows}</table>")

    if s["flags"]:
        rows = "".join(
            f"<tr><td style='color:{AMBER};white-space:nowrap'>{_esc(fl['flag'], 30)}</td>"
            f"<td>{_esc(fl['evidence'], 180)}</td></tr>" for fl in s["flags"])
        parts.append(_sub("Commentary quality flags")
                     + f"<table cellpadding='2' cellspacing='0' style='{_TBL}'>{rows}</table>")

    if s["filings"]:
        rows = ""
        for a in s["filings"]:
            col = _MAT_COLOUR.get(a["materiality"], MUTED)
            dcol = _DIR_COLOUR.get(a["direction"], "")
            tag = (f"<span style='color:{dcol};font-weight:700'>{_esc(a['event_type'], 22)}</span>"
                   if dcol else f"<span style='color:{MUTED}'>{_esc(a['event_type'], 22)}</span>")
            text = a["summary"] or a["headline"]
            rows += (f"<tr><td style='color:{MUTED};white-space:nowrap'>{_esc(a['date'], 10)}</td>"
                     f"<td style='color:{col};white-space:nowrap'>{tag}</td>"
                     f"<td>{_esc(text, 200)}</td></tr>")
        seen = s.get("n_filings_seen", 0)
        more = (f"{s['n_filings']} material of {seen} filed" if seen
                else (f"{s['n_filings']} in the window"
                      if s["n_filings"] > len(s["filings"]) else ""))
        parts.append(_sub("What else moved this quarter", more)
                     + f"<table cellpadding='2' cellspacing='0' style='{_TBL}'>{rows}</table>")

    if include_deck and s["deck_built"] and any(s["deck"].values()):
        d = s["deck"]
        rows = "".join(
            f"<tr><td><b>{_esc(x.get('change_type', ''), 24)}</b></td>"
            f"<td>{_esc(x.get('item', ''), 60)}</td>"
            f"<td style='color:{MUTED}'>{_esc(x.get('prior_state', ''), 40)} &rarr; "
            f"{_esc(x.get('current_state', ''), 40)}</td></tr>"
            for x in d.get("diff", []))
        rows += "".join(
            f"<tr><td><b>{_esc(x.get('flag_type', ''), 24)}</b></td>"
            f"<td colspan='2'>slide {_esc(x.get('slide_ref', ''), 12)} &middot; "
            f"{_esc(x.get('severity', ''), 10)}</td></tr>"
            for x in d.get("flags", []))
        if rows:
            parts.append(_sub("Deck vs the last deck")
                         + f"<table cellpadding='2' cellspacing='0' style='{_TBL}'>{rows}</table>")
    elif include_deck and (s["highlights"] or s["guidance"]):
        # MISSING IS NOT CLEAN. Say the pass has not run rather than imply the deck was
        # read and found honest.
        parts.append(f"<div style='color:{MUTED};font-size:11px;margin:4px 0 8px'>"
                     f"Deck teardown (KPIs dropped, targets moved, framing flags) has "
                     f"not produced rows for this company yet &mdash; not an all-clear.</div>")

    if not parts:
        return ""
    start, end = s.get("window", ("", ""))
    return ("<div style='margin:2px 0 10px'>"
            + "".join(parts)
            + (f"<div style='color:{MUTED};font-size:10.5px'>Filings windowed "
               f"{start} to {end}.</div>" if start else "")
            + "</div>")


# ------------------------------------------------------------------ #
#  Self-test                                                          #
# ------------------------------------------------------------------ #

def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    # ---- the FY-to-calendar mapping is the easiest thing here to get backwards
    check("Q1 FY27 ended Jun 2026", quarter_end("Q1FY27") == date(2026, 6, 30))
    check("Q2 FY27 ended Sep 2026", quarter_end("Q2FY27") == date(2026, 9, 30))
    check("Q3 FY27 ended Dec 2026", quarter_end("Q3FY27") == date(2026, 12, 31))
    check("Q4 FY27 ends Mar 2027 — the one in the next calendar year",
          quarter_end("Q4FY27") == date(2027, 3, 31))
    check("spaced form parses too", quarter_end("Q1 FY27") == date(2026, 6, 30))
    check("junk returns None", quarter_end("nonsense") is None)

    start, end = season_window("Q1FY27", today=date(2026, 8, 14))
    check("window opens when the quarter closes", start == "2026-06-30")
    check("window does not run past today", end == "2026-08-14")

    def _df(rows, cols):
        return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])

    HL = ["isin", "quarter", "category", "statement", "value", "unit"]
    hl = _df([{"isin": "INE1", "quarter": "Q1 FY27", "category": "demand",
               "statement": "Order book at a record", "value": "1200", "unit": "Cr"},
              {"isin": "INE1", "quarter": "Q4 FY26", "category": "demand",
               "statement": "PREVIOUS QUARTER — must not appear"},
              {"isin": "INE2", "quarter": "Q1 FY27", "category": "cost",
               "statement": "OTHER COMPANY — must not appear"}], HL)
    ANN = ["isin", "ann_date", "headline", "summary", "event_type", "materiality",
           "direction"]
    ann = _df([{"isin": "INE1", "ann_date": "2026-07-15", "headline": "Order win",
                "summary": "Bagged a 300 Cr order", "event_type": "order_win",
                "materiality": "high", "direction": "bull"},
               {"isin": "INE1", "ann_date": "2026-05-02",
                "headline": "BEFORE THE QUARTER ENDED — must not appear"}], ANN)

    s = build_summary("INE1", "Q1FY27", ppt_highlights=hl, announcements=ann,
                      today=date(2026, 8, 14))
    check("highlight for this quarter kept", len(s["highlights"]) == 1)
    check("another quarter excluded",
          all("PREVIOUS" not in h["statement"] for h in s["highlights"]))
    check("another company excluded",
          all("OTHER COMPANY" not in h["statement"] for h in s["highlights"]))
    check("in-window filing kept", len(s["filings"]) == 1)
    check("pre-quarter filing excluded",
          all("BEFORE" not in a["headline"] for a in s["filings"]))

    # ---- RELEVANCE, not volume. 1,225 of 2,924 live announcements are `results`,
    # 898 are low materiality and 461 are AGM/EGM — a summary that lists them all is
    # a firehose, and the results ones duplicate the teardown.
    def _a(**kw):
        base = {"headline": "", "summary": "", "event_type": "other",
                "category": "Company Update", "materiality": "med", "direction": ""}
        base.update(kw)
        return base
    check("order win is kept", _is_relevant(_a(event_type="order_win")))
    check("capex is kept", _is_relevant(_a(event_type="capex")))
    check("a downgrade is kept even at low materiality",
          _is_relevant(_a(event_type="rating", materiality="low")))
    check("litigation is kept even at low materiality",
          _is_relevant(_a(event_type="litigation", materiality="low")))
    check("results are NOT repeated here (the teardown covers them)",
          not _is_relevant(_a(event_type="results", materiality="high")))
    check("low-materiality generic filing dropped",
          not _is_relevant(_a(event_type="other", materiality="low")))
    check("high-materiality generic filing kept",
          _is_relevant(_a(event_type="other", materiality="high")))
    check("AGM/EGM dropped", not _is_relevant(_a(category="AGM/EGM")))
    check("a HIGH-materiality AGM item still gets through",
          _is_relevant(_a(category="AGM/EGM", materiality="high",
                          event_type="mna")))
    for junk in ("Copy of Newspaper Publication", "Trading Window closure",
                 "Intimation of Board Meeting", "Duplicate share certificate",
                 "Compliance Certificate under Regulation 74(5)"):
        check(f"routine filing dropped: {junk[:28]}",
              not _is_relevant(_a(headline=junk, materiality="high",
                                  event_type="regulatory")))

    ann2 = _df([{"isin": "INE1", "ann_date": "2026-07-20", "headline": "Order win",
                 "summary": "300 Cr order", "event_type": "order_win",
                 "materiality": "high", "direction": "bull"},
                {"isin": "INE1", "ann_date": "2026-07-21",
                 "headline": "Copy of Newspaper Publication", "summary": "",
                 "event_type": "regulatory", "materiality": "med", "direction": ""},
                {"isin": "INE1", "ann_date": "2026-07-22", "headline": "Q1 results",
                 "summary": "", "event_type": "results", "materiality": "high",
                 "direction": ""}], ANN + ["category"])
    s3 = build_summary("INE1", "Q1FY27", announcements=ann2, today=date(2026, 8, 14))
    check("only the material filing survives", len(s3["filings"]) == 1)
    check("the survivor is the order win", s3["filings"][0]["event_type"] == "order_win")
    check("the count reports material-of-total", s3.get("n_filings_seen") == 3)

    html = render_summary(s)
    check("renders the highlight", "Order book at a record" in html)
    check("renders the filing summary", "300 Cr order" in html)
    check("states the window", "2026-06-30" in html)
    # THE HONESTY RULE: no deck rows must read as "not run", never as "clean".
    check("absent deck teardown is declared, not implied clean",
          "not an all-clear" in html)
    check("no deck section fabricated", "Deck vs the last deck" not in html)

    # A company with nothing published produces nothing, not an empty shell.
    check("nothing published renders empty",
          render_summary(build_summary("INE_NONE", "Q1FY27")) == "")

    # Missing tables must never raise — this is the normal state early in a season.
    try:
        build_summary("INE1", "Q1FY27", ppt_highlights=None, gf1=pd.DataFrame(),
                      announcements=None, deck={"metrics": None, "diff": None})
        check("missing tables do not raise", True)
    except Exception as exc:                                    # pragma: no cover
        check(f"missing tables do not raise ({exc})", False)

    # Results release renders off the validated filing dict.
    s2 = build_summary("INE1", "Q1FY27",
                       filing={"revenue_cr": 555.51, "pat_cr": None, "eps": None,
                               "ebitda_cr": None, "title": "Board outcome"})
    h2 = render_summary(s2)
    check("results release rendered", "555.51" in h2)
    check("absent PAT is simply not printed", "PAT</b>" not in h2)

    print(f"\nseason_summary self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
