"""
pf_coverage.py — what SHOULD exist for each PF holding, what does, and who is due a mail.

Nothing in this repo could previously answer "is my data complete?". Coverage was only
ever implied — by the absence of a row somewhere — so a holding whose investor deck was
never fetched looked identical to one that simply never published a deck. This module
makes the expectation explicit, which is the only way a gap can be reported rather than
silently rendered as a blank section.

Three questions, three functions:

    coverage(...)        per holding x doc_type: present / pending / failed / missing
    reporting_on(...)    which PF holdings have a board meeting on a given date
    mail_due(...)        which holdings should be mailed now, and for which doc types

THE TRIGGER RULE (user, 2026-08-15). A mail goes out when the holding is in PF **and**
its results-calendar date has arrived **and** it has not already been mailed for that
document and period. It is deliberately NOT a rolling time window: a window makes a
missed CI run lose a company permanently, whereas "not yet mailed" is self-healing — the
next run picks it up whenever it runs.

CALENDAR LIMIT, STATED NOT HIDDEN. `results_calendar.parquet` is accumulated from the NSE
and BSE forthcoming-meeting APIs, both of which are STRICTLY FORWARD-LOOKING. Its history
begins 2026-08-06 and nothing earlier is recoverable from that source. So the calendar is
authoritative for "who reports today or tomorrow" and useless for backfill; anything
historic must come from the date cascade in quarterly_table / pf_results_digest instead.
`reporting_on()` therefore returns an empty set rather than a wrong answer for old dates,
and callers fall back.

Pure — no Drive, no network, no Gemini. Self-test:
    python scripts/pf_coverage.py --self-test
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import quarterly_table as QT

# The calendar cannot know anything before this — both its source APIs only ever return
# forthcoming meetings, so history had to be accumulated from the day persistence began.
CALENDAR_EPOCH = "2026-08-06"

PRESENT, PENDING, FAILED, MISSING = "present", "pending", "failed", "missing"

# What a complete picture looks like, per holding.
#   results       — one per quarter
#   presentation  — one per quarter where the company publishes one at all
#   rating        — event-driven; no per-quarter expectation, so never "missing"
EXPECTED_PER_QUARTER = ("results", "presentation")
EVENT_DRIVEN = ("rating",)

_DONE = {"done", "superseded"}
_FAILED = {"error", "expired"}


def _norm(s) -> str:
    return str(s or "").strip().upper()


def coverage(pf, queue: pd.DataFrame, season: str,
             doc_types=("results", "presentation", "rating")) -> pd.DataFrame:
    """One row per (holding, doc_type) for the season: status + counts.

    `pf` is [(isin, symbol, name)] as daily_brief.load_pf returns it.
    Status is the BEST state found, because one good document is enough:
        present  a document reached done/superseded
        pending  queued, not yet processed
        failed   every attempt errored or expired
        missing  nothing queued at all
    """
    rows = []
    q = queue if queue is not None and not queue.empty else pd.DataFrame(
        columns=["isin", "doc_type", "status", "announcement_date"])
    q = q.copy()
    for c in ("isin", "doc_type", "status"):
        if c not in q.columns:
            q[c] = ""
    q["_isin"] = q["isin"].astype(str).str.strip()

    for isin, sym, name in pf:
        sub = q[q["_isin"] == str(isin).strip()]
        for dt in doc_types:
            s = sub[sub["doc_type"].astype(str) == dt]
            st = s["status"].astype(str).str.lower()
            n_done, n_pend = int(st.isin(_DONE).sum()), int(st.eq("pending").sum())
            n_fail = int(st.isin(_FAILED).sum())
            if n_done:
                status = PRESENT
            elif n_pend:
                status = PENDING
            elif n_fail:
                status = FAILED
            else:
                status = MISSING
            rows.append({"isin": isin, "symbol": sym, "name": name, "doc_type": dt,
                         "season": season, "status": status, "n_done": n_done,
                         "n_pending": n_pend, "n_failed": n_fail,
                         "expected": dt in EXPECTED_PER_QUARTER})
    return pd.DataFrame(rows)


def gaps(cov: pd.DataFrame) -> pd.DataFrame:
    """Only the rows a human should act on: expected but not present."""
    if cov is None or cov.empty:
        return pd.DataFrame(columns=list(cov.columns) if cov is not None else [])
    return cov[(cov["expected"]) & (cov["status"] != PRESENT)].sort_values(
        ["status", "symbol"])


def reporting_on(calendar: pd.DataFrame, pf, on: date | None = None,
                 window_days: int = 0) -> dict[str, str]:
    """{isin: meeting_date} for PF holdings with a board meeting on/near `on`.

    `window_days` widens BACKWARDS only — a meeting yesterday still counts today, so a
    skipped run does not lose the company. Never forwards: a meeting that has not
    happened yet has no numbers to mail.
    """
    on = on or date.today()
    if calendar is None or calendar.empty or "meeting_date" not in calendar.columns:
        return {}
    lo = (on - timedelta(days=max(0, window_days))).isoformat()
    hi = on.isoformat()
    if hi < CALENDAR_EPOCH:
        return {}                       # before the calendar knows anything — caller falls back
    c = calendar.copy()
    c["_d"] = c["meeting_date"].astype(str).str.slice(0, 10)
    c = c[(c["_d"] >= lo) & (c["_d"] <= hi)]
    if c.empty:
        return {}
    by_sym = {}
    for _, r in c.iterrows():
        by_sym.setdefault(_norm(r.get("symbol")), str(r["_d"]))
    out = {}
    for isin, sym, _name in pf:
        hit = by_sym.get(_norm(sym))
        if hit:
            out[str(isin).strip()] = hit
    return out


def latest_doc_per_type(pf, queue: pd.DataFrame,
                        doc_types=("results", "presentation", "rating")) -> dict:
    """{(isin, doc_type): {doc_id, date, title}} — the NEWEST processed document.

    This is what makes "a new document arrives" a mailable event. Keying the mail ledger
    on (isin, doc_type, season) alone means the FIRST deck or rating of a quarter is
    mailed and everything after it is silently suppressed — so a rating DOWNGRADE landing
    a week after a routine reaffirmation would never reach the reader. Identity has to be
    the document, not the slot it occupies.
    """
    out = {}
    if queue is None or queue.empty:
        return out
    q = queue.copy()
    for c in ("isin", "doc_type", "status", "announcement_date", "doc_id", "title"):
        if c not in q.columns:
            q[c] = ""
    q = q[q["status"].astype(str).str.lower().isin(_DONE)]
    q["_isin"] = q["isin"].astype(str).str.strip()
    q["_d"] = q["announcement_date"].astype(str).str.slice(0, 10)
    pf_isins = {str(i).strip() for i, _s, _n in pf}
    q = q[q["_isin"].isin(pf_isins)]
    for (isin, dt), grp in q.groupby(["_isin", "doc_type"]):
        if dt not in doc_types:
            continue
        g = grp.sort_values("_d")
        r = g.iloc[-1]
        out[(isin, dt)] = {"doc_id": str(r["doc_id"]).strip(),
                           "date": str(r["_d"]), "title": str(r["title"])[:160]}
    return out


def already_mailed_docs(ledger: pd.DataFrame, season: str) -> set[str]:
    """Set of doc_ids already mailed this season."""
    if ledger is None or ledger.empty or "doc_id" not in ledger.columns:
        return set()
    l = ledger[ledger["season"].astype(str) == str(season)]
    return {str(x).strip() for x in l["doc_id"] if str(x).strip()
            and str(x).strip().lower() not in ("none", "nan")}


def already_mailed(ledger: pd.DataFrame, season: str) -> set[tuple[str, str]]:
    """{(isin, doc_type)} already delivered for this season."""
    if ledger is None or ledger.empty:
        return set()
    l = ledger.copy()
    for c in ("season", "isin", "doc_type"):
        if c not in l.columns:
            l[c] = ""
    l = l[l["season"].astype(str) == str(season)]
    return {(str(r["isin"]).strip(), str(r["doc_type"]))
            for _, r in l.iterrows()}


#  Season status — the deterministic "where are we, out of 51" picture.
DELIVERED, DUE, AWAITING, NO_INFO = "delivered", "due", "awaiting", "no information"


def season_status(pf, queue, calendar, ledger, season, on=None, window_days=2,
                  doc_types=("results", "presentation", "rating")) -> list[dict]:
    """Per holding x doc_type: delivered / due / awaiting / no information, WITH a reason.

    The four states are deliberately distinct, because they need different actions:
      delivered      a mail went out for the newest document
      due            the document is processed and the mail has not gone yet
      awaiting       the exchange calendar says this company reports, nothing has landed
      no information nothing filed and nothing scheduled — for ratings and decks this is
                     normal (not every company issues either), so it is NOT a failure
    """
    on = on or date.today()
    cov = coverage(pf, queue, season, doc_types=doc_types)
    latest = latest_doc_per_type(pf, queue, doc_types)
    mailed_docs = already_mailed_docs(ledger, season)
    legacy = already_mailed(ledger, season)
    reporting = reporting_on(calendar, pf, on, window_days=120)   # whole season so far

    rows = []
    for _, r in cov.iterrows():
        isin, dt = str(r["isin"]).strip(), r["doc_type"]
        doc = latest.get((isin, dt)) or {}
        doc_id = doc.get("doc_id", "")
        sent = bool(doc_id and doc_id in mailed_docs) or \
            (not mailed_docs and (isin, dt) in legacy) or \
            (not doc_id and (isin, dt) in legacy)
        if r["status"] == PRESENT and sent:
            state, why = DELIVERED, f"mailed · {doc.get('date','')}"
        elif r["status"] == PRESENT:
            state, why = DUE, "processed, mail not sent yet"
        elif r["status"] == PENDING:
            state, why = AWAITING, "document fetched, extraction pending"
        elif r["status"] == FAILED:
            state, why = AWAITING, "extraction failed — will retry"
        elif dt == "results" and isin in reporting:
            state, why = AWAITING, f"calendar says reported {reporting[isin]}, no numbers yet"
        elif dt == "results":
            state, why = AWAITING, "no results filed this season yet"
        else:
            state, why = NO_INFO, ("no rating issued this season" if dt == "rating"
                                   else "no deck published this season")
        rows.append({"isin": isin, "symbol": r["symbol"], "name": r["name"],
                     "doc_type": dt, "state": state, "reason": why,
                     "doc_date": doc.get("date", "")})
    return rows


def season_rollup(rows: list[dict]) -> dict:
    """Counts per state, and how many holdings are fully covered on results."""
    out = {}
    for dt in ("results", "presentation", "rating"):
        sub = [r for r in rows if r["doc_type"] == dt]
        out[dt] = {s: sum(1 for r in sub if r["state"] == s)
                   for s in (DELIVERED, DUE, AWAITING, NO_INFO)}
    res = [r for r in rows if r["doc_type"] == "results"]
    out["_companies"] = len({r["isin"] for r in rows})
    out["_results_done"] = sum(1 for r in res if r["state"] == DELIVERED)
    out["_results_total"] = len(res)
    return out


def mail_due(pf, queue: pd.DataFrame, calendar: pd.DataFrame, ledger: pd.DataFrame,
             season: str, on: date | None = None, window_days: int = 2,
             doc_types=("results", "presentation", "rating"),
             require_calendar: bool = False) -> list[dict]:
    """Which holdings should be mailed now, and for what.

    A (holding, doc_type) is due when the document is PRESENT and it has not already been
    mailed this season. `require_calendar=True` additionally demands a board-meeting date
    on/near today — right for the daily results mail, wrong for presentations and ratings,
    which arrive on no calendar at all.
    """
    on = on or date.today()
    cov = coverage(pf, queue, season, doc_types=doc_types)
    latest = latest_doc_per_type(pf, queue, doc_types)
    mailed_docs = already_mailed_docs(ledger, season)
    legacy = already_mailed(ledger, season)          # rows written before doc_id existed
    reporting = reporting_on(calendar, pf, on, window_days)

    out = []
    for _, r in cov.iterrows():
        if r["status"] != PRESENT:
            continue
        isin = str(r["isin"]).strip()
        doc = latest.get((isin, r["doc_type"])) or {}
        doc_id = doc.get("doc_id", "")
        # A document is due when THIS document has not been mailed. Falling back to the
        # old (isin, doc_type) key only for ledger rows written before doc_id existed
        # stops a one-off flood of re-sends on the first run under the new scheme.
        if doc_id and doc_id in mailed_docs:
            continue
        if not doc_id and (isin, r["doc_type"]) in legacy:
            continue
        if doc_id and not mailed_docs and (isin, r["doc_type"]) in legacy:
            continue                                  # legacy-only ledger: treat as sent
        if require_calendar and r["doc_type"] == "results":
            if isin not in reporting:
                continue
        out.append({"isin": r["isin"], "symbol": r["symbol"], "name": r["name"],
                    "doc_type": r["doc_type"], "season": season,
                    "doc_id": doc_id, "doc_date": doc.get("date", ""),
                    "doc_title": doc.get("title", ""),
                    "reported_on": reporting.get(isin, "")})
    out.sort(key=lambda d: (d["doc_type"], d["symbol"]))
    return out


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

    pf = [("INE1", "AAA", "A Ltd"), ("INE2", "BBB", "B Ltd"), ("INE3", "CCC", "C Ltd")]
    Q = ["isin", "doc_type", "status", "announcement_date", "doc_id", "title"]

    def _q(rows):
        return pd.DataFrame([{c: r.get(c) for c in Q} for r in rows])

    queue = _q([
        {"isin": "INE1", "doc_type": "results", "status": "done"},
        {"isin": "INE1", "doc_type": "presentation", "status": "pending"},
        {"isin": "INE2", "doc_type": "results", "status": "error"},
        {"isin": "INE2", "doc_type": "presentation", "status": "done"},
        # INE3 has nothing at all
    ])
    cov = coverage(pf, queue, "Q1FY27")
    def st(isin, dt):
        m = cov[(cov.isin_ == isin) if False else (cov["isin"] == isin)]
        return m[m["doc_type"] == dt]["status"].iloc[0]
    check("done -> present", st("INE1", "results") == PRESENT)
    check("pending -> pending", st("INE1", "presentation") == PENDING)
    check("error -> failed", st("INE2", "results") == FAILED)
    check("nothing queued -> missing", st("INE3", "results") == MISSING)
    check("rating is never 'expected'",
          not cov[cov["doc_type"] == "rating"]["expected"].any())
    check("results IS expected", cov[cov["doc_type"] == "results"]["expected"].all())

    # one good document is enough, even beside failures
    q2 = _q([{"isin": "INE1", "doc_type": "results", "status": "error"},
             {"isin": "INE1", "doc_type": "results", "status": "done"}])
    c2 = coverage([pf[0]], q2, "Q1FY27")
    check("a later success beats an earlier failure",
          c2[c2["doc_type"] == "results"]["status"].iloc[0] == PRESENT)

    g = gaps(cov)
    check("gaps lists only expected-and-absent", set(g["doc_type"]) <= set(EXPECTED_PER_QUARTER))
    check("gaps excludes present rows", PRESENT not in set(g["status"]))

    # ---- calendar
    cal = pd.DataFrame([
        {"symbol": "AAA", "meeting_date": "2026-08-14", "purpose": "Financial Results"},
        {"symbol": "BBB", "meeting_date": "2026-08-10", "purpose": "Financial Results"},
        {"symbol": "ZZZ", "meeting_date": "2026-08-14", "purpose": "Financial Results"},
    ])
    r = reporting_on(cal, pf, date(2026, 8, 14))
    check("today's reporter found", "INE1" in r)
    check("non-PF symbol ignored", len(r) == 1)
    check("older meeting excluded with no window", "INE2" not in r)
    r2 = reporting_on(cal, pf, date(2026, 8, 14), window_days=5)
    check("window reaches backwards", "INE2" in r2)
    r3 = reporting_on(cal, pf, date(2026, 8, 20), window_days=0)
    check("a future-only date yields nothing today", r3 == {})
    # THE LIMIT: the calendar cannot answer for dates before it began accumulating.
    check("pre-epoch date returns empty, not a wrong answer",
          reporting_on(cal, pf, date(2026, 7, 1)) == {})

    # ---- mail_due
    led = pd.DataFrame([{"season": "Q1FY27", "isin": "INE2", "doc_type": "presentation"}])
    due = mail_due(pf, queue, cal, led, "Q1FY27", on=date(2026, 8, 14))
    kinds = {(d["isin"], d["doc_type"]) for d in due}
    check("present + unmailed is due", ("INE1", "results") in kinds)
    check("already mailed is suppressed", ("INE2", "presentation") not in kinds)
    check("pending is not due", ("INE1", "presentation") not in kinds)
    check("missing is not due", ("INE3", "results") not in kinds)

    # require_calendar gates RESULTS only
    due2 = mail_due(pf, queue, cal, pd.DataFrame(), "Q1FY27",
                    on=date(2026, 8, 14), require_calendar=True)
    k2 = {(d["isin"], d["doc_type"]) for d in due2}
    check("results needs a calendar date when required", ("INE1", "results") in k2)
    q3 = _q([{"isin": "INE2", "doc_type": "results", "status": "done"}])
    due3 = mail_due(pf, q3, cal, pd.DataFrame(), "Q1FY27",
                    on=date(2026, 8, 14), require_calendar=True)
    check("a holding not reporting today is gated out",
          ("INE2", "results") not in {(d["isin"], d["doc_type"]) for d in due3})
    due4 = mail_due(pf, q3, cal, pd.DataFrame(), "Q1FY27",
                    on=date(2026, 8, 14), require_calendar=False)
    check("without the gate it is due", ("INE2", "results")
          in {(d["isin"], d["doc_type"]) for d in due4})

    # ---- THE DOWNGRADE CASE: a SECOND document of the same type in one season must
    # still mail. Keying on (isin, doc_type) alone suppressed it, so a rating downgrade
    # arriving after a routine reaffirmation would never have reached the reader.
    q_two = _q([
        {"isin": "INE1", "doc_type": "rating", "status": "done",
         "announcement_date": "2026-07-01", "doc_id": "old_reaffirm"},
        {"isin": "INE1", "doc_type": "rating", "status": "done",
         "announcement_date": "2026-08-10", "doc_id": "new_downgrade"},
    ])
    lat = latest_doc_per_type([pf[0]], q_two)
    check("latest document wins", lat[("INE1", "rating")]["doc_id"] == "new_downgrade")
    led_old = pd.DataFrame([{"season": "Q1FY27", "isin": "INE1", "doc_type": "rating",
                             "doc_id": "old_reaffirm"}])
    due_new = mail_due([pf[0]], q_two, pd.DataFrame(), led_old, "Q1FY27",
                       on=date(2026, 8, 14))
    check("a NEW rating after one was mailed is still due",
          any(d["doc_type"] == "rating" for d in due_new))
    check("the due item is the new document",
          due_new[0]["doc_id"] == "new_downgrade")
    led_new = pd.DataFrame([{"season": "Q1FY27", "isin": "INE1", "doc_type": "rating",
                             "doc_id": "new_downgrade"}])
    check("once the new document is mailed it stops being due",
          not mail_due([pf[0]], q_two, pd.DataFrame(), led_new, "Q1FY27",
                       on=date(2026, 8, 14)))
    # Legacy rows (written before doc_id existed) must not cause a mass re-send.
    led_legacy = pd.DataFrame([{"season": "Q1FY27", "isin": "INE1",
                                "doc_type": "rating"}])
    check("legacy ledger rows still suppress, no re-send flood",
          not mail_due([pf[0]], q_two, pd.DataFrame(), led_legacy, "Q1FY27",
                       on=date(2026, 8, 14)))

    # ---- season status: four distinct states, each with a reason
    q_st = _q([
        {"isin": "INE1", "doc_type": "results", "status": "done",
         "announcement_date": "2026-08-01", "doc_id": "r1"},
        {"isin": "INE2", "doc_type": "results", "status": "pending"},
    ])
    cal_st = pd.DataFrame([{"symbol": "CCC", "meeting_date": "2026-08-12",
                            "purpose": "Financial Results"}])
    led_st = pd.DataFrame([{"season": "Q1FY27", "isin": "INE1",
                            "doc_type": "results", "doc_id": "r1"}])
    st = season_status(pf, q_st, cal_st, led_st, "Q1FY27", on=date(2026, 8, 14))
    def _state(sym, dt):
        return [r for r in st if r["symbol"] == sym and r["doc_type"] == dt][0]
    check("mailed document -> delivered", _state("AAA", "results")["state"] == DELIVERED)
    check("fetched but unprocessed -> awaiting",
          _state("BBB", "results")["state"] == AWAITING)
    check("calendar says reported, nothing landed -> awaiting",
          _state("CCC", "results")["state"] == AWAITING)
    check("the calendar reason is explicit",
          "calendar says reported" in _state("CCC", "results")["reason"])
    check("no deck is 'no information', not a failure",
          _state("AAA", "presentation")["state"] == NO_INFO)
    check("a missing rating explains itself",
          "no rating issued" in _state("AAA", "rating")["reason"])
    roll = season_rollup(st)
    check("rollup counts companies", roll["_companies"] == 3)
    check("rollup counts delivered results", roll["_results_done"] == 1)
    check("rollup covers every state key",
          set(roll["results"]) == {DELIVERED, DUE, AWAITING, NO_INFO})

    # ---- empties must never raise
    check("no queue is all-missing",
          set(coverage(pf, pd.DataFrame(), "Q1FY27")["status"]) == {MISSING})
    check("no calendar is empty, not an error", reporting_on(None, pf, date(2026, 8, 14)) == {})
    check("no ledger means nothing mailed", already_mailed(None, "Q1FY27") == set())

    print(f"\npf_coverage self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
