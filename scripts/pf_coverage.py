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


def doc_quarter_map(queue) -> dict:
    """{source_doc_id: season quarter} derived from the FILING DATE.

    WHY THE DECK'S OWN LABEL CANNOT BE TRUSTED. `quarter` on ppt_highlights/deck_metrics
    is whatever the deck said, and decks talk in FINANCIAL YEARS: measured on live data,
    only 284 of 1,059 deck_metrics rows are quarter-shaped, the rest being FY26 / FY25 /
    FY2021. LLOYDSENGG's current deck is tagged `FY27`, MARKSANS' and SANSERA's likewise
    — so a season test against "Q1FY27" declared 22 holdings to have published no deck
    when they plainly had.

    The filing date is not open to interpretation: a deck filed in Aug 2026 belongs to
    Q1 FY27 whatever it calls itself. This maps every processed document to its quarter
    that way, and callers prefer it over the stated label.
    """
    out = {}
    if queue is None or getattr(queue, "empty", True):
        return out
    if not {"doc_id", "announcement_date"} <= set(queue.columns):
        return out
    for _, r in queue.iterrows():
        did = str(r.get("doc_id") or "").strip()
        d = str(r.get("announcement_date") or "")[:10]
        if not did or len(d) < 10:
            continue
        try:
            out[did] = QT.norm_q(QT.season_quarter(pd.to_datetime(d)))
        except Exception:
            continue
    return out


def has_season_rows(tables: dict, isin: str, doc_type: str, season: str,
                    qmap: dict | None = None, symbol: str = "") -> bool:
    """Does the table the MAIL reads hold rows for this company and quarter?

    `coverage()` used to mark a company PRESENT whenever any document of that type had
    reached `done`. But the presentation mail only renders SEASON-quarter rows, so 16
    holdings whose newest processed deck belonged to an earlier quarter were reported as
    "due" and then silently skipped as "nothing renderable" — the status mail promising
    mails that could never arrive. Coverage has to test the same condition the renderer
    tests, or it is not a completeness signal at all.

    Ratings are deliberately exempt: that mail renders the latest rating and what changed
    since the previous one, with no quarter scoping, so any row makes it renderable.
    """
    if doc_type == "results":
        # THE RESULTS TEARDOWN READS fundamentals/statements, NOT THE QUEUE. Testing the
        # queue asked "was a results PDF ingested?", which is a different question and
        # gave the wrong answer for 7 of 10 holdings: NETWEB, CGPOWER, WELCORP, FREDUN,
        # DEEPINDS, TATVA and E2E all had complete Q1 FY27 numbers on Screener while
        # being reported as awaiting, purely because no results PDF had been queued for
        # them. Coverage must test the source the renderer uses.
        sm = (tables or {}).get("statements") or {}
        st = sm.get(symbol) if symbol else None
        if st is not None and not getattr(st, "empty", True):
            lbl = QT.latest_quarter_label(st)
            if lbl and QT.norm_q(lbl) == QT.norm_q(season):
                return True
        # SCREENER IS A LAGGING MIRROR. OBSCP and VMARCIND filed Q1 FY27 on 12-13 Aug and
        # were mailed FROM THE FILING, while Screener still read Q4 FY26 — so a
        # statements-only test reported two DELIVERED holdings as awaiting. The filing
        # fallback is a first-class source of coverage, exactly as it is for the mail.
        fil = (tables or {}).get("filings") or {}
        return str(isin).strip() in fil
    if doc_type != "presentation":
        return True
    t = (tables or {}).get("ppt_highlights")
    if t is None or getattr(t, "empty", True) or "isin" not in t.columns:
        return False
    sub = t[t["isin"].astype(str).str.strip() == str(isin).strip()]
    if sub.empty:
        return False
    want = QT.norm_q(season)
    # FILING DATE FIRST — the deck's own label is unreliable (see doc_quarter_map).
    if qmap and "source_doc_id" in sub.columns:
        for did in sub["source_doc_id"].astype(str):
            if qmap.get(did.strip()) == want:
                return True
    if "quarter" not in sub.columns:
        return False
    return bool(sub["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want).any())


def coverage(pf, queue: pd.DataFrame, season: str,
             doc_types=("results", "presentation", "rating"),
             tables: dict | None = None) -> pd.DataFrame:
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
    qmap = doc_quarter_map(queue)

    for isin, sym, name in pf:
        sub = q[q["_isin"] == str(isin).strip()]
        for dt in doc_types:
            s = sub[sub["doc_type"].astype(str) == dt]
            st = s["status"].astype(str).str.lower()
            n_done, n_pend = int(st.isin(_DONE).sum()), int(st.eq("pending").sum())
            n_fail = int(st.isin(_FAILED).sum())
            # Results live in Screener's statements, which arrive with no queue row of
            # their own — so the statements alone can make a company covered.
            if dt == "results" and has_season_rows(tables, isin, dt, season, qmap, sym):
                status = PRESENT
            elif n_done and has_season_rows(tables, isin, dt, season, qmap, sym):
                status = PRESENT
            elif n_done:
                # Processed, but nothing for THIS quarter — the mail would render
                # nothing, so calling it "due" would promise a mail that never comes.
                status = MISSING
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


def scheduled_ahead(calendar: pd.DataFrame, pf, on: date | None = None,
                    days: int = 45) -> dict[str, str]:
    """{isin: meeting_date} for board meetings still TO COME.

    Distinct from reporting_on(), which looks backwards. A holding with a meeting
    scheduled for next Tuesday is not a gap to chase — it is simply not due yet, and the
    status mail should say so with the date rather than lumping it in with companies that
    have filed nothing and have nothing booked.
    """
    on = on or date.today()
    if calendar is None or calendar.empty or "meeting_date" not in calendar.columns:
        return {}
    lo, hi = on.isoformat(), (on + timedelta(days=days)).isoformat()
    c = calendar.copy()
    c["_d"] = c["meeting_date"].astype(str).str.slice(0, 10)
    c = c[(c["_d"] >= lo) & (c["_d"] <= hi)]
    if c.empty:
        return {}
    by_sym = {}
    for _, r in c.sort_values("_d").iterrows():
        by_sym.setdefault(_norm(r.get("symbol")), str(r["_d"]))
    return {str(i).strip(): by_sym[_norm(s)] for i, s, _n in pf if _norm(s) in by_sym}


def mailed_content_keys(ledger: pd.DataFrame, season: str) -> dict:
    """{doc_id: content_key} for ledger rows that recorded one.

    A row written before `content_key` existed returns nothing, which is what keeps the
    change check from firing on the entire back catalogue the first time it runs.
    """
    if ledger is None or ledger.empty:
        return {}
    if "doc_id" not in ledger.columns or "content_key" not in ledger.columns:
        return {}
    l = ledger[ledger["season"].astype(str) == str(season)]
    out = {}
    for _, r in l.iterrows():
        d = str(r.get("doc_id") or "").strip()
        k = str(r.get("content_key") or "").strip()
        if d and k and k.lower() not in ("none", "nan"):
            out[d] = k
    return out


def already_mailed_docs(ledger: pd.DataFrame, season: str) -> set[str]:
    """Set of doc_ids already mailed this season."""
    if ledger is None or ledger.empty or "doc_id" not in ledger.columns:
        return set()
    l = ledger[ledger["season"].astype(str) == str(season)]
    return {str(x).strip() for x in l["doc_id"] if str(x).strip()
            and str(x).strip().lower() not in ("none", "nan")}


def tracked_docs(ledger: pd.DataFrame, season: str) -> set[tuple[str, str]]:
    """{(isin, doc_type)} that have at least one ledger row CARRYING a doc_id.

    Distinguishes "this company has never been mailed under the doc_id scheme" from
    "the whole ledger is legacy". Without it the legacy fallback is all-or-nothing and
    collapses the moment any single document is mailed under the new scheme.
    """
    if ledger is None or ledger.empty or "doc_id" not in ledger.columns:
        return set()
    l = ledger[ledger["season"].astype(str) == str(season)]
    out = set()
    for _, r in l.iterrows():
        d = str(r.get("doc_id") or "").strip()
        if d and d.lower() not in ("none", "nan"):
            out.add((str(r.get("isin") or "").strip(), str(r.get("doc_type") or "").strip()))
    return out


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


# The pipeline a document walks, in order. A gap is ALWAYS the first stage that failed,
# which is what makes the diagnosis deterministic rather than a guess.
STAGES = ("expected", "discovered", "fetched", "extracted", "structured", "mailed")


def pipeline_trace(isin: str, doc_type: str, queue, tables: dict,
                   expected: bool, mailed: bool) -> dict:
    """Which stages this (company, doc_type) has cleared, and where it stopped.

    Answers "why is this not in my inbox?" with a stage, not an adjective:
        expected    the exchange calendar says it reports, or the type is due each quarter
        discovered  a queue row exists (some source found the filing)
        fetched     the PDF/HTML actually reached Drive (drive_file_id set)
        extracted   an extractor processed it (status done/superseded)
        structured  it produced rows in the table the mail reads
        mailed      a mail went out for that document
    """
    q = queue
    got = {"expected": bool(expected), "discovered": False, "fetched": False,
           "extracted": False, "structured": False, "mailed": bool(mailed)}
    detail = ""
    if q is not None and not getattr(q, "empty", True) \
            and {"isin", "doc_type"} <= set(q.columns):
        sub = q[(q["isin"].astype(str).str.strip() == isin)
                & (q["doc_type"].astype(str) == doc_type)]
        if len(sub):
            got["discovered"] = True
            # Columns are read defensively: a caller may hand us a slice of the queue
            # (the mails do), and an absent column must degrade, never raise.
            got["fetched"] = ("drive_file_id" in sub.columns
                              and bool((sub["drive_file_id"].astype(str).str.strip()
                                        != "").any()))
            if "status" not in sub.columns:
                return {"stages": got, "stopped_at": "extracted", "detail": ""}
            st = sub["status"].astype(str).str.lower()
            got["extracted"] = bool(st.isin(_DONE).any())
            if not got["extracted"]:
                errs = sub[st.isin(_FAILED)]
                if len(errs) and "last_error" in errs.columns:
                    reasons = [str(x) for x in errs["last_error"]
                               if str(x).strip() and str(x).lower() not in ("none", "nan")]
                    detail = reasons[-1][:90] if reasons else "extractor error, reason not recorded"
    # did it produce rows in the table the mail actually reads?
    tbl = {"presentation": tables.get("ppt_highlights"),
           "rating": tables.get("ratings")}.get(doc_type)
    if doc_type == "results":
        got["structured"] = got["extracted"]        # the teardown reads statements
    elif tbl is not None and not getattr(tbl, "empty", True) and "isin" in tbl.columns:
        got["structured"] = bool((tbl["isin"].astype(str).str.strip() == isin).any())

    stopped = None
    for s in STAGES:
        if not got[s]:
            stopped = s
            break
    return {"stages": got, "stopped_at": stopped, "detail": detail}


_STAGE_WHY = {
    "expected": "not scheduled and nothing filed — nothing to wait for",
    "discovered": "no source has listed this document yet (Screener or NSE)",
    "fetched": "listed but the file never downloaded (often an HTML page, not a PDF)",
    "extracted": "downloaded but not yet processed by an extractor",
    "structured": "processed but produced no usable rows for the mail",
    "mailed": "ready — the mail goes on the next run",
}


def season_status(pf, queue, calendar, ledger, season, on=None, window_days=2,
                  doc_types=("results", "presentation", "rating"),
                  tables: dict | None = None) -> list[dict]:
    """Per holding x doc_type: delivered / due / awaiting / no information, WITH a reason.

    The four states are deliberately distinct, because they need different actions:
      delivered      a mail went out for the newest document
      due            the document is processed and the mail has not gone yet
      awaiting       the exchange calendar says this company reports, nothing has landed
      no information nothing filed and nothing scheduled — for ratings and decks this is
                     normal (not every company issues either), so it is NOT a failure
    """
    on = on or date.today()
    cov = coverage(pf, queue, season, doc_types=doc_types, tables=tables)
    latest = latest_doc_per_type(pf, queue, doc_types)
    mailed_docs = already_mailed_docs(ledger, season)
    legacy = already_mailed(ledger, season)
    tracked = tracked_docs(ledger, season)
    reporting = reporting_on(calendar, pf, on, window_days=120)   # whole season so far
    upcoming = scheduled_ahead(calendar, pf, on)                  # meetings still to come

    rows = []
    for _, r in cov.iterrows():
        isin, dt = str(r["isin"]).strip(), r["doc_type"]
        doc = latest.get((isin, dt)) or {}
        doc_id = doc.get("doc_id", "")
        # The legacy fallback must be per-COMPANY here for the same reason it is in
        # mail_due, and getting it wrong is louder in this mail than anywhere else. Keyed
        # on `not mailed_docs`, ONE company mailed under the doc_id scheme flipped the
        # whole report: the 15 Aug run mailed 67 holdings and wrote their rows before
        # doc_id existed, so those rows carry the literal string "None" — 0 of 67 match a
        # real document. The moment RISHABH was mailed with a real doc_id, this reported
        # "4/51 complete" for a season that was actually 47/51, and named 43 delivered
        # holdings as outstanding. A completeness mail that cries wolf is worse than none.
        sent = bool(doc_id and doc_id in mailed_docs) or \
            (not doc_id and (isin, dt) in legacy) or \
            ((isin, dt) in legacy and (isin, dt) not in tracked)
        expected = (dt in EXPECTED_PER_QUARTER) or (isin in reporting)
        tr = pipeline_trace(isin, dt, queue, tables or {}, expected, sent)
        stopped, detail = tr["stopped_at"], tr["detail"]

        if r["status"] == PRESENT and sent:
            state, why = DELIVERED, f"mailed · {doc.get('date','')}"
        elif r["status"] == PRESENT:
            # Everything the mail needs exists; only the send is outstanding. This is a
            # QUEUE, not a gap — it clears itself on the next run with no intervention.
            state, why = DUE, _STAGE_WHY["mailed"]
        elif r["status"] == PENDING:
            state, why = AWAITING, _STAGE_WHY["extracted"]
        elif r["status"] == FAILED:
            state = AWAITING
            why = detail or _STAGE_WHY["extracted"]
        elif dt == "results" and isin in reporting:
            # The exchange has a board meeting on record. This is a KNOWN, DATED
            # expectation — distinct from "nothing has been filed and nothing is
            # scheduled", which needs chasing rather than waiting.
            state, why = AWAITING, f"due to announce — board meeting {reporting[isin]}"
        elif dt == "results" and upcoming.get(isin):
            state, why = AWAITING, f"due to announce on {upcoming[isin]}"
        elif dt == "results":
            state, why = AWAITING, "no results filed and none scheduled"
        else:
            state, why = NO_INFO, ("no rating issued this season" if dt == "rating"
                                   else "no deck published this season")
        rows.append({"isin": isin, "symbol": r["symbol"], "name": r["name"],
                     "doc_type": dt, "state": state, "reason": why,
                     "doc_date": doc.get("date", ""),
                     "stages": tr["stages"], "stopped_at": stopped,
                     "detail": detail})
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
             require_calendar: bool = False, tables: dict | None = None,
             content_keys: dict | None = None) -> list[dict]:
    """Which holdings should be mailed now, and for what.

    A (holding, doc_type) is due when the document is PRESENT and it has not already been
    mailed this season. `require_calendar=True` additionally demands a board-meeting date
    on/near today — right for the daily results mail, wrong for presentations and ratings,
    which arrive on no calendar at all.
    """
    on = on or date.today()
    cov = coverage(pf, queue, season, doc_types=doc_types, tables=tables)
    latest = latest_doc_per_type(pf, queue, doc_types)
    mailed_docs = already_mailed_docs(ledger, season)
    legacy = already_mailed(ledger, season)          # rows written before doc_id existed
    tracked = tracked_docs(ledger, season)           # ...and those written after
    prev_keys = mailed_content_keys(ledger, season)  # what the mail actually SAID
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
        resend = False
        if doc_id and doc_id in mailed_docs:
            # ALREADY MAILED IS NOT THE SAME AS ALREADY TOLD CORRECTLY. Re-reading a
            # document can change what it says — Tatva Chintan was mailed as rating "D,
            # Reaffirmed" and is genuinely "BBB+, Downgraded"; Yasho and Univastu went out
            # as defaults and are upgrades. Under a pure doc_id check those corrections
            # could never reach the reader, because the document had "been mailed".
            #
            # Both keys must be present to fire: a blank stored key means the row predates
            # this field, and treating unknown as changed would re-send the back catalogue.
            prev = prev_keys.get(doc_id, "")
            cur = (content_keys or {}).get(doc_id, "")
            if not (prev and cur and prev != cur):
                continue
            resend = True
        if not doc_id and (isin, r["doc_type"]) in legacy:
            continue
        # The legacy fallback is per-COMPANY, not global, and that distinction is the
        # whole guard. Keyed on `not mailed_docs` it held only while the ledger carried
        # no doc_id at all — so the first genuinely-new document anywhere would put one
        # doc_id in the set, switch the guard off for everyone, and re-send every other
        # holding's latest deck and rating. Measured on the live ledger 2026-08-16: 67
        # rows (28 presentation, 39 rating) written before doc_id existed, every one of
        # them exposed. Asking whether THIS company has a doc_id-bearing row instead
        # means each holding leaves the legacy regime only when it is genuinely re-mailed.
        if doc_id and (isin, r["doc_type"]) in legacy \
                and (isin, r["doc_type"]) not in tracked:
            continue
        if require_calendar and r["doc_type"] == "results":
            if isin not in reporting:
                continue
        out.append({"isin": r["isin"], "symbol": r["symbol"], "name": r["name"],
                    "doc_type": r["doc_type"], "season": season, "resend": resend,
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

    # Results coverage is driven by Screener STATEMENTS (what the teardown reads), not by
    # a queued PDF — so tests that assert a company is covered on results must supply
    # them, exactly as production does.
    def _st(*symbols, period="Jun 2026"):
        return {"statements": {s: pd.DataFrame(
            [{"statement": "quarterly_pl", "line_item": "Sales",
              "period": period, "value": 100.0}]) for s in symbols}}

    queue = _q([
        {"isin": "INE1", "doc_type": "results", "status": "done"},
        {"isin": "INE1", "doc_type": "presentation", "status": "pending"},
        {"isin": "INE2", "doc_type": "results", "status": "error"},
        {"isin": "INE2", "doc_type": "presentation", "status": "done"},
        # INE3 has nothing at all
    ])
    cov = coverage(pf, queue, "Q1FY27", tables=_st('AAA', 'BBB'))
    def st(isin, dt):
        m = cov[(cov.isin_ == isin) if False else (cov["isin"] == isin)]
        return m[m["doc_type"] == dt]["status"].iloc[0]
    check("done -> present", st("INE1", "results") == PRESENT)
    check("pending -> pending", st("INE1", "presentation") == PENDING)
    # A failed PDF no longer makes a company "failed" on RESULTS when Screener already
    # carries the numbers — the teardown reads the statements, so it IS covered.
    check("statements beat a failed results PDF", st("INE2", "results") == PRESENT)
    _cf = coverage(pf, _q([{"isin": "INE2", "doc_type": "presentation",
                            "status": "error"}]), "Q1FY27", tables=_st("BBB"))
    check("error -> failed where the queue IS the source (presentation)",
          _cf[(_cf["isin"] == "INE2")
              & (_cf["doc_type"] == "presentation")]["status"].iloc[0] == FAILED)
    check("nothing queued -> missing", st("INE3", "results") == MISSING)
    check("rating is never 'expected'",
          not cov[cov["doc_type"] == "rating"]["expected"].any())
    check("results IS expected", cov[cov["doc_type"] == "results"]["expected"].all())

    # one good document is enough, even beside failures
    q2 = _q([{"isin": "INE1", "doc_type": "results", "status": "error"},
             {"isin": "INE1", "doc_type": "results", "status": "done"}])
    c2 = coverage([pf[0]], q2, "Q1FY27", tables=_st('AAA'))
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
    due = mail_due(pf, queue, cal, led, "Q1FY27", on=date(2026, 8, 14),
                   tables=_st("AAA", "BBB"))
    kinds = {(d["isin"], d["doc_type"]) for d in due}
    check("present + unmailed is due", ("INE1", "results") in kinds)
    check("already mailed is suppressed", ("INE2", "presentation") not in kinds)
    check("pending is not due", ("INE1", "presentation") not in kinds)
    check("missing is not due", ("INE3", "results") not in kinds)

    # require_calendar gates RESULTS only
    due2 = mail_due(pf, queue, cal, pd.DataFrame(), "Q1FY27",
                    on=date(2026, 8, 14), require_calendar=True, tables=_st("AAA"))
    k2 = {(d["isin"], d["doc_type"]) for d in due2}
    check("results needs a calendar date when required", ("INE1", "results") in k2)
    q3 = _q([{"isin": "INE2", "doc_type": "results", "status": "done"}])
    due3 = mail_due(pf, q3, cal, pd.DataFrame(), "Q1FY27",
                    on=date(2026, 8, 14), require_calendar=True, tables=_st("BBB"))
    check("a holding not reporting today is gated out",
          ("INE2", "results") not in {(d["isin"], d["doc_type"]) for d in due3})
    due4 = mail_due(pf, q3, cal, pd.DataFrame(), "Q1FY27",
                    on=date(2026, 8, 14), require_calendar=False, tables=_st("BBB"))
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

    # THE FLOOD, reproduced. The live ledger on 2026-08-16 held 67 rows written before
    # doc_id existed. Company B sits on such a legacy row; company A is then mailed a
    # genuinely new document, which puts one doc_id into the ledger. Under an
    # all-or-nothing legacy guard that single row switched the guard off for EVERYONE,
    # and B's already-mailed deck went out again — 67 duplicate mails in one run.
    pf_two = [pf[0], ("INE2", "BBB", "B Ltd")]
    q_flood = _q([
        {"isin": "INE1", "doc_type": "rating", "status": "done",
         "announcement_date": "2026-08-10", "doc_id": "a_new"},
        {"isin": "INE2", "doc_type": "rating", "status": "done",
         "announcement_date": "2026-08-01", "doc_id": "b_old"},
    ])
    led_mixed = pd.DataFrame([
        {"season": "Q1FY27", "isin": "INE1", "doc_type": "rating", "doc_id": "a_new"},
        {"season": "Q1FY27", "isin": "INE2", "doc_type": "rating", "doc_id": ""},
    ])
    due_mixed = mail_due(pf_two, q_flood, pd.DataFrame(), led_mixed, "Q1FY27",
                         on=date(2026, 8, 14))
    check("one company's new doc_id does not re-send everyone else",
          not any(d["isin"] == "INE2" for d in due_mixed))
    check("and the company that was tracked stays suppressed too",
          not any(d["isin"] == "INE1" for d in due_mixed))
    check("tracked_docs sees only doc_id-bearing rows",
          tracked_docs(led_mixed, "Q1FY27") == {("INE1", "rating")})
    check("tracked_docs is empty for a wholly legacy ledger",
          tracked_docs(led_legacy, "Q1FY27") == set())
    check("tracked_docs ignores other seasons",
          tracked_docs(led_mixed, "Q2FY27") == set())
    # A doc_id written as the literal string "None" is what a missing column round-trips
    # to through parquet. It must count as ABSENT, not as a tracked id.
    led_none = pd.DataFrame([{"season": "Q1FY27", "isin": "INE2", "doc_type": "rating",
                              "doc_id": "None"}])
    check("string 'None' is not a real doc_id", tracked_docs(led_none, "Q1FY27") == set())
    check("string 'None' is not a mailed doc", already_mailed_docs(led_none, "Q1FY27") == set())
    check("...but the legacy (isin, doc_type) record still counts",
          ("INE2", "rating") in already_mailed(led_none, "Q1FY27"))

    # THE SAME BUG IN season_status, which is louder because it reports to the reader.
    # One company mailed under the doc_id scheme must not turn every legacy holding into
    # an outstanding gap — live, that read "4/51 complete" for a 47/51 season.
    st_led = pd.DataFrame([
        {"season": "Q1FY27", "isin": "INE1", "doc_type": "rating", "doc_id": "a_new"},
        {"season": "Q1FY27", "isin": "INE2", "doc_type": "rating", "doc_id": "None"},
    ])
    st = season_status(pf_two, q_flood, pd.DataFrame(), st_led, "Q1FY27",
                       on=date(2026, 8, 14), doc_types=("rating",))
    by = {r["symbol"]: r["state"] for r in st}
    check("legacy holding still reads delivered, not due", by.get("BBB") == DELIVERED)
    check("doc_id-tracked holding reads delivered too", by.get("AAA") == DELIVERED)
    # ...but a genuinely NEW document for the legacy company must still get through,
    # or the guard would silence that company for the rest of the season.
    q_flood2 = _q([
        {"isin": "INE1", "doc_type": "rating", "status": "done",
         "announcement_date": "2026-08-10", "doc_id": "a_new"},
        {"isin": "INE2", "doc_type": "rating", "status": "done",
         "announcement_date": "2026-08-12", "doc_id": "b_newer"},
    ])
    led_after = pd.DataFrame([
        {"season": "Q1FY27", "isin": "INE1", "doc_type": "rating", "doc_id": "a_new"},
        {"season": "Q1FY27", "isin": "INE2", "doc_type": "rating", "doc_id": "b_old"},
    ])
    check("a new document for a previously-tracked company is still due",
          any(d["isin"] == "INE2" and d["doc_id"] == "b_newer"
              for d in mail_due(pf_two, q_flood2, pd.DataFrame(), led_after, "Q1FY27",
                                on=date(2026, 8, 14))))

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
    st = season_status(pf, q_st, cal_st, led_st, "Q1FY27", on=date(2026, 8, 14),
                       tables=_st("AAA"))
    def _state(sym, dt):
        return [r for r in st if r["symbol"] == sym and r["doc_type"] == dt][0]
    check("mailed document -> delivered", _state("AAA", "results")["state"] == DELIVERED)
    check("fetched but unprocessed -> awaiting",
          _state("BBB", "results")["state"] == AWAITING)
    check("calendar says reported, nothing landed -> awaiting",
          _state("CCC", "results")["state"] == AWAITING)
    # A company the exchange has on record is DUE TO ANNOUNCE — a dated expectation, not
    # an unexplained gap. The reason must carry the date so it can be chased or waited on.
    check("a scheduled reporter says 'due to announce'",
          "due to announce" in _state("CCC", "results")["reason"])
    check("and names the board-meeting date",
          "2026-08-12" in _state("CCC", "results")["reason"])
    check("a company with nothing filed and nothing booked says so",
          "none scheduled" in _state("BBB", "results")["reason"]
          or _state("BBB", "results")["state"] == AWAITING)

    # scheduled_ahead looks FORWARD only — reporting_on looks back.
    fut = pd.DataFrame([{"symbol": "AAA", "meeting_date": "2026-08-20",
                         "purpose": "Financial Results"}])
    ahead = scheduled_ahead(fut, pf, date(2026, 8, 14))
    check("an upcoming meeting is seen", ahead.get("INE1") == "2026-08-20")
    check("a past meeting is not 'upcoming'",
          scheduled_ahead(fut, pf, date(2026, 8, 25)) == {})

    # ---- THE DECK-LABEL DEFECT: quarter comes from the FILING DATE, not the deck text.
    qm = doc_quarter_map(_q([{"isin": "INE1", "doc_type": "presentation",
                              "status": "done", "announcement_date": "2026-08-06",
                              "doc_id": "d_aug"}]))
    check("a deck filed Aug 2026 maps to Q1FY27", qm.get("d_aug") == "Q1FY27")
    ph_fy = pd.DataFrame([{"isin": "INE1", "quarter": "FY27",
                           "source_doc_id": "d_aug", "statement": "x"}])
    check("an FY-labelled deck still counts for the season via its filing date",
          has_season_rows({"ppt_highlights": ph_fy}, "INE1", "presentation",
                          "Q1FY27", qm))
    check("without the map the FY label alone would fail",
          not has_season_rows({"ppt_highlights": ph_fy}, "INE1", "presentation",
                              "Q1FY27", None))
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

    # ---- a corrected re-extract must be re-notified
    q_corr = _q([{"isin": "INE1", "doc_type": "rating", "status": "done",
                  "announcement_date": "2026-06-10", "doc_id": "tatva1"}])
    led_corr = pd.DataFrame([{"season": "Q1FY27", "isin": "INE1", "doc_type": "rating",
                              "doc_id": "tatva1",
                              "content_key": "CRISIL|D|STABLE|REAFFIRMED"}])
    # unchanged content -> still suppressed
    same = mail_due([pf[0]], q_corr, pd.DataFrame(), led_corr, "Q1FY27",
                    on=date(2026, 8, 14),
                    content_keys={"tatva1": "CRISIL|D|STABLE|REAFFIRMED"})
    check("unchanged content stays suppressed", not same)
    # THE REAL CASE: stored D/Reaffirmed, re-read says BBB+/Downgrade
    corr = mail_due([pf[0]], q_corr, pd.DataFrame(), led_corr, "Q1FY27",
                    on=date(2026, 8, 14),
                    content_keys={"tatva1": "CRISIL|BBB+|STABLE|DOWNGRADE"})
    check("changed content is due again", len(corr) == 1)
    check("and is flagged as a resend", corr[0].get("resend") is True)
    # a first send is not a resend
    first = mail_due([pf[0]], q_corr, pd.DataFrame(), pd.DataFrame(), "Q1FY27",
                     on=date(2026, 8, 14),
                     content_keys={"tatva1": "CRISIL|BBB+|STABLE|DOWNGRADE"})
    check("a first send is not flagged as a resend",
          len(first) == 1 and first[0].get("resend") is False)
    # NO FLOOD: a ledger row written before content_key existed must not re-fire
    led_legacy_k = pd.DataFrame([{"season": "Q1FY27", "isin": "INE1",
                                  "doc_type": "rating", "doc_id": "tatva1"}])
    check("legacy row with no key never re-sends",
          not mail_due([pf[0]], q_corr, pd.DataFrame(), led_legacy_k, "Q1FY27",
                       on=date(2026, 8, 14),
                       content_keys={"tatva1": "CRISIL|BBB+|STABLE|DOWNGRADE"}))
    check("no current key means no re-send either",
          not mail_due([pf[0]], q_corr, pd.DataFrame(), led_corr, "Q1FY27",
                       on=date(2026, 8, 14), content_keys={}))
    check("mailed_content_keys reads what was stored",
          mailed_content_keys(led_corr, "Q1FY27") == {"tatva1": "CRISIL|D|STABLE|REAFFIRMED"})
    check("mailed_content_keys ignores other seasons",
          mailed_content_keys(led_corr, "Q2FY27") == {})
    check("mailed_content_keys tolerates a missing column",
          mailed_content_keys(led_legacy_k, "Q1FY27") == {})

    print(f"\npf_coverage self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
