"""
filing_results.py — the quarter's numbers as taken from the EXCHANGE FILING, for use
when Screener has not republished the company's statements yet.

Why this exists
---------------
Both results mails define "reported" as `fundamentals/statements/<SYM>.parquet` showing
the season quarter in `quarterly_pl`. Screener is a lagging mirror of the filing: on
2026-08-12 VMARCIND and OBSCP filed Q1 FY27 and were invisible in BOTH mails for days,
because Screener still read Q4 FY26.

`results_gemini.parquet` already holds revenue / EBITDA / PAT / EPS / margins extracted
from the filing PDF itself (extract_results.py). This module is the read side of that
table: one validated view, imported by pf_results_digest.py and quarter_teardown.py so
the two mails cannot drift apart. Same reason quarterly_table.py was factored out.

It is NOT a substitute for the statements. It carries ONE quarter with no history, and
callers must render it as its own filing-sourced card — never as a column inside a
Screener table, where a consolidated-vs-standalone or Revenue-vs-Sales difference would
be invisible.

Why the validation below is not paranoia
----------------------------------------
Measured against live Drive data on 2026-08-14 (123 rows / 81 companies):
  * `pat_cr` was numeric in only 25 of 123 rows. `_extractor_base.METRIC_ALIASES` is
    first-match-wins with ("pat","pat") ahead of ("margin","margin"), so a row labelled
    "PAT Margin %" identified as `pat` and wrote "5.13%" into `pat_cr`, clobbering the
    real PAT. `ebitda_cr` was corrupted the same way.
  * `revenue_cr` was numeric in 74 of 123; the rest held the literal header
    'Revenue (Cr)' — a header row parsed as data.
  * 23 rows had a blank `quarter` and are unusable by construction.
The writer bug is fixed in extract_results.py, but rows already stored stay wrong, so
every field is validated here on the way in.

Note `_extractor_base.try_float` STRIPS '%' — try_float("5.13%") returns 5.13. A percent
sign therefore has to be rejected BEFORE coercion, not after, or the mis-parse sails
through looking like a crore figure.

Self-test:  python scripts/filing_results.py --self-test
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import quarterly_table as QT

FILING_NAME = "results_gemini.parquet"

# Mirrors extract_results.RESULTS_GEMINI_COLS. Duplicated rather than imported because
# importing extract_results pulls in google-genai and the whole Gemini pool at module
# scope, which a mail renderer has no business loading.
FILING_COLS = [
    "isin", "symbol", "company_name", "quarter", "fy_year",
    "revenue_cr", "ebitda_cr", "pat_cr", "eps",
    "ebitda_margin_pct", "pat_margin_pct",
    "revenue_yoy_pct", "pat_yoy_pct",
    "processed_at", "source_doc_id",
]

# Absolute-value fields, in crore. A '%' in any of these is the margin-row mis-parse.
_CR_FIELDS = ("revenue_cr", "ebitda_cr", "pat_cr")
# Fields that legitimately carry a percentage.
_PCT_FIELDS = ("ebitda_margin_pct", "pat_margin_pct", "revenue_yoy_pct", "pat_yoy_pct")

_BLANK = {"", "none", "nan", "nat", "<na>", "null", "na"}


def _num(v, *, allow_pct: bool) -> float | None:
    """Coerce one stored value to a float, or None.

    `allow_pct=False` REJECTS anything containing '%' outright — in a crore field that
    is a mis-parsed margin row, not a number, and silently keeping it would put a
    percentage where the mail prints rupees.
    """
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in _BLANK:
        return None
    if "%" in s and not allow_pct:
        return None
    cleaned = re.sub(r"[,%₹$\s]", "", s)
    if cleaned in ("", "-", "–"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _row_to_facts(row) -> dict:
    """One parquet row -> validated dict. Numeric fields are float or None."""
    out = {
        "isin": str(row.get("isin") or "").strip(),
        "symbol": str(row.get("symbol") or "").strip(),
        "company_name": str(row.get("company_name") or "").strip(),
        "quarter": str(row.get("quarter") or "").strip(),
        "fy_year": str(row.get("fy_year") or "").strip(),
        "processed_at": str(row.get("processed_at") or "").strip(),
        "source_doc_id": str(row.get("source_doc_id") or "").strip(),
    }
    for f in _CR_FIELDS + ("eps",):
        out[f] = _num(row.get(f), allow_pct=False)
    for f in _PCT_FIELDS:
        out[f] = _num(row.get(f), allow_pct=True)
    return out


def usable(facts: dict) -> bool:
    """A row is usable only if it names a quarter and carries a real revenue.

    Revenue is the gate because a row whose revenue failed to parse is the
    header-as-value case, where every other field is suspect too. PAT / EBITDA / EPS
    are carried independently: a corrupt PAT suppresses one line, not the company.
    """
    return bool(facts.get("quarter")) and facts.get("revenue_cr") is not None


def select_season(df: pd.DataFrame, season: str) -> dict[str, dict]:
    """{isin: facts} for the season quarter. Pure — no Drive, no network.

    `quarter` is stored with a space ("Q1 FY27") while season_quarter() has none
    ("Q1FY27"), so both sides go through QT.norm_q.

    The upsert key in extract_results is (isin, quarter, source_doc_id), so two filings
    for one quarter — a board outcome and a press release, say — each leave a row. The
    newest `processed_at` wins.
    """
    if df is None or df.empty:
        return {}
    want = QT.norm_q(season)
    best: dict[str, dict] = {}
    for _, row in df.iterrows():
        facts = _row_to_facts(row)
        if not facts["isin"] or not facts["quarter"]:
            continue
        if QT.norm_q(facts["quarter"]) != want:
            continue
        if not usable(facts):
            continue
        prev = best.get(facts["isin"])
        if prev is None or facts["processed_at"] >= prev["processed_at"]:
            best[facts["isin"]] = facts
    return best


def attach_filing_dates(found: dict[str, dict], queue: pd.DataFrame | None) -> dict[str, dict]:
    """Join source_doc_id -> processing_queue for the filing date, title and url.

    results_gemini carries no filing date at all — only `processed_at`, which is when
    Gemini ran, not when the company filed. The queue is the only source for that.
    """
    if queue is None or queue.empty or not found:
        for f in found.values():
            f.setdefault("filed_on", "")
            f.setdefault("title", "")
            f.setdefault("pdf_url", "")
        return found
    q = queue.copy()
    q["_id"] = q["doc_id"].astype(str).str.strip()
    by_id = {r["_id"]: r for _, r in q.iterrows()}
    for f in found.values():
        row = by_id.get(f.get("source_doc_id", ""))
        f["filed_on"] = str(row["announcement_date"])[:10] if row is not None else ""
        f["title"] = str(row["title"]) if row is not None else ""
        f["pdf_url"] = str(row["pdf_url"]) if row is not None else ""
    return found


def load_filing_results(drive, idx, season: str, queue: pd.DataFrame | None = None,
                        log=None) -> dict[str, dict]:
    """{isin: validated facts} for the season, straight from the filings."""
    from _extractor_base import load_parquet
    df = load_parquet(drive, idx, FILING_NAME, FILING_COLS)
    found = select_season(df, season)
    found = attach_filing_dates(found, queue)
    if log:
        log(f"filing results: {len(df)} rows in {FILING_NAME} -> "
            f"{len(found)} usable for {season}")
    return found


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

    # ---- _num
    check("plain float", _num("555.51", allow_pct=False) == 555.51)
    check("comma stripped", _num("7,252.14", allow_pct=False) == 7252.14)
    check("rupee stripped", _num("₹1,234", allow_pct=False) == 1234.0)
    # THE REGRESSION: try_float() strips '%' and would return 5.13 here. A percentage in
    # a crore column is the "PAT Margin %" mis-parse — it must be rejected, not coerced.
    check("percent rejected in a Cr field", _num("5.13%", allow_pct=False) is None)
    check("negative percent rejected too", _num("-2362.76%", allow_pct=False) is None)
    check("percent kept where it belongs", _num("15.2%", allow_pct=True) == 15.2)
    check("header text is not a number", _num("Revenue (Cr)", allow_pct=False) is None)
    check("NA is not a number", _num("NA", allow_pct=False) is None)
    check("blank is not a number", _num("", allow_pct=False) is None)
    check("None is not a number", _num(None, allow_pct=False) is None)

    def _df(rows):
        return pd.DataFrame([{c: r.get(c) for c in FILING_COLS} for r in rows])

    # ---- the three real corruption shapes seen on Drive
    df = _df([
        {"isin": "INE001", "symbol": "GOOD", "quarter": "Q1 FY27",
         "revenue_cr": "555.51", "pat_cr": "42.0", "eps": "4.2",
         "processed_at": "2026-08-13T06:31:21", "source_doc_id": "d1"},
        {"isin": "INE002", "symbol": "HEADER", "quarter": "Q1 FY27",
         "revenue_cr": "Revenue (Cr)", "pat_cr": "PAT Margin %", "eps": "EPS (Rs)",
         "processed_at": "2026-08-12T14:22:47", "source_doc_id": "d2"},
        {"isin": "INE003", "symbol": "NOQTR", "quarter": "",
         "revenue_cr": "100.0", "processed_at": "2026-06-05T08:13:39",
         "source_doc_id": "d3"},
    ])
    got = select_season(df, "Q1FY27")
    check("clean row survives", "INE001" in got)
    check("header-as-value row dropped", "INE002" not in got)
    check("blank-quarter row dropped", "INE003" not in got)
    check("revenue parsed", got["INE001"]["revenue_cr"] == 555.51)
    check("pat parsed", got["INE001"]["pat_cr"] == 42.0)

    # ---- a corrupt PAT must suppress only PAT, never the company
    df = _df([{"isin": "INE010", "symbol": "VMARCIND", "quarter": "Q1 FY27",
               "revenue_cr": "555.51", "pat_cr": "5.13%", "eps": None,
               "pat_margin_pct": "5.13", "processed_at": "2026-08-13", "source_doc_id": "x"}])
    got = select_season(df, "Q1FY27")
    check("company kept despite corrupt PAT", "INE010" in got)
    check("corrupt PAT suppressed", got["INE010"]["pat_cr"] is None)
    check("revenue still available", got["INE010"]["revenue_cr"] == 555.51)
    check("margin field still read", got["INE010"]["pat_margin_pct"] == 5.13)

    # ---- season matching across the spacing difference
    df = _df([{"isin": "INE020", "symbol": "S", "quarter": "Q1 FY27",
               "revenue_cr": "10", "processed_at": "2026-08-01", "source_doc_id": "a"}])
    check("spaced quarter matches unspaced season", "INE020" in select_season(df, "Q1FY27"))
    check("wrong quarter excluded", select_season(df, "Q4FY26") == {})

    # ---- two filings for one quarter -> newest wins, never two rows
    df = _df([
        {"isin": "INE030", "symbol": "D", "quarter": "Q1 FY27", "revenue_cr": "100",
         "processed_at": "2026-08-12T10:00:00", "source_doc_id": "old"},
        {"isin": "INE030", "symbol": "D", "quarter": "Q1 FY27", "revenue_cr": "111",
         "processed_at": "2026-08-13T10:00:00", "source_doc_id": "new"},
    ])
    got = select_season(df, "Q1FY27")
    check("one row per company", len(got) == 1)
    check("newest processed_at wins", got["INE030"]["revenue_cr"] == 111.0)
    check("newest doc id kept", got["INE030"]["source_doc_id"] == "new")

    # ---- empty / missing input must not raise
    check("empty frame is empty result", select_season(pd.DataFrame(), "Q1FY27") == {})
    check("None frame is empty result", select_season(None, "Q1FY27") == {})

    # ---- queue join supplies the filing date results_gemini does not carry
    found = {"INE030": {"source_doc_id": "new"}}
    q = pd.DataFrame([{"doc_id": "new", "announcement_date": "2026-08-12",
                       "title": "Outcome of Board Meeting", "pdf_url": "http://x/y.pdf"}])
    out = attach_filing_dates(found, q)
    check("filing date joined", out["INE030"]["filed_on"] == "2026-08-12")
    check("title joined", out["INE030"]["title"].startswith("Outcome"))
    check("no queue is not a crash", attach_filing_dates({"A": {}}, None)["A"]["filed_on"] == "")

    print(f"\nfiling_results self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
