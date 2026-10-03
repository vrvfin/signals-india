r"""
test_both_reports.py — offline checks for the BOTH-reports path (deep dive + story).

Plain python, no pytest, no Drive, no network: every check runs on in-memory frames.

  1. quarter_status — the story judges financials by "is the latest DUE quarter stored"
     (SEBI 45 days / 60 for Q4 + board-meeting calendar), not by days since download.
  2. narrative_queue — `--add X --with-deepdive` marks a row BOTH; a plain --add never
     downgrades it; re-queueing a done row resets it.
  3. _queued_part_b — a BOTH row waits while its deep dive is pending, attaches the
     Drive copy once built, and says why when there is none.
  4. deep_dive_queue `--requeue` — a DONE company goes back to pending (flipped in place,
     so _dedup_queue keeps it); without --requeue the old behaviour is unchanged.

Run:
    python scripts/tests/test_both_reports.py
Exit code is the number of failures.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import narrative_preflight as PRE          # noqa: E402
import company_narrative_report as CNR     # noqa: E402
import company_deep_report as CDR          # noqa: E402

FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class FakeStore:
    """Store stand-in: parquet(path, name) from a dict; folder() returns the path."""
    def __init__(self, tables: dict):
        self.tables = tables
        self.drive = object()

    def parquet(self, path, name):
        return self.tables.get(name, pd.DataFrame()).copy()

    def folder(self, path):
        return path

    def by_isin(self, name, isin, symbol):
        df = self.tables.get(name, pd.DataFrame())
        if df.empty:
            return df
        m = pd.Series(False, index=df.index)
        for col, val in (("isin", isin), ("symbol", symbol)):
            if col in df.columns:
                m |= df[col].astype(str) == val
        return df[m].copy()


def cal(*rows):
    return pd.DataFrame([{"symbol": s, "meeting_date": d} for s, d in rows])


# ------------------------------------------------------------------ 1 quarter --
print("1. quarter_status (latest DUE quarter, SEBI 45/60 + board meeting)")
OCT2 = date(2026, 10, 2)
cases = [
    # (label, today, stored, calendar rows, expect status, expect due, text in detail)
    ("2 Oct, Jun stored, meeting 8 Oct -> not due", OCT2, "Jun 2026",
     [("TCS", "2026-10-08")], "PASS", False, "board meeting 08 Oct 2026"),
    ("2 Oct, Jun stored, no meeting -> not due (deadline)", OCT2, "Jun 2026",
     [], "PASS", False, "SEBI deadline 14 Nov 2026"),
    ("2 Oct, Jun stored, OLD meeting 11 Aug ignored", OCT2, "Jun 2026",
     [("TCS", "2026-08-11")], "PASS", False, "not due yet"),
    ("2 Oct, only Mar stored -> Jun overdue", OCT2, "Mar 2026",
     [], "WARN", True, "Jun 2026 results are due"),
    ("10 Oct, Jun stored, meeting 8 Oct held -> Sep out", date(2026, 10, 10), "Jun 2026",
     [("TCS", "2026-10-08")], "WARN", True, "Sep 2026 results are out"),
    ("20 Nov, Jun stored, no meeting -> Sep overdue", date(2026, 11, 20), "Jun 2026",
     [], "WARN", True, "Sep 2026 results are due"),
    ("20 Nov, Sep stored -> latest", date(2026, 11, 20), "Sep 2026",
     [], "PASS", False, "the latest quarter"),
    ("Q4: 20 May 2027, Dec stored -> not due (60-day deadline 30 May)", date(2027, 5, 20),
     "Dec 2026", [], "PASS", False, "SEBI deadline 30 May 2027"),
    ("Q4: 1 Jun 2027, Dec stored -> Mar overdue", date(2027, 6, 1),
     "Dec 2026", [], "WARN", True, "Mar 2027 results are due"),
    ("nothing stored -> due", OCT2, None, [], "WARN", True, "no stored quarter"),
    ("calendar keyed by ISIN also matches", OCT2, "Jun 2026",
     [("INE467B01029", "2026-10-08")], "PASS", False, "board meeting 08 Oct 2026"),
]
for label, today, stored, rows, exp_st, exp_due, text in cases:
    st = FakeStore({"results_calendar.parquet": cal(*rows)})
    status, detail, due = PRE.quarter_status(st, "INE467B01029", "TCS", stored, today=today)
    check(label, status == exp_st and due == exp_due and text in detail,
          f"got {status} due={due}: {detail}")

# integrity(): statements + summary checks carry `due`, never FAIL on age alone
print("   integrity() wiring")
stmts = pd.DataFrame([{"symbol": "TCS", "statement": "quarterly_pl", "line_item": "Sales",
                       "period": p, "value": 1.0, "fetched_at": "2026-08-03T10:00:00"}
                      for p in ("Sep 2025", "Dec 2025", "Mar 2026", "Jun 2026")])
summ = pd.DataFrame([{"symbol": "TCS", "latest_quarter_label": "Jun 2026",
                      "fetched_at": "2026-08-03T10:00:00"}])
st = FakeStore({"TCS.parquet": stmts, "summary.parquet": summ,
                "results_calendar.parquet": cal(("TCS", "2026-10-08"))})
checks = {c["id"]: c for c in PRE.integrity(st, "INE467B01029", "TCS")}
for src in ("STALE.statements", "STALE.summary"):
    c = checks.get(src, {})
    check(f"{src}: 60-day-old download but latest quarter stored -> PASS, not due",
          c.get("status") == "PASS" and c.get("due") is False, str(c))

# ------------------------------------------------------------------ 2 nqueue ---
print("2. narrative_queue --with-deepdive")
_q = {"df": pd.DataFrame(columns=CNR.NQUEUE_COLS)}
CNR._load_nqueue = lambda store: _q["df"].copy()
CNR._save_nqueue = lambda store, df: _q.__setitem__("df", df[CNR.NQUEUE_COLS].copy())
n = CNR.enqueue_narrative(None, ["TCS", "MANINDS"], with_deepdive=True)
df = _q["df"]
check("two BOTH rows added", n == 2 and df["with_deepdive"].tolist() == [True, True], df.to_string())
CNR.enqueue_narrative(None, ["TCS"])                       # plain add on a pending BOTH row
check("plain --add does not downgrade a pending BOTH row",
      bool(_q["df"].loc[_q["df"]["token"] == "TCS", "with_deepdive"].iloc[0]) is True)
_q["df"] = pd.DataFrame([{"token": "INFY", "status": "pending", "added_at": "x",
                          "done_at": None, "error": None, "with_deepdive": None}])
CNR.enqueue_narrative(None, ["INFY"], with_deepdive=True)  # upgrade story-only -> BOTH
check("BOTH on a pending story-only row upgrades it",
      _q["df"]["with_deepdive"].iloc[0] is True or _q["df"]["with_deepdive"].iloc[0] == True)
_q["df"] = pd.DataFrame([{"token": "SYRMA", "status": "done", "added_at": "x",
                          "done_at": "y", "error": None, "with_deepdive": True}])
CNR.enqueue_narrative(None, ["SYRMA"])                     # re-queue done row, story only
r = _q["df"].iloc[0]
check("re-queue of a done row -> pending, story-only, done_at cleared",
      r["status"] == "pending" and r["with_deepdive"] == False and r["done_at"] is None, str(r.to_dict()))
old = pd.DataFrame([{"token": "OLD", "status": "pending", "added_at": "x",
                     "done_at": None, "error": None}])        # file written before the column
_q["df"] = old.assign(with_deepdive=None)
CNR.enqueue_narrative(None, ["NEW1"])
check("old rows without the column keep working (None = story only)",
      len(_q["df"]) == 2 and _q["df"]["with_deepdive"].iloc[0] is None)

# ------------------------------------------------------------------ 3 part B ---
print("3. _queued_part_b (wait / attach from Drive)")
facts = pd.DataFrame([{"isin": "INE993A01026", "symbol": "MANINDS",
                       "name": "Man Industries (India) Limited", "mcap_cr": 1.0}])
ddi = pd.DataFrame([{"isin": "INE993A01026", "symbol": "MANINDS",
                     "report_path": "company_repo/INE993A01026/company_deepdive_03Oct26.md"}])
CNR.find_file = lambda drive, folder, name: f"fid:{folder}/{name}"
CNR.download_bytes = lambda drive, fid: b"# Deep Dive - Man Industries\n"
tmp = tempfile.mkdtemp()
st = FakeStore({"company_facts.parquet": facts, "deep_dive_index.parquet": ddi,
                "deep_dive_queue.parquet": pd.DataFrame(
                    [{"token": "Man Industries", "status": "pending"}])})
p, why = CNR._queued_part_b(st, "Man Industries", tmp)
check("deep dive still pending -> wait", p is None and "still queued" in why, why)
st.tables["deep_dive_queue.parquet"] = pd.DataFrame([{"token": "Man Industries", "status": "done"}])
p, why = CNR._queued_part_b(st, "Man Industries", tmp)
check("deep dive done -> Drive copy attached",
      bool(p) and os.path.exists(p) and open(p, encoding="utf-8").read().startswith("# Deep Dive"), str(p))
st.tables["deep_dive_index.parquet"] = pd.DataFrame(columns=["isin", "report_path"])
st.tables["deep_dive_queue.parquet"] = pd.DataFrame(
    [{"token": "Man Industries", "status": "error", "error": "unresolved"}])
p, why = CNR._queued_part_b(st, "Man Industries", tmp)
check("no deep dive + queue error -> wait, reason names the error",
      p is None and "error" in why, why)
p, why = CNR._queued_part_b(st, "NOSUCHCO", tmp)
check("unresolvable token -> no wait (run_one reports it)", p == "" and why == "")

# ------------------------------------------------------------------ 4 requeue --
print("4. deep_dive_queue --requeue")
dq = {"df": None}
def fake_update(svc, root, mutate, owner="x"):
    dq["df"] = CDR._dedup_queue(mutate(dq["df"].copy()))
    return dq["df"]
CDR.queue_update = fake_update
base = pd.DataFrame([{"token": "TCS", "status": "done", "added_at": "a", "done_at": "b", "error": None},
                     {"token": "TCS", "status": "done", "added_at": "c", "done_at": "d", "error": None},
                     {"token": "INFY", "status": "pending", "added_at": "e", "done_at": None, "error": None}])
dq["df"] = base.copy()
n = CDR.enqueue_tokens(None, None, ["TCS", "INFY", "SYRMA"], owner="add")
check("without --requeue: done TCS + pending INFY skipped, SYRMA added (old behaviour)",
      n == 1 and set(dq["df"][dq["df"]["status"] == "pending"]["token"]) == {"INFY", "SYRMA"}
      and (dq["df"]["token"] == "TCS").sum() == 1, dq["df"].to_string())
dq["df"] = base.copy()
n = CDR.enqueue_tokens(None, None, ["TCS", "INFY"], owner="add", requeue=True)
pend = dq["df"][dq["df"]["status"] == "pending"]
check("with --requeue: done TCS back to ONE pending row; INFY untouched",
      n == 1 and sorted(pend["token"]) == ["INFY", "TCS"] and (dq["df"]["token"] == "TCS").sum() == 1
      and pend.loc[pend["token"] == "TCS", "done_at"].isna().all(), dq["df"].to_string())

# ------------------------------------------------------------------ 5 mails ----
print("5. PDF-only mails")
import format_deepdive_pdf as FMT                      # noqa: E402
import notify_deepdive as ND                           # noqa: E402
md = ("# Deep Dive — X\n\n### 0.4 COST\n• first point [Concall]\n• second point\n"
      "\n| a | b |\n|---|---|\n| • not a bullet | 2 |\n")
html = FMT._md_to_html_body(md)
check("'• ' lines become a real list (not one run-on paragraph)",
      html.count("<li>") == 2 and "first point" in html and "• first" not in html, html[:200])
check("a '•' inside a table cell is left alone", "• not a bullet" in html)
try:
    FMT._ensure_native_libs()
    import weasyprint  # noqa: F401
    have_wp = True
except Exception:
    have_wp = False
if have_wp:
    msg = ND._build_email("X Ltd", "XLTD", "INE000X01010", md, "https://drive")
    parts = [(p.get_filename(), p.get_content_type()) for p in msg.walk() if p.get_filename()]
    check("deep dive mail: exactly ONE attachment, the PDF",
          len(parts) == 1 and parts[0][0].endswith(".pdf") and parts[0][1] == "application/pdf",
          str(parts))
    body = [p for p in msg.walk() if p.get_content_type() == "text/html"][0].get_payload(decode=True).decode()
    check("deep dive mail body is formatted HTML, not raw markdown",
          "<li>" in body and "• first" not in body)
else:
    print("  [skip] weasyprint not importable here — PDF mail checks run in CI")
_real = FMT.md_to_pdf
FMT.md_to_pdf = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no pango"))
msg = ND._build_email("X Ltd", "XLTD", "INE000X01010", md, "https://drive")
FMT.md_to_pdf = _real
parts = [(p.get_filename(), p.get_content_type()) for p in msg.walk() if p.get_filename()]
body = [p for p in msg.walk() if p.get_content_type() == "text/html"][0].get_payload(decode=True).decode()
check("PDF failure -> the styled .html is attached instead, and the mail says so",
      len(parts) == 1 and parts[0][0].endswith(".html") and "PDF could not be made" in body,
      str(parts))

print(f"\n{len(FAILS)} failure(s)" + (": " + "; ".join(FAILS) if FAILS else ""))
sys.exit(len(FAILS))
