r"""
teardown_docs.py — "what the documents say" for the quarterly teardown (user step 6).

The teardown's own blocks are this repo's reading of the numbers. This block puts the
COMPANY'S documents next to them, condensed: this quarter's concall and results release,
and this financial year's annual report. Each carries the gallery-style freshness badge
(mail_freshness), so the reader sees at a glance which one landed today.

No new extraction and no model call. The text is what Phase 2 already wrote:
  1. the exact per-document report (doc_reports, keyed by doc_id) - with the same
     failed-generation guards the per-company mails use (real_report / prompt echo /
     degenerate), then
  2. the company_page.md section for that document (run_pf_docs_digest._find_region),
and it is CONDENSED with the PF documents digest's own extractor (_lift_summary, ~1,600
chars) so the teardown stays inside Gmail's 90 KB clip. The full reports stay in the
per-company concall / annual-report mails.

WHICH documents:
  concall, results  newest `done` row filed inside the season window
                    (season_summary.season_window - the same rule the rest of the
                    teardown uses, so another quarter can never appear under this one)
  annual_report     newest `done` report for the current financial year
                    (pf_company_mails.current_ar_fy / ar_fy_year)

    python scripts/teardown_docs.py --self-test
"""
from __future__ import annotations

import sys
from datetime import date

import pandas as pd

KINDS = [("results", "📊 Results release"), ("concall", "🎙 Concall"),
         ("annual_report", "📗 Annual report")]
LIMIT = 1600


def pick_docs(queue: pd.DataFrame, isin: str, season: str, today: date,
              window: tuple[str, str], ar_fy: int, ar_fy_of) -> dict:
    """{kind: row-dict} for the documents this teardown should summarise. Pure."""
    if queue is None or queue.empty:
        return {}
    q = queue[(queue["isin"].astype(str).str.strip() == str(isin).strip())
              & (queue["status"].astype(str) == "done")].copy()
    if q.empty:
        return {}
    q["_d"] = q["announcement_date"].astype(str).str.slice(0, 10)
    start, end = window
    out = {}
    for kind, _ in KINDS:
        k = q[q["doc_type"].astype(str) == kind]
        if k.empty:
            continue
        if kind == "annual_report":
            k = k[[ar_fy_of(d, p) == ar_fy for d, p in
                   zip(k["_d"], k.get("period", pd.Series([""] * len(k), index=k.index)))]]
        else:
            k = k[(k["_d"] >= start) & (k["_d"] <= end)]
        if k.empty:
            continue
        sort_col = "processed_at" if "processed_at" in k.columns else "_d"
        r = k.sort_values(sort_col).iloc[-1]
        # the date a reader cares about: when it was filed - except an AR, whose
        # announcement_date is the FY-END; there, when it reached us
        when = (str(r.get("discovered_at") or r.get("processed_at") or "")[:10]
                if kind == "annual_report" else r["_d"])
        out[kind] = {"doc_id": str(r.get("doc_id") or ""), "date": when,
                     "title": str(r.get("title") or ""), "period": str(r.get("period") or "")}
    return out


def condense(md: str, split_sections, lift_summary, limit: int = LIMIT) -> str:
    """Full report markdown -> its best narrative subsection, <= limit chars."""
    if not str(md or "").strip():
        return ""
    return lift_summary(split_sections(md), limit) or ""


def doc_text(drive, repo_id, index_id, isin, kind, row, cache) -> str:
    """Condensed text for one document; '' when nothing trustworthy exists."""
    from _extractor_base import load_doc_report, is_prompt_echo, degenerate_reason
    from run_pf_docs_digest import _split_sections, _lift_summary, _find_region, _company_page
    from pf_company_mails import real_report
    try:
        exact = load_doc_report(drive, index_id, row["doc_id"]) if row["doc_id"] else ""
    except Exception:
        exact = ""
    if exact.strip():
        exact, hollow = real_report(exact)
        if not (is_prompt_echo(exact) or degenerate_reason(exact) or hollow):
            t = condense(exact, _split_sections, _lift_summary)
            if t:
                return t
    try:
        reg = _find_region(_company_page(drive, repo_id, isin, cache), row.get("period", ""),
                           kind, row["doc_id"], allow_newest=False)
    except Exception:
        reg = None
    return _lift_summary(reg, LIMIT) if reg else ""


def render(docs: dict, texts: dict, today: date, h, badge, md_to_html,
           compact: bool = False) -> str:
    """The block. `h` / `badge` / `md_to_html` are the caller's helpers."""
    parts = []
    for kind, label in KINDS:
        row, txt = docs.get(kind), texts.get(kind, "")
        if not row:
            continue
        head = (f"<div style='margin:10px 0 2px;font-size:13px'>{badge(row['date'], today)}"
                f"<b>{label}</b> <span style='color:#888;font-size:12px'>&middot; filed "
                f"{row['date']}</span></div>")
        if txt:
            body = md_to_html(txt)
        elif compact:
            # the page was near Gmail's clip: say so, never "nothing stored"
            body = ("<div style='color:#8a97a0;font-size:12px'>Summary left out to keep "
                    "this mail inside Gmail's size limit — the full report comes in the "
                    "per-company mail.</div>")
        else:
            body = ("<div style='color:#8a97a0;font-size:12px'>Processed, but no readable "
                    "summary was stored for it.</div>")
        parts.append(head + f"<div style='border-left:3px solid #ccc;padding-left:10px;"
                            f"font-size:12.5px'>{body}</div>")
    if not parts:
        return ""
    return (h("What the documents say",
              "The company's own documents for this quarter, condensed. Full reports "
              "come in the per-company concall and annual-report mails.")
            + "".join(parts))


def _self_test() -> int:
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    q = pd.DataFrame([
        {"isin": "I1", "doc_type": "concall", "status": "done", "doc_id": "c_old",
         "announcement_date": "2026-08-05", "processed_at": "2026-08-06", "title": "Q1 call"},
        {"isin": "I1", "doc_type": "concall", "status": "done", "doc_id": "c_new",
         "announcement_date": "2026-10-13", "processed_at": "2026-10-14", "title": "Q2 call"},
        {"isin": "I1", "doc_type": "results", "status": "pending", "doc_id": "r_pend",
         "announcement_date": "2026-10-14", "processed_at": "", "title": "Q2 results"},
        {"isin": "I1", "doc_type": "annual_report", "status": "done", "doc_id": "ar26",
         "announcement_date": "2026-03-31", "processed_at": "2026-09-01",
         "discovered_at": "2026-08-30", "title": "AR 2026", "period": ""},
        {"isin": "I2", "doc_type": "concall", "status": "done", "doc_id": "other",
         "announcement_date": "2026-10-13", "processed_at": "2026-10-14", "title": "x"}])

    def fy_of(d, p=""):
        return 2026 if str(d)[:10] in ("2026-03-31",) else None
    docs = pick_docs(q, "I1", "Q2FY27", date(2026, 10, 14), ("2026-10-01", "2026-10-14"),
                     2026, fy_of)
    check("concall from THIS quarter's window only", docs.get("concall", {}).get("doc_id") == "c_new")
    check("pending results not shown", "results" not in docs)
    check("AR of the current FY, dated by when it reached us",
          docs.get("annual_report", {}).get("date") == "2026-08-30")
    check("another company's docs never leak in", all(v["doc_id"] != "other" for v in docs.values()))

    split = lambda md: [("## A", md)]
    lift = lambda region, limit: region[0][1][:limit]
    check("condense caps length", len(condense("x" * 5000, split, lift)) == LIMIT)
    check("condense empty -> ''", condense("  ", split, lift) == "")

    html = render(docs, {"concall": "Revenue up **20%**"}, date(2026, 10, 14),
                  lambda t, s="": f"<h3>{t}</h3>", lambda d, t: f"[{d}]",
                  lambda m: f"<p>{m}</p>")
    check("render shows each doc with its badge and date",
          "What the documents say" in html and "[2026-10-13]" in html and "filed 2026-08-30" in html)
    check("doc without text says so", "no readable summary" in html)
    check("nothing to show -> ''", render({}, {}, date(2026, 10, 14), None, None, None) == "")
    hc = render(docs, {}, date(2026, 10, 14), lambda t, s="": f"<h3>{t}</h3>",
                lambda d, t: "", lambda m: m, compact=True)
    check("compact mode says the summary was left out, not missing",
          "size limit" in hc and "no readable" not in hc)
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
