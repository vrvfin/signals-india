"""
pf_season_status.py — the deterministic "where are we, out of 51" mail.

THE QUESTION THIS ANSWERS. Individual company mails tell you what arrived. They cannot
tell you what DIDN'T — a holding with no deck looks exactly like one that is fully
covered, because nothing is sent either way. This mail is the completeness signal:
every PF holding, every document type, one of four states, each with a reason.

    delivered        a mail went out for the newest document
    due              processed, mail not sent yet
    awaiting         the exchange calendar says it reports, or a document is mid-flight
    no information   nothing filed and nothing scheduled — normal for decks and ratings,
                     which not every company issues, so it is NOT counted as a failure

THE WINDOW IS EXCHANGE-DETERMINED, not a fixed calendar. `results_calendar.parquet` is
accumulated from the NSE and BSE board-meeting APIs, so "who reports this season" comes
from the exchanges themselves. Note the calendar's history begins 2026-08-06 and both its
APIs are forward-only — for anything earlier the date cascade in quarterly_table fills in,
and a holding with no date anywhere is reported as such rather than assumed absent.

It runs daily through the season and converges: at the start almost everything is
`awaiting`, and by the end every holding should be `delivered` on results. If it is not,
this mail names the company and says why.

Usage:
    python scripts/pf_season_status.py --dry-run
    python scripts/pf_season_status.py             # send
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import date, datetime

import pandas as pd
from dotenv import load_dotenv

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import quarterly_table as QT
import pf_coverage as COV
from pf_company_mails import LEDGER_NAME, LEDGER_COLS

MAIL_KEY = "pf_season_status"

GREEN, AMBER, RED, GREY = "#1a7a3a", "#b8860b", "#c0392b", "#8a97a0"
_STATE_COLOUR = {COV.DELIVERED: GREEN, COV.DUE: AMBER,
                 COV.AWAITING: RED, COV.NO_INFO: GREY}
_WRAP = "font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222"
_TBL = ("border-collapse:collapse;width:100%;font:12px Arial,Helvetica,sans-serif;"
        "color:#222;margin:6px 0 14px")


def _esc(s, n=200):
    from mailer import esc
    return esc(s, n)


def render(rows: list[dict], roll: dict, season: str, today: str) -> str:
    n_co = roll["_companies"]
    done, tot = roll["_results_done"], roll["_results_total"]
    pct = (100 * done // tot) if tot else 0

    head = (f"<h2 style='margin:0 0 2px'>&#128202; {_esc(QT.qtr_label(season) if ' ' not in season else season, 12)} "
            f"season status &mdash; {done} of {tot} holdings complete on results</h2>"
            f"<div style='color:#888;font-size:12px;margin:0 0 10px'>"
            f"{n_co} PF holdings &middot; as of {_esc(today, 12)} &middot; "
            f"the window is set by the exchange calendar, not a fixed date</div>")

    # progress bar — the one-glance answer
    head += (f"<div style='background:#eceff1;border-radius:3px;height:14px;width:100%;"
             f"margin:0 0 14px'><div style='background:{GREEN};height:14px;"
             f"border-radius:3px;width:{pct}%'></div></div>")

    # rollup per doc type
    hdr = "".join(f"<th style='text-align:left;border-bottom:1px solid #ccc;color:{GREY}'>"
                  f"{h}</th>" for h in ("", "delivered", "due", "awaiting", "no info"))
    body = ""
    for dt in ("results", "presentation", "rating"):
        c = roll[dt]
        body += (f"<tr><td><b>{dt}</b></td>"
                 f"<td style='color:{GREEN};font-weight:700'>{c[COV.DELIVERED]}</td>"
                 f"<td style='color:{AMBER};font-weight:700'>{c[COV.DUE]}</td>"
                 f"<td style='color:{RED};font-weight:700'>{c[COV.AWAITING]}</td>"
                 f"<td style='color:{GREY}'>{c[COV.NO_INFO]}</td></tr>")
    out = [head, f"<table cellpadding='5' cellspacing='0' style='{_TBL}'>"
                 f"<tr>{hdr}</tr>{body}</table>"]

    # What the two states MEAN, stated up front rather than left to inference.
    out.append(
        f"<div style='background:#f6f8f9;border:1px solid #e0e6ea;padding:9px 12px;"
        f"margin:0 0 12px;font-size:12px'>"
        f"<b style='color:{AMBER}'>due</b> &mdash; the document is in hand and processed; "
        f"only the send is outstanding. <b>Clears itself on the next run, no action "
        f"needed.</b><br>"
        f"<b style='color:{RED}'>awaiting</b> &mdash; the document itself is missing, "
        f"undownloaded or unprocessed. <b>This is the real gap</b>; the table names the "
        f"exact stage it stopped at.<br>"
        f"<b style='color:{GREY}'>no information</b> &mdash; the company published nothing "
        f"of that kind this quarter. Normal for decks and ratings; not a failure."
        f"</div>")

    # THE PIPELINE TABLE. A tick per stage, so the gap is visible rather than asserted:
    # the first empty column IS the reason.
    pend = [r for r in rows if r["state"] in (COV.DUE, COV.AWAITING)]
    if pend:
        order = {COV.AWAITING: 0, COV.DUE: 1}
        pend.sort(key=lambda r: (order.get(r["state"], 9), r["doc_type"], r["symbol"]))
        stage_hdr = "".join(
            f"<th style='border-bottom:1px solid #ccc;color:{GREY};font-size:10.5px;"
            f"text-align:center'>{s}</th>" for s in COV.STAGES)
        rws = ""
        for r in pend:
            st = r.get("stages", {})
            cells = ""
            for s in COV.STAGES:
                if st.get(s):
                    cells += f"<td style='text-align:center;color:{GREEN}'>&#10003;</td>"
                elif s == r.get("stopped_at"):
                    cells += (f"<td style='text-align:center;color:{RED};"
                              f"font-weight:700'>&#10007;</td>")
                else:
                    cells += "<td style='text-align:center;color:#dfe4e7'>&middot;</td>"
            rws += (f"<tr><td><b>{_esc(r['symbol'], 16)}</b></td>"
                    f"<td>{_esc(r['doc_type'], 13)}</td>"
                    f"<td style='color:{_STATE_COLOUR[r['state']]};font-weight:700'>"
                    f"{_esc(r['state'], 14)}</td>{cells}"
                    f"<td style='color:#666;font-size:11.5px'>"
                    f"{_esc(r['reason'], 95)}</td></tr>")
        out.append(
            f"<h3 style='margin:14px 0 4px;font-size:13px;color:#34495e;"
            f"border-bottom:2px solid #34495e;padding-bottom:2px'>"
            f"Not delivered yet ({len(pend)}) &mdash; where each one stopped</h3>"
            f"<div style='overflow-x:auto'>"
            f"<table cellpadding='5' cellspacing='0' style='{_TBL}'>"
            f"<tr><th style='text-align:left;border-bottom:1px solid #ccc;color:{GREY}'>Company</th>"
            f"<th style='text-align:left;border-bottom:1px solid #ccc;color:{GREY}'>Document</th>"
            f"<th style='text-align:left;border-bottom:1px solid #ccc;color:{GREY}'>State</th>"
            f"{stage_hdr}"
            f"<th style='text-align:left;border-bottom:1px solid #ccc;color:{GREY}'>Why</th></tr>"
            f"{rws}</table></div>"
            f"<div style='color:{GREY};font-size:11px;margin:-8px 0 12px'>"
            f"&#10003; cleared &middot; <span style='color:{RED}'>&#10007;</span> stopped "
            f"here &mdash; this is the gap &middot; &middot; not reached. Stages run left "
            f"to right: the exchange expects it &rarr; a source listed it &rarr; the file "
            f"downloaded &rarr; an extractor processed it &rarr; it produced rows the mail "
            f"can read &rarr; the mail went out.</div>")
    else:
        out.append(f"<div style='color:{GREEN};font-weight:700;margin:10px 0'>"
                   f"Every holding is delivered or has nothing outstanding.</div>")
    return f"<div style='{_WRAP}'>{''.join(out)}</div>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(_self_test())

    from _extractor_base import (get_drive, get_or_create_subfolder, load_queue,
                                 load_parquet, find_file, download_bytes, log)
    from daily_brief import load_pf

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")

    season = QT.season_quarter()
    today = date.today().isoformat()
    pf = load_pf(drive, root, idx)
    queue = load_queue(drive, idx)
    fid = find_file(drive, idx, "results_calendar.parquet")
    calendar = pd.read_parquet(io.BytesIO(download_bytes(drive, fid))) if fid \
        else pd.DataFrame()
    # TWO LEDGERS, ONE PICTURE. Presentation and rating mails are stamped in
    # pf_company_mails.parquet; the results teardown keeps its own
    # quarter_teardown_mailed.parquet. Reading only the first reported results as
    # 0 of 51 delivered while 47 mails had actually gone out — the status mail has to
    # merge both or it misreports the very thing it exists to report.
    ledger = load_parquet(drive, idx, LEDGER_NAME, LEDGER_COLS)
    td = load_parquet(drive, idx, "quarter_teardown_mailed.parquet",
                      ["season_quarter", "isin", "symbol", "quarter_label",
                       "reported_on", "date_source", "data_source", "mailed_at",
                       "mail_mode"])
    if td is not None and not td.empty:
        conv = pd.DataFrame({
            "season": td["season_quarter"].astype(str),
            "isin": td["isin"].astype(str),
            "symbol": td["symbol"].astype(str),
            "doc_type": "results",
            "period": td["quarter_label"].astype(str),
            "doc_id": "",                    # teardown ledger predates per-doc identity
            "mailed_at": td["mailed_at"].astype(str),
            "subject": "",
        })
        ledger = pd.concat([ledger, conv], ignore_index=True) \
            if ledger is not None and not ledger.empty else conv
        log(f"ledgers merged: {len(conv)} teardown row(s) + company-mail rows")

    # The tables the mails actually read — needed so the "structured" stage means
    # "produced rows a mail can use", not merely "an extractor ran".
    tables = {}
    for name in ("ppt_highlights", "ratings"):
        f = find_file(drive, idx, f"{name}.parquet")
        tables[name] = (pd.read_parquet(io.BytesIO(download_bytes(drive, f)))
                        if f else pd.DataFrame())

    rows = COV.season_status(pf, queue, calendar, ledger, season, on=date.today(),
                             tables=tables)
    roll = COV.season_rollup(rows)
    log(f"season={season} holdings={roll['_companies']} "
        f"results delivered={roll['_results_done']}/{roll['_results_total']}")
    for dt in ("results", "presentation", "rating"):
        log(f"  {dt:<13} {roll[dt]}")

    html = render(rows, roll, season, today)
    subject = (f"📊 {QT.qtr_label(season) if ' ' not in season else season} season status "
               f"— {roll['_results_done']}/{roll['_results_total']} complete")

    if args.dry_run:
        p = os.path.join(args.out_dir, "pf_season_status_preview.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(html)
        log(f"DRY RUN — preview {p} ({len(html.encode()):,} B); no mail.")
        return

    from mailer import send_email, load_mail_settings
    if not load_mail_settings(drive, idx).get(MAIL_KEY, True):
        log(f"mail toggle '{MAIL_KEY}' OFF — not sending.")
        return
    if not (os.getenv("GMAIL_USER") and os.getenv("GMAIL_APP_PASSWORD")
            and os.getenv("NOTIFY_EMAIL")):
        log("mail NOT sent — GMAIL_* not set in this environment.")
        return
    log(f"sent={send_email(subject, html)}")


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    rows = [
        {"isin": "I1", "symbol": "AAA", "name": "A", "doc_type": "results",
         "state": COV.DELIVERED, "reason": "mailed · 2026-08-01", "doc_date": ""},
        {"isin": "I2", "symbol": "BBB", "name": "B", "doc_type": "results",
         "state": COV.AWAITING, "reason": "calendar says reported 2026-08-12, no numbers yet",
         "doc_date": ""},
        {"isin": "I3", "symbol": "CCC", "name": "C", "doc_type": "results",
         "state": COV.DUE, "reason": "processed, mail not sent yet", "doc_date": ""},
        {"isin": "I1", "symbol": "AAA", "name": "A", "doc_type": "rating",
         "state": COV.NO_INFO, "reason": "no rating issued this season", "doc_date": ""},
    ]
    roll = COV.season_rollup(rows)
    html = render(rows, roll, "Q1FY27", "2026-08-16")
    check("headline counts delivered results", "1 of 3" in html)
    check("a pending company is NAMED", "BBB" in html)
    check("its reason is given", "no numbers yet" in html)
    check("a due company is named too", "CCC" in html)
    check("no-information is not treated as missing", "AAA" not in html.split(
        "Not delivered yet")[-1] or "rating" not in html.split("Not delivered yet")[-1])
    check("due vs awaiting is explained up front, not left to inference",
          "Clears itself on the next run" in html and "This is the real gap" in html)
    check("no information is explained as normal, not a failure",
          "not a failure" in html)
    # The pipeline table is what makes a gap checkable rather than asserted.
    rows2 = list(rows)
    rows2[1] = dict(rows2[1], stages={"expected": True, "discovered": True,
                                      "fetched": False, "extracted": False,
                                      "structured": False, "mailed": False},
                    stopped_at="fetched")
    h3 = render(rows2, COV.season_rollup(rows2), "Q1FY27", "2026-08-16")
    check("stage columns are rendered", "discovered" in h3 and "structured" in h3)
    check("the stopped stage is marked", "&#10007;" in h3)
    check("cleared stages are ticked", "&#10003;" in h3)

    all_done = [{"isin": "I1", "symbol": "AAA", "name": "A", "doc_type": "results",
                 "state": COV.DELIVERED, "reason": "mailed", "doc_date": ""}]
    h2 = render(all_done, COV.season_rollup(all_done), "Q1FY27", "2026-08-16")
    check("a complete season says so", "nothing outstanding" in h2)
    check("complete season shows no pending table", "Not delivered yet" not in h2)

    print(f"\npf_season_status self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    main()
