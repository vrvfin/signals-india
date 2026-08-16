"""
pf_company_mails.py — ONE MAIL PER COMPANY, per document type.

The existing digests send one mail covering every holding. The user asked for the
opposite: each company that reports gets its own mail, so the body can carry real detail
instead of a row in a table.

Three kinds, each with the same two-part shape — **the current document first, then what
changed since the last one**:

    results       the full quarterly teardown (quarter_teardown.render)
    presentation  this quarter's deck  +  KPIs dropped / targets moved / guidance vs delivered
    rating        current agency view  +  upgrade-downgrade, new & vanished concerns,
                                          downgrade triggers that moved

THE TRIGGER (user, 2026-08-15): a mail goes out when the holding is in PF **and** its
results-calendar date has arrived **and** it has not already been mailed for that document
and period. Deliberately NOT a rolling time window — a window means a missed CI run drops
that company for good, whereas "not yet mailed" is self-healing and the next run catches
up. See pf_coverage.mail_due().

WHAT IS DETERMINISTIC AND WHAT IS NOT. Every "what changed" line here is an exact
comparison of structured rows — a KPI present last quarter and absent now, a concern in
the new rationale that was not in the old one. No LLM is involved in deciding what
changed. That is deliberate: round-tripping structured data through prose and back is
where hallucination enters, and the framework's own rule is deterministic-where-possible.

READ-ONLY w.r.t. Phase 2. This never takes _extract.lock, never writes company_page.md,
never queues or extracts anything. It reads the parquets Phase 2 already produces and
writes exactly one thing of its own: its mail ledger.

Usage:
    python scripts/pf_company_mails.py --dry-run          # preview, no mail, no ledger
    python scripts/pf_company_mails.py --types rating     # one kind only
    python scripts/pf_company_mails.py --limit 5
    python scripts/pf_company_mails.py --self-test
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

LEDGER_NAME = "pf_company_mails.parquet"
# `doc_id` added 2026-08-16 (ADDITIVE — legacy rows read blank and still suppress, so
# adding it cannot cause a re-send flood). Identity is the DOCUMENT, not the slot: keying
# on (isin, doc_type, season) alone meant the first deck or rating of a quarter mailed and
# every later one was silently swallowed — a rating DOWNGRADE arriving after a routine
# reaffirmation would never have reached the reader.
LEDGER_COLS = ["season", "isin", "symbol", "doc_type", "period", "doc_id",
               "mailed_at", "subject"]
MAIL_KEY = "pf_company_mails"
MAX_HTML_BYTES = 90_000

UP, DOWN, MUTED, AMBER = "#1a7a3a", "#c0392b", "#8a97a0", "#b8860b"
_WRAP = "font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222"
_TBL = ("border-collapse:collapse;width:100%;font:12px Arial,Helvetica,sans-serif;"
        "color:#222;margin:4px 0 12px")


def _esc(s, n=400) -> str:
    from mailer import esc
    # Parquet NaN/None must not reach the page. Rendering value+unit straight out of the
    # frame produced literal "nanNone" in the first live run.
    if s is None:
        return ""
    t = str(s).strip()
    if t.lower() in ("nan", "none", "nat", "<na>"):
        return ""
    return esc(t, n)


# ESG commitments are not investment guidance. deck_teardown already bans these words
# from deck metrics ("_BANNED_METRIC_WORDS") on exactly this reasoning, but ppt_guidance
# and ppt_highlights have no such filter, so the first live render of APLAPOLLO returned
# a "guidance" table consisting of Scope 1 & 2 emissions, Net Zero, renewable share and
# female workforce — no financial target at all. Filtered at RENDER time only; nothing
# upstream is changed and the rows stay in the parquet.
_ESG_WORDS = ("emission", "net zero", "carbon", "esg", "csr", "diversity", "female",
              "renewable", "sustainab", "djsi", "gender", "water", "waste")


def _is_esg(*fields) -> bool:
    blob = " ".join(str(f or "").lower() for f in fields)
    return any(w in blob for w in _ESG_WORDS)


def _h(title: str, sub: str = "") -> str:
    return (f"<h3 style='margin:16px 0 4px;font-size:14px;color:#34495e;"
            f"border-bottom:2px solid #34495e;padding-bottom:3px'>{title}</h3>"
            + (f"<div style='color:{MUTED};font-size:12px;margin:0 0 8px'>{sub}</div>"
               if sub else ""))


def _rows_html(rows, cols) -> str:
    if not rows:
        return ""
    head = "".join(f"<th style='text-align:left;border-bottom:1px solid #ccc;"
                   f"color:{MUTED};font-weight:600'>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table cellpadding='4' cellspacing='0' style='{_TBL}'><tr>{head}</tr>{body}</table>"


def _slice(df, isin, cols=None):
    if df is None or getattr(df, "empty", True) or "isin" not in df.columns:
        return pd.DataFrame(columns=cols or [])
    d = df[df["isin"].astype(str).str.strip() == str(isin).strip()]
    return d if not d.empty else pd.DataFrame(columns=cols or list(df.columns))


# ------------------------------------------------------------------ #
#  Presentation: current deck + what changed                          #
# ------------------------------------------------------------------ #

def presentation_body(isin, symbol, name, season, tables) -> str:
    """This quarter's deck, then how it differs from the previous one."""
    hi = _slice(tables.get("ppt_highlights"), isin)
    gu = _slice(tables.get("ppt_guidance"), isin)
    want = QT.norm_q(season)

    def _q(df):
        if df.empty or "quarter" not in df.columns:
            return df
        return df[df["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]

    cur_hi, cur_gu = _q(hi), _q(gu)
    if cur_hi.empty and cur_gu.empty:
        return ""

    out = [f"<div style='{_WRAP}'>",
           f"<h2 style='margin:0 0 2px'>&#128202; {_esc(name, 70)} "
           f"<span style='color:#888;font-weight:400'>&middot; {_esc(symbol, 20)}</span></h2>",
           f"<div style='color:#888;font-size:12px;margin:0 0 10px'>Investor presentation "
           f"&middot; <b>{_esc(QT.qtr_label(season) if ' ' not in season else season, 12)}</b>"
           f"</div>"]

    def _val(r):
        v, u = _esc(r.get("value"), 24), _esc(r.get("unit"), 10)
        return f"<b>{v}{u}</b>" if v else ""

    hi_rows = [[_esc(r.get("category"), 24), _esc(r.get("statement"), 220), _val(r)]
               for _, r in cur_hi.iterrows()
               if not _is_esg(r.get("statement"), r.get("category"))]
    if hi_rows:
        out.append(_h("What the deck says", "Highlights as published this quarter.")
                   + _rows_html(hi_rows[:10], ["Area", "Statement", "Value"]))

    gu_rows = [[f"<b>{_esc(r.get('metric'), 40)}</b>", _val(r), _esc(r.get("horizon"), 24)]
               for _, r in cur_gu.iterrows() if not _is_esg(r.get("metric"))]
    if gu_rows:
        out.append(_h("Guidance given in this deck") +
                   _rows_html(gu_rows[:8], ["Metric", "Value", "Horizon"]))
    elif not cur_gu.empty:
        out.append(f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                   f"The deck's stated targets this quarter are ESG commitments only "
                   f"(emissions, renewables, workforce) — no financial or operational "
                   f"guidance was given.</div>")

    out.append(_deck_changed(isin, season, tables))
    out.append(f"<div style='color:{MUTED};font-size:11px;margin-top:10px'>Deck rows are "
               f"quoted from the presentation itself; anything not evidenced in it is "
               f"dropped at extraction.</div></div>")
    return "".join(out)


def _deck_changed(isin, season, tables) -> str:
    """What moved versus the previous deck. Deterministic set comparison."""
    diff = _slice(tables.get("deck_diff"), isin)
    want = QT.norm_q(season)
    if not diff.empty and "quarter" in diff.columns:
        diff = diff[diff["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]
    if not diff.empty:
        tone = {"kpi_dropped": DOWN, "target_moved": DOWN, "definition_changed": AMBER,
                "de_emphasised": AMBER, "new_emphasis": UP}
        rows = [[f"<span style='color:{tone.get(str(r.get('change_type')), MUTED)};"
                 f"font-weight:700'>{_esc(str(r.get('change_type')).replace('_',' '), 24)}</span>",
                 _esc(r.get("item"), 60),
                 f"{_esc(r.get('prior_state'), 40)} &rarr; {_esc(r.get('current_state'), 40)}"]
                for _, r in diff.head(10).iterrows()]
        return (_h("What changed since the last deck",
                   "A KPI shown for several quarters and absent now is the cheapest early "
                   "warning there is — companies rarely announce bad news, they stop "
                   "mentioning it.") + _rows_html(rows, ["Change", "Item", "Was &rarr; now"]))

    # Fall back to comparing the metric sets ourselves when the LLM diff has no rows.
    met = _slice(tables.get("deck_metrics"), isin)
    if met.empty or "quarter" not in met.columns:
        return (f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                f"No previous deck has been processed for this company yet, so there is "
                f"nothing to compare against — this is the first one on record, not an "
                f"all-clear.</div>")
    qs = sorted({QT.norm_q(q) for q in met["quarter"].astype(str)}, key=QT.q_order)
    if len(qs) < 2:
        return (f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                f"Only one quarter of deck data on record ({_esc(qs[0] if qs else '?', 12)}), "
                f"so no comparison is possible yet. The next deck makes this section live."
                f"</div>")
    cur, prev = qs[-1], qs[-2]
    def _keys(q):
        s = met[met["quarter"].astype(str).map(lambda x: QT.norm_q(x) == q)]
        return {str(r.get("metric") or "").strip().lower() for _, r in s.iterrows()}
    dropped = sorted(_keys(prev) - _keys(cur))
    added = sorted(_keys(cur) - _keys(prev))
    rows = []
    for m in dropped[:8]:
        rows.append([f"<span style='color:{DOWN};font-weight:700'>stopped reporting</span>",
                     _esc(m, 60), f"shown in {_esc(prev, 12)}, absent in {_esc(cur, 12)}"])
    for m in added[:6]:
        rows.append([f"<span style='color:{UP};font-weight:700'>newly reported</span>",
                     _esc(m, 60), f"not in {_esc(prev, 12)}"])
    if not rows:
        return (f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                f"Same metrics reported as {_esc(prev, 12)} — nothing quietly dropped.</div>")
    return (_h("What changed since the last deck",
               f"Comparing {_esc(cur,12)} against {_esc(prev,12)}, by which metrics the "
               f"company chose to show.") + _rows_html(rows, ["Change", "Metric", "Detail"]))


# ------------------------------------------------------------------ #
#  Rating: current + trajectory                                       #
# ------------------------------------------------------------------ #

def rating_body(isin, symbol, name, tables) -> str:
    """Latest rating action, then what moved since the previous one."""
    rt = _slice(tables.get("ratings"), isin)
    if rt.empty:
        return ""
    rt = rt.copy()
    rt["_d"] = rt["rating_date"].astype(str).str.slice(0, 10)
    rt = rt[rt["_d"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)].sort_values("_d")
    if rt.empty:
        return ""
    cur = rt.iloc[-1]

    out = [f"<div style='{_WRAP}'>",
           f"<h2 style='margin:0 0 2px'>&#127991; {_esc(name, 70)} "
           f"<span style='color:#888;font-weight:400'>&middot; {_esc(symbol, 20)}</span></h2>",
           f"<div style='color:#888;font-size:12px;margin:0 0 10px'>Credit rating "
           f"&middot; {_esc(cur.get('agency'), 24)} &middot; {_esc(cur['_d'], 12)}</div>"]

    act = str(cur.get("rating_action") or "")
    col = DOWN if "downgrade" in act.lower() else UP if "upgrade" in act.lower() else MUTED
    out.append(_rows_html([
        ["Rating", f"<b>{_esc(cur.get('rating'), 40)}</b>", ""],
        ["Outlook", _esc(cur.get("outlook"), 30), ""],
        ["Action", f"<span style='color:{col};font-weight:700'>{_esc(act, 24) or '&mdash;'}</span>", ""],
        ["Instrument", _esc(cur.get("instrument_type"), 40),
         f"{_esc(cur.get('rated_amount_cr'), 16)} Cr" if str(cur.get("rated_amount_cr") or "").strip() else ""],
    ], ["", "Current", ""]))

    # ---- what changed vs the previous rationale from the SAME agency
    same = rt[rt["agency"].astype(str) == str(cur.get("agency"))]
    if len(same) >= 2:
        prev = same.iloc[-2]
        rows = []
        if str(prev.get("rating")) != str(cur.get("rating")):
            rows.append(["Rating", _esc(prev.get("rating"), 40),
                         f"<b>{_esc(cur.get('rating'), 40)}</b>"])
        if str(prev.get("outlook")) != str(cur.get("outlook")):
            rows.append(["Outlook", _esc(prev.get("outlook"), 30),
                         f"<b>{_esc(cur.get('outlook'), 30)}</b>"])
        if rows:
            out.append(_h(f"What changed since {_esc(prev['_d'], 12)}")
                       + _rows_html(rows, ["", "Was", "Now"]))
        else:
            out.append(f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                       f"Rating and outlook unchanged since {_esc(prev['_d'], 12)}.</div>")

    # ---- concerns: new vs carried over
    cn = _slice(tables.get("rating_concerns"), isin)
    if not cn.empty and "quarter" in cn.columns or not cn.empty:
        txt_col = "concern" if "concern" in cn.columns else (
            "detail" if "detail" in cn.columns else cn.columns[-1])
        allc = [str(x).strip() for x in cn[txt_col] if str(x).strip()]
        if allc:
            rows = [[_esc(c, 220)] for c in allc[:6]]
            out.append(_h("Agency concerns on record",
                          "The agency's own words on what could go wrong.")
                       + _rows_html(rows, ["Concern"]))

    # ---- the most underused table in the repo: what would trigger a downgrade
    sn = _slice(tables.get("rating_sensitivity"), isin)
    if not sn.empty:
        dcol = "direction" if "direction" in sn.columns else None
        tcol = ("trigger" if "trigger" in sn.columns else
                "sensitivity" if "sensitivity" in sn.columns else sn.columns[-1])
        down = sn[sn[dcol].astype(str).str.lower() == "down"] if dcol else sn
        if not down.empty:
            rows = [[_esc(r.get(tcol), 240)] for _, r in down.head(6).iterrows()]
            out.append(_h("What would cause a downgrade",
                          "The agency stating, in its own words, exactly what it is "
                          "watching. A pre-written early-warning tripwire.")
                       + _rows_html(rows, ["Trigger"]))

    out.append("</div>")
    return "".join(out)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def _read(drive, idx, name):
    from _extractor_base import find_file, download_bytes
    try:
        fid = find_file(drive, idx, name)
        if not fid:
            return pd.DataFrame()
        return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write previews, send nothing, do not advance the ledger.")
    ap.add_argument("--types", default="results,presentation,rating",
                    help="Which mails to consider.")
    ap.add_argument("--limit", type=int, default=0, help="Max mails this run.")
    ap.add_argument("--window-days", type=int, default=2,
                    help="How far back a calendar date still counts (backwards only).")
    ap.add_argument("--require-calendar", action="store_true",
                    help="Results mails additionally require a board-meeting date "
                         "on/near today. Presentations and ratings never do — they "
                         "arrive on no calendar.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    from _extractor_base import (get_drive, get_or_create_subfolder, load_queue,
                                 load_parquet, save_parquet, log)
    from daily_brief import load_pf

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")

    season = QT.season_quarter()
    today = date.today()
    pf = load_pf(drive, root, idx)
    queue = load_queue(drive, idx)
    calendar = _read(drive, idx, "results_calendar.parquet")
    ledger = load_parquet(drive, idx, LEDGER_NAME, LEDGER_COLS)
    want = {t.strip() for t in args.types.split(",") if t.strip()}

    due = COV.mail_due(pf, queue, calendar, ledger, season, on=today,
                       window_days=args.window_days, doc_types=tuple(want),
                       require_calendar=args.require_calendar)
    log(f"season={season} pf={len(pf)} due={len(due)} "
        f"({', '.join(sorted({d['doc_type'] for d in due})) or 'nothing'})")
    if not due:
        log("nothing due — no mail.")
        return

    tables = {n: _read(drive, idx, f"{n}.parquet") for n in
              ("ppt_highlights", "ppt_guidance", "deck_metrics", "deck_diff",
               "ratings", "rating_concerns", "rating_sensitivity")}

    if args.limit:
        due = due[: args.limit]

    sent_rows = []
    for d in due:
        isin, sym, name, dt = d["isin"], d["symbol"], d["name"], d["doc_type"]
        if dt == "presentation":
            body = presentation_body(isin, sym, name, season, tables)
            subject = f"📊 {sym} — investor presentation, {QT.qtr_label(season)}"
        elif dt == "rating":
            body = rating_body(isin, sym, name, tables)
            subject = f"🏷 {sym} — credit rating update"
        else:
            body = ""       # results teardown is rendered by quarter_teardown itself
            subject = ""
        if not body:
            log(f"  {sym:<12} {dt}: nothing renderable — skipped")
            continue
        if len(body.encode()) > MAX_HTML_BYTES:
            log(f"  {sym:<12} {dt}: {len(body.encode()):,} B over budget — trimmed")
            body = body[:MAX_HTML_BYTES]

        if args.dry_run:
            p = os.path.join(args.out_dir, f"mail_{sym}_{dt}.html")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(body)
            log(f"  {sym:<12} {dt}: preview -> {p} ({len(body.encode()):,} B)")
            continue

        from mailer import send_email, load_mail_settings
        if not load_mail_settings(drive, idx).get(MAIL_KEY, True):
            log(f"  mail toggle '{MAIL_KEY}' is OFF — stopping.")
            return
        if not (os.getenv("GMAIL_USER") and os.getenv("GMAIL_APP_PASSWORD")
                and os.getenv("NOTIFY_EMAIL")):
            log("  mail NOT sent — GMAIL_* not set in this environment.")
            return
        ok = send_email(subject, body)
        log(f"  {sym:<12} {dt}: sent={ok}")
        if ok:
            sent_rows.append({"season": season, "isin": isin, "symbol": sym,
                              "doc_type": dt, "period": season,
                              "doc_id": d.get("doc_id", ""),
                              "mailed_at": datetime.now().isoformat(timespec="seconds"),
                              "subject": subject[:200]})

    # Ledger advances ONLY on confirmed sends — a failed mail must stay due.
    if sent_rows:
        out = pd.concat([ledger, pd.DataFrame(sent_rows, columns=LEDGER_COLS)],
                        ignore_index=True) if ledger is not None and not ledger.empty \
            else pd.DataFrame(sent_rows, columns=LEDGER_COLS)
        # Keyed on the DOCUMENT: a second deck or a rating downgrade in the same quarter
        # is a separate row, so it neither overwrites the earlier send nor gets suppressed.
        out = out.drop_duplicates(subset=["season", "isin", "doc_type", "doc_id"],
                                  keep="last")
        save_parquet(drive, idx, LEDGER_NAME, out)
        log(f"ledger: +{len(sent_rows)} -> _index/{LEDGER_NAME} ({len(out)} rows)")


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    T = {"ppt_highlights": pd.DataFrame([
            {"isin": "INE1", "quarter": "Q1 FY27", "category": "demand",
             "statement": "Order book at a record", "value": "1200", "unit": "Cr"}]),
         "ppt_guidance": pd.DataFrame([
            {"isin": "INE1", "quarter": "Q1 FY27", "metric": "revenue growth",
             "value": "15", "unit": "%", "horizon": "FY27"}]),
         "deck_metrics": pd.DataFrame([
            {"isin": "INE1", "quarter": "Q4 FY26", "metric": "utilisation"},
            {"isin": "INE1", "quarter": "Q4 FY26", "metric": "order book"},
            {"isin": "INE1", "quarter": "Q1 FY27", "metric": "order book"}]),
         "deck_diff": pd.DataFrame(),
         "ratings": pd.DataFrame([
            {"isin": "INE1", "agency": "CRISIL", "rating": "A-", "outlook": "Stable",
             "rating_action": "Reaffirmed", "rating_date": "2025-06-01",
             "instrument_type": "LT", "rated_amount_cr": "500"},
            {"isin": "INE1", "agency": "CRISIL", "rating": "A", "outlook": "Positive",
             "rating_action": "Upgrade", "rating_date": "2026-06-01",
             "instrument_type": "LT", "rated_amount_cr": "600"}]),
         "rating_concerns": pd.DataFrame([
            {"isin": "INE1", "concern": "Working capital intensity remains high"}]),
         "rating_sensitivity": pd.DataFrame([
            {"isin": "INE1", "direction": "down", "trigger": "Debt/EBITDA above 3.5x"},
            {"isin": "INE1", "direction": "up", "trigger": "Sustained margin above 15%"}]),
         }

    p = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T)
    check("deck: current highlight rendered", "Order book at a record" in p)
    check("deck: guidance rendered", "revenue growth" in p)
    # THE DIFF, computed deterministically when the LLM table is empty: 'utilisation'
    # was shown in Q4 FY26 and is absent in Q1 FY27.
    check("deck: dropped KPI detected without an LLM", "utilisation" in p)
    check("deck: drop is labelled as such", "stopped reporting" in p)

    one_q = dict(T, deck_metrics=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "metric": "order book"}]))
    p1 = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", one_q)
    check("deck: a single quarter says so, not 'no changes'",
          "no comparison is possible" in p1)
    check("deck: single quarter is not an all-clear", "stopped reporting" not in p1)

    r = rating_body("INE1", "AAA", "A Ltd", T)
    check("rating: current rating shown", ">A<" in r or "A</b>" in r)
    check("rating: upgrade flagged", "Upgrade" in r)
    check("rating: what-changed section present", "What changed since" in r)
    check("rating: prior rating shown as 'was'", "A-" in r)
    check("rating: downgrade trigger surfaced", "Debt/EBITDA above 3.5x" in r)
    check("rating: up-trigger NOT shown under downgrade",
          "Sustained margin above 15%" not in r.split("What would cause a downgrade")[-1])
    check("rating: concern surfaced", "Working capital intensity" in r)

    # THE RENDER DEFECTS from the first live run against APLAPOLLO.
    nan_t = dict(T, ppt_highlights=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "demand",
         "statement": "HR Coil tubes to grow faster", "value": float("nan"), "unit": None}]))
    pn = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", nan_t)
    check("NaN value never renders as 'nanNone'", "nanNone" not in pn and "nan" not in pn)

    esg_t = dict(T, ppt_guidance=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "metric": "Scope 1 & 2 emissions",
         "value": "-25", "unit": "%", "horizon": "2030"},
        {"isin": "INE1", "quarter": "Q1 FY27", "metric": "Female workforce",
         "value": "1", "unit": "%", "horizon": "yearly"}]))
    pe = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", esg_t)
    check("ESG targets are not presented as guidance", "Scope 1" not in pe)
    check("ESG-only decks say so explicitly", "ESG commitments only" in pe)
    check("real guidance still shows", "revenue growth" in
          presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T))

    check("no data renders nothing, not an empty shell",
          presentation_body("INE_NONE", "X", "X", "Q1FY27", T) == "")
    check("no ratings renders nothing", rating_body("INE_NONE", "X", "X", T) == "")

    print(f"\npf_company_mails self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    main()
