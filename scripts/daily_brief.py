r"""
daily_brief.py — daily per-PF-company brief email: A (exchange announcements) + B (research)
+ C (news), materiality-ranked, date-wise. Pure assembly of EXISTING features:
  A: announcement_ledger.parquet (ingest_announcements — per-company BSE + LLM summary)
  B: research_index.parquet      (daily_research_summary — your broker/research PDFs)
  C: company_deep_report.news_block (Google News RSS, reputable-source whitelist)
  mail: mailer.send_email        (the shared Gmail sender + toggle)

Usage:
  python scripts/daily_brief.py --dry-run                 # write HTML locally, no mail
  python scripts/daily_brief.py --dry-run --limit-companies 5 --no-news
  python scripts/daily_brief.py                           # build + email PF brief
"""
from __future__ import annotations
import os, sys, io, time, argparse, html as _html
import datetime as dt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))
import pandas as pd
from ingest_company_docs import get_drive, get_or_create_subfolder, find_file, download_bytes
from _extractor_base import load_parquet, load_portfolio_isins, save_parquet
from mailer import send_email

try:                                                       # deep-dive news feed (C) —
    from alt_sources import news_block                     # light module (requests+bs4);
except Exception:                                          # company_deep_report's chain
    news_block = None                                      # needs fitz, absent in Phase-1 CI

try:                                                       # D — sector-gated alt sources
    from alt_sources import fda_recalls, rbi_circulars, is_pharma, is_finance, FDA_CLASS_MAT
except Exception:
    fda_recalls = rbi_circulars = is_pharma = is_finance = None
    FDA_CLASS_MAT = {}

try:                                                       # sector taxonomy for the FDA/RBI gate
    from build_classification import load_classification
except Exception:
    load_classification = None

LEDGER_COLS = ["newsid", "isin", "symbol", "ann_date", "category", "headline",
               "summary", "status", "materiality", "event_type", "direction",
               "processed_at"]
RESEARCH_COLS = ["file_name", "source", "doc_date", "doc_type", "companies", "isins",
                 "sectors", "themes", "summary_md"]
_MAT = {"high": 0, "medium": 1, "": 2, "none": 2, "low": 3}
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))       # CI runs UTC; report in IST


def _esc(s):
    return _html.escape(str(s or ""))


def _short(s, n=320):
    s = " ".join(str(s or "").split())
    return (s[:n] + "…") if len(s) > n else s


def _research_snippet(md, name, symbol, maxlen=340):
    """Substantive company snippet ("" = drop the item) — shared extractor, see
    research_snippet.py (section-body harvest, word-bounded keys, label-row filter)."""
    from research_snippet import research_snippet
    return research_snippet(md, name, symbol, maxlen=maxlen)


def load_pf(drive, root, index_id):
    """[(isin, symbol, name)] for portfolio companies (universe-resolved)."""
    isins = [str(x).strip() for x in (load_portfolio_isins(drive, root) or [])]
    m = {}
    fid = find_file(drive, index_id, "company_universe.csv")
    if fid:
        u = pd.read_csv(io.BytesIO(download_bytes(drive, fid))).fillna("")
        for _, r in u.iterrows():
            i = str(r.get("isin", "")).strip()
            if i:
                sym = str(r.get("nse_symbol") or r.get("bse_symbol") or "").strip().upper()
                m[i] = (sym, str(r.get("name", "")).strip())
    out = []
    for i in isins:
        sym, name = m.get(i, ("", ""))
        out.append((i, sym or i, name or sym or i))
    return out


def _cls_row(cls_df, isin, sym, name):
    """The company's classification row as a dict, for the pharma/finance gate.
    Falls back to a name-only dict so gating still works if the taxonomy is missing."""
    if cls_df is not None and not cls_df.empty:
        row = cls_df[cls_df["isin"].astype(str) == isin]
        if row.empty and sym:
            row = cls_df[cls_df["symbol"].astype(str).str.upper() == sym.upper()]
        if not row.empty:
            return row.iloc[0].to_dict()
    return {"name": name, "symbol": sym}


def rbi_html(rbi, fin_syms):
    """One shared RBI section (circulars are sector-wide, not company-specific)."""
    if not rbi:
        return ""
    who = ", ".join(fin_syms) if fin_syms else "finance holdings"
    p = ["<h3 style='margin:16px 0 4px'>🏦 RBI circulars & notifications</h3>",
         f"<p style='color:#555;margin:0 0 6px'>Relevant to your finance holdings: "
         f"{_esc(who)}</p><ul style='margin:4px 0'>"]
    for x in rbi[:8]:
        link, title = _esc(x.get("link", "")), _esc(x.get("title", ""))
        title_html = f"<a href='{link}'>{title}</a>" if link else title
        p.append(f"<li>{_esc(x.get('date'))} — [{_esc(x.get('kind'))}] {title_html}</li>")
    p.append("</ul>")
    return "\n".join(p)


# ---- "new since the last brief" (user 2026-10-13) --------------------------------
# The look-back windows (research 30d, news 21d, FDA 60d, RBI 7d) stay, because research
# and FDA items are often processed days after their date - but each item now reaches
# the reader ONCE. Identity per item lives in this ledger; it advances only after a
# confirmed send. Shared by daily.yml and weekend_announcements.yml (same script).
SEEN_NAME = "daily_brief_seen.parquet"
SEEN_COLS = ["item_id", "kind", "symbol", "first_mailed"]
SEEN_KEEP_DAYS = 120          # > the longest look-back (60d), so nothing re-qualifies


def item_id(kind: str, x) -> str:
    """Stable identity of one brief item. News lines are matched on their text."""
    import hashlib
    if kind == "ann":
        key = x.get("newsid") or f"{x.get('ann_date')}|{x.get('headline')}"
    elif kind == "res":
        key = x.get("file_name") or f"{x.get('doc_date')}|{x.get('source')}"
    elif kind == "fda":
        key = (x.get("recall_number") or x.get("event_id")
               or f"{x.get('date')}|{x.get('product')}")
    elif kind == "rbi":
        key = x.get("link") or f"{x.get('date')}|{x.get('title')}"
    else:                                              # news: one text line
        key = " ".join(str(x).lstrip("- ").lower().split())
    return kind + ":" + hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:16]


def only_new(kind: str, items: list, seen: set, out_ids: list, symbol: str = "") -> list:
    """Items whose id is not in `seen`; their ids are queued in `out_ids`."""
    keep = []
    for x in items or []:
        i = item_id(kind, x)
        if i in seen:
            continue
        keep.append(x)
        out_ids.append((i, kind, symbol))
    return keep


def new_news_lines(news_txt: str, seen: set, out_ids: list, symbol: str) -> str:
    lines = [ln for ln in (news_txt or "").splitlines() if ln.strip()]
    return "\n".join(only_new("news", lines, seen, out_ids, symbol))


def load_seen(drive, idx) -> tuple[pd.DataFrame, set]:
    df = load_parquet(drive, idx, SEEN_NAME, SEEN_COLS)
    return df, set(df["item_id"].astype(str)) if not df.empty else set()


def save_seen(drive, idx, prev: pd.DataFrame, ids: list, now_iso: str) -> int:
    add = pd.DataFrame([{"item_id": i, "kind": k, "symbol": s, "first_mailed": now_iso}
                        for i, k, s in ids], columns=SEEN_COLS)
    out = add if prev is None or prev.empty else pd.concat([prev, add], ignore_index=True)
    cut = (dt.date.today() - dt.timedelta(days=SEEN_KEEP_DAYS)).isoformat()
    out = (out[out["first_mailed"].astype(str).str[:10] >= cut]
           .drop_duplicates("item_id", keep="first"))
    save_parquet(drive, idx, SEEN_NAME, out)
    return len(out)


def company_announcements(ledger, isin, symbol, days):
    if ledger is None or ledger.empty:
        return []
    cut = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    # window on ann_date OR processed_at: a backlog filing summarised only today
    # (quota-starved earlier runs) is NEW information — it must reach the next mail
    # even though its ann_date is older than the window.
    recent = ledger["ann_date"].astype(str) >= cut
    if "processed_at" in ledger.columns:
        recent = recent | (ledger["processed_at"].astype(str).str[:10] >= cut)
    m = (((ledger["isin"].astype(str) == isin)
          | (ledger["symbol"].astype(str).str.upper() == symbol.upper()))
         & (ledger["status"].astype(str) == "done")
         & recent)
    sub = ledger[m].copy()
    if sub.empty:
        return []
    sub["_mr"] = sub["materiality"].astype(str).str.lower().map(lambda x: _MAT.get(x, 2))
    return sub.sort_values(["_mr", "ann_date"], ascending=[True, False]).to_dict("records")


def company_research(research, isin, symbol, name, days):
    if research is None or research.empty:
        return []
    il, sl, nl = isin.lower(), (symbol or "").lower(), (name or "").lower()

    def hit(r):
        blob = f"{r.get('isins','')}{r.get('companies','')}".lower()
        return (il and il in blob) or (len(sl) > 2 and sl in blob) or (len(nl) > 3 and nl in blob)
    sub = research[research.apply(hit, axis=1)].copy()
    if sub.empty:
        return []
    if "doc_date" in sub.columns:
        cut = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        sub = sub[sub["doc_date"].astype(str) >= cut]
        sub = sub.sort_values("doc_date", ascending=False)
    return sub.to_dict("records")


def company_html(sym, name, anns, res, news_txt, fda=None):
    hdr = f"<h3 style='margin:16px 0 4px'>{_esc(sym)} — {_esc(name)}</h3>"
    p = [hdr]
    if fda:
        p.append("<b>💊 US FDA recalls</b><ul style='margin:4px 0'>")
        for x in fda[:5]:
            cls = str(x.get("classification", ""))
            mat = FDA_CLASS_MAT.get(cls, "")
            tag = (f"<span style='color:#b00;font-weight:bold'>[{mat}]</span> " if mat == "high"
                   else (f"<span style='color:#a60'>[{mat}]</span> " if mat == "medium" else ""))
            p.append(f"<li>{_esc(x.get('date'))} — {tag}<b>{_esc(cls)}</b>: "
                     f"{_esc(_short(x.get('product'), 90))}"
                     f"<br><span style='color:#444'>{_esc(_short(x.get('reason'), 200))}</span></li>")
        p.append("</ul>")
    if anns:
        p.append("<b>📌 Exchange announcements</b><ul style='margin:4px 0'>")
        for a in anns[:8]:
            mat = str(a.get("materiality", "")).lower()
            tag = (f"<span style='color:#b00;font-weight:bold'>[{mat}]</span> " if mat == "high"
                   else (f"<span style='color:#a60'>[{mat}]</span> " if mat == "medium" else ""))
            p.append(f"<li>{_esc(a.get('ann_date'))} — {tag}<b>{_esc(a.get('headline'))[:120]}</b>"
                     f"<br><span style='color:#444'>{_esc(_short(a.get('summary'), 300))}</span></li>")
        p.append("</ul>")
    if res:
        rli = []
        for r in res[:6]:
            snip = _research_snippet(r.get("summary_md"), name, sym)
            if not snip:                                   # skip rows with empty summary_md
                continue
            _tv = str(r.get("themes") or "").strip()
            th = (f"<br><i style='color:#777'>themes: {_esc(_tv[:120])}</i>"
                  if _tv not in ("", "[]", "nan") else "")
            rli.append(f"<li>{_esc(r.get('doc_date'))} — [{_esc(r.get('source'))}] "
                       f"{_esc(snip)}{th}</li>")
        if rli:
            p.append("<b>📄 Research</b><ul style='margin:4px 0'>")
            p.extend(rli)
            p.append("</ul>")
    if news_txt:
        p.append("<b>📰 News</b><ul style='margin:4px 0'>")
        for line in news_txt.splitlines()[:8]:
            p.append(f"<li>{_esc(line.lstrip('- '))}</li>")
        p.append("</ul>")
    return "\n".join(p) if len(p) > 1 else ""       # header-only ⇒ nothing to show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-days", type=int, default=0,
                    help="exchange-announcement window in days; 0 = auto (Mon=3 to cover "
                         "the weekend, other weekdays=1 ≈ last 24-48h)")
    ap.add_argument("--research-days", type=int, default=30, help="research window")
    ap.add_argument("--news-days", type=int, default=30, help="news window")
    ap.add_argument("--no-news", action="store_true", help="skip the Google-News pass")
    ap.add_argument("--no-alt", action="store_true", help="skip FDA (pharma) + RBI (finance)")
    ap.add_argument("--fda-days", type=int, default=60, help="US FDA recall window (pharma)")
    ap.add_argument("--rbi-days", type=int, default=7, help="RBI circular window (finance)")
    ap.add_argument("--limit-companies", type=int, default=0, help="cap PF companies (testing)")
    ap.add_argument("--dry-run", action="store_true", help="write HTML locally, do not email")
    args = ap.parse_args()

    drive = get_drive(); root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")

    ledger = load_parquet(drive, idx, "announcement_ledger.parquet", LEDGER_COLS)
    research = load_parquet(drive, idx, "research_index.parquet", RESEARCH_COLS)
    pf = load_pf(drive, root, idx)
    if args.limit_companies:
        pf = pf[:args.limit_companies]
    # D — sector taxonomy for the FDA (pharma) / RBI (finance) gate. Fail-soft: if the
    # classification frame is missing, _cls_row falls back to name-keyword gating.
    alt_on = not args.no_alt and is_pharma is not None
    cls_df = load_classification(drive, root) if (alt_on and load_classification) else None
    now_ist = dt.datetime.now(dt.timezone.utc).astimezone(IST)
    # 24-48h on weekdays, 72h on Monday (weekday()==0) so the weekend's filings aren't missed
    ann_days = args.ann_days or (3 if now_ist.weekday() == 0 else 1)
    print(f"PF companies: {len(pf)} | {now_ist:%d %b %Y %H:%M IST} | ann_days={ann_days} "
          f"research_days={args.research_days} news={not args.no_news} alt={alt_on}")

    try:
        seen_df, seen = load_seen(drive, idx)
    except Exception as e:                     # fail-open: a brief with repeats beats none
        print(f"seen-ledger unavailable ({str(e)[:60]}) - showing the full windows")
        seen_df, seen = pd.DataFrame(columns=SEEN_COLS), set()
    new_ids: list = []
    print(f"seen ledger: {len(seen)} item(s) already mailed")

    sections, n_ann, n_res, n_co, n_fda = [], 0, 0, 0, 0
    fin_syms = []
    for isin, sym, name in pf:
        anns = only_new("ann", company_announcements(ledger, isin, sym, ann_days),
                        seen, new_ids, sym)
        res = only_new("res", company_research(research, isin, sym, name,
                                               args.research_days), seen, new_ids, sym)
        news_txt = ""
        if not args.no_news and news_block:
            try:
                nt = news_block(name, sym, days=args.news_days, limit=8)
                if nt and "No recent news" not in nt and not nt.startswith("DATA_MISSING"):
                    news_txt = new_news_lines(nt, seen, new_ids, sym)
            except Exception:
                news_txt = ""
            time.sleep(0.4)                                # be gentle with Google News
        fda = []
        if alt_on:                                         # D — pharma -> FDA, finance -> RBI
            crow = _cls_row(cls_df, isin, sym, name)
            if is_pharma(crow) and fda_recalls:
                try:
                    fda = only_new("fda", fda_recalls(name, days=args.fda_days),
                                   seen, new_ids, sym)
                except Exception:
                    fda = []
            if is_finance(crow):
                fin_syms.append(sym)
        if not (anns or res or news_txt or fda):
            continue
        sec = company_html(sym, name, anns, res, news_txt, fda)
        if not sec:                                    # all rows empty after filtering
            continue
        sections.append(sec)
        n_ann += len(anns); n_res += len(res); n_fda += len(fda); n_co += 1

    # RBI circulars are sector-wide — fetch ONCE and render one shared finance section.
    rbi = []
    if alt_on and rbi_circulars and fin_syms:
        try:
            rbi = only_new("rbi", rbi_circulars(days=args.rbi_days), seen, new_ids)
        except Exception:
            rbi = []
    rbi_sec = rbi_html(rbi, sorted(set(fin_syms)))
    if rbi_sec:
        sections.append(rbi_sec)

    today = now_ist.strftime("%d %b %Y")
    stamp = now_ist.strftime("%d %b %Y, %H:%M IST")
    alt_bits = (f"{n_fda} FDA · {len(rbi)} RBI" if (n_fda or rbi) else "")
    subject = (f"📋 PF Daily Brief — {today} — {n_co} cos · {n_ann} announcements · "
               f"{n_res} research" + (f" · {alt_bits}" if alt_bits else ""))
    if sections:
        intro = (f"<b>Only what is new since the last brief</b> — {n_co} companies · "
                 f"{n_ann} exchange announcements · {n_res} research items · news from "
                 f"reputable sources. Anything already mailed is left out")
        if alt_bits:
            intro += f" · {n_fda} US-FDA recalls (pharma) · {len(rbi)} RBI circulars (finance)"
        body = (f"<div style='font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222'>"
                f"<h2 style='margin:0 0 2px'>📋 PF Daily Brief</h2>"
                f"<div style='color:#888;font-size:12px;margin:0 0 6px'>as of {stamp}</div>"
                f"<p style='color:#555;margin:0 0 8px'>{intro}.</p><hr>"
                + "<hr>".join(sections) + "</div>")
    else:
        body = (f"<p>Nothing new for your holdings since the last brief "
                f"(as of {stamp}).</p>")
    print(f"brief: {n_co} companies with updates, {n_ann} announcements, {n_res} research items, "
          f"{n_fda} FDA recalls, {len(rbi)} RBI circulars")

    if args.dry_run:
        outp = os.path.join(os.path.dirname(_HERE), "company_reports", "pf_daily_brief.html")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, "w", encoding="utf-8") as fh:
            fh.write(body)
        _safe = subject.encode("ascii", "replace").decode()   # Windows console is cp1252
        print(f"DRY-RUN - wrote {outp} (no mail). Subject: {_safe}")
        return
    if not sections:
        # nothing new: no mail, and nothing to add to the ledger
        print("nothing new since the last brief - no mail")
        return
    if send_email(subject, body) and new_ids:
        # the ledger advances ONLY after a confirmed send
        n = save_seen(drive, idx, seen_df, new_ids,
                      dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        print(f"seen ledger: +{len(new_ids)} -> {SEEN_NAME} ({n} rows)")


def _self_test() -> int:
    """Offline: identities, the new-only filter, news lines. No Drive, no mail."""
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    a1 = {"newsid": "N1", "headline": "Order win"}
    a2 = {"newsid": "N2", "headline": "Board meeting"}
    check("same item -> same id", item_id("ann", a1) == item_id("ann", dict(a1)))
    check("kinds never collide", item_id("ann", {"newsid": "X"}) != item_id("rbi", {"link": "X"}))
    ids: list = []
    seen = {item_id("ann", a1)}
    got = only_new("ann", [a1, a2], seen, ids, "SYM")
    check("already-mailed item dropped", got == [a2])
    check("new item queued for the ledger", [i[0] for i in ids] == [item_id("ann", a2)])
    r1 = {"file_name": "broker_note.pdf", "doc_date": "2026-09-20"}
    check("research keyed on file name",
          item_id("res", r1) == item_id("res", {"file_name": "broker_note.pdf"}))
    news = "- 2026-10-12 · ET: X wins order\n- 2026-10-11 · BS: X results"
    seen2 = {item_id("news", "- 2026-10-11 · BS: X results")}
    ids2: list = []
    out = new_news_lines(news, seen2, ids2, "X")
    check("news: only unseen lines kept", out == "- 2026-10-12 · ET: X wins order")
    check("news: case/space-insensitive", item_id("news", "-  2026-10-11 · bs: x RESULTS")
          == item_id("news", "- 2026-10-11 · BS: X results"))
    check("fda prefers recall_number",
          item_id("fda", {"recall_number": "D-1", "date": "a"})
          == item_id("fda", {"recall_number": "D-1", "date": "b"}))
    check("empty input -> empty", only_new("rbi", [], set(), []) == [])
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    main()
