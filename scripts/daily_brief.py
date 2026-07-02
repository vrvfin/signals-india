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
from _extractor_base import load_parquet, load_portfolio_isins
from mailer import send_email

try:                                                       # reuse the deep-dive news feed (C)
    from company_deep_report import news_block
except Exception:
    news_block = None

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
               "summary", "status", "materiality", "event_type", "direction"]
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


def company_announcements(ledger, isin, symbol, days):
    if ledger is None or ledger.empty:
        return []
    cut = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    m = (((ledger["isin"].astype(str) == isin)
          | (ledger["symbol"].astype(str).str.upper() == symbol.upper()))
         & (ledger["status"].astype(str) == "done")
         & (ledger["ann_date"].astype(str) >= cut))
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

    sections, n_ann, n_res, n_co, n_fda = [], 0, 0, 0, 0
    fin_syms = []
    for isin, sym, name in pf:
        anns = company_announcements(ledger, isin, sym, ann_days)
        res = company_research(research, isin, sym, name, args.research_days)
        news_txt = ""
        if not args.no_news and news_block:
            try:
                nt = news_block(name, sym, days=args.news_days, limit=8)
                if nt and "No recent news" not in nt and not nt.startswith("DATA_MISSING"):
                    news_txt = nt
            except Exception:
                news_txt = ""
            time.sleep(0.4)                                # be gentle with Google News
        fda = []
        if alt_on:                                         # D — pharma -> FDA, finance -> RBI
            crow = _cls_row(cls_df, isin, sym, name)
            if is_pharma(crow) and fda_recalls:
                try:
                    fda = fda_recalls(name, days=args.fda_days)
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
            rbi = rbi_circulars(days=args.rbi_days)
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
        intro = (f"{n_co} companies with updates · {n_ann} exchange announcements "
                 f"(last {ann_days}d) · {n_res} research items (last {args.research_days}d) · "
                 f"news from reputable sources")
        if alt_bits:
            intro += f" · {n_fda} US-FDA recalls (pharma) · {len(rbi)} RBI circulars (finance)"
        body = (f"<div style='font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222'>"
                f"<h2 style='margin:0 0 2px'>📋 PF Daily Brief</h2>"
                f"<div style='color:#888;font-size:12px;margin:0 0 6px'>as of {stamp}</div>"
                f"<p style='color:#555;margin:0 0 8px'>{intro}.</p><hr>"
                + "<hr>".join(sections) + "</div>")
    else:
        body = f"<p>No new PF announcements/research in the window (as of {stamp}).</p>"
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
    send_email(subject, body)


if __name__ == "__main__":
    main()
