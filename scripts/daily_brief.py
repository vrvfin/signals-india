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

LEDGER_COLS = ["newsid", "isin", "symbol", "ann_date", "category", "headline",
               "summary", "status", "materiality", "event_type", "direction"]
RESEARCH_COLS = ["file_name", "source", "doc_date", "doc_type", "companies", "isins",
                 "sectors", "themes", "summary_md"]
_MAT = {"high": 0, "medium": 1, "": 2, "none": 2, "low": 3}


def _esc(s):
    return _html.escape(str(s or ""))


def _short(s, n=320):
    s = " ".join(str(s or "").split())
    return (s[:n] + "…") if len(s) > n else s


def _research_snippet(md, name, symbol, maxlen=340):
    """Pull the COMPANY-RELEVANT bit out of a research summary_md (which starts with a
    generic 'OUTPUT SECTION / DOCUMENT HEADER' table). Prefer lines that mention the company
    (e.g. a sector note's Bajaj-Consumer row); else the first substantive prose line."""
    md = str(md or "")
    key = (name.split()[0] if name else (symbol or "")).lower()
    _skip = ("output section", "document header", "field", "source/author", "companies |",
             "| companies", "document type", "document date", "|---", "---|", "===")
    rel = []
    for ln in md.splitlines():
        s = ln.strip()
        if not s or not key or key not in s.lower():
            continue
        if any(s.lower().startswith(p) or p in s.lower()[:14] for p in _skip):
            continue
        rel.append(s.strip("|").strip())
    if rel:
        return _short(" | ".join(rel[:3]), maxlen)
    for ln in md.splitlines():                                # fallback: first real prose
        s = ln.strip()
        if len(s) > 55 and not s.startswith(("|", "#", "=", "-")) \
                and "OUTPUT SECTION" not in s and "DOCUMENT HEADER" not in s:
            return _short(s, maxlen)
    return _short(md.replace("=", ""), maxlen)


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


def company_html(sym, name, anns, res, news_txt):
    hdr = f"<h3 style='margin:16px 0 4px'>{_esc(sym)} — {_esc(name)}</h3>"
    p = [hdr]
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
            th = f"<br><i style='color:#777'>themes: {_esc(str(r.get('themes'))[:120])}</i>" if r.get("themes") else ""
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
    ap.add_argument("--ann-days", type=int, default=3, help="exchange-announcement window")
    ap.add_argument("--research-days", type=int, default=30, help="research window")
    ap.add_argument("--news-days", type=int, default=30, help="news window")
    ap.add_argument("--no-news", action="store_true", help="skip the Google-News pass")
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
    print(f"PF companies: {len(pf)} | ann_days={args.ann_days} research_days={args.research_days} "
          f"news={not args.no_news}")

    sections, n_ann, n_res, n_co = [], 0, 0, 0
    for isin, sym, name in pf:
        anns = company_announcements(ledger, isin, sym, args.ann_days)
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
        if not (anns or res or news_txt):
            continue
        sec = company_html(sym, name, anns, res, news_txt)
        if not sec:                                    # all rows empty after filtering
            continue
        sections.append(sec)
        n_ann += len(anns); n_res += len(res); n_co += 1

    today = dt.date.today().strftime("%d %b %Y")
    subject = f"📋 PF Daily Brief — {today} — {n_co} cos · {n_ann} announcements · {n_res} research"
    if sections:
        body = (f"<div style='font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222'>"
                f"<h2 style='margin:0 0 4px'>📋 PF Daily Brief — {today}</h2>"
                f"<p style='color:#555;margin:0 0 8px'>{n_co} companies with updates · "
                f"{n_ann} exchange announcements (last {args.ann_days}d) · {n_res} research items "
                f"(last {args.research_days}d) · news from reputable sources.</p><hr>"
                + "<hr>".join(sections) + "</div>")
    else:
        body = f"<p>No new PF announcements/research in the window ({today}).</p>"
    print(f"brief: {n_co} companies with updates, {n_ann} announcements, {n_res} research items")

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
