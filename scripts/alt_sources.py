r"""
alt_sources.py — sector-gated alternative catalyst sources for the daily brief (D).
Pure readers, NO keys, NO Gemini. Importable + `python scripts/alt_sources.py --selftest`.

  fda_recalls(name, days, limit)   -> US FDA drug recalls (openFDA)   — PHARMA only
  rbi_circulars(days, limit)       -> RBI notifications/circulars (RSS) — FINANCE only
  is_pharma(row) / is_finance(row) -> gate on a company_classification row (or {name})

Design (user 2026-07-01): FDA is per-company (recalls keyed by recalling_firm); RBI is
sector-wide (one circular affects all banks/NBFCs), fetched once and shown as a shared
finance section. Both endpoints verified live: openFDA drug/enforcement returns dated
Class I/II/III recalls; RBI notifications_rss.xml / pressreleases_rss.xml are RSS 2.0 with
a UTF-8 BOM (decode utf-8-sig) parsed the same way as company_deep_report.news_block.
"""
from __future__ import annotations
import re, sys, argparse, datetime as dt
import requests

FDA_ENFORCE = "https://api.fda.gov/drug/enforcement.json"
RBI_FEEDS = (
    ("https://www.rbi.org.in/notifications_rss.xml", "Notification"),
    ("https://www.rbi.org.in/pressreleases_rss.xml", "Press release"),
)
_UA = {"User-Agent": "Mozilla/5.0"}

# ── sector gate ────────────────────────────────────────────────────────────────
_PHARMA_KW = ("pharma", "healthcare", "health care", "life scien", "biotech",
              "drug", "hospital", "diagnostic", "medical", "api ")
_FINANCE_KW = ("financ", "bank", "nbfc", "insur", "broker", "asset manag",
               "housing finance", "capital market", "fintech", "lending", "amc",
               "wealth", "depository")
# FDA class -> materiality (mirrors the announcement colour scheme in daily_brief)
FDA_CLASS_MAT = {"Class I": "high", "Class II": "medium", "Class III": "low"}


def _blob(row) -> str:
    return " ".join(str(row.get(k, "")) for k in
                    ("macro_sector", "sector", "industry", "subsector",
                     "peer_group", "name")).lower()


def is_pharma(row) -> bool:
    return any(k in _blob(row) for k in _PHARMA_KW)


def is_finance(row) -> bool:
    return any(k in _blob(row) for k in _FINANCE_KW)


# ── US FDA (openFDA drug enforcement / recalls) ─────────────────────────────────
_DROP = re.compile(r"(?i)\b(limited|ltd|company|co|the|india|inc|corporation|corp)\b\.?")


def fda_search_name(name: str) -> str:
    """Distinctive firm phrase for recalling_firm search: strip corporate suffixes,
    keep the first two tokens (e.g. 'Aurobindo Pharma Limited' -> 'Aurobindo Pharma',
    'Sun Pharmaceutical Industries Ltd' -> 'Sun Pharmaceutical')."""
    n = _DROP.sub(" ", str(name or ""))
    n = re.sub(r"[^A-Za-z0-9 ]", " ", n)
    toks = [t for t in n.split() if len(t) > 1]
    return " ".join(toks[:2]).strip()


def fda_recalls(name: str, days: int = 60, limit: int = 5) -> list[dict]:
    """Recent US FDA drug recalls for `name` (openFDA). [] on no match / any error.
    openFDA returns HTTP 404 when the search matches nothing — treated as empty."""
    q = fda_search_name(name)
    if len(q) < 3:
        return []
    since = (dt.date.today() - dt.timedelta(days=days)).strftime("%Y%m%d")
    today = dt.date.today().strftime("%Y%m%d")
    params = {"search": f'recalling_firm:"{q}" AND report_date:[{since} TO {today}]',
              "sort": "report_date:desc", "limit": limit}
    try:
        r = requests.get(FDA_ENFORCE, params=params, headers=_UA, timeout=25)
        if r.status_code != 200:                       # 404 = no results
            return []
        results = r.json().get("results", [])
    except Exception:
        return []
    out = []
    for x in results:
        d = str(x.get("report_date", ""))
        d = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d
        out.append({
            "date": d,
            "classification": str(x.get("classification", "")).strip(),
            "product": (x.get("product_description") or "").strip(),
            "reason": (x.get("reason_for_recall") or "").strip(),
            "status": str(x.get("status", "")).strip(),
        })
    return out


# ── RBI (notifications + press releases RSS) ─────────────────────────────────────
def _parse_pubdate(pub: str):
    pub = (pub or "").strip()
    for fmt, n in (("%a, %d %b %Y %H:%M:%S", 25), ("%a, %d %b %Y", 16)):
        try:
            return dt.datetime.strptime(pub[:n].strip(), fmt)
        except Exception:
            continue
    return None


def rbi_circulars(days: int = 7, limit: int = 6) -> list[dict]:
    """Recent RBI notifications/circulars + press releases (RSS), newest-first,
    deduped by title. Sector-wide — the caller decides which finance holdings it
    applies to. [] on any error."""
    from bs4 import BeautifulSoup
    cut = dt.datetime.now() - dt.timedelta(days=days)
    rows = []
    for url, kind in RBI_FEEDS:
        try:
            r = requests.get(url, headers=_UA, timeout=25)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content.decode("utf-8-sig", "ignore"), "lxml-xml")
            for it in soup.find_all("item"):
                title = (it.title.get_text() if it.title else "").strip()
                if not title:
                    continue
                link = (it.link.get_text() if it.link else "").strip()
                d = _parse_pubdate((it.pubDate.get_text() if it.pubDate else "").strip())
                if d is not None and d < cut:
                    continue
                rows.append({"date": d.strftime("%Y-%m-%d") if d else "",
                             "title": title, "link": link, "kind": kind})
        except Exception:
            continue
    seen, out = set(), []
    for x in sorted(rows, key=lambda z: z["date"], reverse=True):
        k = x["title"].lower()[:80]
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
        if len(out) >= limit:
            break
    return out


# ── Google-News RSS (C) ──────────────────────────────────────────────────────
# Moved here from company_deep_report.py (2026-07-02): that module's import chain
# pulls pdf_ocr -> fitz (PyMuPDF), which Phase-1 CI does not install — so
# daily_brief's `from company_deep_report import news_block` failed silently in CI
# and the mail never had news. This module is requests+bs4 only.

# Reputable sources only (user-chosen whitelist: dailies + markets + wires,
# plus the Economic Times pharma/business verticals which are the same publisher).
NEWS_WHITELIST = (
    "economic times", "etmarkets", "etpharma", "express pharma",       # ET family
    "business standard", "mint", "livemint", "hindu businessline",
    "businessline", "financial express",                               # dailies
    "moneycontrol", "cnbc", "ndtv profit", "bq prime", "quint",        # markets
    "reuters", "press trust", "pti", "bloomberg",                      # wires
)


def news_block(name, symbol, limit=25, days=365):
    """Recent company news headlines from Google News RSS, filtered to reputable
    sources. Headlines + source + date only (article bodies are JS-redirected and
    not reliably fetchable). External signal — corroborate/contrast vs financials."""
    import urllib.parse
    try:
        from bs4 import BeautifulSoup
        q = urllib.parse.quote(f'"{name}" OR {symbol}')
        url = (f"https://news.google.com/rss/search?q={q}%20when:{days}d"
               "&hl=en-IN&gl=IN&ceid=IN:en")
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
        if r.status_code != 200:
            return f"DATA_MISSING (news fetch failed: HTTP {r.status_code})."
        soup = BeautifulSoup(r.text, "lxml-xml")
        seen, out = set(), []
        for it in soup.find_all("item"):
            title = (it.title.get_text() if it.title else "").strip()
            src = (it.source.get_text() if it.source else "").strip()
            pub = (it.pubDate.get_text() if it.pubDate else "")
            if not title or not src:
                continue
            if not any(w in src.lower() for w in NEWS_WHITELIST):
                continue
            headline = re.sub(r"\s*-\s*[^-]+$", "", title).strip()   # drop " - Source"
            key = headline.lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            try:
                d = dt.datetime.strptime(pub[:16], "%a, %d %b %Y").strftime("%Y-%m-%d")
            except Exception:
                d = pub[:16]
            out.append((d, f"- {d} | {headline} [{src}]"))
            if len(out) >= limit:
                break
        if not out:
            return "No recent news from whitelisted sources."
        out.sort(reverse=True)
        return "\n".join(line for _, line in out)
    except Exception as e:
        return f"DATA_MISSING (news fetch failed: {type(e).__name__})"


def _selftest():
    print("FDA fda_search_name samples:",
          [fda_search_name(n) for n in
           ("Aurobindo Pharma Limited", "Sun Pharmaceutical Industries Ltd",
            "Sai Life Sciences Limited")])
    fda = fda_recalls("Aurobindo Pharma", days=365)
    print(f"FDA Aurobindo recalls (365d): {len(fda)}")
    for x in fda[:3]:
        print("  ", x["date"], "|", x["classification"], "|", x["product"][:60])
    rbi = rbi_circulars(days=30, limit=6)
    print(f"RBI circulars (30d): {len(rbi)}")
    for x in rbi[:4]:
        print("  ", x["date"], "|", x["kind"], "|", x["title"][:70])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        ap.print_help()
