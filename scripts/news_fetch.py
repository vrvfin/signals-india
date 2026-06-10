"""
Shared Google News RSS fetch layer — used by T4.4 Stage 2 (fraud news scan in
build_investigative_fraud.py) and T5 (build_catalyst_notes.py).

Why this exists alongside the gitignored social_signals.py: that module is
local-only (PF tracking). This one is COMMITTED and CI-safe — stdlib XML parse,
no API key, no new pip deps. Only whitelisted (reliable) sources count.

Google News RSS is free and unauthenticated:
    https://news.google.com/rss/search?q=<query>&hl=en-IN&gl=IN&ceid=IN:en
The query supports `when:7d` style recency filters.

Be polite: module-level throttle (>=1.2s between requests) + per-process call
cap so a buggy loop can never hammer Google.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

RSS_URL = "https://news.google.com/rss/search"

# Reliable Indian business/financial press (+ wires). Source domains outside
# this list are dropped — rumour blogs and content farms never count as a hit.
TRUSTED_DOMAINS = frozenset({
    "moneycontrol.com", "economictimes.indiatimes.com", "business-standard.com",
    "livemint.com", "thehindubusinessline.com", "financialexpress.com",
    "reuters.com", "ndtvprofit.com", "moneylife.in", "businesstoday.in",
    "cnbctv18.com", "zeebiz.com", "thehindu.com", "outlookbusiness.com",
    "bloomberg.com", "timesofindia.indiatimes.com", "telegraphindia.com",
    "tribuneindia.com", "deccanherald.com", "newindianexpress.com",
})

THROTTLE_S = 1.2          # min gap between Google requests (polite)
MAX_CALLS_PER_RUN = 700   # hard per-process cap — refuse beyond this

_last_call = 0.0
_calls_made = 0


class NewsFetchBudgetExceeded(RuntimeError):
    """Raised when a run tries to exceed MAX_CALLS_PER_RUN RSS requests."""


def calls_made() -> int:
    return _calls_made


def _domain(url: str) -> str:
    m = re.match(r"https?://(?:www\.)?([^/]+)", str(url).strip(), re.I)
    return m.group(1).lower() if m else ""


def is_trusted(source_url: str) -> bool:
    d = _domain(source_url)
    return any(d == t or d.endswith("." + t) for t in TRUSTED_DOMAINS)


def fetch_news(query: str, days_back: int = 30,
               trusted_only: bool = True, timeout: int = 20) -> list[dict]:
    """One Google News RSS call. Returns [{title, link, source, source_url,
    published}] newest-first, filtered to TRUSTED_DOMAINS by default.
    Returns [] on any fetch/parse failure (callers treat news as best-effort).
    Raises NewsFetchBudgetExceeded only when the per-run cap is hit."""
    global _last_call, _calls_made
    if _calls_made >= MAX_CALLS_PER_RUN:
        raise NewsFetchBudgetExceeded(
            f"news_fetch: {MAX_CALLS_PER_RUN} RSS calls already made this run")

    wait = THROTTLE_S - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()
    _calls_made += 1

    try:
        r = requests.get(
            RSS_URL,
            params={"q": f"{query} when:{days_back}d",
                    "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            headers={"User-Agent": UA},
            timeout=timeout,
        )
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
    except Exception:
        return []

    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        source_url = src_el.get("url", "") if src_el is not None else ""
        if not title:
            continue
        if trusted_only and not is_trusted(source_url):
            continue
        items.append({
            "title": title, "link": link, "source": source,
            "source_url": source_url, "published": pub,
        })
    return items


def keyword_hits(items: list[dict], keywords: list[str]) -> list[dict]:
    """Subset of items whose title matches any keyword (case-insensitive).
    Each returned dict gains a 'matched' list of the keywords that fired."""
    out = []
    for it in items:
        t = it["title"].lower()
        matched = [kw for kw in keywords if kw.lower() in t]
        if matched:
            out.append({**it, "matched": matched})
    return out


def parse_rss_bytes(data: bytes, trusted_only: bool = True) -> list[dict]:
    """Parse a canned RSS payload (offline fixtures / tests) with the same
    extraction + whitelist rules as fetch_news. No network, no throttle."""
    try:
        root = ET.fromstring(data)
    except Exception:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        source_url = src_el.get("url", "") if src_el is not None else ""
        if not title:
            continue
        if trusted_only and not is_trusted(source_url):
            continue
        items.append({
            "title": title, "link": link, "source": source,
            "source_url": source_url, "published": pub,
        })
    return items


if __name__ == "__main__":
    # Ad-hoc smoke test: python scripts/news_fetch.py "TCS"
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else '"Tata Consultancy"'
    for n in fetch_news(q, days_back=7):
        print(f"  [{n['source']}] {n['title']}")
    print(f"({calls_made()} call(s), {datetime.now().isoformat(timespec='seconds')})")
