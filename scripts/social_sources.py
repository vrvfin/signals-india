"""
social_sources.py — community/blog/social coverage for a company (2026-06-12).

Three source families, ONE item shape, every item carries a DIRECT URL
(emails link to it; the deep dive cites `source`):

    {source, author, date(YYYY-MM-DD), text, url}

1. ValuePickr (free Discourse JSON): search the company's thread, take the
   latest posts from TOP CONTRIBUTORS (trust_level >= MIN_TRUST or score >=
   MIN_SCORE). forum.valuepickr.com/search.json -> /t/<id>/posts.json.
2. Curated Indian investing blogs via RSS (BLOG_FEEDS below — edit freely);
   items mentioning the company by name.
3. Twitter/X: REAL fetch only when X_BEARER_TOKEN is set (X API v2 recent
   search over curated handles); the free tier has no search, so without a
   paid key this returns [] and logs once. Handle list: X_HANDLES.

All fetchers fail soft ([]), throttle politely, and never raise.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Accept": "application/json"}
VP_BASE = "https://forum.valuepickr.com"
MIN_TRUST = 2          # Discourse: 2=member, 3=regular, 4=leader
MIN_SCORE = 50.0       # OR a high post score counts as "top contributor"
_THROTTLE_S = 0.8
_last = 0.0

# Curated blog RSS feeds (Indian fundamental-investing writers). Extend freely.
BLOG_FEEDS = [
    ("Dr Vijay Malik", "https://www.drvijaymalik.com/feed/"),
    ("Safal Niveshak", "https://www.safalniveshak.com/feed/"),
    ("AlphaIdeas", "https://alphaideas.in/feed/"),
]

# Curated X/Twitter handles (used ONLY when X_BEARER_TOKEN is set).
X_HANDLES = ["unseenvalue", "safalniveshak", "drvijaymalik", "Vivek_Investor",
             "varinder_bansal"]


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _throttle():
    global _last
    wait = _THROTTLE_S - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.time()


def _plain(html: str, cap: int = 400) -> str:
    txt = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", txt).strip()[:cap]


# ------------------------------------------------------------------ #
#  ValuePickr                                                          #
# ------------------------------------------------------------------ #

def vp_recent_posts(company_name: str, days: int = 7,
                    max_posts: int = 4) -> list[dict]:
    """Latest top-contributor posts on the company's ValuePickr thread."""
    name = (company_name or "").strip()
    if len(name) < 4:
        return []
    try:
        _throttle()
        r = requests.get(f"{VP_BASE}/search.json", params={"q": name},
                         headers=UA, timeout=20)
        topics = r.json().get("topics", [])
        # the company's own thread: title must contain the name's first token(s)
        tok = name.lower().split()[0]
        topic = next((t for t in topics
                      if tok in str(t.get("title", "")).lower()), None)
        if not topic:
            return []
        tid = topic["id"]
        _throttle()
        td = requests.get(f"{VP_BASE}/t/{tid}.json", headers=UA,
                          timeout=20).json()
        stream = td.get("post_stream", {}).get("stream", [])
        if not stream:
            return []
        _throttle()
        pr = requests.get(f"{VP_BASE}/t/{tid}/posts.json",
                          params=[("post_ids[]", i) for i in stream[-15:]],
                          headers=UA, timeout=20)
        posts = pr.json().get("post_stream", {}).get("posts", [])
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        out = []
        for p in posts:
            if str(p.get("created_at", "")) < cutoff:
                continue
            trust = int(p.get("trust_level") or 0)
            score = float(p.get("score") or 0)
            if trust < MIN_TRUST and score < MIN_SCORE:
                continue
            out.append({
                "source": "ValuePickr",
                "author": str(p.get("username", "")),
                "date": str(p.get("created_at", ""))[:10],
                "text": _plain(p.get("cooked", "")),
                "url": f"{VP_BASE}/t/{topic.get('slug', tid)}/{tid}/"
                       f"{p.get('post_number', '')}",
            })
        return sorted(out, key=lambda x: x["date"], reverse=True)[:max_posts]
    except Exception as e:
        _log(f"  VP fetch failed for {name[:25]} ({str(e)[:60]})")
        return []


# ------------------------------------------------------------------ #
#  Curated blogs (RSS)                                                 #
# ------------------------------------------------------------------ #

_feed_cache: dict[str, list[dict]] = {}


def _feed_items(label: str, url: str) -> list[dict]:
    if url in _feed_cache:
        return _feed_cache[url]
    items: list[dict] = []
    try:
        _throttle()
        r = requests.get(url, headers={"User-Agent": UA["User-Agent"]},
                         timeout=15)
        if r.status_code == 200:
            import xml.etree.ElementTree as ET
            for it in ET.fromstring(r.content).iter("item"):
                g = {c.tag.split("}")[-1].lower(): (c.text or "")
                     for c in it}
                dt = ""
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(g.get("pubdate", "")) \
                        .strftime("%Y-%m-%d")
                except Exception:
                    pass
                items.append({"source": label,
                              "author": label,
                              "date": dt,
                              "title": _plain(g.get("title", ""), 200),
                              "text": _plain(g.get("description", ""), 400),
                              "url": (g.get("link") or "").strip()})
    except Exception as e:
        _log(f"  blog feed {label} failed ({str(e)[:50]})")
    _feed_cache[url] = items
    return items


def blog_items(company_name: str, days: int = 21) -> list[dict]:
    """Curated-blog posts mentioning the company (feeds cached per run)."""
    name = (company_name or "").strip().lower()
    if len(name) < 4:
        return []
    key = name.split()[0]
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    out = []
    for label, url in BLOG_FEEDS:
        for it in _feed_items(label, url):
            blob = (it["title"] + " " + it["text"]).lower()
            if key in blob and (not it["date"] or it["date"] >= cutoff):
                out.append({**it, "text": it["title"] + " — " + it["text"]})
    return out[:3]


# ------------------------------------------------------------------ #
#  Twitter / X (real only with a paid API key)                         #
# ------------------------------------------------------------------ #

_x_warned = False


def twitter_items(company_name: str, days: int = 7) -> list[dict]:
    """X recent-search over curated handles. Needs X_BEARER_TOKEN (X API v2
    Basic+ — the free tier has NO search endpoint). Without it: [] + one log."""
    global _x_warned
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if not token:
        if not _x_warned:
            _log("  X/Twitter source dormant — set X_BEARER_TOKEN to enable.")
            _x_warned = True
        return []
    try:
        _throttle()
        froms = " OR ".join(f"from:{h}" for h in X_HANDLES)
        q = f'"{company_name}" ({froms}) -is:retweet'
        r = requests.get("https://api.twitter.com/2/tweets/search/recent",
                         params={"query": q, "max_results": 10,
                                 "tweet.fields": "created_at,author_id"},
                         headers={"Authorization": f"Bearer {token}"},
                         timeout=20)
        out = []
        for t in r.json().get("data", []) or []:
            out.append({"source": "X", "author": str(t.get("author_id", "")),
                        "date": str(t.get("created_at", ""))[:10],
                        "text": _plain(t.get("text", "")),
                        "url": f"https://x.com/i/web/status/{t.get('id')}"})
        return out[:4]
    except Exception as e:
        _log(f"  X fetch failed ({str(e)[:60]})")
        return []


# ------------------------------------------------------------------ #
#  Combined                                                            #
# ------------------------------------------------------------------ #

def community_items(company_name: str, days: int = 7) -> list[dict]:
    """VP + blogs + X for one company, newest first."""
    items = (vp_recent_posts(company_name, days)
             + blog_items(company_name, max(days, 21))
             + twitter_items(company_name, days))
    return sorted(items, key=lambda x: x.get("date", ""), reverse=True)


def community_block(company_name: str, days: int = 7) -> str:
    """Text block for prompts — every line names the source (deep-dive rule)."""
    items = community_items(company_name, days)
    if not items:
        return "DATA_MISSING"
    return "\n".join(
        f"- [{i['source']} · {i['author']}] {i['date']}: {i['text'][:300]} "
        f"({i['url']})" for i in items)
