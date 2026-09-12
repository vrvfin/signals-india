r"""
fetch_mgmt_interviews.py — the latest MANAGEMENT interviews (MD / CEO / CFO / Chairman /
founder) for PF + watchlist companies, each with a link and a short summary.

INDEPENDENT WORK STREAM (user decision 2026-09-11). Own ledger, own daily page, own
evening mail. It does NOT write processing_queue, company_page.md or the deep-dive
yt_summaries/ sidecars, because two live queue readers would treat a YouTube link as a
PDF: narrative_sources.queue_rows keeps one 'done' row of EVERY doc type and re-fetches
its pdf_url, and the deep dive re-summarises pending/error rows. Merging interview
summaries into company pages is a later, separate step.

WHERE INTERVIEWS COME FROM (all keyless; every feed below verified 2026-09-11)
  YouTube channel RSS  youtube.com/feeds/videos.xml?channel_id=… returns only the LATEST
                       15 uploads. On the busiest channels that is about an hour (CNBC
                       Awaaz ~1 h, NDTV Profit ~1.5 h, CNBC-TV18 ~2 h), so --poll runs every
                       30 min. Konexio Network is the opposite: almost every upload is a
                       small-cap or SME management interview, and 15 uploads span ~12 days.
  Google News RSS      print interviews and TV write-ups, via the shared news_fetch.py.
                       Its links are encoded news.google.com/rss/articles/CBMi… URLs that
                       do not redirect server-side, so a news hit carries headline +
                       outlet + clickable link only.
Transcripts are NOT scraped: YouTube blocks cloud-provider IPs for that
(youtube-transcript-api README). Instead Gemini watches the video from its URL
(BucketPool.call_video) — Google fetches it, so nothing is blocked. The free tier allows
8 h of YouTube video a day; VIDEO_BUDGET keeps well inside that.

With a YOUTUBE_API_KEY in env, the poll reads each channel's full upload list
(playlistItems.list, 1 quota unit per 50 videos) instead of the 15-item RSS window.

LEDGER  company_repo/_index/interview_ledger.parquet
  status: candidate -> not_interview | unmatched | matched -> done | error
  --poll only appends candidates. --run classifies, matches, summarises, publishes.
  Written under its OWN lock (interviews.lock), never Phase 2's _extract.lock.

OUTPUTS
  company_repo/_daily/mgmt_interviews_DD_MonYYYY.md  -> Streamlit Company Intel ->
      Daily Digests -> "Mgmt Interviews" (app.py load_daily_index parses this name)
  one evening mail (mailer toggle key 'mgmt_interviews'), only when there is news

Usage:
  python scripts/fetch_mgmt_interviews.py --self-test                  # offline
  python scripts/fetch_mgmt_interviews.py --poll --dry-run             # feeds only
  python scripts/fetch_mgmt_interviews.py --poll                       # one poll
  python scripts/fetch_mgmt_interviews.py --poll --loop-min 340 --every-min 30   # CI
  python scripts/fetch_mgmt_interviews.py --run --dry-run [--classify] # no writes
  python scripts/fetch_mgmt_interviews.py --run --names TCS --limit 1  # live, 1 co
  python scripts/fetch_mgmt_interviews.py --run --pf-only              # holdings only
  python scripts/fetch_mgmt_interviews.py --run                        # daily (CI)
  python scripts/fetch_mgmt_interviews.py --probe-video <youtube url>  # model check
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# A Windows console is cp1252; titles carry Hindi, rupee signs and em dashes. Degrade
# the characters, never the run (same guard as aggregate_signals.py).
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:          # pragma: no cover - not every stream supports it
    pass

import news_fetch
from _extractor_base import (log, salvage_json_objects, sstr)

# ------------------------------------------------------------------ #
#  Configuration                                                     #
# ------------------------------------------------------------------ #

# (channel_id, display name). Every feed verified 2026-09-11. Window = how far back the
# 15 newest uploads reached that evening; it is why --poll runs every 30 minutes.
CHANNELS = [
    ("UCmRbHAgG2k2vDUvb3xsEunQ", "CNBC-TV18"),            # ~2 h
    ("UCI_mwTKUhicNzFrhm33MzBQ", "ET Now"),               # ~5-6 h
    ("UC3uJIdRFTGgLWrUziaHbzrg", "NDTV Profit"),          # ~1.5 h (ex-Bloomberg TV India)
    ("UCQIycDaLsBpMKjOCeaKUYVg", "CNBC Awaaz"),           # ~1 h
    ("UCD3CdwT8lTCe5ZGHbUBxmWA", "ET Now Swadesh"),       # ~2 h
    ("UCkXopQ3ubd-rnXnStZqCl2w", "Zee Business"),         # ~7 h
    ("UChftTVI0QJmyXkajQYt2tiQ", "Moneycontrol"),         # ~8 h
    ("UCQsob4fGjHWhYHW0OLb6rew", "Business Standard"),    # ~31 h
    ("UCdOflwQsgRV_RbgRGRpr_TQ", "BusinessLine"),         # ~3.5 days
    ("UCIALMKvObZNtJ6AmdCLP7Lg", "Bloomberg Television"),  # ~10 h, global
    ("UCkvsXDQM24TFHUgZQJlJSMg", "Konexio Network"),      # ~12 days, SME/small-cap MDs
]
# Bloomberg TV India (UCcC8Yq1ka5NbqDWHJqCnhpA) is deliberately absent: its feed is 404.
# The channel became BloombergQuint -> BQ Prime -> NDTV Profit, which is listed above.

LEDGER_NAME = "interview_ledger.parquet"
LEDGER_COLS = ["item_id", "source", "channel", "title", "description", "url",
               "published_at", "first_seen_at", "status", "isin", "symbol",
               "company_name", "in_pf", "person", "role", "summary", "model",
               "processed_at", "attempts", "last_error", "mailed_at",
               # another upload of the SAME interview (status 'duplicate') points at
               # the row that carries the summary
               "dup_of"]
LOCK_NAME = "interviews.lock"            # own lock: never contends with _extract.lock
LOCK_OWNER = "interviews"
LOCK_MAX_AGE_MIN = 30                    # holds last seconds; a crashed holder is stolen
DAILY_PREFIX = "mgmt_interviews"
DAILY_KEEP_DAYS = 90
# TWO mails (user 2026-09-12): holdings and watchlist names never share an inbox item,
# and each can be switched off on its own.
MAIL_KEY_PF = "mgmt_interviews"
MAIL_KEY_WL = "mgmt_interviews_watchlist"
MAX_HTML_BYTES = 90_000                  # Gmail clips a body at ~102 KB
MAIL_ITEMS_PER_CO = 8                    # bounds one company's block; the rest wait a day
MAIL_MAX_AGE_DAYS = 7                    # never mail a backlog older than a week
# Non-interview rows are kept long enough to dedupe the slowest feed: Konexio's 15
# uploads spanned ~12 days, so a shorter retention would re-admit its old videos.
CANDIDATE_KEEP_DAYS = 21
DESC_CHARS = 400                         # description kept per row (classify + fallback)
# API mode look-back. A routine 30-min poll re-reads 3 h (one page on every channel,
# ~11 quota units); after an outage it catches up to 36 h. Measured 2026-09-11: a
# fixed 4-page cap reached only 12-22 h on Zee Business / ET Now / CNBC-TV18.
POLL_LOOKBACK_MIN_H = 3
POLL_LOOKBACK_MAX_H = 36
API_MAX_PAGES = 10
# Video spend. Probe 2026-09-11 (Jindal Supreme CMD interview, ~17 min): the whole video
# cost 100,223 prompt tokens on gemini-3.7-flash, a 120 s clip 10,943, so clipping is
# honoured. 18 videos x at most 20 min = 6 h/day worst case, inside the free tier's 8 h.
VIDEO_BUDGET = 18                        # Gemini video calls per rolling 24 h
CLIP_MIN_DEFAULT = 20                    # watch at most the first N minutes of a video
NEWS_DAYS = 2
NAMES_NEWS_DAYS = 30
UNLABELLED_MAX_ATTEMPTS = 2              # classifier skipped it twice -> give up
CLASSIFY_CHUNK = 120
STATIC_LITE = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite"]
STATIC_MEDIA = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
                "gemini-2.5-flash"]
KEY_PREFIXES = "FREE_POOL,BACKFILL_GEMINI_KEY,GEMINI_API_KEY"   # CLAUDE.md default order
PROMPT_FILE = Path(_SCRIPTS_DIR) / "mgmt_interview_prompt.txt"
PREVIEW_FILE = Path(_SCRIPTS_DIR).parent / "mgmt_interviews_preview.html"  # gitignored

YT_FEED = "https://www.youtube.com/feeds/videos.xml"
YT_API = "https://www.googleapis.com/youtube/v3/playlistItems"
IST = timedelta(hours=5, minutes=30)
UTC_FMT_LEN = 19                          # 'YYYY-MM-DDTHH:MM:SS'

# Outlets whose interview write-ups count. The shared whitelist plus ET Now's own site,
# added HERE rather than in news_fetch.py so the live catalyst/fraud scans are untouched.
INTERVIEW_DOMAINS = frozenset(news_fetch.TRUSTED_DOMAINS | {"etnownews.com"})
NEWS_CUES = '(CEO OR MD OR CFO OR chairman OR founder OR "managing director" OR interview)'
NEWS_SWEEP_SITES = ("cnbctv18.com", "ndtvprofit.com", "etnownews.com", "moneycontrol.com",
                    "economictimes.indiatimes.com", "business-standard.com",
                    "thehindubusinessline.com", "livemint.com")

# A management voice in a title/description. Word-bounded, so "MD" never fires inside
# "AMD" and "ED" only as its own word.
MGMT_CUE = re.compile(
    r"(?i)\b(md|ceo|cfo|coo|cmd|cto|ed|chairman|chairperson|chairwoman|"
    r"managing director|whole[- ]?time director|executive director|founder|co-?founder|"
    r"promoter|management|mgmt|vice[- ]chairman|business head|in conversation)\b")
# Headlines only: a news story built on an interview names it as one.
NEWS_CUE = re.compile(r"(?i)\b(interview|exclusive|in conversation)\b")
# A channel's short cut of a longer interview (CNBC-TV18 tags these N18S); the full
# upload is preferred as the one that gets summarised.
SEGMENT_MARK = re.compile(r"(?i)\bN18S\b|#shorts|\bshorts\b")
# How media names a company when the name is not its NSE symbol. Resolves only to
# companies in scope; extend as the near-miss log shows new ones.
ALIASES = {
    "SBI": "SBIN", "STATE BANK": "SBIN", "HUL": "HINDUNILVR", "L&T": "LT",
    "LARSEN": "LT", "RIL": "RELIANCE", "RELIANCE": "RELIANCE", "HPCL": "HINDPETRO",
    "IOCL": "IOC", "INDIAN OIL": "IOC", "POLICYBAZAAR": "POLICYBZR", "ZOMATO": "ETERNAL",
    "MAHINDRA & MAHINDRA": "M&M", "M&M": "M&M", "INFOSYS": "INFY", "AIRTEL": "BHARTIARTL",
    "BHARTI AIRTEL": "BHARTIARTL", "HDFC AMC": "HDFCAMC", "ICICI PRU": "ICICIPRULI",
    "MARUTI SUZUKI": "MARUTI",
}
# Titles that are never a watchable management interview: live streams and market
# shows run for hours, and tip/wrap segments carry no management voice.
NOT_INTERVIEW = re.compile(
    r"(?i)(\blive\b|24x7|24/7|livestream|stocks? to (buy|watch|sell)|share market live|"
    r"market (open|close|wrap)|nifty (live|prediction)|stock tips|#shorts|\bshorts\b)")

CLASSIFY_PROMPT = (
    "You label videos and headlines from Indian business media.\n"
    "For EACH item decide: is it the MANAGEMENT of one specific company speaking about "
    "THAT company - an interview, an on-air conversation, or a news story built on what "
    "they told the media? Management = MD, CEO, CFO, COO, Chairman, founder, promoter, "
    "whole-time or executive director, or a named business head.\n"
    "NOT management: analysts, strategists, fund managers giving market views, brokers, "
    "economists, regulators, ministers, market wraps, stock tips, and news ABOUT a "
    "company in which its management does not speak.\n"
    "Return one compact JSON object per item, one per line, and nothing else - no prose, "
    "no code fences:\n"
    '{"id":"<item id>","mgmt":true|false,"company":"<company as named, else empty>",'
    '"person":"<speaker name, else empty>","role":"<speaker role, else empty>"}\n\n'
    "ITEMS (id | outlet | title | description):\n")

# Name words that never identify a company on their own (the shared news_fetch list,
# plus corporate suffixes that survive punctuation stripping).
_GENERIC = frozenset(set(news_fetch._GENERIC_NAME_TOKENS)
                     | {"ltd", "limited", "inc", "plc", "llp", "pvt"})


# ------------------------------------------------------------------ #
#  Time — stored UTC, shown IST (CLAUDE.md rule 8)                    #
# ------------------------------------------------------------------ #

def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).isoformat(timespec="seconds")


def to_utc_iso(s) -> str:
    """Feed/API timestamp ('...+00:00', '...Z', RFC-822) -> naive UTC ISO, '' if unparseable."""
    s = str(s or "").strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(s)
        except Exception:
            return ""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat(timespec="seconds")


def parse_utc(s) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s)[:UTC_FMT_LEN])
    except Exception:
        return None


def ist_label(utc_s) -> str:
    dt = parse_utc(utc_s)
    return (dt + IST).strftime("%d %b %H:%M IST") if dt else ""


def ist_date(utc_s) -> date | None:
    dt = parse_utc(utc_s)
    return (dt + IST).date() if dt else None


def daily_name(d: date) -> str:
    """mgmt_interviews_11_Sep2026.md — the {type}_{dd}_{mon}{yyyy}.md shape that
    app.py:load_daily_index parses into a date."""
    return f"{DAILY_PREFIX}_{d.strftime('%d')}_{d.strftime('%b')}{d.strftime('%Y')}.md"


# ------------------------------------------------------------------ #
#  YouTube discovery                                                 #
# ------------------------------------------------------------------ #

_ATOM = {"a": "http://www.w3.org/2005/Atom",
         "yt": "http://www.youtube.com/xml/schemas/2015",
         "media": "http://search.yahoo.com/mrss/"}


def _yt_item(vid: str, channel: str, title: str, desc: str, published) -> dict:
    return {"item_id": f"yt:{vid}", "source": "youtube", "channel": channel,
            "title": str(title or "").strip()[:300],
            "description": str(desc or "").strip()[:DESC_CHARS],
            "url": f"https://www.youtube.com/watch?v={vid}",
            "published_at": to_utc_iso(published)}


def parse_youtube_feed(data: bytes, channel: str) -> list[dict]:
    """Atom feed bytes -> items. [] on a malformed feed (the caller logs the channel)."""
    try:
        root = ET.fromstring(data)
    except Exception:
        return []
    out = []
    for e in root.findall("a:entry", _ATOM):
        vid = (e.findtext("yt:videoId", "", _ATOM) or "").strip()
        title = (e.findtext("a:title", "", _ATOM) or "").strip()
        desc = ""
        grp = e.find("media:group", _ATOM)
        if grp is not None:
            desc = grp.findtext("media:description", "", _ATOM) or ""
        if vid and title:
            out.append(_yt_item(vid, channel, title, desc,
                                e.findtext("a:published", "", _ATOM)))
    return out


def fetch_feed(channel_id: str, timeout: int = 20) -> tuple[bytes | None, str]:
    try:
        r = requests.get(YT_FEED, params={"channel_id": channel_id},
                         headers={"User-Agent": news_fetch.UA}, timeout=timeout)
    except Exception as e:
        return None, type(e).__name__
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    return r.content, ""


def parse_uploads_api(js: dict, channel: str) -> list[dict]:
    """playlistItems.list JSON -> items (optional key path)."""
    out = []
    for it in (js or {}).get("items", []) or []:
        sn = it.get("snippet") or {}
        cd = it.get("contentDetails") or {}
        vid = cd.get("videoId") or (sn.get("resourceId") or {}).get("videoId", "")
        title = str(sn.get("title") or "").strip()
        if not vid or not title or title in ("Private video", "Deleted video"):
            continue
        out.append(_yt_item(vid, channel, title, sn.get("description", ""),
                            cd.get("videoPublishedAt") or sn.get("publishedAt", "")))
    return out


def fetch_uploads_api(channel_id: str, channel: str, key: str, since: datetime,
                      max_pages: int = API_MAX_PAGES) -> tuple[list[dict], str]:
    """Every upload since `since` from the channel's uploads playlist ("UU" + id[2:]).
    1 quota unit per page of 50. The key travels only in params and is never logged."""
    items, token = [], ""
    for _ in range(max_pages):
        params = {"part": "snippet,contentDetails", "playlistId": "UU" + channel_id[2:],
                  "maxResults": 50, "key": key}
        if token:
            params["pageToken"] = token
        try:
            r = requests.get(YT_API, params=params, timeout=20)
        except Exception as e:
            return items, type(e).__name__
        if r.status_code != 200:
            return items, f"HTTP {r.status_code}"
        js = r.json()
        page = parse_uploads_api(js, channel)
        items += page
        dates = [d for d in (parse_utc(i["published_at"]) for i in page) if d]
        token = js.get("nextPageToken", "")
        if not token or (dates and min(dates) < since):
            break
    return [i for i in items if (parse_utc(i["published_at"]) or since) >= since], ""


def _span_hours(items: list[dict]) -> float | None:
    ds = [d for d in (parse_utc(i["published_at"]) for i in items) if d]
    return round((max(ds) - min(ds)).total_seconds() / 3600, 1) if len(ds) > 1 else None


def api_since(ledger: pd.DataFrame, now: datetime | None = None) -> datetime:
    """How far back the API poll reads: since the last new upload the ledger saw (+1 h),
    never less than POLL_LOOKBACK_MIN_H nor more than POLL_LOOKBACK_MAX_H."""
    now = now or utc_now()
    last = pd.to_datetime(ledger["first_seen_at"], errors="coerce").max() \
        if len(ledger) else pd.NaT
    if pd.isna(last):
        return now - timedelta(hours=POLL_LOOKBACK_MAX_H)
    gap_h = (now - last.to_pydatetime()).total_seconds() / 3600 + 1
    return now - timedelta(hours=min(POLL_LOOKBACK_MAX_H, max(POLL_LOOKBACK_MIN_H, gap_h)))


def fetch_all_channels(since: datetime) -> tuple[list[dict], list[tuple]]:
    """Latest uploads across CHANNELS -> (items, per-channel report rows). `since` only
    bounds the API path; RSS always returns its fixed 15."""
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    found, report = [], []
    for cid, name in CHANNELS:
        if key:
            items, err = fetch_uploads_api(cid, name, key, since)
        else:
            data, err = fetch_feed(cid)
            items = parse_youtube_feed(data, name) if data else []
            if data and not items and not err:
                err = "unparseable feed"
        report.append((name, len(items), _span_hours(items), err))
        found += items
    return found, report


# ------------------------------------------------------------------ #
#  Google News discovery (print interviews + TV write-ups)            #
# ------------------------------------------------------------------ #

def _clean_name(name: str) -> str:
    s = re.sub(r"(?i)\b(limited|ltd\.?)\s*$", "", str(name or "")).strip()
    return re.sub(r"(?i)\s*\((india)\)\s*", " ", s).strip()


def _strip_source(title: str, source: str) -> str:
    """Google News titles end ' - <outlet>'; drop it for display."""
    t = str(title or "").strip()
    if source and t.endswith(f" - {source}"):
        t = t[: -len(source) - 3].strip()
    return t


def _interview_domain(source_url: str) -> bool:
    d = news_fetch._domain(source_url)
    return any(d == t or d.endswith("." + t) for t in INTERVIEW_DOMAINS)


def _news_item(it: dict, hint_isin: str = "") -> dict:
    link = str(it.get("link") or "")
    src = str(it.get("source") or "") or news_fetch._domain(it.get("source_url"))
    return {"item_id": "gn:" + hashlib.sha1(link.encode("utf-8")).hexdigest()[:16],
            "source": "gnews", "channel": src,
            "title": _strip_source(it.get("title"), it.get("source"))[:300],
            "description": "", "url": link,
            "published_at": to_utc_iso(it.get("published")), "_hint_isin": hint_isin}


def news_candidates(targets: pd.DataFrame, days: int, sweep: bool) -> list[dict]:
    """Per-company queries for `targets` (PF, or the --names set) + optional outlet
    sweeps whose headlines are matched to the watchlist later. Best-effort: a failed
    or empty query contributes nothing."""
    out: list[dict] = []
    try:
        for r in targets.itertuples():
            name = _clean_name(r.name)
            if not name:
                continue
            items = news_fetch.fetch_news(f'"{name}" {NEWS_CUES}', days_back=days,
                                          trusted_only=False)
            items = [i for i in items if _interview_domain(i.get("source_url"))]
            for it in news_fetch.relevant_items(items, r.name, r.symbol):
                out.append(_news_item(it, hint_isin=r.isin))
        if sweep:
            for site in NEWS_SWEEP_SITES:
                for it in news_fetch.fetch_news(f"site:{site} {NEWS_CUES}", days_back=days,
                                                trusted_only=False):
                    if _interview_domain(it.get("source_url")):
                        out.append(_news_item(it))
    except news_fetch.NewsFetchBudgetExceeded as e:
        log(f"  news: {e} — remaining queries skipped")
    uniq: dict[str, dict] = {}
    for it in out:                       # per-company hit wins over a sweep duplicate
        if it["item_id"] not in uniq or it.get("_hint_isin"):
            uniq[it["item_id"]] = it
    return list(uniq.values())


# ------------------------------------------------------------------ #
#  Watchlist + company matching                                      #
# ------------------------------------------------------------------ #

# Every spelling a null takes across pandas/pyarrow versions, lower-cased.
_NULLISH = frozenset({"", "none", "nan", "nat", "<na>", "null"})


def is_null(v) -> bool:
    """True for every spelling of 'no value' pandas/pyarrow can hand back."""
    return _s(v).strip().lower() in _NULLISH


def _s(v) -> str:
    """str() that maps None/NaN to '' (a NaN name must not become the word 'nan')."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v)


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9&]+", _s(s).lower())


def name_tokens(name: str) -> list[str]:
    """Distinctive words of a company name (>=3 chars, not a generic suffix)."""
    return [w for w in _words(name) if len(w) >= 3 and w not in _GENERIC]


class Matcher:
    """Map a company name as spoken ('Tata Motors', 'Persistent') to ONE watchlist ISIN.

    Pass 1: every distinctive word of a watchlist name appears in the spoken name; the
            most specific such company wins (HDFC Bank beats Bank of India for 'HDFC
            Bank'). Pass 2: the spoken name holds a watchlist name's FIRST distinctive
            word and no other watchlist company starts with that word ('Dixon' ->
            Dixon Technologies; 'Tata' -> ambiguous -> None). Also an exact NSE symbol.
    A one-word identity ('Bank of India' -> 'bank') only matches when the spoken name
    adds no distinctive word of its own, so 'Yes Bank' is never filed as Bank of India;
    pass 2 carries the same guard.
    Ambiguity returns None: an interview filed under the wrong company is worse than an
    'unmatched' row someone can review."""

    def __init__(self, wl: pd.DataFrame):
        self.rows = []
        first_count: dict[str, int] = {}
        for r in wl.itertuples():
            toks = name_tokens(r.name)
            self.rows.append((_s(r.isin), _s(r.symbol).upper(), toks, set(_words(r.name))))
            if toks:
                first_count[toks[0]] = first_count.get(toks[0], 0) + 1
        self.unique_first = {t for t, n in first_count.items() if n == 1}
        self.by_symbol = {s: isin for isin, s, _, _ in self.rows if s}

    def match(self, company: str) -> str | None:
        words = set(_words(company))
        if not words:
            return None
        spoken = set(name_tokens(company))
        sym = re.sub(r"\s+", " ", _s(company)).strip().upper()
        sym = re.sub(r"\s+(LTD\.?|LIMITED)$", "", sym)
        if len(sym) >= 3 and sym in self.by_symbol:
            return self.by_symbol[sym]
        if ALIASES.get(sym) in self.by_symbol:
            return self.by_symbol[ALIASES[sym]]
        full = [(len(t), isin) for isin, _, t, allw in self.rows
                if t and all(w in words for w in t) and (len(t) > 1 or spoken <= allw)]
        if full:
            best = max(n for n, _ in full)
            top = {isin for n, isin in full if n == best}
            return top.pop() if len(top) == 1 else None
        # The spoken name is a PREFIX of the listed one: media say "Sri Lotus
        # Developers", the exchange lists "Sri Lotus Developers & Realty Ltd". Needs >= 2
        # distinctive words and exactly one candidate, so "Tata Motors" can never land on
        # "Tata Motors Finance" while both are listed. Surfaced by the near-miss log.
        if len(spoken) >= 2:
            sub = {isin for isin, _, t, _ in self.rows if t and spoken < set(t)}
            if len(sub) == 1:
                return sub.pop()
        first = {isin for isin, _, t, allw in self.rows
                 if t and t[0] in self.unique_first and t[0] in words and spoken <= allw}
        return first.pop() if len(first) == 1 else None

    def near_miss(self, company: str) -> str | None:
        """Symbol of a watchlist company whose FIRST distinctive word equals the spoken
        name's — the unmatched rows worth a human look ('HDFC Life' vs HDFC Bank, a
        missing alias, a renamed firm). Any-shared-word was tried first and measured as
        noise on 2026-09-11 ('Hero Motors ~ KETOMOTORS', 'HDFC Life ~ SUVEN')."""
        spoken = name_tokens(company)
        if not spoken:
            return None
        for isin, s, toks, _ in self.rows:
            if toks and toks[0] == spoken[0]:
                return s or isin
        return None

    def mentions(self, text: str, strict: bool = False) -> bool:
        """Does the text name a watchlist company? Loose (default, for the small --names
        set): the full name, the NSE symbol as a word, or a unique first word. Strict
        (market-wide sweeps): the full name (a one-word name must be >= 5 letters) or the
        symbol written in capitals, >= 4 letters. Measured 2026-09-11: media call MCX
        'MCX', never 'Multi Commodity Exchange', so a name-only rule missed its CEO
        interview; and a first-word rule let every 'Global Lens' video through."""
        words = set(_words(text))
        full = any(t and all(w in words for w in t) and (not strict or len(t) > 1
                                                           or len(t[0]) >= 5)
                   for _, _, t, _ in self.rows)
        caps = set(re.findall(r"[A-Z0-9&]+", _s(text)))
        sym = any(len(s) >= 3 and (s in caps if strict and len(s) >= 4 else
                                   (not strict and s.lower() in words))
                  for _, s, _, _ in self.rows)
        if full or sym or strict:
            return full or sym
        return any(t and t[0] in self.unique_first and t[0] in words
                   for _, _, t, _ in self.rows)


def _read_csv_from(drive, parent_id: str, folder: str, name: str) -> pd.DataFrame:
    from _extractor_base import get_or_create_subfolder, find_file, download_bytes
    fid = find_file(drive, get_or_create_subfolder(drive, parent_id, folder), name)
    if not fid:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid))).fillna("")


def load_watchlist(drive, root_id: str, top_n: int) -> pd.DataFrame:
    """PF ∪ top-N signal names -> [isin, symbol, name, in_pf, signal_rank].

    watchlist.build_watchlist supplies the signal half. Its PF half reads portfolio/
    only, but the holdings truth is the NEWEST file across pf_tracking/ AND portfolio/
    (_extractor_base.load_portfolio_isins; sync_pf.bat uploads to pf_tracking/). So PF
    membership comes from load_portfolio_isins and stale portfolio/-only names drop."""
    import watchlist as WLM
    from _extractor_base import load_portfolio_isins
    wl = WLM.build_watchlist(drive, top_n)
    pf = {str(i).strip().upper() for i in (load_portfolio_isins(drive, root_id) or set())}
    ml = _read_csv_from(drive, root_id, "universe", "master_list.csv")
    cols = ["isin", "symbol", "name", "in_pf", "signal_rank"]
    if wl.empty:
        wl = pd.DataFrame(columns=cols + ["in_signal"])
    wl["isin"] = wl["isin"].astype(str).str.strip().str.upper()
    wl["in_pf"] = wl["isin"].isin(pf)
    wl = wl[(wl.get("in_signal", False) == True) | wl["in_pf"]]          # noqa: E712
    missing = pf - set(wl["isin"])
    if missing and not ml.empty:
        m = ml[ml["isin"].astype(str).str.upper().isin(missing)]
        add = pd.DataFrame({"isin": m["isin"].astype(str).str.upper(),
                            "symbol": m["symbol"].astype(str), "name": m["name"].astype(str),
                            "in_pf": True, "signal_rank": None})
        wl = pd.concat([wl, add], ignore_index=True)
    unnamed = int((wl["name"].astype(str).str.strip() == "").sum())
    if unnamed:
        log(f"  watchlist: {unnamed} name(s) without a company name — cannot be matched")
    return wl[cols].drop_duplicates("isin").reset_index(drop=True)


def resolve_names(drive, root_id: str, tokens: list[str], pf: set) -> pd.DataFrame:
    """--names: ISIN, NSE symbol or name fragment -> master_list rows (any listed name,
    watchlist or not)."""
    ml = _read_csv_from(drive, root_id, "universe", "master_list.csv")
    if ml.empty:
        return pd.DataFrame(columns=["isin", "symbol", "name", "in_pf", "signal_rank"])
    hits = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        m = ml[(ml["isin"].astype(str).str.upper() == t.upper())
               | (ml["symbol"].astype(str).str.upper() == t.upper())]
        if m.empty:
            m = ml[ml["name"].astype(str).str.lower().str.contains(re.escape(t.lower()))]
        if m.empty:
            log(f"  --names: '{t}' not found in master_list")
        hits.append(m.head(3))
    if not hits:
        return pd.DataFrame(columns=["isin", "symbol", "name", "in_pf", "signal_rank"])
    out = pd.concat(hits).drop_duplicates("isin")
    return pd.DataFrame({"isin": out["isin"].astype(str).str.upper(),
                         "symbol": out["symbol"].astype(str), "name": out["name"].astype(str),
                         "in_pf": out["isin"].astype(str).str.upper().isin(pf),
                         "signal_rank": None}).reset_index(drop=True)


# ------------------------------------------------------------------ #
#  Classify + summarise                                              #
# ------------------------------------------------------------------ #

def prefilter(item: dict, matcher: Matcher, focused: bool) -> bool:
    """Free first cut before any LLM call.
      video             a management role word in the title or description
      per-company news  a role word or 'interview'/'exclusive' in the headline
      sweep news        that, AND a watchlist company named in the headline (strict)
      focused mode      (--names / --pf-only) the target must be NAMED; a role word is
                        not required, because a holding's interview is worth a classify
                        call even when the title only says "... on Q1 numbers"."""
    title = str(item.get("title", ""))
    text = f"{title} {item.get('description', '')}"
    if item.get("source") == "youtube":
        if NOT_INTERVIEW.search(title):
            return False
        if focused:
            return matcher.mentions(text)
        return bool(MGMT_CUE.search(text))
    cue = bool(MGMT_CUE.search(title) or NEWS_CUE.search(title))
    if item.get("_hint_isin") or focused:
        return cue
    return cue and matcher.mentions(title, strict=True)


def _one_line(s, n: int) -> str:
    return re.sub(r"[\s|]+", " ", str(s or "")).strip()[:n]


def build_classify_prompt(items: list[dict]) -> str:
    lines = [f"{it['item_id']} | {_one_line(it.get('channel'), 40)} | "
             f"{_one_line(it.get('title'), 200)} | {_one_line(it.get('description'), 300)}"
             for it in items]
    return CLASSIFY_PROMPT + "\n".join(lines)


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "yes", "1")


def parse_classification(text: str) -> dict[str, dict]:
    out = {}
    for o in salvage_json_objects(text):
        iid = str(o.get("id") or "").strip()
        if iid:
            out[iid] = {"mgmt": _truthy(o.get("mgmt")),
                        "company": sstr(o.get("company")) or "",
                        "person": sstr(o.get("person")) or "",
                        "role": sstr(o.get("role")) or ""}
    return out


def classify(pool, items: list[dict]) -> dict[str, dict]:
    """One LITE call per CLASSIFY_CHUNK items. A failed chunk leaves its items
    unlabelled; they are retried next run (attempts cap in apply_labels)."""
    from gemini_pool import AllBucketsExhausted, FatalCallError
    labels: dict[str, dict] = {}
    for i in range(0, len(items), CLASSIFY_CHUNK):
        batch = items[i:i + CLASSIFY_CHUNK]
        try:
            text, model = pool.call_text(build_classify_prompt(batch), max_output_tokens=8192)
        except (AllBucketsExhausted, FatalCallError) as e:
            log(f"  classify: chunk {i // CLASSIFY_CHUNK + 1} failed ({str(e)[:100]})")
            continue
        got = parse_classification(text)
        labels.update(got)
        log(f"  classify: chunk {i // CLASSIFY_CHUNK + 1}: {len(got)}/{len(batch)} labelled "
            f"({model})")
    return labels


def apply_labels(items: list[dict], labels: dict[str, dict], matcher: Matcher,
                 wl_by_isin: dict, now_s: str) -> dict[str, dict]:
    """item_id -> ledger fields, from classifier labels + the watchlist match."""
    out = {}
    for it in items:
        iid = it["item_id"]
        lab = labels.get(iid)
        if lab is None:
            n = int(it.get("attempts") or 0) + 1
            out[iid] = {"attempts": n}
            if n >= UNLABELLED_MAX_ATTEMPTS:
                out[iid].update(status="not_interview", processed_at=now_s,
                                last_error="classifier returned no label")
            continue
        if not lab["mgmt"]:
            out[iid] = {"status": "not_interview", "processed_at": now_s,
                        "company_name": lab["company"]}
            continue
        isin = matcher.match(lab["company"]) if lab["company"] else None
        # The per-company news query's company is used ONLY when the classifier names
        # no company. Measured 2026-09-11: the CG Power query returned an MSEDCL story
        # (the shared headline filter keeps any headline containing 'power'); the
        # classifier said MSEDCL, and a hint that overrode it filed the MSEDCL CMD's
        # remarks under CG Power.
        hint = it.get("_hint_isin") or ""
        if hint and not lab["company"] and hint in wl_by_isin:
            isin = hint
        w = wl_by_isin.get(isin) if isin else None
        if w is None:
            out[iid] = {"status": "unmatched", "processed_at": now_s,
                        "company_name": lab["company"], "person": lab["person"],
                        "role": lab["role"]}
            continue
        out[iid] = {"status": "matched", "isin": isin, "symbol": w["symbol"],
                    "company_name": w["name"], "in_pf": bool(w["in_pf"]),
                    "person": lab["person"], "role": lab["role"]}
    return out


def collapse_clips(videos: list[dict], done: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Several uploads of ONE interview -> one summary. Channels post the full talk and
    short cuts of it (2026-09-11, 36 h of uploads: five PhonePe clips, five Karix clips,
    four MobiKwik). Key = (company, speaker, IST day of upload). A group that already has
    a summarised row keeps it as the representative, so a late clip never re-spends the
    video budget; otherwise a non-segment title wins, then the earliest upload. Rows with
    no speaker name are never grouped.
    -> (videos to summarise, {duplicate item_id: representative item_id})"""
    def key(r):
        person = re.sub(r"[^a-z]", "", _s(r.get("person")).lower())
        return (_s(r.get("isin")), person, ist_date(r.get("published_at"))) if person else None

    groups: dict = {}
    for r in done:
        k = key(r)
        if k is not None:
            groups.setdefault(k, {"rep": r, "new": []})
    reps, dups = [], {}
    for r in videos:
        k = key(r)
        if k is None:
            reps.append(r)
        else:
            groups.setdefault(k, {"rep": None, "new": []})["new"].append(r)
    for g in groups.values():
        new = sorted(g["new"], key=lambda r: (1 if SEGMENT_MARK.search(_s(r.get("title"))) else 0,
                                             _s(r.get("published_at"))))
        rep = g["rep"]
        if rep is None and new:
            rep, new = new[0], new[1:]
            reps.append(rep)
        for d in new:
            dups[d["item_id"]] = rep["item_id"]
    return reps, dups


def fill_prompt(tmpl: str, **kw) -> str:
    for k, v in kw.items():
        tmpl = tmpl.replace("{{" + k + "}}", str(v or ""))
    return tmpl


def summarise_video(pool, row: dict, tmpl: str, clip_s: int | None) -> dict:
    """-> ledger fields. NOT_AN_INTERVIEW from the model demotes the row; a
    deterministic model failure falls back to the video's own description."""
    from gemini_pool import FatalCallError
    prompt = fill_prompt(tmpl, COMPANY=row.get("company_name"), SYMBOL=row.get("symbol"),
                         CHANNEL=row.get("channel"), TITLE=row.get("title"),
                         PUBLISHED=ist_label(row.get("published_at")))
    now_s = utc_iso()
    try:
        text, model = pool.call_video(row["url"], prompt, max_output_tokens=8192,
                                      end_offset_s=clip_s)
    except FatalCallError as e:
        desc = _one_line(row.get("description"), 300)
        return {"status": "done", "processed_at": now_s, "model": "fallback",
                "last_error": str(e)[:200],
                "summary": ("Summary unavailable: the model could not read this video."
                            + (f" From the video description: {desc}" if desc else ""))}
    text = (text or "").strip()
    if text.upper().startswith("NOT_AN_INTERVIEW"):
        return {"status": "not_interview", "processed_at": now_s, "model": model,
                "summary": text[:300]}
    return {"status": "done", "processed_at": now_s, "model": model, "summary": text[:3000]}


# ------------------------------------------------------------------ #
#  Ledger                                                            #
# ------------------------------------------------------------------ #

def load_ledger(drive, idx: str) -> pd.DataFrame:
    """The ledger, or an empty frame when it does not exist yet. A file that EXISTS but
    cannot be read raises — writing back an empty frame would wipe the history."""
    from _extractor_base import find_file, download_bytes
    fid = find_file(drive, idx, LEDGER_NAME)
    if not fid:
        return pd.DataFrame(columns=LEDGER_COLS)
    df = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    for c in LEDGER_COLS:
        if c not in df.columns:
            df[c] = None
    # Normalise on READ, not only on write: pandas/pyarrow hand an all-null text column
    # back as None here and as NaN or pd.NA on the CI runner, and a null test that knew
    # only one spelling silently changed behaviour there (2026-09-12: every recent row
    # counted against the video budget, and no row looked unmailed).
    return normalize_ledger(df[LEDGER_COLS])


def normalize_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """One dtype per column so pyarrow never meets a mixed column."""
    df = df.copy()
    for c in LEDGER_COLS:
        if c == "attempts":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        elif c == "in_pf":
            df[c] = df[c].map(lambda v: bool(v) if isinstance(v, (bool, int)) else
                              str(v).strip().lower() == "true")
        else:
            # is_null(), not a None check: pd.NA would otherwise be stored as the
            # literal string "<NA>" and later print as a speaker's name.
            df[c] = df[c].map(lambda v: None if is_null(v) else str(v))
    return df


def prune_ledger(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    seen = pd.to_datetime(df["first_seen_at"], errors="coerce")
    old = seen < (now - timedelta(days=CANDIDATE_KEEP_DAYS))
    junk = df["status"].astype(str).isin(["candidate", "not_interview", "unmatched"])
    return df[~(old & junk)].reset_index(drop=True)


def merge_ledger(ledger: pd.DataFrame, updates: dict[str, dict], new_rows: list[dict],
                 now: datetime) -> pd.DataFrame:
    """Pure: append rows whose item_id is new, THEN apply field updates by item_id (so a
    row created this run takes its final status in the same commit), then prune.
    Updates for ids that vanished (pruned meanwhile) are dropped, never re-created."""
    df = ledger[LEDGER_COLS].astype(object).reset_index(drop=True)
    add, have = [], set(df["item_id"].astype(str))
    for r in new_rows or []:
        iid = str(r.get("item_id"))
        if iid in have:
            continue
        have.add(iid)
        add.append({c: r.get(c) for c in LEDGER_COLS})
    if add:
        extra = pd.DataFrame(add, columns=LEDGER_COLS).astype(object)
        df = extra if df.empty else pd.concat([df, extra], ignore_index=True)
    pos = {iid: i for i, iid in enumerate(df["item_id"].astype(str))}
    for iid, fields in (updates or {}).items():
        i = pos.get(iid)
        if i is None:
            continue
        for k, v in fields.items():
            if k in LEDGER_COLS:
                df.at[i, k] = v
    return prune_ledger(normalize_ledger(df), now)


def commit_ledger(drive, idx: str, updates: dict, new_rows: list[dict],
                  tries: int = 6) -> bool:
    """Read-modify-write under interviews.lock. The ledger is RE-READ inside the lock,
    so a poll that appended while the daily run was summarising is never lost."""
    from _extractor_base import acquire_lock, release_lock, save_parquet
    if not updates and not new_rows:
        return True
    for _ in range(tries):
        if acquire_lock(drive, idx, LOCK_NAME, LOCK_OWNER, max_age_min=LOCK_MAX_AGE_MIN):
            try:
                merged = merge_ledger(load_ledger(drive, idx), updates, new_rows, utc_now())
                save_parquet(drive, idx, LEDGER_NAME, merged)
                return True
            finally:
                release_lock(drive, idx, LOCK_NAME)
        time.sleep(15)
    log(f"  ledger commit SKIPPED — {LOCK_NAME} stayed busy; the next run retries")
    return False


# ------------------------------------------------------------------ #
#  Rendering — Streamlit page + mail                                 #
# ------------------------------------------------------------------ #

def _order(rows: pd.DataFrame, rank: dict) -> pd.DataFrame:
    r = rows.copy()
    r["_pf"] = r["in_pf"].map(lambda v: 0 if v is True or str(v).lower() == "true" else 1)
    r["_rank"] = r["isin"].map(lambda i: rank.get(i, 10**6))
    r["_pub"] = pd.to_datetime(r["published_at"], errors="coerce")
    return r.sort_values(["_pf", "_rank", "_pub"], ascending=[True, True, False])


def _summary_lines(summary: str) -> list[str]:
    """Keep the labelled lines that carry content; NOT_STATED lines are noise here."""
    out = []
    for ln in _s(summary).splitlines():
        ln = ln.strip().lstrip("-* ").strip()
        if ln and "NOT_STATED" not in ln.upper():
            out.append(ln)
    return out[:8]


def _who(r) -> str:
    person = _s(r.get("person")).strip()
    role = _s(r.get("role")).strip()
    return f"{person} ({role})" if person and role else (person or role)


def company_blocks(rows: pd.DataFrame, rank: dict) -> list[dict]:
    """One block per company per IST day, PF first, then watchlist rank, newest first;
    inside a block the videos (with summaries) come before the headlines."""
    blocks: dict = {}
    for r in _order(rows, rank).to_dict("records"):
        k = (_s(r["isin"]), ist_date(r["published_at"]))
        b = blocks.setdefault(k, {"pf": r["_pf"] == 0, "symbol": _s(r["symbol"]),
                                  "name": _s(r["company_name"]), "items": []})
        b["items"].append(r)
    for b in blocks.values():
        b["items"].sort(key=lambda r: (0 if r["source"] == "youtube" else 1,
                                       _s(r["published_at"])))
    return list(blocks.values())


def _meta(r) -> str:
    return " · ".join(x for x in (_who(r), _s(r["channel"]), ist_label(r["published_at"])) if x)


def render_page_md(rows: pd.DataFrame, day: date, rank: dict,
                   clips: dict | None = None) -> str:
    """The Streamlit Daily Digests page. `clips` = {representative item_id: [duplicate
    rows]} — other uploads of the same interview, listed as extra links."""
    clips = clips or {}
    blocks = company_blocks(rows, rank)
    n_pf = sum(1 for b in blocks if b["pf"])
    out = [f"# Management interviews — {day.strftime('%d %b %Y')}",
           f"_{len(blocks)} compan(ies) with management on air or in print "
           f"({n_pf} in PF). Video notes are written by Gemini from the video itself; "
           "news items are headline-only. Links open the source._", ""]
    for label, want in (("Portfolio", True), ("Watchlist", False)):
        part = [b for b in blocks if b["pf"] == want]
        if not part:
            continue
        out += [f"## {label}", ""]
        for b in part:
            out += [f"### {b['symbol'] or b['name']} — {b['name']}", ""]
            for r in b["items"]:
                if r["source"] == "youtube":
                    out.append(f"- 🎥 {_meta(r)} — [{r['title']}]({r['url']})")
                    out += [f"  - {ln}" for ln in _summary_lines(r["summary"])]
                    more = clips.get(r["item_id"], [])
                    if more:
                        out.append("  - More clips: " + " · ".join(
                            f"[{i + 1}]({d['url']})" for i, d in enumerate(more)))
                else:
                    out.append(f"- 📰 {_meta(r)} — [{r['title']}]({r['url']})")
            out.append("")
    return "\n".join(out)


def render_mail_html(rows: pd.DataFrame, day: date, rank: dict, clips: dict | None = None,
                     scope: str = "", max_bytes: int = MAX_HTML_BYTES) -> tuple[str, str, list[str]]:
    """-> (subject, html, item_ids included). Whole company blocks are added until the
    body would pass max_bytes (Gmail clips at ~102 KB); the rest stay unmailed and go
    out the next evening."""
    from mailer import esc
    clips = clips or {}
    blocks = company_blocks(rows, rank)
    n_pf = sum(1 for b in blocks if b["pf"])
    n = len(blocks)
    if scope == "pf":
        subject = f"💼 PF mgmt interviews — {n} compan(ies) · {day.strftime('%d %b')}"
        title, lead = "Management interviews — your holdings", "Companies you hold. "
    elif scope == "watchlist":
        subject = f"🎙️ Watchlist mgmt interviews — {n} compan(ies) · {day.strftime('%d %b')}"
        title, lead = "Management interviews — watchlist", "Watchlist names you do NOT hold. "
    else:
        subject = (f"🎙️ Mgmt interviews — {n} compan(ies)"
                   + (f" (PF {n_pf})" if n_pf else "") + f" · {day.strftime('%d %b')}")
        title, lead = "Management interviews", "PF first, then watchlist. "
    head = ("<div style='font-family:Arial,sans-serif;font-size:14px;color:#222'>"
            f"<h2 style='margin:0 0 6px'>{esc(title, 60)} — {esc(day.strftime('%d %b %Y'))}</h2>"
            f"<p style='color:#666;margin:0 0 14px'>{lead}Video notes are "
            "written by Gemini from the video itself; news items are headline-only. Also on "
            "the dashboard: Company Intel → Daily Digests → Mgmt Interviews.</p>")
    tail = "</div>"
    body, ids, cur, sent = [], [], None, 0
    for b in blocks:
        part = []
        group = "Portfolio" if b["pf"] else "Watchlist"
        if group != cur and not scope:        # a single-scope mail needs no divider
            part.append(f"<h3 style='margin:18px 0 6px;color:#0b5394'>{group}</h3>")
        part.append("<div style='border-left:3px solid #0b5394;padding:4px 10px;margin:8px 0'>"
                    f"<b>{esc(b['symbol'] or b['name'], 30)}</b> — {esc(b['name'], 80)}"
                    "<ul style='margin:4px 0 0 18px;padding:0'>")
        shown = b["items"][:MAIL_ITEMS_PER_CO]
        for r in shown:
            icon = "🎥" if r["source"] == "youtube" else "📰"
            part.append(f"<li>{icon} <span style='color:#666;font-size:12px'>{esc(_meta(r), 160)}"
                        f"</span><br><a href='{esc(r['url'], 800)}'>{esc(r['title'], 200)}</a>")
            if r["source"] == "youtube":
                lines = _summary_lines(r["summary"])
                more = clips.get(r["item_id"], [])
                if more:
                    lines.append("More clips: " + " · ".join(
                        f"<a href='{esc(d['url'], 800)}'>{i + 1}</a>" for i, d in enumerate(more)))
                if lines:
                    part.append("<ul style='margin:2px 0 6px 16px;padding:0'>" + "".join(
                        f"<li>{ln if ln.startswith('More clips') else esc(ln, 400)}</li>"
                        for ln in lines) + "</ul>")
            part.append("</li>")
        if len(b["items"]) > len(shown):
            part.append(f"<li>+{len(b['items']) - len(shown)} more on the dashboard</li>")
        part.append("</ul></div>")
        size = len((head + "".join(body) + "".join(part) + tail).encode("utf-8"))
        if size > max_bytes - 600:
            break
        body += part
        ids += [str(r["item_id"]) for r in shown]
        cur, sent = group, sent + 1
    if sent < len(blocks):
        body.append(f"<p style='color:#a00'>…and {len(blocks) - sent} more compan(ies) on "
                    "the dashboard (mailed tomorrow if still new).</p>")
    return subject, head + "".join(body) + tail, ids


# ------------------------------------------------------------------ #
#  Drive context + daily page                                        #
# ------------------------------------------------------------------ #

def drive_ctx():
    from _extractor_base import get_drive, get_or_create_subfolder
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    return (drive, root, get_or_create_subfolder(drive, repo, "_index"),
            get_or_create_subfolder(drive, repo, "_daily"))


def write_daily_page(drive, daily_id: str, day: date, md: str) -> None:
    from _extractor_base import find_file, upload_bytes
    name = daily_name(day)
    upload_bytes(drive, daily_id, name, md.encode("utf-8"), "text/markdown",
                 existing_id=find_file(drive, daily_id, name))
    log(f"  page written -> company_repo/_daily/{name}")


def prune_daily_pages(drive, daily_id: str, today: date) -> None:
    """Delete OUR pages older than DAILY_KEEP_DAYS. Matches our prefix only, so other
    pipelines' _daily files are never touched (same shape as ingest_announcements)."""
    cutoff = today - timedelta(days=DAILY_KEEP_DAYS)
    try:
        files = drive.files().list(
            q=f"'{daily_id}' in parents and name contains '{DAILY_PREFIX}_' and trashed=false",
            fields="files(id,name)", pageSize=500).execute().get("files", [])
        for f in files:
            m = re.fullmatch(rf"{DAILY_PREFIX}_(\d{{2}})_([A-Za-z]{{3}})(\d{{4}})\.md", f["name"])
            if not m:
                continue
            try:
                d = datetime.strptime(" ".join(m.groups()), "%d %b %Y").date()
            except ValueError:
                continue
            if d < cutoff:
                drive.files().delete(fileId=f["id"]).execute()
    except Exception as e:
        log(f"  daily-page prune skipped ({str(e)[:60]})")


# ------------------------------------------------------------------ #
#  Modes                                                             #
# ------------------------------------------------------------------ #

def poll_once(drive, idx: str | None, dry_run: bool) -> dict:
    ledger = load_ledger(drive, idx) if drive is not None else pd.DataFrame(columns=LEDGER_COLS)
    since = api_since(ledger)
    items, report = fetch_all_channels(since)
    api = bool(os.environ.get("YOUTUBE_API_KEY", "").strip())
    mode = (f"API uploads since {ist_label(utc_iso(since))}" if api else "RSS (latest 15)")
    log(f"poll: {len(items)} upload(s) across {len(CHANNELS)} channels via {mode}")
    for name, n, span, err in report:
        if err:
            log(f"  {name:<22} FAILED ({err})")
        else:
            log(f"  {name:<22} {n:>3} item(s)" + (f" · window {span} h" if span is not None else ""))
    seen = set(ledger["item_id"].astype(str))
    now_s = utc_iso()
    new, batch = [], set()
    for it in items:
        if it["item_id"] in seen or it["item_id"] in batch:
            continue
        batch.add(it["item_id"])
        new.append(dict(it, status="candidate", first_seen_at=now_s, attempts=0))
    failed = [r[0] for r in report if r[3]]
    if dry_run or drive is None:
        log(f"  DRY RUN — would add {len(new)} new candidate(s); ledger has {len(ledger)} row(s)")
    elif new:
        commit_ledger(drive, idx, {}, new)
        log(f"  +{len(new)} candidate(s) -> {LEDGER_NAME}")
    else:
        log("  nothing new")
    return {"fetched": len(items), "new": len(new), "failed": failed, "new_items": new}


def poll_loop(loop_min: float, every_min: float) -> int:
    """CI: poll every `every_min` for `loop_min`. One long run instead of 48 short ones
    keeps check_infra_health's 40-most-recent-runs view on the other workflows."""
    end = time.monotonic() + loop_min * 60
    n = ok = 0
    while True:
        n += 1
        t0 = time.monotonic()
        try:
            drive, _root, idx, _daily = drive_ctx()
            s = poll_once(drive, idx, dry_run=False)
            ok += 1 if len(s["failed"]) < len(CHANNELS) else 0
        except Exception as e:
            log(f"poll #{n} failed: {type(e).__name__}: {str(e)[:160]}")
        nxt = t0 + every_min * 60
        if nxt >= end:
            break
        time.sleep(max(1.0, nxt - time.monotonic()))
    log(f"poll loop done: {ok}/{n} poll(s) reached YouTube and Drive")
    return 0 if ok else 1


def _make_pool(chain: str, static: list[str], drive, idx, call_timeout_s: float):
    from gemini_pool import BucketPool, load_keys_multi
    try:
        from model_registry import resolve
        models = resolve(chain, drive, idx) or list(static)
    except Exception as e:
        log(f"  model registry unavailable ({str(e)[:60]}) — static {chain} chain")
        models = list(static)
    keys = load_keys_multi(os.environ, KEY_PREFIXES)
    if not keys:
        log(f"  no Gemini keys ({KEY_PREFIXES}) — {chain} pool unavailable")
        return None
    return BucketPool(keys, models, inter_call_s=4.0, call_timeout_s=call_timeout_s,
                      logger=log)


def _video_used_24h(ledger: pd.DataFrame) -> int:
    t = pd.to_datetime(ledger["processed_at"], errors="coerce")
    recent = t >= (utc_now() - timedelta(hours=24))
    real = ~ledger["model"].map(lambda v: is_null(v) or _s(v).strip().lower() == "fallback")
    return int((recent & real & (ledger["source"] == "youtube")).sum())


def do_run(args) -> int:
    deadline = time.monotonic() + args.deadline_min * 60
    names_mode = bool(args.names)
    # focused = a specific target set (named companies, or the portfolio). It widens the
    # prefilter to "is my company named?" and never re-labels items outside that set.
    focused = names_mode or args.pf_only
    if args.dry_run:
        log("DRY RUN — no Drive writes, no mail, no video calls"
            + (" (one classify call allowed: --classify)" if args.classify else " (no LLM)"))
    drive, root, idx, daily_id = drive_ctx()

    # 1. one poll, so the run never works from a stale candidate list (a dry run holds
    #    the fresh uploads in memory instead of writing them)
    polled = poll_once(drive, idx, dry_run=args.dry_run)
    ledger = load_ledger(drive, idx)
    if args.dry_run and polled["new_items"]:
        ledger = merge_ledger(ledger, {}, polled["new_items"], utc_now())

    # 2. who we care about
    from _extractor_base import load_portfolio_isins
    if names_mode:
        pf = {str(i).upper() for i in (load_portfolio_isins(drive, root) or set())}
        wl = resolve_names(drive, root, args.names.split(","), pf)
    else:
        wl = load_watchlist(drive, root, args.top)
        if args.pf_only:
            wl = wl[wl["in_pf"].astype(bool)].reset_index(drop=True)
    if wl.empty:
        log("no target companies (watchlist empty / names not found) — stopping")
        return 1
    log(f"targets: {len(wl)} compan(ies), {int(wl['in_pf'].sum())} in PF")
    matcher = Matcher(wl)
    wl_by_isin = {r.isin: {"symbol": r.symbol, "name": r.name, "in_pf": bool(r.in_pf)}
                  for r in wl.itertuples()}
    rank = {r.isin: (r.signal_rank if pd.notna(r.signal_rank) else 10**5)
            for r in wl.itertuples()}

    # 3. Google News: per-company for PF (or the named set) + outlet sweeps
    news_targets = wl if focused else wl[wl["in_pf"].astype(bool)]
    news = news_candidates(news_targets, NAMES_NEWS_DAYS if names_mode else NEWS_DAYS,
                           sweep=not focused)
    seen = set(ledger["item_id"].astype(str))
    news = [n for n in news if n["item_id"] not in seen]
    log(f"news: {len(news)} new headline(s) ({news_fetch.calls_made()} RSS call(s))")

    # 4. the free prefilter
    cands = ledger[ledger["status"].astype(str) == "candidate"].to_dict("records")
    if focused:
        # Rows the cheap prefilter dropped were never shown to the classifier. For a
        # named set or the portfolio, reconsider them: a holding's interview can carry
        # no role word at all in its title.
        rej = ledger[(ledger["status"].astype(str) == "not_interview")
                     & (ledger["last_error"].astype(str) == "prefilter")]
        cands += rej.to_dict("records")
    pool_items, updates = [], {}
    now_s = utc_iso()
    for it in cands + news:
        if prefilter(it, matcher, focused):
            pool_items.append(it)
        elif not focused and it["source"] == "youtube":
            updates[it["item_id"]] = {"status": "not_interview", "processed_at": now_s,
                                      "last_error": "prefilter"}
    n_vid = sum(1 for it in pool_items if it["source"] == "youtube")
    log(f"prefilter: {n_vid} of {len(cands)} video(s) + {len(pool_items) - n_vid} of "
        f"{len(news)} headline(s) go to the classifier")

    # 5. classify (one LITE call per chunk)
    labels = {}
    if pool_items and (not args.dry_run or args.classify):
        lite = _make_pool("LITE_UTILITY", STATIC_LITE, drive, idx, call_timeout_s=180)
        if lite is not None:
            labels = classify(lite, pool_items)
    elif pool_items:
        for it in pool_items[:25]:
            log(f"    would classify: [{it['channel']}] {it['title'][:90]}")
    applied = apply_labels(pool_items, labels, matcher, wl_by_isin, now_s) if labels else {}
    for iid, f in applied.items():
        updates.setdefault(iid, {}).update(f)
    # Headlines are stored only once labelled, so tomorrow's run never re-classifies
    # them; an unlabelled one stays 'candidate' and gets its second chance.
    new_rows = []
    for n in news:
        if n["item_id"] not in applied:
            continue
        row = {k: v for k, v in n.items() if not k.startswith("_")}
        row.update(status="candidate", first_seen_at=now_s, attempts=0)
        row.update(applied[n["item_id"]])
        new_rows.append(row)
    counts: dict[str, int] = {}
    for f in applied.values():
        if f.get("status"):
            counts[f["status"]] = counts.get(f["status"], 0) + 1
    log(f"labels: {counts or 'none'}  (unmatched = management of a company outside PF + "
        "watchlist, or not a listed company)")
    # Only the unmatched names that share a word with a watchlist company can be a
    # matching failure (a missing alias, a renamed firm); the rest are simply out of scope.
    near = sorted({f"{f['company_name']} ~ {m}" for f in applied.values()
                   if f.get("status") == "unmatched"
                   and (m := matcher.near_miss(f.get("company_name", "")))})
    for n in near[:15]:
        log(f"    near-miss (review): {n}")

    # 6. summarise matched videos (budget + deadline); news hits are headline-only
    rows = {r["item_id"]: r for r in ledger.to_dict("records")}
    rows.update({r["item_id"]: r for r in new_rows})
    for iid, f in updates.items():
        if iid in rows:
            rows[iid] = dict(rows[iid], **f)
    matched = [r for r in rows.values() if r.get("status") == "matched"
               and (not focused or r.get("isin") in wl_by_isin)]
    for r in matched:
        if r["source"] != "youtube":
            updates.setdefault(r["item_id"], {}).update(status="done", processed_at=now_s)
            rows[r["item_id"]].update(status="done", processed_at=now_s)
    # one summary per interview: other uploads of it become 'duplicate' links
    reps, dups = collapse_clips(
        [r for r in matched if r["source"] == "youtube"],
        [r for r in rows.values() if r.get("status") == "done" and r.get("source") == "youtube"])
    for d, rep in dups.items():
        f = {"status": "duplicate", "dup_of": rep, "processed_at": now_s}
        updates.setdefault(d, {}).update(f)
        rows[d] = dict(rows[d], **f)
    if dups:
        log(f"clips: {len(dups)} extra upload(s) folded into {len(set(dups.values()))} interview(s)")
    videos = sorted(reps, key=lambda r: (0 if r.get("in_pf") in (True, "True") else 1,
                                         rank.get(r.get("isin"), 10**6),
                                         str(r.get("published_at") or "")))
    if args.resummarise_days:
        # Deliberate re-run of notes that already exist, after a prompt change. The rows
        # keep their place in the ledger; only summary and model are rewritten.
        since = utc_now() - timedelta(days=args.resummarise_days)
        redo = [r for r in rows.values()
                if r.get("status") == "done" and r.get("source") == "youtube"
                and (parse_utc(r.get("processed_at")) or datetime(1970, 1, 1)) >= since
                and (not focused or r.get("isin") in wl_by_isin)]
        have = {r["item_id"] for r in redo}
        videos = redo + [v for v in videos if v["item_id"] not in have]
        log(f"resummarise: {len(redo)} existing note(s) to regenerate")
    budget = max(0, VIDEO_BUDGET - _video_used_24h(ledger))
    if args.limit:
        videos = videos[: args.limit]
    log(f"videos to summarise: {len(videos)} (budget left today: {budget})")
    if videos and args.dry_run:
        for r in videos[:budget]:
            log(f"    would summarise: {r.get('symbol')} ({'PF' if r.get('in_pf') else 'WL'}) · "
                f"{_s(r.get('person'))} {_s(r.get('role'))} · {r['title'][:80]}")
    if videos and not args.dry_run:
        from gemini_pool import AllBucketsExhausted
        media = _make_pool("MEDIA", STATIC_MEDIA, drive, idx, call_timeout_s=300)
        tmpl = PROMPT_FILE.read_text(encoding="utf-8")
        clip_s = int(args.clip_min * 60) if args.clip_min else None
        if media is not None:
            media.probe_models()
        for r in videos[:budget] if media is not None else []:
            if time.monotonic() >= deadline:
                log("  deadline reached — remaining videos stay 'matched' for the next run")
                break
            try:
                f = summarise_video(media, r, tmpl, clip_s)
            except AllBucketsExhausted as e:
                log(f"  video pool exhausted ({str(e)[:80]}) — rest stay 'matched'")
                break
            f["attempts"] = int(r.get("attempts") or 0) + 1
            updates.setdefault(r["item_id"], {}).update(f)
            rows[r["item_id"]].update(f)
            log(f"  {f['status']:<13} {r.get('symbol')} · {r['title'][:70]} ({f.get('model')})")

    # 7. commit
    if args.dry_run:
        log(f"DRY RUN — would commit {len(updates)} update(s) + {len(new_rows)} new row(s)")
    else:
        commit_ledger(drive, idx, updates, new_rows)

    # 8. today's page (IST) + mail
    today = (utc_now() + IST).date()
    done = pd.DataFrame([r for r in rows.values() if r.get("status") == "done"],
                        columns=LEDGER_COLS)
    todays = done[done["processed_at"].map(lambda s: ist_date(s) == today)]
    clips: dict[str, list[dict]] = {}
    for r in rows.values():
        if r.get("status") == "duplicate" and r.get("dup_of"):
            clips.setdefault(str(r["dup_of"]), []).append(r)
    md = render_page_md(todays, today, rank, clips) if not todays.empty else ""
    proc = pd.to_datetime(done["processed_at"], errors="coerce")
    if args.resend_days:
        # deliberate one-off: mail what was already sent, e.g. the first portfolio-only
        # mail after the watchlist-wide one has gone out
        pick = proc >= (utc_now() - timedelta(days=args.resend_days))
    else:
        pick = done["mailed_at"].map(is_null) & (proc >= utc_now()
                                                 - timedelta(days=MAIL_MAX_AGE_DAYS))
    unmailed = done[pick]
    if focused:
        unmailed = unmailed[unmailed["isin"].isin(wl_by_isin)]
        for _, r in unmailed.iterrows():
            print(f"\n{r['symbol']} · {r['channel']} · {ist_label(r['published_at'])}\n"
                  f"{r['title']}\n{r['url']}\n" + "\n".join(_summary_lines(r["summary"])))
    # TWO mails: holdings and watchlist never share one (user 2026-09-12).
    in_pf = unmailed["in_pf"].astype(bool) if not unmailed.empty else pd.Series(dtype=bool)
    mails = [("pf", MAIL_KEY_PF, unmailed[in_pf] if not unmailed.empty else unmailed),
             ("watchlist", MAIL_KEY_WL, unmailed[~in_pf] if not unmailed.empty else unmailed)]
    built = [(scope, key, *render_mail_html(part, today, rank, clips, scope=scope))
             for scope, key, part in mails if not part.empty]
    if args.dry_run:
        PREVIEW_FILE.write_text(
            ("<hr>".join(h for _, _, _, h, _ in built) or "<p>(no unmailed interviews)</p>")
            + "<hr><pre style='white-space:pre-wrap'>"
            + (md or "(no page for today yet)").replace("<", "&lt;") + "</pre>",
            encoding="utf-8")
        log(f"DRY RUN — {len(built)} mail(s) + page preview -> {PREVIEW_FILE.name}: "
            + ", ".join(f"{sc}={len(i)}" for sc, _, _, _, i in built))
        return 0
    if md:
        write_daily_page(drive, daily_id, today, md)
    prune_daily_pages(drive, daily_id, today)
    if not built:
        log("  no new interviews to mail")
    elif not names_mode and not args.no_mail:
        from mailer import send_email, load_mail_settings
        settings = load_mail_settings(drive, idx)
        for scope, key, subject, html, ids in built:
            if not settings.get(key, True):
                log(f"  {scope} mail toggled OFF ('{key}') — {len(ids)} interview(s) held")
            elif send_email(subject, html):
                commit_ledger(drive, idx, {i: {"mailed_at": utc_iso()} for i in ids}, [])
    return 0


def probe_video(url: str, models: list[str], clip_min: float) -> int:
    """Stage-1 check: which models accept a YouTube URL with our keys, how long each
    takes, what it costs in prompt tokens, and whether clipping is honoured. Uses the
    pool's own client (a diagnostic only — production calls go through call_video)."""
    from gemini_pool import BucketPool, load_keys_multi, classify_error, KEY_DEAD
    from google.genai import types as T
    keys = load_keys_multi(os.environ, KEY_PREFIXES)
    if not keys:
        print(f"no keys under {KEY_PREFIXES}")
        return 1
    pool = BucketPool(keys, models, logger=log)
    ask = T.Part.from_text(text="In two sentences: who is speaking (name, role, company) "
                                "and what is the main topic?")
    cfg = T.GenerateContentConfig(temperature=0, max_output_tokens=2048)

    def one(m: str, clip: int | None) -> bool:
        fd = T.FileData(file_uri=url)
        vid = (T.Part(file_data=fd, video_metadata=T.VideoMetadata(end_offset=f"{clip}s"))
               if clip else T.Part(file_data=fd))
        label = f"clip {clip}s" if clip else "full"
        for ki in range(1, len(keys) + 1):
            t0 = time.time()
            try:
                r = pool._client(ki).models.generate_content(model=m, contents=[vid, ask],
                                                             config=cfg)
                um = getattr(r, "usage_metadata", None)
                print(f"OK    {m:<24} {label:<10} key{ki} {time.time() - t0:5.1f}s "
                      f"prompt_tokens={getattr(um, 'prompt_token_count', '?')} | "
                      f"{_one_line(r.text, 160)}")
                return True
            except Exception as e:
                kind, _ = classify_error(e)
                print(f"FAIL  {m:<24} {label:<10} key{ki} {kind}: {_one_line(e, 160)}")
                if kind != KEY_DEAD:
                    return False     # a model-level answer; a dead key just tries the next
        return False

    # Every model gets one SHORT clipped call (cheap acceptance test); only the first
    # model that accepts also watches the whole video, so the token difference shows
    # whether clipping is honoured - about len(models) + 1 calls in total.
    clip = int(clip_min * 60) if clip_min else None
    accepted = [m for m in models if one(m, clip)]
    if accepted and clip:
        one(accepted[0], None)
    print(f"\naccepted YouTube URLs: {accepted or 'NONE'}")
    return 0 if accepted else 1


# ------------------------------------------------------------------ #
#  Self-test (offline)                                               #
# ------------------------------------------------------------------ #

def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns:media="http://search.yahoo.com/mrss/" xmlns="http://www.w3.org/2005/Atom">
 <title>CNBC-TV18</title>
 <entry><yt:videoId>AAAAAAAAAA1</yt:videoId>
  <title>Tata Motors CFO PB Balaji On JLR Margins | CNBC TV18</title>
  <published>2026-09-11T13:57:08+00:00</published>
  <media:group><media:description>In conversation with the CFO of Tata Motors</media:description></media:group></entry>
 <entry><yt:videoId>AAAAAAAAAA2</yt:videoId>
  <title>LIVE: Stock Market Updates | Nifty &amp; Sensex</title>
  <published>2026-09-11T11:51:45+00:00</published></entry>
 <entry><yt:videoId>AAAAAAAAAA3</yt:videoId>
  <title>Persistent Systems eyes mid-teen growth after Nagarro deal</title>
  <published>2026-09-11T12:00:00Z</published></entry>
</feed>"""
    items = parse_youtube_feed(feed, "CNBC-TV18")
    check("feed: all 3 entries parsed", len(items) == 3)
    check("feed: id/url shape", items[0]["item_id"] == "yt:AAAAAAAAAA1"
          and items[0]["url"].endswith("v=AAAAAAAAAA1"))
    check("feed: description read", "CFO" in items[0]["description"])
    check("feed: '+00:00' -> naive UTC", items[0]["published_at"] == "2026-09-11T13:57:08")
    check("feed: 'Z' -> naive UTC", items[2]["published_at"] == "2026-09-11T12:00:00")
    check("feed: window span", _span_hours(items) == 2.1)
    check("feed: garbage -> []", parse_youtube_feed(b"<html>nope", "x") == [])
    check("rfc822 (Google News pubDate) -> UTC",
          to_utc_iso("Fri, 11 Sep 2026 09:08:25 GMT") == "2026-09-11T09:08:25")

    api = {"items": [
        {"snippet": {"title": "Dixon MD on capex", "description": "d",
                     "publishedAt": "2026-09-11T10:00:00Z"},
         "contentDetails": {"videoId": "BBBBBBBBBB1", "videoPublishedAt": "2026-09-11T09:59:00Z"}},
        {"snippet": {"title": "Private video"}, "contentDetails": {"videoId": "CCCCCCCCCC1"}}]}
    ap = parse_uploads_api(api, "ET Now")
    check("api: private video skipped", len(ap) == 1)
    check("api: videoPublishedAt preferred", ap[0]["published_at"] == "2026-09-11T09:59:00")
    t = datetime(2026, 9, 11, 12, 0, 0)

    def _led(*seen):
        return pd.DataFrame({"first_seen_at": list(seen)}, columns=LEDGER_COLS)
    check("look-back: empty ledger catches up the max",
          api_since(pd.DataFrame(columns=LEDGER_COLS), t) == t - timedelta(hours=36))
    check("look-back: routine poll floors at 3 h",
          api_since(_led("2026-09-11T11:30:00"), t) == t - timedelta(hours=3))
    check("look-back: 10 h outage reads 11 h",
          api_since(_led("2026-09-11T02:00:00"), t) == t - timedelta(hours=11))
    check("look-back: long outage caps at 36 h",
          api_since(_led("2026-09-01T00:00:00"), t) == t - timedelta(hours=36))

    check("cue: CFO", bool(MGMT_CUE.search("Tata Motors CFO PB Balaji On JLR")))
    check("cue: MD word-bounded (AMD is not MD)", not MGMT_CUE.search("AMD shares jump"))
    check("cue: 'ED' only as a word", not MGMT_CUE.search("Shares closed higher"))
    check("not-interview: LIVE", bool(NOT_INTERVIEW.search(items[1]["title"])))
    check("not-interview: stock tips",
          bool(NOT_INTERVIEW.search("Stocks to buy: Dixon, Tata Motors")))

    wl = pd.DataFrame({
        "isin": ["INE155A01022", "INE081A01020", "INE084A01016", "INE040A01034",
                 "INE262H01021", "INE935N01020", "INE467B01029"],
        "symbol": ["TATAMOTORS", "TATASTEEL", "BANKINDIA", "HDFCBANK", "PERSISTENT",
                   "DIXON", "TCS"],
        "name": ["Tata Motors Ltd", "Tata Steel Ltd", "Bank of India", "HDFC Bank Ltd",
                 "Persistent Systems Ltd", "Dixon Technologies (India) Ltd",
                 "Tata Consultancy Services Ltd"],
        "in_pf": [True, False, False, False, False, True, False],
        "signal_rank": [None, 3, 9, 1, 2, None, 4]})
    mt = Matcher(wl)
    check("match: full name", mt.match("Tata Motors") == "INE155A01022")
    check("match: most specific wins (HDFC Bank, not Bank of India)",
          mt.match("HDFC Bank") == "INE040A01034")
    check("match: Bank of India", mt.match("Bank of India") == "INE084A01016")
    check("match: unique first word (Dixon)", mt.match("Dixon") == "INE935N01020")
    check("match: shared first word is ambiguous (Tata)", mt.match("Tata") is None)
    check("match: exact NSE symbol", mt.match("TCS") == "INE467B01029")
    check("match: unknown company", mt.match("Pixxel Space") is None)
    check("match: one-word identity is guarded ('Yes Bank' is not Bank of India)",
          mt.match("Yes Bank") is None)
    check("match: first-word pass is guarded too ('Dixon Motors' is not Dixon)",
          mt.match("Dixon Motors") is None)
    check("match: NaN name never becomes the word 'nan'",
          Matcher(pd.DataFrame({"isin": ["X1"], "symbol": [float("nan")],
                                "name": [float("nan")]})).match("nan") is None)
    check("mentions: watchlist name in a title",
          mt.mentions("Persistent Systems eyes mid-teen growth"))
    check("mentions: nothing", not mt.mentions("BRICS summit live updates"))

    check("prefilter: LIVE video dropped", not prefilter(items[1], mt, False))
    check("prefilter: video with a role word passes", prefilter(items[0], mt, False))
    check("prefilter: video without a role word is dropped", not prefilter(items[2], mt, False))
    check("prefilter: --names needs the target named",
          not prefilter({"source": "youtube", "title": "ONDC CBO on lending",
                         "description": ""}, mt, True))
    gn = {"source": "gnews", "description": ""}
    check("prefilter: per-company headline with a role word passes",
          prefilter(dict(gn, title="Dixon MD says capex on track", _hint_isin="X"), mt, False))
    check("prefilter: per-company headline without a cue is dropped",
          not prefilter(dict(gn, title="Dixon shares rise 4%", _hint_isin="X"), mt, False))
    check("prefilter: sweep headline naming a watchlist company passes",
          prefilter(dict(gn, title="Tata Motors CFO on JLR demand"), mt, False))
    check("prefilter: sweep headline for a non-watchlist company is dropped",
          not prefilter(dict(gn, title="IndusInd Bank CEO on lending risks"), mt, False))
    glob = Matcher(pd.DataFrame({"isin": ["G1"], "symbol": ["GLOBALHEAL"],
                                 "name": ["Global Health Ltd"]}))
    check("mentions: strict rejects a lone common first word ('Global Lens')",
          not glob.mentions("BRICS summit | Global Lens", strict=True))
    check("mentions: loose still allows it (the --names path)",
          glob.mentions("BRICS summit | Global Lens"))
    mcx = Matcher(pd.DataFrame({"isin": ["INE745G01035"], "symbol": ["MCX"],
                                "name": ["Multi Commodity Exchange of India Limited"]}))
    check("mentions: the NSE symbol as a word finds MCX's CEO interview (--names MCX)",
          mcx.mentions("AI To Reshape Commodity Markets? MCX CEO On India's Next Big Opportunity"))
    idea = Matcher(pd.DataFrame({"isin": ["V1"], "symbol": ["IDEA"],
                                 "name": ["Vodafone Idea Limited"]}))
    check("mentions strict: a symbol that is a common word needs capitals",
          not idea.mentions("A great idea from the CEO", strict=True)
          and idea.mentions("IDEA CEO on tariff hikes", strict=True))
    check("summary: an empty (NaN) summary renders as nothing, never 'nan'",
          _summary_lines(float("nan")) == [])

    raw = ('{"id":"yt:AAAAAAAAAA1","mgmt":true,"company":"Tata Motors","person":"PB Balaji",'
           '"role":"CFO"}\n{"id":"yt:AAAAAAAAAA3","mgmt":"true","company":"Persistent Systems",'
           '"person":"","role":""}\n{"id":"yt:X","mgmt":tr')
    lab = parse_classification(raw)
    check("classify: truncated tail salvaged", len(lab) == 2)
    check("classify: 'true' string is truthy", lab["yt:AAAAAAAAAA3"]["mgmt"] is True)
    prompt = build_classify_prompt([dict(items[0], title="a | b\nc")])
    check("classify: pipes/newlines flattened in item lines", "a b c" in prompt)

    wlb = {r.isin: {"symbol": r.symbol, "name": r.name, "in_pf": bool(r.in_pf)}
           for r in wl.itertuples()}
    news_hint = {"item_id": "gn:1", "source": "gnews", "_hint_isin": "INE935N01020",
                 "attempts": 0}
    got = apply_labels(
        [items[0], items[2], {"item_id": "yt:Q", "attempts": 1}, news_hint,
         {"item_id": "yt:R", "attempts": 0}],
        {**lab, "gn:1": {"mgmt": True, "company": "", "person": "Atul Lall", "role": "MD"},
         "yt:R": {"mgmt": True, "company": "Pixxel", "person": "A", "role": "CEO"}},
        mt, wlb, "2026-09-11T14:00:00")
    check("apply: matched + PF flag", got["yt:AAAAAAAAAA1"]["status"] == "matched"
          and got["yt:AAAAAAAAAA1"]["in_pf"] is True)
    check("apply: unlabelled twice -> not_interview",
          got["yt:Q"]["status"] == "not_interview" and got["yt:Q"]["attempts"] == 2)
    check("apply: news hint used when company empty", got["gn:1"]["isin"] == "INE935N01020")
    check("apply: unknown company -> unmatched", got["yt:R"]["status"] == "unmatched")

    led = pd.DataFrame([
        {"item_id": "yt:old", "status": "not_interview", "first_seen_at": "2026-08-01T00:00:00"},
        {"item_id": "yt:keep", "status": "done", "first_seen_at": "2026-08-01T00:00:00"},
        {"item_id": "yt:a", "status": "candidate", "first_seen_at": "2026-09-11T10:00:00"},
    ], columns=LEDGER_COLS)
    merged = merge_ledger(led, {"yt:a": {"status": "matched", "attempts": 1},
                                "yt:gone": {"status": "done"}},
                          [{"item_id": "yt:a"}, {"item_id": "yt:b", "status": "candidate",
                                                 "first_seen_at": "2026-09-11T11:00:00"}],
                          datetime(2026, 9, 11, 15))
    ids = list(merged["item_id"])
    check("merge: old non-interview pruned", "yt:old" not in ids)
    check("merge: old DONE interview kept forever", "yt:keep" in ids)
    check("merge: update applied", merged.set_index("item_id").loc["yt:a", "status"] == "matched")
    check("merge: duplicate new row ignored", ids.count("yt:a") == 1)
    check("merge: vanished id never re-created", "yt:gone" not in ids)
    check("merge: new row appended", "yt:b" in ids)
    check("merge: attempts is int", str(merged["attempts"].dtype).startswith("int"))
    buf = io.BytesIO()
    merged.to_parquet(buf, index=False)
    check("merge: parquet round-trips", len(pd.read_parquet(io.BytesIO(buf.getvalue()))) == 3)
    m2 = merge_ledger(pd.DataFrame(columns=LEDGER_COLS), {"gn:n": {"status": "done"}},
                      [{"item_id": "gn:n", "status": "matched",
                        "first_seen_at": "2026-09-11T11:00:00"}], datetime(2026, 9, 11, 15))
    check("merge: a row created this run takes its final status in the same commit",
          list(m2["status"]) == ["done"])
    check("merge: empty ledger + new rows (no concat onto an empty frame)", len(m2) == 1)

    for null in (None, float("nan"), pd.NA, "", "  ", "None", "nan", "<NA>"):
        check(f"null spelling {null!r} reads as empty", is_null(null))
    check("a real value is not null", not is_null("gemini-3.7-flash"))
    nulls = pd.DataFrame([
        dict(item_id="yt:a", source="youtube", status="done", model="gemini-3.7-flash",
             processed_at=utc_iso(), mailed_at=None, person="A", role="CEO"),
        dict(item_id="yt:b", source="youtube", status="not_interview", model=pd.NA,
             processed_at=utc_iso(), mailed_at=pd.NA, person=pd.NA, role=pd.NA),
        dict(item_id="yt:c", source="youtube", status="done", model=float("nan"),
             processed_at=utc_iso(), mailed_at="2026-09-11T10:00:00", person=float("nan"),
             role=None),
    ], columns=LEDGER_COLS)
    norm = normalize_ledger(nulls)
    check("budget counts only rows a model actually ran on (CI counted 17 of 18)",
          _video_used_24h(norm) == 1 and _video_used_24h(nulls) == 1)
    check("a NaN/<NA> mailed_at still reads as unmailed",
          list(norm["mailed_at"].map(is_null)) == [True, True, False])
    check("a null speaker never prints the word 'nan'",
          _who(norm.iloc[1]) == "" and _who(nulls.iloc[2]) == "")
    check("daily name matches app.py's parser",
          re.match(r"^(.+)_(\d{2})_([a-z]{3})(\d{4})\.md$", daily_name(date(2026, 9, 11)),
                   re.IGNORECASE) is not None
          and daily_name(date(2026, 9, 11)) == "mgmt_interviews_11_Sep2026.md")
    check("ist label", ist_label("2026-09-11T13:30:00") == "11 Sep 19:00 IST")
    check("gnews title suffix stripped",
          _strip_source("X eyes growth - CNBC TV18", "CNBC TV18") == "X eyes growth")
    check("gnews id is stable", _news_item({"link": "u"})["item_id"]
          == _news_item({"link": "u"})["item_id"])

    summ = ("SPEAKER: PB Balaji, CFO, Tata Motors\nTOPIC: JLR margins\nGUIDANCE: NOT_STATED\n"
            "TONE: confident — order book")
    done = pd.DataFrame([dict(item_id=f"yt:{i}", source="youtube", channel="CNBC-TV18",
                              title=f"Interview {i}", url=f"https://www.youtube.com/watch?v={i}",
                              published_at="2026-09-11T10:00:00", status="done",
                              isin=f"INE{i:09d}", symbol=f"SYM{i}",
                              company_name=f"Company {i} Ltd", in_pf=bool(i % 2),
                              person="PB Balaji", role="CFO", summary=summ,
                              processed_at="2026-09-11T12:00:00")
                         for i in range(400)], columns=LEDGER_COLS)
    rank = {"INE000000002": 3}
    subj, html, inc = render_mail_html(done, date(2026, 9, 11), rank)
    check("mail: under the Gmail clip size", len(html.encode("utf-8")) <= MAX_HTML_BYTES)
    check("mail: overflow is reported, not dropped silently", 0 < len(inc) < 400
          and "more compan(ies) on the dashboard" in html)
    check("mail: NOT_STATED lines hidden", "NOT_STATED" not in html)
    check("mail: subject counts companies and PF", "400 compan(ies) (PF 200)" in subj)
    _, small, inc_small = render_mail_html(done.head(10), date(2026, 9, 11), rank)
    check("mail: PF section comes first",
          -1 < small.find(">Portfolio<") < small.find(">Watchlist<"))
    check("mail: a small day fits whole", len(inc_small) == 10
          and "on the dashboard (mailed" not in small)
    one_co = done.head(30).assign(isin="INE000000001", symbol="SYM1", in_pf=True)
    _, big1, inc1 = render_mail_html(one_co, date(2026, 9, 11), rank)
    check("mail: one busy company is capped, never blocks the mail",
          len(inc1) == MAIL_ITEMS_PER_CO and "+22 more on the dashboard" in big1)
    dup_row = {"item_id": "yt:d1", "url": "https://www.youtube.com/watch?v=d1"}
    md = render_page_md(done.head(4), date(2026, 9, 11), rank, clips={"yt:1": [dup_row]})
    check("page: PF before watchlist", md.find("## Portfolio") < md.find("## Watchlist"))
    check("page: link present", "(https://www.youtube.com/watch?v=1)" in md)
    check("page: other clips of the interview are linked",
          "More clips: [1](https://www.youtube.com/watch?v=d1)" in md)
    _, mclip, _ = render_mail_html(done.head(4), date(2026, 9, 11), rank,
                                   clips={"yt:1": [dup_row]})
    check("mail: other clips linked, not escaped",
          "More clips: <a href='https://www.youtube.com/watch?v=d1'>1</a>" in mclip)
    two = pd.concat([done.head(1).assign(isin="INE000000001", symbol="SYM1", in_pf=True,
                                         item_id="gn:a", source="gnews", title="CMD on growth"),
                     done.iloc[[1]]])
    check("page: a company's video and headline share one block",
          render_page_md(two, date(2026, 9, 11), rank).count("### SYM1") == 1)

    # clips of one interview -> one summary
    clip = lambda i, t, p, pub: {"item_id": i, "title": t, "person": p, "isin": "X1",
                                 "published_at": pub, "source": "youtube"}
    vids = [clip("yt:c1", "PhonePe's Nigam On UPI | N18S", "Sameer Nigam", "2026-09-11T09:00:00"),
            clip("yt:c2", "Sameer Nigam: Full Interview", "Sameer Nigam", "2026-09-11T10:00:00"),
            clip("yt:c3", "Nigam on lending | N18S", "Sameer Nigam", "2026-09-11T11:00:00"),
            clip("yt:c4", "Other exec", "Rahul X", "2026-09-11T11:00:00"),
            clip("yt:c5", "No speaker named", "", "2026-09-11T11:00:00")]
    reps, dups = collapse_clips(vids, [])
    check("clips: the non-segment upload is the one summarised",
          [r["item_id"] for r in reps if r["person"] == "Sameer Nigam"] == ["yt:c2"])
    check("clips: the segments point at it", dups == {"yt:c1": "yt:c2", "yt:c3": "yt:c2"})
    check("clips: a different speaker and an unnamed one stay separate",
          {"yt:c4", "yt:c5"} <= {r["item_id"] for r in reps})
    reps2, dups2 = collapse_clips([vids[2]], [dict(vids[1], status="done")])
    check("clips: a late clip joins an already-summarised interview (no new spend)",
          reps2 == [] and dups2 == {"yt:c3": "yt:c2"})

    # the CG Power / MSEDCL case: a named company beats the query's company
    hint_lab = {"gn:h": {"mgmt": True, "company": "MSEDCL", "person": "Lokesh Chandra",
                         "role": "CMD"}}
    got_h = apply_labels([{"item_id": "gn:h", "_hint_isin": "INE935N01020", "attempts": 0}],
                         hint_lab, mt, wlb, "2026-09-11T14:00:00")
    check("hint: never overrides a company the classifier named", got_h["gn:h"]["status"] == "unmatched")
    check("alias: SBI -> SBIN when in scope",
          Matcher(pd.DataFrame({"isin": ["S1"], "symbol": ["SBIN"],
                                "name": ["State Bank of India"]})).match("SBI") == "S1")
    check("alias: trailing 'Ltd' ignored for symbol match", mt.match("TCS Ltd") == "INE467B01029")
    lotus = Matcher(pd.DataFrame({
        "isin": ["L1", "T1", "T2"], "symbol": ["LOTUSDEV", "TATAMOTORS", "TMF"],
        "name": ["Sri Lotus Developers & Realty Ltd", "Tata Motors Ltd",
                 "Tata Motors Finance Ltd"]}))
    check("prefix: the spoken name is a prefix of the listed one",
          lotus.match("Sri Lotus Developers") == "L1")
    check("prefix: exact listed name still wins over the longer sibling",
          lotus.match("Tata Motors") == "T1")
    check("prefix: two possible longer names stay ambiguous",
          Matcher(pd.DataFrame({"isin": ["A", "B"], "symbol": ["X", "Y"],
                                "name": ["Sri Lotus Developers & Realty Ltd",
                                         "Sri Lotus Developers & Infra Ltd"]}))
          .match("Sri Lotus Developers") is None)
    check("prefix: one distinctive word is not enough to pick a longer name",
          Matcher(pd.DataFrame({"isin": ["R1", "R2"], "symbol": ["RPOWER", "RELINFRA"],
                                "name": ["Reliance Power Ltd",
                                         "Reliance Infrastructure Ltd"]}))
          .match("Reliance") is None)
    check("focused prefilter: a holding named without any role word still passes",
          prefilter({"source": "youtube", "title": "Dixon on Q1 numbers", "description": ""},
                    mt, True)
          and not prefilter({"source": "youtube", "title": "Dixon on Q1 numbers",
                             "description": ""}, mt, False))
    check("near-miss: same first word as a watchlist name",
          mt.near_miss("HDFC Life") == "HDFCBANK")
    check("near-miss: a shared LATER word is not a near-miss ('Hero Motors' vs Tata Motors)",
          mt.near_miss("Hero Motors") is None)
    check("near-miss: unrelated name is not a near-miss", mt.near_miss("PhonePe") is None)
    _, pf_html, pf_ids = render_mail_html(done.head(4), date(2026, 9, 11), rank, scope="pf")
    pf_subj, _, _ = render_mail_html(done.head(4), date(2026, 9, 11), rank, scope="pf")
    wl_subj, wl_html, _ = render_mail_html(done.head(4), date(2026, 9, 11), rank,
                                           scope="watchlist")
    check("mail: a PF-scoped mail says so and drops the divider",
          pf_subj.startswith("💼 PF mgmt interviews") and "your holdings" in pf_html
          and ">Portfolio<" not in pf_html)
    check("mail: a watchlist-scoped mail says so",
          wl_subj.startswith("🎙️ Watchlist mgmt interviews")
          and "do NOT hold" in wl_html)
    check("mail: default scope still shows both sections",
          ">Portfolio<" in render_mail_html(done.head(4), date(2026, 9, 11), rank)[1])
    check("prompt asks for growth guidance AND strategy",
          all(k in PROMPT_FILE.read_text(encoding="utf-8")
              for k in ("GROWTH & GUIDANCE:", "STRATEGY:")))
    check("prompt file present with all placeholders",
          PROMPT_FILE.exists() and all(
              "{{" + k + "}}" in PROMPT_FILE.read_text(encoding="utf-8")
              for k in ("COMPANY", "SYMBOL", "CHANNEL", "TITLE", "PUBLISHED")))
    check("channels: 11, no duplicates, no retired Bloomberg TV India",
          len(CHANNELS) == 11 and len({c for c, _ in CHANNELS}) == 11
          and "UCcC8Yq1ka5NbqDWHJqCnhpA" not in {c for c, _ in CHANNELS})

    print(f"\nfetch_mgmt_interviews self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


# ------------------------------------------------------------------ #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--poll", action="store_true", help="Append new uploads as candidates.")
    mode.add_argument("--run", action="store_true",
                      help="Classify, summarise, write the page, send the mail.")
    mode.add_argument("--probe-video", metavar="URL",
                      help="Which MEDIA models accept this public YouTube URL.")
    mode.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="No Drive writes, no mail, no video calls (reads allowed).")
    ap.add_argument("--classify", action="store_true",
                    help="With --run --dry-run: allow the one classifier call.")
    ap.add_argument("--loop-min", type=float, default=0,
                    help="--poll: keep polling for this many minutes (CI).")
    ap.add_argument("--every-min", type=float, default=30,
                    help="--poll: minutes between polls in a loop.")
    ap.add_argument("--resummarise-days", type=float, default=0,
                    help="--run: regenerate notes written within this many days "
                         "(use after changing the prompt).")
    ap.add_argument("--resend-days", type=float, default=0,
                    help="--run: also mail interviews already sent within this many days.")
    ap.add_argument("--pf-only", action="store_true",
                    help="--run: portfolio holdings only (no watchlist names).")
    ap.add_argument("--names", default="",
                    help="--run: comma-separated ISIN / NSE symbol / name fragment.")
    ap.add_argument("--limit", type=int, default=0,
                    help="--run: cap the videos summarised this run.")
    ap.add_argument("--top", type=int, default=500, help="Signal names in the watchlist.")
    ap.add_argument("--deadline-min", type=float, default=50,
                    help="--run: stop summarising after this many minutes.")
    ap.add_argument("--clip-min", type=float, default=CLIP_MIN_DEFAULT,
                    help="Watch at most the first N minutes of a video (0 = whole video).")
    ap.add_argument("--no-mail", action="store_true", help="--run: skip the mail.")
    ap.add_argument("--models", default=",".join(STATIC_MEDIA),
                    help="--probe-video: comma-separated models to try.")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.probe_video:
        return probe_video(args.probe_video,
                           [m.strip() for m in args.models.split(",") if m.strip()],
                           args.clip_min)
    if args.poll:
        if args.loop_min and not args.dry_run:
            return poll_loop(args.loop_min, args.every_min)
        try:
            drive, _root, idx, _daily = drive_ctx()
        except Exception as e:
            if not args.dry_run:
                raise
            log(f"  Drive unavailable ({type(e).__name__}) — dry run continues without the ledger")
            drive, idx = None, None
        poll_once(drive, idx, dry_run=args.dry_run)
        return 0
    return do_run(args)


if __name__ == "__main__":
    sys.exit(main())
