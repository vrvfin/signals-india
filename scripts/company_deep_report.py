r"""
company_deep_report.py  —  signals-india / Workflow B (OT7)

On-demand company deep dive. Drains deep_dive_queue.parquet (populated by Excel /
bat / Streamlit), and for each pending company:

  resolve name/NSE/BSE/ISIN -> ISIN (via universe)
   -> PHASE 0 rule-based coverage check (read company_page.md; what exists vs missing)
   -> assemble context from RELIABLE sources only:
        Screener  (fundamentals/summary.parquet + company_repo/_index/results.parquet)
        Company Page brief (company_repo/<ISIN>/company_page.md)
        Research Index filtered to this ISIN / sector / promoter (research_index.parquet)
        BSE announcements (BSE Direct API, best-effort, no auth)
   -> comapnydeepdive_prompt.txt  (single call; inputs are summaries, so it fits free tier)
   -> company_repo/<ISIN>/company_deepdive_DDMMMYY.md
   -> update deep_dive_index.parquet (Streamlit dropdown + last_update) and mark queue done

Runs in CI (deepdive.yml) or locally. Reuses the Drive helpers from the daily script.

Run:   python scripts/company_deep_report.py                  # drain the queue
       python scripts/company_deep_report.py --names "TCS,INFY"   # ad-hoc, no queue
       python scripts/company_deep_report.py --add "INE467B01029" # enqueue only

Env:   GEMINI_API_KEY (comma-separated allowed), GDRIVE_FOLDER_ID, GDRIVE_OAUTH_TOKEN_JSON
Deps:  google-generativeai pandas pyarrow requests + the Drive stack
"""
from __future__ import annotations
import os, io, re, sys, json, argparse, datetime as dt, tempfile, webbrowser

# Ensure scripts/ is on sys.path whether run as `python scripts/foo.py` (CI/root)
# or as `python foo.py` (local, already in scripts/)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests

from daily_research_summary import (drive_service, drive_download, drive_upload, _folder_id)
from gemini_pool import BucketPool, AllBucketsExhausted, FatalCallError, load_keys

SCRIPTS_DIR = _SCRIPTS_DIR
INTER_CALL_SLEEP = 6.0

# Best model first; pool only downgrades when current model is dead on ALL keys.
# 5 models × N keys = N*5 independent daily buckets.
DEEPDIVE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
]

DRIVE = dict(
    queue        = "company_repo/_index/deep_dive_queue.parquet",
    index        = "company_repo/_index/deep_dive_index.parquet",
    research_idx = "company_repo/_index/research_index.parquet",
    proc_queue   = "company_repo/_index/processing_queue.parquet",
    fundamentals = "fundamentals/summary.parquet",
    results      = "company_repo/_index/results.parquet",
    universe     = "universe/master_list.csv",
    company_page = "company_repo",   # /<ISIN>/company_page.md  &  output report
)

# context-size caps so the single call stays well inside the free-tier budget
MAX_PAGE_CHARS   = 30_000
MAX_RESEARCH_ROWS = 25

# ---- document-grounded assembly (Phase 2) ---------------------------------
# doc_type (from processing_queue) -> its dedicated extraction prompt
DOC_PROMPTS = {
    "concall":       "concall_prompt.txt",
    "annual_report": "annual_report_prompt.txt",
    "rating":        "rating_prompt.txt",
    "results":       "results_prompt.txt",
    "presentation":  "presentation_prompt.txt",
}
MAX_INLINE_PDF      = 18 * 1024 * 1024   # Gemini inline-data ceiling (~20MB)
MAX_DOC_TEXT_CHARS  = 80_000             # cap text extracted from html / per chunk
MAX_DOC_SUMMARY_CHARS = 6_000            # cap each per-doc summary fed to deep dive
DO_BACKFILL         = True               # pull full Screener doc history before a dive
# Chunking — long annual reports (financial-statement notes / RPT schedules sit at
# the BACK) are read shallowly in a single pass. Split into page-range chunks so
# every page is actually attended to, then merge the partials into one summary.
CHUNK_TRIGGER_PAGES = 45                 # only chunk docs longer than this
CHUNK_PAGES         = 35                 # pages per sub-PDF chunk

# --------------------------------------------------------------------------
def _read_parquet(svc, path, root):
    b = drive_download(svc, path, root)
    return pd.read_parquet(io.BytesIO(b)) if b else pd.DataFrame()

def _read_csv(svc, path, root):
    b = drive_download(svc, path, root)
    return pd.read_csv(io.BytesIO(b)) if b else pd.DataFrame()

def resolve_isin(token, universe, interactive=False):
    """token may be ISIN / NSE symbol / BSE code / name -> (isin, symbol, name, bse_code).

    Falls back to fuzzy (contains) name match. If multiple hits and interactive=True,
    prompts user to pick; otherwise returns first hit.
    """
    if universe is None or universe.empty:
        return (token, token, token, None)
    cols = {c.lower(): c for c in universe.columns}
    isin_c = cols.get("isin"); sym_c = cols.get("symbol") or cols.get("nse_symbol")
    name_c = cols.get("name") or cols.get("company") or cols.get("company_name")
    bse_c  = cols.get("bse_code") or cols.get("scrip_code") or cols.get("bsecode")
    t = str(token).strip()
    def row_out(r):
        raw_bse = r[bse_c] if bse_c else None
        if bse_c and pd.notna(raw_bse):
            s = str(raw_bse)
            bse_out = str(int(float(s))) if s.replace(".", "").isdigit() else s
        else:
            bse_out = None
        return (str(r[isin_c]) if isin_c else t,
                str(r[sym_c]) if sym_c else t,
                str(r[name_c]) if name_c else t,
                bse_out)
    # exact matches first
    if isin_c and t.upper().startswith("INE"):
        hit = universe[universe[isin_c].astype(str) == t.upper()]
        if not hit.empty: return row_out(hit.iloc[0])
    if sym_c:
        hit = universe[universe[sym_c].astype(str).str.upper() == t.upper()]
        if not hit.empty: return row_out(hit.iloc[0])
    if bse_c and t.isdigit():
        # bse_code is numeric in CSV so pandas reads it as float -> "522101.0"
        # normalise by converting to Int64 string before comparing
        bse_norm = universe[bse_c].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace(".", "").isdigit() else str(x))
        hit = universe[bse_norm == t]
        if not hit.empty: return row_out(hit.iloc[0])
    if name_c:
        hit = universe[universe[name_c].astype(str).str.lower() == t.lower()]
        if not hit.empty: return row_out(hit.iloc[0])
    # fuzzy fallback — partial name contains
    if name_c:
        fuzzy = universe[universe[name_c].astype(str).str.contains(t, case=False, na=False)]
        if not fuzzy.empty:
            if len(fuzzy) == 1 or not interactive:
                return row_out(fuzzy.iloc[0])
            print(f"\nMultiple matches for '{t}':")
            for i, (_, r) in enumerate(fuzzy.head(10).iterrows(), 1):
                n = str(r[name_c]) if name_c else "?"
                s = str(r[sym_c]) if sym_c else "?"
                print(f"  {i}. {n} ({s})")
            while True:
                try:
                    pick = int(input("Pick number: ").strip()) - 1
                    if 0 <= pick < min(len(fuzzy), 10):
                        return row_out(fuzzy.iloc[pick])
                except (ValueError, KeyboardInterrupt):
                    pass
                print("Invalid — enter a number from the list.")
    return (t, t, t, None)

# ---- PHASE 0: rule-based coverage check on company_page.md ----------------
def coverage_check(page_md: str) -> dict:
    if not page_md:
        return dict(has_page=False, ar_years=[], n_concall=0, n_rating=0,
                    n_presentation=0, n_research=0)
    return dict(
        has_page=True,
        ar_years=sorted(set(re.findall(r"FY\s?(\d{2})", page_md))),
        n_concall=len(re.findall(r"(?i)concall", page_md)),
        n_rating=len(re.findall(r"(?i)rating", page_md)),
        n_presentation=len(re.findall(r"(?i)presentation", page_md)),
        n_research=len(re.findall(r"research_\d{4}", page_md)),
    )

# ---- Source assemblers ----------------------------------------------------
def screener_block(fund, results, isin, symbol):
    out = []
    def pick(df):
        if df is None or df.empty: return None
        for col in ("isin", "ISIN"):
            if col in df.columns:
                r = df[df[col].astype(str) == isin]
                if not r.empty: return r
        for col in ("symbol", "Symbol", "nse_symbol"):
            if col in df.columns:
                r = df[df[col].astype(str).str.upper() == symbol.upper()]
                if not r.empty: return r
        return None
    f = pick(fund)
    if f is not None:
        out.append("FUNDAMENTALS (Screener):")
        out.append(f.iloc[0].dropna().astype(str).to_string())
    r = pick(results)
    if r is not None:
        out.append("\nRECENT RESULTS (Screener):")
        out.append(r.head(8).to_string(index=False))
    return "\n".join(out) if out else "DATA_MISSING"

# ---- Screener structured financials (independent cross-check source) -------
SCREENER_SECTIONS = [
    ("profit-loss",  "PROFIT & LOSS"),
    ("balance-sheet", "BALANCE SHEET"),
    ("cash-flow",    "CASH FLOW"),
    ("ratios",       "RATIOS"),
    ("quarters",     "QUARTERLY"),
    ("shareholding", "SHAREHOLDING"),
]

def _screener_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.screener.in/"})
    for part in os.environ.get("SCREENER_SESSION_COOKIE", "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            s.cookies.set(k.strip(), v.strip(), domain=".screener.in")
    return s

def _fmt_screener_table(table) -> str:
    from bs4 import BeautifulSoup  # noqa
    heads = [th.get_text(strip=True) for th in table.select("thead th")]
    period = " | ".join(h for h in heads[1:] if h)
    lines = [f"Period: {period}"]
    for tr in table.select("tbody tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.select("td")]
        if not cells:
            continue
        label = cells[0].replace("\xa0", " ").strip().rstrip("+")
        vals = " | ".join(cells[1:])
        lines.append(f"{label}: {vals}")
    return "\n".join(lines)

def _scrape_screener_financials(symbol: str) -> str | None:
    """Live-scrape the 6 Screener financial tables into a compact text block."""
    from bs4 import BeautifulSoup
    sess = _screener_session()
    for view in ("consolidated/", ""):
        url = f"https://www.screener.in/company/{symbol}/{view}"
        try:
            r = sess.get(url, timeout=30)
        except Exception:
            continue
        if r.status_code != 200 or 'id="profit-loss"' not in r.text:
            continue
        soup = BeautifulSoup(r.text, "lxml")
        blocks = [f"SCREENER STRUCTURED FINANCIALS ({'consolidated' if view else 'standalone'}) "
                  f"— fetched {dt.date.today().isoformat()}:"]
        got = False
        for sec_id, label in SCREENER_SECTIONS:
            sec = soup.find(id=sec_id)
            tbl = sec.find("table") if sec else None
            if tbl is None:
                continue
            blocks.append(f"\n== {label} ==\n{_fmt_screener_table(tbl)}")
            got = True
        return "\n".join(blocks) if got else None
    return None

def screener_financials_block(svc, root, isin, symbol) -> str:
    """Always try a LIVE Screener fetch; cache it to Drive on success; fall back
    to the cached copy if live fails (Screener down / no cookie / unreachable)."""
    cache_path = f"{DRIVE['company_page']}/{isin}/screener_financials.txt"
    live = _scrape_screener_financials(symbol)
    if live:
        try:
            drive_upload(svc, cache_path, root, live.encode("utf-8"), "text/plain")
        except Exception:
            pass
        return live
    cached = drive_download(svc, cache_path, root)
    if cached:
        return ("[STALE CACHE — live Screener fetch failed this run]\n"
                + cached.decode("utf-8", "ignore"))
    return "DATA_MISSING (Screener live fetch failed and no cache available)."

def research_block(ridx, isin, symbol, name):
    if ridx is None or ridx.empty:
        return "No external research context provided."
    def hit(row):
        blob = f"{row.get('isins','')}{row.get('companies','')}".lower()
        return isin.lower() in blob or symbol.lower() in blob or name.lower() in blob
    sel = ridx[ridx.apply(hit, axis=1)].tail(MAX_RESEARCH_ROWS)
    if sel.empty:
        return "No external research context provided."
    lines = []
    for _, r in sel.iterrows():
        lines.append(f"- [{r.get('doc_type','?')} | {r.get('source','?')} | "
                     f"{r.get('doc_date','NA')}] {str(r.get('file_name',''))}")
    return "\n".join(lines)

def bse_announcements(bse_code, limit=40):
    """Recent BSE corporate announcements via BSE Direct API.
    Uses AnnSubCategoryGetData (AnnGetData was retired -> 'No Record Found').
    The meaningful text is NEWSSUB (subject), not HEADLINE ('PDF enclosed')."""
    if not bse_code:
        return "DATA_MISSING (no BSE scrip code resolved)."
    try:
        url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        params = {"pageno": 1, "strCat": "-1", "subcategory": "-1",
                  "strPrevDate": "", "strToDate": "", "strSearch": "P",
                  "strScrip": str(bse_code), "strType": "C"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                   "Accept": "application/json, text/plain, */*",
                   "Referer": "https://www.bseindia.com/corporates/ann.html",
                   "Origin": "https://www.bseindia.com"}
        rsp = requests.get(url, params=params, headers=headers, timeout=30)
        rsp.raise_for_status()
        data = rsp.json()
        rows = data.get("Table", []) if isinstance(data, dict) else []
        if not rows:
            return "No announcements returned."
        out = []
        for r in rows[:limit]:
            subj = (r.get("NEWSSUB") or r.get("HEADLINE") or "").strip()
            cat = (r.get("CATEGORYNAME") or "").strip()
            tag = f" [{cat}]" if cat and cat.lower() not in subj.lower() else ""
            out.append(f"- {str(r.get('NEWS_DT',''))[:10]} | {subj[:160]}{tag}")
        return "\n".join(out)
    except Exception as e:
        return f"DATA_MISSING (BSE fetch failed: {type(e).__name__})"

# Reputable sources only (user-chosen whitelist: dailies + markets + wires,
# plus the Economic Times pharma/business verticals which are the same publisher).
NEWS_WHITELIST = (
    "economic times", "etmarkets", "etpharma", "express pharma",       # ET family
    "business standard", "mint", "livemint", "hindu businessline",
    "businessline", "financial express",                               # dailies
    "moneycontrol", "cnbc", "ndtv profit", "bq prime", "quint",        # markets
    "reuters", "press trust", "pti", "bloomberg",                      # wires
)

def nse_announcements(symbol, limit=20):
    """Best-effort NSE corporate announcements. NSE blocks datacenter IPs often
    (CI) and needs cookie bootstrap — failures are silent (returns '')."""
    try:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                          "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
        s.get("https://www.nseindia.com", timeout=12)               # bootstrap cookies
        r = s.get("https://www.nseindia.com/api/corporate-announcements",
                  params={"index": "equities", "symbol": symbol.upper()}, timeout=18)
        data = r.json()
        rows = data if isinstance(data, list) else data.get("data", [])
        out = []
        for a in rows[:limit]:
            raw = str(a.get("an_dt") or a.get("sort_date") or "").strip()
            d = raw.split(" ")[0]                       # "DD-Mon-YYYY HH:MM" -> date
            subj = (a.get("desc") or "").strip()
            detail = (a.get("attchmntText") or "").strip()
            line = f"- {d} | {subj}" + (f": {detail[:120]}" if detail else "")
            out.append(line)
        return "\n".join(out)
    except Exception:
        return ""

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
            return "DATA_MISSING (news fetch failed)."
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

# Trusted equity-research / business-media YouTube channels (edit to taste).
# Match is a case-insensitive substring of the channelTitle; the company's own
# official channel is always allowed in addition to these.
YOUTUBE_CHANNEL_WHITELIST = (
    # business media
    "cnbc-tv18", "cnbctv18", "et now", "etnow", "moneycontrol", "ndtv profit",
    "zee business", "bloomberg", "livemint", "mint", "business standard",
    "bqprime", "bq prime", "the economic times",
    # reputable research / analysis
    "soic", "finology", "rachana ranade", "equitymaster", "zerodha",
    "sahil bhadviya", "convexity", "valueinvesting", "sysematix", "smart sync",
)

def _fetch_yt_transcript(vid) -> str:
    """Fetch a transcript across youtube-transcript-api versions (>=1.0 uses an
    instance .fetch(); <1.0 used the classmethod .get_transcript())."""
    from youtube_transcript_api import YouTubeTranscriptApi
    langs = ["en", "en-IN", "hi"]
    try:                                  # new API (>= 1.0)
        fetched = YouTubeTranscriptApi().fetch(vid, languages=langs)
        try:
            return " ".join(s.text for s in fetched)
        except Exception:
            return " ".join(s["text"] for s in fetched.to_raw_data())
    except AttributeError:                # old API (< 1.0)
        segs = YouTubeTranscriptApi.get_transcript(vid, languages=langs)
        return " ".join(s["text"] for s in segs)

def _youtube_summary(svc, root, isin, pool, vid, channel, title, pub) -> str | None:
    """Transcript -> research summary, cached as a sidecar per video id."""
    sidecar = f"{DRIVE['company_page']}/{isin}/yt_summaries/{vid}.md"
    cached = drive_download(svc, sidecar, root)
    if cached:
        return cached.decode("utf-8", "ignore")
    try:
        text = _fetch_yt_transcript(vid)[:MAX_DOC_TEXT_CHARS]
    except Exception:
        return None                       # no transcript / blocked
    if len(text.strip()) < 200:
        return None
    prompt = (
        "Summarise this YouTube video transcript about a listed INDIAN company as "
        "concise EQUITY-RESEARCH notes: management guidance, key claims, numbers cited, "
        "risks, and sentiment. Be strictly factual; explicitly flag any promotional, "
        "speculative, or unverified 'tip/target' claims as [UNVERIFIED]. Treat this as a "
        "LOW-confidence external view, not audited fact.\n"
        f"Channel: {channel} | Title: {title} | Published: {pub}\n\nTRANSCRIPT:\n{text}")
    try:
        summ, _ = pool.call_text(prompt)
    except FatalCallError:
        return None
    try:
        drive_upload(svc, sidecar, root, summ.encode("utf-8"), "text/markdown")
    except Exception:
        pass
    return summ

def youtube_block(svc, root, isin, symbol, name, pool, max_videos=8, months=24):
    """Search YouTube for the company, keep videos from whitelisted/official
    channels, summarise their transcripts (cached). External LOW-confidence source."""
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        return "DATA_MISSING (no YouTube API key)."
    try:
        import urllib.parse
        after = (dt.datetime.utcnow() - dt.timedelta(days=months * 30)
                 ).strftime("%Y-%m-%dT%H:%M:%SZ")
        q = urllib.parse.quote(f"{name} {symbol}")
        u = ("https://www.googleapis.com/youtube/v3/search?part=snippet&type=video"
             f"&order=relevance&maxResults=25&publishedAfter={after}&q={q}"
             f"&relevanceLanguage=en&key={key}")
        r = requests.get(u, timeout=20)
        if r.status_code != 200:
            return f"DATA_MISSING (YouTube API {r.status_code})."
        name_tok = re.sub(r"[^a-z]", "", name.split()[0].lower())
        picked = []
        for it in r.json().get("items", []):
            sn = it.get("snippet", {})
            vid = it.get("id", {}).get("videoId", "")
            ch = sn.get("channelTitle", "")
            chl = ch.lower()
            official = (name_tok and name_tok in re.sub(r"[^a-z]", "", chl)) \
                or symbol.lower() in chl
            if not vid or not (official or any(w in chl for w in YOUTUBE_CHANNEL_WHITELIST)):
                continue
            picked.append((vid, ch, sn.get("title", ""), sn.get("publishedAt", "")[:10]))
            if len(picked) >= max_videos:
                break
        if not picked:
            return "No videos from whitelisted/official channels."
        out = []
        for vid, ch, title, pub in picked:
            summ = _youtube_summary(svc, root, isin, pool, vid, ch, title, pub)
            if summ:
                out.append(f"### [YouTube | {ch} | {pub}] {title}\n"
                           f"{summ.strip()[:MAX_DOC_SUMMARY_CHARS]}")
        if not out:
            return "Whitelisted videos found but no usable transcripts."
        return "\n\n".join(out)
    except Exception as e:
        return f"DATA_MISSING (YouTube fetch failed: {type(e).__name__})."

# ---- PHASE 2: document-grounded summarisation -----------------------------
def _download_file_id(svc, file_id: str) -> bytes | None:
    from googleapiclient.http import MediaIoBaseDownload
    try:
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=file_id))
        done = False
        while not done:
            _, done = dl.next_chunk()
        return buf.getvalue()
    except Exception as e:
        print(f"      file fetch failed ({type(e).__name__})")
        return None

def _load_doc_prompt(doc_type: str) -> str:
    fname = DOC_PROMPTS.get(doc_type, "research_doc_prompt.txt")
    path = os.path.join(SCRIPTS_DIR, fname)
    if not os.path.exists(path):       # fallback if a prompt file is missing
        path = os.path.join(SCRIPTS_DIR, "research_doc_prompt.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()

def _sidecar_path(isin: str, row) -> str:
    d = str(row["announcement_date"])[:10]
    return (f"{DRIVE['company_page']}/{isin}/doc_summaries/"
            f"{row['doc_type']}__{d}__{row['doc_id']}.md")

def _summarise_pdf_chunked(pool, prompt, pdf_bytes, label) -> str | None:
    """Summarise a PDF completely. Short docs -> single call_pdf. Long docs ->
    split into CHUNK_PAGES page-range sub-PDFs, summarise each (so every page is
    read), then merge the partials into one coherent summary. Guarantees the back
    of the annual report (notes, RPT schedules, auditor remarks) is covered."""
    import fitz
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = src.page_count
    # short enough -> one pass (best table fidelity)
    if n <= CHUNK_TRIGGER_PAGES and len(pdf_bytes) <= MAX_INLINE_PDF:
        src.close()
        try:
            return pool.call_pdf(pdf_bytes, prompt)[0]
        except FatalCallError as e:
            print(f"      {label}: FATAL ({str(e)[:70]}) — skip"); return None

    partials = []
    for start in range(0, n, CHUNK_PAGES):
        end = min(start + CHUNK_PAGES, n)
        cprompt = (prompt + f"\n\n>>> This is PAGES {start+1}-{end} of {n} of the "
                   f"{label}. Summarise THIS portion faithfully and completely; "
                   f"other passes cover the remaining pages. Keep ALL figures, "
                   f"tables, related-party items, and auditor notes verbatim.")
        try:
            sub = fitz.open(); sub.insert_pdf(src, from_page=start, to_page=end - 1)
            sub_bytes = sub.tobytes(); sub.close()
            if len(sub_bytes) <= MAX_INLINE_PDF:
                txt, _ = pool.call_pdf(sub_bytes, cprompt)
            else:
                page_text = "\n".join(src[p].get_text()
                                      for p in range(start, end))[:MAX_DOC_TEXT_CHARS]
                txt, _ = pool.call_text(cprompt + "\n\n=== TEXT ===\n" + page_text)
            partials.append(f"--- pages {start+1}-{end} ---\n{txt.strip()}")
            print(f"      {label}: chunk pages {start+1}-{end}/{n} ok")
        except FatalCallError as e:
            print(f"      {label}: chunk {start+1}-{end} FATAL ({str(e)[:50]}) — skip chunk")
    src.close()
    if not partials:
        return None
    if len(partials) == 1:
        return partials[0].split("\n", 1)[-1]

    merge_prompt = (prompt + "\n\nThe sections below are page-by-page summaries "
        f"covering the ENTIRE {label} (no page skipped). Synthesise them into ONE "
        "coherent document summary in the format above. Preserve every material "
        "financial figure, multi-year trend, related-party transaction, accounting "
        "policy change, and auditor/rating remark. Do not omit the back-of-report "
        "notes.\n\n=== PAGE-RANGE SUMMARIES ===\n" + "\n\n".join(partials))
    try:
        return pool.call_text(merge_prompt)[0]
    except FatalCallError:
        # merge failed — return the concatenated partials rather than nothing
        return "\n\n".join(partials)[:MAX_DOC_TEXT_CHARS]

def summarise_doc(svc, root, pool, isin, row) -> str | None:
    """Return a per-doc summary. Reuse cached sidecar if present, else fetch the
    raw file (PDF or extracted-HTML text) and run its doc-type prompt. Caches the
    result as a sidecar so future dives reuse it for free. AllBucketsExhausted
    propagates (caller stops); FatalCallError on one doc -> skip that doc."""
    sidecar = _sidecar_path(isin, row)
    cached = drive_download(svc, sidecar, root)
    if cached:
        return cached.decode("utf-8", "ignore")

    fid = str(row.get("drive_file_id") or "").strip()
    if not fid:
        return None
    data = _download_file_id(svc, fid)
    if not data:
        return None

    doc_type = str(row["doc_type"])
    prompt = _load_doc_prompt(doc_type)
    label = f"{doc_type} {str(row['announcement_date'])[:10]}"
    is_pdf = data[:5].startswith(b"%PDF")
    if is_pdf:
        # complete-read with page-range chunking for long reports
        summ = _summarise_pdf_chunked(pool, prompt, data, label)
        if not summ:
            return None
    else:
        text = data.decode("utf-8", "ignore")[:MAX_DOC_TEXT_CHARS]
        if len(text.strip()) < 100:
            print(f"      {label}: no extractable text — skip (needs OCR?)")
            return None
        try:
            summ, _ = pool.call_text(
                prompt + f"\n\n=== DOCUMENT CONTENT ({label}) ===\n{text}")
        except FatalCallError as e:
            print(f"      {label}: FATAL ({str(e)[:80]}) — skip")
            return None

    try:
        drive_upload(svc, sidecar, root, summ.encode("utf-8"), "text/markdown")
    except Exception:
        pass
    return summ

def assemble_doc_summaries(svc, root, pool, isin) -> tuple[str, list[dict]]:
    """Summarise every actual document for this ISIN (reuse-or-generate) and
    return (combined_block, used_docs). Docs already folded into company_page.md
    (status=done) are skipped — the COMPANY_PAGE_BRIEF already carries them."""
    q = _read_parquet(svc, DRIVE["proc_queue"], root)
    if q.empty or "isin" not in q.columns:
        return "DATA_MISSING (no document index).", []
    rows = q[(q["isin"].astype(str) == isin) &
             (q["status"].astype(str) != "download_failed")]
    if rows.empty:
        return "DATA_MISSING (no documents ingested for this company).", []

    blocks, used = [], []
    for _, r in rows.sort_values("announcement_date").iterrows():
        if str(r.get("status")) == "done":
            continue                       # already in COMPANY_PAGE_BRIEF
        summ = summarise_doc(svc, root, pool, isin, r)
        if not summ:
            continue
        d = str(r["announcement_date"])[:10]
        title = str(r.get("title", ""))[:90]
        blocks.append(f"### [{r['doc_type']} | {d} | {title}]\n"
                      f"{summ.strip()[:MAX_DOC_SUMMARY_CHARS]}")
        used.append({"doc_type": str(r["doc_type"]), "date": d,
                     "title": title, "doc_id": str(r["doc_id"])})
    if not blocks:
        return "DATA_MISSING (documents present but none summarisable).", []
    return "\n\n".join(blocks), used

# ---- prompt assembly ------------------------------------------------------
def fill_section(tpl, tag, content):
    # function replacement -> content is inserted literally (no \g/\1 backref
    # interpretation, which would crash on summaries containing backslashes).
    return re.sub(rf"\[{tag}\].*?\[/{tag}\]",
                  lambda m: f"[{tag}]\n{content}\n[/{tag}]",
                  tpl, flags=re.DOTALL)

def build_prompt(name, symbol, isin, screener, page, research, bse,
                 docs="DATA_MISSING", screener_cross="DATA_MISSING", news="DATA_MISSING",
                 youtube="DATA_MISSING"):
    tpl = open(os.path.join(SCRIPTS_DIR, "comapnydeepdive_prompt.txt"),
               encoding="utf-8").read()
    tpl = (tpl.replace("[COMPANY_NAME]", name)
              .replace("[NSE_SYMBOL]", symbol)
              .replace("[ISIN]", isin))
    tpl = fill_section(tpl, "SCREENER_FINANCIAL_DATA", screener)
    tpl = fill_section(tpl, "SCREENER_CROSSCHECK", screener_cross)
    tpl = fill_section(tpl, "COMPANY_PAGE_BRIEF", page[:MAX_PAGE_CHARS] or "DATA_MISSING")
    tpl = fill_section(tpl, "RESEARCH_INDEX_CONTEXT", research)
    tpl = fill_section(tpl, "DOCUMENT_SUMMARIES", docs)
    tpl = fill_section(tpl, "BSE_ANNOUNCEMENTS", bse)
    tpl = fill_section(tpl, "NEWS_CONTEXT", news)
    tpl = fill_section(tpl, "YOUTUBE_RESEARCH", youtube)
    tpl = fill_section(tpl, "RAW_ANNUAL_REPORTS_FOR_GAPS",
                       "None supplied. Rely on Screener + Document Summaries + Company Page.")
    return tpl

def _clean_report_md(md: str) -> str:
    """Strip raw template artifacts the model echoes: ====/---- divider bars,
    'END OF LAYER' separators, $$ LaTeX, and [BRACKET] labels used as headings.
    Keeps inline citations like [Annual Report FY25] and tokens [COMFORT]/[WARNING]."""
    out = []
    for ln in md.splitlines():
        s = ln.strip()
        # title wrapped in divider bars:  ==== TITLE ====  /  ---- TITLE ----
        m = re.match(r"^[=\-_]{3,}\s*(.+?)\s*[=\-_]{3,}$", s)
        if m:
            title = m.group(1).strip()
            if not title or re.search(r"END OF (LAYER|PHASE)", title, re.I):
                continue                       # drop pure dividers / END markers
            out.append(f"## {title}")
            continue
        if re.match(r"^[=_]{3,}$", s):         # pure ==== / ____ line -> drop
            continue
        if re.match(r"^-{4,}$", s):            # ---- (4+) -> markdown hr
            out.append("---"); continue
        out.append(ln)
    text = "\n".join(out)
    # $$ ... $$ block math -> remove (the weighted matrix table already shows it)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    # inline $...$ -> drop the $ delimiters
    text = re.sub(r"\$(?!\$)([^$\n]{1,120})\$", r"\1", text)
    # full-line [ALL CAPS LABEL] -> ### Label  (inline citations are safe)
    text = re.sub(r"(?m)^\s*\[([A-Z][A-Z &/]{3,})\]\s*$",
                  lambda m: f"### {m.group(1).title()}", text)
    # collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

# --------------------------------------------------------------------------
def process_one(svc, root, pool, universe, fund, results, ridx, token,
                interactive=False, do_backfill=DO_BACKFILL):
    isin, symbol, name, bse_code = resolve_isin(token, universe, interactive=interactive)
    print(f"  deep dive: {token} -> {name} ({symbol} / {isin})")

    # Phase 1 — pull full Screener document history (annual reports, ratings,
    # concalls) into processing_queue before we assemble. Best-effort; never fatal.
    if do_backfill and isin.startswith("INE"):
        try:
            from backfill_company_docs import backfill as _backfill
            c = _backfill(symbol, isin)
            print(f"    backfill: {c.get('downloaded',0)} new doc(s), "
                  f"{c.get('found',0)} on Screener")
        except Exception as e:
            print(f"    backfill skipped ({type(e).__name__}: {str(e)[:80]})")

    page_b = drive_download(svc, f"{DRIVE['company_page']}/{isin}/company_page.md", root)
    page = page_b.decode("utf-8") if page_b else ""
    cov = coverage_check(page)
    print(f"    coverage: page={cov['has_page']} ar={cov['ar_years']} "
          f"concall={cov['n_concall']} research={cov['n_research']}")

    # Phase 2 — summarise every actual document (reuse cached sidecar, else
    # run the doc-type prompt) into a provenance-tagged block.
    doc_block, used_docs = assemble_doc_summaries(svc, root, pool, isin)
    print(f"    documents: {len(used_docs)} summarised/reused")

    # Screener structured financials — LIVE fetch (cache to Drive), used as an
    # independent cross-check the model reconciles against the Annual Reports.
    screener_cross = screener_financials_block(svc, root, isin, symbol)
    print(f"    screener cross-check: "
          f"{'live' if not screener_cross.startswith(('DATA_MISSING','[STALE')) else screener_cross[:40]}")

    # Exchange announcements — BSE Direct + best-effort NSE.
    bse = bse_announcements(bse_code)
    nse = nse_announcements(symbol)
    exchange = bse if not nse else f"BSE:\n{bse}\n\nNSE:\n{nse}"
    # Recent news from reputable sources (headlines only).
    news = news_block(name, symbol)
    # YouTube research — whitelisted/official channels, transcript summaries (cached).
    youtube = youtube_block(svc, root, isin, symbol, name, pool)
    yt_ok = not youtube.startswith(("DATA_MISSING", "No videos", "Whitelisted"))
    print(f"    exchange: BSE={'ok' if not bse.startswith('DATA_MISSING') else 'miss'} "
          f"NSE={'ok' if nse else 'skip'} · news={'ok' if not news.startswith(('DATA_MISSING','No recent')) else 'none'} "
          f"· youtube={'ok' if yt_ok else 'none'}")

    prompt = build_prompt(name, symbol, isin,
                          screener_block(fund, results, isin, symbol),
                          page,
                          research_block(ridx, isin, symbol, name),
                          exchange,
                          docs=doc_block,
                          screener_cross=screener_cross,
                          news=news,
                          youtube=youtube)
    report, model_used = pool.call_text(prompt)
    report = _clean_report_md(report)

    stamp = dt.datetime.now().strftime("%d%b%y")
    out_path = f"{DRIVE['company_page']}/{isin}/company_deepdive_{stamp}.md"
    prov = ", ".join(f"{u['doc_type']}:{u['date']}" for u in used_docs) or "none"
    header = (f"# Deep Dive — {name} ({symbol} / {isin})\n"
              f"*Generated {dt.datetime.now():%Y-%m-%d %H:%M} · "
              f"coverage: AR{cov['ar_years']} concalls~{cov['n_concall']} "
              f"research~{cov['n_research']} · docs used: {len(used_docs)}*\n\n"
              f"*Documents fed: {prov}*\n\n---\n\n")
    full_md = header + report
    drive_upload(svc, out_path, root, full_md.encode("utf-8"), "text/markdown")
    print(f"    wrote {out_path}")

    # generate Word + PPT and upload alongside the .md
    base_path = out_path.rsplit(".", 1)[0]
    try:
        from format_deepdive_docx import md_to_docx
        docx_bytes = md_to_docx(full_md, name, symbol, isin,
                                coverage=str(cov.get("ar_years", "")))
        drive_upload(svc, base_path + ".docx", root, docx_bytes,
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        print(f"    wrote {base_path}.docx")
    except Exception as e:
        print(f"    docx skipped: {e}")
    try:
        from format_deepdive_pptx import md_to_pptx
        pptx_bytes = md_to_pptx(full_md, name, symbol, isin,
                                 coverage=str(cov.get("ar_years", "")))
        drive_upload(svc, base_path + ".pptx", root, pptx_bytes,
                     "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        print(f"    wrote {base_path}.pptx")
    except Exception as e:
        print(f"    pptx skipped: {e}")
    return dict(isin=isin, symbol=symbol, name=name, report_path=out_path,
                last_update=dt.datetime.now().isoformat(),
                coverage=json.dumps(cov),
                _report_md=full_md,   # internal — not persisted to parquet
                _slug=f"{symbol.lower()}_{dt.datetime.now().strftime('%d%b%y').lower()}")

def open_report_local(report_md: str, slug: str,
                      name: str = "", symbol: str = "", isin: str = ""):
    """Open a markdown report locally — Obsidian if available, else HTML in browser.
    Also saves .docx and .pptx to the local reports folder.
    """
    obsidian_vault = os.path.join(
        os.environ.get("OBSIDIAN_VAULT", r"D:\EMA_Screener\Obsidian"),
        "signals-india", "deepdive")
    local_dir = os.path.join(
        os.environ.get("REPORTS_DIR", r"D:\EMA_Screener\Reports\signals-india"),
        "deepdive")
    os.makedirs(local_dir, exist_ok=True)

    # save .md
    md_path = os.path.join(local_dir, f"{slug}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # save .docx
    try:
        from format_deepdive_docx import md_to_docx
        docx_bytes = md_to_docx(report_md, name, symbol, isin)
        docx_path = os.path.join(local_dir, f"{slug}.docx")
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)
        print(f"    saved docx: {docx_path}")
    except Exception as e:
        print(f"    docx local save skipped: {e}")

    # save .pptx
    try:
        from format_deepdive_pptx import md_to_pptx
        pptx_bytes = md_to_pptx(report_md, name, symbol, isin)
        pptx_path = os.path.join(local_dir, f"{slug}.pptx")
        with open(pptx_path, "wb") as f:
            f.write(pptx_bytes)
        print(f"    saved pptx: {pptx_path}")
    except Exception as e:
        print(f"    pptx local save skipped: {e}")

    # try Obsidian for .md
    try:
        os.makedirs(obsidian_vault, exist_ok=True)
        obs_path = os.path.join(obsidian_vault, f"{slug}.md")
        with open(obs_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"    saved to Obsidian: {obs_path}")
        return
    except Exception:
        pass

    # HTML fallback
    try:
        import markdown as md_lib
        html_body = md_lib.markdown(report_md, extensions=["tables", "fenced_code"])
    except ImportError:
        html_body = f"<pre>{report_md}</pre>"
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto;"
            f"line-height:1.6}}table{{border-collapse:collapse;width:100%}}"
            f"td,th{{border:1px solid #ccc;padding:6px 10px}}</style></head>"
            f"<body>{html_body}</body></html>")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                     prefix=f"deepdive_{slug}_", mode="w", encoding="utf-8")
    tmp.write(html); tmp.close()
    webbrowser.open(f"file://{tmp.name}")
    print(f"    opened in browser: {tmp.name}")


def update_index(svc, root, recs):
    idx = _read_parquet(svc, DRIVE["index"], root)
    new = pd.DataFrame(recs)
    if not idx.empty:
        idx = idx[~idx["isin"].isin(new["isin"])]      # keep latest per company
    idx = pd.concat([idx, new], ignore_index=True)
    buf = io.BytesIO(); idx.to_parquet(buf, index=False)
    drive_upload(svc, DRIVE["index"], root, buf.getvalue(), "application/octet-stream")
    print(f"  deep_dive_index updated ({len(idx)} companies).")

# --------------------------------------------------------------------------
def _strip_internal(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names",        help="comma-separated tokens, ad-hoc run (bypass queue)")
    ap.add_argument("--add",          help="comma-separated tokens, enqueue only (no processing)")
    ap.add_argument("--open",         action="store_true",
                    help="open report locally in Obsidian / browser after writing")
    ap.add_argument("--interactive",  action="store_true",
                    help="prompt to pick when fuzzy name match returns multiple results")
    ap.add_argument("--resolve-only", action="store_true",
                    help="resolve and print company name then exit (used by bat for confirmation)")
    ap.add_argument("--no-backfill", action="store_true",
                    help="skip pulling full Screener document history before the dive")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(SCRIPTS_DIR), ".env"))

    svc = drive_service(); root = os.environ["GDRIVE_FOLDER_ID"]

    if args.resolve_only and args.names:
        universe = _read_csv(svc, DRIVE["universe"], root)
        for t in [x.strip() for x in args.names.split(",") if x.strip()]:
            isin, symbol, name, _ = resolve_isin(t, universe, interactive=args.interactive)
            if isin == t and symbol == t:
                print(f"Could not resolve: {t}"); sys.exit(1)
            print(f"Resolved: {name} ({symbol} / {isin})")
        return

    if args.add:
        q = _read_parquet(svc, DRIVE["queue"], root)
        rows = [dict(token=t.strip(), status="pending",
                     added_at=dt.datetime.now().isoformat()) for t in args.add.split(",")]
        q = pd.concat([q, pd.DataFrame(rows)], ignore_index=True)
        buf = io.BytesIO(); q.to_parquet(buf, index=False)
        drive_upload(svc, DRIVE["queue"], root, buf.getvalue(), "application/octet-stream")
        print(f"Enqueued {len(rows)}."); return

    api_keys = load_keys(os.environ)
    if not api_keys:
        print("ERROR: no GEMINI_API_KEY or GEMINI_API_KEY_* found in .env")
        sys.exit(1)
    pool = BucketPool(api_keys, DEEPDIVE_MODELS, inter_call_s=INTER_CALL_SLEEP)
    print(f"Pool: {len(api_keys)} key(s) × {len(DEEPDIVE_MODELS)} model(s) "
          f"= {len(api_keys) * len(DEEPDIVE_MODELS)} daily buckets")

    universe = _read_csv(svc, DRIVE["universe"], root)
    fund     = _read_parquet(svc, DRIVE["fundamentals"], root)
    results  = _read_parquet(svc, DRIVE["results"], root)
    ridx     = _read_parquet(svc, DRIVE["research_idx"], root)

    if args.names:
        tokens = [t.strip() for t in args.names.split(",") if t.strip()]
        recs = []
        for t in tokens:
            try:
                recs.append(process_one(svc, root, pool, universe, fund, results, ridx, t,
                                        interactive=args.interactive,
                                        do_backfill=not args.no_backfill))
            except AllBucketsExhausted as exc:
                print(f"  All Gemini buckets exhausted — stopping. ({exc})")
                break
            except FatalCallError as exc:
                print(f"  Fatal error for '{t}' (bad prompt/auth) — skipping. ({exc})")
        if recs:
            if args.open:
                for r in recs:
                    open_report_local(r["_report_md"], r["_slug"],
                                      r.get("name",""), r.get("symbol",""), r.get("isin",""))
            update_index(svc, root, [_strip_internal(r) for r in recs])
        return

    queue = _read_parquet(svc, DRIVE["queue"], root)
    if queue.empty or "status" not in queue:
        print("Queue empty. Nothing to do."); return
    pending = queue[queue["status"] == "pending"]
    if pending.empty:
        print("No pending companies."); return

    recs = []
    for i in pending.index:
        try:
            rec = process_one(svc, root, pool, universe, fund, results, ridx,
                              queue.at[i, "token"], interactive=args.interactive,
                              do_backfill=not args.no_backfill)
            recs.append(rec)
            queue.at[i, "status"] = "done"
            queue.at[i, "done_at"] = dt.datetime.now().isoformat()
        except AllBucketsExhausted as exc:
            # Quota exhausted — leave remaining rows pending for next run
            print(f"  All Gemini buckets exhausted — stopping queue drain. ({exc})")
            break
        except FatalCallError as exc:
            print(f"  FATAL (this company): {str(exc)[:120]}")
            queue.at[i, "status"] = "error"
            queue.at[i, "error"] = str(exc)[:300]
        except Exception as e:
            print(f"    FAILED {queue.at[i,'token']}: {e}")
            queue.at[i, "status"] = "error"
            queue.at[i, "error"] = str(e)[:300]

    buf = io.BytesIO(); queue.to_parquet(buf, index=False)
    drive_upload(svc, DRIVE["queue"], root, buf.getvalue(), "application/octet-stream")
    if recs:
        if args.open:
            for r in recs:
                open_report_local(r["_report_md"], r["_slug"],
                                  r.get("name",""), r.get("symbol",""), r.get("isin",""))
        update_index(svc, root, [_strip_internal(r) for r in recs])
    print(f"Done. {len(recs)} report(s) generated.")


if __name__ == "__main__":
    main()
