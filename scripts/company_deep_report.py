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
import os, io, re, sys, json, time, argparse, datetime as dt, tempfile, webbrowser

# Ensure scripts/ is on sys.path whether run as `python scripts/foo.py` (CI/root)
# or as `python foo.py` (local, already in scripts/)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests

from daily_research_summary import (drive_service, drive_download, drive_upload,
                                    drive_find, _folder_id)

# gemini_pool pulls in google-genai. The queue/resolve paths (used by Streamlit to
# enqueue a company) do NOT need it — only the actual dive does, which runs in CI.
# Guard the import so importing this module works in environments without
# google-genai (e.g. Streamlit Cloud): "cannot import name 'genai' from 'google'".
try:
    from gemini_pool import (BucketPool, AllBucketsExhausted, FatalCallError,
                             load_keys, load_keys_multi)
except Exception:  # google-genai not installed here
    BucketPool = None
    load_keys = None
    load_keys_multi = None

    class AllBucketsExhausted(Exception):
        pass

    class FatalCallError(Exception):
        pass


class DeadlineReached(Exception):
    """Raised inside a company's doc-summarisation when the wall-clock deadline is
    hit, so the run exits cleanly (queue + sidecars flushed) and the company stays
    pending to resume next run from its cached chunk/doc sidecars."""
    pass

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
    # FULL listed universe (NSE main + NSE Emerge SME + BSE, with bse_code) built by
    # build_company_universe.py. Use this, NOT universe/master_list.csv, which Phase 1
    # (build_universe.py) overwrites DAILY with an NSE-only list (no bse_code, no SME).
    universe     = "company_repo/_index/company_universe.csv",
    universe_fallback = "universe/master_list.csv",
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
# Completeness (user 2026-06-22): a `done` AR/concall whose stored extraction
# (response_chars in quarterly_facts) is below these is a partial/thin read — the deep
# dive re-summarises it IN FULL rather than trusting the thin company_page section. The
# rich summary is cached as a sidecar, so this is a one-time cost per doc (anti-loop).
THIN_RESPONSE_CHARS = {"annual_report": 8_000, "concall": 5_500}
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

def _load_universe(svc, root):
    """Full listed universe (NSE+SME+BSE w/ bse_code); fall back to the NSE-only
    master_list if the full file is missing."""
    uni = _read_csv(svc, DRIVE["universe"], root)
    if uni is None or uni.empty:
        uni = _read_csv(svc, DRIVE["universe_fallback"], root)
    return uni

# ── deep_dive_queue: ONE coordinated writer (Streamlit-safe lock + dedup) ───────
# deep_dive_queue is written by several processes (Streamlit "Add to Queue", --add,
# the drain). Without coordination, writes clobbered each other (a manual clean got
# overwritten) and the same token was processed 2-3x. These helpers serialize every
# read-modify-write behind a best-effort Drive-file lock AND dedup on every write, so
# a lost lock can never corrupt the queue (dedup is the guarantee; the lock just
# reduces lost-updates). No google-genai dependency → app.py (Streamlit) can use them.
QUEUE_LOCK_PATH = "company_repo/_index/_deep_dive_queue.lock"
QUEUE_COLS = ["token", "status", "added_at", "done_at", "error"]

def _safe_err(e) -> str:
    """Scrub anything resembling an API key before a message is STORED in the queue.
    The earlier key leak happened because raw str(exception) (containing the env key
    blob) was written to the error column."""
    s = f"{type(e).__name__}: {e}"
    s = re.sub(r"AIza[0-9A-Za-z_\-]{16,}", "[REDACTED_KEY]", s)
    s = re.sub(r"AQ\.[0-9A-Za-z_\-]{16,}", "[REDACTED_KEY]", s)
    s = re.sub(r"(GEMINI_API_KEY|FREE_POOL|BACKFILL_GEMINI_KEY)[0-9A-Za-z_=\-]*",
               "[REDACTED]", s)
    return s[:200]

def _dedup_queue(df):
    """One pending row per token; collapse duplicate done (keep latest done_at); a
    token that is done is not also left pending. Preserves other statuses."""
    if df is None or df.empty or "token" not in df.columns:
        return df if df is not None else pd.DataFrame(columns=QUEUE_COLS)
    df = df.copy()
    df["status"] = df["status"].astype(str)
    done = df[df["status"] == "done"]
    if "done_at" in done.columns:
        done = done.sort_values("done_at").drop_duplicates("token", keep="last")
    else:
        done = done.drop_duplicates("token", keep="last")
    done_tokens = set(done["token"].astype(str))
    pend = df[df["status"] == "pending"].drop_duplicates("token", keep="first")
    pend = pend[~pend["token"].astype(str).isin(done_tokens)]
    other = df[~df["status"].isin(["pending", "done"])]
    return pd.concat([done, pend, other], ignore_index=True)

def _acquire_queue_lock(svc, root, owner="deepdive", max_age_s=600, wait_s=90, poll_s=3):
    """Best-effort Drive-file lock for deep_dive_queue. Steals a lock older than
    max_age_s (a crashed holder can't starve forever). Polls up to wait_s."""
    deadline = time.monotonic() + wait_s
    while True:
        cur = drive_download(svc, QUEUE_LOCK_PATH, root)
        fresh = False
        if cur:
            try:
                meta = json.loads(cur.decode("utf-8", "ignore"))
                fresh = (time.time() - float(meta.get("ts", 0))) < max_age_s
            except Exception:
                fresh = False
        if not fresh:
            drive_upload(svc, QUEUE_LOCK_PATH, root,
                         json.dumps({"owner": owner, "ts": time.time()}).encode(),
                         "application/json")
            chk = drive_download(svc, QUEUE_LOCK_PATH, root)
            try:
                if chk and json.loads(chk.decode("utf-8", "ignore")).get("owner") == owner:
                    return True
            except Exception:
                pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)

def _release_queue_lock(svc, root):
    try:
        fid = drive_find(svc, QUEUE_LOCK_PATH, root)
        if fid:
            svc.files().delete(fileId=fid).execute()
    except Exception:
        pass

def queue_update(svc, root, mutate, owner="deepdive"):
    """Locked read-modify-write of deep_dive_queue: acquire lock, RE-READ the current
    queue from Drive, apply mutate(df)->df, dedup, write, release. Re-reading under the
    lock is what prevents a concurrent enqueue from being lost."""
    got = _acquire_queue_lock(svc, root, owner=owner)
    try:
        df = _read_parquet(svc, DRIVE["queue"], root)
        if df is None or df.empty:
            df = pd.DataFrame(columns=QUEUE_COLS)
        df = _dedup_queue(mutate(df))
        buf = io.BytesIO(); df.to_parquet(buf, index=False)
        drive_upload(svc, DRIVE["queue"], root, buf.getvalue(), "application/octet-stream")
        return df
    finally:
        if got:
            _release_queue_lock(svc, root)

def enqueue_tokens(svc, root, tokens, owner="streamlit") -> int:
    """Add pending rows for tokens, skipping any already pending OR done (locked +
    deduped). Returns how many were actually added. Use this from EVERY enqueue path
    (Streamlit, --add, synthesise) so the queue can never be clobbered or duplicated."""
    toks = list(dict.fromkeys(str(t).strip() for t in tokens if str(t).strip()))
    added = {"n": 0}
    def m(df):
        seen = (set(df[df["status"].astype(str).isin(["pending", "done"])]["token"].astype(str))
                if not df.empty else set())
        new = [dict(token=t, status="pending", added_at=dt.datetime.now().isoformat())
               for t in toks if t not in seen]
        added["n"] = len(new)
        return pd.concat([df, pd.DataFrame(new)], ignore_index=True) if new else df
    queue_update(svc, root, m, owner=owner)
    return added["n"]

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
        sym = str(r[sym_c]) if sym_c else t
        if sym.lower() in ("nan", "none", ""):        # BSE-only co: no NSE symbol
            sym = ""
        return (str(r[isin_c]) if isin_c else t,
                sym,
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

def _scrape_screener_financials(symbol: str, bse_code=None) -> str | None:
    """Live-scrape the 6 Screener financial tables into a compact text block.
    Screener accepts NSE symbol OR BSE scrip code as the URL token — BSE-only
    companies (no NSE symbol) resolve via bse_code (e.g. /company/539730/)."""
    from bs4 import BeautifulSoup
    sess = _screener_session()
    tokens = [t for t in (str(symbol or "").strip(), str(bse_code or "").strip())
              if t and t.lower() != "nan"]
    candidates = []          # (n_periods, rendered_block) — pick the view with MORE
    for tok in tokens:
      for view in ("consolidated/", ""):
        url = f"https://www.screener.in/company/{tok}/{view}"
        try:
            r = sess.get(url, timeout=30)
        except Exception:
            continue
        if r.status_code != 200 or 'id="profit-loss"' not in r.text:
            continue
        soup = BeautifulSoup(r.text, "lxml")
        blocks = [f"SCREENER STRUCTURED FINANCIALS ({'consolidated' if view else 'standalone'}) "
                  f"— fetched {dt.date.today().isoformat()}:"]
        # Top ratios strip (Market Cap / P/E / Book Value / ROE / ROCE ...) — the page
        # header list, not a table; supplies mcap etc. for BSE-only names missing from
        # the NSE-keyed fundamentals parquet.
        top = soup.find(id="top-ratios")
        if top:
            kv = []
            for li in top.find_all("li"):
                nm = li.find(class_="name"); vl = li.find(class_="value") or li.find(class_="number")
                if nm and vl:
                    kv.append(f"{nm.get_text(' ', strip=True)}: {vl.get_text(' ', strip=True)}")
            if kv:
                blocks.append("\n== KEY METRICS ==\n" + "\n".join(kv))
        got = False
        n_periods = 0
        for sec_id, label in SCREENER_SECTIONS:
            sec = soup.find(id=sec_id)
            tbl = sec.find("table") if sec else None
            if tbl is None:
                continue
            if sec_id == "profit-loss":          # history depth = P&L year columns
                n_periods = max(0, len(tbl.select("thead th")) - 1)
            blocks.append(f"\n== {label} ==\n{_fmt_screener_table(tbl)}")
            got = True
        if got:
            # Don't return the first hit: consolidated view can be near-empty for a
            # company that only recently consolidated (2 columns) while standalone
            # carries the full 10-year history. Keep both, pick the deeper one.
            candidates.append((n_periods, "\n".join(blocks)))
      if candidates:
          break                                  # this token worked — don't retry via bse_code
    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]
    return None

def screener_financials_block(svc, root, isin, symbol, bse_code=None) -> str:
    """Always try a LIVE Screener fetch; cache it to Drive on success; fall back
    to the cached copy if live fails (Screener down / no cookie / unreachable)."""
    cache_path = f"{DRIVE['company_page']}/{isin}/screener_financials.txt"
    live = _scrape_screener_financials(symbol, bse_code=bse_code)
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

# NEWS_WHITELIST + news_block moved to alt_sources.py (light deps: requests+bs4
# only) so daily_brief/CI can import them WITHOUT this module's heavy chain
# (daily_research_summary -> pdf_ocr -> fitz, absent in Phase-1 CI). Re-exported
# here so existing callers keep working.
from alt_sources import NEWS_WHITELIST, news_block  # noqa: F401

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

# ---- DRHP / RHP prospectus (auto-discovery + content guardrail) ------------
PROSPECTUS_AUTH_DOMAINS = ("nseindia.com", "bseindia.com", "sebi.gov.in")
MAX_PROSPECTUS_BYTES = 280 * 1024 * 1024     # cap a single download (~280MB)
MAX_PROSPECTUS_SUMMARY_CHARS = 11_000        # richer than other docs

def _ddg_results(query):
    """DuckDuckGo HTML search -> list of (title, url). No API key."""
    import urllib.parse
    from bs4 import BeautifulSoup
    try:
        r = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                          headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                          timeout=25)
    except Exception:
        return []
    soup = BeautifulSoup(r.text, "lxml")
    out = []
    for a in soup.select("a.result__a"):
        h = a.get("href", "")
        m = re.search(r"uddg=([^&]+)", h)
        url = urllib.parse.unquote(m.group(1)) if m else h
        out.append((a.get_text(" ", strip=True), url))
    return out

def _screener_drhp_urls(symbol: str) -> list[str]:
    """Scrape Screener company page #documents section for DRHP/prospectus PDF links.
    Screener often lists the DRHP under the Annual Reports subsection, linking
    directly to the BSE/NSE-hosted PDF — very reliable when present."""
    if not symbol:
        return []
    from bs4 import BeautifulSoup
    out = []
    for view in ("consolidated/", ""):
        url = f"https://www.screener.in/company/{symbol}/{view}"
        try:
            r = requests.get(url,
                             headers={"User-Agent": "Mozilla/5.0",
                                      "Referer": "https://www.screener.in/"},
                             timeout=25)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            # Look inside #documents section (and anywhere on the page as fallback)
            doc_sec = soup.find(id="documents") or soup
            for a in doc_sec.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(" ", strip=True).lower()
                low = href.lower()
                is_prosp = (
                    "drhp" in low or "drhp" in text
                    or "red herring" in text
                    or "prospectus" in low or "prospectus" in text
                    or "rhp" in text
                )
                is_pdf = ".pdf" in low or "pdf" in low
                if is_prosp and is_pdf:
                    if href.startswith("//"):
                        href = "https:" + href
                    elif href.startswith("/"):
                        href = "https://www.screener.in" + href
                    out.append(href)
        except Exception:
            pass
        if out:
            break
    return out[:4]


def _bse_offer_urls(bse_code: str) -> list[str]:
    """BSE Prospectus/Offer Document API — returns direct PDF links for the company's
    filed offer documents (DRHP, RHP, Prospectus). No rate-limit risk."""
    if not bse_code:
        return []
    out = []
    try:
        # BSE ProspectusData endpoint (documented in BSE developer portal)
        api = (f"https://api.bseindia.com/BseIndiaAPI/api/ProspectusData/w"
               f"?scripcd={bse_code}&type=FP")
        r = requests.get(api, headers={"User-Agent": "Mozilla/5.0",
                                        "Referer": "https://www.bseindia.com/"},
                         timeout=20)
        if r.status_code == 200:
            data = r.json()
            # Response is usually a list of {PDFLINKDATA, DOCUMENT_TYPE, ...}
            items = data if isinstance(data, list) else data.get("Table", [])
            for item in items:
                link = item.get("PDFLINKDATA") or item.get("PDF_LINK") or ""
                dtype = (item.get("DOCUMENT_TYPE") or item.get("Doc_Type") or "").upper()
                if not link:
                    continue
                # Prefer DRHP/RHP entries; include Prospectus entries too
                if any(k in dtype for k in ("DRHP", "RED HERRING", "PROSP", "OFFER")):
                    if not link.startswith("http"):
                        link = "https://www.bseindia.com" + link
                    out.append(link)
    except Exception:
        pass
    if not out:
        # Fallback: scrape BSE Prospectus page directly
        try:
            from bs4 import BeautifulSoup
            page = (f"https://www.bseindia.com/corporates/"
                    f"Prospectus_Subs.aspx?scripcd={bse_code}")
            r = requests.get(page, headers={"User-Agent": "Mozilla/5.0",
                                             "Referer": "https://www.bseindia.com/"},
                             timeout=20)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "lxml")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    text = a.get_text(" ", strip=True).lower()
                    low = href.lower()
                    if (".pdf" in low or "pdf" in low) and any(
                        k in low + text for k in ("drhp", "herring", "prosp", "offer")
                    ):
                        if not href.startswith("http"):
                            href = "https://www.bseindia.com" + href
                        out.append(href)
        except Exception:
            pass
    return out[:4]


def _nse_emerge_urls(symbol: str) -> list[str]:
    """NSE Emerge archive: well-known patterns for SME DRHP/RHP PDFs.
    Also queries the NSE Emerge listing documents API."""
    if not symbol:
        return []
    out = []
    sym = symbol.upper()
    # Direct archive patterns that NSE Emerge uses for listing documents
    for pat in (
        f"https://nsearchives.nseindia.com/emerge/content/{sym}_DRHP.pdf",
        f"https://nsearchives.nseindia.com/emerge/content/{sym}_RHP.pdf",
        f"https://nsearchives.nseindia.com/emerge/content/{sym}-DRHP.pdf",
        f"https://nsearchives.nseindia.com/emerge/content/{sym}-RHP.pdf",
    ):
        try:
            r = requests.head(pat, headers={"User-Agent": "Mozilla/5.0"},
                              timeout=15, allow_redirects=True)
            if r.status_code == 200:
                out.append(pat)
        except Exception:
            pass
    # NSE Emerge API for listing documents
    try:
        api = (f"https://www.nseindia.com/api/emerge-content-details"
               f"?symbol={sym}&type=drhp")
        r = requests.get(api, headers={"User-Agent": "Mozilla/5.0",
                                        "Referer": "https://www.nseindia.com/"},
                         timeout=20)
        if r.status_code == 200:
            items = r.json() if isinstance(r.json(), list) else []
            for item in items:
                link = item.get("link") or item.get("url") or ""
                if link and ".pdf" in link.lower():
                    if not link.startswith("http"):
                        link = "https://www.nseindia.com" + link
                    out.append(link)
    except Exception:
        pass
    return out[:4]


def _discover_prospectus_urls(name, symbol=None, bse_code=None):
    """Ranked candidate PDF URLs for a company's DRHP/RHP.
    Priority: Screener docs (user-visible, reliable) → BSE offer API → NSE Emerge
    archives → DDG as last resort. Returns deduplicated list, authoritative first."""
    import urllib.parse
    seen: set[str] = set()
    results: list[tuple[int, str]] = []   # (score, url)

    def _add(url: str, score: int):
        if url and url not in seen:
            seen.add(url)
            results.append((score, url))

    # --- Tier 1: Screener #documents section (user's suggestion; links to exchange PDFs) ---
    for u in _screener_drhp_urls(symbol or ""):
        _add(u, 12)

    # --- Tier 2: BSE offer documents API ---
    for u in _bse_offer_urls(str(bse_code) if bse_code else ""):
        _add(u, 11)

    # --- Tier 3: NSE Emerge archive (SME) ---
    for u in _nse_emerge_urls(symbol or ""):
        _add(u, 10)

    # --- Tier 4: DDG fallback (rate-limited; only if nothing found yet) ---
    if not results:
        cands = []
        for q in (f'"{name}" red herring prospectus site:bseindia.com OR site:nseindia.com',
                  f'"{name}" DRHP filetype:pdf'):
            cands += _ddg_results(q)
        name_tok = re.sub(r"[^a-z]", "", name.lower().split()[0]) if name.split() else ""
        for text, url in cands:
            low = url.lower()
            if ".pdf" not in low and "pdf" not in low:
                continue
            if url in seen:
                continue
            dom = urllib.parse.urlparse(url).netloc.lower()
            score = 0
            if any(d in dom for d in PROSPECTUS_AUTH_DOMAINS):
                score += 10
            if name_tok and len(name_tok) >= 4 and name_tok in dom.replace(".", ""):
                score += 8
            if any(b in dom for b in ("scribd", "helpstudent", "hellobanker", "ssjfinance")):
                score -= 12
            if re.search(r"rhp|herring|prospectus", low + text.lower()):
                score += 3
            _add(url, score)

    results.sort(key=lambda x: x[0], reverse=True)
    return [u for s, u in results if s > -5][:6]

def _download_prospectus(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},
                         timeout=120, stream=True)
        if r.status_code != 200:
            return None
        buf = bytearray()
        for chunk in r.iter_content(131072):
            buf += chunk
            if len(buf) > MAX_PROSPECTUS_BYTES:
                break
        return bytes(buf) if buf[:5].startswith(b"%PDF") else None
    except Exception:
        return None

def _verify_prospectus(pdf_bytes, name):
    """GUARDRAIL: confirm this PDF is a DRHP/RHP AND names THIS company on the
    cover (defends against symbol/name mismatch grabbing the wrong document).
    Returns 'drhp' | 'rhp' | None."""
    try:
        import fitz
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        head = " ".join(d[i].get_text() for i in range(min(6, d.page_count)))
        d.close()
    except Exception:
        return None
    H = head.upper()
    is_drhp = "DRAFT RED HERRING PROSPECTUS" in H
    is_rhp = (not is_drhp) and ("RED HERRING PROSPECTUS" in H
                                or ("PROSPECTUS" in H and "PUBLIC ISSUE" in H))
    if not (is_drhp or is_rhp):
        return None
    core = re.sub(r"\b(limited|ltd|private|pvt)\b", "", name.lower())
    core = re.sub(r"[^a-z ]", "", core)
    flat_head = re.sub(r"[^a-z]", "", head.lower())
    toks = [t for t in core.split() if len(t) >= 4]
    contiguous = core.replace(" ", "") and core.replace(" ", "") in flat_head
    all_tokens = bool(toks) and all(t in head.lower() for t in toks)
    if not (contiguous or all_tokens):
        return None                                 # wrong company — reject
    return "drhp" if is_drhp else "rhp"

def drhp_block(svc, root, isin, symbol, name, pool, bse_code=None):
    """Best-effort: discover the company's prospectus, verify it's the right
    document, summarise (chunked) with the risk/fraud prompt, cache the summary.
    RHP-PRIMARY: the final RHP supersedes the draft, so we summarise the RHP and
    only fall back to the DRHP when no RHP is found — this halves the cost. Raw PDF
    is never stored; only the summary sidecar persists."""
    found = {}
    for typ in ("rhp", "drhp"):
        side = f"{DRIVE['company_page']}/{isin}/doc_summaries/prospectus_{typ}.md"
        c = drive_download(svc, side, root)
        if c:
            found[typ] = c.decode("utf-8", "ignore")
    if found:                                       # already have a cached prospectus
        return _fmt_prospectus(found, name)
    try:
        # Verify candidates WITHOUT summarising; prefer an RHP, hold a DRHP as backup.
        chosen = None                               # (typ, data)
        for url in _discover_prospectus_urls(name, symbol=symbol, bse_code=bse_code):
            data = _download_prospectus(url)
            if not data:
                continue
            typ = _verify_prospectus(data, name)
            if typ == "rhp":
                chosen = ("rhp", data)
                break                               # final doc found — stop
            if typ == "drhp" and chosen is None:
                chosen = ("drhp", data)             # keep looking for an RHP
        if chosen:
            prompt = open(os.path.join(SCRIPTS_DIR, "drhp_prompt.txt"),
                          encoding="utf-8").read()
            typ, data = chosen
            summ = _summarise_pdf_chunked(pool, prompt, data, f"{typ.upper()} {name}",
                                          prefer_text=True)
            if summ:
                found[typ] = summ
                try:
                    drive_upload(svc, f"{DRIVE['company_page']}/{isin}/doc_summaries/"
                                 f"prospectus_{typ}.md", root, summ.encode("utf-8"),
                                 "text/markdown")
                except Exception:
                    pass
    except Exception as e:
        if not found:
            return f"DATA_MISSING (prospectus fetch failed: {type(e).__name__})."
    return _fmt_prospectus(found, name) if found else "DATA_MISSING (no DRHP/RHP found)."

def _fmt_prospectus(found, name):
    return "\n\n".join(
        f"### [{t.upper()} | {name}]\n{s.strip()[:MAX_PROSPECTUS_SUMMARY_CHARS]}"
        for t, s in found.items())

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

def _partial_prefix(isin: str, row) -> str:
    """Drive prefix for this doc's per-chunk resume sidecars."""
    return f"{DRIVE['company_page']}/{isin}/doc_summaries/_partials/{row['doc_id']}"

def _doc_sidecar_cached(svc, root, isin, row) -> bool:
    """True if the doc's FULL summary sidecar already exists (so reusing it costs no
    Gemini calls — used to let cached docs bypass the deadline gate)."""
    try:
        return drive_find(svc, _sidecar_path(isin, row), root) is not None
    except Exception:
        return False

def _refetch_doc_bytes(row):
    """Re-fetch a doc's bytes from its source URL (the raw PDF on Drive is deleted by the
    2-day retention). Reuses the backfill fetcher (handles BSE/NSE PDFs, zips, ICRA HTML).
    Best-effort — returns bytes or None."""
    url = str(row.get("pdf_url") or "").strip()
    if not url:
        return None
    try:
        from backfill_company_docs import fetch_document, screener_session
        out = fetch_document(screener_session(), url)
        return out[0] if out else None
    except Exception:
        return None

def _writeback_thin_section(svc, root, isin, row, summary) -> bool:
    """Write a freshly re-summarised thin AR/concall back into its company_page.md section
    (rule 7C), under _extract.lock, BEST-EFFORT. Safety: REPLACE-ONLY — if the exact
    section header isn't already present it SKIPS (never appends → no duplicate sections);
    if the lock is held by Phase 2 / the nightly backfill it SKIPS (report is already
    complete from the sidecar). Updates response_chars so the doc isn't re-flagged thin.
    Never raises. Returns True only if the page was actually updated."""
    dt_ = str(row["doc_type"])
    if dt_ not in ("annual_report", "concall"):
        return False
    try:
        qf = _read_parquet(svc, "company_repo/_index/quarterly_facts.parquet", root)
        if qf.empty or "source_doc_id" not in qf.columns:
            return False
        m = qf[qf["source_doc_id"].astype(str) == str(row["doc_id"])]
        if m.empty:
            return False
        period = str(m.iloc[0].get("quarter") or "").strip()   # exact label the extractor used
        if not period:
            return False
        label = "Annual Report" if dt_ == "annual_report" else "Concall"
        page_b = drive_download(svc, f"{DRIVE['company_page']}/{isin}/company_page.md", root)
        if not page_b:
            return False
        if not re.search(rf'##\s+{re.escape(period)}\s+{re.escape(label)}\b',
                         page_b.decode("utf-8", "ignore")):
            return False                       # section not found → skip (no append/dup)
        from _extractor_base import acquire_lock, release_lock
        index_id = _folder_id(svc, "company_repo/_index", root)
        repo_id  = _folder_id(svc, "company_repo", root)
        if not index_id or not repo_id:
            return False
        if not acquire_lock(svc, index_id, "_extract.lock", "deepdive_writeback",
                            wait_min=0.5, defer_to_phase2=True):
            print(f"      writeback: lock busy — skip page update for {period} {label}")
            return False
        try:
            title = str(row.get("title") or "")[:90]
            if dt_ == "annual_report":
                from extract_annual_report import _replace_ar_section
                _replace_ar_section(svc, repo_id, isin, period, summary, title)
            else:
                from extract_concall import replace_company_page_section
                replace_company_page_section(svc, repo_id, isin, period, summary, title)
            qf2 = _read_parquet(svc, "company_repo/_index/quarterly_facts.parquet", root)
            mask = qf2["source_doc_id"].astype(str) == str(row["doc_id"])
            if mask.any():
                qf2.loc[mask, "response_chars"] = len(summary)
                buf = io.BytesIO(); qf2.to_parquet(buf, index=False)
                drive_upload(svc, "company_repo/_index/quarterly_facts.parquet", root,
                             buf.getvalue(), "application/octet-stream")
            print(f"      writeback: enriched {period} {label} section in company_page.md")
            return True
        finally:
            release_lock(svc, index_id, "_extract.lock")
    except Exception as e:
        print(f"      writeback skipped ({type(e).__name__}: {str(e)[:60]})")
        return False

def _thin_doc_ids(svc, root, isin) -> set:
    """doc_ids for THIS isin whose stored extraction (response_chars in
    quarterly_facts) is below the per-type threshold — i.e. a partial/thin read the
    deep dive should re-summarise in full. AR<8000, concall<5500 (user 2026-06-22).
    Only concall/AR carry response_chars; other types are presence-only."""
    try:
        qf = _read_parquet(svc, "company_repo/_index/quarterly_facts.parquet", root)
        if qf.empty or "source_doc_id" not in qf.columns or "response_chars" not in qf.columns:
            return set()
        if "isin" in qf.columns:
            qf = qf[qf["isin"].astype(str) == isin]
        if qf.empty:
            return set()
        pq = _read_parquet(svc, DRIVE["proc_queue"], root)
        id2type = (dict(zip(pq["doc_id"].astype(str), pq["doc_type"].astype(str)))
                   if not pq.empty and "doc_id" in pq.columns else {})
        thin = set()
        for _, r in qf.iterrows():
            did = str(r.get("source_doc_id"))
            thr = THIN_RESPONSE_CHARS.get(id2type.get(did, ""))
            rc = pd.to_numeric(r.get("response_chars"), errors="coerce")
            if thr and pd.notna(rc) and rc < thr:
                thin.add(did)
        return thin
    except Exception:
        return set()

def _sweep_partials(svc, root, isin, row):
    """Delete this doc's per-chunk resume partials once its full sidecar exists.
    Best-effort: lists the _partials folder, removes files named {doc_id}_*. Never raises."""
    try:
        folder = f"{DRIVE['company_page']}/{isin}/doc_summaries/_partials"
        fid = _folder_id(svc, folder, root)
        if not fid:
            return
        doc_id = str(row["doc_id"])
        res = svc.files().list(q=f"'{fid}' in parents and trashed=false",
                               fields="files(id,name)", pageSize=200).execute().get("files", [])
        for f in res:
            if f["name"].startswith(f"{doc_id}_"):
                try:
                    svc.files().delete(fileId=f["id"]).execute()
                except Exception:
                    pass
    except Exception:
        pass

def _summarise_pdf_chunked(pool, prompt, pdf_bytes, label, prefer_text=False,
                           partial_cache=None, deadline_ts=None) -> str | None:
    """Summarise a PDF completely. Short docs -> single pass. Long docs -> split
    into CHUNK_PAGES page-range chunks, summarise each, then merge.

    prefer_text=True extracts TEXT per chunk and uses call_text (fast, light) —
    right for large text-based documents like DRHP/RHP where uploading PDF chunks
    via call_pdf is slow. Default (False) uses call_pdf for best table fidelity
    (annual reports).

    partial_cache=(svc, root, prefix): persist each page-range partial as a Drive
    sidecar and REUSE it on a later run, so a killed/deadline-stopped run resumes
    mid-document instead of restarting it. None = no caching (e.g. drhp_block).
    deadline_ts: if set and reached before a NOT-yet-cached chunk, raise
    DeadlineReached so the caller can stop cleanly with progress preserved."""
    import fitz
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = src.page_count
    # short enough -> one pass
    if n <= CHUNK_TRIGGER_PAGES and len(pdf_bytes) <= MAX_INLINE_PDF and not prefer_text:
        src.close()
        try:
            return pool.call_pdf(pdf_bytes, prompt)[0]
        except FatalCallError as e:
            print(f"      {label}: FATAL ({str(e)[:70]}) — skip"); return None

    svc = root = pfx = None
    if partial_cache:
        svc, root, pfx = partial_cache

    # In text mode, summarise larger page windows per call (text is cheap).
    step = (CHUNK_PAGES * 2) if prefer_text else CHUNK_PAGES
    partials = []
    for start in range(0, n, step):
        end = min(start + step, n)
        # Reuse a cached partial if we have one (free, instant resume).
        cached = None
        if partial_cache:
            cb = drive_download(svc, f"{pfx}_{start+1}-{end}.md", root)
            if cb:
                cached = cb.decode("utf-8", "ignore")
        if cached is not None:
            partials.append(cached)
            print(f"      {label}: chunk pages {start+1}-{end}/{n} cached")
            continue
        # Not cached — this chunk costs Gemini calls, so enforce the deadline here.
        if deadline_ts and time.monotonic() >= deadline_ts:
            src.close()
            raise DeadlineReached(f"{label}: chunk budget exhausted at page {start+1}")
        cprompt = (prompt + f"\n\n>>> This is PAGES {start+1}-{end} of {n} of the "
                   f"{label}. Summarise THIS portion faithfully and completely; "
                   f"other passes cover the remaining pages. Keep ALL figures, "
                   f"tables, related-party items, litigation and auditor notes verbatim.")
        try:
            if prefer_text:
                page_text = "\n".join(src[p].get_text()
                                      for p in range(start, end))[:MAX_DOC_TEXT_CHARS]
                if len(page_text.strip()) < 100:
                    continue                      # scanned/empty page range
                txt, _ = pool.call_text(cprompt + "\n\n=== TEXT ===\n" + page_text)
            else:
                sub = fitz.open(); sub.insert_pdf(src, from_page=start, to_page=end - 1)
                sub_bytes = sub.tobytes(); sub.close()
                if len(sub_bytes) <= MAX_INLINE_PDF:
                    txt, _ = pool.call_pdf(sub_bytes, cprompt)
                else:
                    page_text = "\n".join(src[p].get_text()
                                          for p in range(start, end))[:MAX_DOC_TEXT_CHARS]
                    txt, _ = pool.call_text(cprompt + "\n\n=== TEXT ===\n" + page_text)
            part = f"--- pages {start+1}-{end} ---\n{txt.strip()}"
            partials.append(part)
            if partial_cache:                     # persist so next run resumes here
                try:
                    drive_upload(svc, f"{pfx}_{start+1}-{end}.md", root,
                                 part.encode("utf-8"), "text/markdown")
                except Exception:
                    pass
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

def summarise_doc(svc, root, pool, isin, row, deadline_ts=None) -> str | None:
    """Return a per-doc summary. Reuse cached sidecar if present, else fetch the
    raw file (PDF or extracted-HTML text) and run its doc-type prompt. Caches the
    result as a sidecar so future dives reuse it for free. Long PDFs additionally
    cache per-chunk partials so a killed/deadline-stopped run resumes mid-document.
    AllBucketsExhausted / DeadlineReached propagate (caller stops); FatalCallError
    on one doc -> skip that doc."""
    sidecar = _sidecar_path(isin, row)
    cached = drive_download(svc, sidecar, root)
    if cached:
        return cached.decode("utf-8", "ignore")

    fid = str(row.get("drive_file_id") or "").strip()
    data = _download_file_id(svc, fid) if fid else None
    if not data:
        # PDF aged out (2-day retention) or no stored file — re-fetch from source so a
        # thin/old doc can still be re-summarised. Best-effort; None if unreachable.
        data = _refetch_doc_bytes(row)
    if not data:
        return None

    doc_type = str(row["doc_type"])
    prompt = _load_doc_prompt(doc_type)
    label = f"{doc_type} {str(row['announcement_date'])[:10]}"
    pfx = _partial_prefix(isin, row)
    is_pdf = data[:5].startswith(b"%PDF")
    if is_pdf:
        # complete-read with page-range chunking for long reports; partials cached
        # on Drive so a stopped run resumes here rather than restarting the doc.
        summ = _summarise_pdf_chunked(pool, prompt, data, label,
                                      partial_cache=(svc, root, pfx),
                                      deadline_ts=deadline_ts)
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
        # Full summary persisted -> the per-chunk partials are now redundant; sweep
        # them to respect Drive space (best-effort, never fatal).
        if is_pdf:
            _sweep_partials(svc, root, isin, row)
    except Exception:
        pass
    return summ

def assemble_doc_summaries(svc, root, pool, isin, deadline_ts=None) -> tuple[str, list[dict]]:
    """Summarise every actual document for this ISIN (reuse-or-generate) and
    return (combined_block, used_docs). Docs already folded into company_page.md
    (status=done) are skipped — the COMPANY_PAGE_BRIEF already carries them.

    Cached doc/chunk sidecars are always reused (cheap); only docs needing a FRESH
    Gemini summary are gated by deadline_ts. If the wall-clock budget is hit before a
    not-yet-cached doc, raise DeadlineReached so the run exits cleanly and resumes
    next time from the sidecars written so far."""
    q = _read_parquet(svc, DRIVE["proc_queue"], root)
    if q.empty or "isin" not in q.columns:
        return "DATA_MISSING (no document index).", []
    rows = q[(q["isin"].astype(str) == isin) &
             (q["status"].astype(str) != "download_failed")]
    if rows.empty:
        return "DATA_MISSING (no documents ingested for this company).", []

    # Thin AR/concall (partial reads) are re-summarised IN FULL even though they are
    # `done` — so the deep dive never trusts a thin company_page section (user 2026-06-22).
    thin = _thin_doc_ids(svc, root, isin)
    blocks, used, n_thin = [], [], 0
    for _, r in rows.sort_values("announcement_date").iterrows():
        is_thin = str(r["doc_id"]) in thin
        if str(r.get("status")) == "done" and not is_thin:
            continue                       # rich done doc — already in COMPANY_PAGE_BRIEF
        # Reuse-or-generate. A cached summary returns instantly; a fresh one costs
        # Gemini calls, so only THEN enforce the deadline (let cached docs through).
        if deadline_ts and time.monotonic() >= deadline_ts \
                and not _doc_sidecar_cached(svc, root, isin, r):
            raise DeadlineReached(f"{isin}: doc budget exhausted")
        summ = summarise_doc(svc, root, pool, isin, r, deadline_ts=deadline_ts)
        if not summ:
            continue
        if is_thin and str(r.get("status")) == "done":
            n_thin += 1
            # Persist the richer read back into company_page.md's section (7C, best-effort,
            # replace-only + lock-guarded). The report already has it via the block above.
            _writeback_thin_section(svc, root, isin, r, summ)
        d = str(r["announcement_date"])[:10]
        title = str(r.get("title", ""))[:90]
        blocks.append(f"### [{r['doc_type']} | {d} | {title}]\n"
                      f"{summ.strip()[:MAX_DOC_SUMMARY_CHARS]}")
        used.append({"doc_type": str(r["doc_type"]), "date": d,
                     "title": title, "doc_id": str(r["doc_id"])})
    if n_thin:
        print(f"    completeness: re-summarised {n_thin} thin AR/concall doc(s) in full")
    if not blocks:
        return "DATA_MISSING (documents present but none summarisable).", []
    return "\n\n".join(blocks), used

# ---- prompt assembly ------------------------------------------------------
def _fmt(v, spec="{:.0f}"):
    try:
        f = float(v)
        return spec.format(f) if f == f else "?"
    except (TypeError, ValueError):
        return "?"


def phase3_block(svc, root, isin, symbol) -> str:
    """Pre-computed Phase 3 nightly metrics for THIS company — scorecard, fraud
    tracker, mgmt credibility, valuation, current guidance, guided-vs-actual,
    catalysts, derived ratios. Injected as authoritative quant facts the model
    must reconcile its narrative against (zero extra Gemini calls)."""
    sym = str(symbol).upper()

    def _by_co(path):
        df = _read_parquet(svc, path, root)
        if df.empty:
            return df
        if "isin" in df.columns:
            m = df[df["isin"].astype(str) == isin]
            if not m.empty:
                return m
        if "symbol" in df.columns:
            return df[df["symbol"].astype(str).str.upper() == sym]
        return df.iloc[0:0]

    P = "company_repo/_index"
    parts = []
    try:
        import build_classification as bcl
        blk = bcl.classification_block(svc, root, isin, sym)
        if blk != "DATA_MISSING":
            parts.append("CLASSIFICATION & PEERS: " + blk.replace("\n", " "))
    except Exception:
        pass
    # FACT row (mcap / valuation / price moves / growth) + peer medians
    facts = _by_co(f"{P}/company_facts.parquet")
    if not facts.empty:
        r = facts.iloc[0]
        def _g(k):
            v = r.get(k)
            if v is None or (isinstance(v, float) and v != v):
                return "?"
            return round(v, 1) if isinstance(v, float) else v
        parts.append(
            f"MARKET & FINANCIALS (latest): mcap={_g('mcap_cr')} cr | "
            f"P/E={_g('pe')} | P/B={_g('pb')} | price ret 3m/6m/12m="
            f"{_g('ret_3m_pct')}/{_g('ret_6m_pct')}/{_g('ret_12m_pct')}% | "
            f"latest {_g('latest_q')}: rev={_g('rev_q')} (YoY {_g('rev_q_yoy')}%, "
            f"QoQ {_g('rev_q_qoq')}%), PAT={_g('pat_q')} (YoY {_g('pat_q_yoy')}%), "
            f"EPS={_g('eps_q')} (YoY {_g('eps_q_yoy')}%) | TTM rev/PAT/EPS="
            f"{_g('rev_ttm')}/{_g('pat_ttm')}/{_g('eps_ttm')}")
        pg = str(r.get("peer_group", "")).strip()
        if pg:
            pa = _read_parquet(svc, f"{P}/peer_aggregates.parquet", root)
            if not pa.empty:
                m = pa[(pa["level"] == "peer_group") & (pa["group"] == pg)]
                if not m.empty:
                    a = m.iloc[0]
                    parts.append(
                        f"PEER MEDIANS ({pg}, n={int(a['n'])}): "
                        f"P/E={a.get('pe_median')} | 12m ret={a.get('ret_12m_pct_median')}% "
                        f"| rev YoY={a.get('rev_q_yoy_median')}% | "
                        f"PAT YoY={a.get('pat_q_yoy_median')}% — compare vs the company above")
    sc = _by_co(f"{P}/company_scorecard.parquet")
    if not sc.empty:
        r = sc.iloc[0]
        facs = ", ".join(f"{c[6:]}={_fmt(r[c])}" for c in sc.columns
                         if c.startswith("score_") and pd.notna(r[c]))
        parts.append(f"SCORECARD: composite={_fmt(r.get('composite_score'))}/100 "
                     f"(data completeness {_fmt(r.get('data_completeness_pct'))}%)"
                     f" | factors: {facs}")
    # Fraud & surveillance — exact reasons, split by NATURE of the signal:
    # exchange surveillance = price/volatility control (NOT fraud per se);
    # regulatory orders / forensic flags / fraud-news = integrity signals.
    ft = _by_co(f"{P}/fraud_tracker.parquet")
    inv = _by_co(f"{P}/investigative_fraud.parquet")
    fr = _by_co(f"{P}/fraud_risk.parquet")
    fl = []
    if not ft.empty:
        r = ft.iloc[0]
        fl.append(f"score: {r.get('band')} {_fmt(r.get('fraud_score'))}/100 "
                  f"(driver: {r.get('score_driver', '?')}; "
                  f"flagged since {r.get('first_flagged_at')})")
    if not inv.empty:
        r = inv.iloc[0]
        surv = []
        for v in (r.get("asm_level"), r.get("esm_level")):
            if str(v) not in ("none", "", "nan", "None"):
                surv.append(str(v))
        gsm = pd.to_numeric(r.get("gsm_stage"), errors="coerce")
        if pd.notna(gsm) and int(gsm):
            surv.append(f"GSM-{int(gsm)}")
        if bool(r.get("t2t")):
            surv.append("T2T")
        if str(r.get("bse_group", "")).strip() not in ("", "nan"):
            surv.append(f"BSE group {r.get('bse_group')}")
        fl.append("exchange surveillance (price/volatility control measures — "
                  "NOT fraud per se): " + (", ".join(surv) if surv else "none"))
        integ = []
        for col, lbl in (("sebi_actions", "SEBI order match(es)"),
                         ("nfra_actions", "NFRA order(s)")):
            n = pd.to_numeric(r.get(col), errors="coerce")
            if pd.notna(n) and int(n):
                integ.append(f"{int(n)} {lbl}")
        n_news = pd.to_numeric(r.get("news_hits"), errors="coerce")
        if pd.notna(n_news) and int(n_news):
            heads = ""
            try:
                snips = json.loads(str(r.get("news_snippets") or "[]"))
                heads = " — " + "; ".join(s.get("headline", "")[:90]
                                          for s in snips[:3])
            except Exception:
                pass
            integ.append(f"{int(n_news)} fraud-keyword news hit(s){heads}")
        fl.append("integrity signals (adverse): "
                  + ("; ".join(integ) if integ else "none"))
    if not fr.empty:
        r = fr.iloc[0]
        flags = str(r.get("forensic_flags", "")).strip()
        if flags:
            fl.append(f"forensic accounting flags (score "
                      f"{_fmt(r.get('fraud_risk_score'))}/100, higher=worse): {flags}")
    if fl:
        parts.append("FRAUD & SURVEILLANCE:\n  - " + "\n  - ".join(fl))
    else:
        parts.append("FRAUD & SURVEILLANCE: clean — no surveillance listing, no "
                     "regulatory order match, no forensic flag.")
    mc = _by_co(f"{P}/mgmt_credibility.parquet")
    if not mc.empty and "cred_score" in mc.columns:
        r = mc.sort_values("quarter").iloc[-1]
        parts.append(f"MGMT CREDIBILITY (said-vs-delivered, {r.get('quarter')}): "
                     f"score={r.get('cred_score')} pattern={r.get('pattern')} "
                     f"strongest={r.get('strongest_area')} "
                     f"recurring_miss={r.get('recurring_miss')}")
    val = _by_co(f"{P}/valuation.parquet")
    if not val.empty:
        r = val.iloc[0]
        parts.append(f"VALUATION: P/E={_fmt(r.get('pe'), '{:.1f}')} "
                     f"({r.get('mcap_segment')}); cheaper than "
                     f"{_fmt(r.get('pe_pctile_segment'))}% of segment peers; "
                     f"PEG~{_fmt(r.get('peg_proxy'), '{:.2f}')}; "
                     f"valuation_score={_fmt(r.get('valuation_score'))}/100")
    gt = _by_co(f"{P}/guidance_tracker.parquet")
    if not gt.empty and "quarter" in gt.columns:
        latest_q = gt.sort_values("quarter")["quarter"].iloc[-1]
        rows = gt[gt["quarter"] == latest_q].head(8)
        gl = "; ".join(f"{r.get('metric')}={r.get('value')}"
                       f" ({r.get('horizon_fy')})" for _, r in rows.iterrows())
        parts.append(f"CURRENT GUIDANCE ({latest_q}): {gl}")
    pead = _by_co(f"{P}/pead_flags.parquet")
    if not pead.empty:
        rows = pead.sort_values("as_of").tail(6)
        pl = "; ".join(f"{r.get('metric')}: guided {r.get('guided_value')} vs "
                       f"actual {r.get('actual_value')} = {r.get('verdict')}"
                       for _, r in rows.iterrows())
        parts.append(f"GUIDED vs ACTUAL (PEAD): {pl}")
    cat = _by_co(f"{P}/catalyst_index.parquet")
    if not cat.empty:
        rows = cat.sort_values("as_of").tail(2)
        cl = " | ".join(f"[{r.get('as_of')}] {r.get('catalyst_type')}: "
                        f"{str(r.get('headline', ''))[:120]}" for _, r in rows.iterrows())
        parts.append(f"RECENT CATALYSTS: {cl}")
    der = _by_co(f"{P}/financials_derived.parquet")
    if not der.empty and "metric" in der.columns:
        latest = (der.sort_values("period").groupby("metric").tail(1))
        dl = "; ".join(f"{r.get('metric')}={_fmt(r.get('value'), '{:.1f}')}"
                       for _, r in latest.head(12).iterrows())
        parts.append(f"DERIVED RATIOS (latest period each): {dl}")
    return "\n".join(parts) if parts else "DATA_MISSING"


def fill_section(tpl, tag, content):
    # function replacement -> content is inserted literally (no \g/\1 backref
    # interpretation, which would crash on summaries containing backslashes).
    return re.sub(rf"\[{tag}\].*?\[/{tag}\]",
                  lambda m: f"[{tag}]\n{content}\n[/{tag}]",
                  tpl, flags=re.DOTALL)

def build_prompt(name, symbol, isin, screener, page, research, bse,
                 docs="DATA_MISSING", screener_cross="DATA_MISSING", news="DATA_MISSING",
                 youtube="DATA_MISSING", drhp="DATA_MISSING", phase3="DATA_MISSING",
                 community="DATA_MISSING"):
    tpl = open(os.path.join(SCRIPTS_DIR, "comapnydeepdive_prompt.txt"),
               encoding="utf-8").read()
    tpl = (tpl.replace("[COMPANY_NAME]", name)
              .replace("[NSE_SYMBOL]", symbol)
              .replace("[ISIN]", isin))
    tpl = fill_section(tpl, "SCREENER_FINANCIAL_DATA", screener)
    tpl = fill_section(tpl, "SCREENER_CROSSCHECK", screener_cross)
    tpl = fill_section(tpl, "PHASE3_QUANT_SNAPSHOT", phase3)
    tpl = fill_section(tpl, "COMMUNITY_RESEARCH", community)
    tpl = fill_section(tpl, "COMPANY_PAGE_BRIEF", page[:MAX_PAGE_CHARS] or "DATA_MISSING")
    tpl = fill_section(tpl, "RESEARCH_INDEX_CONTEXT", research)
    tpl = fill_section(tpl, "DRHP_PROSPECTUS", drhp)
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
                interactive=False, do_backfill=DO_BACKFILL, deadline_ts=None):
    isin, symbol, name, bse_code = resolve_isin(token, universe, interactive=interactive)
    print(f"  deep dive: {token} -> {name} ({symbol} / {isin})")

    # Phase 1 — pull full Screener document history (annual reports, ratings,
    # concalls) into processing_queue before we assemble. Best-effort; never fatal.
    if do_backfill and isin.startswith("INE"):
        try:
            from backfill_company_docs import backfill as _backfill
            c = _backfill(symbol, isin, bse_code=str(bse_code or ""))
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
    doc_block, used_docs = assemble_doc_summaries(svc, root, pool, isin,
                                                  deadline_ts=deadline_ts)
    print(f"    documents: {len(used_docs)} summarised/reused")

    # Phase 3 nightly tables — pre-computed scorecard/fraud/credibility/guidance
    # facts the report must reconcile against (no extra Gemini calls).
    p3 = phase3_block(svc, root, isin, symbol)
    print(f"    phase3 snapshot: "
          f"{'ok (' + str(p3.count(chr(10)) + 1) + ' lines)' if p3 != 'DATA_MISSING' else 'missing'}")

    # Screener structured financials — LIVE fetch (cache to Drive), used as an
    # independent cross-check the model reconciles against the Annual Reports.
    screener_cross = screener_financials_block(svc, root, isin, symbol, bse_code=bse_code)
    print(f"    screener cross-check: "
          f"{'live' if not screener_cross.startswith(('DATA_MISSING','[STALE')) else screener_cross[:40]}")

    # Exchange announcements — BSE Direct + best-effort NSE.
    bse = bse_announcements(bse_code)
    nse = nse_announcements(symbol)
    exchange = bse if not nse else f"BSE:\n{bse}\n\nNSE:\n{nse}"
    # Recent news from reputable sources (headlines only).
    news = news_block(name, symbol)
    # Community research — VP top contributors / blogs / X (source-named lines).
    try:
        import social_sources
        community = social_sources.community_block(name, days=30)
    except Exception as e:
        community = "DATA_MISSING"
        print(f"    community fetch failed: {str(e)[:60]}")
    # YouTube research — whitelisted/official channels, transcript summaries (cached).
    youtube = youtube_block(svc, root, isin, symbol, name, pool)
    yt_ok = not youtube.startswith(("DATA_MISSING", "No videos", "Whitelisted"))
    # DRHP/RHP prospectus — auto-discover + content guardrail, summary cached (best-effort).
    drhp = drhp_block(svc, root, isin, symbol, name, pool, bse_code=bse_code)
    drhp_ok = not drhp.startswith("DATA_MISSING")
    print(f"    exchange: BSE={'ok' if not bse.startswith('DATA_MISSING') else 'miss'} "
          f"NSE={'ok' if nse else 'skip'} · news={'ok' if not news.startswith(('DATA_MISSING','No recent')) else 'none'} "
          f"· youtube={'ok' if yt_ok else 'none'} · drhp={'ok' if drhp_ok else 'none'}")

    prompt = build_prompt(name, symbol, isin,
                          screener_block(fund, results, isin, symbol),
                          page,
                          research_block(ridx, isin, symbol, name),
                          exchange,
                          docs=doc_block,
                          screener_cross=screener_cross,
                          news=news,
                          youtube=youtube,
                          drhp=drhp,
                          phase3=p3,
                          community=community)
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
    ap.add_argument("--key-prefix",
                    default=os.environ.get("DEEPDIVE_KEY_PREFIX",
                                           "FREE_POOL,BACKFILL_GEMINI_KEY,GEMINI_API_KEY"),
                    help="comma-separated env prefixes for the Gemini pool, in priority "
                         "order (default FREE_POOL,BACKFILL_GEMINI_KEY,GEMINI_API_KEY)")
    ap.add_argument("--deadline-min", type=float,
                    default=float(os.environ.get("DEEPDIVE_DEADLINE_MIN", "0") or 0),
                    help="wall-clock budget; stop starting new docs/companies past it so "
                         "the queue + sidecars flush cleanly (0 = no cap). Resumes next run.")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(SCRIPTS_DIR), ".env"))

    svc = drive_service(); root = os.environ["GDRIVE_FOLDER_ID"]

    if args.resolve_only and args.names:
        universe = _load_universe(svc, root)
        for t in [x.strip() for x in args.names.split(",") if x.strip()]:
            isin, symbol, name, _ = resolve_isin(t, universe, interactive=args.interactive)
            if isin == t and symbol == t:
                print(f"Could not resolve: {t}"); sys.exit(1)
            print(f"Resolved: {name} ({symbol} / {isin})")
        return

    if args.add:
        n = enqueue_tokens(svc, root, args.add.split(","), owner="add")
        print(f"Enqueued {n} (skipped any already pending/done)."); return

    if BucketPool is None or load_keys is None:
        print("ERROR: google-genai not installed — cannot run the deep dive here. "
              "Install it (pip install google-genai) or run via CI/local with deps.")
        sys.exit(1)
    # Default to the big FREE_POOL+BACKFILL quota (separate Cloud projects, lightly
    # used outside the nightly window) and fall back to the Phase-2 GEMINI_API_KEY
    # pool — dedup keeps order, so a key shared across prefixes is counted once.
    _load = load_keys_multi or (lambda env, pfx: load_keys(env, prefix=pfx.split(",")[0]))
    api_keys = _load(os.environ, args.key_prefix)
    if not api_keys:
        print(f"ERROR: no Gemini keys found for prefixes '{args.key_prefix}' in .env")
        sys.exit(1)
    pool = BucketPool(api_keys, DEEPDIVE_MODELS, inter_call_s=INTER_CALL_SLEEP)
    print(f"Pool: {len(api_keys)} key(s) × {len(DEEPDIVE_MODELS)} model(s) "
          f"= {len(api_keys) * len(DEEPDIVE_MODELS)} daily buckets "
          f"[{args.key_prefix}]")
    deadline_ts = (time.monotonic() + args.deadline_min * 60) if args.deadline_min > 0 else None

    universe = _load_universe(svc, root)
    fund     = _read_parquet(svc, DRIVE["fundamentals"], root)
    results  = _read_parquet(svc, DRIVE["results"], root)
    ridx     = _read_parquet(svc, DRIVE["research_idx"], root)

    if args.names:
        tokens = [t.strip() for t in args.names.split(",") if t.strip()]
        recs = []
        for t in tokens:
            if deadline_ts and time.monotonic() >= deadline_ts:
                print("  Deadline reached — stopping (remaining tokens not processed)."); break
            try:
                recs.append(process_one(svc, root, pool, universe, fund, results, ridx, t,
                                        interactive=args.interactive,
                                        do_backfill=not args.no_backfill,
                                        deadline_ts=deadline_ts))
            except AllBucketsExhausted as exc:
                print(f"  All Gemini buckets exhausted — stopping. ({exc})")
                break
            except DeadlineReached:
                print(f"  Deadline reached mid-'{t}' — will resume from cache next run."); break
            except FatalCallError as exc:
                print(f"  Fatal error for '{t}' (bad prompt/auth) — skipping. ({exc})")
        if recs:
            if args.open:
                for r in recs:
                    open_report_local(r["_report_md"], r["_slug"],
                                      r.get("name",""), r.get("symbol",""), r.get("isin",""))
            update_index(svc, root, [_strip_internal(r) for r in recs])
        return

    queue = _dedup_queue(_read_parquet(svc, DRIVE["queue"], root))
    if queue.empty or "status" not in queue:
        print("Queue empty. Nothing to do."); return
    pending = queue[queue["status"] == "pending"]
    if pending.empty:
        print("No pending companies."); return

    def _mark(token, added_at, status, **fields):
        """Locked re-read-merge: set this token's row status on the CURRENT queue, so a
        concurrent enqueue is never clobbered (the bug that re-populated the queue)."""
        def m(df):
            mask = (df["token"].astype(str) == str(token))
            if "added_at" in df.columns:
                mask &= (df["added_at"].astype(str) == str(added_at))
            if not mask.any():       # row vanished (e.g. cleaned) — re-add it as resolved
                row = dict(token=token, status=status, added_at=added_at, **fields)
                return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.loc[mask, "status"] = status
            for k, v in fields.items():
                df.loc[mask, k] = v
            return df
        queue_update(svc, root, m, owner="drain")

    recs = []
    for i in pending.index:
        token, added = queue.at[i, "token"], queue.at[i, "added_at"]
        if deadline_ts and time.monotonic() >= deadline_ts:
            print("  Deadline reached — leaving remaining companies pending for next run.")
            break
        try:
            rec = process_one(svc, root, pool, universe, fund, results, ridx,
                              token, interactive=args.interactive,
                              do_backfill=not args.no_backfill, deadline_ts=deadline_ts)
            recs.append(rec)
            # Mark done on the CURRENT queue (locked) AFTER each company, so a kill (no
            # time restriction) loses at most the in-flight one and concurrent enqueues
            # are preserved.
            _mark(token, added, "done", done_at=dt.datetime.now().isoformat())
            update_index(svc, root, [_strip_internal(rec)])
            if args.open:
                open_report_local(rec["_report_md"], rec["_slug"],
                                  rec.get("name",""), rec.get("symbol",""), rec.get("isin",""))
        except AllBucketsExhausted as exc:
            print(f"  All Gemini buckets exhausted — stopping queue drain. ({exc})")
            break
        except DeadlineReached:
            print(f"  Deadline reached mid-'{token}' — pending; resumes next run.")
            break
        except FatalCallError as exc:
            print(f"  FATAL (this company): {_safe_err(exc)}")
            _mark(token, added, "error", error=_safe_err(exc))
        except Exception as e:
            print(f"    FAILED {token}: {_safe_err(e)}")
            _mark(token, added, "error", error=_safe_err(e))

    print(f"Done. {len(recs)} report(s) generated.")


if __name__ == "__main__":
    main()
