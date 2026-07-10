"""
Phase 2 / Stage A — Company document ingestion.

Scrapes Screener's announcement user-filter feeds (concall transcripts, annual
reports), downloads each NEW document PDF to company_repo/<KEY>/documents/, and
records it in company_repo/_index/processing_queue.parquet with status=pending
for the Stage-B extractor to pick up.

Idempotent: a document already in the queue is skipped — safe to run many times
a day. <KEY> is the company's ISIN where resolvable (from company_universe.csv),
else its Screener symbol.

Usage:
    python scripts/ingest_company_docs.py                  # all feeds
    python scripts/ingest_company_docs.py --feed concall   # one feed
    python scripts/ingest_company_docs.py --max-pages 8    # look further back
    python scripts/ingest_company_docs.py --lookback-days 4
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import io
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# T12: shared _extract.lock so this Stage-A ingester's queue write can never race a
# backfill fetch's queue write (both rewrite processing_queue.parquet wholesale).
from _extractor_base import acquire_lock, release_lock, load_portfolio_isins

SCOPES = ["https://www.googleapis.com/auth/drive"]
_LOCK_NAME = "_extract.lock"
_LOCK_MAX_AGE_MIN = 360

# ============================================================
# CONFIG — Screener announcement user-filter feeds.
# filter_id is the number in the Screener URL:
#   https://www.screener.in/announcements/user-filters/<filter_id>/
# If you recreate the filters on Screener, update these IDs.
# ============================================================
FEEDS = {
    "concall":       {"filter_id": "76106",  "doc_type": "concall"},
    "annual_report": {"filter_id": "103635", "doc_type": "annual_report"},
    "presentation":  {"filter_id": "76295",  "doc_type": "presentation"},
    "rating":        {"filter_id": "215435", "doc_type": "rating"},
    "results":       {"path": "/announcements/results/", "doc_type": "results"},
}
MAX_PAGES = 1          # Screener's announcement feeds ignore ?page= (extra
                       # pages just repeat page 1). Same-day completeness comes
                       # from running the pipeline several times a day.
LOOKBACK_DAYS = 2      # ingest announcements within this many days of today

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive():
    import json
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    cred_data = json.loads(cs_path.read_text())

    # Service account key — no browser flow needed
    if cred_data.get("type") == "service_account":
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            str(cs_path), scopes=SCOPES
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # Saved OAuth token (Credentials.to_json() format) — has refresh_token directly
    if "refresh_token" in cred_data:
        creds = Credentials.from_authorized_user_file(str(cs_path), SCOPES)
        if not creds.valid:
            creds.refresh(Request())
            cs_path.write_text(creds.to_json())
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # Standard OAuth installed-app flow (proper client_secrets.json)
    token_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_or_create_subfolder(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def find_file(drive, folder_id, name):
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_bytes(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    return fh.getvalue()


def upload_bytes(drive, folder_id, filename, data, mimetype, existing_id=None):
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mimetype, resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ---------- Screener scraping ----------

def screener_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA,
                      "Accept-Language": "en-US,en;q=0.9"})
    cookie = os.environ.get("SCREENER_SESSION_COOKIE", "").strip()
    if cookie:
        s.cookies.set("sessionid", cookie, domain=".screener.in")
    return s


def fetch_feed_page(session, cfg, page) -> str:
    """A feed is either a Screener user-filter (filter_id) or a fixed path."""
    if cfg.get("path"):
        url = f"https://www.screener.in{cfg['path']}"
    else:
        url = f"https://www.screener.in/announcements/user-filters/{cfg['filter_id']}/"
    r = session.get(url, params={"page": page} if page > 1 else None, timeout=30)
    if r.status_code != 200:
        log(f"  HTTP {r.status_code} for {url} page {page}")
        return ""
    return r.text


def label_to_date(label: str, run_date: date) -> date:
    l = (label or "").strip().lower()
    if l == "today":
        return run_date
    if l == "yesterday":
        return run_date - timedelta(days=1)
    dt = pd.to_datetime(label, errors="coerce", dayfirst=True)
    return dt.date() if pd.notna(dt) else run_date


def parse_feed_html(html: str, doc_type: str, run_date: date) -> list[dict]:
    """Parse one announcements page into a list of announcement dicts."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    cards = soup.select("div.card.card-medium")
    # fall back: if no date cards, treat the whole page as one undated group
    groups = [(c, c.select_one("div.sub.margin-bottom-16")) for c in cards] \
        if cards else [(soup, None)]
    for card, label_el in groups:
        ann_date = label_to_date(label_el.get_text(strip=True) if label_el else "",
                                 run_date)
        for item in card.select("div.announcement-item"):
            comp_a = item.select_one("a.sub-link")
            if not comp_a:
                continue
            href = comp_a.get("href", "")
            m = re.search(r"/company/([^/]+)/", href)
            symbol = m.group(1).strip() if m else ""
            name_el = comp_a.select_one("span")
            company_name = name_el.get_text(strip=True) if name_el else symbol

            # the doc link is the <a> in the item that is not the company link
            doc_a = next((a for a in item.find_all("a") if a is not comp_a), None)
            if not doc_a:
                continue
            pdf_url = doc_a.get("href", "")
            desc_el = doc_a.select_one("div.sub")
            description = desc_el.get_text(" ", strip=True) if desc_el else ""
            if desc_el:
                desc_el.extract()                       # so title excludes desc
            title = doc_a.get_text(" ", strip=True)
            title = re.sub(r"\s*\d+\s*[hmd] ago\.?\s*$", "", title).strip()

            uid = re.search(r"Pname=([^&]+?)(?:\.pdf)?$", pdf_url)
            # NSE / other URLs have no Pname param — use a stable md5 so the
            # dedup key survives across runs (Python's hash() is per-process).
            doc_id = uid.group(1) if uid else hashlib.md5(
                pdf_url.encode()).hexdigest()[:20]

            rows.append({
                "doc_id": doc_id, "symbol": symbol, "company_name": company_name,
                "doc_type": doc_type, "title": title, "description": description,
                "announcement_date": ann_date, "pdf_url": pdf_url,
            })
    return rows


def _get_once(session, url, timeout=60):
    """One GET with the standard headers; None on network error."""
    try:
        return session.get(url, headers={"User-Agent": UA,
                                         "Referer": "https://www.bseindia.com/"},
                           timeout=timeout, allow_redirects=True)
    except requests.RequestException as e:
        log(f"    download error: {str(e)[:100]}")
        return None


def download_pdf(session, url) -> bytes | None:
    """Fetch a document PDF. BSE's Akamai serves an ~8KB HTML block page from
    AnnPdfOpen.aspx for REAL, existing PDFs (verified 2026-07-09: 4 of the last
    20 queue errors were fresh Q4FY26 concalls bounced this way — RALLIS/TANLA/
    EPIGRAL/CRAMC; cookie-refresh retries do NOT help). The SAME attachment is
    served un-blocked at xml-data/corpfiling/AttachLive/<Pname> (fallback
    AttachHis) — the endpoint the announcement pipeline already uses. So: try
    the stored URL once (fast path unchanged), and on a BSE non-PDF response
    rewrite AnnPdfOpen.aspx?Pname=X to AttachLive/X then AttachHis/X."""
    r = _get_once(session, url)
    if r is not None and r.status_code == 200 and r.content[:5].startswith(b"%PDF"):
        return r.content

    m = re.search(r"(?i)bseindia\.com/.*[?&]Pname=([^&]+)", str(url))
    if m:
        pname = m.group(1)
        for base in ("https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
                     "https://www.bseindia.com/xml-data/corpfiling/AttachHis/"):
            r2 = _get_once(session, base + pname, timeout=60)
            if r2 is not None and r2.status_code == 200 \
                    and r2.content[:5].startswith(b"%PDF"):
                log(f"    BSE AnnPdfOpen blocked — recovered via "
                    f"{base.rsplit('/', 2)[-2]}")
                return r2.content
            time.sleep(1)

    if r is None:
        return None
    if r.status_code != 200 or not r.content:
        log(f"    download HTTP {r.status_code}")
        return None
    log(f"    not a PDF (got {r.content[:20]!r})")
    return None


# ---------- Queue ----------

QUEUE_COLS = ["doc_id", "key", "isin", "symbol", "company_name", "doc_type",
              "title", "description", "announcement_date", "pdf_url",
              "drive_file_id", "status", "discovered_at", "processed_at",
              # Phase 3 T1: set only when a row is processed by a backfill run.
              "backfill_process_date",
              # Phase 3 T1: who ENQUEUED this row — "live" (Phase 2 recent feed) or
              # "backfill" (run_backfill/backfill_company_docs). Set at enqueue time
              # so each extractor only drains its own rows (live -> daily digest +
              # main key pool; backfill -> daily_backfill digest + BACKFILL pool).
              # A blank/absent value is treated as "live" (legacy rows).
              "source",
              # T12: natural-grain period label of the doc ("Q2FY25" concall/
              # results/presentation, "FY24" annual_report, ISO date rating).
              # Stamped at fetch time; used by the coverage view + supersede grain.
              "period",
              # T12 (reserved for Stage C): sha256 of the document bytes where
              # available — cross-phase content identity. Blank until populated.
              "content_sha256"]


def load_queue(drive, index_id) -> pd.DataFrame:
    fid = find_file(drive, index_id, "processing_queue.parquet")
    if not fid:
        return pd.DataFrame(columns=QUEUE_COLS)
    try:
        df = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
        for c in QUEUE_COLS:
            if c not in df.columns:
                df[c] = None
        return df
    except Exception as e:
        log(f"  WARNING: could not read existing queue ({str(e)[:80]}) — fresh.")
        return pd.DataFrame(columns=QUEUE_COLS)


def save_queue(drive, index_id, df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="application/octet-stream", resumable=False)
    fid = find_file(drive, index_id, "processing_queue.parquet")
    if fid:
        drive.files().update(fileId=fid, media_body=media).execute()
    else:
        drive.files().create(body={"name": "processing_queue.parquet",
                                   "parents": [index_id]},
                             media_body=media, fields="id").execute()


def load_symbol_isin_map(drive, index_id) -> dict:
    fid = find_file(drive, index_id, "company_universe.csv")
    if not fid:
        return {}
    try:
        uni = pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
        uni = uni[uni["nse_symbol"].astype(str) != ""]
        return dict(zip(uni["nse_symbol"].astype(str), uni["isin"].astype(str)))
    except Exception:
        return {}


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", choices=list(FEEDS.keys()), default=None,
                        help="Process only one feed (default: all).")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()

    print("Phase 2 / Stage A — Company document ingestion")
    print("-" * 56)

    run_date = date.today()
    oldest = run_date - timedelta(days=args.lookback_days)
    feeds = {args.feed: FEEDS[args.feed]} if args.feed else FEEDS

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    # T12 Phase-2 safety: serialize the queue write via the global _extract.lock.
    # On contention exit cleanly — the next scheduled run re-ingests (the feeds are
    # a 2-day lookback, so a skipped slot loses nothing).
    # Phase-2 priority: wait up to 15 min for the lock (backfill yields to us
    # within ~one document) instead of skipping the ingest on contention.
    if not acquire_lock(drive, index_id, _LOCK_NAME, "ingest",
                        max_age_min=_LOCK_MAX_AGE_MIN, wait_min=15):
        log("Lock unavailable after wait — exiting cleanly.")
        return
    atexit.register(release_lock, drive, index_id, _LOCK_NAME)

    queue = load_queue(drive, index_id)
    # Dedup key: (doc_id, announcement_date) — not just doc_id.
    # Same URL reappearing on a different date (e.g. MP3 on day 1, PDF on day 2)
    # is treated as a new document and processed independently.
    if not queue.empty:
        known_keys: set[str] = set(
            queue["doc_id"].astype(str) + "__"
            + queue["announcement_date"].astype(str).str[:10]
        )
    else:
        known_keys = set()
    sym2isin = load_symbol_isin_map(drive, index_id)
    # T12: extract_results/rating/presentation/annual_report process PORTFOLIO
    # companies only; extract_concall processes ALL companies. So we ingest those
    # four types for portfolio names only — downloading a non-PF copy just creates
    # a PDF no extractor drains, which cleanup deletes in 2 days, leaving an orphaned
    # 'pending' row. Concall is always ingested (whole universe). Fail-safe: if the
    # portfolio list can't be loaded we ingest everything (old behaviour).
    portfolio_isins = load_portfolio_isins(drive, folder_id) or set()
    log(f"Queue has {len(queue)} existing rows · universe map: {len(sym2isin)} symbols "
        f"· portfolio: {len(portfolio_isins)} ISINs (non-concall ingested PF-only)")

    session = screener_session()
    new_rows = []
    counts = {"seen": 0, "new": 0, "downloaded": 0, "skipped_old": 0,
              "dup": 0, "download_fail": 0, "ingest_error": 0, "skipped_nonpf": 0}

    for feed_name, cfg in feeds.items():
        src = cfg.get("path") or f"filter {cfg.get('filter_id')}"
        log(f"Feed '{feed_name}' ({src})")
        for page in range(1, args.max_pages + 1):
            html = fetch_feed_page(session, cfg, page)
            if not html:
                break
            anns = parse_feed_html(html, cfg["doc_type"], run_date)
            if not anns:
                if page == 1:
                    log("  page 1 had 0 announcements — check the Screener "
                        "cookie / filter id.")
                break
            page_all_old = True
            for a in anns:
                counts["seen"] += 1
                if a["announcement_date"] >= oldest:
                    page_all_old = False
                else:
                    counts["skipped_old"] += 1
                    continue
                dedup_key = f"{a['doc_id']}__{str(a['announcement_date'])[:10]}"
                if dedup_key in known_keys:
                    counts["dup"] += 1
                    continue

                isin = sym2isin.get(a["symbol"], "")
                key = isin if isin else a["symbol"]

                # T12: PF-only ingest for non-concall types (see note above). Concall
                # is always ingested. Skipped non-PF rows are never queued, so no
                # orphan is created; if the company joins the portfolio later, the
                # 2-day feed lookback re-surfaces its recent docs.
                if (a["doc_type"] != "concall" and portfolio_isins
                        and isin not in portfolio_isins):
                    counts["skipped_nonpf"] += 1
                    continue

                counts["new"] += 1

                # Per-document isolation: a failed download/upload for ONE doc
                # must never abort the feed. On hard error we skip the row
                # WITHOUT marking it known, so it is retried on the next run.
                # Rule: never stop creating the queue.
                drive_file_id, status = "", "pending"
                try:
                    pdf = download_pdf(session, a["pdf_url"])
                    if pdf is None:
                        counts["download_fail"] += 1
                        status = "download_failed"
                    else:
                        comp_id = get_or_create_subfolder(drive, repo_id, key)
                        docs_id = get_or_create_subfolder(drive, comp_id, "documents")
                        fname = f"{a['doc_type']}__{a['announcement_date']}__{a['doc_id']}.pdf"
                        drive_file_id = upload_bytes(drive, docs_id, fname, pdf,
                                                     "application/pdf")
                        counts["downloaded"] += 1
                        log(f"  + {a['symbol']:<14} {a['doc_type']:<14} "
                            f"{a['announcement_date']}")
                except Exception as exc:
                    counts.setdefault("ingest_error", 0)
                    counts["ingest_error"] += 1
                    counts["new"] -= 1   # not actually queued
                    log(f"  ! {a['symbol']:<14} ingest error "
                        f"({str(exc)[:80]}) — will retry next run")
                    continue

                known_keys.add(dedup_key)
                new_rows.append({
                    "doc_id": a["doc_id"], "key": key, "isin": isin,
                    "symbol": a["symbol"], "company_name": a["company_name"],
                    "doc_type": a["doc_type"], "title": a["title"],
                    "description": a["description"],
                    "announcement_date": str(a["announcement_date"]),
                    "pdf_url": a["pdf_url"], "drive_file_id": drive_file_id,
                    "status": status,
                    "discovered_at": datetime.now().isoformat(timespec="seconds"),
                    "processed_at": "",
                    "source": "live",          # Phase 2 recent-feed origin
                })
                time.sleep(0.4)            # polite to BSE
            if page_all_old:
                break
            time.sleep(0.6)

    if new_rows:
        queue = pd.concat([queue, pd.DataFrame(new_rows)], ignore_index=True)
        save_queue(drive, index_id, queue)

    print("-" * 56)
    print(f"Announcements seen     : {counts['seen']}")
    print(f"Already in queue (dup) : {counts['dup']}")
    print(f"Outside lookback window: {counts['skipped_old']}")
    print(f"Non-PF (non-concall)   : {counts['skipped_nonpf']}  (skipped — extractor is PF-only)")
    print(f"New documents queued   : {counts['new']}")
    print(f"  PDFs downloaded      : {counts['downloaded']}")
    print(f"  download failures    : {counts['download_fail']}")
    print(f"  ingest errors (retry): {counts['ingest_error']}")
    print(f"Queue total rows       : {len(queue)}")
    print("Output: company_repo/_index/processing_queue.parquet")


if __name__ == "__main__":
    main()
