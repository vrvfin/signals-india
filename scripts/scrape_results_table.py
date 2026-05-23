"""
Phase 2 / Stage A — Quarterly results numbers (Screener /results/latest/).

Scrapes Screener's "Latest quarterly results" page — the structured numbers
table (Sales / EBIDT / Net profit / EPS x last quarters + YoY) — and upserts
them into company_repo/_index/results.parquet.

No LLM: the numbers are already structured. Pairs with ingest_company_docs.py's
`results` announcement feed (which gives the publish DATE); this gives the
NUMBERS. Join on slug for "best results of the day".

Idempotent: re-scraping a company-quarter just updates its row.

Usage:
    python scripts/scrape_results_table.py
    python scripts/scrape_results_table.py --max-pages 20   # results season
"""

from __future__ import annotations

import argparse
import io
import os
import re
import time
from datetime import datetime
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

SCOPES = ["https://www.googleapis.com/auth/drive"]
RESULTS_URL = "https://www.screener.in/results/latest/"
MAX_PAGES = 10

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

OUT_COLS = ["slug", "isin", "company_name", "metric",
            "latest_q", "latest_val", "prev_q", "prev_val",
            "yearago_q", "yearago_val", "yoy_pct", "qoq_pct", "scraped_at"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive():
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    token_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
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


def upload_parquet(drive, folder_id, filename, df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="application/octet-stream", resumable=False)
    fid = find_file(drive, folder_id, filename)
    if fid:
        drive.files().update(fileId=fid, media_body=media).execute()
    else:
        drive.files().create(body={"name": filename, "parents": [folder_id]},
                             media_body=media, fields="id").execute()


# ---------- Parsing ----------

def _num(text) -> float | None:
    """Clean a Screener numeric cell: strips Rs, commas, %, arrows, spaces."""
    if text is None:
        return None
    s = re.sub(r"[^0-9.\-]", "", str(text).replace(",", ""))
    if s in ("", "-", ".", "-."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _yoy(text) -> float | None:
    """A YoY cell like '⇡ 10%' or '⇣ 5%' -> signed percent."""
    if text is None:
        return None
    s = str(text)
    val = _num(s)
    if val is None:
        return None
    if "⇣" in s or "-" in s:
        return -abs(val)
    return abs(val)


def screener_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    cookie = os.environ.get("SCREENER_SESSION_COOKIE", "").strip()
    if cookie:
        s.cookies.set("sessionid", cookie, domain=".screener.in")
    return s


def parse_results_page(html: str, run_ts: str) -> list[dict]:
    """One /results/latest/ page -> list of (company, metric) result rows."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    
    # Target every structured data table directly
    for table in soup.select("table.data-table"):
        
        # 1. Look upwards to find the container card holding this table
        card_container = table.find_parent("div", class_="responsive-holder")
        if not card_container:
            continue
            
        # 2. Find the header layout block immediately preceding this table card container
        # Screener places the company name anchor inside the flex-row container directly above it
        header_block = card_container.find_previous("div", class_="flex-row")
        if not header_block:
            continue
            
        # 3. Secure the specific company anchor element inside that header block
        comp_a = header_block.select_one("a[href*='/company/']")
        if not comp_a:
            # Fallback if structural classes shift slightly: lookup nearest preceding company link
            comp_a = table.find_previous("a", href=lambda x: x and "/company/" in x)
            
        if comp_a is None:
            continue
            
        # Extract ticker code/slug without trailing routing endpoints
        m = re.search(r"/company/([^/]+)", comp_a.get("href", ""))
        slug = m.group(1).strip() if m else ""
        name = comp_a.get_text(strip=True)

        trs = table.select("tr")
        if not trs:
            continue
            
        header = [c.get_text(" ", strip=True) for c in trs[0].select("th, td")]
        qlabels = header[2:] if len(header) > 2 else []
        latest_q = qlabels[0] if len(qlabels) > 0 else ""
        prev_q = qlabels[1] if len(qlabels) > 1 else ""
        yearago_q = qlabels[2] if len(qlabels) > 2 else ""

        for tr in trs[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.select("th, td")]
            if len(cells) < 3 or not cells[0]:
                continue
            metric = cells[0]
            yoy = _yoy(cells[1]) if len(cells) > 1 else None
            latest_val = _num(cells[2]) if len(cells) > 2 else None
            prev_val = _num(cells[3]) if len(cells) > 3 else None
            yearago_val = _num(cells[4]) if len(cells) > 4 else None
            
            qoq = None
            if latest_val is not None and prev_val not in (None, 0):
                qoq = round((latest_val / prev_val - 1) * 100, 2)
                
            rows.append({
                "slug": slug, "isin": "", "company_name": name, "metric": metric,
                "latest_q": latest_q, "latest_val": latest_val,
                "prev_q": prev_q, "prev_val": prev_val,
                "yearago_q": yearago_q, "yearago_val": yearago_val,
                "yoy_pct": yoy, "qoq_pct": qoq, "scraped_at": run_ts,
            })
    return rows

def load_slug_isin_map(drive, index_id) -> dict:
    fid = find_file(drive, index_id, "company_universe.csv")
    if not fid:
        return {}
    try:
        uni = pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
        m = {}
        for _, r in uni.iterrows():
            isin = str(r.get("isin", ""))
            if str(r.get("nse_symbol", "")):
                m[str(r["nse_symbol"])] = isin
            if str(r.get("bse_code", "")) and str(r["bse_code"]) != "nan":
                m[str(r["bse_code"])] = isin
        return m
    except Exception:
        return {}


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    args = parser.parse_args()

    print("Phase 2 / Stage A — Quarterly results table scrape")
    print("-" * 56)

    # CRITICAL FIX: Load the environment file before initiating the session
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    run_ts = datetime.now().isoformat(timespec="seconds")
    session = screener_session()

    new_rows, seen_keys = [], set()
    for page in range(1, args.max_pages + 1):
        try:
            r = session.get(RESULTS_URL,
                            params={"page": page} if page > 1 else None, timeout=30)
        except requests.RequestException as e:
            log(f"page {page}: network error {str(e)[:90]}")
            break
        if r.status_code != 200:
            log(f"page {page}: HTTP {r.status_code} — stopping.")
            break
        page_rows = parse_results_page(r.text, run_ts)
        if not page_rows:
            log(f"page {page}: 0 companies — stopping.")
            break
        # detect pagination that just repeats page 1
        page_keys = {(x["slug"], x["metric"], x["latest_q"]) for x in page_rows}
        fresh = page_keys - seen_keys
        if not fresh:
            log(f"page {page}: all repeats — pagination exhausted, stopping.")
            break
        seen_keys |= page_keys
        new_rows.extend(page_rows)
        log(f"page {page}: {len(page_rows)} rows "
            f"({len({x['slug'] for x in page_rows})} companies)")
        time.sleep(0.6)

    if not new_rows:
        print("No results scraped. Check the Screener cookie.")
        return

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    # resolve ISIN
    slug2isin = load_slug_isin_map(drive, index_id)
    fresh = pd.DataFrame(new_rows)
    fresh["isin"] = fresh["slug"].map(lambda s: slug2isin.get(s, ""))

    # upsert into existing results.parquet, keyed by (slug, metric, latest_q)
    existing_id = find_file(drive, index_id, "results.parquet")
    if existing_id:
        try:
            old = pd.read_parquet(io.BytesIO(download_bytes(drive, existing_id)))
            combined = pd.concat([old, fresh], ignore_index=True)
        except Exception:
            combined = fresh
    else:
        combined = fresh
    combined = (combined.drop_duplicates(subset=["slug", "metric", "latest_q"],
                                         keep="last")
                .reset_index(drop=True))
    combined = combined[OUT_COLS]
    upload_parquet(drive, index_id, "results.parquet", combined)

    n_companies = fresh["slug"].nunique()
    print("-" * 56)
    print(f"Companies scraped this run : {n_companies}")
    print(f"Rows scraped this run      : {len(fresh)}")
    print(f"results.parquet total rows : {len(combined)}")
    print("Output: company_repo/_index/results.parquet")


if __name__ == "__main__":
    main()