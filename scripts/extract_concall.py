"""
Phase 2 / Stage B — Concall extraction via Gemini.

Consumes pending concall entries from the Drive processing queue, runs each PDF
through Gemini (File API), extracts structured quarterly facts and guidance into
parquet tables, appends a markdown brief to the company's company_page.md, and
marks queue rows done.

On Gemini 429 with all keys exhausted: stops cleanly (exit 0) so the next
scheduled run resumes from remaining pending rows.

Usage:
    python scripts/extract_concall.py
    python scripts/extract_concall.py --limit 5       # at most N rows
    python scripts/extract_concall.py --dry-run        # parse, no Drive writes
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
GEMINI_MODEL = "gemini-2.5-flash"

# ---- Output toggles ----
OUTPUT_COMPANY_MD   = True   # append to company_repo/<ISIN>/company_page.md
OUTPUT_DAY_MD       = True   # append to company_repo/_daily/concall_DD_MMMYYYY.md
OUTPUT_COMPANY_DOCX = False  # .docx alongside company_page.md  [Stage C]
OUTPUT_DAY_DOCX     = False  # .docx alongside day page          [Stage C]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ------------------------------------------------------------------ #
#  Drive helpers  (same pattern as ingest_company_docs.py)            #
# ------------------------------------------------------------------ #

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
            # Persist refreshed token back to the same file
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


def download_bytes(drive, file_id) -> bytes:
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


# ------------------------------------------------------------------ #
#  Queue helpers                                                       #
# ------------------------------------------------------------------ #

QUEUE_COLS = ["doc_id", "key", "isin", "symbol", "company_name", "doc_type",
              "title", "description", "announcement_date", "pdf_url",
              "drive_file_id", "status", "discovered_at", "processed_at"]


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
        log(f"  WARNING: could not read queue ({str(e)[:80]}) — returning empty.")
        return pd.DataFrame(columns=QUEUE_COLS)


def save_queue(drive, index_id, df: pd.DataFrame) -> None:
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


# ------------------------------------------------------------------ #
#  Parquet schemas and upsert helpers                                  #
# ------------------------------------------------------------------ #

QFACTS_COLS = [
    "isin", "symbol", "company_name", "quarter", "fy_year",
    "revenue_q", "ebitda_q", "pat_q", "margin_pct", "volume_q", "capacity_q",
    "revenue_12m", "pat_12m", "processed_at", "source_doc_id",
]

GUIDANCE_COLS = [
    "isin", "symbol", "company_name", "quarter", "metric",
    "guidance_type", "horizon_fy", "value", "unit", "cagr_pct", "notes",
    "processed_at", "source_doc_id",
]


def _load_parquet(drive, index_id, filename, cols) -> pd.DataFrame:
    fid = find_file(drive, index_id, filename)
    if not fid:
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols]
    except Exception as e:
        log(f"  WARNING: could not read {filename} ({str(e)[:80]}) — fresh.")
        return pd.DataFrame(columns=cols)


def _save_parquet(drive, index_id, filename, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="application/octet-stream", resumable=False)
    fid = find_file(drive, index_id, filename)
    if fid:
        drive.files().update(fileId=fid, media_body=media).execute()
    else:
        drive.files().create(body={"name": filename, "parents": [index_id]},
                             media_body=media, fields="id").execute()


def upsert_facts(drive, index_id, facts: dict) -> None:
    df = _load_parquet(drive, index_id, "quarterly_facts.parquet", QFACTS_COLS)
    mask = (
        (df["isin"].astype(str) == str(facts["isin"])) &
        (df["quarter"].astype(str) == str(facts["quarter"])) &
        (df["source_doc_id"].astype(str) == str(facts["source_doc_id"]))
    )
    df = df[~mask]
    new_row = pd.DataFrame([{c: facts.get(c) for c in QFACTS_COLS}])
    df = pd.concat([df, new_row], ignore_index=True)
    _save_parquet(drive, index_id, "quarterly_facts.parquet", df)


def upsert_guidance(drive, index_id, guidance_rows: list[dict]) -> None:
    if not guidance_rows:
        return
    df = _load_parquet(drive, index_id, "guidance_tracker.parquet", GUIDANCE_COLS)
    source_doc_id = str(guidance_rows[0]["source_doc_id"])
    df = df[df["source_doc_id"].astype(str) != source_doc_id]
    new_df = pd.DataFrame([{c: r.get(c) for c in GUIDANCE_COLS} for r in guidance_rows])
    df = pd.concat([df, new_df], ignore_index=True)
    _save_parquet(drive, index_id, "guidance_tracker.parquet", df)


# ------------------------------------------------------------------ #
#  company_page.md helper                                              #
# ------------------------------------------------------------------ #

def _day_filename(announcement_date: str) -> str:
    """Return e.g. 'concall_26_may2026.md' from '2026-05-26'."""
    try:
        dt = datetime.strptime(str(announcement_date)[:10], "%Y-%m-%d")
        return f"concall_{dt.day:02d}_{dt.strftime('%b').lower()}{dt.year}.md"
    except Exception:
        return f"concall_{str(announcement_date)[:10].replace('-', '_')}.md"


def append_day_page(drive, repo_id, announcement_date: str, symbol: str,
                    company_name: str, quarter: str, content: str) -> None:
    """Append this company's analysis to the daily digest file in _daily/.

    Format:
        # Daily Concall Digest — 2026-05-26
        *Total: 12 concalls — last updated 26 May 2026 14:30 IST*

        ---
        ## concall_1 — TCS · Tata Consultancy Services | Q4 FY26
        ...
        ---
        ## concall_12 — INFY · Infosys | Q4 FY26
        ...

    The numbered heading (concall_N) lets you Ctrl+F / search for a specific
    entry by number.  The Total line in the header is rewritten on every append.
    """
    daily_id = get_or_create_subfolder(drive, repo_id, "_daily")
    fname = _day_filename(announcement_date)
    now_str = datetime.now().strftime("%d %b %Y %H:%M")

    fid = find_file(drive, daily_id, fname)
    if fid:
        existing = download_bytes(drive, fid).decode("utf-8", errors="replace")

        # Count existing numbered entries to assign the next number
        existing_count = len(re.findall(r'^## concall_\d+', existing, re.MULTILINE))
        new_num = existing_count + 1

        # Rewrite the Total line in the header (handles both new and old format)
        new_total = (f"*Total: {new_num} concall{'s' if new_num != 1 else ''}"
                     f" — last updated {now_str} IST*")
        updated = re.sub(r'\*Total: \d+ concalls?[^*]*\*', new_total, existing)
        if updated == existing:
            # Old format file (no Total line yet) — insert after the first line
            first_nl = updated.find('\n')
            if first_nl != -1:
                updated = (updated[:first_nl + 1]
                           + new_total + '\n'
                           + updated[first_nl + 1:])

        entry = (
            f"\n\n---\n"
            f"## concall_{new_num} — {symbol} · {company_name} | {quarter}\n\n"
            + content
        )
        upload_bytes(drive, daily_id, fname,
                     (updated + entry).encode("utf-8"), "text/markdown",
                     existing_id=fid)
    else:
        header = (
            f"# Daily Concall Digest — {str(announcement_date)[:10]}\n"
            f"*Total: 1 concall — last updated {now_str} IST*\n"
        )
        entry = (
            f"\n\n---\n"
            f"## concall_1 — {symbol} · {company_name} | {quarter}\n\n"
            + content
        )
        upload_bytes(drive, daily_id, fname,
                     (header + entry).encode("utf-8"), "text/markdown")


def append_company_page(drive, repo_id, key: str, content: str,
                        doc_title: str, quarter: str) -> None:
    if not key:
        log("  WARN: empty key — skipping company_page.md update")
        return
    comp_id = get_or_create_subfolder(drive, repo_id, key)
    header = (
        f"\n\n---\n## {quarter} Concall — {doc_title}\n"
        f"*Processed: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
    )
    fid = find_file(drive, comp_id, "company_page.md")
    if fid:
        existing = download_bytes(drive, fid).decode("utf-8", errors="replace")
        updated = existing + header + content
        upload_bytes(drive, comp_id, "company_page.md",
                     updated.encode("utf-8"), "text/markdown", existing_id=fid)
    else:
        initial = f"# {key} — Company Intelligence\n" + header + content
        upload_bytes(drive, comp_id, "company_page.md",
                     initial.encode("utf-8"), "text/markdown")


# ------------------------------------------------------------------ #
#  Gemini key pool with round-robin rotation on 429                   #
# ------------------------------------------------------------------ #

class RateLimitExhausted(Exception):
    pass


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc)
    return any(t in s for t in ("429", "503", "Resource has been exhausted",
                                "RESOURCE_EXHAUSTED", "UNAVAILABLE", "quota"))


class GeminiKeyPool:
    def __init__(self, api_keys: list[str]):
        self.keys = api_keys
        self.idx = 0

    def call(self, pdf_bytes: bytes, prompt: str, display_name: str) -> str:
        """Generate content using inline PDF bytes. Rotates keys on 429."""
        import base64
        b64 = base64.standard_b64encode(pdf_bytes).decode()
        backoff = 30
        total_attempts = len(self.keys)
        attempted = 0

        while attempted < total_attempts:
            client = genai.Client(api_key=self.keys[self.idx])
            try:
                log(f"  Generating response (key {self.idx + 1}/{len(self.keys)})...")
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                mime_type="application/pdf",
                                data=b64,
                            )
                        ),
                        genai_types.Part.from_text(text=prompt),
                    ],
                    config=genai_types.GenerateContentConfig(temperature=0.1),
                )
                self.idx = (self.idx + 1) % len(self.keys)
                return response.text
            except Exception as exc:
                if _is_rate_limit(exc):
                    attempted += 1
                    next_idx = (self.idx + 1) % len(self.keys)
                    log(f"  429 on key {self.idx + 1}/{len(self.keys)} "
                        f"(attempt {attempted}/{total_attempts}) — "
                        f"waiting {backoff}s, next key {next_idx + 1}...")
                    time.sleep(backoff)
                    self.idx = next_idx
                    backoff = min(backoff * 2, 120)
                    continue
                raise
        raise RateLimitExhausted(
            f"All {len(self.keys)} Gemini keys exhausted after backoff"
        )


# ------------------------------------------------------------------ #
#  Markdown table parser                                               #
# ------------------------------------------------------------------ #

def _extract_md_tables(text: str) -> list[dict]:
    """Return list of {headers: [...], rows: [[...], ...]} from all tables in text.

    Handles both fenced tables (leading |) and unfenced (cells separated by |
    without a leading pipe), which is what Gemini typically outputs.
    """
    text = text.replace("||", "|")

    def _is_sep(line: str) -> bool:
        """True for markdown separator rows like '--- | --- | ---'."""
        parts = [p.strip() for p in line.strip("|").split("|")]
        return bool(parts) and all(re.match(r"^[\-:\s]+$", p) for p in parts if p)

    def _is_table_row(line: str) -> bool:
        return line.count("|") >= 1 and not _is_sep(line)

    def _parse_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]

    tables: list[dict] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if _is_table_row(line):
            block = []
            while i < len(lines):
                l = lines[i].strip()
                if _is_table_row(l) or _is_sep(l):
                    block.append(l)
                    i += 1
                else:
                    break
            non_sep = [l for l in block if not _is_sep(l)]
            if len(non_sep) >= 2:
                headers = _parse_row(non_sep[0])
                rows = []
                for row_line in non_sep[1:]:
                    cells = _parse_row(row_line)
                    while len(cells) < len(headers):
                        cells.append("")
                    rows.append(cells[: len(headers)])
                if headers:
                    tables.append({"headers": headers, "rows": rows})
        else:
            i += 1
    return tables


def _clean_val(s: str) -> str:
    v = (s or "").strip()
    return v if v and v not in ("-", "–", "") else "NA"


def _try_float(s: str):
    if not s or s == "NA":
        return None
    cleaned = re.sub(r"[,%₹$]", "", s).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


# Maps row label tokens → canonical metric name
_METRIC_ALIASES: list[tuple[str, str]] = [
    ("revenue", "revenue"),
    ("ebidta", "ebitda"),
    ("ebitda", "ebitda"),
    ("ebita",  "ebitda"),
    ("ebit",   "ebitda"),
    ("pat",    "pat"),
    ("margin", "margin"),
    ("volume", "volume"),
    ("capacity", "capacity"),
]


def _identify_metric(label: str) -> str | None:
    low = label.lower().strip()
    for token, name in _METRIC_ALIASES:
        if token in low:
            return name
    return None


def parse_gemini_response(
    text: str, row: pd.Series
) -> tuple[dict, list[dict]]:
    """Parse Gemini markdown response into (quarterly_facts dict, guidance_rows list)."""
    now_str = datetime.now().isoformat(timespec="seconds")
    isin = str(row.get("isin") or "")
    symbol = str(row.get("symbol") or "")
    company_name = str(row.get("company_name") or "")
    source_doc_id = str(row.get("doc_id") or "")

    facts: dict = {
        "isin": isin, "symbol": symbol, "company_name": company_name,
        "quarter": "", "fy_year": "",
        "revenue_q": None, "ebitda_q": None, "pat_q": None,
        "margin_pct": None, "volume_q": None, "capacity_q": None,
        "revenue_12m": None, "pat_12m": None,
        "processed_at": now_str, "source_doc_id": source_doc_id,
    }
    guidance_rows: list[dict] = []

    tables = _extract_md_tables(text)
    if not tables:
        return facts, guidance_rows

    # ---- Table 1: actuals (Q + 12M) + explicit FY guidance ----
    t1 = tables[0]
    hdrs = t1["headers"]

    # Locate quarter column: first header matching Q\d FY\d+
    q_col = next((i for i, h in enumerate(hdrs)
                  if re.search(r"Q\d\s+FY\d{2,4}", h, re.IGNORECASE)), None)
    quarter_name = ""
    if q_col is not None:
        m = re.search(r"(Q\d\s+FY\d{2,4})", hdrs[q_col], re.IGNORECASE)
        if m:
            quarter_name = m.group(1).strip()
            facts["quarter"] = quarter_name
            fy_m = re.search(r"FY(\d{2,4})", quarter_name, re.IGNORECASE)
            if fy_m:
                facts["fy_year"] = f"FY{fy_m.group(1)}"

    # 12M column
    m12_col = next((i for i, h in enumerate(hdrs)
                    if re.search(r"12\s*[Mm]", h)), None)

    # Explicit guidance columns → FY label
    fy_explicit: dict[str, int] = {}
    for i, h in enumerate(hdrs):
        if "explicit" in h.lower() or "guidance" in h.lower():
            m = re.search(r"FY(\d{2,4})", h, re.IGNORECASE)
            if m:
                fy_explicit[f"FY{m.group(1)}"] = i

    # Q-fact field map: metric → schema column
    Q_FIELD = {"revenue": "revenue_q", "ebitda": "ebitda_q", "pat": "pat_q",
               "margin": "margin_pct", "volume": "volume_q", "capacity": "capacity_q"}
    M12_FIELD = {"revenue": "revenue_12m", "pat": "pat_12m"}

    for cells in t1["rows"]:
        if not cells:
            continue
        metric = _identify_metric(cells[0])
        if not metric:
            continue

        if q_col is not None and q_col < len(cells) and metric in Q_FIELD:
            facts[Q_FIELD[metric]] = _clean_val(cells[q_col])

        if m12_col is not None and m12_col < len(cells) and metric in M12_FIELD:
            facts[M12_FIELD[metric]] = _clean_val(cells[m12_col])

        for fy, col in fy_explicit.items():
            if col < len(cells):
                guidance_rows.append({
                    "isin": isin, "symbol": symbol, "company_name": company_name,
                    "quarter": quarter_name, "metric": metric,
                    "guidance_type": "explicit", "horizon_fy": fy,
                    "value": _clean_val(cells[col]), "unit": "",
                    "cagr_pct": None, "notes": "",
                    "processed_at": now_str, "source_doc_id": source_doc_id,
                })

    # ---- Table 2: derived CAGR guidance ----
    if len(tables) >= 2:
        t2 = tables[1]
        hdrs2 = t2["headers"]
        fy_derived: dict[str, int] = {}
        for i, h in enumerate(hdrs2):
            if i == 0:
                continue
            m = re.search(r"FY(\d{2,4})", h, re.IGNORECASE)
            if m:
                fy_derived[f"FY{m.group(1)}"] = i

        for cells in t2["rows"]:
            if not cells:
                continue
            metric = _identify_metric(cells[0])
            if not metric:
                continue
            for fy, col in fy_derived.items():
                if col < len(cells):
                    raw = _clean_val(cells[col])
                    guidance_rows.append({
                        "isin": isin, "symbol": symbol, "company_name": company_name,
                        "quarter": quarter_name, "metric": metric,
                        "guidance_type": "derived", "horizon_fy": fy,
                        "value": raw, "unit": "%",
                        "cagr_pct": _try_float(raw), "notes": "",
                        "processed_at": now_str, "source_doc_id": source_doc_id,
                    })

    return facts, guidance_rows


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 / Stage B — Concall extraction via Gemini"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N pending rows then stop.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run Gemini but skip all Drive writes.")
    args = parser.parse_args()

    print("Phase 2 / Stage B — Concall extraction via Gemini")
    print("-" * 56)

    # .env is loaded inside get_drive(); call it first so all env vars are set
    drive = get_drive()

    # Load Gemini API keys — accepts GEMINI_API_KEY_1..N or plain GEMINI_API_KEY
    api_keys = [
        v for _, v in sorted(
            ((k, v) for k, v in os.environ.items()
             if re.match(r"GEMINI_API_KEY_\d+$", k) and v.strip()),
            key=lambda kv: kv[0],
        )
    ]
    plain = os.environ.get("GEMINI_API_KEY", "").strip()
    if plain and plain not in api_keys:
        api_keys.append(plain)
    if not api_keys:
        print("ERROR: no GEMINI_API_KEY or GEMINI_API_KEY_* found in .env")
        sys.exit(1)
    log(f"Loaded {len(api_keys)} Gemini API key(s)")

    gemini = GeminiKeyPool(api_keys)

    # Load prompt
    prompt_path = Path(__file__).resolve().parent / "concall_prompt.txt"
    if not prompt_path.exists():
        print(f"ERROR: prompt file not found: {prompt_path}")
        sys.exit(1)
    prompt = prompt_path.read_text(encoding="utf-8")

    # Drive folder layout
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    # Load queue and filter to pending concalls
    queue = load_queue(drive, index_id)
    pending_mask = (queue["status"] == "pending") & (queue["doc_type"] == "concall")
    pending_idx = queue.index[pending_mask].tolist()

    log(f"Queue: {len(queue)} total rows, {len(pending_idx)} pending concalls")

    if args.limit:
        pending_idx = pending_idx[: args.limit]

    counts = {"processed": 0, "error": 0, "skipped": 0}

    for queue_idx in pending_idx:
        row = queue.loc[queue_idx]
        label = f"{row.get('symbol', '?')!s:<14} {str(row.get('title', ''))[:55]}"
        log(f"Processing: {label}")

        drive_fid = str(row.get("drive_file_id") or "").strip()
        if not drive_fid:
            log("  SKIP: no drive_file_id in queue row")
            counts["skipped"] += 1
            continue

        try:
            # 1. Download PDF from Drive
            pdf_bytes = download_bytes(drive, drive_fid)
            log(f"  PDF: {len(pdf_bytes):,} bytes")

            display_name = f"{row.get('symbol', 'DOC')}_{str(row.get('doc_id', ''))[:12]}.pdf"

            # 2-3. Upload to Gemini + generate
            markdown_text = gemini.call(pdf_bytes, prompt, display_name)
            log(f"  Gemini response: {len(markdown_text):,} chars")

            if args.dry_run:
                print(f"\n{'='*60}\nDRY RUN — {row.get('symbol')}\n"
                      f"{markdown_text[:800]}\n{'='*60}\n")
                counts["processed"] += 1
                continue

            # 4. Parse markdown → structured data
            facts, guidance_rows = parse_gemini_response(markdown_text, row)
            log(f"  Parsed: quarter={facts['quarter'] or 'unknown'}, "
                f"guidance_rows={len(guidance_rows)}")

            # 5a. Company page (persisted forever)
            if OUTPUT_COMPANY_MD:
                append_company_page(
                    drive, repo_id,
                    key=str(row.get("key") or row.get("isin") or row.get("symbol") or ""),
                    content=markdown_text,
                    doc_title=str(row.get("title", "")),
                    quarter=facts["quarter"],
                )

            # 5b. Day page (auto-deleted after 30 days)
            if OUTPUT_DAY_MD:
                append_day_page(
                    drive, repo_id,
                    announcement_date=str(row.get("announcement_date", "")),
                    symbol=str(row.get("symbol", "")),
                    company_name=str(row.get("company_name", "")),
                    quarter=facts["quarter"],
                    content=markdown_text,
                )

            # 6-7. Upsert parquets
            upsert_facts(drive, index_id, facts)
            upsert_guidance(drive, index_id, guidance_rows)

            # 8. Mark done
            queue.loc[queue_idx, "status"] = "done"
            queue.loc[queue_idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            save_queue(drive, index_id, queue)

            counts["processed"] += 1
            log(f"  Done: {row.get('symbol')}")

        except RateLimitExhausted:
            log("All Gemini keys rate-limited — stopping cleanly. "
                "Remaining pending rows will be picked up on the next run.")
            # Queue already has progress saved from prior iterations
            break

        except Exception as exc:
            log(f"  ERROR: {str(exc)[:120]}")
            queue.loc[queue_idx, "status"] = "error"
            queue.loc[queue_idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            save_queue(drive, index_id, queue)
            counts["error"] += 1

    print("-" * 56)
    print(f"Processed : {counts['processed']}")
    print(f"Errors    : {counts['error']}")
    print(f"Skipped   : {counts['skipped']}")
    if not args.dry_run:
        print("Output: company_repo/_index/quarterly_facts.parquet")
        print("Output: company_repo/_index/guidance_tracker.parquet")
        print("Output: company_repo/<key>/company_page.md")


if __name__ == "__main__":
    main()
