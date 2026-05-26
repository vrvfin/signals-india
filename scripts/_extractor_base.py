"""
Shared infrastructure for Phase 2 extractors.

Imported by extract_concall.py, extract_results.py, extract_annual_report.py,
extract_presentation.py, and extract_rating.py. Not run directly.

Provides:
  - get_drive()                3-path auth (service account / saved token / OAuth)
  - GeminiKeyPool              inline-PDF Gemini calls with key rotation + backoff
  - Drive helpers              get_or_create_subfolder, find_file, download_bytes, upload_bytes
  - Queue helpers              load_queue, save_queue
  - Parquet helpers            load_parquet, save_parquet
  - Markdown table parser      extract_md_tables, clean_val, try_float, identify_metric
  - Dual-output helpers        day_filename, append_day_page, append_company_page
  - Key loader                 load_api_keys
"""

from __future__ import annotations

import io
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ------------------------------------------------------------------ #
#  Drive auth  (3 paths: service account / saved token / OAuth flow) #
# ------------------------------------------------------------------ #

def get_drive():
    import json
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    cred_data = json.loads(cs_path.read_text())

    if cred_data.get("type") == "service_account":
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            str(cs_path), scopes=SCOPES
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    if "refresh_token" in cred_data:
        creds = Credentials.from_authorized_user_file(str(cs_path), SCOPES)
        if not creds.valid:
            creds.refresh(Request())
            cs_path.write_text(creds.to_json())
        return build("drive", "v3", credentials=creds, cache_discovery=False)

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


# ------------------------------------------------------------------ #
#  Drive file helpers                                                  #
# ------------------------------------------------------------------ #

def get_or_create_subfolder(drive, parent_id: str, name: str) -> str:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def find_file(drive, folder_id: str, name: str) -> str | None:
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_bytes(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    return fh.getvalue()


def upload_bytes(drive, folder_id: str, filename: str, data: bytes,
                 mimetype: str, existing_id: str | None = None) -> str:
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


def load_queue(drive, index_id: str) -> pd.DataFrame:
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


def save_queue(drive, index_id: str, df: pd.DataFrame) -> None:
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
#  Parquet helpers                                                     #
# ------------------------------------------------------------------ #

def load_parquet(drive, index_id: str, filename: str,
                 cols: list[str]) -> pd.DataFrame:
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


def save_parquet(drive, index_id: str, filename: str,
                 df: pd.DataFrame) -> None:
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
    def __init__(self, api_keys: list[str], model: str):
        self.keys = api_keys
        self.model = model
        self.idx = 0

    def _run(self, contents: list, label: str) -> str:
        """Internal: call Gemini with given contents, rotate keys on 429."""
        backoff = 30
        total_attempts = len(self.keys)
        attempted = 0
        while attempted < total_attempts:
            client = genai.Client(api_key=self.keys[self.idx])
            try:
                log(f"  {label} (key {self.idx + 1}/{len(self.keys)})...")
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
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

    def call(self, pdf_bytes: bytes, prompt: str, display_name: str) -> str:
        """Generate content from inline PDF bytes + prompt. Rotates keys on 429."""
        import base64
        b64 = base64.standard_b64encode(pdf_bytes).decode()
        contents = [
            genai_types.Part(
                inline_data=genai_types.Blob(mime_type="application/pdf", data=b64)
            ),
            genai_types.Part.from_text(text=prompt),
        ]
        return self._run(contents, f"Generating response [{display_name}]")

    def call_text(self, prompt: str, display_name: str) -> str:
        """Generate content from text prompt only (no PDF). Used for synthesis passes."""
        contents = [genai_types.Part.from_text(text=prompt)]
        return self._run(contents, f"Synthesising [{display_name}]")


# ------------------------------------------------------------------ #
#  Markdown table parser (handles fenced and unfenced pipe tables)    #
# ------------------------------------------------------------------ #

def extract_md_tables(text: str) -> list[dict]:
    """Return list of {headers: [...], rows: [[...], ...]} from all tables in text."""
    text = text.replace("||", "|")

    def _is_sep(line: str) -> bool:
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


def clean_val(s: str) -> str:
    v = (s or "").strip()
    return v if v and v not in ("-", "–", "") else "NA"


def try_float(s: str):
    if not s or s == "NA":
        return None
    cleaned = re.sub(r"[,%₹$()bpsBPS]", "", s).strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


METRIC_ALIASES: list[tuple[str, str]] = [
    ("revenue",  "revenue"),
    ("sales",    "revenue"),
    ("ebidta",   "ebitda"),
    ("ebitda",   "ebitda"),
    ("ebita",    "ebitda"),
    ("ebit",     "ebitda"),
    ("pat",      "pat"),
    ("net profit", "pat"),
    ("profit after", "pat"),
    ("margin",   "margin"),
    ("volume",   "volume"),
    ("capacity", "capacity"),
    ("eps",      "eps"),
]


def identify_metric(label: str) -> str | None:
    low = label.lower().strip()
    for token, name in METRIC_ALIASES:
        if token in low:
            return name
    return None


# ------------------------------------------------------------------ #
#  Dual-output helpers  (company page + _daily/ day digest)          #
# ------------------------------------------------------------------ #

def day_filename(doc_type: str, announcement_date: str) -> str:
    """Return e.g. 'results_26_may2026.md' from 'results', '2026-05-26'."""
    try:
        dt = datetime.strptime(str(announcement_date)[:10], "%Y-%m-%d")
        return f"{doc_type}_{dt.day:02d}_{dt.strftime('%b').lower()}{dt.year}.md"
    except Exception:
        return f"{doc_type}_{str(announcement_date)[:10].replace('-', '_')}.md"


def append_day_page(drive, repo_id: str, doc_type: str,
                    announcement_date: str, symbol: str,
                    company_name: str, quarter: str, content: str) -> None:
    """Append analysis to _daily/<doc_type>_DD_MMMYYYY.md (auto-deleted after 30 days)."""
    daily_id = get_or_create_subfolder(drive, repo_id, "_daily")
    fname = day_filename(doc_type, announcement_date)
    entry = f"\n\n---\n## {symbol} — {company_name} | {quarter}\n\n" + content
    fid = find_file(drive, daily_id, fname)
    if fid:
        existing = download_bytes(drive, fid).decode("utf-8", errors="replace")
        upload_bytes(drive, daily_id, fname,
                     (existing + entry).encode("utf-8"), "text/markdown",
                     existing_id=fid)
    else:
        doc_label = doc_type.replace("_", " ").title()
        header = (f"# Daily {doc_label} Digest — "
                  f"{str(announcement_date)[:10]}\n"
                  f"*Auto-deleted after 30 days.*\n")
        upload_bytes(drive, daily_id, fname,
                     (header + entry).encode("utf-8"), "text/markdown")


def append_company_page(drive, repo_id: str, key: str,
                        doc_type_label: str, content: str,
                        doc_title: str, quarter: str) -> None:
    """Append a section to company_repo/<key>/company_page.md (persisted forever)."""
    if not key:
        log("  WARN: empty key — skipping company_page.md update")
        return
    comp_id = get_or_create_subfolder(drive, repo_id, key)
    header = (
        f"\n\n---\n## {quarter} {doc_type_label} — {doc_title}\n"
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
#  Gemini key loader                                                   #
# ------------------------------------------------------------------ #

def load_api_keys() -> list[str]:
    """Return sorted GEMINI_API_KEY_1..N keys, plus plain GEMINI_API_KEY if set."""
    keys = [
        v for _, v in sorted(
            ((k, v) for k, v in os.environ.items()
             if re.match(r"GEMINI_API_KEY_\d+$", k) and v.strip()),
            key=lambda kv: kv[0],
        )
    ]
    plain = os.environ.get("GEMINI_API_KEY", "").strip()
    if plain and plain not in keys:
        keys.append(plain)
    return keys
