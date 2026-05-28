"""
Phase 2 / Stage B — Concall extraction via Gemini.

Consumes pending concall entries from the Drive processing queue, runs each PDF
through Gemini (File API), extracts structured quarterly facts and guidance into
parquet tables, appends a markdown brief to the company's company_page.md, and
marks queue rows done.

Outputs per processed concall
  1. company_repo/<key>/company_page.md             — per-company brief (cumulative)
  2. company_repo/_daily/concall_DD_MMMYYYY.md      — daily digest with index
  3. company_repo/_quarterly/QXFY_mgmt_guidance.md  — quarterly guidance tracker
  4. company_repo/_index/quarterly_facts.parquet    — actuals (Table_A)
     company_repo/_index/guidance_tracker.parquet   — Table_A guidance rows
     company_repo/_index/gf1_guidance_statements.parquet  — raw forward stmts
     company_repo/_index/gf2_historical_guidance.parquet  — past guidance vs actuals
     company_repo/_index/gf3_operational_visibility.parquet
     company_repo/_index/gf4_quality_flags.parquet

CSV snapshots written once per run (overwrite):
     company_repo/_index/gf1_guidance_statements.csv
     company_repo/_index/gf2_historical_guidance.csv
     company_repo/_index/gf3_operational_visibility.csv
     company_repo/_index/gf4_quality_flags.csv
     company_repo/_index/guidance_tracker.csv

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

# Minimum seconds to sleep between consecutive Gemini calls.
# Keeps us at ≤10 calls/min per key, eliminating most RPM 429 cascades.
INTER_CALL_SLEEP = 6

# ---- Output toggles ----
OUTPUT_COMPANY_MD          = True   # append to company_repo/<ISIN>/company_page.md
OUTPUT_DAY_MD              = True   # append to company_repo/_daily/concall_DD_MMMYYYY.md
OUTPUT_QUARTERLY_MD        = True   # append to company_repo/_quarterly/QXFY_mgmt_guidance.md
OUTPUT_GF_PARQUETS         = True   # write GF1-4 parquets to _index/
OUTPUT_GF_CSV              = True   # write CSV snapshots at end of run


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ------------------------------------------------------------------ #
#  Drive helpers                                                       #
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
#  Parquet schemas                                                     #
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

# GF1 — Raw forward-looking statements (exact text from transcript)
GF1_COLS = [
    "isin", "symbol", "company_name", "quarter",
    "statement_id", "exact_statement", "metric_type", "timeframe",
    "explicitness_type", "quantifiable", "numeric_value", "range_val",
    "operational_anchor", "supporting_evidence",
    "processed_at", "source_doc_id",
]

# GF2 — Historical guidance vs actuals
GF2_COLS = [
    "isin", "symbol", "company_name", "quarter",
    "financial_qtr", "historical_reference", "original_guidance",
    "actual_mentioned_outcome", "context_source", "management_self_assessment",
    "processed_at", "source_doc_id",
]

# GF3 — Operational visibility drivers
GF3_COLS = [
    "isin", "symbol", "company_name", "quarter",
    "visibility_driver", "evidence_type", "timeframe",
    "quantified", "commentary",
    "processed_at", "source_doc_id",
]

# GF4 — Guidance quality flags
GF4_COLS = [
    "isin", "symbol", "company_name", "quarter",
    "flag_type", "evidence",
    "processed_at", "source_doc_id",
]


# ------------------------------------------------------------------ #
#  Parquet helpers                                                     #
# ------------------------------------------------------------------ #

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


def _upsert_gf(drive, index_id, filename: str, cols: list, rows: list[dict]) -> None:
    """Generic GF table upsert: delete existing rows for this source_doc_id, append new."""
    if not rows:
        return
    df = _load_parquet(drive, index_id, filename, cols)
    source_doc_id = str(rows[0].get("source_doc_id", ""))
    if source_doc_id:
        df = df[df["source_doc_id"].astype(str) != source_doc_id]
    new_df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    df = pd.concat([df, new_df], ignore_index=True)
    _save_parquet(drive, index_id, filename, df)


def upsert_gf1(drive, index_id, rows: list[dict]) -> None:
    _upsert_gf(drive, index_id, "gf1_guidance_statements.parquet", GF1_COLS, rows)


def upsert_gf2(drive, index_id, rows: list[dict]) -> None:
    _upsert_gf(drive, index_id, "gf2_historical_guidance.parquet", GF2_COLS, rows)


def upsert_gf3(drive, index_id, rows: list[dict]) -> None:
    _upsert_gf(drive, index_id, "gf3_operational_visibility.parquet", GF3_COLS, rows)


def upsert_gf4(drive, index_id, rows: list[dict]) -> None:
    _upsert_gf(drive, index_id, "gf4_quality_flags.parquet", GF4_COLS, rows)


def write_csv_exports(drive, index_id) -> None:
    """Write CSV snapshots of all guidance parquets to Drive. Called once per run."""
    exports = [
        ("guidance_tracker.parquet",         "guidance_tracker.csv",         GUIDANCE_COLS),
        ("gf1_guidance_statements.parquet",  "gf1_guidance_statements.csv",  GF1_COLS),
        ("gf2_historical_guidance.parquet",  "gf2_historical_guidance.csv",  GF2_COLS),
        ("gf3_operational_visibility.parquet","gf3_operational_visibility.csv",GF3_COLS),
        ("gf4_quality_flags.parquet",        "gf4_quality_flags.csv",        GF4_COLS),
    ]
    for pq_name, csv_name, cols in exports:
        try:
            df = _load_parquet(drive, index_id, pq_name, cols)
            if df.empty:
                continue
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            fid = find_file(drive, index_id, csv_name)
            upload_bytes(drive, index_id, csv_name, csv_bytes, "text/csv", existing_id=fid)
            log(f"  CSV export: {csv_name} ({len(df)} rows)")
        except Exception as e:
            log(f"  WARNING: CSV export failed for {csv_name}: {str(e)[:80]}")


# ------------------------------------------------------------------ #
#  Markdown output helpers                                             #
# ------------------------------------------------------------------ #

# Regex to detect the *(Run HH:MM IST)* suffix embedded in ## concall_N headings.
# Used by _build_index_with_runs to reconstruct run grouping on every Index rebuild.
_RUN_SUFFIX = re.compile(r'\s*\*\(Run (.+?)\)\*$')


def _build_index_with_runs(entries: list[tuple[str, str]]) -> list[str]:
    """Build the **Index:** body lines, grouping consecutive entries by run time.

    entries: list of (number_str, full_heading_description).
             Descriptions produced by append_day_page end with *(Run HH:MM IST)*.
             Pre-feature entries without that suffix are shown ungrouped (no header).

    Returns a flat list of strings ready for '\\n'.join().
    Example output:
        ['*Run 1 — 08:12 IST*',
         '- concall_1: TCS · Tata Consultancy Services | Q4 FY26',
         '*Run 2 — 11:05 IST*',
         '- concall_2: INFY · Infosys | Q4 FY26']
    """
    # Parse each entry into (num, clean_description, run_ts)
    parsed: list[tuple[str, str, str]] = []
    for n, desc in entries:
        m = _RUN_SUFFIX.search(desc)
        rt = m.group(1) if m else ""
        parsed.append((n, _RUN_SUFFIX.sub("", desc).strip(), rt))

    # Collect run-time labels in order of first appearance (for numbering)
    seen_rts: list[str] = []
    for _, _, rt in parsed:
        if rt and rt not in seen_rts:
            seen_rts.append(rt)

    lines: list[str] = []
    current_rt = "__unset__"
    for n, clean, rt in parsed:
        if rt != current_rt:
            if rt:
                run_num = seen_rts.index(rt) + 1
                lines.append(f"*Run {run_num} — {rt}*")
            # empty rt = legacy entry written before this feature — no run header
            current_rt = rt
        lines.append(f"- concall_{n}: {clean}")
    return lines


def _day_filename(announcement_date: str) -> str:
    """Return e.g. 'concall_26_may2026.md' from '2026-05-26'."""
    try:
        dt = datetime.strptime(str(announcement_date)[:10], "%Y-%m-%d")
        return f"concall_{dt.day:02d}_{dt.strftime('%b').lower()}{dt.year}.md"
    except Exception:
        return f"concall_{str(announcement_date)[:10].replace('-', '_')}.md"


def _quarter_filename(quarter: str) -> str:
    """Return e.g. 'Q4_FY26_mgmt_guidance.md' from 'Q4 FY26'."""
    if not quarter:
        return "QX_FYxx_mgmt_guidance.md"
    safe = re.sub(r"\s+", "_", quarter.strip())
    safe = re.sub(r"[^A-Za-z0-9_]", "", safe)
    return f"{safe}_mgmt_guidance.md"


def append_day_page(drive, repo_id, announcement_date: str, symbol: str,
                    company_name: str, quarter: str, content: str,
                    run_time: str = "") -> None:
    """Append this company's analysis to the daily digest file in _daily/.

    File format (header rebuilt on every append):

        # Daily Concall Digest — 2026-05-26
        *Total: 3 concalls — last updated 26 May 2026 14:30 IST*

        **Index:**
        *Run 1 — 08:12 IST*
        - concall_1: TCS · Tata Consultancy Services | Q4 FY26
        - concall_2: RELIANCE · Reliance Industries | Q4 FY26
        *Run 2 — 11:05 IST*
        - concall_3: INFY · Infosys | Q4 FY26

        ---
        ## concall_1 — TCS · Tata Consultancy Services | Q4 FY26 *(Run 08:12 IST)*
        ...content...

    Run time is embedded in the ## heading as *(Run HH:MM IST)* for reconstruction
    on subsequent appends. The Index shows a clean description under a run-group header.
    Entries written before this feature have no suffix and appear ungrouped.
    Ctrl+F / search "concall_3" jumps directly to the third entry.
    The Index list and Total count are fully rewritten on every append.
    """
    daily_id = get_or_create_subfolder(drive, repo_id, "_daily")
    fname = _day_filename(announcement_date)
    now_str = datetime.now().strftime("%d %b %Y %H:%M")
    date_str = str(announcement_date)[:10]

    # Description stored in ## heading includes run suffix (for index reconstruction).
    # Index lines show the clean version; run group headers show the time.
    entry_clean  = f"{symbol} · {company_name} | {quarter}"
    entry_stored = f"{entry_clean} *(Run {run_time})*" if run_time else entry_clean

    fid = find_file(drive, daily_id, fname)
    if fid:
        existing = download_bytes(drive, fid).decode("utf-8", errors="replace")

        existing_entries = re.findall(
            r'^## concall_(\d+) — (.+)$', existing, re.MULTILINE
        )
        new_num = len(existing_entries) + 1

        # Build index including the new entry; _build_index_with_runs groups by run
        all_entries = existing_entries + [(str(new_num), entry_stored)]
        index_lines = _build_index_with_runs(all_entries)

        total = (f"*Total: {new_num} concall{'s' if new_num != 1 else ''}"
                 f" — last updated {now_str} IST*")
        new_header = (
            f"# Daily Concall Digest — {date_str}\n"
            f"{total}\n\n"
            f"**Index:**\n"
            + "\n".join(index_lines)
            + "\n"
        )

        split_match = re.search(r'\n---\n## concall_', existing)
        if split_match:
            entries_part = existing[split_match.start():]
        else:
            entries_part = "\n\n" + existing.lstrip("# \n")

        new_entry = (
            f"\n---\n"
            f"## concall_{new_num} — {entry_stored}\n\n"
            + content
        )
        upload_bytes(drive, daily_id, fname,
                     (new_header + entries_part + new_entry).encode("utf-8"),
                     "text/markdown", existing_id=fid)
    else:
        # First entry of the day — write fresh file
        index_run_header = f"*Run 1 — {run_time}*\n" if run_time else ""
        header = (
            f"# Daily Concall Digest — {date_str}\n"
            f"*Total: 1 concall — last updated {now_str} IST*\n\n"
            f"**Index:**\n"
            f"{index_run_header}"
            f"- concall_1: {entry_clean}\n"
        )
        entry = (
            f"\n---\n"
            f"## concall_1 — {entry_stored}\n\n"
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


def append_quarterly_guidance_page(drive, repo_id, symbol: str, company_name: str,
                                   quarter: str, guidance_content: str) -> None:
    """Append compact guidance summary to the quarterly guidance tracker.

    File: company_repo/_quarterly/Q4_FY26_mgmt_guidance.md
    Format: indexed like daily digest, one entry per company.
    Contains: Table_A + GF1 + GF4 + A-3 summary extracted from full response.
    """
    if not quarter:
        return
    quarterly_id = get_or_create_subfolder(drive, repo_id, "_quarterly")
    fname = _quarter_filename(quarter)
    now_str = datetime.now().strftime("%d %b %Y %H:%M")

    fid = find_file(drive, quarterly_id, fname)
    if fid:
        existing = download_bytes(drive, fid).decode("utf-8", errors="replace")

        existing_entries = re.findall(
            r'^## co_(\d+) — (.+)$', existing, re.MULTILINE
        )
        new_num = len(existing_entries) + 1

        index_lines = [f"- co_{n}: {desc}" for n, desc in existing_entries]
        index_lines.append(f"- co_{new_num}: {symbol} · {company_name}")
        total = (f"*Total: {new_num} compan{'ies' if new_num != 1 else 'y'}"
                 f" — last updated {now_str} IST*")
        new_header = (
            f"# {quarter} Management Guidance Tracker\n"
            f"{total}\n\n"
            f"**Index:**\n"
            + "\n".join(index_lines)
            + "\n"
        )

        split_match = re.search(r'\n---\n## co_', existing)
        entries_part = existing[split_match.start():] if split_match else (
            "\n\n" + existing.lstrip("# \n")
        )

        new_entry = (
            f"\n---\n"
            f"## co_{new_num} — {symbol} · {company_name}\n\n"
            + guidance_content
        )
        upload_bytes(drive, quarterly_id, fname,
                     (new_header + entries_part + new_entry).encode("utf-8"),
                     "text/markdown", existing_id=fid)
    else:
        header = (
            f"# {quarter} Management Guidance Tracker\n"
            f"*Total: 1 company — last updated {now_str} IST*\n\n"
            f"**Index:**\n"
            f"- co_1: {symbol} · {company_name}\n"
        )
        entry = (
            f"\n---\n"
            f"## co_1 — {symbol} · {company_name}\n\n"
            + guidance_content
        )
        upload_bytes(drive, quarterly_id, fname,
                     (header + entry).encode("utf-8"), "text/markdown")


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
        self._last_call_ts: float = 0.0

    def call(self, pdf_bytes: bytes, prompt: str, display_name: str) -> str:
        """Generate content using inline PDF bytes. Rotates keys on 429.

        Enforces INTER_CALL_SLEEP between consecutive calls to stay within RPM.
        """
        import base64
        # Enforce minimum gap between calls (RPM protection)
        elapsed = time.time() - self._last_call_ts
        if elapsed < INTER_CALL_SLEEP and self._last_call_ts > 0:
            time.sleep(INTER_CALL_SLEEP - elapsed)

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
                self._last_call_ts = time.time()
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


def _find_section_table(text: str, *patterns: str) -> dict | None:
    """Find the first markdown table appearing after any of the given regex patterns."""
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        after = text[m.start():]
        tables = _extract_md_tables(after)
        if tables:
            return tables[0]
    return None


def _find_section_text(text: str, *patterns: str, max_chars: int = 4000) -> str:
    """Return up to max_chars of text starting from the first matching pattern."""
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return text[m.start(): m.start() + max_chars]
    return ""


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


# ------------------------------------------------------------------ #
#  GF section parsers                                                  #
# ------------------------------------------------------------------ #

def _norm_header(h: str) -> str:
    """Normalise a table header to a Python identifier for mapping."""
    return re.sub(r"\W+", "_", h.strip().lower()).strip("_")


def _parse_gf_table(table: dict | None, col_map: dict,
                    fixed: dict) -> list[dict]:
    """Convert a parsed markdown table → list of row dicts using col_map.

    col_map: {normalised_header_str → schema_column_name}
    fixed:   values always injected (isin, symbol, company_name, quarter, ...)
    """
    if not table:
        return []
    hdrs = [_norm_header(h) for h in table["headers"]]
    rows = []
    for cells in table["rows"]:
        row_dict = dict(fixed)
        for i, hdr in enumerate(hdrs):
            if i < len(cells):
                mapped = col_map.get(hdr)
                if mapped:
                    row_dict[mapped] = _clean_val(cells[i])
        rows.append(row_dict)
    return rows


# GF1 header → schema column mapping
_GF1_MAP = {
    "company_name":       "company_name",
    "current_qtr":        "quarter",
    "statement_id":       "statement_id",
    "exact_statement":    "exact_statement",
    "metric_type":        "metric_type",
    "timeframe":          "timeframe",
    "explicitness_type":  "explicitness_type",
    "quantifiable":       "quantifiable",
    "numeric_value":      "numeric_value",
    "range":              "range_val",
    "operational_anchor": "operational_anchor",
    "supporting_evidence":"supporting_evidence",
}

# GF2 header → schema column mapping
_GF2_MAP = {
    "company_name":              "company_name",
    "financial_qtr":             "financial_qtr",
    "historical_reference":      "historical_reference",
    "original_guidance":         "original_guidance",
    "actual_mentioned_outcome":  "actual_mentioned_outcome",
    "context_source":            "context_source",
    "management_self_assessment":"management_self_assessment",
}

# GF3 header → schema column mapping
_GF3_MAP = {
    "company_name":      "company_name",
    "financial_qtr":     "quarter",
    "visibility_driver": "visibility_driver",
    "evidence_type":     "evidence_type",
    "timeframe":         "timeframe",
    "quantified":        "quantified",
    "commentary":        "commentary",
}

# GF4 header → schema column mapping
_GF4_MAP = {
    "company_name":  "company_name",
    "financial_qtr": "quarter",
    "flag_type":     "flag_type",
    "evidence":      "evidence",
}


def parse_gf_sections(
    text: str, row: pd.Series, quarter: str, now_str: str
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Extract GF1–GF4 rows from the full Gemini markdown response.

    Returns (gf1_rows, gf2_rows, gf3_rows, gf4_rows).
    """
    isin = str(row.get("isin") or "")
    symbol = str(row.get("symbol") or "")
    company_name = str(row.get("company_name") or "")
    source_doc_id = str(row.get("doc_id") or "")

    fixed = {
        "isin": isin, "symbol": symbol, "company_name": company_name,
        "quarter": quarter, "processed_at": now_str, "source_doc_id": source_doc_id,
    }

    # GF1
    gf1_tbl = _find_section_table(
        text,
        r"Section\s+GF1", r"GF1\s*[—–\-]", r"Raw\s+Guidance\s+Extraction"
    )
    gf1_rows = _parse_gf_table(gf1_tbl, _GF1_MAP, fixed)

    # GF2
    gf2_tbl = _find_section_table(
        text,
        r"Section\s+GF2", r"GF2\s*[—–\-]", r"Historical\s+Guidance\s+References"
    )
    gf2_rows = _parse_gf_table(gf2_tbl, _GF2_MAP, fixed)

    # GF3
    gf3_tbl = _find_section_table(
        text,
        r"Section\s+GF3", r"GF3\s*[—–\-]", r"Operational\s+Visibility\s+Extraction"
    )
    gf3_rows = _parse_gf_table(gf3_tbl, _GF3_MAP, fixed)

    # GF4
    gf4_tbl = _find_section_table(
        text,
        r"Section\s+GF4", r"GF4\s*[—–\-]", r"Guidance\s+Quality\s+Flags"
    )
    gf4_rows = _parse_gf_table(gf4_tbl, _GF4_MAP, fixed)

    return gf1_rows, gf2_rows, gf3_rows, gf4_rows


def build_quarterly_guidance_content(text: str, symbol: str, company_name: str,
                                     quarter: str) -> str:
    """Extract compact guidance summary for the quarterly tracker.

    Pulls: A-3 summary + Table_A + GF1 + GF4 sections.
    Falls back to first 3,000 chars if no sections found.
    """
    parts = []

    # A-3 key guidance summary
    a3 = _find_section_text(
        text,
        r"A-3\)", r"A[\-\s]3\b", r"forward\s+guidance.*summary",
        r"Key\s+Forward\s+Guidance",
        max_chars=600
    )
    if a3:
        parts.append("#### Key Forward Guidance (A-3)\n" + a3.strip())

    # Table_A financial grid
    ta = _find_section_text(
        text,
        r"Table_A", r"Table\s+A\b", r"Unified\s+Financial",
        r"Financial\s+Intelligence",
        max_chars=2500
    )
    if ta:
        parts.append("#### Financial Grid (Table_A)\n" + ta.strip())

    # GF1 raw guidance
    gf1 = _find_section_text(
        text,
        r"Section\s+GF1", r"GF1\s*[—–\-]", r"Raw\s+Guidance\s+Extraction",
        max_chars=3000
    )
    if gf1:
        parts.append("#### GF1 — Raw Guidance Statements\n" + gf1.strip())

    # GF4 quality flags
    gf4 = _find_section_text(
        text,
        r"Section\s+GF4", r"GF4\s*[—–\-]", r"Guidance\s+Quality\s+Flags",
        max_chars=1500
    )
    if gf4:
        parts.append("#### GF4 — Guidance Quality Flags\n" + gf4.strip())

    if parts:
        return "\n\n".join(parts)
    return text[:3000] + "\n\n*(truncated — full analysis in company_page.md)*"


# ------------------------------------------------------------------ #
#  Main Gemini response parser (Table_A + guidance rows)              #
# ------------------------------------------------------------------ #

def parse_gemini_response(
    text: str, row: pd.Series
) -> tuple[dict, list[dict]]:
    """Parse Gemini markdown → (quarterly_facts dict, guidance_rows list) from Table_A."""
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

    m12_col = next((i for i, h in enumerate(hdrs)
                    if re.search(r"12\s*[Mm]", h)), None)

    fy_explicit: dict[str, int] = {}
    for i, h in enumerate(hdrs):
        if "explicit" in h.lower() or "guidance" in h.lower():
            m = re.search(r"FY(\d{2,4})", h, re.IGNORECASE)
            if m:
                fy_explicit[f"FY{m.group(1)}"] = i

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

    drive = get_drive()

    # Load Gemini API keys
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

    counts = {"processed": 0, "error": 0, "skipped": 0, "gf1": 0, "gf2": 0, "gf3": 0, "gf4": 0}

    # Fixed run-time label for this execution — used to group today's digest entries
    # by run (Run 1 / Run 2 / …) so the user can see what each scheduled slot added.
    run_time = datetime.now().strftime("%H:%M IST")

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

            # 2-3. Upload to Gemini + generate (includes inter-call sleep)
            markdown_text = gemini.call(pdf_bytes, prompt, display_name)
            log(f"  Gemini response: {len(markdown_text):,} chars")

            if args.dry_run:
                print(f"\n{'='*60}\nDRY RUN — {row.get('symbol')}\n"
                      f"{markdown_text[:800]}\n{'='*60}\n")
                counts["processed"] += 1
                continue

            now_str = datetime.now().isoformat(timespec="seconds")

            # 4. Parse Table_A → quarterly facts + guidance rows
            facts, guidance_rows = parse_gemini_response(markdown_text, row)
            quarter = facts["quarter"]
            log(f"  Parsed: quarter={quarter or 'unknown'}, "
                f"guidance_rows={len(guidance_rows)}")

            # 4b. Parse GF1-4 sections
            gf1_rows, gf2_rows, gf3_rows, gf4_rows = parse_gf_sections(
                markdown_text, row, quarter, now_str
            )
            counts["gf1"] += len(gf1_rows)
            counts["gf2"] += len(gf2_rows)
            counts["gf3"] += len(gf3_rows)
            counts["gf4"] += len(gf4_rows)
            log(f"  GF parsed: GF1={len(gf1_rows)} GF2={len(gf2_rows)} "
                f"GF3={len(gf3_rows)} GF4={len(gf4_rows)}")

            # 5a. Company page (persisted forever)
            if OUTPUT_COMPANY_MD:
                append_company_page(
                    drive, repo_id,
                    key=str(row.get("key") or row.get("isin") or row.get("symbol") or ""),
                    content=markdown_text,
                    doc_title=str(row.get("title", "")),
                    quarter=quarter,
                )

            # 5b. Daily digest
            if OUTPUT_DAY_MD:
                append_day_page(
                    drive, repo_id,
                    announcement_date=str(row.get("announcement_date", "")),
                    symbol=str(row.get("symbol", "")),
                    company_name=str(row.get("company_name", "")),
                    quarter=quarter,
                    content=markdown_text,
                    run_time=run_time,
                )

            # 5c. Quarterly guidance tracker
            if OUTPUT_QUARTERLY_MD and quarter:
                guidance_content = build_quarterly_guidance_content(
                    markdown_text,
                    symbol=str(row.get("symbol", "")),
                    company_name=str(row.get("company_name", "")),
                    quarter=quarter,
                )
                append_quarterly_guidance_page(
                    drive, repo_id,
                    symbol=str(row.get("symbol", "")),
                    company_name=str(row.get("company_name", "")),
                    quarter=quarter,
                    guidance_content=guidance_content,
                )

            # 6. Upsert parquets (Table_A)
            upsert_facts(drive, index_id, facts)
            upsert_guidance(drive, index_id, guidance_rows)

            # 6b. Upsert GF1-4 parquets
            if OUTPUT_GF_PARQUETS:
                upsert_gf1(drive, index_id, gf1_rows)
                upsert_gf2(drive, index_id, gf2_rows)
                upsert_gf3(drive, index_id, gf3_rows)
                upsert_gf4(drive, index_id, gf4_rows)

            # 7. Mark done
            queue.loc[queue_idx, "status"] = "done"
            queue.loc[queue_idx, "processed_at"] = now_str
            save_queue(drive, index_id, queue)

            counts["processed"] += 1
            log(f"  Done: {row.get('symbol')}")

        except RateLimitExhausted:
            log("All Gemini keys rate-limited — stopping cleanly. "
                "Remaining pending rows will be picked up on the next run.")
            break

        except Exception as exc:
            log(f"  ERROR: {str(exc)[:120]}")
            queue.loc[queue_idx, "status"] = "error"
            queue.loc[queue_idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            save_queue(drive, index_id, queue)
            counts["error"] += 1

    # CSV snapshots (once per run, after all processing)
    if OUTPUT_GF_CSV and counts["processed"] > 0 and not args.dry_run:
        log("Writing CSV snapshots...")
        write_csv_exports(drive, index_id)

    print("-" * 56)
    print(f"Processed : {counts['processed']}")
    print(f"Errors    : {counts['error']}")
    print(f"Skipped   : {counts['skipped']}")
    print(f"GF rows   : GF1={counts['gf1']} GF2={counts['gf2']} "
          f"GF3={counts['gf3']} GF4={counts['gf4']}")
    if not args.dry_run and counts["processed"] > 0:
        print("Output: company_repo/_index/quarterly_facts.parquet")
        print("Output: company_repo/_index/guidance_tracker.parquet")
        print("Output: company_repo/_index/gf1_guidance_statements.parquet (.csv)")
        print("Output: company_repo/_index/gf2_historical_guidance.parquet (.csv)")
        print("Output: company_repo/_index/gf3_operational_visibility.parquet (.csv)")
        print("Output: company_repo/_index/gf4_quality_flags.parquet (.csv)")
        print("Output: company_repo/<key>/company_page.md")
        print("Output: company_repo/_daily/concall_DD_MMMYYYY.md")
        print("Output: company_repo/_quarterly/QXFY_mgmt_guidance.md")


if __name__ == "__main__":
    main()
