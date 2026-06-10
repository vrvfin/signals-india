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
import atexit
import io
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from gemini_pool import (BucketPool, AllBucketsExhausted, FatalCallError,
                         load_keys)

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Concall is P0 (best quality). Models tried best-first; the pool only
# downgrades to the next model once the current one is exhausted on ALL keys.
# Each (key, model) is a separate daily free-tier quota bucket.
# All confirmed to have a free tier as of 2026-06 (pro-tier models do NOT).
# Quality-only chain. Disjoint from P1_MODELS (lite) so P1 extraction can never
# consume concall's premium buckets. 3 models × 6 keys = 18 daily buckets.
CONCALL_MODELS = [
    "gemini-3.5-flash",        # best free-tier quality
    "gemini-3-flash-preview",
    "gemini-2.5-flash",        # 20/day measured
]

# Minimum seconds to sleep between consecutive successful Gemini calls (RPM hygiene).
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
              "drive_file_id", "status", "discovered_at", "processed_at",
              # Phase 3 T1: stamped (=date) only when a row is processed by a
              # backfill run; blank for normal Phase 2 live processing.
              "backfill_process_date",
              # Phase 3 T1: enqueue origin — "live" or "backfill". Each extractor
              # drains ONLY its own rows so backfill never lands in the daily digest
              # or burns the main key pool. Blank/absent => treated as "live".
              "source"]


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
#  Mutual-exclusion lock (T1.4)                                        #
#  One extractor (live OR backfill) writes the shared _index parquets  #
#  at a time, so concurrent runs can't clobber each other. Night-run   #
#  scheduling is the primary defense; this is the safety guard. Stale   #
#  locks (crashed run) are auto-stolen after LOCK_MAX_AGE_MIN.         #
# ------------------------------------------------------------------ #

_LOCK_NAME = "_extract.lock"
LOCK_MAX_AGE_MIN = 360   # 6h — long enough for a big backfill night run


def acquire_extract_lock(drive, index_id, mode: str) -> bool:
    """Try to claim the extractor lock. Returns True on success.

    A fresh lock owned by another run blocks acquisition; a lock older than
    LOCK_MAX_AGE_MIN is considered stale and stolen."""
    fid = find_file(drive, index_id, _LOCK_NAME)
    if fid:
        try:
            content = download_bytes(drive, fid).decode("utf-8", errors="replace")
            ts_str = content.split("|", 2)[1] if "|" in content else ""
            ts = datetime.fromisoformat(ts_str) if ts_str else None
            age_min = ((datetime.now() - ts).total_seconds() / 60.0) if ts else 1e9
            if age_min < LOCK_MAX_AGE_MIN:
                log(f"  LOCK held by '{content.split('|')[0]}' "
                    f"({age_min:.0f} min old) — another extraction is running. "
                    f"Exiting cleanly; will resume next run.")
                return False
            log(f"  LOCK is stale ({age_min:.0f} min) — stealing it.")
        except Exception as e:
            log(f"  LOCK read failed ({str(e)[:60]}) — overwriting.")
    payload = f"{mode}|{datetime.now().isoformat(timespec='seconds')}".encode("utf-8")
    upload_bytes(drive, index_id, _LOCK_NAME, payload, "text/plain", existing_id=fid)
    return True


def release_extract_lock(drive, index_id) -> None:
    try:
        fid = find_file(drive, index_id, _LOCK_NAME)
        if fid:
            drive.files().delete(fileId=fid).execute()
    except Exception as e:
        log(f"  LOCK release failed ({str(e)[:60]}) — will be auto-stolen later.")


# ------------------------------------------------------------------ #
#  Parquet schemas                                                     #
# ------------------------------------------------------------------ #

QFACTS_COLS = [
    "isin", "symbol", "company_name", "quarter", "fy_year",
    "revenue_q", "ebitda_q", "pat_q", "margin_pct", "volume_q", "capacity_q",
    "revenue_12m", "pat_12m", "processed_at", "source_doc_id",
    # Gemini response length (chars). Used as a richness proxy: if a second
    # document for the same company+quarter is >20% longer, it supersedes the
    # existing entry (full transcript beats a short announcement). Legacy rows
    # that pre-date this column have response_chars=0 and are always superseded.
    "response_chars",
]

# Minimum ratio new/old response length to supersede an existing quarter entry.
SUPERSEDE_THRESHOLD = 1.2   # new must be >20% longer than existing


def _norm_quarter(q: str) -> str:
    """Canonical form for quarter strings used in dedup comparisons.

    Converts any of 'Q4FY26', 'Q4 FY26', 'Q4 FY 26', 'q4fy26' → 'Q4 FY26'.
    Handles half-year 'H1 FY26' similarly.  Returns the input unchanged if the
    pattern is not recognised (so non-quarter strings don't get mangled).
    """
    s = str(q).strip().upper()
    normed = re.sub(r"([QH]\d)\s*(FY\s*\d+)", lambda m: m.group(1) + " " + re.sub(r"\s+", "", m.group(2)), s)
    return normed

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

# GF_TRACK — Mgmt Said vs Delivered (credibility). One row per verdict-table row;
# the four summary fields (cred_score/pattern/strongest_area/recurring_miss) are
# duplicated onto every row of the same concall for easy per-company lookup.
# Produced only when a [HISTORICAL_CONTEXT] block was injected (history exists).
MGMT_CRED_COLS = [
    "isin", "symbol", "company_name", "quarter",
    "qtr_guided", "metric", "guidance_given", "target_period",
    "actual_delivered", "delta", "verdict",
    "cred_score", "pattern", "strongest_area", "recurring_miss",
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


def upsert_mgmt_credibility(drive, index_id, rows: list[dict]) -> None:
    _upsert_gf(drive, index_id, "mgmt_credibility.parquet", MGMT_CRED_COLS, rows)


def write_csv_exports(drive, index_id) -> None:
    """Write CSV snapshots of all guidance parquets to Drive. Called once per run."""
    exports = [
        ("guidance_tracker.parquet",         "guidance_tracker.csv",         GUIDANCE_COLS),
        ("gf1_guidance_statements.parquet",  "gf1_guidance_statements.csv",  GF1_COLS),
        ("gf2_historical_guidance.parquet",  "gf2_historical_guidance.csv",  GF2_COLS),
        ("gf3_operational_visibility.parquet","gf3_operational_visibility.csv",GF3_COLS),
        ("gf4_quality_flags.parquet",        "gf4_quality_flags.csv",        GF4_COLS),
        ("mgmt_credibility.parquet",         "mgmt_credibility.csv",         MGMT_CRED_COLS),
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


def _day_filename(announcement_date: str, prefix: str = "concall") -> str:
    """Return e.g. 'concall_26_may2026.md' from '2026-05-26'.

    prefix='daily_backfill' routes backfill output to its own digest file so it
    stays distinct from the live Phase 2 daily concall digest (T1.3)."""
    try:
        dt = datetime.strptime(str(announcement_date)[:10], "%Y-%m-%d")
        return f"{prefix}_{dt.day:02d}_{dt.strftime('%b').lower()}{dt.year}.md"
    except Exception:
        return f"{prefix}_{str(announcement_date)[:10].replace('-', '_')}.md"


def _quarter_filename(quarter: str) -> str:
    """Return e.g. 'Q4_FY26_mgmt_guidance.md' from 'Q4 FY26'."""
    if not quarter:
        return "QX_FYxx_mgmt_guidance.md"
    safe = re.sub(r"\s+", "_", quarter.strip())
    safe = re.sub(r"[^A-Za-z0-9_]", "", safe)
    return f"{safe}_mgmt_guidance.md"


def append_day_page(drive, repo_id, announcement_date: str, symbol: str,
                    company_name: str, quarter: str, content: str,
                    run_time: str = "", backfill: bool = False) -> None:
    """Append this company's analysis to the daily digest file in _daily/.

    UPDATED: Uses EXTRACTION_DATE (today) for filename, not announcement_date.
    This shows "what was processed today" vs "what was announced in the past".

    File format (header rebuilt on every append):

        # Daily Concall Digest — 2026-06-06 (extraction date, not announcement date)
        *Total: 3 concalls — last updated 06 Jun 2026 14:30 IST*

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
    # USE EXTRACTION DATE (today) for the daily digest filename
    # This groups concalls by "when processed" not "when announced"
    extraction_date = datetime.now().strftime("%Y-%m-%d")  # e.g., "2026-06-06"
    # Backfill output goes to its own digest file/title (T1.3); live Phase 2 unchanged.
    digest_prefix = "daily_backfill" if backfill else "concall"
    digest_title = "Daily Backfill Digest" if backfill else "Daily Concall Digest"
    fname = _day_filename(extraction_date, digest_prefix)
    now_str = datetime.now().strftime("%d %b %Y %H:%M")
    date_str = extraction_date[:10]  # "2026-06-06"

    # Description stored in ## heading includes run suffix (for index reconstruction).
    # Index lines show the clean version; run group headers show the time.
    entry_clean  = f"{symbol} · {company_name} | {quarter} (announced {announcement_date})"
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
            f"# {digest_title} — {date_str}\n"
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
            f"# {digest_title} — {date_str}\n"
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


def replace_company_page_section(drive, repo_id, key: str, quarter: str,
                                  content: str, doc_title: str) -> None:
    """Replace an existing quarter's section in company_page.md in-place.

    Called when a richer document supersedes a previously processed one for the
    same quarter. Finds the section by its ## header, replaces everything up to
    the next section separator (---) or end of file, and re-uploads.
    Falls back to append if the section header is not found (e.g. first time).
    """
    if not key:
        return
    comp_id = get_or_create_subfolder(drive, repo_id, key)
    new_header = (
        f"\n\n---\n## {quarter} Concall — {doc_title}\n"
        f"*Processed: {datetime.now().strftime('%Y-%m-%d')} (superseded)*\n\n"
    )
    new_section = new_header + content
    fid = find_file(drive, comp_id, "company_page.md")
    if not fid:
        initial = f"# {key} — Company Intelligence\n" + new_section
        upload_bytes(drive, comp_id, "company_page.md",
                     initial.encode("utf-8"), "text/markdown")
        return
    existing = download_bytes(drive, fid).decode("utf-8", errors="replace")
    # Match from the section's --- separator + ## header through to the next ---
    # separator or end of file. re.DOTALL so .* crosses newlines.
    pattern = (rf'\n\n---\n## {re.escape(quarter)} Concall[^\n]*\n'
               rf'\*Processed:[^\n]*\n\n'
               rf'.*?'
               rf'(?=\n\n---\n##|\Z)')
    if re.search(pattern, existing, re.DOTALL):
        updated = re.sub(pattern, new_section, existing, count=1, flags=re.DOTALL)
        log(f"  Replaced {quarter} section in company_page.md (superseded).")
    else:
        # Section not found — append (handles edge cases like different date format)
        updated = existing + new_section
        log(f"  WARN: {quarter} section not found in company_page.md — appending.")
    upload_bytes(drive, comp_id, "company_page.md",
                 updated.encode("utf-8"), "text/markdown", existing_id=fid)


def _purge_quarter_records(drive, index_id, isin: str, quarter: str) -> None:
    """Remove ALL records for (isin, quarter) from every structured parquet table.

    Called before writing a superseding document so old data is fully replaced,
    not accumulated. The new document's write path then adds fresh rows as normal.
    """
    def _drop(fname, cols):
        df = _load_parquet(drive, index_id, fname, cols)
        if df.empty:
            return
        mask = ((df["isin"].astype(str) == str(isin)) &
                (df["quarter"].astype(str) == str(quarter)))
        if mask.any():
            _save_parquet(drive, index_id, fname, df[~mask].reset_index(drop=True))

    _drop("quarterly_facts.parquet",              QFACTS_COLS)
    _drop("guidance_tracker.parquet",             GUIDANCE_COLS)
    _drop("gf1_guidance_statements.parquet",      GF1_COLS)
    _drop("gf2_historical_guidance.parquet",      GF2_COLS)
    _drop("gf3_operational_visibility.parquet",   GF3_COLS)
    _drop("gf4_quality_flags.parquet",            GF4_COLS)
    log(f"  Purged old {quarter} records from all parquets.")


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
#  Gemini calls are handled by the shared BucketPool engine            #
#  (see gemini_pool.py) — bounded, error-typed (key, model) fallback.  #
# ------------------------------------------------------------------ #


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

# GF_TRACK (Mgmt Said vs Delivered) header → schema column mapping.
# Prompt table header: Qtr Guided | Metric | Guidance Given | Target Period |
#                      Actual Delivered | Delta | Verdict
_GF_TRACK_MAP = {
    "qtr_guided":       "qtr_guided",
    "metric":           "metric",
    "guidance_given":   "guidance_given",
    "target_period":    "target_period",
    "actual_delivered": "actual_delivered",
    "delta":            "delta",
    "verdict":          "verdict",
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


# Summary lines under "## Mgmt Credibility Summary" (see concall_prompt.txt §GF_TRACK)
_CRED_SCORE_RE = re.compile(r"Overall\s+score:\s*([\d.]+)\s*/\s*5", re.IGNORECASE)
_CRED_PATTERN_RE = re.compile(r"Pattern:\s*(.+)", re.IGNORECASE)
_CRED_STRONG_RE = re.compile(r"Strongest\s+area:\s*(.+)", re.IGNORECASE)
_CRED_MISS_RE = re.compile(r"Recurring\s+miss:\s*(.+)", re.IGNORECASE)


def _first_line(s: str) -> str:
    """First non-empty line of a regex capture, stripped of markdown bullets/emphasis."""
    line = (s or "").splitlines()[0] if s else ""
    return line.strip().strip("*[]").strip()


def parse_gf_track(text: str, row: pd.Series, quarter: str, now_str: str) -> list[dict]:
    """Extract the GF_TRACK 'Said vs Delivered' verdict table + credibility summary.

    Returns one dict per verdict-table row (MGMT_CRED_COLS), with the four summary
    fields duplicated onto each. Empty list if the section is absent (it is only
    produced when a [HISTORICAL_CONTEXT] block was injected).
    """
    fixed = {
        "isin": str(row.get("isin") or ""),
        "symbol": str(row.get("symbol") or ""),
        "company_name": str(row.get("company_name") or ""),
        "quarter": quarter,
        "processed_at": now_str,
        "source_doc_id": str(row.get("doc_id") or ""),
    }

    tbl = _find_section_table(
        text,
        r"Section\s+GF_TRACK", r"GF_TRACK\s*[—–\-]", r"Mgmt\s+Said\s+vs\s+Delivered",
    )
    rows = _parse_gf_table(tbl, _GF_TRACK_MAP, fixed)

    # Credibility summary block (after the table)
    summary_txt = _find_section_text(
        text, r"Mgmt\s+Credibility\s+Summary", r"Credibility\s+Summary", max_chars=600
    )
    cred = {"cred_score": "NA", "pattern": "NA",
            "strongest_area": "NA", "recurring_miss": "NA"}
    if summary_txt:
        m = _CRED_SCORE_RE.search(summary_txt)
        if m:
            cred["cred_score"] = m.group(1)
        for key, rx in (("pattern", _CRED_PATTERN_RE),
                        ("strongest_area", _CRED_STRONG_RE),
                        ("recurring_miss", _CRED_MISS_RE)):
            m = rx.search(summary_txt)
            if m:
                cred[key] = _first_line(m.group(1))

    # If there is a summary but no parseable table rows, still record one summary row
    # so the company's credibility score is queryable.
    if not rows and cred["cred_score"] != "NA":
        rows = [dict(fixed)]
    for r in rows:
        r.update(cred)
    return rows


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

    # GF_TRACK credibility section (present only when historical context was injected)
    gf_track = _find_section_text(
        text,
        r"Section\s+GF_TRACK", r"GF_TRACK\s*[—–\-]", r"Mgmt\s+Said\s+vs\s+Delivered",
        max_chars=2500
    )
    if gf_track:
        parts.append("#### Mgmt Said vs Delivered (GF_TRACK)\n" + gf_track.strip())

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
                  if re.search(r"Q\d\s*FY\d{2,4}", h, re.IGNORECASE)), None)
    quarter_name = ""
    if q_col is not None:
        m = re.search(r"(Q\d\s*FY\d{2,4})", hdrs[q_col], re.IGNORECASE)
        if m:
            quarter_name = _norm_quarter(m.group(1).strip())
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
#  Historical context helpers (GF_TRACK credibility section)         #
# ------------------------------------------------------------------ #

def _current_india_quarter() -> str:
    """Return current India FY quarter tag, e.g. 'Q1FY27'.
    India FY: Apr-Jun = Q1, Jul-Sep = Q2, Oct-Dec = Q3, Jan-Mar = Q4."""
    m, y = datetime.now().month, datetime.now().year
    if m in (4, 5, 6):    return f"Q1FY{str(y + 1)[2:]}"
    if m in (7, 8, 9):    return f"Q2FY{str(y + 1)[2:]}"
    if m in (10, 11, 12): return f"Q3FY{str(y + 1)[2:]}"
    return f"Q4FY{str(y)[2:]}"  # Jan–Mar


def _build_historical_context(
    gf1_df: pd.DataFrame,
    qfacts_df: pd.DataFrame,
    results_df: pd.DataFrame,
    row: pd.Series,
) -> str | None:
    """Build a [HISTORICAL_CONTEXT] block for the Gemini prompt.

    Returns None if there is no GF1 history for this company (first-ever
    processed concall) — the prompt is then sent unchanged and Gemini will
    NOT produce a GF_TRACK section (the conditional rule in the prompt handles this).

    When history exists, the block contains:
      - Past GF1 guidance statements (up to 30 rows, last 8 quarters)
      - Actual delivered results from quarterly_facts and results parquets
    """
    isin = str(row.get("isin") or "").strip()
    company_name = str(row.get("company_name") or "").strip()
    symbol = str(row.get("symbol") or "").strip()

    if gf1_df.empty:
        return None

    # --- Filter GF1 to this company (ISIN preferred, fall back to name) ---
    if isin:
        co_gf1 = gf1_df[gf1_df["isin"].astype(str).str.strip() == isin].copy()
    else:
        co_gf1 = gf1_df[
            gf1_df["company_name"].astype(str).str.lower().str.strip()
            == company_name.lower().strip()
        ].copy()

    if co_gf1.empty:
        return None  # No history yet — skip GF_TRACK for this concall

    # Sort chronologically, cap to last 30 rows
    if "quarter" in co_gf1.columns:
        co_gf1 = co_gf1.sort_values("quarter", na_position="first")
    co_gf1 = co_gf1.tail(30)

    # Drop rows with no real statement
    co_gf1 = co_gf1[
        co_gf1["exact_statement"].astype(str).str.strip().str.upper() != "NA"
    ]
    if co_gf1.empty:
        return None

    # --- Build GF1 compact table ---
    gf1_lines = [
        "| Quarter | Metric | Exact Statement | Timeframe | Value |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for _, r in co_gf1.iterrows():
        stmt = str(r.get("exact_statement", "NA"))[:180].replace("|", "/")
        gf1_lines.append(
            f"| {r.get('quarter', 'NA')} | {r.get('metric_type', 'NA')} | "
            f"{stmt} | {r.get('timeframe', 'NA')} | {r.get('numeric_value', 'NA')} |"
        )

    parts = [
        "[HISTORICAL_CONTEXT]",
        f"Company: {company_name or symbol}  |  ISIN: {isin or 'unknown'}",
        "",
        "=== Past GF1 Guidance Statements (last 8 quarters) ===",
        "\n".join(gf1_lines),
    ]

    # --- Actuals from quarterly_facts parquet ---
    actuals_lines: list[str] = []
    if not qfacts_df.empty and isin:
        co_facts = qfacts_df[qfacts_df["isin"].astype(str).str.strip() == isin].copy()
        if not co_facts.empty:
            if "quarter" in co_facts.columns:
                co_facts = co_facts.sort_values("quarter", na_position="first")
            for _, r in co_facts.tail(8).iterrows():
                actuals_lines.append(
                    f"| {r.get('quarter', 'NA')} | "
                    f"Rev={r.get('revenue_q', 'NA')} | "
                    f"EBITDA={r.get('ebitda_q', 'NA')} | "
                    f"PAT={r.get('pat_q', 'NA')} | "
                    f"Margin={r.get('margin_pct', 'NA')} |"
                )

    # --- Actuals from results.parquet (Screener YoY numbers) ---
    if not results_df.empty:
        if isin and "isin" in results_df.columns:
            co_res = results_df[results_df["isin"].astype(str).str.strip() == isin]
        elif symbol and "slug" in results_df.columns:
            co_res = results_df[
                results_df["slug"].astype(str).str.strip().str.lower()
                == symbol.lower()
            ]
        else:
            co_res = pd.DataFrame()

        for _, r in co_res.iterrows():
            actuals_lines.append(
                f"| {r.get('metric', 'NA')} | "
                f"Latest: {r.get('latest_q', 'NA')}={r.get('latest_val', 'NA')} | "
                f"YoY: {r.get('yoy_pct', 'NA')}% |"
            )

    if actuals_lines:
        parts += [
            "",
            "=== Actual Delivered Results (recent quarters) ===",
            "| Metric / Quarter | Values | YoY |",
            "| :--- | :--- | :--- |",
        ] + actuals_lines

    parts.append("[/HISTORICAL_CONTEXT]")
    return "\n".join(parts)


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
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill mode: use the dedicated key pool "
                             "(BACKFILL_GEMINI_KEY*), route the daily digest to "
                             "daily_backfill_*.md, and stamp backfill_process_date "
                             "on each processed queue row. Same extractor/prompt/"
                             "tables as Phase 2 otherwise.")
    parser.add_argument("--key-prefix", default=None,
                        help="Env-var prefix for the Gemini key pool. Defaults to "
                             "GEMINI_API_KEY (Phase 2), or BACKFILL_GEMINI_KEY when "
                             "--backfill is set. Override here if needed.")
    parser.add_argument("--no-lock", action="store_true",
                        help="Skip the Drive mutual-exclusion lock (testing only).")
    parser.add_argument("--check-keys", action="store_true",
                        help="Print the selected key-pool size and exit (no Gemini, no Drive).")
    args = parser.parse_args()

    # Backfill defaults to the dedicated key pool unless an explicit prefix is given.
    key_prefix = args.key_prefix or ("BACKFILL_GEMINI_KEY" if args.backfill
                                     else "GEMINI_API_KEY")

    # Zero-cost pool verification: confirm the keys loaded, then exit.
    if args.check_keys:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        ks = load_keys(os.environ, prefix=key_prefix)
        print(f"Key pool '{key_prefix}': {len(ks)} key(s) × {len(CONCALL_MODELS)} "
              f"model(s) = {len(ks) * len(CONCALL_MODELS)} daily buckets")
        sys.exit(0 if ks else 1)

    print("Phase 2 / Stage B — Concall extraction via Gemini")
    print("-" * 56)

    drive = get_drive()

    # Load Gemini API keys and build the bucket pool (keys × CONCALL_MODELS).
    # Backfill uses a dedicated pool (separate Cloud projects) so its quota is
    # fully independent of the live Phase 2 pool — T1.4.
    api_keys = load_keys(os.environ, prefix=key_prefix)
    if not api_keys:
        print(f"ERROR: no {key_prefix} or {key_prefix}_* found in .env")
        sys.exit(1)
    log(f"Key pool: prefix '{key_prefix}'"
        + ("  [BACKFILL MODE]" if args.backfill else ""))
    gemini = BucketPool(api_keys, CONCALL_MODELS,
                        inter_call_s=INTER_CALL_SLEEP, logger=log)
    log(f"Loaded {len(api_keys)} key(s) × {len(CONCALL_MODELS)} model(s) "
        f"= {len(api_keys) * len(CONCALL_MODELS)} buckets")

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

    # Mutual-exclusion lock so a live run and a backfill run can't clobber the
    # shared _index parquets concurrently (T1.4). Released on exit via atexit.
    if not args.no_lock and not args.dry_run:
        mode = "backfill" if args.backfill else "live"
        if not acquire_extract_lock(drive, index_id, mode):
            sys.exit(0)
        atexit.register(release_extract_lock, drive, index_id)

    # Load queue and filter to pending concalls
    queue = load_queue(drive, index_id)
    # ORIGIN ROUTING (T1 fix): each extractor drains ONLY its own rows so a
    # backfill-fetched concall never lands in the live daily digest or burns the
    # main key pool (and vice-versa). A blank/absent source is legacy => "live".
    _src = queue["source"].astype(str).str.strip().str.lower()
    is_backfill_row = _src.eq("backfill")
    origin_mask = is_backfill_row if args.backfill else (~is_backfill_row)
    pending_mask = ((queue["status"] == "pending")
                    & (queue["doc_type"] == "concall")
                    & origin_mask)
    # Process OLDEST -> NEWEST per company so each company's GF1 guidance history
    # accrues before its latest concall is processed; only then does the latest
    # concall see prior history and emit GF_TRACK. Harmless for Phase 2's 2-day
    # window; essential for backfill. Sort key: (isin, announcement_date ASC).
    pending = queue[pending_mask].copy()
    pending["_ann"] = pd.to_datetime(
        pending["announcement_date"].astype(str).str[:10], errors="coerce"
    )
    pending = pending.sort_values(["isin", "_ann"], na_position="first")
    pending_idx = pending.index.tolist()

    log(f"Queue: {len(queue)} total rows, {len(pending_idx)} pending "
        f"{'BACKFILL' if args.backfill else 'LIVE'} concalls "
        f"(processed oldest->newest per company)")

    if args.limit:
        pending_idx = pending_idx[: args.limit]

    # ---- Load context caches once per run (GF_TRACK historical credibility) ----
    # Loaded here so we read Drive once, not once per document.
    _gf1_cache = _load_parquet(drive, index_id, "gf1_guidance_statements.parquet", GF1_COLS)
    _qfacts_cache = _load_parquet(drive, index_id, "quarterly_facts.parquet", QFACTS_COLS)
    _results_cache: pd.DataFrame = pd.DataFrame()
    try:
        _res_fid = find_file(drive, index_id, "results.parquet")
        if _res_fid:
            _results_cache = pd.read_parquet(io.BytesIO(download_bytes(drive, _res_fid)))
    except Exception as _e:
        log(f"  NOTE: results.parquet not loaded for context ({str(_e)[:60]})")
    log(f"Context caches: GF1={len(_gf1_cache)} rows · "
        f"facts={len(_qfacts_cache)} rows · results={len(_results_cache)} rows")

    counts = {"processed": 0, "error": 0, "skipped": 0, "skipped_dup": 0,
              "superseded": 0, "gf1": 0,
              "gf2": 0, "gf3": 0, "gf4": 0, "gf_track": 0, "mgmt_cred": 0}

    # Run-local guard: (isin, quarter) already written THIS run. Combined with the
    # quarterly_facts cache below, this is the backstop that stops the same concall
    # quarter being summarised twice (e.g. a doc queued via both live feed and
    # backfill with different doc_ids) even if it slips past enqueue-time dedup.
    _seen_quarter_keys: set[tuple[str, str]] = set()

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

        # Download PDF outside the main try so a transient Drive error doesn't
        # get misclassified as a permanent row 'error'.
        try:
            pdf_bytes = download_bytes(drive, drive_fid)
            log(f"  PDF: {len(pdf_bytes):,} bytes")
        except Exception as exc:
            log(f"  Drive download failed ({str(exc)[:80]}) — leaving pending")
            counts["skipped"] += 1
            continue

        try:
            # Build historical context for GF_TRACK (returns None if no prior history)
            _hist = _build_historical_context(_gf1_cache, _qfacts_cache, _results_cache, row)
            effective_prompt = prompt + "\n\n" + _hist if _hist else prompt
            if _hist:
                log(f"  GF_TRACK: injecting {len(_hist):,}-char historical context")

            # Generate via the bucket pool (best model first; bounded fallback).
            markdown_text, model_used = gemini.call_pdf(pdf_bytes, effective_prompt)
            log(f"  Gemini response: {len(markdown_text):,} chars [{model_used}]")

            if args.dry_run:
                print(f"\n{'='*60}\nDRY RUN — {row.get('symbol')}\n"
                      f"{markdown_text[:800]}\n{'='*60}\n")
                counts["processed"] += 1
                continue

            now_str = datetime.now().isoformat(timespec="seconds")

            # 4. Parse Table_A → quarterly facts + guidance rows
            facts, guidance_rows = parse_gemini_response(markdown_text, row)
            quarter = facts["quarter"]
            facts["response_chars"] = len(markdown_text)   # richness proxy
            log(f"  Parsed: quarter={quarter or 'unknown'}, "
                f"guidance_rows={len(guidance_rows)}, "
                f"response_chars={len(markdown_text):,}")

            # DUP-QUARTER / SUPERSEDE: if this company+quarter was already summarised
            # by a DIFFERENT document, decide whether to replace or skip.
            #
            # Rule (mirrors user policy):
            #   • Same run (_seen_quarter_keys): always skip — no comparison possible.
            #   • Cross-run: compare response_chars.
            #       new >= existing * SUPERSEDE_THRESHOLD → SUPERSEDE (richer doc wins,
            #         old .md section replaced in-place, old parquet rows purged).
            #       new < threshold → skip (true duplicate, shorter/same doc).
            #       Legacy rows with no response_chars → always supersede (unknown richness).
            _this_doc = str(facts.get("source_doc_id") or row.get("doc_id") or "")
            _isin_key = str(row.get("isin") or row.get("key") or "").strip()
            _this_chars = len(markdown_text)
            _dup_quarter = False
            _supersede = False
            _old_source_doc_id = ""

            _norm_q = _norm_quarter(str(quarter)) if quarter else ""
            if _norm_q and _isin_key:
                _qkey = (_isin_key, _norm_q)
                if _qkey in _seen_quarter_keys:
                    _dup_quarter = True   # same run — always skip
                elif not _qfacts_cache.empty:
                    _m = ((_qfacts_cache["isin"].astype(str) == _isin_key)
                          & (_qfacts_cache["quarter"].astype(str).map(_norm_quarter) == _norm_q)
                          & (_qfacts_cache["source_doc_id"].astype(str) != _this_doc))
                    if _m.any():
                        _old_chars = pd.to_numeric(
                            _qfacts_cache.loc[_m, "response_chars"], errors="coerce"
                        ).fillna(0).max()
                        _old_source_doc_id = str(
                            _qfacts_cache.loc[_m, "source_doc_id"].iloc[0])
                        if _old_chars == 0 or _this_chars >= _old_chars * SUPERSEDE_THRESHOLD:
                            _supersede = True
                            log(f"  SUPERSEDE: {row.get('symbol')} {quarter} — "
                                f"new {_this_chars:,} chars vs existing "
                                f"{int(_old_chars):,} — replacing.")
                        else:
                            _dup_quarter = True
                            log(f"  DUP-QUARTER (skip): {row.get('symbol')} {quarter} — "
                                f"new {_this_chars:,} not >{SUPERSEDE_THRESHOLD:.0%} "
                                f"of existing {int(_old_chars):,} — skipping.")
                _seen_quarter_keys.add(_qkey)

            if _supersede:
                # Purge old quarter data from all parquets + mark old queue row
                _purge_quarter_records(drive, index_id, _isin_key, str(quarter))
                _old_mask = queue["doc_id"].astype(str) == _old_source_doc_id
                if _old_mask.any():
                    queue.loc[_old_mask, "status"] = "superseded"
                counts["superseded"] = counts.get("superseded", 0) + 1
                # Fall through — normal write path below handles the new document.

            elif _dup_quarter:
                queue.loc[queue_idx, "status"] = "done"
                queue.loc[queue_idx, "processed_at"] = now_str
                if args.backfill:
                    queue.loc[queue_idx, "backfill_process_date"] = \
                        datetime.now().strftime("%Y-%m-%d")
                save_queue(drive, index_id, queue)
                counts["skipped_dup"] += 1
                continue

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

            # 4c. Parse GF_TRACK "Said vs Delivered" -> mgmt_credibility rows.
            # Only present when historical context was injected (history exists).
            mgmt_cred_rows: list[dict] = []
            if _hist:
                mgmt_cred_rows = parse_gf_track(markdown_text, row, quarter, now_str)
                if mgmt_cred_rows:
                    counts["gf_track"] += 1
                    counts["mgmt_cred"] += len(mgmt_cred_rows)
                    log(f"  GF_TRACK section: present "
                        f"({len(mgmt_cred_rows)} credibility row(s))")

            # 5a. Company page (persisted forever)
            if OUTPUT_COMPANY_MD:
                _cp_key = str(
                    row.get("key") or row.get("isin") or row.get("symbol") or "")
                if _supersede and quarter:
                    # Replace the existing quarter section in-place (richer doc wins)
                    replace_company_page_section(
                        drive, repo_id,
                        key=_cp_key,
                        quarter=quarter,
                        content=markdown_text,
                        doc_title=str(row.get("title", "")),
                    )
                else:
                    append_company_page(
                        drive, repo_id,
                        key=_cp_key,
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
                    backfill=args.backfill,
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
                # GF_TRACK credibility (only populated when history existed)
                if mgmt_cred_rows:
                    upsert_mgmt_credibility(drive, index_id, mgmt_cred_rows)

            # 7. Mark done
            queue.loc[queue_idx, "status"] = "done"
            queue.loc[queue_idx, "processed_at"] = now_str
            if args.backfill:
                queue.loc[queue_idx, "backfill_process_date"] = \
                    datetime.now().strftime("%Y-%m-%d")
            save_queue(drive, index_id, queue)

            counts["processed"] += 1
            log(f"  Done: {row.get('symbol')}")

        except AllBucketsExhausted as exc:
            # Transient/quota: every remaining row faces the same dead buckets,
            # so stop now. Row stays 'pending' (untouched) and resumes next run.
            counts["deferred"] = len(pending_idx) - counts["processed"] \
                - counts["error"] - counts["skipped"]
            log(f"Stopping: {exc}. {counts['deferred']} concall(s) deferred — "
                f"resume after next free bucket / reset (~13:30 IST).")
            break

        except FatalCallError as exc:
            # Deterministic failure for THIS document (bad PDF, blocked, 400).
            log(f"  FATAL (this doc): {str(exc)[:120]}")
            queue.loc[queue_idx, "status"] = "error"
            queue.loc[queue_idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            save_queue(drive, index_id, queue)
            counts["error"] += 1

        except Exception as exc:
            log(f"  ERROR (post-processing): {str(exc)[:120]}")
            queue.loc[queue_idx, "status"] = "error"
            queue.loc[queue_idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            save_queue(drive, index_id, queue)
            counts["error"] += 1

    # CSV snapshots (once per run, after all processing)
    if OUTPUT_GF_CSV and counts["processed"] > 0 and not args.dry_run:
        log("Writing CSV snapshots...")
        write_csv_exports(drive, index_id)

    pool = gemini.summary()
    print("-" * 56)
    print(f"Processed : {counts['processed']}")
    print(f"Deferred  : {counts.get('deferred', 0)}  (still pending — quota/transient)")
    print(f"Errors    : {counts['error']}  (bad PDF / deterministic)")
    print(f"Skipped   : {counts['skipped']}  (no drive_file_id / download fail)")
    print(f"Dup-qtr   : {counts['skipped_dup']}  (quarter already summarised — marked done)")
    print(f"Superseded: {counts.get('superseded', 0)}  (richer doc replaced existing quarter section)")
    print(f"GF rows   : GF1={counts['gf1']} GF2={counts['gf2']} "
          f"GF3={counts['gf3']} GF4={counts['gf4']}")
    print(f"GF_TRACK  : {counts['gf_track']} credibility sections · "
          f"{counts['mgmt_cred']} mgmt_credibility rows "
          f"(0 = no prior history yet; expected once backfill is processed)")
    print(f"Calls OK by model : {pool['by_model'] or '{}'}")
    print(f"Buckets   : {pool['buckets_alive']} alive · "
          f"{pool['buckets_dead_today']} dead-today · "
          f"{pool['buckets_parked']} parked  (of {pool['buckets_total']}) "
          f"· {pool['elapsed_s']}s")
    if not args.dry_run and counts["processed"] > 0:
        print("Output: company_repo/_index/quarterly_facts.parquet")
        print("Output: company_repo/_index/guidance_tracker.parquet")
        print("Output: company_repo/_index/gf1_guidance_statements.parquet (.csv)")
        print("Output: company_repo/_index/gf2_historical_guidance.parquet (.csv)")
        print("Output: company_repo/_index/gf3_operational_visibility.parquet (.csv)")
        print("Output: company_repo/_index/gf4_quality_flags.parquet (.csv)")
        print("Output: company_repo/_index/mgmt_credibility.parquet (.csv)")
        print("Output: company_repo/<key>/company_page.md")
        print("Output: company_repo/_daily/concall_DD_MMMYYYY.md")
        print("Output: company_repo/_quarterly/QXFY_mgmt_guidance.md")


if __name__ == "__main__":
    main()
