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
import json
import os
import re
import time
from datetime import datetime, timedelta
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
#  Structured-tabulation helpers (shared by AR/presentation/rating)   #
#  Generic version of the proven AR pattern: a SEPARATE bounded JSON-  #
#  only pass over a produced report, parsed by SALVAGE (every flat     #
#  {...} object) + dedupe — robust to a lite model that truncates or   #
#  loops. Pure functions; additive (existing extractors are unchanged).#
# ------------------------------------------------------------------ #

_FLAT_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
# Absence-of-concern / clean / missing-data phrasing — these are NOT real flags.
_CLEAN_FLAG_RE = re.compile(
    r"\b(no material|no significant|no adverse|no emphasis|no eom|no qualif\w*|"
    r"no red[- ]?flag|no concern|no issue|no change[s]? in|no instance|"
    r"consistent with (the )?prior|in line with prior|within (the )?(acceptable|normal)|"
    r"generally align\w*|no unusual|is (minimal|negligible|immaterial)|"
    r"are (minimal|negligible|immaterial)|marked as|"
    r"none (were |was )?(noted|disclosed|observed|reported))\b", re.IGNORECASE)


def sstr(v):
    """Trimmed string, or None for empty/null (keeps parquet nulls clean)."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def fnum(v):
    """Coerce to float, guarding None/blank/garbage -> None."""
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None


def clamp(v, allowed: set, default: str) -> str:
    """Lower-cased value if in `allowed`, else `default`."""
    s = str(v or "").strip().lower()
    return s if s in allowed else default


def is_real_flag(evidence) -> bool:
    """True only for an ACTUAL concern. Drops blank, short DATA_MISSING, and
    clean/absence statements (a clean entity should surface few/no flags)."""
    e = (evidence or "").strip()
    if not e:
        return False
    if "DATA_MISSING" in e.upper() and len(e) < 60:
        return False
    if _CLEAN_FLAG_RE.search(e):
        return False
    return True


def upsert_structured(drive, index_id: str, filename: str, cols: list,
                      rows: list) -> None:
    """Delete existing rows for this source_doc_id, append new (idempotent re-extract).
    Generic version of the AR _upsert_ar; used by presentation/rating tabulation."""
    if not rows:
        return
    df = load_parquet(drive, index_id, filename, cols)
    sdid = str(rows[0].get("source_doc_id", ""))
    if sdid and "source_doc_id" in df.columns:
        df = df[df["source_doc_id"].astype(str) != sdid]
    new_df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    df = pd.concat([df, new_df], ignore_index=True)
    save_parquet(drive, index_id, filename, df)


def call_over_doc(gemini, prompt: str, doc_bytes: bytes, *,
                  max_output_tokens: int | None = None, max_text_chars: int = 120000,
                  name: str = "doc") -> str:
    """Call Gemini with `prompt` over a source document, AUTO-DETECTING bytes:
    `%PDF…` → call_pdf (inline PDF); otherwise → call_text(prompt + decoded text).
    This stops HTML rationales (CRISIL/SMERA/Brickwork) being sent as application/pdf
    (a fatal error that marks the doc 'error'). Empty string when no doc/prompt."""
    if not prompt or not doc_bytes:
        return ""
    if doc_bytes[:5].startswith(b"%PDF"):
        return gemini.call(doc_bytes, prompt, name, max_output_tokens=max_output_tokens)
    text = doc_bytes.decode("utf-8", "replace")[:max_text_chars]
    return gemini.call_text(prompt + "\n\nDOCUMENT:\n" + text, name,
                            max_output_tokens=max_output_tokens)


def run_structured_over_doc(gemini, struct_prompt: str, doc_bytes: bytes, *,
                            max_output_tokens: int = 4096, max_text_chars: int = 60000,
                            name: str = "struct") -> str:
    """Structured JSON-only pass over the source doc (back-compat wrapper over
    call_over_doc) — feeds the source straight to the focused JSON prompt."""
    return call_over_doc(gemini, struct_prompt, doc_bytes,
                         max_output_tokens=max_output_tokens,
                         max_text_chars=max_text_chars, name=name)


GEMINI_USAGE_COLS = ["ts", "doc_type", "source", "key_idx", "model",
                     "ok", "fail", "rpm_cool", "overload_503", "state"]


def persist_gemini_usage(drive, index_id: str, summary: dict, doc_type: str,
                         source: str, keep_days: int = 30) -> None:
    """Append the pool's per-(key, model) attribution from BucketPool.summary() to
    gemini_usage.parquet, so ops-mail can show exactly which keys/models summarised
    how many docs and why others stopped (rpm_cool = PerMinute, overload_503, or
    state=dead_today = PerDay). Best-effort; never raise (logging must not break a run)."""
    try:
        rows = (summary or {}).get("buckets") or []
        ts = datetime.now().isoformat(timespec="seconds")
        recs = [{"ts": ts, "doc_type": doc_type, "source": source,
                 "key_idx": r.get("key_idx"), "model": r.get("model"),
                 "ok": int(r.get("ok", 0)), "fail": int(r.get("fail", 0)),
                 "rpm_cool": int(r.get("rpm_cool", 0)),
                 "overload_503": int(r.get("overload_503", 0)),
                 "state": str(r.get("state", ""))}
                for r in rows]
        # keep only buckets that did something this run (lean table)
        recs = [r for r in recs if r["ok"] or r["fail"] or r["rpm_cool"] or r["overload_503"]]
        if not recs:
            return
        df = load_parquet(drive, index_id, "gemini_usage.parquet", GEMINI_USAGE_COLS)
        df = pd.concat([df, pd.DataFrame(recs)], ignore_index=True)
        cut = (datetime.now() - timedelta(days=keep_days)).isoformat()
        df = df[df["ts"].astype(str) >= cut].reset_index(drop=True)
        save_parquet(drive, index_id, "gemini_usage.parquet", df)
        log(f"  gemini_usage: logged {len(recs)} bucket rows "
            f"(ok={sum(r['ok'] for r in recs)}).")
    except Exception as e:
        log(f"  WARNING: gemini_usage logging failed ({str(e)[:80]}).")


def salvage_json_objects(text: str) -> list[dict]:
    """Parse every FLAT {...} object in the text individually — robust to a truncated
    or repetition-looped response (complete objects recovered, trailing partial one
    skipped). Structured rows are flat (no nesting), so they survive an unterminated
    enclosing array/object."""
    out: list[dict] = []
    for m in _FLAT_OBJ_RE.findall(text or ""):
        try:
            o = json.loads(m)
            if isinstance(o, dict):
                out.append(o)
        except Exception:
            continue
    return out


# ------------------------------------------------------------------ #
#  Gemini key pool with round-robin rotation on 429                   #
# ------------------------------------------------------------------ #

# P1 doc-types (results / rating / presentation / annual_report) are PF-only and
# low-volume. They run on LITE models, kept disjoint from concall's quality chain
# so P1 can never consume concall's premium (key, model) buckets.
# gemini-2.0-flash ADDED (additive) — a separate per-(project,model) daily-quota
# bucket that is reliably up (the catalyst pool uses it), so the pool has more total
# free-tier quota to draw on. Nothing removed; the startup probe drops any that flap.
# The new gemini_usage.parquet log shows which models actually contribute.
P1_MODELS = ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.0-flash"]

# Extra models added to the BACKFILL chain ONLY (not Phase-2 PF) for more per-(project,
# model) daily-quota buckets. Measured live (2026-06-22) as having free quota on the
# FREE_POOL projects. The startup probe drops any that flap. Phase-2 PF extracts keep
# exactly P1_MODELS, so Phase 2 is unchanged.
BACKFILL_EXTRA_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash"]


class RateLimitExhausted(Exception):
    """Raised when every (key, model) bucket is exhausted/transient. Callers
    treat this as 'stop the stage, leave rows pending for the next run'.
    Kept for backward compatibility with the P1 extractors' except clauses."""


class GeminiKeyPool:
    """Back-compat adapter over the bounded BucketPool engine (gemini_pool.py).

    Preserves the (api_keys, model) constructor and the string-returning
    .call()/.call_text() API the P1 extractors already use, so those files need
    no structural change. `model` may be a single model string or a list (a
    fallback chain). All the checks-and-balances — error-typed handling,
    per-(key,model) daily-quota tracking, model fallback, bounded retries, and
    the stage wall-clock ceiling — come from BucketPool.

    Mapping of the engine's outcomes onto the legacy contract:
      * AllBucketsExhausted (transient/quota) -> RateLimitExhausted  (defer)
      * FatalCallError      (bad PDF / 400)   -> propagates as Exception (row error)
    """

    def __init__(self, api_keys: list[str], model):
        from gemini_pool import BucketPool
        models = model if isinstance(model, (list, tuple)) else [model]
        self._pool = BucketPool(list(api_keys), list(models), logger=log)

    def _guard(self, fn):
        from gemini_pool import AllBucketsExhausted
        try:
            text, _model_used = fn()
            return text
        except AllBucketsExhausted as exc:
            raise RateLimitExhausted(str(exc))

    def call(self, pdf_bytes: bytes, prompt: str, display_name: str,
             max_output_tokens: int | None = None) -> str:
        """Generate from inline PDF bytes + prompt (bounded key/model fallback).
        `max_output_tokens` (default None) bounds the response — unchanged when None."""
        return self._guard(
            lambda: self._pool.call_pdf(pdf_bytes, prompt,
                                        max_output_tokens=max_output_tokens))

    def call_text(self, prompt: str, display_name: str,
                  max_output_tokens: int | None = None) -> str:
        """Generate from a text-only prompt (synthesis passes)."""
        return self._guard(
            lambda: self._pool.call_text(prompt,
                                         max_output_tokens=max_output_tokens))

    def summary(self) -> dict:
        return self._pool.summary()

    def prime_from_health(self, drive, index_id: str) -> None:
        """Phase 1 — pre-mark PerDay-dead buckets from gemini_usage.parquet (see
        prime_pool_from_health). Convenience so AR/rating extractors can prime their
        wrapped pool directly."""
        prime_pool_from_health(self._pool, drive, index_id)


def prime_pool_from_health(pool, drive, index_id: str) -> None:
    """Phase 1 — read gemini_usage.parquet and pre-mark (key, model) buckets that hit
    PerDay quota since the last reset as DEAD_TODAY, so this run skips them instead of
    burning a real call to re-discover each. Best-effort: never raises (a logging/cache
    miss must not break extraction). `pool` may be a BucketPool or a GeminiKeyPool."""
    try:
        from datetime import datetime
        from bucket_health import dead_buckets_since_reset
        target = getattr(pool, "_pool", pool)   # unwrap GeminiKeyPool if needed
        df = load_parquet(drive, index_id, "gemini_usage.parquet", GEMINI_USAGE_COLS)
        dead = dead_buckets_since_reset(df, datetime.utcnow())
        if dead:
            n = target.prime_dead_buckets(dead)
            log(f"  bucket-health: primed {n} PerDay-dead bucket(s) since last reset "
                f"(skipping re-discovery this run).")
    except Exception as e:                       # never break a run over the cache
        log(f"  WARNING: bucket-health priming skipped ({str(e)[:80]}).")


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
    """Append analysis to _daily/<doc_type>_DD_MMMYYYY.md (persisted forever)."""
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
                  f"{str(announcement_date)[:10]}\n")
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
#  Portfolio filter                                                    #
# ------------------------------------------------------------------ #

# PF holdings live in two folders historically: pf_tracking/ (where sync_pf.bat
# uploads the LIVE list) and portfolio/ (older). The SINGLE source of truth is the
# most-recently-modified holdings file across BOTH — so whichever you upload last
# wins, every program stays consistent, and a stale folder can't mislead anything.
PF_FOLDERS = ("pf_tracking", "portfolio")


def find_latest_portfolio_file(drive, folder_id: str, folder_names=PF_FOLDERS):
    """Newest .xls/.xlsx/.csv across the given subfolders (by modifiedTime).
    Returns the Drive file dict {id,name,modifiedTime,_folder} or None.
    Read-only: never creates a folder."""
    best = None
    for fn in folder_names:
        q = (f"name='{fn}' and '{folder_id}' in parents "
             f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
        fol = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not fol:
            continue
        files = drive.files().list(
            q=f"'{fol[0]['id']}' in parents and trashed=false",
            fields="files(id, name, modifiedTime)", orderBy="modifiedTime desc",
        ).execute().get("files", [])
        cand = next((f for f in files
                     if f["name"].lower().endswith((".xls", ".xlsx", ".csv"))), None)
        if cand and (best is None or cand["modifiedTime"] > best["modifiedTime"]):
            cand = dict(cand, _folder=fn)
            best = cand
    return best


def isin_symbol_map(*frames) -> dict:
    """isin -> symbol unioned across several tables (first NON-BLANK wins) so a
    gap in one source is auto-healed by another. Pass DataFrames that have `isin`
    + `symbol` columns (master_list, screener_grades, guidance_tracker,
    announcement_ledger…). screener_grades leaves SME symbols blank; guidance /
    master_list fill them. Pure (no I/O) — callers pass already-loaded frames."""
    m: dict = {}
    for df in frames:
        if df is None or getattr(df, "empty", True):
            continue
        if not {"isin", "symbol"} <= set(df.columns):
            continue
        for i, s in zip(df["isin"].astype(str), df["symbol"].astype(str)):
            i, s = i.strip(), s.strip()
            if i and s and s.lower() != "nan" and i not in m:
                m[i] = s
    return m


def load_portfolio_isins(drive, folder_id: str,
                         folder_name=None) -> set[str] | None:
    """Return ISIN set from the LIVE portfolio holdings file.

    Single source of truth = the most-recently-modified .xls/.xlsx/.csv across
    BOTH pf_tracking/ and portfolio/ (auto-heals — newest upload wins, no matter
    the folder). Pass folder_name='x' to restrict to one folder.
      - Auto-detects header row (Screener exports have ~13 blank rows first)
      - Returns frozenset of ISIN strings, or None when no file found
        (callers fall back to processing all companies when None is returned)

    Called by extract_results/rating/presentation/annual_report, ingest_announcements,
    build_catalyst_notes, build_investigative_fraud, run_backfill, run_pf_digest,
    ingest_company_docs — NOT by extract_concall (concall stays universal)."""
    names = (folder_name,) if folder_name else PF_FOLDERS
    target = find_latest_portfolio_file(drive, folder_id, names)
    if not target:
        log(f"  Portfolio filter: no holdings file in {'/'.join(names)} — processing all companies")
        return None

    log(f"  Portfolio filter: reading '{target['name']}' (newest across {','.join(names)}, "
        f"from {target.get('_folder', '?')}/)")
    raw = download_bytes(drive, target["id"])
    fn = target["name"].lower()

    try:
        if fn.endswith(".csv"):
            df_raw = pd.read_csv(io.BytesIO(raw), header=None)
            engine = "csv"
        else:
            engine = "xlrd" if fn.endswith(".xls") else "openpyxl"
            df_raw = pd.read_excel(io.BytesIO(raw), engine=engine, header=None)

        # Find header row: first row containing an "ISIN" cell
        header_row = None
        for i, row in df_raw.iterrows():
            if any(str(v).strip().upper() == "ISIN" for v in row.dropna()):
                header_row = i
                break
        if header_row is None:
            log("  Portfolio filter: ISIN column not found in file — processing all companies")
            return None

        if fn.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw), header=header_row)
        else:
            df = pd.read_excel(io.BytesIO(raw), engine=engine, header=header_row)

        df = df.dropna(subset=["ISIN"]).copy()
        isins: set[str] = set(df["ISIN"].astype(str).str.strip())
        log(f"  Portfolio filter: {len(isins)} ISINs loaded — non-portfolio rows skipped "
            f"(stay pending; processed if added to portfolio later)")
        return isins

    except Exception as exc:
        log(f"  Portfolio filter: ERROR reading file ({str(exc)[:120]}) — processing all companies")
        return None


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


# ------------------------------------------------------------------ #
#  Generic Drive mutual-exclusion lock (Phase 3)                      #
#  For scripts that read-modify-write the SAME _index parquet so two  #
#  runs can't clobber each other. Stale locks auto-steal after        #
#  max_age_min (crashed run). Acquire returns True on success.        #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  PHASE-2 PRIORITY BEACON                                            #
#  Phase 2 (high priority) drops this marker while it wants the lock; #
#  backfill (low priority) sees it and steps aside — at acquire time  #
#  AND between documents — so Phase 2 always wins within ~one doc.    #
# ------------------------------------------------------------------ #
_PHASE2_BEACON = "_phase2_active"
PHASE2_BEACON_MAX_AGE_MIN = 20   # stale (crashed Phase 2) after this → ignored


def set_phase2_beacon(drive, index_id: str) -> None:
    """Announce 'Phase 2 wants the lock'. Idempotent; refreshes the timestamp."""
    try:
        fid = find_file(drive, index_id, _PHASE2_BEACON)
        payload = datetime.now().isoformat(timespec="seconds").encode("utf-8")
        upload_bytes(drive, index_id, _PHASE2_BEACON, payload, "text/plain",
                     existing_id=fid)
    except Exception as e:
        log(f"  BEACON set failed ({str(e)[:60]}) — continuing.")


def clear_phase2_beacon(drive, index_id: str) -> None:
    try:
        fid = find_file(drive, index_id, _PHASE2_BEACON)
        if fid:
            drive.files().delete(fileId=fid).execute()
    except Exception as e:
        log(f"  BEACON clear failed ({str(e)[:60]}) — will go stale on its own.")


def phase2_beacon_fresh(drive, index_id: str,
                        max_age_min: int = PHASE2_BEACON_MAX_AGE_MIN) -> bool:
    """True if Phase 2 has a fresh active beacon (so backfill should step aside)."""
    try:
        fid = find_file(drive, index_id, _PHASE2_BEACON)
        if not fid:
            return False
        ts_str = download_bytes(drive, fid).decode("utf-8", errors="replace").strip()
        ts = datetime.fromisoformat(ts_str) if ts_str else None
        if not ts:
            return False
        return (datetime.now() - ts).total_seconds() / 60.0 < max_age_min
    except Exception:
        return False                                 # fail-open: don't block backfill


def acquire_lock(drive, index_id: str, lock_name: str, owner: str,
                 max_age_min: int = 180, grace_sec: int = 8,
                 wait_min: float = 0.0, defer_to_phase2: bool = False) -> bool:
    """Claim the shared lock; return False if another process holds it.

    Priority coordination (T12):
      • defer_to_phase2 (backfill callers): if Phase 2 has a fresh beacon, yield the
        lock immediately so Phase 2 can take it.
      • wait_min > 0 (Phase 2 callers): announce the beacon and POLL for the lock up
        to wait_min minutes before giving up — backfill yields within ~one document,
        so Phase 2 normally acquires in well under 2 min; wait_min is just the cap.

    A genuinely-held lock persists; a just-released one disappears once Drive's
    file-list index settles, so a fresh foreign lock is re-checked after a short nap
    before yielding (absorbs sequential hand-off lag). Uncontended acquires pay
    nothing."""
    if defer_to_phase2 and phase2_beacon_fresh(drive, index_id):
        log(f"  Phase 2 active — '{owner}' yields {lock_name}, exiting cleanly.")
        return False
    if wait_min > 0:
        set_phase2_beacon(drive, index_id)           # backfill, step aside

    deadline = time.time() + max(wait_min * 60.0, float(grace_sec))
    fid = None
    while True:
        fid = find_file(drive, index_id, lock_name)
        if not fid:
            break                                    # free — acquire below
        try:
            content = download_bytes(drive, fid).decode("utf-8", errors="replace")
            ts_str = content.split("|", 2)[1] if "|" in content else ""
            ts = datetime.fromisoformat(ts_str) if ts_str else None
            age_min = ((datetime.now() - ts).total_seconds() / 60.0) if ts else 1e9
        except Exception as e:
            log(f"  LOCK {lock_name} read failed ({str(e)[:60]}) — overwriting.")
            break
        if age_min >= max_age_min:
            log(f"  LOCK {lock_name} stale ({age_min:.0f} min) — stealing.")
            break
        now = time.time()
        if now < deadline:
            if wait_min > 0:
                set_phase2_beacon(drive, index_id)   # keep announcing while we wait
                log(f"  LOCK {lock_name} held by '{content.split('|')[0]}' "
                    f"({age_min:.0f} min) — Phase 2 waiting…")
                time.sleep(min(30.0, max(1.0, deadline - now)))
            else:
                time.sleep(min(float(grace_sec), max(0.5, deadline - now)))
            continue                                 # re-check
        log(f"  LOCK {lock_name} held by '{content.split('|')[0]}' "
            f"({age_min:.0f} min) — exiting cleanly.")
        return False
    payload = f"{owner}|{datetime.now().isoformat(timespec='seconds')}".encode("utf-8")
    upload_bytes(drive, index_id, lock_name, payload, "text/plain", existing_id=fid)
    return True


def release_lock(drive, index_id: str, lock_name: str) -> None:
    try:
        fid = find_file(drive, index_id, lock_name)
        if fid:
            drive.files().delete(fileId=fid).execute()
    except Exception as e:
        log(f"  LOCK {lock_name} release failed ({str(e)[:60]}) — auto-stolen later.")
