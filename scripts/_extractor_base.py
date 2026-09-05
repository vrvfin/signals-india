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
              "drive_file_id", "status", "discovered_at", "processed_at",
              # Failure diagnostics (added 2026-08-15, ADDITIVE — old rows read None).
              # The queue recorded status='error' and nothing else, so 2,326 failures
              # carried no reason at all and their causes had to be reverse-engineered
              # from filing titles and date histograms. Any writer that sets an error
              # status should also record WHY, via mark_queue_error() below.
              # Safe to add: load_queue back-fills missing columns and never slices,
              # and save_queue writes the frame as-is, so extra columns survive every
              # load/save cycle in every pipeline (this is how backfill_process_date
              # and source already work).
              "attempts", "last_error", "last_attempt_at"]


# A model sometimes returns the PROMPT instead of an answer. Measured 2026-09-02:
# CG Power's stored annual-report analysis opens with annual_report_prompt.txt itself
# ("Generate the final report immediately... The ENTIRE report must stay under ~1,200
# lines"). That text is long and prose-like, so every length-based quality test rates it
# highly and every "best passage" heuristic picks it first. Two or more of these phrases
# together is the signal; one alone can appear in genuine prose.
_PROMPT_ECHO_MARKERS = (
    "generate the final report", "visible report structure", "no individual paragraph",
    "must stay under", "output must be", "formatting style", "make absolutely zero",
    "do not continue or loop", "report structure", "you are a lead forensic",
    "your task is", "never repeat a sentence",
)


# THE MARKER LIST ABOVE ONLY KNOWS THE ANNUAL-REPORT PROMPT, and a model can echo any
# of them. AVALON's stored concall was 260,730 chars that opened with concall_prompt.txt
# verbatim — "[FINANCIAL GRID ENFORCEMENT RULES]", "[CROSS-INDUSTRY METRIC ADAPTATION
# MAPPING]", the whole RATE/ABS/LVL instruction block — and only THEN gave a real
# Table_A of Avalon's numbers. Zero of the twelve markers matched, so the mail carried
# the instructions to the reader.
#
# A response that quotes its own prompt back is provable rather than guessable: compare
# it against the prompt FILES on disk. A genuine summary does not reproduce three
# separate 45-character instruction lines word for word.
_PROMPT_LINE_MIN = 45      # short lines are shared by chance; long ones are not
_PROMPT_LINE_HITS = 3      # three verbatim INSTRUCTION lines is not coincidence
_prompt_lines_cache: list | None = None

# A LINE THE MODEL IS TOLD TO REPRODUCE IS NOT AN ECHO. The concall prompt names the
# output sections and the model is required to write those headings out; measured across
# 831 stored concall sections on 2026-09-05, the distribution of verbatim matches is
# bimodal — 416 sections at ZERO and a spike of 355 at exactly FOUR, and the four are:
#     "section gf3 - operational visibility extraction"        (398 sections)
#     "e-1) forward visibility & monitoring framework"          (397)
#     "section 1) unified financial intelligence & guidance..." (389)
#     "section 2) executive summaries & commentary section"     (382)
# Counting those made 45% of every concall look like an echo. Excluding heading-shaped
# lines leaves only real instruction prose, so the threshold means what it says.
_PROMPT_HEADING_RE = re.compile(
    r"^(?:output\s+)?section\b"          # "Section 1) ...", "OUTPUT SECTION C ..."
    r"|^[a-z]-?\d*[\)\.]\s"              # "e-1) ...", "a-2) ...", "c) ..."
    r"|^table_[a-z0-9]", re.I)


def _prompt_lines() -> list:
    """Distinctive instruction lines from every prompt file next to this module."""
    global _prompt_lines_cache
    if _prompt_lines_cache is not None:
        return _prompt_lines_cache
    import glob
    out: list = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        for path in glob.glob(os.path.join(here, "*prompt*.txt")):
            try:
                txt = open(path, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for ln in txt.splitlines():
                s = " ".join(ln.split())
                # skip table rows and rule bars - those DO appear in real reports
                if (len(s) >= _PROMPT_LINE_MIN and not s.startswith("|")
                        and not set(s) <= set("=-_* ")
                        and not _PROMPT_HEADING_RE.match(s)):
                    out.append(s.lower())
    except Exception:
        pass
    _prompt_lines_cache = out
    return out


def echoes_prompt_verbatim(text: str) -> int:
    """How many distinct prompt instruction lines this response reproduces word for word."""
    t = " ".join(str(text or "").split()).lower()
    if not t:
        return 0
    return sum(1 for ln in _prompt_lines() if ln in t)


def is_prompt_echo(text: str) -> bool:
    """True when a model response carries its own prompt rather than only an answer.

    Two independent tests, either of which condemns it: the phrase markers above, and
    verbatim reproduction of lines from the prompt files themselves. The second catches
    a PARTIAL echo — instructions followed by a genuine answer — which the first cannot,
    because such a response also contains real content.
    """
    t = str(text or "").lower()
    if sum(1 for k in _PROMPT_ECHO_MARKERS if k in t) >= 2:
        return True
    return echoes_prompt_verbatim(text) >= _PROMPT_LINE_HITS


# A run this long is never typesetting. MEASURED on the eight stored annual-report
# narratives, 2026-09-04 - the longest space run in each:
#     SHADOWFAX 0 · SENORES 5 · RISHABH 18 || RATEGAIN 4,712 · INDOBORAX 9,393 ·
#     GOLDIAM 64,883 · WELSPUNLIV 67,934 · RATEGAIN 76,272
# The gap between 18 and 4,712 is empty, so 40 separates real markdown-table alignment
# from a degenerate generation with a wide margin on both sides.
PAD_RUN_MIN = 40


def squeeze_padding(text: str) -> str:
    """Collapse the runaway space runs these models emit, leaving tables intact.

    THE BUG THIS FIXES. A model that loses its stop condition mid-report emits tens of
    thousands of consecutive spaces and then stops. Nothing noticed, because every check
    downstream measured len(): the MIN_REPORT_CHARS gate scored RATEGAIN's report as
    80,039 chars and passed it, when it held 3,528 chars of report followed by 76,272
    spaces. Worse, DOC_REPORT_MAX_CHARS then truncated INSIDE the padding, so whatever
    the model wrote after the run was thrown away.

    Runs shorter than PAD_RUN_MIN are left exactly as they are, so markdown table
    alignment survives. Four or more blank lines collapse to two, which markdown treats
    identically.
    """
    s = str(text or "")
    s = re.sub(r"[ \t]{%d,}" % PAD_RUN_MIN, " ", s)
    return re.sub(r"\n{4,}", "\n\n", s)


# A REPEAT LOOP is the other way one of these generations dies, and it is the one that
# reached the inbox. Navin Fluorine's mailed annual report was 66 KB of:
#       | FY2025 | 64 | N/A |
#       | FY2026 | 64 | N/A |
# repeated 99 times - seven pages of "NA" - and it sailed through every check, because
# a loop is LONG. Measured across the 45 rendered annual-report mails on 2026-09-04:
#       healthy  92-100% distinct rows, worst row repeated 1-2 times  (44 reports)
#       NAVINFLUOR         14% distinct, worst row repeated 99 times  ( 1 report)
# The gap is enormous, so the thresholds below sit far from both edges: a report is
# only condemned when it is BOTH mostly duplicates AND has one line repeated 8+ times.
# Models sometimes decorate markdown with literal HTML - SGFIN's report carried
#     <p style="font-size:0.8em; color:#808080;"><i>Source: ... (Page 56)</i></p>
#     <span style="color:purple;">Structural Shift in Operations</span>
# The mail renderer escapes anything it did not build itself, so the reader sees the tag
# soup verbatim instead of the sentence - and the PDF carries it too. Measured on the 43
# rendered annual reports, 2026-09-05: 1 report affected, and it was the one drawn at
# random to show the user. Keep the TEXT, drop the tags.
_INLINE_HTML_RE = re.compile(
    r"</?(?:p|span|div|i|b|em|strong|u|small|font|br|sub|sup)\b[^>]*>", re.I)


# A MARKDOWN TABLE THE MODEL WROTE ON ONE LINE IS NOT A TABLE TO ANY RENDERER.
# Measured on TD Power Systems' v2 report, 2026-09-05 — the whole revenue table arrived
# as a single line:
#     | Geography | FY 2026 | FY 2025 | | :--- | :--- | :--- | | Domestic | 73,438 | ...
# so the mail rendered a wall of pipes instead of rows. The giveaway is a separator row
# (:--- / ---) sitting INSIDE a line that also holds data cells; a real markdown table
# always puts it on its own line. Row boundaries in the flattened form are "| |".
_SEP_INLINE_RE = re.compile(r"\|\s*:?-{2,}:?\s*\|")
_ROW_BREAK_RE = re.compile(r"\|\s+\|")


def unflatten_tables(text: str) -> str:
    """Restore row breaks in a markdown table the model emitted on a single line."""
    out = []
    for ln in str(text or "").splitlines():
        if ln.count("|") >= 6 and _SEP_INLINE_RE.search(ln):
            # A LABEL IN FRONT OF THE TABLE KEEPS IT FROM BEING ONE. md_to_html
            # only treats a line as a table row when it STARTS with "|", so
            # "Revenue by Geography: | Geography | ..." still rendered as prose.
            head, sep, rest = ln.partition("|")
            if head.strip():
                out.append(head.rstrip())
            out.append(_ROW_BREAK_RE.sub("|\n|", sep + rest))
        else:
            out.append(ln)
    return "\n".join(out)


# THE RUPEE SIGN ARRIVES AS A BACKTICK. Many Indian annual reports embed the rupee glyph
# in a font whose extraction maps it to U+0060, so the report reads "` 2,400 Crore".
# Only a backtick immediately in front of a number is converted — code spans and any
# other backtick use are untouched.
_RUPEE_BACKTICK_RE = re.compile(r"`\s?(?=\d)")


def fix_rupee_glyph(text: str) -> str:
    return _RUPEE_BACKTICK_RE.sub("\u20b9", str(text or ""))


def strip_inline_html(text: str) -> str:
    """Remove decorative HTML tags a model emitted inside markdown, keeping the words."""
    s = _INLINE_HTML_RE.sub("", str(text or ""))
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#x27;", "'"), ("&#39;", "'")):
        s = s.replace(a, b)
    # a second pass: unescaping may have revealed tags that were double-encoded
    return _INLINE_HTML_RE.sub("", s)


REPEAT_MIN_LINES = 20      # too few lines to judge; the length gate covers those
REPEAT_MAX_RUN = 8         # one line repeated this often is not a report
REPEAT_MIN_DISTINCT = 0.70  # healthy floor is 0.92; this is 22 points of headroom


def degenerate_reason(text: str) -> str:
    """Why this response is a failed generation rather than a short report, or "".

    Length cannot answer this. A padded response and a looping response are both long,
    and both are worthless - so they are judged on SHAPE, not size.
    """
    from collections import Counter
    lines = [ln.strip() for ln in squeeze_padding(text).splitlines()
             if len(ln.strip()) > 8]
    if len(lines) < REPEAT_MIN_LINES:
        return ""
    counts = Counter(lines)
    line, run = counts.most_common(1)[0]
    distinct = len(counts) / len(lines)
    if run >= REPEAT_MAX_RUN and distinct < REPEAT_MIN_DISTINCT:
        return (f"repetition loop: {distinct:.0%} distinct lines, one repeated "
                f"{run}x ({line[:40]!r})")
    return ""


# ADDITIVE (2026-09-03). A narrative keyed by the DOCUMENT that produced it.
#
# WHY THIS EXISTS. Until now the only durable copy of an extraction's prose was the
# section it appended to company_repo/<isin>/company_page.md, and that page is an
# APPEND LOG: a re-extraction adds a second section for the same document rather than
# replacing the first, and the supersede path drops the <!-- doc:... --> marker, so the
# STALE copy keeps the only exact key. Measured on APL Apollo 2026-09-03: 21 annual
# report sections, two of them for the same FY2026 document - one 764 chars holding the
# marker, one 4,474 chars without it. Any reader then has to GUESS which is current, and
# two readers guessing differently is exactly what produced a mail with the wrong report.
#
# Rather than change how that page is written - 25+ scripts parse it, including
# ar_scorecard, daily_ar_summary, company_deep_report, ask_company and the Obsidian
# fetchers - this INTRODUCES a second, exact record. company_page.md is untouched and
# every existing reader keeps working. Readers that want "the report for document X"
# can now ask for it by id instead of parsing a page.
DOC_REPORT_COLS = ["source_doc_id", "isin", "symbol", "company_name", "doc_type",
                   "period", "report_md", "chars", "processed_at"]
DOC_REPORT_FILE = "doc_reports.parquet"
DOC_REPORT_MAX_CHARS = 80000       # a runaway lite-model response must not bloat the table


def save_doc_report(drive, index_id: str, row: dict, report_md: str) -> None:
    """Record one extraction's narrative against its doc_id. Never raises.

    upsert_structured deletes any existing rows for this source_doc_id before appending,
    so a re-extraction REPLACES its own record - one row per document, always current.
    Failure here must never fail an extraction that has already succeeded, so everything
    is caught: the page write remains the system of record.
    """
    try:
        did = str(row.get("doc_id") or "").strip()
        # Squeeze BEFORE the cap: truncating inside a padding run discards real report.
        md = squeeze_padding(report_md)
        if not did or not md.strip():
            return
        if len(md) > DOC_REPORT_MAX_CHARS:
            md = md[:DOC_REPORT_MAX_CHARS] + "\n\n_[truncated at DOC_REPORT_MAX_CHARS]_"
        upsert_structured(drive, index_id, DOC_REPORT_FILE, DOC_REPORT_COLS, [{
            "source_doc_id": did,
            "isin": str(row.get("isin") or ""),
            "symbol": str(row.get("symbol") or ""),
            "company_name": str(row.get("company_name") or ""),
            "doc_type": str(row.get("doc_type") or ""),
            "period": str(row.get("period") or ""),
            "report_md": md,
            "chars": len(md),
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }])
    except Exception as e:
        log(f"  NOTE: doc_reports write skipped ({str(e)[:70]})")


def load_doc_report(drive, index_id: str, doc_id: str) -> str:
    """The stored narrative for one document, or "" when there is none."""
    try:
        did = str(doc_id or "").strip()
        if not did:
            return ""
        df = load_parquet(drive, index_id, DOC_REPORT_FILE, DOC_REPORT_COLS)
        if df is None or df.empty:
            return ""
        hit = df[df["source_doc_id"].astype(str).str.strip() == did]
        return str(hit.iloc[-1]["report_md"]) if not hit.empty else ""
    except Exception:
        return ""


def mark_queue_error(queue, idx, reason: str, status: str = "error") -> None:
    """Record a failure ON the queue row: status, reason, attempt count, timestamp.

    In-place on the caller's frame; the caller still owns save_queue and the lock.
    Truncates the reason — the point is triage ("HTML not PDF", "no drive_file_id"),
    not a stack trace.
    """
    try:
        prev = queue.loc[idx, "attempts"]
        n = int(prev) if str(prev).strip() not in ("", "None", "nan", "<NA>") else 0
    except Exception:
        n = 0
    queue.loc[idx, "status"] = status
    queue.loc[idx, "attempts"] = n + 1
    queue.loc[idx, "last_error"] = str(reason)[:300]
    queue.loc[idx, "last_attempt_at"] = datetime.now().isoformat(timespec="seconds")


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
    """Delete existing rows for EVERY source_doc_id present in `rows`, then append the new
    rows (idempotent re-extract). `rows` may span ONE doc (per-doc call) or MANY docs (one
    batched write covering several source_doc_ids) — both dedupe correctly with a single
    load+save, which is what the batch-write path uses to cut Drive round-trips."""
    if not rows:
        return
    df = load_parquet(drive, index_id, filename, cols)
    sdids = {str(r.get("source_doc_id", "")) for r in rows if str(r.get("source_doc_id", ""))}
    if sdids and "source_doc_id" in df.columns:
        df = df[~df["source_doc_id"].astype(str).isin(sdids)]
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
# STATIC FALLBACK ONLY. model_registry.CHAINS["P1"] is the declared source of truth and
# resolve() filters it by what a daily probe found alive; these literals are what the
# system falls back to when the registry is missing or stale, so they must stay valid.
# gemini-2.0-flash was retired by Google and 404s - it sat LAST here, so it only failed
# once the two ahead of it were overloaded, i.e. exactly on the busy days.
P1_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.5-flash-lite"]

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
        from bucket_health import dead_buckets_since_reset, dead_models_keys_since_reset
        target = getattr(pool, "_pool", pool)   # unwrap GeminiKeyPool if needed
        df = load_parquet(drive, index_id, "gemini_usage.parquet", GEMINI_USAGE_COLS)
        now = datetime.utcnow()
        dead = set(dead_buckets_since_reset(df, now))          # PerDay-dead (key, model)
        # Also skip whole MODELS / KEYS that only failed this window (>=10 fails, 0 ok) —
        # e.g. gemini-2.0-flash flooding 429s. Expand to this pool's (key, model) buckets.
        dead_models, dead_keys = dead_models_keys_since_reset(df, now)
        if dead_models or dead_keys:
            for b in getattr(target, "buckets", []):
                if b.model in dead_models or b.key_idx in dead_keys:
                    dead.add((b.key_idx, b.model))
        if dead:
            n = target.prime_dead_buckets(dead)
            log(f"  bucket-health: primed {n} dead bucket(s) since reset — PerDay + "
                f"{len(dead_models)} dead model(s) {sorted(dead_models)} + "
                f"{len(dead_keys)} dead key(s).")
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
                    company_name: str, quarter: str, content: str,
                    dedup_marker: str | None = None) -> None:
    """Append analysis to _daily/<doc_type>_DD_MMMYYYY.md (persisted forever).

    `dedup_marker` (e.g. doc_id) makes the append IDEMPOTENT: if the file already contains
    that marker, the call is a no-op. This lets the batch-write path mark the queue in
    batches without a CI kill re-appending duplicate sections on the next run."""
    daily_id = get_or_create_subfolder(drive, repo_id, "_daily")
    fname = day_filename(doc_type, announcement_date)
    mark = f"<!-- doc:{dedup_marker} -->" if dedup_marker else ""
    entry = f"\n\n---\n## {symbol} — {company_name} | {quarter}\n{mark}\n" + content
    fid = find_file(drive, daily_id, fname)
    if fid:
        existing = download_bytes(drive, fid).decode("utf-8", errors="replace")
        if dedup_marker and mark in existing:
            return                      # already written — idempotent no-op
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
                        doc_title: str, quarter: str,
                        dedup_marker: str | None = None) -> None:
    """Append a section to company_repo/<key>/company_page.md (persisted forever).

    `dedup_marker` (e.g. doc_id) makes the append IDEMPOTENT: if the file already contains
    that marker, the call is a no-op. This is what lets the batch-write path defer the queue
    mark-done into batches safely — a CI kill mid-batch re-processes those docs, but the
    re-append is skipped (and the structured upserts dedupe), so no duplicate sections."""
    if not key:
        log("  WARN: empty key — skipping company_page.md update")
        return
    comp_id = get_or_create_subfolder(drive, repo_id, key)
    mark = f"<!-- doc:{dedup_marker} -->" if dedup_marker else ""
    header = (
        f"\n\n---\n## {quarter} {doc_type_label} — {doc_title}\n"
        f"*Processed: {datetime.now().strftime('%Y-%m-%d')}*\n{mark}\n\n"
    )
    fid = find_file(drive, comp_id, "company_page.md")
    if fid:
        existing = download_bytes(drive, fid).decode("utf-8", errors="replace")
        if dedup_marker and mark in existing:
            return                      # already written — idempotent no-op
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
        # UTC — see the note on acquire_lock. Writer and reader may sit in different
        # timezones (CI is UTC, a dev box is IST), and a naive local stamp makes this
        # beacon read 330 min old the moment a local process looks at it.
        payload = datetime.utcnow().isoformat(timespec="seconds").encode("utf-8")
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
        return (datetime.utcnow() - ts).total_seconds() / 60.0 < max_age_min
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
    nothing.

    TIMESTAMPS ARE UTC ON BOTH SIDES — do not "simplify" this back to datetime.now().
    The stamp is written by one process and aged by another, and those two need not
    share a timezone: CI runners are UTC, a dev box here is IST (+5:30). With naive
    local stamps, a local run reading a CI-written lock computes an age 330 minutes
    too large. Observed 2026-08-14: a live Phase-2 lock 30 minutes old was reported as
    360 minutes — exactly max_age_min — so the next local run of ANY lock-taking
    script would have declared it stale and stolen it mid-extraction, putting two
    writers on processing_queue.parquet and company_page.md. The beacon had the same
    fault in the opposite direction: a CI beacon looked permanently stale to a local
    reader, so backfill would never yield to Phase 2.

    In CI nothing changes — utcnow() == now() on a UTC runner. A lock written by the
    OLD code on a local box reads as negative age here, which errs toward never
    stealing: the safe direction."""
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
            age_min = ((datetime.utcnow() - ts).total_seconds() / 60.0) if ts else 1e9
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
    payload = f"{owner}|{datetime.utcnow().isoformat(timespec='seconds')}".encode("utf-8")
    upload_bytes(drive, index_id, lock_name, payload, "text/plain", existing_id=fid)
    return True


def release_lock(drive, index_id: str, lock_name: str) -> None:
    try:
        fid = find_file(drive, index_id, lock_name)
        if fid:
            drive.files().delete(fileId=fid).execute()
    except Exception as e:
        log(f"  LOCK {lock_name} release failed ({str(e)[:60]}) — auto-stolen later.")
