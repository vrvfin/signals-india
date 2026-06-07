r"""
daily_research_summary.py  —  signals-india / Workflow A (OT8)

LOCAL, manual trigger. Summarises the day's research PDFs and produces THREE outputs
per document. Source PDFs NEVER leave the machine; only the daily .md and the index
parquet sync to Drive.

  intake (\intake\<source>\*.pdf)
    -> dedup (sha256 + first-page/page-count fingerprint)        [research_ledger.parquet, LOCAL]
    -> classify doc_type (rule-based, tiny-Gemini fallback)
    -> route to mapped prompt (typed docs -> dedicated prompt; else research_doc_prompt)
    -> summarise on the DEDICATED 2-key pool (DAILY_GEMINI_KEY_1/2, separate projects)
    -> normalise tags vs tag_vocabulary.parquet + resolve company -> ISIN vs universe
    -> 1) entry in research_DD_MMMYYYY.md   (running counter research_N; human read)
       2) row in research_index.parquet      (machine feed for the OT7 deep dive)
       3) if company-mapped: append block to company_repo/<ISIN>/company_page.md
    -> upload ONLY the daily .md + research_index.parquet to Drive

Run:   python scripts/daily_research_summary.py
       python scripts/daily_research_summary.py --dry-run   (no Gemini, no upload; lists work)

Deps (local):  google-generativeai pandas pyarrow PyMuPDF pdfplumber
               google-api-python-client google-auth google-auth-oauthlib
Env:   DAILY_GEMINI_KEY_1, DAILY_GEMINI_KEY_2, GDRIVE_FOLDER_ID,
       GDRIVE_OAUTH_TOKEN_JSON  (same token the rest of the system uses)

INTEGRATION NOTE: the Drive helpers below are self-contained so this runs standalone.
If you prefer, replace drive_service()/drive_* with the equivalents in _extractor_base.py
— the rest of the module is agnostic to which Drive layer it calls.
"""
from __future__ import annotations
import os, sys, io, re, json, time, hashlib, argparse, datetime as dt, shutil
from pathlib import Path

import pandas as pd

# load project .env (same pattern as all pipeline scripts) so env vars are available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# Phase-2 bucket engine: (key, model) daily-quota buckets with error-typed fallback.
# Guarded: gemini_pool needs google-genai. This module is imported by
# company_deep_report for its Drive helpers (drive_service/_folder_id), which the
# Streamlit queue path uses — and Streamlit Cloud has no google-genai. Don't let the
# pool import break those consumers; only build_daily_pool() actually needs it.
try:
    from gemini_pool import BucketPool, AllBucketsExhausted, FatalCallError
except Exception:  # google-genai not installed here
    BucketPool = None

    class AllBucketsExhausted(Exception):
        pass

    class FatalCallError(Exception):
        pass

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
INTAKE_DIR   = Path(os.getenv("RESEARCH_INTAKE_DIR", r"D:\EMA_Screener\research_intake"))
SCRIPTS_DIR  = Path(__file__).resolve().parent
LEDGER_PATH   = INTAKE_DIR / "_ledger" / "research_ledger.parquet"   # LOCAL only
MENTIONS_PATH    = INTAKE_DIR / "_ledger" / "company_mentions.parquet"  # LOCAL only
NEW_ISINS_PATH   = INTAKE_DIR / "_ledger" / "new_isins_latest.json"     # ISINs with new docs this run
LOCAL_INDEX_PATH = INTAKE_DIR / "_ledger" / "research_index.parquet"    # LOCAL copy WITH summary_md

# Quality-first model chain. On the DEDICATED daily keys (separate Cloud projects),
# each (key, model) is its own free-tier daily bucket. 2 keys × 5 models = 10 buckets.
# Best model first; the pool only downgrades when the current model is dead on ALL keys.
DAILY_MODELS = [
    "gemini-3.5-flash",        # best free-tier quality
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",   # lite fallback — quality degrades last
    "gemini-3.1-flash-lite",
]
INTER_CALL_SLEEP = 6.0            # seconds between successful calls (RPM hygiene)
MAX_CHARS_CALL   = 400_000        # ~100k tokens; chunk above this (map-reduce)
MIN_TEXT_CHARS   = 200            # below this -> scanned/empty -> NEEDS_OCR, skip

# Drive layout (reuse existing repo)
DRIVE = dict(
    universe_csv       = "universe/master_list.csv",
    vocab_parquet      = "company_repo/_index/tag_vocabulary.parquet",
    index_parquet      = "company_repo/_index/research_index.parquet",
    daily_dir          = "company_repo/_daily",
    company_page_dir   = "company_repo",   # /<ISIN>/company_page.md
    mentions_parquet   = "company_repo/_index/company_mentions.parquet",
)

# doc_type -> prompt file. Typed docs reuse the live extractors; rest -> generic.
PROMPT_FILE = {
    "single_company_ar":     "annual_report_prompt.txt",
    "concall":               "concall_prompt.txt",
    "results":               "results_prompt.txt",
    "single_company_rating": "rating_prompt.txt",
    "presentation":          "presentation_prompt.txt",
    "_generic":              "research_doc_prompt.txt",
}
GENERIC_TYPES = {  # these always go through research_doc_prompt
    "single_company_note", "single_company_drhp", "single_company_policy",
    "multi_company_seminar", "multi_company_sector", "govt_policy",
    "macro_report", "other",
}

# Rule-based classifier keywords (checked against subfolder + filename + first page)
CLASSIFY_RULES = [
    ("concall",               ["concall", "earnings call", "conference call", "transcript"]),
    ("single_company_ar",     ["annual report", "integrated report", "_ar_", "annualreport"]),
    ("results",               ["financial results", "results filing", "reg 33", "unaudited results"]),
    ("single_company_rating", ["rating rationale", "credit rating", "crisil", "icra", "care ratings", "india ratings"]),
    ("presentation",          ["investor presentation", "earnings presentation", "_ppt", "investor ppt"]),
    ("single_company_drhp",   ["draft red herring", "drhp", "red herring prospectus"]),
    ("multi_company_seminar", ["seminar", "conference notes", "investor day", "analyst meet"]),
    ("multi_company_sector",  ["sector report", "thematic", "industry report"]),
    ("govt_policy",           ["ministry of", "gazette", "union budget", "policy notification", "circular"]),
    ("macro_report",          ["rbi", "monetary policy", "macro", "gdp", "cpi", "global markets"]),
]

# ----------------------------------------------------------------------------
# GEMINI — dedicated DAILY keys × model chain via the shared BucketPool engine
# (Phase 2 gemini_pool.py). Separate Cloud projects keep this quota independent
# of Phase 2. PerDay 429 -> bucket dead-today, instant rotate; AllBucketsExhausted
# -> caller stops and resumes next run (work already persisted per-file).
# ----------------------------------------------------------------------------
def daily_keys() -> list[str]:
    keys = [os.getenv("DAILY_GEMINI_KEY_1"), os.getenv("DAILY_GEMINI_KEY_2")]
    keys = [k.strip() for k in keys if k and k.strip()]
    if not keys:
        raise RuntimeError("Set DAILY_GEMINI_KEY_1 / DAILY_GEMINI_KEY_2")
    return keys

def build_daily_pool() -> BucketPool:
    """BucketPool over the dedicated daily keys × DAILY_MODELS (best model first)."""
    if BucketPool is None:
        raise RuntimeError("google-genai not installed — cannot build the Gemini pool here.")
    return BucketPool(daily_keys(), DAILY_MODELS, inter_call_s=INTER_CALL_SLEEP)

# ----------------------------------------------------------------------------
# DRIVE — minimal helper (swap for _extractor_base if preferred)
# ----------------------------------------------------------------------------
def drive_service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    # CI: inline JSON in env var; local: file path (matches _extractor_base.py pattern)
    if os.environ.get("GDRIVE_OAUTH_TOKEN_JSON"):
        creds = Credentials.from_authorized_user_info(
            json.loads(os.environ["GDRIVE_OAUTH_TOKEN_JSON"]))
    else:
        token_path = os.environ.get("GDRIVE_OAUTH_TOKEN_PATH", "")
        if not token_path:
            raise RuntimeError("Set GDRIVE_OAUTH_TOKEN_JSON or GDRIVE_OAUTH_TOKEN_PATH")
        creds = Credentials.from_authorized_user_file(token_path)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _folder_id(svc, path: str, root: str, create=False) -> str | None:
    parent = root
    for part in path.strip("/").split("/"):
        q = (f"name='{part}' and '{parent}' in parents and "
             "mimeType='application/vnd.google-apps.folder' and trashed=false")
        res = svc.files().list(q=q, fields="files(id)", pageSize=1).execute().get("files", [])
        if res:
            parent = res[0]["id"]
        elif create:
            meta = {"name": part, "parents": [parent],
                    "mimeType": "application/vnd.google-apps.folder"}
            parent = svc.files().create(body=meta, fields="id").execute()["id"]
        else:
            return None
    return parent

def drive_find(svc, full_path: str, root: str) -> str | None:
    *dirs, name = full_path.strip("/").split("/")
    fid = _folder_id(svc, "/".join(dirs), root) if dirs else root
    if fid is None:
        return None
    q = f"name='{name}' and '{fid}' in parents and trashed=false"
    res = svc.files().list(q=q, fields="files(id)", pageSize=1).execute().get("files", [])
    return res[0]["id"] if res else None

def drive_download(svc, full_path: str, root: str) -> bytes | None:
    fid = drive_find(svc, full_path, root)
    if not fid:
        return None
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO(); dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=fid))
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

def drive_upload(svc, full_path: str, root: str, data: bytes, mime: str):
    from googleapiclient.http import MediaIoBaseUpload
    *dirs, name = full_path.strip("/").split("/")
    fid_dir = _folder_id(svc, "/".join(dirs), root, create=True) if dirs else root
    existing = drive_find(svc, full_path, root)
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)
    if existing:
        svc.files().update(fileId=existing, media_body=media).execute()
    else:
        svc.files().create(body={"name": name, "parents": [fid_dir]},
                           media_body=media, fields="id").execute()

def check_drive_service(svc) -> bool:
    """Validate that the Drive service is accessible via a cheap list() call."""
    try:
        svc.files().list(q="trashed=false", fields="files(id)", pageSize=1).execute()
        return True
    except Exception as e:
        print(f"   Drive service check failed: {e}")
        return False

def drive_upload_with_retry(svc, full_path: str, root: str, data: bytes, mime: str,
                            max_retries: int = 3) -> None:
    """Upload with exponential backoff retry for transient 5xx errors.
    Non-5xx errors (auth, quota, etc.) fail immediately."""
    from googleapiclient.errors import HttpError
    backoff_secs = [1, 2, 4]
    for attempt in range(max_retries):
        try:
            drive_upload(svc, full_path, root, data, mime)
            return
        except HttpError as e:
            status = e.resp.status if hasattr(e, 'resp') else None
            is_transient = status and 500 <= status < 600
            if not is_transient or attempt >= max_retries - 1:
                print(f"   Upload failed ({full_path}, {len(data)} bytes): {e}")
                raise
            print(f"   Transient error (HTTP {status}), retrying in {backoff_secs[attempt]}s...")
            time.sleep(backoff_secs[attempt])
        except Exception as e:
            print(f"   Upload failed ({full_path}, {len(data)} bytes): {e}")
            raise

# ----------------------------------------------------------------------------
# PDF text extraction (fitz -> pdfplumber -> pypdf)
# ----------------------------------------------------------------------------
def extract_pdf(path: Path):
    try:
        import fitz
        doc = fitz.open(path)
        pages = [p.get_text() for p in doc]
        return "\n".join(pages), len(pages)
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages = [(pg.extract_text() or "") for pg in pdf.pages]
        return "\n".join(pages), len(pages)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        pages = [(pg.extract_text() or "") for pg in r.pages]
        return "\n".join(pages), len(pages)
    except Exception as e:
        print(f"   PDF extract failed: {e}")
        return "", 0

# ----------------------------------------------------------------------------
# Dedup keys
# ----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

def fuzzy_fingerprint(text: str, n_pages: int) -> str:
    head = re.sub(r"\s+", " ", text[:2000]).strip().lower()
    return hashlib.sha256(f"{head}|{n_pages}".encode()).hexdigest()

def content_key(companies, doc_date, n_pages) -> str:
    names = "|".join(sorted(c.get("name", "").lower() for c in companies)) or "na"
    return hashlib.sha256(f"{names}|{doc_date}|{n_pages}".encode()).hexdigest()

# ----------------------------------------------------------------------------
# Classify + summarise
# ----------------------------------------------------------------------------
def classify(source: str, filename: str, first_page: str) -> str:
    """Rule-based only (no Gemini call) — saves a daily bucket per doc.
    Unmatched docs route to 'other' -> the generic research_doc_prompt."""
    hay = f"{source} {filename} {first_page[:1500]}".lower()
    for doc_type, kws in CLASSIFY_RULES:
        if any(k in hay for k in kws):
            return doc_type
    return "other"

def load_prompt(doc_type: str, vocab_block: str) -> str:
    fname = PROMPT_FILE.get(doc_type if doc_type not in GENERIC_TYPES else "_generic",
                            PROMPT_FILE["_generic"])
    text = (SCRIPTS_DIR / fname).read_text(encoding="utf-8")
    if "[CONTROLLED_VOCABULARY]" in text:
        text = text.replace("[CONTROLLED_VOCABULARY]", vocab_block)
    return text

def summarise(doc_type: str, body: str, vocab_block: str, pool: BucketPool) -> str:
    """One pool.call_text per chunk. Raises AllBucketsExhausted / FatalCallError
    (handled by the caller). call_text returns (text, model_used)."""
    prompt = load_prompt(doc_type, vocab_block)
    if len(body) <= MAX_CHARS_CALL:
        return pool.call_text(prompt + "\n\n===== DOCUMENT =====\n" + body)[0]
    # map-reduce for large docs
    chunks = [body[i:i + MAX_CHARS_CALL] for i in range(0, len(body), MAX_CHARS_CALL)]
    partials = []
    for j, ch in enumerate(chunks, 1):
        partials.append(pool.call_text(
            f"Extract the key facts, figures and statements from PART {j}/{len(chunks)} "
            f"of a document. Be exhaustive and faithful; do not interpret.\n\n{ch}")[0])
    merged = "\n\n".join(partials)
    return pool.call_text(prompt + "\n\n===== DOCUMENT (assembled extracts) =====\n" + merged)[0]

# ----------------------------------------------------------------------------
# Tag extraction + normalisation
# ----------------------------------------------------------------------------
def parse_json_tail(md: str) -> dict | None:
    m = list(re.finditer(r"```json\s*(\{.*?\})\s*```", md, re.DOTALL))
    if not m:
        return None
    try:
        return json.loads(m[-1].group(1))
    except Exception:
        return None

def build_vocab_block(vocab: pd.DataFrame) -> str:
    def slugs(t): return ", ".join(vocab[vocab.tag_type == t].tag_slug.tolist())
    return ("SECTORS: " + slugs("sector") + "\nSUBSECTORS: " + slugs("subsector") +
            "\nDOC_TYPES: " + slugs("doc_type") + "\nTHEMES: " + slugs("theme"))

def alias_map(vocab: pd.DataFrame) -> dict:
    m = {}
    for _, r in vocab.iterrows():
        if r.status != "closed":
            continue
        m[r.tag_slug] = r.tag_slug
        for a in str(r.aliases or "").split("|"):
            if a.strip():
                m[re.sub(r"\s+", " ", a.strip().lower())] = r.tag_slug
    return m

def normalise_list(vals, amap, closed_set, fallback):
    out = []
    for v in (vals or []):
        k = re.sub(r"\s+", " ", str(v).strip().lower())
        slug = amap.get(k, k if k in closed_set else fallback)
        if slug and slug not in out:
            out.append(slug)
    return out

def resolve_isins(companies, universe: pd.DataFrame) -> list:
    if universe is None or universe.empty:
        return []
    name_col = next((c for c in universe.columns if c.lower() in ("name", "company", "company_name")), None)
    isin_col = next((c for c in universe.columns if c.lower() == "isin"), None)
    sym_col  = next((c for c in universe.columns if c.lower() in ("symbol", "nse_symbol")), None)
    isins = []
    for c in (companies or []):
        tok = str(c.get("ticker_or_isin", "")).strip().upper()
        if isin_col and tok.startswith("INE") and tok in set(universe[isin_col].astype(str)):
            isins.append(tok); continue
        if sym_col and tok and tok in set(universe[sym_col].astype(str).str.upper()):
            row = universe[universe[sym_col].astype(str).str.upper() == tok]
            if isin_col and not row.empty:
                isins.append(str(row.iloc[0][isin_col])); continue
        if name_col:
            nm = str(c.get("name", "")).strip().lower()
            if nm:
                hit = universe[universe[name_col].astype(str).str.lower() == nm]
                if isin_col and not hit.empty:
                    isins.append(str(hit.iloc[0][isin_col]))
    return [i for i in dict.fromkeys(isins) if i and i != "nan"]

def company_slug(name: str) -> str:
    """Normalise a company name for cross-document search (strip legal suffixes)."""
    s = name.lower().strip()
    for sfx in (" limited", " ltd.", " ltd", " private limited", " pvt. ltd.", " pvt. ltd",
                " pvt ltd", " private", " pvt.", " pvt", " incorporated", " inc.",
                " inc", " corporation", " corp", " llp", " llc"):
        if s.endswith(sfx):
            s = s[:-len(sfx)].strip()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------
def daily_md_path() -> str:
    return f"{DRIVE['daily_dir']}/research_{dt.datetime.now():%d_%b%Y}.md"

def render_entry(n, tags, meta, summary_md, run_hhmm) -> str:
    tagline = " ".join(f"#{t}" for t in tags) or "#untagged"
    comp = ", ".join(f"{c.get('name','?')}" for c in meta.get("companies", [])) or "NA"
    body = re.sub(r"```json.*?```", "", summary_md, flags=re.DOTALL).strip()
    return (f"\n## research_{n:04d}  *(Run {run_hhmm} IST)*\n"
            f"**Source:** {meta.get('source','NA')} · **Date:** {meta.get('doc_date','NA')} · "
            f"**Type:** {meta.get('doc_type','other')} · **Companies:** {comp}\n"
            f"**Tags:** {tagline}\n\n{body}\n\n---\n")

def append_company_page(svc, isin, n, meta, summary_md, root):
    path = f"{DRIVE['company_page_dir']}/{isin}/company_page.md"
    cur = drive_download(svc, path, root)
    cur = cur.decode("utf-8") if cur else f"# Company Page — {isin}\n"
    if "## Research Notes" not in cur:
        cur += "\n## Research Notes\n"
    block = (f"\n### research_{n:04d} · {meta.get('doc_date','NA')} · {meta.get('source','NA')} "
             f"· {meta.get('doc_type','other')}\n"
             + re.sub(r"```json.*?```", "", summary_md, flags=re.DOTALL).strip() + "\n")
    drive_upload_with_retry(svc, path, root, (cur + block).encode("utf-8"), "text/markdown")

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def _move_to(subdir: str, pdf: Path) -> None:
    """Move a PDF into <intake>/<subdir>/<source>/ (collision-safe)."""
    dest_dir = INTAKE_DIR / subdir / pdf.parent.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / pdf.name
    if dest.exists():                       # already a copy there — disambiguate
        dest = dest_dir / f"{pdf.stem}_{int(time.time())}{pdf.suffix}"
    try:
        shutil.move(str(pdf), str(dest))
        print(f"   moved -> {subdir}/{pdf.parent.name}/{dest.name}")
    except Exception as e:
        print(f"   WARN: could not move {pdf.name}: {e}")

def _move_processed(pdf: Path) -> None:
    """A finished, summarised PDF -> _processed/ (kept; never reprocessed)."""
    _move_to("_processed", pdf)

def _move_duplicate(pdf: Path) -> None:
    """A duplicate (already-seen / near-dup) -> _duplicates/ (safe to delete)."""
    _move_to("_duplicates", pdf)


def _daily_header(run_hhmm: str) -> str:
    return (f"# Daily Research Digest — {dt.datetime.now():%d %b %Y}\n\n"
            f"*Started {run_hhmm} IST · appended per document*\n\n---\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no Gemini, no Drive; classify + list only")
    ap.add_argument("--limit", type=int, default=0, help="process at most N new docs then stop (0 = all)")
    args = ap.parse_args()

    if not INTAKE_DIR.exists():
        sys.exit(f"Intake dir not found: {INTAKE_DIR}")
    _SKIP_DIRS = {"_processed", "_duplicates", "_ledger"}
    pdfs = sorted(p for p in INTAKE_DIR.rglob("*.pdf")
                  if _SKIP_DIRS.isdisjoint(p.parts))
    if not pdfs:
        print("No PDFs in intake."); return

    # ledger (local) — dedup + resume state (persists across ALL runs, forever)
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame(
        columns=["research_n", "doc_hash", "fuzzy_fp", "content_key", "file_name",
                 "source", "doc_type", "processed_at", "status"])
    seen_hash = set(ledger.doc_hash); seen_fp = set(ledger.fuzzy_fp)
    seen_ck = set(ledger.get("content_key", pd.Series(dtype=str)))
    prior_status = dict(zip(ledger.doc_hash, ledger.status)) if len(ledger) else {}
    counter = int(ledger.research_n.max()) if len(ledger) else 0

    def add_ledger(**row):
        ledger.loc[len(ledger)] = row
        ledger.to_parquet(LEDGER_PATH, index=False)

    # ---- DRY RUN: classify + list only, no Gemini, no Drive ----
    if args.dry_run:
        for pdf in pdfs:
            if sha256_file(pdf) in seen_hash:
                continue
            text, npages = extract_pdf(pdf)
            if len(text) < MIN_TEXT_CHARS:
                print(f"   SKIP (needs OCR / empty): {pdf.name}"); continue
            dtp = classify(pdf.parent.name, pdf.name, text[:3000])
            print(f" {pdf.name}  ->  {dtp}  ({npages}p, src={pdf.parent.name})")
        print("Dry run complete (no Gemini, no Drive)."); return

    # ---- LIVE ----
    svc  = drive_service()
    pool = build_daily_pool()
    root = os.environ.get("GDRIVE_FOLDER_ID", "")
    nbuckets = len(daily_keys()) * len(DAILY_MODELS)
    print(f"Pool: {len(daily_keys())} key(s) × {len(DAILY_MODELS)} model(s) = {nbuckets} daily buckets")

    if not check_drive_service(svc):
        sys.exit("Drive service is not accessible; check credentials and network.")

    # vocab + universe (from Drive)
    vocab = universe = None; vblock = ""
    vb = drive_download(svc, DRIVE["vocab_parquet"], root)
    if vb:
        vocab = pd.read_parquet(io.BytesIO(vb)); vblock = build_vocab_block(vocab)
    uv = drive_download(svc, DRIVE["universe_csv"], root)
    if uv:
        universe = pd.read_csv(io.BytesIO(uv))
    amap = alias_map(vocab) if vocab is not None else {}
    closed = {t: set(vocab[vocab.tag_type == t].tag_slug) for t in
              ("sector", "subsector", "theme", "doc_type")} if vocab is not None else {}

    run_hhmm = dt.datetime.now().strftime("%H:%M")

    # running state loaded ONCE; appended + persisted after EACH doc
    dpath = daily_md_path()
    dprev = drive_download(svc, dpath, root)
    daily_md = dprev.decode("utf-8") if dprev else _daily_header(run_hhmm)
    iraw = drive_download(svc, DRIVE["index_parquet"], root)
    index_df = pd.read_parquet(io.BytesIO(iraw)) if iraw else pd.DataFrame()
    mraw = drive_download(svc, DRIVE["mentions_parquet"], root)
    mentions_df = pd.read_parquet(io.BytesIO(mraw)) if mraw else pd.DataFrame()
    LOCAL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    new_isins_run: set[str] = set()
    counts = dict(processed=0, skipped=0, deferred=0, error=0, ocr=0, dup=0)

    for pdf in pdfs:
        if args.limit and counts["processed"] >= args.limit:
            print(f"Reached --limit {args.limit}; stopping."); break

        source = pdf.parent.name
        h = sha256_file(pdf)
        if h in seen_hash:
            if prior_status.get(h) == "needs_ocr":
                # still unreadable — leave in intake for future OCR, don't treat as dup
                counts["skipped"] += 1; continue
            print(f"   duplicate (already processed): {pdf.name}")
            _move_duplicate(pdf); counts["dup"] += 1; continue

        text, npages = extract_pdf(pdf)
        if len(text) < MIN_TEXT_CHARS:
            print(f"   SKIP (needs OCR / empty): {pdf.name}")
            add_ledger(research_n=0, doc_hash=h, fuzzy_fp="", content_key="",
                       file_name=pdf.name, source=source, doc_type="needs_ocr",
                       processed_at=dt.datetime.now().isoformat(), status="needs_ocr")
            seen_hash.add(h); prior_status[h] = "needs_ocr"; counts["ocr"] += 1; continue

        fp = fuzzy_fingerprint(text, npages)
        if fp in seen_fp:
            print(f"   duplicate (same content, different file): {pdf.name}")
            _move_duplicate(pdf); counts["dup"] += 1; continue

        doc_type = classify(source, pdf.name, text[:3000])
        print(f" [{counts['processed']+1}] {pdf.name}  ->  {doc_type}  ({npages}p, src={source})")

        # ---- Gemini summarise via bucket pool ----
        try:
            summary = summarise(doc_type, text, vblock, pool)
        except AllBucketsExhausted as exc:
            counts["deferred"] = sum(1 for p in pdfs if sha256_file(p) not in seen_hash)
            print(f"\nStopping: {exc}")
            print(f"{counts['deferred']} PDF(s) remain in intake — they resume on the next "
                  "run (buckets refill at the daily reset).")
            break
        except FatalCallError as exc:
            print(f"   FATAL (this doc): {str(exc)[:120]}")
            add_ledger(research_n=0, doc_hash=h, fuzzy_fp=fp, content_key="",
                       file_name=pdf.name, source=source, doc_type=doc_type,
                       processed_at=dt.datetime.now().isoformat(), status="error")
            seen_hash.add(h); counts["error"] += 1; continue

        meta = parse_json_tail(summary) or {}
        meta.setdefault("doc_type", doc_type)
        meta.setdefault("source", source)
        meta.setdefault("doc_date", "NA")
        meta.setdefault("companies", [])

        ck = content_key(meta.get("companies", []), meta.get("doc_date", "NA"), npages)
        if ck in seen_ck:
            print("   near-duplicate (content_key) -> recorded, not re-emitted")
            add_ledger(research_n=0, doc_hash=h, fuzzy_fp=fp, content_key=ck,
                       file_name=pdf.name, source=source, doc_type=doc_type,
                       processed_at=dt.datetime.now().isoformat(), status="dup")
            seen_hash.add(h); _move_duplicate(pdf); counts["dup"] += 1; continue

        sectors = normalise_list(meta.get("sectors"),  amap, closed.get("sector", set()),  "other_sector")
        subs    = normalise_list(meta.get("subsectors"), amap, closed.get("subsector", set()), "")
        themes  = normalise_list(meta.get("themes"),   amap, closed.get("theme", set()),   "")
        isins   = resolve_isins(meta.get("companies", []), universe)
        tags    = [meta["doc_type"]] + sectors + themes

        counter += 1
        md_ref = f"{dpath}#research_{counter:04d}"
        entry  = render_entry(counter, tags, meta, summary, run_hhmm)
        clean  = re.sub(r"```json.*?```", "", summary, flags=re.DOTALL).strip()
        index_row = dict(
            research_n=counter, doc_hash=h, fuzzy_fp=fp, content_key=ck,
            file_name=pdf.name, source=meta["source"], doc_date=meta["doc_date"],
            doc_type=meta["doc_type"], companies=json.dumps(meta.get("companies", [])),
            isins=json.dumps(isins), sectors=json.dumps(sectors), subsectors=json.dumps(subs),
            themes=json.dumps(themes), promoters=json.dumps(meta.get("promoters", [])),
            policies=json.dumps(meta.get("policies", [])),
            fy_or_quarter=meta.get("fy_or_quarter", "NA"),
            confidence=meta.get("confidence", "NA"),
            summary_md=clean, daily_md_ref=md_ref,
            processed_at=dt.datetime.now().isoformat())
        ment_rows = []
        for c in meta.get("companies", []):
            raw = c.get("name", "").strip()
            if not raw:
                continue
            tok = str(c.get("ticker_or_isin", "")).strip().upper()
            isin_for_c = tok if tok.startswith("INE") and tok in isins else (isins[0] if isins else "")
            ment_rows.append(dict(
                research_n=counter, file_name=pdf.name, source=meta["source"],
                doc_date=meta["doc_date"], doc_type=meta["doc_type"],
                company_name_raw=raw, company_name_slug=company_slug(raw),
                isin=isin_for_c, daily_md_ref=md_ref))

        # ---- PERSIST THIS DOC (append-each-time: Drive + local) ----
        try:
            daily_md += entry
            drive_upload_with_retry(svc, dpath, root, daily_md.encode("utf-8"), "text/markdown")

            index_df = pd.concat([index_df, pd.DataFrame([index_row])], ignore_index=True)
            ibuf = io.BytesIO(); index_df.to_parquet(ibuf, index=False)
            drive_upload_with_retry(svc, DRIVE["index_parquet"], root, ibuf.getvalue(), "application/octet-stream")
            index_df.to_parquet(LOCAL_INDEX_PATH, index=False)

            if ment_rows:
                mentions_df = pd.concat([mentions_df, pd.DataFrame(ment_rows)], ignore_index=True)
                mbuf = io.BytesIO(); mentions_df.to_parquet(mbuf, index=False)
                drive_upload_with_retry(svc, DRIVE["mentions_parquet"], root, mbuf.getvalue(), "application/octet-stream")
                mentions_df.to_parquet(MENTIONS_PATH, index=False)

            for isin in isins:
                append_company_page(svc, isin, counter, meta, summary, root)
        except Exception as e:
            print(f"   FATAL (upload): {pdf.name}: {str(e)[:120]}")
            add_ledger(research_n=counter, doc_hash=h, fuzzy_fp=fp, content_key=ck,
                       file_name=pdf.name, source=source, doc_type=meta["doc_type"],
                       processed_at=dt.datetime.now().isoformat(), status="upload_error")
            seen_hash.add(h); seen_fp.add(fp); seen_ck.add(ck)
            _move_processed(pdf)
            raise

        add_ledger(research_n=counter, doc_hash=h, fuzzy_fp=fp, content_key=ck,
                   file_name=pdf.name, source=source, doc_type=meta["doc_type"],
                   processed_at=dt.datetime.now().isoformat(), status="ok")

        new_isins_run.update(i for i in isins if i.startswith("INE"))
        NEW_ISINS_PATH.write_text(json.dumps(sorted(new_isins_run)), encoding="utf-8")

        _move_processed(pdf)
        seen_hash.add(h); seen_fp.add(fp); seen_ck.add(ck)
        counts["processed"] += 1
        print(f"   persisted research_{counter:04d}  ({len(isins)} ISIN, {len(ment_rows)} mention)")

    print("-" * 56)
    print(f"Processed : {counts['processed']}")
    print(f"Deferred  : {counts['deferred']}  (still in intake — quota/transient; resume next run)")
    print(f"Duplicates: {counts['dup']}   OCR-skip: {counts['ocr']}   Errors: {counts['error']}")
    print(f"Other skip: {counts['skipped']}  (already processed / fingerprint dup)")
    try:
        print("Buckets:", pool.summary())
    except Exception:
        pass


if __name__ == "__main__":
    main()
