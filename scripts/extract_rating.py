"""
Phase 2 / Stage D — Credit rating extraction via Gemini.

Consumes pending rating entries from the Drive processing queue, runs each PDF
through Gemini with rating_prompt.txt (solvency + credit forensic lens), appends
the full analysis to company_page.md and the daily digest, upserts extracted rating
metadata into _index/ratings.parquet, and marks queue rows done.

On Gemini 429 with all keys exhausted: stops cleanly (exit 0).

Usage:
    python scripts/extract_rating.py
    python scripts/extract_rating.py --limit 5
    python scripts/extract_rating.py --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _extractor_base import (
    RateLimitExhausted, GeminiKeyPool, get_drive, load_api_keys, P1_MODELS,
    log, get_or_create_subfolder,
    load_queue, save_queue,
    load_parquet, save_parquet,
    download_bytes,
    extract_md_tables, clean_val,
    append_company_page, append_day_page,
    load_portfolio_isins,
    acquire_lock, release_lock,
    salvage_json_objects, clamp, sstr, upsert_structured,   # Stage 3 tabulation
    call_over_doc,                                          # detect PDF vs HTML/text
    run_structured_over_doc,                                # Stage 3b: extract from source
)

# T12: SAME lock the concall extractor uses → one global mutex on the shared queue /
# company_page.md / parquets (Phase-2 live vs backfill can never write concurrently).
_LOCK_NAME = "_extract.lock"
_LOCK_MAX_AGE_MIN = 360

# ---- Config ----
DOC_TYPE        = "rating"
GEMINI_MODEL    = P1_MODELS          # lite chain, disjoint from concall (P0)
PROMPT_FILE     = "rating_prompt.txt"
DOC_TYPE_LABEL  = "Credit Rating"

OUTPUT_COMPANY_MD   = True
OUTPUT_DAY_MD       = True
OUTPUT_COMPANY_DOCX = False  # [Stage C]
OUTPUT_DAY_DOCX     = False  # [Stage C]

# ---- Parquet schema ----
RATINGS_COLS = [
    "isin", "symbol", "company_name",
    "agency", "rating", "outlook", "rating_action",
    "instrument_type", "rated_amount_cr",
    "rating_date", "processed_at", "source_doc_id",
]

# ---- Stage 3: structured tabulation (ADDITIVE — ratings.parquet unchanged) ----
STRUCT_PROMPT_FILE = "rating_structured_prompt.txt"
STRUCT_INPUT_CHARS = 60000
# Bound the lite model's report (it rambles non-linearly — measured ~2M chars even on a
# 4KB rating doc — which starves the structured pass). Mirrors AR. ENHANCEMENT only:
# output is still the same markdown report + ratings.parquet, just not a runaway blob.
RATING_MAX_OUTPUT_TOKENS = 4096
_RATING_BASE = ["isin", "symbol", "company_name", "agency", "rating_date",
                "processed_at", "source_doc_id"]
RATING_DRIVERS_COLS    = _RATING_BASE[:5] + ["driver", "evidence"] + _RATING_BASE[5:]
RATING_CONCERNS_COLS   = _RATING_BASE[:5] + ["concern", "severity", "evidence"] + _RATING_BASE[5:]
RATING_SENSITIVITY_COLS = _RATING_BASE[:5] + ["direction", "trigger"] + _RATING_BASE[5:]
_SEV = {"low", "medium", "high"}
_DIR = {"up", "down"}


def parse_rating_structured(text, row, agency, rating_date, now_str):
    """Salvage + dedupe → (drivers, concerns, sensitivity) rows."""
    objs = salvage_json_objects(text)
    if not objs:
        return [], [], []
    base = {"isin": str(row.get("isin") or ""), "symbol": str(row.get("symbol") or ""),
            "company_name": str(row.get("company_name") or ""), "agency": agency or "",
            "rating_date": rating_date or "", "processed_at": now_str,
            "source_doc_id": str(row.get("doc_id") or "")}
    dr, co, se = [], [], []
    sd, sc, ss = set(), set(), set()
    for o in objs:
        if "concern" in o:
            d = {**base, "concern": sstr(o.get("concern")),
                 "severity": clamp(o.get("severity"), _SEV, "medium"),
                 "evidence": sstr(o.get("evidence"))}
            k = (d["concern"] or "")[:120]
            if d["concern"] and k not in sc:
                sc.add(k); co.append(d)
        elif "driver" in o:
            d = {**base, "driver": sstr(o.get("driver")), "evidence": sstr(o.get("evidence"))}
            k = (d["driver"] or "")[:120]
            if d["driver"] and k not in sd:
                sd.add(k); dr.append(d)
        elif "trigger" in o or "direction" in o:
            d = {**base, "direction": clamp(o.get("direction"), _DIR, "up"),
                 "trigger": sstr(o.get("trigger"))}
            k = (d["direction"], (d["trigger"] or "")[:120])
            if d["trigger"] and k not in ss:
                ss.add(k); se.append(d)
    return dr[:10], co[:10], se[:8]


def tabulate_rating(gemini, struct_prompt, doc_bytes, row, agency, rating_date, now_str):
    """Stage 3b: run the structured JSON pass DIRECTLY over the source rating doc
    (small/clean) — not the lite model's rambling report — so drivers/concerns/
    sensitivity actually populate. Best-effort."""
    resp = run_structured_over_doc(gemini, struct_prompt, doc_bytes,
                                   name=f"{row.get('symbol', 'DOC')}_RATING_struct")
    return parse_rating_structured(resp, row, agency, rating_date, now_str)

# Common rating agency name variants
_AGENCY_PATTERNS = [
    (r"crisil",   "CRISIL"),
    (r"icra",     "ICRA"),
    (r"care",     "CARE"),
    (r"india ratings|ind-ra", "India Ratings"),
    (r"brickwork", "Brickwork"),
    (r"acuite",   "Acuite"),
]


# ------------------------------------------------------------------ #
#  Parser                                                              #
# ------------------------------------------------------------------ #

def _detect_agency(text: str) -> str:
    low = text.lower()
    for pattern, name in _AGENCY_PATTERNS:
        if re.search(pattern, low):
            return name
    return "DATA_MISSING"


# Canonical outlooks. Anything else in the outlook slot is NOT an outlook — most often
# it is the short-term rating from a combined "AA+(Stable)/A1+" symbol, which is how
# 107+ rows ended up with an "outlook" of A1+, A2+ or [ICRA]A1+.
_VALID_OUTLOOKS = {
    "stable": "Stable", "positive": "Positive", "negative": "Negative",
    "developing": "Developing", "watch": "Watch",
    "watch developing": "Watch Developing", "watch negative": "Watch Negative",
    "watch positive": "Watch Positive", "rating watch": "Watch",
    "cwn": "Watch Negative", "cwp": "Watch Positive", "rwn": "Watch Negative",
    "rwp": "Watch Positive",
}

# Agencies rename and abbreviate themselves; CARE alone appeared as CARE, "CARE Ratings"
# and CareEdge, splitting one agency's history into three and breaking any
# "what changed since last time" comparison, which keys on the agency.
_AGENCY_CANON = (
    ("crisil", "CRISIL"), ("icra", "ICRA"), ("care", "CARE"), ("careedge", "CARE"),
    ("india ratings", "India Ratings"), ("ind-ra", "India Ratings"),
    ("fitch", "Fitch"), ("brickwork", "Brickwork"), ("acuite", "Acuite"),
    ("infomerics", "Infomerics"), ("smera", "SMERA"), ("crest", "CREST"),
)


# A rating action is only ever claimed when the document SAYS it. Both detection branches
# used to end in `else: "Reaffirmed"`, so a rationale whose keyword fell outside the search
# window was recorded as a reaffirmation — an assertion, not a gap.
#
# Measured on live PF data 2026-08-18: 160 rating rows held 146 "Reaffirmed", 14 "Upgrade"
# and ZERO downgrades. Tatva Chintan's 10 Jun 2025 rationale opens "Long Term Rating Crisil
# BBB+/Stable (Downgraded from 'Crisil A-/Negative')" and was stored as Reaffirmed. A
# downgrade reported as a reaffirmation is the worst error this pipeline can make: it is
# precisely the alert the reader needs, turned into an all-clear.
#
# "downgrad" is tested first because a rationale that mentions both actions ("upgraded in
# 2023, downgraded now") must report the current one, and the current one leads the header.
def _source_text(blob: bytes) -> str:
    """Readable text from the source document, PDF or HTML.

    The rating ACTION is stated in the document header and nowhere else reliable, so this
    has to work for both shapes: CRISIL and Brickwork serve HTML, while ICRA's real
    document sits behind a PDF endpoint. Handling only HTML left every ICRA rationale with
    no action at all. Six pages is ample — the action is on page one, and reading a
    120-page annexure would cost time for nothing.
    """
    if not blob:
        return ""
    if blob[:4] == b"%PDF":
        try:
            import fitz
            with fitz.open(stream=blob, filetype="pdf") as doc:
                return chr(10).join(pg.get_text() for pg in list(doc)[:6])
        except Exception:
            return ""
    return blob.decode("utf-8", "ignore")


def _fetch_doc(drive, drive_fid: str, row) -> bytes:
    """The document bytes, from Drive if it is still there and from the SOURCE if
    it is not.

    Retention rule 1 deletes stored documents two days after processing, so a
    re-extract months later finds a `drive_file_id` that 404s. Measured on a random
    12 of the 410 PF rating rows (2026-08-21): 5 dead, 7 alive — about 42% of the
    history unreachable through Drive alone. Every queue row still carries the
    agency URL it came from, and CRISIL/ICRA rationale pages stay published, so the
    source is the reliable copy for anything older than two days.

    Nothing is re-uploaded: the bytes are used for this one call and dropped, which
    keeps retention satisfied by construction rather than by a later cleanup.
    """
    try:
        return download_bytes(drive, drive_fid)
    except Exception as exc:
        url = str(row.get("pdf_url") or "").strip()
        if not url or url.lower() == "nan":
            raise
        log(f"  Drive copy gone ({str(exc)[:40]}) - refetching from source")
        from ingest_company_docs import download_pdf, screener_session
        global _SRC_SESSION
        if _SRC_SESSION is None:
            _SRC_SESSION = screener_session()
        # ICRA serves a JS-rendered page at the rationale URL; the actual document
        # sits behind a separate PDF endpoint. backfill_company_docs already worked
        # this out, so reuse it rather than rediscovering it (rule 4).
        from backfill_company_docs import _resolve_doc_url
        blob = download_pdf(_SRC_SESSION, _resolve_doc_url(url))
        if not blob:
            raise RuntimeError(f"Drive copy deleted and source refetch failed: {url[:80]}")
        return blob


def _rating_col(hdrs) -> int | None:
    """Index of the column holding the RATING, never the one naming the agency.

    The old test was `"rating" in h or "outlook" in h`, and the model's own table leads
    with a column headed "Rating Agency" — which contains "rating". So the agency name was
    read as the rating, which is how 115 of 160 PF rows came to hold "ICRA", "CRISIL" and
    "CareEdge" instead of a rating. Verified against a live ICRA table whose headers are:
    Rating Agency | Instrument Type | Rated Amount | Current Rating/Outlook | FY2025 ...

    A "current" column wins over a historical one, because the mail reports what the
    rating IS, not what it was two years ago.
    """
    cands = [i for i, h in enumerate(hdrs)
             if ("rating" in h or "outlook" in h) and "agency" not in h]
    if not cands:
        return None
    for i in cands:
        if "current" in hdrs[i]:
            return i
    return cands[0]


def _strip_markup(blob: str) -> str:
    """Visible text from an HTML rationale. CRISIL/ICRA/Brickwork serve these as
    HTML, so the action line is buried in tags that a substring search would miss."""
    import html as _html
    t = re.sub(r"<script[^>]*>.*?</script>", " ", str(blob or ""),
               flags=re.S | re.I)
    t = re.sub(r"<style[^>]*>.*?</style>", " ", t, flags=re.S | re.I)
    t = _html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t).strip()


_SRC_SESSION = None   # lazily created; one HTTP session for the whole run


# PARTICIPLES, not stems. Every rationale carries a "Rating Sensitivities" section
# saying things like "sustained weakening MAY LEAD TO A DOWNGRADE" — the noun describes a
# hypothetical, the participle describes something that happened. Matching the stem
# "downgrad" would fire on the hypothetical and invent a downgrade that never occurred.
_ACTION_WORDS = (("downgraded", "Downgrade"), ("upgraded", "Upgrade"),
                 ("reaffirmed", "Reaffirmed"), ("assigned", "Assigned"),
                 ("withdrawn", "Withdrawn"), ("suspended", "Suspended"))


def _action_from(blob: str) -> str:
    """The rating action a document states, or "" when it states none."""
    low = str(blob or "").lower()
    for word, action in _ACTION_WORDS:
        if word in low:
            return action
    return ""


def canon_agency(raw: str) -> str:
    """One agency, one name — so a rating history is not split across spellings."""
    low = str(raw or "").strip().lower()
    if not low or low in ("na", "none", "data_missing"):
        return "DATA_MISSING"
    for token, name in _AGENCY_CANON:
        if token in low:
            return name
    return str(raw).strip()[:40]


def canon_outlook(raw: str) -> str | None:
    """Return a real outlook, or None. A short-term rating is not an outlook."""
    low = str(raw or "").strip().lower().strip("()[]. ")
    if not low:
        return None
    low = re.sub(r"^(rating\s+)?outlook[:\s]*", "", low).strip()
    if low in _VALID_OUTLOOKS:
        return _VALID_OUTLOOKS[low]
    for k, v in _VALID_OUTLOOKS.items():
        if re.search(rf"\b{re.escape(k)}\b", low):
            return v
    return None


def _detect_rating(text: str) -> str:
    """Find the first SEBI/NIC rating symbol in text (e.g. AAA, AA+, BBB-).

    TWO REGEX BUGS FIXED (2026-08-16), both silent and both at scale:

    * `C` and `D` had a LEADING word boundary and no trailing one, so `\bC` matched the
      C of "CARE", "CRISIL" and "Cash Credit". 294 rows were stored as rating "C" and
      100 as "D" — the single commonest values in the table, and almost all wrong.
    * The alternation was not longest-first: `AA` preceded `AA-`, and `BBB` preceded
      `BBB-`, so every minus-notch rating was silently recorded a notch higher.

    Symbols now match longest-first and are anchored on both sides.
    """
    m = re.search(
        r"(?<![A-Za-z0-9+-])"
        r"(AAA|AA\+|AA-|AA|A\+|A-|BBB\+|BBB-|BBB|BB\+|BB-|BB|B\+|B-|B|C|D|A)"
        r"(?![A-Za-z0-9])"
        r"(?:\s*/\s*(Stable|Positive|Negative|Watch|CWN|CWP))?",
        text, re.IGNORECASE
    )
    if m:
        rating = m.group(1).upper()
        outlook = m.group(2).title() if m.group(2) else ""
        return f"{rating}/{outlook}" if outlook else rating
    return "DATA_MISSING"


def parse_gemini_response(text: str, row: pd.Series,
                          source_text: str = "") -> dict:
    """Extract rating metadata from the forensic credit report.

    The rating_prompt produces a Section 1 table with Agency, Instrument,
    Rated Amount, Rating/Outlook, historical migrations. We read that table.
    """
    now_str = datetime.now().isoformat(timespec="seconds")
    isin    = str(row.get("isin") or "")
    symbol  = str(row.get("symbol") or "")
    company = str(row.get("company_name") or "")
    doc_id  = str(row.get("doc_id") or "")

    facts: dict = {c: None for c in RATINGS_COLS}
    facts.update({
        "isin": isin, "symbol": symbol, "company_name": company,
        "agency": canon_agency(_detect_agency(text)),
        "rating": _detect_rating(text),
        "rating_date": str(row.get("announcement_date", ""))[:10] or None,
        "processed_at": now_str, "source_doc_id": doc_id,
    })

    # Try to parse the Section 1 chronology table for richer data
    tables = extract_md_tables(text)
    for t in tables:
        hdrs = [h.lower() for h in t["headers"]]
        # Identify columns
        agency_col    = next((i for i, h in enumerate(hdrs) if "agency" in h), None)
        rating_col    = _rating_col(hdrs)
        instrument_col = next((i for i, h in enumerate(hdrs)
                               if "instrument" in h or "facility" in h), None)
        amount_col    = next((i for i, h in enumerate(hdrs)
                              if "amount" in h or "size" in h or "limit" in h), None)

        if rating_col is None:
            continue

        if t["rows"]:
            # Use the first (most recent) row
            cells = t["rows"][0]
            if agency_col is not None and agency_col < len(cells):
                raw = clean_val(cells[agency_col])
                if raw != "NA":
                    facts["agency"] = canon_agency(raw)

            if rating_col < len(cells):
                raw = clean_val(cells[rating_col])
                if raw != "NA":
                    # "AA+/Stable" -> rating + outlook. But the second half is NOT always
                    # an outlook: "[ICRA]AA+(Stable)/A1+" carries the SHORT-TERM rating
                    # there, which is how A1+ / A2+ / [ICRA]A1+ ended up stored as
                    # outlooks on 107+ rows. Only a recognised outlook is accepted, and
                    # a parenthesised one is preferred since that is where agencies put it.
                    parts = raw.split("/")
                    # Through _detect_rating, not raw. The cell is "[ICRA]AA+(Stable)"
                    # or "Crisil BBB+" depending on the agency, and storing it verbatim is
                    # how the agency's own name ended up in the rating field. The detector
                    # already returns DATA_MISSING for a cell holding no rating symbol,
                    # which is the honest answer for a mis-picked column.
                    facts["rating"] = _detect_rating(parts[0].strip())
                    ol = canon_outlook(
                        (re.search(r"\(([^)]*)\)", raw) or [None, ""])[1]
                        if re.search(r"\(([^)]*)\)", raw) else "")
                    if not ol and len(parts) > 1:
                        ol = canon_outlook(parts[1])
                    if not ol:
                        ol = canon_outlook(raw)
                    if ol:
                        facts["outlook"] = ol
                    # Detect action from the text around the rating. UNKNOWN is a
                    # valid answer — see _action_from() for why asserting "Reaffirmed"
                    # by default is the most dangerous thing this file can do.
                    act = _action_from(raw)
                    if act:
                        facts["rating_action"] = act

            if instrument_col is not None and instrument_col < len(cells):
                raw = clean_val(cells[instrument_col])
                if raw != "NA":
                    facts["instrument_type"] = raw

            if amount_col is not None and amount_col < len(cells):
                raw = clean_val(cells[amount_col])
                if raw != "NA":
                    facts["rated_amount_cr"] = raw
        break

    # Detect outlook separately if not already set
    if not facts.get("outlook"):
        m = re.search(r"\b(Stable|Positive|Negative|Watch|CWN|CWP)\b", text, re.IGNORECASE)
        if m:
            facts["outlook"] = m.group(1).title()

    # Detect the action from the header. The window was 500 chars, shorter than the
    # preamble on a CRISIL rationale, so the word that matters routinely fell outside it.
    # Widened, and silence now means UNKNOWN rather than "Reaffirmed".
    if not facts.get("rating_action"):
        # The model's SUMMARY is searched first, then the source document itself.
        # The action lives in the document header ("Downgraded from 'Crisil
        # A-/Negative'") and a 123k-char summary often paraphrases it away —
        # which is how a real Tatva Chintan downgrade came back as no action at all.
        # THE SOURCE IS AUTHORITATIVE and is searched FIRST. Measured on two live
        # documents: APL Apollo's ICRA rationale (truth: reaffirmed) contains the string
        # "downgrad" ZERO times, yet the model's 123k-char summary discusses downgrades
        # in the abstract — searching the summary first labelled a reaffirmation as a
        # downgrade. Tatva Chintan's CRISIL rationale (truth: downgraded) carries
        # "downgraded to"/"downgraded from" four times, all inside the header.
        facts["rating_action"] = (_action_from(_strip_markup(source_text)[:4000])
                                 or _action_from(text[:4000]))

    return facts


def upsert_ratings(drive, index_id: str, facts) -> None:
    """Idempotent by (isin, source_doc_id). `facts` may be a single dict (per-doc) or a
    LIST of dicts (one batched write covering many docs — used by the batch-write path)."""
    items = facts if isinstance(facts, list) else [facts]
    if not items:
        return
    df = load_parquet(drive, index_id, "ratings.parquet", RATINGS_COLS)
    keys = {(str(f.get("isin")), str(f.get("source_doc_id"))) for f in items}
    if not df.empty and {"isin", "source_doc_id"} <= set(df.columns):
        df = df[~df.apply(lambda r: (str(r["isin"]), str(r["source_doc_id"])) in keys, axis=1)]
    new_df = pd.DataFrame([{c: f.get(c) for c in RATINGS_COLS} for f in items])
    df = pd.concat([df, new_df], ignore_index=True)
    save_parquet(drive, index_id, "ratings.parquet", df)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 / Stage D — Credit rating extraction via Gemini"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    # Backfill flags (defaults leave Phase 2 live behaviour identical).
    parser.add_argument("--all-companies", action="store_true",
                        help="Skip the portfolio filter — process every pending "
                             "rating (universe backfill; use with --max-age-hours).")
    parser.add_argument("--key-prefix", type=str, default=None,
                        help="Load keys from this env prefix / comma list (e.g. "
                             "FREE_POOL,BACKFILL_GEMINI_KEY) instead of GEMINI_API_KEY.")
    parser.add_argument("--max-age-hours", type=float, default=None,
                        help="Only rows discovered within N hours (guards quota on "
                             "stale legacy rows).")
    parser.add_argument("--deadline-min", type=float, default=None,
                        help="Wall-clock cap (min): exit cleanly after this so the "
                             "shared _extract.lock is released before the CI job "
                             "timeout (prevents a killed step leaving a stale lock).")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Commit the queue + structured parquets once per N docs "
                             "instead of per-doc (cuts whole-file Drive rewrites). "
                             "company_page/day_page stay per-doc but idempotent, so a "
                             "kill mid-batch re-processes <=N docs without duplicates.")
    args = parser.parse_args()

    print(f"Phase 2 / Stage D — {DOC_TYPE_LABEL} extraction via Gemini")
    print("-" * 56)

    drive = get_drive()

    if args.key_prefix:
        from gemini_pool import load_keys_multi
        api_keys = load_keys_multi(os.environ, args.key_prefix)   # comma list ok
        if not api_keys:
            print(f"ERROR: no {args.key_prefix}* keys found in env")
            sys.exit(1)
    else:
        api_keys = load_api_keys()
        if not api_keys:
            print("ERROR: no GEMINI_API_KEY or GEMINI_API_KEY_* found in .env")
            sys.exit(1)
    log(f"Loaded {len(api_keys)} Gemini API key(s)"
        + (f" [{args.key_prefix}]" if args.key_prefix else ""))

    # Backfill (--all-companies) gets extra quota-bucket models; Phase-2 PF keeps P1_MODELS.
    from _extractor_base import BACKFILL_EXTRA_MODELS
    from provider_router import make_extraction_pool
    _models = list(GEMINI_MODEL) + (BACKFILL_EXTRA_MODELS if args.all_companies else [])
    # Phase 2: BACKFILL-ONLY Groq/Cerebras fallback when Gemini is exhausted (PF path =
    # pure Gemini, unchanged). Plain GeminiKeyPool unless --all-companies AND alt keys exist.
    gemini = make_extraction_pool(api_keys, _models, enable_fallback=args.all_companies)

    prompt_path = Path(__file__).resolve().parent / PROMPT_FILE
    if not prompt_path.exists():
        print(f"ERROR: prompt file not found: {prompt_path}")
        sys.exit(1)
    prompt = prompt_path.read_text(encoding="utf-8")
    _struct_path = Path(__file__).resolve().parent / STRUCT_PROMPT_FILE
    struct_prompt = _struct_path.read_text(encoding="utf-8") if _struct_path.exists() else ""

    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id   = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id  = get_or_create_subfolder(drive, repo_id,   "_index")

    # Phase 1 (BACKFILL ONLY): pre-mark PerDay-dead buckets from the health cache so
    # this run skips them instead of burning a call each to re-discover. PF path unchanged.
    if args.all_companies:
        gemini.prime_from_health(drive, index_id)

    # T12 Phase-2 safety: serialize shared-file writes via the global _extract.lock.
    # On contention exit cleanly — the next run resumes (rows stay pending).
    _is_backfill = args.all_companies          # universe backfill path -> yield to Phase 2
    if not args.dry_run:
        # PF path = Phase 2 (wait for the lock); backfill yields immediately.
        if _is_backfill:
            got = acquire_lock(drive, index_id, _LOCK_NAME, DOC_TYPE,
                               max_age_min=_LOCK_MAX_AGE_MIN, defer_to_phase2=True)
        else:
            got = acquire_lock(drive, index_id, _LOCK_NAME, DOC_TYPE,
                               max_age_min=_LOCK_MAX_AGE_MIN, wait_min=15)
        if not got:
            log("  Lock unavailable (yielded to Phase 2 / timed out) — exiting cleanly.")
            sys.exit(0)
        atexit.register(release_lock, drive, index_id, _LOCK_NAME)

    queue = load_queue(drive, index_id)
    pending_mask = (queue["status"] == "pending") & (queue["doc_type"] == DOC_TYPE)
    pending_idx = queue.index[pending_mask].tolist()
    log(f"Queue: {len(queue)} total rows, {len(pending_idx)} pending {DOC_TYPE}")

    # Optional freshness window (discovered_at within N hours) — backfill quota guard.
    if args.max_age_hours and "discovered_at" in queue.columns:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(hours=args.max_age_hours)) \
            .isoformat(timespec="seconds")
        before = len(pending_idx)
        pending_idx = [i for i in pending_idx
                       if str(queue.loc[i, "discovered_at"]) >= cutoff]
        log(f"  After {args.max_age_hours:.0f}h freshness filter: "
            f"{len(pending_idx)}/{before} to process")

    # Portfolio filter: skip non-portfolio companies (rows stay pending). Universe
    # backfill (--all-companies) bypasses so every fresh rating gets judged.
    if args.all_companies:
        log("  --all-companies: portfolio filter bypassed (rating backfill)")
    else:
        portfolio_isins = load_portfolio_isins(drive, folder_id)
        if portfolio_isins:
            before = len(pending_idx)
            pending_idx = [i for i in pending_idx
                           if str(queue.loc[i, "isin"]).strip() in portfolio_isins]
            log(f"  After portfolio filter: {len(pending_idx)}/{before} to process")

    if args.limit:
        pending_idx = pending_idx[: args.limit]

    counts = {"processed": 0, "error": 0, "skipped": 0}
    _t0 = time.time()
    BATCH = max(1, args.batch_size)

    # BATCH-WRITE: company_page/day_page stay per-doc (idempotent via the doc_id marker —
    # they are per-company/day files). The idempotent parquet upserts (ratings + 3 structured)
    # and the queue mark-done are BUFFERED and committed once per BATCH, cutting ~12s of
    # whole-file Drive rewrites per doc to roughly once per batch. A CI kill mid-batch
    # re-processes <=BATCH docs: the company_page/day_page re-append is skipped (marker) and
    # the upserts dedupe by source_doc_id → no duplicates, no lost structured rows.
    b_idx: list = []
    b_facts: list = []
    b_dr: list = []
    b_co: list = []
    b_se: list = []

    def _flush(force_save: bool = False) -> None:
        if b_idx:
            upsert_ratings(drive, index_id, b_facts)        # list form -> one load+save
            if b_dr:
                upsert_structured(drive, index_id, "rating_drivers.parquet",
                                  RATING_DRIVERS_COLS, b_dr)
            if b_co:
                upsert_structured(drive, index_id, "rating_concerns.parquet",
                                  RATING_CONCERNS_COLS, b_co)
            if b_se:
                upsert_structured(drive, index_id, "rating_sensitivity.parquet",
                                  RATING_SENSITIVITY_COLS, b_se)
            now_s = datetime.now().isoformat(timespec="seconds")
            for qi in b_idx:
                queue.loc[qi, "status"] = "done"
                queue.loc[qi, "processed_at"] = now_s
            log(f"  [flush] committed {len(b_idx)} doc(s): 1 queue write + batched upserts.")
        if b_idx or force_save:
            save_queue(drive, index_id, queue)              # persists done + any error marks
        b_idx.clear(); b_facts.clear(); b_dr.clear(); b_co.clear(); b_se.clear()

    for queue_idx in pending_idx:
        # Wall-clock cap: the final flush below commits the partial batch before the lock
        # is released, so deadline never loses buffered work.
        if args.deadline_min and (time.time() - _t0) / 60.0 >= args.deadline_min:
            log(f"  Deadline {args.deadline_min:.0f} min reached — flushing + exiting.")
            break
        row = queue.loc[queue_idx]
        label = f"{row.get('symbol', '?')!s:<14} {str(row.get('title', ''))[:55]}"
        log(f"Processing: {label}")

        drive_fid = str(row.get("drive_file_id") or "").strip()
        if not drive_fid:
            log("  SKIP: no drive_file_id")
            counts["skipped"] += 1
            continue
        doc_id = str(row.get("doc_id", ""))

        try:
            pdf_bytes = _fetch_doc(drive, drive_fid, row)
            log(f"  PDF: {len(pdf_bytes):,} bytes")

            display_name = f"{row.get('symbol', 'DOC')}_{doc_id[:12]}.pdf"
            # Detect PDF vs HTML/text: CRISIL/SMERA/Brickwork rationales arrive as text;
            # sending them as application/pdf was fatal-erroring ~36% of ratings.
            markdown_text = call_over_doc(gemini, prompt, pdf_bytes, name=display_name,
                                          max_output_tokens=RATING_MAX_OUTPUT_TOKENS)
            log(f"  Gemini response: {len(markdown_text):,} chars")

            if args.dry_run:
                print(f"\n{'='*60}\nDRY RUN — {row.get('symbol')}\n"
                      f"{markdown_text[:800]}\n{'='*60}\n")
                counts["processed"] += 1
                continue

            # HTML/text rationales only: a real PDF would need a text extraction pass,
            # and the agencies that bury the action in markup are exactly the ones
            # serving HTML.
            _src = _source_text(pdf_bytes)
            facts = parse_gemini_response(markdown_text, row, source_text=_src)
            log(f"  Parsed: agency={facts['agency']}, rating={facts['rating']}, "
                f"outlook={facts['outlook']}, action={facts['rating_action']}")

            if OUTPUT_COMPANY_MD:
                append_company_page(
                    drive, repo_id,
                    key=str(row.get("key") or row.get("isin") or row.get("symbol") or ""),
                    doc_type_label=DOC_TYPE_LABEL,
                    content=markdown_text,
                    doc_title=str(row.get("title", "")),
                    quarter=f"{facts['agency']} {facts['rating']}",
                    dedup_marker=doc_id,
                )

            if OUTPUT_DAY_MD:
                append_day_page(
                    drive, repo_id,
                    doc_type=DOC_TYPE,
                    announcement_date=str(row.get("announcement_date", "")),
                    symbol=str(row.get("symbol", "")),
                    company_name=str(row.get("company_name", "")),
                    quarter=f"{facts['agency']} {facts['rating']}",
                    content=markdown_text,
                    dedup_marker=doc_id,
                )

            # Stage 3 tabulation (ADDITIVE, best-effort): separate JSON-only pass →
            # rating_drivers / rating_concerns / rating_sensitivity. Buffered, flushed in batch.
            try:
                dr, co, se = tabulate_rating(
                    gemini, struct_prompt, pdf_bytes, row,
                    facts.get("agency"), facts.get("rating_date"),
                    datetime.now().isoformat(timespec="seconds"))
                b_dr.extend(dr or []); b_co.extend(co or []); b_se.extend(se or [])
                log(f"  Tabulated: drivers={len(dr)}, concerns={len(co)}, "
                    f"sensitivity={len(se)}")
            except RateLimitExhausted:
                log("  Structured pass: keys exhausted — tabulation skipped.")
            except Exception as _e:
                log(f"  WARNING: rating tabulation failed ({str(_e)[:90]}).")

            # Buffer the idempotent parquet rows + queue row; committed together at flush.
            b_facts.append(facts)
            b_idx.append(queue_idx)
            counts["processed"] += 1
            log(f"  Buffered: {row.get('symbol')} ({len(b_idx)}/{BATCH})")
            if len(b_idx) >= BATCH:
                _flush()

        except RateLimitExhausted:
            log("All Gemini keys rate-limited — flushing + stopping cleanly.")
            break

        except Exception as exc:
            log(f"  ERROR: {str(exc)[:120]}")
            queue.loc[queue_idx, "status"] = "error"
            queue.loc[queue_idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            counts["error"] += 1

    if not args.dry_run:
        _flush(force_save=True)     # commit the final partial batch + any error marks

    if not args.dry_run:
        from _extractor_base import persist_gemini_usage
        persist_gemini_usage(drive, index_id, gemini.summary(), DOC_TYPE,
                             "backfill" if getattr(args, "all_companies", False) else "phase2")
    print("-" * 56)
    print(f"Processed : {counts['processed']}")
    print(f"Errors    : {counts['error']}")
    print(f"Skipped   : {counts['skipped']}")
    if not args.dry_run:
        print("Output: company_repo/_index/ratings.parquet")
        print("Output: company_repo/<key>/company_page.md")
        print("Output: company_repo/_daily/rating_DD_MMMYYYY.md")


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    # THE BUG THAT PRODUCED 294 "C" RATINGS: C with no trailing boundary matched the
    # C of CARE / CRISIL / Cash Credit.
    check("agency name is not read as a rating",
          _detect_rating("CARE Ratings has assigned") != "C")
    check("CRISIL is not read as a rating",
          _detect_rating("CRISIL Ratings Limited") != "C")
    check("'Cash Credit' is not read as a rating",
          _detect_rating("Cash Credit facility of Rs 50 crore") != "C")
    check("'Debt' is not read as rating D",
          _detect_rating("Debt service coverage remains") != "D")
    # A real C or D must still be found.
    check("a genuine C rating is found", _detect_rating("rated CRISIL C") in ("C", "C/"))
    check("a genuine D rating is found", "D" in _detect_rating("downgraded to D"))

    # LONGEST-FIRST: a minus notch must not be recorded a notch higher.
    check("AA- stays AA-", _detect_rating("[ICRA]AA-").startswith("AA-"))
    check("BBB- stays BBB-", _detect_rating("CARE BBB-").startswith("BBB-"))
    check("AA+ stays AA+", _detect_rating("CRISIL AA+").startswith("AA+"))
    check("AAA is not truncated", _detect_rating("IND AAA").startswith("AAA"))

    # OUTLOOK: a short-term rating is not an outlook.
    check("A1+ is rejected as an outlook", canon_outlook("A1+") is None)
    check("[ICRA]A1+ is rejected", canon_outlook("[ICRA]A1+") is None)
    check("Stable is accepted", canon_outlook("Stable") == "Stable")
    check("(Stable) is accepted", canon_outlook("(Stable)") == "Stable")
    check("'Outlook: Positive' is accepted", canon_outlook("Outlook: Positive") == "Positive")
    check("CWN maps to Watch Negative", canon_outlook("CWN") == "Watch Negative")
    check("empty is None", canon_outlook("") is None)

    # AGENCY: one agency, one name.
    check("CARE Ratings canonicalises", canon_agency("CARE Ratings") == "CARE")
    check("CareEdge canonicalises", canon_agency("CareEdge Ratings") == "CARE")
    check("ICRA canonicalises", canon_agency("ICRA Limited") == "ICRA")
    check("Ind-Ra canonicalises", canon_agency("Ind-Ra") == "India Ratings")
    check("blank is DATA_MISSING", canon_agency("") == "DATA_MISSING")

    # END TO END on the shape that caused the live defect.
    md = ("| Agency | Instrument | Rating/Outlook |\n|---|---|---|\n"
          "| CareEdge Ratings | Cash Credit | [ICRA]AA+(Stable)/A1+ |\n")
    f = parse_gemini_response(md, pd.Series({"isin": "I", "symbol": "S",
                                             "announcement_date": "2026-06-01"}))
    check("e2e agency canonical", f["agency"] == "CARE")
    check("e2e outlook is the real outlook", f["outlook"] == "Stable")
    check("e2e outlook is NOT the short-term rating", f["outlook"] != "A1+")

    # ---- rating action: never assert what the document does not say
    check("downgrade detected",
          _action_from("Long Term Rating Crisil BBB+/Stable "
                       "(Downgraded from 'Crisil A-/Negative')") == "Downgrade")
    check("upgrade detected", _action_from("Rating upgraded to CRISIL A") == "Upgrade")
    check("reaffirmation only when stated",
          _action_from("CRISIL A-/Stable Reaffirmed") == "Reaffirmed")
    check("assignment detected", _action_from("Rating assigned at CARE BBB") == "Assigned")
    check("withdrawal detected", _action_from("The rating has been withdrawn") == "Withdrawn")
    # THE BUG: silence used to become "Reaffirmed", turning Tatva Chintan's real
    # A-/Negative -> BBB+/Stable downgrade into an all-clear.
    check("silence is UNKNOWN, not Reaffirmed",
          _action_from("Total Bank Loan Facilities Rated Rs.245 Crore") == "")
    check("empty input safe", _action_from("") == "" and _action_from(None) == "")
    check("downgrade wins when both words appear",
          _action_from("upgraded in 2023, downgraded now") == "Downgrade")
    check("case insensitive", _action_from("DOWNGRADED FROM CRISIL A-") == "Downgrade")
    # The header window was 500 chars; on a real CRISIL rationale the word that matters
    # sits past it, which is why the live data held zero downgrades across 160 rows.
    hdr = ("Rating Action Total Bank Loan Facilities Rated Rs.245 Crore " + ("x " * 260)
           + "Long Term Rating Crisil BBB+/Stable (Downgraded from A-/Negative)")
    check("old 500-char window would have missed it", _action_from(hdr[:500]) == "")
    check("widened window catches it", _action_from(hdr[:4000]) == "Downgrade")

    # ---- the action lives in the SOURCE document, not in the model summary
    crisil = ("<html><body><table><tr><td>Rating Action</td></tr>"
              "<tr><td>Long Term Rating</td><td>Crisil BBB+/Stable "
              "(Downgraded from &#39;Crisil A-/Negative&#39;)</td></tr>"
              "</table></body></html>")
    check("markup stripped to visible text",
          "Downgraded from" in _strip_markup(crisil))
    check("entities unescaped", "'Crisil A-/Negative'" in _strip_markup(crisil))
    check("script and style discarded",
          "hidden" not in _strip_markup("<style>.a{x:hidden}</style><p>ok</p>"))
    check("action read out of the HTML source",
          _action_from(_strip_markup(crisil)[:4000]) == "Downgrade")
    # A summary that paraphrases the action away must NOT block the source lookup.
    row_ = pd.Series({"isin": "I", "symbol": "TATVA", "announcement_date": "2025-06-10"})
    quiet = ("| Agency | Instrument | Rating/Outlook |" + chr(10)
             + "|---|---|---|" + chr(10)
             + "| CRISIL | CC | BBB+/Stable |" + chr(10))
    f_no = parse_gemini_response(quiet, row_)
    check("no source, no invented action", f_no["rating_action"] == "")
    f_src = parse_gemini_response(quiet, row_, source_text=crisil)
    check("source supplies the downgrade", f_src["rating_action"] == "Downgrade")
    check("the rating itself is unchanged by the source lookup",
          f_src["rating"] == f_no["rating"])

    # ---- document fetch: Drive first, SOURCE when retention has deleted it
    class _OKDrive:
        pass
    _live = {"id1": b"%PDF-1.4 real bytes"}
    import extract_rating as _ER
    _orig_db = _ER.download_bytes
    try:
        _ER.download_bytes = lambda drv, fid: _live[fid]
        r_ok = pd.Series({"pdf_url": "http://agency/x.html"})
        check("Drive copy used when alive",
              _ER._fetch_doc(None, "id1", r_ok) == b"%PDF-1.4 real bytes")
        # Drive gone + no url -> the original error must surface, not a silent empty doc
        def _boom(drv, fid):
            raise RuntimeError("HttpError 404")
        _ER.download_bytes = _boom
        raised = False
        try:
            _ER._fetch_doc(None, "id1", pd.Series({"pdf_url": ""}))
        except Exception:
            raised = True
        check("no source url -> error surfaces", raised)
        raised2 = False
        try:
            _ER._fetch_doc(None, "id1", pd.Series({"pdf_url": "nan"}))
        except Exception:
            raised2 = True
        check("literal nan url is not a url", raised2)
    finally:
        _ER.download_bytes = _orig_db

    # ---- the rating column is never the agency column
    live = ["rating agency", "instrument type", "rated amount (rs. crore)",
            "current rating/outlook (feb 26, 2026)", "fy2025 rating/outlook"]
    check("agency column rejected", _rating_col(live) == 3)
    check("current beats historical",
          _rating_col(["fy2024 rating", "current rating/outlook"]) == 1)
    check("plain rating column still found",
          _rating_col(["instrument", "rating/outlook"]) == 1)
    check("outlook-only header works", _rating_col(["instrument", "outlook"]) == 1)
    check("no rating column -> None", _rating_col(["instrument", "amount"]) is None)
    check("agency-only header is not a rating column",
          _rating_col(["rating agency", "amount"]) is None)

    # ---- hypothetical downgrades in Rating Sensitivities must not become actions
    sens = ("Rating Sensitivities Negative factors: sustained weakening in margins "
            "may lead to a downgrade of the ratings")
    check("hypothetical downgrade ignored", _action_from(sens) == "")
    check("hypothetical upgrade ignored",
          _action_from("an upgrade could follow sustained improvement") == "")
    check("real downgrade still caught",
          _action_from("Ratings downgraded to Crisil BBB+/Stable") == "Downgrade")
    check("real reaffirmation still caught",
          _action_from("[ICRA]AA+(Stable); reaffirmed for enhanced amount") == "Reaffirmed")

    # ---- the SOURCE outranks the model summary, which paraphrases
    src_reaff = "<p>[ICRA]AA+(Stable) /[ICRA]A1+; reaffirmed for enhanced amount</p>"
    summary_says_downgrade = ("| Rating Agency | Current Rating/Outlook |" + chr(10)
                              + "|---|---|" + chr(10)
                              + "| ICRA | [ICRA]AA+(Stable) |" + chr(10)
                              + "A downgraded peer was discussed at length." + chr(10))
    row2 = pd.Series({"isin": "I", "symbol": "APLAPOLLO", "announcement_date": "2026-06-28"})
    f2 = parse_gemini_response(summary_says_downgrade, row2, source_text=src_reaff)
    check("source reaffirmation beats summary downgrade talk",
          f2["rating_action"] == "Reaffirmed")
    check("agency name is NOT stored as the rating", f2["rating"] != "ICRA")
    check("the real rating is stored", "AA+" in f2["rating"])

    # ---- the rating cell is normalised, never stored verbatim
    icra_tbl = ("| Rating Agency | Current Rating/Outlook |" + chr(10)
                + "|---|---|" + chr(10)
                + "| ICRA | [ICRA]AA+(Stable)/[ICRA]A1+ |" + chr(10))
    row3 = pd.Series({"isin": "I", "symbol": "APLAPOLLO", "announcement_date": "2026-06-28"})
    f3 = parse_gemini_response(icra_tbl, row3)
    check("ICRA bracket prefix stripped", f3["rating"] == "AA+")
    check("short-term rating not stored as outlook", f3["outlook"] != "A1+")
    check("outlook still read", f3["outlook"] == "Stable")
    check("agency still ICRA", f3["agency"] == "ICRA")
    # A mis-picked column holding no rating symbol reports DATA_MISSING, not the text.
    check("agency name never becomes a rating", _detect_rating("CareEdge") == "DATA_MISSING")

    # ---- source text works for PDFs as well as HTML
    check("html source decoded", "reaffirmed" in _source_text(b"<p>reaffirmed</p>"))
    check("empty source safe", _source_text(b"") == "" and _source_text(None) == "")
    check("undecodable pdf degrades to empty, not a crash",
          _source_text(b"%PDF-1.4 not really a pdf") == "")

    print(f"\nextract_rating self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    main()
