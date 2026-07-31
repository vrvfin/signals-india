r"""
extract_mgmt_quotes.py — N8. The verbatim management quote spine for section 20.

This is the source deck's strongest device: the same commitment tracked across four
consecutive earnings calls, quoted exactly, then set against what the numbers did. It
only works if the quotes are real, so:

  EVERY QUOTE IS VERIFIED BY STRING MATCH against the transcript it came from, and any
  quote not found verbatim is DISCARDED before writing. A fabricated or paraphrased
  quotation cannot enter the store. (Ground truth check: the Landmark Q4 FY26 transcript
  really does contain "this is the year where we want to get the profits back", which is
  what the deck quoted — so this approach reproduces the real artefact.)

Does NOT touch `extract_concall.py` or `concall_prompt.txt`. Memory
`phase2-concall-untouchable` records the live concall path as P0; this re-fetches the same
transcripts from `pdf_url` and writes its own table.

Schema: `company_repo/_index/mgmt_quotes.parquet`
  isin · symbol · company_name · quarter · call_date · speaker · role · topic · quote
  · commitment · source_doc_id · extracted_at

Usage:
  python scripts/extract_mgmt_quotes.py --names LANDMARK --dry-run
  python scripts/extract_mgmt_quotes.py --names LANDMARK
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from _extractor_base import (find_file, download_bytes, upload_bytes, log)
from gemini_pool import load_keys
import narrative_sources as NS
import verify_grounding as VG
from narrative_factpack import Store, resolve, IDX

QUOTES_FILE = "mgmt_quotes.parquet"
QUOTE_COLS = ["isin", "symbol", "company_name", "quarter", "call_date", "speaker",
              "role", "topic", "quote", "commitment", "source_doc_id", "extracted_at"]

MODELS = ["gemini-2.5-flash", "gemini-flash-latest"]
MAX_OUTPUT_TOKENS = 14000        # thinking models: see narrative_generate
CHUNK_CHARS = 70_000
MAX_QUOTES_PER_CALL = 12
INTER_CALL_SLEEP = 3.0
MIN_TRANSCRIPT_CHARS = 5_000     # below this it is a notice, not a transcript

TOPICS = ("guidance", "margin", "capex", "demand", "expansion", "consolidation",
          "cost", "debt", "competition", "strategy", "other")

PROMPT = """You are extracting VERBATIM management quotations from an Indian company's \
earnings call transcript. You are not summarising. You are copying sentences.

THE BINDING RULE
Each `quote` must be copied EXACTLY from the transcript, character for character. Quotes \
are verified by string match against the transcript and any quote not found verbatim is \
DISCARDED. Do not paraphrase, do not tidy grammar, do not join sentences that are not \
adjacent, do not fix transcription errors.

WHAT TO EXTRACT
Prefer statements that a reader could later hold management to:
- forward commitments (margin, capex, growth, debt, timelines)
- explicit refusals to guide ("we are not giving guidance on...")
- explanations of a miss or a one-off
- strategy shifts (expansion vs consolidation)
- answers to pointed analyst questions

Skip pleasantries, operator instructions, and analyst statements — quote MANAGEMENT only \
(the analyst's question may go in `topic` context, not in `quote`).

For each quote also record:
  speaker      the person's name as printed in the transcript
  role         their title if the transcript states it, else ""
  topic        one of: {topics}
  commitment   if the quote contains a checkable commitment, state it in a few words \
(e.g. "capex ~INR 50 cr FY27"); else ""

OUTPUT — STRICT JSON, NO MARKDOWN FENCE, NO PROSE
{{"quotes":[{{"speaker":"<name>","role":"<title or \\"\\">","topic":"<one topic>",
            "quote":"<VERBATIM sentence(s) from the transcript>",
            "commitment":"<checkable commitment or \\"\\">"}}]}}

At most {maxq} quotes, the most consequential first. If this document is not an earnings \
call transcript, return {{"quotes":[]}}.

=== COMPANY ===
{company}

=== TRANSCRIPT {doc_id} ({call_date}) ===
{document}
"""


def _quarter_from(text: str, call_date: str) -> str:
    """Quarter label from the transcript header if stated, else derived from the call
    date (Indian FY: Apr-Mar, calls happen after quarter end)."""
    m = re.search(r"\bQ([1-4])\s*FY\s*'?(\d{2,4})\b", text[:6000], re.I)
    if m:
        yr = m.group(2)[-2:]
        return f"Q{m.group(1)} FY{yr}"
    d = pd.to_datetime(call_date, errors="coerce")
    if pd.isna(d):
        return ""
    # a call in May/Jun reports the Jan-Mar quarter (Q4) of the FY just ended
    q, fy = {1: (3, 0), 2: (4, 0), 3: (4, 0), 4: (4, 0), 5: (4, 0), 6: (4, 0),
             7: (1, 1), 8: (1, 1), 9: (1, 1), 10: (2, 1), 11: (2, 1), 12: (3, 1)}[d.month]
    return f"Q{q} FY{str(d.year + fy)[2:]}"


def _parse(raw: str) -> list[dict]:
    t = (raw or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    for cand in (t, t[t.find("{"):] if "{" in t else ""):
        if not cand:
            continue
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and isinstance(d.get("quotes"), list):
                return d["quotes"]
        except json.JSONDecodeError:
            continue
    out = []
    for m in re.finditer(r'\{[^{}]*"quote"\s*:\s*"(?:[^"\\]|\\.)*"[^{}]*\}', t):
        try:
            out.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return out


def extract_call(pool, company: dict, doc_id: str, call_date: str,
                 text: str) -> tuple[list[dict], dict]:
    stats = {"returned": 0, "dropped_verbatim": 0, "dropped_shape": 0}
    if len(text) < MIN_TRANSCRIPT_CHARS:
        log(f"      {len(text):,} chars — too short to be a transcript, skipped")
        return [], stats
    quarter = _quarter_from(text, call_date)
    norm_doc = VG.normalise(text)
    prompt = (PROMPT
              .replace("{topics}", ", ".join(TOPICS))
              .replace("{maxq}", str(MAX_QUOTES_PER_CALL))
              .replace("{company}", f"{company['name']} ({company['symbol']})")
              .replace("{doc_id}", doc_id)
              .replace("{call_date}", str(call_date))
              .replace("{document}", text[:CHUNK_CHARS]))
    try:
        raw = pool.call_text(prompt, f"quotes_{doc_id}",
                            max_output_tokens=MAX_OUTPUT_TOKENS)
    except Exception as e:
        log(f"      call failed — {str(e)[:130]}")
        return [], stats
    kept = []
    for q in _parse(raw):
        stats["returned"] += 1
        quote = str(q.get("quote", "")).strip()
        speaker = str(q.get("speaker", "")).strip()
        if not quote or len(quote) < 25:
            stats["dropped_shape"] += 1
            continue
        nq = VG.normalise(quote)
        if nq not in norm_doc and nq.lower() not in norm_doc.lower():
            stats["dropped_verbatim"] += 1
            continue
        topic = str(q.get("topic", "other")).strip().lower()
        kept.append({"quarter": quarter, "call_date": str(call_date),
                     "speaker": speaker[:120],
                     "role": str(q.get("role", ""))[:120],
                     "topic": topic if topic in TOPICS else "other",
                     "quote": quote[:1200],
                     "commitment": str(q.get("commitment", ""))[:300],
                     "source_doc_id": doc_id})
    return kept, stats


def save(store: Store, rows: pd.DataFrame, dry_run: bool) -> str:
    drive = store.drive
    folder = store.folder(IDX)
    existing = pd.DataFrame(columns=QUOTE_COLS)
    fid = find_file(drive, folder, QUOTES_FILE)
    if fid:
        try:
            existing = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
        except Exception as e:
            log(f"  WARNING: could not read existing {QUOTES_FILE} ({str(e)[:70]})")
    for c in QUOTE_COLS:
        if c not in existing.columns:
            existing[c] = None
    isins = set(rows["isin"].astype(str)) if not rows.empty else set()
    other = existing[~existing["isin"].astype(str).isin(isins)]
    merged = pd.concat([other, rows], ignore_index=True)[QUOTE_COLS]
    merged = merged.drop_duplicates(subset=["isin", "source_doc_id", "quote"],
                                    keep="first")
    if dry_run:
        return (f"DRY RUN — would write {len(merged)} rows "
                f"({len(rows)} this run, {len(other)} preserved)")
    buf = io.BytesIO()
    merged.to_parquet(buf, index=False)
    # signature is (drive, folder_id, filename, data, mimetype, existing_id) — passing
    # the file id positionally put it in the mimetype slot and crashed the upload AFTER
    # the extraction had already been paid for.
    upload_bytes(drive, folder, QUOTES_FILE, buf.getvalue(),
                 "application/octet-stream", existing_id=fid)
    return f"wrote {QUOTES_FILE}: {len(merged)} rows total, {len(rows)} from this run"


def run_one(store: Store, pool, token: str, cache: Path, dry_run: bool) -> pd.DataFrame:
    r = resolve(store, token)
    if r is None:
        log(f"  could not resolve '{token}'")
        return pd.DataFrame(columns=QUOTE_COLS)
    isin, symbol, name = r
    company = {"isin": isin, "symbol": symbol, "name": name}
    log(f"  {name} ({symbol})")

    rows = NS.queue_rows(store, isin, symbol, limits={"concall": 6})
    rows = rows[rows["doc_type"].astype(str) == "concall"] if not rows.empty else rows
    if rows.empty:
        log("    no concalls in the queue — run backfill_company_docs.py first")
        return pd.DataFrame(columns=QUOTE_COLS)
    log(f"    {len(rows)} concall(s) in the queue"
        + ("  <-- section 20 wants >= 4 for a claim-across-calls spine"
           if len(rows) < 4 else ""))

    out, tot = [], {"returned": 0, "dropped_verbatim": 0, "dropped_shape": 0}
    for _, row in rows.iterrows():
        did = NS.doc_id_for(row)
        cached = cache / f"{did}.txt"
        if cached.exists():
            text = cached.read_text(encoding="utf-8", errors="ignore")
        else:
            from company_deep_report import _refetch_doc_bytes
            data = _refetch_doc_bytes(row)
            text = NS._clean(NS._pdf_text(data)) if data else ""
            if text:
                cache.mkdir(parents=True, exist_ok=True)
                cached.write_text(text, encoding="utf-8")
        if not text:
            log(f"    {did}: no text — skipped")
            continue
        log(f"    {did}: {len(text):,} chars")
        if dry_run:
            log(f"      DRY RUN — would extract up to {MAX_QUOTES_PER_CALL} quotes")
            continue
        qs, stats = extract_call(pool, company, did,
                                 str(row.get("announcement_date")), text)
        for k in tot:
            tot[k] += stats.get(k, 0)
        log(f"      {len(qs)} quote(s) kept of {stats['returned']} returned "
            f"({stats['dropped_verbatim']} failed verbatim check)")
        out += qs
        time.sleep(INTER_CALL_SLEEP)

    if not out:
        return pd.DataFrame(columns=QUOTE_COLS)
    df = pd.DataFrame(out)
    df["isin"], df["symbol"], df["company_name"] = isin, symbol, name
    df["extracted_at"] = datetime.now().isoformat(timespec="seconds")
    df = df[QUOTE_COLS]
    log(f"    TOTAL {len(df)} quotes across {df['quarter'].nunique()} quarter(s); "
        f"topics: {df['topic'].value_counts().to_dict()}")
    if tot["returned"]:
        rate = 100.0 * tot["dropped_verbatim"] / tot["returned"]
        log(f"    verbatim discard rate: {rate:.0f}%"
            + ("  <-- HIGH: the model is paraphrasing quotes" if rate > 40 else ""))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache", default="")
    a = ap.parse_args()

    store = Store()
    pool = None
    if not a.dry_run:
        keys = (load_keys(os.environ, prefix="FREE_POOL")
                or load_keys(os.environ, prefix="GEMINI_API_KEY"))
        if not keys:
            print("no Gemini keys — set FREE_POOL_n")
            return 1
        from _extractor_base import GeminiKeyPool
        pool = GeminiKeyPool(keys, MODELS)

    frames = [run_one(store, pool, t, Path(a.cache or f"./_src_{t}"), a.dry_run)
              for t in a.names]
    allrows = (pd.concat(frames, ignore_index=True) if frames
               else pd.DataFrame(columns=QUOTE_COLS))
    if allrows.empty:
        print("no quotes extracted — nothing written")
        return 0
    print(save(store, allrows, a.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
