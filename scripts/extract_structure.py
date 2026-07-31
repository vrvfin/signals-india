r"""
extract_structure.py — N7. Extracts the structural facts sections 1/3/4/6/9/12/23 need
(milestones, subsidiaries, management, segments, portfolio units, company-named risks)
into `company_repo/_index/company_structure.parquet`.

WHY THIS IS A SEPARATE PASS, NOT A NEW FIELD IN THE LIVE AR EXTRACTOR
  `extract_annual_report.py` / `ar_structured_prompt.txt` are shared by Phase 2 (live)
  and Phase 3 (backfill). CLAUDE.md rule 3 makes any change there a both-pipelines
  problem, and memory `phase2-concall-untouchable` records the live path as P0. This
  reads the same documents by re-fetching them from `pdf_url` and writes its OWN table,
  so nightly CI behaviour is bit-for-bit unchanged.

THE GUARANTEE THAT MATTERS
  Every record carries an `evidence_span`, and a record whose span is NOT found verbatim
  in the source is DISCARDED before it is written. Garbage never enters the store, so
  downstream sections inherit grounded facts by construction rather than by trusting the
  model. The discard count is reported — a high rate means the prompt or the document is
  the problem, and you can see it.

Schema (NEW table, additive — nothing else reads it yet):
  isin · symbol · company_name · kind · item · field · value · unit · period
  · evidence_span · source_doc_id · doc_date · extracted_at

Usage:
  python scripts/extract_structure.py --names LANDMARK --dry-run
  python scripts/extract_structure.py --names LANDMARK
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

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)
from gemini_pool import load_keys
import narrative_sources as NS
import verify_grounding as VG
from narrative_factpack import Store, resolve, IDX

STRUCT_FILE = "company_structure.parquet"
STRUCT_COLS = ["isin", "symbol", "company_name", "kind", "item", "field", "value",
               "unit", "period", "evidence_span", "source_doc_id", "doc_date",
               "extracted_at"]
KINDS = ("milestone", "subsidiary", "management", "portfolio_unit", "segment", "risk")

MODELS = ["gemini-2.5-flash", "gemini-flash-latest"]
# Thinking models spend this budget on reasoning too — see narrative_generate for the
# truncation this caused at 3000.
MAX_OUTPUT_TOKENS = 14000
# Annual reports run to 400k chars; a single call cannot carry that. Chunk and merge.
CHUNK_CHARS = 90_000
CHUNK_OVERLAP = 2_000
MAX_CHUNKS_PER_DOC = 4
INTER_CALL_SLEEP = 3.0
# Which doc types are worth this pass. Concalls have their own extractor (N8).
WANTED = ("annual_report", "presentation")


def _chunks(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, i = [], 0
    while i < len(text) and len(out) < MAX_CHUNKS_PER_DOC:
        out.append(text[i:i + CHUNK_CHARS])
        i += CHUNK_CHARS - CHUNK_OVERLAP
    return out


def _parse(raw: str) -> list[dict]:
    t = (raw or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    for cand in (t, t[t.find("{"):] if "{" in t else ""):
        if not cand:
            continue
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and isinstance(d.get("records"), list):
                return d["records"]
        except json.JSONDecodeError:
            continue
    # truncated array — salvage whole objects
    recs = []
    for m in re.finditer(r'\{[^{}]*"evidence_span"\s*:\s*"(?:[^"\\]|\\.)*"[^{}]*\}', t):
        try:
            recs.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return recs


def extract_doc(pool, tpl: str, company: dict, doc_id: str, doc_type: str,
                doc_date: str, text: str, log=log) -> tuple[list[dict], dict]:
    """-> (kept records, stats). Records failing verbatim verification are dropped."""
    kept, stats = [], {"returned": 0, "dropped_span": 0, "dropped_shape": 0,
                       "chunks": 0}
    norm_doc = VG.normalise(text)
    for ci, chunk in enumerate(_chunks(text)):
        stats["chunks"] += 1
        prompt = (tpl
                  .replace("{company}", f"{company['name']} ({company['symbol']} / "
                                        f"{company['isin']})")
                  .replace("{doc_id}", doc_id)
                  .replace("{doc_type}", doc_type)
                  .replace("{doc_date}", str(doc_date))
                  .replace("{document}", chunk))
        try:
            raw = pool.call_text(prompt, f"struct_{doc_id}_{ci}",
                                 max_output_tokens=MAX_OUTPUT_TOKENS)
        except Exception as e:
            log(f"      chunk {ci}: call failed — {str(e)[:120]}")
            continue
        recs = _parse(raw)
        stats["returned"] += len(recs)
        for r in recs:
            kind = str(r.get("kind", "")).strip().lower()
            item = str(r.get("item", "")).strip()
            span = str(r.get("evidence_span", "")).strip()
            if kind not in KINDS or not item or not span:
                stats["dropped_shape"] += 1
                continue
            # THE GATE: the span must actually be in the document.
            nspan = VG.normalise(span)
            if nspan not in norm_doc and nspan.lower() not in norm_doc.lower():
                stats["dropped_span"] += 1
                continue
            kept.append({"kind": kind, "item": item[:200],
                         "field": str(r.get("field", ""))[:60],
                         "value": str(r.get("value", ""))[:400],
                         "unit": str(r.get("unit", ""))[:20],
                         "period": str(r.get("period", ""))[:20],
                         "evidence_span": span[:500],
                         "source_doc_id": doc_id, "doc_date": str(doc_date)})
        if ci < len(_chunks(text)) - 1:
            time.sleep(INTER_CALL_SLEEP)
    return kept, stats


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Same (kind,item,field,period) from several documents: keep the newest doc_date.
    Chunk overlap also produces exact duplicates — those collapse here."""
    if df.empty:
        return df
    df = df.sort_values("doc_date", ascending=False)
    return df.drop_duplicates(subset=["isin", "kind", "item", "field", "period"],
                              keep="first")


def save(store: Store, rows: pd.DataFrame, dry_run: bool) -> str:
    drive, root = store.drive, store.root
    fid_folder = store.folder(IDX)
    existing = pd.DataFrame(columns=STRUCT_COLS)
    fid = find_file(drive, fid_folder, STRUCT_FILE)
    if fid:
        try:
            existing = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
        except Exception as e:
            log(f"  WARNING: could not read existing {STRUCT_FILE} ({str(e)[:70]})")
    for c in STRUCT_COLS:
        if c not in existing.columns:
            existing[c] = None
    # replace this company's rows wholesale; other companies untouched
    isins = set(rows["isin"].astype(str)) if not rows.empty else set()
    kept_other = existing[~existing["isin"].astype(str).isin(isins)]
    merged = _dedupe(pd.concat([kept_other, rows], ignore_index=True)[STRUCT_COLS])
    if dry_run:
        return (f"DRY RUN — would write {len(merged)} rows "
                f"({len(rows)} for this run, {len(kept_other)} preserved)")
    buf = io.BytesIO()
    merged.to_parquet(buf, index=False)
    # signature is (drive, folder_id, filename, data, mimetype, existing_id) — see the
    # same bug fixed in extract_mgmt_quotes.py.
    upload_bytes(drive, fid_folder, STRUCT_FILE, buf.getvalue(),
                 "application/octet-stream", existing_id=fid)
    return f"wrote {STRUCT_FILE}: {len(merged)} rows total, {len(rows)} from this run"


def run_one(store: Store, pool, tpl: str, token: str, cache: Path,
            dry_run: bool) -> pd.DataFrame:
    r = resolve(store, token)
    if r is None:
        log(f"  could not resolve '{token}'")
        return pd.DataFrame(columns=STRUCT_COLS)
    isin, symbol, name = r
    company = {"isin": isin, "symbol": symbol, "name": name}
    log(f"  {name} ({symbol})")

    rows = store.by_isin("processing_queue.parquet", isin, symbol)
    rows = rows[rows["doc_type"].astype(str).isin(WANTED)] if not rows.empty else rows
    if rows.empty:
        log(f"    no {'/'.join(WANTED)} documents in the queue — nothing to extract")
        return pd.DataFrame(columns=STRUCT_COLS)
    rows = NS.queue_rows(store, isin, symbol,
                         limits={k: NS.DEFAULT_LIMITS.get(k, 2) for k in WANTED})
    rows = rows[rows["doc_type"].astype(str).isin(WANTED)]

    out, total = [], {"returned": 0, "dropped_span": 0, "dropped_shape": 0}
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
            log(f"    {did}: no text (re-fetch failed or scanned) — skipped")
            continue
        log(f"    {did}: {len(text):,} chars")
        if dry_run:
            log(f"      DRY RUN — would run {len(_chunks(text))} chunk(s), no LLM call")
            continue
        recs, stats = extract_doc(pool, tpl, company, did,
                                 str(row.get("doc_type")),
                                 str(row.get("announcement_date")), text)
        for k in total:
            total[k] += stats.get(k, 0)
        log(f"      {len(recs)} kept of {stats['returned']} returned "
            f"({stats['dropped_span']} failed verbatim check, "
            f"{stats['dropped_shape']} malformed)")
        out += recs

    if not out:
        return pd.DataFrame(columns=STRUCT_COLS)
    df = pd.DataFrame(out)
    df["isin"], df["symbol"], df["company_name"] = isin, symbol, name
    df["extracted_at"] = datetime.now().isoformat(timespec="seconds")
    df = _dedupe(df[STRUCT_COLS])
    by_kind = df.groupby("kind").size().to_dict()
    log(f"    TOTAL {len(df)} records after dedupe: {by_kind}")
    if total["returned"]:
        rate = 100.0 * total["dropped_span"] / total["returned"]
        log(f"    verbatim-check discard rate: {rate:.0f}% "
            f"({total['dropped_span']}/{total['returned']})"
            + ("  <-- HIGH: the model is paraphrasing spans; tighten the prompt"
               if rate > 40 else ""))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="list documents and chunk counts; no LLM calls, no writes")
    ap.add_argument("--cache", default="", help="text cache dir (default: scratch)")
    a = ap.parse_args()

    store = Store()
    tpl = (Path(_HERE) / "narrative_structure_prompt.txt").read_text(encoding="utf-8")
    pool = None
    if not a.dry_run:
        keys = (load_keys(os.environ, prefix="FREE_POOL")
                or load_keys(os.environ, prefix="GEMINI_API_KEY"))
        if not keys:
            print("no Gemini keys — set FREE_POOL_n")
            return 1
        from _extractor_base import GeminiKeyPool
        pool = GeminiKeyPool(keys, MODELS)

    frames = []
    for token in a.names:
        cache = Path(a.cache or f"./_src_{token}")
        frames.append(run_one(store, pool, tpl, token, cache, a.dry_run))
    allrows = (pd.concat(frames, ignore_index=True) if frames
               else pd.DataFrame(columns=STRUCT_COLS))
    if allrows.empty:
        print("no records extracted — nothing written")
        return 0
    print(save(store, allrows, a.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
