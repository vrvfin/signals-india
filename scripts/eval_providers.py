r"""
eval_providers.py — measure extraction + summary quality across FREE tiers on REAL docs.

Pulls a live concall / annual-report / rating doc from Screener, extracts text, then runs
the SAME production prompt through Gemini (baseline) and the Groq/Cerebras free models, so
we compare apples-to-apples (identical text input) on two task shapes:
  • SUMMARY  — concall_prompt.txt, annual_report_prompt.txt  (narrative quality)
  • JSON     — rating_structured_prompt.txt                  (structured-extraction discipline)

It does NOT touch the live extractors. Outputs raw responses to scripts/_eval_out/ for
manual side-by-side reading, and prints a metrics table (latency, output size, JSON yield).

Also: --parallel-test times sequential vs threaded alt-provider calls (the Phase-3 enabler).

Usage:
  python scripts/eval_providers.py
  python scripts/eval_providers.py --skip-gemini      # save Gemini quota
  python scripts/eval_providers.py --parallel-test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))

import fitz  # PyMuPDF
from ingest_company_docs import screener_session
from backfill_company_docs import fetch_company_page, parse_company_documents, fetch_document
from _extractor_base import salvage_json_objects
from gemini_pool import BucketPool, load_keys_multi
from altllm_pool import AltPool, AltLLMError

MAX_INPUT_CHARS = 80_000          # fair, fits every model's context (~20k tokens)
MAX_OUT_TOKENS = 4096
OUT_DIR = os.path.join(_HERE, "_eval_out")

# doc_type -> (prompt file, task label). rating uses the STRUCTURED (JSON) prompt.
TASKS = {
    "concall":       ("concall_prompt.txt",          "summary"),
    "annual_report": ("annual_report_prompt.txt",    "summary"),
    "rating":        ("rating_structured_prompt.txt", "json"),
}
CANDIDATES = ["TATAMOTORS", "RELIANCE", "LT", "TCS", "HDFCBANK"]

# (label, provider, model) — provider 'gemini' handled specially
ALT_MODELS = [
    ("groq:gpt-oss-120b",   "groq",     "openai/gpt-oss-120b"),
    ("groq:llama-3.3-70b",  "groq",     "llama-3.3-70b-versatile"),
    ("cerebras:gpt-oss-120b", "cerebras", "gpt-oss-120b"),
    ("cerebras:glm-4.7",    "cerebras", "zai-glm-4.7"),
]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3.1-flash-lite"]


def log(m): print(f"[{dt.datetime.now():%H:%M:%S}] {m}")


def _text_from(data: bytes, mime: str) -> str:
    if mime == "application/pdf":
        d = fitz.open(stream=data, filetype="pdf")
        try:
            return "\n".join(p.get_text() for p in d)
        finally:
            d.close()
    return data.decode("utf-8", "ignore")


def fetch_one_doc(session, doc_type: str, symbol_override: str | None):
    """Return (symbol, title, text) for the newest doc of `doc_type`, walking candidates."""
    cands = [symbol_override] if symbol_override else CANDIDATES
    for sym in cands:
        html = fetch_company_page(session, sym)
        if not html:
            continue
        docs = parse_company_documents(html, dt.date.today(), {doc_type})
        docs = [d for d in docs if d["doc_type"] == doc_type]
        docs.sort(key=lambda d: str(d["announcement_date"]), reverse=True)
        for d in docs:
            fetched = fetch_document(session, d["pdf_url"])
            if not fetched:
                continue
            data, mime, _ = fetched
            text = _text_from(data, mime)
            if len(text) >= 400:
                return sym, d["title"], text[:MAX_INPUT_CHARS]
    return None, None, None


def run_quality(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    session = screener_session()
    gpool = None
    if not args.skip_gemini:
        gkeys = load_keys_multi(os.environ, "FREE_POOL,BACKFILL_GEMINI_KEY")
        gpool = BucketPool(gkeys, GEMINI_MODELS) if gkeys else None
        log(f"Gemini pool: {len(gkeys)} keys" if gkeys else "Gemini: NO keys — skipping")
    pools = {p: AltPool(p, os.environ) for p in ("groq", "cerebras")}

    rows = []
    for doc_type, (prompt_file, task) in TASKS.items():
        prompt = open(os.path.join(_HERE, prompt_file), encoding="utf-8").read()
        log(f"=== fetching a {doc_type} doc ===")
        sym, title, text = fetch_one_doc(session, doc_type, args.symbol)
        if not text:
            log(f"  could not get a {doc_type} doc — skipping")
            continue
        log(f"  using {sym}: {title[:50]!r} ({len(text)} chars, task={task})")
        full = f"{prompt}\n\n---DOCUMENT TEXT---\n{text}"

        configs = list(ALT_MODELS)
        if gpool is not None:
            configs = [("gemini", "gemini", "|".join(GEMINI_MODELS))] + configs

        for label, provider, model in configs:
            t0 = time.time()
            status, out = "ok", ""
            try:
                if provider == "gemini":
                    out, used = gpool.call_text(full, max_output_tokens=MAX_OUT_TOKENS)
                    label = f"gemini:{used}"
                else:
                    out = pools[provider].call_text(full, model,
                                                    max_output_tokens=MAX_OUT_TOKENS)
            except (AltLLMError, Exception) as e:
                status, out = f"ERR {type(e).__name__}", str(e)[:200]
            secs = round(time.time() - t0, 1)
            njson = len(salvage_json_objects(out)) if status == "ok" else 0
            fn = f"{doc_type}__{label.replace(':','_').replace('/','_')}.txt"
            try:
                with open(os.path.join(OUT_DIR, fn), "w", encoding="utf-8") as fh:
                    fh.write(out)
            except Exception:
                pass
            rows.append((doc_type, task, label, secs, len(out), njson, status))
            log(f"    {label:<28} {secs:>5}s  {len(out):>6} chars  "
                f"{njson:>3} json  [{status}]")

    print("\n" + "=" * 92)
    print(f"{'doc_type':<14}{'task':<8}{'model':<26}{'secs':>6}{'chars':>8}{'json':>6}  status")
    print("-" * 92)
    for dt_, task, label, secs, ch, nj, st in rows:
        print(f"{dt_:<14}{task:<8}{label:<26}{secs:>6}{ch:>8}{nj:>6}  {st}")
    print("=" * 92)
    print(f"Raw outputs saved to {OUT_DIR} — read side-by-side to judge quality.")


def run_parallel_test(args):
    """Sequential vs threaded timing on one alt model — proves the alt path is
    parallel-safe (plain HTTP, no grpc), the Phase-3 throughput enabler."""
    pool = AltPool("groq", os.environ)
    model = "llama-3.3-70b-versatile"
    prompt = "Summarise in one sentence: Indian equities had a strong quarter on earnings."
    n, workers = 8, 4

    log(f"Parallel test: {n} calls to groq:{model}, sequential then {workers}-threaded")
    t0 = time.time()
    for _ in range(n):
        pool.call_text(prompt, model, max_output_tokens=64)
    seq = time.time() - t0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda _: pool.call_text(prompt, model, max_output_tokens=64), range(n)))
    par = time.time() - t0

    print("\n" + "=" * 60)
    print(f"  sequential ({n} calls): {seq:6.1f}s  ({seq/n:.2f}s/call)")
    print(f"  {workers}-threaded ({n} calls): {par:6.1f}s  ({par/n:.2f}s/call)")
    print(f"  speedup: {seq/par:.1f}x   (no errors = thread-safe)")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, help="force a specific Screener symbol")
    ap.add_argument("--skip-gemini", action="store_true", help="skip Gemini baseline (save quota)")
    ap.add_argument("--parallel-test", action="store_true", help="run only the parallel timing test")
    args = ap.parse_args()
    if args.parallel_test:
        run_parallel_test(args)
    else:
        run_quality(args)
        run_parallel_test(args)


if __name__ == "__main__":
    main()
