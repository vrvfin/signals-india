"""
check_models.py — standalone Gemini model health validator (no Drive, no queue).

Independent of the extractor's Step-A probe: polls each model several rounds over a
short window so you can SEE the live/flap rate yourself, per key-pool.

Usage:
    python scripts/check_models.py                       # BACKFILL pool, 4 rounds
    python scripts/check_models.py --prefix GEMINI_API_KEY --rounds 6
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google import genai
from google.genai import types as gt
from gemini_pool import load_keys
from extract_concall import CONCALL_MODELS
from _extractor_base import P1_MODELS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="BACKFILL_GEMINI_KEY",
                    help="Key-pool env prefix (BACKFILL_GEMINI_KEY | GEMINI_API_KEY).")
    ap.add_argument("--rounds", type=int, default=4, help="Polls per model.")
    ap.add_argument("--gap", type=float, default=5.0, help="Seconds between rounds.")
    args = ap.parse_args()

    keys = load_keys(os.environ, prefix=args.prefix)
    if not keys:
        print(f"No keys for prefix {args.prefix}"); return
    models = CONCALL_MODELS + [m for m in P1_MODELS if m not in CONCALL_MODELS]
    cfg = gt.GenerateContentConfig(max_output_tokens=8)
    ho = gt.HttpOptions(timeout=60000, retry_options=gt.HttpRetryOptions(attempts=1))
    part = [gt.Part.from_text(text="ping")]

    res = {m: [] for m in models}
    for rnd in range(args.rounds):
        for mi, m in enumerate(models):
            c = genai.Client(api_key=keys[(rnd * len(models) + mi) % len(keys)],
                             http_options=ho)
            t = time.time()
            try:
                c.models.generate_content(model=m, contents=part, config=cfg)
                res[m].append("OK")
            except Exception as exc:                       # noqa: BLE001
                s = str(exc)
                res[m].append("503" if ("503" in s or "UNAVAILABLE" in s)
                              else ("429" if "429" in s or "RESOURCE_EXH" in s
                                    else "ERR"))
        if rnd < args.rounds - 1:
            time.sleep(args.gap)

    print(f"\nPool '{args.prefix}' · {len(keys)} keys · {args.rounds} rounds")
    print("-" * 64)
    print(f"{'model':30} {'rounds':<22} live")
    for m in models:
        r = res[m]
        print(f"{m:30} {' '.join(x for x in r):<22} {r.count('OK')}/{len(r)}")
    print("-" * 64)
    healthy = [m for m in models if res[m].count("OK") >= max(1, args.rounds // 2)]
    print(f"Usable now ({len(healthy)}/{len(models)}): {healthy}")


if __name__ == "__main__":
    main()
