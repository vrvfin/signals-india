r"""
check_altllm_keys.py — verify Groq / Cerebras API keys exist and are VALID.

Both providers are OpenAI-compatible. Validity is checked with a Bearer-auth GET on the
provider's /models endpoint — this does NOT consume any generation quota. Keys are loaded
with the SAME mechanism as the Gemini pool (`gemini_pool.load_keys`), so the secret format
is identical:

  GROQ_API_KEY            (one secret holding 1..N keys, comma/newline/space separated)
  GROQ_API_KEY_1 .. _N    (or numbered secrets — either works, both are picked up)
  CEREBRAS_API_KEY        (same)
  CEREBRAS_API_KEY_1 .. _N

Usage:
  python scripts/check_altllm_keys.py            # check both providers
  python scripts/check_altllm_keys.py --provider groq
Exit code: 0 if (no keys configured) OR (all configured keys valid); 1 if any key invalid.

Importable: `validate_provider(name)` returns [(masked_key, ok, detail), ...] so Phase 2
(the provider integration) can gate on "process only if valid".
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))

import requests
from gemini_pool import load_keys

# provider -> (env prefix, models endpoint). Add a provider here to extend.
PROVIDERS = {
    "groq":     ("GROQ_API_KEY",     "https://api.groq.com/openai/v1/models"),
    "cerebras": ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1/models"),
}


def _mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) >= 4 else "****"


def validate_provider(name: str) -> list[tuple[str, bool, str]]:
    """Return [(masked_key, ok, detail)] for every configured key of `name`.
    ok=True iff the /models endpoint accepts the key (HTTP 200). No quota consumed."""
    prefix, url = PROVIDERS[name]
    keys = load_keys(os.environ, prefix=prefix)
    out: list[tuple[str, bool, str]] = []
    for k in keys:
        try:
            r = requests.get(url, headers={"Authorization": f"Bearer {k}"}, timeout=20)
            if r.status_code == 200:
                try:
                    n = len(r.json().get("data", []))
                    out.append((_mask(k), True, f"valid · {n} models visible"))
                except Exception:
                    out.append((_mask(k), True, "valid (200)"))
            elif r.status_code in (401, 403):
                out.append((_mask(k), False, f"INVALID/unauthorized (HTTP {r.status_code})"))
            else:
                out.append((_mask(k), False, f"unexpected HTTP {r.status_code}: {r.text[:80]}"))
        except Exception as e:
            out.append((_mask(k), False, f"request error: {type(e).__name__}: {str(e)[:80]}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS), default=None,
                    help="check just one provider (default: both)")
    args = ap.parse_args()

    names = [args.provider] if args.provider else list(PROVIDERS)
    any_invalid = False
    for name in names:
        prefix = PROVIDERS[name][0]
        print(f"\n=== {name.upper()}  (secret prefix: {prefix}) ===")
        results = validate_provider(name)
        if not results:
            print(f"  no keys configured — add secret(s) {prefix} or {prefix}_1..{prefix}_N")
            continue
        for masked, ok, detail in results:
            flag = "OK " if ok else "BAD"
            print(f"  [{flag}] {masked}  {detail}")
            any_invalid = any_invalid or (not ok)
        n_ok = sum(1 for _, ok, _ in results if ok)
        print(f"  -> {n_ok}/{len(results)} valid")
    return 1 if any_invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
