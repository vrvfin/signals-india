r"""
provider_router.py — Phase 2: Gemini-primary extraction with Groq/Cerebras fallback.

Gemini is tried FIRST on every call (quality + "Gemini primary"). ONLY when Gemini is
fully exhausted (RateLimitExhausted = every (key, model) bucket dead) does it fall back to
Groq/Cerebras on the document's extracted TEXT — turning would-be-DEFERRED rows into done
using independent free quota. This is strictly additive throughput: the fallback can never
replace a Gemini result, only rescue one Gemini couldn't serve.

`FallbackPool` implements the GeminiKeyPool call surface (call / call_text / summary /
prime_from_health), so it is a drop-in for call_over_doc / run_structured_over_doc — the
AR and rating extractors need a one-line construction change only.

Eval-driven routing (2026-06-25): Groq free tier caps ~8k tokens/min → small docs only
(big prompts 413); Cerebras gpt-oss-120b handles long text. Cerebras glm-4.7 is excluded
(non-standard reasoning response shape). Alt providers are TEXT-only (no PDF/scanned).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _extractor_base import GeminiKeyPool, RateLimitExhausted, log
from altllm_pool import AltPool, AltLLMError

# Route by prompt size. Groq's ~8k-tokens/min free cap → only small docs; big prompts go
# straight to Cerebras gpt-oss-120b (proven on whole concalls/ARs in the eval).
_SMALL_CHARS = 24_000   # ≈ 6k tokens
# Cap text sent to alt providers — eval showed ~80k chars (~20k tok) is safe on Cerebras;
# bigger requests 429 on the free tier. Gemini (primary) still sees the full doc; only the
# fallback trims the doc TAIL (instruction stays at the front of the prompt).
_ALT_MAX_CHARS = 80_000
_ALT_SMALL = [("groq", "openai/gpt-oss-120b"),
              ("groq", "llama-3.3-70b-versatile"),
              ("cerebras", "gpt-oss-120b")]
_ALT_LARGE = [("cerebras", "gpt-oss-120b")]


def _pdf_text(pdf_bytes: bytes, max_chars: int = 120_000) -> str:
    """Best-effort text from a PDF for the text-only alt providers. '' if scanned/no text."""
    try:
        import fitz
        d = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            return "\n".join(p.get_text() for p in d)[:max_chars]
        finally:
            d.close()
    except Exception as e:
        log(f"  [fallback] PDF text extraction failed ({type(e).__name__}) — defer to Gemini")
        return ""


class FallbackPool:
    """Gemini-primary pool with Groq/Cerebras text fallback on exhaustion."""

    def __init__(self, primary: GeminiKeyPool, alt_pools: dict):
        self.primary = primary
        self.alt_pools = alt_pools          # {"groq": AltPool, "cerebras": AltPool}
        self._alt_ok: dict[str, int] = {}
        self._alt_fail: dict[str, int] = {}

    # -- alt fallback over already-assembled TEXT prompt --
    def _alt_call(self, prompt: str, max_output_tokens: int | None) -> str:
        order = _ALT_SMALL if len(prompt) <= _SMALL_CHARS else _ALT_LARGE
        if len(prompt) > _ALT_MAX_CHARS:        # keep instruction (front), trim doc tail
            prompt = prompt[:_ALT_MAX_CHARS]
        last = "no alt provider available"
        for prov, model in order:
            pool = self.alt_pools.get(prov)
            if not pool or not pool.keys:
                continue
            key = f"{prov}:{model}"
            try:
                out = pool.call_text(prompt, model, max_output_tokens=max_output_tokens)
                if out and out.strip():
                    self._alt_ok[key] = self._alt_ok.get(key, 0) + 1
                    log(f"  [fallback] {key} handled doc ({len(out)} chars)")
                    return out
                last = f"{key}: empty response"
            except AltLLMError as e:
                self._alt_fail[key] = self._alt_fail.get(key, 0) + 1
                last = f"{key}: {str(e)[:80]}"
        # Gemini already exhausted AND no alt could serve → defer (row stays pending).
        raise RateLimitExhausted(f"Gemini exhausted + alt fallback failed ({last})")

    # -- GeminiKeyPool-compatible surface --
    def call(self, pdf_bytes: bytes, prompt: str, display_name: str,
             max_output_tokens: int | None = None) -> str:
        try:
            return self.primary.call(pdf_bytes, prompt, display_name,
                                     max_output_tokens=max_output_tokens)
        except RateLimitExhausted:
            text = _pdf_text(pdf_bytes)
            if not text.strip():
                raise        # scanned/no-text PDF — only Gemini vision can read it
            return self._alt_call(prompt + "\n\nDOCUMENT:\n" + text, max_output_tokens)

    def call_text(self, prompt: str, display_name: str = "",
                  max_output_tokens: int | None = None) -> str:
        try:
            return self.primary.call_text(prompt, display_name,
                                          max_output_tokens=max_output_tokens)
        except RateLimitExhausted:
            # prompt already carries the document text (call_over_doc embeds it)
            return self._alt_call(prompt, max_output_tokens)

    def summary(self) -> dict:
        s = self.primary.summary()
        buckets = list(s.get("buckets", []))
        # log alt contribution as synthetic buckets (key_idx=0, model="prov:model",
        # state="alt") so persist_gemini_usage shows exactly what the fallback did.
        for key in set(self._alt_ok) | set(self._alt_fail):
            buckets.append({"key_idx": 0, "model": key,
                            "ok": self._alt_ok.get(key, 0),
                            "fail": self._alt_fail.get(key, 0),
                            "rpm_cool": 0, "overload_503": 0, "state": "alt"})
        s["buckets"] = buckets
        s["alt_ok"] = sum(self._alt_ok.values())
        return s

    def prime_from_health(self, drive, index_id: str) -> None:
        self.primary.prime_from_health(drive, index_id)


def build_alt_pools(env: dict | None = None) -> dict:
    """{'groq': AltPool, 'cerebras': AltPool} for whichever providers have keys."""
    env = env if env is not None else os.environ
    pools: dict[str, AltPool] = {}
    for prov in ("groq", "cerebras"):
        try:
            p = AltPool(prov, env)
            if p.keys:
                pools[prov] = p
        except Exception:
            pass
    return pools


def make_extraction_pool(api_keys: list[str], models, *,
                         enable_fallback: bool, env: dict | None = None):
    """A plain GeminiKeyPool, or a FallbackPool wrapping it when fallback is enabled AND
    alt keys exist. Pass enable_fallback=True ONLY on the BACKFILL path (PF/Phase-2 stays
    pure Gemini)."""
    primary = GeminiKeyPool(api_keys, models)
    if not enable_fallback:
        return primary
    alt = build_alt_pools(env)
    if not alt:
        return primary
    log(f"  Phase 2: Gemini-primary + fallback {sorted(alt)} "
        f"(alt used ONLY when Gemini quota is exhausted)")
    return FallbackPool(primary, alt)
