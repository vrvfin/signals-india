r"""
altllm_pool.py — minimal OpenAI-compatible pool for Groq / Cerebras (TEXT-only).

Both providers speak the OpenAI /chat/completions API, so one tiny requests-based client
covers both (no new pip deps, no provider SDKs). Keys load via the SAME mechanism as the
Gemini pool (`gemini_pool.load_keys`), so secret format is identical:
  GROQ_API_KEY / GROQ_API_KEY_1..N      CEREBRAS_API_KEY / CEREBRAS_API_KEY_1..N

These models have NO native PDF input — callers must pass extracted text. Unlike Gemini's
grpc client, this is plain HTTP, so it is thread-safe and safe to run in parallel.

This module is standalone — it does NOT touch the live Gemini extractors. It is used by the
eval harness now, and will back the Phase-2 provider router once routing thresholds are set.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import requests
from gemini_pool import load_keys

PROVIDERS = {
    "groq":     {"prefix": "GROQ_API_KEY",     "base": "https://api.groq.com/openai/v1"},
    "cerebras": {"prefix": "CEREBRAS_API_KEY", "base": "https://api.cerebras.ai/v1"},
}

# transient HTTP statuses worth rotating to another key / retrying
_TRANSIENT = {429, 500, 502, 503, 504}


class AltLLMError(Exception):
    """All attempts failed (transient) or a deterministic 4xx (bad request/auth)."""


class AltPool:
    def __init__(self, provider: str, env: dict | None = None, *, timeout: float = 180.0):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.base = cfg["base"]
        self.keys = load_keys(env if env is not None else os.environ, prefix=cfg["prefix"])
        self.timeout = timeout
        self._i = 0

    def call_text(self, prompt: str, model: str, *,
                  max_output_tokens: int | None = None,
                  temperature: float = 0.1) -> str:
        """Run `prompt` on `model`, rotating keys on transient errors. Returns the
        response text. Raises AltLLMError if every attempt fails."""
        if not self.keys:
            raise AltLLMError(f"no {self.provider} keys configured")
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_output_tokens:
            body["max_tokens"] = int(max_output_tokens)
        attempts = max(len(self.keys) * 2, 4)
        last = ""
        for _ in range(attempts):
            key = self.keys[self._i % len(self.keys)]
            self._i += 1
            try:
                r = requests.post(
                    f"{self.base}/chat/completions",
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(0.8)
                continue
            if r.status_code == 200:
                try:
                    msg = r.json()["choices"][0]["message"]
                except Exception as e:                       # malformed 200
                    raise AltLLMError(f"bad 200 body: {str(e)[:120]}")
                # reasoning models (gpt-oss, glm) may put the answer in content, or leave
                # content empty and use reasoning/reasoning_content — accept whichever has text.
                return (msg.get("content")
                        or msg.get("reasoning_content")
                        or msg.get("reasoning") or "")
            if r.status_code in _TRANSIENT:
                last = f"HTTP {r.status_code}"
                time.sleep(1.0)
                continue
            # deterministic (400 bad request, 401 auth, 404 model) — don't retry
            raise AltLLMError(f"HTTP {r.status_code}: {r.text[:200]}")
        raise AltLLMError(f"all {self.provider} attempts failed: {last}")
