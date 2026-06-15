"""
gemini_pool.py — bucket-based Gemini call engine with bounded, error-typed
fallback across (key, model) combinations.

WHY THIS EXISTS
---------------
Free-tier Gemini enforces two stacked quotas, confirmed live from the API's
own quota fields:
  * GenerateRequestsPerMinutePerProjectPerModel-FreeTier   (e.g. 5/min)
  * GenerateRequestsPerDayPerProjectPerModel-FreeTier      (e.g. 20/day)
plus an intermittent 503 UNAVAILABLE ("model experiencing high demand").

Crucially the quota is PER-PROJECT-PER-MODEL. With N independent keys (= N
projects) and M models, there are N*M independent daily buckets. This engine
treats each (key, model) pair as a bucket and walks them in a strict priority
order (best model first, across all keys) with these guarantees:

  Validate-before-escalate
    - 503  -> model is fine, just busy: try SAME model on another key.
    - PerMinute -> this key's minute window is full: try SAME model, other key.
    - PerDay -> only THIS bucket is dead-for-today: try SAME model, other keys.
    - Downgrade to the next (lower-quality) model ONLY when the current model
      is unusable on EVERY key. Quality degrades last.

  Cannot loop (three independent bounds)
    1. DEAD_TODAY is permanent for the run; COOLING has a finite budget. So the
       set of usable buckets only shrinks -> total attempts are bounded by
       buckets * (cooling_budget + overload_budget + 1).
    2. Per-call: one pass over live buckets; if none serve, raise — the caller
       defers the document (status stays pending) and moves on.
    (An optional stage wall-clock cap exists but is OFF by default: concall is
     P0 and must never be cut off while still making progress. Bounds 1-2 alone
     guarantee termination; the GitHub job timeout is the only outer limit.)

  No long sleeps
    - PerDay / PerMinute cost ~milliseconds (rotate to a fresh bucket).
    - 503 gets a short capped backoff only.
    - We sleep-to-recover at most once, bounded, and only when nothing else is
      live and the wait fits inside the stage deadline.

The engine never marks a row 'error' for a quota/transient reason — those raise
AllBucketsExhausted so the caller keeps the row 'pending' for the next run.
Only genuine, deterministic failures (bad PDF, 400, auth) raise FatalCallError.
"""

from __future__ import annotations

import base64
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from google import genai
from google.genai import types as genai_types


# ── exceptions ────────────────────────────────────────────────────────────────

class AllBucketsExhausted(Exception):
    """No live (key, model) bucket can currently serve the request.

    Transient/quota condition — caller should leave the row pending and stop the
    stage (every remaining row faces the same dead buckets)."""


class FatalCallError(Exception):
    """Deterministic, non-retryable failure (bad PDF, 400, auth). Caller marks
    the specific row 'error' and continues with the next row."""


# ── error classification ────────────────────────────────────────────────────────

PERDAY, PERMIN, OVERLOAD, FATAL = "perday", "permin", "overload", "fatal"


def classify_error(exc: Exception) -> tuple[str, float]:
    """Return (kind, retry_after_seconds) from the real exception text."""
    s = str(exc)
    # explicit quota dimension wins
    if "PerDay" in s or "per day" in s.lower():
        return PERDAY, 0.0
    if "PerMinute" in s:
        return PERMIN, _retry_delay(s, default=60.0)
    if "503" in s or "UNAVAILABLE" in s or "high demand" in s:
        return OVERLOAD, 0.0
    # generic 429 with no dimension: treat as a transient minute-class throttle
    if "429" in s or "RESOURCE_EXHAUSTED" in s or "Resource has been exhausted" in s:
        return PERMIN, _retry_delay(s, default=45.0)
    # connection-level failures (server hangup, reset, read timeout, network blip)
    # are TRANSIENT — retry on another bucket, never burn the row as a fatal error.
    # Seen live: "Server disconnected without sending a response" / WinError 10053.
    _low = s.lower()
    if any(t in s for t in ("Server disconnected", "RemoteDisconnected",
                            "ConnectionAborted", "ConnectionReset", "10053", "10054")) \
            or any(t in _low for t in ("connection aborted", "connection reset",
                                       "connection error", "timed out", "timeout",
                                       "read timed out", "temporarily unavailable",
                                       "500 internal", "502", "504")):
        return OVERLOAD, _retry_delay(s, default=8.0)
    return FATAL, 0.0


def _retry_delay(s: str, default: float) -> float:
    m = re.search(r"retry in ([\d.]+)s", s) or re.search(r"retryDelay'?:?\s*'?(\d+)s", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


# ── bucket state ────────────────────────────────────────────────────────────────

ALIVE, DEAD_TODAY, DEAD_RUN = "alive", "dead_today", "dead_run"


@dataclass
class _Bucket:
    key_idx: int          # index into the key list (1-based for logs)
    model: str
    model_rank: int       # 0 = best/preferred
    state: str = ALIVE
    not_before: float = 0.0      # epoch secs; cooling until then
    cool_used: int = 0           # PerMinute cooldowns consumed
    overload_used: int = 0       # 503s consumed
    ok: int = 0
    fail: int = 0
    in_flight: bool = False      # a worker is mid-call on this bucket (parallel mode)
    last_call_ts: float = 0.0    # epoch of this bucket's last successful call (RPM pace)

    @property
    def label(self) -> str:
        return f"key{self.key_idx}/{self.model}"


# ── the pool ──────────────────────────────────────────────────────────────────

class BucketPool:
    def __init__(
        self,
        api_keys: list[str],
        models: list[str],
        *,
        cooling_budget: int = 2,      # PerMinute cooldowns allowed per bucket
        overload_budget: int = 2,     # 503s allowed per bucket
        stage_deadline_s: float | None = None,  # optional wall-clock cap; None = no cap
        inter_call_s: float = 6.0,    # min gap between successful calls (RPM hygiene)
        overload_backoff_s: float = 8.0,
        call_timeout_s: float = 180.0,  # hard per-call HTTP timeout (no infinite hangs)
        logger=print,
    ):
        # NOTE: there is deliberately NO wall-clock cap by default. Termination is
        # already guaranteed by the bucket state machine (DEAD_TODAY is permanent;
        # COOLING has a finite budget), so a clock cap would only stop productive
        # work. Concall (P0) must never be cut off while still making progress —
        # it runs until all rows are done or every free bucket is exhausted, with
        # the GitHub job timeout as the only outer bound. Pass stage_deadline_s
        # explicitly only if a specific caller wants a soft cap.
        self.keys = api_keys
        self.models = models
        self.cooling_budget = cooling_budget
        self.overload_budget = overload_budget
        self.inter_call_s = inter_call_s
        self.overload_backoff_s = overload_backoff_s
        # google-genai HttpOptions.timeout is in MILLISECONDS. A hard ceiling so a
        # stalled connection raises (transient -> retried) instead of hanging the
        # whole run. Critical under --workers: one stuck call would otherwise block
        # consumption of every completed result behind it (head-of-line stall).
        self._call_timeout_ms = int(call_timeout_s * 1000)
        self._log = logger
        self._last_call_ts = 0.0
        # Guards all mutable bucket/pool state so K worker threads can share one
        # pool (T12 #1). RLock: _client() may be called while the lock is held.
        # The slow generate_content() network call happens OUTSIDE this lock, so
        # workers genuinely run in parallel; only selection + bookkeeping is
        # serialized. In the default single-worker path the lock is uncontended,
        # so behaviour is identical to the original sequential engine.
        self._lock = threading.RLock()
        self._started = time.time()
        self._deadline = (self._started + stage_deadline_s
                          if stage_deadline_s else None)
        self._clients: dict[int, genai.Client] = {}
        # buckets ordered best-model-first, then by key
        self.buckets: list[_Bucket] = [
            _Bucket(key_idx=ki + 1, model=m, model_rank=rank)
            for rank, m in enumerate(models)
            for ki in range(len(api_keys))
        ]

    # -- client cache --
    def _client(self, key_idx: int) -> genai.Client:
        # RLock-guarded: safe to call while already holding self._lock.
        with self._lock:
            if key_idx not in self._clients:
                self._clients[key_idx] = genai.Client(
                    api_key=self.keys[key_idx - 1],
                    http_options=genai_types.HttpOptions(timeout=self._call_timeout_ms),
                )
            return self._clients[key_idx]

    # -- bucket selection --
    def _live(self, b: _Bucket, now: float) -> bool:
        # A bucket already serving a worker (in_flight) is not selectable, so two
        # workers never hammer the same (key, model) inside its RPM window.
        return b.state == ALIVE and not b.in_flight and b.not_before <= now

    def _any_nonterminal(self) -> bool:
        return any(b.state == ALIVE for b in self.buckets)

    def _any_in_flight(self) -> bool:
        return any(b.in_flight for b in self.buckets)

    def _next_bucket(self, now: float) -> _Bucket | None:
        """Lowest (model_rank, key_idx) bucket that is live right now."""
        live = [b for b in self.buckets if self._live(b, now)]
        if not live:
            return None
        live.sort(key=lambda b: (b.model_rank, b.key_idx))
        return live[0]

    def _earliest_wakeup(self, now: float) -> float | None:
        """Soonest not_before among ALIVE-but-cooling buckets, else None."""
        future = [b.not_before for b in self.buckets
                  if b.state == ALIVE and b.not_before > now]
        return min(future) if future else None

    # -- public API --
    def call_pdf(self, pdf_bytes: bytes, prompt: str) -> tuple[str, str]:
        """Run prompt over the PDF. Returns (response_text, model_used).

        Raises AllBucketsExhausted (transient -> defer row) or FatalCallError
        (deterministic -> mark row error)."""
        b64 = base64.standard_b64encode(pdf_bytes).decode()
        parts = [
            genai_types.Part(inline_data=genai_types.Blob(
                mime_type="application/pdf", data=b64)),
            genai_types.Part.from_text(text=prompt),
        ]
        return self._run(parts)

    def call_text(self, prompt: str) -> tuple[str, str]:
        return self._run([genai_types.Part.from_text(text=prompt)])

    def _run(self, parts) -> tuple[str, str]:
        # Thread-safe call engine. Selection + bookkeeping run under self._lock;
        # the slow generate_content() network call runs OUTSIDE the lock so K
        # workers overlap. RPM hygiene is now per-bucket (last_call_ts) instead of
        # one global gate — correct under parallelism and equivalent for one
        # worker (a single thread rotates buckets and rarely re-hits one inside
        # inter_call_s anyway).
        while True:
            b = None
            client = None
            nap = 0.0
            with self._lock:
                now = time.time()
                if self._deadline is not None and now >= self._deadline:
                    raise AllBucketsExhausted("stage wall-clock ceiling reached")

                b = self._next_bucket(now)
                if b is not None:
                    # per-bucket RPM pace: leave inter_call_s between this bucket's
                    # own successful calls.
                    gap = now - b.last_call_ts
                    if b.last_call_ts and gap < self.inter_call_s:
                        b.not_before = max(b.not_before,
                                           b.last_call_ts + self.inter_call_s)
                        b = None            # cooling — fall through to wait/retry
                    else:
                        b.in_flight = True
                        client = self._client(b.key_idx)

                if b is None:
                    # nothing selectable now — wait for the soonest wakeup, unless
                    # an in-flight worker may free/refresh a bucket first.
                    wake = self._earliest_wakeup(now)
                    if wake is None or (self._deadline is not None and wake >= self._deadline):
                        if not self._any_in_flight():
                            raise AllBucketsExhausted(self._state_summary())
                        nap = 0.25          # let an in-flight call complete
                    else:
                        nap = max(0.0, wake - now)
                        if self._any_in_flight():
                            nap = min(nap, 0.5)   # re-check completions promptly

            # ---- outside the lock ----
            if b is None:
                if nap > 0:
                    time.sleep(nap)
                continue

            self._log(f"  calling {b.label} ...")
            try:
                resp = client.models.generate_content(
                    model=b.model,
                    contents=parts,
                    config=genai_types.GenerateContentConfig(temperature=0.1),
                )
                text = resp.text
            except Exception as exc:
                with self._lock:
                    b.in_flight = False
                    b.fail += 1
                    kind, retry_after = classify_error(exc)
                    self._apply_failure(b, kind, retry_after, exc)
                if kind == FATAL:
                    raise FatalCallError(str(exc)[:300])
                continue   # loop: pick the next live bucket
            if not text:
                # empty/blocked response is deterministic for this doc
                with self._lock:
                    b.in_flight = False
                    b.fail += 1
                raise FatalCallError(f"empty response from {b.label}")
            with self._lock:
                b.in_flight = False
                b.ok += 1
                b.last_call_ts = time.time()
                self._last_call_ts = b.last_call_ts
            return text, b.model

    def _apply_failure(self, b: _Bucket, kind: str, retry_after: float, exc: Exception):
        now = time.time()
        if kind == PERDAY:
            b.state = DEAD_TODAY
            self._log(f"  {b.label}: PerDay exhausted — dead until reset (~13:30 IST)")
        elif kind == PERMIN:
            b.cool_used += 1
            if b.cool_used > self.cooling_budget:
                b.state = DEAD_RUN
                self._log(f"  {b.label}: PerMinute budget spent — parked for this run")
            else:
                b.not_before = now + retry_after
                self._log(f"  {b.label}: PerMinute — cooling {retry_after:.0f}s "
                          f"({b.cool_used}/{self.cooling_budget})")
        elif kind == OVERLOAD:
            b.overload_used += 1
            if b.overload_used > self.overload_budget:
                b.state = DEAD_RUN
                self._log(f"  {b.label}: 503 budget spent — parked for this run")
            else:
                b.not_before = now + self.overload_backoff_s
                self._log(f"  {b.label}: 503 overload — backoff {self.overload_backoff_s:.0f}s "
                          f"({b.overload_used}/{self.overload_budget})")

    # -- reporting --
    def _state_summary(self) -> str:
        from collections import Counter
        c = Counter(b.state for b in self.buckets)
        return (f"all buckets exhausted "
                f"(dead_today={c.get(DEAD_TODAY,0)}, parked={c.get(DEAD_RUN,0)}, "
                f"alive={c.get(ALIVE,0)})")

    def summary(self) -> dict:
        from collections import Counter
        states = Counter(b.state for b in self.buckets)
        ok = sum(b.ok for b in self.buckets)
        by_model = {}
        for b in self.buckets:
            if b.ok:
                by_model[b.model] = by_model.get(b.model, 0) + b.ok
        return {
            "calls_ok": ok,
            "by_model": by_model,
            "buckets_total": len(self.buckets),
            "buckets_dead_today": states.get(DEAD_TODAY, 0),
            "buckets_parked": states.get(DEAD_RUN, 0),
            "buckets_alive": states.get(ALIVE, 0),
            "elapsed_s": round(time.time() - self._started, 1),
        }


def load_keys(env: dict, prefix: str = "GEMINI_API_KEY") -> list[str]:
    """Collect <prefix>_<n> (sorted) + plain <prefix>, de-duped.

    Default prefix preserves the original Phase 2 behaviour (GEMINI_API_KEY_<n>
    + GEMINI_API_KEY). Pass prefix="BACKFILL_GEMINI_KEY" to load the dedicated
    backfill pool (separate Cloud projects) so its quota is independent of Phase 2.
    """
    num_re = re.compile(rf"{re.escape(prefix)}_\d+$")
    raw = [v for _, v in sorted(
        ((k, v) for k, v in env.items()
         if num_re.match(k) and v.strip()),
        key=lambda kv: kv[0])]
    plain = env.get(prefix, "").strip()
    if plain:
        raw.append(plain)
    # A single var may hold MANY keys separated by comma, semicolon, newline or
    # spaces (one git secret with 8 keys = the common case). Gemini keys contain
    # none of those, so splitting on any separator is safe. The secret may also
    # be pasted as .env-style lines ("BACKFILL_GEMINI_KEY_1=AIza..."): strip the
    # NAME= prefix and drop bare NAME tokens (real keys always have lowercase,
    # never '='). De-dupe, keep order.
    keys: list[str] = []
    for v in raw:
        for part in re.split(r"[,\s;]+", v):
            p = part.strip().strip('"').strip("'")
            if "=" in p:
                p = p.split("=")[-1].strip().strip('"').strip("'")
            if not p or re.fullmatch(r"[A-Z0-9_]+", p):
                continue  # empty, or an env-var NAME fragment — not a key
            if p not in keys:
                keys.append(p)
    return keys
