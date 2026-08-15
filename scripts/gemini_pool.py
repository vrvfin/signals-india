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
# KEY_DEAD: a per-KEY permanent auth failure (deleted/disabled service account,
# invalid/revoked key). NOT per-document — condemn that key for the run and rotate
# to the next key; only AllBucketsExhausted if EVERY key is dead. The key stays in
# config, so a later run re-probes it (e.g. once the account is re-enabled).
KEY_DEAD = "key_dead"


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
                            "ConnectionAborted", "ConnectionReset", "10053", "10054",
                            "499", "CANCELLED")) \
            or any(t in _low for t in ("connection aborted", "connection reset",
                                       "connection error", "timed out", "timeout",
                                       "read timed out", "temporarily unavailable",
                                       "operation was cancelled", "cancelled",
                                       "500 internal", "502", "504")):
        return OVERLOAD, _retry_delay(s, default=8.0)
    # per-KEY permanent auth failure: deleted/disabled service account, invalid or
    # revoked key. This condemns the KEY (rotate to the next), not the document.
    if any(t in s for t in ("ACCOUNT_STATE_INVALID", "API_KEY_INVALID",
                            "UNAUTHENTICATED", "PERMISSION_DENIED")) \
            or any(t in _low for t in ("service account is deleted or disabled",
                                       "api key not valid", "401", "403")):
        return KEY_DEAD, 0.0
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
        model_overload_keys: int | None = None,  # 503 on this many DISTINCT keys ->
                                        # drop the model for the run. None = scale with
                                        # pool size; see the note in __init__.
        model_fail_drop: int = 10,      # >=N fails of ANY type with 0 ok -> the model is
                                        # dead for the workload; drop it for the run
        key_fail_drop: int = 10,        # >=N fails of ANY type with 0 ok -> the key is
                                        # dead for the run; drop it
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
        # SCALE THE CIRCUIT BREAKER WITH THE POOL (2026-08-15).
        # A fixed 3 was written for a small pool. With 28 keys it condemns a model after
        # 503s on keys 1-3 and the run then has no live bucket left — so keys 4..28 are
        # never tried at all. Measured on the deck backfill: 28 keys loaded, only
        # key_idx 1/2/3 ever appear in gemini_usage, 120 successful calls, then
        # "All Gemini keys rate-limited" with ~140 documents still pending. That message
        # is misleading: nothing was rate-limited, two models had been circuit-broken on
        # three transient 503s each.
        #
        # A 503 is a SERVER-side "high demand" signal, not a property of the key, so
        # sampling more keys before condemning a model is the correct read.
        #
        # Small pools are unchanged: max(3, n // 3) is 3 for anything up to 11 keys, so
        # the Phase-2 concall path keeps exactly today's behaviour.
        self.model_overload_keys = (model_overload_keys if model_overload_keys is not None
                                    else max(3, len(api_keys) // 3))
        self.model_fail_drop = model_fail_drop
        self.key_fail_drop = key_fail_drop
        # Distinct key indices that have hit 503 per model (circuit-breaker signal).
        # A tiny "ping" probe can pass while real PDF calls 503 (model overloaded for
        # the actual workload), so this is driven by REAL calls, not the probe.
        self._model_503_keys: dict[str, set] = {}
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
                # attempts=1 -> the SDK does NOT retry 503/overload internally; it
                # surfaces in ~1-2s so OUR engine does the failover fast (was ~50s
                # of hidden SDK retries inside the timeout, which crippled throughput
                # when a model flapped). timeout still caps a genuinely hung call.
                self._clients[key_idx] = genai.Client(
                    api_key=self.keys[key_idx - 1],
                    http_options=genai_types.HttpOptions(
                        timeout=self._call_timeout_ms,
                        retry_options=genai_types.HttpRetryOptions(attempts=1),
                    ),
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
    def probe_models(self, *, max_keys_per_model: int = 2) -> list[str]:
        """STEP A — pre-flight model health check. Ping each model once and DROP
        (for this run) any that is unresponsive — 503/UNAVAILABLE or not-found — so
        no document later wastes time failing over a dead model, and the chain
        self-heals when a model recovers (no hardcoded removals).

        A per-KEY condition (PerDay/PerMinute/auth) does NOT condemn the model — only
        a model-level outage, confirmed on up to `max_keys_per_model` different keys,
        drops it. Cheap: ~1 call per live model. Returns the dropped model names."""
        cfg = genai_types.GenerateContentConfig(temperature=0, max_output_tokens=8)
        part = [genai_types.Part.from_text(text="ping")]
        dropped: list[str] = []
        for mi, model in enumerate(list(self.models)):
            alive = False
            for kj in range(min(max_keys_per_model, len(self.keys))):
                key_idx = ((mi * max_keys_per_model + kj) % len(self.keys)) + 1
                try:
                    # A non-exception response == the model is up. Do NOT inspect
                    # r.text: a tiny max_output_tokens can finish as MAX_TOKENS with
                    # empty text on a perfectly healthy model (this false-dropped
                    # live models before).
                    self._client(key_idx).models.generate_content(
                        model=model, contents=part, config=cfg)
                    alive = True
                    break
                except Exception as exc:                       # noqa: BLE001
                    kind, _ = classify_error(exc)
                    s = str(exc).lower()
                    model_down = (kind == OVERLOAD) or "404" in s \
                        or "not found" in s or "not supported" in s
                    if not model_down:
                        alive = True   # per-key quota/auth/transient — model is fine
                        break
                    # model-level error -> try the next key to confirm it's really down
            if not alive:
                dropped.append(model)
        if dropped:
            with self._lock:
                self.models = [m for m in self.models if m not in dropped]
                self.buckets = [b for b in self.buckets if b.model not in dropped]
            for m in dropped:
                self._log(f"  STEP A: model '{m}' unresponsive (503/404) — DROPPED for this run")
        self._log(f"  STEP A model health: {len(self.models)} live model(s) {self.models}"
                  + (f" · dropped {dropped}" if dropped else ""))
        if not self.models:
            self._log("  STEP A WARNING: every model failed the probe — nothing live!")
        return dropped

    def prime_dead_buckets(self, pairs) -> int:
        """Phase 1 — pre-mark (key_idx, model) buckets DEAD_TODAY from the persisted
        health cache, so a fresh run does NOT re-discover yesterday-style PerDay deaths by
        burning one real call per dead bucket. Returns how many were marked.

        Safety floor: if priming would leave NO live bucket (e.g. a stale/corrupt cache),
        revive the best-rank bucket per model as a self-test so the run can still probe
        whether quota has actually reset. Additive — nothing calls this unless wired."""
        pairs = {(int(k), str(m)) for k, m in (pairs or set())}
        if not pairs:
            return 0
        with self._lock:
            n = 0
            for b in self.buckets:
                if (b.key_idx, b.model) in pairs and b.state == ALIVE:
                    b.state = DEAD_TODAY
                    n += 1
            if n and not any(b.state == ALIVE for b in self.buckets) and self.buckets:
                seen = set()
                for b in sorted(self.buckets, key=lambda x: (x.model_rank, x.key_idx)):
                    if b.model not in seen:
                        b.state = ALIVE
                        seen.add(b.model)
                self._log(f"  prime: cache would kill ALL buckets — revived "
                          f"{len(seen)} best-rank bucket(s) as a self-test")
            self._log(f"  prime: marked {n} bucket(s) DEAD_TODAY from health cache")
            return n

    def call_pdf(self, pdf_bytes: bytes, prompt: str,
                 max_output_tokens: int | None = None) -> tuple[str, str]:
        """Run prompt over the PDF. Returns (response_text, model_used).

        `max_output_tokens` (default None) bounds the response — used by the AR
        extractor to stop the lite model running away; None = unbounded (unchanged).

        Raises AllBucketsExhausted (transient -> defer row) or FatalCallError
        (deterministic -> mark row error)."""
        b64 = base64.standard_b64encode(pdf_bytes).decode()
        parts = [
            genai_types.Part(inline_data=genai_types.Blob(
                mime_type="application/pdf", data=b64)),
            genai_types.Part.from_text(text=prompt),
        ]
        return self._run(parts, max_output_tokens=max_output_tokens)

    def call_text(self, prompt: str,
                  max_output_tokens: int | None = None) -> tuple[str, str]:
        return self._run([genai_types.Part.from_text(text=prompt)],
                         max_output_tokens=max_output_tokens)

    def _run(self, parts, max_output_tokens: int | None = None) -> tuple[str, str]:
        # Build the generation config once. Default (None) is byte-identical to before.
        _cfg_kw = {"temperature": 0.1}
        if max_output_tokens:
            _cfg_kw["max_output_tokens"] = int(max_output_tokens)
        gen_config = genai_types.GenerateContentConfig(**_cfg_kw)
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
                    config=gen_config,
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
        if kind == KEY_DEAD:
            # per-KEY permanent auth failure — condemn EVERY bucket on this key for
            # the run and rotate. The key stays in config, so a fresh run re-probes it.
            killed = [bb for bb in self.buckets if bb.key_idx == b.key_idx
                      and bb.state == ALIVE]
            for bb in killed:
                bb.state = DEAD_RUN
            self._log(f"  key{b.key_idx}: auth failure ({str(exc)[:80]}) — "
                      f"key DEAD for this run ({len(killed)} bucket(s) dropped), rotating")
        elif kind == PERDAY:
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
            # MODEL CIRCUIT BREAKER — if this model has now 503'd on enough DISTINCT
            # keys, it's overloaded for the real (PDF) workload regardless of what the
            # ping-probe said: drop the ENTIRE model for this run and move on, instead
            # of burning ~25s/503 across all 10 keys one-by-one.
            ks = self._model_503_keys.setdefault(b.model, set())
            ks.add(b.key_idx)
            if len(ks) >= self.model_overload_keys:
                killed = [bb for bb in self.buckets
                          if bb.model == b.model and bb.state == ALIVE]
                for bb in killed:
                    bb.state = DEAD_RUN
                if killed:
                    self._log(f"  CIRCUIT-BREAK: model '{b.model}' 503'd on "
                              f"{len(ks)} keys — dropping all {len(killed)} remaining "
                              f"buckets for this run; failing over to next model.")

        # GENERALIZED DEAD breaker (ANY error type): a model or key that has ONLY failed
        # this run (>= threshold fails, 0 ok) is dead for the workload — park its remaining
        # buckets so we stop retrying it. Catches e.g. gemini-2.0-flash flooding 429s (which
        # the 503-only breaker missed: ~700 wasted fails/day). The 0-ok guard guarantees a
        # productive model/key (some fails but real successes) is NEVER dropped.
        mfail = sum(bb.fail for bb in self.buckets if bb.model == b.model)
        mok = sum(bb.ok for bb in self.buckets if bb.model == b.model)
        if mok == 0 and mfail >= self.model_fail_drop:
            killed = [bb for bb in self.buckets
                      if bb.model == b.model and bb.state == ALIVE]
            for bb in killed:
                bb.state = DEAD_RUN
            if killed:
                self._log(f"  MODEL-DEAD: '{b.model}' {mfail} fails / 0 ok — dropping "
                          f"{len(killed)} bucket(s) for this run.")
        # KEY breaker requires failure across >=2 DISTINCT models (else a single dead
        # model would wrongly condemn the whole key + its still-good models). A genuine
        # dead key fails on everything; a pure auth-dead key is already handled by KEY_DEAD.
        kfail = sum(bb.fail for bb in self.buckets if bb.key_idx == b.key_idx)
        kok = sum(bb.ok for bb in self.buckets if bb.key_idx == b.key_idx)
        kmodels = {bb.model for bb in self.buckets if bb.key_idx == b.key_idx and bb.fail > 0}
        if kok == 0 and kfail >= self.key_fail_drop and len(kmodels) >= 2:
            killed = [bb for bb in self.buckets
                      if bb.key_idx == b.key_idx and bb.state == ALIVE]
            for bb in killed:
                bb.state = DEAD_RUN
            if killed:
                self._log(f"  KEY-DEAD: key{b.key_idx} {kfail} fails / 0 ok across "
                          f"{len(kmodels)} models — dropping {len(killed)} bucket(s) this run.")

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
            # Per-(key, model) attribution so we can SEE who summarised what and why a
            # bucket stopped: state dead_today = PerDay-quota; rpm_cool = PerMinute
            # cooldowns (the known RPM cap); overload_503 = model 503s.
            "buckets": [
                {"key_idx": b.key_idx, "model": b.model, "ok": b.ok, "fail": b.fail,
                 "rpm_cool": b.cool_used, "overload_503": b.overload_used,
                 "state": b.state}
                for b in self.buckets
            ],
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


def load_keys_multi(env: dict, prefixes_csv: str) -> list[str]:
    """Load keys across a comma-separated list of env prefixes, concatenated and
    de-duped (first occurrence wins, order preserved). A single prefix behaves exactly
    like load_keys(env, prefix=...). Backfill uses 'FREE_POOL,BACKFILL_GEMINI_KEY' — a
    missing prefix simply contributes nothing (graceful fallback to whatever IS
    present). Shared by the concall + AR backfill extractors."""
    out: list[str] = []
    for p in (x.strip() for x in str(prefixes_csv).split(",") if x.strip()):
        for k in load_keys(env, prefix=p):
            if k not in out:
                out.append(k)
    return out
