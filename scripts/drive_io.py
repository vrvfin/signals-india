"""Drive I/O hardening for plain scripts: retry + a content-addressed parquet cache.

WHY THIS EXISTS
---------------
app.py has a `_drive_call` retry wrapper, but it is Streamlit-coupled
(`st.cache_resource` / `drive_service.clear()`) and a plain script cannot import
it. So scripts/ had NO Drive retry at all. On 2026-08-25 a momentary DNS failure

    socket.gaierror: [Errno 11001] getaddrinfo failed
    httplib2.error.ServerNotFoundError: Unable to find the server at www.googleapis.com

killed build_gallery.py one statement AFTER 45 minutes of OHLCV downloads had
finished; the result lived only in memory, so the whole build was lost and
gallery.html silently stayed a day stale.

Note the subtlety that made the crash possible: ServerNotFoundError is NOT an
OSError subclass (MRO: ServerNotFoundError -> HttpLib2Error -> Exception), so the
OSError-based transient tuple used elsewhere in this repo would not have caught
it. Same for google.auth's TransportError. Both are listed explicitly below.

This module is OPT-IN. `_extractor_base.py` is deliberately NOT touched, so the
live Phase-2 / backfill path keeps byte-identical behaviour (CLAUDE.md rules 3+5).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import sys
import shutil
import socket
import ssl
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:                                    # optional at import time
    from googleapiclient.errors import HttpError
except ImportError:                     # pragma: no cover
    HttpError = None

_EXTRA_ERR: tuple = ()
try:
    import httplib2
    _EXTRA_ERR += (httplib2.HttpLib2Error,)     # ServerNotFoundError lives here
except ImportError:                     # pragma: no cover
    pass
try:
    from google.auth.exceptions import TransportError
    _EXTRA_ERR += (TransportError,)             # also not an OSError
except ImportError:                     # pragma: no cover
    pass

# socket.error / ConnectionError are OSError aliases; listed to mirror the
# existing convention in daily_research_summary.py and to stay self-documenting.
_TRANSIENT: tuple = (
    ConnectionAbortedError, ConnectionResetError, ConnectionError,
    socket.error, ssl.SSLError, TimeoutError, OSError, BrokenPipeError,
) + _EXTRA_ERR

# Unattended local batch, not an interactive app: a Wi-Fi reconnect or DNS
# hiccup routinely outlasts app.py's 9s total, so wait meaningfully longer.
_BACKOFF = [2, 5, 15, 30]

_DEFAULT_CACHE_ROOT = (Path(r"D:\EMA_Screener\Reports\signals-india\.cache")
                       / "gallery_parquet")


# Console encoding. Several scripts here log the rupee sign, a delta or an em
# dash, and a Windows console is cp1252 — so a run could complete all its work
# and then die in a log line. It cost three separate crashes before being fixed
# in one place. Degrade the characters, never the run.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:          # pragma: no cover - not every stream supports it
    pass

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------- retry ----

# A 403 is normally permanent (no permission, file not shared). Google's
# per-minute quota is the exception: "Quota exceeded for quota metric 'Total
# Query Cost' and limit 'Units per minute per user'" arrives as a 403 with reason
# rateLimitExceeded, and it clears when the minute rolls over. 2026-10-05 it
# dropped 536 stocks from features; 2026-11-09 it killed PEAD's folder lookup and
# aggregate_signals' download, and Phase 1 went red. Decided by the REASON in the
# error body, not the status: dailyLimitExceeded, storageQuotaExceeded and
# insufficientPermissions are 403s too, and waiting fixes none of them.
_RATE_LIMIT_REASONS = {"rateLimitExceeded", "userRateLimitExceeded"}


def _http_reasons(exc) -> set:
    """The errors[].reason values of a googleapiclient HttpError."""
    reasons = set()
    try:
        body = json.loads((getattr(exc, "content", b"") or b"").decode("utf-8", "ignore"))
        err = body.get("error", {}) if isinstance(body, dict) else {}
        for item in (err.get("errors") or []) if isinstance(err, dict) else []:
            if isinstance(item, dict) and item.get("reason"):
                reasons.add(str(item["reason"]))
    except Exception:                                # noqa: BLE001 - not JSON
        pass
    if not reasons:   # no JSON body: the "Details" text of str(e) still names it
        reasons.update(re.findall(r"'reason': '(\w+)'", str(exc)))
    return reasons


def _classify(exc) -> tuple[bool, str]:
    """-> (is_transient, short_reason). Permanent errors fail fast: retrying a
    401/403/404 just burns the backoff and hides a real problem. The one 403
    that is retried is Google's per-minute rate limit (see _RATE_LIMIT_REASONS)."""
    if HttpError is not None and isinstance(exc, HttpError):
        status = getattr(getattr(exc, "resp", None), "status", None)
        try:
            status = int(status)
        except (TypeError, ValueError):
            return False, "HTTP ?"
        if 500 <= status < 600 or status == 429:
            return True, f"HTTP {status}"
        if status == 403 and _http_reasons(exc) & _RATE_LIMIT_REASONS:
            return True, "HTTP 403 rate limit"
        return False, f"HTTP {status}"
    if isinstance(exc, _TRANSIENT):
        return True, type(exc).__name__
    return False, type(exc).__name__


def is_transient(exc) -> bool:
    """True when drive_call would have retried this error."""
    return _classify(exc)[0]


def drive_call(fn, attempts: int = 5, on_reconnect=None, label: str = ""):
    """Run a Drive operation, retrying transient network failures.

    fn           zero-arg callable performing the Drive work.
    on_reconnect optional zero-arg callable invoked before each retry so the
                 caller can rebuild a poisoned connection (the script-side
                 equivalent of the `drive_service.clear()` in app.py).
    Permanent errors (auth, 404, bad request) re-raise immediately.
    """
    last = attempts - 1
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:                       # noqa: BLE001 - reclassified
            transient, why = _classify(e)
            if not transient or i == last:
                raise
            wait = _BACKOFF[min(i, len(_BACKOFF) - 1)]
            tag = f" [{label}]" if label else ""
            log(f"  drive retry {i + 1}/{last}{tag}: {why} - waiting {wait}s")
            if on_reconnect is not None:
                try:
                    on_reconnect()
                except Exception as re_err:          # reconnect is best-effort
                    log(f"  reconnect failed ({type(re_err).__name__}) - retrying anyway")
            time.sleep(wait)


# Phase 1's Drive lookups and reads. 2026-11-06 Drive answered every files.list
# with a 500 after ~30 s for ~7.5 minutes (11:22-11:29:30 UTC). PEAD's first
# lookup came at 11:22:55, so it needed ~7 minutes of patience; drive_call's
# default 5 tries gives up after ~3.4 min of 30 s hangs. 10 tries wait
# 2+5+15+30x6 = 202 s: ~3.4 min when Drive fails fast, ~8.4 min when every call
# hangs 30 s as it did that day.
LOOKUP_ATTEMPTS = 10


def retrying(fn, attempts: int = LOOKUP_ATTEMPTS):
    """Wrap a READ-ONLY Drive helper (find_file, get_or_create_subfolder,
    list_files_in_folder, a bytes download ...) so every call goes through
    drive_call. No retry logic of its own: same classifier, same backoff.

    Wrap the fetch, not the parse: pyarrow raises a plain OSError for a corrupt
    parquet footer, which _classify would take for a dropped connection.

    get_or_create_subfolder is safe to wrap whole: a retry lists again first, so
    a folder whose create landed before a 5xx is found, not created twice (the
    same look-again rule as compute_features.upload_parquet)."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # (drive, parent, NAME): label with the name; never with a bare Drive id
        name = args[2] if len(args) > 2 and isinstance(args[2], str) else ""
        label = f"{fn.__name__} {name}".strip()
        return drive_call(lambda: fn(*args, **kwargs), attempts=attempts, label=label)
    return wrapped


# ---------------------------------------------------------------- cache ----

class ParquetCache:
    """Content-addressed cache of Drive parquets.

    The key is (file_id, the modifiedTime Drive itself reports). Callers re-list
    the folder live on every run, so each run learns the CURRENT modifiedTime
    before consulting the cache: if the file changed on Drive the key changes and
    it re-downloads. Staleness is therefore designed out, not merely bounded —
    unlike a TTL cache (cf. the CACHE_HOURS in fetch_gf_filtered.py).
    """

    def __init__(self, root=None, enabled: bool = True):
        self.enabled = enabled
        self.root = Path(root or os.environ.get("GALLERY_CACHE_DIR")
                         or _DEFAULT_CACHE_ROOT)
        self.hits = 0
        self.misses = 0
        self.stores = 0
        if self.enabled:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                log(f"  cache disabled (cannot create {self.root}: {type(e).__name__})")
                self.enabled = False

    @staticmethod
    def _key(file_id: str, mtime: str) -> str:
        return hashlib.sha1(f"{file_id}|{mtime}".encode()).hexdigest()[:16]

    def _paths(self, file_id: str, mtime: str):
        k = self._key(file_id, mtime)
        return self.root / f"{k}.parquet", self.root / f"{k}.meta"

    def get(self, file_id: str, mtime: str):
        """-> DataFrame, or None on any miss. Every integrity problem (missing
        sidecar, size mismatch, unparseable parquet) counts as a miss."""
        if not self.enabled or not mtime:
            return None
        p, m = self._paths(file_id, mtime)
        if not (p.exists() and m.exists()):
            self.misses += 1
            return None
        try:
            meta = json.loads(m.read_text())
            if meta.get("file_id") != file_id or meta.get("mtime") != mtime:
                raise ValueError("sidecar mismatch")
            if int(meta.get("size", -1)) != p.stat().st_size:
                raise ValueError("size mismatch")
            df = pd.read_parquet(p)
        except Exception:
            self.misses += 1
            return None
        self.hits += 1
        return df

    def put(self, file_id: str, mtime: str, data: bytes) -> None:
        """Store atomically: write .tmp beside the target, then os.replace, so a
        crash mid-write can never leave a truncated entry behind."""
        if not self.enabled or not mtime or not data:
            return
        p, m = self._paths(file_id, mtime)
        tmp_p, tmp_m = Path(f"{p}.tmp"), Path(f"{m}.tmp")
        try:
            tmp_p.write_bytes(data)
            tmp_m.write_text(json.dumps({"file_id": file_id, "mtime": mtime,
                                         "size": len(data)}))
            os.replace(tmp_p, p)
            os.replace(tmp_m, m)
            self.stores += 1
        except Exception as e:
            log(f"  cache write skipped ({type(e).__name__})")
            for t in (tmp_p, tmp_m):
                try:
                    t.unlink()
                except Exception:
                    pass

    def purge(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def summary(self) -> str:
        if not self.enabled:
            return "cache off"
        return f"cache {self.hits} hit / {self.misses} miss / {self.stores} stored"
