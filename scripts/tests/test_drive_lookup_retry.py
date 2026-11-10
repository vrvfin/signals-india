r"""
test_drive_lookup_retry.py — offline checks for the Phase 1 Drive retry (no Drive, no network).

  1. drive_io._classify: 5xx / 429 / network errors and a RATE-LIMIT 403 are retried;
     every other 403 (permissions, daily quota, storage), 401 and 404 fail at once
  2. drive_io.retrying: passes results through, retries a 500 and a rate-limit 403,
     raises a 404 at once, gives up after LOOKUP_ATTEMPTS calls (waits 202 s in total)
  3. coverage: every Drive files().list on the Phase 1 path (daily.yml) runs inside a
     @retrying helper or a drive_call lambda, and every call of a shared _extractor_base /
     ingest_company_docs lookup is re-bound through retrying() first. Exceptions are
     listed in ALLOWED with the reason.
  4. strategy_pead scan, on a fake Drive: a refused read recovers; a missing file, a corrupt
     file (pyarrow's plain OSError) and a detector error are counted and skipped; a read that
     never comes back stops the scan and fails the step without publishing a partial list
  5. gallery_sync._drive (chart publish, galleries.yml): the company_repo lookup waits out a
     rate-limit 403; a permission 403 still fails at once

Run:
    python scripts/tests/test_drive_lookup_retry.py
Exit code = number of failures.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
os.environ.setdefault("GDRIVE_FOLDER_ID", "ROOT")

import httplib2                                   # noqa: E402
import pandas as pd                               # noqa: E402
from googleapiclient.errors import HttpError      # noqa: E402

import drive_io                                   # noqa: E402

_FAILS: list[str] = []


def check(label, ok, extra=""):
    ok = bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {extra}" if extra else ""))
    if not ok:
        _FAILS.append(label)


def http_error(status: int, reason: str | None = None, message: str = "x") -> HttpError:
    """An HttpError shaped like Drive v3's JSON error body."""
    if reason is None:
        content = b""
    else:
        content = json.dumps({"error": {"code": status, "message": message, "errors": [
            {"message": message, "domain": "usageLimits", "reason": reason}]}}).encode()
    return HttpError(httplib2.Response({"status": status}), content, uri="https://x/files")


# The exact 403 Drive sent on 2026-11-09 (Phase 1 run 42602826310)
QUOTA_MSG = ("Quota exceeded for quota metric 'Total Query Cost' and limit 'Units per "
             "minute per user' of service 'drive.googleapis.com' for consumer "
             "'project_number:645357929207'.")


# ----------------------------------------------------------------- 1 classify
def test_classify():
    print("1. drive_io._classify")
    cases = [
        ("500", http_error(500), True),
        ("503", http_error(503), True),
        ("429", http_error(429, "rateLimitExceeded"), True),
        ("403 rateLimitExceeded (9 Nov body)", http_error(403, "rateLimitExceeded", QUOTA_MSG), True),
        ("403 userRateLimitExceeded", http_error(403, "userRateLimitExceeded"), True),
        ("403 insufficientPermissions", http_error(403, "insufficientPermissions"), False),
        ("403 dailyLimitExceeded", http_error(403, "dailyLimitExceeded"), False),
        ("403 storageQuotaExceeded", http_error(403, "storageQuotaExceeded"), False),
        ("403 forbidden", http_error(403, "forbidden"), False),
        ("403 with no body", http_error(403), False),
        ("401", http_error(401, "authError"), False),
        ("404", http_error(404, "notFound"), False),
        ("ConnectionResetError", ConnectionResetError("reset"), True),
        ("ValueError", ValueError("bad"), False),
    ]
    for label, exc, want in cases:
        got, why = drive_io._classify(exc)
        check(f"{label} -> {'retry' if want else 'fail fast'}", got == want, why)
    check("is_transient agrees with _classify",
          drive_io.is_transient(http_error(403, "rateLimitExceeded"))
          and not drive_io.is_transient(http_error(404, "notFound")))


# ----------------------------------------------------------------- 2 retrying
class Flaky:
    """Raises the queued errors in order, then returns `result`."""
    def __init__(self, errors, result="OK"):
        self.errors, self.result, self.calls = list(errors), result, 0
        self.__name__ = "flaky"                   # retrying() labels with fn.__name__

    def __call__(self, drive, parent, name):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def test_retrying():
    print("2. drive_io.retrying")
    sleeps, results = [], []
    real_sleep = drive_io.time.sleep
    drive_io.time.sleep = sleeps.append

    def check(label, ok, extra=""):               # held back: stdout is captured below
        results.append((label, ok, extra))
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            def find_file(drive, folder_id, name):
                """doc"""
                return f"id-of-{name}"
            w = drive_io.retrying(find_file)
            check("result passes through", w("D", "F", "a.csv") == "id-of-a.csv")
            check("name/doc kept (functools.wraps)",
                  w.__name__ == "find_file" and w.__doc__ == "doc" and w.__wrapped__ is find_file)

            f = Flaky([http_error(500), http_error(500)])
            sleeps.clear()
            check("two 500s then OK -> OK", drive_io.retrying(f)("D", "F", "x") == "OK")
            check("  ... 3 calls, waits 2 s + 5 s", f.calls == 3 and sleeps == [2, 5],
                  f"calls={f.calls} sleeps={sleeps}")

            f = Flaky([http_error(403, "rateLimitExceeded", QUOTA_MSG)])
            check("rate-limit 403 then OK -> OK", drive_io.retrying(f)("D", "F", "x") == "OK")

            f = Flaky([http_error(404, "notFound")])
            sleeps.clear()
            try:
                drive_io.retrying(f)("D", "F", "x")
                check("404 raises at once", False, "no exception")
            except HttpError:
                check("404 raises at once", f.calls == 1 and not sleeps)

            f = Flaky([http_error(403, "insufficientPermissions")])
            try:
                drive_io.retrying(f)("D", "F", "x")
                check("permission 403 raises at once", False, "no exception")
            except HttpError:
                check("permission 403 raises at once", f.calls == 1)

            n = drive_io.LOOKUP_ATTEMPTS
            f = Flaky([http_error(500)] * 100)
            sleeps.clear()
            try:
                drive_io.retrying(f)("D", "F", "x")
                check("endless 500 gives up", False, "no exception")
            except HttpError:
                check(f"endless 500 gives up after {n} calls", f.calls == n, f"calls={f.calls}")
                check("  ... total wait 202 s (2+5+15+30x6)", sum(sleeps) == 202, f"{sum(sleeps)}")
        log = out.getvalue()
        check("each retry is logged with the helper and file name",
              "drive retry 1/9 [flaky x]: HTTP 500 - waiting 2s" in log, log.splitlines()[:1])
        check("a rate-limit retry says so in the log", "HTTP 403 rate limit" in log)
    finally:
        drive_io.time.sleep = real_sleep
    for label, ok, extra in results:
        globals()["check"](label, ok, extra)


# ----------------------------------------------------------------- 3 coverage
SHARED = {"get_or_create_subfolder", "find_file", "load_portfolio_isins",
          "find_latest_portfolio_file", "load_parquet"}
SHARED_MODULES = {"_extractor_base", "ingest_company_docs"}
RETRY_WRAPPERS = {"drive_call", "_dc"}

# (script, function, what) -> why it is deliberately left as it is
ALLOWED = {
    ("pipeline_skip_check.py", "find_sub", "files().list"):
        "already fail-safe: Drive unreachable -> run anyway",
    ("pipeline_skip_check.py", "find_file", "files().list"):
        "already fail-safe: Drive unreachable -> run anyway",
    ("build_universe.py", "upload_csv", "files().list"):
        "write helper (lookup right before update/create); writes are out of scope",
    ("signal_tracker.py", "main", "files().list"):
        "the --replay listing only; daily.yml never passes --replay",
    ("ingest_announcements.py", "_prune_daily", "files().list"):
        "best-effort pruning inside try/except",
    ("daily_brief.py", "load_pf", "load_portfolio_isins"):
        "main() calls load_pf through retrying(); load_pf stays plain for its 12 importers",
    ("daily_brief.py", "load_pf", "find_file"):
        "main() calls load_pf through retrying(); load_pf stays plain for its 12 importers",
    ("daily_brief.py", "load_seen", "load_parquet"):
        "main() calls it fail-open in try/except: a brief with repeats beats none",
}


def daily_scripts() -> list[str]:
    yml = open(os.path.join(REPO, ".github", "workflows", "daily.yml"), encoding="utf-8").read()
    return sorted(set(re.findall(r"python scripts/([a-z_0-9]+\.py)", yml)))


def _scopes(tree):
    """parent map + enclosing function chain for every node"""
    parent = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[c] = n
    return parent


def _enclosing(node, parent):
    out, p = [], parent.get(node)
    while p is not None:
        out.append(p)
        p = parent.get(p)
    return out


def _decorated_retrying(fn) -> bool:
    return any((isinstance(d, ast.Name) and d.id == "retrying") for d in fn.decorator_list)


def _in_retry_lambda(chain) -> bool:
    for i, n in enumerate(chain):
        if isinstance(n, ast.Lambda) and i + 1 < len(chain):
            call = chain[i + 1]
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                    and call.func.id in RETRY_WRAPPERS:
                return True
    return False


def _binds(scope, name):
    """how `scope` (a FunctionDef or Module) binds `name`: 'retrying', 'shared', 'other' or None"""
    how = None
    for st in _own_scope_nodes(scope):
        if isinstance(st, ast.ImportFrom) and any((a.asname or a.name) == name for a in st.names):
            how = how or ("shared" if st.module in SHARED_MODULES else "other")
        if isinstance(st, ast.Assign):
            for tgt in st.targets:
                names = [tgt] if isinstance(tgt, ast.Name) else (
                    list(tgt.elts) if isinstance(tgt, ast.Tuple) else [])
                vals = [st.value] if isinstance(tgt, ast.Name) else (
                    list(st.value.elts) if isinstance(st.value, ast.Tuple) else [])
                for t, v in zip(names, vals):
                    if isinstance(t, ast.Name) and t.id == name and isinstance(v, ast.Call) \
                            and getattr(v.func, "id", "") == "retrying":
                        how = "retrying"
    return how


def _own_scope_nodes(scope):
    """nodes that belong to `scope` itself, not to a nested def"""
    out, stack = [], list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        out.append(n)
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(ast.iter_child_nodes(n))
    return out


def _binding_scope(chain, tree, name):
    """innermost function (or the module) whose own body binds `name`"""
    for n in chain:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any((isinstance(x, ast.ImportFrom) and any((a.asname or a.name) == name for a in x.names))
                   or (isinstance(x, ast.Assign) and any(
                       isinstance(t, ast.Name) and t.id == name for t in x.targets))
                   for x in _own_scope_nodes(n)):
                return n
    return tree


def test_coverage():
    print("3. coverage of the Phase 1 path (daily.yml)")
    scripts = daily_scripts()
    check("daily.yml lists the Phase 1 scripts", len(scripts) >= 25, f"{len(scripts)} scripts")
    uncovered = []
    for s in scripts:
        src = open(os.path.join(SCRIPTS, s), encoding="utf-8").read()
        tree = ast.parse(src)
        parent = _scopes(tree)
        local_defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _enclosing(node, parent)
            fns = [n for n in chain if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            where = fns[0].name if fns else "<module>"
            covered = any(_decorated_retrying(f) for f in fns) or _in_retry_lambda(chain)
            f = node.func
            # (a) a raw drive.files().list(...)
            raw = (isinstance(f, ast.Attribute) and f.attr == "list"
                   and isinstance(f.value, ast.Call)
                   and getattr(f.value.func, "attr", "") == "files")
            # (b) a call of a shared helper imported from _extractor_base / ingest_company_docs
            shared = False
            if isinstance(f, ast.Name) and f.id in SHARED:
                scope = _binding_scope(chain, tree, f.id)
                how = _binds(scope, f.id)
                covered = covered or how == "retrying"
                # a local module-level def is this script's own helper: (a) covers its body
                shared = how in ("shared", "retrying") and not (
                    scope is tree and f.id in local_defs)
            what = "files().list" if raw else (f.id if shared else "")
            if what and not covered and (s, where, what) not in ALLOWED:
                uncovered.append(f"{s}:{node.lineno} in {where} ({what})")
    check("every lookup on the path is retried (or allow-listed)", not uncovered,
          "; ".join(uncovered[:12]) + (f" (+{len(uncovered) - 12} more)" if len(uncovered) > 12 else ""))
    for (s, fn), why in {(s, fn): why for (s, fn, _w), why in ALLOWED.items()}.items():
        check(f"allow-listed {s}:{fn} still exists", s in scripts and
              any(isinstance(n, ast.FunctionDef) and n.name == fn
                  for n in ast.walk(ast.parse(open(os.path.join(SCRIPTS, s), encoding="utf-8").read()))),
              why)
    for s in ("strategy_pead.py", "aggregate_signals.py"):
        tree = ast.parse(open(os.path.join(SCRIPTS, s), encoding="utf-8").read())
        defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        fetch = defs.get("_fetch_bytes")
        uses = all(any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_fetch_bytes"
                       for c in ast.walk(defs[d])) for d in ("download_parquet", "download_csv"))
        check(f"{s}: downloads fetch through a retried _fetch_bytes, parse outside it",
              fetch is not None and _decorated_retrying(fetch) and uses
              and not any(_decorated_retrying(defs[d]) for d in ("download_parquet", "download_csv")))
    for shared_file in ("_extractor_base.py", "ingest_company_docs.py", "extract_concall.py"):
        src = open(os.path.join(SCRIPTS, shared_file), encoding="utf-8").read()
        check(f"{shared_file} untouched (no drive_io / retrying)",
              "retrying" not in src and "drive_io" not in src)


# ----------------------------------------------------------------- 4 PEAD scan
def _run_pead(stocks: dict, workers: int):
    """Run strategy_pead.main() on a fake Drive. stocks: symbol -> behaviour
    ('ok' | 'ratelimit_once' | 'missing' | 'bug' | 'down'). -> (exit, log, uploads, reads)"""
    import importlib
    import strategy_pead as P
    P = importlib.reload(P)
    today = pd.Timestamp.today().normalize()
    syms = list(stocks)
    reads: dict[str, int] = {}
    uploads: list[str] = []

    def fake_bytes(drive, fid):
        reads[fid] = reads.get(fid, 0) + 1
        if fid == "F_summary":
            return pd.DataFrame({"symbol": syms}).to_parquet()
        if fid == "F_results":
            return pd.DataFrame({"isin": [f"IN{s}" for s in syms],
                                 "first_seen_at": [today - pd.Timedelta(days=2)] * len(syms),
                                 "latest_q": [pd.Timestamp("2026-09-30")] * len(syms)}).to_parquet()
        if fid == "F_master":
            return pd.DataFrame({"isin": [f"IN{s}" for s in syms], "symbol": syms}).to_csv(
                index=False).encode()
        sym = fid[2:]
        how = stocks[sym]
        if how == "missing":
            raise http_error(404, "notFound")
        if how == "down":
            raise http_error(500)
        if how == "ratelimit_once" and reads[fid] == 1:
            raise http_error(403, "rateLimitExceeded", QUOTA_MSG)
        good = pd.DataFrame({"date": pd.date_range(end=today, periods=80), "close": 100.0,
                             "open": 100.0, "high": 101.0, "low": 99.0, "volume": 1000}).to_parquet()
        if how == "corrupt":         # bad footer: pyarrow raises a plain OSError on read
            n = int.from_bytes(good[-8:-4], "little")
            return good[:-8 - n] + b"\xff" * n + good[-8:]
        return good

    def fake_detect(sym, ohlcv, anchor, today_dt):
        if stocks[sym] == "bug":
            raise KeyError("earnings_close")
        return None

    P.get_drive = lambda: "DRIVE"
    P._thread_drive = lambda: "DRIVE"
    P.get_or_create_subfolder = lambda d, p, n: f"D_{n}"
    P.find_file = lambda d, f, n: {"summary.parquet": "F_summary", "results.parquet": "F_results",
                                   "master_list.csv": "F_master"}.get(n, f"F_{n}")
    P.list_files_in_folder = lambda d, f: {f"{s}.parquet": f"S_{s}" for s in syms}
    P._fetch_bytes = drive_io.retrying(fake_bytes)    # download_parquet/_csv parse what it returns
    P.detect_pead_at = fake_detect
    P.upload_csv = lambda d, f, name, df, existing_id=None: uploads.append(name)

    real_sleep, real_argv = drive_io.time.sleep, sys.argv
    drive_io.time.sleep = lambda s: None
    sys.argv = ["strategy_pead.py", "--workers", str(workers)]
    code = 0
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                P.main()
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
    finally:
        drive_io.time.sleep, sys.argv = real_sleep, real_argv
    return code, out.getvalue(), uploads, reads


def test_pead_scan():
    print("4. strategy_pead scan on a fake Drive")
    code, log, uploads, reads = _run_pead(
        {"AAA": "ok", "RLIM": "ratelimit_once", "MISS": "missing", "BUG": "bug",
         "CORR": "corrupt"}, workers=4)
    check("a refused read is retried and recovers", reads.get("S_RLIM") == 2, reads.get("S_RLIM"))
    check("a missing file is not retried", reads.get("S_MISS") == 1, reads.get("S_MISS"))
    check("a corrupt file (OSError on parse) is read once, not taken for an outage",
          reads.get("S_CORR") == 1, reads.get("S_CORR"))
    check("run completes and publishes", code == 0 and "latest.csv" in uploads, f"exit={code} {uploads}")
    check("skips are counted in the log, by error type",
          "PEAD scan: 3 stock(s) skipped" in log and "HttpError" in log and "KeyError" in log
          and "OSError" in log,
          next((ln for ln in log.splitlines() if "skipped" in ln), "no line")[:200])

    # workers=1 scans in ISIN order, so ADOWN (INADOWN) is read first
    stocks = {"ADOWN": "down", "B1": "ok", "B2": "ok", "B3": "ok"}
    code, log, uploads, reads = _run_pead(stocks, workers=1)
    check("Drive down after the full budget -> step fails", code == 1, f"exit={code}")
    check("  ... nothing published", not uploads, uploads)
    check(f"  ... the failing read was tried {drive_io.LOOKUP_ATTEMPTS} times",
          reads.get("S_ADOWN") == drive_io.LOOKUP_ATTEMPTS, reads.get("S_ADOWN"))
    check("  ... the other stocks are not ground through the same budget",
          not any(reads.get(f"S_{s}") for s in ("B1", "B2", "B3")),
          {k: v for k, v in reads.items() if k.startswith("S_")})
    check("  ... and the log says why", "PEAD scan STOPPED" in log and "3 stock(s) were not scanned" in log,
          next((ln for ln in log.splitlines() if "STOPPED" in ln), "no line")[:160])


# ----------------------------------------------------------------- 5 gallery_sync
def test_gallery_sync_drive():
    """The one bare lookup in gallery_sync: on 2026-11-09 (galleries run 42660520815) the
    rate-limit 403 on it crashed "Publish the setups page"."""
    print("5. gallery_sync._drive (chart publish, galleries.yml)")
    import _extractor_base as eb
    import gallery_sync
    sleeps = []
    real = (drive_io.time.sleep, eb.get_drive, eb.get_or_create_subfolder)
    drive_io.time.sleep = sleeps.append
    got, raised = None, False
    limited = Flaky([http_error(403, "rateLimitExceeded", QUOTA_MSG)], result="REPO")
    denied = Flaky([http_error(403, "insufficientPermissions")], result="REPO")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            eb.get_drive, eb.get_or_create_subfolder = (lambda: "D"), limited
            try:
                got = gallery_sync._drive()
            except HttpError:
                pass
            eb.get_or_create_subfolder = denied
            try:
                gallery_sync._drive()
            except HttpError:
                raised = True
    finally:
        drive_io.time.sleep, eb.get_drive, eb.get_or_create_subfolder = real
    check("a rate-limit 403 on the company_repo lookup is waited out",
          got is not None and got[:2] == ("D", "REPO") and limited.calls == 2,
          f"calls={limited.calls} waits={sleeps}")
    check("the helper handed to callers is unchanged", got is not None and got[2] is limited)
    check("a permission 403 still fails at once", raised and denied.calls == 1, f"calls={denied.calls}")


if __name__ == "__main__":
    test_classify()
    test_retrying()
    test_coverage()
    test_pead_scan()
    test_gallery_sync_drive()
    print(f"\n{len(_FAILS)} failure(s)" + (": " + "; ".join(_FAILS) if _FAILS else ""))
    sys.exit(len(_FAILS))
