r"""
dispatch_due.py — start the PF mail workflow on time from a job that is already running.

WHY. GitHub's scheduler ran pf_daily_mails 3 times a day instead of 5, and 4-5 h late
(measured 21-24 Oct 2026: ~14:00, ~19:30, ~01:00 IST against 09:05 / 12:05 / 15:05 / 18:05 /
21:05). The mgmt_interviews POLL job, by contrast, stays up ~5.5 h at a time and between
its four daily runs covers nearly the whole day. User 2026-10-25: "can you manage it with
any of the existing jobs?" — so this runs in the BACKGROUND of that poll job (the interview
code is untouched) and, every few minutes, dispatches pf_daily_mails when a slot has passed
and no pf_daily_mails run has been created since.

    A slot fires only within WINDOW_MIN after it (an old slot is left to the workflow's own
    late cron + its guard, which skips once a run succeeded in the last 150 min).
    Any pf_daily_mails run created since the slot (scheduled, dispatched, manual) counts.

Uses the job's own GITHUB_TOKEN (GH_TOKEN env) via the REST API — standard library only.

    python scripts/dispatch_due.py --self-test
    python scripts/dispatch_due.py --once --dry-run       # what it would do right now
    python scripts/dispatch_due.py --loop-min 340 --every-min 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
WORKFLOW = "pf_daily_mails.yml"
SLOTS_IST = ("09:05", "12:05", "15:05", "18:05", "21:05")      # pf_daily_mails cron, in IST
WINDOW_MIN = 120          # a slot is "due" for this long after it; later = left to the cron


def log(m):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z] {m}", flush=True)


def due_slot(now: datetime, slots=SLOTS_IST, window_min: int = WINDOW_MIN) -> datetime | None:
    """The latest slot at or before `now` (tz-aware) if it is within window_min, else None.
    Returned in UTC. Pure."""
    n = now.astimezone(IST)
    best = None
    for day in (n.date() - timedelta(days=1), n.date()):
        for s in slots:
            h, m = map(int, s.split(":"))
            t = datetime(day.year, day.month, day.day, h, m, tzinfo=IST)
            if t <= n and (best is None or t > best):
                best = t
    if best is None or (n - best) > timedelta(minutes=window_min):
        return None
    return best.astimezone(timezone.utc)


def _api(method: str, path: str, token: str, body: dict | None = None):
    req = urllib.request.Request(
        f"https://api.github.com{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        return r.status, (json.loads(raw) if raw else None)


def runs_since(repo: str, token: str, since_utc: datetime) -> int:
    q = since_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    _, j = _api("GET", f"/repos/{repo}/actions/workflows/{WORKFLOW}/runs"
                       f"?created=%3E%3D{q}&per_page=5", token)
    return int((j or {}).get("total_count", 0))


def check_once(repo: str, token: str, dry_run: bool, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    slot = due_slot(now)
    if slot is None:
        return "no slot due"
    n = runs_since(repo, token, slot)
    if n:
        return f"slot {slot.astimezone(IST):%H:%M} IST already has {n} run(s)"
    if dry_run:
        return f"slot {slot.astimezone(IST):%H:%M} IST has no run - WOULD dispatch {WORKFLOW}"
    status, _ = _api("POST", f"/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches",
                     token, {"ref": "main"})
    return f"slot {slot.astimezone(IST):%H:%M} IST had no run - dispatched (HTTP {status})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop-min", type=float, default=0)
    ap.add_argument("--every-min", type=float, default=5)
    ap.add_argument("--dry-run", action="store_true", help="report, never dispatch")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    token, repo = os.environ.get("GH_TOKEN", ""), os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo:
        log("GH_TOKEN / GITHUB_REPOSITORY not set - nothing to do")
        return 0
    end = time.time() + a.loop_min * 60
    while True:
        try:
            log(check_once(repo, token, a.dry_run))
        except Exception as e:                       # one failed check never ends the loop
            log(f"check failed: {type(e).__name__}: {str(e)[:120]}")
        if a.once or time.time() + a.every_min * 60 > end:
            return 0
        time.sleep(a.every_min * 60)


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    def ist(h, m, d=26):
        return datetime(2026, 10, d, h, m, tzinfo=IST)

    s = due_slot(ist(9, 10))
    check("09:10 IST -> the 09:05 slot is due", s and s.astimezone(IST).strftime("%H:%M") == "09:05")
    check("09:00 IST -> nothing due (21:05 yesterday is > 2 h old)", due_slot(ist(9, 0)) is None)
    check("11:00 IST -> 09:05 slot still due (within 2 h)", due_slot(ist(11, 0)) is not None)
    check("11:10 IST -> 09:05 slot too old", due_slot(ist(11, 10)) is None)
    s2 = due_slot(ist(22, 30))
    check("22:30 IST -> the 21:05 slot", s2 and s2.astimezone(IST).strftime("%H:%M") == "21:05")
    check("00:30 IST -> yesterday's 21:05 is too old", due_slot(ist(0, 30, 27)) is None)
    check("slot returned in UTC", s.tzinfo == timezone.utc and s.strftime("%H:%M") == "03:35")
    print(f"\ndispatch_due self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
