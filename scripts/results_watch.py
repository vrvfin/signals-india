r"""
results_watch.py — send a PF holding's full results teardown as soon as Screener has
the quarter (user 2026-10-16: "a financial teardown same day or next day is critical").

WHY. On 15 Oct CYIENTDLM's numbers were on Screener by 19:51 IST, but the full teardown
reached the user at 01:03: the statements refresh (Fundamentals Refresh) and the PF mail
job are both cron-scheduled and GitHub started each of them ~4 h late. This loop does the
two steps itself, only for holdings that need it, every few minutes through the evening.

ONE PASS:
  1. holdings with a season report date in the last --hours that have NOT yet had the
     full teardown (quarter_teardown.recent_reporters + its mail ledger; a holding mailed
     only a filing card is still owed one - quarter_teardown.owed_full_teardown);
  2. re-read just those companies from Screener (ingest_fundamentals --symbols);
  3. those whose statements now show the season quarter -> quarter_teardown
     --per-company --symbols <them> (the teardown's own ledger stops any repeat).
Nothing is sent for a company Screener has not caught up with; pf_daily_mails keeps
sending the "from the filing" card for those.

LOOP (CI): --loop-min / --every-min, results season only (seasons.in_results_mail_window),
and it sits out the Phase-1 Drive window (10:30-11:45 UTC).

    python scripts/results_watch.py --once --dry-run
    python scripts/results_watch.py --loop-min 330 --every-min 20
    python scripts/results_watch.py --once --refresh-only   (pf_daily_mails, before its teardown)
    python scripts/results_watch.py --self-test
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

PHASE1_UTC = ((10, 30), (11, 45))      # shared per-user Drive quota window


def _log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def in_phase1_window(now_utc: datetime) -> bool:
    hm = (now_utc.hour, now_utc.minute)
    return PHASE1_UTC[0] <= hm < PHASE1_UTC[1]


def pick_candidates(recent: list, owed: set, done: set) -> list:
    """(isin, sym, name) owed a full teardown: reported recently or mailed only a filing
    card, and not already mailed in full. Pure."""
    out, seen = [], set()
    for t in recent:
        isin = str(t[0])
        if isin in done or isin in seen:
            continue
        out.append((isin, t[1], t[2]))
        seen.add(isin)
    return out


def ready(cands: list, latest_q: dict, season_label: str) -> list:
    """Candidates whose statements now show the season quarter. Pure."""
    import quarterly_table as QT
    want = QT.norm_q(season_label)
    return [c for c in cands if QT.norm_q(latest_q.get(c[1]) or "") == want]


def _done_full(ledger: pd.DataFrame, season: str) -> set:
    if ledger is None or ledger.empty:
        return set()
    l = ledger[ledger["season_quarter"].astype(str) == season]
    if "mail_mode" in l.columns:
        l = l[l["mail_mode"].astype(str) == "per_company"]
    else:
        return set()
    src = l["data_source"].fillna("screener").astype(str).replace(
        {"None": "screener", "nan": "screener", "": "screener"})
    return set(l[src == "screener"]["isin"].astype(str))


def one_pass(hours: float, dry_run: bool, refresh_only: bool = False) -> dict:
    from _extractor_base import get_drive, get_or_create_subfolder, find_file, download_bytes
    from _extractor_base import load_parquet
    from daily_brief import load_pf
    import quarter_teardown as QTD
    import quarterly_table as QT

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")
    season = QT.season_quarter()
    pf = load_pf(drive, root, idx)
    recent = QTD.recent_reporters(drive, idx, pf, season, hours, _log)
    led = load_parquet(drive, idx, QTD.LEDGER_NAME, QTD.LEDGER_COLS)
    owed = QTD.owed_full_teardown(led, season, "per_company")
    by_isin = {str(i): (i, s, n) for i, s, n in pf}
    recent_all = list(recent) + [by_isin[i] for i in owed if i in by_isin]
    cands = pick_candidates(recent_all, owed, _done_full(led, season))
    _log(f"season {season}: {len(recent)} recent reporter(s), {len(owed)} owed after a "
         f"filing card -> {len(cands)} not yet mailed in full"
         + (f": {', '.join(c[1] for c in cands)}" if cands else ""))
    if not cands:
        return {"candidates": 0, "ready": 0}

    syms = ",".join(c[1] for c in cands)
    cmd = [sys.executable, os.path.join(_SCRIPTS_DIR, "ingest_fundamentals.py"),
           "--symbols", syms] + (["--dry-run"] if dry_run else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    _log("  statements refresh: " + (r.stdout.strip().splitlines() or ["(no output)"])[-1][:160])

    fund = get_or_create_subfolder(drive, root, "fundamentals")
    st = get_or_create_subfolder(drive, fund, "statements")
    latest = {}
    for _i, sym, _n in cands:
        fid = find_file(drive, st, f"{sym}.parquet")
        if fid:
            try:
                latest[sym] = QT.latest_quarter_label(
                    pd.read_parquet(io.BytesIO(download_bytes(drive, fid))))
            except Exception:
                latest[sym] = None
    go = ready(cands, latest, season)
    _log(f"  Screener has {season} for {len(go)} of {len(cands)}"
         + (f": {', '.join(c[1] for c in go)}" if go else ""))
    if not go or refresh_only:
        return {"candidates": len(cands), "ready": len(go)}

    cmd = [sys.executable, os.path.join(_SCRIPTS_DIR, "quarter_teardown.py"),
           "--per-company", "--since-hours", str(int(hours)),
           "--symbols", ",".join(c[1] for c in go)] + (["--dry-run"] if dry_run else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = r.stdout.splitlines()
    sent = [ln for ln in out if "mail sent=True" in ln]
    for ln in [ln for ln in out if " facts " in ln or "mail sent=" in ln
               or "teardown mail(s)" in ln or "--dry-run: not sending" in ln][-12:]:
        _log("  " + ln.split("] ", 1)[-1].strip()[:160])
    if r.returncode:
        _log("  teardown exited " + str(r.returncode) + ": "
             + (r.stderr.strip().splitlines() or [""])[-1][:160])
    return {"candidates": len(cands), "ready": len(go), "sent": len(sent)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--loop-min", type=float, default=0)
    ap.add_argument("--every-min", type=float, default=20)
    ap.add_argument("--hours", type=float, default=96,
                    help="how far back a report date counts as recent")
    ap.add_argument("--dry-run", action="store_true",
                    help="statements refresh and teardown both in --dry-run: no writes, no mail")
    ap.add_argument("--refresh-only", action="store_true",
                    help="refresh statements only; leave the sending to the teardown step "
                         "that follows (pf_daily_mails)")
    ap.add_argument("--any-season", action="store_true",
                    help="ignore the results-season gate (testing)")
    args = ap.parse_args()

    from seasons import in_results_mail_window
    from mail_freshness import IST
    if not args.any_season and not in_results_mail_window(datetime.now(IST)):
        _log("outside the results season — nothing to watch")
        return 0
    end = time.time() + args.loop_min * 60
    while True:
        now = datetime.now(timezone.utc)
        if in_phase1_window(now):
            _log("Phase-1 Drive window — sitting this pass out")
        else:
            try:
                one_pass(args.hours, args.dry_run, args.refresh_only)
            except Exception as e:               # one bad pass never ends the watch
                _log(f"pass failed: {type(e).__name__}: {str(e)[:150]}")
        if args.once or time.time() + args.every_min * 60 > end:
            return 0
        time.sleep(args.every_min * 60)


def _self_test() -> int:
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    recent = [("I1", "AAA", "A"), ("I2", "BBB", "B"), ("I1", "AAA", "A")]
    check("not-yet-mailed recent reporters, de-duplicated",
          pick_candidates(recent, set(), {"I2"}) == [("I1", "AAA", "A")])
    check("already mailed in full -> not a candidate",
          pick_candidates([("I2", "BBB", "B")], set(), {"I2"}) == [])
    check("ready only when statements show the season",
          ready([("I1", "AAA", "A"), ("I2", "BBB", "B")],
                {"AAA": "Q2 FY27", "BBB": "Q1 FY27"}, "Q2FY27") == [("I1", "AAA", "A")])
    check("missing statements -> not ready", ready([("I3", "CCC", "C")], {}, "Q2FY27") == [])
    led = pd.DataFrame([
        {"season_quarter": "Q2FY27", "isin": "I1", "data_source": "screener", "mail_mode": "per_company"},
        {"season_quarter": "Q2FY27", "isin": "I2", "data_source": "filing", "mail_mode": "per_company"},
        {"season_quarter": "Q2FY27", "isin": "I3", "data_source": "screener", "mail_mode": "combined"}])
    check("done = full per-company teardowns only", _done_full(led, "Q2FY27") == {"I1"})
    check("Phase-1 window respected",
          in_phase1_window(datetime(2026, 10, 16, 10, 45, tzinfo=timezone.utc))
          and not in_phase1_window(datetime(2026, 10, 16, 14, 0, tzinfo=timezone.utc)))
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
