r"""
fetch_nse_sme_eod.py — daily prices for NSE SME (Emerge) names that Yahoo lacks.

WHY (2026-10-14). ingest_ohlcv.py prices SME names via Yahoo `<sym>.NS`, and Yahoo
has no data for many of them: 276 of the 593 SME names in universe/master_list.csv
had NO price file at all (PF holdings ANONDITA and OBSCP among them), so every
strategy, gallery and the Guru screen were blind to them. NSE's own daily file (the
CM bhavcopy) carries every SME row (series SM/ST) with OHLC and volume.

HOW — the BSE-only daily step (fetch_bse_eod_api.py) already solves this exact
problem for BSE names, and this reuses it rather than rebuilding it:
  * _merge_bars       official bars replace same-date bars, dedup on date
  * catch-up window   every run re-takes the last --catchup-days from official files
  * split / bonus     a >20% 1-day move is rescaled ONLY on an OFFICIAL corporate-
                      action record (_apply_split + junction_ok, PR #121), never
                      price-implied. For NSE-only names the record comes from NSE's
                      corporate-actions list (index=sme), passed in as corp_fn.
NSE's bhavcopy is UNADJUSTED (its PrvsClsgPric is the raw previous close — verified:
0 adjusted rows across 23 sessions, including AISL's 1:1 bonus on 9 Oct), which is
why the official corporate-action list is required.

WHICH NAMES. master_list SME rows whose price file is MISSING, or was not updated by
today's Yahoo ingest (modifiedTime older than --stale-hours). Names Yahoo keeps fresh
are never touched. The six Yahoo-collision symbols (ingest_ohlcv.COLLISION_SKIP) are
left out: their files have held another company's prices before.

Phase 1 step: after the Yahoo + BSE ingests, before compute_features. A history
backfill for names that have none: --from YYYY-MM-DD (one-off, outside 10:30-11:45 UTC).

    python scripts/fetch_nse_sme_eod.py --dry-run
    python scripts/fetch_nse_sme_eod.py --names ANONDITA,OBSCP --from 2025-10-01
    python scripts/fetch_nse_sme_eod.py --self-test
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

NSE_BHAV = ("https://nsearchives.nseindia.com/content/cm/"
            "BhavCopy_NSE_CM_0_0_0_{d}_F_0000.csv.zip")
NSE_HOME = "https://www.nseindia.com/"
NSE_CA = ("https://www.nseindia.com/api/corporates-corporateActions?index=sme"
          "&from_date={a}&to_date={b}")
SME_SERIES = {"SM", "ST"}
EXDATE_SLACK_DAYS = 5          # same tolerance as the BSE step
WAIT_POLL_SEC = 120
MIN_STALE_HOURS = 6.0           # below this, Yahoo-fresh names would become targets

_sess = None


def _session():
    global _sess
    if _sess is None:
        from curl_cffi import requests as rq      # impersonate: plain requests is refused
        _sess = rq.Session(impersonate="chrome")
    return _sess


def _log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


# --------------------------------------------------------------------------- #
#  pure parts (self-tested)                                                   #
# --------------------------------------------------------------------------- #

def parse_bhav(content: bytes) -> pd.DataFrame:
    """Zipped UDiFF bhavcopy -> SME rows only."""
    z = zipfile.ZipFile(io.BytesIO(content))
    df = pd.read_csv(z.open(z.namelist()[0]), dtype=str)
    return df[df["SctySrs"].astype(str).str.strip().isin(SME_SERIES)]


def _f(v):
    try:
        x = float(str(v).replace(",", ""))
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def sme_bars(df: pd.DataFrame, keys: set) -> dict:
    """key -> bar, in the exact shape fetch_bse_eod_api._merge_bars expects."""
    out = {}
    for r in df.to_dict("records"):
        k = str(r.get("TckrSymb", "")).strip().upper()
        if k not in keys:
            continue
        o, h, l, c = (_f(r.get(x)) for x in ("OpnPric", "HghPric", "LwPric", "ClsPric"))
        if not c or c <= 0:
            continue
        out[k] = {"date": pd.Timestamp(r["TradDt"]).normalize(), "open": o or c,
                  "high": h or c, "low": l or c, "close": c,
                  "volume": float(_f(r.get("TtlTradgVol")) or 0)}
    return out


def action_factor(subject: str) -> float | None:
    """NSE corporate-action subject -> price divisor. Bonus X:Y -> (X+Y)/Y;
    face-value split 'From Rs A ... To Rs/Re B' -> A/B. Anything else -> None."""
    s = " ".join(str(subject or "").lower().split())
    if "bonus" in s:
        m = re.search(r"(\d+)\s*:\s*(\d+)", s)
        if m and int(m.group(2)) > 0:
            x, y = int(m.group(1)), int(m.group(2))
            return (x + y) / y
    if "split" in s or "sub-division" in s or "sub division" in s:
        fv = [float(x) for x in re.findall(r"r[se]\.?\s*(\d+(?:\.\d+)?)", s)]
        if len(fv) >= 2 and fv[1] > 0 and fv[0] > fv[1]:
            return fv[0] / fv[1]
    return None


_RIGHTS = re.compile(r"rights\s+(\d+)\s*:\s*(\d+)\s*@\s*(?:premium|prem)\s*r[se]?\.?\s*"
                     r"(\d+(?:\.\d+)?)", re.I)


def rights_terms(subject: str, face_val) -> tuple | None:
    """'RIGHTS a:b @ PREMIUM Rs p' + face value -> (a new per b held, issue price)."""
    m = _RIGHTS.search(str(subject or ""))
    fv = _f(face_val)
    if not m or fv is None:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b, fv + float(m.group(3))) if a > 0 and b > 0 else None


def rights_factor(terms: tuple, cum_close: float) -> float | None:
    """Price divisor for a rights issue = cum price / theoretical ex-rights price,
    TERP = (b*P + a*I) / (a+b). Verified on DHARIWAL (1:1 @ Rs 2.50, cum 79.45 ->
    TERP 40.98, first ex close 41.80) and MPEL (2:1 @ Rs 16, cum 38.70 -> TERP
    23.57, ex close 24.70 = TERP + its 5% band). A rights issue priced at or above
    the market changes nothing -> None."""
    a, b, issue = terms
    if not cum_close or cum_close <= issue:
        return None
    terp = (b * cum_close + a * issue) / (a + b)
    return cum_close / terp if terp > 0 else None


def parse_actions(rows: list) -> dict:
    """{symbol: [(ex_date, factor | ("rights", terms))]} from NSE's corporate-actions
    JSON rows. Bonus/split factors are fixed; a rights factor needs the cum price,
    so it is resolved later against the symbol's own bars (corp_fn_for)."""
    out: dict = {}
    for x in rows or []:
        f = action_factor(x.get("subject"))
        if not f:
            t = rights_terms(x.get("subject"), x.get("faceVal"))
            f = ("rights", t) if t else None
        if not f:
            continue
        try:
            exd = datetime.strptime(str(x.get("exDate", "")).strip(), "%d-%b-%Y").date()
        except ValueError:
            continue
        out.setdefault(str(x.get("symbol", "")).strip().upper(), []).append((exd, f))
    return out


def corp_fn_for(sym: str, actions: dict, bars: list | None = None):
    """corp_fn for _merge_bars: (factor, ex_date) within the ex-date slack, else None.
    `bars`: this symbol's official bars, for a rights issue's cum-rights close."""
    evs = actions.get(sym, [])
    closes = sorted(((pd.Timestamp(b_["date"]).date(), float(b_["close"]))
                     for b_ in (bars or [])), key=lambda x: x[0])

    def factor(f, exd):
        if not isinstance(f, tuple):
            return f
        cum = [c for d_, c in closes if d_ < exd]
        return rights_factor(f[1], cum[-1]) if cum else None

    def fn(around: date):
        # A thinly traded SME stock may not trade for days after an ex-date: the
        # first ex-bar can land past the slack (JAKHARIA bonus 8 Oct 2025, next trade
        # 15 Oct; ARIHANTACA bonus 1 Oct 2026, next trade 7 Oct). An ex-date inside
        # the no-trade gap - after the previous bar, on or before this one - is the
        # action behind this jump; it wins over a mere slack match.
        prev = [d_ for d_, _ in closes if d_ < around]
        prev_d = prev[-1] if prev else None
        best = None
        for exd, f in evs:
            in_gap = prev_d is not None and prev_d < exd <= around
            gap = abs((exd - around).days)
            if not (in_gap or gap <= EXDATE_SLACK_DAYS):
                continue
            rank = (0 if in_gap else 1, gap)
            if best is None or rank < best[0]:
                val = factor(f, exd)
                if val:
                    best = (rank, val, exd)
        return (best[1], best[2]) if best else None
    return fn


def source_mismatch(merge_result: dict, tol_pct: float = 25.0) -> bool:
    """_merge_bars reported a jump it could not explain with an official record AND
    the jump is bigger than an SME circuit can make. SME bands are 5/10/20%: a
    20% day is a real move, and treating it as a mismatch (2026-10-14) would have
    stopped a stock updating whenever it hit a circuit inside the 14-day re-take.
    tol = the BSE step's own JUNCTION_TOL (25%)."""
    for e in merge_result.get("events", []):
        m = re.search(r"jump ([+-]?\d+(?:\.\d+)?)%.*no official record", str(e))
        if m and abs(float(m.group(1))) > tol_pct:
            return True
    return False


def pick_targets(sme_syms, files: dict, now: datetime, stale_hours: float,
                 skip: set) -> list:
    """SME names Yahoo did not keep fresh: no file, or file not modified recently."""
    out = []
    for s in sorted(set(sme_syms) - set(skip)):
        meta = files.get(f"{s}.parquet")
        if meta is None:
            out.append(s)
            continue
        try:
            mod = datetime.fromisoformat(meta["modifiedTime"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            out.append(s)
            continue
        if (now - mod).total_seconds() > stale_hours * 3600:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
#  network + Drive                                                            #
# --------------------------------------------------------------------------- #

def fetch_bhav(day: date):
    """(frame, status): ok | none (not published / holiday) | blocked."""
    try:
        r = _session().get(NSE_BHAV.format(d=day.strftime("%Y%m%d")), timeout=40)
    except Exception:
        return None, "blocked"
    if r.status_code == 404:
        return None, "none"
    if r.status_code != 200:
        return None, "blocked"
    if not r.content.startswith(b"PK"):
        return None, "none"
    try:
        return parse_bhav(r.content), "ok"
    except Exception:
        return None, "blocked"


def fetch_actions(start: date, end: date) -> tuple[dict, str]:
    try:
        s = _session()
        s.get(NSE_HOME, timeout=30)                 # NSE's API needs the home cookies
        r = s.get(NSE_CA.format(a=start.strftime("%d-%m-%Y"), b=end.strftime("%d-%m-%Y")),
                  headers={"Referer": NSE_HOME + "companies-listing/corporate-filings-actions",
                           "Accept": "application/json, text/plain, */*"}, timeout=40)
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}"
        j = r.json()
        rows = j if isinstance(j, list) else j.get("data", [])
        return parse_actions(rows), f"{len(rows)} rows"
    except Exception as e:
        return {}, f"failed ({str(e)[:60]})"


def _files_with_mtime(drive, folder_id) -> dict:
    out, tok = {}, None
    while True:
        r = drive.files().list(q=f"'{folder_id}' in parents and trashed=false",
                               fields="nextPageToken, files(id, name, modifiedTime)",
                               pageSize=1000, pageToken=tok).execute()
        for f in r.get("files", []):
            out[f["name"]] = f
        tok = r.get("nextPageToken")
        if not tok:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fetch + report; NO Drive writes")
    ap.add_argument("--names", default="", help="restrict to these SME symbols")
    ap.add_argument("--from", dest="from_date", default="",
                    help="history backfill: every session from this date (YYYY-MM-DD)")
    ap.add_argument("--catchup-days", type=int, default=14,
                    help="calendar days re-taken from official files every run")
    ap.add_argument("--stale-hours", type=float, default=12.0,
                    help="a file Yahoo did not touch for this long counts as uncovered")
    ap.add_argument("--wait-min", type=float, default=0,
                    help="minutes to wait for today's file (0: let tomorrow's catch-up)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true",
                    help="discard the stored history and rebuild from NSE only. "
                         "ONLY for ingest_ohlcv.NSE_OWNED names, which --names must list")
    args = ap.parse_args()
    if args.stale_hours < MIN_STALE_HOURS:
        # 2026-10-14: --stale-hours 0 made every Yahoo-covered SME name a target and
        # overwrote 282 of their files with raw NSE bars (restored from Drive
        # revisions). The Yahoo/NSE ownership boundary is not an option to tune away.
        ap.error(f"--stale-hours must be >= {MIN_STALE_HOURS:g}: lower would take over "
                 f"names Yahoo keeps fresh")

    from _extractor_base import get_drive, find_file, download_bytes
    from fetch_bse_eod_api import _folder, _merge_bars, _thread_drive, _today_ist
    from ingest_ohlcv import COLLISION_SKIP, NSE_OWNED
    from fetch_bse_eod_api import JUNCTION_TOL
    if args.rebuild:
        want = {x.strip().upper() for x in args.names.split(",") if x.strip()}
        if not want or not want <= set(NSE_OWNED):
            # never a scope wider than the NSE-owned list (2026-10-14 incident)
            sys.exit(f"--rebuild needs --names, all in NSE_OWNED {sorted(NSE_OWNED)}")

    drive = get_drive()
    uf = _folder(drive, "universe")
    mid = find_file(drive, uf, "master_list.csv")
    if not mid:
        _log("universe/master_list.csv missing — nothing to do")
        return
    ml = pd.read_csv(io.BytesIO(download_bytes(drive, mid)), dtype=str).fillna("")
    sme = set(ml[ml["series"].str.strip() == "SME"]["symbol"].str.strip().str.upper()) - {""}
    live_fid = _folder(drive, "data/ohlcv")
    files = _files_with_mtime(drive, live_fid)
    now = datetime.now(timezone.utc)
    targets = pick_targets(sme, files, now, args.stale_hours, set(COLLISION_SKIP))
    # NSE-owned names are updated every run, whatever Yahoo did (it skips them)
    targets = sorted(set(targets) | (set(NSE_OWNED) & sme))
    if args.names:
        want = {x.strip().upper() for x in args.names.split(",") if x.strip()}
        targets = [t for t in targets if t in want] or sorted(want & sme)
    n_missing = sum(1 for t in targets if f"{t}.parquet" not in files)
    _log(f"SME names {len(sme)} | Yahoo-uncovered targets {len(targets)} "
         f"({n_missing} with no file) | mode={'DRY-RUN' if args.dry_run else 'LIVE'}")
    if not targets:
        return

    today = _today_ist()
    if args.from_date:
        start = date.fromisoformat(args.from_date)
        days = [start + timedelta(n) for n in range((today - start).days + 1)]
    else:
        days = [today - timedelta(n) for n in range(max(1, args.catchup_days))]
    days = sorted(d for d in days if d.weekday() < 5)

    official, per_day, keys = {}, [], set(targets)
    for d in days:
        df, st = fetch_bhav(d)
        if st == "none" and d == today and args.wait_min > 0:
            deadline = time.time() + args.wait_min * 60
            while st == "none" and time.time() < deadline:
                time.sleep(WAIT_POLL_SEC)
                df, st = fetch_bhav(d)
        if st == "ok":
            bars = sme_bars(df, keys)
            for k, b in bars.items():
                official.setdefault(k, []).append(b)
            per_day.append(f"{d}:{len(bars)}")
        elif st == "blocked":
            per_day.append(f"{d}:BLOCKED")
        time.sleep(0.5)
    _log(f"sessions {len(days)}: " + " ".join(per_day[-10:])
         + (" ..." if len(per_day) > 10 else ""))
    if all(p.endswith("BLOCKED") for p in per_day):
        _log("NSE refused every request — no prices written (Yahoo-uncovered names stay as they are)")
        return

    actions, ca_note = fetch_actions(days[0] - timedelta(days=EXDATE_SLACK_DAYS),
                                     today + timedelta(days=EXDATE_SLACK_DAYS))
    _log(f"NSE SME corporate actions: {ca_note}, {sum(map(len, actions.values()))} "
         f"bonus/split records")

    def _one(k):
        try:
            fid = (files.get(f"{k}.parquet") or {}).get("id")
            kw = dict(corp_fn=corp_fn_for(k, actions, official.get(k, [])))
            if args.rebuild and fid and not args.dry_run:
                # empty the file IN PLACE (same id: no duplicate), then the normal
                # merge rebuilds it from official bars with every corporate action
                empty = pd.DataFrame(columns=["date", "open", "high", "low", "close",
                                              "volume"])
                from _extractor_base import upload_bytes
                upload_bytes(_thread_drive(), live_fid, f"{k}.parquet",
                             empty.to_parquet(index=False), "application/octet-stream",
                             existing_id=fid)
            if fid and not args.dry_run and not args.rebuild and k not in NSE_OWNED:
                # A file Yahoo wrote before: merge only if NSE's prices JOIN it. An
                # unexplained jump means the stored history is another company's or
                # on another scale (RAJPUTANA: stored ~30, NSE ~200, +564%) - never
                # glue the two together; skip and report instead.
                probe = _merge_bars(_thread_drive(), live_fid, k, official.get(k, []), [],
                                    fid, write=False, **kw)
                if source_mismatch(probe, JUNCTION_TOL * 100):
                    probe["events"] = ["SKIPPED - stored history does not join NSE prices: "
                                       + "; ".join(probe["events"])]
                    probe["added"] = probe["replaced"] = 0
                    return probe
            # a --rebuild dry run previews the rebuild: as if the file were empty
            merge_fid = None if (args.rebuild and args.dry_run) else fid
            return _merge_bars(_thread_drive(), live_fid, k, official.get(k, []), [],
                               merge_fid, write=not args.dry_run, **kw)
        except Exception as e:
            return {"key": k, "added": 0, "replaced": 0,
                    "events": [f"error {str(e)[:80]}"], "new_dates": []}

    t0, res = time.time(), []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for f in as_completed([pool.submit(_one, k) for k in sorted(official)]):
            res.append(f.result())
    added = sum(r["added"] for r in res)
    ev = [(r["key"], e) for r in res for e in r["events"]]
    _log(f"SME: {len(res)} names | bars added {added} | replaced "
         f"{sum(r['replaced'] for r in res)} | corporate-action events {len(ev)} | "
         f"{'DRY-RUN - no writes' if args.dry_run else 'written'} | {time.time() - t0:.0f}s")
    for k, e in sorted(ev)[:30]:
        _log(f"    {k:<12} {e}")
    no_bar = sorted(set(targets) - set(official))
    if no_bar:
        _log(f"  {len(no_bar)} target(s) not in any NSE file in the window "
             f"(suspended/migrated?): {', '.join(no_bar[:15])}")


def _self_test() -> int:
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    check("bonus 1:1 -> /2", action_factor("BONUS 1:1") == 2.0)
    check("bonus 2:1 (2 new per 1 held) -> /3, as the BSE step", action_factor("Bonus 2:1") == 3.0)
    check("bonus 1:3 -> /1.333", abs(action_factor("Bonus 1:3") - 4 / 3) < 1e-9)
    check("split Rs 10 -> Rs 2 -> /5", action_factor(
        "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share") == 5.0)
    check("split Rs 10 -> Re 1 -> /10", action_factor(
        "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share") == 10.0)
    check("dividend ignored", action_factor("Interim Dividend - Rs 2 Per Share") is None)
    acts = parse_actions([
        {"symbol": "AISL", "subject": "BONUS 1:1", "exDate": "09-Oct-2026"},
        {"symbol": "AISL", "subject": "Dividend - Rs 1", "exDate": "01-Aug-2026"},
        {"symbol": "X", "subject": "BONUS 1:1", "exDate": "bad"}])
    check("actions parsed, dividends and bad dates dropped",
          acts == {"AISL": [(date(2026, 10, 9), 2.0)]})
    fn = corp_fn_for("AISL", acts)
    check("corp_fn finds the ex-date within slack", fn(date(2026, 10, 10)) == (2.0, date(2026, 10, 9)))
    check("corp_fn: nothing far away", fn(date(2026, 9, 1)) is None)
    check("corp_fn: unknown symbol", corp_fn_for("ZZZ", acts)(date(2026, 10, 9)) is None)
    # rights issues (DHARIWAL / MPEL, real NSE records and closes)
    t = rights_terms("RIGHTS 1:1 @ PREMIUM RE 0.50/-", "2")
    check("rights terms parsed", t == (1, 1, 2.5))
    check("DHARIWAL TERP factor", abs(rights_factor(t, 79.45) - 79.45 / 40.975) < 1e-9)
    t2 = rights_terms("RIGHTS 2:1 @ PREMIUM RS 6/-", "10")
    check("MPEL: 2 new per 1 held at Rs 16", t2 == (2, 1, 16.0)
          and abs(38.70 / rights_factor(t2, 38.70) - 23.5667) < 1e-3)
    check("rights at/above market -> no adjustment", rights_factor((1, 1, 100.0), 90.0) is None)
    acts2 = parse_actions([{"symbol": "DHARIWAL", "exDate": "27-Feb-2026", "faceVal": "2",
                            "subject": "RIGHTS 1:1 @ PREMIUM RE 0.50/-"}])
    bars_d = [{"date": pd.Timestamp("2026-02-26"), "close": 79.45},
              {"date": pd.Timestamp("2026-02-27"), "close": 41.80}]
    got = corp_fn_for("DHARIWAL", acts2, bars_d)(date(2026, 2, 27))
    check("corp_fn resolves a rights factor from the cum close",
          got is not None and abs(got[0] - 79.45 / 40.975) < 1e-9 and got[1] == date(2026, 2, 27))
    jk = parse_actions([{"symbol": "JAKHARIA", "exDate": "08-Oct-2025", "faceVal": "10",
                         "subject": "BONUS 2:1"}])
    bars_j = [{"date": pd.Timestamp("2025-10-07"), "close": 300.0},
              {"date": pd.Timestamp("2025-10-15"), "close": 105.0}]
    check("ex-date inside a no-trade gap matches (JAKHARIA, 7 days late)",
          corp_fn_for("JAKHARIA", jk, bars_j)(date(2025, 10, 15)) == (3.0, date(2025, 10, 8)))
    check("but not when a bar already traded after the ex-date",
          corp_fn_for("JAKHARIA", jk, bars_j + [{"date": pd.Timestamp("2025-10-09"),
                                                  "close": 100.0}])(date(2025, 10, 20)) is None)
    check("rights without a cum bar -> no factor",
          corp_fn_for("DHARIWAL", acts2, [])(date(2026, 2, 27)) is None)

    raw = pd.DataFrame([
        {"TradDt": "2026-10-13", "TckrSymb": "ANONDITA", "SctySrs": "SM", "OpnPric": "1049.95",
         "HghPric": "1058", "LwPric": "991", "ClsPric": "1023.75", "TtlTradgVol": "63800"},
        {"TradDt": "2026-10-13", "TckrSymb": "RELIANCE", "SctySrs": "EQ", "OpnPric": "1",
         "HghPric": "1", "LwPric": "1", "ClsPric": "1166.7", "TtlTradgVol": "1"},
        {"TradDt": "2026-10-13", "TckrSymb": "OBSCP", "SctySrs": "SM", "OpnPric": "",
         "HghPric": "", "LwPric": "", "ClsPric": "829.4", "TtlTradgVol": ""}])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", raw.to_csv(index=False))
    sm = parse_bhav(buf.getvalue())
    check("only SME series kept", set(sm["TckrSymb"]) == {"ANONDITA", "OBSCP"})
    bars = sme_bars(sm, {"ANONDITA", "OBSCP"})
    check("bar shape for _merge_bars", bars["ANONDITA"]["close"] == 1023.75
          and bars["ANONDITA"]["volume"] == 63800.0
          and bars["ANONDITA"]["date"] == pd.Timestamp("2026-10-13"))
    check("missing OHLV filled from close", bars["OBSCP"]["open"] == 829.4
          and bars["OBSCP"]["volume"] == 0.0)
    check("names outside the target set ignored", set(sme_bars(sm, {"OBSCP"})) == {"OBSCP"})

    now = datetime(2026, 10, 14, 11, 0, tzinfo=timezone.utc)
    files = {"FRESH.parquet": {"modifiedTime": "2026-10-14T10:40:00Z"},
             "STALE.parquet": {"modifiedTime": "2026-10-10T10:40:00Z"},
             "KEL.parquet": {"modifiedTime": "2026-10-01T10:40:00Z"}}
    t = pick_targets({"FRESH", "STALE", "NEW", "KEL"}, files, now, 12, {"KEL"})
    check("targets: missing + stale, never fresh or collision", t == ["NEW", "STALE"])
    check("unexplained jump = source mismatch", source_mismatch(
        {"events": ["jump +564% on 2026-10-01 - no official record, history untouched"]}))
    check("a 20% circuit day is NOT a mismatch", not source_mismatch(
        {"events": ["jump +20% on 2026-03-12 - no official record, history untouched"]}))
    check("a -21% day is within the 25% tolerance", not source_mismatch(
        {"events": ["jump -21% on 2026-02-02 - no official record, history untouched"]}))
    check("-65% is a mismatch", source_mismatch(
        {"events": ["jump -65% on 2025-10-15 - no official record, history untouched"]}))
    check("official rescale is not a mismatch", not source_mismatch(
        {"events": ["rescaled /2 (official 2026-10-09)"]}))
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    main()
