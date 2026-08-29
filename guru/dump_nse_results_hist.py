r"""
P5b (NSE route) — dump_nse_results_hist.py  (Project Guru, STANDALONE, RESUMABLE)

Deep quarterly financials from NSE's corporates-financial-results archive —
structured, fixed-template, FREE, no account. Verified live 2026-07-06:
TCS quarterly records exist from 2007; detail pages parse with pandas.read_html.

Two phases in one script (both resumable):
  PHASE 1 (index): per NSE symbol, walk 2-year windows 2007->today against
    /api/corporates-financial-results?index=equities&symbol=S&period=Quarterly
    &from_date=&to_date= and store EVERY field of every record ->
    guru/data/nse_results_index.parquet   (ledger: per symbol)
  PHASE 2 (details): fetch each record's resultDetailedDataLink (nsearchives
    fixed-template HTML, no cookie needed) and store EVERY line item ->
    guru/data/nse_results_facts/<SYMBOL>.parquet  (long: line_item, value)
    (ledger: per record seq id)

NSE quirks handled:
  * list API needs cookie warmup on nseindia.com; session is refreshed every
    WARMUP_EVERY calls and on any 401/403/timeout.
  * detail pages live on nsearchives.nseindia.com — static host, no cookies.
  * consolidated AND standalone filings both kept (flagged via 'consolidated').

Usage:
    python guru/dump_nse_results_hist.py --dry-run
    python guru/dump_nse_results_hist.py --phase index --limit 3     # pilot
    python guru/dump_nse_results_hist.py --phase details --limit 20  # pilot
    python guru/dump_nse_results_hist.py                             # both, resume
    python guru/dump_nse_results_hist.py --status
"""
from __future__ import annotations

import argparse
import io
import os
import re
import time
from datetime import date, datetime

import pandas as pd
import requests

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
STATUS_DIR = os.path.join(DATA_DIR, "_dump_status")
FACTS_DIR = os.path.join(DATA_DIR, "nse_results_facts")
INDEX_PATH = os.path.join(DATA_DIR, "nse_results_index.parquet")
SYM_LEDGER = os.path.join(STATUS_DIR, "nse_results_sym_ledger.parquet")
DET_LEDGER = os.path.join(STATUS_DIR, "nse_results_det_ledger.parquet")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
LIST_URL = ("https://www.nseindia.com/api/corporates-financial-results"
            "?index=equities&symbol={sym}&period=Quarterly"
            "&from_date={frm}&to_date={to}")
START_YEAR = 2007          # verified: nothing before 2007
WARMUP_EVERY = 250         # refresh NSE session every N list calls
LIST_PAUSE = 0.35
DET_PAUSE = 0.15
FLUSH_EVERY_SYM = 20
FLUSH_EVERY_DET = 200


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def new_nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json,text/plain,*/*",
                      "Accept-Language": "en-US,en;q=0.9"})
    s.get("https://www.nseindia.com/", timeout=20)
    s.get("https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
          timeout=20)
    return s


def windows() -> list[tuple[str, str]]:
    out = []
    y = START_YEAR
    today = date.today()
    while y <= today.year:
        frm = f"01-01-{y}"
        to_y = min(y + 1, today.year)
        to = f"31-12-{to_y}" if to_y < today.year else today.strftime("%d-%m-%Y")
        out.append((frm, to))
        y += 2
    return out


def nse_symbols() -> pd.DataFrame:
    uni = pd.read_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"))
    m = uni[uni["nse_symbol"].notna() & uni["nse_symbol"].astype(str).str.strip().ne("")]
    return m[["guru_key", "nse_symbol"]].drop_duplicates("nse_symbol").reset_index(drop=True)


def load_sym_ledger(syms: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(SYM_LEDGER):
        led = pd.read_parquet(SYM_LEDGER)
        new = syms[~syms["nse_symbol"].isin(led["nse_symbol"])]
        if not new.empty:
            add = new.copy(); add["status"] = "pending"; add["records"] = 0; add["error"] = ""
            led = pd.concat([led, add], ignore_index=True)
        return led
    led = syms.copy(); led["status"] = "pending"; led["records"] = 0; led["error"] = ""
    return led


def phase_index(args) -> None:
    syms = nse_symbols()
    led = load_sym_ledger(syms)
    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo = led[todo_mask]
    if args.limit:
        todo = todo.head(args.limit)
    log(f"PHASE 1 (index): {len(todo)} of {len(led)} symbols | windows: {len(windows())}")
    if args.dry_run:
        for _, r in todo.head(10).iterrows():
            print("   ", r["nse_symbol"])
        return

    idx_rows = []
    if os.path.exists(INDEX_PATH):
        idx_rows = pd.read_parquet(INDEX_PATH).to_dict("records")
    seen = {(r.get("symbol"), r.get("seqNumber")) for r in idx_rows}
    s = new_nse_session()
    calls = 0
    n_ok = n_empty = n_err = 0

    def flush():
        os.makedirs(STATUS_DIR, exist_ok=True)
        if idx_rows:
            pd.DataFrame(idx_rows).to_parquet(INDEX_PATH, index=False)
        led.to_parquet(SYM_LEDGER, index=False)

    for i, (li, r) in enumerate(todo.iterrows(), 1):
        sym = r["nse_symbol"]
        got = 0
        try:
            for frm, to in windows():
                calls += 1
                if calls % WARMUP_EVERY == 0:
                    s = new_nse_session()
                for attempt in (1, 2):
                    try:
                        resp = s.get(LIST_URL.format(sym=requests.utils.quote(sym),
                                                     frm=frm, to=to), timeout=30)
                        if resp.status_code in (401, 403):
                            raise requests.RequestException(f"http {resp.status_code}")
                        break
                    except requests.RequestException:
                        if attempt == 2:
                            raise
                        s = new_nse_session()
                if not resp.text.strip().startswith(("[", "{")):
                    continue
                recs = resp.json()
                if not isinstance(recs, list):
                    continue
                for rec in recs:
                    key = (rec.get("symbol"), rec.get("seqNumber"))
                    if key in seen:
                        continue
                    seen.add(key)
                    rec["guru_key"] = r["guru_key"]
                    idx_rows.append(rec)
                    got += 1
                time.sleep(LIST_PAUSE)
            led.at[li, "status"] = "done" if got else "empty"
            led.at[li, "records"] = got
            led.at[li, "error"] = ""
            n_ok += bool(got); n_empty += (not got)
        except Exception as e:
            led.at[li, "status"] = "error"; led.at[li, "error"] = str(e)[:150]; n_err += 1
            s = new_nse_session()
        if i % FLUSH_EVERY_SYM == 0:
            flush()
            log(f"  {i}/{len(todo)} symbols (with-data={n_ok} empty={n_empty} "
                f"err={n_err}) | index rows: {len(idx_rows):,}")
    flush()
    log(f"PHASE 1 COMPLETE: with-data={n_ok} empty={n_empty} err={n_err} | "
        f"total index rows: {len(idx_rows):,}")


def parse_detail_html(html: str) -> list[tuple[str, str]]:
    """fixed-template NSE results page -> [(line_item, value)] — keeps EVERYTHING."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return []
    out = []
    for t in tables:
        if t.shape[1] != 2 or len(t) < 5:
            continue
        for _, row in t.iterrows():
            k = str(row.iloc[0]).strip()
            v = str(row.iloc[1]).strip()
            if not k or k.lower() == "nan" or k == v:   # section headers repeat k==v
                continue
            out.append((k, v))
    return out


def phase_details(args) -> None:
    if not os.path.exists(INDEX_PATH):
        log("no index yet — run phase index first"); return
    idx = pd.read_parquet(INDEX_PATH)
    idx["det_id"] = idx["symbol"].astype(str) + "_" + idx["seqNumber"].astype(str)
    idx = idx[idx["resultDetailedDataLink"].notna()
              & idx["resultDetailedDataLink"].astype(str).str.strip().ne("")
              & idx["resultDetailedDataLink"].astype(str).str.strip().ne("-")]
    if os.path.exists(DET_LEDGER):
        led = pd.read_parquet(DET_LEDGER)
        new = idx[~idx["det_id"].isin(led["det_id"])]
        if not new.empty:
            add = pd.DataFrame({"det_id": new["det_id"], "symbol": new["symbol"],
                                "status": "pending", "error": ""})
            led = pd.concat([led, add], ignore_index=True)
    else:
        led = pd.DataFrame({"det_id": idx["det_id"], "symbol": idx["symbol"],
                            "status": "pending", "error": ""})
    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo_ids = set(led.loc[todo_mask, "det_id"])
    work = idx[idx["det_id"].isin(todo_ids)]
    if args.limit:
        work = work.head(args.limit)
    log(f"PHASE 2 (details): {len(work)} of {len(idx)} records")
    if args.dry_run:
        for _, r in work.head(10).iterrows():
            print("   ", r["det_id"], str(r["resultDetailedDataLink"])[:70])
        return

    s = requests.Session(); s.headers.update({"User-Agent": UA})
    led_idx = {d: i for i, d in enumerate(led["det_id"])}
    buf: dict[str, list] = {}
    n_ok = n_err = 0

    def flush():
        os.makedirs(STATUS_DIR, exist_ok=True); os.makedirs(FACTS_DIR, exist_ok=True)
        for sym, rows in buf.items():
            fp = os.path.join(FACTS_DIR, f"{sym}.parquet")
            df = pd.DataFrame(rows)
            if os.path.exists(fp):
                old = pd.read_parquet(fp)
                df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                    subset=["det_id", "line_item"], keep="last")
            df.to_parquet(fp, index=False)
        buf.clear()
        led.to_parquet(DET_LEDGER, index=False)

    for i, (_, r) in enumerate(work.iterrows(), 1):
        li = led_idx[r["det_id"]]
        try:
            url = str(r["resultDetailedDataLink"])
            if url.startswith("/"):
                url = "https://www.nseindia.com" + url
            resp = s.get(url, timeout=30)
            pairs = parse_detail_html(resp.text) if resp.status_code == 200 else []
            if pairs:
                sym = str(r["symbol"])
                buf.setdefault(sym, []).extend(
                    {"det_id": r["det_id"], "guru_key": r.get("guru_key"),
                     "symbol": sym, "period_to": r.get("toDate"),
                     "relating_to": r.get("relatingTo"),
                     "consolidated": r.get("consolidated"),
                     "audited": r.get("audited"),
                     "filing_date": r.get("filingDate") or r.get("broadCastDate"),
                     "line_item": k, "value": v} for k, v in pairs)
                led.at[li, "status"] = "done"; led.at[li, "error"] = ""; n_ok += 1
            else:
                led.at[li, "status"] = "empty"; led.at[li, "error"] = f"no table ({resp.status_code})"
        except Exception as e:
            led.at[li, "status"] = "error"; led.at[li, "error"] = str(e)[:150]; n_err += 1
        if i % FLUSH_EVERY_DET == 0:
            flush()
            log(f"  {i}/{len(work)} details (ok={n_ok} err={n_err})")
        time.sleep(DET_PAUSE)
    flush()
    log(f"PHASE 2 COMPLETE: ok={n_ok} err={n_err}")


XBRL_DIR = os.path.join(DATA_DIR, "nse_results_xbrl")
XBRL_LEDGER = os.path.join(STATUS_DIR, "nse_results_xbrl_ledger.parquet")
_XBRL_TAG_RE = re.compile(r"<((?:in-[a-z]+|ind-as)[^:>]*:[A-Za-z0-9_]+)"
                          r"(?:\s+[^>]*contextRef=\"([^\"]*)\")?[^>]*>([^<]+)<")


def phase_xbrl(args) -> None:
    """2018+ era: filings carry XBRL .xml links instead of HTML detail pages.
    Fetch each .xml (nsearchives static host, no cookie) and store EVERY tagged
    leaf value (tag, contextRef, value) — the machine-native superset."""
    if not os.path.exists(INDEX_PATH):
        log("no index yet — run phase index first"); return
    idx = pd.read_parquet(INDEX_PATH)
    idx["det_id"] = idx["symbol"].astype(str) + "_" + idx["seqNumber"].astype(str)
    xb = idx[idx["xbrl"].astype(str).str.strip().str.endswith(".xml")].copy()
    xb = xb.drop_duplicates(subset=["det_id"], keep="first")
    if os.path.exists(XBRL_LEDGER):
        led = pd.read_parquet(XBRL_LEDGER)
        new = xb[~xb["det_id"].isin(led["det_id"])]
        if not new.empty:
            led = pd.concat([led, pd.DataFrame(
                {"det_id": new["det_id"], "symbol": new["symbol"],
                 "status": "pending", "error": ""})], ignore_index=True)
    else:
        led = pd.DataFrame({"det_id": xb["det_id"], "symbol": xb["symbol"],
                            "status": "pending", "error": ""})
    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo_ids = set(led.loc[todo_mask, "det_id"])
    work = xb[xb["det_id"].isin(todo_ids)]
    if args.limit:
        work = work.head(args.limit)
    log(f"PHASE 3 (xbrl): {len(work)} of {len(xb)} filings")
    if args.dry_run:
        for _, r in work.head(10).iterrows():
            print("   ", r["det_id"], str(r["xbrl"])[-45:])
        return

    s = requests.Session(); s.headers.update({"User-Agent": UA})
    led_idx = {d: i for i, d in enumerate(led["det_id"])}
    buf: dict[str, list] = {}
    n_ok = n_err = 0

    def flush():
        os.makedirs(STATUS_DIR, exist_ok=True); os.makedirs(XBRL_DIR, exist_ok=True)
        for sym, rows in buf.items():
            fp = os.path.join(XBRL_DIR, f"{sym}.parquet")
            df = pd.DataFrame(rows)
            if os.path.exists(fp):
                old = pd.read_parquet(fp)
                df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                    subset=["det_id", "tag", "context"], keep="last")
            df.to_parquet(fp, index=False)
        buf.clear()
        led.to_parquet(XBRL_LEDGER, index=False)

    for i, (_, r) in enumerate(work.iterrows(), 1):
        li = led_idx[r["det_id"]]
        try:
            resp = s.get(str(r["xbrl"]), timeout=30)
            hits = _XBRL_TAG_RE.findall(resp.text) if resp.status_code == 200 else []
            if hits:
                sym = str(r["symbol"])
                buf.setdefault(sym, []).extend(
                    {"det_id": r["det_id"], "guru_key": r.get("guru_key"),
                     "symbol": sym, "period_to": r.get("toDate"),
                     "relating_to": r.get("relatingTo"),
                     "consolidated": r.get("consolidated"),
                     "audited": r.get("audited"),
                     "filing_date": r.get("filingDate") or r.get("broadCastDate"),
                     "tag": t.split(":", 1)[1], "context": c or "",
                     "value": v.strip()} for t, c, v in hits)
                led.at[li, "status"] = "done"; led.at[li, "error"] = ""; n_ok += 1
            else:
                led.at[li, "status"] = "empty"
                led.at[li, "error"] = f"no tags ({resp.status_code})"
        except Exception as e:
            led.at[li, "status"] = "error"; led.at[li, "error"] = str(e)[:150]; n_err += 1
        if i % FLUSH_EVERY_DET == 0:
            flush()
            log(f"  {i}/{len(work)} xbrl (ok={n_ok} err={n_err})")
        time.sleep(DET_PAUSE)
    flush()
    log(f"PHASE 3 COMPLETE: ok={n_ok} err={n_err}")


def phase_recent(args) -> None:
    """Targeted re-pull of the 2025->today window for every symbol (the original
    run's last windows silently skipped on a stale session). Appends NEW records
    to the index (dedup on symbol+seqNumber); details/xbrl phases then pick the
    new rows up as pending automatically."""
    syms = nse_symbols()
    if args.limit:
        syms = syms.head(args.limit)
    frm, to = "01-01-2025", date.today().strftime("%d-%m-%Y")
    log(f"PHASE recent: window {frm} -> {to} for {len(syms)} symbols")
    idx_rows = []
    if os.path.exists(INDEX_PATH):
        idx_rows = pd.read_parquet(INDEX_PATH).to_dict("records")
    seen = {(r.get("symbol"), r.get("seqNumber")) for r in idx_rows}
    if args.dry_run:
        log("DRY RUN — would fetch one window per symbol, appending new records.")
        return
    s = new_nse_session()
    added = 0; errs = 0
    for i, (_, r) in enumerate(syms.iterrows(), 1):
        sym = r["nse_symbol"]
        try:
            if i % WARMUP_EVERY == 0:
                s = new_nse_session()
            for attempt in (1, 2):
                try:
                    resp = s.get(LIST_URL.format(sym=requests.utils.quote(sym),
                                                 frm=frm, to=to), timeout=30)
                    if resp.status_code in (401, 403):
                        raise requests.RequestException(f"http {resp.status_code}")
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    s = new_nse_session()
            if resp.text.strip().startswith(("[", "{")):
                recs = resp.json()
                if isinstance(recs, list):
                    for rec in recs:
                        key = (rec.get("symbol"), rec.get("seqNumber"))
                        if key in seen:
                            continue
                        seen.add(key)
                        rec["guru_key"] = r["guru_key"]
                        idx_rows.append(rec)
                        added += 1
            time.sleep(LIST_PAUSE)
        except Exception:
            errs += 1
            s = new_nse_session()
        if i % 100 == 0:
            pd.DataFrame(idx_rows).to_parquet(INDEX_PATH, index=False)
            log(f"  {i}/{len(syms)} symbols | +{added} new records (errs={errs})")
    pd.DataFrame(idx_rows).to_parquet(INDEX_PATH, index=False)
    log(f"PHASE recent COMPLETE: +{added} new records (errs={errs}) | "
        f"index now {len(idx_rows):,} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["index", "details", "xbrl", "both", "all",
                                        "recent"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        print("=" * 58)
        print(" Project Guru - NSE results dump progress")
        print("=" * 58)
        if os.path.exists(SYM_LEDGER):
            sl = pd.read_parquet(SYM_LEDGER)
            vc = sl["status"].value_counts().to_dict()
            total = len(sl); done = total - vc.get("pending", 0)
            pct = 100 * done / total if total else 0
            idx_n = len(pd.read_parquet(INDEX_PATH)) if os.path.exists(INDEX_PATH) else 0
            print(f" PHASE 1 (index)  : {done}/{total} symbols  ({pct:5.1f}% done)")
            print(f"    with-data={vc.get('done',0)}  empty={vc.get('empty',0)}  "
                  f"error={vc.get('error',0)}  pending={vc.get('pending',0)}")
            print(f"    filings indexed : {idx_n:,}")
        else:
            print(" PHASE 1 (index)  : not started")
        if os.path.exists(DET_LEDGER):
            dl = pd.read_parquet(DET_LEDGER)
            vc = dl["status"].value_counts().to_dict()
            total = len(dl); done = total - vc.get("pending", 0)
            pct = 100 * done / total if total else 0
            import glob as _g
            fact_files = _g.glob(os.path.join(FACTS_DIR, "*.parquet")) if os.path.isdir(FACTS_DIR) else []
            print(f" PHASE 2 (details): {done}/{total} filings  ({pct:5.1f}% done)")
            print(f"    fact files      : {len(fact_files)} company parquets")
            print(f"    ok={vc.get('done',0)}  empty={vc.get('empty',0)}  "
                  f"error={vc.get('error',0)}  pending={vc.get('pending',0)}")
        else:
            print(" PHASE 2 (details): not started (needs phase 1 first)")
        if os.path.exists(XBRL_LEDGER):
            xl_ = pd.read_parquet(XBRL_LEDGER)
            vc = xl_["status"].value_counts().to_dict()
            total = len(xl_); done = total - vc.get("pending", 0)
            pct = 100 * done / total if total else 0
            print(f" PHASE 3 (xbrl)   : {done}/{total} filings  ({pct:5.1f}% done)")
            print(f"    ok={vc.get('done',0)}  empty={vc.get('empty',0)}  "
                  f"error={vc.get('error',0)}  pending={vc.get('pending',0)}")
        else:
            print(" PHASE 3 (xbrl)   : not started (2018+ era)")
        print("=" * 58)
        return

    if args.phase == "recent":
        phase_recent(args)
        phase_details(args)
        phase_xbrl(args)
        return
    if args.phase in ("index", "both", "all"):
        phase_index(args)
    if args.phase in ("details", "both", "all"):
        phase_details(args)
    if args.phase in ("xbrl", "all"):
        phase_xbrl(args)


if __name__ == "__main__":
    main()
