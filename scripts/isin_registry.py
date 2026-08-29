r"""
isin_registry.py — the ISIN change registry (user 2026-08-31).

THE PROBLEM
    A company's ISIN can change (bonus, sub-division, reissue) while its ticker
    stays the same. Everything here is filed by ISIN, so the system sees a brand-new
    company and orphans the old history. 8 companies are already split this way, all
    within a seven-week window, carrying ~4,639 rows of history with them.

WHY THIS SCRIPT EXISTS RATHER THAN AN INFERENCE
    Noticing that one ISIN vanished from OUR data while a similar one appeared is a
    GUESS. It cannot tell a real change from a delisting or from a failed download.
    This script uses the two authorities instead:

      DISCOVER  the exchange's own security list (NSE EQUITY_L + SME, BSE scrips).
                That file IS the authority on which ticker carries which ISIN. It is
                downloaded weekly today and overwritten, so nothing can be compared.
                `--snapshot` keeps a dated copy; two copies make a change self-evident.

      CONFIRM   BSE's CorporateAction API — the same feed `fetch_bse_eod_api.py`
                already trusts for split/bonus factors. It returns the official event
                type, ratio and ex-date, so every registry row carries its reason.

    Discovery is cheap and swept across the whole universe; confirmation is one call
    per candidate, so it is only ever spent on the handful the list turns up.

WRITES NOTHING EXCEPT THE REGISTRY. No existing table, folder or ledger is modified
by any mode in this file. Repairing the orphaned rows is a separate, later step.

USAGE
    python scripts/isin_registry.py --snapshot              # daily: keep a dated copy
    python scripts/isin_registry.py --detect                # compare last 2 snapshots
    python scripts/isin_registry.py --confirm-known --dry-run   # seed the 8 already broken
    python scripts/isin_registry.py --show                  # print the registry

LIMIT: --snapshot only starts helping from the day it first runs. The 8 changes that
already happened have no "before" copy, which is exactly why --confirm-known exists.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

from _extractor_base import (  # noqa: E402
    log, get_drive, get_or_create_subfolder, find_file, download_bytes,
    load_parquet, save_parquet,
)

# ── Sources ──────────────────────────────────────────────────────────────────
NSE_MAIN = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_SME = "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
      "Accept": "text/csv,*/*"}
# The same feed fetch_bse_eod_api.py already trusts for official split/bonus factors.
CORP_URL = "https://api.bseindia.com/BseIndiaAPI/api/CorporateAction/w?scripcode={code}"

SNAP_DIR = ("universe", "isin_history")          # dated copies live here
REGISTRY = "isin_alias.parquet"                   # in company_repo/_index/
REG_COLS = ["old_isin", "new_isin", "symbol", "exchange", "changed_between",
            "event_type", "ratio", "ex_date", "confirmed", "source", "detected_on"]

# The 8 splits found by sweeping company_repo against master_list. They pre-date any
# snapshot, so they can only be confirmed one at a time against the corporate feed.
# Third value is when the NEW isin's folder first appeared — our best proxy for the
# change date, used to demand that any corporate action actually LINES UP in time.
KNOWN = [
    ("INE063E01053", "INE063E01061", "POCL",       "2026-07-24"),
    ("INE0JYY01011", "INE0JYY01029", "MWL",        "2026-07-16"),
    ("INE419M01027", "INE419M01035", "TDPOWERSYS", "2026-08-23"),
    ("INE506W01012", "INE506W01020", "KRISHANA",   "2026-07-08"),
    ("INE682M01012", "INE682M01020", "JLHL",       "2026-07-27"),
    ("INE811A01020", "INE811A01038", "KIRLPNU",    "2026-08-24"),
    ("INE869Y01010", "INE869Y01028", "TEMBO",      "2026-08-25"),
    ("INE900L01010", "INE900L01028", "MBAPL",      "2026-07-08"),
]
# A corporate action only EXPLAINS an ISIN change if its ex-date is near it. The feed
# returns a company's whole history, and POCL's most recent bonus is from 2022 — four
# years before its 2026 change. Without this window that row would read "confirmed".
EXDATE_WINDOW_DAYS = 45


# ── Snapshot ─────────────────────────────────────────────────────────────────

def _fetch(url: str, exchange: str) -> pd.DataFrame:
    r = requests.get(url, headers=UA, timeout=45)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    col = next((c for c in ("isin_number", "isin") if c in df.columns), None)
    if not col or "symbol" not in df.columns:
        raise ValueError(f"{exchange}: no symbol/isin column in {list(df.columns)}")
    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str).str.strip().str.upper(),
        "isin": df[col].astype(str).str.strip().str.upper(),
        "exchange": exchange,
    })
    return out[out["isin"].str.match(r"^IN[A-Z0-9]{10}$", na=False)]


def snapshot(drive, root_id, asof: date, dry: bool) -> pd.DataFrame:
    """Keep a dated copy of the exchange's own ticker -> ISIN mapping.

    This is the whole point: the official list already says which ticker carries
    which ISIN, but it is overwritten on every refresh. One dated copy per day turns
    a change into a fact you can point at instead of a pattern you inferred."""
    frames = []
    for url, ex in ((NSE_MAIN, "NSE"), (NSE_SME, "NSE_SME")):
        try:
            f = _fetch(url, ex)
            log(f"  {ex}: {len(f)} securities")
            frames.append(f)
        except Exception as e:
            # A partial snapshot is worse than none: a list that failed to download
            # looks exactly like every one of its tickers being delisted.
            log(f"  ERROR fetching {ex}: {str(e)[:120]}")
            raise SystemExit(f"aborting — refusing to write a partial snapshot")
    snap = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "exchange"])
    name = f"{asof.isoformat()}.csv"
    if dry:
        log(f"  [DRY-RUN] would write {name}: {len(snap)} rows")
        return snap
    fid = root_id
    for n in SNAP_DIR:
        fid = get_or_create_subfolder(drive, fid, n)
    from googleapiclient.http import MediaInMemoryUpload
    buf = snap.to_csv(index=False).encode()
    existing = find_file(drive, fid, name)
    media = MediaInMemoryUpload(buf, mimetype="text/csv", resumable=False)
    if existing:
        drive.files().update(fileId=existing, media_body=media).execute()
    else:
        drive.files().create(body={"name": name, "parents": [fid]},
                             media_body=media, fields="id").execute()
    log(f"  wrote {'/'.join(SNAP_DIR)}/{name}  ({len(snap)} securities)")
    return snap


def _list_snapshots(drive, root_id) -> list[tuple[str, str]]:
    fid = root_id
    for n in SNAP_DIR:
        fid = get_or_create_subfolder(drive, fid, n)
    files = drive.files().list(q=f"'{fid}' in parents and trashed=false",
                               fields="files(id,name)", pageSize=1000
                               ).execute().get("files", [])
    got = [(f["name"][:-4], f["id"]) for f in files if f["name"].endswith(".csv")]
    return sorted(got)


def _read_snapshot(drive, fid: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid)))


# ── Confirm against the official corporate-action feed ───────────────────────

def corp_action(scrip_or_symbol: str, around: str | None = None) -> dict | None:
    """Official corporate actions for a scrip code, as the NEAREST bonus /
    sub-division to `around`. Returns None if the feed is unavailable or has no
    such record — an unreachable feed leaves a row unconfirmed, never guessed.

    `near` in the result says whether the ex-date is within EXDATE_WINDOW_DAYS of
    the observed change. Only a `near` action explains the ISIN change; a distant
    one is returned as context but must NOT be treated as confirmation."""
    try:
        r = requests.get(CORP_URL.format(code=str(scrip_or_symbol).strip()),
                         headers={"User-Agent": UA["User-Agent"],
                                  "Referer": "https://www.bseindia.com/"}, timeout=25)
        if r.status_code != 200:
            return None
        j = r.json()
    except Exception:
        return None
    anchor = None
    if around:
        try:
            anchor = datetime.strptime(around, "%Y-%m-%d").date()
        except ValueError:
            anchor = None
    best, best_gap = None, None
    for row in (j.get("Table1") or []) if isinstance(j, dict) else []:
        xtype = str(row.get("XTYPE", "")).lower()
        if not ("bonus" in xtype or "sub" in xtype or "split" in xtype):
            continue
        m = re.match(r"(\d{1,2})\s+(\w{3})\s+(\d{4})", str(row.get("BCRD_FROM", "")))
        exd = None
        if m:
            try:
                exd = datetime.strptime(m.group(0), "%d %b %Y").date()
            except ValueError:
                pass
        gap = abs((exd - anchor).days) if (exd and anchor) else 10**6
        cand = {"event_type": str(row.get("XTYPE", "")).strip(),
                "ratio": str(row.get("VALUE", "")).strip(),
                "ex_date": exd.isoformat() if exd else None,
                "near": bool(exd and anchor and gap <= EXDATE_WINDOW_DAYS),
                "gap_days": (gap if gap < 10**6 else None)}
        if best is None or gap < best_gap:
            best, best_gap = cand, gap
    return best


# ── Registry ─────────────────────────────────────────────────────────────────

def _bse_code_map(drive, root_id) -> dict:
    """symbol -> bse scrip code, needed to call the corporate-action feed.
    Lives in company_repo/_index/company_universe.csv (4,975 codes), NOT under
    universe/ — master_list carries no scrip code at all."""
    for folder, fname in ((("company_repo", "_index"), "company_universe.csv"),
                          (("universe",), "company_universe.csv"),
                          (("universe",), "master_list.csv")):
        try:
            fid = root_id
            for n in folder:
                fid = get_or_create_subfolder(drive, fid, n)
            f = find_file(drive, fid, fname)
            if not f:
                continue
            df = pd.read_csv(io.BytesIO(download_bytes(drive, f)))
            df.columns = [c.strip().lower() for c in df.columns]
            code = next((c for c in ("bse_code", "scrip_code", "securitycode")
                         if c in df.columns), None)
            # The ticker column is `nse_symbol` here, not `symbol`.
            sym = next((c for c in ("nse_symbol", "symbol", "bse_symbol")
                        if c in df.columns), None)
            if code and sym:
                m = {}
                for r in df.itertuples():
                    s = str(getattr(r, sym, "")).strip().upper()
                    v = getattr(r, code, None)
                    if not s or s == "NAN" or pd.isna(v):
                        continue
                    # stored as a float (532626.0) — the API wants a bare integer
                    m[s] = str(int(float(v)))
                if m:
                    log(f"  bse code map from {fname}: {len(m)} symbols")
                    return m
        except Exception:
            continue
    log("  WARN: no bse_code column found — rows will stay unconfirmed.")
    return {}


def add_rows(reg: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Append, never duplicating an (old,new) pair already recorded."""
    if not rows:
        return reg
    have = set(zip(reg["old_isin"], reg["new_isin"])) if not reg.empty else set()
    fresh = [r for r in rows if (r["old_isin"], r["new_isin"]) not in have]
    if not fresh:
        return reg
    return pd.concat([reg, pd.DataFrame(fresh, columns=REG_COLS)], ignore_index=True)


def detect(drive, root_id, reg: pd.DataFrame, asof: date) -> tuple[pd.DataFrame, list]:
    """Compare the two most recent snapshots. A ticker carrying a DIFFERENT ISIN
    between two official lists is the exchange stating the change — not an inference.

    A ticker that merely DISAPPEARS is ignored: that is a delisting or a failed
    download, and merging on it would be a guess."""
    snaps = _list_snapshots(drive, root_id)
    if len(snaps) < 2:
        log(f"  only {len(snaps)} snapshot(s) — need 2 to compare. "
            f"Run --snapshot daily; detection begins tomorrow.")
        return reg, []
    (d_prev, id_prev), (d_cur, id_cur) = snaps[-2], snaps[-1]
    prev, cur = _read_snapshot(drive, id_prev), _read_snapshot(drive, id_cur)
    log(f"  comparing {d_prev} ({len(prev)}) -> {d_cur} ({len(cur)})")
    p = prev.set_index(["symbol", "exchange"])["isin"].to_dict()
    codes = None
    found = []
    for r in cur.itertuples():
        old = p.get((r.symbol, r.exchange))
        if old and old != r.isin:
            if codes is None:
                codes = _bse_code_map(drive, root_id)
            ca = corp_action(codes.get(r.symbol, ""), d_cur) if codes.get(r.symbol) else None
            found.append(dict(
                old_isin=old, new_isin=r.isin, symbol=r.symbol, exchange=r.exchange,
                changed_between=f"{d_prev}..{d_cur}",
                event_type=(ca or {}).get("event_type", ""),
                ratio=(ca or {}).get("ratio", ""), ex_date=(ca or {}).get("ex_date"),
                confirmed=bool(ca and ca.get("near")), source="exchange-list", detected_on=asof.isoformat()))
    for f in found:
        log(f"  ISIN CHANGE  {f['symbol']:<14} {f['old_isin']} -> {f['new_isin']}"
            + (f"  [{f['event_type']} {f['ratio']} ex {f['ex_date']}]" if f["confirmed"]
               else "  [UNCONFIRMED — no matching corporate action]"))
    if not found:
        log("  no ISIN changes between those two snapshots.")
    return add_rows(reg, found), found


def confirm_known(drive, root_id, reg: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Seed the 8 splits that pre-date any snapshot.

    They have no "before" copy to diff, so they are corroborated against the LIVE
    official list instead, which is the same authority `--detect` uses: the new ISIN
    must be the one the exchange currently publishes for that ticker, and the old one
    must appear nowhere in it. That is the exchange stating the mapping today.

    The corporate-action feed is asked as well, but only for COLOUR. In practice none
    of these 8 was a bonus or sub-division — they are identity-only reissues, which is
    why TDPOWERSYS's prices never rescaled. A missing action means "no bonus explains
    this", NOT "this mapping is wrong"."""
    live = snapshot(drive, root_id, asof, dry=True)      # fetch only, writes nothing
    by_sym = live.groupby("symbol")["isin"].apply(set).to_dict()
    all_isins = set(live["isin"])
    codes = _bse_code_map(drive, root_id)

    rows, n_ok = [], 0
    for old, new, sym, seen_on in KNOWN:
        cur = by_sym.get(sym, set())
        new_is_live = new in cur
        old_is_gone = old not in all_isins
        ok = new_is_live and old_is_gone
        n_ok += int(ok)
        code = codes.get(sym, "")
        ca = corp_action(code, seen_on) if code else None
        near = bool(ca and ca.get("near"))
        rows.append(dict(
            old_isin=old, new_isin=new, symbol=sym, exchange="NSE",
            changed_between=f"~{seen_on}",
            event_type=(ca or {}).get("event_type", "") if near else "",
            ratio=(ca or {}).get("ratio", "") if near else "",
            ex_date=(ca or {}).get("ex_date") if near else None,
            confirmed=ok, source="exchange-list-live", detected_on=asof.isoformat()))
        if ok:
            log(f"  {sym:<12} {old} -> {new}  CONFIRMED by the live official list"
                + (f"  [{ca['event_type']} {ca['ratio']} ex {ca['ex_date']}]" if near
                   else "  (no bonus/split — identity-only reissue)"))
        else:
            why = []
            if not new_is_live:
                why.append(f"exchange lists {sorted(cur) or 'nothing'} for {sym}")
            if not old_is_gone:
                why.append(f"{old} is STILL live — not superseded")
            log(f"  {sym:<12} {old} -> {new}  REJECTED — " + "; ".join(why))
    log(f"  {n_ok}/{len(KNOWN)} corroborated by the exchange's own current list.")
    return add_rows(reg, [r for r in rows if r["confirmed"]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="store_true",
                    help="Keep a dated copy of the exchange ticker->ISIN list.")
    ap.add_argument("--detect", action="store_true",
                    help="Compare the two most recent snapshots and record changes.")
    ap.add_argument("--confirm-known", action="store_true",
                    help="Seed the 8 pre-registry splits, confirming each against the feed.")
    ap.add_argument("--show", action="store_true", help="Print the registry.")
    ap.add_argument("--mail", action="store_true",
                    help="Email ONLY when a new ISIN change is recorded (silent otherwise).")
    ap.add_argument("--dry-run", action="store_true", help="Read-only: write nothing.")
    ap.add_argument("--asof", default=None, help="Override today's date (YYYY-MM-DD).")
    args = ap.parse_args()
    if not (args.snapshot or args.detect or args.confirm_known or args.show):
        ap.error("give --snapshot, --detect, --confirm-known or --show")

    asof = (datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else date.today())
    log("=" * 64)
    log(f"ISIN registry — {asof}{'  [DRY-RUN]' if args.dry_run else ''}")
    log("=" * 64)

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    index_id = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root_id, "company_repo"), "_index")
    reg = load_parquet(drive, index_id, REGISTRY, REG_COLS)
    before = len(reg)

    if args.snapshot:
        snapshot(drive, root_id, asof, args.dry_run)
    if args.confirm_known:
        reg = confirm_known(drive, root_id, reg, asof)
    if args.detect:
        reg, _ = detect(drive, root_id, reg, asof)

    if len(reg) != before and not args.dry_run:
        save_parquet(drive, index_id, REGISTRY, reg)
        log(f"  registry: {before} -> {len(reg)} rows")
    elif len(reg) != before:
        log(f"  [DRY-RUN] registry would go {before} -> {len(reg)} rows")

    added = len(reg) - before
    if args.mail and added > 0 and not args.dry_run:
        # Silent unless something actually changed — a daily "nothing happened" mail
        # trains you to ignore it, and this fires roughly once a week.
        from mailer import send_email, esc
        new_rows = reg.tail(added)
        body = ["<h2 style='margin:0'>ISIN change detected</h2>",
                f"<p style='color:#666'>{added} change(s) recorded on {asof}. "
                f"Nothing has been repaired — the registry only records them.</p>",
                "<table cellpadding='6' cellspacing='0' style='border-collapse:collapse;"
                "font-size:13px'><tr><th align='left'>Ticker</th><th align='left'>Old ISIN</th>"
                "<th align='left'>New ISIN</th><th align='left'>Why</th></tr>"]
        for r in new_rows.itertuples():
            why = (f"{esc(str(r.event_type))} {esc(str(r.ratio))} ex {esc(str(r.ex_date))}"
                   if r.confirmed and r.event_type else "identity-only reissue")
            body.append(f"<tr><td><b>{esc(str(r.symbol))}</b></td><td>{esc(str(r.old_isin))}"
                        f"</td><td>{esc(str(r.new_isin))}</td><td>{why}</td></tr>")
        body.append("</table><p style='color:#888;font-size:12px'>Source: the exchange's own "
                    "security list, compared against yesterday's copy.</p>")
        send_email(f"ISIN change — {added} company(ies) on {asof}", "".join(body))

    if args.show or len(reg) != before:
        if reg.empty:
            log("  registry is empty.")
        else:
            print()
            print(reg[["symbol", "old_isin", "new_isin", "event_type", "ratio",
                       "ex_date", "confirmed", "source"]].to_string(index=False))
    log("Done. NOTHING ELSE WAS MODIFIED — repairing orphaned rows is a later step.")


if __name__ == "__main__":
    main()
