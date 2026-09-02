"""build_listing_dates.py — ONE table holding every security's listing date.

WHY
---
The IPO view was wrong twice over. `master_list.listing_date` is populated only
for NSE MAIN BOARD rows: 2,558 of 5,613 names (46%). BSE carries 0 of 2,479 and
NSE SME carries 0 of 576, because build_universe.py hardcodes `"listing_date": ""`
on both of those paths. So "listed in the last year" was blind to 54% of the
universe, including 87 genuine SME IPOs.

DESIGN — reuse the raw sources already pulled; only this final table is new.
Nothing else is modified: build_universe.py, master_list.csv and every live
pipeline are untouched (CLAUDE.md rules 3+5).

  tier 1  NSE main board  master_list.listing_date, already fetched by
                          build_universe.py from NSE's EQUITY_L.csv.  AUTHORITATIVE
  tier 2  NSE SME/Emerge  the SME sibling of that same NSE archive file. Note the
                          2-DIGIT YEAR ('24-Aug-26'): parsing it with the main
                          board's '%d-%b-%Y' silently yields ZERO rows.  AUTHORITATIVE
  tier 3  everything else first bar of data/ohlcv/<SYM>.parquet, which we already
                          store for every name.  INFERRED, and only when it is
                          trustworthy — see the cliff rule below.

THE CLIFF RULE (why tier 3 is not naive)
----------------------------------------
A first OHLCV bar is NOT automatically a listing date: when a batch of names is
added to ingest, they all start on the same day. Measured on a 109-name sample:
100% of NSE SME and 82% of BSE names start on one of 2024-06-12 / 2025-12-19 /
2016-05-18-19. Treating those as listing dates would invent a fake IPO cohort —
exactly the bug being fixed. So any date shared by more than CLIFF_MIN names in a
run is declared an ingest cliff and those names get NO date rather than a wrong
one. Cliffs are detected from the data each run, never hardcoded, so they stay
correct as ingest changes.

Precedence is strict: an authoritative date is never overwritten by an inferred
one, and `source` + `confidence` travel with every row so any tier can be
re-derived later without guessing where a date came from.

Run:
  python scripts/build_listing_dates.py --backfill --dry-run   # full sweep, no writes
  python scripts/build_listing_dates.py --backfill             # once, seeds the table
  python scripts/build_listing_dates.py                        # DAILY: new names only
"""
from __future__ import annotations

import argparse
import io
import re
import os
import sys
from datetime import datetime, date

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, save_parquet, log,
                             acquire_lock, release_lock)
from drive_io import drive_call, ParquetCache

TABLE = "listing_dates.parquet"
LEDGER = "listing_seen_ledger.parquet"      # every security ever observed
INDEX_PATH = "company_repo/_index"
LOCK_NAME = "_listing_dates.lock"

LEDGER_COLS = ["isin", "symbol", "name_norm", "name", "exchange", "scrip_code",
               "first_seen", "last_seen"]

# BSE's active-equity master. A security absent yesterday and present today has
# just listed — this is how same-day IPO alerts are actually done, and it needs
# no listing-date field at all.
BSE_MASTER_URL = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
                  "?Group=&Scripcode=&industry=&segment=Equity&status=Active")
# NOTE: the header dict is built in _fetch_bse_master(), not here — UA is defined
# further down the file and referencing it at this point would NameError on import.

# If a fetched master comes back smaller than this fraction of what the ledger
# already holds for that exchange, treat the fetch as broken and skip it. Without
# this, one bad response would look like a mass delisting and, worse, would make
# every surviving name look "new" on the following run.
MASTER_MIN_FRACTION = 0.90

COLS = ["isin", "symbol", "exchange", "listing_date", "source", "confidence",
        "first_seen", "updated_at",
        # set when a date is shared by more names than could plausibly list on
        # one day, WHATEVER the source said (see the global cliff check)
        "on_date_cliff",
        # --classify pass (additive; blank for rows never classified)
        "listing_type", "type_evidence", "classified_at"]

# listing_type — ONLY ever set from positive proof. Everything else stays
# UNCLASSIFIED. The IPO view filters on this rather than on listing_date alone.
#
#   migration     PROVEN: the stock has price bars well before its listing_date.
#                 You cannot trade a share that has not listed, so prior trading
#                 is proof the date marks a re-listing, not a debut. This is what
#                 an SME moving to the NSE main board looks like — NSE stamps a
#                 fresh date_of_listing on the EQ series and the stock appears to
#                 be brand new. 165 of 510 recent NSE names are this.
#   etf           PROVEN: a fund instrument, not a company (two-factor: fund
#                 naming AND no company financials on file).
#   ipo           PROVEN: a SEBI public-issue prospectus is on file for this
#                 security. Only a genuine public issue files one; a demerger
#                 files a scheme of arrangement with the NCLT instead.
#   unclassified  Not provable from the data we hold. NOTE this is where
#                 demergers land (TMCV, VAML, PIRAMALFIN) — verified 2026-08-28
#                 that neither financial history nor anything else we pull can
#                 separate a demerger from an IPO, so they are NOT guessed at.
#                 See _classify_rows for the two heuristics tested and rejected.
MIGRATION, ETF, IPO, UNCLASSIFIED = "migration", "etf", "ipo", "unclassified"

# A share cannot trade before it lists, so any gap beyond a settlement-sized
# window is real. 30 days is deliberately generous — it is there to absorb data
# noise, not to make a judgement call.
PRIOR_TRADE_DAYS = 30

_FUND_PAT = (r"ETF|BEES|NIFTY|SENSEX|LIQUID|GOLD|SILVER|GILT|BHARATBOND|"
             r"MOMENTUM|LOWVOL|MIDCAP|SMALLCAP|LARGEMID|MID50|MID150|IVZ|AXIS")

# NSE archive files. The main board one is already fetched by build_universe.py;
# the Emerge sibling is the same host and the same column layout.
NSE_MAIN_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_EMERGE_URL = ("https://nsearchives.nseindia.com/emerge/corporates/content/"
                  "SME_EQUITY_L.csv")
# The plain /content/equities/SME_EQUITY_L.csv path exists but is a 2-row stub —
# the real Emerge list (565 rows) lives under /emerge/corporates/.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# More than this many names sharing one first-bar date = an ingest batch, not a
# listing day. Real listing days put a handful of names on the tape at once; the
# measured cliffs put 21-38 out of a 109-name sample on a single date.
CLIFF_MIN = 8

AUTHORITATIVE, INFERRED, UNKNOWN = "authoritative", "inferred", "unknown"


# ----------------------------------------------------------------- Drive ----
_DRIVE = None
_FOLDER_IDS: dict = {}


def _drive():
    global _DRIVE
    if _DRIVE is None:
        _DRIVE = get_drive()
    return _DRIVE


def _reconnect():
    global _DRIVE
    _DRIVE = get_drive()


def _dc(fn, label=""):
    return drive_call(fn, on_reconnect=_reconnect, label=label)


def _folder(parts: str) -> str:
    if parts in _FOLDER_IDS:
        return _FOLDER_IDS[parts]
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = _dc(lambda p=p, fid=fid: get_or_create_subfolder(_drive(), fid, p),
                  label=parts)
    _FOLDER_IDS[parts] = fid
    return fid


def _read_csv_drive(folder: str, name: str) -> pd.DataFrame:
    fid = _dc(lambda: find_file(_drive(), _folder(folder), name), label=name)
    if not fid:
        return pd.DataFrame()
    raw = _dc(lambda: download_bytes(_drive(), fid), label=name)
    return pd.read_csv(io.BytesIO(raw))


def _list_ohlcv() -> dict:
    """{name: {"id", "mtime"}} for data/ohlcv — one listing, reused for every read."""
    fid = _folder("data/ohlcv")
    out, tok = {}, None
    while True:
        resp = _dc(lambda tok=tok: _drive().files().list(
            q=f"'{fid}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,modifiedTime)",
            pageSize=1000, pageToken=tok).execute(), label="ohlcv-list")
        for f in resp.get("files", []):
            out[f["name"]] = {"id": f["id"], "mtime": f.get("modifiedTime", "")}
        tok = resp.get("nextPageToken")
        if not tok:
            break
    return out


# ------------------------------------------------------------ NSE sources ----
def _fetch_nse(url: str, two_digit_year: bool) -> pd.DataFrame:
    """-> DataFrame[symbol, isin, listing_date]. Empty frame on any failure: a
    dead NSE archive must degrade this run, never abort it."""
    try:
        r = requests.get(url, headers={"User-Agent": UA,
                                       "Referer": "https://www.nseindia.com/"},
                         timeout=45)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        log(f"  NSE fetch failed ({url.rsplit('/', 1)[-1]}): "
            f"{type(e).__name__} {str(e)[:70]} — skipped.")
        return pd.DataFrame(columns=["symbol", "isin", "listing_date"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "date_of_listing" not in df.columns or "symbol" not in df.columns:
        log(f"  {url.rsplit('/', 1)[-1]}: unexpected columns {list(df.columns)[:6]} — skipped.")
        return pd.DataFrame(columns=["symbol", "isin", "listing_date"])
    # Emerge stamps a 2-digit year ('24-Aug-26'); the main board uses 4
    # ('06-OCT-2008'). Parsing one with the other's format yields ZERO rows
    # silently, which is exactly how the SME gap went unnoticed.
    fmt = "%d-%b-%y" if two_digit_year else "%d-%b-%Y"
    ld = pd.to_datetime(df["date_of_listing"], format=fmt, errors="coerce")
    if ld.notna().sum() == 0 and len(df):
        ld = pd.to_datetime(df["date_of_listing"], errors="coerce", dayfirst=True)
    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str).str.strip().str.upper(),
        "isin": (df["isin_number"].astype(str).str.strip()
                 if "isin_number" in df.columns else ""),
        "listing_date": ld.dt.date,
    })
    out = out[out["listing_date"].notna() & out["symbol"].ne("")]
    log(f"  {url.rsplit('/', 1)[-1]}: {len(out)} dated rows")
    return out.drop_duplicates("symbol")


def _fetch_nse_full(url: str, two_digit_year: bool) -> pd.DataFrame:
    """As _fetch_nse, but keeps the company NAME too — the watch guardrails need
    it to test whether a company (not just an ISIN) is genuinely new."""
    base = _fetch_nse(url, two_digit_year)
    if base.empty:
        return pd.DataFrame(columns=["symbol", "isin", "listing_date", "name"])
    try:
        r = requests.get(url, headers={"User-Agent": UA,
                                       "Referer": "https://www.nseindia.com/"},
                         timeout=45)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        nm = dict(zip(df["symbol"].astype(str).str.strip().str.upper(),
                      df.get("name_of_company", "").astype(str).str.strip()))
    except Exception:
        nm = {}
    return base.assign(name=base["symbol"].map(nm).fillna(""))


# ---------------------------------------------------------- OHLCV tier 3 ----
def _first_bar_dates(symbols, files, cache) -> dict:
    """{symbol: first bar date}. Reads only the symbols asked for."""
    out = {}
    for i, sym in enumerate(symbols, 1):
        ent = files.get(f"{sym}.parquet")
        if not ent:
            continue
        try:
            df = cache.get(ent["id"], ent["mtime"]) if cache else None
            if df is None:
                raw = _dc(lambda e=ent: download_bytes(_drive(), e["id"]), label=sym)
                df = pd.read_parquet(io.BytesIO(raw))
                if cache:
                    cache.put(ent["id"], ent["mtime"], raw)
            if df.empty or "date" not in df.columns:
                continue
            d = pd.to_datetime(df["date"], errors="coerce").min()
            if pd.notna(d):
                out[sym] = d.date()
        except Exception as e:
            log(f"    {sym}: OHLCV read failed ({type(e).__name__})")
        if i % 250 == 0:
            log(f"    …{i}/{len(symbols)} OHLCV read")
    return out


def _find_cliffs(first_bars: dict) -> set:
    """Dates shared by > CLIFF_MIN names — an ingest batch, not a listing day."""
    if not first_bars:
        return set()
    vc = pd.Series(list(first_bars.values())).value_counts()
    return set(vc[vc > CLIFF_MIN].index)


# -------------------------------------------------------------- classify ----
def _classify_rows(rows: pd.DataFrame, first_bars: dict, sebi_syms: set,
                   sebi_isins: set, has_financials: dict) -> pd.DataFrame:
    """Assign listing_type from POSITIVE proof only.

    Two heuristics were tested against known cases on 2026-08-28 and REJECTED —
    do not reintroduce them without re-testing:

    (a) "long financial history => demerger". Backwards. Genuine IPOs carry 3-6
        years of prior statements from their prospectus (LENSKART 2020, LGEINDIA
        2020, TATACAP 2021) while the demergers TMCV and VAML carry only ~1 year,
        because carve-out financials begin at the demerger.
    (b) "absent from SEBI filings => demerger". The drhp_seeds scrape covers only
        68 of 572 recent listings, so absence means 'not in our sample', not 'no
        prospectus exists'. Presence is proof; absence is not.

    Order matters: migration outranks ipo, because a name that migrated from SME
    to the main board legitimately HAS an old prospectus on file (BETA, SHIVAUM)
    — but the recent date still marks a re-listing, not a debut.
    """
    out_type, out_ev = [], []
    for _, r in rows.iterrows():
        sym = str(r["symbol"]).upper()
        isin = str(r.get("isin", "") or "")
        ld = pd.to_datetime(r["listing_date"], errors="coerce")
        fb = first_bars.get(sym)
        typ, ev = UNCLASSIFIED, ""

        if fb is not None and pd.notna(ld):
            gap = (ld.date() - fb).days
            if gap > PRIOR_TRADE_DAYS:
                typ = MIGRATION
                ev = f"traded from {fb}, {gap}d before the listing date"

        if typ == UNCLASSIFIED:
            looks_fund = bool(pd.Series([sym, str(r.get("name", "") or "").upper()])
                              .str.contains(_FUND_PAT, regex=True, na=False).any())
            if looks_fund and has_financials.get(sym) is False:
                typ, ev = ETF, "fund-style instrument with no company financials on file"

        if typ == UNCLASSIFIED and (sym in sebi_syms or (isin and isin in sebi_isins)):
            typ, ev = IPO, "SEBI public-issue prospectus on file"

        if typ == UNCLASSIFIED:
            bits = []
            if fb is None:
                bits.append("no price history to test against")
            else:
                bits.append("no trading before the listing date")
            bits.append("no SEBI prospectus in our sample")
            ev = "; ".join(bits) + " — cannot prove either way"
        out_type.append(typ)
        out_ev.append(ev)
    rows = rows.copy()
    rows["listing_type"] = out_type
    rows["type_evidence"] = out_ev
    return rows


def _norm_name(s) -> str:
    """Company name reduced to a comparable stem: upper, no punctuation, and the
    corporate suffixes dropped so 'Gaja Alternative Asset Management Ltd' and
    'GAJA ALTERNATIVE ASSET MANAGEMENT LIMITED' collapse to the same key."""
    t = re.sub(r"[^A-Z0-9 ]+", " ", str(s or "").upper())
    t = re.sub(r"\b(LIMITED|LTD|PRIVATE|PVT|PUBLIC|COMPANY|CO|CORPORATION|CORP|"
               r"INDIA|INDIAN|THE|AND)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _fetch_bse_master() -> pd.DataFrame:
    """BSE active-equity master -> [isin, symbol, name, scrip_code]. Empty frame
    on any failure — a dead BSE must never look like a mass delisting."""
    hdr = {"User-Agent": UA, "Accept": "application/json, text/plain, */*",
           "Referer": "https://www.bseindia.com/",
           "Origin": "https://www.bseindia.com"}
    try:
        r = requests.get(BSE_MASTER_URL, headers=hdr, timeout=90)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        log(f"  BSE master fetch failed: {type(e).__name__} {str(e)[:70]} — skipped.")
        return pd.DataFrame(columns=["isin", "symbol", "name", "scrip_code"])
    b = pd.DataFrame(rows)
    if b.empty or "ISIN_NUMBER" not in b.columns:
        log("  BSE master returned nothing usable — skipped.")
        return pd.DataFrame(columns=["isin", "symbol", "name", "scrip_code"])
    out = pd.DataFrame({
        "isin": b["ISIN_NUMBER"].astype(str).str.strip(),
        "symbol": b.get("scrip_id", "").astype(str).str.strip().str.upper(),
        "name": b.get("Scrip_Name", "").astype(str).str.strip(),
        "scrip_code": b.get("SCRIP_CD", "").astype(str).str.strip(),
    })
    out = out[out["isin"].str.match(r"^IN", na=False)]
    log(f"  BSE master: {len(out)} active equity securities")
    return out.drop_duplicates("isin")


def _bse_corp_actions(scrip_code: str) -> list:
    """-> [purpose_name] for the scrip. Used only to explain a candidate away:
    a security carrying a fresh Bonus/Split/Merger record is a corporate action,
    not a debut. Failure returns [] (the check simply does not fire)."""
    if not scrip_code:
        return []
    hdr = {"User-Agent": UA, "Accept": "application/json, text/plain, */*",
           "Referer": "https://www.bseindia.com/",
           "Origin": "https://www.bseindia.com"}
    try:
        r = requests.get(
            f"https://api.bseindia.com/BseIndiaAPI/api/CorporateAction/w"
            f"?scripcode={scrip_code}", headers=hdr, timeout=25)
        j = r.json() if r.ok else {}
        rows = j if isinstance(j, list) else (j.get("Table") or [])
        return [str(x.get("purpose_name", "")) for x in rows]
    except Exception:
        return []


_CA_PAT = re.compile(r"BONUS|SPLIT|SUB.?DIVISION|MERGER|AMALGAM|DEMERG|"
                     r"SCHEME OF ARRANGEMENT|CONSOLIDAT|FACE VALUE", re.I)


def _watch_verdict(symbol: str, name_norm: str, seen_sym: set, seen_name: set,
                   first_bar, corp_actions, asof=None) -> list:
    """-> list of reasons this candidate is NOT a new listing (empty = it IS one).

    Pure function so every guardrail is testable without touching the network.
    All reasons are collected, so the log can explain a rejection fully.

    IDENTITY RULE — symbol and name are judged TOGETHER, never separately:

      symbol KNOWN + name KNOWN   the same company under a new ISIN. That is an
                                  ISIN reissue, i.e. a corporate action. REJECT.
      name KNOWN, symbol new      the same company under a new ticker (a symbol
                                  change). Still not a debut. REJECT.
      symbol KNOWN, name NEW      TICKER RECYCLING — an exchange reassigning a
                                  freed-up ticker from a delisted company to a
                                  different one. This IS a genuine new listing, so
                                  it must NOT be rejected. An earlier version of
                                  this function rejected on symbol alone and would
                                  have silently dropped exactly these IPOs.
    """
    asof = asof or date.today()
    reasons = []
    sym_hit = bool(symbol) and symbol in seen_sym
    name_hit = bool(name_norm) and name_norm in seen_name
    if sym_hit and name_hit:
        reasons.append(f"same symbol AND name as an existing ISIN ({symbol}) "
                       f"— ISIN reissue, not a debut")
    elif name_hit:
        reasons.append("company name already seen on another ISIN "
                       "— same company, new ticker")
    # sym_hit alone is deliberately NOT a rejection; it is surfaced by the caller
    # as a recycled ticker so it can be eyeballed without being suppressed.
    if first_bar is not None and (asof - first_bar).days > PRIOR_TRADE_DAYS:
        reasons.append(f"already had price history from {first_bar}")
    hits = [a for a in (corp_actions or []) if _CA_PAT.search(a or "")]
    if hits:
        reasons.append("corporate action on file: " + "; ".join(hits[:2]))
    return reasons


def _run_watch(args) -> None:
    """DAILY IPO TRACKER — detect newly listed securities by diffing the exchange
    masters against a cumulative ledger of everything ever seen.

    Why a LEDGER and not yesterday's snapshot: a missed CI run would silently
    lose a day of listings if we compared day-over-day. Against a cumulative
    ledger, a skipped run costs nothing — the new name is still absent from the
    ledger tomorrow.

    GUARDRAILS (a candidate must clear all of them to be called a new listing):
      G0 warm-up      on the very first run the ledger is empty, so EVERYTHING
                      looks new. That run records only and emits nothing.
      G1 source sanity a master smaller than MASTER_MIN_FRACTION of what the
                      ledger holds for that exchange is treated as a broken fetch
                      and skipped — never diffed.
      G2 isin novelty  the primary key. Verified 2026-08-28: a bonus does NOT
                      change the ISIN (FREDUN kept INE194R01017 and scrip code
                      539730 across its 2:1 bonus on 2026-07-16), so splits and
                      bonuses cannot manufacture a phantom listing.
      G3 symbol novelty a known symbol on a new ISIN means a re-issue or a
                      security reconstitution, not a debut.
      G4 name novelty  same, on a normalised company-name stem. G3+G4 are the
                      belt-and-braces behind G2: even if some exotic action did
                      mint a fresh ISIN, the company is not new.
      G5 no prior price the security must have no OHLCV bars predating detection.
                      This is the forward-looking version of the migration test.
      G6 corp action   BSE CorporateAction carrying Bonus/Split/Merger/Demerger
                      explains the candidate away.
    Anything caught by G3-G6 is still RECORDED, with the reason — it is just not
    reported as a new listing.
    """
    idx = _folder(INDEX_PATH)
    today = date.today().isoformat()

    led = pd.DataFrame(columns=LEDGER_COLS)
    lfid = _dc(lambda: find_file(_drive(), idx, LEDGER), label=LEDGER)
    if lfid:
        try:
            led = pd.read_parquet(io.BytesIO(
                _dc(lambda: download_bytes(_drive(), lfid), label=LEDGER)))
            for c in LEDGER_COLS:
                if c not in led.columns:
                    led[c] = None
        except Exception as e:
            log(f"  ledger unreadable ({type(e).__name__}) — treating as first run.")
    first_run = led.empty
    log(f"seen-ledger: {len(led)} securities"
        + ("  [FIRST RUN — record only, emit nothing]" if first_run else ""))

    # ---- gather the three masters -------------------------------------
    frames = []
    for url, two_digit in ((NSE_MAIN_URL, False), (NSE_EMERGE_URL, True)):
        m = _fetch_nse_full(url, two_digit)
        if not m.empty:
            frames.append(m.assign(exchange="NSE"))
    bse = _fetch_bse_master()
    if not bse.empty:
        bse = bse.assign(exchange="BSE", listing_date=pd.NaT)
        frames.append(bse)
    if not frames:
        log("  every master failed — nothing to do this run."); return
    cur = pd.concat(frames, ignore_index=True)
    cur["isin"] = cur["isin"].astype(str).str.strip()
    cur = cur[cur["isin"].str.match(r"^IN", na=False)].drop_duplicates("isin")
    cur["symbol"] = cur["symbol"].astype(str).str.upper().str.strip()
    cur["name_norm"] = cur["name"].map(_norm_name) if "name" in cur.columns else ""
    if "scrip_code" not in cur.columns:
        cur["scrip_code"] = ""

    # ---- G1: source sanity, per exchange ------------------------------
    if not first_run:
        for ex in cur["exchange"].unique():
            n_now = int((cur["exchange"] == ex).sum())
            n_led = int((led["exchange"] == ex).sum())
            if n_led and n_now < n_led * MASTER_MIN_FRACTION:
                log(f"  !! {ex} master returned {n_now} vs {n_led} in the ledger "
                    f"(<{MASTER_MIN_FRACTION:.0%}) — refusing to diff this exchange.")
                cur = cur[cur["exchange"] != ex]
    if cur.empty:
        log("  nothing left to diff after the sanity guard."); return

    seen_isin = set(led["isin"].astype(str))
    seen_sym = set(led["symbol"].astype(str))
    seen_name = set(led["name_norm"].astype(str)) - {""}

    cand = cur[~cur["isin"].isin(seen_isin)].copy()
    log(f"  masters: {len(cur)} securities; {len(cand)} ISINs not in the ledger")

    # DISAPPEARANCES. Additions alone cannot tell an ISIN reissue from a debut.
    # Pairing them can: if (symbol, name) LEFT the master under one ISIN and
    # ARRIVED under another, that is positive proof of a corporate action, and it
    # also hands us the old->new mapping so the stored ISIN can be corrected.
    # Only exchanges that passed the G1 sanity guard are considered, otherwise a
    # bad fetch would read as a mass delisting.
    live_ex = set(cur["exchange"].unique())
    gone = led[led["exchange"].isin(live_ex) & ~led["isin"].isin(set(cur["isin"]))]
    reissue = {}
    if not gone.empty:
        log(f"  {len(gone)} security(s) left the master(s) since the last run")
        by_ident = {(str(r["symbol"]), str(r["name_norm"])): str(r["isin"])
                    for _, r in gone.iterrows()}
        by_name = {str(r["name_norm"]): str(r["isin"])
                   for _, r in gone.iterrows() if str(r["name_norm"])}
        for _, c in cand.iterrows():
            k = (str(c["symbol"]), str(c["name_norm"]))
            old = by_ident.get(k) or by_name.get(str(c["name_norm"]))
            if old:
                reissue[str(c["isin"])] = old
        if reissue:
            log(f"  {len(reissue)} of them reappeared under a NEW ISIN "
                f"— confirmed reissue, not a listing:")
            for new_i, old_i in list(reissue.items())[:10]:
                sym = cand[cand["isin"].eq(new_i)]["symbol"].iloc[0]
                log(f"     {sym:<14}{old_i} -> {new_i}")

    results = []
    if not first_run and not cand.empty:
        # G5 needs price history — one folder listing, reads only the candidates.
        cache = ParquetCache(enabled=not args.no_cache)
        files = _list_ohlcv()
        bars = _first_bar_dates(sorted(cand["symbol"].astype(str)), files, cache)
        for _, c in cand.iterrows():
            sym, nn = str(c["symbol"]), str(c["name_norm"])
            # The corporate-action lookup is one HTTP call per candidate, so only
            # ask when nothing cheaper has already explained the name away.
            cheap = _watch_verdict(sym, nn, seen_sym, seen_name, bars.get(sym), [])
            acts = ([] if cheap or str(c.get("exchange")) != "BSE"
                    else _bse_corp_actions(str(c.get("scrip_code", ""))))
            reasons = _watch_verdict(sym, nn, seen_sym, seen_name,
                                     bars.get(sym), acts)
            # Positive proof beats inference: a confirmed disappear/reappear pair
            # is a corporate action regardless of what the other guardrails said.
            if str(c["isin"]) in reissue:
                reasons.insert(0, f"reissue of {reissue[str(c['isin'])]} "
                                  f"(same security left the master under that ISIN)")
            # A recycled ticker is NOT suppressed — it is a real listing on a
            # freed-up symbol — but it is called out so it can be eyeballed.
            elif sym in seen_sym and nn not in seen_name and not reasons:
                log(f"     note: {sym} reuses a ticker last held by a different "
                    f"company — treated as a genuine listing")
            results.append({"isin": c["isin"], "symbol": sym,
                            "name": c.get("name", ""), "exchange": c["exchange"],
                            "scrip_code": c.get("scrip_code", ""),
                            "listing_date": c.get("listing_date"),
                            "verdict": "NEW LISTING" if not reasons else "not new",
                            "why": "; ".join(reasons)})

    res = pd.DataFrame(results)
    new = res[res["verdict"].eq("NEW LISTING")] if not res.empty else pd.DataFrame()
    if not first_run:
        log(f"  ---- {len(new)} NEW LISTING(S) ----")
        for _, r in new.iterrows():
            ld = r["listing_date"]
            when = str(ld)[:10] if pd.notna(ld) else "date TBC (first bar pending)"
            log(f"     {r['symbol']:<14}{r['exchange']:<5}{when:<28}{str(r['name'])[:40]}")
        if not res.empty and len(res) > len(new):
            log(f"  ---- {len(res) - len(new)} candidate(s) explained away ----")
            for _, r in res[res["verdict"].ne("NEW LISTING")].head(12).iterrows():
                log(f"     {r['symbol']:<14}{r['why'][:80]}")

    if args.dry_run:
        log("[DRY-RUN] ledger not updated, listing_dates not written.")
        return

    # ---- persist: ledger always, listing_dates only for real new listings ----
    fresh_led = pd.DataFrame({
        "isin": cur["isin"], "symbol": cur["symbol"], "name_norm": cur["name_norm"],
        "name": cur.get("name", ""), "exchange": cur["exchange"],
        "scrip_code": cur["scrip_code"], "first_seen": today, "last_seen": today})
    if not led.empty:
        prev = dict(zip(led["isin"].astype(str), led["first_seen"]))
        fresh_led["first_seen"] = [prev.get(i, today) for i in fresh_led["isin"]]
        keep = led[~led["isin"].isin(set(fresh_led["isin"]))]
        out_led = pd.concat([keep, fresh_led], ignore_index=True)
    else:
        out_led = fresh_led
    out_led = out_led.drop_duplicates("isin", keep="last").reset_index(drop=True)

    if not acquire_lock(_drive(), idx, LOCK_NAME, "listing_watch",
                        wait_min=args.lock_wait_min):
        log("  lock busy — skipping write."); return
    try:
        _dc(lambda: save_parquet(_drive(), idx, LEDGER, out_led), label=LEDGER)
        log(f"  ledger -> {len(out_led)} securities")
        if not new.empty:
            tfid = _dc(lambda: find_file(_drive(), idx, TABLE), label=TABLE)
            t = pd.DataFrame(columns=COLS)
            if tfid:
                t = pd.read_parquet(io.BytesIO(
                    _dc(lambda: download_bytes(_drive(), tfid), label=TABLE)))
                for c in COLS:
                    if c not in t.columns:
                        t[c] = None
            now = datetime.now().isoformat(timespec="seconds")
            add = pd.DataFrame([{
                "isin": r["isin"], "symbol": r["symbol"], "exchange": r["exchange"],
                # NSE states the date; BSE does not, so detection day stands in
                # until the first traded bar confirms it on a later run.
                "listing_date": (pd.to_datetime(r["listing_date"]).date()
                                 if pd.notna(r["listing_date"]) else date.today()),
                "source": ("exchange_watch_nse" if pd.notna(r["listing_date"])
                           else "exchange_watch_bse_detected"),
                "confidence": (AUTHORITATIVE if pd.notna(r["listing_date"])
                               else INFERRED),
                "first_seen": now, "updated_at": now,
                "listing_type": UNCLASSIFIED,
                "type_evidence": "detected by exchange-master diff; "
                                 "run --classify to label it",
                "classified_at": None} for _, r in new.iterrows()], columns=COLS)
            merged = (pd.concat([t[~t["symbol"].isin(add["symbol"])], add],
                                ignore_index=True)
                        .drop_duplicates("symbol", keep="last")
                        .sort_values("symbol").reset_index(drop=True))
            _dc(lambda: save_parquet(_drive(), idx, TABLE, merged), label=TABLE)
            log(f"  listing_dates += {len(add)} new listing(s)")
    finally:
        release_lock(_drive(), idx, LOCK_NAME)


def _run_classify(args) -> None:
    """Second pass over the existing table — no date sweep, no re-download of
    anything already cached."""
    idx = _folder(INDEX_PATH)
    fid = _dc(lambda: find_file(_drive(), idx, TABLE), label=TABLE)
    if not fid:
        log(f"{TABLE} not found — run --backfill first."); return
    raw = _dc(lambda: download_bytes(_drive(), fid), label=TABLE)
    t = pd.read_parquet(io.BytesIO(raw))
    for c in COLS:
        if c not in t.columns:
            t[c] = None
    ld = pd.to_datetime(t["listing_date"], errors="coerce")
    cut = pd.Timestamp.today().normalize() - pd.Timedelta(days=args.classify_days)
    target = t[ld >= cut].copy()
    log(f"classify: {len(target)} rows dated within {args.classify_days}d")
    if target.empty:
        return

    uni = _read_csv_drive("universe", "master_list.csv")
    if not uni.empty:
        uni["symbol"] = uni["symbol"].astype(str).str.upper()
        target = target.merge(uni[["symbol", "name"]].drop_duplicates("symbol"),
                              on="symbol", how="left")

    syms = target["symbol"].astype(str).str.upper().tolist()
    cache = ParquetCache(enabled=not args.no_cache)
    log("  reading first OHLCV bar (the prior-trading proof)…")
    first_bars = _first_bar_dates(syms, _list_ohlcv(), cache)

    # Financials presence — only needed for names that LOOK like a fund, so this
    # is a few dozen reads, not a few hundred.
    fund_like = [s for s in syms
                 if pd.Series([s]).str.contains(_FUND_PAT, regex=True, na=False).any()]
    has_fin = {}
    if fund_like:
        log(f"  checking financials for {len(fund_like)} fund-style names…")
        stmt_files = {}
        sid = _folder("fundamentals/statements")
        tok = None
        while True:
            resp = _dc(lambda tok=tok: _drive().files().list(
                q=f"'{sid}' in parents and trashed=false",
                fields="nextPageToken, files(id,name)", pageSize=1000,
                pageToken=tok).execute(), label="stmt-list")
            for f in resp.get("files", []):
                stmt_files[f["name"]] = f["id"]
            tok = resp.get("nextPageToken")
            if not tok:
                break
        for s in fund_like:
            has_fin[s] = f"{s}.parquet" in stmt_files

    seeds = pd.DataFrame()
    sfid = _dc(lambda: find_file(_drive(), idx, "drhp_seeds.parquet"), label="seeds")
    if sfid:
        try:
            seeds = pd.read_parquet(io.BytesIO(
                _dc(lambda: download_bytes(_drive(), sfid), label="seeds")))
        except Exception as e:
            log(f"  drhp_seeds unreadable ({type(e).__name__}) — IPO proof unavailable.")
    sebi_syms = set(seeds["symbol"].astype(str).str.upper()) if "symbol" in seeds else set()
    sebi_isins = set(seeds["isin"].astype(str)) if "isin" in seeds else set()
    log(f"  SEBI prospectus index: {len(sebi_isins)} securities")

    done = _classify_rows(target, first_bars, sebi_syms, sebi_isins, has_fin)
    now = datetime.now().isoformat(timespec="seconds")
    done["classified_at"] = now

    log("  ---- listing_type ----")
    for k, n in done["listing_type"].value_counts().items():
        log(f"     {k:<14} {n:>5}")
    yr = pd.to_datetime(done["listing_date"], errors="coerce") >= (
        pd.Timestamp.today().normalize() - pd.Timedelta(days=365))
    y = done[yr]
    log(f"  ---- of the {len(y)} dated inside 365d ----")
    for k, n in y["listing_type"].value_counts().items():
        log(f"     {k:<14} {n:>5}")
    log(f"     => NOT a new listing (migration+etf): "
        f"{int(y['listing_type'].isin([MIGRATION, ETF]).sum())}")

    if args.dry_run:
        log("[DRY-RUN] no Drive write.")
        for k in (MIGRATION, ETF, IPO):
            ex = y[y["listing_type"].eq(k)]["symbol"].head(8).tolist()
            if ex:
                log(f"  {k}: " + ", ".join(ex))
        return

    keep = t[~t["symbol"].isin(done["symbol"])]
    out = (pd.concat([keep, done[COLS]], ignore_index=True)
             .drop_duplicates("symbol", keep="last")
             .sort_values("symbol").reset_index(drop=True))
    if not acquire_lock(_drive(), idx, LOCK_NAME, "listing_dates",
                        wait_min=args.lock_wait_min):
        log("  lock busy — skipping write."); return
    try:
        _dc(lambda: save_parquet(_drive(), idx, TABLE, out), label=TABLE)
        log(f"  wrote {INDEX_PATH}/{TABLE} ({len(out)} rows)")
    finally:
        release_lock(_drive(), idx, LOCK_NAME)


# ------------------------------------------------------------------ main ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="full sweep over the whole universe (run once to seed). "
                         "Default is the DAILY mode: only names missing a date.")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, write nothing to Drive")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap how many names get an OHLCV read (testing)")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass the local parquet cache")
    ap.add_argument("--classify", action="store_true",
                    help="second pass: label listing_type on rows dated inside "
                         "--classify-days. Reads the existing table; does NOT "
                         "redo the date sweep.")
    ap.add_argument("--classify-days", type=int, default=730,
                    help="how far back --classify labels (default 730)")
    ap.add_argument("--watch", action="store_true",
                    help="DAILY IPO TRACKER: diff the NSE/BSE security masters "
                         "against the cumulative seen-ledger and report genuinely "
                         "new listings. First run records only (warm-up).")
    ap.add_argument("--lock-wait-min", type=float, default=5.0)
    args = ap.parse_args()

    if args.watch:
        _run_watch(args)
        return
    if args.classify:
        _run_classify(args)
        return

    started = datetime.now()
    log(f"listing dates — {'BACKFILL' if args.backfill else 'DAILY'}"
        f"{' [DRY-RUN]' if args.dry_run else ''}")

    idx = _folder(INDEX_PATH)
    uni = _read_csv_drive("universe", "master_list.csv")
    if uni.empty or "symbol" not in uni.columns:
        log("master_list.csv missing or malformed — nothing to do."); return
    uni["symbol"] = uni["symbol"].astype(str).str.strip().str.upper()
    uni = uni[uni["symbol"].ne("")].drop_duplicates("symbol")
    log(f"  universe: {len(uni)} names")

    existing = pd.DataFrame(columns=COLS)
    fid = _dc(lambda: find_file(_drive(), idx, TABLE), label=TABLE)
    if fid:
        try:
            raw = _dc(lambda: download_bytes(_drive(), fid), label=TABLE)
            existing = pd.read_parquet(io.BytesIO(raw))
            for c in COLS:
                if c not in existing.columns:
                    existing[c] = None
        except Exception as e:
            log(f"  existing table unreadable ({type(e).__name__}) — rebuilding.")
    conf_by = {} if args.backfill else {
        str(r["symbol"]): str(r["confidence"]) for _, r in existing.iterrows()}
    settled = {s for s, c in conf_by.items() if c == AUTHORITATIVE}
    log(f"  existing table: {len(existing)} rows"
        + ("" if args.backfill else
           f"; {len(settled)} settled (authoritative), {len(conf_by) - len(settled)} open"))

    # Two different target sets, because the two halves cost wildly different
    # amounts. Tiers 1-2 are two cheap CSV reads, so re-run them over every name
    # that is not yet AUTHORITATIVE — that is how an inferred or unknown row gets
    # upgraded when a name later appears on an NSE list (e.g. SME -> main board).
    # Tier 3 downloads one parquet per name, so it runs ONLY for names with no row
    # at all. Without that split, every daily run would re-download the ~2,400
    # cliff-blocked names forever, since a first bar never changes.
    todo = uni if args.backfill else uni[~uni["symbol"].isin(settled)]
    ohlcv_todo = (set(uni["symbol"]) if args.backfill
                  else set(uni["symbol"]) - set(conf_by))
    log(f"  to resolve: {len(todo)} (of which {len(ohlcv_todo)} may need an OHLCV read)")
    if todo.empty:
        log("  nothing new — table already covers the universe."); return

    resolved: dict = {}          # symbol -> (date, source, confidence)

    # ---- tier 1: NSE main board, already in master_list -------------------
    if "listing_date" in todo.columns:
        ld = pd.to_datetime(todo["listing_date"], format="%d-%b-%Y", errors="coerce")
        for sym, d in zip(todo["symbol"], ld):
            if pd.notna(d):
                resolved[sym] = (d.date(), "master_list_nse", AUTHORITATIVE)
    log(f"  tier 1 NSE main board (master_list): {len(resolved)}")

    # ---- tier 2: NSE Emerge / SME ----------------------------------------
    want = set(todo["symbol"]) - set(resolved)
    if want:
        eme = _fetch_nse(NSE_EMERGE_URL, two_digit_year=True)
        n0 = len(resolved)
        for sym, d in zip(eme["symbol"], eme["listing_date"]):
            if sym in want:
                resolved[sym] = (d, "nse_emerge", AUTHORITATIVE)
        log(f"  tier 2 NSE Emerge: +{len(resolved) - n0}")

    # ---- tier 3: first OHLCV bar, cliff-aware -----------------------------
    want = sorted((set(todo["symbol"]) - set(resolved)) & ohlcv_todo)
    if args.limit:
        want = want[:args.limit]
    if want:
        log(f"  tier 3: reading first bar for {len(want)} names…")
        cache = ParquetCache(enabled=not args.no_cache)
        files = _list_ohlcv()
        first = _first_bar_dates(want, files, cache)
        cliffs = _find_cliffs(first)
        if cliffs:
            log(f"  ingest cliffs detected (dates shared by >{CLIFF_MIN} names): "
                + ", ".join(str(c) for c in sorted(cliffs)))
        n0, on_cliff = len(resolved), 0
        for sym, d in first.items():
            if d in cliffs:
                on_cliff += 1          # ingest batch — no date beats a wrong date
                continue
            resolved[sym] = (d, "ohlcv_first_bar", INFERRED)
        log(f"  tier 3 OHLCV: +{len(resolved) - n0} dated, "
            f"{on_cliff} skipped on a cliff, "
            f"{len(want) - len(first)} with no OHLCV file [{cache.summary()}]")

    # Names we tried and could not date get a row too, marked UNKNOWN. That row is
    # what stops tier 3 re-downloading them on every future run: a first bar never
    # changes, so a cliff-blocked name is permanently undatable from OHLCV. They
    # stay eligible for the cheap tiers 1-2 in case NSE lists them later.
    undated = sorted((set(todo["symbol"]) - set(resolved)) & ohlcv_todo)
    for s in undated:
        resolved[s] = (None, "none", UNKNOWN)
    log(f"  RESOLVED {len([1 for v in resolved.values() if v[2] != UNKNOWN])}"
        f" / {len(todo)}; {len(undated)} recorded as undatable")

    # ---- date cliffs, ACROSS EVERY SOURCE ---------------------------------
    # The tier-3 cliff rule above only ever guarded dates INFERRED from a first
    # price bar. An authoritative source can publish a bulk date too: NSE's own
    # EQUITY_L.csv carries 2026-04-20 for 103 different names, which is not 103
    # companies listing on one day — India's busiest genuine IPO day is a
    # handful. Those rows sailed through tier 1 unchallenged and made up a THIRD
    # of the "listed in the last year" cohort feeding the IPO engine.
    #
    # The date is NOT discarded: it may well mean something real (a
    # re-registration, a series migration, a corporate action). It is flagged, so
    # consumers that need "when did this actually start trading" can exclude it
    # while anything wanting the raw exchange field still has it. Additive column,
    # per the schema-first rule.
    dated = {s_: d for s_, (d, _, _) in resolved.items() if d is not None}
    global_cliffs = _find_cliffs(dated)
    if global_cliffs:
        from collections import Counter
        cnt = Counter(dated.values())
        detail = ", ".join(f"{c} ({cnt[c]} names)" for c in sorted(global_cliffs))
        log(f"  DATE CLIFFS across all sources (> {CLIFF_MIN} names on one date): "
            f"{detail}")
        by_src = Counter(src for s_, (d, src, _) in resolved.items()
                         if d in global_cliffs)
        log(f"    by source: {dict(by_src)} — flagged on_date_cliff, date kept")

    # ---- assemble ---------------------------------------------------------
    now = datetime.now().isoformat(timespec="seconds")
    isin_by = dict(zip(uni["symbol"], uni.get("isin", pd.Series(dtype=object))))
    exch_by = dict(zip(uni["symbol"], uni.get("exchange", pd.Series(dtype=object))))
    seen_by = dict(zip(existing.get("symbol", pd.Series(dtype=object)),
                       existing.get("first_seen", pd.Series(dtype=object))))
    rows = [{"isin": isin_by.get(s, ""), "symbol": s, "exchange": exch_by.get(s, ""),
             "listing_date": d, "source": src, "confidence": conf,
             "first_seen": seen_by.get(s) or now, "updated_at": now,
             "on_date_cliff": bool(d is not None and d in global_cliffs)}
            for s, (d, src, conf) in sorted(resolved.items())]
    fresh = pd.DataFrame(rows, columns=COLS)

    # Strict precedence: authoritative > inferred > unknown. A better source always
    # wins; an equal-or-worse one never overwrites what is already there. Ties go
    # to the new row so a corrected NSE date does land.
    RANK = {AUTHORITATIVE: 3, INFERRED: 2, UNKNOWN: 1}
    if not existing.empty:
        both = pd.concat([existing.assign(_new=0), fresh.assign(_new=1)],
                         ignore_index=True)
    else:
        both = fresh.assign(_new=1)
    both["_rank"] = both["confidence"].map(RANK).fillna(0)
    before = both[both["_new"].eq(1)].shape[0]
    out = (both.sort_values(["symbol", "_rank", "_new"])
                .drop_duplicates("symbol", keep="last")
                .drop(columns=["_new", "_rank"])
                .dropna(subset=["symbol"])
                .sort_values("symbol").reset_index(drop=True))
    kept_new = out["updated_at"].eq(now).sum() if "updated_at" in out else 0
    if before and kept_new < before:
        log(f"  {before - int(kept_new)} new row(s) rejected — an equal or better "
            f"source is already recorded")

    # ---- report -----------------------------------------------------------
    ld = pd.to_datetime(out["listing_date"], errors="coerce")
    cut = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
    recent = out[ld >= cut]
    n_cliff = (int(out["on_date_cliff"].astype("boolean").fillna(False).sum())
               if "on_date_cliff" in out.columns else 0)
    log(f"  TABLE: {len(out)} rows "
        f"({int(out['confidence'].eq(AUTHORITATIVE).sum())} authoritative, "
        f"{int(out['confidence'].eq(INFERRED).sum())} inferred, "
        f"{n_cliff} flagged on_date_cliff)")
    log(f"  listed in the last 365d: {len(recent)}")
    if not recent.empty and "exchange" in recent.columns:
        for ex, n in recent["exchange"].value_counts().items():
            log(f"      {ex}: {n}")

    if args.dry_run:
        log("[DRY-RUN] no Drive write.")
        if not recent.empty:
            newest = recent.assign(_d=ld[ld >= cut]).sort_values("_d", ascending=False)
            log("  newest 10: " + ", ".join(
                f"{r.symbol}({r.listing_date})" for r in newest.head(10).itertuples()))
        return

    owner = "listing_dates"
    if not acquire_lock(_drive(), idx, LOCK_NAME, owner,
                        wait_min=args.lock_wait_min):
        log("  lock busy — another run is writing; skipping this pass."); return
    try:
        _dc(lambda: save_parquet(_drive(), idx, TABLE, out), label=TABLE)
        log(f"  wrote {INDEX_PATH}/{TABLE}")
    finally:
        release_lock(_drive(), idx, LOCK_NAME)
    log(f"done in {(datetime.now() - started).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
