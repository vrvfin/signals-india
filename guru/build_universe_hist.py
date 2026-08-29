r"""
P1 — build_universe_hist.py  (Project Guru, STANDALONE pipeline)

Builds the point-in-time historical universe INCLUDING delisted companies, plus the
master trading calendar. Everything writes LOCALLY under guru/data/ — no Drive, no
shared tables, no Phase-1/2/3 coupling (spec §10.5 standalone constraint).

Outputs
-------
guru/data/universe_hist.parquet
    One row per security. Columns:
      guru_key   <- STABLE join key for ALL guru tables: valid ISIN, else BSE<code>,
                    else NSE:<symbol>. Price files are stored as <guru_key>.parquet,
                    so rows WITHOUT an ISIN still join price history cleanly.
      yf_ticker  <- price-fetch ticker: <nse_symbol>.NS when NSE-listed (incl. NSE
                    delisted names — Yahoo often retains their history), else
                    <bse_code>.BO.
      isin, bse_code, bse_symbol, nse_symbol, name, exchanges (NSE|BSE|NSE+BSE),
      bse_group, bse_status (Active/Delisted/Suspended), nse_active (bool),
      listing_date (NSE where known), delisting_flag (bool),
      nse_delisting_date, nse_delisting_type (from NSE delisted.csv), source, fetched_at
    Delisted/suspended BSE securities AND NSE-only delisted names (delisted.csv)
    ARE included — that is the survivorship fix.
guru/data/trading_calendar.parquet
    Union of NSE (^NSEI) and BSE (^BSESN) trading days over the last ~25 years,
    columns: date, nse_open, bse_open.
guru/data/_universe_coverage.txt
    Human-readable coverage report (counts per source/status).

Sources (all public, no auth)
-----------------------------
1. BSE ListofScripData API — status=Active / Deactive (delisted+suspended), segment
   Equity. Gives scrip code, ISIN, name, group, status. This is the backbone: BSE
   historically carries nearly every Indian listed company incl. dead ones.
2. NSE EQUITY_L.csv (nsearchives) — active NSE mainboard: symbol, ISIN, listing date.
3. yfinance ^NSEI / ^BSESN — trading-day calendar.

Usage
-----
    python guru/build_universe_hist.py --dry-run     # no writes, fetch + report only
    python guru/build_universe_hist.py               # full build
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
BSE_HDR = {"User-Agent": UA, "Accept": "application/json, text/plain, */*",
           "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"}
NSE_HDR = {"User-Agent": UA, "Accept": "*/*"}

BSE_LIST_URL = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
                "?Group=&Scripcode=&industry=&segment=Equity&status={status}")
NSE_EQUITY_L = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_DELISTED = "https://nsearchives.nseindia.com/content/equities/delisted.csv"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_bse_scrips(status: str) -> pd.DataFrame:
    """status: 'Active', 'Delisted' or 'Suspended' (verified live 2026-07-03)."""
    url = BSE_LIST_URL.format(status=status)
    r = requests.get(url, headers=BSE_HDR, timeout=60)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"BSE ListofScripData returned 0 rows for status={status}")
    # Observed schema: SCRIP_CD, Scrip_Name, Status, GROUP, FACE_VALUE, ISIN_NUMBER,
    # INDUSTRY, scrip_id (symbol), Segment, NSURL, Issuer_Name, Mktcap
    df = df.rename(columns={
        "SCRIP_CD": "bse_code", "Scrip_Name": "bse_symbol", "ISIN_NUMBER": "isin",
        "Issuer_Name": "name", "GROUP": "bse_group", "Status": "bse_status",
        "scrip_id": "bse_scrip_id",
    })
    keep = [c for c in ["bse_code", "bse_symbol", "bse_scrip_id", "isin", "name",
                        "bse_group", "bse_status"] if c in df.columns]
    df = df[keep].copy()
    df["bse_code"] = df["bse_code"].astype(str).str.strip()
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    log(f"BSE {status}: {len(df)} scrips")
    return df


def fetch_nse_equity_l() -> pd.DataFrame:
    r = requests.get(NSE_EQUITY_L, headers=NSE_HDR, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "SYMBOL": "nse_symbol", "NAME OF COMPANY": "name_nse",
        "ISIN NUMBER": "isin", "DATE OF LISTING": "listing_date", "SERIES": "nse_series",
    })
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    df["nse_symbol"] = df["nse_symbol"].astype(str).str.strip()
    df["listing_date"] = pd.to_datetime(df["listing_date"], format="%d-%b-%Y",
                                        errors="coerce")
    log(f"NSE EQUITY_L: {len(df)} symbols")
    return df[["nse_symbol", "name_nse", "isin", "listing_date", "nse_series"]]


def fetch_nse_delisted() -> pd.DataFrame:
    r = requests.get(NSE_DELISTED, headers=NSE_HDR, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Symbol": "nse_symbol", "Company": "name_nse_del",
                            "Delisted Date": "nse_delisting_date",
                            "Type of Delisting": "nse_delisting_type"})
    df = df[[c for c in ["nse_symbol", "name_nse_del", "nse_delisting_date",
                         "nse_delisting_type"] if c in df.columns]].copy()
    df["nse_symbol"] = df["nse_symbol"].astype(str).str.strip()
    df = df[df["nse_symbol"].ne("") & df["nse_symbol"].notna()]
    df["nse_delisting_date"] = pd.to_datetime(df["nse_delisting_date"],
                                              format="%d-%b-%y", errors="coerce")
    df["nse_delisting_type"] = df["nse_delisting_type"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["nse_symbol"], keep="last")
    log(f"NSE delisted.csv: {len(df)} symbols")
    return df


def build_universe() -> pd.DataFrame:
    parts = []
    for status in ["Active", "Delisted", "Suspended"]:
        parts.append(fetch_bse_scrips(status))
        time.sleep(1)
    bse = pd.concat(parts, ignore_index=True)
    bse = bse.drop_duplicates(subset=["bse_code"], keep="first")

    nse = fetch_nse_equity_l()

    # Merge on ISIN where both have a real ISIN; BSE rows without ISIN stay as-is.
    valid_isin = bse["isin"].str.match(r"^IN[A-Z0-9]{10}$", na=False)
    log(f"BSE rows with valid ISIN: {valid_isin.sum()} / {len(bse)}")
    uni = bse.merge(nse, on="isin", how="outer", indicator=True)

    uni["exchanges"] = uni["_merge"].map(
        {"left_only": "BSE", "right_only": "NSE", "both": "NSE+BSE"})
    uni = uni.drop(columns=["_merge"])
    uni["name"] = uni["name"].fillna(uni["name_nse"])
    uni = uni.drop(columns=["name_nse"])
    uni["nse_active"] = uni["nse_symbol"].notna()
    uni["bse_status"] = uni["bse_status"].fillna("")
    uni["delisting_flag"] = uni["bse_status"].str.lower().isin(
        ["delisted", "deactive", "suspended"]) & ~uni["nse_active"]
    # --- NSE delisted names (delisted.csv): enrich matches, append the rest ---
    nse_del = fetch_nse_delisted()
    uni = uni.merge(nse_del, on="nse_symbol", how="left")
    matched = uni["nse_delisting_date"].notna().sum()
    extra = nse_del[~nse_del["nse_symbol"].isin(uni["nse_symbol"].dropna())].copy()
    log(f"NSE delisted: {matched} matched existing rows, {len(extra)} NSE-only additions")
    if not extra.empty:
        extra = extra.rename(columns={"name_nse_del": "name"})
        extra["exchanges"] = "NSE"
        extra["nse_active"] = False
        extra["bse_status"] = ""
        extra["delisting_flag"] = True
        uni = pd.concat([uni, extra], ignore_index=True)
    uni = uni.drop(columns=["name_nse_del"], errors="ignore")
    # NSE-delisted rows that never traded on BSE: flag them delisted too
    uni.loc[uni["nse_delisting_date"].notna() & ~uni["nse_active"].fillna(False),
            "delisting_flag"] = True

    # --- guru_key: THE stable join key for every guru table (price files use it) ---
    valid = uni["isin"].str.match(r"^IN[A-Z0-9]{10}$", na=False)
    bse_ok = uni["bse_code"].notna() & uni["bse_code"].astype(str).str.strip().ne("") \
        & uni["bse_code"].astype(str).str.lower().ne("nan")
    uni["guru_key"] = None
    uni.loc[valid, "guru_key"] = uni.loc[valid, "isin"]
    uni.loc[~valid & bse_ok, "guru_key"] = "BSE" + uni.loc[~valid & bse_ok,
                                                           "bse_code"].astype(str)
    rest = uni["guru_key"].isna() & uni["nse_symbol"].notna()
    uni.loc[rest, "guru_key"] = "NSE:" + uni.loc[rest, "nse_symbol"]
    uni = uni[uni["guru_key"].notna()].copy()
    dups = uni["guru_key"].duplicated().sum()
    if dups:
        log(f"WARN: {dups} duplicate guru_keys — keeping first (NSE+BSE rows win)")
        uni = uni.sort_values("exchanges").drop_duplicates("guru_key", keep="first")

    # --- yf_ticker: how the price dump fetches each row ---
    has_nse = uni["nse_symbol"].notna() & uni["nse_symbol"].astype(str).str.strip().ne("")
    uni["yf_ticker"] = None
    uni.loc[has_nse, "yf_ticker"] = uni.loc[has_nse, "nse_symbol"].astype(str) + ".NS"
    bse_fallback = ~has_nse & bse_ok
    uni.loc[bse_fallback, "yf_ticker"] = uni.loc[bse_fallback,
                                                 "bse_code"].astype(str) + ".BO"

    uni["source"] = "bse_listofscrip+nse_equity_l+nse_delisted_csv"
    uni["fetched_at"] = datetime.utcnow().isoformat(timespec="seconds")
    return uni


def build_calendar() -> pd.DataFrame:
    import yfinance as yf
    frames = {}
    for label, ticker in [("nse_open", "^NSEI"), ("bse_open", "^BSESN")]:
        h = yf.download(ticker, period="25y", interval="1d", progress=False,
                        auto_adjust=True)
        if h.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        idx = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        frames[label] = pd.Series(True, index=idx)
        log(f"{ticker}: {len(idx)} trading days ({idx.min().date()} -> {idx.max().date()})")
    cal = pd.DataFrame(frames).astype("boolean").fillna(False).astype(bool).reset_index()
    cal.columns = ["date", "nse_open", "bse_open"]
    return cal.sort_values("date").reset_index(drop=True)


def coverage_report(uni: pd.DataFrame, cal: pd.DataFrame) -> str:
    lines = ["Project Guru — universe_hist coverage report",
             f"generated {datetime.now().isoformat(timespec='seconds')}", ""]
    lines.append(f"total securities: {len(uni)}")
    lines.append(f"by exchange: {uni['exchanges'].value_counts().to_dict()}")
    lines.append(f"by bse_status: {uni['bse_status'].value_counts().to_dict()}")
    lines.append(f"delisting_flag=True (survivorship additions): {int(uni['delisting_flag'].sum())}")
    lines.append(f"valid ISIN: {uni['isin'].str.match(r'^IN[A-Z0-9]{10}$', na=False).sum()}")
    lines.append(f"guru_key coverage: {uni['guru_key'].notna().sum()} / {len(uni)} "
                 f"(isin={uni['guru_key'].str.startswith('IN').sum()}, "
                 f"bse={uni['guru_key'].str.startswith('BSE').sum()}, "
                 f"nse={uni['guru_key'].str.startswith('NSE:').sum()})")
    lines.append(f"yf_ticker coverage: {uni['yf_ticker'].notna().sum()} / {len(uni)} "
                 f"(.NS={uni['yf_ticker'].str.endswith('.NS').fillna(False).sum()}, "
                 f".BO={uni['yf_ticker'].str.endswith('.BO').fillna(False).sum()})")
    lines.append(f"NSE delisting_date known: {uni['nse_delisting_date'].notna().sum()}")
    lines.append(f"NSE listing_date known: {uni['listing_date'].notna().sum()}")
    lines.append("")
    lines.append(f"calendar days: {len(cal)}  span {cal['date'].min().date()} -> {cal['date'].max().date()}")
    lines.append(f"NSE-only days: {int((cal['nse_open'] & ~cal['bse_open']).sum())}, "
                 f"BSE-only days: {int((cal['bse_open'] & ~cal['nse_open']).sum())}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + print coverage, write NOTHING")
    args = ap.parse_args()

    uni = build_universe()
    cal = build_calendar()
    report = coverage_report(uni, cal)
    print("\n" + report + "\n")

    if args.dry_run:
        log("DRY RUN — nothing written.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    uni.to_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"), index=False)
    cal.to_parquet(os.path.join(DATA_DIR, "trading_calendar.parquet"), index=False)
    with open(os.path.join(DATA_DIR, "_universe_coverage.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    log(f"written: {DATA_DIR}\\universe_hist.parquet ({len(uni)} rows), "
        f"trading_calendar.parquet ({len(cal)} rows), _universe_coverage.txt")


if __name__ == "__main__":
    main()
