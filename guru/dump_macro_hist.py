r"""
P2c — dump_macro_hist.py  (Project Guru, STANDALONE)

One-time ~25y daily history for indices / FX / commodities via yfinance.
Ticker map borrowed from scripts/ingest_indices_macro.py (read-only reuse);
writes ONLY guru/data/macro_hist/<NAME>.parquet + _macro_coverage.txt.

Small enough to refetch whole every time — a per-row resume ledger is overkill;
rerunning simply overwrites (idempotent). Sensex added for pre-2007 regime
coverage (Yahoo's ^NSEI only starts 2007; ^BSESN reaches 2001 in our calendar).

Usage:
    python guru/dump_macro_hist.py --dry-run
    python guru/dump_macro_hist.py
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_DIR = os.path.join(GURU_DIR, "data", "macro_hist")

SERIES = {
    # Indian indices
    "NIFTY_50": "^NSEI", "NIFTY_BANK": "^NSEBANK", "INDIA_VIX": "^INDIAVIX",
    "NIFTY_500": "^CRSLDX", "NIFTY_MIDCAP_100": "^CNXMIDCAP",
    "NIFTY_SMALLCAP_100": "^CNXSC", "NIFTY_IT": "^CNXIT", "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_PHARMA": "^CNXPHARMA", "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_METAL": "^CNXMETAL", "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_ENERGY": "^CNXENERGY", "NIFTY_INFRA": "^CNXINFRA",
    "SENSEX": "^BSESN",            # pre-2007 Indian-market regime anchor
    # Macro
    "USD_INR": "INR=X", "BRENT_CRUDE": "BZ=F", "WTI_CRUDE": "CL=F", "GOLD": "GC=F",
    "DOW_JONES": "^DJI", "NASDAQ": "^IXIC", "SP500": "^GSPC",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        log(f"DRY RUN — would fetch {len(SERIES)} series (25y daily) into {MACRO_DIR}:")
        for name, tkr in SERIES.items():
            print(f"   {name:<20} <- {tkr}")
        return

    import yfinance as yf
    os.makedirs(MACRO_DIR, exist_ok=True)
    report = []
    for name, tkr in SERIES.items():
        try:
            h = yf.download(tkr, period="25y", interval="1d", progress=False,
                            auto_adjust=True)
            if h is None or h.empty:
                report.append((name, tkr, "EMPTY", 0, "", ""))
                log(f"{name}: EMPTY")
                continue
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.get_level_values(0)
            h = h.reset_index()
            h.columns = [str(c).lower() for c in h.columns]
            h = h.rename(columns={"index": "date"})
            h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None).dt.normalize()
            h = h[["date", "open", "high", "low", "close", "volume"]].dropna(
                subset=["close"])
            h.to_parquet(os.path.join(MACRO_DIR, f"{name}.parquet"), index=False)
            report.append((name, tkr, "ok", len(h),
                           str(h['date'].iloc[0].date()), str(h['date'].iloc[-1].date())))
            log(f"{name}: {len(h)} bars {h['date'].iloc[0].date()} -> "
                f"{h['date'].iloc[-1].date()}")
        except Exception as e:
            report.append((name, tkr, f"ERR {str(e)[:60]}", 0, "", ""))
            log(f"{name}: ERROR {str(e)[:80]}")

    rep = pd.DataFrame(report, columns=["name", "ticker", "status", "bars",
                                        "first", "last"])
    txt = ("Project Guru - macro/indices dump coverage (P2c)\n"
           f"generated {datetime.now().isoformat(timespec='seconds')}\n\n"
           + rep.to_string(index=False))
    with open(os.path.join(GURU_DIR, "data", "_macro_coverage.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print()
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
