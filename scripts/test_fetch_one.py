"""
Stage 2b — Verify single-symbol OHLCV fetch via yfinance.

Usage:
    python scripts/test_fetch_one.py             # defaults to RELIANCE, last 1 year
    python scripts/test_fetch_one.py TCS         # any NSE symbol
    python scripts/test_fetch_one.py TCS 5       # symbol + years of history

yfinance is used because NSE rate-limits jugaad-data's User-Agent. Yahoo Finance
aggregates the same OHLCV (split-adjusted by default) and is far more reliable.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import yfinance as yf


def main() -> None:
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "RELIANCE"
    years = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    yf_ticker = f"{symbol}.NS"

    print(f"yfinance version: {yf.__version__}")
    print(f"Fetching {yf_ticker} | period={years}y")
    print("-" * 60)

    # Try Ticker.history first
    df = yf.Ticker(yf_ticker).history(period=f"{years}y", auto_adjust=True)

    # Fallback to yf.download if empty
    if df is None or len(df) == 0:
        print("Ticker.history returned empty; trying yf.download…")
        df = yf.download(yf_ticker, period=f"{years}y",
                         auto_adjust=True, progress=False)

    if df is None or len(df) == 0:
        print(f"ERROR: No data returned for {yf_ticker} via either method.")
        print("Diagnostic — trying AAPL (US sanity check):")
        sanity = yf.download("AAPL", period="1mo", progress=False)
        print(f"  AAPL rows: {len(sanity)}")
        if len(sanity) > 0:
            print("  → yfinance works; problem is NSE-specific. We'll need NSE bhavcopy as fallback.")
        else:
            print("  → yfinance broken entirely. Try `pip install -U yfinance` again.")
        sys.exit(1)

    df = df.reset_index()  # move Date out of the index
    df.columns = [c.lower() for c in df.columns]

    print(f"Rows           : {len(df)}")
    print(f"Columns        : {list(df.columns)}")
    print(f"Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Last close     : ₹{df['close'].iloc[-1]:,.2f}")
    print(f"Period high    : ₹{df['high'].max():,.2f}")
    print(f"Period low     : ₹{df['low'].min():,.2f}")
    print(f"Avg daily vol  : {df['volume'].mean():,.0f}")
    print("\nLast 5 rows:")
    print(df.tail()[["date", "open", "high", "low", "close", "volume"]].to_string(index=False))


if __name__ == "__main__":
    main()