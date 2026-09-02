# Long-horizon price highs — `price_highs.parquet`

*Built 2026-09-02. Lives at `company_repo/_index/price_highs.parquet` on Drive.*

## What this is

One row per stock holding its highest price over 3, 5, 10, 20 and 25 years, plus
its all-time high and how much history backs each figure.

**233 KB for 4,374 stocks.**

## Why it exists

Phase 1's own price store cannot answer "how far below its high is this stock?"
in any consistent way.

| | Median history | Deepest | Clear 5 years |
|---|---|---|---|
| NSE | 6.9 years | 10.3 | 52% |
| **BSE** | **2.2 years** | **2.2** | **0%** |
| Overall | 2.2 years | 10.3 | **26%** |

Every BSE name sits at *exactly* 2.22 years — that is the hard cap in
`fetch_bse_only_ohlcv.py` (`"2y" if args.backfill`), not a coincidence. 114 of a
300-file sample share a first bar of 2024-06-12: the day that batch entered the
feed, not the day those companies listed.

Computing a "high" from that store means a 10-year high for one stock and an
8-month high for the next, **and the bias runs one way**: less history makes
"near its high" trivially easy to satisfy, so the screen fills with the newest
names. This is exactly the bug found in `guru/daily_screen.py`, where a 4-year
window was labelled `pct_from_all_time_high`.

## Why not just fetch deeper history?

We measured that too. Deepening every stock to 10 years costs **+239 MB** (237 MB
→ 476 MB), which is affordable — but it also means rewriting live price files on
the code path that once corrupted 59 parquets, and `ingest_ohlcv.py` cannot do it
anyway: `merge_and_upload` keeps only bars *newer* than what is stored, so
`--backfill` would run for an hour and deepen nothing.

**A high is a number, not a history.** Storing 25 years of bars to derive one
figure costs ~239 MB; storing the figure costs 233 KB — about **1/1000th**.

## Where the depth came from

`guru/data/ohlcv_hist/` — 5,062 files built during the backtest work: median 15
years, deepest 25.2, oldest bar 2001-07-03.

This was a **one-time extraction, not a runtime dependency.** Phase 1 owns the
resulting table and maintains it from its own daily feed. If the guru store
vanished tomorrow the table keeps working. That distinction is why this is
acceptable despite the plan's "do not take a dependency on guru's store".

### The join was the whole problem

Guru keys its files by `guru_key`. Matching those to Phase 1's symbols:

| Join key | Guru rows reached |
|---|---|
| `nse_symbol` alone | 2,386 — **and nothing on BSE** |
| `bse_scrip_id` | 4,847 |
| `isin` | 4,958 |
| **cascade: isin → nse_symbol → bse_scrip_id** | **5,017 → 88.8% of Phase 1's universe** |

Matching on symbol alone would have reached 42% of the universe and delivered
almost no gain over what we already had. ISIN is the key that makes this worth
doing.

## What it delivers

| Window | Stocks with a figure | Share of the 5,617-name universe |
|---|---|---|
| 3 years | 3,798 | 67.6% |
| **5 years** | **3,475** | **61.9%** |
| 10 years | 2,801 | 49.9% |
| 20 years | 1,619 | 28.8% |
| 25 years | 508 | 9.0% |

**5-year coverage goes from 26% to 62%** — 2.4× — for 233 KB and no fetching.

## How it is used

`compute_features.py` joins the table and computes `pct_from_high_{3,5,10,20,25}y`
and `pct_from_all_time_high` from it. The table wins wherever it has a figure;
the shallow window-based calculation survives only where it does not.

`history_years` is taken from the table too, because the **≥5-year floor**
(`ATH_MIN_YEARS`) is what decides whether a long-horizon signal may fire at all.
A stock below that floor produces no long-horizon signal — never a silent
fallback to a shorter window.

## Maintenance, and its one honest limitation

```bash
python scripts/build_price_highs.py --update
```

Runs daily in `daily.yml` between *Compute features* and *Market state*. It
raises any stored high that today's bar exceeded.

- **Exact** for `high_all_time` — a running maximum needs only one comparison.
- **Approximate** for the N-year windows. They decay: a high set 5.5 years ago
  should drop out of the 5-year window, and that cannot be done incrementally
  without the history. Between re-seeds an N-year high can only move **up**.

That error is in the safe direction — a stale, too-high figure makes a stock look
*further* from its high than it really is, so the screen under-fires rather than
over-fires. `as_of` records when each window was last truly computed.

**Re-seed periodically** to clear the drift:

```bash
python scripts/build_price_highs.py --seed --guru-dir <path-to>/guru/data/ohlcv_hist
```

`--seed` is local-only: `guru/data/` is gitignored, so it is absent from CI and
from a fresh clone. Always `--dry-run` first.

## Known gaps

- **1,246 of 5,617 Phase 1 symbols have no row** (77.8% joined). Mostly names in
  Phase 1's universe that guru's 2026-07 snapshot never carried.
- **BSE 25-year depth is inherited, not verified.** The figures come from guru's
  store; they were not re-fetched or cross-checked against the exchange.
- **The guru store is a snapshot** last refreshed 2026-08-28 and is not updated
  by CI. Barely matters for a 25-year high; the daily `--update` covers the
  recent end.
- **`as_of` drift is not yet alarmed.** Nothing warns when the windows are stale
  enough to matter. A re-seed cadence has not been decided.
