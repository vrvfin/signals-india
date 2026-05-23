# Indian Equity Signal System — Design Document

**Version:** 0.4
**Owner:** va
**Last updated:** 2026-05-21
**Status:** Live — Phase 1 (price/volume system) complete; Phase 2 next.

> This is the **design** doc — goals, architecture, decisions, rationale.
> `PROJECT_STATUS.md` is the companion **current-state** doc (what's built /
> pending). `RUNBOOK.md` covers failure recovery. To resume work in a new chat,
> paste this file + `PROJECT_STATUS.md` + one `strategy_*.py` as a template.

---

## 1. Goals

A daily, automated trading-signal system for Indian listed equities, built on a
family of independently-pluggable strategies. Concretely:

- Produce daily, ranked, end-of-day signals across multiple methodologies
  (Qullamaggie, momentum, MA-respect, Minervini, Darvas, CANSLIM, PEAD, volume).
- Cover the broad NSE universe (~2,070 liquid symbols after filters); all
  filters configurable.
- Persist full price history (Google Drive) so features can be recomputed and
  strategies backtested.
- Deliver results through a Streamlit dashboard — per-stock charts, market-state
  overview, per-signal explainability.
- Run automatically every trading day on GitHub Actions, with no manual steps,
  so it keeps working while the user travels.
- Be **modular**: a new strategy is a new file; nothing existing changes.
- Be **transparent**: every run is logged; a health-check verifies output
  freshness and surfaces breakage.
- Emit **actionable zones** per signal (buy / add / hold / stop_loss / exit /
  sell) and a cross-strategy **Multi-Strategy Conviction** view.

## 2. Non-goals (v1)

No intraday signals (EOD only). No automated order placement — signals only.
No options / futures / commodities. No multi-user accounts or hosting for
others. Outputs are research artifacts, not investment advice.

---

## 3. Signal model & zone markers

Each strategy, on flagging a stock, emits a row with a `zone_type`:

| zone_type | meaning                          | chart rendering        |
|-----------|----------------------------------|------------------------|
| buy       | entry candidate                  | green band             |
| add       | high-conviction continuation     | teal up-arrow          |
| hold      | constructive, no fresh entry     | grey dot               |
| stop_loss | protective stop level            | red line               |
| exit      | over-extension / take-profit     | orange line            |
| sell      | breakdown / exit signal          | red band               |

Standard per-strategy CSV schema: `symbol, date, strategy, zone_type, score,
entry, stop, reason` plus strategy-specific extras. The dashboard unions zone
markers from every strategy onto each stock's chart, and the `reason` column
drives per-signal explainability.

---

## 4. Architecture & data flow

```
GitHub Actions  (daily 17:30 IST Mon-Fri; Monday run also does weekly steps)
  build_universe -> ingest_ohlcv -> ingest_indices_macro -> fix_indices_nse
                 -> ingest_fii_dii -> compute_features
                 -> 8 strategy scripts -> aggregate_signals -> market_state
                 -> pipeline_healthcheck   (truth gate - only gating step)
  weekly (Mon):  ingest_fundamentals, enrich_market_cap
        |
        v  every stage reads from / writes to Google Drive
Google Drive  signals-india/
        |
        v
Streamlit dashboard (app.py) - reads Drive, 5-min cache, live on Streamlit Cloud
```

Key idea: each stage reads from Drive and writes to Drive, and is independently
runnable. One stage failing does not stop the others — `continue-on-error` is
set on every workflow step except the final health-check, which is the single
pass/fail gate.

---

## 5. Storage layout (Google Drive)

```
signals-india/
  data/
    ohlcv/            one parquet per symbol (RELIANCE.parquet, ...)
    indices/          NIFTY_50, NIFTY_500, NIFTY_MIDCAP_100, ...
    macro/            USD_INR, BRENT_CRUDE, ..., FII_DII.csv
    market_state/     latest.parquet, history.csv, sector_rotation
  universe/
    master_list.csv   tradeable universe (~2,365 symbols)
    market_cap.csv    symbol, market_cap_cr, mcap_segment
  features/
    latest.parquet    one row per symbol, ~48 columns
  signals/
    per_strategy/<name>/latest.csv   (+ dated CSV)
    aggregated/{latest,conviction,diff_vs_yesterday}.csv
  fundamentals/summary.parquet
  logs/health/latest.json
```

Storage is a personal Google Drive folder (15 GB free, well under quota).
Auth is **OAuth user-delegation** (Desktop client), not a service account —
service accounts have no storage quota on a personal Drive. The app runs in
Google "Production" mode so refresh tokens do not expire weekly.

History retention: OHLCV kept from listing date; signals and logs kept forever
(tiny); old dated snapshots pruned beyond ~90 days.

---

## 6. Configuration convention

There is **no central `config.yaml`** (it was in the original design but never
built). In practice each strategy carries its own clearly-marked config
constants at the top of its file — e.g. `strategy_volume.py` has a `CONFIG`
block, `strategy_momentum.py` uses function-default parameters. This keeps the
modularity rule intact: a strategy is one self-contained file. Tuning a strategy
is a one-line edit inside that file.

Health-check thresholds live in `pipeline_healthcheck.py`
(`MIN_FEATURE_ROWS`, `OHLCV_MAX_STALE_DAYS`, `FRESH_WINDOW_HOURS`); the schedule
lives in `daily.yml`.

---

## 7. Pipeline stages

**Data pull**
- `build_universe.py` — NSE EQUITY_L.csv -> `universe/master_list.csv`.
- `ingest_ohlcv.py` — per-symbol OHLCV via batched `yf.download` (yfinance >= 1.3.0).
- `ingest_indices_macro.py` — index OHLCV + macro series.
- `fix_indices_nse.py` — NIFTY MIDCAP 100 / SMALLCAP 100 via the NSE index
  bhavcopy on `archives.nseindia.com` (yfinance tickers for these are broken;
  the indicesHistory API is bot-blocked).
- `ingest_fii_dii.py` — FII/DII cash flows -> `data/macro/FII_DII.csv`.
- `ingest_fundamentals.py` (+ `screener_client.py`) — screener.in scrape,
  cookie-authenticated -> `fundamentals/summary.parquet`. Weekly.
- `enrich_market_cap.py` — yfinance market cap -> `universe/market_cap.csv`
  with a segment label (Largecap / Midcap / Smallcap / Microcap). Weekly.

**Features** — `compute_features.py` computes ~48 features per symbol into
`features/latest.parquet`: EMAs (10/20/50/100/200), SMAs (50/200), ATR(14),
ADR%(20), 52-week high/low + distances, returns 1d/1m/2m/3m/6m/12m, RS rank
(cross-sectional percentile) per lookback, `vol_today_ratio`, consecutive
`days_above_ema_*`, trend flags, `prior_close`. Sub-Rs.10 names are dropped.

**Aggregation** — `aggregate_signals.py` auto-discovers every
`signals/per_strategy/*/latest.csv`, builds the unified table, the
Multi-Strategy Conviction list (flagged by >=2 strategies), and a
diff-vs-yesterday.

**Market state** — `market_state.py` computes a 6-component Market Health Score
(0-100) and sector rotation.

**Health-check** — `pipeline_healthcheck.py` runs last, verifies output
freshness (features < 24 h and >= 1,500 rows; OHLCV sample bars recent; each
strategy and index file fresh), writes `logs/health/latest.json`, and exits
non-zero on a CRITICAL failure — turning the run RED and emailing the user.

---

## 8. Strategies (all built)

Each is one file, writing `signals/per_strategy/<name>/latest.csv`.

- **Momentum** (`strategy_momentum.py`) — 5 timeframes (1m/2m/3m/6m/12m); buy =
  top-decile RS rank above 200 SMA; add = also near 52w high.
- **MA-respect** (`strategy_ma_respect.py`) — configurable family; default
  20EMA-30d, 20EMA-60d, 50EMA-60d; flags clean uninterrupted trends.
- **Qullamaggie** — prior run-up + tight consolidation + volume-confirmed breakout.
- **Minervini** — 8-point SEPA trend template.
- **Darvas** — tight box near 52w highs, breakout trigger.
- **CANSLIM** — 7-rule screen, needs fundamentals.
- **PEAD** — post-earnings-announcement drift.
- **Volume** (`strategy_volume.py`) — `volume_breakout` (volume surge + uptrend
  near highs) and `volume_vcp` (volume dry-up in a tight base — pre-breakout coil).

---

## 9. Dashboard (`app.py`, live on Streamlit Cloud)

Six pages: **Market Overview** (macro strip, FII/DII panel, Market Health
Score, breadth, market-cap segment filter, Nifty chart, sector rotation),
**Today's Signals** (filterable table), **Graphs** (conviction-ordered,
paginated chart gallery with multi-strategy chips per stock), **Stock Detail**
(3-panel chart + per-signal explainability — intent, rules, reason),
**My Portfolio** (reads a Screener "My Investments" .xls export, ISIN-matched),
**Strategy Docs**.

A data-freshness banner + sidebar status read `logs/health/latest.json`. All
Drive calls go through a `_drive_call` retry wrapper that rebuilds the
connection on transient TLS / `BrokenPipeError` failures.

Cloud auth uses the same OAuth credentials passed as Streamlit Cloud secrets.

---

## 10. Monitoring & failure handling

- Every workflow step except the health-check has `continue-on-error: true`, so
  one failure never stops the pipeline.
- `pipeline_healthcheck.py` is the truth gate: it independently verifies the
  output files on Drive, so even a silently-failing step is caught.
- A RED run emails the user (GitHub -> "failed workflows only").
- `RUNBOOK.md` documents the five common failure symptoms (OAuth/Drive auth,
  Screener cookie expiry, data-source block, a strategy warning, Drive filling
  up) with ~2-minute fixes each.

---

## 11. Auto-run

GitHub Actions, `.github/workflows/daily.yml`. Two cron entries: `0 12 * * 1`
(Monday 17:30 IST — includes the weekly fundamentals + market-cap steps) and
`0 12 * * 2-5` (Tue-Fri). Also `workflow_dispatch` for manual runs. Strategies
are isolated into separate steps so one failing doesn't block the others.

---

## 12. Adding a new strategy — the recipe

1. Create `scripts/strategy_<name>.py` (copy an existing one). It writes
   `signals/per_strategy/<name>/latest.csv` with the standard schema (sec. 3).
2. Add a `Strategy - <Name>` step to `daily.yml` with `continue-on-error: true`.
3. Add a `STRATEGY_DOCS` entry in `app.py` keyed by the strategy name.

`aggregate_signals.py`, `pipeline_healthcheck.py`, and the dashboard Graphs page
auto-discover the new folder — no edits needed there.

---

## 13. Key design decisions (decision log)

- **Universe:** NSE + BSE intended; NSE prioritised on dual-listed names. Built
  NSE-only (~2,070 symbols with sufficient history); BSE deferred.
- **Data sources:** free only. yfinance for OHLCV; NSE archives bhavcopy for the
  two broken indices; NSE public pages for FII/DII; screener.in (user's
  subscription) for fundamentals. TradingView used only for visual verification.
- **Auth:** OAuth user-delegation, not a service account (personal-Drive quota).
- **Storage:** Google Drive — visible, free, simple.
- **Repo:** private GitHub repo.
- **Modularity:** new strategy = new file; nothing existing is edited.
- **Process:** stage-gate sign-off, no fixed weekly timeline — each stage moves
  on the user's explicit approval.
- **Dashboard flow:** macro-first (Market Overview is the landing page), then
  drill into stocks.
- **Portfolio:** tracked via a Screener .xls export the user uploads; cost / P&L
  data is never exposed to git or any LLM.
- **ASM/GSM:** surveillance stocks are included, flagged not excluded.
- **Security (binding):** code never accesses passwords or PII; secrets live
  only in GitHub Secrets / Streamlit secrets / local `.env` (gitignored) and are
  read via `os.environ`.
- **LLM summaries:** deferred to Phase 2, using Gemini's free tier.

---

## 14. Progress Log

Most-recent first. `YYYY-MM-DD | author | summary`.

### 2026-05-21 | va | Stage 12 + monitoring + dashboard hardening — Phase 1 complete
Volume strategy (volume_breakout + volume_vcp) added — closes the original
8-strategy list. `pipeline_healthcheck.py` truth-gate + `RUNBOOK.md` added.
`daily.yml` reworked: continue-on-error on every step except the health-check,
isolated strategy steps, two-cron schedule (fixes a latent bug where weekly
fundamentals never ran). `fix_indices_nse.py` rewritten to the NSE index
bhavcopy. Dashboard: Graphs page (conviction-ordered, paginated chart gallery),
FII/DII panel, data-freshness banner, `_drive_call` Drive retry wrapper.
Streamlit Cloud deployed live. `compute_features.py`: confirmed penny-stock
filter + added `return_1d_pct` / `prior_close`. `enrich_market_cap.py` run.
`PROJECT_STATUS.md` created as the live current-state handover doc.

### 2026-05-18..21 | va | Stages 8-11 (dashboard, aggregation, cloud, fundamentals)
Streamlit dashboard live. `aggregate_signals.py` — unified table,
Multi-Strategy Conviction, diff-vs-yesterday, auto-discovers strategy folders.
Cloud auto-run on GitHub Actions confirmed end-to-end. Fundamentals scrape
(2,363 ok) + CANSLIM + PEAD.

### 2026-05-18 | va | Stage 7 (Market State + Health Score) approved
6-component health score live (Nifty trend, breadth, highs/lows, VIX, FII,
A/D). Sector rotation working. Outputs to `data/market_state/`.

### 2026-05-17 | va | Stages 3-6 approved
Stage 3 feature engine (~2,070 symbols x ~46 cols). Stage 4 momentum (5
timeframes). Stage 5 MA-respect (3 instances). Stage 6 Qullamaggie. Stages 6b/6c
Minervini + Darvas.

### 2026-05-16 | va | Stages 1-2 approved
Infra (conda env, private repo, OAuth Drive). Universe build (2,365 symbols).
Bulk OHLCV ingest (~2.5M rows). Switched OHLCV source jugaad-data -> yfinance.

### 2026-05-14..15 | Claude + va | Design v0.1 -> v0.3
Goals, assumptions, architecture, storage, config, stage-gate plan. Signal
zone-marker spec. 21 design questions resolved.

---

## 15. Phase 2 roadmap (next)

- Company repository: business overview, detailed reports, concall summaries.
- 12-quarter financials + valuation overlays on stock charts.
- LLM narrative summaries (Gemini free tier).
- Remaining dashboard pages: dedicated Multi-Strategy Conviction page,
  Watchlist, Backtest, Build Journal.
- Telegram alerts.

---

## 16. Disclaimer

This system produces analytical research artifacts. It does not constitute
investment advice, and no signal it generates is a recommendation to buy or sell
any security. The user is solely responsible for trading decisions. Indian
securities trading involves risk of loss.
