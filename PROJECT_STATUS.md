  # PROJECT_STATUS — signals-india

**Purpose of this file.** Single source of truth for *current state*. DESIGN.md
records what was *designed*; this file records what is *built, broken, and pending*.
Keep it in the repo root. To resume work in any future chat session, paste this
file plus one existing `strategy_*.py` as a template — that is enough context.

**Last updated:** 2026-05-21
**Repo:** github.com/vrvfin/signals-india (private)
**Local:** `D:\EMA_Screener\claude\signals-india`  ·  conda env `signals-india` (Python 3.11)
**Storage:** Google Drive folder `signals-india/` (OAuth, app in Production mode)
**Version**: 0.4 and **Last updated**: 2026-05-21, **Status**: Live — Stage 12 complete, monitoring layer added.  
---

## 1. What this project is

A daily, automated trading-signal system for the Indian equity market (NSE
universe, ~2,070 liquid symbols). It pulls end-of-day data, computes features,
runs a family of independent strategy engines, aggregates them, and presents
everything in a Streamlit dashboard. Signals only — no order execution. Runs
unattended on GitHub Actions so it works while the user travels.

Original principles (from DESIGN.md, still binding): modular (a new strategy is
a new file, nothing existing changes); free tiers only; no PII / passwords ever
touched by code; end-of-day positional + swing horizon; stage-gate sign-off.

---

## 2. Architecture / data flow

```
GitHub Actions (daily, 17:30 IST, Mon-Fri)
  build_universe → ingest_ohlcv → ingest_indices_macro → fix_indices_nse
                 → ingest_fii_dii → compute_features
                 → 8 strategy scripts → aggregate_signals → market_state
                 → pipeline_healthcheck  (truth gate)
        │
        ▼  all reads/writes go to Google Drive
Drive: data/ohlcv, data/indices, data/macro, data/market_state,
       features/latest.parquet, signals/per_strategy/<name>/latest.csv,
       signals/aggregated/, fundamentals/, logs/health/latest.json
        │
        ▼
Streamlit dashboard (app.py) reads Drive, 5-min cache
```

Every stage reads from Drive and writes to Drive; stages are independently
runnable. One stage failing does not stop the others (see §6 monitoring).

---

## 3. File map

**Data pull**
- `build_universe.py` — NSE EQUITY_L.csv → `universe/master_list.csv` (~2,365 symbols).
- `ingest_ohlcv.py` — per-symbol OHLCV via yfinance (batched `yf.download`). Needs yfinance ≥ 1.3.0.
- `ingest_indices_macro.py` — index OHLCV + macro series (USD/INR, Brent, Dow, Nasdaq, VIX…).
- `fix_indices_nse.py` — NIFTY MIDCAP 100 / SMALLCAP 100 via NSE **index bhavcopy** (`archives.nseindia.com`). yfinance tickers for these are broken; the old indicesHistory API is blocked.
- `ingest_fii_dii.py` — FII/DII cash flows from NSE → `data/macro/FII_DII.csv`.
- `ingest_fundamentals.py` + `screener_client.py` — screener.in scrape (cookie auth) → `fundamentals/summary.parquet`. Weekly (Mondays).
- `enrich_market_cap.py` — market-cap segment tagging → `universe/market_cap.csv`. **Not run / not in workflow yet.**

**Features**
- `compute_features.py` — ~46 features per symbol → `features/latest.parquet`. EMAs/SMAs, ATR, ADR%, 52w high/low, returns 1/2/3/6/12m, RS rank, `vol_today_ratio`, `days_above_ema_*`, trend flags.

**Strategy engines** (each writes `signals/per_strategy/<name>/latest.csv` + dated CSV)
- `strategy_momentum.py` → momentum_1m / 2m / 3m / 6m / 12m
- `strategy_ma_respect.py` → ma_respect_20ema_30d / 20ema_60d / 50ema_60d
- `strategy_qullamaggie.py` → qullamaggie
- `strategy_minervini.py` → minervini
- `strategy_darvas.py` → darvas
- `strategy_canslim.py` → canslim (needs fundamentals)
- `strategy_pead.py` → pead
- `strategy_volume.py` → volume_breakout + volume_vcp  *(newest, Stage 12)*

**Aggregate / state / monitoring**
- `aggregate_signals.py` — unions all `per_strategy/*/latest.csv` (**auto-discovers** — no strategy list), builds `signals/aggregated/{latest,conviction,diff_vs_yesterday}.csv`. Multi-Strategy Conviction = flagged by ≥2 strategies.
- `market_state.py` — 6-component Market Health Score (0-100) + sector rotation → `data/market_state/`.
- `pipeline_healthcheck.py` — final workflow step ("truth gate"). Verifies output freshness; writes `logs/health/latest.json`; exits non-zero on CRITICAL → workflow RED → GitHub email. **Auto-discovers** strategies and indices — no per-strategy wiring.

**Dashboard / infra**
- `app.py` — Streamlit dashboard (repo root).
- `.github/workflows/daily.yml` — the workflow.
- `requirements.txt`, `DESIGN.md`, `SETUP.md`, `RUNBOOK.md`, `.env` (gitignored).
- Helpers / one-offs: `auth_helper.py`, `verify_features.py`, `verify_ohlcv.py`, `test_fetch_one.py`, `smoke_test.py`.

**Signal output schema** (every strategy CSV): `symbol, date, strategy, zone_type,
score, entry, stop, reason` + strategy-specific extras. `zone_type` ∈
buy / add / hold / stop_loss / exit / sell.

---

## 4. What's DONE

- **Infra** — conda env, private repo, Google OAuth (Production mode, tokens don't expire weekly), Drive folder.
- **Data pipeline** — universe, OHLCV (~2,070 symbols with sufficient history), indices incl. the midcap/smallcap bhavcopy fix, macro, FII/DII, fundamentals (2,363 ok).
- **Features** — `compute_features.py`, 46 columns, ~2,070 symbols.
- **Strategies — all 8 original ideas + Minervini built:** momentum (×5), MA-respect (×3), Qullamaggie, Minervini, Darvas, CANSLIM, PEAD, Volume (breakout + VCP = the most recent, locally tested: 69 + 96 signals).
- **Aggregation** — unified table, Multi-Strategy Conviction, diff-vs-yesterday.
- **Market state** — Health Score + sector rotation.
- **Auto-run** — GitHub Actions daily; confirmed working end-to-end.
- **Monitoring** — `pipeline_healthcheck.py` truth gate + `RUNBOOK.md`; health banner + sidebar status live in `app.py`; healthcheck verified HEALTHY on 2026-05-21 (all 18 checks PASS incl. both indices).
- **Dashboard pages working:** Market Overview (macro strip, health score, breadth incl. multi-MA %, 52w-high distribution, golden-cross / 200-SMA-rising / new-highs-lows, Nifty chart, sector rotation), Today's Signals (filterable), Stock Detail (3-panel chart + per-signal explainability with intent/rules/reason), My Portfolio (Screener .xls ISIN upload), Strategy Docs.
### 2026-05-21 | va | Stage 12 (Volume strategy) + monitoring layer
strategy_volume.py adds volume_breakout (surge + uptrend + near-high) and
volume_vcp (dry-up in a tight base) — closes the original 8-strategy list.
pipeline_healthcheck.py added as the workflow truth-gate (verifies output
freshness, writes logs/health/latest.json, turns the run RED on critical
failure); RUNBOOK.md added (5-symptom recovery guide). daily.yml reworked:
continue-on-error on every step except the healthcheck, strategies isolated
into separate steps, two-cron schedule (fixes a latent bug where weekly
fundamentals never ran). fix_indices_nse.py rewritten to use the NSE index
bhavcopy after the indicesHistory API proved blocked. Dashboard gained a
data-freshness banner + sidebar status. PROJECT_STATUS.md created as the live
current-state handover doc.

### 2026-05-18..21 | va | Stages 8-11 (dashboard, aggregation, cloud, fundamentals)
Stage 8 Streamlit dashboard live (Market Overview, Today's Signals, Stock
Detail, My Portfolio, Strategy Docs). Stage 9 aggregate_signals.py — unified
table, Multi-Strategy Conviction, diff-vs-yesterday (auto-discovers strategy
folders). Stage 10 cloud auto-run on GitHub Actions confirmed. Stage 11
fundamentals scrape (2363 ok) + CANSLIM + PEAD.
---

## 5. What's PENDING

**P1 — broken, fix next**
1. **Graphs page is dead.** `app.py` sidebar lists "Graphs" but `main()` has no `elif page == "Graphs"` branch — clicking it renders nothing. The per-strategy chart gallery (DESIGN §11 / user request) was never built. Needs a `page_graphs()` function.
2. **`volume_breakout` / `volume_vcp` missing from `STRATEGY_DOCS`** in `app.py`. Stock Detail shows "(no doc)" for volume signals until two dict entries are added.

**P2 — functionality gaps**
3. **Streamlit Cloud deployment unconfirmed.** Until the app is live on share.streamlit.io it does not auto-refresh while travelling. (`app.py` already supports cloud auth via `GDRIVE_OAUTH_*_JSON` env vars.)
4. **FII/DII not shown on the dashboard.** Data is pulled to `data/macro/FII_DII.csv`; Market Overview's macro strip shows USD/INR, Brent, Dow, Nasdaq only. DESIGN §11 wanted an FII/DII panel.
5. **`enrich_market_cap.py` not run and not in the workflow.** The Market Overview market-cap segment filter (Largecap/Midcap/Smallcap/Microcap) is inert until `universe/market_cap.csv` exists. Relevant since the user focuses on sub-5,000 cr names.

**P3 — cleanup / minor**
6. **Penny-stock filter not centralised.** `strategy_volume.py` filters sub-₹10 internally; older strategies and `compute_features.py` do not. DESIGN §5 / Stage-6 TODO wanted `min_price` enforced once upstream.
7. **`config.yaml` was never implemented.** DESIGN §5 envisioned one central `config.yaml`; in reality each strategy carries its own constants (e.g. `strategy_volume.py` CONFIG block, `strategy_momentum.py` function defaults). Not wrong, but DESIGN.md and reality disagree — decide whether to build the central config or update DESIGN.md.
8. **DESIGN.md Progress Log is stale** — last entry is Stage 7 (2026-05-18). Stages 8–12 (dashboard, aggregation, cloud auto-run, fundamentals, healthcheck, volume) are not logged there.
9. Optional feature column `return_1d_pct` / `prior_close` not added to `compute_features.py` (would let `volume_breakout` also require an up-day).

**Deferred — Phase 2 (parked by the user)**
- Dashboard pages from DESIGN §11 not built: Multi-Strategy Conviction (dedicated page — data exists), Watchlist, Backtest, Build Journal.
- Company repository: business overview, detailed reports, concall summaries.
- 12-quarter financials + valuation overlays on charts.
- LLM narrative summaries (decided: Gemini free tier, not Claude API).
- Telegram alerts (DESIGN Stage 12).

---

## 6. How to add a new strategy

Three spots — nothing else changes:

1. **`scripts/strategy_<name>.py`** — copy an existing one as a template. It must
   write `signals/per_strategy/<name>/latest.csv` with the columns in §3.
2. **`.github/workflows/daily.yml`** — add a step:
   `- name: Strategy - <Name>` / `run: python scripts/strategy_<name>.py` /
   `continue-on-error: true`.
3. **`app.py` → `STRATEGY_DOCS`** — add a dict entry keyed by the exact strategy
   name (title / intent / rules / best_for / caveat), or Stock Detail shows
   "(no doc)".

`aggregate_signals.py` and `pipeline_healthcheck.py` auto-discover any new
`per_strategy/` folder — **no edit needed** in either.

---

## 7. Config reference (where the knobs live)

There is no central config file today. Tunables are in-file:
- `strategy_volume.py` — a clearly-marked `CONFIG` block (volume multiples, ADR, near-high %).
- `strategy_momentum.py` — defaults in the `momentum_signals()` signature.
- Other strategies — constants near the top of each file.
- `pipeline_healthcheck.py` — `MIN_FEATURE_ROWS`, `OHLCV_MAX_STALE_DAYS`, `FRESH_WINDOW_HOURS`.
- `daily.yml` — schedule crons; Monday cron triggers weekly fundamentals.

---

## 8. Known fragilities & how breakage surfaces

- **OAuth token / Drive auth** — RUNBOOK Symptom 1. Healthcheck flags CRITICAL.
- **Screener cookie expiry** (~30 days) — RUNBOOK Symptom 2. Affects fundamentals/CANSLIM/PEAD only.
- **NSE / yfinance rate-limiting** — usually transient; RUNBOOK Symptom 3.
- **A strategy erroring** — RUNBOOK Symptom 4; healthcheck WARNING, run still passes.
- Detection chain: workflow step logs → `pipeline_healthcheck.py` truth gate →
  `logs/health/latest.json` → red/green banner in the dashboard → GitHub email
  on a RED (CRITICAL) run. Notifications set to "failed workflows only".

---

## 9. Environment & secrets

- conda env `signals-india`, Python 3.11. Deps in `requirements.txt`
  (key: yfinance ≥ 1.3.0, pandas 2.2.2, streamlit 1.38.0, plotly 5.24.0).
- Secrets live in 3 places, never seen by code except via `os.environ`:
  GitHub Actions secrets, Streamlit Cloud secrets, local `.env` (gitignored).
  Keys: `GDRIVE_FOLDER_ID`, `GDRIVE_OAUTH_CLIENT_SECRET_JSON` / `_TOKEN_JSON`
  (cloud) or `_PATH` (local), `SCREENER_SESSION_COOKIE`.
- Constraint (binding): code never accesses passwords/PII; portfolio cost/P&L
  data is not exposed to git or any LLM.
