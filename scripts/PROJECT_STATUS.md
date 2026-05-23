# PROJECT_STATUS — signals-india

**Purpose of this file.** Single source of truth for *current state*. DESIGN.md
records what was *designed*; this file records what is *built and pending*.
Keep it in the repo root. To resume work in any future chat session, paste this
file plus one existing `strategy_*.py` as a template — that is enough context.

**Last updated:** 2026-05-21 (Phase 1 complete — price/volume system done)
**Repo:** github.com/vrvfin/signals-india (private)
**Local:** `D:\EMA_Screener\claude\signals-india`  ·  conda env `signals-india` (Python 3.11)
**Storage:** Google Drive folder `signals-india/` (OAuth, app in Production mode)

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
GitHub Actions (daily 17:30 IST Mon-Fri; Mon run also does weekly steps)
  build_universe → ingest_ohlcv → ingest_indices_macro → fix_indices_nse
                 → ingest_fii_dii → compute_features
                 → 8 strategy scripts → aggregate_signals → market_state
                 → pipeline_healthcheck  (truth gate)
  weekly (Mon):  ingest_fundamentals, enrich_market_cap
        │
        ▼  all reads/writes go to Google Drive
Drive: data/ohlcv, data/indices, data/macro, data/market_state,
       features/latest.parquet, signals/per_strategy/<name>/latest.csv,
       signals/aggregated/, fundamentals/, universe/, logs/health/latest.json
        │
        ▼
Streamlit dashboard (app.py) reads Drive, 5-min cache. Live on Streamlit Cloud.
```

Every stage reads from Drive and writes to Drive; stages are independently
runnable. One stage failing does not stop the others (see §8).

---

## 3. File map

**Data pull**
- `build_universe.py` — NSE EQUITY_L.csv → `universe/master_list.csv` (~2,365 symbols).
- `ingest_ohlcv.py` — per-symbol OHLCV via yfinance (batched `yf.download`). Needs yfinance ≥ 1.3.0.
- `ingest_indices_macro.py` — index OHLCV + macro series (USD/INR, Brent, Dow, Nasdaq, VIX…).
- `fix_indices_nse.py` — NIFTY MIDCAP 100 / SMALLCAP 100 via NSE **index bhavcopy** (`archives.nseindia.com`).
- `ingest_fii_dii.py` — FII/DII cash flows from NSE → `data/macro/FII_DII.csv` (cols: date, category, buy, sell, net).
- `ingest_fundamentals.py` + `screener_client.py` — screener.in scrape (cookie auth) → `fundamentals/summary.parquet`. Weekly (Mondays).
- `enrich_market_cap.py` — yfinance market-cap → `universe/market_cap.csv` (symbol, market_cap_cr, mcap_segment). Weekly (Mondays); has resume logic.

**Features**
- `compute_features.py` — ~48 features per symbol → `features/latest.parquet`. EMAs/SMAs, ATR, ADR%, 52w high/low, returns 1d/1m/2m/3m/6m/12m, RS rank, `vol_today_ratio`, `days_above_ema_*`, `prior_close`, trend flags. Penny-stock filter (close ≥ ₹10) applied before upload.

**Strategy engines** (each writes `signals/per_strategy/<name>/latest.csv` + dated CSV)
- `strategy_momentum.py` → momentum_1m / 2m / 3m / 6m / 12m
- `strategy_ma_respect.py` → ma_respect_20ema_30d / 20ema_60d / 50ema_60d
- `strategy_qullamaggie.py` → qullamaggie
- `strategy_minervini.py` → minervini
- `strategy_darvas.py` → darvas
- `strategy_canslim.py` → canslim (needs fundamentals)
- `strategy_pead.py` → pead
- `strategy_volume.py` → volume_breakout + volume_vcp

**Aggregate / state / monitoring**
- `aggregate_signals.py` — unions all `per_strategy/*/latest.csv` (**auto-discovers** — no strategy list), builds `signals/aggregated/{latest,conviction,diff_vs_yesterday}.csv`. Multi-Strategy Conviction = flagged by ≥2 strategies.
- `market_state.py` — 6-component Market Health Score (0-100) + sector rotation → `data/market_state/`.
- `pipeline_healthcheck.py` — final workflow step ("truth gate"). Verifies output freshness; writes `logs/health/latest.json`; exits non-zero on CRITICAL → workflow RED → GitHub email. **Auto-discovers** strategies and indices.

**Dashboard / infra**
- `app.py` — Streamlit dashboard (repo root). Has a `_drive_call` retry wrapper for transient Drive errors.
- `.github/workflows/daily.yml` — the workflow (two crons: Mon, Tue–Fri).
- `requirements.txt`, `DESIGN.md`, `SETUP.md`, `RUNBOOK.md`, `.env` (gitignored).
- Helpers / one-offs: `auth_helper.py`, `verify_features.py`, `verify_ohlcv.py`, `test_fetch_one.py`, `smoke_test.py`.

**Signal output schema** (every strategy CSV): `symbol, date, strategy, zone_type,
score, entry, stop, reason` + strategy-specific extras. `zone_type` ∈
buy / add / hold / stop_loss / exit / sell.

---

## 4. What's DONE (Phase 1 — complete)

- **Infra** — conda env, private repo, Google OAuth (Production mode), Drive folder.
- **Data pipeline** — universe, OHLCV (~2,070 symbols), indices incl. the midcap/smallcap bhavcopy fix, macro, FII/DII, fundamentals (2,363 ok), market-cap enrichment.
- **Features** — `compute_features.py`, ~48 columns incl. `return_1d_pct` / `prior_close`; penny filter applied.
- **Strategies — all 8 original ideas + Minervini:** momentum (×5), MA-respect (×3), Qullamaggie, Minervini, Darvas, CANSLIM, PEAD, Volume (breakout + VCP).
- **Aggregation** — unified table, Multi-Strategy Conviction, diff-vs-yesterday.
- **Market state** — Health Score + sector rotation.
- **Auto-run** — GitHub Actions daily; confirmed working end-to-end.
- **Monitoring** — `pipeline_healthcheck.py` truth gate + `RUNBOOK.md`; health banner + sidebar status in `app.py`.
- **Dashboard (live on Streamlit Cloud):** Market Overview (macro strip, FII/DII panel, health score, breadth, market-cap segment filter, Nifty chart, sector rotation), Today's Signals (filterable table), Graphs (conviction-ordered, paginated chart gallery with multi-strategy chips), Stock Detail (3-panel chart + per-signal explainability), My Portfolio (Screener .xls ISIN upload), Strategy Docs.
- **Resilience** — `_drive_call` retry wrapper: transient TLS / `BrokenPipeError` / stale-connection failures drop the cached Drive connection and retry with a fresh one.

---

## 5. What's PENDING

**Phase 1 (price/volume): nothing outstanding.** All P1/P2 items from earlier
versions of this file were closed on 2026-05-21 — FII/DII panel, market-cap
filter, `return_1d_pct`/`prior_close`, penny filter (confirmed already present),
`config.yaml` discrepancy (DESIGN §5 now documents the per-file convention),
DESIGN Progress Log brought current.

**Phase 2 — Company Repository — IN PROGRESS (see `PHASE2_SPEC.md`).**
- **Stage A DONE (2026-05-23):** `build_company_universe.py` (5,525 companies,
  NSE main + NSE Emerge SME + BSE, by ISIN), `ingest_company_docs.py` (5 Screener
  feeds → PDFs + processing queue), `scrape_results_table.py` (quarterly numbers),
  `cleanup_company_docs.py` (10-day raw-file purge).
- **Stage B next:** Gemini extraction (concall / presentation / results-filing
  prompts) — needs `GEMINI_API_KEY`.
- Then: company page generator, guidance scorecard, deep reports, history backfill.
- **PENDING — insider buy/sell source:** Screener's insider feeds aren't suitable
  (one-line facts, not documents); an alternative structured source is TBD.
- Deferred still: 12-quarter financials + valuation overlays on charts; remaining
  DESIGN §11 dashboard pages (Conviction page, Watchlist, Backtest, Build Journal);
  Telegram alerts.

**Verify on next runs:** `compute_features.py` emits the new `return_1d_pct` /
`prior_close` columns; the FII/DII panel renders on Market Overview; the
market-cap segment filter populates once `universe/market_cap.csv` exists.

---

## 6. How to add a new strategy

Three spots — nothing else changes:

1. **`scripts/strategy_<name>.py`** — copy an existing one as a template. It must
   write `signals/per_strategy/<name>/latest.csv` with the columns in §3.
2. **`.github/workflows/daily.yml`** — add a step:
   `- name: Strategy - <Name>` / `run: python scripts/strategy_<name>.py` /
   `continue-on-error: true`.
3. **`app.py` → `STRATEGY_DOCS`** — add a dict entry keyed by the exact strategy
   name, or Stock Detail shows "(no doc)".

`aggregate_signals.py`, `pipeline_healthcheck.py`, and the dashboard Graphs page
auto-discover any new `per_strategy/` folder — no edit needed.

---

## 7. Config reference (where the knobs live)

No central config file — each strategy carries its own clearly-marked constants:
- `strategy_volume.py` — a `CONFIG` block (volume multiples, ADR, near-high %).
- `strategy_momentum.py` — defaults in the `momentum_signals()` signature.
- Other strategies — constants near the top of each file.
- `pipeline_healthcheck.py` — `MIN_FEATURE_ROWS`, `OHLCV_MAX_STALE_DAYS`, `FRESH_WINDOW_HOURS`.
- `daily.yml` — schedule crons; Monday cron triggers the weekly steps.

---

## 8. Known fragilities & how breakage surfaces

- **OAuth token / Drive auth** — RUNBOOK Symptom 1. Healthcheck flags CRITICAL.
- **Screener cookie expiry** (~30 days) — RUNBOOK Symptom 2. Affects fundamentals/CANSLIM/PEAD only.
- **NSE / yfinance rate-limiting** — usually transient; RUNBOOK Symptom 3.
- **A strategy erroring** — RUNBOOK Symptom 4; healthcheck WARNING, run still passes.
- **Transient Drive TLS / `BrokenPipeError`** — handled by the dashboard's `_drive_call` retry (fresh connection per retry).
- **Streamlit Cloud ~1 GB memory ceiling** — a one-off segfault on 2026-05-21 was cleared by a single app reboot (environmental, not a code bug). If segfaults *recur*, it points to memory and the dashboard's chart/cache load needs trimming.
- Detection chain: workflow step logs → `pipeline_healthcheck.py` truth gate → `logs/health/latest.json` → red/green banner in the dashboard → GitHub email on a RED (CRITICAL) run. Notifications set to "failed workflows only".

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
