# PROJECT_STATUS — signals-india

**Purpose of this file.** Single source of truth for *current state*. DESIGN.md
records what was *designed*; this file records what is *built and pending*.
To resume work in any future chat session, paste this file as context — it is
enough to pick up where we left off.

**Last updated:** 2026-05-28
**Repo:** github.com/vrvfin/signals-india (**public** — unlimited GitHub Actions minutes)
**Local:** `D:\EMA_Screener\claude\signals-india` · conda env `signals-india` (Python 3.11)
**Storage:** Google Drive folder `signals-india/` (OAuth, app in Production mode)

---

## 1. What this project is

A daily, automated trading-signal + company-intelligence system for the Indian
equity market (NSE universe, ~2,070 liquid symbols). Phase 1 pulls EOD price
data, computes features, runs strategy engines, and presents signals in a
Streamlit dashboard. Phase 2 ingests Screener PDFs (concalls, results,
ratings, presentations, annual reports), runs them through Gemini, and builds
a company intelligence repository. Signals only — no order execution. Runs
unattended on GitHub Actions.

Binding principles (from DESIGN.md): modular; free tiers only (Drive, GitHub
Actions public repo, Gemini free tier, Streamlit Cloud); no PII/passwords ever
in code; portfolio cost/P&L never exposed to git or LLM; all I/O through
Google Drive.

---

## 2. Architecture / data flow

```
GitHub Actions — two independent workflows:

PHASE 1 (daily.yml) — Mon–Fri UTC 10:30 (IST 16:00)
  skip_check (skip if already ran today IST; bypass on manual trigger)
  build_universe → ingest_ohlcv → ingest_indices_macro → fix_indices_nse
  → ingest_fii_dii → compute_features
  → 10 strategy scripts → aggregate_signals → market_state
  → pipeline_healthcheck  (truth gate — exits non-zero on CRITICAL)
  weekly (Mon only): ingest_fundamentals, enrich_market_cap

PHASE 2 (phase2.yml) — 29 runs/week
  Mon–Fri: 4×/day (IST 10/13/16/19) + 1 overnight (IST 02:00)
  Sat: 3×/day (IST 10/13/16)   Sun: 1× (IST 13:00)
  skip_check (skip if queue empty AND last run < 45 min ago)
  ingest_company_docs → scrape_results_table
  → extract_concall → extract_results → extract_rating
  → extract_presentation → extract_annual_report
  → cleanup_company_docs → write_phase2_status

Both workflows: concurrency group = sequential (never parallel runs).
Manual workflow_dispatch always bypasses skip check.

        ▼  all reads/writes go to Google Drive
Drive folder layout:
  data/ohlcv, data/indices, data/macro, data/market_state/
  features/latest.parquet
  signals/per_strategy/<name>/latest.csv, signals/aggregated/
  fundamentals/, universe/
  company_repo/_index/{processing_queue, quarterly_facts, guidance_tracker}.parquet
  company_repo/<ISIN>/company_page.md
  company_repo/_daily/concall_DD_MMMYYYY.md  (daily digest, kept forever)
  logs/health/latest.json          (Phase 1 health report)
  logs/health/phase2_latest.json   (Phase 2 queue snapshot)

        ▼
Streamlit dashboard (app.py) reads Drive, 5-min cache. Live on Streamlit Cloud.
```

---

## 3. File map

### Phase 1 — Data pipeline
- `build_universe.py` — NSE EQUITY_L.csv → `universe/master_list.csv` (~2,365 symbols)
- `ingest_ohlcv.py` — per-symbol OHLCV via yfinance (batched). yfinance installed separately in workflow (not in root requirements.txt — avoids curl_cffi malloc crash on Streamlit Cloud)
- `ingest_indices_macro.py` — index OHLCV + macro (USD/INR, Brent, Dow, Nasdaq, VIX…)
- `fix_indices_nse.py` — NIFTY MIDCAP 100 / SMALLCAP 100 via NSE index bhavcopy
- `ingest_fii_dii.py` — FII/DII cash flows from NSE → `data/macro/FII_DII.csv`
- `ingest_fundamentals.py` + `screener_client.py` — screener.in scrape (cookie auth) → `fundamentals/summary.parquet`. Weekly (Mondays)
- `enrich_market_cap.py` — yfinance market-cap → `universe/market_cap.csv`. Weekly (Mondays)
- `compute_features.py` — ~48 features per symbol → `features/latest.parquet`
- `pipeline_healthcheck.py` — truth gate; writes `logs/health/latest.json`; OHLCV_MAX_STALE_DAYS = 6 (handles India holiday weekends + yfinance lag)
- `pipeline_skip_check.py` — pre-pipeline guard; reads Drive status; writes skip=true/false to $GITHUB_OUTPUT

### Phase 1 — Strategy engines
Each writes `signals/per_strategy/<name>/latest.csv` + dated CSV.
- `strategy_momentum.py` → momentum_1m / 2m / 3m / 6m / 12m
- `strategy_ma_respect.py` → ma_respect_20ema_30d / 20ema_60d / 50ema_60d
- `strategy_qullamaggie.py` → qullamaggie
- `strategy_minervini.py` → minervini
- `strategy_darvas.py` → darvas
- `strategy_canslim.py` → canslim (needs fundamentals)
- `strategy_pead.py` → pead
- `strategy_volume.py` → volume_breakout + volume_vcp
- `aggregate_signals.py` — auto-discovers all per_strategy/ folders; builds aggregated/ + conviction + diff-vs-yesterday
- `market_state.py` — 6-component Market Health Score + sector rotation

### Phase 2 — Company Intelligence
- `build_company_universe.py` — 5,525 companies (NSE main + SME + BSE) by ISIN
- `ingest_company_docs.py` — 5 Screener feeds → PDFs into Drive processing queue
- `scrape_results_table.py` — quarterly numbers from Screener HTML (no LLM)
- `extract_concall.py` — universal (all companies); no time limit; all pending processed every run
- `extract_results.py` — portfolio-filtered
- `extract_rating.py` — portfolio-filtered
- `extract_presentation.py` — portfolio-filtered
- `extract_annual_report.py` — portfolio-filtered
- `cleanup_company_docs.py` — deletes raw PDFs > 10 days; daily digests kept forever
- `write_phase2_status.py` — writes `logs/health/phase2_latest.json` after every run
- `_extractor_base.py` — shared Drive/Gemini/queue/parquet helpers for all extractors

### Dashboard / infra
- `app.py` — Streamlit dashboard; reads Drive; `_drive_call` retry wrapper for transient errors
- `.github/workflows/daily.yml` — Phase 1 workflow
- `.github/workflows/phase2.yml` — Phase 2 workflow (concurrency: sequential, cancel-in-progress: false)
- `requirements.txt` — root (Streamlit Cloud); yfinance intentionally excluded
- `scripts/requirements.txt` — Phase 2 pipeline dependencies
- `scripts/concall_prompt.txt`, `results_prompt.txt`, `rating_prompt.txt`, `presentation_prompt.txt`, `annual_report_prompt.txt`, `comapnydeepdive_prompt.txt` — Gemini prompts

---

## 4. What's DONE

### Phase 1 — Complete ✅
- Full data pipeline (universe, OHLCV ~2,070 symbols, indices, macro, FII/DII, fundamentals, market-cap)
- 10 strategy engines (momentum ×5, MA-respect ×3, Qullamaggie, Minervini, Darvas, CANSLIM, PEAD, Volume ×2)
- Aggregation, conviction scoring, diff-vs-yesterday
- Market Health Score + sector rotation
- Dashboard live on Streamlit Cloud: Market Overview, Today's Signals, Graphs (conviction-ordered chart gallery), Stock Detail, My Portfolio, Strategy Docs
- Health monitoring: truth-gate healthcheck + banners + sidebar in app.py

### Phase 2 — Stage A + B Complete ✅
- Company universe (5,525 ISINs)
- PDF ingestion pipeline + processing queue
- All 5 extractors running (concall universal; results/rating/presentation/AR portfolio-filtered)
- **5 output files per concall** (as of 2026-05-28):
  1. `company_repo/<key>/company_page.md` — per-company cumulative brief
  2. `company_repo/_daily/concall_DD_MMMYYYY.md` — daily digest (indexed, kept forever)
  3. `company_repo/_quarterly/QXFY_mgmt_guidance.md` — quarterly guidance tracker (new)
  4. Parquet outputs: `quarterly_facts`, `guidance_tracker` (Table_A), `gf1_guidance_statements`, `gf2_historical_guidance`, `gf3_operational_visibility`, `gf4_quality_flags`
  5. CSV snapshots of all parquets written to Drive at end of each run (for Excel filtering)
- GF1–GF4 fully parsed from Gemini output into structured parquets (section-aware extraction)
- Daily digest: numbered entries (`concall_N`), live index list, running total with timestamp
- Quarterly guidance tracker: indexed by company, contains Table_A + GF1 + GF4 + A-3 summary
- Inter-call sleep (6 sec) to reduce Gemini RPM 429 cascades
- Daily digests kept forever; raw PDFs deleted after 10 days
- Phase 2 health report written to Drive after every run; visible in dashboard

### Infrastructure ✅
- Repo public (unlimited GitHub Actions minutes; no quota risk)
- Skip check: Phase 1 skips if already ran today (IST); Phase 2 skips if queue empty + <45min ago; manual trigger always bypasses
- OHLCV stale threshold: 6d (handles holiday weekends + yfinance NSE data lag)
- Streamlit malloc crash fixed (yfinance removed from root requirements.txt)
- Security: no credentials in repo; `*.parquet`, `*.csv`, `*.json` all gitignored; results.parquet removed from git history; `.env.example` has placeholders only; portfolio data never in git

---

## 5. What's PENDING

### Stage C — Dashboard enhancements
- **Guidance page** in app.py — show `guidance_tracker.parquet` per company (management guidance vs actuals)
- **Results filter-by-growth** — filter signals by revenue/PAT growth criteria from `quarterly_facts.parquet`

### Stage E — Deep dive report
- `company_deep_report.py` — prompt file `comapnydeepdive_prompt.txt` already exists; script not yet built

### Blocked
- **Insider buy/sell** — no reliable free structured data source identified

### Low-priority / deferred
- Dashboard pages from DESIGN §11: Conviction (dedicated page), Watchlist, Backtest, Build Journal
- 12-quarter financials + valuation overlays on charts
- Telegram alerts
- Move LLM prompts from repo to private Drive files (currently in git as .txt)
- Update GitHub Actions to Node.js 24 (current v4 actions deprecated June 2026; still functional until Sep 2026)

---

## 6. How to add a new strategy

1. **`scripts/strategy_<name>.py`** — copy an existing one. Must write `signals/per_strategy/<name>/latest.csv` with columns: `symbol, date, strategy, zone_type, score, entry, stop, reason`.
2. **`daily.yml`** — add a step with `continue-on-error: true` and `if: steps.fresh.outputs.skip != 'true'`.
3. **`app.py` → `STRATEGY_DOCS`** — add a dict entry or Stock Detail shows "(no doc)".

`aggregate_signals.py` and `pipeline_healthcheck.py` auto-discover — no edit needed.

---

## 7. Config reference (where the knobs live)

- `strategy_volume.py` — `CONFIG` block (volume multiples, ADR, near-high %)
- `strategy_momentum.py` — defaults in `momentum_signals()` signature
- Other strategies — constants near top of each file
- `pipeline_healthcheck.py` — `MIN_FEATURE_ROWS=1500`, `OHLCV_MAX_STALE_DAYS=6`, `FRESH_WINDOW_HOURS=24`
- `pipeline_skip_check.py` — `PHASE2_SKIP_MINUTES=45`
- `extract_concall.py` — `INTER_CALL_SLEEP=6` (seconds between Gemini calls, RPM protection)
- `daily.yml` — schedule crons; Monday cron triggers weekly steps
- `phase2.yml` — 29-run/week schedule; `timeout-minutes: 180`

---

## 8. Known behaviour & fragilities

- **OHLCV latest bar** — yfinance NSE data has 1–2 day lag; a 4-day India holiday weekend pushes the latest bar up to 5d old. The 6d threshold handles this. Data catches up on the next trading day's run.
- **Phase 1 skip showing ✅ in <1 min** — correct; skip check found today's run already done. Full pipeline only runs once per day (IST date).
- **OAuth token** — refresh_token stays valid indefinitely unless app deleted or access manually revoked. No browser flow needed in CI.
- **Screener cookie expiry** (~30 days) — affects fundamentals/CANSLIM/PEAD. Renew via RUNBOOK.
- **NSE/yfinance rate-limiting** — transient; retry on next run.
- **A strategy erroring** — `continue-on-error: true`; healthcheck WARNING, run still GREEN.
- Detection chain: step logs → healthcheck truth gate → `logs/health/latest.json` → dashboard banner → GitHub email on RED run.

---

## 9. Environment & secrets

- conda env `signals-india`, Python 3.11
- `requirements.txt` (root) — Streamlit Cloud; excludes yfinance (installed separately in daily.yml)
- `scripts/requirements.txt` — Phase 2 pipeline
- Secrets: GitHub Actions secrets + Streamlit Cloud secrets + local `.env` (gitignored)
- Keys: `GDRIVE_FOLDER_ID`, `GDRIVE_OAUTH_TOKEN_JSON`, `GDRIVE_OAUTH_CLIENT_SECRET_JSON`, `SCREENER_SESSION_COOKIE`, `GEMINI_API_KEY`
- Constraint (binding): code never accesses passwords/PII; portfolio cost/P&L never in git or LLM context
