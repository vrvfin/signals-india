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

PHASE 2 (phase2.yml) — 47 runs/week  [updated 2026-05-28]
  Mon–Fri: 7×/day (IST 10/12/14/16/18/20/22, every 2h) + 1 overnight (IST 03:00)
  Sat: 5×/day (IST 10/12/14/16/18)   Sun: 2× (IST 10/14)
  Reliability: (a) denser cron; (b) daily.yml triggers Phase 2 after Phase 1
               via ACTIONS_PAT; (c) optional external cron via cron-job.org
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
  company_repo/_index/processing_queue.parquet
  company_repo/_index/quarterly_facts.parquet
  company_repo/_index/guidance_tracker.parquet        ← Table_A guidance rows
  company_repo/_index/gf1_guidance_statements.parquet ← raw forward statements
  company_repo/_index/gf2_historical_guidance.parquet ← past guidance vs actuals
  company_repo/_index/gf3_operational_visibility.parquet
  company_repo/_index/gf4_quality_flags.parquet
  company_repo/_index/*.csv                           ← CSV snapshots (per run)
  company_repo/<ISIN>/company_page.md
  company_repo/_daily/concall_DD_MMMYYYY.md   (daily digest, kept forever)
  company_repo/_quarterly/QXFY_mgmt_guidance.md  (quarterly tracker, per quarter)
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
- **Guidance page** in app.py — show `guidance_tracker.parquet` + `gf1_guidance_statements.parquet` per company; management guidance vs actuals view (data now available in parquets)
- **Results filter-by-growth** — filter signals by revenue/PAT growth criteria from `quarterly_facts.parquet`
- **High-guidance watchlist** — companies with active explicit guidance; separate chart section in app (OT5)
- **Guidance-backed momentum score** — internal rank order combining price action + guidance quality (OT6)
- **Mgmt said vs delivered tracker** — GF2 cross-quarter comparison showing guidance credibility per company (OT3; needs 2-3 quarters of GF2 data to be meaningful)

### Stage E — Deep dive report
- `company_deep_report.py` — prompt file `comapnydeepdive_prompt.txt` already exists; script not yet built (OT7)
- **Local doc summarisation → Drive context store** — summarise user's local docs (industry reports, sell-side), store on Drive, inject into deep research (OT8; depends on OT7)

### Infrastructure fixes pending
- **MP3/transcript dedup** — date-based dedup in `ingest_company_docs.py` (same company, different received_date = new doc) (InfraFix 1)
- **Seasonal Phase 2 frequency** — higher run frequency during Q-end concall season (45 days after Dec/Mar/Jun/Sep quarter close); lower off-season (InfraFix 2)
- **Run-block separator in daily digest** — group `concall_N` entries by run within the daily digest (low priority)
- **ACTIONS_PAT secret** — one-time manual step: create GitHub PAT (workflow scope) → store as `ACTIONS_PAT` repo secret → activates Phase 1→Phase 2 backup trigger in daily.yml

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
- `phase2.yml` — 47-run/week schedule (every 2h, IST 10–22 + overnight 03:00); `timeout-minutes: 180`; `repository_dispatch: trigger-phase2` added
- `daily.yml` — Phase 2 backup trigger step at end of Phase 1 (uses `ACTIONS_PAT` secret); safe if secret not yet set

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
- `ACTIONS_PAT` — optional but recommended: GitHub PAT with `workflow` scope; enables Phase 1→Phase 2 backup trigger in daily.yml; create at GitHub → Settings → Developer settings → Personal access tokens (classic)
- Constraint (binding): code never accesses passwords/PII; portfolio cost/P&L never in git or LLM context

---

## 10. Session change log

Full specification detail in `scripts/Concall_Extractor.txt`.

### Session 2026-05-26 — Prompt review + operational fixes

**Asked:**
- Review `concall_prompt.txt` and confirm if information is clear or if there are prompt-level challenges
- Fix 1: MP3 without transcript — date-based dedup so same company on different dates = new doc
- Fix 2: Seasonal Phase 2 frequency tuning aligned to India Q-end concall seasons
- Fix 3: Increase daily digest retention to 90 days
- Confirmation 1: Confirm run-tracker / `Run1_concall_XXX` style grouping in daily digest

**Delivered:**
- `concall_prompt.txt` fully reviewed and rewritten: added zero-hallucination block, readability block, cross-industry metric adaptation (Manufacturing/BFSI/IT), strict Explicit/Derived guidance classification rules, mandatory derivation footnotes, completed GF1-4 sections, A1/A2/A3/B/C/E1 all present
- Fix 3 resolved: daily digests kept forever (better than 90 days)
- Confirmation 1 partial: sequential `concall_N` numbering with live **Index:** list and timestamp is live; per-run block grouping deferred

**Still pending from this session:** Fix 1 (MP3 dedup), Fix 2 (seasonal frequency), run-block separator

---

### Session 2026-05-28 — Capacity analysis + 5-output pipeline

**Asked:**
- Check Gemini key count and remaining daily/monthly capacity
- "Your assumption were wrong yesterday — 8 companies took 40 mins, you said 8-10 mins. Wrong by 4x."
- If 100 concalls + 100 presentations/results per day, is there bandwidth?
- Plan for the full month — avoid hitting limits like GitHub Actions quota
- GF1-4 must be parsed into structured data; produce 5 output files per concall in .md and .csv

**Delivered:**
- Corrected timing model: ~2 min/concall all-in (was 8-10 min estimate — acknowledged wrong)
- Capacity confirmed safe: 6 keys × 250 RPD = 1,500/day; peak load ~130 req/day = 8.7% utilisation
- Monthly risk = zero: Gemini 2.5 Flash free tier has no monthly request cap (daily RPD is the only limit)
- 429 root cause identified: RPM (not RPD). Fix: `INTER_CALL_SLEEP = 6s` added to `GeminiKeyPool`
- `extract_concall.py` fully rewritten with 5 outputs:
  1. `company_page.md` (existing, per company)
  2. `concall_DD_MMMYYYY.md` daily digest (existing)
  3. `QXFY_mgmt_guidance.md` quarterly guidance tracker (NEW)
  4. `gf1/gf2/gf3/gf4_*.parquet` (NEW — section-aware GF extraction)
  5. CSV snapshots of all 5 parquets written per run (NEW)
- All committed and pushed; active from next scheduled Phase 2 run

---

### Session 2026-05-28 (addendum) — Phase 2 run reliability

**Asked:**
- "Last run today at 1:30 although we should have run at 10 PM — problem will persist."
- Can run 5 times during 10 AM–10 PM and once around 3 AM; if code doesn't
  really execute that's a problem.

**Root cause identified:**
- No 10 PM IST slot existed in old schedule (last slot was 19:00 = 7 PM)
- GitHub Actions scheduled jobs are queued, not guaranteed — delays of 30–90 min
  are a known GitHub platform limitation, not a code bug

**Delivered (InfraFix 3):**
- phase2.yml: schedule upgraded from 29→47 runs/week: every 2h 10 AM–10 PM IST
  + 3 AM overnight on weekdays; Sat 5×; Sun 2×
- daily.yml: backup trigger step added at end of Phase 1 — dispatches Phase 2
  via `gh workflow run` using `ACTIONS_PAT` secret (safe if secret not yet set)
- phase2.yml: `repository_dispatch: types: [trigger-phase2]` added for the backup
- Three-layer reliability: (1) denser cron, (2) Phase 1 backup trigger, (3) optional external cron (cron-job.org)

**Pending action from user (1 task):**
- Create GitHub PAT (workflow scope) → store as `ACTIONS_PAT` repo secret
  to activate the Phase 1→Phase 2 backup trigger

**Still pending from this session:**
- OT3: Mgmt said vs delivered cross-quarter view (needs 2-3 quarters of GF2 data)
- OT4: Confirm results summary pull via screener
- OT5: High-guidance watchlist + chart overlay in app.py
- OT6: Guidance-backed momentum score
- OT7: Deep research report via user input (Streamlit UI)
- OT8: Local document summarisation → Drive context store
- InfraFix 1: MP3/transcript date-based dedup (carried from 2026-05-26)
- InfraFix 2: Seasonal Phase 2 frequency tuning (carried from 2026-05-26)
