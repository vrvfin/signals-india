# PROJECT_STATUS — signals-india

**Purpose of this file.** Single source of truth for *current state*. DESIGN.md
records what was *designed*; this file records what is *built and pending*.
To resume work in any future chat session, paste this file as context — it is
enough to pick up where we left off.

**IMPORTANT — single file rule.** This is the ONLY status/tracking document.
Whether new items arrive in chat, from an uploaded document, or verbally — they
must be added here in the appropriate section (Section 5 pending or Section 10
session log) before the session ends. No separate tracking spreadsheets or docs.

**Last updated:** 2026-06-06
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

PHASE 2 (phase2.yml) — seasonal schedule  [updated 2026-05-28]
  PEAK season (17 Jan–28 Feb, 17 Apr–30 May, 17 Jul–30 Aug, 17 Oct–30 Nov):
    Weekday: 6 daytime (IST 08/11/14/17/20/22) + overnight 03:00 = 7/day
    Saturday: 5 daytime (IST 08/11:30/14/16:30/20) + overnight   = 6/day
    Sunday: 3 daytime (IST 08/14/20)                             = 3/day
  OFF-SEASON: 3 runs/day all day types (IST 08/14/20 ±75 min windows)
  skip_check enforces seasonal thresholds (peak: 45 min; off-season window: 90 min;
             outside off-season window → always skip)
  Reliability: (a) seasonal cron; (b) daily.yml backup trigger (ACTIONS_PAT);
               (c) optional external cron via cron-job.org
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
  company_repo/_daily/research_DD_MMMYYYY.md  (OT8 daily research digest)
  company_repo/_daily/daily_focus_DD_MMMYYYY.md (OT8 daily focus brief)
  company_repo/_quarterly/QXFY_mgmt_guidance.md  (quarterly tracker, per quarter)
  company_repo/_index/deep_dive_queue.parquet    (OT7 pending companies)
  company_repo/_index/deep_dive_index.parquet    (OT7 completed reports)
  company_repo/_index/research_index.parquet     (OT8 doc index)
  company_repo/_index/company_mentions.parquet   (OT8 cross-company mentions)
  company_repo/<ISIN>/company_deepdive_DDMMMYY.md  (OT7 report)
  company_repo/<ISIN>/company_deepdive_DDMMMYY.docx/.pdf  (OT7 formatted)
  company_repo/<ISIN>/synthesis_latest.md + dated archive  (OT8 synthesis)
  universe/master_list.csv         (NSE+BSE+SME, 5,542 companies, weekly refresh)
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

### Workflow A — OT8 Daily research summarisation (LOCAL)
- `daily_research_summary.py` — dedup → classify → route → summarise → 3 outputs + Drive upload
- `find_company_docs.py` + `find_company_docs.bat` — search mentions by name/ISIN/doc_type
- `synthesise_company_docs.py` + `synthesise_company.bat` — per-company synthesis via Gemini
- `daily_focus.py` + `daily_focus_prompt.txt` — cross-report triage brief
- `research_doc_prompt.txt`, `company_synthesis_prompt.txt` — Gemini prompts
- `get_latest_research.bat`, `run_daily_research.bat` — local trigger + Task Scheduler

### Workflow B — OT7 Company deep dive (CI + local)
- `company_deep_report.py` — queue drain → resolve → coverage check → assemble → Gemini → Drive
  Uses `gemini_pool.BucketPool` + `load_keys()` (consistent with all other scripts)
  `DEEPDIVE_MODELS` = 5 models × N keys = N×5 daily buckets
- `fetch_deepdive.py` + `get_deepdive.bat` — fetch completed report → Obsidian/browser
- `run_deepdive.bat` — interactive local trigger (run now or queue for CI)
- `format_deepdive_docx.py` — md → Word (.docx) with cover, TOC, styled tables
- `format_deepdive_pdf.py` — md → styled HTML → PDF via weasyprint (A4, navy tables, page numbers)
- `notify_deepdive.py` — Gmail email with .md + .pdf attachments after CI run
- `.github/workflows/deepdive.yml` — daily 08:00 IST + manual dispatch + repository_dispatch
- `.github/workflows/universe_refresh.yml` — weekly Sunday 03:00 IST rebuilds master_list.csv

### Phase 2 — Company Intelligence
- `build_company_universe.py` — 5,542 companies (NSE main + SME + BSE) by ISIN
  Writes BOTH `company_repo/_index/company_universe.csv` AND `universe/master_list.csv`
  CI auth supported (GDRIVE_OAUTH_TOKEN_JSON inline)
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

### Dashboard — Stage C Complete ✅ (2026-05-29)
- **Mgmt Guidance page** — new sidebar nav item with 3 tabs:
  - Guidance Tracker: `guidance_tracker.parquet` + `gf1_guidance_statements.parquet`, 4 filters (company / metric / horizon / type)
  - Active Watchlist: current/future FY explicit guidance, GF4 quality scores (+1/−1 per flag), drill-down per company
  - Guidance × Momentum: combined score = n_strategies × max(0.5, 1 + quality × 0.3) + n_guidance × 0.2; threshold sliders
- **Results growth filter** in Today's Signals — opt-in expander (checkbox), min Revenue YoY% + PAT YoY% from `results.parquet`, joins via ISIN→symbol from universe
- **Management Guidance section** in Stock Detail — active guidance table, all-quarters expander, GF4 quality badge, raw GF1 statements expander
- 4 new cached data loaders (`load_guidance_tracker`, `load_gf1_statements`, `load_gf4_flags`, `load_results_summary`)
- GF4 helpers: `_gf4_quality_score()`, `_guidance_is_active()`, `_GF4_POSITIVE`/`_GF4_NEGATIVE` frozensets

### Infrastructure ✅
- Repo public (unlimited GitHub Actions minutes; no quota risk)
- Skip check: Phase 1 skips if already ran today (IST); Phase 2 skips if queue empty + <45min ago; manual trigger always bypasses
- OHLCV stale threshold: 6d (handles holiday weekends + yfinance NSE data lag)
- Streamlit malloc crash fixed (yfinance removed from root requirements.txt)
- Security: no credentials in repo; `*.parquet`, `*.csv`, `*.json` all gitignored; results.parquet removed from git history; `.env.example` has placeholders only; portfolio data never in git
- **MP3/transcript date-based dedup** — composite key `doc_id + announcement_date` in `ingest_company_docs.py` (Fix 1, 2026-05-28)
- **Run-block separator in daily digest** — `extract_concall.py` groups `concall_N` entries by run time with `*(Run HH:MM IST)*` suffix; backward-compatible (InfraFix 1 / Confirmation 1, 2026-05-28)
- **Three-layer Phase 2 reliability** — (1) dense seasonal cron 6-7×/weekday peak; (2) daily.yml ACTIONS_PAT backup trigger; (3) cron-job.org external HTTP trigger every 2h. ACTIONS_PAT secret created ✅
- **Phase 1 external trigger** — cron-job.org sends `repository_dispatch` to daily.yml at 16:00 IST weekdays; daily.yml updated to receive it. cron-job.org Phase 1 job created ✅

---

## 5. What's PENDING

<!-- Priority order as set 2026-05-29: P0→P1→P2→P3→P4 -->

### OT8 — Document Library (context store) ✅ DONE (2026-05-29)
- New "Doc Library" Streamlit page with 2 tabs:
  - ⬆ Upload: company key, doc type, year/label, multi-file uploader → Drive
  - 📋 My Library: browse all uploaded docs across companies, metrics, per-company readiness
- Drive structure: `user_docs/<company_key>/<doc_type>_<label>.pdf` + `_manifest.json` per company
- Manifest tracks: file_id, original_name, doc_type, label, size_kb, uploaded_at, deep_dive_status
- `deep_dive_status: "pending"` flags docs not yet processed by OT7
- Supports PDF, DOCX, TXT, XLSX; graceful name-collision handling
- Helper functions: `_get_or_create_folder`, `_upload_bytes_to_drive`, `_read_manifest`, `_write_manifest`

### OT7 — Company deep-dive report ✅ DONE (2026-06-06)
See session log 2026-06-06 for full detail.

### P2 — OT3: Mgmt said vs delivered (GF2 credibility tracker)
- Cross-quarter comparison of guidance_tracker vs quarterly_facts
- Shows guidance credibility per company per metric
- Deferred: needs 2-3 quarters of GF2 data (earliest useful: ~Nov 2026)

### P3 — OT9: Streamlit performance optimisation
- Audit cache TTLs, reduce redundant Drive calls, lazy-load heavy pages

### OT11 — Phase 1 HTML report ✅ DONE (2026-05-29)
- `scripts/fetch_phase1_report.py` + `get_phase1_report.bat`
- Section 1: Conviction signals (≥N strategies, buy/add zones) — table + collapsible Plotly charts
- Section 2: Portfolio — holdings overlaid with features + signal zones + collapsible charts
- Charts: candlestick + 20/50 EMA + 200 SMA + volume + signal entry/stop lines
- Interactive (hover/zoom/pan), CDN Plotly.js, collapsible `<details>` so 300 charts don't freeze browser
- Bat options: tables-only (~15s) through top 300 charts (~90 min); portfolio charts always included with --with-charts
- Saves to `D:\EMA_Screener\Reports\signals-india\phase1_reports\`; print to PDF via Ctrl+P → Landscape

### P4 — Documentation + Fix 2B
- **End-to-end runbook** — local + Streamlit, all processes, how to fix common issues
- **Fix 2B** — BSE Direct API secondary ingestion (eliminates Screener 25-item cap)
  API: `https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w` — public, no auth, no cap

### Infrastructure — Node.js 24 ✅ DONE (2026-05-29)
- `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` added to both `daily.yml` and `phase2.yml`.

### OT10 — Doc Viewer + Local Bat Tools ✅ DONE (2026-05-29)

**Streamlit:**
- "Doc Viewer" page — 4 tabs: Company Page, GF Tables (dataframes), Results, Quarterly Guidance
- "Mgmt Guidance" page — Tracker / Active Watchlist / Guidance × Momentum tabs

**Local bat files (all in project root, double-click to run):**

| Bat file | Purpose | Saves to |
|---|---|---|
| `get_latest_concall.bat` | Latest daily digest → Obsidian | `signals-india\concalls\` |
| `get_concalls_range.bat` | Date-range digests → Obsidian | `signals-india\concalls\` |
| `list_concalls.bat` | Browse + pick by date → Obsidian | `signals-india\concalls\` |
| `get_company_intel.bat` | Company page by ISIN/symbol → Obsidian | `signals-india\company_intel\` |
| `get_quarterly_guidance.bat` | Quarterly .md (growing season summary) → Obsidian | `signals-india\quarterly_guidance\` |
| `get_gf_csv.bat` | GF1-GF4 + Table A as CSV → Explorer/Excel | `signals-india\gf_tables_csv\` |
| `get_gf_filtered.bat` | Filter by guidance criteria → matching company pages → Obsidian | `signals-india\company_intel\` |

**Supporting scripts:**
- `scripts/_md_utils.py` — shared Obsidian table fixer (joins split rows, strips leading whitespace, unwraps fenced tables)
- `scripts/md_viewer.py` — CLI: `python scripts/md_viewer.py file.md` → HTML in browser
- `scripts/fetch_latest_concall.py`, `fetch_concalls_range.py`, `fetch_company_intel.py`
- `scripts/fetch_quarterly_guidance.py`, `fetch_gf_csv.py`, `fetch_gf_filtered.py`
- `scripts/RUNBOOK.md` — operations guide, FAQ, troubleshooting

**Known data quality issue:** Some companies in guidance_tracker have BSE scrip code as `symbol` (e.g. `544675`) and NaN ISIN — company_page.md lookup falls back gracefully with clear error message.

### Blocked
- **Insider buy/sell** — no reliable free structured data source identified

### Low-priority / long-term
- Dashboard pages from DESIGN §11: Conviction, Watchlist, Backtest, Build Journal
- 12-quarter financials + valuation overlays on charts
- Telegram alerts
- Move prompts to private Drive files (currently in git as .txt)

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
- `phase2.yml` — seasonal schedule (peak: 6-7/day weekday, 3/day off-season); `timeout-minutes: 180`; `repository_dispatch: trigger-phase2` added
- `pipeline_skip_check.py` — `PEAK_SEASONS` date ranges, `_is_peak_season()`, `_in_offseason_window()`; peak threshold 45 min, off-season 90 min; outside window → hard skip
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

---

### Session 2026-05-28 (continued) — Reliability + app.py Stage C

**Asked:**
- Enforce Phase 1 runs at 4PM IST via external trigger (since GitHub cron is unreliable)
- Confirm OT4 (screener results summary pull)
- Close all pending app.py items (OT5-a, OT5-b, OT6, Stage C-1, Stage C-2)

**Delivered:**
- Phase 1 cron-job.org: `repository_dispatch` trigger added to `daily.yml`; cron-job.org job configured for weekdays 16:00 IST
- ACTIONS_PAT GitHub secret created (Phase 1→Phase 2 backup trigger now active)
- OT4 confirmed: `scrape_results_table.py` already covers this; `results.parquet` at `company_repo/_index/results.parquet` with YoY% columns
- OT3 deferred: needs real data from 2-3 quarters before it's meaningful

**Stage C-1 — Mgmt Guidance page (app.py):**
- New "Mgmt Guidance" nav item + `page_guidance()` with 3 tabs
- Tab 1 Guidance Tracker: 4-filter table from `guidance_tracker.parquet` + GF1 raw statements expander
- Tab 2 Active Watchlist: current/future FY guidance ranked by GF4 quality score, drill-down per company
- Tab 3 Guidance × Momentum: combined score formula, min-strategy/guidance threshold sliders

**Stage C-2 — Results growth filter (app.py):**
- Collapsed expander in Today's Signals with opt-in checkbox
- Filters by min Revenue YoY% + PAT YoY%; joins `results.parquet` → universe via ISIN

**OT5-b — Stock Detail guidance section (app.py):**
- Management Guidance section at bottom of Stock Detail page
- Shows active (current/future FY) guidance table + all-quarters expander + GF4 quality badge + raw GF1 expander

**Still pending / deferred:**
- OT3: Deferred to Nov 2026 (needs 2-3 quarters of GF2 data)
- OT7: Deep research report (Streamlit input → Gemini) — not yet built
- OT8: Local doc summarisation → Drive context store (depends on OT7)
- Fix 2B: BSE Direct API secondary ingestion (eliminates Screener 25-item cap)

---

### Session 2026-05-29 — Status review

**Run health check:**
- Phase 2 running successfully: 6 successful runs in last 24h (cron + repository_dispatch), 2 cancelled via timeout (3h limit hit — queue is large during peak season). Normal for peak concall season.
- Phase 1 last run: 2026-05-28T14:07 UTC (manual trigger, all steps passed except Health check truth gate — isolated failure, likely data freshness issue on that specific run). No scheduled Phase 1 run yet today (will fire at 10:30 UTC = 16:00 IST).
- Node.js 20 deprecation warning visible on all runs — actions/checkout, setup-python, cache all on Node.js 20. **Forces to Node.js 24 from June 2, 2026** — must fix within 4 days.

**Documentation updated:**
- Section 4 (Done) updated with Stage C completion and all infra items
- Section 5 (Pending) refreshed — removed completed items, added Fix 2B and Node.js 24 as priority items

---

### Session 2026-05-29 — P0 + stale fix + OT10

**Asked:**
- Fix Node.js 24 migration (P0, deadline Jun 2)
- Investigate Phase 1 stale warning (yfinance has latest data)
- Maintain single status file for all tracking
- OT9: Streamlit optimisation (P2, defer)
- OT10: Tool to read .md files with proper table rendering (P1)

**Delivered:**

**P0 — Node.js 24 (both workflows):**
- Added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` as job-level env var in `daily.yml` and `phase2.yml`
- Silences all deprecation warnings; opts in now before Jun 2 forced migration

**Stale warning root cause found and fixed:**
- `pipeline_healthcheck.py` line 205: `stale_days = (datetime.now() - latest_bar).days`
- `datetime.now()` is timezone-naive; `latest_bar` from yfinance can be timezone-aware (IST tz)
- This raises `TypeError` → uncaught exception → `sys.exit(1)` → CRITICAL failure → "stale" banner in app
- Fix: compare `.date()` only using `pd.Timestamp(latest_bar).date()` vs `datetime.now(timezone.utc).date()`
- This is timezone-safe regardless of how yfinance stores NSE timestamps

**OT10 — Doc Viewer:**
- `scripts/md_viewer.py`: local CLI tool; `python scripts/md_viewer.py <file.md>` opens HTML in browser with proper table rendering. Falls back to `<pre>` if `markdown` package not installed.
- New "Doc Viewer" page in `app.py` (sidebar nav): 4 tabs — Company Page (.md rendered), GF Tables (parquets as dataframes), Results, Quarterly Guidance (.md browser)
- Matches by both ISIN and symbol

**Single file rule:** Added explicit note to top of this file — all tracking lives here.

**Still pending:**
-Priority 1 - pick my portfolio from drive & create the output specification PF_trackingperforamnce - this is WIP, we will focus on OT7/OT8
- OT7: Deep research report (company_deep_report.py)
- OT8: Local doc summarisation → Drive context store
- Fix 2B: BSE Direct API (Screener 25-item cap)
- OT9: Streamlit optimisation (P2, deferred)
- User to add new items

---

### Session 2026-06-03 — Workflow A wiring + new features (OT8)

**Asked:**
- Get Workflow A (OT8 daily summariser) actually running against the real local env.
- Add: (1) move processed PDFs within intake; (2) capture company mentions so any past
  doc mentioning e.g. "Venus Remedies" can be found later; (3) bat to find+open those docs;
  (4) bat to synthesise all a company's docs into one note (existing or new prompt);
  (5) upload synthesis to Drive in company folder, dated + standardised; (6) auto-refresh a
  rolling per-company "latest" synthesis; (7) auto-trigger OT7 deep dive.
- Then: a cross-report **Daily Focus** triage note (today + rolling 7d) ranking strong
  growth / guidance / risk / valuation, plus new info, order-size-vs-mcap, fraud risk,
  macro→sector. Open daily digest + focus note only (NOT 100 per-company tabs).

**Environment setup (this machine):**
- Python env: `C:\Users\vaido\.conda\envs\signals-india\python.exe` (3.11). `conda` not on PATH.
- Installed missing deps: **PyMuPDF, pdfplumber** (rest already present).
- Created intake folder `D:\EMA_Screener\research_intake` (+ `_ledger/`); was missing.
- `build_tag_vocab.py` → 138 tag rows; `tag_vocabulary.csv/.parquet` generated.

**Delivered — daily_research_summary.py (OT8 core, now LIVE-ready):**
- `load_dotenv(.env)` added (was missing → bat runs had no env vars; root cause of failures).
- Drive auth resolves `GDRIVE_OAUTH_TOKEN_JSON` (CI) else `GDRIVE_OAUTH_TOKEN_PATH` (local file).
- `summary_md` now stored in research_index/ledger (so synthesis needs no PDF re-read).
- **Move processed PDFs** → `research_intake/_processed/<source>/` after ledger save
  (dupes / needs_ocr stay in place).
- **Company mentions index** → `_ledger/company_mentions.parquet` (1 row per doc×company,
  with `company_name_slug` legal-suffix-stripped) + uploaded to
  `company_repo/_index/company_mentions.parquet`.
- Writes `_ledger/new_isins_latest.json` (ISINs with new docs this run).

**New scripts:**
- `find_company_docs.py` — search mentions by name / ISIN / doc_type; `--open` opens the PDFs.
- `synthesise_company_docs.py` — find all of a company's summaries → 1 Gemini call →
  `synthesis_<slug>_DDMmmYYYY.md` (dated) + `synthesis_latest.md` to `company_repo/<ISIN>/`;
  flags: `--upload`, `--queue` (enqueue OT7 deep_dive_queue), `--all-new`, `--no-open`.
- `company_synthesis_prompt.txt` — thesis / metrics / mgmt signals / bull-bear / contradictions
  / story evolution / analyst views.
- `daily_focus.py` + `daily_focus_prompt.txt` — cross-report triage, PART 1 Today + PART 2
  Rolling(7d); sections: Must Focus, Strong Growth, Guidance & Order Book (order vs mcap),
  Valuation/Analyst, New Info, Red Flags & Fraud Risk, Macro→Sector. Local + Drive `_daily/`.

**Bat files (use full python.exe path — NOT `conda activate`; that fails on this machine):**
- `get_latest_research.bat` (root, interactive) — 4 steps: summarise → synthesise
  (`--all-new --upload --queue --no-open`) → daily_focus → fetch digest. Opens focus + digest only.
- `run_daily_research.bat` (scripts/, Task Scheduler) — same chain, `--no-open`, logged.
- `find_company_docs.bat`, `synthesise_company.bat` (root, interactive prompts).

**Drive outputs (OT8):**
- `company_repo/_daily/research_DD_MMMYYYY.md` (digest) + `daily_focus_DD_MMMYYYY.md` (NEW)
- `company_repo/_index/research_index.parquet` + `company_mentions.parquet` (NEW)
- `company_repo/<ISIN>/synthesis_latest.md` + dated archive (NEW)
- `company_repo/_index/deep_dive_queue.parquet` (auto-enqueued for OT7)
- Source PDFs NEVER uploaded (stay local in `_processed/`).

**Status:** all scripts compile; Drive auth + Gemini 2-key pool verified loading from .env;
`--dry-run` clean (detected 12 real PDFs in intake). **Awaiting user's first LIVE run** of
`get_latest_research.bat` (~3-4 min; makes Gemini calls + Drive writes).

**Still pending / next:**
- Confirm live run output quality (digest, focus note, per-company synthesis).
- OT7 deep dive (`company_deep_report.py`) end-to-end test (queue now auto-populated by OT8).
- Known gaps unchanged: typed prompts don't emit JSON tag tail; BSE code col; no OCR.

---

### Session 2026-06-04 — First LIVE run + Obsidian + brief-style prompts

**Asked:**
- Finish OT8: real run, Obsidian output, appealing synthesis + doc-wise summary auto-open,
  confirm Drive uploads. Checked Claude API key option (declined — staying free-tier Gemini).

**LIVE run completed (11 docs):**
- First real `daily_research_summary.py` run processed 11 PDFs → digest `research_03_Jun2026.md`,
  `research_index.parquet`, `company_mentions.parquet` on Drive; PDFs moved to `_processed/`.
- High-value docs captured: Man Industries (InCred BUY ₹768/+53%), Yatharth (Sunidhi BUY ₹1,078/+33%),
  LTM (+41%), Fiem (HOLD), Tega (downgrade NEUTRAL), banks/autos constructive, Manappuram RoA red flag.

**Obsidian integration (all 3 outputs):**
- Outputs now save into the Obsidian vault `D:\EMA_Screener\Reports\signals-india\` and open via
  `obsidian://`, run through `_md_utils.fix_markdown_for_obsidian` (matches concall/intel tools).
  - Daily digest → `...\signals-india\research\`
  - Daily focus  → `...\signals-india\daily_focus\`
  - Per-company synthesis → `...\signals-india\company_synthesis\`
- Switched from md_viewer (browser) → Obsidian. Bat `--outdir` updated to vault path.

**Prompts rewritten to "Daily Research Brief" house style (the appealing format user approved):**
- `daily_focus_prompt.txt` — 🗞️ header + desk-view one-liner · ⚡ TL;DR · 🎯 ranked conviction
  table · 📈 growth scorecard · 🏦 sector pulse · 🌍 macro · ⚠️ risk radar (🚩 fraud) · 📋 doc-by-doc.
- `company_synthesis_prompt.txt` — 🏢 verdict banner · snapshot · thesis · metrics · mgmt signals ·
  bull/bear · contradictions · story evolution · analyst views.
- Verified placeholders match script replacements; drop-in, no code change.

**BUG FIXED — summary_md not available locally:**
- `summary_md` was written only to the DRIVE index, but `daily_focus` / `synthesise` read a LOCAL
  parquet → "no summary_md" error. Fix: `daily_research_summary.py` now ALSO writes
  `_ledger/research_index.parquet` (local, with summary_md); both consumers read this local index
  (was research_ledger.parquet). Backfilled today's 11 rows from Drive (one-time).

**Auto-open confirmed:** `get_latest_research.bat` opens TWO Obsidian tabs each run —
[3/4] Daily Focus brief + [4/4] digest. Per-company syntheses go to Drive silently.

**Env/quota note:** No Claude/Anthropic key (user declined). Anthropic API would be separate billing
(not Gemini, not Claude Code quota) but breaks the "free-tier Gemini only" golden rule — not adopted.

**Drive uploads (unchanged, confirmed with user):** digest, daily_focus, research_index,
company_mentions, company_page, synthesis_latest + dated, deep_dive_queue. Source PDFs NEVER uploaded.

**Still pending / next:**
- Normal same-day run will populate "Today" section (04-Jun regen showed all 11 under Rolling —
  date artifact only, not a bug).
- OT7 deep dive end-to-end test (queue auto-populated by OT8 `--queue`).
- Duplicate bat cleanup: `get_latest_research.bat` exists in BOTH root and scripts/ (both fixed/identical);
  decide single canonical location. Desktop `research.bat` (OneDrive) not yet reviewed.
- Known gaps unchanged: typed prompts don't emit JSON tag tail (→ some Date/Companies = NA); BSE code col (now fixed — see session 2026-06-06); no OCR.

---

### Session 2026-06-06 — OT7 full build, CI fix, email attachments

**Asked:**
- Status check on OT7; how to trigger locally and via Streamlit
- Make Gemini key handling consistent with concall/OT8 (use gemini_pool.BucketPool)
- Confirm app.py in git; fix Deep Dive page missing from sidebar
- Fix BSE code resolve (539730 / 522101 not resolving)
- Auto-refresh universe/master_list.csv weekly
- Fix deepdive CI not running; email with .md + PDF attachments

**Delivered:**

**OT7 — company_deep_report.py (fully built + consistent):**
- Replaced inline `KeyPool` with `gemini_pool.BucketPool` + `load_keys()`
  → reads `GEMINI_API_KEY_1…6` automatically, no .env changes needed
- `DEEPDIVE_MODELS` chain: 5 models × N keys = N×5 daily buckets
- Proper `AllBucketsExhausted` / `FatalCallError` handling
- Multi-company: `--names "TCS,INFY,VENUSREM"` or `--add "TCS,INFY"`
- Local trigger: `run_deepdive.bat` (interactive, run-now or queue)
- Fetch completed: `get_deepdive.bat` → Obsidian/browser

**Streamlit Deep Dive page:**
- Fixed `drive_download not defined` → `_app_drive_download/_app_drive_upload` helpers
- Fixed `No module named 'daily_research_summary'` → `scripts/` added to `sys.path`
- Queue input accepts comma-separated companies; resolves all, warns on failures

**BSE code resolution fixed:**
- Root cause: pandas reads numeric CSV as float64 → `"522101.0"` not `"522101"`
- Fix: normalise to `int(float(x))` string in `resolve_isin()` and `row_out()`

**Universe / master_list.csv:**
- `build_company_universe.py` now writes BOTH `company_repo/_index/company_universe.csv`
  AND `universe/master_list.csv`; CI auth updated; 5,542 companies
- `universe_refresh.yml` — weekly Sunday 03:00 IST + manual dispatch

**deepdive.yml CI bug fixed:**
- Root cause: `NOTIFY_SINCE: ${{ steps.run_deepdive.outputs.started_at }}` at job-level
  env → step outputs invalid there → workflow failed to parse (0s, 0 jobs) on every run
- Fix: moved `NOTIFY_SINCE` to the notify step's own `env` block

**Email attachments:**
- `format_deepdive_pdf.py` — md → styled HTML → PDF (weasyprint): A4, cover page,
  navy table headers, alternating row shading, page numbers
- `notify_deepdive.py` — attaches `.md` + `.pdf` to each email (MIMEBase)
- `deepdive.yml` + `requirements.txt` — weasyprint + system deps added

**Queue status 2026-06-06:**
- `venusrem` pending (queued 05-Jun 23:08 via Streamlit)
- `522101` pending (queued 06-Jun 16:40 via local bat)
- Will process at CI run 08:00 IST 07-Jun-2026

**Commits:** cff95a0 · 32b893d · c7192c2 · 28caf90 · bf709bf · 59ab7c2 · c172862 · 5db3b7d

**Still pending:**
- OT7 first successful end-to-end CI run + email confirmation
- Fix 2B: BSE Direct API secondary ingestion
- OT9: Streamlit performance optimisation (P2, deferred)
- Known gaps: typed prompts no JSON tag tail; no OCR for scanned PDFs
