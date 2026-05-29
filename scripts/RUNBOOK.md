# Signals India — Runbook & FAQ

**Single source of truth for operations, troubleshooting, and daily use.**
Last updated: 2026-05-29

---

## 1. Daily Workflow (what to do each day)

| Time | Action |
|---|---|
| 16:00 IST | Phase 1 auto-runs (data pipeline). No action needed. |
| 16:00 IST | cron-job.org also fires Phase 1 as backup. |
| 17:00–23:00 IST | Phase 2 runs 3-6× automatically (concall processing). |
| Evening | Double-click `get_latest_concall.bat` to read today's concalls in Obsidian. |
| Anytime | Open Streamlit app to view signals, portfolio, guidance tables. |

---

## 2. Local Tools (bat files — double-click from project root or desktop shortcut)

| File | What it does | Expected output |
|---|---|---|
| `get_latest_concall.bat` | Downloads latest daily digest from Drive, fixes Obsidian table rendering, opens in Obsidian | Window shows download progress, file opens in Obsidian automatically |
| `list_concalls.bat` | Shows last 10 available digest files with age, lets you pick one by date | List printed, then prompts for date e.g. `29may2026` |
| `get_company_intel.bat` | Prompts for ISIN/symbol, downloads company_page.md (Table A + GF1-4 + summaries), opens in Obsidian | Prompts for input, shows download, opens file |

**One-time setup for bat files:**
Edit `OUTPUT_DIR` in both scripts to point inside your Obsidian vault:
- `scripts/fetch_latest_concall.py` line ~32: `OUTPUT_DIR = Path(r"C:\Users\vaido\Documents\YourVault\concalls")`
- `scripts/fetch_company_intel.py` line ~32: `OUTPUT_DIR = Path(r"C:\Users\vaido\Documents\YourVault\company_intel")`

**To put on desktop:** Right-click any `.bat` → Send to → Desktop (create shortcut)

---

## 3. Streamlit App — Page Guide

| Page | What to use it for |
|---|---|
| Market Overview | Market health score, breadth, FII/DII, Nifty chart, sector rotation |
| Today's Signals | All buy/add/hold signals across strategies. Use growth filter (expander) to add revenue/PAT filter. |
| Company Intel | Browse AI-generated daily digests by date/type. Company Page tab for full company brief. |
| **Mgmt Guidance** | **Table A + GF1-4 data.** Tab 1: filter by company/metric/horizon. Tab 2: active watchlist ranked by quality score. Tab 3: guidance × momentum combined score. |
| **Doc Viewer** | **Per-company view.** Enter ISIN or symbol → tabs for Company Page (rendered markdown), GF Tables (dataframes), Results, Quarterly Guidance. Best for reading tables. |
| My Portfolio | Holdings from your `.xls` on Drive. Overlays signals, features, RS rank. |
| Graphs | Chart gallery for all buy/add signals. Filter by strategy + zone. |
| Stock Detail | Single stock chart + signals + management guidance section. |
| Strategy Docs | Explanation of each strategy's rules and intent. |

---

## 4. Viewing Table A / GF1-GF4

**Three ways:**

1. **Obsidian** (best for reading) → `get_latest_concall.bat` or `get_company_intel.bat`
2. **Streamlit Doc Viewer** → enter symbol → GF Tables tab → proper dataframe grids
3. **Streamlit Mgmt Guidance** → Guidance Tracker tab → filter by company

---

## 5. Common Issues & Fixes

### Phase 1 — "stale data" warning in app
**Symptom:** App banner shows "Phase 1 has not run for Xh — signals may be STALE"
**Cause:** Phase 1 last ran >30h ago (usually means today's run hasn't fired yet, or failed)
**Fix:**
1. Check GitHub Actions: `gh run list --limit 5` or go to github.com/vrvfin/signals-india/actions
2. If last run failed at "Health check" step → usually a transient data issue, re-trigger: GitHub → Actions → Daily Pipeline → Run workflow
3. If no run at all today → check cron-job.org dashboard to confirm the 16:00 IST trigger fired

### Phase 2 — "cancelled" runs
**Symptom:** GitHub Actions shows Phase 2 run as "cancelled"
**Cause:** Two scenarios: (a) run hit the 3-hour timeout — normal during peak concall season when queue is large; (b) manual cancellation
**Fix:** Nothing to do — next scheduled run picks up remaining queue items automatically

### Phase 2 — company not being processed
**Symptom:** Expected concall not appearing in daily digest
**Cause A:** Company not in your portfolio (portfolio-filtered extractors skip non-portfolio companies; concalls are universal but capped at Screener's 25-item list)
**Cause B:** Screener 25-item cap — concalls beyond position 25 are missed (Fix 2B in backlog)
**Fix:** Ensure company ISIN is in your portfolio file on Drive

### Screener cookie expired
**Symptom:** `strategy_canslim.py`, `strategy_pead.py`, or `ingest_fundamentals.py` fail; healthcheck shows WARNING
**Fix:**
1. Log into screener.in in browser
2. Open DevTools → Application → Cookies → copy `sessionid` value
3. GitHub → repo → Settings → Secrets → `SCREENER_SESSION_COOKIE` → Update
4. Re-run Phase 1 manually from GitHub Actions

### `get_latest_concall.bat` — "conda is not recognized" or "Python was not found"
**Symptom:** Error about conda or Python not found when running bat from desktop/Explorer
**Cause:** conda and Python are not on the system PATH outside Anaconda Prompt
**Fix (already applied 2026-05-29):** Bat files use hardcoded Python path:
`C:\Users\vaido\.conda\envs\signals-india\python.exe`
No conda activation needed — calls Python directly.
If the env is ever rebuilt at a different path, update the `set PYTHON=` line in each bat file.

### Obsidian tables not rendering (pipe characters visible)
**Symptom:** GF1-4 tables show as raw `| col | col |` text instead of grids
**Cause:** Gemini splits wide table rows across 2 physical lines; Obsidian can't parse split rows
**Fix:** Re-run `get_latest_concall.bat` — the fixer now joins split rows before saving. Files are always re-fixed on every run.

### Drive auth error / OAuth token expired
**Symptom:** Scripts fail with "credentials" or "token" error
**Fix:** Token auto-refreshes via refresh_token (lasts indefinitely). If it fails completely:
1. Delete `_sec/tk.json` locally
2. Re-run any script — it will open a browser for re-auth
3. On GitHub Actions, update `GDRIVE_OAUTH_TOKEN_JSON` secret with new token JSON

### Node.js deprecation warnings in GitHub Actions
**Status:** Fixed 2026-05-29. `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` added to both workflows.

---

## 6. Pipeline Architecture (quick reference)

```
Phase 1 (daily.yml) — weekdays 16:00 IST
  Triggers: GitHub cron (10:30 UTC) + cron-job.org (16:00 IST) + manual
  Steps: universe → OHLCV → indices → macro → FII/DII → features
         → 10 strategy scripts → aggregate → market_state → healthcheck
  Weekly (Mon): fundamentals + market_cap enrichment

Phase 2 (phase2.yml) — 3-7× per day (peak season)
  Triggers: dense cron + Phase1 backup (ACTIONS_PAT) + cron-job.org every 2h
  Steps: ingest_company_docs → scrape_results_table → extract_concall
         → extract_results → extract_rating → extract_presentation
         → extract_annual_report → cleanup → write_phase2_status

Streamlit app — reads Drive, 5-min cache
  Live at: [check share.streamlit.io for your app URL]
```

---

## 7. Key File Locations

| What | Where |
|---|---|
| Daily concall digests | `company_repo/_daily/concall_DD_MMMYYYY.md` |
| Company pages | `company_repo/<ISIN>/company_page.md` |
| Quarterly guidance | `company_repo/_quarterly/QXFY_mgmt_guidance.md` |
| Guidance parquet | `company_repo/_index/guidance_tracker.parquet` |
| GF1 statements | `company_repo/_index/gf1_guidance_statements.parquet` |
| GF4 quality flags | `company_repo/_index/gf4_quality_flags.parquet` |
| Results summary | `company_repo/_index/results.parquet` |
| Features | `features/latest.parquet` |
| Signals | `signals/per_strategy/<name>/latest.csv` |
| Health report | `logs/health/latest.json` |
| Phase 2 status | `logs/health/phase2_latest.json` |
| Portfolio file | `portfolio/<your-file>.xls` (most recent picked automatically) |

---

## 8. Secrets Reference

| Secret | Where used | Expiry |
|---|---|---|
| `GDRIVE_FOLDER_ID` | All scripts | Never |
| `GDRIVE_OAUTH_TOKEN_JSON` | All scripts (Drive auth) | Auto-refreshes |
| `GDRIVE_OAUTH_CLIENT_SECRET_JSON` | All scripts (Drive auth) | Never |
| `SCREENER_SESSION_COOKIE` | fundamentals, CANSLIM, PEAD | ~30 days — renew manually |
| `GEMINI_API_KEY` | Phase 2 extractors | Never (check GCP console) |
| `ACTIONS_PAT` | Phase 1→Phase 2 backup trigger | Check GitHub token expiry |

---

## 9. How to Add a New Strategy

1. Copy an existing strategy script (e.g. `strategy_momentum.py`) → `strategy_<name>.py`
2. Output must write: `signals/per_strategy/<name>/latest.csv` with columns: `symbol, date, strategy, zone_type, score, entry, stop, reason`
3. Add step to `daily.yml` with `continue-on-error: true`
4. Add entry to `STRATEGY_DOCS` dict in `app.py` for the Stock Detail page
5. `aggregate_signals.py` and `pipeline_healthcheck.py` auto-discover — no changes needed

---

## 10. Pending Work (current priority order)

| Priority | Item | Notes |
|---|---|---|
| P0 | OT8: Local doc summarisation → Drive context store | Upload local PDFs → Gemini summary → Drive |
| P1 | OT7: Deep research report (company_deep_report.py) | Prompt exists; UI + script to build |
| P2 | OT3: Mgmt said vs delivered (GF2 tracker) | Needs data — earliest Nov 2026 |
| P3 | OT9: Streamlit performance optimisation | After P1 items done |
| P4 | End-to-end documentation expansion | This doc + in-app help |
| P4 | Fix 2B: BSE Direct API (Screener 25-item cap) | Medium effort |
