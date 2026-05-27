# PHASE 2 SPEC — Company Repository & Concall Intelligence

**Status:** active · **Created:** 2026-05-22 · **Last updated:** 2026-05-27
Companion to `DESIGN.md` / `PROJECT_STATUS.md`.

---

## 1. Objective

Automate the user's manual company-research loop: collect corporate documents
(concalls, annual reports, research reports, credit ratings, DRHPs), summarise
them with Gemini, extract structured guidance & financial facts, keep a living
per-company page on Google Drive, and surface selected views in Streamlit —
without overrunning free-tier limits.

---

## 2. Document types in scope

| doc_type        | Source feed / approach                          | Prompt file               | Size range   | Gemini strategy        |
|-----------------|------------------------------------------------|---------------------------|--------------|------------------------|
| `concall`       | Screener filter 76106                          | `concall_prompt.txt`      | 10–40 pages  | File API               |
| `annual_report` | Screener filter 103635                         | `annual_report_prompt.txt`| 100–400 pages| File API + map-reduce  |
| `presentation`  | Screener filter 76295                          | `presentation_prompt.txt` | 10–60 pages  | File API               |
| `rating`        | Screener filter 215435                         | `rating_prompt.txt`       | 5–20 pages   | File API               |
| `drhp`          | BSE/NSE DRHP filings (new — see §10.1)         | `annual_report_prompt.txt`| 200–600 pages| File API + map-reduce  |
| `results`       | Screener `/results/latest/` (structured table) | No LLM — numbers only     | HTML table   | Direct scrape          |

**Map-reduce rule:** if PDF > 300 pages (annual report or DRHP), split into
~100-page chunks, summarise each chunk independently, then run a final synthesis
pass over the chunk summaries. This avoids context-window overflow.

**File API:** all PDFs (regardless of size) are uploaded to Gemini's File API
as a temp file, then referenced in the prompt — not passed as inline base64.
Gemini auto-deletes File API uploads after 48 hours; our Drive copy is the
permanent record.

---

## 3. Priority order (token-budget driven)

1. **P1 — Daily concall summariser (MVP).** New concall transcripts → Gemini →
   structured page note + guidance JSON. Ship first.
2. **P1 — Daily results storage.** During results season, `scrape_results_table.py`
   runs daily (already built in Stage A) and upserts rows into
   `_index/results.parquet`. No LLM — pure structured scrape.
3. **P2 — Guidance extraction.** From each processed concall, extract explicit
   + derived guidance into `_index/guidance_tracker.parquet`.
4. **P2 — Annual report & rating summariser.** Same pipeline as concall but with
   the AR/rating prompts and map-reduce for large docs.
5. **P2 — Research report & presentation summariser.** Lower volume, on-demand
   or opportunistic.
6. **P2 — DRHP summariser.** Triggered by new IPO filings. Map-reduce mandatory.
7. **P2 — History backfill.** Walk concalls backward from 2025-05-20 overnight.
8. **P2 — Deep research reports.** Full forensic report for portfolio companies
   + on-demand via Streamlit or Drive upload.

---

## 4. P1 — Daily concall pipeline (MVP)

```
ingest_company_docs.py   [already built — Stage A]
  scrape Screener announcements → download new PDFs to
  company_repo/<ISIN>/documents/ → add to _index/processing_queue.parquet
  (status=pending)

extract_concall.py       [Stage B — build next]
  for each pending concall in the queue:
    1. download PDF bytes from Drive
    2. upload to Gemini File API (temp)
    3. call Gemini with concall_prompt.txt
    4. parse response into:
       (a) detailed markdown note  → append to company_repo/<ISIN>/company_page.md
       (b) structured JSON tail    → upsert into _index/quarterly_facts.parquet
                                     and _index/guidance_tracker.parquet
    5. mark queue row status=done
  on Gemini 429: stop cleanly, leave remaining rows pending (next run resumes)
  on Gemini 500/503: retry once with exponential back-off, then mark status=error

extract_results.py       [already built as scrape_results_table.py — Stage A]
  results season: pull quarterly result numbers → _index/results.parquet

cleanup_company_docs.py  [already built — Stage A]
  delete raw PDFs older than 10 days (keeps company_page.md + index parquets)
```

One Gemini call per concall produces **both** the detailed page note and the
structured facts — no double spend.

---

## 5. Guidance extraction — schema

The structured JSON tail produced by `extract_concall.py` feeds two parquets:

**`_index/quarterly_facts.parquet`** — one row per (isin, quarter):
```
isin, symbol, company_name, quarter, fy_year,
revenue_q, ebitda_q, pat_q, margin_pct,
volume_q, capacity_q,
revenue_12m, pat_12m,
processed_at, source_doc_id
```

**`_index/guidance_tracker.parquet`** — one row per (isin, quarter, metric, horizon):
```
isin, symbol, company_name, quarter,
metric,               # revenue / ebitda / pat / margin / volume / capacity
guidance_type,        # explicit | derived
horizon_fy,           # FY27 / FY28 / FY29 / FY30
value, unit,          # absolute value + unit string (e.g. "cr", "%", "MT")
cagr_pct,             # derived CAGR if guidance_type=derived
notes,                # calc notes from prompt (annualisation method, etc.)
processed_at, source_doc_id
```

`guidance_type=explicit` = management stated a direct number.
`guidance_type=derived` = we computed a CAGR from absolute guidance following
the concall prompt's calculation protocol.

---

## 6. Output format — dual write design

Every processed document is written to **two places simultaneously**:

### 6a. Company page (persisted forever)
`company_repo/<ISIN>/company_page.md`
- One file per company, summaries appended chronologically (newest at bottom)
- Reliance Q4 FY26 concall → appended to Reliance's page forever
- Optional: `company_page.docx` alongside (toggle `OUTPUT_COMPANY_DOCX`)

### 6b. Day page (auto-deleted after 30 days)
`company_repo/_daily/concall_26_may2026.md`
- One file per calendar date, all companies with concalls that day
- Reliance Q4 FY26 also appears here alongside the other 199 companies
- Format: `## RELIANCE — Q4 FY26\n<full Gemini analysis>\n---\n`
- Auto-deleted by `cleanup_company_docs.py` after 30 days
- Optional: `concall_26_may2026.docx` alongside (toggle `OUTPUT_DAY_DOCX`)

### 6c. Toggle flags (in each extractor script's CONFIG block)
```python
OUTPUT_COMPANY_MD   = True    # write company_page.md
OUTPUT_DAY_MD       = True    # write _daily/concall_DD_MMMYYYY.md
OUTPUT_COMPANY_DOCX = False   # .docx alongside company_page.md  [Stage C]
OUTPUT_DAY_DOCX     = False   # .docx alongside day page          [Stage C]
```
Set any to False to skip that output. Both MD flags True = full dual write.

### 6d. Day page naming convention
`concall_DD_MMMYYYY.md` — e.g. `concall_26_may2026.md`, `concall_01_jan2027.md`
(lowercase month, no leading zero suppression on day, full 4-digit year)

---

## 7. Storage layout (Google Drive)

```
company_repo/
  <ISIN>/
    documents/          raw PDFs — auto-deleted after 10 days
    company_page.md     living page: per-doc summaries appended chronologically
    company_page.docx   optional Drive-native copy [Stage C, toggle]
    deep_report.md      full forensic report [Stage E]
    deep_report.docx    Drive-native copy    [Stage E]
  _daily/
    concall_26_may2026.md    all concall summaries for 26 May 2026
    concall_26_may2026.docx  optional [Stage C, toggle]
    results_26_may2026.md    all results summaries for 26 May 2026 [future]
    ...                      auto-deleted after 30 days by cleanup script
  _index/
    company_universe.csv      ISIN, symbol, exchange, name, aliases
    processing_queue.parquet  (isin, doc_id, doc_type) → status, drive_file_id
    quarterly_facts.parquet   isin × quarter → financial actuals (concall + AR + presentation)
    guidance_tracker.parquet  isin × quarter × metric × horizon → guidance
    results.parquet           Screener HTML-scraped structured numbers (scrape_results_table.py)
    results_gemini.parquet    Gemini-extracted financials from results PDFs (extract_results.py)
    ratings.parquet           isin × source_doc_id → agency, rating, outlook, action
    deep_research_requests.csv  user-submitted companies for deep report [Stage E]
```

---

## 8. P2 — Extractors for other doc types (Stage D)

All extractors share identical design: queue-driven, dual output (company page +
`_daily/` day digest with 30-day TTL), toggle flags, 3-path Drive auth, multi-key
Gemini rotation. Shared infrastructure lives in `_extractor_base.py`.

**Shared infrastructure — `scripts/_extractor_base.py`**
- `get_drive()` — 3-path auth (service account, saved token, OAuth flow)
- `GeminiKeyPool` — inline PDF calls, round-robin key rotation, backoff
- `load_queue / save_queue` — queue parquet R/W
- `load_parquet / save_parquet` — generic parquet upsert helpers
- `extract_md_tables` — fenced + unfenced pipe-table parser
- `append_day_page(doc_type, ...)` — parameterised day digest writer
- `append_company_page(doc_type_label, ...)` — parameterised company page writer
- `load_api_keys()` — reads `GEMINI_API_KEY_1..N` and plain `GEMINI_API_KEY`

**`extract_results.py`** *(Stage B — 94 pending)*
- Uses `results_prompt.txt` (focused quarterly financials table)
- Upserts into `_index/results_gemini.parquet` (complements HTML-scraped `results.parquet`)
- Day files: `_daily/results_DD_MMMYYYY.md`

**`extract_annual_report.py`** *(Stage D — 50 pending)*
- Uses `annual_report_prompt.txt` (forensic lens)
- Map-reduce for PDFs > 12 MB: chunk → per-chunk Gemini calls → synthesis pass
  (uses `pypdf` for page-accurate splitting; falls back to byte-chunking)
- Upserts into `_index/quarterly_facts.parquet` (annual totals stored as FY-labelled rows)
- Day files: `_daily/annual_report_DD_MMMYYYY.md`

**`extract_presentation.py`** *(Stage D — 37 pending)*
- Uses `presentation_prompt.txt` (operational KPIs + narrative forensics)
- Best-effort upsert into `_index/quarterly_facts.parquet`
- Day files: `_daily/presentation_DD_MMMYYYY.md`

**`extract_rating.py`** *(Stage D — 25 pending)*
- Uses `rating_prompt.txt` (solvency + credit forensics)
- Upserts into `_index/ratings.parquet` (agency, rating, outlook, action, instrument)
- Day files: `_daily/rating_DD_MMMYYYY.md`

**`extract_drhp.py`** — planned Stage G; map-reduce mandatory; produces standalone
`drhp_report.md` (not appended to `company_page.md`).

---

## 8. P2 — Deep-research reports (Stage E)

`company_deep_report.py` — for a given company:
- Feeds all available documents + `company_page.md` to Gemini using the
  `comapnydeepdive_prompt.txt` forensic chain (Phases 1–4).
- Produces `deep_report.md` + `deep_report.docx` on Drive.

**Two trigger paths (both supported):**

1. **Streamlit input (preferred):** a company multiselect on the dashboard
   writes ISINs to `_index/deep_research_requests.csv`. The next scheduled run
   consumes the list.
2. **Drive upload fallback:** user drops a file (one ISIN per line) into
   `company_repo/_requests/`. The script checks this folder each run.
3. **Portfolio auto-trigger:** companies in the user's portfolio (My Portfolio
   page) are automatically queued for a deep report on first encounter.

---

## 9. LLM & quota strategy

- **Primary model:** `gemini-1.5-flash` — fast, handles PDF via File API,
  generous free tier. Switch to `gemini-1.5-pro` for deep reports (more
  reasoning depth) via a per-extractor config flag.
- **Multi-key rotation:** support `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, …
  `GEMINI_API_KEY_N` (as many as the user adds). The extractor round-robins
  across keys and, on 429 from one key, immediately tries the next. Only stops
  when all keys are rate-limited. This multiplies the effective free-tier quota
  by the number of keys.
- **Resumable by design:** the persistent queue holds every pending task; each
  run keeps processing subject to quota, then the next run continues. Nothing is
  lost or re-done.
- **Expected volume:** ~50 concalls/day in peak season — comfortably within one
  free key's daily limit. Multi-key absorbs backfill surges.
- **Night backfill window:** 00:00–05:00 IST (GitHub Actions cron). Multi-key
  rotation makes the most of this quiet window.
- **Second provider (future option):** a paid Anthropic Claude API key, behind
  a `LLM_PROVIDER` config flag, off by default. Automated pipelines can only
  use API keys — not Claude.ai / Gemini app subscriptions.

---

## 10. Dashboard additions — what to show and what NOT to show

### What to build (NSE subset only, load from Drive on 5-min cache)

**Concall Feed page (Stage C)**
- SHOW: today's concall summaries (company name, quarter, 2-line A-1 summary,
  score card link, link to full `company_page.md` on Drive)
- SHOW: "Top concall of the day" = highest conviction based on explicit guidance
  beat vs. prior quarter (when enough history exists; else fallback to most
  recent concall)
- SHOW: "Top results of the day" = highest YoY PAT growth from `results.parquet`
  (linked to the structured numbers table)
- DO NOT SHOW: raw PDF links, internal doc IDs, processing queue internals

**Guidance page (Stage D)**
- SHOW: guidance leaderboard — companies with highest explicit FY guidance
  upgrades vs. prior quarter
- SHOW: guidance-missed list — companies where actuals came in below prior
  explicit guidance
- SHOW: serial over/under-promiser ranking (needs ≥ 4 quarters of history;
  hide the widget entirely if < 4 quarters available — no empty tables)
- DO NOT SHOW: derived guidance in the leaderboard (mark it visually as
  "estimated" if shown at all; never mix with explicit guidance in rankings)

**Stock Detail page — Fundamentals strip (Stage C)**
- SHOW: latest quarter actuals (revenue, PAT, margin, YoY growth) from
  `quarterly_facts.parquet`
- SHOW: latest explicit guidance summary (1–2 rows, clearly labelled)
- SHOW: link to `company_page.md` and `deep_report.md` on Drive (if they exist)
- SHOW: "Request Deep Report" button (triggers Stage E pipeline)
- DO NOT SHOW: derived guidance numbers without a clear "estimated" label
- DO NOT SHOW: the Fundamentals strip at all if no quarterly_facts exist yet
  for that company (hide gracefully, don't show empty cards)

**My Portfolio page — additions**
- SHOW: inline guidance status per holding (met / missed / pending)
- DO NOT SHOW: cost price, P&L, or any personally identifiable position data
  (existing constraint — unchanged)

### What NOT to build in the dashboard
- Raw processing queue viewer (internal plumbing — use Drive directly)
- DRHP page (DRHPs are pre-IPO one-offs; link to `drhp_report.md` from Stock
  Detail if it exists, no separate page)
- Backfill progress bar (scheduled job internals — visible in GitHub Actions
  logs, not the dashboard)
- Research report feed (low volume; surface via Stock Detail link only)

---

## 11. DRHP source — open item §10.1

Screener does not have a DRHP feed. DRHP PDFs must be sourced from:
- **BSE:** `https://www.bseindia.com/markets/PublicIssues/DRHPs.aspx` (scrape)
- **SEBI:** `https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecent=yes` (scrape)
- **Alternative:** NSE emerging companies page

`ingest_company_docs.py` needs a new `drhp` feed entry pointing to one of the
above. Build alongside Stage D (not a Stage B blocker).

---

## 12. Scheduling / auto-run (GitHub Actions)

`company_repo.yml` — new workflow, separate from the Phase 1 `daily.yml`:

```yaml
# runs several times daily + a heavy backfill window overnight
on:
  schedule:
    - cron: '30 4,7,10,13 * * 1-5'   # 10:00, 13:00, 16:00, 19:00 IST (Mon-Fri)
    - cron: '30 18 * * 0'            # 00:00 IST Sun (backfill window start)
```

Steps (all `continue-on-error: true`):
1. `ingest_company_docs.py` — refresh the queue
2. `scrape_results_table.py` — daily results numbers
3. `extract_concall.py` — consume pending concalls
4. `extract_annual_report.py` — consume pending ARs (when built)
5. `extract_rating.py` / `extract_presentation.py` (when built)
6. `cleanup_company_docs.py` — purge old raw PDFs
7. `company_healthcheck.py` — verify index freshness (to be built)

---

## 13. Build order

- **Stage A** — `build_company_universe.py`, `ingest_company_docs.py`,
  `scrape_results_table.py`, `cleanup_company_docs.py`. **DONE 2026-05-23.**
- **Stage B** — `extract_concall.py` + multi-key Gemini rotation. **DONE 2026-05-26.**
  `extract_results.py` + `_extractor_base.py`. **DONE 2026-05-26.**
- **Stage D extractors** — `extract_annual_report.py`, `extract_presentation.py`,
  `extract_rating.py`. **DONE 2026-05-26.**
- **Portfolio filter** — all Stage D extractors + `extract_results.py` restricted to
  portfolio ISINs from Drive. **DONE 2026-05-27.** Reduces Phase 2 run time 51 → 5–8 min.
- **Pipeline health dashboard** — `write_phase2_status.py` + `app.py` sidebar/banner
  showing Phase 1 + Phase 2 status (queue counts, age, errors). **DONE 2026-05-27.**
- **Streamlit fix** — yfinance removed from root `requirements.txt`; app.py crash
  from `curl_cffi==0.15.0` resolved. **DONE 2026-05-27.**
- **Stage C** — `company_page_generator.py` (regenerates `.md` header/overview
  section) + Concall Feed dashboard page + Stock Detail Fundamentals strip.
  *Pending — next after Phase 2 running stably.*
- **Stage D dashboard** — `build_guidance_scorecard.py` + Guidance dashboard page
  (leaderboard, missed guidance).  *Pending.*
- **Stage D results filter** — filterable YoY/QoQ results table in app.py.  *Pending.*
- **Stage E** — `company_deep_report.py` (portfolio + on-request via Streamlit
  input or Drive upload). Prompt `comapnydeepdive_prompt.txt` already exists.  *Pending.*
- **Insider buy/sell** — source not identified (BSE/NSE feeds need evaluation).  *Blocked.*
- **Stage F** — history backfill from 2025-05-20 (overnight cron window).  *Pending.*
- **Stage G** — DRHP ingestion + `extract_drhp.py` (needs source identified
  per §11).  *Pending.*

---

## 14. Open items / decisions

1. **`GEMINI_API_KEY_1` … `_N`** — user to create (free, AI Studio) and add as
   GitHub Actions + Streamlit Cloud secrets.
2. **DRHP source URL** — BSE scrape is the most reliable path; needs a one-off
   HTML inspection to confirm selectors. Deferred to Stage G.
3. **"Top of the day" definition** — decided at Stage C build: highest YoY PAT
   growth for results; most recent with explicit guidance upgrade for concalls.
4. **Google Workspace account** — not required (10-day cleanup keeps Drive lean).
5. **Insider buy/sell source** — Screener insider feeds (76296/76297) are
   one-line facts, not documents. Alternative structured source (NSE/BSE insider
   disclosures) is TBD. Company page promoter-activity section stays empty.
6. **Screener feed pagination** — feeds show only ~25–50 items; same-day
   completeness relies on running the pipeline several times daily. Peak-season
   risk noted and mitigated by the multi-run schedule.
7. **Model per doc type** — `gemini-1.5-flash` default for concall / rating /
   presentation; `gemini-1.5-pro` for annual report / DRHP / deep report. Flag
   in each extractor script's CONFIG block.
8. **Schedule** ✅ **RESOLVED 2026-05-27.**
   Final schedule: Mon–Fri every 3h (10:00/13:00/16:00/19:00 IST) + once 02:00 IST
   overnight = 5 runs/day. Saturday 3 runs (10/13/16 IST). Sunday 1 run (13:00 IST).
   = 29 runs/week. Combined Phase 1 + Phase 2 budget: ~1,360–1,710 min/month
   (within 2,000 free). Portfolio filter reduced Phase 2 avg run time from ~51 min
   to ~5–8 min — this is what made the schedule viable without hitting the quota.
9. **Deep dive report script** *(pending — Stage E)* — `comapnydeepdive_prompt.txt`
   exists. `company_deep_report.py` not yet written. Trigger paths: Streamlit button
   on Stock Detail page + Drive upload fallback. Priority: after Phase 2 running stably.
10. **Insider buy/sell** *(blocked)* — Screener feeds 76296/76297 are one-line facts,
    not document PDFs. Need to evaluate BSE/NSE structured insider disclosure feeds.
    Company page "buying/selling" section will remain empty until resolved.
