# PHASE 2 SPEC — Company Repository & Concall Intelligence

**Status:** spec for review · **Created:** 2026-05-22 · move to repo root after review.
Companion to `DESIGN.md` / `PROJECT_STATUS.md`.

---

## 1. Objective

Automate the user's manual company-research loop: collect concalls (and later
annual reports / research / ratings), summarise them with an LLM, keep a living
per-company page on Google Drive, and surface selected views in Streamlit —
without overrunning free-tier limits.

## 2. Priority order (token-budget driven)

Everything is gated by daily LLM quota. Order, highest first:

1. **P1 — Daily concall summariser (the MVP).** Each day, pick up every new
   concall transcript from Screener's announcements feed, summarise it, update
   the company page + structured indexes. *This is what we build and ship first.*
2. **P2 — History backfill.** Once the daily loop runs comfortably and spare
   quota exists, walk concalls backward from **2025-05-20**.
3. **P2 — Deep-research reports.** The full forensic report (`prompt.txt`
   Phase 1-4) for: (a) portfolio companies automatically; (b) any company the
   user requests — via a Streamlit input (preferred) or a Drive-upload fallback.
   The company's existing `company_page.md` is an input to the deep report.

If quota is tight, P2 items become **on-demand**: backfill a company, then run
its deep report.

## 3. Universe & documents

- **Daily pipeline** needs no universe file — it processes whatever appears on
  Screener's announcements feed (covers NSE + BSE alike).
- `build_company_universe.py` builds an NSE + BSE master list keyed by **ISIN**
  (handles renames; never orphans history). Used for identity mapping and the
  Streamlit NSE-subset.
- Documents in scope: concall transcripts (P1), then annual reports, research
  reports, credit ratings, investor presentations (P2 / deep research).
- Web / social / YouTube / EPFO sections of `prompt.txt` are **not API-automatable**
  → marked `DATA_MISSING`; done manually by the user. ("Deep Research" is a
  Gemini *app* feature, no API — cannot be scripted.)

## 4. P1 — Daily concall pipeline (MVP)

```
ingest_company_docs.py   scrape Screener announcements (filter: Transcript/
                         Concall) for recent days → download new transcript
                         PDFs to company_repo/<ISIN>/documents/ → add to
                         _index/processing_queue.parquet (status=pending)
extract_concall.py       take pending concalls → Gemini (prompt.txt #1, the
                         detailed concall prompt) → produces (a) the detailed
                         quarter note .md, (b) a structured JSON tail.
                         Appends the note to company_repo/<SYMBOL>/company_page.md;
                         writes JSON to _index/quarterly_facts.parquet +
                         guidance_tracker.parquet. Marks queue entry done.
                         On a Gemini rate-limit: stop gracefully, leave the
                         rest pending — the next scheduled run resumes.
extract_results.py       results season: pull quarterly result numbers →
                         _index/results.parquet → "top results of the day".
cleanup_company_docs.py  delete raw PDFs older than 10 days (keep page + facts).
```

One Gemini call per concall produces **both** the detailed page note and the
structured facts — no double spend.

## 5. Storage layout (Google Drive)

```
company_repo/
  <ISIN>/
    documents/        raw PDFs — auto-deleted after 10 days
    company_page.md   living page: Exec Summary + Business Overview (regenerated)
                      + append-only Quarterly History
    company_page.docx Drive-native copy (regenerated with the .md)
    deep_report.md / .docx   the forensic report (P2, when generated)
  _index/
    company_universe.csv      ISIN, symbol, exchange, name, aliases
    processing_queue.parquet  what's pending / done, per (ISIN, quarter, doc)
    quarterly_facts.parquet   symbol x quarter -> growth, margin, ...
    guidance_tracker.parquet  symbol x quarter -> guidance, actual, met?
    results.parquet           results-season numbers
```

## 6. P2 — Deep-research reports

- `company_deep_report.py` — for a company: feed its collected documents +
  `company_page.md` to Gemini using the `prompt.txt` forensic prompts (AR
  chunked map-reduce) → `deep_report.md` + `.docx` on Drive.
- **Triggers:** portfolio companies automatically; a user request list at
  `_index/deep_research_requests.csv` (written by a Streamlit input — a company
  multiselect) or, as a fallback, a file dropped in `company_repo/_requests/`.
- The next scheduled run consumes the request list.

## 7. LLM & quota strategy

- **Primary:** Gemini free tier (`GEMINI_API_KEY` from Google AI Studio — a new
  secret, same handling as the others).
- **Resumable by design:** a persistent daily queue (`processing_queue.parquet`)
  holds every pending task; each run keeps working through it subject to quota,
  then the next run continues. Nothing is lost or re-done.
- **Expected volume:** ~50 concalls/day, ~300-400/week in season — comfortably
  within the daily free tier. The queue mostly absorbs the backfill surge and
  catch-up after a quota hit.
- **Multiple runs per day:** the workflow is scheduled several times daily plus
  a **heavy-backfill window 00:00-05:00 IST** (quiet hours), so backfill chews
  through history overnight.
- **Optional second provider** (if you want more daily throughput): a paid
  Gemini tier or an Anthropic Claude API key, behind a config flag, off by
  default. NOTE: an automated pipeline can only use an LLM via an **API key** —
  it cannot tap a Claude.ai / Gemini-app subscription. "Using Claude at night"
  therefore means adding a (paid) Anthropic API key; the night *schedule* itself
  is included regardless.

## 8. Scheduling / auto-run

All on GitHub Actions, all output to Drive — same as Phase 1. New workflow
`company_repo.yml`: daily concall ingest+extract at a few fixed times; the
backfill job in the 00:00-05:00 IST window; `continue-on-error` on every step;
the existing health-check pattern extended to cover repo freshness.

## 9. Dashboard additions (light — NSE subset only)

- **Concall Feed** page: today's concall summaries, top results / top concall of
  the day (definition of "top" decided at build).
- **Guidance** page: best-guidance leaderboard, guidance-missed list, serial
  over/under-promiser ranking (matures after ~4 quarters of data).
- **Stock Detail:** a Fundamentals strip — latest quarter facts + a link to the
  full `company_page` / `deep_report` on Drive.
- A Streamlit input to request deep-research for one or more companies.

## 10. Open items / decisions

1. **`GEMINI_API_KEY`** — user to create (free, AI Studio) and add as a secret.
   Needed from the extraction step onward; not for document ingestion.
2. **Second LLM provider?** Gemini-only (free) is the default. A paid Anthropic
   key would add night-time throughput — decide later, not a blocker.
3. **"Top of the day"** definition — deferred to build.
4. **Google Workspace account** — not required (10-day cleanup keeps Drive lean).
5. **PENDING — insider buy/sell source.** Screener's insider buy/sell feeds
   (filters 76296 / 76297) are NOT used here: insider trades are one-line facts,
   not documents to LLM-summarise. An alternative structured source (NSE/BSE
   insider-trading disclosures, or a data feed) is to be identified. Until then
   the company page's promoter-activity section stays empty.
6. **Screener feed pagination** — the announcement feeds ignore `?page=` and the
   "SHOW MORE" button does nothing; each feed shows only its latest ~25–50.
   Same-day completeness depends on running the pipeline several times a day,
   not deep pagination. Peak-season risk noted.

## 11. Build order

- **Stage A** — `build_company_universe.py`, `ingest_company_docs.py` (5 feeds:
  concall, annual_report, presentation, rating, results — scrape + download),
  `scrape_results_table.py` (quarterly numbers), `cleanup_company_docs.py`.
  *No Gemini key needed.* **DONE & verified — 2026-05-23.**
- **Stage B** — `extract_concall.py` (Gemini) → the **daily MVP is live**.
- **Stage C** — company page generator + Concall Feed dashboard page.
- **Stage D** — `build_guidance_scorecard.py` + Guidance dashboard page.
- **Stage E** — `company_deep_report.py` (portfolio + on-request).
- **Stage F** — history backfill from 2025-05-20 (night window).
