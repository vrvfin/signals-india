# Narrative report — defect log and fix provenance

Every defect found while building the narrative pipeline, how it was **detected**, the
**evidence**, and the **fix**. Nothing here was found by reading code and guessing; each
row names the observation that exposed it.

The ground truth throughout is TheWrap Weekly Members Meet #161 (26 July 2026), slides
8–33, a 26-slide analysis of Landmark Cars. Its published figures are an answer key: if
our arithmetic disagrees with a slide, one of us is wrong and it is worth finding out
which.

---

## 1 · Exceptional items missing from the fixed block

| | |
|---|---|
| **Detected by** | Comparing `narrative_compute` output against deck slides 28 and 30 |
| **Evidence** | Our implied P/E cell read 51.8x; the deck's read 55.6x. Our sensitivity row at 6.28% margin gave PAT ₹59.6 Cr against the deck's ₹57 Cr |
| **Root cause** | Screener's `annual_pl` has no exceptional-items line. Working backwards from the deck: 149.2 (D&A) + 79.8 (finance) + **3.6 (exceptional)** = 232.6, not the 229 we computed. Tax was also 24.4% actual vs Screener's rounded 24 |
| **Why it mattered** | A 7% error in every sensitivity cell, with nothing to signal it. Loose test tolerances (`tol=0.12`) were hiding it — the selftest passed |
| **Fix** | `narrative_compute.operating_leverage()` takes `exceptionals`; adds `residual_vs_disclosed_gap` and a `reconciles` flag tying the fixed block to the disclosed EBITDA−PBT gap. `effective_tax_pct()` derives the rate from the PBT/PAT pair in preference to Screener's rounded line |
| **Verified** | Selftest `[6]`/`[7]` now reproduce the deck exactly — EPS 13.39 vs 13.39, P/E 55.8x vs 55.6x. Test `[7b]` asserts that omitting exceptionals produces the −6.7% drift and that `reconciles` goes False |
| **Lesson applied** | Tolerances were tightened from 0.06/0.12 to 0.01. A tolerance wide enough to hide a real defect is not a test |

## 2 · CAGR measured over a window that inverted the conclusion

| | |
|---|---|
| **Detected by** | Reading the rendered markdown and noticing §5 said the opposite of deck slide 10 |
| **Evidence** | Ours: revenue CAGR 14.91%, EBITDA CAGR 22.56% → "profit outgrew scale". Deck: 13.3% / 10.9% → "profit lagged scale badly" |
| **Root cause** | We measured over all available history (FY18→FY26). FY18 EBITDA was ₹55 Cr against a series median of ₹187 Cr, and FY19–FY20 were loss years. The depressed base flattered the whole series |
| **Why it mattered** | Arithmetically correct, analytically backwards — the worst kind of error, because nothing looks broken |
| **Fix** | `cagr_pct()` takes `window` (default 4 steps = a 5-point FY span) and returns `base_warning` when the start year sits below half the series median, plus a note when the series contains negative years. The full-history figure is still reported, explicitly caveated |
| **Verified** | §5 now reads 13.24% / 10.62% / −2.63pp, matching the deck's reported-basis 13.3%. The 22.56% appears labelled "full history" with the caveat attached |

## 3 · Stability claimed across loss-making years

| | |
|---|---|
| **Detected by** | Reading §8 — its heading contradicted its own number |
| **Evidence** | Section titled "the quiet compounder — the steadiest series" reporting a 400.73bps margin band spanning COVID and two loss years |
| **Fix** | `sec8_stable_series` restricts the headline band to the last 5 FYs and to profitable years only, labels the span (`FY22-FY26`), adds `years_in_band`, and reports the full-history range separately. If fewer than 3 profitable years exist it emits a gap saying no stability claim is supportable |
| **Verified** | §8 now reads 144.27bps over FY22–FY26 with the 400.73bps full range shown separately |

## 4 · Renderer duplicated 93 facts above the table holding them

| | |
|---|---|
| **Detected by** | Reading the rendered markdown |
| **Evidence** | §18 printed a 93-row "Metric \| Value \| Basis" list ("Revenue FY18, Revenue FY19, …") directly above the clean 9×10 `tbl.financials`. Two rows both read "EBITDA FY18" at different values (55 and 51) |
| **Fix** | `_facts_in(..., for_display=True)` withholds per-period series facts when a table in the same section already renders them; provenance is unaffected because `_sources_for` still reads the full set. The two EBITDA bases got distinct labels ("incl. other income" / "operating profit only") |
| **Verified** | §18 now shows 3 CAGR facts plus the matrix; no duplicate labels |

## 5 · Chart drew the margin line against an invisible axis

| | |
|---|---|
| **Detected by** | Suspicion, then reading the library source rather than assuming |
| **Evidence** | `curl unpkg.com/lightweight-charts@4.1.3/...` → the default price-scale object is `pe={autoScale:!0,…,visible:!1,…}`. The left scale is **invisible by default**, so a series on `priceScaleId:'left'` plots with no readable axis. Separately, revenue (~4,896) and EBITDA (~280) as two overlapping histograms rendered EBITDA as a sliver behind revenue |
| **Fix** | `leftPriceScale:{visible:true}` set explicitly in the chart options, with a comment recording why; EBITDA changed from a second histogram to a line so it stays legible at any ratio |
| **Not verified** | The Browser pane reports `innerWidth: 0` and screenshots fail with "the pane is not displayed". Confirmed via injected JS that the library loads, the chart initialises, 10 canvases are created and all 4 wide tables sit in working `overflow-x` containers — but **the visual has not been seen** |

## 6 · Gemini truncation misdiagnosed as a prompt problem

| | |
|---|---|
| **Detected by** | Both live sections returning "unparseable JSON" on retry |
| **Evidence** | Captured the raw response: 1,721 chars (~430 tokens) ending mid-sentence at "Management aims to", against `max_output_tokens=3000`. Re-running the identical prompt at 12,000 returned 5,057 chars and parsed with all 9 keys |
| **Root cause** | Gemini 2.5 models are **thinking** models: `max_output_tokens` covers reasoning tokens as well as the visible answer. The budget was spent thinking |
| **Fix** | `MAX_OUTPUT_TOKENS = 12000` in `narrative_generate.py` (14000 in the extractors, which emit longer JSON), each with a comment recording the measurement so nobody lowers it again |

## 7 · Retry kept the worse attempt

| | |
|---|---|
| **Detected by** | Inspecting the generation output after fixing #6 |
| **Evidence** | Section 5 attempt 1 produced good prose with 5 gate findings; attempt 2 returned unparseable output; the saved section was **empty**. `out = _parse(raw)` overwrote the good attempt with `None` |
| **Fix** | `generate_section` scores each attempt (`-1` for unparseable, else `100 − hard failures`) and keeps the best, recording `attempts` and which one was used |

## 8 · Gate 1 numeric regex split four-digit years

| | |
|---|---|
| **Detected by** | Live generation produced findings for a number `'202'` that appears nowhere |
| **Evidence** | `[m.group('num') for m in _NUM_RE.finditer('concall_2026-06-03')]` → `['202','6','06','03']` |
| **Root cause** | The comma-grouped branch `\d{1,3}(?:,\d{2,3})*` used `*`, so it matched the first three digits of any long run and won the alternation |
| **Fix** | Changed `*` to `+` so that branch requires at least one comma group; full digit runs fall to the second branch |
| **Verified** | `finditer('2026')` → `['2026']`; the selftest's genuine catches still fire |

## 9 · Gate 1 demanded facts for inline citations

| | |
|---|---|
| **Detected by** | Same live run — three findings pointing at `(concall_2026-06-03)` |
| **Root cause** | `_DATE_RE` is `\b(?:19|20)\d{2}\b`, and `\b` does not fire between `_` and a digit, so a doc_id's year survived masking |
| **Fix** | Added `_DOCID_RE`, `_ISODATE_RE` and `_FACTID_RE`, applied **before** `_DATE_RE` so the whole citation is blanked rather than leaving digit fragments |
| **Verified** | `_mask_labels('per lev.absorption_pct (concall_2026-06-03)')` blanks both tokens |

## 10 · Gate 1 rejected correct English about a negative fact

| | |
|---|---|
| **Detected by** | Same live run: "lagged revenue growth by 2.63 percentage points" was flagged against a stored `scale.gap_pp = -2.63` |
| **Fix** | Numeric comparison now also matches on absolute value. Direction carried in words is correct writing; a genuinely inverted claim is a semantic error for the auditor, not something a string check can adjudicate |
| **Also** | `_label_numbers()` licenses numbers appearing in fact **labels**, so "for every 1% change in EBITDA" is legal against the fact "PBT move per 1% EBITDA move" |
| **Verified** | All three phrasings now pass; `Rs 9,999 Cr` is still caught |

## 11 · Gate 2 case-sensitivity would have generated noise

| | |
|---|---|
| **Detected by** | Writing the selftest and disliking the result |
| **Reasoning** | Lowercasing a sentence-initial word when embedding a quote mid-sentence is correct practice. Failing it trains the reader to ignore the gate — the worst outcome for a safety check. The risk being guarded against is fabrication, and a fabricated quote will not match case-insensitively either |
| **Fix** | Case-folded match passes with severity `warn`, so the drift is visible but does not fail the run |

## 12 · Vahan is not machine-reachable

| | |
|---|---|
| **Detected by** | Probing endpoints before writing a fetcher |
| **Evidence** | `vahan.parivahan.gov.in` → `ConnectionResetError(10054)`; `analytics.parivahan.gov.in` → HTTP 403; `cea.nic.in` → 200 but unstructured HTML; `api.data.gov.in` → 400 (needs a key) |
| **Corroboration** | Deck slide 25's own footer: the figures are "the author's own extraction and aggregation from the raw files" — manual work, not an API |
| **Fix** | `alt_registry.py` labels every dataset `api` / `keyed` / `manual` and makes manual CSV intake first-class (`ALT_INTAKE_DIR/<dataset>/*.csv`). A manual dataset with no file yields DATA_MISSING with the reason and the how-to, never an estimate |

## 13 · Audit 429'd on every section, and reported it as a clean result

| | |
|---|---|
| **Detected by** | The first full live run: "audit 0/8 verified · 0 unsupported · 0 contradicted" |
| **Evidence** | `audit_LANDMARK.json` → all 8 sections `adjudicator call failed: all cerebras attempts failed: HTTP 429`. Calibration had passed because its fixtures were ~700 chars; the real run sent 3 documents at 60k each ≈ 45k tokens |
| **Root cause** | `MAX_SOURCE_CHARS` was a PER-DOCUMENT cap with no total budget. `provider_router.py` had already measured the real limit — `_ALT_MAX_CHARS = 80_000`, "bigger requests 429" — and I did not apply the constraint my own repo had established |
| **Second, worse defect** | The summary line read "0 verified · 0 unsupported · 0 contradicted", which is indistinguishable from a clean audit. The only clue was `not_in_source: 8`. I had guarded the degraded-Gemini-fallback case but not the every-call-failed case |
| **Fix** | `MAX_TOTAL_SOURCE_CHARS = 70_000` across all documents, packed most-relevant-first (`_fit_sources`) with any dropped document NAMED in the prompt; on a 429 the budget halves and retries once. New verdict `AUDIT_FAILED`, plus `ran` / `sections_audited` / `sections_failed` on the summary. Renderer prints a red **AUDIT DID NOT RUN** banner in md and HTML; the orchestrator logs it |
| **Verified** | Re-run: 8/8 sections audited, 0 failures |

## 14 · The auditor was denied the accounts it was asked to check

| | |
|---|---|
| **Detected by** | Reading the re-run's verdicts after fixing #13 |
| **Evidence** | Correct claims were marked UNSUPPORTED — "Revenue reached Rs 4,896 crore in FY26", "Revenue grew at a 13% CAGR from FY22 to FY26". Both are straight from the fact pack |
| **Root cause** | My own design. To protect independence I gave the auditor only the source documents — but the financial figures come from Screener statements, not from any filing in that bundle, so there was nothing to verify them against. Independence means not sharing the generator's FRAMING, not being refused the underlying DATA. A fact-checker denied the accounts cannot check the accounts |
| **Fix** | The section-scoped fact table is now supplied as a labelled `DATA TABLE` evidence block (`fact_table_text()`), and the prompt states that numeric claims are checked against it while claims about what management said are checked against the documents. The auditor still never sees the generator's prompt or reasoning |
| **Verified** | 84/91 verified, 3 partial, 4 unsupported, 0 contradicted — against 0/91 before |
| **Follow-on** | Of the 4 remaining UNSUPPORTED, 3 were not claims about the company at all (a "this is not a forecast" disclaimer, two methodology notes). The prompt now tells the auditor to skip disclaimers and meta-commentary, since noise buries real findings |

## 15 · Unicode crash after all the LLM spend

| | |
|---|---|
| **Detected by** | `UnicodeEncodeError: 'charmap' codec can't encode character '₹'` — a traceback at the very end of an audit run |
| **Root cause** | Windows consoles default to cp1252, which has no rupee sign. Printing an audited claim containing ₹ killed the process *after* every API call had been paid for |
| **Fix** | `_safe()` re-encodes with `errors="replace"` for console output only; the JSON on disk keeps the real characters |

## 16 · Extracted data was written to Drive and never read back

| | |
|---|---|
| **Detected by** | The user asking "what data sources is it using" — I checked instead of answering from memory |
| **Evidence** | `company_structure.parquet` (233 rows) and `mgmt_quotes.parquet` (12 rows) existed on Drive, but `grep company_structure narrative_factpack.py` returned nothing. Sections 1/3/4/6/9/12/20/23 were still hard-coded in `UNCOVERED` |
| **Root cause** | N7 and N8 built the extractors and wrote the tables; nothing consumed them. The report kept rendering DATA_MISSING for sections whose data had already been extracted and paid for |
| **Fix** | Section builders `sec1_history`, `sec3_entity_map`, `sec4_management`, `sec6_segment_economics`, `sec7_ratings`, `sec9_portfolio`, `sec12_unit_deepdives`, `sec20_quote_spine`, `sec23_structural_risks`. `UNCOVERED` now holds only 10 (narrative-derived) and 16/17 (no feed exists) |
| **Verified** | LANDMARK 8 → 17 populated sections, 141 → 154 facts, 4 → 12 tables |
| **Lesson** | "Extractor built" is not "feature delivered". The pipeline had a silent seam between producing data and consuming it, and only an end-to-end check across that seam would have caught it |

## 17 · Two API contracts assumed rather than read

| | |
|---|---|
| **Detected by** | Running it — both crashed on first contact |
| **Bug A** | `upload_bytes(drive, folder, name, data, fid)` — the real signature is `(drive, folder_id, filename, data, mimetype, existing_id)`. The file id landed in `mimetype`, so `MediaIoBaseUpload` got `None` and raised `AttributeError: 'NoneType' object has no attribute 'split'` — **after** the extraction LLM calls had been paid for |
| **Bug B** | `bse_announcements()` / `nse_announcements()` return a FORMATTED STRING of `"- date \| subject [cat]"` lines, not a list of dicts. `'str' object has no attribute 'get'` |
| **Fix** | A: pass `"application/octet-stream"` with `existing_id=fid`, matching `build_derived_from_statements.py`. B: parse the string rather than re-implement the call, so the BSE header dance and NSE cookie bootstrap stay in one place |
| **Follow-on** | BSE emits ISO dates, NSE emits `DD-MMM-YYYY`. Truncating to 10 chars chopped NSE's year (`30-May-202`), and sorting two formats together is meaningless — both are normalised to ISO now, and `dayfirst` is applied only to the non-ISO form so pandas stops warning on every row |
| **Verified** | 40 filings retrieved, correctly dated and sorted |

---

## Operational findings (the pipeline's data, not this code)

- **`company_facts.parquet` is globally stale.** All 2,959 rows carry
  `updated_at = 2026-07-22` — 7.7 days at time of writing, against a 5-day policy.
  Consequence for a fast grower: MONOLITH's stored TTM revenue is ₹137 Cr while
  `statements` implies ₹155 Cr, a 13.4% divergence, because `company_facts` stops at
  Mar 2026 while `statements` has Jun 2026 at ₹47 Cr. Preflight returns **DO NOT
  PUBLISH**. Slow compounders (TCS, Reliance) only WARN on the same staleness, so
  severity scales with real distortion.
- **`FREE_POOL_2` returns HTTP 401** ("The bound service account..."), dropping 3 buckets
  every run. `FREE_POOL_1` was PerDay-exhausted during testing.
- **Landmark has only 2 concalls and 3 ARs in the queue**, and zero rows in
  `pead_flags` / `mgmt_credibility`. Section 20 wants ≥4 calls for a
  claim-across-calls spine, so this company is a poor end-to-end test of it.

## Things deliberately NOT done

- **The live Phase-2 path is untouched.** `extract_concall.py`,
  `ar_structured_prompt.txt`, `concall_prompt.txt` and `_extractor_base.py` are
  unmodified. N7/N8 re-fetch the same documents from `pdf_url` and write their own
  tables, per CLAUDE.md rule 3 and memory `phase2-concall-untouchable`.
- **Part B is attached, not invoked.** `company_deep_report.py` writes to Drive and owns
  a queue lifecycle; calling it from the orchestrator would duplicate side effects. Pass
  its markdown with `--forensic-md`.
- **Nothing is committed.** All files are new and uncommitted.
