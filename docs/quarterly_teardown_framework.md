# Quarterly Teardown Framework

How to read a quarter — the results filing *and* the investor deck — so the nuance
survives. This is both the human checklist and the machine contract: every block below
names its exact source column and formula, and `scripts/quarter_teardown.py` implements
it block for block.

**Design principle:** deterministic where the data exists, LLM only where prose is the
only source. Six of the eight blocks need no Gemini call at all.

---

## Source of truth

`fundamentals/statements/<SYM>.parquet` — long format, columns
`symbol · statement · line_item · period · value · fetched_at`.

Verified against live Drive data (2026-08-08, EMMVEE + PF sample):

| `statement` | grain | `line_item` values |
|---|---|---|
| `quarterly_pl` | **quarterly** | Sales · Expenses · Operating Profit · OPM % · Other Income · Interest · Depreciation · Profit before tax · Tax % · Net Profit · EPS in Rs |
| `annual_pl` | annual (+ a `TTM` period) | same, plus Dividend Payout % |
| `balance_sheet` | **annual only** | Equity Capital · Reserves · Borrowings · Other Liabilities · Total Liabilities · Fixed Assets · CWIP · Investments · Other Assets · Total Assets |
| `cash_flow` | **annual only** | Cash from Operating Activity · Cash from Investing Activity · Cash from Financing Activity · Net Cash Flow · Free Cash Flow · CFO/OP |
| `ratios` | annual | *(added by the ratios unlock)* Debtor Days · Inventory Days · Days Payable · Working Capital Days · ROCE % |

Three facts that constrain every formula below:

1. **The full P&L ladder is quarterly.** Other Income, Tax %, Depreciation and Interest
   are all present per quarter — so the entire "is this profit real?" analysis is
   arithmetic, not an LLM call.
2. **Balance sheet and cash flow are ANNUAL.** Any check built on them is a once-a-year
   read. It must carry an FY stamp in the render, never sit unlabelled beside a quarterly
   number.
3. **Receivables and Inventories are not balance-sheet rows.** Screener exposes only the
   *days* ratios, and only in the `#ratios` section. Everything receivables-related depends
   on the ratios unlock.

Filter note: `quarterly_pl` contains a junk `Raw PDF` line item — drop it.

Quarter labelling follows `quarterly_table.qtr_label()` — the quarter that just **ENDED**
(`Jun 2026` → `Q1 FY27`). Never `extract_concall._current_india_quarter()`, which returns
the quarter in progress and is a different concept.

---

## Block A — Verdict strip

**Answers:** what happened, in one line.

Quarter label · Revenue, PAT, EPS with YoY and QoQ · OPM in percentage points · one
verdict chip.

`BEAT · INLINE · MISS` is measured against the **company's own prior guidance**, from
`guidance_vs_actual.parquet` (`guided`, `actual`, `delta`, `verdict`, band ±2.0). Not
against consensus — there is no consensus feed in this repo, and implying one would be
fiction. When no prior guidance exists the chip reads `NO GUIDANCE`, not `INLINE`.

Reuses `quarterly_table.headline()`, which already returns period, quarter, revenue, pat,
eps, opm, npm each as `{value, yoy, qoq}`.

---

## Source of truth, part 2 — when Screener has not caught up

Screener is a **lagging mirror** of the exchange filing. On 2026-08-12 VMARCIND and
OBSCP filed Q1 FY27; two days later both still showed `Q4 FY26` as their newest
`quarterly_pl` column, so both mails — which define "reported" as that column matching
the season — could not see them at all.

`results_gemini.parquet` already holds revenue / EBITDA / PAT / EPS / margins read from
the filing PDF itself. `scripts/filing_results.py` is the validated read side of it, used
by `pf_results_digest.py` and `quarter_teardown.py` alike.

A filing-sourced company is rendered as **its own single-quarter card**, never as a column
inside a Screener table: the filing may be consolidated where Screener is standalone, and
its "Revenue" need not tie to Screener's "Sales". Blocks B and H1 and the six-quarter
ladder are **withheld** for such a company — every one of them needs the quarterly ladder
that does not exist yet. Withheld, not estimated.

The stored table is dirtier than its schema suggests, so every field is validated on read
(measured 2026-08-14, 123 rows):

| Symptom | Cause | Handling |
|---|---|---|
| `pat_cr` numeric in only 25/123 | `METRIC_ALIASES` is first-match-wins with `("pat","pat")` ahead of `("margin","margin")`, so `PAT Margin %` identified as `pat` and wrote `"5.13%"` into `pat_cr`. `ebitda_cr` corrupted identically. | writer fixed in `extract_results.py`; a `%` in an absolute field is rejected on read |
| `revenue_cr` numeric in 74/123 | a repeated header row parsed as data → the literal `'Revenue (Cr)'` | non-numeric drops the row |
| 23 rows with a blank `quarter` | Gemini's table header carried no `Q# FY##` | dropped — unassignable to a quarter |

Note `_extractor_base.try_float` **strips** `%`, so `try_float("5.13%")` returns `5.13`.
A percent must be rejected *before* coercion or the mis-parse passes as a crore figure.

Rows already stored are not rewritten — re-extraction is the honest repair.

---

## Block B — Quality of the number  *(arithmetic, no LLM)*

**Answers:** is this profit real, or engineered?

Rendered as a **three-panel card inline in the mail body** — adjusted-PAT bridge ·
balance-sheet movement · cash-flow quality — so the question is answerable in about ten
seconds without opening an attachment. Panel 1 is quarterly; panels 2 and 3 are annual and
carry an FY stamp. They are never combined into one derived number: `Fact.derive` refuses
mixed grains by design.

Every holding gets the card, including those whose full teardown does not fit the mail's
size budget. Reducing most of the night's reporters to a single summary line is what made
this block invisible in the first place.

All from `quarterly_pl`. `q` = current quarter, `q-1` = QoQ, `q-4` = YoY.

| Check | Formula | Flag when |
|---|---|---|
| **Margin walk** | `OPM% [q] − OPM% [q-4]`, and vs `[q-1]` | report always, in pp |
| — gross vs opex split | `(Sales−Expenses)/Sales` move vs `Expenses/Sales` move | narrative only |
| **Other-income share of PBT** | `Other Income / Profit before tax × 100` | rises > 5 pp YoY |
| **Tax rate** | `Tax %`, vs mean of prior 4 quarters | deviates > 5 pp |
| **Interest step** | `Interest [q] / Interest [q-1] − 1` | > 25% jump |
| **Depreciation step** | `Depreciation [q] / Depreciation [q-1] − 1` | > 25% jump |
| **Adjusted PAT** | reported PAT − post-tax effect of the YoY *rise* in Other Income + amount over-taxed vs the prior-4Q rate | show beside reported PAT, with the bridge |
| **Growth attribution** | revenue growth split into volume vs realisation | only when the deck gives volume |

**Why these:** a jump in Interest or Depreciation usually means capitalisation ended — the
capex story just landed in the P&L. A PAT beat carried by Other Income or a low tax rate is
not an operating beat. **Adjusted PAT is the single most useful number on the page**: it
is what the business earned before the two easiest levers were pulled.

It is built as an explicit bridge FROM reported PAT, never recomputed from PBT, so every
line can be shown and the total ties out exactly:

```
  Reported PAT                                          263.0
  Less: rise in other income, post-tax                  −10.5   (₹14 Cr pre-tax @ 25.0%)
  Add : over-taxed vs the prior-4Q average              + 8.4   (25.0% vs 22.50% on ₹338 Cr)
  = Adjusted PAT                                        260.9   (−0.8% vs reported)
```

Two deliberate choices. Only the **increase** in other income is removed — a business is
entitled to its usual treasury income, and stripping the whole line would understate every
company that holds cash. And the tax line **adds back** what was over-taxed rather than
silently re-taxing, because a quarter taxed above its own recent average was penalised,
not flattered.

Recomputing as `PBT × (1 − normal rate)` instead leaves an unexplained residual, since
Screener rounds PAT and PBT independently to whole crore. Anchoring on reported avoids it.

---

## Block C — Operating engine  *(deck, LLM)*

**Answers:** what physically drove the quarter.

Capacity · utilisation % · order book / pipeline · volume · realisation · segment mix ·
geography mix. Each trended over 4–6 quarters, so a deck number is read against its own
history rather than in isolation.

→ `deck_metrics.parquet`, `category ∈ capacity | utilisation | orderbook | volume |
realisation | segment_mix | geo_mix`.

---

## Block D — Said vs delivered  *(already computed — rendering only)*

**Answers:** was the last promise kept?

`guidance_vs_actual.parquet` — `period · metric · guided · actual · delta · verdict ·
source · cred_score`. `source` distinguishes `pead_concall` / `pead_presentation` /
`pead_annual_report` / `gf_track`.

Plus the credibility pattern from `mgmt_credibility.parquet`: `cred_score` (EXCEEDED=5,
DELIVERED=4, PARTIAL=3, MISSED=1) and `pattern ∈ Calibrated | Optimistic Bias |
Conservative Bias | Erratic | Insufficient Data`.

A management team that is **Calibrated** is worth more than one that beats erratically.
Consistency of guidance is itself a signal about the business's visibility.

---

## Block E — Deck-vs-deck diff  *(deck, LLM)* ← highest signal

**Answers:** what changed in the telling.

This quarter's deck against the same company's previous deck:

| `change_type` | Meaning |
|---|---|
| `kpi_dropped` | a metric shown for several quarters is absent now |
| `target_moved` | a deadline or number shifted out |
| `definition_changed` | same label, different basis (e.g. order book now includes LOIs) |
| `new_emphasis` | promoted to the front of the deck |
| `de_emphasised` | demoted to the annexure |

**Why it matters most:** companies rarely announce bad news. They stop mentioning it. A KPI
that quietly disappears after four quarters of prominence is the cheapest early warning
available, and it is invisible to anyone reading only the current deck.

→ `deck_diff.parquet`. Implemented by passing the prior quarter's stored rows in as a
`[PRIOR_DECK]` context block, mirroring `concall_prompt.txt`'s `[HISTORICAL_CONTEXT]`.

---

## Block F — Framing flags  *(deck, LLM)*

**Answers:** how is the story being staged?

Truncated or rebased chart axes · switching absolute ↔ % exactly when growth slows ·
cherry-picked peer sets · a non-GAAP metric introduced this quarter · caveats moved into
footnotes.

Every flag carries a slide reference and a verbatim quote, or it is not emitted. Severity
`high | medium | low`. → `deck_flags.parquet`.

---

## Block G — Open questions  *(LLM)*

3–5 questions the deck leaves unanswered, each tied to a specific flag above. These are the
questions to carry into the next concall.

---

## Block H — Early warnings & confirmations

The forensic layer. **H2 needs no new extraction at all.**

### H1 — Divergence engine  *(arithmetic)*

The framing that matters is not "are receivables high?" but **"is receivables growing
faster than the earnings it is supposed to produce?"** Each check is a paired growth-rate
gap in percentage points, rendered as one diverging bar — red adverse, green favourable.

| Check | Source | Adverse when |
|---|---|---|
| **CFO vs PAT** ← anchor | `financials_derived.cfo_pat_ratio` (annual) | `< 1` two years running |
| Receivable days vs revenue growth | `ratios` → `receivable_days` | days rising while revenue flat or down |
| Inventory days vs revenue growth | `ratios` → `inventory_days` | days rising → demand softening |
| Working-capital days | `ratios` → `wc_days` | `> 1.5 ×` mean of prior 3 years |
| Cash conversion cycle | `receivable + inventory − payable` | rising year on year |
| Interest coverage | `financials_derived.interest_coverage` | `< 1.5`, or falling |
| Leverage | `financials_derived.net_debt_ebitda` | `> 4` |
| Other income vs operating profit | block B | OI growing faster than OP |
| CWIP not converting | `balance_sheet`: CWIP / Total Assets | rising 3 FY running while Fixed Assets flat |
| Borrowings rising | `balance_sheet.Borrowings` | rising while presented as cash-rich |
| ROCE | `financials_derived.roce_pct` | three consecutive declines |

**CFO vs PAT is the anchor.** Historically the single best forensic predictor; everything
else here is corroboration. Profit that never becomes cash is either an accounting choice
or a collection problem, and both eventually show up in the price.

> **Note on `net_debt_ebitda`:** despite the name, `build_derived_metrics` computes it as
> `Borrowings / Operating Profit` — a **gross**-debt proxy. Label it honestly in the render.

### H2 — Red-flag register  *(pure rendering, zero new extraction)*

One dated, severity-sorted register per company, merged from seven existing tables. Live
PF row counts, verified 2026-08-08:

| Source | Contributes | PF rows / names |
|---|---|---|
| `ar_red_flags.parquet` | 16 auditor & governance flags + `severity` + `page_ref` | 1,127 / 52 |
| `rating_concerns.parquet` | agency-authored weaknesses + `severity` | 578 / 43 |
| `rating_sensitivity.parquet` | `direction='down'` = explicit downgrade triggers | 585 / 42 |
| `ratings.parquet` | `rating_action='Downgrade'` as a dated event | 159 / 43 |
| `gf4_quality_flags.parquet` | `Contradictory` / `Promotional Commentary` | 212 / 19 |
| announcement ledger | `litigation` / `management_change` / `regulatory` × `bear` × `high` | — |
| `fraud_tracker.parquet` | band RED/ALERT/WATCH + 7-day trend — **link, don't recompute** | 37 / 37 |

The 16 `ar_red_flags.flag_type` values: `auditor_qualification · emphasis_of_matter ·
caro_adverse · accounting_policy_change · accounting_estimate_change ·
revenue_recognition_change · notes_to_accounts_deviation · cfo_pat_divergence ·
working_capital_stretch · cwip_buildup · related_party_transaction · promoter_pledge ·
kmp_churn · contingent_liability · tax_variance · other`.

**`rating_sensitivity` is the most underused table in the repo.** It is the rating agency
stating, in its own words, exactly what would cause a downgrade. That is a pre-written
early-warning tripwire per company, and nothing has ever displayed it.

**Positive column**, same treatment: CFO/PAT above 1 and rising · debtor and inventory days
falling while revenue grows · utilisation climbing toward capacity · CWIP converting into
Fixed Assets · debt falling and coverage rising · rating upgrade or outlook to positive ·
gross margin expanding **with** volume · `rating_sensitivity.direction='up'` triggers being
met · guidance beaten consistently (`Calibrated`).

### H3 — AR forensic score  *(parse what already exists)*

`annual_report_prompt.txt` §6 generates a 5-dimension weighted 0–10 matrix every run —
Cash Flow Quality 25% · Accounting Integrity 25% · Balance Sheet Risk 20% · Governance &
Promoter 20% · Regulatory & Auditor Transparency 10% — with a
`Clean (0–2) / Monitor (3–5) / Elevated (6–7) / Avoid (8–10)` label. It is written into
`company_page.md` as prose and **never parsed**. Tabulating it is the cheapest high-value
addition in this framework.

---

## Block S — The quarter as the company told it  *(rendering only)*

**Answers:** what did the company itself publish about this quarter?

Blocks A–H are this repo's reading of the quarter. Block S is the company's own account,
assembled by `scripts/season_summary.py` from tables that already exist and had never
been read together:

| Section | Source | PF coverage for Q1 FY27 (2026-08-14) |
|---|---|---|
| Results release | `results_gemini` via `filing_results` | 25 companies |
| Investor presentation | `ppt_highlights`, `ppt_guidance` | 23 / 12 companies |
| Concall | `gf1_guidance_statements`, `gf4_quality_flags` | 28 companies |
| Other filings | `announcement_ledger` (`summary`, `event_type`, `materiality`, `direction`) | 43 companies |
| Deck teardown | `deck_metrics/diff/flags` | **0 — the files do not exist** |

Announcements are dated rather than quarter-tagged, so they are windowed from the quarter
end to at most 110 days after it. Everything else is matched on `quarter`.

`--teardown` was enabled in `phase2.yml` on 13 Aug 2026 and has not yet written a row.
`deck_diff` additionally needs **two consecutive quarters** of deck coverage before it can
ever be non-empty, because `deck_teardown.parse_teardown` drops `diff_without_prior_deck`.
Until rows land, the section says the pass has not run — it never renders as an all-clear.

---

## Honesty constraints the render must state, not paper over

These are correctness rules, not style preferences.

- **Never mix grains.** Balance-sheet and cash-flow checks are annual; AR flags are
  FY-stamped. Both must show their period beside the value. Placing an annual ratio
  unlabelled next to a quarterly number is the same error `pf_results_digest.py` was built
  to prevent.
- **AR flags go stale mid-year.** An FY25 flag in Q3 FY27 is nine months old. Show the FY
  and let the reader discount it.
- **Going concern is not captured anywhere** — zero occurrences repo-wide across all
  prompts. The page must not imply this was checked.
- **`rating_action` recognises only Upgrade / Downgrade / Reaffirmed.** No Withdrawn,
  Suspended, or Rating Watch — all strong real-world warnings, all invisible here. A known
  blind spot.
- **Missing ≠ clean.** A company with no `ar_red_flags` rows may simply have no annual
  report processed. Render `not covered`, never an implied all-clear.
- **Join on `isin`, not `symbol`.** `ar_red_flags.symbol` sometimes carries a BSE code
  (e.g. `539730`), and `fy_year` mixes `FY25` with `FY25-26` — normalise both.

---

## What a good and a bad quarter look like

**Quietly compounding:** revenue growth carried by volume, not price · OPM up with gross
margin, not one-off opex relief · Other Income share of PBT flat · CFO/PAT above 1 and
rising · debtor and inventory days falling · CWIP converting into Fixed Assets · deck keeps
showing the same KPIs and hits its prior targets · agency lists upgrade triggers being met.

**Heading south:** PAT beat carried by Other Income or a low tax rate · OPM up only because
a cost line was deferred · CFO below PAT for a second year · receivable days climbing while
revenue flattens · CWIP rising for three years without commissioning · a KPI shown for four
quarters silently dropped from the deck · a target quietly moved out · agency downgrade
trigger language sharpening between reports.

The single strongest pairing on this page: **Adjusted PAT (block B) beside CFO/PAT (H1) beside
the deck diff (E).** Together they answer whether the profit is operating, whether it
became cash, and whether management is still willing to be measured on it.
