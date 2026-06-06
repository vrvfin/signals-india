# Quarterly Concall Gap Filler — Implementation Guide

## Overview

Two-part system to automatically fill historical Q1/Q2/Q3/Q4 concall gaps caused by Screener's 25-item/day feed cap:

1. **find_quarterly_gaps.py** — Discover companies with Q4 concalls not yet in the queue
2. **merge_quarterly_gaps_to_queue.py** — Intelligently merge gaps into processing_queue.parquet

Both scripts:
- Auto-detect current quarter (Q1/Q2/Q3/Q4) from calendar date
- Query full 3-month window per quarter (not just peak 60-day season)
- Filter audio (3-layer detection: URL extension → Content-Type → PDF magic bytes)
- Dedup against entire queue history (prevents re-adding done/error items)
- Respect Gemini quota (keep pending ≤ 75 for safe daily processing)

---

## Script 1: find_quarterly_gaps.py

**Purpose:** Discover companies with published Q4 concalls not yet in the queue.

**Inputs:**
- Screener cookie (from `SCREENER_SESSION_COOKIE` env var)
- Current quarter (auto-detected from date)

**Outputs:**
- `concall_gaps_{QUARTER_LABEL}.parquet` (e.g., `concall_gaps_Q4FY.parquet`)
  - Columns: isin, symbol, company_name, announcement_date, concall_url, content_type, source, discovered_at
  - Uploaded to Drive: `company_repo/_index/concall_gaps_Q4FY.parquet`

**Usage:**
```bash
# Auto-detect quarter, upload to Drive
python scripts/find_quarterly_gaps.py

# Dry-run (don't upload to Drive, save locally)
python scripts/find_quarterly_gaps.py --dry-run

# Manual quarter specification
python scripts/find_quarterly_gaps.py --quarter Q4FY --start 2026-04-01 --end 2026-06-30
```

**Logic:**
1. Auto-detect current quarter (Q1/Q2/Q3/Q4) from `date.today().month`
2. Query Screener's concall filter for the full 3-month window
3. Handle pagination (Screener shows ~25 items/page max)
4. Detect MP3 vs PDF (skip audio files)
5. Cross-check against `processing_queue.parquet` (skip already-queued)
6. Save gaps registry to Drive

**Key Features:**
- ✅ 3-layer MP3 detection (extension → HTTP Content-Type → magic bytes)
- ✅ Dedup against entire queue (all statuses: pending, done, error)
- ✅ Handles pagination to bypass 25-item cap
- ✅ Fallback to local file if Drive upload fails

---

## Script 2: merge_quarterly_gaps_to_queue.py

**Purpose:** Intelligently merge gaps into processing_queue.parquet, respecting quota and queue depth.

**Inputs:**
- `concall_gaps_{QUARTER_LABEL}.parquet` (from find_quarterly_gaps.py)
- `processing_queue.parquet` (entire history, all statuses)

**Outputs:**
- Updated `processing_queue.parquet` on Drive (appended with new rows)

**Usage:**
```bash
# Smart merge (auto-detects quarter, respects queue depth)
python scripts/merge_quarterly_gaps_to_queue.py

# Dry-run (print what would be added, don't update queue)
python scripts/merge_quarterly_gaps_to_queue.py --dry-run

# Force (skip queue depth check, add regardless)
python scripts/merge_quarterly_gaps_to_queue.py --force
```

**Logic:**
1. Load gaps registry for current quarter
2. Load entire queue (all rows, all statuses)
3. Build dedup key: (isin, announcement_date)
4. Find new gaps not in queue
5. Check queue depth:
   - If pending >= 150 → skip (queue backlogged from Gemini exhaustion)
   - If pending < 20 → push ~50–75 items (aim for total ~75)
   - If 20 <= pending < 150 → skip (let queue drain)
6. Append new rows to queue
7. Save updated queue to Drive

**Key Features:**
- ✅ **Safe dedup:** Loads ENTIRE queue (all statuses) — prevents re-adding done items on future runs
- ✅ **Quota-aware:** Respects 2-day PDF storage (75 pending × ~2 min/concall = safe 24h processing)
- ✅ **Intelligent merging:** Adjusts batch size based on current queue depth
- ✅ **Weekly-safe:** Can run every weekend; won't re-add done items

---

## Integration into phase2.yml

Add the following job to `.github/workflows/phase2.yml`:

```yaml
jobs:
  # ... existing Phase 2 jobs ...

  quarterly_gap_filler:
    runs-on: ubuntu-latest
    name: "Quarterly concall gap filler (weekends only)"
    # Only run on Saturday (6) or Sunday (0)
    if: contains('0,6', github.event.schedule.day_of_week)
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: "[DEBUG] Check current quarter"
        run: python -c "
from datetime import date
quarter_map = {
    (4, 5, 6): 'Q4FY',
    (7, 8, 9): 'Q1FY',
    (10, 11, 12): 'Q2FY',
    (1, 2, 3): 'Q3FY',
}
m = date.today().month
q = next((label for months, label in quarter_map.items() if m in months), 'NONE')
print(f'Current quarter: {q}')
"
        env:
          TZ: 'Asia/Kolkata'

      - name: "Find quarterly gaps"
        id: find_gaps
        run: python scripts/find_quarterly_gaps.py
        env:
          SCREENER_SESSION_COOKIE: ${{ secrets.SCREENER_SESSION_COOKIE }}
          GDRIVE_FOLDER_ID: ${{ secrets.GDRIVE_FOLDER_ID }}
          GDRIVE_OAUTH_TOKEN_PATH: ${{ secrets.GDRIVE_OAUTH_TOKEN_PATH }}
        continue-on-error: true

      - name: "[DEBUG] Log gaps discovery"
        if: always()
        run: echo "Gaps discovery completed (check logs above for details)"

      - name: "Merge gaps into queue"
        id: merge_gaps
        run: python scripts/merge_quarterly_gaps_to_queue.py
        env:
          GDRIVE_FOLDER_ID: ${{ secrets.GDRIVE_FOLDER_ID }}
          GDRIVE_OAUTH_TOKEN_PATH: ${{ secrets.GDRIVE_OAUTH_TOKEN_PATH }}
        continue-on-error: true

      - name: "[DEBUG] Log merge result"
        if: always()
        run: echo "Merge completed (check logs above for queue depth and items pushed)"
```

**Key Settings:**
- `if: contains('0,6', github.event.schedule.day_of_week)` — Run only on weekends (Saturday=6, Sunday=0)
- `continue-on-error: true` — Don't block Phase 2 if gap filler fails
- Secrets needed:
  - `SCREENER_SESSION_COOKIE` (from your .env)
  - `GDRIVE_FOLDER_ID` (from your .env)
  - `GDRIVE_OAUTH_TOKEN_PATH` (from your .env)

---

## How It Works End-to-End

### Weekly Execution (Saturday Evening → Sunday Morning)

**Saturday 18:00 IST:**
```
phase2.yml triggers → quarterly_gap_filler job starts
  ↓
find_quarterly_gaps.py runs:
  - Auto-detects quarter (Q4FY if April–June)
  - Queries Screener (Apr 1 – Jun 30)
  - Finds ~450 concalls
  - Dedup checks → identifies ~250 gaps (not in queue)
  - Saves gaps to Drive: concall_gaps_Q4FY.parquet
```

**Sunday 09:00 IST:**
```
phase2.yml triggers → quarterly_gap_filler job runs again
  ↓
merge_quarterly_gaps_to_queue.py runs:
  - Loads gaps (250 items)
  - Checks queue depth (say, 30 pending)
  - Calculates available slots: 75 - 30 = 45
  - Merges 45 new gaps to processing_queue
  - Saves updated queue to Drive
  - Logs: "Added 45 gaps, 205 remaining for next weekend"
```

**Tue–Fri (normal Phase 2 runs):**
```
extract_concall.py processes the 45 new queue items normally:
  - Downloads PDFs
  - Extracts via Gemini (quarter, guidance, facts)
  - Updates company_page.md
  - Stores in parquets
  - Marks status = done when complete
```

**Next Saturday 18:00 IST:**
```
Gap filler runs again:
  - find_quarterly_gaps.py finds 250 gaps again (some already queued from previous week)
  - Dedup check: "TCS 2026-05-15" is in queue (status=done), skip it
  - Only NEW gaps are added to new registry
  - Saves updated gaps

merge_quarterly_gaps_to_queue.py runs:
  - Checks queue depth (~20 pending, others done from previous run)
  - Merges next 50–55 gaps
  - Remaining: ~195 gaps for future weekends
```

---

## Safety Guarantees

### 1. **No Double-Processing**

Even if the same gap is discovered multiple times:
- Queue dedup key: `(isin, announcement_date)`
- Merge logic loads ENTIRE queue (pending + done + error)
- Done items are visible to dedup check
- Won't be re-added ✅

**Scenario:**
```
Week 1: TCS 2026-05-15 added to queue → status=done
Week 2: Gap filler finds TCS 2026-05-15 again
        Dedup check: (TCS, 2026-05-15) in queue_keys? YES
        Result: SKIP (not re-added) ✅
```

### 2. **Quarter-Safe History**

Dedup works across years:
```
quarterly_facts.parquet dedup key: (isin, quarter, source_doc_id)

TCS Q1 FY25 (2025-07-15) + TCS Q4 FY29 (2029-01-25):
  - Both rows coexist permanently
  - Different quarters → different dedup keys
  - No collision ✅
```

### 3. **Quota-Safe Processing**

Smart batch sizing prevents Gemini quota exhaustion:
- Current pending < 20 → push ~50–75 (aim for 75 total)
- Current pending ≥ 20 and < 150 → wait (let queue drain)
- Current pending ≥ 150 → dormant (queue backlogged) ✅

### 4. **Storage-Safe**

PDF retention: 2 days max on Drive
- 75 pending items × ~2 min/concall = 150 min = 2.5 hours
- Processed within 24h of download (safe before 2-day cleanup) ✅

---

## Monitoring

### Logs to Check

After gap filler runs, check GitHub Actions workflow logs:

```
[DEBUG] Quarter: Q4FY
[DEBUG] Window: 2026-04-01 – 2026-06-30
[...] Querying Screener for concalls...
[OK] Found 450 concalls on Screener
[OK] Found 250 gaps (new, not yet in queue)
[INFO] Skipped 10 audio files
[OK] Created concall_gaps_Q4FY.parquet

[SUMMARY]
  Pushed to queue: 45
  Remaining unpushed: 205
  New queue depth: 75
```

### Drive Files to Monitor

- `company_repo/_index/concall_gaps_Q4FY.parquet` — Current gaps registry
- `company_repo/_index/processing_queue.parquet` — Updated queue (new rows appended)
- Logs in future Phase 2 runs show these items being processed

---

## Troubleshooting

### "SCREENER_SESSION_COOKIE not set"
→ Add to `.env` and GitHub Secrets (see screener_client.py instructions)

### "GDRIVE auth failed"
→ Run `python scripts/auth_helper.py` to refresh token

### "Found 0 concalls on Screener"
→ Screener's HTML may have changed
→ Verify filter still works: https://www.screener.in/announcements/user-filters/76106/
→ If needed, update parsing logic in find_quarterly_gaps.py

### "Queue backlogged (XXX pending)"
→ Gemini quota likely exhausted
→ Gap filler will skip and retry next weekend
→ Monitor Phase 2 logs to see queue draining

---

## Future Enhancements

1. **BSE API Fallback**: If Screener parsing fails, query BSE Direct API
2. **Revised Concall Handling**: If MP3 published, then PDF → track both, use latest
3. **Per-Quota Batch Sizing**: Adjust batch based on daily Gemini quota remaining
4. **Email Notifications**: Alert user when gaps discovered/merged (for quarterly reviews)

---

## Files

- `scripts/find_quarterly_gaps.py` — Discovery script (250+ lines)
- `scripts/merge_quarterly_gaps_to_queue.py` — Merge script (280+ lines)
- `.github/workflows/phase2.yml` — Add `quarterly_gap_filler` job (see Integration section above)

---

## Summary

✅ **Automatic**, **weekly**, **quarter-aware** concall backfill system
✅ **Safe** dedup (prevents re-adding done items)
✅ **Quota-aware** (respects Gemini + 2-day storage)
✅ **Integrated** into phase2.yml (no separate workflow)
✅ **Reusable** for Q1/Q2/Q3/Q4 (auto-parametrized)

Expected result by end of Q4 FY26 (June 30, 2026): ~400–500 Q4 concalls backfilled into the system.
