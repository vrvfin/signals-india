"""
Phase 3 — T8: daily AR digest + FOCUS / DEFOCUS lists (user spec 2026-06-12).

Runs AFTER extract_annual_report --all-companies in each 4h backfill slot:

  1. Queue rows doc_type=annual_report processed in the last 24h (any company)
     -> pull each company's freshest AR section from its company_page.md.
  2. Write _daily/daily_AR_summary_DD_MMMYYYY.md (focus lists on top, then the
     per-company AR sections). Re-runs the same day REWRITE the file with the
     day's full set (idempotent across the 6 slots).
  3. One Gemini pass (BACKFILL pool, lite models) judges every AR:
       FOCUS   — chairman's letter guiding positive + good growth + capacity
                 addition + industry tailwind + financials improving across
                 ALL THREE statements.
       DEFOCUS — financial shenanigans + auditor qualifications/EoM + notes-to-
                 accounts red flags + accounting method/policy changes.
  4. Fan-out: FOCUS names -> deep_dive_queue (auto deep dive next run; doc-level
     dedup is inherited — summaries are cached sidecars the dive reuses);
     both lists -> _index/ar_focus.parquet (+csv) for app/fraud/catalyst use;
     mail digest (toggle 'ar_focus').

Usage:
    python scripts/daily_ar_summary.py --dry-run     # no Gemini, no writes
    python scripts/daily_ar_summary.py --hours 24
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, load_parquet,
                             save_parquet, log)
from mailer import send_email, load_mail_settings, esc

AR_FOCUS_COLS = ["isin", "symbol", "as_of", "list", "reasons", "computed_at"]
PER_AR_CAP = 4000          # chars of each company's AR section fed to the judge
TOTAL_CAP = 90_000         # max chars per judge call; chunked beyond this

JUDGE_PROMPT = """You are a buy-side gatekeeper reading TODAY'S fresh Annual Report
summaries (one section per company, produced by a forensic-analyst pipeline).
Sort the companies into FOCUS / DEFOCUS / NEITHER using ONLY these criteria:

FOCUS (must satisfy MOST of these, not just one):
  - Chairman's/MD's letter guiding POSITIVE with explicit forward commentary
  - Good revenue/profit growth trajectory
  - Capacity addition / capex pipeline with substance (numbers, timelines)
  - Management cites improving industry dynamics (tailwinds)
  - Financials improving across ALL THREE statements (P&L growth, balance
    sheet deleveraging/strengthening, operating cash flow backing profits)

DEFOCUS (ANY ONE of these is enough):
  - Signs of financial shenanigans (CFO-PAT divergence, CWIP games,
    receivable/inventory stretching, RPT loops)
  - Auditor qualification, Emphasis of Matter, or auditor change
  - Notes-to-accounts red flags (contingent liabilities, write-offs)
  - Change in accounting method/policy/estimates that flatters results

Reply EXACTLY in this format, one line per company you classify (skip NEITHER):
FOCUS|<SYMBOL>|<one-line why, naming the specific criteria met>
DEFOCUS|<SYMBOL>|<one-line why, naming the specific red flag>

Nothing else. Be selective — an empty list is a valid answer.

--- TODAY'S AR SUMMARIES ---
{summaries}
"""


def _build_pool():
    from gemini_pool import BucketPool, load_keys
    keys = load_keys(os.environ, prefix="BACKFILL_GEMINI_KEY")
    for prefix in ("GEMINI_API_KEY",):
        keys += [k for k in load_keys(os.environ, prefix=prefix) if k not in keys]
    if not keys:
        log("no Gemini keys — judge pass skipped.")
        return None
    return BucketPool(keys, ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"],
                      inter_call_s=6.0, logger=log)


def _ar_section(page_md: str, cap: int = PER_AR_CAP) -> str:
    """Freshest 'Annual Report' section of a company_page.md (best-effort)."""
    if not page_md:
        return ""
    hits = [m.start() for m in re.finditer(r"annual report", page_md, re.I)]
    if not hits:
        return page_md[-cap:]
    start = hits[-1]
    head = page_md.rfind("\n#", 0, start)          # back up to its heading
    return page_md[max(0, head):start + cap][:cap]


def _judge(pool, blocks: list[tuple[str, str]]) -> list[dict]:
    """blocks = [(symbol, section_text)] -> [{symbol, list, reasons}]."""
    out: list[dict] = []
    batch, size = [], 0
    batches = []
    for sym, txt in blocks:
        t = f"=== {sym} ===\n{txt}\n"
        if size + len(t) > TOTAL_CAP and batch:
            batches.append(batch)
            batch, size = [], 0
        batch.append(t)
        size += len(t)
    if batch:
        batches.append(batch)
    for b in batches:
        try:
            text, _ = pool.call_text(JUDGE_PROMPT.format(summaries="".join(b)))
        except Exception as e:
            log(f"  judge call failed ({str(e)[:80]}) — batch skipped")
            continue
        for line in text.strip().splitlines():
            m = re.match(r"\s*(FOCUS|DEFOCUS)\s*\|\s*([A-Z0-9._&-]+)\s*\|\s*(.+)",
                         line.strip(), re.I)
            if m:
                out.append({"list": m.group(1).upper(),
                            "symbol": m.group(2).upper(),
                            "reasons": m.group(3).strip()[:300]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="List today's ARs; no Gemini, no writes.")
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")
    today = datetime.now()
    as_of = today.strftime("%Y-%m-%d")

    qfid = find_file(drive, index_id, "processing_queue.parquet")
    queue = (pd.read_parquet(io.BytesIO(download_bytes(drive, qfid)))
             if qfid else pd.DataFrame())
    if queue.empty or "doc_type" not in queue.columns:
        log("processing_queue empty — nothing to do.")
        return
    cutoff = (today - timedelta(hours=args.hours)).isoformat(timespec="seconds")
    done_ar = queue[(queue["doc_type"].astype(str) == "annual_report")
                    & (queue["status"].astype(str) == "done")
                    & (queue["processed_at"].astype(str) >= cutoff)]
    done_ar = done_ar.drop_duplicates("isin")
    log(f"ARs summarised in last {args.hours:.0f}h: {len(done_ar)} company(ies)")
    if done_ar.empty:
        return

    blocks: list[tuple[str, str]] = []
    names: dict[str, str] = {}
    for _, r in done_ar.iterrows():
        isin, sym = str(r.get("isin", "")), str(r.get("symbol", "")).upper()
        if not isin or not sym:
            continue
        comp_id = find_file(drive, repo_id, isin)   # company folder
        if not comp_id:
            continue
        page_fid = find_file(drive, comp_id, "company_page.md")
        if not page_fid:
            continue
        try:
            page = download_bytes(drive, page_fid).decode("utf-8", errors="replace")
        except Exception:
            continue
        sec = _ar_section(page)
        if sec.strip():
            blocks.append((sym, sec))
            names[sym] = str(r.get("company_name", ""))

    log(f"AR sections pulled: {len(blocks)}")
    if args.dry_run:
        for sym, sec in blocks:
            log(f"  {sym:<14} {len(sec)} chars")
        return
    if not blocks:
        return

    pool = _build_pool()
    verdicts = _judge(pool, blocks) if pool else []
    focus = [v for v in verdicts if v["list"] == "FOCUS"]
    defocus = [v for v in verdicts if v["list"] == "DEFOCUS"]
    log(f"judge: FOCUS={len(focus)} DEFOCUS={len(defocus)} "
        f"of {len(blocks)} ARs")

    # ---- daily md (rewritten each slot with the day's full set) ----
    md = [f"# Daily Annual-Report digest — {today.strftime('%d %b %Y')}",
          f"\n*{len(blocks)} AR(s) summarised in the last {args.hours:.0f}h.*\n",
          "## 🎯 FOCUS list"]
    md += [f"- **{v['symbol']}** ({names.get(v['symbol'], '')}): {v['reasons']}"
           for v in focus] or ["- (none today)"]
    md += ["\n## 🚫 DEFOCUS list"]
    md += [f"- **{v['symbol']}** ({names.get(v['symbol'], '')}): {v['reasons']}"
           for v in defocus] or ["- (none today)"]
    md += ["\n---\n## Per-company AR sections\n"]
    md += [f"\n### {sym} — {names.get(sym, '')}\n\n{sec}\n" for sym, sec in blocks]
    daily_id = get_or_create_subfolder(drive, repo_id, "_daily")
    fname = f"daily_AR_summary_{today.strftime('%d_%b%Y')}.md"
    existing = find_file(drive, daily_id, fname)
    upload_bytes(drive, daily_id, fname, "\n".join(md).encode("utf-8"),
                 "text/markdown", existing_id=existing)
    log(f"wrote _daily/{fname}")

    # ---- ar_focus.parquet upsert (symbol, as_of) ----
    if verdicts:
        sym2isin = {str(r.get("symbol", "")).upper(): str(r.get("isin", ""))
                    for _, r in done_ar.iterrows()}
        new = pd.DataFrame([{
            "isin": sym2isin.get(v["symbol"], ""), "symbol": v["symbol"],
            "as_of": as_of, "list": v["list"], "reasons": v["reasons"],
            "computed_at": today.isoformat(timespec="seconds"),
        } for v in verdicts], columns=AR_FOCUS_COLS)
        old = load_parquet(drive, index_id, "ar_focus.parquet", AR_FOCUS_COLS)
        keep = old[~(old["symbol"].isin(new["symbol"]) & (old["as_of"] == as_of))] \
            if not old.empty else old
        out = pd.concat([keep, new], ignore_index=True)
        save_parquet(drive, index_id, "ar_focus.parquet", out)
        log(f"ar_focus.parquet: {len(out)} rows total")

    # ---- FOCUS -> deep_dive_queue (skip if already pending/done today) ----
    if focus:
        dq = load_parquet(drive, index_id, "deep_dive_queue.parquet",
                          ["token", "status", "added_at", "done_at", "error"])
        have = set(dq[dq["status"].astype(str) == "pending"]["token"]
                   .astype(str).str.upper()) if not dq.empty else set()
        adds = [{"token": v["symbol"], "status": "pending",
                 "added_at": today.isoformat(timespec="seconds")}
                for v in focus if v["symbol"] not in have]
        if adds:
            dq = pd.concat([dq, pd.DataFrame(adds)], ignore_index=True)
            save_parquet(drive, index_id, "deep_dive_queue.parquet", dq)
            log(f"deep_dive_queue: +{len(adds)} FOCUS name(s) enqueued")

    # ---- mail ----
    if not load_mail_settings(drive, index_id).get("ar_focus", True):
        log("ar_focus mail toggled OFF — skipped.")
        return
    if not verdicts and len(blocks) == 0:
        return
    rows = "".join(
        f"<tr><td>{'🎯' if v['list'] == 'FOCUS' else '🚫'} <b>{esc(v['symbol'])}</b>"
        f"</td><td>{esc(names.get(v['symbol'], ''), 30)}</td>"
        f"<td>{esc(v['reasons'], 280)}</td></tr>"
        for v in focus + defocus)
    body = (f"<p><b>{len(blocks)} Annual Report(s)</b> summarised in the last "
            f"{args.hours:.0f}h → 🎯 {len(focus)} FOCUS · 🚫 {len(defocus)} DEFOCUS."
            f"</p>"
            + (f"<table border=1 cellpadding=4 cellspacing=0><tr><th></th>"
               f"<th>Company</th><th>Why</th></tr>{rows}</table>" if rows else
               "<p>No company met the focus/defocus bar today.</p>")
            + f"<p style='font-size:11px;color:#999'>FOCUS names are auto-queued "
              f"for a deep dive. Full digest: _daily/{fname}. Toggle this mail "
              f"in the app sidebar.</p>")
    send_email(f"📚 AR digest — 🎯{len(focus)} / 🚫{len(defocus)} of "
               f"{len(blocks)} — {as_of}", body)


if __name__ == "__main__":
    main()
