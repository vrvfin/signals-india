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


def _md_to_pdf(md_text: str) -> bytes | None:
    """Render our own markdown digest to a simple A4 PDF via PyMuPDF Story
    (already a CI dep — no weasyprint needed). None on failure (fail-soft:
    caller attaches the .md instead)."""
    try:
        import fitz
        out = []
        for ln in md_text.splitlines():
            s = ln.rstrip()
            if s.startswith("### "):
                out.append(f"<h3>{esc(s[4:], 300)}</h3>")
            elif s.startswith("## "):
                out.append(f"<h2>{esc(s[3:], 300)}</h2>")
            elif s.startswith("# "):
                out.append(f"<h1>{esc(s[2:], 300)}</h1>")
            elif s.startswith("- "):
                out.append(f"<p>&bull; {esc(s[2:], 600)}</p>")
            elif s.strip() == "---":
                out.append("<hr>")
            elif s.strip():
                out.append(f"<p>{esc(s, 1200)}</p>")
        html = ("<body style='font-family:sans-serif;font-size:9pt;"
                "line-height:1.35'>" + "\n".join(out) + "</body>")
        story = fitz.Story(html=html)
        buf = io.BytesIO()
        writer = fitz.DocumentWriter(buf)
        page = fitz.paper_rect("a4")
        where = page + (36, 36, -36, -36)
        more = 1
        while more:
            dev = writer.begin_page(page)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
        writer.close()
        return buf.getvalue()
    except Exception as e:
        log(f"  PDF render failed ({str(e)[:80]}) — will attach .md instead")
        return None


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


def fetch_range(date_from: str, date_to: str) -> None:
    """Range-wise AR focus/defocus pull from ar_focus.parquet (get_ar_focus.bat).
    Writes ar_focus_<from>_<to>.md (+ .pdf when renderable) to the project root.
    Read-only on Drive; no Gemini."""
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")
    df = load_parquet(drive, index_id, "ar_focus.parquet", AR_FOCUS_COLS)
    sel = df[(df["as_of"] >= date_from) & (df["as_of"] <= date_to)] \
        .sort_values(["as_of", "list", "symbol"])
    log(f"ar_focus rows {date_from}..{date_to}: {len(sel)} "
        f"(FOCUS {int((sel['list'] == 'FOCUS').sum())} / "
        f"DEFOCUS {int((sel['list'] == 'DEFOCUS').sum())})")
    md = [f"# AR Focus/Defocus — {date_from} to {date_to}\n"]
    for lst, icon, mark in (("FOCUS", "🎯", "+"), ("DEFOCUS", "🚫", "-")):
        md.append(f"\n## {icon} {lst}")
        part = sel[sel["list"] == lst]
        md += [f"- **{r['symbol']}** [{r['as_of']}]: {r['reasons']}"
               for _, r in part.iterrows()] or ["- (none)"]
        for _, r in part.iterrows():   # console: ASCII only (cp1252-safe)
            print(f"  {mark} {str(r['as_of'])} {str(r['symbol']):<14} "
                  f"{str(r['reasons'])[:90]}".encode('ascii', 'replace').decode())
    out = Path(__file__).resolve().parent.parent / \
        f"ar_focus_{date_from}_{date_to}.md"
    text = "\n".join(md)
    out.write_text(text, encoding="utf-8")
    pdf = _md_to_pdf(text)
    if pdf:
        out.with_suffix(".pdf").write_bytes(pdf)
    log(f"wrote {out.name}" + (" + .pdf" if pdf else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="List today's ARs; no Gemini, no writes.")
    ap.add_argument("--email", action="store_true",
                    help="Send the digest mail. OFF in the 4h backfill slots; "
                         "the 19:00 IST t4_nightly batch passes it once a day.")
    ap.add_argument("--range-from", type=str, default=None, metavar="YYYY-MM-DD",
                    help="With --range-to: range-wise focus-list pull, then exit.")
    ap.add_argument("--range-to", type=str, default=None, metavar="YYYY-MM-DD")
    args = ap.parse_args()

    if args.range_from:
        fetch_range(args.range_from, args.range_to
                    or datetime.now().strftime("%Y-%m-%d"))
        return

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
    # PAUSED 2026-06-22 (user): deep_dive_queue is USER-PUSH-ONLY (Streamlit "Add to
    # Queue" / company_deep_report.py --add). This automatic FOCUS fan-out was an
    # oversight that filled the queue with non-PF names (e.g. Pennar). Re-enable by
    # setting env ENABLE_AR_FOCUS_DEEPDIVE_ENQUEUE=1.
    if focus and os.environ.get("ENABLE_AR_FOCUS_DEEPDIVE_ENQUEUE") == "1":
        # Coordinated enqueue (lock + dedup) so even if re-enabled this can't clobber
        # the queue or duplicate a name.
        from company_deep_report import enqueue_tokens
        n = enqueue_tokens(drive, os.environ["GDRIVE_FOLDER_ID"],
                           [v["symbol"] for v in focus], owner="ar_focus")
        log(f"deep_dive_queue: +{n} FOCUS name(s) enqueued (coordinated)")

    # ---- mail: body = FOCUS quick summaries with the WHY; full digest PDF attached ----
    if not args.email:
        log("no --email — digest/judge/enqueue done, mail left to the 19:00 batch.")
        return
    if not load_mail_settings(drive, index_id).get("ar_focus", True):
        log("ar_focus mail toggled OFF — skipped.")
        return
    if not verdicts and len(blocks) == 0:
        return
    focus_html = "".join(
        f"<p style='margin:6px 0'>🎯 <b>{esc(v['symbol'])}</b> · "
        f"{esc(names.get(v['symbol'], ''), 40)}<br>"
        f"<i>why focus:</i> {esc(v['reasons'], 320)}</p>"
        for v in focus) or "<p>(no FOCUS name today)</p>"
    defocus_html = "".join(
        f"<tr><td>🚫 <b>{esc(v['symbol'])}</b></td>"
        f"<td>{esc(names.get(v['symbol'], ''), 30)}</td>"
        f"<td>{esc(v['reasons'], 240)}</td></tr>" for v in defocus)
    body = (f"<p><b>{len(blocks)} Annual Report(s)</b> summarised in the last "
            f"{args.hours:.0f}h → 🎯 {len(focus)} FOCUS · 🚫 {len(defocus)} "
            f"DEFOCUS.</p><h3>🎯 Focus — why</h3>{focus_html}"
            + (f"<h3>🚫 Defocus</h3><table border=1 cellpadding=4 cellspacing=0>"
               f"<tr><th></th><th>Company</th><th>Red flag</th></tr>{defocus_html}"
               f"</table>" if defocus_html else "")
            + f"<p style='font-size:11px;color:#999'>Full per-company digest "
              f"attached as PDF (also at _daily/{fname}). FOCUS names are "
              f"auto-queued for deep dives. Toggle this mail in the app sidebar."
              f"</p>")
    md_text = "\n".join(md)
    pdf = _md_to_pdf(md_text)
    atts = ([(fname.replace(".md", ".pdf"), pdf, "pdf")] if pdf
            else [(fname, md_text.encode("utf-8"), "octet-stream")])
    send_email(f"📚 AR digest — 🎯{len(focus)} / 🚫{len(defocus)} of "
               f"{len(blocks)} — {as_of}", body, attachments=atts)


if __name__ == "__main__":
    main()
