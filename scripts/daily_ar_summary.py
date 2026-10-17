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

from _extractor_base import (get_drive, get_or_create_subfolder, find_file, load_table,
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


_VERDICT_RE = re.compile(r"\s*(FOCUS|DEFOCUS)\s*\|\s*([A-Z0-9._&-]+)\s*\|\s*(.+)", re.I)
JUDGE_MAX_TOKENS = 3000     # ~1 line per company; a looping lite model is cut off here


def parse_verdicts(text: str, allowed: set) -> list[dict]:
    """Judge reply -> one verdict per company. Pure.

    16 Oct 2026: the lite judge LOOPED and printed DEFOCUS lines thousands of times;
    every line was counted ("🎯14 / 🚫4023 of 20"). Only companies that were IN the
    batch count, and only their FIRST verdict (a loop repeats, it does not correct)."""
    out, seen = [], set()
    for line in str(text or "").strip().splitlines():
        m = _VERDICT_RE.match(line.strip())
        if not m:
            continue
        sym = m.group(2).upper()
        if sym not in allowed or sym in seen:
            continue
        seen.add(sym)
        out.append({"list": m.group(1).upper(), "symbol": sym,
                    "reasons": m.group(3).strip()[:300]})
    return out


# ---- one line each: growth guidance / risk / balance sheet (user 2026-10-17) ----------
_GROWTH_WORDS = ("revenue", "sales", "top line", "topline", "ebitda", "pat", "profit",
                 "growth", "volume", "earnings")
_SEV_RANK = {"high": 0, "critical": 0, "medium": 1, "moderate": 1, "low": 2}


def _blank(v) -> bool:
    return v is None or str(v).strip().lower() in ("", "nan", "none", "null", "na")


def horizon_end_fy(h) -> int | None:
    """'FY21' -> 2021, 'FY2025' -> 2025, 'FY26-27' / 'H1 2026-27' -> 2027. Pure."""
    txt = str(h or "")
    yrs = [int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", txt)]
    yrs += [2000 + int(y) for y in re.findall(r"(?i)fy\s?'?(\d{2})(?!\d)", txt)]
    tail = re.search(r"(20\d{2}|\d{2})\s*[-/]\s*(\d{2})(?!\d)", txt)
    if tail:
        yrs.append(2000 + int(tail.group(2)))
    return max(yrs) if yrs else None


def current_fy(today=None) -> int:
    """Indian FY ending year: Oct 2026 -> 2027. Pure."""
    t = today or datetime.now()
    return t.year + 1 if t.month >= 4 else t.year


def guidance_line(g: pd.DataFrame, now_fy: int | None = None) -> str:
    """The AR's growth guidance in one line, from ar_guidance rows. Pure.

    Guidance whose target year has PASSED is dropped: a backfilled FY2019-20 AR's
    "capex by FY21" is history, not guidance (seen 17 Oct 2026)."""
    if g is None or g.empty:
        return "not stated in the AR"
    now_fy = now_fy or current_fy()
    hz_col = g["horizon_fy"] if "horizon_fy" in g.columns else pd.Series([None] * len(g), index=g.index)
    keep = []
    for h in hz_col:
        end = horizon_end_fy(h)
        keep.append(not (end is not None and end < now_fy))
    g = g[keep]
    if g.empty:
        return "only past-dated guidance in this AR"
    def score(r):
        txt = f"{r.get('metric', '')} {r.get('guidance_type', '')}".lower()
        return (0 if any(w in txt for w in _GROWTH_WORDS) else 1,
                0 if not _blank(r.get("cagr_pct")) or not _blank(r.get("value")) else 1)
    rows = sorted((r for _, r in g.iterrows()), key=score)
    bits = []
    for r in rows[:2]:
        val = (f"{float(r['cagr_pct']):g}% CAGR" if not _blank(r.get("cagr_pct"))
               else f"{r.get('value')}{'' if _blank(r.get('unit')) else ' ' + str(r.get('unit'))}"
               if not _blank(r.get("value")) else "")
        hz = "" if _blank(r.get("horizon_fy")) else f" by {r.get('horizon_fy')}"
        note = "" if _blank(r.get("notes")) else f" ({str(r.get('notes'))[:90]})"
        body = " ".join(x for x in (val, hz.strip(), note.strip()) if x)
        bits.append(f"{str(r.get('metric') or '').strip()}: {body}" if body
                    else str(r.get("metric") or "").strip())
    return "; ".join(b for b in bits if b) or "not stated in the AR"


def risk_line(f: pd.DataFrame) -> str:
    """The AR's most severe red flag in one line, from ar_red_flags rows. Pure."""
    if f is None or f.empty:
        return "no red flag found in the AR"
    rows = sorted((r for _, r in f.iterrows()),
                  key=lambda r: _SEV_RANK.get(str(r.get("severity") or "").strip().lower(), 3))
    r = rows[0]
    sev = "" if _blank(r.get("severity")) else f"[{str(r.get('severity')).upper()}] "
    cat = str(r.get("flag_type") or r.get("category") or "").strip()
    ev = "" if _blank(r.get("evidence")) else str(r.get("evidence")).strip()[:160]
    pg = "" if _blank(r.get("page_ref")) else f" (p. {r.get('page_ref')})"
    more = f" · +{len(rows) - 1} more" if len(rows) > 1 else ""
    return f"{sev}{cat}{': ' if cat and ev else ''}{ev}{pg}{more}".strip() or \
        "no red flag found in the AR"


def _fy_key(p) -> int:
    m = re.search(r"(\d{4})", str(p))
    return int(m.group(1)) if m else -1


def bs_line(stmts: pd.DataFrame) -> str:
    """Balance sheet in one line from Screener statements: latest FY vs prior FY —
    borrowings, net worth (equity + reserves), debt/equity, investments. Pure."""
    if stmts is None or stmts.empty or "statement" not in stmts.columns:
        return "no balance sheet on file"
    b = stmts[stmts["statement"].astype(str) == "balance_sheet"]
    if b.empty:
        return "no balance sheet on file"
    def item(name):
        s = b[b["line_item"].astype(str).str.strip() == name]
        return {str(p): pd.to_numeric(v, errors="coerce") for p, v in zip(s["period"], s["value"])}
    borr, eq, res, inv = item("Borrowings"), item("Equity Capital"), item("Reserves"), item("Investments")
    periods = sorted(set(eq) | set(res) | set(borr), key=_fy_key)
    if not periods:
        return "no balance sheet on file"
    cur = periods[-1]
    prv = periods[-2] if len(periods) > 1 else None
    def nw(p):
        e, r = eq.get(p), res.get(p)
        return None if e is None or r is None or pd.isna(e) or pd.isna(r) else float(e + r)
    def cr(v):
        return "–" if v is None or pd.isna(v) else f"₹{v:,.0f} cr"
    def de(p):
        n, d = nw(p), borr.get(p)
        return None if not n or d is None or pd.isna(d) or n <= 0 else float(d) / n
    def vs(cur_v, prv_v, fmt):
        return "" if prv is None or prv_v is None or pd.isna(prv_v) else f" (vs {fmt(prv_v)})"
    d_c, d_p = de(cur), de(prv) if prv else None
    parts = [f"{cur}: debt {cr(borr.get(cur))}{vs(None, borr.get(prv) if prv else None, cr)}",
             f"net worth {cr(nw(cur))}{vs(None, nw(prv) if prv else None, cr)}",
             "D/E " + ("–" if d_c is None else f"{d_c:.2f}")
             + ("" if d_p is None else f" (vs {d_p:.2f})")]
    if inv.get(cur) is not None and not pd.isna(inv.get(cur)) and inv.get(cur) > 0:
        parts.append(f"investments {cr(inv.get(cur))}")
    return " · ".join(parts)


def _order(blocks, focus, defocus) -> list:
    """FOCUS first, then DEFOCUS, then every other AR of the day. Pure."""
    f = [v["symbol"] for v in focus]
    d = [v["symbol"] for v in defocus if v["symbol"] not in f]
    rest = [s for s, _ in blocks if s not in f and s not in d]
    return f + d + rest


def company_cards(order, names, lines, focus, defocus) -> str:
    """One card per AR: verdict + why, then growth guidance / risk / balance sheet. Pure."""
    why = {v["symbol"]: (v["list"], v["reasons"]) for v in focus + defocus}
    out = []
    for sym in order:
        g, rk, bs, pf, fy = _line(lines, sym)
        lst, rs = why.get(sym, ("", ""))
        tag, col = (("🎯 FOCUS", "#1a7a3a") if lst == "FOCUS" else
                    ("🚫 DEFOCUS", "#c0392b") if lst == "DEFOCUS" else ("", "#8a97a0"))
        pf_chip = (" <span style='background:#1f4d6b;color:#fff;border-radius:3px;"
                   "padding:0 5px;font-size:11px'>PF</span>" if pf else "")
        out.append(
            f"<div style='border:1px solid #e0e6ea;border-left:4px solid {col};"
            f"padding:6px 10px;margin:8px 0;font-family:Arial,sans-serif;font-size:13px'>"
            f"<div><b>{esc(sym)}</b> · {esc(names.get(sym, ''), 50)}{pf_chip}"
            + (f" <span style='color:#8a97a0;font-size:11.5px'>· {esc(fy, 20)}</span>" if fy else "")
            + (f" <span style='color:{col};font-weight:700'>{tag}</span>" if tag else "")
            + "</div>"
            + (f"<div style='color:#444;margin:2px 0'><i>why:</i> {esc(rs, 300)}</div>" if rs else "")
            + "<table cellpadding='2' style='font-size:12.5px;border-collapse:collapse'>"
            f"<tr><td style='color:#6b4d8f;width:120px;vertical-align:top'><b>Growth guidance</b></td>"
            f"<td>{esc(g, 300)}</td></tr>"
            f"<tr><td style='color:#c0392b;vertical-align:top'><b>Risk</b></td><td>{esc(rk, 300)}</td></tr>"
            f"<tr><td style='color:#1f4d6b;vertical-align:top'><b>Balance sheet</b></td>"
            f"<td>{esc(bs, 300)}</td></tr></table></div>")
    return "".join(out)


_AR_GUIDANCE_COLS = ["isin", "symbol", "fy_year", "metric", "guidance_type", "horizon_fy",
                     "value", "unit", "cagr_pct", "notes", "source_doc_id"]
_AR_REDFLAG_COLS = ["isin", "symbol", "fy_year", "category", "flag_type", "severity",
                    "evidence", "page_ref", "source_doc_id"]


def _rows_for(df: pd.DataFrame, isin: str, doc_id: str) -> pd.DataFrame:
    """This AR's rows: by its own document id, else the company's latest FY."""
    if df is None or df.empty:
        return pd.DataFrame()
    if doc_id and "source_doc_id" in df.columns:
        x = df[df["source_doc_id"].astype(str) == doc_id]
        if not x.empty:
            return x
    x = df[df["isin"].astype(str) == isin]
    if x.empty or "fy_year" not in x.columns:
        return x
    fy = pd.to_numeric(x["fy_year"], errors="coerce")
    return x[fy == fy.max()] if fy.notna().any() else x


def ar_label(*frames) -> str:
    """'FY2019-20 AR' from the rows' fy_year, so a backfilled old report is obvious. Pure."""
    for x in frames:
        if x is not None and not x.empty and "fy_year" in x.columns:
            ys = [y for y in (horizon_end_fy(v) for v in x["fy_year"]) if y]   # "FY25" -> 2025
            if ys:
                y = max(ys)
                return f"FY{y - 1}-{str(y)[2:]} AR"
    return ""


def _line(lines: dict, sym: str) -> tuple:
    """(growth, risk, balance sheet, is_pf, AR label) with blanks for a missing entry."""
    t = tuple(lines.get(sym, ()))
    return t + ("–", "–", "–", False, "")[len(t):]


def _one_liners(drive, root, index_id, done_ar, syms) -> dict:
    """{symbol: (growth guidance, risk, balance sheet, is_pf)} — read only."""
    out = {}
    try:
        gd = load_parquet(drive, index_id, "ar_guidance.parquet", _AR_GUIDANCE_COLS)
        rf = load_parquet(drive, index_id, "ar_red_flags.parquet", _AR_REDFLAG_COLS)
    except Exception as e:
        log(f"  AR guidance/red-flag tables unavailable ({str(e)[:60]})")
        gd = rf = pd.DataFrame()
    try:
        from daily_brief import load_pf
        pf_isins = {str(i) for i, _s, _n in load_pf(drive, root, index_id)}
    except Exception:
        pf_isins = set()
    fund = get_or_create_subfolder(drive, root, "fundamentals")
    st_id = get_or_create_subfolder(drive, fund, "statements")
    want = set(syms)
    for _, r in done_ar.iterrows():
        sym = str(r.get("symbol", "")).upper()
        if sym not in want:
            continue
        isin, doc = str(r.get("isin", "")), str(r.get("doc_id", "") or "")
        stmts = pd.DataFrame()
        try:
            fid = find_file(drive, st_id, f"{sym}.parquet")
            if fid:
                stmts = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
        except Exception:
            pass
        g_rows, f_rows = _rows_for(gd, isin, doc), _rows_for(rf, isin, doc)
        out[sym] = (guidance_line(g_rows), risk_line(f_rows), bs_line(stmts),
                    isin in pf_isins, ar_label(g_rows, f_rows))
    return out


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
    done = set()
    for b in batches:
        allowed = {t.split("===")[1].strip().upper() for t in b}
        try:
            text, _ = pool.call_text(JUDGE_PROMPT.format(summaries="".join(b)),
                                     max_output_tokens=JUDGE_MAX_TOKENS)
        except Exception as e:
            log(f"  judge call failed ({str(e)[:80]}) — batch skipped")
            continue
        got = [v for v in parse_verdicts(text, allowed - done)]
        n_lines = sum(1 for ln in str(text).splitlines() if _VERDICT_RE.match(ln.strip()))
        if n_lines > len(got):
            log(f"  judge: {n_lines} verdict lines for {len(allowed)} companies — "
                f"kept {len(got)} (one per company in the batch)")
        done |= {v["symbol"] for v in got}
        out += got
    return out


def fetch_range(date_from: str, date_to: str) -> None:
    """Range-wise AR focus/defocus pull from ar_focus.parquet (get_ar_focus.bat).
    Writes ar_focus_<from>_<to>.md (+ .pdf when renderable) to the project root.
    Read-only on Drive; no Gemini."""
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")
    # load_table, NOT load_parquet: this frame is written straight back, and
    # load_parquet ends in `return df[cols]` - it SLICES - so any column this
    # module's list does not name would be DELETED for every other pipeline.
    df = load_table(drive, index_id, "ar_focus.parquet", AR_FOCUS_COLS)
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
    lines = _one_liners(drive, root, index_id, done_ar, [s for s, _ in blocks])
    if args.dry_run:
        for sym, sec in blocks:
            g, rk, bs, pf, fy = _line(lines, sym)
            log(f"  {sym:<14} {fy:<13} {len(sec)} chars{'  [PF]' if pf else ''}\n"
                f"      growth guidance: {g}\n      risk: {rk}\n      balance sheet: {bs}")
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
    md += ["\n## In one line each — growth guidance · risk · balance sheet"]
    for sym in _order(blocks, focus, defocus):
        g, rk, bs, pf, fy = _line(lines, sym)
        md += [f"\n**{sym}** ({names.get(sym, '')}){' · PF' if pf else ''}"
               f"{' · ' + fy if fy else ''}",
               f"- Growth guidance: {g}", f"- Risk: {rk}", f"- Balance sheet: {bs}"]
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
        # written back a few lines below — must not slice. See load_table.
        old = load_table(drive, index_id, "ar_focus.parquet", AR_FOCUS_COLS)
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
    body = (f"<p><b>{len(blocks)} Annual Report(s)</b> summarised in the last "
            f"{args.hours:.0f}h → 🎯 {len(focus)} FOCUS · 🚫 {len(defocus)} "
            f"DEFOCUS.</p>"
            + company_cards(_order(blocks, focus, defocus), names, lines, focus, defocus)
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


def _self_test() -> int:
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    loop = "FOCUS|AAA|capex + tailwinds\n" + "DEFOCUS|BBB|auditor EoM\n" * 4000 + \
           "DEFOCUS|ZZZ|not in this batch\nFOCUS|BBB|contradiction later\nnoise line"
    v = parse_verdicts(loop, {"AAA", "BBB", "CCC"})
    check("looping judge -> one verdict per company (16 Oct: 4,023 of 20)", len(v) == 2)
    check("first verdict wins", [x["list"] for x in v] == ["FOCUS", "DEFOCUS"])
    check("company outside the batch ignored", all(x["symbol"] != "ZZZ" for x in v))

    g = pd.DataFrame([
        {"metric": "Dividend", "guidance_type": "other", "value": "2", "unit": "Rs",
         "cagr_pct": None, "horizon_fy": None, "notes": None},
        {"metric": "Revenue", "guidance_type": "growth", "value": None, "unit": None,
         "cagr_pct": 18, "horizon_fy": "FY28", "notes": "led by exports"}])
    gl = guidance_line(g)
    check("growth guidance first: metric, CAGR, horizon, note",
          gl.startswith("Revenue: 18% CAGR by FY28 (led by exports)"))
    check("no guidance rows -> says so", guidance_line(pd.DataFrame()) == "not stated in the AR")
    check("horizon parse", (horizon_end_fy("FY21"), horizon_end_fy("FY2025"),
                            horizon_end_fy("FY26-27"), horizon_end_fy("H1 2026-27"),
                            horizon_end_fy("medium term")) == (2021, 2025, 2027, 2027, None))
    old = pd.DataFrame([{"metric": "capex", "guidance_type": "capex", "value": None, "unit": None,
                         "cagr_pct": None, "horizon_fy": "FY21", "notes": "refurbish"}])
    check("past-dated guidance dropped", guidance_line(old, 2027) == "only past-dated guidance in this AR")
    check("current FY: Oct 2026 -> 2027", current_fy(datetime(2026, 10, 17)) == 2027)
    check("AR label from fy_year", ar_label(pd.DataFrame([{"fy_year": "FY20"}])) == "FY2019-20 AR")

    f = pd.DataFrame([
        {"category": "governance", "flag_type": "RPT", "severity": "low", "evidence": "x", "page_ref": 4},
        {"category": "audit", "flag_type": "Emphasis of Matter", "severity": "high",
         "evidence": "auditor flagged going concern", "page_ref": 112}])
    rl = risk_line(f)
    check("most severe flag first, with evidence + page + count",
          rl.startswith("[HIGH] Emphasis of Matter: auditor flagged going concern (p. 112)")
          and "+1 more" in rl)
    check("no flags -> says so", risk_line(None) == "no red flag found in the AR")

    st = pd.DataFrame([
        {"statement": "balance_sheet", "line_item": it, "period": p, "value": v}
        for it, vals in {"Equity Capital": (10, 10), "Reserves": (990, 1190),
                         "Borrowings": (500, 400), "Investments": (0, 120)}.items()
        for p, v in zip(("Mar 2025", "Mar 2026"), vals)])
    bl = bs_line(st)
    check("balance sheet: latest FY vs prior, debt / net worth / D/E / investments",
          bl == "Mar 2026: debt ₹400 cr (vs ₹500 cr) · net worth ₹1,200 cr (vs ₹1,000 cr) · "
                "D/E 0.33 (vs 0.50) · investments ₹120 cr")
    check("no statements -> says so", bs_line(pd.DataFrame()) == "no balance sheet on file")

    order = _order([("AAA", ""), ("BBB", ""), ("CCC", "")],
                   [{"symbol": "CCC"}], [{"symbol": "AAA"}])
    check("order: FOCUS, DEFOCUS, rest", order == ["CCC", "AAA", "BBB"])
    cards = company_cards(["AAA"], {"AAA": "A Ltd"}, {"AAA": ("g1", "r1", "b1", True, "FY2025-26 AR")},
                          [], [{"symbol": "AAA", "list": "DEFOCUS", "reasons": "EoM"}])
    check("card carries PF chip, verdict, why and the three lines",
          all(s in cards for s in ("PF", "DEFOCUS", "EoM", "Growth guidance", "g1", "FY2025-26 AR",
                                   "Risk", "r1", "Balance sheet", "b1")))
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    main()
