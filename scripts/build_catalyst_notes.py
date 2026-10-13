"""
Phase 3 — T5: Catalyst notes for strong stocks ("why is it moving?").

For names where >= 2 strategies agree (user decision 2026-06-10: n_strategies > 1),
fetch the last 7 days of TRUSTED news (shared news_fetch.py whitelist), add the
tail of company_page.md when present, and ask a lite Gemini model for a short
catalyst note. Writes:

  company_repo/<ISIN>/company_catalyst_DDMMMYY.md     — the note (downloadable)
  company_repo/_index/catalyst_index.parquet (+.csv)  — one row per note,
      CATALYST_COLS = isin, symbol, as_of, headline, catalyst_type, tags,
                      md_path, n_sources, computed_at

Quota discipline:
  - No trusted news in the window -> NO Gemini call, no note (nothing moving).
  - A note already written today for a symbol -> skipped (idempotent nightly).
  - --limit caps Gemini calls per run (default 30); rotation is stalest-first
    by previous note date so all eligible names cycle through over days.
  - Pool: GEMINI_API_KEY -> BACKFILL_GEMINI_KEY cascade, lite models only.

Usage:
    python scripts/build_catalyst_notes.py --dry-run          # list eligible, no network
    python scripts/build_catalyst_notes.py --names "TCS"      # ad-hoc, bypasses cap
    python scripts/build_catalyst_notes.py                    # nightly run (cap 30)
"""

from __future__ import annotations

import argparse
import html as html_mod
import io
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from _extractor_base import (
    get_drive, get_or_create_subfolder, find_file, download_bytes, upload_bytes,
    load_portfolio_isins,
)
import news_fetch

DATA_MISSING = "DATA_MISSING"

CATALYST_COLS = [
    "isin", "symbol", "as_of", "headline", "catalyst_type", "tags",
    "what_to_track",          # 2026-06-12: concrete monitorables (user ask)
    "note_text",              # 2026-06-17: note body inline, so the app can diff
                              # today's note vs the prior as_of without fetching md
    "md_path", "n_sources", "computed_at",
    # 2026-10-12 (ADDITIVE; old rows read None): 12-char fingerprint of the evidence
    # the note was written from. Same evidence next time -> no Gemini call, no note.
    "evidence_hash",
]

CATALYST_TYPES = {"order_win", "mgmt_change", "policy", "sector",
                  "results", "corporate_action", "unknown"}

MIN_STRATEGIES = 2          # n_strategies > 1 per user decision
NEWS_DAYS = 7


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ------------------------------------------------------------------ #
#  Storage abstraction (same pattern as the other T4/T5 scripts)      #
# ------------------------------------------------------------------ #

from _t4_store import Store




# ------------------------------------------------------------------ #
#  Gemini pool + note generation                                      #
# ------------------------------------------------------------------ #

# Static fallback, used only when the registry cannot be read. Kept in step with
# CHAINS["LITE_UTILITY"] in model_registry.py.
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite",
                   "gemini-3.5-flash-lite"]


def _build_gemini_pool(drive=None, index_id: str = ""):
    """GEMINI_API_KEY -> BACKFILL_GEMINI_KEY cascade, lite models (rule #4).

    WAS ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]. The second is 404 — retired
    by Google, probed 2026-09-04 — so this nightly job had no working fallback when
    the first model was busy. The chain now comes from model_registry, which drops a
    model the day after it dies.
    """
    try:
        from gemini_pool import BucketPool, load_keys
        keys = load_keys(os.environ, prefix="GEMINI_API_KEY")
        keys += [k for k in load_keys(os.environ, prefix="BACKFILL_GEMINI_KEY")
                 if k not in keys]
        if not keys:
            log("no GEMINI keys found — cannot generate notes.")
            return None
        try:
            from model_registry import resolve
            models = resolve("LITE_UTILITY", drive, index_id) or FALLBACK_MODELS
        except Exception as e:
            log(f"  model registry unavailable ({str(e)[:60]}) — static chain")
            models = FALLBACK_MODELS
        log(f"  models: {', '.join(models)}")
        return BucketPool(keys, models, inter_call_s=6.0, logger=log)
    except Exception as e:
        log(f"Gemini pool init failed: {str(e)[:80]}")
        return None


PROMPT = """You are a senior equity research analyst covering Indian listed companies.
Your readers scan many charts quickly — they need the sharpest possible answer to
"why is {company} ({symbol}) moving, does it matter, and what exactly do I watch
next". Work ONLY from the recent trusted headlines and the optional company brief
below. Never invent facts that are not in them.

Reply in EXACTLY this format (4 metadata lines, then markdown bullets):
TYPE=<one of: order_win|mgmt_change|policy|sector|results|corporate_action|unknown>
HEADLINE=<one factual line, <=120 chars, no hype, include the key number if available>
TAGS=<2-4 comma-separated lowercase tags>
WHAT_TO_TRACK=<2-4 concrete monitorables separated by " | ". Each must be CHECKABLE —
  a number, a date, an event or a filing. GOOD: "order-book conversion in Q1FY27
  results (Jul-26)" / "promoter pledge % in next shareholding filing" / "capacity
  commissioning timeline for the new line". BAD: "watch performance", "monitor
  sentiment". If TYPE=unknown, give the disconfirming check instead, e.g.
  "verify any exchange filing behind the price move".>

Then 4-6 markdown bullets, in this priority order:
- WHAT happened — the catalyst itself with its specific numbers (deal size, %,
  capacity, stake) and which headline supports it [source].
- HOW MATERIAL — size it against the company (vs revenue / market cap when the brief
  allows). Needle-moving or routine?
- DURABILITY — one-off (single order, settlement, block deal) vs structural (new
  segment, policy tailwind, recurring demand). Say which and why.
- SKEPTICISM — what makes this LESS bullish than the headline reads: promotional
  tone, unnamed sources, re-announcement of old news, missing counterparty, or a
  small-cap price-action story dressed up as fundamentals. Flag it explicitly.
- If the headlines show NO clear catalyst, say so honestly (TYPE=unknown) and state
  what the noise actually is (routine AGM coverage, sector listicle, etc.).

--- RECENT TRUSTED HEADLINES (last {days} days) ---
{headlines}

--- EXCHANGE FILINGS (BSE corporate announcements, last {days} days) ---
{filings}
(Filings are PRIMARY evidence — an order win or approval filed with the
exchange outranks a news story. BUT most filings are routine compliance:
investor-meet intimations, reg. 74(5) certificates, trading-window closures,
newspaper-publication copies — these are NOT catalysts. Only treat a filing
as a catalyst if it discloses new business substance.)

--- COMMUNITY COVERAGE (ValuePickr top contributors / curated blogs / X) ---
{community}
(Community posts are OPINION from experienced retail researchers — useful for
catalysts the press missed, but verify substance; never treat price talk as a
catalyst. Each line carries its direct link.)

--- INTERNAL RESEARCH NOTES (the user's own research intake; may be empty) ---
{research}
(Curated analyst/sector notes the user collected — treat as informed internal
perspective: corroborate or contrast with the headlines/filings, cite as
[internal research].)

--- COMPANY BRIEF (may be empty) ---
{brief}
"""


from mailer import esc as _esc_base


def _esc(s, n=120) -> str:
    return _esc_base(s, n)


def _research_notes(ridx, isin: str, days: int = 14) -> list[str]:
    """User's own research intake (Workflow A research_index) mentioning this
    company — third evidence source for catalysts (user 2026-06-12)."""
    if ridx is None or ridx.empty or not isin:
        return []
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        hit = ridx[ridx["isins"].astype(str).str.contains(isin, na=False)
                   & (ridx["processed_at"].astype(str) >= cutoff)]
        return [f"- [{r.get('doc_type', '?')} {str(r.get('doc_date', ''))[:10]}] "
                f"{str(r.get('file_name', ''))[:60]}: "
                f"{str(r.get('summary_md', ''))[:280]}"
                for _, r in hit.tail(3).iterrows()]
    except Exception:
        return []


def _recent_filings(bse_code, days: int) -> list[str]:
    """BSE corporate announcements for the prompt, last `days` only (user
    2026-06-12: filings often ARE the catalyst — order wins, approvals,
    capacity — and reach the exchange before the press). Reuses the deep
    dive's fetcher. Fail-soft []."""
    if not bse_code or str(bse_code).lower() in ("", "nan", "none"):
        return []
    try:
        from company_deep_report import bse_announcements
        block = bse_announcements(bse_code)
    except Exception as e:
        log(f"  filings fetch failed ({str(e)[:60]})")
        return []
    if not block or block.startswith(("DATA_MISSING", "No announcements")):
        return []
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    out = [ln for ln in block.splitlines()
           if (m := re.match(r"- (\d{4}-\d{2}-\d{2})", ln)) and m.group(1) >= cutoff]
    return out[:12]


def evidence_hash(items, filings, research, community) -> str:
    """Order-independent fingerprint of everything a note is written FROM. Headlines
    are matched on their title text only (the RSS link and timestamp change between
    fetches of the same story)."""
    import hashlib
    lines = sorted({str(i.get("title", "")).strip().lower() for i in items or []}
                   | {str(x).strip().lower() for x in (filings or [])}
                   | {str(x).strip().lower() for x in (research or [])}
                   | {str(x).strip().lower() for x in (community or [])})
    lines = [x for x in lines if x]
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:12] if lines else ""


def _norm_headline(h) -> str:
    return " ".join(str(h or "").lower().split())


# SAME STORY (user-approved 2026-10-12). The model rewords the same event every night:
# WELCORP's one US$412.5 mn order was re-mailed on 3, 9, 10 and 11 Oct as four
# different headlines. Two notes are the same story when they share the catalyst type
# AND an AMOUNT - a number tied to a currency or a unit (Rs 2,026 Cr, US$412.5 mn,
# 24%, 122,662 tons). Bare numbers (dates, "FY27") are ignored: matching on them
# dropped a genuinely new plant commissioning in the replay.
SAME_STORY_DAYS = 14
_CUR = r"(?:rs\.?|inr|₹|\$|us\$|usd|sar|eur)"
_UNIT = r"(?:cr|crs?|crore|crores|mn|million|bn|billion|lakh|%|tons?|tonnes?|mw|gw)"
# (?![a-z]) not \b: "24%" followed by a space has no word boundary after the "%"
_AMOUNT = re.compile(rf"{_CUR}\s*(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s*{_UNIT}(?![a-z])",
                     re.I)


def amounts(headline) -> set[str]:
    out = set()
    for m in _AMOUNT.finditer(str(headline or "")):
        v = (m.group(1) or m.group(2)).replace(",", "")
        if v.replace(".", "").strip("0"):
            out.add(v)
    return out


def same_story(recent: list[tuple], ctype: str, headline: str) -> bool:
    """recent = [(catalyst_type, headline)] for this symbol's notes in the last
    SAME_STORY_DAYS days (before today)."""
    mine = amounts(headline)
    return bool(mine) and any(t == ctype and (mine & amounts(h)) for t, h in recent)


def note_changed(prev: dict | None, ctype: str, headline: str) -> bool:
    """A note is NEWS when there was no earlier note, or its catalyst type or headline
    differs from the symbol's previous note (user 2026-10-10: 'only where there is a
    change')."""
    if not prev:
        return True
    return (str(prev.get("catalyst_type") or "") != str(ctype or "")
            or _norm_headline(prev.get("headline")) != _norm_headline(headline))


def actionable(rows: list[dict]) -> list[dict]:
    """Rows worth a reader's time: a real catalyst (not 'unknown') that is new."""
    return [r for r in rows
            if str(r.get("catalyst_type") or "") != "unknown" and r.get("_changed")]


def catalyst_mail_html(rows: list[dict], counts: dict) -> str:
    """Only the notes that changed and name a real catalyst. Everything else is one
    line of counts, so a quiet night costs the reader one glance (or no mail)."""
    rows = sorted(rows, key=lambda r: (not r.get("_in_pf"), r["symbol"]))
    td = "padding:6px 8px;border-bottom:1px solid #eee;vertical-align:top;font-size:13px"
    body = "".join(
        f"<tr><td style='{td}'>{'💼 ' if r.get('_in_pf') else ''}<b>"
        f"{_esc(r['symbol'])}</b><br><span style='color:#888;font-size:11px'>"
        f"{_esc(r['catalyst_type'])}</span></td>"
        f"<td style='{td}'><b>{_esc(r['headline'], 200)}</b>"
        + (f"<br>👁 <i>{_esc(r.get('what_to_track', ''), 260)}</i>"
           if r.get("what_to_track") else "")
        + (f"<br><span style='color:#999;font-size:11px'>previous note "
           f"({_esc(r.get('_prev_as_of'), 10)}): {_esc(r.get('_prev_headline'), 140)}"
           f"</span>" if r.get("_prev_headline") else
           "<br><span style='color:#1a7a3a;font-size:11px'>first note for this stock"
           "</span>")
        + "</td></tr>" for r in rows)
    c = counts
    return (
        f"<div style='font-family:Arial,sans-serif;max-width:720px'>"
        f"<p style='font-size:13px'><b>{len(rows)} new catalyst"
        f"{'' if len(rows) == 1 else 's'}</b> tonight — only notes that name a real "
        f"catalyst AND differ from that stock's previous note. Holdings first.</p>"
        f"<table style='border-collapse:collapse;width:100%'>{body}</table>"
        f"<p style='font-size:11px;color:#999;margin-top:10px'>Also scanned "
        f"{c.get('selected', 0)} stocks: {c.get('quiet', 0)} had no news at all · "
        f"{c.get('unchanged', 0)} had the same news as last time (no AI call spent) · "
        f"{c.get('no_catalyst', 0)} had news but no clear catalyst · "
        f"{c.get('same_note', 0)} repeated a story already mailed in the last "
        f"{SAME_STORY_DAYS} days. Full notes: Stock "
        f"Detail / Scorecard pages. Toggle this mail in the app sidebar.</p></div>")


def make_note(pool, company: str, symbol: str, items: list[dict],
              filings: list[str], research: list[str], community: list[str],
              brief: str) -> tuple[str, str, str, str, str] | None:
    """Returns (catalyst_type, headline, tags, what_to_track, md_body) or None."""
    headlines = "\n".join(
        f"- [{i['source']}] {i['title']} ({i['published'][:16]})"
        for i in items) or "(none in window)"
    prompt = PROMPT.format(company=company or symbol, symbol=symbol,
                           days=NEWS_DAYS, headlines=headlines,
                           filings="\n".join(filings) or "(none in window)",
                           community="\n".join(community) or "(none)",
                           research="\n".join(research) or "(none)",
                           brief=(brief or "")[:6000])
    try:
        text, _ = pool.call_text(prompt)
    except Exception as e:
        log(f"  Gemini failed for {symbol}: {str(e)[:80]}")
        return None
    ctype, headline, tags, track = "unknown", "", "", ""
    body_lines = []
    for line in text.strip().splitlines():
        s = line.strip()
        if s.upper().startswith("TYPE="):
            cand = s[5:].strip().lower()
            ctype = cand if cand in CATALYST_TYPES else "unknown"
        elif s.upper().startswith("HEADLINE="):
            headline = s[9:].strip()[:160]
        elif s.upper().startswith("TAGS="):
            tags = s[5:].strip()[:120]
        elif s.upper().startswith("WHAT_TO_TRACK="):
            track = s[14:].strip()[:400]
        else:
            body_lines.append(line)
    if not headline:
        headline = (items[0]["title"][:160] if items else "")
    return ctype, headline, tags, track, "\n".join(body_lines).strip()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=str, default=None,
                    help="Ad-hoc comma list — bypasses eligibility + cap.")
    ap.add_argument("--limit", type=int, default=30,
                    help="Max Gemini notes per run (rotation is stalest-first).")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--local-dir", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="List the eligible/selected names. No network, no writes.")
    ap.add_argument("--email", action="store_true",
                    help="Mail tonight's notes digest. Respects the 'catalyst' "
                         "toggle in mail_settings.json.")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    local_dir = Path(args.local_dir) if args.local_dir else \
        Path(__file__).resolve().parent.parent / ".t4_local"
    store = Store(args.local, local_dir)
    today = datetime.now()
    as_of = today.strftime("%Y-%m-%d")
    log(f"build_catalyst_notes — mode={'LOCAL' if args.local else 'DRIVE'} "
        f"{'(dry-run)' if args.dry_run else ''}")

    # ---- eligibility: n_strategies >= 2 from aggregated signals ----
    sig = store.read_csv(["signals", "aggregated", "latest.csv"])
    if sig is None or sig.empty or "symbol" not in sig.columns:
        log("signals/aggregated/latest.csv absent — nothing to do.")
        return
    sig = sig.copy()
    sig["symbol"] = sig["symbol"].astype(str).str.upper()
    sig["n_strategies"] = pd.to_numeric(sig.get("n_strategies"), errors="coerce")
    eligible = (sig[sig["n_strategies"] >= MIN_STRATEGIES]
                .sort_values("n_strategies", ascending=False))
    elig_syms = list(dict.fromkeys(eligible["symbol"].tolist()))

    # ---- universe lookup (symbol -> isin, name, bse_code) ----
    cu = store.read_csv(["company_repo", "_index", "company_universe.csv"])
    isin_map = {}
    bse_map: dict[str, str] = {}
    alias_map: dict[str, str] = {}   # NSE symbol -> BSE ticker (media often uses it)
    if cu is not None and not cu.empty:
        sc = "nse_symbol" if "nse_symbol" in cu.columns else "symbol"
        for _, r in cu.iterrows():
            s = str(r.get(sc, "")).strip().upper()
            if s:
                isin_map[s] = (str(r.get("isin", "")).strip(),
                               str(r.get("name", "")).strip())
                bse_map[s] = str(r.get("bse_code", "")).strip()
                alias_map[s] = str(r.get("bse_symbol", "")).strip()

    # ---- user research intake (third evidence source) ----
    ridx = store.read_parquet(["company_repo", "_index", "research_index.parquet"])

    # ---- previous index: idempotency + stalest-first rotation ----
    idx = store.read_parquet(["company_repo", "_index", "catalyst_index.parquet"])
    last_note: dict[str, str] = {}
    prev_row: dict[str, dict] = {}       # newest note BEFORE today, per symbol
    if idx is not None and not idx.empty:
        for _, r in idx.sort_values("as_of").iterrows():
            last_note[str(r["symbol"]).upper()] = str(r["as_of"])
            if str(r["as_of"]) < as_of:
                prev_row[str(r["symbol"]).upper()] = r.to_dict()
    recent: dict[str, list] = {}         # same-story window, per symbol
    _since = (today - timedelta(days=SAME_STORY_DAYS)).strftime("%Y-%m-%d")
    if idx is not None and not idx.empty:
        for _, r in idx.iterrows():
            if _since <= str(r["as_of"]) < as_of:
                recent.setdefault(str(r["symbol"]).upper(), []).append(
                    (str(r.get("catalyst_type") or ""), str(r.get("headline") or "")))

    # ---- PF companies: scanned EVERY run on top of the cap (user 2026-06-11) ----
    pf_syms: list[str] = []
    if not args.local:
        try:
            pf_isins = load_portfolio_isins(store.drive, store.root) or set()
            sym_by_isin = {v[0]: k for k, v in isin_map.items() if v[0]}
            pf_syms = sorted(sym_by_isin[i] for i in pf_isins if i in sym_by_isin)
        except Exception as e:
            log(f"portfolio load failed ({str(e)[:60]}) — PF tier skipped")

    if args.names:
        selected = [s.strip().upper() for s in args.names.split(",") if s.strip()]
    else:
        pf_todo = [s for s in pf_syms if last_note.get(s, "") != as_of]
        todo = [s for s in elig_syms
                if last_note.get(s, "") != as_of and s not in set(pf_todo)]
        todo.sort(key=lambda s: last_note.get(s, ""))      # never-noted first
        selected = pf_todo + todo[:max(0, args.limit)]
    log(f"eligible (n_strategies>={MIN_STRATEGIES}): {len(elig_syms)}; "
        f"PF daily: {len(pf_syms)}; selected this run: {len(selected)}")

    if args.dry_run:
        log("DRY-RUN — would generate notes for:")
        for s in selected:
            isin, cname = isin_map.get(s, ("", ""))
            log(f"  {s:12s} isin={isin or '?':14s} last_note={last_note.get(s, 'never')}")
        return
    if not selected:
        log("Nothing to do.")
        return

    # Local mode has no Drive handle; resolve() then returns the declared chain,
    # which is exactly the intended fail-safe.
    pool = _build_gemini_pool(
        None if store.local else store.drive,
        "" if store.local else store._folder(["company_repo", "_index"]))
    if pool is None:
        return

    new_rows, skipped_quiet, skipped_unchanged = [], 0, 0
    for sym in selected:
        isin, cname = isin_map.get(sym, ("", ""))
        if not isin:
            log(f"  {sym}: no ISIN in universe — skipped")
            continue
        try:
            items = news_fetch.fetch_news(f'"{cname or sym}"', days_back=NEWS_DAYS)
        except news_fetch.NewsFetchBudgetExceeded:
            log("  news RSS budget hit — stopping note generation for this run")
            break
        # Relevance double-check (user 2026-07-09): Google News matches article
        # BODIES, so passing-mention pieces (MF holding lists, sector round-ups)
        # slip through the quoted-name search. Keep only headlines that actually
        # name the company (distinctive name token or NSE symbol).
        n_raw = len(items)
        items = news_fetch.relevant_items(items, cname, sym,
                                          aliases=(alias_map.get(sym, ""),))
        if n_raw and len(items) < n_raw:
            log(f"  {sym}: relevance filter kept {len(items)}/{n_raw} headlines")
        # Exchange filings (user 2026-06-12): often the catalyst itself.
        filings = _recent_filings(bse_map.get(sym, ""), NEWS_DAYS)
        research = _research_notes(ridx, isin)
        # Community coverage: VP top contributors / blogs / X (user 2026-06-12).
        import social_sources
        comm = social_sources.community_items(cname, NEWS_DAYS, blog_days=21)
        community = [f"- [{c['source']} · {c['author']}] {c['date']}: "
                     f"{c['text'][:250]} ({c['url']})" for c in comm]
        if not items and not filings and not research and not community:
            skipped_quiet += 1
            continue           # nothing moving anywhere -> save the Gemini call
        ev = evidence_hash(items[:10], filings, research, community)
        prev = prev_row.get(sym)
        if ev and prev and str(prev.get("evidence_hash") or "") == ev:
            skipped_unchanged += 1
            log(f"  {sym}: same evidence as the {prev.get('as_of')} note — no call")
            continue
        brief = store.read_text(["company_repo", isin, "company_page.md"]) or ""
        res = make_note(pool, cname, sym, items[:10], filings, research,
                        community, brief[-6000:])
        if res is None:
            continue
        ctype, headline, tags, track, body = res
        md_name = f"company_catalyst_{today.strftime('%d%b%y')}.md"
        md_path = f"company_repo/{isin}/{md_name}"
        md = (f"# Catalyst note — {cname or sym} ({sym})\n\n"
              f"*As of {as_of} · type: **{ctype}** · tags: {tags or '-'}*\n\n"
              f"**{headline}**\n\n"
              + (f"**👁 What to track:** {track}\n\n" if track else "")
              + f"{body}\n\n---\n"
              f"### Sources (last {NEWS_DAYS} days)\n"
              + "\n".join(f"- [{i['source']}] {i['title']}" for i in items[:10])
              + (("\n**BSE filings:**\n" + "\n".join(filings)) if filings else "")
              + (("\n**Community:**\n" + "\n".join(community)) if community else "")
              + f"\n\n*Generated {datetime.now().isoformat(timespec='seconds')}*\n")
        store.write_text(["company_repo", isin, md_name], md)
        new_rows.append({
            "isin": isin, "symbol": sym, "as_of": as_of,
            "headline": headline, "catalyst_type": ctype, "tags": tags,
            "what_to_track": track,
            # Inline note body (headline + what-to-track + narrative) so the app can
            # diff today vs the prior as_of cheaply — no per-render md fetch.
            "note_text": (f"{headline}\n\n"
                          + (f"What to track: {track}\n\n" if track else "")
                          + (body or "")).strip(),
            "md_path": md_path, "n_sources": len(items) + len(filings),
            "computed_at": datetime.now().isoformat(timespec="seconds"),
            "evidence_hash": ev,
            # mail-only (dropped by CATALYST_COLS before any write)
            "_changed": (note_changed(prev, ctype, headline)
                         and not same_story(recent.get(sym, []), ctype, headline)),
            "_prev_headline": (prev or {}).get("headline"),
            "_prev_as_of": (prev or {}).get("as_of"),
            "_in_pf": sym in set(pf_syms),
        })
        log(f"  {sym}: note written ({ctype}) — {headline[:70]}")

    news = actionable(new_rows)
    counts = {"selected": len(selected), "quiet": skipped_quiet,
              "unchanged": skipped_unchanged,
              "no_catalyst": sum(1 for r in new_rows
                                 if str(r.get("catalyst_type")) == "unknown"),
              "same_note": sum(1 for r in new_rows
                               if str(r.get("catalyst_type")) != "unknown"
                               and not r.get("_changed"))}
    log(f"notes written: {len(new_rows)} ({len(news)} new catalysts); quiet names "
        f"skipped (no news, no filings): {skipped_quiet}; same evidence as last note "
        f"(no call): {skipped_unchanged}; RSS calls: {news_fetch.calls_made()}")

    if new_rows:
        # ---- upsert index: replace same (symbol, as_of) rows, append the rest ----
        new_df = pd.DataFrame(new_rows, columns=CATALYST_COLS)
        if idx is not None and not idx.empty:
            keep = idx[~((idx["symbol"].astype(str).str.upper().isin(new_df["symbol"]))
                         & (idx["as_of"] == as_of))]
            out = pd.concat([keep, new_df], ignore_index=True)
        else:
            out = new_df
        store.write_df(["company_repo", "_index", "catalyst_index.parquet"], out)
        store.write_df(["company_repo", "_index", "catalyst_index.csv"], out)
        log(f"catalyst_index updated: {len(out)} total rows.")

    if args.email:
        if args.local or store.drive is None:
            log("--email: local mode, no Drive/toggles — mail skipped.")
            return
        from mailer import send_email, load_mail_settings
        index_id = store._folder(["company_repo", "_index"])
        if not load_mail_settings(store.drive, index_id).get("catalyst", True):
            log("catalyst mail toggled OFF — skipped.")
            return
        if not news:
            log("no new catalyst tonight — no mail (counts: "
                + ", ".join(f"{k}={v}" for k, v in counts.items()) + ")")
            return
        n_pf = sum(1 for r in news if r.get("_in_pf"))
        send_email(f"💡 Catalysts — {len(news)} new"
                   + (f" · 💼 {n_pf} PF" if n_pf else "")
                   + f" — {datetime.now():%d-%b}",
                   catalyst_mail_html(news, counts))


def _self_test() -> int:
    """Offline: evidence fingerprint, change detection, mail filtering. No network."""
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    a = [{"title": "X wins Rs 500 cr order", "link": "u1"}, {"title": "X Q2 results"}]
    b = [{"title": "x q2 results", "link": "other"}, {"title": "X wins Rs 500 cr order"}]
    h = evidence_hash(a, ["- filing 1"], [], [])
    check("hash ignores order, case and links", h == evidence_hash(b, ["- filing 1"], [], []))
    check("new filing changes the hash", h != evidence_hash(a, ["- filing 1", "- f2"], [], []))
    check("no evidence -> empty hash", evidence_hash([], [], [], []) == "")
    check("12 chars", len(h) == 12)
    prev = {"catalyst_type": "order_win", "headline": "X wins Rs 500 cr order"}
    check("first note is news", note_changed(None, "order_win", "anything"))
    check("same type + same headline (case/space) is not news",
          not note_changed(prev, "order_win", "x wins  rs 500 cr ORDER"))
    check("new headline is news", note_changed(prev, "order_win", "X wins Rs 900 cr order"))
    check("new type is news", note_changed(prev, "results", "X wins Rs 500 cr order"))
    welcorp = [("order_win", "Welspun Corp secures US$412.5 million order, order book US$4.7 billion")]
    check("reworded same order = same story", same_story(
        welcorp, "order_win", "Welspun Corp secured record USD 412.5 mn HFIW pipe order"))
    check("different amount = new story", not same_story(
        welcorp, "order_win", "Associate East Pipes wins SAR 771 million Aramco contract"))
    check("same amount, different type = new story", not same_story(
        welcorp, "results", "Revenue US$412.5 million"))
    check("dates and FY labels are not amounts",
          amounts("Board meets on 29 Sept 2026 for Q2 FY27 results") == set())
    check("no amount in the new headline = never same story",
          not same_story(welcorp, "order_win", "Syrma inaugurates new plant in Jodhpur"))
    check("amount forms", amounts("Rs 2,026 Cr; 24.2% YoY; 122,662 tons; ₹8.83 crore")
          == {"2026", "24.2", "122662", "8.83"})
    rows = [{"symbol": "A", "catalyst_type": "unknown", "_changed": True},
            {"symbol": "B", "catalyst_type": "order_win", "_changed": False},
            {"symbol": "C", "catalyst_type": "results", "_changed": True, "_in_pf": True,
             "headline": "C beats", "what_to_track": "margin in Q3"},
            {"symbol": "D", "catalyst_type": "policy", "_changed": True,
             "headline": "D gets PLI", "_prev_headline": "D applies", "_prev_as_of": "2026-10-01"}]
    news = actionable(rows)
    check("unknown and unchanged notes are dropped", [r["symbol"] for r in news] == ["C", "D"])
    html = catalyst_mail_html(news, {"selected": 30, "quiet": 12, "unchanged": 9,
                                     "no_catalyst": 1, "same_note": 1})
    check("PF first", html.index(">C<") < html.index(">D<"))
    check("previous note shown", "D applies" in html and "first note" in html)
    check("counts line", "9 had the same news" in html)
    check("mail-only keys never persisted",
          not any(c.startswith("_") for c in CATALYST_COLS)
          and CATALYST_COLS[-1] == "evidence_hash")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys as _sys
    if "--self-test" in _sys.argv:
        raise SystemExit(_self_test())
    main()
