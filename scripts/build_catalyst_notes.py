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
    "md_path", "n_sources", "computed_at",
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

def _build_gemini_pool():
    """GEMINI_API_KEY -> BACKFILL_GEMINI_KEY cascade, lite models (rule #4)."""
    try:
        from gemini_pool import BucketPool, load_keys
        keys = load_keys(os.environ, prefix="GEMINI_API_KEY")
        keys += [k for k in load_keys(os.environ, prefix="BACKFILL_GEMINI_KEY")
                 if k not in keys]
        if not keys:
            log("no GEMINI keys found — cannot generate notes.")
            return None
        return BucketPool(keys, ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"],
                          inter_call_s=6.0, logger=log)
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


def _catalyst_mail_html(new_rows: list[dict], n_eligible: int, n_pf: int,
                        n_selected: int, skipped_quiet: int) -> str:
    parts = [f"<p><b>Catalyst notes — nightly digest.</b><br>"
             f"Eligible (≥{MIN_STRATEGIES} strategies): {n_eligible} · "
             f"PF daily: {n_pf} · scanned: {n_selected} · "
             f"quiet (no trusted news): {skipped_quiet} · "
             f"notes written: <b>{len(new_rows)}</b></p>"]
    if new_rows:
        rows = "".join(
            f"<tr><td><b>{_esc(r['symbol'])}</b></td><td>{_esc(r['catalyst_type'])}</td>"
            f"<td>{_esc(r['headline'], 160)}"
            + (f"<br><i style='color:#777'>👁 {_esc(r.get('what_to_track', ''), 200)}"
               f"</i>" if r.get("what_to_track") else "")
            + f"</td><td>{_esc(r['tags'], 60)}</td>"
            f"<td align=center>{r['n_sources']}</td></tr>" for r in new_rows)
        parts.append("<table border=1 cellpadding=4 cellspacing=0>"
                     "<tr><th>Symbol</th><th>Type</th><th>Headline</th>"
                     "<th>Tags</th><th>Src</th></tr>" + rows + "</table>")
    else:
        parts.append("<p>No catalysts found tonight — every scanned name was quiet.</p>")
    parts.append("<p style='font-size:11px;color:#999'>Full notes live on the Stock "
                 "Detail / Scorecard pages. Quiet = no trusted-source news in "
                 f"{NEWS_DAYS} days (no Gemini call spent). "
                 "Toggle this mail in the app sidebar (📧 Email toggles).</p>")
    return "".join(parts)


def make_note(pool, company: str, symbol: str, items: list[dict],
              filings: list[str], research: list[str],
              brief: str) -> tuple[str, str, str, str, str] | None:
    """Returns (catalyst_type, headline, tags, what_to_track, md_body) or None."""
    headlines = "\n".join(
        f"- [{i['source']}] {i['title']} ({i['published'][:16]})"
        for i in items) or "(none in window)"
    prompt = PROMPT.format(company=company or symbol, symbol=symbol,
                           days=NEWS_DAYS, headlines=headlines,
                           filings="\n".join(filings) or "(none in window)",
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
    if cu is not None and not cu.empty:
        sc = "nse_symbol" if "nse_symbol" in cu.columns else "symbol"
        for _, r in cu.iterrows():
            s = str(r.get(sc, "")).strip().upper()
            if s:
                isin_map[s] = (str(r.get("isin", "")).strip(),
                               str(r.get("name", "")).strip())
                bse_map[s] = str(r.get("bse_code", "")).strip()

    # ---- user research intake (third evidence source) ----
    ridx = store.read_parquet(["company_repo", "_index", "research_index.parquet"])

    # ---- previous index: idempotency + stalest-first rotation ----
    idx = store.read_parquet(["company_repo", "_index", "catalyst_index.parquet"])
    last_note: dict[str, str] = {}
    if idx is not None and not idx.empty:
        for _, r in idx.sort_values("as_of").iterrows():
            last_note[str(r["symbol"]).upper()] = str(r["as_of"])

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

    pool = _build_gemini_pool()
    if pool is None:
        return

    new_rows, skipped_quiet = [], 0
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
        # Exchange filings (user 2026-06-12): often the catalyst itself.
        filings = _recent_filings(bse_map.get(sym, ""), NEWS_DAYS)
        research = _research_notes(ridx, isin)
        if not items and not filings and not research:
            skipped_quiet += 1
            continue           # nothing moving anywhere -> save the Gemini call
        brief = store.read_text(["company_repo", isin, "company_page.md"]) or ""
        res = make_note(pool, cname, sym, items[:10], filings, research,
                        brief[-6000:])
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
              + f"\n\n*Generated {datetime.now().isoformat(timespec='seconds')}*\n")
        store.write_text(["company_repo", isin, md_name], md)
        new_rows.append({
            "isin": isin, "symbol": sym, "as_of": as_of,
            "headline": headline, "catalyst_type": ctype, "tags": tags,
            "what_to_track": track,
            "md_path": md_path, "n_sources": len(items) + len(filings),
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        })
        log(f"  {sym}: note written ({ctype}) — {headline[:70]}")

    log(f"notes written: {len(new_rows)}; quiet names skipped (no news, no filings): "
        f"{skipped_quiet}; RSS calls: {news_fetch.calls_made()}")

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
        send_email(f"💡 Catalyst notes — {len(new_rows)} new — {datetime.now():%d-%b}",
                   _catalyst_mail_html(new_rows, len(elig_syms), len(pf_syms),
                                       len(selected), skipped_quiet))


if __name__ == "__main__":
    main()
