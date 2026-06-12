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
from datetime import datetime
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


PROMPT = """You are an equity research assistant covering Indian listed companies.
Based ONLY on the recent headlines and the optional company brief below, write a short
"why is it moving / what changed" catalyst note for {company} ({symbol}).

Reply in EXACTLY this format (3 lines of metadata, then markdown):
TYPE=<one of: order_win|mgmt_change|policy|sector|results|corporate_action|unknown>
HEADLINE=<one factual line, <=120 chars, no hype>
TAGS=<2-4 comma-separated lowercase tags>

Then 3-5 markdown bullets explaining the catalyst, citing which headline supports each
point. If the headlines do not show a clear catalyst, say so honestly (TYPE=unknown).

--- RECENT TRUSTED HEADLINES (last {days} days) ---
{headlines}

--- COMPANY BRIEF (may be empty) ---
{brief}
"""


def _esc(s, n=120) -> str:
    return html_mod.escape(str(s)[:n])


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
            f"<td>{_esc(r['headline'], 160)}</td><td>{_esc(r['tags'], 60)}</td>"
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
              brief: str) -> tuple[str, str, str, str] | None:
    """Returns (catalyst_type, headline, tags, md_body) or None on failure."""
    headlines = "\n".join(
        f"- [{i['source']}] {i['title']} ({i['published'][:16]})" for i in items)
    prompt = PROMPT.format(company=company or symbol, symbol=symbol,
                           days=NEWS_DAYS, headlines=headlines,
                           brief=(brief or "")[:6000])
    try:
        text, _ = pool.call_text(prompt)
    except Exception as e:
        log(f"  Gemini failed for {symbol}: {str(e)[:80]}")
        return None
    ctype, headline, tags = "unknown", "", ""
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
        else:
            body_lines.append(line)
    if not headline:
        headline = (items[0]["title"][:160] if items else "")
    return ctype, headline, tags, "\n".join(body_lines).strip()


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

    # ---- universe lookup (symbol -> isin, name) ----
    cu = store.read_csv(["company_repo", "_index", "company_universe.csv"])
    isin_map = {}
    if cu is not None and not cu.empty:
        sc = "nse_symbol" if "nse_symbol" in cu.columns else "symbol"
        for _, r in cu.iterrows():
            s = str(r.get(sc, "")).strip().upper()
            if s:
                isin_map[s] = (str(r.get("isin", "")).strip(),
                               str(r.get("name", "")).strip())

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
        if not items:
            skipped_quiet += 1
            continue                       # nothing moving -> save the Gemini call
        brief = store.read_text(["company_repo", isin, "company_page.md"]) or ""
        res = make_note(pool, cname, sym, items[:10], brief[-6000:])
        if res is None:
            continue
        ctype, headline, tags, body = res
        md_name = f"company_catalyst_{today.strftime('%d%b%y')}.md"
        md_path = f"company_repo/{isin}/{md_name}"
        md = (f"# Catalyst note — {cname or sym} ({sym})\n\n"
              f"*As of {as_of} · type: **{ctype}** · tags: {tags or '-'}*\n\n"
              f"**{headline}**\n\n{body}\n\n---\n"
              f"### Sources (trusted whitelist, last {NEWS_DAYS} days)\n"
              + "\n".join(f"- [{i['source']}] {i['title']}" for i in items[:10])
              + f"\n\n*Generated {datetime.now().isoformat(timespec='seconds')}*\n")
        store.write_text(["company_repo", isin, md_name], md)
        new_rows.append({
            "isin": isin, "symbol": sym, "as_of": as_of,
            "headline": headline, "catalyst_type": ctype, "tags": tags,
            "md_path": md_path, "n_sources": len(items),
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        })
        log(f"  {sym}: note written ({ctype}) — {headline[:70]}")

    log(f"notes written: {len(new_rows)}; quiet names skipped (no trusted news): "
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
