r"""
ingest_announcements.py — INDEPENDENT BSE corporate-announcement pipeline.

Distinct from Phase 2 / backfill (which own concall/AR/results/rating/presentation
via the global processing_queue). This handles EVERY OTHER announcement: pulls the
day's filings market-wide, keeps the ones for our watchlist (PF first, then
n_strategies>=2), downloads the actual PDF, LLM-summarises it (lite model), and
writes a daily digest + per-company section. Its own ledger — NOT the global queue
(user decision 2026-06-17: avoid coupling complexity).

Excludes doc types already covered elsewhere: annual_report, concall/transcript,
credit rating, investor presentation (+ audio/video). Every kept filing is
categorised + flagged so categories can be toggled off later.

Outputs:
  company_repo/_index/announcement_ledger.parquet   (dedup + audit, own ledger)
  company_repo/_daily/daily_update_summary_<DDMMMYY>.md   (daily digest)
  company_repo/<isin>/company_page.md   (append an "Announcements" section)
  PDFs: company_repo/<isin>/documents/  (2-day retention, golden rule)

Usage:
  python scripts/ingest_announcements.py --dry-run          # discovery + counts, no writes/Gemini
  python scripts/ingest_announcements.py --limit 150        # live, capped
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log,
                             load_portfolio_isins, append_company_page,
                             salvage_json_objects, clamp, sstr)  # Stage 3 event tags
from gemini_pool import (BucketPool, load_keys, AllBucketsExhausted,
                         FatalCallError)

SUMMARY_PROMPT = (
    "You are an equity analyst. This is a BSE corporate filing. In 3-5 crisp "
    "lines for a portfolio manager: (1) what is this filing, (2) why it matters / "
    "the catalyst, (3) one concrete thing to track next. No preamble, no "
    "boilerplate. If it is routine/immaterial, say so in one line.\n"
    "Then on a FINAL separate line output ONLY a compact JSON object (no code "
    "fences, no other text) classifying the filing:\n"
    '{"event_type":"<one of: results|order_win|capex|mna|fundraise|debt|rating|'
    'litigation|management_change|buyback|dividend|expansion|regulatory|other>",'
    '"materiality":"high|med|low","direction":"bull|bear|neutral"}')

# Stage 3 event-tag vocab (LLM-derived, ADDITIVE to the BSE category/subcategory).
_EVENT_TYPES = {"results", "order_win", "capex", "mna", "fundraise", "debt", "rating",
                "litigation", "management_change", "buyback", "dividend", "expansion",
                "regulatory", "other"}
_MATERIALITY = {"high", "med", "low"}
_DIRECTION = {"bull", "bear", "neutral"}


def _parse_event_tags(resp: str) -> tuple[dict, str]:
    """Return (tags, clean_summary). Pull the JSON tail (event_type/materiality/
    direction) via the shared salvage helper; strip it from the stored summary text.
    Defaults are safe ('other'/'low'/'neutral') so a missing/garbled tail never breaks
    the existing summary path."""
    tags = {"event_type": "other", "materiality": "low", "direction": "neutral"}
    summary = resp or ""
    for o in salvage_json_objects(resp):
        if "event_type" in o or "direction" in o:
            tags = {
                "event_type": clamp(o.get("event_type"), _EVENT_TYPES, "other"),
                "materiality": clamp(o.get("materiality"), _MATERIALITY, "low"),
                "direction": clamp(o.get("direction"), _DIRECTION, "neutral"),
            }
            break
    # strip any {...} json object(s) from the human summary
    summary = re.sub(r"\{[^{}]*\}", "", summary).strip()
    return tags, summary
DAILY_KEEP_DAYS = 35      # keep ~1 month+ of daily_update_summary_*.md history

ANN_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
ATTACH_HOSTS = ("https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
                "https://www.bseindia.com/xml-data/corpfiling/AttachHis/")
API_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
           "Accept": "application/json, text/plain, */*",
           "Referer": "https://www.bseindia.com/corporates/ann.html",
           "Origin": "https://www.bseindia.com"}

# Phase-2 / backfill already own these — match on sub-category / category / flags
# and DROP so we never duplicate that work.
EXCLUDE_PAT = re.compile(
    r"transcript|annual report|investor presentation|presentation|"
    r"credit rating|rating|earnings call|con(ference)? ?call|audio|video",
    re.I)

# High-confidence ROUTINE noise (pure logistics, no catalyst value) — dropped so the
# per-company sweep doesn't flood the summariser/banner. Deliberately conservative: keeps
# postal-ballot NOTICES (where things like "increase in borrowing power" live), orders,
# board outcomes, fundraising, results, resignations, M&A, Reg-30 substantive disclosures.
ROUTINE_DROP = re.compile(
    r"(?i)(trading window|newspaper (advertisement|publication|clipping|ad\b)|"
    r"scrutin|loss of (share|certificate)|duplicate (share|certificate)|"
    r"issue of duplicate|share certificate|compliance certificate|"
    r"certificate under (regulation|reg)|reg(ulation)?\.? *74\b|book closure|"
    r"sub-?division|split of|dividend (distribution )?tax|record date for)")

LEDGER_COLS = ["newsid", "isin", "symbol", "scrip_cd", "ann_date", "category",
               "subcategory", "flag", "headline", "attachment", "pdf_sha",
               "summary", "status", "discovered_at", "processed_at",
               # Stage 3: LLM event tags (ADDITIVE; old rows read NaN).
               "event_type", "materiality", "direction"]


def _folder(drive, parts: str) -> str:
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _read_csv(drive, folder_id, name) -> pd.DataFrame:
    fid = find_file(drive, folder_id, name)
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid))) if fid else pd.DataFrame()


def _read_parquet(drive, folder_id, name) -> pd.DataFrame:
    fid = find_file(drive, folder_id, name)
    return pd.read_parquet(io.BytesIO(download_bytes(drive, fid))) if fid else pd.DataFrame()


# ---------------------------------------------------------------- watchlist

def _turnover_map(drive) -> dict:
    """symbol -> avg 20d ₹-turnover in cr, mirroring app.py's liquidity floor
    (prefers avg_turnover_20d_cr; falls back to volume*close/1e7)."""
    feats = _read_parquet(drive, _folder(drive, "features"), "latest.parquet")
    if feats.empty or "symbol" not in feats.columns:
        return {}
    if "avg_turnover_20d_cr" in feats.columns:
        turn = pd.to_numeric(feats["avg_turnover_20d_cr"], errors="coerce")
    elif {"volume", "close"}.issubset(feats.columns):
        turn = (pd.to_numeric(feats["volume"], errors="coerce")
                * pd.to_numeric(feats["close"], errors="coerce")) / 1e7
    else:
        return {}
    return dict(zip(feats["symbol"].astype(str).str.upper(), turn))


def build_watchlist(drive, min_turnover_cr: float = 1.0):
    """(order, meta) — PF first, then n_strategies>=2 from the latest Phase-1
    output, with a ₹-turnover liquidity floor on the conviction tier (matches
    app.py's default min_turnover_cr=1.0). PF is kept regardless of liquidity.
    Returns ordered list of bse_code and maps for isin/symbol/name."""
    idx = _folder(drive, "company_repo/_index")
    uni = _read_csv(drive, idx, "company_universe.csv").fillna("")
    # bse_code -> (isin, symbol, name)
    meta = {}
    for _, r in uni.iterrows():
        code = str(r.get("bse_code", "")).strip().replace(".0", "")
        if not code or code.lower() == "nan":
            continue
        sym = str(r.get("nse_symbol") or r.get("bse_symbol") or "").strip().upper()
        meta[code] = (str(r.get("isin", "")).strip(), sym, str(r.get("name", "")).strip())
    isin_to_code = {v[0]: k for k, v in meta.items() if v[0]}
    sym_to_code = {v[1]: k for k, v in meta.items() if v[1]}

    order, seen = [], set()
    # PF first
    try:
        for isin in (load_portfolio_isins(drive, os.environ["GDRIVE_FOLDER_ID"]) or []):
            c = isin_to_code.get(str(isin).strip())
            if c and c not in seen:
                order.append(c); seen.add(c)
    except Exception as e:
        log(f"  PF load failed ({str(e)[:60]}) — PF tier skipped")
    n_pf = len(order)
    # then n_strategies >= 2 (latest aggregated signals) WITH the ₹-turnover floor
    turn = _turnover_map(drive) if min_turnover_cr > 0 else {}
    sig = _read_csv(drive, _folder(drive, "signals/aggregated"), "latest.csv")
    dropped_illiquid = 0
    if not sig.empty and "symbol" in sig.columns:
        sig["n"] = pd.to_numeric(sig.get("n_strategies"), errors="coerce")
        conv = (sig[sig["n"] >= 2].sort_values("n", ascending=False)["symbol"]
                .astype(str).str.upper())
        for s in conv:
            c = sym_to_code.get(s)
            if not c or c in seen:
                continue
            if min_turnover_cr > 0 and turn.get(s, -1.0) < min_turnover_cr:
                dropped_illiquid += 1
                continue
            order.append(c); seen.add(c)
    log(f"watchlist: {len(order)} companies (PF {n_pf} + conviction "
        f"{len(order)-n_pf}; dropped {dropped_illiquid} illiquid "
        f"<Rs{min_turnover_cr:.0f}cr/day)")
    return order, meta


# ---------------------------------------------------------------- discovery

def fetch_day_announcements(d: date, max_pages: int = 30) -> list[dict]:
    """ALL market-wide filings for day d (strScrip='' + date), paginated."""
    out, s = [], requests.Session()
    s.headers.update(API_HDR)
    ymd = d.strftime("%Y%m%d")
    for pageno in range(1, max_pages + 1):
        params = {"pageno": pageno, "strCat": "-1", "subcategory": "-1",
                  "strPrevDate": ymd, "strToDate": ymd, "strSearch": "P",
                  "strScrip": "", "strType": "C"}
        try:
            r = s.get(ANN_API, params=params, timeout=40)
            rows = r.json().get("Table", []) if r.status_code == 200 else []
        except Exception as e:
            log(f"  page {pageno} fetch failed: {str(e)[:60]}"); break
        if not rows:
            break
        out += rows
        if len(rows) < 50:
            break
        time.sleep(0.3)
    return out


def fetch_company_announcements(code: str, lookback_days: int,
                                max_pages: int = 6) -> list[dict]:
    """PER-COMPANY BSE filings (strScrip=code) over the last lookback_days. Guarantees a
    watchlist company's filings are seen regardless of the market-wide page cap — which was
    dropping PF filings on heavy days (the Kernex borrowing-power miss)."""
    out, s = [], requests.Session()
    s.headers.update(API_HDR)
    to_d = date.today()
    from_d = to_d - timedelta(days=max(0, lookback_days - 1))
    for pageno in range(1, max_pages + 1):
        params = {"pageno": pageno, "strCat": "-1", "subcategory": "-1",
                  "strPrevDate": from_d.strftime("%Y%m%d"),
                  "strToDate": to_d.strftime("%Y%m%d"),
                  "strSearch": "P", "strScrip": str(code), "strType": "C"}
        try:
            r = s.get(ANN_API, params=params, timeout=30)
            rows = r.json().get("Table", []) if r.status_code == 200 else []
        except Exception:
            break
        if not rows:
            break
        for rr in rows:
            rr.setdefault("SCRIP_CD", str(code))     # ensure the watchlist filter matches
        out += rows
        if len(rows) < 50:
            break
        time.sleep(0.2)
    return out


def percompany_scan_codes(drive, meta: dict, top_n: int) -> list[str]:
    """bse_codes to fetch PER-COMPANY (guaranteed coverage): PF first, then Top-N by market
    cap. These bypass the market-wide page cap so their filings are never dropped."""
    isin_to_code = {v[0]: k for k, v in meta.items() if v[0]}
    sym_to_code = {v[1]: k for k, v in meta.items() if v[1]}
    codes, seen = [], set()
    try:
        for isin in (load_portfolio_isins(drive, os.environ["GDRIVE_FOLDER_ID"]) or []):
            c = isin_to_code.get(str(isin).strip())
            if c and c not in seen:
                codes.append(c); seen.add(c)
    except Exception as e:
        log(f"  PF load failed for per-company list ({str(e)[:50]})")
    n_pf = len(codes)
    if top_n > 0:
        mc = _read_csv(drive, _folder(drive, "universe"), "market_cap.csv")
        if not mc.empty and {"symbol", "market_cap_cr"} <= set(mc.columns):
            mc["_mc"] = pd.to_numeric(mc["market_cap_cr"], errors="coerce")
            for s in (mc.sort_values("_mc", ascending=False)["symbol"]
                      .astype(str).str.upper().head(top_n)):
                c = sym_to_code.get(s)
                if c and c not in seen:
                    codes.append(c); seen.add(c)
        else:
            log("  universe/market_cap.csv missing — top-N mcap tier skipped")
    log(f"per-company scan list: {len(codes)} codes (PF {n_pf} + top-{top_n} mcap)")
    return codes


def _download_pdf(attachment: str) -> bytes | None:
    """Fetch the announcement PDF from BSE (Live then His host). None on failure."""
    for base in ATTACH_HOSTS:
        try:
            r = requests.get(base + attachment,
                             headers={"User-Agent": API_HDR["User-Agent"],
                                      "Referer": "https://www.bseindia.com/"},
                             timeout=60)
            if r.status_code == 200 and r.content[:5].startswith(b"%PDF"):
                return r.content
        except Exception:
            continue
    return None


def _build_pool():
    """Lite-model cascade. FREE_POOL is the default pool for new programs
    (user 2026-06-17); BACKFILL -> GEMINI kept as fallback."""
    keys = load_keys(os.environ, prefix="FREE_POOL")
    for pref in ("BACKFILL_GEMINI_KEY", "GEMINI_API_KEY"):
        keys += [k for k in load_keys(os.environ, prefix=pref) if k not in keys]
    if not keys:
        log("No Gemini keys (FREE_POOL/BACKFILL/GEMINI) — cannot summarise.")
        return None
    # gemini-2.5-flash-lite is the live free model; 2.0-flash-lite's free quota is
    # exhausted (429 PerDay) and was making the pool give up early — lead with 2.5
    # and use 2.0-flash (non-lite) as the fallback, not 2.0-flash-lite.
    log(f"Gemini pool: {len(keys)} keys (FREE_POOL then BACKFILL/GEMINI)")
    return BucketPool(keys, ["gemini-2.5-flash-lite", "gemini-2.0-flash"],
                      inter_call_s=0.5, logger=log, overload_budget=3)


def categorise(row: dict) -> tuple[str, str, bool]:
    """(category, subcategory, excluded?) — excluded = already covered by Phase 2."""
    cat = (row.get("CATEGORYNAME") or "").strip()
    sub = (row.get("SUBCATNAME") or "").strip()
    inv_pres = str(row.get("Investor_Presentation") or "").strip()
    av = str(row.get("AUDIO_VIDEO_FILE") or "").strip()
    blob = f"{cat} {sub}"
    excluded = bool(EXCLUDE_PAT.search(blob)) or bool(inv_pres) or bool(av)
    return cat or "Uncategorised", sub, excluded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: today).")
    ap.add_argument("--lookback-days", type=int, default=1,
                    help="Scan the last N days ending at --date (default 1 = today "
                         "only). >1 catches filings on days a run was skipped/late or "
                         "over a weekend; dedup by newsid prevents repeats.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional safety cap on NEW summaries (0 = no cap; dedup "
                         "already bounds it to the day's actual new filings).")
    ap.add_argument("--top-n", type=int, default=300,
                    help="Also fetch PER-COMPANY (strScrip) for PF + top-N by market cap, so "
                         "their filings are never dropped by the market-wide page cap (the "
                         "Kernex miss). 0 = market-wide only.")
    ap.add_argument("--min-turnover", type=float, default=1.0,
                    help="₹-turnover floor (cr/day, 20d) on the conviction tier — "
                         "matches app.py default (1.0). PF kept regardless. 0 = off.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Discovery + filter + categorise + counts. No PDF/Gemini/writes.")
    args = ap.parse_args()

    drive = get_drive()
    d = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()
    lookback = max(1, args.lookback_days)
    days = [d - timedelta(days=i) for i in range(lookback)]
    log(f"BSE announcements for {days[-1]}..{d} ({lookback}d)  "
        f"mode={'DRY-RUN' if args.dry_run else 'LIVE'}")

    order, meta = build_watchlist(drive, min_turnover_cr=args.min_turnover)
    wl_codes = set(order)

    rows = []
    for dd in days:
        day_rows = fetch_day_announcements(dd)
        log(f"market-wide filings on {dd}: {len(day_rows)}")
        rows.extend(day_rows)

    # A (2026-07-01): PER-COMPANY guaranteed fetch for PF + top-N by market cap (strScrip),
    # so their filings are never dropped by the market-wide page cap. Their codes join the
    # watchlist filter below; dedup-by-newsid removes overlap with the market-wide rows.
    pc_codes = percompany_scan_codes(drive, meta, args.top_n)
    if pc_codes:
        wl_codes |= set(pc_codes)
        got = 0
        for c in pc_codes:
            cr = fetch_company_announcements(c, lookback)
            rows.extend(cr); got += len(cr)
        log(f"per-company fetch: {len(pc_codes)} companies -> {got} filing-rows")

    # filter to watchlist + categorise + drop Phase-2-covered types
    kept, excluded_n, offwl_n, routine_n = [], 0, 0, 0
    for r in rows:
        code = str(r.get("SCRIP_CD", "")).strip()
        if code not in wl_codes:
            offwl_n += 1; continue
        cat, sub, excluded = categorise(r)
        if excluded:
            excluded_n += 1; continue
        _subj = (r.get("NEWSSUB") or r.get("HEADLINE") or "").strip()
        if ROUTINE_DROP.search(f"{cat} {sub} {_subj}"):
            routine_n += 1; continue                 # pure logistics — no catalyst value
        isin, sym, name = meta.get(code, ("", "", ""))
        kept.append({"newsid": str(r.get("NEWSID", "")), "scrip_cd": code,
                     "isin": isin, "symbol": sym, "name": name,
                     "ann_date": str(r.get("NEWS_DT", ""))[:10],
                     "category": cat, "subcategory": sub,
                     "flag": "critical" if str(r.get("CRITICALNEWS","")) in ("1","True") else "",
                     "headline": (r.get("NEWSSUB") or r.get("HEADLINE") or "").strip(),
                     "attachment": (r.get("ATTACHMENTNAME") or "").strip()})

    # dedup against our OWN ledger (not the global queue)
    idx = _folder(drive, "company_repo/_index")
    ledger = _read_parquet(drive, idx, "announcement_ledger.parquet")
    known = set(ledger["newsid"].astype(str)) if not ledger.empty and "newsid" in ledger else set()
    new = [k for k in kept if k["newsid"] not in known]

    log("-" * 60)
    log(f"on watchlist: {len(kept)} | off-watchlist dropped: {offwl_n} | "
        f"excluded (AR/concall/rating/pres/audio): {excluded_n} | "
        f"routine-noise dropped: {routine_n}")
    log(f"NEW (not in ledger): {len(new)} | already in ledger: {len(kept)-len(new)}")
    log(f"by category (NEW): {dict(Counter(k['category'] for k in new).most_common())}")
    log(f"critical-flagged (NEW): {sum(1 for k in new if k['flag']=='critical')}")
    log(f"with PDF attachment (NEW): {sum(1 for k in new if k['attachment'])}")

    # order: PF & conviction rank (watchlist order) with critical first within ties
    order_rank = {c: i for i, c in enumerate(order)}
    new.sort(key=lambda k: (0 if k["flag"] == "critical" else 1,
                            order_rank.get(k["scrip_cd"], 1 << 30)))

    if args.dry_run:
        log("\nWould SUMMARISE (first 20 new, critical->PF->conviction):")
        for k in new[:20]:
            log(f"  {k['symbol'][:12]:<12} {k['ann_date']} [{k['category'][:16]:<16}]"
                f"{' *crit' if k['flag'] else '':<6} {k['headline'][:60]}")
        log("\n(DRY-RUN — no writes/Gemini.)")
        return

    new = [k for k in new if k["attachment"]]      # need a PDF to summarise
    if args.limit and args.limit > 0:
        new = new[:args.limit]
    if not new:
        log("No new PDF-bearing announcements — nothing to summarise.")
        return

    pool = _build_pool()
    if pool is None:
        return
    pool.probe_models()

    repo_id = _folder(drive, "company_repo")
    done_rows, fail = [], 0
    digest = []          # (symbol, category, headline, summary) for the daily md
    for i, k in enumerate(new, 1):
        pdf = _download_pdf(k["attachment"])
        if pdf is None:
            fail += 1; continue
        try:
            summary, _model = pool.call_pdf(pdf, SUMMARY_PROMPT)
        except AllBucketsExhausted:
            log("  Gemini buckets exhausted — stopping (remaining picked up next run).")
            break
        except FatalCallError as e:
            log(f"  {k['symbol']}: fatal call ({str(e)[:60]}) — skipped"); fail += 1; continue
        except Exception as e:
            log(f"  {k['symbol']}: call error ({str(e)[:60]}) — skipped"); fail += 1; continue
        # Stage 3: pull the LLM event tags from the response tail; clean_summary is
        # the human prose with the JSON object stripped out.
        _tags, summary = _parse_event_tags(summary or "")
        sha = hashlib.sha256(pdf).hexdigest()
        now = datetime.now().isoformat(timespec="seconds")
        done_rows.append({**{c: k.get(c, "") for c in
                             ("newsid", "isin", "symbol", "category", "subcategory",
                              "flag", "headline", "attachment")},
                          "scrip_cd": k["scrip_cd"], "ann_date": k["ann_date"],
                          "pdf_sha": sha, "summary": summary, "status": "done",
                          "discovered_at": now, "processed_at": now,
                          "event_type": _tags["event_type"],
                          "materiality": _tags["materiality"],
                          "direction": _tags["direction"]})
        digest.append((k["symbol"], k["category"], k["ann_date"],
                       k["headline"], summary))
        # store PDF for the 2-day retention sweep + append company_page section
        if k["isin"]:
            comp_id = get_or_create_subfolder(drive, repo_id, k["isin"])
            docs_id = get_or_create_subfolder(drive, comp_id, "documents")
            upload_bytes(drive, docs_id,
                         f"announcement__{k['ann_date']}__{k['newsid']}.pdf",
                         pdf, "application/pdf")
            append_company_page(
                drive, repo_id, k["isin"], "Announcement",
                f"**{k['headline']}**\n\n{summary}\n", k["headline"][:80], k["ann_date"])
        if i % 25 == 0:
            log(f"  [{i}/{len(new)}] summarised={len(done_rows)} fail={fail}")

    # ---- persist: ledger (append, dedup newsid) ----
    if done_rows:
        new_df = pd.DataFrame(done_rows, columns=LEDGER_COLS)
        out = (pd.concat([ledger, new_df], ignore_index=True)
               if not ledger.empty else new_df)
        out = out.drop_duplicates(subset=["newsid"], keep="last").reset_index(drop=True)
        upload_bytes(drive, idx, "announcement_ledger.parquet",
                     out.to_parquet(index=False), "application/octet-stream",
                     existing_id=find_file(drive, idx, "announcement_ledger.parquet"))
        log(f"announcement_ledger: +{len(done_rows)} -> {len(out)} rows")

        # ---- daily digest md (kept >=1 month) ----
        daily_id = _folder(drive, "company_repo/_daily")
        md = [f"# Daily announcement update — {d}",
              f"*{len(done_rows)} new filings summarised "
              f"(PF + n_strategies>=2 watchlist). Generated {datetime.now():%Y-%m-%d %H:%M}*\n"]
        for sym, cat, adate, head, summ in digest:
            md.append(f"\n## {sym} · {cat} · {adate}\n**{head}**\n\n{summ}\n")
        fn = f"daily_update_summary_{d:%d%b%y}.md"
        upload_bytes(drive, daily_id, fn, "\n".join(md).encode("utf-8"),
                     "text/markdown", existing_id=find_file(drive, daily_id, fn))
        log(f"wrote company_repo/_daily/{fn}")
        _prune_daily(drive, daily_id, d)

    log(f"DONE: summarised={len(done_rows)} fail={fail}")

    # ---- 2-day PDF retention (global golden rule) ----
    try:
        import subprocess
        subprocess.run([sys.executable, os.path.join(_SCRIPTS_DIR,
                        "cleanup_company_docs.py"), "--retain-days", "2"],
                       check=False, timeout=900)
    except Exception as e:
        log(f"  retention cleanup call failed ({str(e)[:60]}) — run cleanup_company_docs.py")


def _prune_daily(drive, daily_id, today: date) -> None:
    """Delete daily_update_summary_*.md older than DAILY_KEEP_DAYS (~1 month)."""
    cutoff = today - timedelta(days=DAILY_KEEP_DAYS)
    try:
        resp = drive.files().list(
            q=f"'{daily_id}' in parents and trashed=false",
            fields="files(id,name)", pageSize=1000).execute()
        for f in resp.get("files", []):
            m = re.match(r"daily_update_summary_(\d{2}\w{3}\d{2})\.md", f["name"])
            if not m:
                continue
            try:
                fdate = datetime.strptime(m.group(1), "%d%b%y").date()
            except ValueError:
                continue
            if fdate < cutoff:
                drive.files().delete(fileId=f["id"]).execute()
    except Exception as e:
        log(f"  daily-md prune skipped ({str(e)[:50]})")


if __name__ == "__main__":
    main()
