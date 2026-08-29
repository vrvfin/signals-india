r"""
P5b — extract_quarterly_hist.py  (Project Guru, STANDALONE, RESUMABLE)

Extends quarterly fundamental history beyond Screener's 13-quarter cap by
extracting 2008-2023 quarterly numbers from the BSE result PDFs already indexed
in results_dates_hist.parquet (P2d). Reuses the repo's Phase-2 extraction
machinery READ-ONLY (results_prompt.txt + gemini_pool.BucketPool on FREE_POOL +
the markdown-table parse helpers); writes ONLY under guru/data/. Never touches
the Phase-2 queue, company_page.md, or any shared parquet.

SCOPE (user 2026-07-05): backtestable core only (keys with BOTH price+fundamentals
metrics), deduped to ONE filing per (guru_key, reported-quarter), RECENT-FIRST so
usable depth accrues incrementally. Deep years only (<= DEEP_MAX_YEAR); 2024+ is
already covered densely by Screener.

Output: guru/data/quarterly_ext_facts.parquet  (one row per company-quarter)
  guru_key, reported_quarter, announcement_date, revenue_cr, ebitda_cr, pat_cr,
  eps, ebitda_margin_pct, pat_margin_pct, revenue_yoy_pct, pat_yoy_pct,
  gemini_quarter, model, newsid
Ledger: guru/data/_dump_status/quarterly_ext_ledger.parquet (per NEWSID)

Usage:
    python guru/extract_quarterly_hist.py --dry-run       # plan only, no LLM
    python guru/extract_quarterly_hist.py --limit 15      # pilot (real LLM calls)
    python guru/extract_quarterly_hist.py                 # full, resumes
    python guru/extract_quarterly_hist.py --status
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
STATUS_DIR = os.path.join(DATA_DIR, "_dump_status")
OUT_PATH = os.path.join(DATA_DIR, "quarterly_ext_facts.parquet")
LEDGER_PATH = os.path.join(STATUS_DIR, "quarterly_ext_ledger.parquet")

SCRIPTS_DIR = os.path.join(os.path.dirname(GURU_DIR), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
load_dotenv(os.path.join(os.path.dirname(GURU_DIR), ".env"))

from gemini_pool import BucketPool, load_keys, AllBucketsExhausted, FatalCallError
from _extractor_base import extract_md_tables, clean_val, identify_metric, P1_MODELS

DEEP_MAX_YEAR = 2023        # 2024+ already dense from Screener
ATTACH_URL = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/{}"
PDF_HDR = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
           "Referer": "https://www.bseindia.com/"}
FLUSH_EVERY = 50
OUT_COLS = ["guru_key", "reported_quarter", "announcement_date", "revenue_cr",
            "ebitda_cr", "pat_cr", "eps", "ebitda_margin_pct", "pat_margin_pct",
            "revenue_yoy_pct", "pat_yoy_pct", "gemini_quarter", "model", "newsid"]

# real results filings only — exclude intimations/delays/newspaper/board-meeting
EXCLUDE_RE = re.compile(r"delay|intimation|newspaper|board meeting|analyst|investor|"
                        r"con-?call|transcript|schedule|press release|clarification",
                        re.I)
INCLUDE_RE = re.compile(r"financial result|results? for|audited|un-?audited", re.I)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def reported_quarter(news_dt: pd.Timestamp) -> str:
    """Map filing date -> the quarter it most likely reports (for dedup)."""
    y, mth = news_dt.year, news_dt.month
    if mth in (1, 2, 3):
        return f"{y-1}-12"      # Q3 (Dec) reported Jan-Mar
    if mth in (4, 5, 6):
        return f"{y}-03"        # Q4 (Mar) reported Apr-Jun
    if mth in (7, 8, 9):
        return f"{y}-06"        # Q1 (Jun) reported Jul-Sep
    return f"{y}-09"            # Q2 (Sep) reported Oct-Dec


def backtestable_core() -> set:
    """keys with BOTH technical and fundamental metrics computed."""
    t = pd.read_parquet(os.path.join(STATUS_DIR, "tech_metrics_ledger.parquet"))
    f = pd.read_parquet(os.path.join(STATUS_DIR, "fund_metrics_ledger.parquet"))
    tk = set(t.loc[t["status"] == "done", "guru_key"])
    fk = set(f.loc[f["status"] == "done", "guru_key"])
    return tk & fk


def build_worklist() -> pd.DataFrame:
    core = backtestable_core()
    uni = pd.read_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"))
    uni["bse_code"] = uni["bse_code"].astype(str)
    bmap = uni[uni.bse_code.str.match(r"^\d+$", na=False)].set_index("bse_code")["guru_key"]
    rd = pd.read_parquet(os.path.join(DATA_DIR, "results_dates_hist.parquet"))
    rd["SCRIP_CD"] = rd["SCRIP_CD"].astype(str)
    rd["guru_key"] = rd["SCRIP_CD"].map(bmap)
    # format='ISO8601': the column mixes millisecond and non-millisecond ISO
    # stamps; default inference locks onto the first row's format and coerces all
    # pre-2017 (no-millisecond) rows to NaT — silently dropping 9 years of data.
    rd["NEWS_DT"] = pd.to_datetime(rd["NEWS_DT"], format="ISO8601",
                                   errors="coerce").dt.tz_localize(None)
    rd = rd.dropna(subset=["guru_key", "NEWS_DT", "ATTACHMENTNAME"])
    rd = rd[rd["guru_key"].isin(core)]
    rd = rd[rd["NEWS_DT"].dt.year <= DEEP_MAX_YEAR]
    sub = rd["NEWSSUB"].fillna("")
    rd = rd[sub.str.contains(INCLUDE_RE) & ~sub.str.contains(EXCLUDE_RE)]
    rd["reported_quarter"] = rd["NEWS_DT"].apply(reported_quarter)
    # dedup: earliest filing per (guru_key, reported_quarter) = the result announcement
    rd = rd.sort_values("NEWS_DT").drop_duplicates(
        subset=["guru_key", "reported_quarter"], keep="first")
    rd = rd.sort_values("NEWS_DT", ascending=False)   # RECENT-FIRST
    return rd[["NEWSID", "guru_key", "reported_quarter", "NEWS_DT",
               "ATTACHMENTNAME", "NEWSSUB"]].reset_index(drop=True)


def parse_response(text: str) -> dict:
    facts = {c: None for c in ("revenue_cr", "ebitda_cr", "pat_cr", "eps",
                               "ebitda_margin_pct", "pat_margin_pct",
                               "revenue_yoy_pct", "pat_yoy_pct", "gemini_quarter")}
    tables = extract_md_tables(text)
    if not tables:
        return facts
    t = tables[0]
    hdrs = t["headers"]
    q_col = next((i for i, h in enumerate(hdrs)
                  if re.search(r"Q\d\s*FY\d{2,4}", h, re.I)), 1 if len(hdrs) > 1 else None)
    if q_col is not None and q_col < len(hdrs):
        m = re.search(r"(Q\d\s*FY\d{2,4})", hdrs[q_col], re.I)
        if m:
            facts["gemini_quarter"] = m.group(1).strip()
    yoy_col = next((i for i, h in enumerate(hdrs) if re.search(r"yoy|%", h, re.I)), None)
    for cells in t["rows"]:
        if not cells:
            continue
        metric = identify_metric(cells[0])
        low = cells[0].lower()
        if q_col is not None and q_col < len(cells):
            raw = clean_val(cells[q_col])
            if raw != "NA":
                if "margin" in low and "ebitda" in low:
                    facts["ebitda_margin_pct"] = raw
                elif "margin" in low and "pat" in low:
                    facts["pat_margin_pct"] = raw
                elif metric in ("revenue", "sales"):
                    facts["revenue_cr"] = raw
                elif metric == "ebitda":
                    facts["ebitda_cr"] = raw
                elif metric == "pat":
                    facts["pat_cr"] = raw
                elif metric == "eps":
                    facts["eps"] = raw
        if yoy_col is not None and yoy_col < len(cells):
            ry = clean_val(cells[yoy_col])
            if metric in ("revenue", "sales"):
                facts["revenue_yoy_pct"] = ry
            elif metric == "pat":
                facts["pat_yoy_pct"] = ry
    return facts


def load_ledger(work: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(LEDGER_PATH):
        led = pd.read_parquet(LEDGER_PATH)
        new = work[~work["NEWSID"].isin(led["newsid"])]
        if not new.empty:
            add = pd.DataFrame({"newsid": new["NEWSID"], "guru_key": new["guru_key"],
                                "status": "pending", "error": ""})
            led = pd.concat([led, add], ignore_index=True)
        return led
    return pd.DataFrame({"newsid": work["NEWSID"], "guru_key": work["guru_key"],
                         "status": "pending", "error": ""})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    work = build_worklist()
    led = load_ledger(work)
    if args.status:
        print(f"{led['status'].value_counts().to_dict()}")
        if os.path.exists(OUT_PATH):
            print(f"facts rows: {len(pd.read_parquet(OUT_PATH)):,}")
        return

    done_ids = set(led.loc[led["status"].isin(["done", "empty"]), "newsid"])
    if args.retry_errors:
        done_ids = set(led.loc[led["status"].isin(["done", "empty"]), "newsid"])
    todo = work[~work["NEWSID"].isin(done_ids)]
    if args.limit:
        todo = todo.head(args.limit)
    log(f"worklist: {len(work)} company-quarters in scope | to extract now: {len(todo)}")
    log(f"  span: {work['NEWS_DT'].min().date()} -> {work['NEWS_DT'].max().date()} "
        f"| distinct companies: {work['guru_key'].nunique()}")

    if args.dry_run:
        log("DRY RUN — no downloads, no LLM. First 10:")
        for _, r in todo.head(10).iterrows():
            print(f"   {r['guru_key']}  {r['reported_quarter']}  {r['NEWS_DT'].date()}  "
                  f"{str(r['NEWSSUB'])[:50]}")
        return

    keys = load_keys(os.environ, prefix="FREE_POOL")
    if not keys:
        log("ERROR: no FREE_POOL keys in .env"); sys.exit(1)
    log(f"FREE_POOL: {len(keys)} keys | models: {P1_MODELS}")
    pool = BucketPool(keys, P1_MODELS, logger=log)
    prompt = (Path(SCRIPTS_DIR) / "results_prompt.txt").read_text(encoding="utf-8")

    facts_rows = []
    if os.path.exists(OUT_PATH):
        facts_rows = pd.read_parquet(OUT_PATH).to_dict("records")
    led_idx = {nid: i for i, nid in enumerate(led["newsid"])}
    sess = requests.Session(); sess.headers.update(PDF_HDR)
    n_ok = n_empty = n_err = 0

    def flush():
        os.makedirs(STATUS_DIR, exist_ok=True)
        pd.DataFrame(facts_rows, columns=OUT_COLS).to_parquet(OUT_PATH, index=False)
        led.to_parquet(LEDGER_PATH, index=False)

    for i, (_, r) in enumerate(todo.iterrows(), 1):
        nid = r["NEWSID"]; li = led_idx[nid]
        try:
            resp = sess.get(ATTACH_URL.format(r["ATTACHMENTNAME"]), timeout=40)
            if resp.status_code != 200 or resp.content[:4] != b"%PDF":
                led.at[li, "status"] = "empty"; led.at[li, "error"] = f"no pdf ({resp.status_code})"
                n_empty += 1
            else:
                text, model = pool.call_pdf(resp.content, prompt)
                f = parse_response(text)
                if f["revenue_cr"] is None and f["pat_cr"] is None and f["eps"] is None:
                    led.at[li, "status"] = "empty"; led.at[li, "error"] = "no numbers parsed"
                    n_empty += 1
                else:
                    row = {"guru_key": r["guru_key"], "reported_quarter": r["reported_quarter"],
                           "announcement_date": r["NEWS_DT"], "model": model, "newsid": nid}
                    row.update(f)
                    facts_rows.append({c: row.get(c) for c in OUT_COLS})
                    led.at[li, "status"] = "done"; led.at[li, "error"] = ""
                    n_ok += 1
        except AllBucketsExhausted:
            flush()
            log(f"All FREE_POOL buckets exhausted — flushed, stopping cleanly "
                f"(done={n_ok} empty={n_empty} err={n_err}). Rerun resumes.")
            return
        except FatalCallError as e:
            led.at[li, "status"] = "error"; led.at[li, "error"] = str(e)[:150]; n_err += 1
        except Exception as e:
            led.at[li, "status"] = "error"; led.at[li, "error"] = str(e)[:150]; n_err += 1
        if i % FLUSH_EVERY == 0:
            flush()
            log(f"  {i}/{len(todo)} (done={n_ok} empty={n_empty} err={n_err})")
    flush()
    log(f"RUN COMPLETE: done={n_ok} empty={n_empty} err={n_err}")


if __name__ == "__main__":
    main()
