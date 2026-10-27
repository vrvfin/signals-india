r"""
results_thesis_mail.py — ONE mail per good result: a ~200-word "why it did well" thesis.

Selection = the Results-surge rule (flag_growth_surge.build_rows: latest-quarter PAT YoY
>= 100%, whole universe), reported in the last --hours (default 48). For each one:
  1. BSE filings of the last few days (per company, bse_http session): the results filing
     and the investor presentation. PDFs are read IN MEMORY only — never saved to Drive.
  2. Past guidance for the ISIN (guidance_tracker / ppt_guidance / gf2 / mgmt_credibility).
  3. Chart facts, rule-based (tech_history: stage, patterns, extremes, vs Nifty 500) +
     features/latest.parquet.
  4. ONE Gemini call (FREE_POOL first) -> <=200-word thesis in 4 parts.
  5. One mail per company (toggle 'results_thesis'); ledger row only after a confirmed send.

Wait rule: results + deck found -> send now. No deck yet -> wait until the result is 24h
old, then send without it. No results PDF after 24h -> numbers + guidance only, flagged.

Own ledger: company_repo/_index/results_thesis_ledger.parquet, keyed (isin, quarter).
It is NOT the global processing_queue — user-approved exception to rule 7 (2026-10-25):
these PDFs are read in memory for a mail, never stored or extracted.

Usage:
  python scripts/results_thesis_mail.py --dry-run                 # candidates + docs + decisions
  python scripts/results_thesis_mail.py --dry-run --hours 168
  python scripts/results_thesis_mail.py --symbols ABC --no-send --preview-dir out   # 1 Gemini call
  python scripts/results_thesis_mail.py                           # live
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, load_parquet, load_table, save_parquet,
                             load_portfolio_isins, log)
from mailer import send_email, load_mail_settings, esc
from bse_http import bse_session
import flag_growth_surge as GS
import tech_history as T

MAIL_KEY = "results_thesis"
LEDGER_NAME = "results_thesis_ledger.parquet"
LEDGER_COLS = ["isin", "symbol", "quarter", "reported_at", "had_results_pdf",
               "had_deck", "pdf_urls", "model", "mailed_at", "subject"]
PROMPT_FILE = os.path.join(_SCRIPTS_DIR, "results_thesis_prompt.txt")

DECK_WAIT_H = 24          # wait this long after the result for the presentation
BSE_LOOKBACK_DAYS = 4     # filings window per company (results day can precede first_seen)
DECK_CHARS = 40000        # presentation text cap in the prompt
MIN_DECK_CHARS = 2500     # below this the "deck" is a cover letter, not a presentation
MAX_OUT_TOKENS = 4096     # thesis is ~200 words; the cap only stops a runaway
MAX_HTML_BYTES = 90_000   # Gmail clips at ~102 KB

ANN_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
ATTACH_HOSTS = ("https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
                "https://www.bseindia.com/xml-data/corpfiling/AttachHis/")
# Referer MUST be the homepage: with .../corporates/ann.html Akamai answers 403 even
# through curl_cffi (measured 2026-10-26); the homepage referer gets 200.
API_HDR = {"Accept": "application/json, text/plain, */*",
           "Referer": "https://www.bseindia.com/",
           "Origin": "https://www.bseindia.com"}
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

RESULT_PAT = re.compile(r"financial result|\bresults?\b", re.I)
DECK_PAT = re.compile(r"investor presentation|earnings presentation|analyst presentation", re.I)
NOT_RESULT_PAT = re.compile(r"intimation|prior intimation|newspaper|trading window|"
                            r"press release.*intimation|record date", re.I)


def _utcnow() -> datetime:
    """Naive UTC — reported_at (results.parquet first_seen_at) is stamped in UTC on CI."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _age_h(stamp, now: datetime) -> float | None:
    t = pd.to_datetime(stamp, errors="coerce")
    if pd.isna(t):
        return None
    return (now - t.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600


# ---------------------------------------------------------------- BSE filings
def fetch_filings(sess, code: str, days: int) -> list[dict]:
    to_d = datetime.now().date()
    from_d = to_d - timedelta(days=days - 1)
    out = []
    for pageno in range(1, 4):
        params = {"pageno": pageno, "strCat": "-1", "subcategory": "-1",
                  "strPrevDate": from_d.strftime("%Y%m%d"), "strToDate": to_d.strftime("%Y%m%d"),
                  "strSearch": "P", "strScrip": str(code), "strType": "C"}
        try:
            r = sess.get(ANN_API, params=params, timeout=30)
            rows = r.json().get("Table", []) if r.status_code == 200 else []
            if r.status_code != 200:
                log(f"    BSE {code}: HTTP {r.status_code}")
        except Exception as e:
            log(f"    BSE {code}: fetch failed ({str(e)[:60]})")
            break
        if not rows:
            break
        out += rows
        if len(rows) < 50:
            break
        time.sleep(0.2)
    return out


def pick_docs(rows: list[dict]) -> tuple[dict | None, dict | None]:
    """(results filing, investor presentation) — newest of each, attachment required."""
    res, deck = [], []
    for r in rows:
        att = (r.get("ATTACHMENTNAME") or "").strip()
        if not att:
            continue
        cat = (r.get("CATEGORYNAME") or "").strip()
        sub = (r.get("SUBCATNAME") or "").strip()
        head = (r.get("NEWSSUB") or r.get("HEADLINE") or "").strip()
        blob = f"{cat} {sub} {head}"
        d = {"attachment": att, "date": str(r.get("NEWS_DT", ""))[:16],
             "headline": head, "category": f"{cat} / {sub}".strip(" /")}
        if DECK_PAT.search(blob) or str(r.get("Investor_Presentation") or "").strip():
            # a filing NAMED investor presentation beats an "Analyst / Investor Meet -
            # Outcome" that BSE merely flags as carrying one
            d["rank"] = 0 if DECK_PAT.search(blob) else 1
            deck.append(d)
        elif (cat.lower() == "result" or RESULT_PAT.search(blob)) and not NOT_RESULT_PAT.search(blob):
            # a Financial Results sub-category beats a board-outcome that mentions results
            d["rank"] = 0 if "financial result" in sub.lower() else 1
            res.append(d)
    res.sort(key=lambda d: d["date"], reverse=True)       # newest first ...
    res.sort(key=lambda d: d["rank"])                     # ... within each rank (stable)
    deck.sort(key=lambda d: d["date"], reverse=True)
    deck.sort(key=lambda d: d["rank"])
    return (res[0] if res else None), (deck[0] if deck else None)


def download_pdf(sess, attachment: str) -> tuple[bytes | None, str]:
    for base in ATTACH_HOSTS:
        url = base + attachment
        try:
            r = sess.get(url, timeout=60)
            if r.status_code == 200 and r.content[:5].startswith(b"%PDF"):
                return r.content, url
        except Exception:
            continue
    return None, ATTACH_HOSTS[0] + attachment


def pdf_text(pdf_bytes: bytes, max_chars: int) -> str:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)[:max_chars]
    except Exception:
        return ""


# ---------------------------------------------------------------- context blocks
def numbers_block(results: pd.DataFrame, isin: str, row: dict) -> tuple[str, list[tuple]]:
    """Text for the prompt + (metric, latest, yoy, qoq) rows for the mail strip."""
    r = results[results["isin"].astype(str).str.strip() == isin]
    lines, strip = [], []
    if not r.empty:
        r = r.copy()
        r["_qd"] = pd.to_datetime(r["latest_q"], format="%b %Y", errors="coerce")
        r = r[r["_qd"] == r["_qd"].max()] if r["_qd"].notna().any() else r
        for _, x in r.drop_duplicates("metric", keep="last").iterrows():
            lv, yv, pv = x.get("latest_val"), x.get("yearago_val"), x.get("prev_val")
            yoy, qoq = x.get("yoy_pct"), x.get("qoq_pct")
            lines.append(f"- {x['metric']}: {x.get('latest_q')} = {lv} | year-ago "
                         f"{x.get('yearago_q')} = {yv} | prev qtr = {pv} | YoY {yoy}% | QoQ {qoq}%")
            strip.append((str(x["metric"]), lv, yoy, qoq))
    lines.append(f"- PAT YoY used for selection: {row['pat_yoy_pct']}% ({row['quarter']})")
    lines.append(f"- Classification: {row['classification']} (quarters with >=100% PAT YoY in "
                 f"last 4: {row['n_surge_4q']}; year-ago base: {row['base_yearago_sign']})")
    return "\n".join(lines), strip


GUIDANCE_TABLES = [   # (file, columns to show) — read-only, filtered to one ISIN
    ("guidance_tracker.parquet", ["quarter", "metric", "guidance_type", "horizon_fy",
                                  "value", "unit", "notes"]),
    ("ppt_guidance.parquet", ["quarter", "metric", "guidance_type", "horizon", "value",
                              "unit", "notes"]),
    ("mgmt_credibility.parquet", ["qtr_guided", "metric", "guidance_given", "target_period",
                                  "actual_delivered", "verdict", "cred_score"]),
    ("gf2_historical_guidance.parquet", ["financial_qtr", "original_guidance",
                                         "actual_mentioned_outcome",
                                         "management_self_assessment"]),
]


def load_guidance(drive, index_id) -> dict[str, pd.DataFrame]:
    out = {}
    for name, cols in GUIDANCE_TABLES:
        df = load_table(drive, index_id, name, ["isin"] + cols)
        out[name] = df
        log(f"  {name}: {len(df)} rows")
    return out


def guidance_block(tables: dict, isin: str, per_table: int = 8) -> tuple[str, list[list[str]]]:
    lines, rows = [], []
    for name, cols in GUIDANCE_TABLES:
        df = tables.get(name)
        if df is None or df.empty or "isin" not in df.columns:
            continue
        d = df[df["isin"].astype(str).str.strip() == isin]
        if d.empty:
            continue
        use = [c for c in cols if c in d.columns]
        d = d[use].tail(per_table)
        lines.append(f"[{name.replace('.parquet', '')}]")
        for _, x in d.iterrows():
            vals = [str(x[c])[:160] for c in use
                    if pd.notna(x[c]) and str(x[c]).strip() not in ("", "None", "nan")]
            lines.append("- " + " | ".join(vals))
            if name in ("guidance_tracker.parquet", "ppt_guidance.parquet",
                        "mgmt_credibility.parquet"):
                rows.append([name.split("_")[0]] + vals[:5])
    return ("\n".join(lines) if lines else "No past guidance on record."), rows[-12:]


PATTERN_NAME = {   # same wording as build_pf_history.PATTERN_NAME (kept local: that
    # script is a local-only page builder with heavy imports)
    "golden_cross_50_200": "Golden cross (50/200)", "death_cross_50_200": "Death cross (50/200)",
    "cross_up_20_50": "20-day crossed above 50-day", "cross_down_20_50": "20-day crossed below 50-day",
    "reclaim_50": "Reclaimed the 50-day average", "lose_50": "Lost the 50-day average",
    "reclaim_200": "Reclaimed the 200-day average", "lose_200": "Lost the 200-day average",
    "higher_highs_lows": "Higher highs & higher lows", "lower_highs_lows": "Lower highs & lower lows",
    "high_52w_on_volume": "52-week high on volume", "low_52w": "New 52-week low",
    "squeeze_break_up": "Squeeze broke UP", "squeeze_break_down": "Squeeze broke DOWN",
    "pocket_pivot": "Pocket pivot", "distribution_5_in_25": "5 distribution days in 25",
    "rsi_bear_divergence": "Bearish RSI divergence", "rsi_bull_divergence": "Bullish RSI divergence",
    "failed_breakout": "Failed breakout", "close_below_21ema_after_run": "Closed below 21-day EMA after a run",
}


def tech_facts(px: pd.DataFrame | None, n500: pd.Series | None, feat: dict | None) -> list[str]:
    facts = []
    if px is not None and not px.empty:
        try:
            df = T._norm(px)
            pat = T.patterns(df)
            if pat.get("stage"):
                facts.append(f"Trend: {pat['stage']}")
            c = df["close"]
            rsi = T.rsi(c).iloc[-1]
            if pd.notna(rsi):
                facts.append(f"RSI(14): {rsi:.0f}")
            unus = T.unusual(df)
            rel = [m for m in [T.vs_market(df, n500, "Nifty 500")] if m]
            for m in rel:
                ex = ", ".join(f"{k} {v:+.0f}%" for k, v in m.get("excess", {}).items())
                if ex:
                    facts.append(f"Excess return vs Nifty 500: {ex}"
                                 + (f" ({m['quadrant']})" if m.get("quadrant") else ""))
            for f in pat.get("fired", [])[:3]:
                facts.append(f"Pattern: {PATTERN_NAME.get(f['key'], f['key'])} "
                             f"({f['bars_ago']} sessions ago)")
            for v in T.verdict(unus, pat, rel):
                if v.get("text"):
                    facts.append(f"Unusual: {v['text']}")
        except Exception as e:
            facts.append(f"(chart history unavailable: {str(e)[:60]})")
    if feat:
        def g(k):
            v = feat.get(k)
            return None if v is None or (isinstance(v, float) and pd.isna(v)) else v
        bits = []
        if g("dist_from_52w_high_pct") is not None:
            bits.append(f"{float(g('dist_from_52w_high_pct')):+.1f}% from 52-week high")
        for k, lbl in (("above_50sma", "50-day"), ("above_200sma", "200-day")):
            if g(k) is not None:
                bits.append(f"{'above' if bool(g(k)) else 'below'} {lbl} average")
        for k, lbl in (("return_1m_pct", "1M"), ("return_3m_pct", "3M"), ("return_12m_pct", "12M")):
            if g(k) is not None:
                bits.append(f"{lbl} return {float(g(k)):+.0f}%")
        rs = [k for k in feat if str(k).startswith("rs_rank")]
        if rs and g(rs[0]) is not None:
            bits.append(f"{rs[0]} {float(g(rs[0])):.0f}")
        if bits:
            facts.append("Levels: " + "; ".join(bits))
    return facts or ["No price history available."]


def load_prices(drive, symbols: list[str]) -> tuple[dict, pd.Series | None, dict]:
    """{sym: ohlcv df}, Nifty-500 close series, {sym: features row} — Drive, read-only."""
    import build_gallery as BG
    px = BG._bulk_parquet(drive, BG._folder(drive, "data/ohlcv"), symbols, what="OHLCV")
    n500 = None
    try:
        idx = BG._read_parquet(drive, BG._folder(drive, "data/indices"), "NIFTY_500.parquet")
        if not idx.empty and {"date", "close"} <= set(idx.columns):
            idx["date"] = pd.to_datetime(idx["date"])
            n500 = idx.drop_duplicates("date").set_index("date")["close"].sort_index()
    except Exception as e:
        log(f"  NIFTY_500 unavailable ({str(e)[:60]})")
    feats = {}
    try:
        f = BG._read_parquet(drive, BG._folder(drive, "features"), "latest.parquet")
        if not f.empty and "symbol" in f.columns:
            f = f[f["symbol"].astype(str).isin(symbols)]
            feats = {str(r["symbol"]): r.to_dict() for _, r in f.iterrows()}
    except Exception as e:
        log(f"  features/latest.parquet unavailable ({str(e)[:60]})")
    return px, n500, feats


# ---------------------------------------------------------------- Gemini
def build_pool():
    from gemini_pool import BucketPool, load_keys_multi
    from model_registry import pick, overload_keys
    keys = load_keys_multi(os.environ, "FREE_POOL,BACKFILL_GEMINI_KEY,GEMINI_API_KEY")
    if not keys:
        log("No Gemini keys (FREE_POOL/BACKFILL/GEMINI).")
        return None
    # BULK, not WRITE: WRITE can lead with a thinking model that burns the whole output
    # cap on long PDF prompts (see gemini thinking runaway) — a 200-word note needs none.
    models = pick("BULK", FALLBACK_MODELS)
    log(f"Gemini pool: {len(keys)} keys · models {models}")
    return BucketPool(keys, models, inter_call_s=2.0, logger=log,
                      model_overload_keys=overload_keys(len(keys)))


def clean_thesis(text: str) -> str:
    t = re.sub(r"^```\w*|```$", "", (text or "").strip(), flags=re.M).strip()
    return t


def thesis_html(text: str) -> str:
    out = []
    for para in [p.strip() for p in re.split(r"\n\s*\n|\n(?=\*\*)", text) if p.strip()]:
        h = esc(para, 4000)
        h = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", h)
        out.append(f'<p style="margin:6px 0;line-height:1.5">{h}</p>')
    return "\n".join(out)


# ---------------------------------------------------------------- mail
def _pct(v) -> str:
    v = pd.to_numeric(v, errors="coerce")
    return "—" if pd.isna(v) else f"{float(v):+.0f}%"


def build_mail(row: dict, thesis: str, strip: list, grows: list, tech: list,
               res_doc, deck_doc, urls: dict, model: str,
               deck_chars: int = 0) -> tuple[str, str]:
    sym = row["symbol"] or row["isin"]
    pf = " 💼PF" if row["in_pf"] else ""
    subject = (f"🧠 Results thesis — {sym} {row['quarter']} · PAT {_pct(row['pat_yoy_pct'])} YoY"
               f"{pf}")
    td = 'style="padding:3px 8px;border-bottom:1px solid #eee;font-size:12.5px"'
    num = "".join(f"<tr><td {td}>{esc(m, 40)}</td><td {td}>{esc(lv, 20)}</td>"
                  f"<td {td}>{_pct(y)}</td><td {td}>{_pct(q)}</td></tr>"
                  for m, lv, y, q in strip)
    gtab = "".join("<tr>" + "".join(f"<td {td}>{esc(c, 90)}</td>" for c in g) + "</tr>"
                   for g in grows)
    missing = []
    if not res_doc:
        missing.append("results filing not found on BSE — thesis from reported numbers only")
    if not deck_doc:
        missing.append("no investor presentation found")
    elif deck_chars < MIN_DECK_CHARS:
        missing.append("the 'presentation' filing is a short letter, not a deck")
    src = []
    for lbl, d in (("Results filing", res_doc), ("Investor presentation", deck_doc)):
        if d:
            src.append(f'<a href="{esc(urls.get(lbl, ""), 300)}">{lbl}</a> '
                       f'({esc(d["date"], 16)} · {esc(d["headline"], 90)})')
    html = f"""<div style="font-family:Arial,sans-serif;max-width:720px;color:#222">
<h2 style="margin:0 0 4px">{esc(row['company_name'] or sym, 80)} ({esc(sym, 20)}){pf}</h2>
<div style="color:#555;font-size:13px">{esc(row['quarter'], 12)} · PAT YoY {_pct(row['pat_yoy_pct'])}
 · {esc(row['classification'], 12)} · reported {esc(str(row['reported_at'])[:16], 16)} UTC</div>
{''.join(f'<div style="color:#a15c00;font-size:12px">⚠ {esc(m, 120)}</div>' for m in missing)}
<div style="background:#f6f9fc;border-left:3px solid #0b5394;padding:6px 12px;margin:10px 0">
{thesis_html(thesis)}</div>
<h4 style="margin:12px 0 4px">Reported numbers</h4>
<table style="border-collapse:collapse"><tr><th {td}>Metric</th><th {td}>Latest</th>
<th {td}>YoY</th><th {td}>QoQ</th></tr>{num}</table>
<h4 style="margin:12px 0 4px">Past guidance</h4>
{f'<table style="border-collapse:collapse">{gtab}</table>' if gtab else '<div style="font-size:12.5px">None on record.</div>'}
<h4 style="margin:12px 0 4px">Chart</h4>
<ul style="font-size:12.5px;margin:0">{''.join(f'<li>{esc(t, 200)}</li>' for t in tech)}</ul>
<div style="font-size:11px;color:#777;margin-top:12px">Sources: {' · '.join(src) or 'none'}
 · model {esc(model, 40)} · signals only, not advice.</div></div>"""
    if len(html.encode("utf-8")) > MAX_HTML_BYTES:
        html = html.replace(gtab, "")
    return subject, html


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=48.0,
                    help="Candidates reported in the last N hours (default 48).")
    ap.add_argument("--dry-run", action="store_true",
                    help="No Gemini, no mail, no Drive write: candidates, docs found, decisions.")
    ap.add_argument("--symbols", default="",
                    help="Comma list: only these symbols, ignoring the --hours window and the "
                         "deck wait.")
    ap.add_argument("--force", action="store_true",
                    help="With --symbols: ignore the ledger (re-mail).")
    ap.add_argument("--no-send", action="store_true",
                    help="Call Gemini but do not mail and do not write the ledger.")
    ap.add_argument("--preview-dir", default="", help="Write each mail's HTML here.")
    ap.add_argument("--max-mails", type=int, default=40)
    args = ap.parse_args()
    if args.force and not args.symbols:
        sys.exit("--force needs --symbols")
    want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    print("Results thesis mail — why the good results were good")
    print("-" * 60)
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    if not args.dry_run and not args.no_send:
        if not load_mail_settings(drive, index_id).get(MAIL_KEY, True):
            log(f"{MAIL_KEY} mail toggled OFF — nothing to do.")
            return

    # 1) candidates — the Results-surge rule, unchanged
    derived = load_parquet(drive, index_id, "financials_derived.parquet", GS.DERIVED_COLS)
    fin3 = load_parquet(drive, index_id, "financials_3stmt.parquet", GS.FIN3_COLS)
    results = load_parquet(drive, index_id, "results.parquet", GS.RESULTS_COLS)
    if derived.empty:
        sys.exit("financials_derived.parquet empty.")
    names, bse_by, nse_by = {}, {}, {}
    uni_id = find_file(drive, index_id, "company_universe.csv")
    if uni_id:
        uni = pd.read_csv(io.BytesIO(download_bytes(drive, uni_id)), dtype=str)
        for _, r in uni.iterrows():
            iso = str(r.get("isin", "")).strip()
            if not iso:
                continue
            names[iso] = str(r.get("name", "") or "").strip()
            b = str(r.get("bse_code", "") or "").strip().split(".")[0]
            if b and b.lower() != "nan":
                bse_by[iso] = b
            n = str(r.get("nse_symbol", "") or "").strip().upper()
            if n and n.lower() != "nan":
                nse_by[iso] = n
    pf = load_portfolio_isins(drive, root_id) or set()
    now = _utcnow()
    rows = GS.build_rows(derived, fin3, results, names, pf, now.isoformat(timespec="seconds"))
    for r in rows:
        r["symbol"] = r["symbol"] or nse_by.get(r["isin"], "") or bse_by.get(r["isin"], "")
    log(f"surging companies (all): {len(rows)}")

    if want:
        cands = [r for r in rows if r["symbol"].upper() in want]
        log(f"--symbols: {len(cands)} of {len(want)} found among surging rows")
    else:
        cands = [r for r in rows
                 if (a := _age_h(r["reported_at"], now)) is not None and a <= args.hours]
    log(f"candidates: {len(cands)}"
        + ("" if want else f" (reported in last {args.hours:.0f}h)"))

    ledger = load_table(drive, index_id, LEDGER_NAME, LEDGER_COLS)
    done = set(zip(ledger["isin"].astype(str), ledger["quarter"].astype(str)))
    if not args.force:
        before = len(cands)
        cands = [r for r in cands if (r["isin"], str(r["quarter"])) not in done]
        log(f"already mailed (ledger): {before - len(cands)}")
    cands.sort(key=lambda r: (not r["in_pf"], -(r["pat_yoy_pct"] or 0)))
    if not cands:
        log("nothing to mail.")
        return

    # 2) docs + wait decision per candidate
    sess = bse_session(API_HDR)
    plan = []
    for r in cands:
        code = bse_by.get(r["isin"])
        age = _age_h(r["reported_at"], now) or 0.0
        days = max(BSE_LOOKBACK_DAYS, int(age // 24) + 3)   # an older result needs a wider window
        filings = fetch_filings(sess, code, days) if code else []
        res_doc, deck_doc = pick_docs(filings)
        # only a filing NAMED a presentation ends the wait — a flagged "Investor Meet -
        # Outcome" is often a 1-page letter (PERMAGN 2026-10-24: 1,194 chars)
        if res_doc and deck_doc and deck_doc["rank"] == 0:
            decision = "BUILD"
        elif age < DECK_WAIT_H and not want:
            decision = "WAIT"
        else:
            decision = "BUILD"
        plan.append((r, code, res_doc, deck_doc, age, decision))
        log(f"  {r['symbol']:<14} {r['quarter']:<9} PAT {_pct(r['pat_yoy_pct']):>6} "
            f"{r['classification']:<10} age {age:5.1f}h bse={code or '-'} "
            f"filings={len(filings)} results={'Y' if res_doc else '-'} "
            f"deck={'Y' if deck_doc else '-'} -> {decision}")
        if args.dry_run:
            for lbl, d in (("results", res_doc), ("deck", deck_doc)):
                if d:
                    log(f"      {lbl}: {d['date']} [{d['category']}] {d['headline'][:90]}")
        time.sleep(0.3)

    todo = [p for p in plan if p[5] == "BUILD"][:args.max_mails]
    log(f"to build now: {len(todo)} · waiting for deck: "
        f"{sum(1 for p in plan if p[5] == 'WAIT')}")
    if args.dry_run:
        print("\nDRY RUN — no Gemini, no mail, no Drive write.")
        return
    if not todo:
        return

    # 3) shared context, loaded once
    gtables = load_guidance(drive, index_id)
    px, n500, feats = load_prices(drive, [p[0]["symbol"] for p in todo])
    prompt_tpl = Path(PROMPT_FILE).read_text(encoding="utf-8")
    pool = build_pool()
    if pool is None:
        sys.exit(1)
    from gemini_pool import AllBucketsExhausted, FatalCallError

    prev_dir = Path(args.preview_dir) if args.preview_dir else None
    if prev_dir:
        prev_dir.mkdir(parents=True, exist_ok=True)
    sent_rows, n_sent, n_fail = [], 0, 0
    for r, code, res_doc, deck_doc, age, _ in todo:
        sym = r["symbol"]
        urls, res_pdf, deck_txt = {}, None, ""
        if res_doc:
            res_pdf, urls["Results filing"] = download_pdf(sess, res_doc["attachment"])
        if deck_doc:
            deck_pdf, urls["Investor presentation"] = download_pdf(sess, deck_doc["attachment"])
            deck_txt = pdf_text(deck_pdf, DECK_CHARS) if deck_pdf else ""
        numbers, strip = numbers_block(results, r["isin"], r)
        guid, grows = guidance_block(gtables, r["isin"])
        tech = tech_facts(px.get(sym), n500, feats.get(sym))
        prompt = prompt_tpl.format(
            company=f"{r['company_name']} ({sym}), ISIN {r['isin']}",
            numbers=numbers, guidance=guid, tech="\n".join(f"- {t}" for t in tech),
            deck=deck_txt or "No investor presentation available.",
            results_note=("Attached as PDF." if res_pdf else
                          "Not available — rely on the reported numbers above."))
        try:
            if res_pdf:
                text, model = pool.call_pdf(res_pdf, prompt, max_output_tokens=MAX_OUT_TOKENS)
            else:
                text, model = pool.call_text(prompt, max_output_tokens=MAX_OUT_TOKENS)
        except AllBucketsExhausted:
            log("  Gemini quota exhausted — the rest wait for the next run.")
            break
        except FatalCallError as e:
            log(f"  {sym}: Gemini fatal error ({str(e)[:80]}) — skipped")
            n_fail += 1
            continue
        thesis = clean_thesis(text)
        if len(thesis) < 200:
            log(f"  {sym}: thesis too short ({len(thesis)} chars) — skipped, retried next run")
            n_fail += 1
            continue
        subject, html = build_mail(r, thesis, strip, grows, tech, res_doc, deck_doc, urls, model,
                                   deck_chars=len(deck_txt))
        words = len(re.sub(r"\*\*", "", thesis).split())
        log(f"  {sym}: thesis {words} words via {model} · results pdf "
            f"{'Y' if res_pdf else '-'} · deck text {len(deck_txt)} chars")
        if prev_dir:
            (prev_dir / f"thesis_{sym}.html").write_text(html, encoding="utf-8")
        if args.no_send:
            continue
        if send_email(subject, html):
            n_sent += 1
            sent_rows.append({"isin": r["isin"], "symbol": sym, "quarter": str(r["quarter"]),
                              "reported_at": str(r["reported_at"]),
                              "had_results_pdf": bool(res_pdf),
                              "had_deck": len(deck_txt) >= MIN_DECK_CHARS,
                              "pdf_urls": " ".join(urls.values()), "model": model,
                              "mailed_at": _utcnow().isoformat(timespec="seconds"),
                              "subject": subject})
        else:
            n_fail += 1
            log(f"  {sym}: mail send FAILED — not in ledger, retried next run")

    try:
        from model_registry import record_usage
        record_usage(pool, "results_thesis", "results", drive=drive, index_id=index_id, log=log)
    except Exception:
        pass
    if sent_rows:
        # re-read just before writing so a parallel run's rows are not lost
        ledger = load_table(drive, index_id, LEDGER_NAME, LEDGER_COLS)
        out = pd.concat([ledger, pd.DataFrame(sent_rows)], ignore_index=True)
        out = out.drop_duplicates(["isin", "quarter"], keep="first")
        save_parquet(drive, index_id, LEDGER_NAME, out)
        log(f"ledger: +{len(sent_rows)} rows ({len(out)} total)")
    log(f"done: sent {n_sent} · failed {n_fail}")


if __name__ == "__main__":
    main()
