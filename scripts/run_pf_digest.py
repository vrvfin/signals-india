"""
run_pf_digest.py — PF daily FOCUS mail (rebuilt 2026-10-14; was a per-holding list).

User 2026-10-13: "it does not have any actionable insights". The old mail listed every
holding's files and flags, and repeated the fraud band and scorecard score every day.

Now: for each holding, only what CHANGED in the window becomes a trigger with its
evidence (filings by materiality, results/concall/rating/deck/AR processed, fraud
band worsened, new catalyst, results due <=3 days, guidance pace
slipping, unusual price move vs Nifty, research, community). Triggers are weighted
into an ATTENTION score; holdings that earn a card get three lines - what changed,
why it matters, what to check - plus a 1-year chart and the evidence. Minor updates
are one line each, quiet holdings one line in total. Rubric, view and grounding:
pf_digest_focus.py.

The three lines come from ONE batched model call (FREE_POOL -> BACKFILL -> GEMINI),
grounded: a view that cites a number missing from its evidence is replaced by the
rule-based one. No keys / dry run / failure -> rule-based for all. Toggle: 'pf_digest'.

Usage:
    python scripts/run_pf_digest.py --dry-run     # no model, no mail, preview html
    python scripts/run_pf_digest.py --preview     # with the model, no mail
    python scripts/run_pf_digest.py --hours 24
    python scripts/run_pf_digest.py --self-test
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, load_portfolio_isins, log)
from mailer import send_email, load_mail_settings, esc
import social_sources
import pf_digest_focus as F
import build_catalyst_notes as CN


def _read(drive, root, parts):
    fid = root
    for p in parts[:-1]:
        fid = get_or_create_subfolder(drive, fid, p)
    f = find_file(drive, fid, parts[-1])
    if not f:
        return pd.DataFrame()
    try:
        raw = download_bytes(drive, f)
        return (pd.read_csv(io.BytesIO(raw)) if parts[-1].endswith(".csv")
                else pd.read_parquet(io.BytesIO(raw)))
    except Exception:
        return pd.DataFrame()


def _txt(v) -> str:
    """Cell -> clean text; NaN / None / 'nan' / 'none' -> ''."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def _by_isin(df, isin, col="isin"):
    if df.empty or col not in df.columns:
        return df.iloc[0:0] if not df.empty else pd.DataFrame()
    return df[df[col].astype(str) == isin]


def _fmt_num(v, prefix="", suffix="") -> str:
    try:
        f = float(v)
        if f != f:
            return ""
        return f"{prefix}{f:,.0f}{suffix}"
    except (TypeError, ValueError):
        return ""


def _doc_snippet(doc_type, doc_id, isin, ctx) -> str | None:
    """Content snippet for a processed doc (user 2026-07-08): the extracted
    numbers/guidance, not just the title. Matches on source_doc_id ONLY: the old
    fallback to "the company's latest rows" put last quarter's guidance under
    today's document, and the focus view then reported it as today's change
    (TATVA, 2026-10-14 preview: an analyst-meet notice shown with ABS guidance).
    Returns HTML or None."""
    doc_id = str(doc_id or "")

    def _match(df):
        if df.empty or not doc_id or "source_doc_id" not in df.columns:
            return df.iloc[0:0] if not df.empty else df
        return df[df["source_doc_id"].astype(str) == doc_id]

    if doc_type == "results":
        r = _match(ctx["rgem"])
        if r.empty:
            return None
        x = r.iloc[-1]
        parts = []
        rev = _fmt_num(x.get("revenue_cr"), "₹", " cr")
        if rev:
            yoy = _fmt_num(x.get("revenue_yoy_pct"), "+", "% YoY")
            parts.append(f"Rev {rev}" + (f" ({yoy})" if yoy else ""))
        pat = _fmt_num(x.get("pat_cr"), "₹", " cr")
        if pat:
            yoy = _fmt_num(x.get("pat_yoy_pct"), "+", "% YoY")
            parts.append(f"PAT {pat}" + (f" ({yoy})" if yoy else ""))
        eps = _fmt_num(x.get("eps"), "₹")
        if eps:
            parts.append(f"EPS {eps}")
        return " · ".join(parts) if parts else None

    if doc_type == "concall":
        g = _match(ctx["guid"])
        if not g.empty:
            parts = [f"{esc(x.get('metric', ''), 14)}: {esc(x.get('value', ''), 24)}"
                     f" ({esc(x.get('horizon_fy', ''), 8)})"
                     for _, x in g.head(3).iterrows() if str(x.get("value", "")).strip()]
            if parts:
                return "guidance — " + " · ".join(parts)
        f1 = _match(ctx["gf1"])
        if not f1.empty:
            x = f1.iloc[-1]
            stmt = str(x.get("exact_statement", "") or "").strip()
            if stmt:
                return f"“{esc(stmt, 180)}”"
        return None

    if doc_type == "presentation":
        p = _match(ctx["ppt"])
        if p.empty:
            return None
        parts = [f"{esc(x.get('metric', ''), 14)}: {esc(x.get('value', ''), 24)}"
                 f" ({esc(x.get('horizon', ''), 8)})"
                 for _, x in p.head(3).iterrows() if str(x.get("value", "")).strip()]
        return ("guidance — " + " · ".join(parts)) if parts else None

    return None


# --------------------------------------------------------------------------- #
#  triggers — only what CHANGED in the window (pf_digest_focus has the rubric)  #
# --------------------------------------------------------------------------- #

_BAND_RANK = {"": 0, "NONE": 0, "WATCH": 1, "ALERT": 2, "RED": 3}
_DOC_KIND = {"results": ("results", F.STRONG), "concall": ("concall", F.STRONG),
             "rating": ("rating", F.STRONG), "presentation": ("presentation", F.MEDIUM),
             "annual_report": ("annual_report", F.MEDIUM)}


_OUTCOME = re.compile(r"outcome of board|board meeting outcome", re.I)
_NOTICE = re.compile(r"intimation|notice of|schedule of|"
                     r"analyst.{0,12}meet|investor.{0,12}meet|newspaper|trading window",
                     re.I)
_FUTURE_RESULTS = re.compile(r"intimation.*(results|financial)|consider.*results", re.I)


def doc_kind(doc_type: str, title) -> tuple[str, int]:
    """Weight a processed document by what it IS, not only by its queue doc_type.
    The queue files board-meeting notices as results/concall (SPORTKING's 'Board
    Meeting Intimation ... to consider results' read as STRONG 'results processed')."""
    t = str(title or "")
    if _OUTCOME.search(t):            # decisions taken (issues, dividends, appointments)
        return "filing_med", F.MEDIUM
    if _NOTICE.search(t):
        if _FUTURE_RESULTS.search(t):
            return "results_due", F.MEDIUM
        return "filing_low", F.MINOR
    return _DOC_KIND.get(doc_type, ("filing_low", F.MINOR))


def _plain(s) -> str:
    """HTML-escaped snippet -> plain text for a trigger fact."""
    import html as _h
    return _h.unescape(re.sub(r"<[^>]+>", "", str(s or "")))


def collect_triggers(isin, sym, name, bse_code, ctx, hours, today) -> list[dict]:
    cut_iso = (datetime.now() - timedelta(hours=hours)).isoformat()
    cut_d = cut_iso[:10]
    T: list[dict] = []
    ann_url = (f"https://www.bseindia.com/corporates/ann.html?scrip={bse_code}"
               if bse_code else "https://www.bseindia.com/corporates/ann.html")

    # 1. exchange filings, weighted by the company-materiality the LLM tagged
    led = ctx["ann"]
    hit = pd.DataFrame()
    if not led.empty:
        m = (((led["isin"].astype(str) == isin)
              | (led["symbol"].astype(str).str.upper() == sym))
             & (led["status"].astype(str) == "done")
             & ((led["processed_at"].astype(str) >= cut_iso)
                | (led["ann_date"].astype(str) >= cut_d)))
        hit = led[m]
    for _, a in hit.head(6).iterrows():
        mat = str(a.get("materiality") or "").lower()
        kind, w = (("filing_high", F.STRONG) if mat == "high" else
                   ("filing_med", F.MEDIUM) if mat == "medium" else
                   ("filing_low", F.MINOR))
        T.append(F.trig(kind, w, f"{a.get('headline', '')} — "
                        f"{str(a.get('summary') or '')[:220]}", ann_url))
    if hit.empty:                     # ledger not caught up: the live list, unweighted
        try:
            from build_catalyst_notes import _recent_filings
            for f in _recent_filings(bse_code, max(1, round(hours / 24)))[:3]:
                T.append(F.trig("filing_low", F.MINOR, f[2:], ann_url))
        except Exception:
            pass

    # 2. research that arrived
    r = ctx["ridx"]
    if not r.empty and "isins" in r.columns:
        h = r[r["isins"].astype(str).str.contains(isin, na=False)
              & (r["processed_at"].astype(str) >= cut_iso)]
        for _, x in h.head(2).iterrows():
            T.append(F.trig("research", F.MINOR,
                            f"[{x.get('doc_type', '?')}] {x.get('file_name', '')}: "
                            f"{str(x.get('summary_md') or '')[:200]}"))

    # 3. PEAD is NOT a daily trigger: build_pead_flags re-stamps every verdict nightly
    #    (8,799 rows, all as_of today, horizons FY21-FY26), so "fresh" meant "always".
    #    Guidance vs actual lives in the per-company results mail (blocks D / D2).

    # 4. AR focus / defocus
    arf = ctx["arf"]
    if not arf.empty and "symbol" in arf.columns:
        h = arf[(arf["symbol"].astype(str).str.upper() == sym)
                & (arf["as_of"].astype(str) >= cut_d)]
        for _, x in h.head(2).iterrows():
            T.append(F.trig("ar_focus", F.MEDIUM,
                            f"AR {x.get('list')}: {str(x.get('reasons') or '')[:200]}"))

    # 5. fraud band WORSENED (not "has a band" — that repeated every day)
    fh = _by_isin(ctx["fth"], isin)
    if len(fh) >= 2:
        fh = fh.sort_values("as_of")
        cur, prev = fh.iloc[-1], fh.iloc[-2]
        if (str(cur["as_of"])[:10] >= cut_d and
                _BAND_RANK.get(str(cur["band"]).upper(), 0)
                > _BAND_RANK.get(str(prev["band"]).upper(), 0)):
            why = ""
            ft = _by_isin(ctx["ft"], isin)
            if not ft.empty:
                why = str(ft.iloc[0].get("reason") or "")[:180]
            T.append(F.trig("fraud", F.STRONG,
                            f"Fraud-tracker band {prev['band']} -> {cur['band']} "
                            f"(score {float(cur['fraud_score']):.0f}). {why}"))

    # 6. documents processed in the window, with their extracted content
    q = ctx["queue"]
    if not q.empty:
        h = q[(q["isin"].astype(str) == isin) & (q["status"].astype(str) == "done")
              & (q["processed_at"].astype(str) >= cut_iso)]
        for _, x in h.head(4).iterrows():
            dt_ = str(x.get("doc_type", ""))
            kind, w = doc_kind(dt_, x.get("title", ""))
            snip = _doc_snippet(dt_, x.get("doc_id"), isin, ctx)
            T.append(F.trig(kind, w, f"{dt_} processed: {x.get('title', '')}"
                            + (f" — {_plain(snip)}" if snip else "")))

    # 7. community
    try:
        for c in social_sources.community_items(name, days=max(1, round(hours / 24)))[:2]:
            T.append(F.trig("community", F.MINOR,
                            f"[{c['source']} · {c['author']}] {c['text'][:200]}", c["url"]))
    except Exception:
        pass

    # 8. a NEW catalyst - same tests as the catalyst mail (build_catalyst_notes):
    #    a real type, different from the previous note, not the same story re-worded
    cat = ctx["cat"]
    if not cat.empty and "symbol" in cat.columns:
        h = cat[cat["symbol"].astype(str).str.upper() == sym].sort_values("as_of")
        if not h.empty:
            x = h.iloc[-1]
            older = h[h["as_of"].astype(str) < str(x.get("as_of", ""))]
            prev = older.iloc[-1].to_dict() if not older.empty else None
            since = (today - timedelta(days=CN.SAME_STORY_DAYS)).isoformat()
            recent = [(str(r_.get("catalyst_type") or ""), str(r_.get("headline") or ""))
                      for _, r_ in older[older["as_of"].astype(str) >= since].iterrows()]
            ctype, hl = str(x.get("catalyst_type") or ""), str(x.get("headline") or "")
            if (str(x.get("as_of", "")) >= cut_d and ctype not in ("", "unknown")
                    and CN.note_changed(prev, ctype, hl)
                    and not CN.same_story(recent, ctype, hl)):
                T.append(F.trig("catalyst", F.MEDIUM, f"[{ctype}] {hl}"))

    # 9. results board meeting in the next 3 days
    cal = ctx["cal"]
    if not cal.empty and "symbol" in cal.columns:
        upto = (today + timedelta(days=3)).isoformat()
        h = cal[(cal["symbol"].astype(str).str.upper() == sym)
                & (cal["meeting_date"].astype(str) >= today.isoformat())
                & (cal["meeting_date"].astype(str) <= upto)]
        if not h.empty:
            T.append(F.trig("results_due", F.MEDIUM,
                            f"Results board meeting on {h['meeting_date'].astype(str).min()}"))

    # 10. guidance pace slipped in this week's snapshot
    gp = _by_isin(ctx["gp"], isin)
    if not gp.empty and "as_of" in gp.columns:
        h = gp[(gp["as_of"].astype(str) >= cut_d)
               & gp["status"].astype(str).isin(["BEHIND", "AT_RISK"])
               & pd.to_numeric(gp["delta_week"], errors="coerce").notna()]
        for _, x in h.head(2).iterrows():
            T.append(F.trig("guidance", F.MEDIUM,
                            f"{x.get('metric')} guidance '{str(x.get('guided_text'))[:80]}' "
                            f"({x.get('target_period')}): {x.get('pct_of_target')}% of target "
                            f"at {x.get('time_pct')}% of the horizon — {x.get('status')}"))

    # 11. unusual price move vs Nifty
    pt = F.price_trigger(ctx["ohlcv"].get(sym), ctx["nifty"], today)
    if pt:
        T.append(pt)
    return T


def _views(focus: dict, drive, index_id, use_llm: bool) -> dict:
    """One batched model call for every focus holding; rules when it is off/fails."""
    if not focus:
        return {}
    raw = ""
    if use_llm:
        try:
            from gemini_pool import load_keys_multi
            from _extractor_base import GeminiKeyPool, salvage_json_objects
            keys = load_keys_multi(os.environ, "FREE_POOL,BACKFILL_GEMINI_KEY,GEMINI_API_KEY")
            if keys:
                try:
                    from model_registry import resolve
                    models = resolve("LITE_UTILITY", drive, index_id) or ["gemini-2.5-flash-lite"]
                except Exception:
                    models = ["gemini-2.5-flash-lite"]
                pool = GeminiKeyPool(keys, models)
                raw = pool.call_text(F.build_prompt(focus), "pf_digest_focus",
                                     max_output_tokens=6000)
                views = F.parse_views(raw, focus, salvage_json_objects)
                n = sum(1 for v in views.values() if v["source"] == "model")
                log(f"view: model wrote {n}/{len(focus)} (rest rule-based)")
                return views
            log("view: no keys (FREE_POOL/BACKFILL/GEMINI) — rule-based")
        except Exception as e:
            log(f"view: model call failed ({str(e)[:80]}) — rule-based")
    return F.parse_views("", focus)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="no model call, no mail; preview html written locally")
    ap.add_argument("--preview", action="store_true",
                    help="WITH the model call, but no mail; preview html written locally")
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    pf = load_portfolio_isins(drive, root) or set()
    if not pf:
        log("no portfolio file — nothing to do.")
        return
    uni = _read(drive, root, ["company_repo", "_index", "company_universe.csv"])
    rows = []
    for _, r in uni.iterrows():
        isin = str(r.get("isin", "")).strip()
        if isin in pf:
            # A missing NSE symbol reads as float NaN, which is TRUTHY and str()s to
            # "nan" - BSE-only LTELEVATOR became "NAN" and lost its chart and name.
            sym = (_txt(r.get("nse_symbol")) or _txt(r.get("bse_symbol"))
                   or _txt(r.get("bse_code")).split(".")[0] or isin).upper()
            rows.append((isin, sym, str(r.get("name", "")).strip(),
                         str(r.get("bse_code", "")).strip()))
    log(f"PF companies resolved: {len(rows)} of {len(pf)} ISINs")

    P = ["company_repo", "_index"]
    qfid = find_file(drive, index_id, "processing_queue.parquet")
    from build_gallery import _bulk_parquet, _folder
    syms = sorted({s for _, s, _, _ in rows})
    ctx = {
        "queue": (pd.read_parquet(io.BytesIO(download_bytes(drive, qfid)))
                  if qfid else pd.DataFrame()),
        "ann":  _read(drive, root, P + ["announcement_ledger.parquet"]),
        "ridx": _read(drive, root, P + ["research_index.parquet"]),
        "arf":  _read(drive, root, P + ["ar_focus.parquet"]),
        "ft":   _read(drive, root, P + ["fraud_tracker.parquet"]),
        "fth":  _read(drive, root, P + ["fraud_tracker_history.parquet"]),
        "cat":  _read(drive, root, P + ["catalyst_index.parquet"]),
        "cal":  _read(drive, root, P + ["results_calendar.parquet"]),
        "gp":   _read(drive, root, P + ["guidance_progress.parquet"]),
        # content snippets for processed docs (user 2026-07-08)
        "rgem": _read(drive, root, P + ["results_gemini.parquet"]),
        "guid": _read(drive, root, P + ["guidance_tracker.parquet"]),
        "gf1":  _read(drive, root, P + ["gf1_guidance_statements.parquet"]),
        "ppt":  _read(drive, root, P + ["ppt_guidance.parquet"]),
        "nifty": _read(drive, root, ["data", "indices", "NIFTY_50.parquet"]),
    }
    try:
        ctx["ohlcv"] = _bulk_parquet(drive, _folder(drive, "data/ohlcv"), syms) or {}
    except Exception as e:
        log(f"OHLCV load failed ({str(e)[:70]}) — no price triggers or charts")
        ctx["ohlcv"] = {}
    log("inputs: " + ", ".join(f"{k}={len(v)}" for k, v in ctx.items()
                               if isinstance(v, pd.DataFrame)))

    today = date.today()
    trig_by, names = {}, {}
    for isin, sym, name, bse_code in rows:
        trig_by[sym] = collect_triggers(isin, sym, name, bse_code, ctx, args.hours, today)
        names[sym] = name
    score = {s: F.attention(t) for s, t in trig_by.items()}
    focus = {s: trig_by[s] for s in sorted(score, key=lambda s: (-score[s], s))
             if score[s] >= F.FOCUS_MIN}
    minor = [s for s in sorted(score) if 0 < score[s] < F.FOCUS_MIN]
    quiet = [s for s in sorted(score) if score[s] == 0]
    for s in focus:
        log(f"  {s:<14} attention {score[s]:>2}  "
            + ", ".join(t["kind"] for t in focus[s]))
    log(f"focus={len(focus)} minor={len(minor)} quiet={len(quiet)}")

    views = _views(focus, drive, index_id, use_llm=not args.dry_run)

    # charts for the focus cards (same cap as the combined-strength mail)
    import mail_chart
    imgs, cards = [], []
    for i, s in enumerate(focus):
        ch = ""
        if i < CHART_N:
            try:
                png = mail_chart.render_png(ctx["ohlcv"].get(s), s)
            except Exception:
                png = b""
            if png:
                cid = "pf_" + re.sub(r"[^A-Za-z0-9]", "_", s)
                imgs.append((cid, png))
                ch = mail_chart.img_tag(cid, s)
        cards.append(F.focus_card(s, names[s], "", focus[s], views[s], ch))

    head = (f"<div style='font:13px Arial,sans-serif;max-width:720px'>"
            f"<h2 style='margin:0 0 4px'>💼 PF focus — {today:%d %b %Y}</h2>"
            f"<p style='margin:0 0 10px;color:#444'><b>{len(focus)} holding"
            f"{'' if len(focus) == 1 else 's'} need a look</b>, ranked by attention "
            f"· {len(minor)} with a minor update · {len(quiet)} quiet. Only what "
            f"changed in the last {args.hours:.0f}h counts.</p>")
    body = (head + ("".join(cards) if cards else
                    "<p>Nothing that needs a look today.</p>")
            + (f"<h3 style='font-size:13px;margin:12px 0 4px'>Minor updates</h3><ul>"
               + "".join(F.minor_line(s, trig_by[s]) for s in minor) + "</ul>"
               if minor else "")
            + (f"<p style='color:#777'><b>Quiet:</b> {esc(', '.join(quiet), 1500)}</p>"
               if quiet else "")
            + "<p style='font-size:11px;color:#999'>Attention = strong 3 (material "
              "filing, results/concall, unusual price move, fraud flag worse, "
              "rating) · medium 2 (other filing, deck/AR, new catalyst, "
              f"results due ≤3 days, guidance pace slipping) · minor 1. A card needs "
              f"{F.FOCUS_MIN}+. Unusual move = ≥{F.PRICE_Z:g}× the stock's own typical "
              f"move vs Nifty, or a ≥{F.PRICE_ABS_5D:g}% move vs Nifty over 5 days. "
              "Views are written only from the evidence listed; a view "
              "marked rule-based is the fallback. Signals only, not advice. Toggle this "
              "mail in the app sidebar.</p></div>")

    if args.dry_run or args.preview:
        import base64
        prev_html = body
        for cid, png in imgs:
            prev_html = prev_html.replace(
                f"cid:{cid}", "data:image/png;base64," + base64.b64encode(png).decode())
        prev = Path(__file__).resolve().parent.parent / "pf_digest_preview.html"
        prev.write_text(prev_html, encoding="utf-8")
        log(f"{'DRY-RUN' if args.dry_run else 'PREVIEW'} — {prev.name} "
            f"({len(body.encode()) / 1024:.0f} KB html, {len(imgs)} charts); no mail.")
        return
    if not load_mail_settings(drive, index_id).get("pf_digest", True):
        log("pf_digest mail toggled OFF — skipped.")
        return
    send_email(f"💼 PF focus — {len(focus)} need a look · {len(minor)} minor — "
               f"{today:%d %b}", body, inline_images=imgs)


CHART_N = 15


def _self_test() -> int:
    fails = F._self_test()
    ok = True
    for title, dt_, want in [
        ("Board Meeting Intimation for Consider And Approve The Unaudited Financial "
         "Results Of The Company For The Quarter", "results", "results_due"),
        ("Announcement under Regulation 30 (LODR)-Analyst / Investor Meet - Intimation",
         "concall", "filing_low"),
        ("Board Meeting Outcome for Outcome Of Board Meeting Held On October 05, 2026",
         "results", "filing_med"),
        ("Q2 FY27 Earnings Call Transcript", "concall", "concall"),
        ("Financial Results for the quarter ended 30 Sep 2026", "results", "results"),
    ]:
        got = doc_kind(dt_, title)[0]
        good = got == want
        ok &= good
        print(("  ok   " if good else "  FAIL ") + f"doc_kind {want:<12} <- {title[:50]}")
    ctx = {"guid": pd.DataFrame([{"isin": "I1", "source_doc_id": "OLD", "metric": "rev",
                                  "value": "70-80 cr", "horizon_fy": "1Y"}])}
    snip = _doc_snippet("concall", "NEW", "I1", ctx | {"gf1": pd.DataFrame()})
    good = snip is None
    ok &= good
    print(("  ok   " if good else "  FAIL ") + "no snippet borrowed from another document")
    print("ALL PASS" if (ok and not fails) else "FAILED")
    return 0 if (ok and not fails) else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    main()
