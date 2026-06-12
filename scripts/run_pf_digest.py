"""
run_pf_digest.py — PF companies daily digest mail (user spec 2026-06-12).

One crisp section per PORTFOLIO company that had ANY activity in the window:
  1. 📜 corporate announcements (BSE, with a direct source link)
  2. 📰 research coverage that arrived (research_index / Workflow A)
  3. 🧪 capability flags: scorecard composite, fraud-tracker band+why,
     fresh PEAD verdicts, AR focus/defocus verdicts
  4. 📄 docs processed (AR / rating / concall / results / presentation titles)
  5. 🗣 community: ValuePickr top contributors / curated blogs / X — each with
     its direct link (social_sources.py)
  6. 💡 latest catalyst note headline + what-to-track

Quiet companies are listed in one summary line. Toggle: 'pf_digest'.
Runs at the end of t4_nightly (catalysts fresh). NO Gemini.

Usage:
    python scripts/run_pf_digest.py --dry-run     # build, save preview, no mail
    python scripts/run_pf_digest.py --hours 24
"""

from __future__ import annotations

import argparse
import io
import os
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


def _by_isin(df, isin, col="isin"):
    if df.empty or col not in df.columns:
        return df.iloc[0:0] if not df.empty else pd.DataFrame()
    return df[df[col].astype(str) == isin]


def company_section(isin, sym, name, bse_code, ctx, hours) -> str | None:
    """HTML <li> block for one PF company, or None if fully quiet."""
    cutoff_iso = (datetime.now() - timedelta(hours=hours)).isoformat()
    bullets = []

    # 1. announcements (BSE, with source link)
    try:
        from build_catalyst_notes import _recent_filings
        # strict 24h block (user 2026-06-12): date-granular sources include
        # only items dated within the block
        filings = _recent_filings(bse_code, max(1, round(hours / 24)))
    except Exception:
        filings = []
    ann_url = (f"https://www.bseindia.com/corporates/ann.html?scrip={bse_code}"
               if bse_code else "https://www.bseindia.com/corporates/ann.html")
    for f in filings[:4]:
        bullets.append(f"📜 {esc(f[2:], 170)} "
                       f"(<a href='{ann_url}'>source</a>)")

    # 2. research coverage
    r = ctx["ridx"]
    if not r.empty and "isins" in r.columns:
        hit = r[r["isins"].astype(str).str.contains(isin, na=False)
                & (r["processed_at"].astype(str) >= cutoff_iso)]
        for _, x in hit.head(3).iterrows():
            bullets.append(f"📰 research: [{esc(x.get('doc_type', '?'), 20)}] "
                           f"{esc(x.get('file_name', ''), 70)} — "
                           f"{esc(x.get('summary_md', ''), 180)}")

    # 3. capability flags
    flags = []
    pead = _by_isin(ctx["pead"], isin)
    if not pead.empty and "as_of" in pead.columns:
        fresh = pead[pead["as_of"].astype(str) >= cutoff_iso[:10]]
        flags += [f"PEAD {x.get('metric')}: <b>{x.get('verdict')}</b> "
                  f"({x.get('delta_pct')}%)" for _, x in fresh.head(3).iterrows()]
    arf = ctx["arf"]
    if not arf.empty:
        hit = arf[(arf["symbol"].astype(str).str.upper() == sym)
                  & (arf["as_of"].astype(str) >= cutoff_iso[:10])]
        flags += [f"AR <b>{x.get('list')}</b>: {esc(x.get('reasons', ''), 150)}"
                  for _, x in hit.iterrows()]
    ft = ctx["ft"]
    if not ft.empty:
        hit = ft[ft["symbol"].astype(str).str.upper() == sym]
        if not hit.empty:
            x = hit.iloc[0]
            flags.append(f"fraud tracker <b>{x.get('band')}</b> "
                         f"{float(x.get('fraud_score', 0)):.0f}: "
                         f"{esc(x.get('reason', ''), 140)}")
    if flags:
        bullets.append("🧪 " + " · ".join(flags))

    # 4. docs processed in window
    q = ctx["queue"]
    if not q.empty:
        hit = q[(q["isin"].astype(str) == isin)
                & (q["status"].astype(str) == "done")
                & (q["processed_at"].astype(str) >= cutoff_iso)]
        for _, x in hit.head(4).iterrows():
            bullets.append(f"📄 {esc(x.get('doc_type', '?'), 16)} processed: "
                           f"{esc(x.get('title', ''), 110)}")

    # 5. community (links included) — same strict 24h block
    for c in social_sources.community_items(name,
                                            days=max(1, round(hours / 24)))[:3]:
        bullets.append(f"🗣 [{esc(c['source'], 14)} · {esc(c['author'], 20)}] "
                       f"{esc(c['text'], 170)} (<a href='{esc(c['url'], 200)}'>"
                       f"source</a>)")

    # 6. latest catalyst
    cat = ctx["cat"]
    if not cat.empty:
        hit = cat[cat["symbol"].astype(str).str.upper() == sym] \
            .sort_values("as_of").tail(1)
        for _, x in hit.iterrows():
            if str(x.get("as_of", "")) >= cutoff_iso[:10]:
                line = (f"💡 catalyst [{x.get('catalyst_type')}]: "
                        f"{esc(x.get('headline', ''), 150)}")
                track = str(x.get("what_to_track", "") or "")
                if track and track.lower() != "nan":
                    line += f"<br>&nbsp;&nbsp;👁 {esc(track, 180)}"
                bullets.append(line)

    if not bullets:
        return None
    sc = ctx["sc"]
    badge = ""
    if not sc.empty:
        hit = sc[sc["symbol"].astype(str).str.upper() == sym]
        if not hit.empty and pd.notna(hit.iloc[0].get("composite_score")):
            badge = f" — composite {float(hit.iloc[0]['composite_score']):.0f}/100"
    return (f"<h3 style='margin:14px 0 4px 0'>{esc(sym)} · {esc(name, 40)}"
            f"{badge}</h3><ul style='margin:2px 0'>"
            + "".join(f"<li style='margin:3px 0'>{b}</li>" for b in bullets)
            + "</ul>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--dry-run", action="store_true")
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
            rows.append((isin, str(r.get("nse_symbol", "")).strip().upper(),
                         str(r.get("name", "")).strip(),
                         str(r.get("bse_code", "")).strip()))
    log(f"PF companies resolved: {len(rows)} of {len(pf)} ISINs")

    P = ["company_repo", "_index"]
    qfid = find_file(drive, index_id, "processing_queue.parquet")
    ctx = {
        "queue": (pd.read_parquet(io.BytesIO(download_bytes(drive, qfid)))
                  if qfid else pd.DataFrame()),
        "ridx": _read(drive, root, P + ["research_index.parquet"]),
        "pead": _read(drive, root, P + ["pead_flags.parquet"]),
        "arf":  _read(drive, root, P + ["ar_focus.parquet"]),
        "ft":   _read(drive, root, P + ["fraud_tracker.parquet"]),
        "cat":  _read(drive, root, P + ["catalyst_index.parquet"]),
        "sc":   _read(drive, root, P + ["company_scorecard.parquet"]),
    }

    sections, quiet = [], []
    for isin, sym, name, bse_code in sorted(rows, key=lambda x: x[1]):
        sec = company_section(isin, sym, name, bse_code, ctx, args.hours)
        (sections if sec else quiet).append(sec or sym)
        log(f"  {sym:<14} {'ACTIVE' if sec else 'quiet'}")

    body = (f"<p><b>Portfolio daily digest</b> — last {args.hours:.0f}h. "
            f"{len(sections)} of {len(rows)} PF companies had activity.</p>"
            + "".join(sections)
            + (f"<p style='color:#777'><b>Quiet:</b> "
               f"{esc(', '.join(quiet), 1500)}</p>" if quiet else "")
            + "<p style='font-size:11px;color:#999'>Sources are linked inline "
              "(BSE filings, ValuePickr, blogs). Toggle this mail in the app "
              "sidebar (📧 Email toggles).</p>")

    if args.dry_run:
        prev = Path(__file__).resolve().parent.parent / "pf_digest_preview.html"
        prev.write_text(body, encoding="utf-8")
        log(f"DRY-RUN — preview saved to {prev.name}; no mail.")
        return
    if not load_mail_settings(drive, index_id).get("pf_digest", True):
        log("pf_digest mail toggled OFF — skipped.")
        return
    send_email(f"💼 PF daily digest — {len(sections)} active / "
               f"{len(rows)} — {date.today()}", body)


if __name__ == "__main__":
    main()
