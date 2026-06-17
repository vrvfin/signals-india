r"""
build_screener_grades.py — precompute the 6-rule attractiveness heatmap.

One row per company with the raw numbers + color tier for each of:
  YOY (PAT, latest qtr) · QOQ (PAT, latest qtr) · Guidance growth ·
  Valuation (PE; PB/EV-EBITDA per sector = v2) · CFO quality · ROE
plus green_count = how many are Good-or-better. app.py renders this as a color
heatmap sorted by green_count (no heavy compute in Streamlit -> OOM-safe).

Inputs (all already on Drive):
  _index/financials_derived.parquet  (full universe now)  -> YOY/QOQ/ROE/CFO
  _index/valuation.parquet           -> PE
  _index/guidance_tracker.parquet    -> guidance growth (cagr_pct)
  fundamentals/summary.parquet       -> company_name

Usage:
  python scripts/build_screener_grades.py --dry-run
  python scripts/build_screener_grades.py
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)
import gradation as G

OUT_COLS = ["isin", "symbol", "company_name",
            "yoy", "yoy_tier", "qoq", "qoq_tier",
            "guidance", "guidance_tier",
            "val_metric", "val_value", "val_tier",
            "cfo_ratio", "cfo_tier", "roe", "roe_tier",
            "green_count", "computed_at"]


def _folder(drive, parts):
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _rp(drive, folder, name):
    fid = find_file(drive, folder, name)
    return pd.read_parquet(io.BytesIO(download_bytes(drive, fid))) if fid else pd.DataFrame()


def _latest_by_metric(derived: pd.DataFrame) -> dict:
    """{isin: {metric: latest_value}} — last row per (isin,metric) preserves
    derive_company's chronological emit order."""
    out: dict = {}
    if derived.empty:
        return out
    d = derived[derived["period_type"] == "quarterly"] if "period_type" in derived else derived
    # quarterly for YOY/QOQ; annual for ROE/CFO -> handle both by metric below
    for _, r in derived.iterrows():
        iso = str(r.get("isin", "")).strip()
        if not iso:
            continue
        out.setdefault(iso, {})[r["metric"]] = r["value"]   # last wins = latest
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    drive = get_drive()
    idx = _folder(drive, "company_repo/_index")
    derived = _rp(drive, idx, "financials_derived.parquet")
    val = _rp(drive, idx, "valuation.parquet")
    guid = _rp(drive, idx, "guidance_tracker.parquet")
    summ = _rp(drive, _folder(drive, "fundamentals"), "summary.parquet")
    log(f"inputs: derived={len(derived)} val={len(val)} guidance={len(guid)} summary={len(summ)}")

    latest = _latest_by_metric(derived)

    pe_by_isin = {}
    if not val.empty and "isin" in val.columns:
        for _, r in val.iterrows():
            pe_by_isin[str(r["isin"]).strip()] = r.get("pe")
    name_by_isin, sym_by_isin = {}, {}
    if not summ.empty:
        for _, r in summ.iterrows():
            pass  # summary is symbol-keyed; names filled from derived/val below

    # guidance: best (max) cagr_pct per isin
    guid_by_isin = {}
    if not guid.empty and "isin" in guid.columns and "cagr_pct" in guid.columns:
        gg = guid.copy()
        gg["cagr_pct"] = pd.to_numeric(gg["cagr_pct"], errors="coerce")
        guid_by_isin = (gg.dropna(subset=["cagr_pct"]).groupby(
            gg["isin"].astype(str).str.strip())["cagr_pct"].max().to_dict())

    # symbol/name lookups from valuation (has both) then derived
    for df in (val, derived):
        if df.empty:
            continue
        for _, r in df.iterrows():
            iso = str(r.get("isin", "")).strip()
            if iso and iso not in sym_by_isin:
                sym_by_isin[iso] = str(r.get("symbol", "")).strip()
                name_by_isin[iso] = str(r.get("company_name", "") or "").strip()

    isins = set(latest) | set(pe_by_isin) | set(guid_by_isin)
    isins.discard("")
    rows = []
    now = datetime.now().isoformat(timespec="seconds")
    for iso in isins:
        m = latest.get(iso, {})
        yoy = m.get("pat_yoy_pct")
        qoq = m.get("pat_qoq_pct")
        roe = m.get("roe_pct")
        cfo = m.get("cfo_pat_ratio")
        gdv = guid_by_isin.get(iso)
        pe = pe_by_isin.get(iso)
        tiers = {
            "yoy": G.grade_growth(yoy), "qoq": G.grade_growth(qoq),
            "guidance": G.grade_growth(gdv) if gdv is not None else "na",
            "val": G.grade_valuation(pe, "pe"),
            "cfo": G.grade_cfo(cfo), "roe": G.grade_roe(roe),
        }
        rows.append({
            "isin": iso, "symbol": sym_by_isin.get(iso, ""),
            "company_name": name_by_isin.get(iso, ""),
            "yoy": yoy, "yoy_tier": tiers["yoy"],
            "qoq": qoq, "qoq_tier": tiers["qoq"],
            "guidance": gdv, "guidance_tier": tiers["guidance"],
            "val_metric": "PE", "val_value": pe, "val_tier": tiers["val"],
            "cfo_ratio": cfo, "cfo_tier": tiers["cfo"],
            "roe": roe, "roe_tier": tiers["roe"],
            "green_count": G.green_count(list(tiers.values())),
            "computed_at": now,
        })

    out = pd.DataFrame(rows, columns=OUT_COLS).sort_values(
        "green_count", ascending=False).reset_index(drop=True)
    log(f"graded {len(out)} companies")
    for col in ("yoy", "qoq", "guidance", "val", "cfo", "roe"):
        gt = out[f"{col if col!='val' else 'val'}_tier"]
        green = gt.isin(G.GREEN_TIERS).sum()
        log(f"  green on {col:<9}: {green}")

    if args.dry_run:
        log("DRY-RUN — no write. Top 5 by green_count:")
        for _, r in out.head(5).iterrows():
            log(f"  {r['symbol'][:12]:<12} green={r['green_count']} "
                f"yoy={r['yoy_tier']} qoq={r['qoq_tier']} val={r['val_tier']} "
                f"cfo={r['cfo_tier']} roe={r['roe_tier']}")
        return

    upload_bytes(drive, idx, "screener_grades.parquet",
                 out.to_parquet(index=False), "application/octet-stream",
                 existing_id=find_file(drive, idx, "screener_grades.parquet"))
    log(f"wrote _index/screener_grades.parquet ({len(out)} rows)")


if __name__ == "__main__":
    main()
