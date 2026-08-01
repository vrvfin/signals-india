r"""
build_company_facts.py — T10: per-company FACT table for the ENTIRE listed
universe (NSE + BSE), ISIN-keyed, plus peer/segment median aggregates.

One raw row per company; numbers assembled from existing parquets (no new
fetch). Fields fill in as the pipelines cover more names; BSE-only names that
no source covers yet stay blank until the rolling Screener enrichment lands.

  company_repo/_index/company_facts.parquet (+ .csv)   — raw, one row/company
  company_repo/_index/peer_aggregates.parquet          — medians by
      peer_group / subsector / sector / segment

Columns (FACT_COLS): identity + classification + the metrics the user asked for
  mcap_cr, pe, pb,
  ret_3m_pct, ret_6m_pct, ret_12m_pct, vol_20d_avg, vol_today_ratio,
  rev_q, rev_q_yoy, rev_q_qoq, pat_q, pat_q_yoy, pat_q_qoq,
  eps_q, eps_q_yoy, eps_q_qoq,
  rev_ttm, pat_ttm, eps_ttm,
  latest_q, source, updated_at

Sources: summary.parquet (mcap/pe/book + q arrays), features/latest (price moves
+ volume), results.parquet (explicit Q value + YoY + QoQ — authoritative),
classification.csv (segment/peer_group/sector/subsector), universe (isin map).

Usage:
    python scripts/build_company_facts.py --dry-run     # coverage + sample
    python scripts/build_company_facts.py
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

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)

FACT_COLS = [
    "isin", "symbol", "name", "segment", "macro_sector", "sector",
    "subsector", "peer_group",
    "mcap_cr", "pe", "pb",
    "ret_3m_pct", "ret_6m_pct", "ret_12m_pct", "vol_20d_avg", "vol_today_ratio",
    "rev_q", "rev_q_yoy", "rev_q_qoq", "pat_q", "pat_q_yoy", "pat_q_qoq",
    "eps_q", "eps_q_yoy", "eps_q_qoq", "rev_ttm", "pat_ttm", "eps_ttm",
    "latest_q", "source", "updated_at",
]
GROUP_LEVELS = ["peer_group", "subsector", "sector", "segment"]
NUM_METRICS = ["mcap_cr", "pe", "pb", "ret_3m_pct", "ret_6m_pct", "ret_12m_pct",
               "rev_q_yoy", "rev_q_qoq", "pat_q_yoy", "pat_q_qoq",
               "eps_q_yoy", "eps_q_qoq"]


def _read(drive, root, parts):
    fid = root
    for p in parts[:-1]:
        fid = get_or_create_subfolder(drive, fid, p)
    f = find_file(drive, fid, parts[-1])
    if not f:
        return None
    raw = download_bytes(drive, f)
    return (pd.read_csv(io.BytesIO(raw)) if parts[-1].endswith(".csv")
            else pd.read_parquet(io.BytesIO(raw)))


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _pct(v):
    """A growth/return %, with an absurd-value guard (Screener parse noise can
    yield e.g. -2e10). Null anything beyond +/-10000%."""
    f = _num(v)
    return None if (f is None or abs(f) > 10000) else f


def _ttm(arr):
    try:
        a = [float(x) for x in arr if str(x) not in ("nan", "None", "")]
        return round(sum(a), 2) if a else None
    except Exception:
        return None


def _q_last(arr):
    """Latest quarter from a Screener 4-quarter array. The array is oldest-first, so
    the LAST element is the most recent quarter — verified against
    summary.latest_quarter_eps, which equals q_eps_last_4q[-1]."""
    try:
        a = [float(x) for x in arr if str(x) not in ("nan", "None", "")]
        return round(a[-1], 2) if a else None
    except Exception:
        return None


def _pivot_results(res: pd.DataFrame) -> dict[str, dict]:
    """isin -> {rev_q, rev_q_yoy, rev_q_qoq, pat_*, eps_*, latest_q} from the
    explicit Screener results table (authoritative Q value + YoY + QoQ)."""
    if res is None or res.empty:
        return {}
    metric_map = {"sales": "rev", "revenue": "rev", "net profit": "pat",
                  "profit after tax": "pat", "pat": "pat", "eps": "eps"}
    out: dict[str, dict] = {}
    for _, r in res.iterrows():
        isin = str(r.get("isin", "")).strip()
        if not isin:
            continue
        m = str(r.get("metric", "")).lower().strip()
        key = next((v for k, v in metric_map.items() if k in m), None)
        if not key:
            continue
        d = out.setdefault(isin, {"latest_q": str(r.get("latest_q", ""))})
        d[f"{key}_q"] = _num(r.get("latest_val"))
        d[f"{key}_q_yoy"] = _pct(r.get("yoy_pct"))
        d[f"{key}_q_qoq"] = _pct(r.get("qoq_pct"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    index_id = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")
    today = datetime.now().strftime("%Y-%m-%d")

    uni = _read(drive, root, ["company_repo", "_index", "company_universe.csv"])
    if uni is None or uni.empty:
        log("universe missing.")
        return
    sym_col = "nse_symbol" if "nse_symbol" in uni.columns else "symbol"
    # BSE-only names have an isin + bse_code + bse_symbol but NO nse_symbol, so keying
    # solely on nse_symbol silently dropped 2,667 of 5,631 universe rows — company_facts
    # covered 53% of the universe while claiming the whole of it. Measured 2026-07-30:
    # 2,604 of those 2,667 already have a fundamentals/summary.parquet row and 2,378
    # already have a fundamentals/statements/ file, so the data was present and merely
    # unjoined. Falling back to bse_symbol takes coverage to ~99%.
    # Existing NSE rows are unaffected: nse_symbol still wins whenever it is present.
    alt_sym_col = "bse_symbol" if "bse_symbol" in uni.columns else None

    summ = _read(drive, root, ["fundamentals", "summary.parquet"])
    summ_by_sym = ({str(r["symbol"]).upper(): r for _, r in summ.iterrows()}
                   if summ is not None else {})
    feat = _read(drive, root, ["features", "latest.parquet"])
    feat_by_sym = ({str(r["symbol"]).upper(): r for _, r in feat.iterrows()}
                   if feat is not None else {})
    res_by_isin = _pivot_results(
        _read(drive, root, ["company_repo", "_index", "results.parquet"]))
    cls = _read(drive, root, ["company_repo", "_index", "company_classification.csv"])
    cls_by_isin = ({str(r["isin"]): r for _, r in cls.fillna("").iterrows()}
                   if cls is not None else {})

    rows = []
    n_alt = 0
    for _, u in uni.iterrows():
        isin = str(u.get("isin", "")).strip()
        sym = str(u.get(sym_col, "")).strip().upper()
        listing = "nse"
        if sym in ("", "NAN") and alt_sym_col:
            alt = str(u.get(alt_sym_col, "")).strip().upper()
            if alt not in ("", "NAN"):
                sym, listing, n_alt = alt, "bse", n_alt + 1
        if not isin or sym in ("", "NAN"):
            continue
        s = summ_by_sym.get(sym, {})
        f = feat_by_sym.get(sym, {})
        rq = res_by_isin.get(isin, {})
        c = cls_by_isin.get(isin, {})

        def sg(src, k):
            try:
                return src[k] if k in src else (src.get(k) if hasattr(src, "get") else None)
            except Exception:
                return None

        price = _num(sg(f, "close"))
        bv = _num(sg(s, "book_value"))
        src = "+".join(x for x, ok in
                       (("summary", sym in summ_by_sym),
                        ("features", sym in feat_by_sym),
                        ("results", bool(rq))) if ok) or "none"
        rows.append({
            "isin": isin, "symbol": sym, "name": str(u.get("name", "")).strip(),
            "segment": str(c.get("segment", "")) if c is not None else "",
            "macro_sector": str(c.get("macro_sector", "")) if c is not None else "",
            "sector": str(c.get("sector", "")) if c is not None else "",
            "subsector": str(c.get("subsector", "")) if c is not None else "",
            "peer_group": str(c.get("peer_group", "")) if c is not None else "",
            "mcap_cr": _num(sg(s, "market_cap_cr")),
            "pe": _num(sg(s, "pe")),
            "pb": round(price / bv, 2) if (price and bv) else None,
            "ret_3m_pct": _num(sg(f, "return_3m_pct")),
            "ret_6m_pct": _num(sg(f, "return_6m_pct")),
            "ret_12m_pct": _num(sg(f, "return_12m_pct")),
            "vol_20d_avg": _num(sg(f, "vol_20d_avg")),
            "vol_today_ratio": _num(sg(f, "vol_today_ratio")),
            # results.parquet is fed by Screener's /results/latest/ feed, which is a
            # 25-item WINDOW — anything scrolling out between runs is never captured
            # (160 of 819 Q1-FY27 reporters were absent on 2026-08-01). Those same
            # quarters ARE present in summary.parquet's 4-quarter arrays, which are
            # already loaded here, so fall back to the last array element rather than
            # leaving the latest-quarter line blank. results.parquet still WINS when
            # present: it carries YoY/QoQ that a 4-quarter array cannot express.
            "rev_q": rq.get("rev_q", _q_last(sg(s, "q_sales_last_4q"))),
            "rev_q_yoy": rq.get("rev_q_yoy"),
            "rev_q_qoq": rq.get("rev_q_qoq"),
            "pat_q": rq.get("pat_q", _q_last(sg(s, "q_netprofit_last_4q"))),
            "pat_q_yoy": rq.get("pat_q_yoy"),
            "pat_q_qoq": rq.get("pat_q_qoq"),
            "eps_q": rq.get("eps_q", _q_last(sg(s, "q_eps_last_4q"))),
            "eps_q_yoy": rq.get("eps_q_yoy", _num(sg(s, "q_eps_yoy_pct"))),
            "eps_q_qoq": rq.get("eps_q_qoq"),
            "rev_ttm": _ttm(sg(s, "q_sales_last_4q")),
            "pat_ttm": _ttm(sg(s, "q_netprofit_last_4q")),
            "eps_ttm": _ttm(sg(s, "q_eps_last_4q")),
            "latest_q": rq.get("latest_q") or str(sg(s, "latest_quarter_label") or ""),
            # `source` records which parquets fed the row; the listing tag records WHICH
            # symbol namespace the row was keyed on, so a consumer can tell an NSE row
            # from a BSE-only one without re-deriving it from the universe.
            "source": (src + "|bse") if listing == "bse" else src,
            "updated_at": today,
        })

    df = pd.DataFrame(rows, columns=FACT_COLS)
    cov = {c: int(df[c].notna().sum()) for c in
           ["mcap_cr", "pe", "ret_12m_pct", "rev_q_yoy", "pat_q_yoy", "rev_ttm"]}
    log(f"company_facts: {len(df)} rows "
        f"({len(df) - n_alt} keyed on {sym_col}, {n_alt} on {alt_sym_col or 'n/a'}). "
        f"coverage {cov}")

    # ---- peer/segment median aggregates ----
    agg_rows = []
    for level in GROUP_LEVELS:
        g = df[df[level].astype(str).str.len() > 0]
        for name, members in g.groupby(level):
            if not str(name) or str(name).lower() == "nan":
                continue
            rec = {"level": level, "group": name, "n": len(members)}
            for m in NUM_METRICS:
                rec[f"{m}_median"] = round(members[m].median(), 2) \
                    if members[m].notna().any() else None
            agg_rows.append(rec)
    agg = pd.DataFrame(agg_rows)

    if args.dry_run:
        log("DRY-RUN sample (covered names):")
        sample = df[df["mcap_cr"].notna() & df["rev_q_yoy"].notna()].head(8)
        cols = ["symbol", "peer_group", "mcap_cr", "pe", "ret_12m_pct",
                "rev_q_yoy", "pat_q_yoy", "eps_q_yoy", "rev_ttm"]
        print(sample[cols].to_string(index=False))
        log("peer-aggregate sample:")
        print(agg[agg["level"] == "peer_group"].head(6).to_string(index=False))
        return

    for name, frame in (("company_facts", df), ("peer_aggregates", agg)):
        buf = io.BytesIO()
        frame.to_parquet(buf, index=False)
        upload_bytes(drive, index_id, f"{name}.parquet", buf.getvalue(),
                     "application/octet-stream",
                     existing_id=find_file(drive, index_id, f"{name}.parquet"))
    upload_bytes(drive, index_id, "company_facts.csv",
                 df.to_csv(index=False).encode("utf-8"), "text/csv",
                 existing_id=find_file(drive, index_id, "company_facts.csv"))
    log(f"wrote company_facts (.parquet/.csv) + peer_aggregates ({len(agg)} groups).")


if __name__ == "__main__":
    main()
