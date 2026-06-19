r"""
coverage_report.py — READ-ONLY backfill / pipeline coverage snapshot.

Pulls every parquet under company_repo/_index/ and fundamentals/ from Drive plus
the universe CSVs, and reports, per table:
  - row count
  - distinct companies (by isin, then symbol/nse_symbol as fallback)
  - for processing_queue: doc_type x status x source breakdown + distinct ISIN done

Then prints a coverage matrix vs the universe (distinct ISIN that appear anywhere).

NO writes. NO Gemini. Safe to run anytime.

    python scripts/coverage_report.py
    python scripts/coverage_report.py --doc-type annual_report --target 3
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Reuse the exact Drive helpers Phase 2 uses — no new auth path.
from write_phase2_status import (
    get_drive, find_sub, find_file, download_bytes,
)

ID_COLS = ["isin", "ISIN", "symbol", "nse_symbol", "Symbol"]

# Per-doc-type "covered to target depth" thresholds (count of status=done docs).
# concall/results/presentation are quarterly (8q ≈ 2y); AR/rating are annual.
DEPTH_TARGET = {"concall": 8, "results": 8, "presentation": 8,
                "annual_report": 3, "rating": 3}
COVERAGE_MD = "coverage_report.md"


def list_files(drive, folder_id, suffix=".parquet"):
    """All non-trashed files in folder_id ending with suffix (handles paging)."""
    out, tok = [], None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name)",
            pageSize=200, pageToken=tok,
        ).execute()
        out += [f for f in resp.get("files", []) if f["name"].endswith(suffix)]
        tok = resp.get("nextPageToken")
        if not tok:
            break
    return sorted(out, key=lambda f: f["name"])


def read_parquet(drive, file_id):
    return pd.read_parquet(io.BytesIO(download_bytes(drive, file_id)))


def read_csv(drive, file_id):
    return pd.read_csv(io.BytesIO(download_bytes(drive, file_id)))


def id_series(df: pd.DataFrame):
    """First available identifier column, cleaned; None if none present."""
    for c in ID_COLS:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            s = s[~s.str.lower().isin(["", "nan", "none"])]
            return c, s
    return None, None


# --------------------------------------------------------------------------- #
#  Tier + depth coverage report (T12 Stage 0)
# --------------------------------------------------------------------------- #
def _co_key(co: dict) -> str:
    """Same key convention as the queue/coverage: isin if present else symbol."""
    return (str(co.get("isin") or "").strip()
            or str(co.get("symbol") or "").strip())


def _tier_keys(companies: list[dict]) -> dict:
    """Group company keys by priority tier (portfolio / conviction / tail)."""
    tiers: dict[str, set] = {"portfolio": set(), "conviction": set(), "tail": set()}
    for co in companies:
        key = _co_key(co)
        if not key:
            continue
        tiers.setdefault(co.get("tier", "tail"), set()).add(key)
    return tiers


def _tier_stats(keys: set, doc_type: str, target: int,
                pc_found: dict, cov_idx: dict) -> dict:
    """Compute page-check + depth stats for one tier's key set.

    page-checked = a key with a page-check row OR an existing coverage row (so the
    report is meaningful from existing coverage even before the ledger fills).
    has-docs     = page-check found>0, or coverage shows any found/pending/done docs.
    depth (over has-docs): full n_done>=target / partial 0<n_done<target / none 0."""
    total = len(keys)
    checked = has = no = full = partial = none = 0
    earliest = ""
    latest = ""
    for k in keys:
        cov = cov_idx.get(k)
        pf = pc_found.get(k)
        was_checked = (pf is not None) or (cov is not None)
        if not was_checked:
            continue
        checked += 1
        cov_n_found = int(cov.get("n_found") or 0) if cov is not None else 0
        cov_n_done = int(cov.get("n_done") or 0) if cov is not None else 0
        cov_n_pend = int(cov.get("n_pending") or 0) if cov is not None else 0
        has_docs = (pf is not None and pf > 0) or (cov_n_found + cov_n_done + cov_n_pend) > 0
        if not has_docs:
            no += 1
            continue
        has += 1
        if cov_n_done >= target:
            full += 1
        elif cov_n_done > 0:
            partial += 1
        else:
            none += 1
        if cov is not None:
            ed = str(cov.get("covered_earliest_date") or "")
            ld = str(cov.get("covered_latest_date") or "")
            if ed and (not earliest or ed < earliest):
                earliest = ed
            if ld and (not latest or ld > latest):
                latest = ld
    return {"total": total, "checked": checked, "has": has, "no": no,
            "full": full, "partial": partial, "none": none,
            "earliest": earliest, "latest": latest}


def tier_depth_report(drive, root_id, index_id, doc_type: str,
                      target: int, upload: bool) -> str:
    """Per-tier page-check + depth-to-target breakdown for one doc_type.
    Prints a console table, returns (and optionally uploads) a markdown summary."""
    # Reuse the orchestrator's exact priority ordering / tiering (no duplication).
    import run_backfill as rb
    from backfill_coverage import build_coverage
    from backfill_pagecheck import load_pagecheck

    companies = rb.build_company_order(drive, root_id)
    tiers = _tier_keys(companies)

    qfid = find_file(drive, index_id, "processing_queue.parquet")
    q = read_parquet(drive, qfid) if qfid else pd.DataFrame()
    cov = build_coverage(q)
    cov_idx = {str(r["key"]): r.to_dict()
               for _, r in cov.iterrows() if str(r["doc_type"]) == doc_type} \
        if not cov.empty else {}

    pc = load_pagecheck(drive, index_id)
    pc_found = {}
    if not pc.empty:
        pcd = pc[pc["doc_type"].astype(str) == doc_type]
        for _, r in pcd.iterrows():
            pc_found[str(r["key"]).strip()] = int(
                pd.to_numeric(r["n_docs_found"], errors="coerce") or 0)

    order = ["portfolio", "conviction", "tail"]
    stats = {t: _tier_stats(tiers.get(t, set()), doc_type, target, pc_found, cov_idx)
             for t in order}
    grand = {"total": 0, "checked": 0, "has": 0, "no": 0,
             "full": 0, "partial": 0, "none": 0, "earliest": "", "latest": ""}
    for t in order:
        s = stats[t]
        for k in ("total", "checked", "has", "no", "full", "partial", "none"):
            grand[k] += s[k]
        if s["earliest"] and (not grand["earliest"] or s["earliest"] < grand["earliest"]):
            grand["earliest"] = s["earliest"]
        if s["latest"] and (not grand["latest"] or s["latest"] > grand["latest"]):
            grand["latest"] = s["latest"]

    def _pct(n, d):
        return f"{(100.0 * n / d):.1f}%" if d else "  n/a"

    def _row(label, s):
        return (f"  {label:<11} {s['total']:>6} "
                f"{s['checked']:>7} ({_pct(s['checked'], s['total']):>6}) "
                f"{s['has']:>7} {s['no']:>6} | "
                f"{s['full']:>5} {s['partial']:>7} {s['none']:>5} | "
                f"{s['earliest'] or '-':>10} {s['latest'] or '-':>10}")

    header = (f"  {'tier':<11} {'total':>6} {'checked':>7} {'(%)':>8} "
              f"{'has-doc':>7} {'no-doc':>6} | "
              f"{'full':>5} {'partial':>7} {'none':>5} | "
              f"{'oldest':>10} {'newest':>10}")
    print("\n" + "=" * 110)
    print(f"TIER + DEPTH COVERAGE — doc_type={doc_type}  target-depth={target} docs")
    print("=" * 110)
    print(header)
    print("  " + "-" * 106)
    for t in order:
        print(_row(t, stats[t]))
    print("  " + "-" * 106)
    print(_row("TOTAL", grand))
    print("\n  full = >= target done | partial = some done | none = checked, doc-bearing,"
          " 0 done\n  has-doc/no-doc = of page-checked companies; oldest/newest = covered"
          " announcement dates")

    # ── Markdown summary ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [f"# Backfill coverage — `{doc_type}`",
          "",
          f"_Generated {ts} · target depth = {target} done docs_",
          "",
          "| tier | total | page-checked | % | has-doc | no-doc | "
          "full | partial | none | oldest | newest |",
          "|---|--:|--:|--:|--:|--:|--:|--:|--:|--|--|"]
    for t in order + ["TOTAL"]:
        s = grand if t == "TOTAL" else stats[t]
        md.append(f"| {t} | {s['total']} | {s['checked']} | "
                  f"{_pct(s['checked'], s['total'])} | {s['has']} | {s['no']} | "
                  f"{s['full']} | {s['partial']} | {s['none']} | "
                  f"{s['earliest'] or '-'} | {s['latest'] or '-'} |")
    md += ["",
           "- **full** = ≥ target `done` docs · **partial** = some `done` · "
           "**none** = page-checked & doc-bearing but 0 `done`",
           "- **has-doc / no-doc** = of page-checked companies (the page-check ledger "
           "is the denominator)",
           "- **oldest / newest** = covered announcement-date span across the tier"]
    md_text = "\n".join(md) + "\n"

    if upload:
        try:
            from _extractor_base import upload_bytes
            existing = find_file(drive, index_id, COVERAGE_MD)
            upload_bytes(drive, index_id, COVERAGE_MD,
                         md_text.encode("utf-8"), "text/markdown", existing)
            print(f"\n  Markdown uploaded -> company_repo/_index/{COVERAGE_MD}")
        except Exception as e:
            print(f"\n  WARNING: markdown upload failed ({str(e)[:80]}).")
    return md_text


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-type", default="concall",
                    help="Doc type for the tier+depth breakdown (default concall).")
    ap.add_argument("--target", type=int, default=0,
                    help="Depth target in done-docs (0 = auto: 8 quarterly, 3 annual).")
    ap.add_argument("--no-upload", action="store_true",
                    help="Skip writing coverage_report.md to Drive (console only).")
    args = ap.parse_args()
    target = args.target or DEPTH_TARGET.get(args.doc_type, 8)

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]

    repo = find_sub(drive, root, "company_repo")
    index_id = find_sub(drive, repo, "_index") if repo else None
    fund_id = find_sub(drive, root, "fundamentals")
    uni_id = find_sub(drive, root, "universe")

    # ── Universe ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("UNIVERSE")
    print("=" * 78)
    uni_isins = set()
    uni_total = 0
    for folder, path in [(index_id, "company_repo/_index/company_universe.csv"),
                         (uni_id, "universe/master_list.csv")]:
        if not folder:
            continue
        fname = path.split("/")[-1]
        fid = find_file(drive, folder, fname)
        if not fid:
            print(f"  {path:<48} MISSING")
            continue
        df = read_csv(drive, fid)
        col, s = id_series(df)
        n_isin = df["isin"].astype(str).str.strip().replace(
            {"nan": "", "none": ""}).pipe(lambda x: x[x != ""]).nunique() \
            if "isin" in df.columns else 0
        print(f"  {path:<48} rows={len(df):>6}  distinct_isin={n_isin:>6}  "
              f"cols={list(df.columns)[:6]}")
        if "isin" in df.columns:
            vals = df["isin"].astype(str).str.strip()
            uni_isins |= set(vals[~vals.str.lower().isin(["", "nan", "none"])])
        uni_total = max(uni_total, len(df))
    uni_isin_n = len(uni_isins) if uni_isins else uni_total
    print(f"\n  -> Universe rows (max): {uni_total}   distinct ISIN: {len(uni_isins)}")

    def pct(n):
        d = len(uni_isins) or uni_total
        return f"{(100.0*n/d):5.1f}%" if d else "  n/a"

    # ── processing_queue (the backfill + live capture ledger) ──────────────────
    print("\n" + "=" * 78)
    print("PROCESSING QUEUE  (company_repo/_index/processing_queue.parquet)")
    print("=" * 78)
    q_done_isins_by_type = {}
    if index_id:
        qfid = find_file(drive, index_id, "processing_queue.parquet")
        if qfid:
            q = read_parquet(drive, qfid)
            print(f"  rows={len(q)}  cols={list(q.columns)}")
            if "status" in q.columns:
                print(f"\n  status totals: {q['status'].value_counts().to_dict()}")
            if "source" in q.columns:
                print(f"  source totals: {q['source'].astype(str).value_counts().to_dict()}")
            if {"doc_type", "status"}.issubset(q.columns):
                print("\n  doc_type x status (rows) | distinct companies DONE:")
                idcol, _ = id_series(q)
                for dt, grp in q.groupby("doc_type"):
                    cnts = grp["status"].value_counts().to_dict()
                    done = grp[grp["status"] == "done"]
                    n_co = 0
                    if idcol and idcol in done.columns:
                        dv = done[idcol].astype(str).str.strip()
                        dv = dv[~dv.str.lower().isin(["", "nan", "none"])]
                        n_co = dv.nunique()
                        q_done_isins_by_type[str(dt)] = set(dv)
                    print(f"    {str(dt):<16} "
                          f"pending={cnts.get('pending',0):>5} "
                          f"done={cnts.get('done',0):>5} "
                          f"error={cnts.get('error',0):>5}  "
                          f"| companies_done={n_co:>5} ({pct(n_co)})")
        else:
            print("  MISSING")

    # ── Every _index and fundamentals parquet ─────────────────────────────────
    for label, folder in [("company_repo/_index", index_id), ("fundamentals", fund_id)]:
        if not folder:
            continue
        print("\n" + "=" * 78)
        print(f"ALL PARQUETS in {label}/")
        print("=" * 78)
        for f in list_files(drive, folder):
            try:
                df = read_parquet(drive, f["id"])
            except Exception as e:
                print(f"  {f['name']:<42} READ ERROR: {str(e)[:60]}")
                continue
            col, s = id_series(df)
            nd = s.nunique() if s is not None else 0
            label_id = f"{col}" if col else "no-id-col"
            print(f"  {f['name']:<42} rows={len(df):>7}  "
                  f"distinct[{label_id}]={nd:>6} ({pct(nd)})")

    # ── Captured-anywhere summary ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY — companies captured (distinct ISIN with status=done in queue)")
    print("=" * 78)
    all_done = set().union(*q_done_isins_by_type.values()) if q_done_isins_by_type else set()
    print(f"  ANY doc captured : {len(all_done):>6} ({pct(len(all_done))} of universe)")
    for dt in sorted(q_done_isins_by_type):
        n = len(q_done_isins_by_type[dt])
        print(f"    {dt:<16} {n:>6} ({pct(n)})")

    # ── Tier + depth breakdown (T12 Stage 0) ──────────────────────────────────
    if index_id:
        tier_depth_report(drive, root, index_id, args.doc_type, target,
                          upload=not args.no_upload)
    else:
        print("\n  (skipped tier+depth report — no company_repo/_index folder.)")


if __name__ == "__main__":
    main()
