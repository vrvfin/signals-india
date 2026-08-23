r"""
audit_guidance_cleaning.py — is the guidance data actually cleaned correctly?

READ-ONLY. Writes one local HTML report and prints a pass/fail summary. No Drive
writes, no lock, no Gemini. Nothing in the pipeline depends on it.

    python scripts/audit_guidance_cleaning.py              # writes the report
    python scripts/audit_guidance_cleaning.py --dry-run    # console only
    python scripts/audit_guidance_cleaning.py --sample 8   # more reject samples

FIVE INDEPENDENT WAYS TO CHECK THE CLEANING
-------------------------------------------
A. INVARIANTS      machine-checkable rules that must hold for every published
                   row (no margin rows, no unresolved horizons, evidence present,
                   metric in {revenue, pat}, ...). Any failure is a real bug.
B. EVIDENCE        every published row shown as
                       raw cell -> cleaned number -> verbatim management quote
                   with the agreement delta. delta 0.00% means the machine and
                   the transcript landed on exactly the same figure.
C. WHAT CHANGED    every company where the INCUMBENT scorer
                   (build_gallery.guidance_scores, still LIVE for
                   gallery_guidance.html) and the cleaner disagree, with both
                   numbers and the rule that moved it. This is the diff that
                   says what cleaning bought.
D. REJECTS         a stratified sample of the cells the cleaner refused, per
                   rule, so an over-aggressive rule is visible rather than
                   silently eating good rows.
E. COVERAGE        how much of the funnel survives each stage, and where the
                   evidence runs out (PAT statements are thin).

WHY THIS IS NOT A UNIT TEST
---------------------------
guidance_strength/--self-test and guidance_validate/--self-test already pin the
KNOWN failures as fixtures. This audit is the complement: it runs the real
pipeline over the real table and shows a human what came out, because a fixture
can only catch a defect someone already thought of.
"""
from __future__ import annotations

import argparse
import html as html_mod
import io
import os
import sys
import webbrowser
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import build_gallery as BG
import build_guidance_watchlist as BW
import guidance_strength as GS
import guidance_validate as GVAL
import quarterly_table as QT
from _extractor_base import (download_bytes, find_file, get_drive,
                             get_or_create_subfolder, load_parquet, log)


def _esc(v) -> str:
    return html_mod.escape(str(v if v is not None else ""))


def _read_any(drive, fid_folder, name) -> pd.DataFrame:
    fid = find_file(drive, fid_folder, name)
    if not fid:
        return pd.DataFrame()
    raw = download_bytes(drive, fid)
    return (pd.read_csv(io.BytesIO(raw)) if name.endswith(".csv")
            else pd.read_parquet(io.BytesIO(raw)))


# --------------------------------------------------------------------------- #
#  A. invariants                                                                #
# --------------------------------------------------------------------------- #
def invariants(wl: pd.DataFrame) -> list:
    """(name, ok, detail) for each rule that must hold on every published row."""
    out = []

    def chk(name, bad_mask, detail_col=None):
        bad = wl[bad_mask] if len(wl) else wl
        ok = len(bad) == 0
        if ok:
            out.append((name, True, f"0 of {len(wl)} rows violate this"))
        else:
            ex = ""
            if detail_col and detail_col in bad.columns:
                ex = " · e.g. " + ", ".join(
                    f"{s}={v}" for s, v in zip(bad["nse_symbol"].head(3),
                                               bad[detail_col].head(3)))
            out.append((name, False, f"{len(bad)} of {len(wl)} rows VIOLATE{ex}"))

    if wl.empty:
        return [("watchlist is non-empty", False, "0 rows — nothing to audit")]

    num = lambda c: pd.to_numeric(wl[c], errors="coerce")  # noqa: E731

    chk("metric is revenue or PAT only (margin never scored)",
        ~wl["score_metric"].isin(["revenue", "pat"]), "score_metric")
    chk("value_type is never a LEVEL percentage",
        wl["value_type"].isin(["margin_pct", "level_pct", "utilisation_pct",
                               "capacity_pct"]), "value_type")
    chk("every published row carries transcript evidence",
        ~wl["validation_verdict"].isin(list(GVAL.PUBLISHABLE)),
        "validation_verdict")
    chk("evidence statement is non-empty",
        wl["evidence_stmt"].astype(str).str.strip() == "", "validation_verdict")
    chk("horizon resolved to a real forward distance (years > 0)",
        ~(num("years_used") > 0), "years_used")
    chk("CAGR clears the configured floor",
        num("cagr_pct") < num("min_cagr_used"), "cagr_pct")
    chk("absolute rows have a positive TTM base",
        (wl["score_kind"] == "absolute") & ~(num("base_ttm_cr") > 0),
        "base_ttm_cr")
    chk("absolute rows target ABOVE their base",
        (wl["score_kind"] == "absolute")
        & ~(num("target_cr") > num("base_ttm_cr")), "target_cr")
    chk("quarter label is parseable",
        wl["quarter"].map(QT.q_order).fillna(-1) <= 0, "quarter")
    chk("no duplicate (isin, quarter)",
        wl.duplicated(subset=["isin", "quarter"], keep=False), "quarter")
    chk("date_added is a real date",
        pd.to_datetime(wl["date_added"], errors="coerce").isna(), "date_added")
    chk("n_quarters >= 1 for every row",
        ~(num("n_quarters") >= 1), "n_quarters")
    chk("streak never exceeds n_quarters",
        num("quarter_streak") > num("n_quarters"), "quarter_streak")
    chk("in_prev_quarter implies n_quarters >= 2",
        wl["in_prev_quarter"].fillna(False) & (num("n_quarters") < 2),
        "n_quarters")
    chk("no published row is scoped to a SEGMENT rather than the company",
        wl["evidence_stmt"].map(lambda s: bool(GVAL.scope_is_segment(s))),
        "evidence_stmt")
    chk("no published row rests on a NEGATED statement",
        wl["evidence_stmt"].map(GVAL.is_negated), "evidence_stmt")
    chk("absolute rows record how many quarters built their base",
        (wl["score_kind"] == "absolute") & num("base_quarters").isna(),
        "base_quarters")
    chk("base scale is the annualisation factor 4/n",
        (wl["score_kind"] == "absolute")
        & (num("base_quarters") * num("base_scale") - 4.0).abs().gt(0.05),
        "base_scale")
    chk("absolute horizons are never sub-annual",
        (wl["score_kind"] == "absolute") & (num("years_used") < 1.0),
        "years_used")
    chk("n_rows_over_min is a real count, never the old hardcoded stub",
        ~(num("n_rows_over_min") >= 1), "n_rows_over_min")
    chk("guidance_source is a known source",
        ~wl["guidance_source"].isin(["concall", "presentation", "annual_report"]),
        "guidance_source")
    chk("no non-revenue concept survived into a revenue row",
        (wl["score_metric"] == "revenue")
        & wl["guided_value"].astype(str).str.lower()
        .str.contains("|".join(GS.NON_REVENUE_QUALIFIERS), regex=True, na=False),
        "guided_value")
    return out


# --------------------------------------------------------------------------- #
#  C. incumbent vs cleaner                                                      #
# --------------------------------------------------------------------------- #
def compare_incumbent(guid, wl, base_rev_sym, base_pat_sym) -> pd.DataFrame:
    """What the LIVE scorer says vs what the cleaner says, per symbol."""
    live = BG.guidance_scores(guid, base_rev_sym, base_pat_sym)
    rows = []
    for _, r in wl.iterrows():
        sym = str(r.get("nse_symbol") or r.get("symbol") or "").upper()
        old = live.get(sym)
        rows.append({
            "symbol": sym,
            "incumbent_pct": round(old[0], 1) if old else None,
            "cleaned_pct": round(float(r["cagr_pct"]), 1),
            "rule": r["score_rule"],
            "verdict": r["validation_verdict"],
            "raw_cell": str(r["guided_value"])[:70],
        })
    d = pd.DataFrame(rows)
    d["delta"] = d["incumbent_pct"] - d["cleaned_pct"]
    return d.sort_values("delta", ascending=False, na_position="last")


def _table(df: pd.DataFrame, cols=None, cls="") -> str:
    cols = cols or list(df.columns)
    th = "".join(f'<th>{_esc(c)}</th>' for c in cols)
    body = []
    for _, r in df.iterrows():
        body.append("<tr>" + "".join(f"<td>{_esc(r.get(c))}</td>" for c in cols)
                    + "</tr>")
    return (f'<table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


_TPL = """<!doctype html><html><head><meta charset="utf-8">
<title>Guidance cleaning audit __DATE__</title><style>
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f4f6f9;margin:0;padding:16px;color:#222}
 h1{font-size:19px;margin:4px 0 2px} h2{font-size:15px;margin:22px 0 6px;color:#1a3d6e}
 .sub{font-size:12px;color:#777;margin-bottom:14px}
 .card{background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:10px 12px;margin-bottom:14px;overflow-x:auto}
 table{border-collapse:collapse;width:100%;font-size:11.5px}
 th{background:#1a3d6e;color:#fff;text-align:left;padding:4px 6px;position:sticky;top:0;white-space:nowrap}
 td{padding:3px 6px;border-bottom:1px solid #eef1f5;vertical-align:top}
 .pass{color:#1a7a3a;font-weight:700}.fail{color:#c0392b;font-weight:700}
 .note{font-size:12px;color:#555;margin:2px 0 8px;line-height:1.5}
 .q{color:#1565c0;font-style:italic}
</style></head><body>
<h1>Guidance cleaning audit</h1>
<div class="sub">__DATE__ · quarter __QTR__ · read-only, nothing was written</div>
<h2>A · Invariants — rules that must hold on every published row</h2>
<div class="card">__INVARIANTS__</div>
<h2>B · Evidence — raw cell &rarr; cleaned number &rarr; what management said</h2>
<div class="note">delta 0.00% means the cleaned figure and the transcript figure are the
same number. This is the section to read if you want to spot-check the cleaning by hand.</div>
<div class="card">__EVIDENCE__</div>
<h2>C · What cleaning changed vs the live scorer</h2>
<div class="note">The incumbent is <code>build_gallery.guidance_scores</code>, still driving
gallery_guidance.html. A blank incumbent value means that name does not reach its list at all.</div>
<div class="card">__CHANGED__</div>
<h2>D · Rejects — a sample of every rule, so an over-aggressive rule is visible</h2>
<div class="card">__REJECTS__</div>
<h2>E · Coverage — where the funnel narrows</h2>
<div class="card">__COVERAGE__</div>
</body></html>"""


def main() -> int:                                            # noqa: C901
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quarter", default="auto")
    ap.add_argument("--sample", type=int, default=5,
                    help="reject rows shown per rule (default 5)")
    ap.add_argument("--out", default="guidance_cleaning_audit.html")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="console summary only; no HTML written")
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    idx = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")
    fund = get_or_create_subfolder(drive, root, "fundamentals")

    log("loading tables…")
    wl = load_parquet(drive, idx, BW.GW_NAME, BW.GW_COLS)
    if wl.empty:
        log("guidance_watchlist.parquet is empty — run build_guidance_watchlist.py.")
        return 1
    guid = load_parquet(drive, idx, BW.GUIDANCE_NAME, BW.GUIDANCE_COLS)
    queue = _read_any(drive, idx, BW.QUEUE_NAME)
    gf1 = _read_any(drive, idx, BW.GF1_NAME)
    uni = _read_any(drive, idx, BW.UNIVERSE_NAME)
    summ = _read_any(drive, fund, "summary.parquet")

    wl["_qo"] = wl["quarter"].map(QT.q_order)
    q = (str(wl.sort_values("_qo")["quarter"].iloc[-1])
         if args.quarter == "auto" else QT.norm_q(args.quarter))
    cur = wl[wl["quarter"] == q].sort_values(
        ["date_added", "cagr_pct"], ascending=[False, False]).reset_index(drop=True)
    log(f"auditing {len(cur)} published rows for {q} (of {len(wl)} total)")

    # ---- A invariants ------------------------------------------------------
    inv = invariants(cur)
    n_fail = sum(1 for _, ok, _ in inv if not ok)
    inv_html = _table(pd.DataFrame(
        [{"check": n, "result": "PASS" if ok else "FAIL", "detail": d}
         for n, ok, d in inv]))
    inv_html = inv_html.replace("<td>PASS</td>", '<td class="pass">PASS</td>')
    inv_html = inv_html.replace("<td>FAIL</td>", '<td class="fail">FAIL</td>')

    # ---- B evidence --------------------------------------------------------
    ev = pd.DataFrame([{
        "symbol": r["nse_symbol"] or r["symbol"],
        "metric": r["score_metric"],
        "raw cell (as extracted)": str(r["guided_value"])[:60],
        "rule applied": r["score_rule"],
        "horizon": r["horizon_fy"],
        "yrs": r["years_used"],
        "base ₹cr": ("" if pd.isna(r["base_ttm_cr"]) else f"{r['base_ttm_cr']:,.0f}"),
        "target ₹cr": ("" if pd.isna(r["target_cr"]) else f"{r['target_cr']:,.0f}"),
        "CLEANED %/yr": f"{r['cagr_pct']:.1f}",
        "verdict": r["validation_verdict"],
        "delta": ("" if pd.isna(r["evidence_delta_pct"])
                  else f"{r['evidence_delta_pct']:.2f}%"),
        "what management said": str(r["evidence_stmt"])[:190],
    } for _, r in cur.iterrows()])

    # ---- C incumbent vs cleaner -------------------------------------------
    base_rev_sym, base_pat_sym = {}, {}
    if not summ.empty and "symbol" in summ.columns:
        for _, r in summ.iterrows():
            s = str(r.get("symbol", "")).upper()
            for col, dest in (("q_sales_last_4q", base_rev_sym),
                              ("q_netprofit_last_4q", base_pat_sym)):
                try:
                    dest[s] = float(pd.Series(r.get(col)).astype(float).sum())
                except Exception:                             # noqa: BLE001
                    pass
    cmp_df = compare_incumbent(guid, cur, base_rev_sym, base_pat_sym)
    moved = cmp_df[cmp_df["delta"].abs() > 1] if "delta" in cmp_df else cmp_df
    newly = cmp_df[cmp_df["incumbent_pct"].isna()]

    # ---- D rejects ---------------------------------------------------------
    g2 = BW.resolve_quarters(guid, queue)
    sym2isin = BW.symbol_isin_map(uni, g2)
    b_rev, b_pat = BW.summary_base_maps(summ, sym2isin)
    qo = QT.q_order(q)
    rp = g2[g2["metric"].astype(str).str.lower()
            .isin(["revenue", "sales", "pat", "profit", "net profit", "earnings"])]
    win = rp[(rp["_qo"] <= qo) & (rp["_qo"] >= qo - 2)]
    _best, rejects, _scored = GS.best_per_key(win, b_rev, b_pat, key="isin")
    rj = pd.DataFrame([{"symbol": x.get("symbol"), "quarter": x.get("quarter"),
                        "metric": x.get("metric"), "reason": x.get("reject"),
                        "cell": str(x.get("raw"))[:74]}
                       for lst in rejects.values() for x in lst])
    if not rj.empty:
        rj["rule"] = rj["reason"].astype(str).str.split(":").str[0]
        samp = pd.concat([d.head(args.sample) for _, d in rj.groupby("rule")],
                         ignore_index=True)
        counts = rj["rule"].value_counts().to_dict()
        samp["total with this rule"] = samp["rule"].map(counts)
    else:
        samp = pd.DataFrame([{"note": "no rejects"}])

    # ---- E coverage --------------------------------------------------------
    cov = pd.DataFrame([
        {"stage": f"revenue/PAT rows within 2 quarters of {q}", "n": len(win)},
        {"stage": "companies with at least one scoreable cell", "n": len(_best)},
        {"stage": f"clear the {cur['min_cagr_used'].iloc[0]:.0f}% floor",
         "n": int(sum(1 for v in _best.values()
                      if v["cagr_pct"] >= cur["min_cagr_used"].iloc[0]))},
        {"stage": "PUBLISHED (transcript CONFIRMED or CONSISTENT)", "n": len(cur)},
        {"stage": "cells refused by the cleaner (all rules)",
         "n": 0 if rj.empty else len(rj)},
        {"stage": "of published: repeat guiders (n_quarters >= 2)",
         "n": int((cur["n_quarters"] >= 2).sum())},
    ])

    # ---- console -----------------------------------------------------------
    print()
    print("=" * 74)
    print(f"  A · INVARIANTS  —  {len(inv) - n_fail}/{len(inv)} PASS"
          + ("" if not n_fail else f"   *** {n_fail} FAILED ***"))
    for n, ok, d in inv:
        if not ok:
            print(f"      FAIL  {n}  ({d})")
    print(f"  B · EVIDENCE    —  {len(cur)} published rows, all with a quote")
    exact = (pd.to_numeric(cur["evidence_delta_pct"], errors="coerce")
             .fillna(99) <= 0.01).sum()
    print(f"                     {exact} match the transcript figure EXACTLY (delta<=0.01%)")
    print(f"  C · CHANGED     —  {len(moved)} names differ from the live scorer by >1pt")
    print(f"                     {len(newly)} are NEW (the live scorer misses them entirely)")
    print(f"  D · REJECTS     —  {0 if rj.empty else len(rj)} cells refused"
          + ("" if rj.empty else f", {rj['rule'].nunique()} distinct rules"))
    print(f"  E · COVERAGE    —  {len(win)} rows -> {len(_best)} scored -> {len(cur)} published")
    print("=" * 74)
    print()

    if args.dry_run:
        log("DRY-RUN — no HTML written.")
        return 1 if n_fail else 0

    html = (_TPL.replace("__DATE__", datetime.now().strftime("%d %b %Y %H:%M"))
            .replace("__QTR__", q)
            .replace("__INVARIANTS__", inv_html)
            .replace("__EVIDENCE__", _table(ev))
            .replace("__CHANGED__", _table(
                cmp_df[["symbol", "incumbent_pct", "cleaned_pct", "delta", "rule",
                        "verdict", "raw_cell"]]))
            .replace("__REJECTS__", _table(samp))
            .replace("__COVERAGE__", _table(cov)))
    out_path = os.path.join(os.path.dirname(_SCRIPTS_DIR), args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"wrote {out_path} ({len(html) / 1e3:.0f} KB)")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out_path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
