r"""
narrative_preflight.py — answers two questions BEFORE a narrative report is generated:

  1. WILL EVERY SECTION BE POPULATED?  A readiness matrix over all 26 sections, each in
     one of three states, because they need different actions:
       READY      — the inputs are on Drive now.
       FETCHABLE  — the pipeline CAN get this, but this company's documents are missing
                    or too few. Emits the exact command to fix it.
       BLOCKED    — the extraction capability does not exist yet (N7/N8/N9). No command
                    will help; the section renders DATA_MISSING until that phase ships.

  2. IS THE DATA CORRECT?  Deterministic integrity checks that run EVERY time, not once:
       RECON  cross-source agreement (quarterly sums vs annual, TTM vs quarters)
       IDENT  internal identities (PAT/EPS implies a stable share count; the fixed block
              ties to the disclosed EBITDA-PBT gap)
       STALE  source age against a stated policy
     These catch the failure mode the grounding gates cannot: a number that is faithfully
     copied from a source that is itself wrong or out of date.

Exit codes: 0 all good · 1 could not resolve · 2 integrity FAIL (do not publish).

Usage:
  python scripts/narrative_preflight.py --names LANDMARK
  python scripts/narrative_preflight.py --names LANDMARK --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import narrative_compute as NC
from narrative_factpack import Store, resolve, IDX, FUND

# ----------------------------------------------------------------- policy -----
# Max age before a source is considered stale. Chosen against how each is refreshed:
# statements/summary come from the weekly fundamentals job, company_facts nightly.
STALE_DAYS = {"statements": 14, "summary": 14, "company_facts": 5}

# Minimum documents a section needs to be worth rendering at all.
MIN_DOCS = {"concall": 4, "annual_report": 3}

# backfill_company_docs takes --token (name / NSE symbol / BSE code / ISIN) and
# resolves it through the universe. NOT --names, which it rejects.
BACKFILL_CMD = "python scripts/backfill_company_docs.py --token \"{name}\""

# section -> (title, kind, requirement)
#   kind "data"     : satisfied from parquets already on Drive
#   kind "docs"     : needs N documents of a doc_type in processing_queue
#   kind "blocked"  : needs an unbuilt phase
# N7/N8 ARE built, so sections that depend on extracted filings are now "docs": they
# need an annual report (or concall) in the queue AND the extractor to have run. A
# company without the document is FETCHABLE — a command fixes it — not BLOCKED.
REQUIREMENTS: list[tuple[int, str, str, dict]] = [
    (1,  "History timeline",              "docs",    {"doc_type": "annual_report",
                                                      "also": ["company_structure"]}),
    (2,  "The company on one page",       "data",    {"tables": ["company_facts", "summary"]}),
    (3,  "Legal entity map",              "docs",    {"doc_type": "annual_report",
                                                      "also": ["company_structure"]}),
    (4,  "Management bench",              "docs",    {"doc_type": "annual_report",
                                                      "also": ["company_structure"]}),
    (5,  "Scale history",                 "data",    {"tables": ["statements"]}),
    (6,  "Segment economics",             "docs",    {"doc_type": "annual_report",
                                                      "also": ["company_structure"]}),
    (7,  "Accounting basis & ratings",    "data",    {"tables": ["ar_red_flags"]}),
    (8,  "The steadiest series",          "data",    {"tables": ["statements"]}),
    (9,  "Portfolio table",               "docs",    {"doc_type": "annual_report",
                                                      "also": ["company_structure"]}),
    (10, "Tier framework",                "blocked", {"phase": "Layer B (derived from s9)"}),
    (11, "Mix shift",                     "docs",    {"doc_type": "annual_report",
                                                      "also": ["company_structure"]}),
    (12, "Unit deep dives",               "docs",    {"doc_type": "annual_report",
                                                      "also": ["company_structure"]}),
    (16, "Alt-data claim test",           "blocked", {"phase": "no automatable feed"}),
    (17, "Alt-data scorecard",            "blocked", {"phase": "no automatable feed"}),
    (18, "Full financial record",         "data",    {"tables": ["statements"]}),
    (19, "Operating leverage",            "data",    {"tables": ["statements"]}),
    (20, "Management's claim tested",     "docs",    {"doc_type": "concall",
                                                      "also": ["mgmt_quotes"]}),
    (21, "Peer set",                      "data",    {"tables": ["company_facts",
                                                                 "peer_aggregates"]}),
    (22, "Sensitivity grid",              "data",    {"tables": ["statements",
                                                                 "company_facts"]}),
    (23, "Structural risks",              "docs",    {"doc_type": "annual_report",
                                                      "also": ["company_structure"]}),
    (24, "Financial & execution risks",   "data",    {"tables": ["fraud_tracker"]}),
    (25, "Policy backdrop",               "data",    {"tables": []}),
    (26, "Findings",                      "data",    {"tables": []}),
    (27, "Recent exchange filings",       "data",    {"tables": []}),
]


def _age_days(ts) -> float | None:
    try:
        t = pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.isna(t):
            return None
        return (datetime.now(timezone.utc) - t.to_pydatetime()).total_seconds() / 86400.0
    except Exception:
        return None


# ------------------------------------------------------------- readiness -----
def readiness(store: Store, isin: str, symbol: str, name: str) -> list[dict]:
    st = store.parquet(f"{FUND}/statements", f"{symbol}.parquet")
    q = store.by_isin("processing_queue.parquet", isin, symbol)
    doc_counts = ({} if q.empty else
                  q[q["status"].astype(str) == "done"]
                  .groupby("doc_type").size().to_dict())

    avail = {
        "statements": not st.empty,
        "company_facts": not store.by_isin("company_facts.parquet", isin, symbol).empty,
        "peer_aggregates": not store.parquet(IDX, "peer_aggregates.parquet").empty,
        "fraud_tracker": not store.by_isin("fraud_tracker.parquet", isin, symbol).empty,
        "ar_red_flags": not store.by_isin("ar_red_flags.parquet", isin, symbol).empty,
        "guidance_tracker": not store.by_isin("guidance_tracker.parquet", isin,
                                              symbol).empty,
        # The N7/N8 outputs — present only once the extractors have run for THIS
        # company, which is what separates "we cannot do this" from "not done yet".
        "company_structure": not store.by_isin("company_structure.parquet", isin,
                                               symbol).empty,
        "mgmt_quotes": not store.by_isin("mgmt_quotes.parquet", isin, symbol).empty,
    }
    s = store.parquet(FUND, "summary.parquet")
    avail["summary"] = (not s.empty and
                        (s["symbol"].astype(str).str.upper() == symbol.upper()).any())

    out = []
    for num, title, kind, req in REQUIREMENTS:
        # `remedy` holds a RUNNABLE command and nothing else; a phase dependency goes in
        # `blocked_by`. Concatenating the two produced un-runnable command lines.
        row = {"section": num, "title": title, "state": "READY", "detail": "",
               "remedy": "", "blocked_by": ""}
        if kind == "blocked":
            row["state"] = "BLOCKED"
            row["detail"] = f"extraction capability not built ({req['phase']})"
            row["blocked_by"] = req["phase"]
        elif kind == "docs":
            dt = req["doc_type"]
            have, need = int(doc_counts.get(dt, 0)), MIN_DOCS.get(dt, 1)
            missing_tbl = [t for t in req.get("also", []) if not avail.get(t)]
            if have < need:
                row["state"] = "FETCHABLE"
                row["detail"] = (f"{have} processed {dt}(s), need >= {need}"
                                 + (f"; also missing {', '.join(missing_tbl)}"
                                    if missing_tbl else ""))
                row["remedy"] = BACKFILL_CMD.format(name=name)
            elif missing_tbl:
                row["state"] = "FETCHABLE"
                row["detail"] = f"{have} {dt}(s) present but {', '.join(missing_tbl)} empty"
                row["remedy"] = BACKFILL_CMD.format(name=name)
            else:
                row["detail"] = f"{have} processed {dt}(s)"
            if req.get("phase"):
                # Documents may be present while the extractor that reads them is not.
                # Such a section stays BLOCKED, but if documents are ALSO short the
                # fetch command is still worth running now — both are reported.
                row["blocked_by"] = req["phase"]
                if row["state"] == "READY":
                    row["state"] = "BLOCKED"
                row["detail"] += f"; renderer needs {req['phase']}"
        else:
            missing = [t for t in req.get("tables", []) if not avail.get(t)]
            if missing:
                row["state"] = "FETCHABLE"
                row["detail"] = f"missing: {', '.join(missing)}"
                row["remedy"] = BACKFILL_CMD.format(name=name)
            else:
                row["detail"] = "inputs present"
            if req.get("partial"):
                row["detail"] += f" (partial — {req['partial']})"
        out.append(row)
    return out


# ------------------------------------------------------------- integrity -----
def integrity(store: Store, isin: str, symbol: str) -> list[dict]:
    """Deterministic correctness checks. Each returns PASS / WARN / FAIL with the
    numbers that produced the verdict, so a failure is actionable rather than a mood."""
    checks: list[dict] = []

    def add(cid, name, status, detail):
        checks.append({"id": cid, "name": name, "status": status, "detail": detail})

    st = store.parquet(f"{FUND}/statements", f"{symbol}.parquet")
    if st.empty:
        add("RECON.statements", "Statements present", "FAIL",
            f"no fundamentals/statements/{symbol}.parquet — every financial section "
            f"would be empty")
        return checks

    # -- RECON 1: quarterly sales should sum to the annual figure for a complete FY ----
    qp = NC.pivot(st, "quarterly_pl")
    ap = NC.pivot(st, "annual_pl")
    if not qp.empty and NC.L_SALES in qp.index:
        qs = {}
        for per in qp.columns:
            v = qp.at[NC.L_SALES, per]
            if pd.notna(v):
                # Screener quarter labels are month-end: Jun/Sep/Dec/Mar. Mar belongs to
                # the FY ending that March; the other three to the following March.
                try:
                    mon, yr = str(per).split()
                    yr = int(yr)
                except ValueError:
                    continue
                fy = yr if mon == "Mar" else yr + 1
                qs.setdefault(f"FY{str(fy)[2:]}", []).append(float(v))
        for fy, vals in sorted(qs.items()):
            if len(vals) != 4:
                continue
            per = next((p for p in NC._annual_periods(st) if NC.fy_label(p) == fy), None)
            ann = NC._get(ap, NC.L_SALES, per) if per else None
            if ann is None or ann == 0:
                continue
            diff = 100.0 * (sum(vals) / ann - 1.0)
            add(f"RECON.q2a.{fy}", f"Quarterly sales sum to annual ({fy})",
                "PASS" if abs(diff) <= 3.0 else "WARN" if abs(diff) <= 8.0 else "FAIL",
                f"4 quarters = {sum(vals):,.0f} vs annual {ann:,.0f} ({diff:+.1f}%)")

    # -- RECON 2: company_facts TTM revenue vs the last four quarters ------------------
    cf = store.by_isin("company_facts.parquet", isin, symbol)
    if not cf.empty and not qp.empty and NC.L_SALES in qp.index:
        ttm = pd.to_numeric(cf.iloc[0].get("rev_ttm"), errors="coerce")
        ordered = []
        for per in qp.columns:
            try:
                mon, yr = str(per).split()
                ordered.append((int(yr), {"Mar": 4, "Jun": 1, "Sep": 2, "Dec": 3}[mon],
                                per))
            except (ValueError, KeyError):
                continue
        ordered.sort()
        last4 = [qp.at[NC.L_SALES, p] for _, _, p in ordered[-4:]]
        last4 = [float(v) for v in last4 if pd.notna(v)]
        if pd.notna(ttm) and len(last4) == 4 and ttm:
            diff = 100.0 * (sum(last4) / float(ttm) - 1.0)
            add("RECON.ttm", "company_facts TTM revenue vs last 4 quarters",
                "PASS" if abs(diff) <= 3.0 else "WARN" if abs(diff) <= 10.0 else "FAIL",
                f"quarters {sum(last4):,.0f} vs stored TTM {float(ttm):,.0f} "
                f"({diff:+.1f}%)")

    # -- IDENT 1: PAT/EPS implies a share count; it should not jump without issuance ---
    pat, eps = NC.pat_series(st).as_map(), NC.eps_series(st).as_map()
    shares = {fy: pat[fy] / eps[fy] for fy in pat
              if pat.get(fy) and eps.get(fy) and pat[fy] > 0 and eps[fy] > 0}
    if len(shares) >= 2:
        keys = list(shares)
        jumps = []
        for a, b in zip(keys, keys[1:]):
            ch = 100.0 * (shares[b] / shares[a] - 1.0)
            if abs(ch) > 10.0:
                jumps.append(f"{a}->{b} {ch:+.0f}% ({shares[a]:.2f}->{shares[b]:.2f} Cr)")
        add("IDENT.shares", "Implied share count is stable across years",
            "PASS" if not jumps else "WARN",
            (f"stable at ~{list(shares.values())[-1]:.2f} Cr"
             if not jumps else
             "share count moves >10%: " + "; ".join(jumps)
             + " — equity issuance, a split, or a data error"))

    # -- IDENT 2: the fixed block must tie to the disclosed EBITDA-PBT gap -------------
    lev = NC.operating_leverage(st)
    if lev:
        add("IDENT.fixedblock", "Fixed block ties to the EBITDA-PBT gap",
            "PASS" if lev["reconciles"] else "WARN",
            f"D&A {lev['depreciation']:,.0f} + finance {lev['finance_cost']:,.0f} "
            f"= {lev['fixed_block']:,.0f} vs disclosed gap "
            f"{lev['ebitda'] - lev['pbt']:,.0f} "
            f"(residual {lev['residual_vs_disclosed_gap']:+.1f} Rs Cr)"
            + ("" if lev["reconciles"] else
               " — an undisclosed item (usually exceptionals) sits between them; "
               "sensitivity cells will be overstated until it is supplied"))

    # -- STALE: source age against policy ---------------------------------------------
    ages = {}
    if "fetched_at" in st.columns and len(st):
        ages["statements"] = _age_days(st["fetched_at"].iloc[0])
    if not cf.empty:
        ages["company_facts"] = _age_days(cf.iloc[0].get("updated_at"))
    s = store.parquet(FUND, "summary.parquet")
    if not s.empty:
        m = s[s["symbol"].astype(str).str.upper() == symbol.upper()]
        if not m.empty:
            ages["summary"] = _age_days(m.iloc[0].get("fetched_at"))
    for src, age in ages.items():
        lim = STALE_DAYS.get(src, 14)
        if age is None:
            add(f"STALE.{src}", f"{src} freshness", "WARN", "no timestamp recorded")
        else:
            add(f"STALE.{src}", f"{src} freshness",
                "PASS" if age <= lim else "WARN" if age <= lim * 2 else "FAIL",
                f"{age:.1f} days old (policy: refresh within {lim}d)")

    # -- STALE cross-source skew: sources disagreeing on "now" produce mixed reports ---
    known = {k: v for k, v in ages.items() if v is not None}
    if len(known) >= 2:
        skew = max(known.values()) - min(known.values())
        add("STALE.skew", "Sources agree on as-of date",
            "PASS" if skew <= 7 else "WARN",
            f"oldest and newest inputs differ by {skew:.1f} days "
            f"({', '.join(f'{k} {v:.0f}d' for k, v in sorted(known.items()))})"
            + ("" if skew <= 7 else
               " — market data and financials describe different moments"))
    return checks


# ---------------------------------------------------------------- report -----
def run(store: Store, token: str) -> dict | None:
    r = resolve(store, token)
    if r is None:
        return None
    isin, symbol, name = r
    rows = readiness(store, isin, symbol, name)
    checks = integrity(store, isin, symbol)
    counts = {s: sum(1 for x in rows if x["state"] == s)
              for s in ("READY", "FETCHABLE", "BLOCKED")}
    verdicts = {s: sum(1 for c in checks if c["status"] == s)
                for s in ("PASS", "WARN", "FAIL")}
    return {"company": {"isin": isin, "symbol": symbol, "name": name},
            "readiness": rows, "readiness_counts": counts,
            "integrity": checks, "integrity_counts": verdicts,
            "publishable": verdicts.get("FAIL", 0) == 0}


def _print(rep: dict):
    c = rep["company"]
    print(f"\n{'=' * 78}\n{c['name']}  ({c['symbol']} / {c['isin']})\n{'=' * 78}")
    print("\nSECTION READINESS")
    print(f"  {'#':>3}  {'STATE':<10} {'SECTION':<30} DETAIL")
    for r in rep["readiness"]:
        print(f"  {r['section']:>3}  {r['state']:<10} {r['title'][:30]:<30} "
              f"{r['detail'][:60]}")
    rc = rep["readiness_counts"]
    print(f"\n  {rc['READY']} READY · {rc['FETCHABLE']} FETCHABLE · "
          f"{rc['BLOCKED']} BLOCKED")
    remedies = sorted({r["remedy"] for r in rep["readiness"] if r["remedy"]})
    if remedies:
        print("\n  Run these now to close the document gaps:")
        for m in remedies:
            print(f"    {m}")
    phases: dict[str, list[int]] = {}
    for r in rep["readiness"]:
        if r["state"] == "BLOCKED" and r["blocked_by"]:
            phases.setdefault(r["blocked_by"], []).append(r["section"])
    for ph, secs in sorted(phases.items()):
        print(f"\n  {ph} unblocks {len(secs)} section(s): "
              f"{', '.join(str(s) for s in secs)}")
    if phases:
        print("  No command populates these until the phase ships.")

    print("\nDATA INTEGRITY")
    for ch in rep["integrity"]:
        mark = {"PASS": "ok  ", "WARN": "WARN", "FAIL": "FAIL"}[ch["status"]]
        print(f"  [{mark}] {ch['name']}")
        print(f"         {ch['detail']}")
    ic = rep["integrity_counts"]
    print(f"\n  {ic['PASS']} pass · {ic['WARN']} warn · {ic['FAIL']} fail")
    print(f"\n  VERDICT: {'publishable' if rep['publishable'] else 'DO NOT PUBLISH'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    store = Store()
    reports, worst = [], 0
    for t in a.names:
        rep = run(store, t)
        if rep is None:
            print(f"could not resolve '{t}'")
            worst = max(worst, 1)
            continue
        reports.append(rep)
        if not rep["publishable"]:
            worst = max(worst, 2)
    if a.json:
        print(json.dumps(reports, indent=2))
    else:
        for rep in reports:
            _print(rep)
    return worst


if __name__ == "__main__":
    sys.exit(main())
