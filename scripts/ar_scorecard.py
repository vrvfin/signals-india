r"""
ar_scorecard.py — tabulate the AR forensic risk score that is already generated and thrown
away.

`annual_report_prompt.txt` section 6 makes every annual-report run produce a weighted
0-10 risk matrix (Cash Flow Quality 25% / Accounting Integrity 25% / Balance Sheet Risk
20% / Governance & Promoter 20% / Regulatory & Auditor Transparency 10%) with a
Clean / Monitor / Elevated / Avoid label. It lands in `company_page.md` as prose and no
code has ever read it back. `.claude/plans/ar-tabulation.md` planned AR_SCORECARD_COLS
and it was never built.

This parses those blocks out of the markdown that already exists on Drive. Nothing is
re-extracted and no model is called, so it covers every annual report ever processed.

THE TRAP THIS GUARDS AGAINST
----------------------------
Some scorecards read 5.0 / MONITOR with every justification saying "DATA_MISSING; neutral
score assigned." That is not a Monitor verdict — it is the absence of one. Presenting it
as a risk assessment would be worse than showing nothing. Every parsed row therefore
carries `n_data_missing` and a `confidence`, and `confidence == "none"` rows must be
rendered as "not assessed", never as a score.

    python scripts/ar_scorecard.py --isin INE702C01027     # live, read-only
    python scripts/ar_scorecard.py --self-test             # no Drive
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd

AR_SCORECARD_COLS = ["isin", "symbol", "company_name", "fy_year", "dimension",
                     "weight_pct", "score", "justification", "overall_score",
                     "risk_label", "n_dims", "n_data_missing", "confidence",
                     "parsed_at", "source_doc"]

OUT_NAME = "ar_scorecard.parquet"

# Canonical dimension names. The model paraphrases the prompt's labels ("Governance",
# "Governance/Promoter", "Transparency"), so match on a distinctive substring.
DIMENSIONS = [
    ("cash_flow_quality", ("cash flow",)),
    ("accounting_integrity", ("accounting",)),
    ("balance_sheet_risk", ("balance sheet",)),
    ("governance_promoter", ("governance", "promoter")),
    ("regulatory_transparency", ("transparency", "regulatory", "auditor disclosure")),
]
CANON_WEIGHT = {"cash_flow_quality": 25, "accounting_integrity": 25,
                "balance_sheet_risk": 20, "governance_promoter": 20,
                "regulatory_transparency": 10}

LABELS = ("clean", "monitor", "elevated", "avoid")

# §6 heading, with or without a markdown prefix and with or without the parenthetical.
_SEC6 = re.compile(r"^[#\s>*-]{0,8}6\.\s*FORENSIC\s+FINANCIAL\s+RISK\s+SCORECARD.*$",
                   re.I | re.M)
# The next top-level section ends the block.
_SEC7 = re.compile(r"^[#\s>*-]{0,8}7\.\s+\S", re.M)
# A company_page.md document header: "## FY24 Annual Report — <title>"
_DOCHDR = re.compile(r"^##\s+(?P<period>\S+)\s+(?P<label>[A-Za-z ]+?)\s+[—-]\s+(?P<title>.*)$",
                     re.M)

_SCORE_LINE = re.compile(
    r"(?:total\s+weighted|weighted\s+overall|overall\s+weighted|total)\s+risk\s+score"
    r"\s*[:\-]?\s*\**\s*(?P<v>\d+(?:\.\d+)?)", re.I)
_LABEL_LINE = re.compile(r"risk\s+label\s*[:\-]?\s*(?P<v>.{0,60})", re.I)
_ROW = re.compile(r"^\s*\|(?P<cells>.+)\|\s*$", re.M)


def _cells(line: str) -> list[str]:
    return [c.strip().strip("*").strip() for c in line.split("|")[1:-1]]


def _num(tok: str):
    m = re.search(r"-?\d+(?:\.\d+)?", str(tok or ""))
    return float(m.group()) if m else None


def _canon_dim(label: str) -> str | None:
    low = str(label or "").lower()
    for canon, keys in DIMENSIONS:
        if any(k in low for k in keys):
            return canon
    return None


def _label_of(text: str) -> str:
    low = text.lower()
    for lab in LABELS:
        if lab in low:
            return lab.capitalize()
    return ""


def parse_block(block: str) -> dict:
    """One §6 block -> {dims: [...], overall_score, risk_label, n_data_missing}."""
    dims, seen = [], set()
    for m in _ROW.finditer(block):
        cells = _cells(m.group(0))
        if len(cells) < 2:
            continue
        canon = _canon_dim(cells[0])
        if not canon or canon in seen:
            continue
        # Column layout varies: [dim, weight, score, justification] or [dim, score, just].
        weight, score, just = None, None, ""
        nums = [(_num(c), c) for c in cells[1:]]
        if len(cells) >= 4 and "%" in cells[1]:
            weight, score, just = _num(cells[1]), _num(cells[2]), cells[3]
        else:
            for val, raw in nums:
                if val is None:
                    continue
                if "%" in raw and weight is None:
                    weight = val
                elif score is None and 0 <= val <= 10:
                    score = val
            just = cells[-1] if not _num(cells[-1]) else ""
        if score is None:
            continue
        seen.add(canon)
        dims.append({"dimension": canon,
                     "weight_pct": weight if weight is not None else CANON_WEIGHT[canon],
                     "score": score, "justification": just[:400]})

    overall = None
    ms = _SCORE_LINE.search(block)
    if ms:
        overall = float(ms.group("v"))
    label = ""
    ml = _LABEL_LINE.search(block)
    if ml:
        label = _label_of(ml.group("v"))
    if not label and overall is not None:
        label = ("Clean" if overall <= 2 else "Monitor" if overall <= 5
                 else "Elevated" if overall <= 7 else "Avoid")

    n_missing = sum(1 for d in dims if "data_missing" in d["justification"].lower())
    return {"dims": dims, "overall_score": overall, "risk_label": label,
            "n_data_missing": n_missing}


def _confidence(n_dims: int, n_missing: int) -> str:
    """A score built on DATA_MISSING is not a verdict.

    none   -> every dimension was DATA_MISSING; render "not assessed", never a score
    low    -> most dimensions unevidenced
    medium -> one gap
    high   -> fully evidenced
    """
    if n_dims == 0:
        return "none"
    if n_missing >= n_dims:
        return "none"
    if n_missing >= max(2, n_dims // 2):
        return "low"
    if n_missing >= 1:
        return "medium"
    return "high"


def _fy_for(md: str, pos: int) -> tuple[str, str]:
    """Walk back to the enclosing '## <period> <Doc Label> — <title>' header."""
    best = ("", "")
    for m in _DOCHDR.finditer(md):
        if m.start() > pos:
            break
        best = (m.group("period").strip(), m.group("title").strip()[:120])
    return best


def parse_company_page(md: str, isin: str = "", symbol: str = "",
                       company_name: str = "") -> pd.DataFrame:
    """Every §6 scorecard in one company_page.md, newest fiscal year last."""
    rows, now = [], datetime.now().isoformat(timespec="seconds")
    for m in _SEC6.finditer(md):
        start = m.end()
        nxt = _SEC7.search(md, start)
        block = md[start: nxt.start() if nxt else start + 4000]
        parsed = parse_block(block)
        if not parsed["dims"]:
            continue
        fy, doc = _fy_for(md, m.start())
        conf = _confidence(len(parsed["dims"]), parsed["n_data_missing"])
        for d in parsed["dims"]:
            rows.append({
                "isin": isin, "symbol": symbol, "company_name": company_name,
                "fy_year": fy, "dimension": d["dimension"],
                "weight_pct": d["weight_pct"], "score": d["score"],
                "justification": d["justification"],
                "overall_score": parsed["overall_score"],
                "risk_label": parsed["risk_label"],
                "n_dims": len(parsed["dims"]),
                "n_data_missing": parsed["n_data_missing"],
                "confidence": conf, "parsed_at": now, "source_doc": doc,
            })
    df = pd.DataFrame(rows, columns=AR_SCORECARD_COLS)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["isin", "fy_year", "dimension"], keep="last")


def latest(df: pd.DataFrame) -> pd.DataFrame:
    """The newest scorecard only, one row per dimension."""
    if df.empty:
        return df
    import redflag_register as RR
    d = df.copy()
    d["_e"] = d["fy_year"].map(lambda f: (RR.period_end(f) or __import__("datetime")
                                          .date(1900, 1, 1)).toordinal())
    return d[d["_e"] == d["_e"].max()].drop(columns=["_e"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    real = """
## FY24 Annual Report — Annual Report 2023-24

### 6. FORENSIC FINANCIAL RISK SCORECARD
| Risk Dimension | Weight | Score (0-10) | Justification |
| :--- | :--- | :--- | :--- |
| Cash Flow Quality | 25% | 2 | CFO consistently exceeds PAT. |
| Accounting Integrity | 25% | 2 | No aggressive capitalization or policy shifts. |
| Balance Sheet Risk | 20% | 3 | Moderate debt levels; RPTs are strategic. |
| Governance/Promoter | 20% | 3 | Strong attendance; no pledging. |
| Transparency | 10% | 2 | High disclosure levels in annual report. |

*   **Weighted Overall Risk Score:** 2.4 / 10
*   **Risk Label:** [CLEAN]

7. INVESTMENT THESIS MATRIX
"""
    df = parse_company_page(real, "INE1", "SYM", "Co")
    check("parses 5 dimensions", len(df) == 5)
    check("overall parsed", df["overall_score"].iloc[0] == 2.4)
    check("label parsed", df["risk_label"].iloc[0] == "Clean")
    check("fy from doc header", df["fy_year"].iloc[0] == "FY24")
    check("weights parsed", set(df["weight_pct"]) == {25, 20, 10})
    check("confidence high", df["confidence"].iloc[0] == "high")

    missing = """
## FY25 Annual Report — AR 2024-25

### 6. FORENSIC FINANCIAL RISK SCORECARD
| Risk Dimension | Weight | Score (0-10) | Justification |
| :--- | :--- | :--- | :--- |
| Cash Flow Quality | 25% | 5 | DATA_MISSING; neutral score assigned. |
| Accounting Integrity | 25% | 5 | DATA_MISSING; neutral score assigned. |
| Balance Sheet Risk | 20% | 5 | DATA_MISSING; neutral score assigned. |
| Governance | 20% | 5 | DATA_MISSING; neutral score assigned. |
| Transparency | 10% | 5 | DATA_MISSING; neutral score assigned. |

*   **Total Weighted Risk Score:** 5.0
*   **Risk Label:** MONITOR (Due to lack of available financial data).

7. NEXT
"""
    dm = parse_company_page(missing, "INE1", "SYM", "Co")
    check("all-DATA_MISSING -> confidence none", dm["confidence"].iloc[0] == "none")
    check("all-DATA_MISSING counted", dm["n_data_missing"].iloc[0] == 5)
    check("score still captured", dm["overall_score"].iloc[0] == 5.0)

    bare = """
6. FORENSIC FINANCIAL RISK SCORECARD (MANDATORY WEIGHTED MATRIX)
    Weighted Overall Risk Score: 4.5 / 10
    Risk Label: Monitor

| Risk Dimension | Weight | Score (0-10) | Justification |
| Cash Flow Quality | 25% | 4 | Volatile CFO. |
| Accounting Integrity | 25% | 5 | Policy change in FY23. |
| Balance Sheet Risk | 20% | 4 | High RPT volume. |
| Governance | 20% | 5 | KMP churn. |
| Transparency | 10% | 4 | Adequate. |
"""
    b = parse_company_page(bare, "INE2", "S2", "C2")
    check("bare heading parsed", len(b) == 5)
    check("score before table parsed", b["overall_score"].iloc[0] == 4.5)
    check("label before table parsed", b["risk_label"].iloc[0] == "Monitor")

    two = real + missing
    t = parse_company_page(two, "INE1", "SYM", "Co")
    check("two scorecards kept apart", set(t["fy_year"]) == {"FY24", "FY25"})
    check("latest picks FY25", set(latest(t)["fy_year"]) == {"FY25"})

    check("no §6 -> empty", parse_company_page("# nothing here").empty)
    check("derived label when absent",
          parse_block("| Cash Flow Quality | 25% | 8 | bad |\n"
                      "Total Weighted Risk Score: 8.5")["risk_label"] == "Avoid")

    print(f"\nar_scorecard self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isin", help="Parse one company's page from Drive (read-only).")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())
    if not args.isin:
        ap.print_help()
        return

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))
    from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                                 download_bytes)
    drive = get_drive()
    repo = get_or_create_subfolder(drive, os.environ["GDRIVE_FOLDER_ID"], "company_repo")
    cdir = get_or_create_subfolder(drive, repo, args.isin)
    fid = find_file(drive, cdir, "company_page.md")
    if not fid:
        print("no company_page.md")
        return
    md = download_bytes(drive, fid).decode("utf-8", "ignore")
    df = parse_company_page(md, args.isin)
    if df.empty:
        print("no §6 scorecard blocks found")
        return
    print(f"{len(df)} dimension rows across {df['fy_year'].nunique()} annual reports\n")
    for fy, g in df.groupby("fy_year"):
        r = g.iloc[0]
        warn = "   <- NOT ASSESSED (all DATA_MISSING)" if r["confidence"] == "none" else ""
        print(f"  {fy:9s} score={r['overall_score']}  {r['risk_label']:9s} "
              f"conf={r['confidence']:6s} missing={r['n_data_missing']}/{r['n_dims']}{warn}")
        for _, d in g.iterrows():
            print(f"        {d['dimension']:26s} {d['weight_pct']:>3.0f}% "
                  f"{d['score']:>4.1f}  {str(d['justification'])[:64]}"
                  .encode("ascii", "ignore").decode())


if __name__ == "__main__":
    main()
