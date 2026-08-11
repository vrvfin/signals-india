r"""
deck_teardown.py — the deck-nuance pass: operating engine, deck-vs-deck diff, framing
flags, open questions.

Runs as an ADDITIVE third structured pass inside extract_presentation.py, guarded by
`--teardown` AND membership of PF u watchlist. Default off. It appends only to its own
parquets, so with the flag absent every existing Phase-2 output is byte-identical.

WHY THIS EXISTS
---------------
The current presentation pass captures slide titles and ESG boilerplate. Measured on
APLAPOLLO's live Q1 FY27 deck: 12 highlights of which 9 are slide headings, and all 4
guidance rows are sustainability targets (Net Zero 2050, female workforce +1%/yr). Zero
volume, zero utilisation, zero order book. The operating engine of the business is simply
not being captured.

GROUNDING
---------
Every row must survive `provenance.grounded()`: the number the model reports has to appear
verbatim inside the evidence span the model copied out of the deck. A model that invents
"utilisation rose to 87%" without 87 appearing in its own quoted text loses that row. This
is enforced here, at parse time, not requested politely in the prompt.

    python scripts/deck_teardown.py --self-test
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd

from provenance import grounded

PROMPT_FILE = "deck_teardown_prompt.txt"

_BASE = ["isin", "symbol", "company_name", "quarter"]
_TAIL = ["processed_at", "source_doc_id"]

DECK_METRICS_COLS = _BASE + ["category", "metric", "value", "unit", "period",
                             "slide_ref", "evidence"] + _TAIL
DECK_DIFF_COLS = _BASE + ["prev_quarter", "change_type", "item", "prior_state",
                          "current_state", "severity", "evidence"] + _TAIL
DECK_FLAGS_COLS = _BASE + ["flag_type", "slide_ref", "severity", "evidence"] + _TAIL
DECK_QUESTIONS_COLS = _BASE + ["question"] + _TAIL

CATEGORIES = {"capacity", "utilisation", "orderbook", "volume", "realisation",
              "segment_mix", "geo_mix"}
CHANGE_TYPES = {"kpi_dropped", "target_moved", "definition_changed", "new_emphasis",
                "de_emphasised"}
FLAG_TYPES = {"axis_truncated", "axis_rebased", "absolute_to_percent",
              "percent_to_absolute", "peer_set_cherry_picked", "non_gaap_introduced",
              "caveat_in_footnote", "metric_redefined", "misleading_scale"}
SEVERITIES = {"high", "medium", "low"}

# Emitted by the model despite the prompt forbidding it. Dropped rather than stored:
# these are what currently crowd out the operating metrics.
_BANNED_METRIC_WORDS = ("emission", "scope 1", "scope 2", "net zero", "renewable",
                        "diversity", "female", "csr", "esg", "djsi", "sustainab",
                        "carbon", "water", "waste")
# Financial summary lines belong to the audited filing, never to a deck row.
_FINANCIAL_WORDS = ("revenue", "ebitda", "pat", "profit after tax", "net profit",
                    "turnover", "eps")

MAX_METRICS, MAX_DIFF, MAX_FLAGS, MAX_Q = 20, 10, 8, 5


def _s(v, n=400) -> str:
    if v is None:
        return ""
    t = str(v).strip()
    return "" if t.lower() in ("nan", "none", "null", "data_missing") else t[:n]


def _f(v):
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _clamp(v, allowed, default):
    t = _s(v, 40).lower().replace(" ", "_").replace("-", "_")
    return t if t in allowed else default


def _banned(metric: str, evidence: str) -> bool:
    blob = f"{metric} {evidence}".lower()
    return (any(w in blob for w in _BANNED_METRIC_WORDS)
            or any(w in metric.lower() for w in _FINANCIAL_WORDS))


def parse_teardown(payload, row, quarter: str, now_str: str,
                   prev_quarter: str = "") -> dict:
    """Model output -> four row lists, with every ungrounded claim dropped.

    `payload` may be the raw response text or an already-parsed dict.
    Returns {"metrics": [...], "diff": [...], "flags": [...], "questions": [...],
             "dropped": {...}} — `dropped` is a per-reason count for the run log, so a
    model that starts hallucinating shows up in the logs instead of in the data.
    """
    if isinstance(payload, dict):
        obj = payload
    else:
        obj = _first_json_object(_s(payload, 400_000))
    if not obj:
        return {"metrics": [], "diff": [], "flags": [], "questions": [],
                "dropped": {"unparseable_response": 1}}

    base = {"isin": _s(row.get("isin"), 20), "symbol": _s(row.get("symbol"), 24),
            "company_name": _s(row.get("company_name"), 120), "quarter": quarter,
            "processed_at": now_str, "source_doc_id": _s(row.get("doc_id"), 80)}
    dropped: dict[str, int] = {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    # ---- metrics
    metrics, seen = [], set()
    for o in (obj.get("metrics") or [])[: MAX_METRICS * 2]:
        if not isinstance(o, dict):
            continue
        ev, name = _s(o.get("evidence"), 300), _s(o.get("metric"), 120)
        val = _f(o.get("value"))
        if not name:
            drop("no_metric_name"); continue
        if not ev:
            drop("no_evidence"); continue
        if _banned(name, ev):
            drop("esg_or_financial_row"); continue
        if val is not None and not grounded(val, ev, tolerance_pct=1.0):
            drop("value_not_in_evidence"); continue
        cat = _clamp(o.get("category"), CATEGORIES, "")
        if not cat:
            drop("bad_category"); continue
        k = (cat, name.lower(), str(val), _s(o.get("period"), 20))
        if k in seen:
            drop("duplicate"); continue
        seen.add(k)
        metrics.append({**base, "category": cat, "metric": name, "value": val,
                        "unit": _s(o.get("unit"), 16), "period": _s(o.get("period"), 20),
                        "slide_ref": _s(o.get("slide_ref"), 60), "evidence": ev})

    # ---- diff
    diff, seen_d = [], set()
    for o in (obj.get("diff") or [])[: MAX_DIFF * 2]:
        if not isinstance(o, dict):
            continue
        ct = _clamp(o.get("change_type"), CHANGE_TYPES, "")
        item = _s(o.get("item"), 160)
        ev = _s(o.get("evidence"), 300)
        if not ct or not item:
            drop("bad_diff_row"); continue
        # A dropped KPI is proven by ABSENCE, so it is the one row that cannot quote
        # this deck. Everything else must.
        if ct != "kpi_dropped" and not ev:
            drop("diff_no_evidence"); continue
        if not prev_quarter:
            drop("diff_without_prior_deck"); continue
        k = (ct, item.lower())
        if k in seen_d:
            drop("duplicate"); continue
        seen_d.add(k)
        diff.append({**base, "prev_quarter": prev_quarter, "change_type": ct,
                     "item": item, "prior_state": _s(o.get("prior_state"), 200),
                     "current_state": _s(o.get("current_state"), 200),
                     "severity": _clamp(o.get("severity"), SEVERITIES, "medium"),
                     "evidence": ev})

    # ---- flags
    flags, seen_f = [], set()
    for o in (obj.get("flags") or [])[: MAX_FLAGS * 2]:
        if not isinstance(o, dict):
            continue
        ft = _clamp(o.get("flag_type"), FLAG_TYPES, "")
        ev = _s(o.get("evidence"), 300)
        if not ft:
            drop("bad_flag_type"); continue
        if len(ev) < 10:
            drop("flag_no_evidence"); continue
        if ft in seen_f:
            drop("duplicate"); continue
        seen_f.add(ft)
        flags.append({**base, "flag_type": ft, "slide_ref": _s(o.get("slide_ref"), 60),
                      "severity": _clamp(o.get("severity"), SEVERITIES, "medium"),
                      "evidence": ev})

    # ---- questions
    questions, seen_q = [], set()
    for q in (obj.get("questions") or [])[: MAX_Q * 2]:
        t = _s(q, 300)
        if len(t) < 15 or t.lower() in seen_q:
            continue
        seen_q.add(t.lower())
        questions.append({**base, "question": t})

    return {"metrics": metrics[:MAX_METRICS], "diff": diff[:MAX_DIFF],
            "flags": flags[:MAX_FLAGS], "questions": questions[:MAX_Q],
            "dropped": dropped}


def _first_json_object(text: str):
    """The outermost JSON object in a response, tolerating fences and trailing prose."""
    import json
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t.lstrip("`")
        t = t[4:] if t[:4].lower() == "json" else t
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def prior_deck_context(metrics_df: pd.DataFrame, isin: str, quarter: str,
                       limit: int = 24) -> tuple[str, str]:
    """([PRIOR_DECK] block, prev_quarter) from previously stored deck_metrics rows.

    Without this the diff cannot run — which is why parse_teardown discards every diff
    row when prev_quarter is empty rather than letting the model speculate.
    """
    if metrics_df is None or metrics_df.empty:
        return "", ""
    import redflag_register as RR
    d = metrics_df[metrics_df["isin"].astype(str) == isin].copy()
    if d.empty:
        return "", ""
    cur = RR.period_end(quarter)
    d["_e"] = d["quarter"].map(lambda q: RR.period_end(q))
    d = d[d["_e"].notna()]
    if cur:
        d = d[d["_e"] < cur]
    if d.empty:
        return "", ""
    newest = d["_e"].max()
    prev = d[d["_e"] == newest]
    pq = str(prev.iloc[0]["quarter"])
    lines = [f"- {r['category']}: {r['metric']} = {r['value']} {r['unit']}"
             f" ({r['period']})".rstrip()
             for _, r in prev.head(limit).iterrows()]
    return ("\n[PRIOR_DECK] — what the previous deck (" + pq + ") showed. Use ONLY this "
            "for prior_state, and report a kpi_dropped for anything here that the "
            "current deck does not show:\n" + "\n".join(lines) + "\n"), pq


def run_teardown(gemini, prompt: str, doc_bytes: bytes, row, quarter: str, now_str: str,
                 prior_block: str = "", prev_quarter: str = "") -> dict:
    """One structured pass over the deck. Shared machinery only (rule 4)."""
    from _extractor_base import run_structured_over_doc
    resp = run_structured_over_doc(
        gemini, prompt + prior_block, doc_bytes, max_output_tokens=4096,
        name=f"{row.get('symbol', 'DOC')}_PPT_teardown")
    return parse_teardown(resp, row, quarter, now_str, prev_quarter)


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

    row = {"isin": "INE702C01027", "symbol": "APLAPOLLO",
           "company_name": "APL Apollo", "doc_id": "doc1"}
    now = "2026-08-08T00:00:00"

    resp = """```json
    {"quarter":"Q1FY27",
     "metrics":[
      {"category":"capacity","metric":"Installed capacity","value":4.5,"unit":"MT",
       "period":"Q1FY27","slide_ref":"12","evidence":"Installed capacity of 4.5 MT as at June 2026"},
      {"category":"utilisation","metric":"Utilisation","value":87,"unit":"%",
       "period":"Q1FY27","slide_ref":"12","evidence":"capacity utilisation stood at 71% for the quarter"},
      {"category":"volume","metric":"Sales volume","value":812000,"unit":"units",
       "period":"Q1FY27","slide_ref":"5","evidence":"Volume of 812,000 tonnes in Q1"},
      {"category":"other","metric":"Scope 1 emissions","value":25,"unit":"%",
       "period":"2030","slide_ref":"40","evidence":"reduce Scope 1 emissions 25% by 2030"},
      {"category":"volume","metric":"Revenue","value":5607,"unit":"cr",
       "period":"Q1FY27","slide_ref":"3","evidence":"Revenue of 5,607 cr"},
      {"category":"orderbook","metric":"Order book","value":null,"unit":null,
       "period":"Q1FY27","slide_ref":"9","evidence":"order book remains healthy"}
     ],
     "diff":[
      {"change_type":"kpi_dropped","item":"Realisation per tonne","prior_state":"shown Q4FY26",
       "current_state":"absent","severity":"high","evidence":""},
      {"change_type":"target_moved","item":"8 MT capacity","prior_state":"FY27",
       "current_state":"FY28","severity":"high","evidence":"8 MT capacity target by FY28"}
     ],
     "flags":[
      {"flag_type":"absolute_to_percent","slide_ref":"7","severity":"medium",
       "evidence":"volume growth shown as % where Q4 deck showed tonnes"},
      {"flag_type":"telepathy","slide_ref":"9","severity":"high","evidence":"seems misleading"}
     ],
     "questions":["What drove the sequential fall in realisation per tonne?","ok"]}
    ```"""

    r = parse_teardown(resp, row, "Q1FY27", now, prev_quarter="Q4FY26")
    names = [m["metric"] for m in r["metrics"]]
    check("keeps grounded metric", "Installed capacity" in names)
    check("drops ungrounded (87 vs 71)", "Utilisation" not in names)
    check("keeps comma'd number", "Sales volume" in names)
    check("drops ESG row", "Scope 1 emissions" not in names)
    check("drops financial row", "Revenue" not in names)
    check("keeps null-value metric", "Order book" in names)
    check("drop reasons logged", r["dropped"].get("value_not_in_evidence") == 1)
    check("esg drop logged", r["dropped"].get("esg_or_financial_row", 0) >= 1)

    check("diff parsed", len(r["diff"]) == 2)
    check("kpi_dropped allowed without evidence",
          any(d["change_type"] == "kpi_dropped" for d in r["diff"]))
    check("prev_quarter stamped", r["diff"][0]["prev_quarter"] == "Q4FY26")
    check("bad flag_type dropped", [f["flag_type"] for f in r["flags"]] ==
          ["absolute_to_percent"])
    check("short question dropped", len(r["questions"]) == 1)

    no_prior = parse_teardown(resp, row, "Q1FY27", now, prev_quarter="")
    check("no prior deck -> no diff", no_prior["diff"] == [])
    check("no-prior drop logged",
          no_prior["dropped"].get("diff_without_prior_deck", 0) >= 1)

    check("garbage -> empty", parse_teardown("not json", row, "Q1FY27", now)["metrics"] == [])
    check("unparseable logged",
          parse_teardown("not json", row, "Q1FY27", now)["dropped"].get(
              "unparseable_response") == 1)

    prev = pd.DataFrame([
        {"isin": "INE702C01027", "quarter": "Q4 FY26", "category": "volume",
         "metric": "Sales volume", "value": 790000, "unit": "t", "period": "Q4FY26"},
        {"isin": "INE702C01027", "quarter": "Q3 FY26", "category": "volume",
         "metric": "Sales volume", "value": 700000, "unit": "t", "period": "Q3FY26"},
    ])
    blk, pq = prior_deck_context(prev, "INE702C01027", "Q1 FY27")
    check("prior deck picks newest prior", pq == "Q4 FY26")
    check("prior block mentions the metric", "Sales volume" in blk)
    blk2, pq2 = prior_deck_context(prev, "INE_NONE", "Q1 FY27")
    check("prior deck empty for unknown isin", pq2 == "" and blk2 == "")

    check("cols wired", len(DECK_METRICS_COLS) == 13 and len(DECK_DIFF_COLS) == 13)

    print(f"\ndeck_teardown self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    ap.print_help()
