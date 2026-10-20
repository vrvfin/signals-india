r"""
peer_check.py — where a reporting company ranks against its peers THIS quarter (O8).

Peer group = company_classification.csv `peer_group` (T9 taxonomy). Peer numbers =
results.parquet (Screener's latest-results feed): sales / PAT / EPS YoY for the SAME
quarter label. The feed does not see every reporter, so the block always says how many
of the group's companies it found — a rank among 3 of 9 is not a rank among 9.

Used by quarter_teardown.py (block "Peer check" in the per-company teardown). No LLM.

    python scripts/peer_check.py --self-test
"""
from __future__ import annotations

import html as _html
import sys

import pandas as pd

MAX_ROWS = 8


def _yoy(rows: pd.DataFrame, *terms: str):
    if rows is None or rows.empty:
        return None
    m = rows["metric"].astype(str).str.lower()
    for t in terms:
        hit = rows[m.str.contains(t, na=False, regex=False)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        y = pd.to_numeric(r.get("yoy_pct"), errors="coerce")
        if pd.isna(y):
            lv = pd.to_numeric(r.get("latest_val"), errors="coerce")
            yv = pd.to_numeric(r.get("yearago_val"), errors="coerce")
            if pd.notna(lv) and pd.notna(yv) and yv > 0:
                y = (lv - yv) / yv * 100.0
        return None if pd.isna(y) else float(y)
    return None


def peer_table(isin: str, cls: pd.DataFrame, res: pd.DataFrame, quarter: str) -> dict:
    """{'group', 'n_group', 'rows': [{isin, symbol, sales, pat, eps, me}], 'rank_pat',
    'rank_sales', 'n_found'} or {} when the company has no peer group. Pure.

    `quarter` is results.parquet's latest_q label for the company (e.g. 'Sep 2026')."""
    if cls is None or cls.empty or res is None or res.empty:
        return {}
    c = cls[cls["isin"].astype(str) == str(isin)]
    if c.empty:
        return {}
    group = str(c["peer_group"].iloc[0] or "").strip()
    if not group or group.lower() in ("nan", "none", "unknown"):
        return {}
    members = cls[cls["peer_group"].astype(str).str.strip() == group]
    sym_of = dict(zip(members["isin"].astype(str), members["symbol"].astype(str)))
    q = res[(res["isin"].astype(str).isin(sym_of)) & (res["latest_q"].astype(str) == str(quarter))]
    rows = []
    for i, g in q.groupby(q["isin"].astype(str)):
        rows.append({"isin": i, "symbol": sym_of.get(i, i),
                     "sales": _yoy(g, "sales", "revenue"), "pat": _yoy(g, "net prof"),
                     "eps": _yoy(g, "eps"), "me": i == str(isin)})
    if not any(r["me"] for r in rows) or len(rows) < 2:
        return {"group": group, "n_group": len(sym_of), "rows": rows, "n_found": len(rows),
                "rank_pat": None, "rank_sales": None}

    def rank(key):
        vals = sorted((r for r in rows if r[key] is not None), key=lambda r: -r[key])
        for k, r in enumerate(vals, 1):
            if r["me"]:
                return (k, len(vals))
        return None

    rows.sort(key=lambda r: (r["pat"] is None, -(r["pat"] or 0)))
    return {"group": group, "n_group": len(sym_of), "rows": rows, "n_found": len(rows),
            "rank_pat": rank("pat"), "rank_sales": rank("sales")}


def _p(v):
    return "–" if v is None else f"{v:+.0f}%"


def render(pt: dict) -> str:
    """HTML block for the teardown; '' when there is nothing to compare. Pure."""
    if not pt or pt.get("n_found", 0) < 2 or not any(r["me"] for r in pt["rows"]):
        return ""
    def _rank_txt(rk, what):
        return (f"<b>{rk[0]}</b> of {rk[1]} on {what}" if rk else f"no {what} figure")
    lead = (f"Ranked {_rank_txt(pt['rank_pat'], 'PAT growth')} and "
            f"{_rank_txt(pt['rank_sales'], 'sales growth')} among the "
            f"<b>{pt['n_found']}</b> of {pt['n_group']} <i>{_html.escape(pt['group'])}</i> "
            f"companies that have reported this quarter on our results feed.")
    trs = []
    for r in pt["rows"][:MAX_ROWS]:
        bg = "background:#fff6d6;font-weight:700;" if r["me"] else ""
        trs.append(f"<tr style='{bg}'>" + "".join(
            f"<td style='padding:2px 8px;border-top:1px solid #eee;text-align:{al}'>{v}</td>"
            for v, al in ((_html.escape(r["symbol"]), "left"), (_p(r["sales"]), "right"),
                          (_p(r["pat"]), "right"), (_p(r["eps"]), "right"))) + "</tr>")
    more = (f"<div style='color:#888;font-size:11px'>+{len(pt['rows']) - MAX_ROWS} more "
            f"peers</div>" if len(pt["rows"]) > MAX_ROWS else "")
    return ("<h3 style='margin:16px 0 4px'>&#128101; Peer check</h3>"
            f"<div style='font-size:12.5px;margin:0 0 4px'>{lead}</div>"
            "<table cellspacing='0' style='border-collapse:collapse;font-size:12px'>"
            "<tr style='background:#f4f6f8'><th style='text-align:left;padding:2px 8px'>Peer"
            "</th><th style='padding:2px 8px'>Sales YoY</th><th style='padding:2px 8px'>PAT YoY"
            "</th><th style='padding:2px 8px'>EPS YoY</th></tr>" + "".join(trs) + "</table>" + more)


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    cls = pd.DataFrame([
        {"isin": "A", "symbol": "AAA", "peer_group": "EMS"},
        {"isin": "B", "symbol": "BBB", "peer_group": "EMS"},
        {"isin": "C", "symbol": "CCC", "peer_group": "EMS"},
        {"isin": "D", "symbol": "DDD", "peer_group": "EMS"},
        {"isin": "X", "symbol": "XXX", "peer_group": "Cement"}])

    def rr(isin, s, p, q="Sep 2026"):
        return [{"isin": isin, "latest_q": q, "metric": "Sales", "yoy_pct": s},
                {"isin": isin, "latest_q": q, "metric": "Net Profit", "yoy_pct": p}]
    res = pd.DataFrame(rr("A", 40, 60) + rr("B", 20, 80) + rr("C", 10, -5)
                       + rr("D", 90, 90, q="Jun 2026") + rr("X", 5, 5))
    pt = peer_table("A", cls, res, "Sep 2026")
    check("only same-quarter peers in the same group", pt["n_found"] == 3 and pt["n_group"] == 4)
    check("rank on PAT: 2nd of 3", pt["rank_pat"] == (2, 3))
    check("rank on sales: 1st of 3", pt["rank_sales"] == (1, 3))
    html = render(pt)
    check("render says found-of-group and highlights the company",
          "3</b> of 4" in html and "#fff6d6" in html and "<b>2</b> of 3 on PAT growth" in html)
    check("no peer group -> nothing", peer_table("Z", cls, res, "Sep 2026") == {}
          and render({}) == "")
    alone = peer_table("X", cls, res, "Sep 2026")
    check("only the company itself reported -> no block", render(alone) == "")
    print(f"\npeer_check self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
