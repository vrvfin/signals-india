r"""
test_pat_growth_gallery.py — offline checks for build_gallery.pat_growth_rows and
the guidance-date sort the watchlist page uses.

Plain python, no pytest (repo convention, see test_phase1_engine.py).
Synthetic rows only: no Drive, no network.

  1. thresholds: exactly 50% YoY and exactly 0% QoQ are both OUT
  2. a year-ago loss or blank is out; a previous-quarter loss is in, with no QoQ %
  3. a blank previous quarter is out
  4. one quarter only, half-year rows never count, --quarter picks an older one
  5. newest reported first (IST date), then higher YoY
  6. other metrics ignored; a blank ISIN survives, keyed by slug
  7. the table and banner render for every kind of row
  8. watchlist sort: guidance_date first, date_added for a row without one
  9. statements top-up: only companies results.parquet lacks for the quarter, same
     rules, one basis per company, same quarter only, marked and dated
 10. quarter display: label + months, banner
 11. 'on both lists': guidance quarter pick, badge maps, both pages render the badge

Run:
    python scripts/tests/test_pat_growth_gallery.py
Exit code = number of failures.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import build_gallery as BG  # noqa: E402

FAILS = []


def chk(label, got, want):
    if got != want:
        FAILS.append(f"{label}: got {got!r}, want {want!r}")


def row(slug, latest, prev, yearago, q="Sep 2026", seen="2026-11-01T04:00:00",
        isin=None, metric="Net profit", name=None):
    return {"slug": slug, "isin": f"INE{slug}" if isin is None else isin,
            "company_name": name or slug, "metric": metric, "latest_q": q,
            "latest_val": latest, "prev_val": prev, "yearago_val": yearago,
            "first_seen_at": seen, "scraped_at": seen}


def slugs(rows, **kw):
    out, _ = BG.pat_growth_rows(pd.DataFrame(rows), **kw)
    return out["slug"].tolist()


# 1. thresholds are strict
got = slugs([row("JUSTOVER", 150.01, 100, 100), row("EXACT50", 150, 100, 100),
             row("FLATQOQ", 200, 200, 100), row("DOWNQOQ", 200, 201, 100),
             row("LOWYOY", 140, 100, 100)])
chk("thresholds", got, ["JUSTOVER"])

# 2. loss rules
got, _ = BG.pat_growth_rows(pd.DataFrame([
    row("YEARLOSS", 50, 40, -10), row("YEARNIL", 50, 40, 0),
    row("YEARBLANK", 50, 40, None), row("PREVLOSS", 50, -5, 10),
    row("PREVNIL", 50, 0, 10), row("NORMAL", 300, 200, 100)]))
chk("loss rules", sorted(got["slug"]), ["NORMAL", "PREVLOSS", "PREVNIL"])
prev_loss = got[got["slug"] == "PREVLOSS"].iloc[0]
chk("prev-loss has no QoQ %", pd.isna(prev_loss["qoq_pct"]), True)
chk("prev-loss wording", BG._pg_qoq_text(prev_loss), "loss → profit")
chk("prev-nil wording", BG._pg_qoq_text(got[got["slug"] == "PREVNIL"].iloc[0]),
    "from nil")
normal = got[got["slug"] == "NORMAL"].iloc[0]
chk("YoY from levels", round(normal["yoy_pct"], 2), 200.0)
chk("QoQ from levels", round(normal["qoq_pct"], 2), 50.0)
chk("normal wording", BG._pg_qoq_text(normal), "+50%")

# 3. a blank previous quarter can't prove QoQ growth
chk("blank prev out", slugs([row("NOPREV", 300, None, 100)]), [])

# 4. one quarter only
rows = [row("NEW", 300, 200, 100), row("OLD", 300, 200, 100, q="Jun 2026"),
        row("HALFYEAR", 300, 200, 100, q="HY-Sep 2026")]
out, label = BG.pat_growth_rows(pd.DataFrame(rows))
chk("auto = newest quarter", out["slug"].tolist(), ["NEW"])
chk("quarter label", label, "Q2 FY27")
out, label = BG.pat_growth_rows(pd.DataFrame(rows), quarter="Q1FY27")
chk("explicit quarter", (out["slug"].tolist(), label), (["OLD"], "Q1 FY27"))
chk("half-year only = nothing",
    slugs([row("HALFYEAR", 300, 200, 100, q="HY-Sep 2026")]), [])

# 5. order. 20:00 UTC on 1 Nov is already 2 Nov in IST.
out, _ = BG.pat_growth_rows(pd.DataFrame([
    row("OLDER", 900, 200, 100, seen="2026-10-30T04:00:00"),
    row("LOWER", 200, 150, 100, seen="2026-11-02T04:00:00"),
    row("HIGHER", 500, 150, 100, seen="2026-11-01T20:00:00")]))
chk("newest first, then YoY", out["slug"].tolist(), ["HIGHER", "LOWER", "OLDER"])
chk("reported shown in IST", out["reported"].iloc[0], "2026-11-02")

# 6. metrics and blank ISINs
got = slugs([row("SALESONLY", 300, 200, 100, metric="Sales"),
             row("EPSONLY", 300, 200, 100, metric="EPS"),
             row("REITA", 300, 200, 100, isin=""),
             row("REITB", 300, 200, 100, isin="nan"),
             row("REITC", 300, 200, 100, isin="None")])
chk("other metrics ignored", "SALESONLY" in got or "EPSONLY" in got, False)
chk("blank ISINs stay separate", sorted(got), ["REITA", "REITB", "REITC"])
out, _ = BG.pat_growth_rows(pd.DataFrame([row("REITA", 300, 200, 100, isin="")]))
chk("blank ISIN is empty, not 'nan'", out["isin"].iloc[0], "")
chk("same ISIN twice = latest row wins", slugs([
    row("TWOA", 300, 200, 100, isin="INE1", seen="2026-11-01T04:00:00"),
    row("TWOB", 300, 200, 100, isin="INE1", seen="2026-11-02T04:00:00")]), ["TWOB"])
chk("empty input", BG.pat_growth_rows(pd.DataFrame())[0].empty, True)
chk("missing columns", BG.pat_growth_rows(pd.DataFrame({"slug": ["X"]}))[0].empty, True)

# 7. rendering, including a name with '&' and a name-only row
out, _ = BG.pat_growth_rows(pd.DataFrame([
    row("AAA", 300, 200, 100, name="Alpha & Sons"),
    row("BBB", 50, -5, 10, isin="", name="Beta REIT")]))
out["symbol"] = ["AAA" if s == "AAA" else "" for s in out["slug"]]
out["in_pf"] = [True if s == "AAA" else None for s in out["slug"]]
table, annot = BG._pg_table(out), BG._pg_annot(out)
chk("name escaped", "Alpha &amp; Sons" in table, True)
chk("charted row links to its card", 'href="#ch' in table, True)
chk("name-only row shows slug, no link", ">BBB</span>" in table, True)
chk("banner only for charted rows", sorted(annot), ["AAA"])
chk("banner carries the quarter", "Q2 FY27" in annot["AAA"], True)
chk("PF marked", "IN PF" in annot["AAA"], True)

# 8. watchlist sort date
wl = pd.DataFrame({"date_added": ["2026-07-29", "2026-09-04", "2026-10-31", "2026-08-01"],
                   "guidance_date": ["2026-10-26", "2026-11-02", None, float("nan")]})
chk("guided date, else added", BG._wl_guided_on(wl).tolist(),
    ["2026-10-26", "2026-11-02", "2026-10-31", "2026-08-01"])
chk("table not rebuilt yet = date_added",
    BG._wl_guided_on(wl.drop(columns="guidance_date")).tolist(),
    wl["date_added"].tolist())

# 9. statements top-up (financials_3stmt) for companies results.parquet lacks
def f3(isin, sym, period, value, basis="consolidated", seen="2026-11-03T20:00:00",
       item="Net Profit", ptype="quarterly"):
    return {"isin": isin, "symbol": sym, "statement": "income", "line_item": item,
            "period": period, "period_type": ptype, "value": value, "basis": basis,
            "scraped_at": seen}


def trio(isin, sym, latest, prev, yearago, **kw):
    return [f3(isin, sym, "Sep 2026", latest, **kw), f3(isin, sym, "Jun 2026", prev, **kw),
            f3(isin, sym, "Sep 2025", yearago, **kw)]


res9 = pd.DataFrame([row("INRES", 300, 200, 100, isin="INE_R"),
                     row("RESLOSS", 300, 200, -5, isin="INE_L")])
fin9 = pd.DataFrame(
    trio("INE_R", "INRES", 999, 1, 1)                    # in results: never added again
    + trio("INE_L", "RESLOSS", 300, 200, 100)            # in results, left out there: stays out
    + trio("INE_A", "ADDME", 300, 200, 100)              # missing from results, passes
    + trio("INE_S", "SLOW", 120, 100, 100)               # missing, YoY only 20%
    + trio("INE_C", "CONS", 300, 200, 100)               # consolidated passes ...
    + trio("INE_C", "CONS", 300, 400, 100, basis="standalone")  # ... standalone ignored
    + trio("INE_O", "ONLYSTD", 300, 200, 100, basis="standalone")  # no consolidated
    + [f3("INE_M", "MIXED", "Sep 2026", 300), f3("INE_M", "MIXED", "Jun 2026", 200),
       f3("INE_M", "MIXED", "Sep 2025", 100, basis="standalone")]  # bases never mixed
    + [f3("INE_J", "JUNONLY", "Jun 2026", 300), f3("INE_J", "JUNONLY", "Mar 2026", 200),
       f3("INE_J", "JUNONLY", "Jun 2025", 100)]          # older quarter only
    + trio("INE_Y", "ANNUAL", 300, 200, 100, ptype="annual")      # not quarterly
    + trio("INE_P", "OPPROFIT", 300, 200, 100, item="Operating Profit"))
out9, label9 = BG.pat_growth_rows(res9, fin3=fin9, names={"INE_A": "Add Me Ltd"})
chk("top-up adds only missing, passing names", sorted(out9["slug"]),
    ["ADDME", "CONS", "INRES", "ONLYSTD"])
chk("source marked", dict(zip(out9["slug"], out9["source"])),
    {"INRES": "results", "ADDME": "statements", "CONS": "statements",
     "ONLYSTD": "statements"})
add = out9[out9["slug"] == "ADDME"].iloc[0]
chk("name from the universe map", add["company_name"], "Add Me Ltd")
chk("name falls back to the symbol", out9[out9["slug"] == "CONS"].iloc[0]["company_name"],
    "CONS")
chk("consolidated wins, figures not mixed",
    round(out9[out9["slug"] == "CONS"].iloc[0]["qoq_pct"], 1), 50.0)
chk("statements date shown in IST", add["reported"], "2026-11-04")
chk("one quarter label", set(out9["quarter"]), {"Q2 FY27"})
chk("results row unchanged by fin3", out9[out9["slug"] == "INRES"].iloc[0]["pat"], 300)
chk("no fin3 = results only", slugs([row("INRES", 300, 200, 100, isin="INE_R")]), ["INRES"])
chk("fin3 without its columns is ignored",
    BG.pat_growth_rows(res9, fin3=pd.DataFrame({"isin": ["X"]}))[0]["slug"].tolist(),
    ["INRES"])
out_q1, lab_q1 = BG.pat_growth_rows(
    pd.DataFrame([row("OLDQ", 300, 200, 100, q="Jun 2026", isin="INE_Q"),
                  row("NEWQ", 300, 200, 100, isin="INE_N")]),
    quarter="Q1FY27",
    fin3=pd.DataFrame([f3("INE_X", "XJUN", "Jun 2026", 300),
                       f3("INE_X", "XJUN", "Mar 2026", 200),
                       f3("INE_X", "XJUN", "Jun 2025", 100)] + trio("INE_W", "WSEP", 300, 200, 100)))
chk("explicit older quarter tops up the same quarter",
    (sorted(out_q1["slug"]), lab_q1), (["OLDQ", "XJUN"], "Q1 FY27"))
r9 = out9.copy()
r9["symbol"], r9["in_pf"], r9["both"] = r9["slug"], None, ""
t9, a9 = BG._pg_table(r9), BG._pg_annot(r9)
chk("statements row starred in the table", t9.count("usually ~2 days after the result"), 3)
chk("statements card says so", "statements read" in a9["ADDME"], True)
chk("results card keeps its wording", "📅 reported" in a9["INRES"], True)

# 10. quarter display
chk("Q2", BG._q_display("Q2FY27"), "Q2 FY27 (Jul–Sep 2026)")
chk("Q1 with a space", BG._q_display("Q1 FY27"), "Q1 FY27 (Apr–Jun 2026)")
chk("Q3", BG._q_display("Q3FY27"), "Q3 FY27 (Oct–Dec 2026)")
chk("Q4 is the next calendar year", BG._q_display("Q4FY27"), "Q4 FY27 (Jan–Mar 2027)")
chk("short form", BG._q_display("Q2FY27", months=False), "Q2 FY27")
chk("other labels pass through", BG._q_display("HY-Sep 2026"), "HY-Sep 2026")
banner = BG._quarter_banner("Q2FY27", "x & y")
chk("banner has quarter and months", "Q2 FY27 (Jul–Sep 2026)" in banner, True)
chk("banner escapes its text", "x &amp; y" in banner, True)

# 11. on both lists
wl11 = pd.DataFrame({"isin": ["INE_A", "INE_B", "INE_Z", ""],
                     "quarter": ["Q2FY27", "Q2FY27", "Q1FY27", "Q2FY27"],
                     "cagr_pct": [62.4, 55.0, 80.0, 70.0],
                     "score_metric": ["revenue", "pat", "revenue", "revenue"]})
wq, gq = BG._wl_pick_quarter(wl11)
chk("guidance: newest quarter only", (gq, sorted(wq["isin"])), ("Q2FY27", ["", "INE_A", "INE_B"]))
chk("guidance: explicit quarter", BG._wl_pick_quarter(wl11, "Q1 FY27")[0]["isin"].tolist(),
    ["INE_Z"])
chk("guidance: empty table", BG._wl_pick_quarter(pd.DataFrame())[1], "")
gmap = BG._both_from_guidance(wq, gq)
chk("guidance badge keys (blank ISIN skipped)", sorted(gmap), ["INE_A", "INE_B"])
chk("guidance badge text", gmap["INE_A"], "guidance Q2 FY27: revenue ~62%/yr guided")
pmap = BG._both_from_pat(out9, label9)
chk("PAT badge keys", sorted(pmap), ["INE_A", "INE_C", "INE_O", "INE_R"])
chk("PAT badge text", pmap["INE_A"], "PAT Q2 FY27: +200% YoY, QoQ +50%")
r9["both"] = [gmap.get(i, "") for i in r9["isin"]]
t11, a11 = BG._pg_table(r9), BG._pg_annot(r9)
chk("PAT table: one badge", t11.count("⭐ both"), 1)
chk("PAT cards: badge only on the name on both",
    ["ON BOTH LISTS" in a11[k] for k in sorted(a11)], [k == "ADDME" for k in sorted(a11)])
wlr = pd.DataFrame({"isin": ["INE_A", "INE_B"], "nse_symbol": ["ADDME", "BBB"],
                    "quarter": ["Q2FY27", "Q2FY27"], "cagr_pct": [62.4, 55.0],
                    "score_metric": ["revenue", "pat"], "date_added": ["2026-11-01"] * 2})
wlr["both"] = [pmap.get(i, "") for i in wlr["isin"]]
tw, aw = BG._wl_table(wlr), BG._wl_annot(wlr)
chk("guidance table: one badge", tw.count("⭐ both"), 1)
chk("guidance cards: badge only on the name on both",
    ("ON BOTH LISTS" in aw["ADDME"], "ON BOTH LISTS" in aw["BBB"]), (True, False))
chk("guidance card shows the quarter", "Q2 FY27" in aw["BBB"], True)
chk("guidance table without the column still renders",
    "⭐" in BG._wl_table(wlr.drop(columns="both")), False)


for f in FAILS:
    print("FAIL " + f)
print(f"{'FAILED' if FAILS else 'OK'}: {len(FAILS)} failure(s)")
sys.exit(len(FAILS))
