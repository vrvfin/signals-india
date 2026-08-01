r"""
render_narrative_deck.py — Layer D of the narrative report. NO LLM.

Consumes the fact pack (Layer A), optionally the narrative (Layer B) and the audit
(Layer C), and emits BOTH artefacts:
  • company_narrative_<SYM>_DDMMMYY.md    — canonical, uploads to Drive
  • company_narrative_<SYM>_DDMMMYY.html  — slide-per-section deck, opens in a browser

The source footer under every section is GENERATED FROM THE PACK, not written by a model.
That is what makes invariant #1 ("every section cites its documents") structurally true
rather than an instruction a model may forget.

Charts are drawn client-side with TradingView lightweight-charts from a CDN, matching the
existing local-deck pattern in build_gallery.py / build_market_state_html.py. No new deps.

Usage:
  python scripts/render_narrative_deck.py --factpack fp.json --outdir . --open
  python scripts/render_narrative_deck.py --factpack fp.json --narrative nar.json \
                                          --audit audit.json --outdir .
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

# Section titles follow the 26-section canonical spec. Sections with no content in this
# pack render as an explicit gap card rather than being silently dropped.
SECTIONS: list[tuple[int, str]] = [
    (1, "Twenty-eight years, one decision at a time"),
    (2, "The company on one page"),
    (3, "One listed parent, its subsidiaries"),
    (4, "A founder-led business with a professional operating layer"),
    (5, "How the business was built — and what it cost"),
    (6, "Segment economics — revenue share vs profit share"),
    (7, "The accounting basis, and what the rating agencies say"),
    (8, "The quiet compounder — the steadiest series"),
    (9, "The portfolio, as disclosed"),
    (10, "A more useful way to read the portfolio"),
    (11, "What moved this year"),
    (12, "Unit deep dives"),
    (14, "Independent research — coverage of this company"),
    (15, "Independent research — the sector view"),
    (16, "Testing a management claim against independent data"),
    (17, "Independent data scorecard — and its limits"),
    (18, "The full financial record"),
    (19, "Operating leverage, stated as arithmetic"),
    (20, "Management's claim, tested across four calls"),
    (21, "The peer set"),
    (22, "Implied multiples across revenue and margin"),
    (23, "Structural and counterparty risks"),
    (24, "Financial, policy and execution risks"),
    (25, "The policy backdrop"),
    (26, "Findings"),
    (27, "Recent exchange filings"),
    (28, "Appendix — unit conversions (*)"),
]

DISCLAIMER = (
    "Prepared for educational and analytical purposes only. Not investment advice, not a "
    "research report within the meaning of any securities regulation, and not a "
    "recommendation to buy or sell any security. No target price, no rating, no "
    "fair-value estimate and no forecast of future financial performance. Sensitivity "
    "tables restate disclosed figures under stated assumptions and are not projections. "
    "Readers should consult the company's own filings and, where relevant, a licensed "
    "financial adviser."
)


# ------------------------------------------------------------------ helpers ---
def _fmt(v, unit: str = "") -> str:
    if v is None:
        return "n/d"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        s = f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
    else:
        s = f"{v:,}"
    u = (unit or "").strip()
    if u in ("%", "x", "bps", "pp"):
        return f"{s}{u}"
    if u in ("count", "label", "/100"):
        return s if u in ("count", "label") else f"{s}/100"
    if u.lower().startswith("rs"):
        return f"Rs {s} Cr" if "cr" in u.lower() else f"Rs {s}"
    return f"{s} {u}".strip()


def _src_str(src: dict) -> str:
    """One human-readable citation from a fact-pack source record."""
    if not src:
        return "unattributed"
    k = src.get("kind")
    if k == "statements":
        li = ", ".join(src.get("line_items", []) or [])
        p = src.get("period", "")
        f = src.get("fetched_at")
        bits = [src.get("table", "")]
        if li:
            bits.append(f"line items: {li}")
        if p and p != "all":
            bits.append(f"period {p}")
        if f:
            bits.append(f"retrieved {str(f)[:10]}")
        return " · ".join(b for b in bits if b)
    if k == "parquet":
        return " · ".join(b for b in (src.get("table", ""), src.get("note", "")) if b)
    if k == "computed":
        return f"computed: {src.get('note', '')}"
    if k == "document":
        return " · ".join(str(src.get(x, "")) for x in ("doc_type", "date", "title") if
                          src.get(x))
    return json.dumps(src, ensure_ascii=False)[:180]


def _sources_for(section: int, pack: dict) -> list[str]:
    """Distinct citations backing everything rendered in this section."""
    seen, out = set(), []
    for f in pack.get("facts", []):
        if f.get("section") == section:
            s = _src_str(f.get("source"))
            if s not in seen:
                seen.add(s)
                out.append(s)
    for t in pack.get("tables", []):
        if t.get("section") == section:
            s = _src_str(t.get("source"))
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


# Facts whose id matches these prefixes are per-period series that a table in the same
# section already presents as a matrix. Listing them again produces dozens of rows like
# "Revenue FY18 / Revenue FY19 / ..." directly above the table holding the same numbers.
_SERIES_PREFIXES = ("fin.",)


def _facts_in(section: int, pack: dict, for_display: bool = True) -> list[dict]:
    """Facts belonging to a section. When `for_display`, per-period series facts are
    withheld IF a table in the same section already renders them — the table is the
    readable form, and showing both is duplication, not thoroughness. Provenance is
    unaffected: `_sources_for` still reads the full set."""
    facts = [f for f in pack.get("facts", []) if f.get("section") == section]
    if not for_display:
        return facts
    has_table = any(t.get("section") == section for t in pack.get("tables", []))
    if not has_table:
        return facts
    kept = [f for f in facts
            if not str(f.get("id", "")).startswith(_SERIES_PREFIXES)
            or str(f.get("id", "")).endswith(".cagr")]
    return kept


def _tables_in(section: int, pack: dict) -> list[dict]:
    return [t for t in pack.get("tables", []) if t.get("section") == section]


def _gaps_in(section: int, pack: dict) -> list[str]:
    out = [g["reason"] for g in pack.get("coverage_gaps", [])
           if g.get("section") == section]
    out += [u["reason"] for u in pack.get("uncovered_sections", [])
            if u.get("section") == section]
    return out


def _narr(section: int, narrative: dict | None) -> dict:
    if not narrative:
        return {}
    for s in narrative.get("sections", []):
        if str(s.get("id")) == str(section):
            return s
    return {}


def _audit_for(section: int, audit: dict | None) -> list[dict]:
    if not audit:
        return []
    return [c for c in audit.get("claims", [])
            if str(c.get("section")) == str(section)]


def _verdict_tag(v: str) -> str:
    return {"VERIFIED": "", "PARTIAL": "[PARTIAL]", "UNSUPPORTED": "[UNSUPPORTED]",
            "CONTRADICTED": "[CONTRADICTED]", "NOT_IN_SOURCE": "[NOT_IN_SOURCE]"
            }.get(str(v).upper(), f"[{v}]")


# -------------------------------------------------------------------- markdown -
def render_markdown(pack: dict, narrative: dict | None = None,
                    audit: dict | None = None) -> str:
    co = pack["company"]
    L: list[str] = []
    L.append(f"# {co['name']} — narrative report")
    L.append("")
    L.append(f"**{co['symbol']} · {co['isin']}** · data current to "
             f"{pack.get('as_of_utc', '')[:10]} · this report will become stale.")
    L.append("")

    if audit and not audit.get("ran", True):
        # An audit that errored on every section must never render as a clean one.
        L.append(f"> ### AUDIT DID NOT RUN")
        L.append(f">")
        L.append(f"> Every section failed adjudication, so **no claim in this report has "
                 f"been independently checked**. Treat it as unaudited.")
        L.append(f">")
        L.append(f"> Adjudicator `{audit.get('model', 'unknown')}` — "
                 f"{audit.get('failure_reason', 'reason not recorded')}")
        L.append("")
    elif audit:
        s = audit.get("summary", {})
        L.append(f"> **Audit:** {s.get('verified', 0)} of {s.get('total', 0)} claims "
                 f"verified against source · {s.get('unsupported', 0)} unsupported · "
                 f"{s.get('contradicted', 0)} contradicted · adjudicated by "
                 f"`{audit.get('model', 'unknown')}`.")
        if s.get("audit_failed"):
            L.append(f">")
            L.append(f"> **{s['sections_failed']} section(s) could not be audited** "
                     f"({s['sections_audited']} were). Claims in those sections are "
                     f"unchecked.")
        L.append("")
    else:
        L.append("> **Audit:** not run. Numbers are fact-pack grounded, but no "
                 "independent re-validation has been performed on this copy.")
        L.append("")

    L.append("## Part A — Narrative report")
    L.append("")
    for num, title in SECTIONS:
        facts, tables = _facts_in(num, pack), _tables_in(num, pack)
        gaps, nar = _gaps_in(num, pack), _narr(num, narrative)
        if not (facts or tables or gaps or nar):
            continue
        L.append(f"### {num}. {title}")
        L.append("")
        if nar.get("takeaway"):
            L.append(f"**{nar['takeaway']}**")
            L.append("")
        if nar.get("body"):
            L.append(nar["body"])
            L.append("")

        if facts:
            L.append("| Metric | Value | Basis |")
            L.append("| --- | --- | --- |")
            for f in facts:
                L.append(f"| {f['label']} | {_fmt(f['value'], f.get('unit', ''))} "
                         f"| {f.get('basis', '')} |")
            L.append("")

        for t in tables:
            L.append(f"**{t['title']}**")
            L.append("")
            cols = t["columns"]
            L.append("| " + " | ".join(str(c) for c in cols) + " |")
            L.append("| " + " | ".join("---" for _ in cols) + " |")
            for row in t["rows"]:
                L.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
            L.append("")
            if t.get("note"):
                L.append(f"*{t['note']}*")
                L.append("")

        for c in _audit_for(num, audit):
            tag = _verdict_tag(c.get("verdict"))
            if tag:
                L.append(f"- {tag} {c.get('claim', '')[:200]} — "
                         f"{c.get('reason', '')[:200]}")
        if _audit_for(num, audit):
            L.append("")

        if gaps:
            for g in gaps:
                L.append(f"- **DATA_MISSING** — {g}")
            L.append("")

        srcs = _sources_for(num, pack)
        if srcs:
            L.append("*Sources: " + "; ".join(srcs) + "*")
            L.append("")

    if narrative and narrative.get("forensic_report"):
        L.append("## Part B — Forensic summary")
        L.append("")
        L.append(narrative["forensic_report"])
        L.append("")

    L.append("## Part C — Audit")
    L.append("")
    if not audit:
        L.append("Not run for this copy.")
    else:
        s = audit.get("summary", {})
        L.append(f"Adjudicator: `{audit.get('model', 'unknown')}`"
                 + (" — DEGRADED: same model family as the generator, so this audit is "
                    "weaker evidence of independence."
                    if audit.get("degraded_fallback") else ""))
        L.append("")
        L.append(f"| Verdict | Claims |")
        L.append("| --- | --- |")
        for k in ("verified", "partial", "unsupported", "contradicted",
                  "not_in_source"):
            L.append(f"| {k.replace('_', ' ').title()} | {s.get(k, 0)} |")
        L.append("")
        bad = [c for c in audit.get("claims", [])
               if str(c.get("verdict", "")).upper() not in ("VERIFIED", "PARTIAL")]
        if bad:
            L.append("### Claims not supported by source")
            L.append("")
            for c in bad:
                L.append(f"- **{_verdict_tag(c.get('verdict'))}** (section "
                         f"{c.get('section')}) “{c.get('claim', '')[:300]}”")
                L.append(f"  - {c.get('reason', '')[:400]}")
            L.append("")
        else:
            L.append("Every claim was supported by the sources checked.")
            L.append("")

    gaps_all = pack.get("coverage_gaps", []) + pack.get("uncovered_sections", [])
    if gaps_all:
        L.append("### Coverage gaps")
        L.append("")
        for g in gaps_all:
            L.append(f"- section {g.get('section')}: {g.get('reason')}")
        L.append("")

    L.append("---")
    L.append("")
    L.append(f"*{DISCLAIMER}*")
    return "\n".join(L)


# ------------------------------------------------------------------- html -----
_TPL = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
 :root{--bg:#fff;--fg:#14161a;--mut:#5b6270;--line:#e4e7ec;--card:#fafbfc;--accent:#2c4bd8}
 @media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e8eaed;--mut:#9aa3b2;--line:#252a33;--card:#161920;--accent:#7d93ff}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 .wrap{max-width:1180px;margin:0 auto;padding:24px 18px 80px}
 header{border-bottom:2px solid var(--fg);padding-bottom:14px;margin-bottom:8px}
 h1{font-size:30px;margin:0 0 6px}
 .sub{color:var(--mut);font-size:13px}
 .audit{margin:14px 0;padding:10px 14px;border-left:3px solid var(--accent);background:var(--card);font-size:13px}
 section{border:1px solid var(--line);border-radius:10px;background:var(--card);padding:18px;margin:18px 0}
 h2{font-size:19px;margin:0 0 4px}
 h2 .n{color:var(--mut);font-weight:400;margin-right:8px}
 .take{font-weight:600;margin:8px 0}
 .tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;margin:12px 0}
 .tile{border:1px solid var(--line);border-radius:8px;padding:10px;background:var(--bg)}
 .tile .l{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.4px}
 .tile .v{font-size:20px;font-weight:600;margin-top:2px}
 .tile .b{font-size:10px;color:var(--mut);margin-top:2px}
 .scroll{overflow-x:auto;margin:12px 0}
 table{border-collapse:collapse;width:100%;font-size:13px;min-width:520px}
 th,td{border-bottom:1px solid var(--line);padding:6px 9px;text-align:right;white-space:nowrap}
 th:first-child,td:first-child{text-align:left}
 th{background:var(--bg);font-size:11px;text-transform:uppercase;letter-spacing:.4px;color:var(--mut)}
 .note{font-size:12px;color:var(--mut);font-style:italic;margin:6px 0}
 .gap{font-size:12px;color:#b4531a;background:rgba(180,83,26,.08);border-radius:6px;padding:6px 9px;margin:5px 0}
 @media(prefers-color-scheme:dark){.gap{color:#f0a06a}}
 .src{font-size:11px;color:var(--mut);border-top:1px dashed var(--line);margin-top:14px;padding-top:8px;word-break:break-word}
 .chart{height:280px;margin:12px 0}
 .flag{font-size:12px;padding:4px 8px;border-radius:5px;margin:4px 0;background:rgba(200,60,60,.1);color:#c33}
 @media(prefers-color-scheme:dark){.flag{color:#ff8b8b}}
 footer{margin-top:32px;padding-top:14px;border-top:1px solid var(--line);font-size:11px;color:var(--mut)}
</style>
<div class="wrap">__BODY__</div>
<script>
const PAYLOAD = __PAYLOAD__;
document.querySelectorAll('.chart').forEach(el=>{
  const spec = PAYLOAD[el.id]; if(!spec) return;
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const ch = LightweightCharts.createChart(el, {
    height: el.clientHeight, autoSize: true,
    layout:{background:{color:'transparent'},textColor:dark?'#9aa3b2':'#5b6270',fontSize:11},
    grid:{vertLines:{visible:false},horzLines:{color:dark?'#252a33':'#eef0f3'}},
    rightPriceScale:{borderVisible:false},
    // The left scale is INVISIBLE by default in lightweight-charts v4 (default
    // priceScale options carry visible:false). Without this the margin line draws
    // against an unlabelled axis and cannot be read off the chart.
    leftPriceScale:{borderVisible:false, visible:true},
    timeScale:{borderVisible:false},
    handleScroll:false, handleScale:false,
  });
  (spec.series||[]).forEach(s=>{
    const opts = Object.assign({priceScaleId: s.axis||'right'}, s.options||{});
    const ser = s.type==='line' ? ch.addLineSeries(opts) : ch.addHistogramSeries(opts);
    ser.setData(s.data);
  });
  ch.timeScale().fitContent();
});
</script>"""


def _chart_payload(pack: dict) -> dict:
    """Series for the section-18 chart: revenue + EBITDA bars, margin line.

    lightweight-charts needs a time axis; FY labels are mapped to 1 Jan of the FY-end
    calendar year purely as an ordering device.
    """
    tbl = next((t for t in pack.get("tables", []) if t["id"] == "tbl.financials"), None)
    if not tbl:
        return {}
    rev, eb, mar = [], [], []
    for row in tbl["rows"]:
        fy = str(row.get("FY", ""))
        if not fy.startswith("FY"):
            continue
        t = f"20{fy[2:]}-01-01"
        if row.get("revenue") is not None:
            rev.append({"time": t, "value": float(row["revenue"])})
        if row.get("ebitda_incl_oi") is not None:
            eb.append({"time": t, "value": float(row["ebitda_incl_oi"])})
        if row.get("margin") is not None:
            mar.append({"time": t, "value": round(float(row["margin"]), 2)})
    if not rev:
        return {}
    # Revenue as bars, EBITDA as a LINE — not a second histogram. EBITDA is ~6% of
    # revenue for a distributor like this, so overlapping bars render it as a sliver
    # hidden behind the revenue bar. A line stays legible at any ratio.
    return {"ch_fin": {"series": [
        {"type": "hist", "data": rev,
         "options": {"color": "rgba(120,150,255,.45)",
                     "priceFormat": {"type": "volume"}}},
        {"type": "line", "data": eb,
         "options": {"color": "#28a06e", "lineWidth": 2,
                     "priceFormat": {"type": "volume"}}},
        {"type": "line", "data": mar, "axis": "left",
         "options": {"color": "#e8833a", "lineWidth": 2, "lineStyle": 2}},
    ]}}


def render_html(pack: dict, narrative: dict | None = None,
                audit: dict | None = None) -> str:
    co = pack["company"]
    e = html.escape
    B: list[str] = []
    B.append("<header>")
    B.append(f"<h1>{e(co['name'])}</h1>")
    B.append(f"<div class='sub'>{e(co['symbol'])} · {e(co['isin'])} · data current to "
             f"{e(pack.get('as_of_utc', '')[:10])} — this report will become stale</div>")
    B.append("</header>")

    if audit and not audit.get("ran", True):
        B.append("<div class='flag'><b>AUDIT DID NOT RUN.</b> Every section failed "
                 "adjudication, so no claim in this report has been independently "
                 f"checked — treat it as unaudited. Adjudicator "
                 f"<code>{e(str(audit.get('model', 'unknown')))}</code>: "
                 f"{e(str(audit.get('failure_reason', 'reason not recorded')))}</div>")
    elif audit:
        s = audit.get("summary", {})
        B.append(f"<div class='audit'><b>Audit:</b> {s.get('verified', 0)} of "
                 f"{s.get('total', 0)} claims verified against source · "
                 f"{s.get('unsupported', 0)} unsupported · "
                 f"{s.get('contradicted', 0)} contradicted · adjudicated by "
                 f"<code>{e(str(audit.get('model', 'unknown')))}</code>"
                 + (f" · <b>{s['sections_failed']} section(s) could NOT be audited</b>"
                    if s.get("audit_failed") else "") + "</div>")
    else:
        B.append("<div class='audit'><b>Audit:</b> not run — numbers are fact-pack "
                 "grounded, but no independent re-validation has been performed.</div>")

    for num, title in SECTIONS:
        facts, tables = _facts_in(num, pack), _tables_in(num, pack)
        gaps, nar = _gaps_in(num, pack), _narr(num, narrative)
        if not (facts or tables or gaps or nar):
            continue
        B.append("<section>")
        B.append(f"<h2><span class='n'>{num}</span>{e(title)}</h2>")
        if nar.get("takeaway"):
            B.append(f"<div class='take'>{e(nar['takeaway'])}</div>")
        if nar.get("body"):
            B.append("<p>" + e(nar["body"]).replace("\n", "<br>") + "</p>")

        if facts:
            B.append("<div class='tiles'>")
            for f in facts[:16]:
                B.append(f"<div class='tile'><div class='l'>{e(str(f['label']))}</div>"
                         f"<div class='v'>{e(_fmt(f['value'], f.get('unit', '')))}</div>"
                         f"<div class='b'>{e(str(f.get('basis', '')))}</div></div>")
            B.append("</div>")

        if num == 18 and _chart_payload(pack):
            B.append("<div class='chart' id='ch_fin'></div>")
            B.append("<div class='note'>Bars: revenue (Rs Cr, right axis). "
                     "Solid line: EBITDA incl. other income (Rs Cr, right axis). "
                     "Dashed line: EBITDA margin % (left axis).</div>")

        for t in tables:
            B.append(f"<div class='note'><b>{e(t['title'])}</b></div><div class='scroll'>"
                     "<table><thead><tr>")
            for c in t["columns"]:
                B.append(f"<th>{e(str(c))}</th>")
            B.append("</tr></thead><tbody>")
            for row in t["rows"]:
                B.append("<tr>" + "".join(f"<td>{e(_fmt(row.get(c)))}</td>"
                                          for c in t["columns"]) + "</tr>")
            B.append("</tbody></table></div>")
            if t.get("note"):
                B.append(f"<div class='note'>{e(t['note'])}</div>")

        for c in _audit_for(num, audit):
            tag = _verdict_tag(c.get("verdict"))
            if tag:
                B.append(f"<div class='flag'>{e(tag)} {e(str(c.get('claim', ''))[:220])}"
                         f" — {e(str(c.get('reason', ''))[:220])}</div>")

        for g in gaps:
            B.append(f"<div class='gap'><b>DATA_MISSING</b> — {e(g)}</div>")

        srcs = _sources_for(num, pack)
        if srcs:
            B.append("<div class='src'><b>Sources:</b> " +
                     e("; ".join(srcs)) + "</div>")
        B.append("</section>")

    if narrative and narrative.get("forensic_report"):
        B.append("<section><h2><span class='n'>B</span>Forensic summary</h2><p>"
                 + e(narrative["forensic_report"]).replace("\n", "<br>") + "</p></section>")

    B.append("<section><h2><span class='n'>C</span>Audit</h2>")
    if not audit:
        B.append("<p>Not run for this copy.</p>")
    else:
        if audit.get("degraded_fallback"):
            B.append("<div class='flag'>DEGRADED: the adjudicator ran on the same model "
                     "family as the generator, so this audit is weaker evidence of "
                     "independence.</div>")
        bad = [c for c in audit.get("claims", [])
               if str(c.get("verdict", "")).upper() not in ("VERIFIED", "PARTIAL")]
        if not bad:
            B.append("<p>Every claim was supported by the sources checked.</p>")
        for c in bad:
            B.append(f"<div class='flag'>{e(_verdict_tag(c.get('verdict')))} "
                     f"(section {e(str(c.get('section')))}) "
                     f"“{e(str(c.get('claim', ''))[:300])}” — "
                     f"{e(str(c.get('reason', ''))[:300])}</div>")
    B.append("</section>")

    gaps_all = pack.get("coverage_gaps", []) + pack.get("uncovered_sections", [])
    if gaps_all:
        B.append("<section><h2><span class='n'>D</span>Coverage gaps</h2>")
        for g in gaps_all:
            B.append(f"<div class='gap'>section {e(str(g.get('section')))}: "
                     f"{e(str(g.get('reason')))}</div>")
        B.append("</section>")

    B.append(f"<footer>{e(DISCLAIMER)}</footer>")

    return (_TPL.replace("__TITLE__", e(f"{co['name']} — narrative report"))
            .replace("__BODY__", "\n".join(B))
            .replace("__PAYLOAD__", json.dumps(_chart_payload(pack))))


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factpack", required=True)
    ap.add_argument("--narrative", help="Layer B output (optional)")
    ap.add_argument("--audit", help="Layer C output (optional)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--open", action="store_true", help="open the HTML when done")
    a = ap.parse_args()

    pack = json.loads(Path(a.factpack).read_text(encoding="utf-8"))
    nar = json.loads(Path(a.narrative).read_text(encoding="utf-8")) if a.narrative else None
    aud = json.loads(Path(a.audit).read_text(encoding="utf-8")) if a.audit else None

    sym = pack["company"]["symbol"]
    stamp = datetime.now().strftime("%d%b%y")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    md_p = out / f"company_narrative_{sym}_{stamp}.md"
    html_p = out / f"company_narrative_{sym}_{stamp}.html"

    md_p.write_text(render_markdown(pack, nar, aud), encoding="utf-8")
    html_p.write_text(render_html(pack, nar, aud), encoding="utf-8")
    print(f"wrote {md_p}")
    print(f"wrote {html_p}")
    if not aud:
        print("NOTE: rendered without an audit pass — Part C is empty.")
    if a.open:
        webbrowser.open(html_p.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
