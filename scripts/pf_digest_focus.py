r"""
pf_digest_focus.py — "why focus" for the PF daily digest (user 2026-10-10/13).

    "PF daily digest does not have any actionable insight — it just quotes some files
     or numbers. I need a view on why to focus."

The old digest printed, for every holding with any activity, every flag it had —
including the fraud band and scorecard score every single day, changed or not. This
module turns the same inputs into a decision aid:

  1. TRIGGERS — only things that CHANGED in the window, each with its evidence line.
       strong (3): high-materiality filing · results/concall processed ·
                   unusual price move vs Nifty · fraud band worse · rating document
       medium (2): medium filing · presentation/AR processed · new catalyst ·
                   results due in <=3 days · guidance pace slipping · AR focus flag
       minor  (1): other filing · research note · community post
  2. ATTENTION — the sum of trigger weights. >= FOCUS_MIN gets a focus card; a lower
     non-zero score gets one line; zero is "quiet".
  3. THE VIEW — three lines per focus holding: what changed, why it matters, what to
     check. Written by ONE batched LLM call from the trigger facts only, then
     GROUNDED: any number in the model's text that is not in that holding's facts
     rejects the model's view for a deterministic one. With no model (dry run, no
     keys, call failure) every holding gets the deterministic view, so the mail
     never depends on the model.

"Unusual price move" = the stock's return relative to Nifty over 1 or 5 days is at
least PRICE_Z times its own typical swing over the previous ~6 months. Relative to
its own volatility, so a quiet large-cap and a volatile small-cap are judged alike.

Signals only, human-in-the-loop: the view says what to look at, never buy/sell.

    python scripts/pf_digest_focus.py --self-test
"""
from __future__ import annotations

import html as _html
import json
import re
import sys

import pandas as pd

STRONG, MEDIUM, MINOR = 3, 2, 1
FOCUS_MIN = 2                 # attention score that earns a focus card
PRICE_Z = 2.0                 # "unusual" = >= 2x the stock's own typical relative swing
PRICE_LOOKBACK = 120          # trading days used for the typical swing
FIELD_MAX = 220               # chars per view line

# Deterministic "why it matters / what to check" per trigger kind — the fallback
# view, and the rubric the model is shown.
WHY = {
    "price": "An unusual move against the market usually has a reason; it matters "
             "whether that reason is in a filing or only in the price.",
    "filing_high": "The company itself rated this filing as material.",
    "results": "New quarterly numbers change the earnings picture the holding was bought on.",
    "concall": "Management has just updated its own outlook.",
    "pead": "The reported numbers landed against management's own guidance.",
    "fraud": "A forensic red-flag score worsened.",
    "rating": "A credit agency has re-assessed the company.",
    "filing_med": "A filing the company marked as moderately important.",
    "presentation": "A new investor deck can move targets and KPIs.",
    "annual_report": "The annual report can change the multi-year story.",
    "catalyst": "There is a fresh, specific reason being cited for the stock.",
    "results_due": "Results are due — know what management promised before the print.",
    "guidance": "Management's own promise is slipping behind pace.",
    "ar_focus": "The annual-report read-through flagged a change in focus.",
    "filing_low": "A routine filing.",
    "research": "New broker or research coverage arrived.",
    "community": "Experienced investors are discussing it.",
}
CHECK = {
    "price": "Check today's filings and news for the cause; compare with the last results.",
    "filing_high": "Read the filing summary and whether it changes the thesis.",
    "results": "Compare revenue/PAT growth with the guidance and the last 4 quarters.",
    "concall": "Read the forward guidance lines and any change from last quarter.",
    "pead": "See which metric missed or beat and by how much.",
    "fraud": "Open the fraud tracker reason and the underlying flag.",
    "rating": "Check the rating action and the agency's stated concerns.",
    "filing_med": "Skim the filing summary.",
    "presentation": "Check targets and KPIs against the previous deck.",
    "annual_report": "Check the red flags and guidance sections of the AR summary.",
    "catalyst": "Verify the catalyst against an exchange filing.",
    "results_due": "Note the open guidance before the results date.",
    "guidance": "Check how far behind the promise is and what management said about it.",
    "ar_focus": "Read the AR focus/defocus reasons.",
    "filing_low": "No action needed unless it recurs.",
    "research": "Skim the research summary for a new number or view.",
    "community": "Skim the post; verify any claim against filings.",
}


def trig(kind: str, weight: int, fact: str, link: str = "") -> dict:
    return {"kind": kind, "weight": weight, "fact": " ".join(str(fact).split()),
            "link": link}


def attention(triggers: list[dict]) -> int:
    return sum(t["weight"] for t in triggers)


# --------------------------------------------------------------------------- #
#  price trigger                                                              #
# --------------------------------------------------------------------------- #

def _closes(df) -> pd.Series:
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", "close"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna().drop_duplicates("date").sort_values("date")
    return d.set_index("date")["close"]


def price_trigger(stock_ohlcv, nifty_ohlcv, today=None, max_age_days: int = 4):
    """The unusual-move trigger, or None. Relative return = stock minus Nifty."""
    s, n = _closes(stock_ohlcv), _closes(nifty_ohlcv)
    if len(s) < PRICE_LOOKBACK // 2 or len(n) < PRICE_LOOKBACK // 2:
        return None
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    if (today - s.index[-1]).days > max_age_days:
        return None                                  # stale bar: no claim about "today"
    rel = (s.pct_change() - n.pct_change().reindex(s.index)).dropna() * 100
    if len(rel) < 30:
        return None
    hist1 = rel.iloc[:-1].tail(PRICE_LOOKBACK)
    roll5 = rel.rolling(5).sum()
    hist5 = roll5.iloc[:-1].dropna().tail(PRICE_LOOKBACK)
    r1, r5 = float(rel.iloc[-1]), float(roll5.iloc[-1])
    sd1, sd5 = float(hist1.std()), float(hist5.std())
    z1 = abs(r1) / sd1 if sd1 > 0 else 0.0
    z5 = abs(r5) / sd5 if sd5 > 0 else 0.0
    if max(z1, z5) < PRICE_Z:
        return None
    if z5 >= z1:
        fact = (f"Stock {r5:+.1f}% vs Nifty over 5 days, {z5:.1f}x its usual 5-day "
                f"swing (1-day {r1:+.1f}%)")
    else:
        fact = (f"Stock {r1:+.1f}% vs Nifty today, {z1:.1f}x its usual daily swing "
                f"(5-day {r5:+.1f}%)")
    return trig("price", STRONG, fact)


# --------------------------------------------------------------------------- #
#  the view                                                                   #
# --------------------------------------------------------------------------- #

def fallback_view(triggers: list[dict]) -> dict:
    top = sorted(triggers, key=lambda t: -t["weight"])
    lead = top[0]["kind"] if top else ""
    return {"what_changed": "; ".join(t["fact"] for t in top[:3])[:FIELD_MAX],
            "why_it_matters": WHY.get(lead, "")[:FIELD_MAX],
            "what_to_check": CHECK.get(lead, "")[:FIELD_MAX],
            "source": "rules"}


PROMPT = """You help a portfolio manager decide which holdings need a look TODAY.
For each holding below you get ONLY the facts that changed in the last day. Write
three short lines per holding:
  what_changed   - the most important change, in plain words
  why_it_matters - why it could affect the investment case
  what_to_check  - one concrete thing to verify next
RULES: use ONLY the facts given; do not add numbers that are not in the facts; no
buy/sell/hold advice; each line under 200 characters; if the facts are routine, say so.
Reply with one JSON object per holding, nothing else:
{"symbol": "...", "what_changed": "...", "why_it_matters": "...", "what_to_check": "..."}

HOLDINGS:
"""


def build_prompt(focus: dict[str, list[dict]]) -> str:
    blocks = []
    for sym, ts in focus.items():
        facts = "\n".join(f"  - [{t['kind']}] {t['fact'][:400]}" for t in
                          sorted(ts, key=lambda t: -t["weight"])[:8])
        blocks.append(f"{sym}:\n{facts}")
    return PROMPT + "\n\n".join(blocks)


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _nums(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(str(text or "")):
        v = m.replace(",", "").rstrip(".")
        if v:
            out.add(v.rstrip("0").rstrip(".") if "." in v else v)
    return out


def grounded(view: dict, triggers: list[dict]) -> bool:
    """Every number the model wrote must appear in this holding's facts."""
    facts = _nums(" ".join(t["fact"] for t in triggers))
    said = _nums(" ".join(str(view.get(k, "")) for k in
                          ("what_changed", "why_it_matters", "what_to_check")))
    return said <= facts


def parse_views(raw: str, focus: dict[str, list[dict]], salvage=None) -> dict[str, dict]:
    """{symbol: view} for every focus holding: the model's when present, complete and
    grounded, else the deterministic one."""
    objs = []
    if raw:
        if salvage is not None:
            objs = salvage(raw)
        else:
            for m in re.findall(r"\{[^{}]*\}", raw):
                try:
                    objs.append(json.loads(m))
                except ValueError:
                    pass
    got = {}
    for o in objs:
        sym = str(o.get("symbol", "")).strip().upper()
        if sym in focus and sym not in got:
            v = {k: " ".join(str(o.get(k, "")).split())[:FIELD_MAX]
                 for k in ("what_changed", "why_it_matters", "what_to_check")}
            if all(v.values()) and grounded(v, focus[sym]):
                v["source"] = "model"
                got[sym] = v
    return {sym: got.get(sym) or fallback_view(ts) for sym, ts in focus.items()}


# --------------------------------------------------------------------------- #
#  html                                                                       #
# --------------------------------------------------------------------------- #

def esc(s, n=400) -> str:
    return _html.escape(str(s if s is not None else "")[:n])


KIND_LABEL = {
    "price": "price move", "filing_high": "material filing", "results": "results",
    "concall": "concall", "pead": "vs guidance", "fraud": "fraud flag",
    "rating": "rating", "filing_med": "filing", "presentation": "deck",
    "annual_report": "annual report", "catalyst": "catalyst",
    "results_due": "results due", "guidance": "guidance pace", "ar_focus": "AR focus",
    "filing_low": "filing", "research": "research", "community": "community",
}


def _chip(t: dict) -> str:
    bg = {STRONG: "#fde2e1", MEDIUM: "#fff3cd", MINOR: "#eef1f4"}[t["weight"]]
    return (f"<span style='background:{bg};padding:1px 6px;border-radius:3px;"
            f"font-size:11px;margin-right:4px'>{esc(KIND_LABEL.get(t['kind'], t['kind']))}"
            f"</span>")


def focus_card(sym: str, name: str, in_pf_weight: str, triggers: list[dict],
               view: dict, chart_html: str = "") -> str:
    score = attention(triggers)
    chips = "".join(_chip(t) for t in sorted(triggers, key=lambda t: -t["weight"])[:6])
    ev = "".join(
        f"<li>{esc(t['fact'], 300)}"
        + (f" (<a href='{esc(t['link'], 300)}'>source</a>)" if t.get("link") else "")
        + "</li>" for t in sorted(triggers, key=lambda t: -t["weight"])[:8])
    src = ("" if view.get("source") == "model" else
           " <span style='color:#999;font-size:10px'>(rule-based)</span>")
    return (
        f"<div style='border:1px solid #dde3e8;border-radius:6px;padding:10px 12px;"
        f"margin:0 0 14px;font:13px Arial,sans-serif'>"
        f"<div style='font-size:15px'><b>{esc(sym, 20)}</b> · {esc(name, 50)}"
        f"{(' · ' + esc(in_pf_weight, 12)) if in_pf_weight else ''}"
        f"<span style='float:right;font-size:12px;color:#555'>attention {score}</span></div>"
        f"<div style='margin:4px 0 6px'>{chips}</div>"
        f"<div><b>What changed:</b> {esc(view.get('what_changed'))}</div>"
        f"<div><b>Why it matters:</b> {esc(view.get('why_it_matters'))}</div>"
        f"<div><b>What to check:</b> {esc(view.get('what_to_check'))}{src}</div>"
        f"{chart_html}"
        f"<div style='font-size:11px;color:#666;margin-top:6px'>Evidence:"
        f"<ul style='margin:2px 0 0 16px;padding:0'>{ev}</ul></div></div>")


def minor_line(sym: str, triggers: list[dict]) -> str:
    return (f"<li><b>{esc(sym, 20)}</b> — "
            + esc("; ".join(t["fact"] for t in triggers)[:240]) + "</li>")


def _self_test() -> int:
    import numpy as np
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    # price trigger: a calm stock that suddenly drops 8% while Nifty is flat
    n = 200
    dates = pd.bdate_range("2026-01-01", periods=n)
    rng = np.random.default_rng(3)
    nifty = pd.DataFrame({"date": dates, "close": 100 * np.cumprod(1 + rng.normal(0, .005, n))})
    calm = 100 * np.cumprod(1 + rng.normal(0, .006, n))
    jolt = calm.copy()
    jolt[-1] = jolt[-2] * 0.92
    today = dates[-1]
    t = price_trigger(pd.DataFrame({"date": dates, "close": jolt}), nifty, today)
    check("sudden -8% on a calm stock is unusual",
          t is not None and t["kind"] == "price" and "vs Nifty" in t["fact"]
          and t["weight"] == STRONG)
    check("ordinary day is not", price_trigger(pd.DataFrame({"date": dates, "close": calm}),
                                               nifty, today) is None)
    check("stale bar makes no claim", price_trigger(
        pd.DataFrame({"date": dates, "close": jolt}), nifty,
        today + pd.Timedelta(days=10)) is None)
    check("too little history -> none", price_trigger(
        pd.DataFrame({"date": dates[:20], "close": jolt[:20]}), nifty, today) is None)

    ts = [trig("results", STRONG, "Q2 FY27 results processed: Rev Rs 1,604 cr (+67% YoY)"),
          trig("research", MINOR, "Broker note arrived")]
    check("attention sums weights", attention(ts) == 4)
    fb = fallback_view(ts)
    check("fallback leads with the strongest trigger",
          fb["what_changed"].startswith("Q2 FY27 results") and fb["why_it_matters"] == WHY["results"])
    good = {"what_changed": "Q2 revenue rose 67% to Rs 1,604 cr",
            "why_it_matters": "Growth is well above trend", "what_to_check": "Margins vs guidance"}
    bad = dict(good, what_changed="Q2 revenue rose 72% to Rs 1,700 cr")
    check("grounded view accepted", grounded(good, ts))
    check("invented numbers rejected", not grounded(bad, ts))
    focus = {"SYRMA": ts, "ABC": [trig("filing_high", STRONG, "Order win of Rs 500 cr")]}
    raw = ('noise {"symbol": "SYRMA", "what_changed": "Q2 revenue rose 67% to Rs 1,604 cr", '
           '"why_it_matters": "Growth well above trend", "what_to_check": "Margins"} '
           '{"symbol": "ABC", "what_changed": "Won a Rs 900 cr order", '
           '"why_it_matters": "x", "what_to_check": "y"} {"symbol": "ZZZ", "what_changed": "a",'
           ' "why_it_matters": "b", "what_to_check": "c"}')
    v = parse_views(raw, focus)
    check("model view used when grounded", v["SYRMA"]["source"] == "model")
    check("ungrounded model view replaced by rules", v["ABC"]["source"] == "rules")
    check("unknown symbol ignored", set(v) == {"SYRMA", "ABC"})
    check("no model output -> all rules",
          all(x["source"] == "rules" for x in parse_views("", focus).values()))
    p = build_prompt(focus)
    check("prompt carries facts, not advice", "Order win of Rs 500 cr" in p
          and "buy/sell/hold advice" in p)
    card = focus_card("SYRMA", "Syrma SGS", "", ts, v["SYRMA"])
    check("card shows the three lines and evidence",
          all(s in card for s in ("What changed", "Why it matters", "What to check",
                                  "Evidence", "attention 4")))
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
