"""
pf_company_mails.py — ONE MAIL PER COMPANY, per document type.

The existing digests send one mail covering every holding. The user asked for the
opposite: each company that reports gets its own mail, so the body can carry real detail
instead of a row in a table.

Three kinds, each with the same two-part shape — **the current document first, then what
changed since the last one**:

    results       the full quarterly teardown (quarter_teardown.render)
    presentation  this quarter's deck  +  KPIs dropped / targets moved / guidance vs delivered
    rating        current agency view  +  upgrade-downgrade, new & vanished concerns,
                                          downgrade triggers that moved

THE TRIGGER (user, 2026-08-15): a mail goes out when the holding is in PF **and** its
results-calendar date has arrived **and** it has not already been mailed for that document
and period. Deliberately NOT a rolling time window — a window means a missed CI run drops
that company for good, whereas "not yet mailed" is self-healing and the next run catches
up. See pf_coverage.mail_due().

WHAT IS DETERMINISTIC AND WHAT IS NOT. Every "what changed" line here is an exact
comparison of structured rows — a KPI present last quarter and absent now, a concern in
the new rationale that was not in the old one. No LLM is involved in deciding what
changed. That is deliberate: round-tripping structured data through prose and back is
where hallucination enters, and the framework's own rule is deterministic-where-possible.

READ-ONLY w.r.t. Phase 2. This never takes _extract.lock, never writes company_page.md,
never queues or extracts anything. It reads the parquets Phase 2 already produces and
writes exactly one thing of its own: its mail ledger.

Usage:
    python scripts/pf_company_mails.py --dry-run          # preview, no mail, no ledger
    python scripts/pf_company_mails.py --types rating     # one kind only
    python scripts/pf_company_mails.py --limit 5
    python scripts/pf_company_mails.py --self-test
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import date, datetime

import pandas as pd
from dotenv import load_dotenv

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import quarterly_table as QT
import pf_coverage as COV
import deck_summary as DS

LEDGER_NAME = "pf_company_mails.parquet"
# `doc_id` added 2026-08-16 (ADDITIVE — legacy rows read blank and still suppress, so
# adding it cannot cause a re-send flood). Identity is the DOCUMENT, not the slot: keying
# on (isin, doc_type, season) alone meant the first deck or rating of a quarter mailed and
# every later one was silently swallowed — a rating DOWNGRADE arriving after a routine
# reaffirmation would never have reached the reader.
LEDGER_COLS = ["season", "isin", "symbol", "doc_type", "period", "doc_id",
               "mailed_at", "subject"]
MAIL_KEY = "pf_company_mails"
MAX_HTML_BYTES = 90_000

UP, DOWN, MUTED, AMBER = "#1a7a3a", "#c0392b", "#8a97a0", "#b8860b"
# Theme colours, defined here with the rest of the palette because both the deck
# summary sections and the theme chips need them.
BLUE, PURPLE = "#1f4d6b", "#6b4d8f"
_WRAP = "font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222"
_TBL = ("border-collapse:collapse;width:100%;font:12px Arial,Helvetica,sans-serif;"
        "color:#222;margin:4px 0 12px")


def _esc(s, n=400) -> str:
    from mailer import esc
    # Parquet NaN/None must not reach the page. Rendering value+unit straight out of the
    # frame produced literal "nanNone" in the first live run.
    if s is None:
        return ""
    t = str(s).strip()
    if t.lower() in ("nan", "none", "nat", "<na>"):
        return ""
    return esc(t, n)


# ESG commitments are not investment guidance. deck_teardown already bans these words
# from deck metrics ("_BANNED_METRIC_WORDS") on exactly this reasoning, but ppt_guidance
# and ppt_highlights have no such filter, so the first live render of APLAPOLLO returned
# a "guidance" table consisting of Scope 1 & 2 emissions, Net Zero, renewable share and
# female workforce — no financial target at all. Filtered at RENDER time only; nothing
# upstream is changed and the rows stay in the parquet.
_ESG_WORDS = ("emission", "net zero", "carbon", "esg", "csr", "diversity", "female",
              "renewable", "sustainab", "djsi", "gender", "water", "waste")


def _is_esg(*fields) -> bool:
    blob = " ".join(str(f or "").lower() for f in fields)
    return any(w in blob for w in _ESG_WORDS)


_NUM_RE = __import__("re").compile(r"^\s*([+-]?)\s*([\d,]+(?:\.\d+)?)\s*(%|x|bps|Cr|cr)?\s*$")


def colour_num(text, good_up: bool = True, bold: bool = True) -> str:
    """Render a number with a colour that carries its MEANING.

    Green/red is not decoration — it is the fastest way to read a table of mixed
    signals. `good_up=False` inverts it for metrics where lower is better (debt, days,
    a downgrade count), so a falling number reads green there.
    Non-numeric text passes through untouched rather than being force-coloured.
    """
    s = _esc(text, 40)
    if not s:
        return ""
    m = _NUM_RE.match(str(text).strip())
    if not m:
        return s
    sign, digits, unit = m.group(1), m.group(2), m.group(3) or ""
    try:
        val = float(digits.replace(",", ""))
    except ValueError:
        return s
    if sign == "-":
        val = -val
    if abs(val) < 1e-9:
        col = MUTED
    else:
        positive = val > 0
        col = UP if (positive == good_up) else DOWN
    shown = f"{sign}{digits}{unit}"
    weight = "font-weight:700;" if bold else ""
    return f"<span style='color:{col};{weight}'>{_esc(shown, 40)}</span>"


def _h(title: str, sub: str = "") -> str:
    return (f"<h3 style='margin:16px 0 4px;font-size:14px;color:#34495e;"
            f"border-bottom:2px solid #34495e;padding-bottom:3px'>{title}</h3>"
            + (f"<div style='color:{MUTED};font-size:12px;margin:0 0 8px'>{sub}</div>"
               if sub else ""))


def _rows_html(rows, cols) -> str:
    if not rows:
        return ""
    head = "".join(f"<th style='text-align:left;border-bottom:1px solid #ccc;"
                   f"color:{MUTED};font-weight:600'>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table cellpadding='4' cellspacing='0' style='{_TBL}'><tr>{head}</tr>{body}</table>"


def _slice(df, isin, cols=None):
    if df is None or getattr(df, "empty", True) or "isin" not in df.columns:
        return pd.DataFrame(columns=cols or [])
    d = df[df["isin"].astype(str).str.strip() == str(isin).strip()]
    return d if not d.empty else pd.DataFrame(columns=cols or list(df.columns))


# ------------------------------------------------------------------ #
#  Presentation: current deck + what changed                          #
# ------------------------------------------------------------------ #

# The eighteen deck_summary sections, with the heading a reader sees and the colour that
# lets them scan for it. Risk is red and guidance purple for the same reason as the theme
# chips: those are the two a reader hunts for. The rest carry the neutral band so the
# colour still means something when it appears.
_SUMMARY_SECTIONS = (
    ("business_overview",     "What the company does",            BLUE),
    ("financials",            "Financials, as the deck states them", BLUE),
    ("balance_sheet",         "Balance sheet",                    BLUE),
    ("segment_performance",   "Segment performance",              BLUE),
    ("geography",             "Geography",                        BLUE),
    ("capacity_expansion",    "Capacity and expansion",           BLUE),
    ("capex",                 "Capex",                            BLUE),
    ("orderbook_pipeline",    "Order book and pipeline",          BLUE),
    ("customers",             "Customers",                        BLUE),
    ("products_rd",           "Products and R&amp;D",             BLUE),
    ("industry_market",       "Industry and market",              BLUE),
    ("strategy",              "Strategy",                         PURPLE),
    ("guidance_outlook",      "Guidance and outlook",             PURPLE),
    ("risks",                 "Risks the deck admits",            DOWN),
    ("management_commentary", "Management commentary",            "#34495e"),
    ("capital_allocation",    "Capital allocation",               BLUE),
    ("subsidiary_ma",         "Subsidiaries and M&amp;A",         BLUE),
    ("esg",                   "ESG",                              MUTED),
)


def standalone_summary(isin, season, tables) -> str:
    """The deck read as a company note in its own right.

    This is the section that matters for a company that never holds a concall: eighteen
    sections spanning what the business does, how it performed, what it is building, what
    it promises and what it admits — every line quoted from the deck at extraction.

    The coverage count is printed deliberately. "7 of 18 sections" tells a reader the deck
    was thin, which is itself information about the company; without it, an absent
    balance sheet reads as a pipeline failure rather than as a 15-page deck.
    """
    ds = _slice(tables.get("deck_summary"), isin)
    if ds.empty:
        return ""
    if "quarter" in ds.columns:
        want = QT.norm_q(season)
        ds = ds[ds["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]
    if ds.empty:
        return ""

    present = {str(s).strip() for s in ds["section"]}
    n_have = sum(1 for s, _, _ in _SUMMARY_SECTIONS if s in present)

    out = [_h("The deck on its own terms",
              f"Everything this presentation discloses, quoted from it. "
              f"<b>{n_have} of 18</b> sections are covered by this deck.")]

    body = []
    for sec, title, colour in _SUMMARY_SECTIONS:
        part = ds[ds["section"].astype(str).str.strip() == sec]
        if part.empty:
            continue
        body.append(f"<tr><td colspan='3' style='padding-top:9px;color:{colour};"
                    f"font-weight:700;border-bottom:1px solid {colour}'>{title}</td></tr>")
        for _, r in part.iterrows():
            val = r.get("value")
            num = ""
            if val is not None and str(val).strip() not in ("", "nan", "None"):
                unit = _esc(DS.display_unit(r.get("unit")), 12)
                # A percentage is directional and gets the up/down colour; an absolute
                # level is not — a bigger number is not automatically better news.
                txt = f"{float(val):g}"
                num = (colour_num(f"{txt}{unit}") if unit == "%"
                       else f"<b>{_esc(txt, 18)}{(' ' + unit) if unit else ''}</b>")
            per = _esc(r.get("period"), 16)
            det = _esc(r.get("detail"), 210)
            body.append(
                f"<tr><td style='width:31%'>{_esc(r.get('label'), 70)}"
                + (f" <span style='color:{MUTED};font-size:11px'>{per}</span>" if per else "")
                + f"</td><td style='width:17%;text-align:right'>{num}</td>"
                  f"<td style='color:#444'>{det}</td></tr>")

    if not body:
        return ""
    out.append(f"<table cellpadding='4' cellspacing='0' style='{_TBL}'>"
               + "".join(body) + "</table>")
    return "".join(out)


def presentation_body(isin, symbol, name, season, tables) -> str:
    """This quarter's deck, then how it differs from the previous one."""
    hi = _slice(tables.get("ppt_highlights"), isin)
    gu = _slice(tables.get("ppt_guidance"), isin)
    want = QT.norm_q(season)

    def _q(df):
        if df.empty or "quarter" not in df.columns:
            return df
        return df[df["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]

    cur_hi, cur_gu = _q(hi), _q(gu)
    # The standalone summary is computed BEFORE the gate: a company that holds no concall
    # may have a full deck_summary and no ppt_highlights at all, and gating on the old
    # two tables would suppress exactly the mail this section was built for.
    standalone = standalone_summary(isin, season, tables)
    if cur_hi.empty and cur_gu.empty and not standalone:
        return ""

    out = [f"<div style='{_WRAP}'>",
           f"<h2 style='margin:0 0 2px'>&#128202; {_esc(name, 70)} "
           f"<span style='color:#888;font-weight:400'>&middot; {_esc(symbol, 20)}</span></h2>",
           f"<div style='color:#888;font-size:12px;margin:0 0 10px'>Investor presentation "
           f"&middot; <b>{_esc(QT.qtr_label(season) if ' ' not in season else season, 12)}</b>"
           f"</div>"]

    def _val(r):
        v, u = _esc(r.get("value"), 24), _esc(r.get("unit"), 10)
        return f"<b>{v}{u}</b>" if v else ""

    out.append(_key_callouts(cur_hi))
    out.append(standalone)

    hi_rows = [[_esc(r.get("category"), 24), _esc(r.get("statement"), 220), _val(r)]
               for _, r in cur_hi.iterrows()
               if not _is_esg(r.get("statement"), r.get("category"))]
    if hi_rows:
        out.append(_h("What the deck says", "Highlights as published this quarter.")
                   + _rows_html(hi_rows[:10], ["Area", "Statement", "Value"]))

    gu_rows = [[f"<b>{_esc(r.get('metric'), 40)}</b>", _val(r), _esc(r.get("horizon"), 24)]
               for _, r in cur_gu.iterrows() if not _is_esg(r.get("metric"))]
    if gu_rows:
        out.append(_h("Guidance given in this deck") +
                   _rows_html(gu_rows[:8], ["Metric", "Value", "Horizon"]))
    elif not cur_gu.empty:
        out.append(f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                   f"The deck's stated targets this quarter are ESG commitments only "
                   f"(emissions, renewables, workforce) — no financial or operational "
                   f"guidance was given.</div>")

    out.append(_business_view(isin, season, tables))
    out.append(_risks_and_changes(isin, season, tables))
    out.append(_consistency(isin, season, tables))
    out.append(_promised_vs_delivered(isin, tables))
    out.append(_deck_changed(isin, season, tables))
    out.append(f"<div style='color:{MUTED};font-size:11px;margin-top:10px'>Deck rows are "
               f"quoted from the presentation itself; anything not evidenced in it is "
               f"dropped at extraction.</div></div>")
    return "".join(out)


_VERDICT_TONE = {"beat": UP, "exceeded": UP, "delivered": UP, "met": UP,
                 "inline": MUTED, "partial": AMBER, "too_early": MUTED, "na": MUTED,
                 "miss": DOWN, "missed": DOWN, "below": DOWN}


_KEY_CATEGORIES = ("orderbook", "order book", "capacity", "utilisation", "utilization",
                   "volume", "realisation", "realization", "margin", "demand",
                   "expansion", "capex", "guidance")


def _key_callouts(cur_hi) -> str:
    """The three or four lines worth reading first.

    A deck highlights table is long and flat. What a reader wants at the top is the
    handful of statements that carry a NUMBER in a category that moves a thesis —
    order book, capacity, utilisation, volume, realisation, margin. Everything else
    stays below in the full table.
    """
    scored = []
    for _, r in cur_hi.iterrows():
        stmt, cat = str(r.get("statement") or ""), str(r.get("category") or "")
        val, unit = str(r.get("value") or ""), str(r.get("unit") or "")
        if _is_esg(stmt, cat):
            continue
        has_num = bool(val.strip()) and val.strip().lower() not in ("nan", "none")
        in_key = any(k in (cat + " " + stmt).lower() for k in _KEY_CATEGORIES)
        if not has_num:
            continue
        scored.append((2 if in_key else 1, stmt, val, unit, cat))
    if not scored:
        return ""
    scored.sort(key=lambda x: -x[0])
    cards = ""
    for _s, stmt, val, unit, cat in scored[:4]:
        cards += (
            f"<td style='vertical-align:top;padding:0 8px 0 0;width:25%'>"
            f"<div style='border:1px solid #e0e6ea;border-left:3px solid {UP};"
            f"background:#f6f8f9;padding:8px 10px'>"
            f"<div style='font-size:10.5px;color:{MUTED};text-transform:uppercase;"
            f"letter-spacing:.3px'>{_esc(cat, 22)}</div>"
            f"<div style='font-size:17px;font-weight:700;color:#111;margin:2px 0'>"
            f"{_esc(val, 18)}{_esc(unit, 8)}</div>"
            f"<div style='font-size:11px;color:#444'>{_esc(stmt, 90)}</div>"
            f"</div></td>")
    return (_h("Key call-outs", "The numbers from this deck worth reading first.")
            + f"<table cellpadding='0' cellspacing='0' style='{_TBL}'><tr>{cards}</tr>"
              f"</table>")


# The deck's job is the OPERATING picture: what the business is physically doing. The
# statement says how the money behaved; these say how the business behaved. Categories
# come from deck_teardown's controlled vocabulary plus the deck's own highlight buckets.
_BUSINESS_VIEWS = (
    ("Capacity and expansion", ("capacity", "expansion", "capex", "plant", "greenfield",
                                "brownfield", "commission")),
    ("Utilisation", ("utilisation", "utilization", "occupancy", "run rate")),
    ("Order book and pipeline", ("orderbook", "order book", "pipeline", "backlog",
                                 "tender", "win")),
    ("Volume and realisation", ("volume", "realisation", "realization", "tonnage",
                                "asp", "price")),
    ("Segment mix", ("segment", "product mix", "category", "vertical")),
    ("Geography", ("geo", "geograph", "export", "domestic", "region")),
    ("Demand and macro", ("demand", "macro", "industry", "market size", "cycle",
                          "consumption")),
)



# Themes an operating line can belong to, each with the colour it is emphasised in.
# Order matters: the first match wins, so the more specific themes are listed first.
_THEMES = (
    ("guidance", PURPLE, ("guidance", "target", "outlook", "expect", "aim", "plan to",
                          "by fy", "we will", "on track")),
    ("risk", DOWN, ("risk", "headwind", "pressure", "decline", "slowdown", "delay",
                    "deferred", "shortfall", "weak", "challenge", "impact of")),
    ("capacity", BLUE, ("capacity", "expansion", "capex", "commission", "plant",
                        "greenfield", "brownfield", "debottleneck", "utilisation",
                        "utilization")),
    ("growth", UP, ("growth", "grew", "increase", "record", "highest", "up ", "demand",
                    "order", "volume", "market share", "new product", "launch")),
)


def theme_of(text: str) -> tuple[str, str]:
    """(theme, colour) for an operating line — so a risk never reads green."""
    low = str(text or "").lower()
    for name, col, keys in _THEMES:
        if any(k in low for k in keys):
            return name, col
    return "", MUTED


def _chip(label: str, colour: str) -> str:
    return (f"<span style='background:{colour};color:#fff;border-radius:3px;"
            f"padding:1px 6px;font-size:10px;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.3px'>{label}</span>")


def _risks_and_changes(isin, season, tables) -> str:
    """Risks and management changes — which the DECK will not tell you.

    deck_metrics bans financial words and decks rarely volunteer bad news, so neither
    risk nor a management change appears there. Both have real sources elsewhere:
      framing flags   deck_flags      — how the deck is staged (truncated axes, etc.)
      commentary      gf4_quality_flags — contradictory or promotional concall language
      management      announcement_ledger event_type='management_change'
    Sourcing them honestly is better than implying the deck disclosed them.
    """
    rows = []
    df = _slice(tables.get("deck_flags"), isin)
    if not df.empty:
        for _, r in df.head(4).iterrows():
            rows.append([_chip("framing", AMBER),
                         _esc(str(r.get("flag_type") or "").replace("_", " "), 40),
                         _esc(r.get("evidence"), 170)])
    gf4 = _slice(tables.get("gf4_quality_flags"), isin)
    if not gf4.empty:
        for _, r in gf4.head(3).iterrows():
            rows.append([_chip("commentary", AMBER),
                         _esc(r.get("flag_type"), 40), _esc(r.get("evidence"), 170)])
    ann = tables.get("announcement_ledger")
    if ann is not None and not getattr(ann, "empty", True) and "event_type" in ann.columns:
        a = ann[(ann["isin"].astype(str).str.strip() == str(isin).strip())
                & (ann["event_type"].astype(str) == "management_change")]
        if "ann_date" in a.columns:
            a = a.sort_values("ann_date", ascending=False)
        for _, r in a.head(3).iterrows():
            rows.append([_chip("management", PURPLE),
                         _esc(str(r.get("ann_date"))[:10], 12),
                         _esc(r.get("summary") or r.get("headline"), 170)])
    if not rows:
        return ""
    return (_h("Risks and changes",
               "Not from the deck — companies rarely put these in one. Framing flags "
               "come from the deck teardown, commentary flags from the concall, and "
               "management changes from exchange filings.")
            + _rows_html(rows, ["", "What", "Detail"]))


def _business_view(isin, season, tables) -> str:
    """How the BUSINESS is behaving — capacity, utilisation, orders, mix, geography, macro.

    The deck is where the operating story lives, and it was being rendered as one flat
    list of highlights. Grouping it into the views an analyst actually asks for turns the
    same rows into an operating picture: is capacity going up, is it being used, is the
    order book covering it, where is the volume coming from, and what is the company
    saying about its market.

    deck_metrics (the grounded teardown pass) is preferred where it exists, because every
    row there had to quote the deck verbatim; ppt_highlights fills in otherwise.
    """
    hi = _slice(tables.get("ppt_highlights"), isin)
    dm = _slice(tables.get("deck_metrics"), isin)
    want = QT.norm_q(season)

    items = []
    if not dm.empty:
        for _, r in dm.iterrows():
            q = QT.norm_q(str(r.get("quarter") or ""))
            if q and q != want:
                continue
            items.append((f"{r.get('category','')} {r.get('metric','')}",
                          str(r.get("metric") or ""),
                          f"{_esc(r.get('value'), 20)}{_esc(r.get('unit'), 10)}",
                          str(r.get("slide_ref") or "")))
    if not items and not hi.empty:
        h = hi
        if "quarter" in h.columns:
            h = h[h["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]
        for _, r in h.iterrows():
            if _is_esg(r.get("statement"), r.get("category")):
                continue
            items.append((f"{r.get('category','')} {r.get('statement','')}",
                          str(r.get("statement") or ""),
                          f"{_esc(r.get('value'), 20)}{_esc(r.get('unit'), 10)}", ""))
    if not items:
        return ""

    out, used = [], set()
    for title, keys in _BUSINESS_VIEWS:
        rows = []
        for i, (blob, label, val, ref) in enumerate(items):
            if i in used:
                continue
            if any(k in blob.lower() for k in keys):
                used.add(i)
                th, tcol = theme_of(label)
                rows.append([_chip(th, tcol) if th else "",
                             _esc(label, 190),
                             colour_num(val) if val else "",
                             _esc(ref, 14)])
        if rows:
            out.append(_h(title) + _rows_html(rows[:6], ["", "What the deck says",
                                                         "Value", "Slide"]))
    if not out:
        return ""
    return (_h("How the business is behaving",
               "The operating picture from the deck — capacity, utilisation, orders, "
               "mix, geography and the market backdrop. The financial teardown covers "
               "how the money behaved; this covers how the business did.")
            + "".join(out))


def _consistency(isin, season, tables) -> str:
    """What the company keeps saying, and what it has quietly stopped saying.

    Two halves, because they fail differently:
      NUMBERS — a metric reported in several quarters, and how its value moved. A target
                restated unchanged is a promise being kept; one revised down without
                comment is the thing worth catching.
      COMMENTARY — a theme repeated quarter after quarter is a consistent message; one
                that ran for several quarters and then vanished is the cheapest early
                warning there is, because companies stop mentioning bad news rather than
                announcing it.
    """
    hi = _slice(tables.get("ppt_highlights"), isin)
    if hi.empty or "quarter" not in hi.columns:
        return ""
    hi = hi.copy()
    hi["_q"] = hi["quarter"].astype(str).map(QT.norm_q)
    quarters = sorted({q for q in hi["_q"] if q}, key=QT.q_order)
    if len(quarters) < 2:
        return ""
    cur, prev = quarters[-1], quarters[-2]

    def _rows(q):
        return hi[hi["_q"] == q]

    # ---- numbers: same statement, different value
    def _numkey(r):
        return _PUNCT.sub(" ", str(r.get("statement") or "").lower()).strip()[:70]

    cur_n = {_numkey(r): (str(r.get("value") or ""), str(r.get("unit") or ""))
             for _, r in _rows(cur).iterrows() if str(r.get("value") or "").strip()}
    prev_n = {_numkey(r): (str(r.get("value") or ""), str(r.get("unit") or ""))
              for _, r in _rows(prev).iterrows() if str(r.get("value") or "").strip()}
    same, moved = [], []
    for k, (v, u) in cur_n.items():
        if k not in prev_n:
            continue
        pv, _pu = prev_n[k]
        if str(v).strip() == str(pv).strip():
            same.append((k, v, u))
        else:
            moved.append((k, pv, v, u))

    # ---- commentary: themes carried forward vs dropped
    def _txt(q):
        return {_PUNCT.sub(" ", str(r.get("statement") or "").lower()).strip()[:70]:
                str(r.get("statement") or "")
                for _, r in _rows(q).iterrows()
                if not _is_esg(r.get("statement"), r.get("category"))}
    ct, pt = _txt(cur), _txt(prev)
    carried = [ct[k] for k in ct if k in pt][:4]
    dropped = [pt[k] for k in pt if k not in ct][:5]

    if not (same or moved or carried or dropped):
        return ""

    out = [_h("Consistent, and not",
              f"{_esc(cur,10)} against {_esc(prev,10)} — the same claim restated, "
              f"revised, or quietly dropped.")]
    rows = []
    for k, pv, v, u in moved[:6]:
        rows.append([f"<span style='color:{AMBER};font-weight:700'>revised</span>",
                     _esc(k, 70), _esc(pv, 20), f"{colour_num(v)}{_esc(u, 8)}"])
    for k, v, u in same[:4]:
        rows.append([f"<span style='color:{UP};font-weight:700'>restated</span>",
                     _esc(k, 70), _esc(v, 20) + _esc(u, 8), "unchanged"])
    if rows:
        out.append(_rows_html(rows, ["Number", "What", f"{_esc(prev,9)}", f"{_esc(cur,9)}"]))
    crows = []
    for s in dropped:
        crows.append([f"<span style='color:{DOWN};font-weight:700'>stopped saying</span>",
                      _esc(s, 200)])
    for s in carried:
        crows.append([f"<span style='color:{UP};font-weight:700'>still saying</span>",
                      _esc(s, 200)])
    if crows:
        out.append(_rows_html(crows, ["Commentary", "Statement"]))
    return "".join(out)


_PUNCT = __import__("re").compile(r"[^a-z0-9]+")


def _promised_vs_delivered(isin, tables) -> str:
    """What management SAID they would do, against what actually happened.

    "A KPI stopped being reported" is only half the signal. The half that matters more is
    a promise that was made and then quietly not kept: a margin target missed, a capacity
    date pushed out, a guidance range revised down. guidance_vs_actual already holds the
    join (guided / actual / delta / verdict) and mgmt_credibility holds the pattern; both
    were computed and never surfaced in a per-company mail.
    """
    gva = _slice(tables.get("guidance_vs_actual"), isin)
    cred = _slice(tables.get("mgmt_credibility"), isin)
    out = []

    if not gva.empty:
        g = gva.copy()
        if "period" in g.columns:
            g = g.sort_values("period")
        rows = []
        for _, r in g.tail(8).iloc[::-1].iterrows():
            v = str(r.get("verdict") or "").strip().lower().replace(" ", "_")
            tone = _VERDICT_TONE.get(v, MUTED)
            delta = r.get("delta")
            rows.append([
                _esc(r.get("period"), 12),
                f"<b>{_esc(r.get('metric'), 34)}</b>",
                _esc(r.get("guided"), 34),
                _esc(r.get("actual"), 34) or "&mdash;",
                colour_num(delta) if str(delta or "").strip() else "",
                f"<span style='color:{tone};font-weight:700'>"
                f"{_esc(str(r.get('verdict') or '').upper(), 12)}</span>",
            ])
        if rows:
            out.append(_h("Promised vs delivered",
                          "Management's own prior guidance, joined to what actually "
                          "happened. A missed promise is a stronger signal than a "
                          "dropped metric, because someone chose to make it.")
                       + _rows_html(rows, ["Period", "Metric", "Guided", "Actual",
                                           "Delta", "Verdict"]))

    hist = _slice(tables.get("gf2_historical_guidance"), isin)
    if not hist.empty:
        rows = []
        for _, r in hist.head(5).iterrows():
            orig = _esc(r.get("original_guidance"), 120)
            outc = _esc(r.get("actual_mentioned_outcome"), 120)
            if not orig:
                continue
            rows.append([_esc(r.get("financial_qtr"), 12), orig, outc or "&mdash;",
                         _esc(r.get("management_self_assessment"), 60)])
        if rows:
            out.append(_h("Historical guidance, revisited",
                          "What management said in earlier quarters and how they have "
                          "since described the outcome — in their own words.")
                       + _rows_html(rows, ["Quarter", "Said then", "Outcome",
                                           "Their assessment"]))

    # Consistency of the message itself — a team that guides accurately every quarter is
    # worth more than one that beats erratically, and the pattern is already computed.
    if not cred.empty:
        pat = ""
        for c in ("pattern", "recurring_miss", "strongest_area"):
            if c in cred.columns:
                vals = [str(x).strip() for x in cred[c]
                        if str(x).strip() and str(x).lower() not in ("none", "nan")]
                if vals:
                    label = {"pattern": "Guidance pattern",
                             "recurring_miss": "Recurring miss",
                             "strongest_area": "Most reliable on"}[c]
                    tone = (UP if c == "strongest_area" else
                            DOWN if c == "recurring_miss" else MUTED)
                    pat += (f"<tr><td style='color:{MUTED}'>{label}</td>"
                            f"<td style='color:{tone};font-weight:700'>"
                            f"{_esc(vals[-1], 90)}</td></tr>")
        if pat:
            out.append(_h("Is the commentary consistent?",
                          "Whether this management guides reliably, and where it "
                          "habitually falls short.")
                       + f"<table cellpadding='4' cellspacing='0' style='{_TBL}'>"
                         f"{pat}</table>")
    return "".join(out)


def _deck_changed(isin, season, tables) -> str:
    """What moved versus the previous deck. Deterministic set comparison."""
    diff = _slice(tables.get("deck_diff"), isin)
    want = QT.norm_q(season)
    if not diff.empty and "quarter" in diff.columns:
        diff = diff[diff["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]
    if not diff.empty:
        tone = {"kpi_dropped": DOWN, "target_moved": DOWN, "definition_changed": AMBER,
                "de_emphasised": AMBER, "new_emphasis": UP}
        rows = [[f"<span style='color:{tone.get(str(r.get('change_type')), MUTED)};"
                 f"font-weight:700'>{_esc(str(r.get('change_type')).replace('_',' '), 24)}</span>",
                 _esc(r.get("item"), 60),
                 f"{_esc(r.get('prior_state'), 40)} &rarr; {_esc(r.get('current_state'), 40)}"]
                for _, r in diff.head(10).iterrows()]
        return (_h("What changed since the last deck",
                   "A KPI shown for several quarters and absent now is the cheapest early "
                   "warning there is — companies rarely announce bad news, they stop "
                   "mentioning it.") + _rows_html(rows, ["Change", "Item", "Was &rarr; now"]))

    # Fall back to comparing the metric sets ourselves when the LLM diff has no rows.
    met = _slice(tables.get("deck_metrics"), isin)
    if met.empty or "quarter" not in met.columns:
        return (f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                f"No previous deck has been processed for this company yet, so there is "
                f"nothing to compare against — this is the first one on record, not an "
                f"all-clear.</div>")
    qs = sorted({QT.norm_q(q) for q in met["quarter"].astype(str)}, key=QT.q_order)
    if len(qs) < 2:
        return (f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                f"Only one quarter of deck data on record ({_esc(qs[0] if qs else '?', 12)}), "
                f"so no comparison is possible yet. The next deck makes this section live."
                f"</div>")
    cur, prev = qs[-1], qs[-2]
    def _keys(q):
        s = met[met["quarter"].astype(str).map(lambda x: QT.norm_q(x) == q)]
        return {str(r.get("metric") or "").strip().lower() for _, r in s.iterrows()}
    dropped = sorted(_keys(prev) - _keys(cur))
    added = sorted(_keys(cur) - _keys(prev))
    rows = []
    for m in dropped[:8]:
        rows.append([f"<span style='color:{DOWN};font-weight:700'>stopped reporting</span>",
                     _esc(m, 60), f"shown in {_esc(prev, 12)}, absent in {_esc(cur, 12)}"])
    for m in added[:6]:
        rows.append([f"<span style='color:{UP};font-weight:700'>newly reported</span>",
                     _esc(m, 60), f"not in {_esc(prev, 12)}"])
    if not rows:
        return (f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                f"Same metrics reported as {_esc(prev, 12)} — nothing quietly dropped.</div>")
    return (_h("What changed since the last deck",
               f"Comparing {_esc(cur,12)} against {_esc(prev,12)}, by which metrics the "
               f"company chose to show.") + _rows_html(rows, ["Change", "Metric", "Detail"]))


# ------------------------------------------------------------------ #
#  Rating: current + trajectory                                       #
# ------------------------------------------------------------------ #

def rating_body(isin, symbol, name, tables) -> str:
    """Latest rating action, then what moved since the previous one."""
    rt = _slice(tables.get("ratings"), isin)
    if rt.empty:
        return ""
    rt = rt.copy()
    rt["_d"] = rt["rating_date"].astype(str).str.slice(0, 10)
    rt = rt[rt["_d"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)].sort_values("_d")
    if rt.empty:
        return ""
    cur = rt.iloc[-1]

    out = [f"<div style='{_WRAP}'>",
           f"<h2 style='margin:0 0 2px'>&#127991; {_esc(name, 70)} "
           f"<span style='color:#888;font-weight:400'>&middot; {_esc(symbol, 20)}</span></h2>",
           f"<div style='color:#888;font-size:12px;margin:0 0 10px'>Credit rating "
           f"&middot; {_esc(cur.get('agency'), 24)} &middot; {_esc(cur['_d'], 12)}</div>"]

    # KEY NUMBERS FIRST. A rating mail is mostly prose; the figures that actually get
    # compared quarter to quarter are the rated amount, how long the rating has stood,
    # and how the agency's own tally of strengths against concerns has moved.
    _dr = _slice(tables.get("rating_drivers"), isin)
    _cn2 = _slice(tables.get("rating_concerns"), isin)
    _sn2 = _slice(tables.get("rating_sensitivity"), isin)
    _amt = str(cur.get("rated_amount_cr") or "").strip()
    _n_same = len(rt[rt["rating"].astype(str) == str(cur.get("rating"))])
    _tiles = ""
    for _lab, _v, _good in (
            ("Rated amount", f"{_amt} Cr" if _amt else "", True),
            ("Strengths cited", str(len(_dr)) if len(_dr) else "", True),
            ("Concerns cited", str(len(_cn2)) if len(_cn2) else "", False),
            ("Downgrade triggers", str(len(_sn2[_sn2.get("direction", pd.Series(dtype=str))
                                               .astype(str).str.lower() == "down"]))
             if len(_sn2) else "", False),
            ("Quarters at this rating", str(_n_same) if _n_same else "", True)):
        if not _v:
            continue
        _tiles += (f"<td style='vertical-align:top;padding:0 8px 0 0'>"
                   f"<div style='border:1px solid #e0e6ea;background:#f6f8f9;"
                   f"padding:7px 10px'>"
                   f"<div style='font-size:10.5px;color:{MUTED};text-transform:uppercase;"
                   f"letter-spacing:.3px'>{_lab}</div>"
                   f"<div style='font-size:16px;margin-top:2px'>"
                   f"{colour_num(_v, good_up=_good)}</div></div></td>")
    if _tiles:
        out.append(f"<table cellpadding='0' cellspacing='0' style='{_TBL}'>"
                   f"<tr>{_tiles}</tr></table>")

    act = str(cur.get("rating_action") or "")
    col = DOWN if "downgrade" in act.lower() else UP if "upgrade" in act.lower() else MUTED
    out.append(_rows_html([
        ["Rating", f"<b>{_esc(cur.get('rating'), 40)}</b>", ""],
        ["Outlook", _esc(cur.get("outlook"), 30), ""],
        ["Action", f"<span style='color:{col};font-weight:700'>{_esc(act, 24) or '&mdash;'}</span>", ""],
        ["Instrument", _esc(cur.get("instrument_type"), 40),
         f"{_esc(cur.get('rated_amount_cr'), 16)} Cr" if str(cur.get("rated_amount_cr") or "").strip() else ""],
    ], ["", "Current", ""]))

    # ---- what changed vs the previous rationale from the SAME agency
    same = rt[rt["agency"].astype(str) == str(cur.get("agency"))]
    if len(same) >= 2:
        prev = same.iloc[-2]
        rows = []
        if str(prev.get("rating")) != str(cur.get("rating")):
            rows.append(["Rating", _esc(prev.get("rating"), 40),
                         f"<b>{_esc(cur.get('rating'), 40)}</b>"])
        if str(prev.get("outlook")) != str(cur.get("outlook")):
            rows.append(["Outlook", _esc(prev.get("outlook"), 30),
                         f"<b>{_esc(cur.get('outlook'), 30)}</b>"])
        if rows:
            out.append(_h(f"What changed since {_esc(prev['_d'], 12)}")
                       + _rows_html(rows, ["", "Was", "Now"]))
        else:
            out.append(f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                       f"Rating and outlook unchanged since {_esc(prev['_d'], 12)}.</div>")

    # ---- concerns: new vs carried over
    cn = _slice(tables.get("rating_concerns"), isin)
    if not cn.empty and "quarter" in cn.columns or not cn.empty:
        txt_col = "concern" if "concern" in cn.columns else (
            "detail" if "detail" in cn.columns else cn.columns[-1])
        allc = [str(x).strip() for x in cn[txt_col] if str(x).strip()]
        if allc:
            rows = [[_esc(c, 220)] for c in allc[:6]]
            out.append(_h("Agency concerns on record",
                          "The agency's own words on what could go wrong.")
                       + _rows_html(rows, ["Concern"]))

    # ---- WHAT IS WORKING. The agency's own stated strengths — order book, market
    # position, promoter support, margin resilience. A rating mail that lists only
    # concerns and downgrade triggers reads as uniformly bearish even when the agency's
    # case is largely favourable, which misrepresents the document.
    dr = _slice(tables.get("rating_drivers"), isin)
    if not dr.empty:
        dcol = ("driver" if "driver" in dr.columns else
                "evidence" if "evidence" in dr.columns else dr.columns[-1])
        if "rating_date" in dr.columns:
            dr = dr.copy()
            dr["_d"] = dr["rating_date"].astype(str).str.slice(0, 10)
            latest_d = dr["_d"].max()
            dr = dr[dr["_d"] == latest_d]          # the current rationale only
        seen, rows = set(), []
        for _, r in dr.iterrows():
            txt = str(r.get(dcol) or "").strip()
            key = txt.lower()[:60]
            if not txt or key in seen:
                continue
            seen.add(key)
            rows.append([f"<span style='color:{UP}'>&#9679;</span> {_esc(txt, 240)}"])
            if len(rows) >= 6:
                break
        if rows:
            out.append(_h("What the agency says is working",
                          "Strengths cited in the current rationale — the other half of "
                          "the credit view.")
                       + _rows_html(rows, ["Strength"]))

    # ---- the most underused table in the repo: what would trigger a downgrade
    sn = _slice(tables.get("rating_sensitivity"), isin)
    if not sn.empty:
        dcol = "direction" if "direction" in sn.columns else None
        tcol = ("trigger" if "trigger" in sn.columns else
                "sensitivity" if "sensitivity" in sn.columns else sn.columns[-1])
        down = sn[sn[dcol].astype(str).str.lower() == "down"] if dcol else sn
        if not down.empty:
            rows = [[f"<span style='color:{DOWN}'>&#9660;</span> {_esc(r.get(tcol), 240)}"]
                    for _, r in down.head(6).iterrows()]
            out.append(_h("What would cause a downgrade",
                          "The agency stating, in its own words, exactly what it is "
                          "watching. A pre-written early-warning tripwire.")
                       + _rows_html(rows, ["Trigger"]))
        # The upgrade side is equally pre-written and never shown anywhere.
        up_ = sn[sn[dcol].astype(str).str.lower() == "up"] if dcol else sn.iloc[0:0]
        if not up_.empty:
            rows = [[f"<span style='color:{UP}'>&#9650;</span> {_esc(r.get(tcol), 240)}"]
                    for _, r in up_.head(5).iterrows()]
            out.append(_h("What would earn an upgrade",
                          "The conditions the agency has committed to rewarding — worth "
                          "tracking against the company's own guidance.")
                       + _rows_html(rows, ["Trigger"]))

    out.append("</div>")
    return "".join(out)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def _read(drive, idx, name):
    from _extractor_base import find_file, download_bytes
    try:
        fid = find_file(drive, idx, name)
        if not fid:
            return pd.DataFrame()
        return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write previews, send nothing, do not advance the ledger.")
    ap.add_argument("--types", default="results,presentation,rating",
                    help="Which mails to consider.")
    ap.add_argument("--limit", type=int, default=0, help="Max mails this run.")
    ap.add_argument("--window-days", type=int, default=2,
                    help="How far back a calendar date still counts (backwards only).")
    ap.add_argument("--require-calendar", action="store_true",
                    help="Results mails additionally require a board-meeting date "
                         "on/near today. Presentations and ratings never do — they "
                         "arrive on no calendar.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    from _extractor_base import (get_drive, get_or_create_subfolder, load_queue,
                                 load_parquet, save_parquet, log)
    from daily_brief import load_pf

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")

    season = QT.season_quarter()
    today = date.today()
    pf = load_pf(drive, root, idx)
    queue = load_queue(drive, idx)
    calendar = _read(drive, idx, "results_calendar.parquet")
    ledger = load_parquet(drive, idx, LEDGER_NAME, LEDGER_COLS)
    want = {t.strip() for t in args.types.split(",") if t.strip()}

    # The same tables the renderers read, so "due" means "will actually send" —
    # coverage tested only that an extractor had run, which reported 16 presentations
    # as due and then skipped every one as "nothing renderable".
    _pre = {n: _read(drive, idx, f"{n}.parquet") for n in ("ppt_highlights", "ratings")}
    due = COV.mail_due(pf, queue, calendar, ledger, season, on=today,
                       window_days=args.window_days, doc_types=tuple(want),
                       require_calendar=args.require_calendar, tables=_pre)
    log(f"season={season} pf={len(pf)} due={len(due)} "
        f"({', '.join(sorted({d['doc_type'] for d in due})) or 'nothing'})")
    if not due:
        log("nothing due — no mail.")
        return

    tables = {n: _read(drive, idx, f"{n}.parquet") for n in
              ("ppt_highlights", "ppt_guidance", "deck_summary", "deck_metrics", "deck_diff",
               "ratings", "rating_concerns", "rating_sensitivity",
               # positives + promise-tracking: all already computed, never surfaced
               "rating_drivers", "guidance_vs_actual", "mgmt_credibility",
               "gf2_historical_guidance",
               # risks + management changes: the deck does not carry these
               "deck_flags", "gf4_quality_flags", "announcement_ledger")}

    if args.limit:
        due = due[: args.limit]

    sent_rows = []
    for d in due:
        isin, sym, name, dt = d["isin"], d["symbol"], d["name"], d["doc_type"]
        if dt == "presentation":
            body = presentation_body(isin, sym, name, season, tables)
            subject = f"📊 {sym} — investor presentation, {QT.qtr_label(season)}"
        elif dt == "rating":
            body = rating_body(isin, sym, name, tables)
            subject = f"🏷 {sym} — credit rating update"
        else:
            body = ""       # results teardown is rendered by quarter_teardown itself
            subject = ""
        if not body:
            log(f"  {sym:<12} {dt}: nothing renderable — skipped")
            continue
        if len(body.encode()) > MAX_HTML_BYTES:
            log(f"  {sym:<12} {dt}: {len(body.encode()):,} B over budget — trimmed")
            body = body[:MAX_HTML_BYTES]

        if args.dry_run:
            p = os.path.join(args.out_dir, f"mail_{sym}_{dt}.html")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(body)
            log(f"  {sym:<12} {dt}: preview -> {p} ({len(body.encode()):,} B)")
            continue

        from mailer import send_email, load_mail_settings
        if not load_mail_settings(drive, idx).get(MAIL_KEY, True):
            log(f"  mail toggle '{MAIL_KEY}' is OFF — stopping.")
            return
        if not (os.getenv("GMAIL_USER") and os.getenv("GMAIL_APP_PASSWORD")
                and os.getenv("NOTIFY_EMAIL")):
            log("  mail NOT sent — GMAIL_* not set in this environment.")
            return
        ok = send_email(subject, body)
        log(f"  {sym:<12} {dt}: sent={ok}")
        if ok:
            sent_rows.append({"season": season, "isin": isin, "symbol": sym,
                              "doc_type": dt, "period": season,
                              "doc_id": d.get("doc_id", ""),
                              "mailed_at": datetime.now().isoformat(timespec="seconds"),
                              "subject": subject[:200]})

    # Ledger advances ONLY on confirmed sends — a failed mail must stay due.
    if sent_rows:
        out = pd.concat([ledger, pd.DataFrame(sent_rows, columns=LEDGER_COLS)],
                        ignore_index=True) if ledger is not None and not ledger.empty \
            else pd.DataFrame(sent_rows, columns=LEDGER_COLS)
        # Keyed on the DOCUMENT: a second deck or a rating downgrade in the same quarter
        # is a separate row, so it neither overwrites the earlier send nor gets suppressed.
        out = out.drop_duplicates(subset=["season", "isin", "doc_type", "doc_id"],
                                  keep="last")
        save_parquet(drive, idx, LEDGER_NAME, out)
        log(f"ledger: +{len(sent_rows)} -> _index/{LEDGER_NAME} ({len(out)} rows)")


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    T = {"ppt_highlights": pd.DataFrame([
            {"isin": "INE1", "quarter": "Q1 FY27", "category": "demand",
             "statement": "Order book at a record", "value": "1200", "unit": "Cr"}]),
         "ppt_guidance": pd.DataFrame([
            {"isin": "INE1", "quarter": "Q1 FY27", "metric": "revenue growth",
             "value": "15", "unit": "%", "horizon": "FY27"}]),
         "deck_metrics": pd.DataFrame([
            {"isin": "INE1", "quarter": "Q4 FY26", "metric": "utilisation"},
            {"isin": "INE1", "quarter": "Q4 FY26", "metric": "order book"},
            {"isin": "INE1", "quarter": "Q1 FY27", "metric": "order book"}]),
         "deck_diff": pd.DataFrame(),
         "ratings": pd.DataFrame([
            {"isin": "INE1", "agency": "CRISIL", "rating": "A-", "outlook": "Stable",
             "rating_action": "Reaffirmed", "rating_date": "2025-06-01",
             "instrument_type": "LT", "rated_amount_cr": "500"},
            {"isin": "INE1", "agency": "CRISIL", "rating": "A", "outlook": "Positive",
             "rating_action": "Upgrade", "rating_date": "2026-06-01",
             "instrument_type": "LT", "rated_amount_cr": "600"}]),
         "rating_concerns": pd.DataFrame([
            {"isin": "INE1", "concern": "Working capital intensity remains high"}]),
         "rating_sensitivity": pd.DataFrame([
            {"isin": "INE1", "direction": "down", "trigger": "Debt/EBITDA above 3.5x"},
            {"isin": "INE1", "direction": "up", "trigger": "Sustained margin above 15%"}]),
         }

    p = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T)
    check("deck: current highlight rendered", "Order book at a record" in p)
    check("deck: guidance rendered", "revenue growth" in p)
    # THE DIFF, computed deterministically when the LLM table is empty: 'utilisation'
    # was shown in Q4 FY26 and is absent in Q1 FY27.
    check("deck: dropped KPI detected without an LLM", "utilisation" in p)
    check("deck: drop is labelled as such", "stopped reporting" in p)

    one_q = dict(T, deck_metrics=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "metric": "order book"}]))
    p1 = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", one_q)
    check("deck: a single quarter says so, not 'no changes'",
          "no comparison is possible" in p1)
    check("deck: single quarter is not an all-clear", "stopped reporting" not in p1)

    r = rating_body("INE1", "AAA", "A Ltd", T)
    check("rating: current rating shown", ">A<" in r or "A</b>" in r)
    check("rating: upgrade flagged", "Upgrade" in r)
    check("rating: what-changed section present", "What changed since" in r)
    check("rating: prior rating shown as 'was'", "A-" in r)
    check("rating: downgrade trigger surfaced", "Debt/EBITDA above 3.5x" in r)
    # Up-triggers must NOT sit under the downgrade heading, but they SHOULD now have
    # their own — the agency's upgrade conditions were computed and never shown.
    _dn = r.split("What would cause a downgrade")[-1].split("What would earn an upgrade")[0]
    check("rating: up-trigger not filed under downgrade",
          "Sustained margin above 15%" not in _dn)
    check("rating: upgrade triggers get their own section",
          "What would earn an upgrade" in r and "Sustained margin above 15%" in r)
    check("rating: concern surfaced", "Working capital intensity" in r)

    # THE RENDER DEFECTS from the first live run against APLAPOLLO.
    nan_t = dict(T, ppt_highlights=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "demand",
         "statement": "HR Coil tubes to grow faster", "value": float("nan"), "unit": None}]))
    pn = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", nan_t)
    # Substring "nan" also occurs inside "financial", so test the DEFECT: a NaN value
    # or None unit reaching the page as literal text.
    check("NaN value never renders as 'nanNone'",
          "nanNone" not in pn and ">nan<" not in pn and ">None<" not in pn
          and "nan</b>" not in pn)

    esg_t = dict(T, ppt_guidance=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "metric": "Scope 1 & 2 emissions",
         "value": "-25", "unit": "%", "horizon": "2030"},
        {"isin": "INE1", "quarter": "Q1 FY27", "metric": "Female workforce",
         "value": "1", "unit": "%", "horizon": "yearly"}]))
    pe = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", esg_t)
    check("ESG targets are not presented as guidance", "Scope 1" not in pe)
    check("ESG-only decks say so explicitly", "ESG commitments only" in pe)
    check("real guidance still shows", "revenue growth" in
          presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T))

    # POSITIVES: a rating mail listing only concerns misreads a favourable rationale.
    T_pos = dict(T, rating_drivers=pd.DataFrame([
        {"isin": "INE1", "agency": "CRISIL", "rating_date": "2026-06-01",
         "driver": "Healthy order book providing revenue visibility"},
        {"isin": "INE1", "agency": "CRISIL", "rating_date": "2026-06-01",
         "driver": "Established market position"}]))
    rp = rating_body("INE1", "AAA", "A Ltd", T_pos)
    check("rating: strengths are surfaced", "Healthy order book" in rp)
    check("rating: strengths get their own heading", "says is working" in rp)

    # PROMISED VS DELIVERED — the half that matters more than a dropped KPI.
    T_gva = dict(T, guidance_vs_actual=pd.DataFrame([
        {"isin": "INE1", "period": "Q4FY26", "metric": "EBITDA margin",
         "guided": "18%", "actual": "16.1%", "delta": "-1.9", "verdict": "MISS"}]),
        mgmt_credibility=pd.DataFrame([
            {"isin": "INE1", "quarter": "Q1 FY27", "pattern": "Optimistic Bias",
             "recurring_miss": "margin", "strongest_area": "revenue"}]))
    pg = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T_gva)
    check("promise vs outcome is shown", "Promised vs delivered" in pg)
    check("the missed promise is named", "EBITDA margin" in pg)
    check("the verdict is carried", "MISS" in pg)
    check("commentary consistency is reported", "Optimistic Bias" in pg)
    check("a recurring miss is called out", "Recurring miss" in pg)

    # COLOUR CARRIES MEANING, and only for real numbers.
    check("a gain is green", UP in colour_num("+12.5%"))
    check("a loss is red", DOWN in colour_num("-3.2%"))
    check("lower-is-better inverts", UP in colour_num("-8", good_up=False))
    check("text is not force-coloured", colour_num("n/a") == "n/a")
    check("empty stays empty", colour_num(None) == "")

    # KEY CALL-OUTS: numbers in thesis-moving categories, lifted to the top.
    T_kc = dict(T, ppt_highlights=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "orderbook",
         "statement": "Order book at record", "value": "1200", "unit": "Cr"},
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "other",
         "statement": "No number here", "value": "", "unit": ""}]))
    kc = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T_kc)
    check("call-outs lift the numeric highlight", "Key call-outs" in kc)
    check("the number is featured", "1200" in kc)
    check("a statement with no number is not a call-out",
          "No number here" not in kc.split("Key call-outs")[1].split("</table>")[0])

    # CONSISTENCY: restated / revised / stopped saying, across two quarters.
    T_cons = dict(T, ppt_highlights=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q4 FY26", "category": "capacity",
         "statement": "Capacity target by FY28", "value": "8.0", "unit": "Mn"},
        {"isin": "INE1", "quarter": "Q4 FY26", "category": "demand",
         "statement": "Exports scaling strongly", "value": "", "unit": ""},
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "capacity",
         "statement": "Capacity target by FY28", "value": "6.5", "unit": "Mn"},
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "demand",
         "statement": "Domestic demand firm", "value": "", "unit": ""}]))
    cs = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T_cons)
    check("consistency section renders", "Consistent, and not" in cs)
    check("a quietly revised target is flagged", "revised" in cs)
    check("both old and new values are shown", "8.0" in cs and "6.5" in cs)
    check("a dropped theme is called out", "stopped saying" in cs)
    check("the dropped statement is named", "Exports scaling strongly" in cs)

    # RATING KEY NUMBERS
    rk = rating_body("INE1", "AAA", "A Ltd", dict(T, rating_drivers=pd.DataFrame([
        {"isin": "INE1", "agency": "CRISIL", "rating_date": "2026-06-01",
         "driver": "Healthy order book"}])))
    check("rated amount is a key number", "Rated amount" in rk)
    check("strengths are counted", "Strengths cited" in rk)
    check("downgrade triggers are counted", "Downgrade triggers" in rk)

    # THEME EMPHASIS: a risk must never read green just because it sits in a deck.
    check("a risk line is themed red", theme_of("Margin pressure from input costs")[1] == DOWN)
    check("a growth line is themed green", theme_of("Order book at record high")[1] == UP)
    check("a capacity fact is themed blue",
          theme_of("Capacity at 8 Mn tonnes across 12 plants")[1] == BLUE)
    # A capacity TARGET is guidance, not a capacity fact — it is a promise to track,
    # and that is the more useful label of the two.
    check("a capacity TARGET is themed as guidance",
          theme_of("Capacity expansion by FY28")[0] == "guidance")
    check("a guidance line is themed distinctly",
          theme_of("We expect 15% growth by FY28")[1] == PURPLE)
    check("guidance beats growth when both words appear",
          theme_of("We expect strong growth")[0] == "guidance")
    check("risk beats capacity when both appear",
          theme_of("Capacity expansion delayed")[0] == "risk")
    check("an unthemed line stays neutral", theme_of("Board met on Tuesday")[1] == MUTED)

    # RISKS AND CHANGES come from real sources, not the deck.
    T_rc = dict(T, deck_flags=pd.DataFrame([
        {"isin": "INE1", "flag_type": "axis_truncated", "evidence": "Chart on slide 7"}]),
        gf4_quality_flags=pd.DataFrame([
            {"isin": "INE1", "flag_type": "Promotional Commentary",
             "evidence": "best ever quarter"}]),
        announcement_ledger=pd.DataFrame([
            {"isin": "INE1", "ann_date": "2026-08-02", "event_type": "management_change",
             "headline": "CFO resigns", "summary": "CFO stepped down"}]))
    rc = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T_rc)
    check("framing flag surfaced", "axis truncated" in rc)
    check("commentary flag surfaced", "Promotional Commentary" in rc)
    check("management change surfaced", "CFO" in rc)
    check("their sources are named, not implied to be the deck", "Not from the deck" in rc)

    # ---- standalone deck summary: the section that carries a no-concall company
    _ds = pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1FY27", "section": "financials",
         "label": "Consolidated PAT", "value": 194.0, "unit": "Rs. Mn",
         "period": "Q1FY27", "detail": "", "evidence": "194"},
        {"isin": "INE1", "quarter": "Q1FY27", "section": "financials",
         "label": "EBITDA growth", "value": 17.3, "unit": "%", "period": "Q1FY27",
         "detail": "", "evidence": "17.3%"},
        {"isin": "INE1", "quarter": "Q1FY27", "section": "segment_performance",
         "label": "HPDC revenue", "value": -41.2, "unit": "%", "period": "Q1FY27",
         "detail": "", "evidence": "-41.2%"},
        {"isin": "INE1", "quarter": "Q1FY27", "section": "risks", "label": "Concentration",
         "value": None, "unit": "", "period": "", "detail": "Top-5 customers are 46%",
         "evidence": "top 5 are 46% of revenue"},
        {"isin": "INE1", "quarter": "Q4FY26", "section": "capex", "label": "Old capex",
         "value": 500.0, "unit": "cr", "period": "FY26", "detail": "",
         "evidence": "500 cr"}])
    T_ds = dict(T, deck_summary=_ds)
    ss = standalone_summary("INE1", "Q1FY27", T_ds)
    check("standalone summary renders", "The deck on its own terms" in ss)
    check("coverage stated honestly", "3 of 18" in ss)
    check("section heading shown", "Financials, as the deck states them" in ss)
    check("unit normalised for display", "Rs mn" in ss and "Rs. Mn" not in ss)
    check("percent gets direction colour", UP in ss and DOWN in ss)
    check("absolute level is not colour-graded", f">194</b>" in ss or "194 Rs mn" in ss)
    check("risk section carries the red band", "Risks the deck admits" in ss)
    check("narrative detail shown", "Top-5 customers are 46%" in ss)
    check("other quarters excluded", "Old capex" not in ss and "Capex" not in ss)
    check("no deck_summary renders nothing", standalone_summary("INE1", "Q1FY27", T) == "")
    check("other company renders nothing", standalone_summary("INE9", "Q1FY27", T_ds) == "")
    check("summary reaches the mail body",
          "The deck on its own terms" in presentation_body("INE1", "AAA", "A", "Q1FY27", T_ds))
    check("18 sections mapped, matching the extractor",
          {s for s, _, _ in _SUMMARY_SECTIONS} == set(DS.SECTIONS))

    # A company with NO concall and NO ppt_highlights is the whole point: the old gate
    # returned "" for it and the mail this section exists for would never have gone out.
    only_ds = {k: pd.DataFrame() for k in T}
    only_ds["deck_summary"] = _ds
    check("deck-only company still gets a mail",
          "The deck on its own terms" in presentation_body("INE1", "AAA", "A", "Q1FY27", only_ds))

    check("no data renders nothing, not an empty shell",
          presentation_body("INE_NONE", "X", "X", "Q1FY27", T) == "")
    check("no ratings renders nothing", rating_body("INE_NONE", "X", "X", T) == "")

    print(f"\npf_company_mails self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    main()
