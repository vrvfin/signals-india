r"""
quarterly_table.py — shared quarterly P&L table (Sales / Net Profit / EPS / OPM% /
NPM%) with YoY + QoQ, rendered as inline-styled HTML that survives Gmail.

Pure pandas — no Drive, no network, no env — so it can be checked offline and reused
by any mail builder. Input is ONE company's `fundamentals/statements/<SYM>.parquet`
frame (long: symbol, statement, line_item, period, value).

Quarter labelling follows the Phase-1/2 RESULTS convention — the quarter that just
ENDED, i.e. period "Jun 2026" -> "Q1 FY27". Same as
  screener_scraper.current_season_key()   (Aug 2026 -> Q1FY27)
  backfill_company_docs._fy_quarter_label()
  build_gallery._qtr_label()
NOT extract_concall._current_india_quarter(), which returns the quarter in PROGRESS
(Aug 2026 -> Q2FY27) and is a different concept (guidance horizon).

Two deliberate differences from build_gallery.Cards.quarterly(), which this is
derived from:
  1. line_item ALIASES — banks/NBFCs label the top line "Revenue" and the margin
     "Financing Margin %"; the gallery matches "Sales"/"OPM %" only and silently
     renders "—" for them.
  2. YoY/QoQ are looked up BY PERIOD off the shared header, not by position in each
     row's own list — a row with a missing quarter can't silently shift the
     comparison onto the wrong quarter.
Margin rows compare in PERCENTAGE POINTS (+1.2 pp), never as a relative % change.

Public API:
    qtr_label(period)                    -> "Q1 FY27"
    q_order(label)                       -> sortable int ("Q1 FY27" -> 2701)
    latest_quarter_label(stmts_df)       -> "Q1 FY27" | None
    quarterly_rows(stmts_df, quarters=6) -> (periods, {label: {period: value}})
    quarterly_table_html(stmts_df, ...)  -> "<table>…</table>" | ""
    headline(stmts_df)                   -> {"period", "quarter", "sales", ...}
"""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

# Screener period month -> (fiscal quarter, FY bump). Verbatim from build_gallery.
_QMAP = {"mar": ("Q4", 0), "jun": ("Q1", 1), "sep": ("Q2", 1), "dec": ("Q3", 1)}

UP, DOWN, MUTED = "#1a7a3a", "#c0392b", "#bbb"

# (display label, Screener line_item aliases | None for derived, kind, decimals)
#   kind "num" -> relative % delta;  kind "pp" -> percentage-point delta
ROWS: list[tuple[str, list[str] | None, str, int]] = [
    ("Revenue",    ["Sales", "Revenue", "Sales +", "Revenue +"], "num", 0),
    ("Net Profit", ["Net Profit", "Net Profit +"],               "num", 0),
    ("EPS",        ["EPS in Rs", "EPS", "EPS in ₹"],        "num", 2),
    ("OPM %",      ["OPM %", "Financing Margin %"],              "pp",  1),
    ("NPM %",      None,                                         "pp",  1),
]


def qtr_label(period) -> str:
    """Screener period column -> fiscal quarter label. 'Jun 2026' -> 'Q1 FY27'.
    Returns the input unchanged when it doesn't parse."""
    p = str(period).strip()
    toks = p.replace("-", " ").split()
    if len(toks) >= 2:
        mon = toks[0][:3].lower()
        yr = "".join(c for c in toks[-1] if c.isdigit())
        if mon in _QMAP and yr:
            q, bump = _QMAP[mon]
            try:
                fy = (int(yr[-2:]) if len(yr) <= 2 else int(yr) % 100) + bump
                return f"{q} FY{fy % 100:02d}"
            except ValueError:
                pass
    return p


def q_order(qs) -> int:
    """Tolerant fiscal-quarter sort key (handles "Q2 FY '26", "Q1FY2026")."""
    m = re.match(r"\s*Q([1-4])\D*?(\d{2,4})", str(qs))
    if not m:
        return -1
    return (int(m.group(2)) % 100) * 100 + int(m.group(1))


def season_quarter(when: datetime | None = None) -> str:
    """Results-season quarter for a date, e.g. 'Q1FY27' in Jul-Sep 2026 — the quarter
    whose results are being ANNOUNCED (the one that just ended).

    Byte-identical output to `screener_scraper.current_season_key()` and to
    `backfill_company_docs._fy_quarter_label()` for the same date. It is reimplemented
    here rather than imported because `screener_scraper.py` is a LOCAL-only module that
    was never committed — importing it works on the dev machine and dies in CI with
    ModuleNotFoundError. Keep this in sync if the FY convention ever changes.
    """
    d = when or datetime.now()
    m, y = d.month, d.year
    if m in (4, 5, 6):
        return f"Q4FY{y % 100:02d}"          # Jan-Mar quarter, FY ending this March
    if m in (7, 8, 9):
        return f"Q1FY{(y + 1) % 100:02d}"
    if m in (10, 11, 12):
        return f"Q2FY{(y + 1) % 100:02d}"
    return f"Q3FY{y % 100:02d}"              # Jan-Mar -> Oct-Dec quarter


def norm_q(qs) -> str:
    """Canonical 'Q1FY27' form, so 'Q1 FY27' and 'Q1FY27' compare equal."""
    m = re.match(r"\s*Q([1-4])\D*?(\d{2,4})", str(qs))
    return f"Q{m.group(1)}FY{int(m.group(2)) % 100:02d}" if m else str(qs).strip()


def _quarterly(stmts_df) -> pd.DataFrame:
    """The quarterly_pl slice, or an empty frame. Annual is NEVER substituted —
    mixing an annual column into a quarter's report is exactly what this digest
    must not do (the gallery falls back to annual_pl; here that's a miss)."""
    if stmts_df is None or getattr(stmts_df, "empty", True):
        return pd.DataFrame()
    if "statement" not in stmts_df.columns or "line_item" not in stmts_df.columns:
        return pd.DataFrame()
    return stmts_df[stmts_df["statement"].astype(str) == "quarterly_pl"]


def _periods(q: pd.DataFrame) -> list[str]:
    """Ordered unique period columns, chronological (Screener writes them in order)."""
    seen: set[str] = set()
    out: list[str] = []
    for p in q["period"].astype(str):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _series(q: pd.DataFrame, keys: list[str]) -> dict[str, float]:
    """{period: value} for the first line_item alias present."""
    li = q["line_item"].astype(str).str.strip()
    for k in keys:
        sub = q[li == k]
        if not sub.empty:
            return dict(zip(sub["period"].astype(str),
                            pd.to_numeric(sub["value"], errors="coerce")))
    return {}


def quarterly_rows(stmts_df, quarters: int = 6):
    """(periods, {display_label: {period: value}}) — periods trimmed to the last
    `quarters`, values keyed by period so nothing can shift out of alignment."""
    q = _quarterly(stmts_df)
    if q.empty:
        return [], {}
    all_periods = _periods(q)
    if not all_periods:
        return [], {}

    data: dict[str, dict[str, float]] = {}
    for label, keys, _kind, _dp in ROWS:
        data[label] = _series(q, keys) if keys else {}

    # NPM % — derived, because Screener publishes OPM but not net margin
    sales, npat = data.get("Revenue", {}), data.get("Net Profit", {})
    npm: dict[str, float] = {}
    for p in all_periods:
        s, n = sales.get(p), npat.get(p)
        if s is not None and n is not None and pd.notna(s) and pd.notna(n) and s != 0:
            npm[p] = n / s * 100
    data["NPM %"] = npm

    if not any(data[k] for k in data):
        return [], {}
    return all_periods[-quarters:], data


def latest_quarter_label(stmts_df) -> str | None:
    """Fiscal label of the newest quarterly_pl column, e.g. 'Q1 FY27'.
    None when the company has no quarterly P&L at all (annual-only filers)."""
    q = _quarterly(stmts_df)
    if q.empty:
        return None
    ps = _periods(q)
    return qtr_label(ps[-1]) if ps else None


# ── HTML ─────────────────────────────────────────────────────────────────────
# Size discipline: font/colour/alignment live on the <table> (all inheritable) and
# spacing comes from the cellpadding attribute, so the great majority of <td>s carry
# NO style attribute. Repeating a full style string on every cell — which is what
# the gallery does — costs ~7 KB per company and pushes a full-portfolio mail past
# Gmail's ~102 KB clip threshold. This shape costs ~1.5 KB per company instead.
_TBL = ("border-collapse:collapse;width:100%;font:11px Arial,Helvetica,sans-serif;"
        "color:#111;text-align:right;margin:2px 0 6px 0")
_HD = "border-bottom:1px solid #ccc"
_EDGE = "border-left:1px solid #ccc"


def _fmt(v, dp: int) -> str:
    if v is None or pd.isna(v):
        return "—"
    return format(float(v), f",.{dp}f")


def _delta_cell(cur, prev, kind: str, first: bool = False) -> str:
    """Relative % for levels, percentage points for margins. Sign-flip guard: a
    negative base makes a relative % meaningless, so print ▲/▼ n/m instead."""
    edge = f"{_EDGE};" if first else ""
    if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
        return f"<td style='{edge}color:{MUTED}'>—</td>"
    cur, prev = float(cur), float(prev)
    up = cur > prev
    col = UP if up else DOWN
    if kind == "pp":
        txt = f"{cur - prev:+.1f} pp"
    elif prev > 0:
        txt = f"{(cur / prev - 1) * 100:+.0f}%"
    else:
        txt = ("▲" if up else "▼") + " n/m"
    return f"<td style='{edge}font-weight:700;color:{col}'>{txt}</td>"


def quarterly_table_html(stmts_df, quarters: int = 6) -> str:
    """Full table: one column per quarter + YoY + QoQ. "" when there is no
    quarterly data to show."""
    periods, data = quarterly_rows(stmts_df, quarters)
    if not periods:
        return ""

    cur_p = periods[-1]
    qoq_p = periods[-2] if len(periods) >= 2 else None
    # YoY = same quarter one year earlier = 4 columns back on the shared header
    yoy_p = periods[-5] if len(periods) >= 5 else None

    head = (f"<tr style='color:#666'><td style='{_HD};text-align:left'></td>"
            + "".join(f"<td style='{_HD}'>{qtr_label(p)}</td>" for p in periods)
            + f"<td style='{_HD};{_EDGE}'>YoY</td>"
            + f"<td style='{_HD}'>QoQ</td></tr>")

    body = ""
    for label, _keys, kind, dp in ROWS:
        d = data.get(label, {})
        if not d:
            continue
        cells = "".join(f"<td>{_fmt(d.get(p), dp)}</td>" for p in periods)
        yoy = _delta_cell(d.get(cur_p), d.get(yoy_p) if yoy_p else None, kind, first=True)
        qoq = _delta_cell(d.get(cur_p), d.get(qoq_p) if qoq_p else None, kind)
        body += (f"<tr><td style='text-align:left'><b>{label}</b></td>"
                 f"{cells}{yoy}{qoq}</tr>")
    if not body:
        return ""
    return (f"<table cellpadding='4' cellspacing='0' style='{_TBL}'>{head}{body}</table>")


def headline(stmts_df) -> dict:
    """Latest-quarter numbers + deltas as plain values — for subject lines, one-line
    summaries and the ledger. Empty dict when there is no quarterly data."""
    periods, data = quarterly_rows(stmts_df)
    if not periods:
        return {}
    cur_p = periods[-1]
    qoq_p = periods[-2] if len(periods) >= 2 else None
    yoy_p = periods[-5] if len(periods) >= 5 else None

    def _d(label, kind):
        d = data.get(label, {})
        cur = d.get(cur_p)
        out = {"value": None if cur is None or pd.isna(cur) else float(cur),
               "yoy": None, "qoq": None}
        for tag, p in (("yoy", yoy_p), ("qoq", qoq_p)):
            prev = d.get(p) if p else None
            if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
                continue
            cur_f, prev_f = float(cur), float(prev)
            if kind == "pp":
                out[tag] = cur_f - prev_f
            elif prev_f > 0:
                out[tag] = (cur_f / prev_f - 1) * 100
        return out

    return {
        "period":  cur_p,
        "quarter": qtr_label(cur_p),
        "revenue": _d("Revenue", "num"),
        "pat":     _d("Net Profit", "num"),
        "eps":     _d("EPS", "num"),
        "opm":     _d("OPM %", "pp"),
        "npm":     _d("NPM %", "pp"),
    }
