r"""
mail_freshness.py — "what is new today" for PF mails, the way the gallery shows it.

User 2026-10-14: "I like the current format of the mail; additionally highlight what is
the latest update for the day — similar to how it works on the daily chart render."

The gallery (build_gallery._fresh_badge) tags every item 🆕 NEW (today) / 🕓 RECENT
(last 7 days) / grey "Nd old", and heads each section with its newest item's badge.
This module gives mails the same look, plus a one-line strip at the top of each mail:

    🆕 New today (14 Oct): concall transcript · 2 filings
    — or —
    Nothing new today · latest update: results release, 12 Oct (2d ago)

DATES ARE IST. The gallery ages items against the runner's clock (UTC); several PF mails
go out after midnight IST, where that would call a same-day filing "RECENT". Same tiers,
same colours, IST calendar day.

    python scripts/mail_freshness.py --self-test
"""
from __future__ import annotations

import html as _html
import sys
from datetime import date, datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
RECENT_DAYS = 7                       # same tier as build_gallery._fresh_badge


def today_ist() -> date:
    return datetime.now(IST).date()


def _d(v) -> date | None:
    try:
        return date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def age_days(v, today: date | None = None) -> int | None:
    d = _d(v)
    return None if d is None else ((today or today_ist()) - d).days


def badge(v, today: date | None = None) -> str:
    """🆕 NEW / 🕓 RECENT / grey age chip — the gallery's exact look, IST-dated."""
    a = age_days(v, today)
    if a is None:
        return ""
    if a <= 0:
        return ('<span style="background:#1a7a3a;color:#fff;padding:1px 6px;'
                'border-radius:6px;font-size:10px;font-weight:700">🆕 NEW</span> ')
    if a <= RECENT_DAYS:
        return ('<span style="background:#1565c0;color:#fff;padding:1px 6px;'
                'border-radius:6px;font-size:10px;font-weight:700">🕓 RECENT</span> ')
    return (f'<span style="background:#9e9e9e;color:#fff;padding:1px 6px;'
            f'border-radius:6px;font-size:10px">↺ {a}d old</span> ')


def _fmt(d: date) -> str:
    return d.strftime("%d %b").lstrip("0")


def today_strip(entries: list[dict], today: date | None = None) -> str:
    """One line for the top of a mail. entries = [{"date": ..., "label": ...}].
    Lists everything dated today; otherwise names the latest update and its age."""
    today = today or today_ist()
    dated = [(e, _d(e.get("date"))) for e in entries or []]
    dated = [(e, d) for e, d in dated if d is not None and d <= today]
    if not dated:
        return ""
    new = [e["label"] for e, d in dated if d == today]
    if new:
        return (f"<div style='background:#e8f5e9;border-left:4px solid #1a7a3a;"
                f"padding:6px 10px;margin:0 0 10px;font:13px Arial,sans-serif'>"
                f"<b>🆕 New today ({_fmt(today)}):</b> "
                f"{_html.escape(' · '.join(dict.fromkeys(new)))}</div>")
    e, d = max(dated, key=lambda x: x[1])
    return (f"<div style='background:#f4f6f8;border-left:4px solid #9e9e9e;"
            f"padding:6px 10px;margin:0 0 10px;font:13px Arial,sans-serif;color:#555'>"
            f"Nothing new today · latest update: {_html.escape(str(e['label']))}, "
            f"{_fmt(d)} ({(today - d).days}d ago)</div>")


def doc_strip(label: str, filed, reached=None, today: date | None = None) -> str:
    """The top line of a single-document mail: how fresh THIS document is.
    'reached' = when the pipeline got it (a late-processed filing is still news)."""
    today = today or today_ist()
    f, r = _d(filed), _d(reached)
    if f is None:
        return ""
    when = "today" if f == today else f"{_fmt(f)} ({(today - f).days}d ago)"
    extra = (" · reached us today" if (r == today and f != today) else "")
    tone = ("#e8f5e9", "#1a7a3a") if f == today else (
        ("#e3f2fd", "#1565c0") if (today - f).days <= RECENT_DAYS else ("#f4f6f8", "#9e9e9e"))
    return (f"<div style='background:{tone[0]};border-left:4px solid {tone[1]};"
            f"padding:6px 10px;margin:0 0 10px;font:13px Arial,sans-serif'>"
            f"{badge(filed, today)}<b>{_html.escape(str(label))}</b> filed {when}{extra}"
            f"</div>")


def count_label(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _self_test() -> int:
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    t = date(2026, 10, 14)
    check("today -> NEW", "NEW" in badge("2026-10-14", t))
    check("3 days -> RECENT", "RECENT" in badge("2026-10-11", t))
    check("7 days -> RECENT (gallery tier)", "RECENT" in badge("2026-10-07", t))
    check("8 days -> grey age", "8d old" in badge("2026-10-06", t))
    check("bad date -> nothing", badge("n/a", t) == "")
    s = today_strip([{"date": "2026-10-14", "label": "concall transcript"},
                     {"date": "2026-10-14 18:30", "label": "2 filings"},
                     {"date": "2026-10-02", "label": "results"}], t)
    check("strip lists today's items", "New today (14 Oct)" in s
          and "concall transcript · 2 filings" in s and "results" not in s)
    s2 = today_strip([{"date": "2026-10-12", "label": "results release"},
                      {"date": "2026-10-02", "label": "deck"}], t)
    check("strip names the latest when nothing is new",
          "Nothing new today" in s2 and "results release, 12 Oct (2d ago)" in s2)
    check("future-dated rows ignored, empty -> no strip",
          today_strip([{"date": "2026-10-20", "label": "x"}], t) == "")
    check("labels escaped", "&lt;b&gt;" in today_strip([{"date": "2026-10-14",
                                                           "label": "<b>"}], t))
    ds = doc_strip("concall transcript", "2026-10-14", "2026-10-14", t)
    check("doc strip: filed today", "NEW" in ds and "filed today" in ds and "reached" not in ds)
    ds2 = doc_strip("annual report", "2026-10-10", "2026-10-14", t)
    check("doc strip: older filing that reached us today",
          "RECENT" in ds2 and "10 Oct (4d ago)" in ds2 and "reached us today" in ds2)
    check("doc strip: no date -> nothing", doc_strip("x", "", None, t) == "")
    # IST: 19:00 UTC on 14 Oct is already 15 Oct in India
    late = datetime(2026, 10, 14, 19, 0, tzinfo=timezone.utc).astimezone(IST).date()
    check("IST day boundary", late == date(2026, 10, 15))
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
