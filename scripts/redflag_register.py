r"""
redflag_register.py — one dated, severity-ranked risk register per company, merged from
tables this repo already writes and never displays.

Nothing here calls a model. Every row is a verbatim claim already stored by an extractor,
re-presented with its source, its period and its age. The value added is the merge and the
ordering, not new analysis.

Sources (all keyed on ISIN — never symbol; ar_red_flags.symbol carries BSE codes for some
companies, so a symbol join silently attributes one company's auditor flags to another):

  ar_red_flags          16 auditor/governance flag types + severity + page_ref   (FY grain)
  rating_concerns       agency-authored weaknesses + severity                    (dated)
  rating_sensitivity    direction=down -> downgrade tripwires; up -> upgrade      (dated)
  ratings               rating_action Downgrade/Upgrade as dated events          (dated)
  gf4_quality_flags     concall guidance-quality flags, both signs               (quarter)
  announcement_ledger   litigation / management_change / regulatory, bear+high   (dated)
  fraud_tracker         band + score + reason                                    (as_of)

Two rules the rest of the framework depends on:
  * MISSING IS NOT CLEAN. `covered()` reports which sources actually had rows, so a page
    can say "no annual report processed" instead of implying a clean audit.
  * AGE IS SHOWN, NOT HIDDEN. An FY16 working-capital flag is history. Rows carry
    `age_days` and `stale`, and the renderer fades rather than drops them.

Offline-testable: build_register() takes plain DataFrames.

    python scripts/redflag_register.py --isin INE702C01027      # live, read-only
    python scripts/redflag_register.py --self-test              # no Drive
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import date, datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd

REGISTER_COLS = ["isin", "symbol", "source", "kind", "label", "detail", "severity",
                 "direction", "period", "period_end", "grain", "ref", "age_days", "stale"]

ADVERSE, FAVOURABLE, NEUTRAL = "adverse", "favourable", "neutral"
SEV_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}

# Staleness is grain-aware, because "old" means different things per source.
#
# Dated/quarterly rows (rating actions, concall flags, announcements) genuinely decay:
# a concall flag from three years ago is history.
#
# Annual-report flags do NOT decay on the calendar — a company files one AR a year, so
# the newest available AR is the current state of the record however old it is. They are
# superseded by a NEWER annual report for the same company, not by elapsed time. Ageing
# them at a flat 730 days made FY24 flags "expire" while FY24 was still among the most
# recent reports on file, which hid live governance findings.
STALE_DAYS = 730
ANNUAL_STALE_DAYS = 365 * 4      # backstop when only ancient ARs exist at all

# gf4_quality_flags carries both signs. Values are not clamped in the extractor, so match
# loosely on lowercase substrings rather than exact equality.
GF4_NEGATIVE = ("promotional commentary", "contradictory commentary", "weak visibility",
                "margin visibility weak", "guidance ambiguous", "no near-term guidance",
                "long-term aspirational")
GF4_POSITIVE = ("strong order book support", "capacity backed")

# announcement event_types that belong on a risk register at all
ANN_RISK_EVENTS = ("litigation", "management_change", "regulatory")

# ar_red_flags severity is clamped by the extractor, but category is free-ish; these are
# the flag_types that deserve to lead a register when recent.
AR_HEADLINE = ("auditor_qualification", "emphasis_of_matter", "caro_adverse",
               "cfo_pat_divergence", "related_party_transaction", "promoter_pledge",
               "kmp_churn", "contingent_liability", "working_capital_stretch",
               "cwip_buildup")


# --------------------------------------------------------------------------- #
# period normalisation
# --------------------------------------------------------------------------- #

_FY_RE = re.compile(r"FY\s*(\d{2,4})(?:\s*[-/]\s*(\d{2,4}))?", re.I)
_Q_RE = re.compile(r"Q([1-4])\s*FY\s*(\d{2,4})", re.I)
_QEND = {1: (6, 30), 2: (9, 30), 3: (12, 31), 4: (3, 31)}


def _yy(tok: str) -> int:
    """'25' / '2025' -> 2025."""
    n = int(tok)
    return n if n >= 1000 else 2000 + n


def period_end(period) -> date | None:
    """Any period label this repo produces -> the date it ENDS on, for sorting.

    'FY25' -> 2025-03-31   'FY25-26' -> 2026-03-31   'Q1 FY27' -> 2026-06-30
    '2026-06-30' -> itself
    """
    if period is None:
        return None
    s = str(period).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None

    m = _Q_RE.search(s)
    if m:
        q, fy = int(m.group(1)), _yy(m.group(2))
        mth, day = _QEND[q]
        yr = fy - 1 if q in (1, 2, 3) else fy   # FY27 Q1 = Jun 2026; Q4 = Mar 2027
        return date(yr, mth, day)

    m = _FY_RE.search(s)
    if m:
        end = _yy(m.group(2)) if m.group(2) else _yy(m.group(1))
        return date(end, 3, 31)

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%b %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s[:10] if "-" in s or "/" in s else s, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, errors="raise").date()
    except Exception:
        return None


def _clean(v, limit: int = 400) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("nan", "none", "nat", "data_missing"):
        return ""
    return s[:limit]


def _sev(v) -> str:
    s = _clean(v).lower()
    return s if s in SEV_RANK else "medium"


# --------------------------------------------------------------------------- #
# the merge
# --------------------------------------------------------------------------- #

def _rows_for(df: pd.DataFrame, isin: str) -> pd.DataFrame:
    """ISIN filter. Never symbol — see module docstring."""
    if df is None or df.empty or "isin" not in df.columns:
        return pd.DataFrame()
    return df[df["isin"].astype(str).str.strip() == isin]


def _latest_per_agency(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the newest rating report from each agency.

    Agencies restate the same concerns and the same tripwires in every review, so the
    raw table holds one copy per report. Without this the register fills with six
    identical "sustained deterioration in profitability" rows and buries everything
    else. A newer report from the same agency supersedes its predecessor; different
    agencies are kept side by side because they genuinely disagree.
    """
    if df is None or df.empty or "rating_date" not in df.columns:
        return df if df is not None else pd.DataFrame()
    d = df.copy()
    d["_d"] = pd.to_datetime(d["rating_date"], errors="coerce")
    ag = d["agency"].astype(str).str.strip().str.upper() if "agency" in d.columns else ""
    d["_ag"] = ag
    newest = d.groupby("_ag")["_d"].transform("max")
    return d[d["_d"].eq(newest)].drop(columns=["_d", "_ag"])


def build_register(isin: str, symbol: str = "", *, ar_red_flags=None,
                   rating_concerns=None, rating_sensitivity=None, ratings=None,
                   gf4=None, announcements=None, fraud_tracker=None,
                   today: date | None = None) -> pd.DataFrame:
    """Merge every source into one register. Each argument is that table's full frame
    (or None); filtering to `isin` happens here so a caller cannot forget it."""
    today = today or date.today()
    out: list[dict] = []

    def add(source, kind, label, detail, severity, direction, period, grain, ref=""):
        pe = period_end(period)
        age = (today - pe).days if pe else None
        out.append({
            "isin": isin, "symbol": symbol, "source": source, "kind": kind,
            "label": _clean(label, 120), "detail": _clean(detail),
            "severity": severity, "direction": direction,
            "period": _clean(period, 24), "period_end": pe, "grain": grain,
            "ref": _clean(ref, 120), "age_days": age,
            "stale": bool(age is not None and age > STALE_DAYS),
        })

    # --- annual report forensic flags
    for _, r in _rows_for(ar_red_flags, isin).iterrows():
        ft = _clean(r.get("flag_type"), 60) or "other"
        add("ar_red_flags", ft, ft, r.get("evidence"), _sev(r.get("severity")),
            ADVERSE, r.get("fy_year"), "annual", r.get("page_ref"))

    # --- agency concerns (latest report per agency only)
    for _, r in _latest_per_agency(_rows_for(rating_concerns, isin)).iterrows():
        add("rating_concerns", "agency_concern", r.get("concern"), "",
            _sev(r.get("severity")), ADVERSE, r.get("rating_date"), "event",
            _clean(r.get("agency"), 20))

    # --- agency tripwires, both directions (latest report per agency only)
    for _, r in _latest_per_agency(_rows_for(rating_sensitivity, isin)).iterrows():
        d = _clean(r.get("direction")).lower()
        if d == "down":
            add("rating_sensitivity", "downgrade_trigger", "Downgrade trigger",
                r.get("trigger"), "high", ADVERSE, r.get("rating_date"), "event",
                _clean(r.get("agency"), 20))
        elif d == "up":
            add("rating_sensitivity", "upgrade_trigger", "Upgrade trigger",
                r.get("trigger"), "low", FAVOURABLE, r.get("rating_date"), "event",
                _clean(r.get("agency"), 20))

    # --- rating actions. Reaffirmed is not an event worth a register row.
    for _, r in _rows_for(ratings, isin).iterrows():
        act = _clean(r.get("rating_action"), 20).lower()
        if act not in ("downgrade", "upgrade"):
            continue
        rating = _clean(r.get("rating"), 40)
        add("ratings", f"rating_{act}", f"Rating {act}",
            f"{rating} ({_clean(r.get('outlook'), 20)})".strip(),
            "high" if act == "downgrade" else "low",
            ADVERSE if act == "downgrade" else FAVOURABLE,
            r.get("rating_date"), "event", _clean(r.get("agency"), 20))

    # --- concall guidance-quality flags
    for _, r in _rows_for(gf4, isin).iterrows():
        ft = _clean(r.get("flag_type"), 60)
        low = ft.lower()
        if any(k in low for k in GF4_POSITIVE):
            direction, sev = FAVOURABLE, "low"
        elif any(k in low for k in GF4_NEGATIVE):
            direction, sev = ADVERSE, "medium"
        else:
            direction, sev = NEUTRAL, "low"
        add("gf4_quality_flags", "concall_flag", ft, r.get("evidence"), sev,
            direction, r.get("quarter"), "quarterly")

    # --- material adverse announcements
    ann = _rows_for(announcements, isin)
    if not ann.empty and "event_type" in ann.columns:
        for _, r in ann.iterrows():
            ev = _clean(r.get("event_type"), 30).lower()
            if ev not in ANN_RISK_EVENTS:
                continue
            if _clean(r.get("direction")).lower() != "bear":
                continue
            mat = _clean(r.get("materiality")).lower()
            if mat not in ("high", "med", "medium"):
                continue
            add("announcement_ledger", ev, _clean(r.get("headline"), 120),
                r.get("summary"), "high" if mat == "high" else "medium",
                ADVERSE, r.get("ann_date"), "event")

    # --- fraud tracker band (one row; link out rather than recompute)
    ft_rows = _rows_for(fraud_tracker, isin)
    if not ft_rows.empty:
        r = ft_rows.iloc[0]
        band = _clean(r.get("band"), 12).upper()
        if band in ("RED", "ALERT", "WATCH"):
            add("fraud_tracker", "fraud_band", f"Fraud tracker: {band}",
                r.get("reason"),
                {"RED": "high", "ALERT": "high", "WATCH": "medium"}[band],
                ADVERSE, r.get("as_of"), "event",
                f"score {_clean(r.get('fraud_score'), 8)}")

    df = pd.DataFrame(out, columns=REGISTER_COLS)
    if df.empty:
        return df

    # Annual-report flags: superseded by a newer AR, not by the calendar. Only the
    # newest fiscal year on file stays live; everything behind it fades.
    ar_mask = df["source"] == "ar_red_flags"
    if ar_mask.any():
        newest = max((d for d in df.loc[ar_mask, "period_end"] if d), default=None)
        if newest:
            df.loc[ar_mask, "stale"] = df.loc[ar_mask].apply(
                lambda r: bool(r["period_end"] and r["period_end"] < newest)
                or bool(r["age_days"] and r["age_days"] > ANNUAL_STALE_DAYS), axis=1)

    df["_sev"] = df["severity"].map(lambda s: SEV_RANK.get(s, 0))
    df["_pe"] = df["period_end"].map(lambda d: d.toordinal() if d else 0)
    df = (df.sort_values(["_pe", "_sev"], ascending=[False, False])
            .drop(columns=["_sev", "_pe"]).reset_index(drop=True))
    return df


def covered(isin: str, **frames) -> dict:
    """Which sources actually had rows for this company.

    The whole point: a source with zero rows means "never processed", not "clean". A page
    that cannot say which is which is misleading, however pretty it looks.
    """
    out = {}
    for name, df in frames.items():
        if df is None or getattr(df, "empty", True) or "isin" not in df.columns:
            out[name] = 0
            continue
        out[name] = int((df["isin"].astype(str).str.strip() == isin).sum())
    return out


def split(reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(adverse, favourable) — the two columns a register renders as."""
    if reg.empty:
        return reg, reg
    return (reg[reg["direction"] == ADVERSE].reset_index(drop=True),
            reg[reg["direction"] == FAVOURABLE].reset_index(drop=True))


def headline_risks(reg: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """The rows worth putting at the top: recent, adverse, and either high severity or a
    flag_type that matters structurally."""
    if reg.empty:
        return reg
    adv = reg[(reg["direction"] == ADVERSE) & (~reg["stale"])]
    if adv.empty:
        return adv
    keep = adv[(adv["severity"] == "high") | (adv["kind"].isin(AR_HEADLINE)) |
               (adv["kind"] == "downgrade_trigger")]
    keep = keep if not keep.empty else adv
    # At most two of any one kind, so a company with four live downgrade triggers still
    # surfaces its auditor and governance flags instead of drowning them.
    return (keep.groupby("kind", sort=False).head(2).head(n).reset_index(drop=True))


# --------------------------------------------------------------------------- #
def _load_live(isin: str):
    from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                                 download_bytes)

    def folder(drive, parts):
        fid = os.environ["GDRIVE_FOLDER_ID"]
        for p in parts.split("/"):
            fid = get_or_create_subfolder(drive, fid, p)
        return fid

    drive = get_drive()
    idx = folder(drive, "company_repo/_index")

    def rd(name):
        f = find_file(drive, idx, name)
        if not f:
            return pd.DataFrame()
        return pd.read_parquet(io.BytesIO(download_bytes(drive, f)))

    return dict(
        ar_red_flags=rd("ar_red_flags.parquet"),
        rating_concerns=rd("rating_concerns.parquet"),
        rating_sensitivity=rd("rating_sensitivity.parquet"),
        ratings=rd("ratings.parquet"),
        gf4=rd("gf4_quality_flags.parquet"),
        announcements=rd("announcement_ledger.parquet"),
        fraud_tracker=rd("fraud_tracker.parquet"),
    )


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    check("FY25 ends Mar 2025", period_end("FY25") == date(2025, 3, 31))
    check("FY25-26 ends Mar 2026", period_end("FY25-26") == date(2026, 3, 31))
    check("Q1 FY27 ends Jun 2026", period_end("Q1 FY27") == date(2026, 6, 30))
    check("Q4 FY27 ends Mar 2027", period_end("Q4 FY27") == date(2027, 3, 31))
    check("ISO date passes", period_end("2026-06-30") == date(2026, 6, 30))
    check("junk -> None", period_end("DATA_MISSING") is None)

    I = "INE702C01027"
    ar = pd.DataFrame([
        {"isin": I, "fy_year": "FY24", "flag_type": "kmp_churn", "severity": "medium",
         "evidence": "CHRO resigned w.e.f. 30 Mar 2024.", "page_ref": "Note 12"},
        {"isin": I, "fy_year": "FY16", "flag_type": "working_capital_stretch",
         "severity": "high", "evidence": "old", "page_ref": ""},
        {"isin": "INE_OTHER", "fy_year": "FY24", "flag_type": "caro_adverse",
         "severity": "high", "evidence": "other company", "page_ref": ""},
    ])
    sens = pd.DataFrame([
        {"isin": I, "agency": "ICRA", "rating_date": "2026-06-30", "direction": "down",
         "trigger": "total debt/OPBDITA exceeding 0.75 times"},
        {"isin": I, "agency": "ICRA", "rating_date": "2026-06-30", "direction": "up",
         "trigger": "significant growth in operating income"},
    ])
    g4 = pd.DataFrame([
        {"isin": I, "quarter": "Q1 FY27", "flag_type": "Promotional Commentary",
         "evidence": "maintaining guidance 101%"},
        {"isin": I, "quarter": "Q1 FY27", "flag_type": "Capacity Backed",
         "evidence": "plants on track"},
    ])
    reg = build_register(I, "APLAPOLLO", ar_red_flags=ar, rating_sensitivity=sens,
                         gf4=g4, today=date(2026, 8, 8))

    check("other company excluded", not (reg["detail"] == "other company").any())
    check("all rows are this isin", (reg["isin"] == I).all())
    adv, fav = split(reg)
    check("adverse counted", len(adv) == 4)          # 2 ar + 1 downgrade + 1 gf4 neg
    check("favourable counted", len(fav) == 2)       # upgrade trigger + capacity backed
    check("FY16 superseded by FY24", bool(reg[reg["period"] == "FY16"]["stale"].iloc[0]))
    check("newest AR stays live",
          not bool(reg[reg["period"] == "FY24"]["stale"].iloc[0]))
    check("Q1 FY27 not stale", not bool(reg[reg["period"] == "Q1 FY27"]["stale"].iloc[0]))
    check("newest first", reg.iloc[0]["period_end"] >= reg.iloc[-1]["period_end"])

    hl = headline_risks(reg)
    check("headline drops stale", not hl["stale"].any())
    check("headline keeps tripwire", "downgrade_trigger" in hl["kind"].tolist())

    cov = covered(I, ar_red_flags=ar, ratings=pd.DataFrame())
    check("coverage counts rows", cov["ar_red_flags"] == 2)
    check("coverage flags empty source", cov["ratings"] == 0)

    empty = build_register("INE_NOTHING", ar_red_flags=ar, today=date(2026, 8, 8))
    check("unknown isin -> empty", empty.empty)

    print(f"\nredflag_register self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isin")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())
    if not args.isin:
        ap.print_help()
        return

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))
    frames = _load_live(args.isin)
    cov = covered(args.isin, **frames)
    print("coverage (rows per source):")
    for k, n in cov.items():
        print(f"   {k:22s} {n:5d}" + ("   <- NOT PROCESSED, not 'clean'" if n == 0 else ""))

    reg = build_register(args.isin, args.symbol, **frames)
    if reg.empty:
        print("\nregister is EMPTY — no source had a row for this ISIN.")
        return
    adv, fav = split(reg)
    print(f"\nregister: {len(reg)} rows  ({len(adv)} adverse / {len(fav)} favourable, "
          f"{int(reg['stale'].sum())} stale)")
    print("\n--- headline risks ---")
    for _, r in headline_risks(reg).iterrows():
        print(f"  [{r['period']:>9s}] {r['severity']:6s} {r['kind']:26s} "
              f"{(r['label'] or '')[:44]:44s} {(r['detail'] or '')[:60]}"
              .encode("ascii", "ignore").decode())
    print("\n--- favourable ---")
    for _, r in fav.head(6).iterrows():
        print(f"  [{r['period']:>9s}] {r['kind']:20s} {(r['detail'] or r['label'])[:80]}"
              .encode("ascii", "ignore").decode())


if __name__ == "__main__":
    main()
