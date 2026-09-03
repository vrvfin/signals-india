r"""
build_gallery.py — LOCAL one-shot chart gallery (no Streamlit, no memory cap).

Reproduces the app's Graphs "Quick Scan" gallery as ONE self-contained .html:
every ranked name as a candlestick+volume chart drawn CLIENT-SIDE (TradingView
lightweight-charts via CDN), with the same mcap / 6-rule grades / 6-quarter
table / guidance / LLM-summary cards. Writes the file locally and opens it in
your default browser. Because rendering happens in the browser, there is no
1 GB Streamlit limit to hit.

Data is pulled from the same Drive tables the app uses:
  signals/per_strategy/<strat>/latest.csv      -> ranking (n_strategies, score)
  features/latest.parquet                       -> turnover floor
  data/ohlcv/<sym>.parquet                      -> candles + volume
  fundamentals/summary.parquet                  -> market cap (screener)
  fundamentals/statements/<sym>.parquet         -> 6-quarter table
  company_repo/_index/screener_grades.parquet   -> 6-rule grade strip + name
  company_repo/_index/guidance_tracker.parquet  -> guidance (1Y + all horizons)
  company_repo/_index/gf1_guidance_statements.parquet
  company_repo/_index/announcement_ledger.parquet -> LLM summary
  universe/master_list.csv                      -> NSE/BSE exchange split

Usage:
  python scripts/build_gallery.py                 # all ranked names, open browser
  python scripts/build_gallery.py --max 200       # cap count
  python scripts/build_gallery.py --min-strats 3 --turnover 2
  python scripts/build_gallery.py --dry-run       # counts only, no html/open
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import webbrowser
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, log, load_portfolio_isins,
                             find_latest_portfolio_file, isin_symbol_map)
from drive_io import drive_call, ParquetCache
import gradation as G

# Tolerance for "how far behind the market may a series be". Mirrors
# OHLCV_MAX_STALE_DAYS in pipeline_healthcheck.py:52 — 4 calendar days is too
# tight for India's long holiday weekends. Kept as a local constant rather than
# an import so this renderer pulls in no pipeline/Drive side effects.
OHLCV_MAX_STALE_DAYS = 6

_EMPTY = pd.DataFrame()


# ─────────────────────────── Drive loaders (plain) ──────────────────────────
# Every Drive entry point here goes through drive_io.drive_call. On 2026-08-25 a
# momentary DNS failure (getaddrinfo -> ServerNotFoundError) killed a build one
# statement AFTER 45 minutes of downloads had finished, and gallery.html silently
# stayed a day stale. A transient blip now costs a short pause, not the run.
_DRIVE = None
_FOLDER_IDS: dict = {}          # path -> folder id, resolved once per process


def _drive(existing=None):
    """The Drive service these helpers actually call.

    `existing` lets an outside caller's service be adopted on first use —
    pf_results_digest.py and pf_season_status.py import _folder/_bulk_parquet and
    pass their own — so behaviour for them is unchanged while reconnects still
    work through the same global.
    """
    global _DRIVE
    if _DRIVE is None:
        _DRIVE = existing if existing is not None else get_drive()
    return _DRIVE


def _reconnect():
    """Drop a poisoned httplib2 connection and build a fresh Drive service."""
    global _DRIVE
    _DRIVE = get_drive()


def _dc(fn, label=""):
    return drive_call(fn, on_reconnect=_reconnect, label=label)


def _folder(drive, parts):
    """Resolve 'a/b/c' to a folder id. Memoised: main() asks for the same handful
    of paths up to 11 times and every hop was a live round-trip — so ~25 needless
    calls, each one its own crash site."""
    if parts in _FOLDER_IDS:
        return _FOLDER_IDS[parts]
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = _dc(lambda p=p, fid=fid: get_or_create_subfolder(_drive(drive), fid, p),
                  label=parts)
    _FOLDER_IDS[parts] = fid
    return fid


def _read_parquet(drive, folder, name):
    fid = _dc(lambda: find_file(_drive(drive), folder, name), label=name)
    if not fid:
        return _EMPTY
    raw = _dc(lambda: download_bytes(_drive(drive), fid), label=name)
    return pd.read_parquet(io.BytesIO(raw))


def _read_csv(drive, folder, name):
    fid = _dc(lambda: find_file(_drive(drive), folder, name), label=name)
    if not fid:
        return _EMPTY
    raw = _dc(lambda: download_bytes(_drive(drive), fid), label=name)
    return pd.read_csv(io.BytesIO(raw))


def _list_folder(drive, folder_id):
    """-> {name: {"id", "mtime", "size"}}.

    modifiedTime/size ride along in the SAME request (no extra round-trip) and
    are what the parquet cache keys on, so a cached copy can never outlive the
    file it came from. NOTE the value is a dict, not a bare id — the callers in
    this file (_bulk_parquet, _load_signals) read ["id"].
    """
    out, tok = {}, None
    while True:
        resp = _dc(lambda tok=tok: _drive(drive).files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,modifiedTime,size)",
            pageSize=1000, pageToken=tok).execute(), label="list")
        for f in resp.get("files", []):
            out[f["name"]] = {"id": f["id"], "mtime": f.get("modifiedTime", ""),
                              "size": f.get("size")}
        tok = resp.get("nextPageToken")
        if not tok:
            break
    return out


def _load_signals(drive):
    sig_id = _folder(drive, "signals/per_strategy")
    subs = _dc(lambda: _drive(drive).files().list(
        q=f"'{sig_id}' in parents and mimeType='application/vnd.google-apps.folder' "
          "and trashed=false", fields="files(id,name)").execute().get("files", []),
        label="signals")
    frames = []
    for s in subs:
        files = _list_folder(drive, s["id"])
        ent = files.get("latest.csv")
        if ent:
            try:
                raw = _dc(lambda e=ent: download_bytes(_drive(drive), e["id"]),
                          label=s["name"])
                df = pd.read_csv(io.BytesIO(raw))
            except pd.errors.EmptyDataError:
                continue  # zero-signal / empty latest.csv — skip, don't crash the gallery
            df["strategy_group"] = s["name"]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else _EMPTY


def _load_pf_holdings(drive):
    """DataFrame[isin, name] from the LIVE holdings file (newest across
    pf_tracking/ + portfolio/, same source as load_portfolio_isins). Scans every
    cell for an ISIN pattern so it is format-agnostic (Screener / broker exports)."""
    import re as _re
    tgt = _dc(lambda: find_latest_portfolio_file(_drive(drive),
                                                 os.environ["GDRIVE_FOLDER_ID"]),
              label="pf-holdings")
    if not tgt:
        return _EMPTY
    raw = _dc(lambda: download_bytes(_drive(drive), tgt["id"]), label="pf-holdings")
    try:
        if tgt["name"].lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw), header=None, dtype=str)
        else:
            df = pd.read_excel(io.BytesIO(raw), header=None, dtype=str)
    except Exception as e:
        log(f"  PF holdings parse failed: {str(e)[:80]}"); return _EMPTY
    isin_re = _re.compile(r"^IN[A-Z0-9]{10}$")
    rows = []
    for _, row in df.iterrows():
        cells = [("" if pd.isna(c) else str(c).strip()) for c in row.tolist()]
        isin = next((c for c in cells if isin_re.match(c)), None)
        if not isin:
            continue
        # name = first cell that looks like a company name (alpha, not the isin)
        name = next((c for c in cells
                     if c and c != isin and any(ch.isalpha() for ch in c)
                     and not isin_re.match(c)), "")
        rows.append({"isin": isin, "name": name})
    log(f"  PF holdings: {len(rows)} rows from '{tgt['name']}' "
        f"({tgt.get('_folder','?')}/)")
    return pd.DataFrame(rows).drop_duplicates("isin")


def _bulk_parquet(drive, folder_id, symbols, cache=None, what="files",
                  max_missing_pct=None, failed_out=None):
    """-> {symbol: DataFrame}   (return contract deliberately UNCHANGED)

    pf_results_digest.py and pf_season_status.py both import this, so the new
    behaviour is strictly opt-in: they keep the plain dict, get retry and better
    logging for free, and — with max_missing_pct=None — never gain an abort that
    could silence a live daily mail. build_gallery passes both extras.

    cache            ParquetCache, or None to always download.
    max_missing_pct  None = never abort (legacy). A number = raise past that
                     share of failures.
    failed_out       optional set, populated with symbols that FAILED to download.

    List the folder once, then fetch only what is needed. Two outcomes that used
    to be indistinguishable are now separated:

      absent  no file for that symbol here at all. Normal and expected — plenty
              of names legitimately have no statements. Counted, not warned.
      failed  the file EXISTS on Drive but could not be downloaded. THIS is the
              real signal, and it used to be swallowed into an empty frame: a
              network blip mid-loop produced blank charts on a build that still
              reported success. Now it retries, names the offenders, and refuses
              to write the page past a floor.
    """
    all_files = _list_folder(drive, folder_id)
    out, absent, failed = {}, [], []
    for sym in symbols:
        ent = all_files.get(f"{sym}.parquet")
        if not ent:
            out[sym] = _EMPTY
            absent.append(sym)
            continue
        df = cache.get(ent["id"], ent["mtime"]) if cache else None
        if df is not None:
            out[sym] = df
            continue
        try:
            raw = _dc(lambda e=ent: download_bytes(_drive(drive), e["id"]), label=sym)
            out[sym] = pd.read_parquet(io.BytesIO(raw))
            if cache:
                cache.put(ent["id"], ent["mtime"], raw)
        except Exception as e:
            out[sym] = _EMPTY
            failed.append((sym, type(e).__name__))
    present = len(symbols) - len(absent)
    if failed_out is not None:
        failed_out.update(s for s, _ in failed)
    log(f"  {what}: {present - len(failed)}/{len(symbols)} loaded"
        + (f", {len(absent)} not on Drive" if absent else "")
        + (f", {len(failed)} FAILED" if failed else "")
        + (f" [{cache.summary()}]" if cache else ""))
    if failed:
        shown = ", ".join(f"{s} ({e})" for s, e in failed[:25])
        more = f" +{len(failed) - 25} more" if len(failed) > 25 else ""
        log(f"  !! {what} download failures: {shown}{more}")
        pct = 100.0 * len(failed) / max(present, 1)
        if max_missing_pct is not None and pct > max_missing_pct:
            raise RuntimeError(
                f"{what}: {len(failed)}/{present} downloads failed ({pct:.1f}% > "
                f"{max_missing_pct}%). Refusing to write a gallery with silently "
                f"blank charts. Re-run (the cache makes it quick), or pass "
                f"--max-missing-pct if this level of loss is expected.")
    return out


def _last_bar_date(odf):
    """Newest date in the RAW frame.

    Must be read BEFORE any resampling: a 'W-FRI' bar is stamped with that week's
    Friday — a FUTURE date midweek — and 'ME' with month-end, so a resampled
    label would report a chart as fresher than it actually is.
    """
    if odf is None or getattr(odf, "empty", True) or "date" not in odf.columns:
        return None
    d = pd.to_datetime(odf["date"], errors="coerce").max()
    return None if pd.isna(d) else d.date()


# ─────────────────────────── card helpers (ported) ──────────────────────────
_QMAP = {"mar": ("Q4", 0), "jun": ("Q1", 1), "sep": ("Q2", 1), "dec": ("Q3", 1)}
_HORIZON_ORDER = ["NEXT_QTR", "1Y", "2Y", "3Y", "3Y+"]
_HLABEL = {"NEXT_QTR": "Nxt-Q", "1Y": "Yr", "2Y": "2Y", "3Y": "3Y", "3Y+": "3Y+"}
_GROWTH_METRICS = ("revenue", "sales", "pat", "profit", "volume", "earnings")
_BARE_PCT_GROWTH = ("revenue", "sales", "volume")
_CUR_TOK = ("inr", "₹", "crore", " cr", "cr.", "cr ", "gigawatt", " gw", " mw",
            " ton", "capex", " bn", " mn")
_KIND_TAG = {"growth": ("growth", "#1a7a3a"), "margin": ("margin", "#8e6e00"),
             "absolute": ("abs", "#5a4bb3"), "qual": ("note", "#777")}
import re


def _qtr_label(period):
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


def _g_kind(metric, value):
    m = str(metric).lower(); v = str(value).lower().strip()
    if v in ("", "na", "nan", "n/a", "-"):
        return "qual"
    if "growth" in v or "yoy" in v or "cagr" in v:
        return "growth"
    if "margin" in v or "of revenue" in v or "of sales" in v or "bps" in v:
        return "margin"
    if any(c in v for c in _CUR_TOK):
        return "absolute"
    if "%" in v or re.fullmatch(r"[\d.\-– ]+", v):
        return "growth" if any(k in m for k in _BARE_PCT_GROWTH) else "margin"
    return "qual"


def _q_order(qs):
    """Tolerant fiscal-quarter sort key (handles "Q2 FY '26", "Q1 FY2026")."""
    m = re.match(r"\s*Q([1-4])\D*?(\d{2,4})", str(qs))
    if not m:
        return -1
    return (int(m.group(2)) % 100) * 100 + int(m.group(1))


def _fresh_badge(date_str):
    """🆕 NEW (today) / 🕓 RECENT (last 7d) / grey age chip — same tiers as app.py."""
    from datetime import date as _d
    try:
        age = (_d.today() - _d.fromisoformat(str(date_str)[:10])).days
    except Exception:
        return ""
    if age <= 0:
        return ('<span style="background:#1a7a3a;color:#fff;padding:1px 6px;'
                'border-radius:6px;font-size:10px;font-weight:700">🆕 NEW</span> ')
    if age <= 7:
        return ('<span style="background:#1565c0;color:#fff;padding:1px 6px;'
                'border-radius:6px;font-size:10px;font-weight:700">🕓 RECENT</span> ')
    return (f'<span style="background:#9e9e9e;color:#fff;padding:1px 6px;'
            f'border-radius:6px;font-size:10px">↺ {age}d old</span> ')


def _research_snip(md, name, symbol="", maxlen=300):
    """Substantive company snippet ("" = drop the item) — shared extractor, see
    research_snippet.py (section-body harvest, word-bounded keys, label-row filter)."""
    from research_snippet import research_snippet
    return research_snippet(md, name, symbol, maxlen=maxlen)


class Cards:
    """Holds the loaded lookups and builds each card's HTML + chart data."""

    def __init__(self, grades, statements_map, guidance, gf1, ann, research=None,
                 news=None):
        self.grades_by = self._group(grades)
        self.stmts = statements_map
        self.guid_by = {k: g for k, g in
                        guidance.groupby(guidance["symbol"].astype(str).str.upper())} \
            if not guidance.empty and "symbol" in guidance.columns else {}
        self.gf1_by = self._group(gf1)
        self.ann_by = self._group(ann)
        self.research = research if research is not None else _EMPTY
        self.news = news or {}                     # symbol.upper() -> news text
        self.name_by = {}
        if not grades.empty and {"symbol", "company_name"} <= set(grades.columns):
            for _, r in grades.iterrows():
                self.name_by[str(r["symbol"]).upper()] = str(r.get("company_name", "") or "")

    @staticmethod
    def _group(df):
        if df is None or df.empty or "symbol" not in df.columns:
            return {}
        return {k: g for k, g in df.groupby(df["symbol"].astype(str).str.upper())}

    # mcap chip
    def mcap(self, sym, mcap_map):
        mc = mcap_map.get(sym.upper())
        if mc is None or pd.isna(mc):
            return ""
        seg = ("Largecap" if mc >= 20000 else "Midcap" if mc >= 5000
               else "Smallcap" if mc >= 500 else "Microcap")
        return (f'<span style="background:#eceff1;color:#333;padding:1px 7px;'
                f'border-radius:6px;font-size:12px;font-weight:600;margin-right:4px">'
                f'₹{mc:,.0f} Cr · {seg}</span>')

    def grades_strip(self, sym):
        g = self.grades_by.get(sym.upper(), _EMPTY)
        if g is None or g.empty:
            return ""
        r = g.iloc[-1]
        tiles = []
        tile = lambda tier, txt: (
            f'<span style="background:{G.TIER_COLOR.get(tier, "#eee")};color:#111;'
            f'padding:1px 7px;border-radius:6px;font-size:12px;font-weight:600;'
            f'margin-right:3px;display:inline-block">{txt}</span>')
        specs = [("yoy_tier", "PAT YoY", "yoy", "{:.0f}%"),
                 ("qoq_tier", "PAT QoQ", "qoq", "{:.0f}%")]
        for tcol, lbl, vcol, vfmt in specs:
            v = r.get(vcol)
            try:
                vs = vfmt.format(float(v)) if pd.notna(v) else "—"
            except (TypeError, ValueError):
                vs = "—"
            tiles.append(tile(str(r.get(tcol, "na")), f"{lbl} {vs}"))
        gm, gv, _ = self.guid_1y_headline(sym)
        if gv is None:
            tiles.append(tile("na", "Guid 1Y —"))
        else:
            tiles.append(tile(G.grade_growth(gv),
                              f"Guid·{(gm.split()[0][:4] if gm else 'Guid')} 1Y +{gv:.0f}%"))
        for tcol, lbl, vcol, vfmt in [("val_tier", "Val", "val_value", "PE {:.0f}"),
                                      ("cfo_tier", "CFO", "cfo_ratio", "{:.1f}x"),
                                      ("roe_tier", "ROE", "roe", "{:.0f}%")]:
            v = r.get(vcol)
            try:
                vs = vfmt.format(float(v)) if pd.notna(v) else "—"
            except (TypeError, ValueError):
                vs = "—"
            tiles.append(tile(str(r.get(tcol, "na")), f"{lbl} {vs}"))
        return "".join(tiles)

    def quarterly(self, sym):
        sdf = self.stmts.get(sym, _EMPTY)
        if sdf is None or sdf.empty or "statement" not in sdf.columns:
            return ""
        q = sdf[sdf["statement"] == "quarterly_pl"]
        is_annual = False
        if q.empty:                          # OMAXAUTO etc. lack quarterly_pl
            q = sdf[sdf["statement"] == "annual_pl"]
            is_annual = True
        if q.empty:
            return ""
        ser = lambda it: list(zip(q[q["line_item"] == it]["period"].astype(str),
                                  pd.to_numeric(q[q["line_item"] == it]["value"], errors="coerce")))
        rowdefs = [("Sales", "Sales"), ("Profit", "Net Profit"), ("EPS", "EPS in Rs")]
        periods = None
        for _, it in rowdefs:
            s = ser(it)
            if s:
                periods = [p for p, _ in s][-6:]
                break
        if not periods:
            return ""

        def pct(cur, prev):
            if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
                return "<td style='font-size:11px;text-align:right;color:#bbb'>—</td>"
            up = cur > prev
            col = "#1a7a3a" if up else "#c0392b"
            txt = f"{(cur / prev - 1) * 100:+.0f}%" if prev > 0 else ("▲" if up else "▼") + " n/m"
            return (f"<td style='font-size:11px;text-align:right;font-weight:700;"
                    f"color:{col}'>{txt}</td>")

        _fy = lambda p: (f"FY{''.join(c for c in str(p) if c.isdigit())[-2:]}"
                         if any(c.isdigit() for c in str(p)) else str(p))
        plabel = _fy if is_annual else _qtr_label
        corner = "<span style='color:#8a6d00;font-size:9px'>annual</span>" if is_annual else ""
        th = (f"<tr><td style='font-size:11px'>{corner}</td>"
              + "".join(f"<td style='font-size:11px;text-align:right;color:#666;"
                        f"font-weight:600'>{plabel(p)}</td>" for p in periods)
              + "<td style='font-size:11px;text-align:right;color:#666;font-weight:600;"
                "border-left:1px solid #ccc'>YoY</td>"
              + "<td style='font-size:11px;text-align:right;color:#666;font-weight:600'>QoQ</td></tr>")
        body = ""
        for lbl, it in rowdefs:
            vals = ser(it); d = dict(vals)
            cells = "".join(
                "<td style='font-size:11px;text-align:right;font-weight:600;color:#111'>"
                + ("—" if d.get(p) is None or pd.isna(d.get(p)) else format(d.get(p), ",.0f"))
                + "</td>" for p in periods)
            seq = [v for _, v in vals]
            cur = seq[-1] if seq else None
            if is_annual:
                yoy = pct(cur, seq[-2] if len(seq) >= 2 else None).replace(
                    "text-align:right;", "text-align:right;border-left:1px solid #ccc;")
                qoq = "<td style='font-size:11px;text-align:right;color:#bbb'>—</td>"
            else:
                yoy = pct(cur, seq[-5] if len(seq) >= 5 else None).replace(
                    "text-align:right;", "text-align:right;border-left:1px solid #ccc;")
                qoq = pct(cur, seq[-2] if len(seq) >= 2 else None)
            body += (f"<tr><td style='font-size:11px;color:#333'><b>{lbl}</b></td>"
                     f"{cells}{yoy}{qoq}</tr>")
        return f"<table style='border-collapse:collapse;width:100%;margin:2px 0 4px 0'>{th}{body}</table>"

    # guidance ------------------------------------------------------------
    def _guid_latest(self, sym):
        g = self.guid_by.get(sym.upper())
        if g is None or g.empty or "metric" not in g.columns:
            return "", None
        g = g.copy()
        g["_qo"] = g["quarter"].map(_q_order) if "quarter" in g.columns else -1
        if g["_qo"].max() >= 0:
            g = g[g["_qo"] == g["_qo"].max()]
        qlab = str(g["quarter"].iloc[0]) if "quarter" in g.columns and len(g) else ""
        return qlab, g.assign(_c=pd.to_numeric(g.get("cagr_pct"), errors="coerce"))

    def guid_1y(self, sym):
        qlab, g = self._guid_latest(sym)
        if g is None:
            return "", []
        kinds = [_g_kind(m, v) for m, v in zip(g["metric"], g.get("value", ""))]
        g = g.assign(_k=kinds)
        g1 = g[g["horizon_fy"].astype(str).str.upper().eq("1Y")
               & g["guidance_type"].astype(str).str.lower().str.contains("explicit")
               & g["_k"].eq("growth")].dropna(subset=["_c"])
        out = [(m, float(s["_c"].median())) for m, s in g1.groupby(g1["metric"].astype(str).str.title())]
        return qlab, sorted(out, key=lambda x: -x[1])

    def guid_1y_headline(self, sym):
        qlab, mets = self.guid_1y(sym)
        if not mets:
            return None, None, qlab
        for m, v in mets:
            if any(k in m.lower() for k in _GROWTH_METRICS):
                return m, v, qlab
        return mets[0][0], mets[0][1], qlab

    def growth_blob(self, sym):
        g = self.grades_by.get(sym.upper(), _EMPTY)
        if g is None or g.empty:
            return ""
        r = g.iloc[-1]
        sdf = self.stmts.get(sym, _EMPTY)
        qlab = ""
        if sdf is not None and not sdf.empty and "statement" in sdf.columns:
            qd = sdf[sdf["statement"] == "quarterly_pl"]
            if not qd.empty and "period" in qd.columns:
                per = list(qd["period"].astype(str))
                qlab = _qtr_label(per[-1]) if per else ""
        hot = []
        for key, lbl in (("yoy", "PAT YoY"), ("qoq", "PAT QoQ")):
            v = pd.to_numeric(r.get(key), errors="coerce")
            if pd.notna(v) and v > 30:
                hot.append(f"{lbl} +{v:.0f}%")
        for m, c in self.guid_1y(sym)[1]:
            if c > 30 and any(k in m.lower() for k in _GROWTH_METRICS):
                hot.append(f"Guid·{m} +{c:.0f}% (1Y growth)")
        if not hot:
            return ""
        qtag = f'<span style="opacity:.85">{qlab} · </span>' if qlab else ""
        return (f'<div style="background:#1a7a3a;color:#fff;padding:4px 10px;'
                f'border-radius:6px;font-size:13.5px;font-weight:700;margin:3px 0">'
                f'🚀 {qtag}{" · ".join(hot)}</div>')

    def guidance_panel(self, sym):
        lines = []
        qlab, g = self._guid_latest(sym)
        src_q = qlab
        if g is not None and not g.empty:
            for m, sub in g.groupby(g["metric"].astype(str).str.title()):
                if not m or m.lower() == "nan":
                    continue
                ks = [_g_kind(m, vv) for vv in sub.get("value", [])]
                kind = max(set(ks), key=ks.count) if ks else "qual"
                klab, kcol = _KIND_TAG.get(kind, ("", "#777"))
                bits, seen = [], set()
                hz_up = sub["horizon_fy"].astype(str).str.upper()
                for h in _HORIZON_ORDER + ["__other__"]:
                    if h == "__other__":          # FY27 / FY28 etc. — show them too
                        hs = sub[~hz_up.isin(_HORIZON_ORDER)]
                    else:
                        hs = sub[hz_up == h]
                    for _, hr in hs.iterrows():
                        t = str(hr.get("value", "")).strip()
                        if not t or t.lower() in ("na", "nan", "n/a", "-") or t in seen:
                            continue
                        seen.add(t)
                        disp = t + "%" if kind in ("growth", "margin") and re.fullmatch(r"[\d.\-– ]+", t) else t
                        disp = disp[:34] + ("…" if len(disp) > 34 else "")
                        tag = _HLABEL.get(h, "") if h != "__other__" else \
                            str(hr.get("horizon_fy", "")).strip()
                        bits.append((f'<span style="color:#888">{tag}</span> ' if tag else "")
                                    + f'<b style="color:#0d2f5c">{disp}</b>')
                        break
                if bits:
                    chip = (f'<span style="background:{kcol};color:#fff;border-radius:4px;'
                            f'padding:0 5px;font-size:10px;margin-left:4px">{klab}</span>')
                    lines.append(f'<b style="color:#1565c0">{m}</b>{chip} ' + " · ".join(bits[:4]))
                if len(lines) >= 7:
                    break
        g1 = self.gf1_by.get(sym.upper(), _EMPTY)
        gf1_latest_q = ""
        if g1 is not None and not g1.empty and "exact_statement" in g1.columns:
            g2 = g1.copy()
            # sort by QUARTER (latest first), not processed_at (ties on a single
            # backfill run hid the newest quarter, e.g. CPPLUS Q4 vs Q3).
            g2["_qo"] = g2["quarter"].map(_q_order) if "quarter" in g2.columns else -1
            sort_cols = ["_qo"] + (["processed_at"] if "processed_at" in g2.columns else [])
            g2 = g2.sort_values(sort_cols, ascending=False)
            if (g2["_qo"] >= 0).any():
                gf1_latest_q = str(g2[g2["_qo"] >= 0]["quarter"].iloc[0])
            shown = 0
            for _, gr in g2.iterrows():
                stmt = str(gr.get("exact_statement", "") or "").strip()
                if not stmt or stmt.lower() == "nan":
                    continue
                qv = str(gr.get("quarter", "") or "").strip()
                mt = str(gr.get("metric_type", "") or "").strip()
                tf = str(gr.get("timeframe", "") or "").strip()
                tag = " · ".join(t for t in (qv, mt, tf) if t and t.lower() != "nan")
                lines.append((f'<i style="color:#1565c0">[{tag}]</i> ' if tag else "") + stmt[:200])
                shown += 1
                if shown >= 3:
                    break
        if not lines:
            return ""
        hdr_q = max([q for q in (src_q, gf1_latest_q) if q], key=_q_order, default=src_q)
        src = (f'<span style="background:#1565c0;color:#fff;border-radius:4px;'
               f'padding:0 6px;font-size:10px;margin-left:6px">concall {hdr_q}</span>'
               if hdr_q else "")
        return (f'<div style="background:#eef6ff;border-left:3px solid #1565c0;'
                f'padding:6px 10px;font-size:12.5px;color:#222;margin:3px 0;line-height:1.55">'
                f'📋 <b style="font-size:13px">Guidance / outlook</b>{src}<br>'
                + "<br>".join(lines[:7]) + "</div>")

    def llm_summary(self, sym, days=90, max_items=6):
        """Exchange-filing panel: EVERY recent announcement (not just the last),
        ordered by materiality then recency, with an actionable roll-up header —
        event-type mix, bull/bear direction tally, and the ₹/% figures pulled out
        of the summaries so growth / guidance / risk numbers are scannable."""
        a = self.ann_by.get(sym.upper(), _EMPTY)
        if a is None or a.empty or "summary" not in a.columns:
            return ""
        d = a.copy()
        if "ann_date" in d.columns:
            d["_dt"] = pd.to_datetime(d["ann_date"], errors="coerce")
            cut = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
            recent = d[d["_dt"] >= cut]
            d = recent if not recent.empty else d          # never render empty
            d = d.sort_values("_dt", ascending=False)
        d = d[d["summary"].astype(str).str.strip().str.lower().ne("nan")]
        d = d[d["summary"].astype(str).str.strip() != ""]
        if d.empty:
            return ""

        # ---- roll-up header: what happened, how material, which way ----
        n = len(d)
        ev = (d["event_type"].dropna().astype(str).value_counts().to_dict()
              if "event_type" in d.columns else {})
        ev_txt = " · ".join(f"{k.replace('_', ' ')} ×{v}" if v > 1 else k.replace("_", " ")
                            for k, v in list(ev.items())[:4])
        bulls = int((d.get("direction") == "bull").sum()) if "direction" in d else 0
        bears = int((d.get("direction") == "bear").sum()) if "direction" in d else 0
        highs = int((d.get("materiality") == "high").sum()) if "materiality" in d else 0
        chips = []
        if highs:
            chips.append(f'<span style="background:#c0392b;color:#fff;padding:0 6px;'
                         f'border-radius:4px;font-weight:700">{highs} high-impact</span>')
        if bulls:
            chips.append(f'<span style="background:#1a7a3a;color:#fff;padding:0 6px;'
                         f'border-radius:4px;font-weight:700">▲ {bulls} bullish</span>')
        if bears:
            chips.append(f'<span style="background:#c0392b;color:#fff;padding:0 6px;'
                         f'border-radius:4px;font-weight:700">▼ {bears} bearish</span>')
        # numbers across ALL filings in the window (growth / guidance / risk figures)
        nums = _extract_numbers(" ".join(d["summary"].astype(str).tolist()))
        if nums:
            chips.append('<span style="background:#0b5394;color:#fff;padding:0 6px;'
                         'border-radius:4px;font-weight:700">' +
                         " · ".join(nums[:6]) + "</span>")

        # ---- per-filing lines, most material first ----
        mrank = {"high": 0, "med": 1, "low": 2}
        if "materiality" in d.columns:
            d = d.assign(_m=d["materiality"].map(lambda x: mrank.get(str(x), 3))) \
                 .sort_values(["_m", "_dt"], ascending=[True, False])
        lines = []
        for _, r in d.head(max_items).iterrows():
            adate = str(r.get("ann_date", ""))[:10]
            mat = str(r.get("materiality", "") or "")
            dirn = str(r.get("direction", "") or "")
            dot = {"bull": "🟢", "bear": "🔴"}.get(dirn, "⚪")
            mcol = {"high": "#c0392b", "med": "#a66300"}.get(mat, "#7f8c8d")
            s = _highlight_numbers(str(r.get("summary", "")).strip()[:320])
            lines.append(
                f'<div style="margin:3px 0">{dot} {_fresh_badge(adate)}'
                f'<b style="color:{mcol}">{adate}'
                f' · {str(r.get("event_type", r.get("category", "")) or "").replace("_", " ")}'
                f'{" · " + mat.upper() if mat else ""}:</b> {s}'
                f'{_ann_pdf_link(r.get("attachment"))}</div>')
        more = f' <span style="color:#888">(+{n - max_items} more)</span>' if n > max_items else ""

        newest_ann = str(d["ann_date"].max())[:10] if "ann_date" in d.columns else ""
        return (f'<div style="background:#fffde7;border-left:3px solid #f9a825;'
                f'padding:6px 10px;font-size:12.5px;color:#222;margin:3px 0;line-height:1.55">'
                f'{_fresh_badge(newest_ann)}'
                f'🧠 <b style="font-size:13px;color:#8a6d00">Exchange filings</b> '
                f'<span style="color:#666">— {n} in {days}d{more}</span>'
                + (f'<div style="margin:3px 0">{" ".join(chips)}</div>' if chips else "")
                + (f'<div style="color:#555;font-size:11.5px;margin-bottom:2px">{ev_txt}</div>'
                   if ev_txt else "")
                + "".join(lines) + "</div>")

    def news_card(self, sym, name="", max_items=5):
        """Recent company news headlines (Google News RSS, reputable sources only)
        pre-fetched into self.news by _fetch_news_map, RANKED BY RELEVANCE to this
        company: direct company news first, generic 'N stocks to buy' roundups
        demoted and marked. Numbers in headlines are highlighted."""
        txt = (self.news or {}).get(sym.upper(), "")
        if not txt or txt.startswith("DATA_MISSING") or txt.startswith("No recent news"):
            return ""
        nm = name or self.name_by.get(sym.upper(), "")
        items = []
        for ln in str(txt).splitlines():
            ln = ln.strip().lstrip("- ").strip()
            if not ln or "|" not in ln:
                continue
            date_part, rest = ln.split("|", 1)
            head = rest.strip()
            items.append((_news_relevance(head, nm, sym), date_part.strip()[:10], head))
        if not items:
            return ""
        # relevance first, then recency — nothing dropped, only ordered.
        # Python's sort is stable, so sorting by date first then by score keeps
        # the newest item on top within each relevance tier.
        items.sort(key=lambda x: x[1], reverse=True)      # date desc
        items.sort(key=lambda x: x[0], reverse=True)      # score desc (stable)
        n_direct = sum(1 for s, _, _ in items if s >= _NEWS_DIRECT_MIN)
        lines = []
        for score, dt_s, head in items[:max_items]:
            tag = ("" if score >= _NEWS_DIRECT_MIN else
                   ' <span style="background:#b0bec5;color:#fff;padding:0 5px;'
                   'border-radius:4px;font-size:10px">unverified</span>')
            lines.append(f'<div style="margin:2px 0">{_fresh_badge(dt_s)}'
                         f'<b style="color:#00695c">{dt_s}</b>'
                         f' · {_highlight_numbers(head)}{tag}</div>')
        newest = max((d for _, d, _ in items), default="")
        cnt = (f'<span style="color:#666"> — {n_direct} direct'
               f'{f", {len(items) - n_direct} unverified" if len(items) > n_direct else ""}'
               f'</span>')
        return (f'<div style="background:#e0f2f1;border-left:3px solid #00897b;'
                f'padding:6px 10px;font-size:12.5px;color:#222;margin:3px 0;line-height:1.55">'
                f'{_fresh_badge(newest)}📰 <b style="font-size:13px;color:#00695c">News</b>{cnt}'
                + "".join(lines) + "</div>")

    def research_card(self, sym, name, isin="", days=45, max_items=3):
        """Recent research (research_index) mentioning this company — same matching
        as daily_brief.company_research (isins/companies blob; no symbol column)."""
        df = self.research
        if df is None or df.empty:
            return ""
        il = str(isin or "").lower()
        sl, nl = str(sym or "").lower(), str(name or "").lower()

        def hit(r):
            blob = f"{r.get('isins','')}{r.get('companies','')}".lower()
            return ((il and il in blob) or (len(sl) > 2 and sl in blob)
                    or (len(nl) > 3 and nl in blob))
        sub = df[df.apply(hit, axis=1)]
        if sub.empty:
            return ""
        if "doc_date" in sub.columns:
            from datetime import date, timedelta
            cut = (date.today() - timedelta(days=days)).isoformat()
            sub = sub[sub["doc_date"].astype(str) >= cut].sort_values(
                "doc_date", ascending=False)
        lines = []
        for _, r in sub.head(max_items).iterrows():
            snip = _research_snip(r.get("summary_md"), name, sym)
            if not snip:
                continue
            # Context so the snippet isn't a naked number: WHAT kind of document
            # this is, and the theme/sector it was filed under. Without these the
            # line reads as figures with no provenance.
            dtype = str(r.get("doc_type", "") or "").replace("_", " ").title()
            theme = str(r.get("themes", "") or "").strip()
            theme = theme.split(",")[0].strip()[:40] if theme and theme.lower() != "nan" else ""
            ddate = str(r.get("doc_date", ""))[:10]
            tags = " · ".join(x for x in [dtype, theme] if x)
            lines.append(
                f'<div style="margin:3px 0">{_fresh_badge(ddate)}'
                f'<b style="color:#4a148c">{ddate} · {str(r.get("source", ""))[:40]}</b>'
                + (f' <span style="background:#ede7f6;color:#4a148c;padding:0 5px;'
                   f'border-radius:4px;font-size:10.5px">{tags}</span>' if tags else "")
                + f'<br>{_highlight_numbers(snip)}</div>')
        if not lines:
            return ""
        newest = str(sub["doc_date"].iloc[0])[:10] if "doc_date" in sub.columns else ""
        return (f'<div style="background:#f3e5f5;border-left:3px solid #7b1fa2;'
                f'padding:6px 10px;font-size:12.5px;color:#222;margin:3px 0;line-height:1.55">'
                f'{_fresh_badge(newest)}📄 <b style="font-size:13px">Research</b>'
                f'<span style="color:#666"> — {len(lines)} note'
                f'{"s" if len(lines) != 1 else ""} in {days}d</span>'
                + "".join(lines) + "</div>")


# NOTE: alternation is first-match-wins, so the LONGER unit spellings must come
# first (crores|crore|cr), else "Rs 450 crore" truncates to "Rs 450 cr" + "ore".
_UNITS = r"(?:crores|crore|cr|billion|bn|million|mn|lakhs|lakh)"
_NUM_PAT = re.compile(
    rf"(?:(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d+)?\s?{_UNITS}?"
    rf"|[\d,]+(?:\.\d+)?\s?{_UNITS}\b"
    r"|[+-]?\d+(?:\.\d+)?\s?%)", re.I)


# BSE serves filing PDFs from these hosts (same pair ingest_announcements uses).
# AttachLive holds recent filings; AttachHis the archive. We link Live — for an
# older filing BSE redirects/404s, which is why this is a "verify" link, not a
# promise the PDF is still hot.
_BSE_ATTACH = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/"


def _ann_pdf_link(attachment) -> str:
    """One-word link to the original exchange filing PDF, so any summary can be
    cross-checked against the source document."""
    a = str(attachment or "").strip()
    if not a or a.lower() in ("nan", "none"):
        return ""
    return (f' <a href="{_BSE_ATTACH}{a}" target="_blank" rel="noopener" '
            f'style="color:#0b5394;font-weight:700;text-decoration:none;'
            f'font-size:11px">[PDF]</a>')


def _extract_numbers(text: str, limit: int = 12) -> list:
    """Distinct ₹-amounts and percentages mentioned across filing summaries —
    the growth / guidance / order-size / margin figures worth scanning."""
    out, seen = [], set()
    for m in _NUM_PAT.findall(str(text)):
        v = " ".join(m.split())
        k = v.lower().replace(" ", "")
        if k in seen or k in {"0%", "100%"}:
            continue
        seen.add(k)
        out.append(v)
        if len(out) >= limit:
            break
    return out


def _highlight_numbers(text: str) -> str:
    """Bold the ₹/% figures inside a filing summary so numbers pop out."""
    return _NUM_PAT.sub(
        lambda m: f'<b style="color:#0b5394">{m.group(0)}</b>', str(text))


# Listicle / roundup patterns: the headline mentions the stock only as one of
# many (Google News matches these on the ticker), so they are demoted — not
# dropped, since a portfolio-action roundup can still be worth a glance.
# A headline counts as DIRECT company news at this score or above (see
# _news_relevance): full-name match (+4) or name+ticker clears it; a lone
# first-word hit (+1, e.g. "Rishab Pant...") or a bare fragment does not.
_NEWS_DIRECT_MIN = 3

# Card ranking tiebreak ladder, applied after n_strategies: shorter horizons
# first, so recent momentum decides between names with equal strategy counts.
_RANK_RETURNS = ("1m", "3m", "6m", "12m")

_LISTICLE_PAT = re.compile(
    r"(\b\d+\s+(?:stocks?|shares?|picks?|multibagger|smallcap|midcap|largecap)"
    r"|\bstocks? to (?:buy|watch|sell)|\btop \d+|\bportfolio\b|\bbuy or sell\b"
    r"|\bmultibagger|\bbrokerage (?:radar|calls)|\bmarket (?:wrap|roundup)"
    r"|\bnifty (?:today|outlook)|\bsensex\b|\bf&o (?:ban|cues)|\bstocks in news)", re.I)


def _news_relevance(headline: str, name: str, symbol: str) -> int:
    """Rank a headline's relevance to THIS company (higher = more relevant).
      +4 the FULL distinctive company name appears ("rishab instruments")
      +1 only the first word matches ("Rishab ..." — could be a person's name,
         so this alone is NOT treated as direct company news)
      +1 ticker appears as a standalone word
      -3 listicle/roundup phrasing (the stock is one of many)
      -2 headline too short to say anything (bare "Cummins India" fragments)
    A headline is DIRECT only at >=3, so a lone first-word hit stays demoted.
    Nothing is dropped on score alone — the caller sorts by score then recency."""
    h = str(headline).lower()
    score = 0
    nm = str(name or "").lower().strip()
    core_words = []
    if nm:
        # strip corporate boilerplate so "Rishab Instruments Limited" -> "rishab instruments"
        core = re.sub(r"\b(ltd|limited|india|indian|industries|enterprises|corp|"
                      r"corporation|company|holdings?|group|technologies|"
                      r"international|and|&|the)\b", " ", nm)
        core_words = [w for w in core.split() if len(w) > 2]
        core = " ".join(core_words)
        if core and core in h:
            score += 4                     # full distinctive name — unambiguous
        elif len(core_words) >= 2 and all(
                re.search(rf"\b{re.escape(w)}", h) for w in core_words[:2]):
            score += 4                     # both distinctive tokens present
        elif core_words and re.search(rf"\b{re.escape(core_words[0])}\b", h):
            score += 1                     # first word only — weak (person/brand)
    sym = str(symbol or "").lower().strip()
    if len(sym) > 2 and re.search(rf"\b{re.escape(sym)}\b", h):
        score += 1
    if _LISTICLE_PAT.search(h):
        score -= 3
    # A headline needs enough words to carry an explanation; bare name fragments
    # ("Cummins India", "Cummins India share price") say nothing actionable.
    if len(re.findall(r"[a-z0-9]+", h)) < 5:
        score -= 2
    return score


def _fetch_news_map(syms, name_by, days=30, workers=8):
    """symbol.upper() -> news text, fetched in parallel via alt_sources.news_block
    (Google News RSS, reputable-source whitelist — same feed daily_brief uses).
    Network-bound and best-effort: any failure yields an empty entry, never raises."""
    try:
        from alt_sources import news_block
    except Exception as e:
        log(f"  news: alt_sources unavailable ({type(e).__name__}) — skipped")
        return {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = {}

    def one(s):
        try:
            return s, news_block(name_by.get(s.upper(), s), s, limit=8, days=days)
        except Exception:
            return s, ""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for f in as_completed([pool.submit(one, s) for s in syms]):
            s, txt = f.result()
            out[s.upper()] = txt
    hit = sum(1 for v in out.values()
              if v and not v.startswith(("DATA_MISSING", "No recent news")))
    log(f"  news: {hit}/{len(syms)} names with headlines ({days}d window)")
    return out


def _resample_ohlc(d, rule):
    """Resample daily OHLCV to weekly ('W-FRI') / monthly ('ME') bars:
    open=first, high=max, low=min, close=last, volume=sum."""
    x = d.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).set_index("date").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    agg = (x.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"})
            .dropna(subset=["close"]).reset_index())
    return agg


def _ohlc_arrays(odf, days, resample=None):
    """-> (candles, volumes, ema20, ema50) for lightweight-charts. EMAs computed
    on the (optionally resampled) close series, so weekly/monthly views carry
    weekly/monthly EMAs. resample: None=daily, 'W-FRI'=weekly, 'ME'=monthly."""
    if odf is None or odf.empty:
        return [], [], [], []
    if resample:
        d = _resample_ohlc(odf, resample)
        nbars = 200 if str(resample).startswith("W") else 120   # ~4y wk / ~10y mo
        d = d.tail(nbars).reset_index(drop=True)
    else:
        d = odf.sort_values("date").tail(days).reset_index(drop=True)
    if d.empty:
        return [], [], [], []
    closes = pd.to_numeric(d["close"], errors="coerce")
    e20 = closes.ewm(span=20, adjust=False).mean()
    e50 = closes.ewm(span=50, adjust=False).mean()
    candles, vols, ema20, ema50 = [], [], [], []
    for i, r in d.iterrows():
        try:
            o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        except (TypeError, ValueError):
            continue
        t = str(r["date"])[:10]
        candles.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        v = pd.to_numeric(r.get("volume"), errors="coerce")
        vols.append({"time": t, "value": float(v) if pd.notna(v) else 0,
                     "color": "#26a69a" if c >= o else "#ef5350"})
        if pd.notna(e20.iloc[i]):
            ema20.append({"time": t, "value": round(float(e20.iloc[i]), 2)})
        if pd.notna(e50.iloc[i]):
            ema50.append({"time": t, "value": round(float(e50.iloc[i]), 2)})
    return candles, vols, ema20, ema50


_TPL = """<!doctype html><html><head><meta charset="utf-8">
<title>Signals gallery __DATE__</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f4f6f9;margin:0;padding:12px}
 h1{font-size:16px;margin:6px 8px}
 .grid{display:grid;grid-template-columns:1fr;gap:16px;max-width:1100px;margin:0 auto}
 .card{background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:8px 10px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
 .hd{font-size:15px;font-weight:800;color:#1a3d6e;border-top:3px solid #1a3d6e;padding-top:4px;margin-bottom:3px}
 .hd .nm{font-size:12px;font-weight:500;color:#666}
 .row{margin:2px 0}
 .chart{height:440px;margin-top:6px}
 table{border-collapse:collapse;width:100%}
 .fresh{font-size:12px;margin:-2px 8px 8px;padding:5px 9px;border-radius:5px;display:inline-block}
 .fresh-ok{background:#e8f5e9;color:#1b5e20;border:1px solid #c3e6c8}
 .fresh-warn{background:#fff4e0;color:#8a5300;border:1px solid #f2d9a8}
 .fresh-bad{background:#fdecea;color:#96231a;border:1px solid #f5c6c0}
 .stale{font-size:10.5px;font-weight:600;color:#8a5300;background:#fff4e0;border:1px solid #f2d9a8;border-radius:4px;padding:1px 5px;margin-left:6px;white-space:nowrap}
 .stale-bad{font-size:10.5px;font-weight:600;color:#96231a;background:#fdecea;border:1px solid #f5c6c0;border-radius:4px;padding:1px 5px;margin-left:6px;white-space:nowrap}
 .dlfail{font-size:11px;color:#96231a;background:#fdecea;border:1px solid #f5c6c0;border-radius:4px;padding:5px 7px;margin:4px 0}
</style></head><body>
<h1>__TITLE__ — __N__ charts — __DATE__ (rendered in your browser)</h1>
__FRESH__
<div style="font-size:11px;color:#888;margin:-2px 8px 10px">Candlesticks + volume · <b style="color:#2962FF">EMA20</b> · <b style="color:#FF6D00">EMA50</b></div>
__PRELUDE__<div class="grid">__CARDS__</div>
<script>
const D=__PAYLOAD__;
function mk(id){
 const el=document.getElementById('ch'+id); if(!el||el.dataset.done)return; el.dataset.done=1;
 const ch=LightweightCharts.createChart(el,{height:440,layout:{textColor:'#333',background:{color:'#fff'}},
   rightPriceScale:{borderColor:'#eee'},timeScale:{borderColor:'#eee'},grid:{horzLines:{color:'#f2f2f2'},vertLines:{color:'#f7f7f7'}}});
 const cs=ch.addCandlestickSeries(); cs.setData((D[id]||{}).c||[]);
 const l20=ch.addLineSeries({color:'#2962FF',lineWidth:2,priceLineVisible:false,lastValueVisible:false});
 l20.setData((D[id]||{}).e20||[]);
 const l50=ch.addLineSeries({color:'#FF6D00',lineWidth:2,priceLineVisible:false,lastValueVisible:false});
 l50.setData((D[id]||{}).e50||[]);
 const vs=ch.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:''});
 vs.priceScale().applyOptions({scaleMargins:{top:0.82,bottom:0}}); vs.setData((D[id]||{}).v||[]);
 ch.timeScale().fitContent();
}
const io=new IntersectionObserver((es)=>{es.forEach(e=>{if(e.isIntersecting)mk(e.target.id.slice(2));});},{rootMargin:'300px'});
document.querySelectorAll('.chart').forEach(el=>io.observe(el));
</script></body></html>"""


def _stale_chip(d, mkt_date):
    """Per-card marker, rendered ONLY when this series is behind the market. Fresh
    cards get nothing — 700 identical badges would just become wallpaper, so mark
    the exception, not the rule."""
    if not d or not mkt_date or d >= mkt_date:
        return ""
    n = (mkt_date - d).days
    cls = "stale-bad" if n > OHLCV_MAX_STALE_DAYS else "stale"
    return (f'<span class="{cls}">&#9888; data to {d.strftime("%d %b %Y")} '
            f'({n}d behind)</span>')


def _freshness_banner(mkt_date, gen_date, n_behind, n_charts):
    """Page-level statement of what the page was built from, so a stale gallery
    announces itself instead of being noticed days later by accident.

    Both dates carry their weekday, because that is what makes the gap readable
    at a glance: Sat-generated/Fri-data is nothing, Wed-generated/Fri-data is a
    stalled feed, and the day names say which is which without any arithmetic.
    """
    if not mkt_date:
        return ('<div class="fresh fresh-bad">&#9888; No price data on this page '
                '&mdash; latest data point unknown.</div>')
    lag = (gen_date - mkt_date).days
    when = mkt_date.strftime("%a %d %b %Y")
    gen = gen_date.strftime("%a %d %b %Y")
    plural = "day" if lag == 1 else "days"
    if lag <= 0:
        cls = "fresh-ok"
        msg = (f"&#10003; Generated {gen} &middot; latest data point {when} "
               f"&mdash; up to date.")
    elif lag > OHLCV_MAX_STALE_DAYS:
        cls = "fresh-bad"
        msg = (f"&#9888; Generated {gen} &middot; latest data point {when} "
               f"&mdash; {lag} {plural} behind. The price feed looks stalled.")
    else:
        # Name the ordinary explanations so a normal gap doesn't read as a fault.
        why = (" (today's close may not be ingested yet)" if lag == 1
               else " (weekend / holiday gap is normal)" if lag <= 4 else "")
        cls = "fresh-warn"
        msg = (f"&#9888; Generated {gen} &middot; latest data point {when} "
               f"&mdash; {lag} {plural} behind{why}.")
    tail = (f" &middot; {n_behind} of {n_charts} charts are behind {when}"
            if n_behind else f" &middot; all {n_charts} charts current to {when}")
    return f'<div class="fresh {cls}">{msg}{tail}</div>'


def build_html(ranked, omap, cards: Cards, mcap_map, days, title="📊 Signals gallery",
               annot=None, resample=None, prelude="", failed_syms=None):
    """`prelude` is optional page furniture rendered ABOVE the card grid (the
    watchlist summary table uses it). It defaults to "" and the placeholder sits
    flush against <div class="grid">, so every existing caller emits a
    byte-identical page.

    `failed_syms` are names whose OHLCV download FAILED this run (as opposed to
    not existing on Drive) — they render a red note instead of the benign
    "no feed" one, so a network problem can never masquerade as missing data."""
    annot = annot or {}
    failed_syms = failed_syms or set()
    card_html, data = [], {}

    # Freshness reference = the MODE of every symbol's last raw bar, i.e. what the
    # market as a whole last traded. Self-calibrating, so there is no NSE holiday
    # calendar to maintain and a long holiday weekend simply looks normal.
    bar_dates = {}
    for _s in ranked.get("symbol", pd.Series(dtype=object)).tolist():
        _s = str(_s or "")
        if _s:
            _d = _last_bar_date(omap.get(_s))
            if _d:
                bar_dates[_s] = _d
    mkt_date = None
    if bar_dates:
        _modes = pd.Series(list(bar_dates.values())).mode()
        if not _modes.empty:
            mkt_date = _modes.iloc[0]
    for j, (_, rr) in enumerate(ranked.iterrows()):
        s = str(rr.get("symbol", "") or "")
        pfname = str(rr.get("_pfname", "") or "")          # PF holding name (raw list)
        c, v, e20, e50 = (_ohlc_arrays(omap.get(s, _EMPTY), days, resample=resample)
                          if s else ([], [], [], []))

        # Truly unresolved (no symbol at all) -> name-only card, nothing dropped.
        if not s:
            label = pfname or "(unknown)"
            card_html.append(
                f'<div class="card"><div class="hd">{j + 1}. <b>{label}</b></div>'
                f'<div style="font-size:12px;color:#999;padding:8px 2px">'
                f'Not in the tracked universe (delisted / suspended / non-equity).'
                f'</div></div>')
            continue

        # Symbol resolved: build the fundamentals panels REGARDLESS of candle
        # data (#4 — a name with no OHLCV, e.g. an SME with no Yahoo feed, still
        # shows its grades/quarterly/guidance instead of being blanked out).
        nmj = cards.name_by.get(s.upper(), "") or pfname
        meta = "".join(x for x in [
            f'<div class="hd">{j + 1}. <b>{s}</b>'
            + (f' <span class="nm">{nmj}</span>' if nmj else "")
            + _decision_chip(rr.get("decision"), rr.get("n_buy"), rr.get("n_vote"))
            + _stale_chip(bar_dates.get(s), mkt_date)
            + "</div>",
            annot.get(s.upper(), ""),
            f'<div class="row">{cards.mcap(s, mcap_map)}{cards.grades_strip(s)}</div>',
            cards.quarterly(s), cards.growth_blob(s), cards.guidance_panel(s),
            cards.llm_summary(s), cards.news_card(s, nmj),
            cards.research_card(s, nmj),
        ] if x)
        if not c:
            # Fundamentals present but no price series — panels + a small note,
            # no chart div. A DOWNLOAD FAILURE is called out separately: it used
            # to render as the benign "no feed" note, so a network problem was
            # indistinguishable from a name that genuinely has no price history.
            note = (f'<div class="dlfail">&#9888; Price data FAILED to download '
                    f'this run — this is a fetch problem, not a missing feed. '
                    f'Re-run to restore this chart.</div>'
                    if s.upper() in failed_syms or s in failed_syms else
                    f'<div style="font-size:11px;color:#999;padding:6px 2px">'
                    f'No chart/price data for this name (no OHLCV feed).</div>')
            card_html.append(f'<div class="card">{meta}{note}</div>')
            continue
        data[str(j)] = {"c": c, "v": v, "e20": e20, "e50": e50}
        # Per-chart legend — the page-level note scrolls away on a long gallery,
        # so each chart says what its two lines are.
        legend = ('<div style="font-size:10.5px;color:#777;margin:2px 0 0">'
                  'Candles + volume · <b style="color:#2962FF">━ EMA20</b> '
                  '(short-term trend) · <b style="color:#FF6D00">━ EMA50</b> '
                  '(medium-term trend)</div>')
        card_html.append(f'<div class="card">{meta}{legend}'
                         f'<div class="chart" id="ch{j}"></div></div>')
    now = datetime.now()
    n_behind = sum(1 for d in bar_dates.values() if mkt_date and d < mkt_date)
    banner = _freshness_banner(mkt_date, now.date(), n_behind, len(bar_dates))
    if mkt_date:
        log(f"  data as of {mkt_date} (generated {now.date()}); "
            f"{n_behind}/{len(bar_dates)} charts behind that date")
    return (_TPL.replace("__PAYLOAD__", json.dumps(data, separators=(",", ":")))
                .replace("__CARDS__", "".join(card_html))
                .replace("__N__", str(len(card_html)))
                .replace("__TITLE__", title)
                .replace("__DATE__", now.strftime("%d %b %Y %H:%M"))
                .replace("__FRESH__", banner)
                .replace("__PRELUDE__", prelude or ""))


# ─────────────────────── guidance-strength scoring ──────────────────────────
def _parse_cr(text):
    """Extract a ₹-crore number from free guidance text. Range -> midpoint.
    'INR3000 crores' -> 3000 ; 'INR 850-900 crores' -> 875 ; '3000' -> 3000."""
    t = str(text).lower().replace(",", "")
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    if not nums:
        return None
    vals = [float(n) for n in nums[:2]] if "-" in t.split("cr")[0] or "–" in t else [float(nums[0])]
    val = sum(vals) / len(vals)
    if "bn" in t or "billion" in t:
        val *= 100         # ₹1 bn = 100 cr
    return val


def _horizon_years(h):
    h = str(h).strip().upper()
    fixed = {"NEXT_QTR": 0.25, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "3Y+": 4.0}
    if h in fixed:
        return fixed[h]
    m = re.search(r"FY\D*?(\d{2,4})", h)
    if m:
        yr = int(m.group(1)) % 100
        return max(0.5, (2000 + yr) - 2026)     # years from ~now (mid-2026)
    return 1.0


def guidance_scores(guid, base_rev, base_pat,
                    min_base_rev=0.0, min_base_pat=0.0, cap_cagr=1e9):
    """{SYMBOL: (score_pct, detail_html)} — for each company's LATEST quarter,
    the MAX implied annual growth across revenue & PAT guidance. Absolute targets
    -> CAGR vs current TTM base; growth-rate guidance -> its cagr_pct directly.

    RAW by default (no base floor, no CAGR cap) so the top list is the unfiltered
    ranking — to be tweaked later. Only the math guard (base>0, target>base)
    applies. Pass min_base_rev/min_base_pat/cap_cagr to re-impose sanity limits."""
    if guid is None or guid.empty:
        return {}
    g = guid.copy()
    g["_qo"] = g["quarter"].map(_q_order) if "quarter" in g.columns else -1
    g["_c"] = pd.to_numeric(g.get("cagr_pct"), errors="coerce")
    out = {}
    for sym, sub in g.groupby(g["symbol"].astype(str).str.upper()):
        if sub["_qo"].max() >= 0:
            sub = sub[sub["_qo"] == sub["_qo"].max()]      # latest quarter only
        best = (-1e9, "")
        for _, r in sub.iterrows():
            met = str(r.get("metric", "")).lower()
            if "revenue" in met or "sales" in met:
                base, mlabel, floor = base_rev.get(sym), "Revenue", min_base_rev
            elif "pat" in met or "profit" in met or "earnings" in met:
                base, mlabel, floor = base_pat.get(sym), "PAT", min_base_pat
            else:
                continue
            if not base or base < floor:        # base too small -> CAGR unreliable
                continue
            kind = _g_kind(met, r.get("value"))
            yrs = _horizon_years(r.get("horizon_fy"))
            g_pct, detail = None, ""
            if kind == "absolute":
                tgt = _parse_cr(r.get("value"))
                if tgt and tgt > base and yrs > 0:
                    g_pct = ((tgt / base) ** (1.0 / yrs) - 1) * 100
                    detail = f"{mlabel} ~{base:,.0f}→{tgt:,.0f} cr / {yrs:.0f}y"
            elif kind == "growth" and pd.notna(r["_c"]):
                g_pct = float(r["_c"])
                detail = f"{mlabel} +{g_pct:.0f}%/yr"
            if g_pct is not None and 0 < g_pct <= cap_cagr and g_pct > best[0]:
                best = (g_pct, detail)
        if best[0] > -1e9:
            out[sym] = best
    return out


def _consensus_decision(sig: pd.DataFrame) -> pd.DataFrame:
    """#22 — per-symbol consensus across ALL strategies that flagged it (every
    zone, before the buy/add view filter). A strategy 'votes buy' if any of its
    rows for the symbol is buy/add, else it votes hold. decision = BUY if >=80%
    of the voting strategies say buy, HOLD if 50-80%, else WATCH.
    Returns [symbol, n_buy, n_vote, buy_share, decision]."""
    d = sig.copy()
    d["_buy"] = d["zone_type"].astype(str).str.lower().isin(["buy", "add"])
    per = d.groupby(["symbol", "strategy_group"])["_buy"].max().reset_index()
    agg = (per.groupby("symbol")["_buy"]
           .agg(n_buy="sum", n_vote="count").reset_index())
    agg["buy_share"] = agg["n_buy"] / agg["n_vote"].clip(lower=1)
    agg["decision"] = agg["buy_share"].map(
        lambda x: "BUY" if x >= 0.8 else ("HOLD" if x >= 0.5 else "WATCH"))
    return agg


def _decision_chip(dec, nb, nv) -> str:
    """#22 consensus chip for the card header. BUY green / HOLD amber / WATCH grey."""
    if not dec or (isinstance(dec, float) and pd.isna(dec)):
        return ""
    color = {"BUY": "#1a7a3a", "HOLD": "#a66300", "WATCH": "#7f8c8d"}.get(str(dec), "#7f8c8d")
    try:
        frac = f" · {int(nb)}/{int(nv)}"
    except (TypeError, ValueError):
        frac = ""
    return (f'<span style="background:{color};color:#fff;padding:2px 10px;'
            f'border-radius:6px;font-weight:800;font-size:13px;margin-left:8px">'
            f'{dec}{frac} strat</span>')


def _select_ipos(drive, args, exch):
    """Recent listings (IPO view): every name whose master_list listing_date is
    within --ipo-days, ranked on PURE RETURNS (1m > 3m > 6m > 12m) — no strategy
    count, because a stock listed months ago has too little history to accumulate
    strategy hits and would be unfairly buried. Strategy count is still shown on
    the card so a name that IS being flagged stands out.

    NOTE: listing_date only exists for NSE rows; BSE-only names carry no date and
    are therefore absent from this view.

    A RECENT DATE IS NOT PROOF OF A RECENT LISTING. Measured 2026-09-02: of the
    577 names dated inside a year, 165 are provable re-listings (the stock has
    price bars from BEFORE its listing date — you cannot trade a share that has
    not listed, so prior trading proves the date marks an SME-to-main-board
    migration, not a debut) and 12 are ETFs. That is 31% of this view that was
    never an IPO. company_repo/_index/listing_dates.parquet carries the proof in
    `listing_type`, so it is preferred over master_list when present; the
    classification travels onto the card rather than being silently applied."""
    uni = _read_csv(drive, _folder(drive, "universe"), "master_list.csv")
    if uni.empty or "listing_date" not in uni.columns:
        log("  master_list has no listing_date — IPO view unavailable")
        return pd.DataFrame()
    ld = pd.to_datetime(uni["listing_date"], format="%d-%b-%Y", errors="coerce")
    cut = pd.Timestamp.today().normalize() - pd.Timedelta(days=args.ipo_days)
    rec = uni.assign(_ld=ld)[ld >= cut].copy()
    log(f"  {len(rec)} names listed in the last {args.ipo_days}d (master_list)")

    # ---- prefer the PROVEN classification where it exists --------------------
    ltype = {}
    try:
        lt = _read_parquet(drive, _folder(drive, "company_repo/_index"),
                           "listing_dates.parquet")
    except Exception as e:
        log(f"  listing_dates.parquet unavailable ({type(e).__name__}) — "
            f"falling back to master_list dates only")
        lt = pd.DataFrame()
    if not lt.empty and "listing_type" in lt.columns and "symbol" in lt.columns:
        lt = lt.copy()
        lt["symbol"] = lt["symbol"].astype(str)
        lt["_ld2"] = pd.to_datetime(lt.get("listing_date"), errors="coerce")
        lt = lt.dropna(subset=["_ld2"])
        # this table covers BSE and NSE SME too, which master_list does not
        wider = lt[lt["_ld2"] >= cut]
        if len(wider) > len(rec):
            log(f"  listing_dates.parquet covers {len(wider)} in the same window "
                f"(master_list has {len(rec)}) — using it")
            rec = pd.DataFrame({"symbol": wider["symbol"], "_ld": wider["_ld2"]})
        ltype = dict(zip(lt["symbol"], lt["listing_type"].fillna("")))
        before = len(rec)
        bad = rec["symbol"].astype(str).map(
            lambda x: str(ltype.get(x, "")).lower() in ("migration", "etf"))
        dropped = rec[bad]
        rec = rec[~bad]
        if len(dropped):
            kinds = (dropped["symbol"].astype(str).map(lambda x: ltype.get(x, ""))
                     .value_counts().to_dict())
            log(f"  listing_type gate: {before} -> {len(rec)} "
                f"({len(dropped)} dropped: {kinds}) — re-listings and funds are "
                f"not IPOs")
    else:
        log("  listing_dates.parquet has no listing_type — showing every recent "
            "DATE, which includes re-listings and ETFs. Run "
            "build_listing_dates.py --classify to fix.")
    if rec.empty:
        return pd.DataFrame()

    out = pd.DataFrame({"symbol": rec["symbol"].astype(str),
                        "_listed": rec["_ld"]})
    # attach returns + liquidity from features; a listing with no features yet
    # (too new to compute) is dropped rather than shown blank
    feats = _read_parquet(drive, _folder(drive, "features"), "latest.parquet")
    if feats.empty or "symbol" not in feats.columns:
        log("  features/latest.parquet missing — IPO view unavailable")
        return pd.DataFrame()
    fsym = feats["symbol"].astype(str)
    for lb in _RANK_RETURNS:
        col = f"return_{lb}_pct"
        out[col] = out["symbol"].map(dict(zip(fsym, pd.to_numeric(
            feats[col], errors="coerce")))) if col in feats.columns else float("nan")
    tcol = ("avg_turnover_30d_cr" if "avg_turnover_30d_cr" in feats.columns
            else "avg_turnover_20d_cr")
    if tcol in feats.columns:
        out["_turn"] = out["symbol"].map(dict(zip(fsym, pd.to_numeric(
            feats[tcol], errors="coerce"))))
        if args.turnover > 0:
            before = len(out)
            out = out[out["_turn"].fillna(-1.0) >= args.turnover]
            log(f"  {len(out)}/{before} pass Rs{args.turnover:.0f}cr turnover floor")
    out = out[out[[f"return_{lb}_pct" for lb in _RANK_RETURNS]].notna().any(axis=1)]

    # strategy hits (shown, not used for ranking) + consensus decision
    sig = _load_signals(drive)
    if not sig.empty:
        dec = _consensus_decision(sig)
        ns = (sig.groupby("symbol")["strategy_group"].nunique()
              .reset_index(name="n_strategies"))
        out = out.merge(ns, on="symbol", how="left").merge(dec, on="symbol", how="left")
    if "n_strategies" not in out.columns:
        out["n_strategies"] = 0
    out["n_strategies"] = out["n_strategies"].fillna(0).astype(int)

    sort_cols = [f"return_{lb}_pct" for lb in _RANK_RETURNS
                 if f"return_{lb}_pct" in out.columns]
    out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols),
                          na_position="last")
    out["_exch"] = out["symbol"].map(exch).fillna("NSE")
    log("  IPO view ranked by: " + " > ".join(sort_cols))
    return out.reset_index(drop=True)


def _select_signals(drive, args, exch):
    sig = _load_signals(drive)
    if sig.empty:
        return pd.DataFrame()
    dec = _consensus_decision(sig)          # ALL zones, before the view filter
    if args.zones.strip():
        sig = sig[sig["zone_type"].isin([z.strip() for z in args.zones.split(",")])]
    conv = (sig.groupby("symbol")["strategy_group"].nunique()
            .reset_index(name="n_strategies"))
    conv = conv[conv["n_strategies"] >= args.min_strats]
    best = sig.groupby("symbol")["score"].max().reset_index(name="best_score")
    conv = conv.merge(best, on="symbol", how="left")
    conv = conv.merge(dec, on="symbol", how="left")
    log(f"  {len(conv)} names with >={args.min_strats} strategies")

    # Features carry both the liquidity floor AND the return ladder used for
    # ranking, so load once regardless of whether the turnover floor is on.
    feats = _read_parquet(drive, _folder(drive, "features"), "latest.parquet")
    if not feats.empty and "symbol" in feats.columns:
        fsym = feats["symbol"].astype(str)
        if args.turnover > 0:
            if "avg_turnover_30d_cr" in feats.columns:      # #24: 30-day floor
                turn = pd.to_numeric(feats["avg_turnover_30d_cr"], errors="coerce")
            elif "avg_turnover_20d_cr" in feats.columns:     # fallback until recompute
                turn = pd.to_numeric(feats["avg_turnover_20d_cr"], errors="coerce")
            elif {"vol_20d_avg", "close"} <= set(feats.columns):
                turn = (pd.to_numeric(feats["vol_20d_avg"], errors="coerce")
                        * pd.to_numeric(feats["close"], errors="coerce")) / 1e7
            else:
                turn = pd.Series(float("nan"), index=feats.index)
            tmap = dict(zip(fsym, turn))
            conv = conv[conv["symbol"].astype(str).map(tmap).fillna(-1.0) >= args.turnover]
            log(f"  {len(conv)} pass Rs{args.turnover:.0f}cr turnover floor")
        # Return ladder for the sort (missing -> -inf so it ranks last, never first)
        for _lb in _RANK_RETURNS:
            col = f"return_{_lb}_pct"
            if col in feats.columns:
                rmap = dict(zip(fsym, pd.to_numeric(feats[col], errors="coerce")))
                conv[col] = conv["symbol"].astype(str).map(rmap)
            else:
                conv[col] = float("nan")

    # RANKING: most strategies first, then the return ladder 1m > 3m > 6m > 12m
    # (each is a tiebreak for the one before it). best_score is deliberately NOT
    # used — it is the max RAW score across strategies whose units differ
    # (RS rank 0-100 vs streak DAYS vs raw 3m return %), so it silently favoured
    # whichever strategy happened to emit big numbers.
    sort_cols = ["n_strategies"] + [f"return_{lb}_pct" for lb in _RANK_RETURNS
                                    if f"return_{lb}_pct" in conv.columns]
    conv = conv.sort_values(sort_cols, ascending=[False] * len(sort_cols),
                            na_position="last")
    conv["_exch"] = conv["symbol"].astype(str).map(exch).fillna("NSE")
    if args.bse_last:      # opt-in: park thinner BSE-only names below NSE
        conv = pd.concat([conv[conv["_exch"] != "BSE"], conv[conv["_exch"] == "BSE"]],
                         ignore_index=True)
    log("  ranked by: " + " > ".join(sort_cols))
    return conv.reset_index(drop=True)


_ADD_BADGE = {"fresh": ("🆕 FRESH", "#8e44ad"), "new": ("NEW", "#2980b9"),
              "recent": ("RECENT", "#16a085")}


def _dh_chip(dh):
    """Bold colored chip for % below 52w high — green near the high (strength),
    amber 5-15% off, red deeper. The single most scanned number on a card."""
    if dh is None or pd.isna(dh):
        return ""
    v = float(dh)
    if v >= -0.5:
        return ('<span style="background:#1a7a3a;color:#fff;padding:1px 8px;'
                'border-radius:6px;font-weight:700">🏔 at 52w high</span>')
    col = "#1a7a3a" if v >= -5 else ("#a66300" if v >= -15 else "#c0392b")
    return (f'<span style="background:{col};color:#fff;padding:1px 8px;'
            f'border-radius:6px;font-weight:700">▼ {abs(v):.0f}% off high</span>')


def _load_open_signals(drive) -> pd.DataFrame:
    """signals/aggregated/open_signals.csv — the entry and stop FROZEN on the day
    each (symbol, family) first fired, written by aggregate_signals.

    This is what lets a card say what you would have RISKED and what the idea has
    done since, rather than only what it looks like today. Absent until the new
    aggregator has run once; every consumer below degrades to the old card."""
    try:
        df = _read_csv(drive, _folder(drive, "signals/aggregated"),
                       "open_signals.csv")
    except Exception as e:
        log(f"  open_signals.csv unavailable ({type(e).__name__}) — cards will "
            f"omit the risk and engine lines")
        return pd.DataFrame()
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    log(f"  open_signals: {len(df):,} rows, {df['symbol'].nunique():,} symbols")
    return df


def _open_map(opens: pd.DataFrame) -> dict:
    """symbol -> {families, first_date, entry, stop, times_seen, events}.

    One symbol can hold several families; the OLDEST first_date is the one that
    matters ("when did this idea first appear"), and the entry/stop come from
    that same first sighting so the risk shown is the risk that was real."""
    if opens is None or opens.empty:
        return {}
    out = {}
    for sym, g in opens.groupby("symbol"):
        g = g.copy()
        g["_fd"] = pd.to_datetime(g.get("first_date"), errors="coerce")
        g = g.sort_values("_fd")
        first = g.iloc[0]
        out[str(sym)] = {
            "families": sorted({str(x) for x in g.get("family", []) if str(x)}),
            "first_date": first.get("first_date"),
            "entry": pd.to_numeric(pd.Series([first.get("entry_at_signal")]),
                                   errors="coerce").iloc[0],
            "stop": pd.to_numeric(pd.Series([first.get("stop_at_signal")]),
                                  errors="coerce").iloc[0],
            "times_seen": int(pd.to_numeric(g.get("times_seen"),
                                            errors="coerce").max() or 0),
            "events": int(pd.to_numeric(g.get("n_events_at_signal"),
                                        errors="coerce").fillna(0).max() or 0),
        }
    return out


def _build_meta_annot(ranked, feats, mem, opens=None) -> dict:
    """symbol.upper() -> meta HTML: strategies · price · 1/3/6/12M returns ·
    tenure (came Nx since <date>) + FRESH/NEW/RECENT/DROPPED badge."""
    fcols = ["close", "dist_from_52w_high_pct",
             "return_1m_pct", "return_3m_pct", "return_6m_pct", "return_12m_pct",
             "rs_rank_1m", "rs_rank_3m", "rs_rank_6m", "rs_rank_12m"]
    fmap = {}
    if feats is not None and not feats.empty and "symbol" in feats.columns:
        keep = [c for c in fcols if c in feats.columns]
        for _, r in feats[["symbol"] + keep].iterrows():
            fmap[str(r["symbol"]).upper()] = r
    omap = _open_map(opens)
    mmap = {}
    if mem is not None and not mem.empty and "symbol" in mem.columns:
        mk = [c for c in ["days_present", "window_snapshots", "first_seen",
                          "add_tier", "dropped_last_7d"] if c in mem.columns]
        for _, r in mem[["symbol"] + mk].iterrows():
            mmap[str(r["symbol"]).upper()] = r
    annot = {}
    for _, rr in ranked.iterrows():
        u = str(rr.get("symbol", "") or "").upper()
        if not u:
            continue
        # ⚡ bit only when n_strategies is known: 0 is meaningful (drops view =
        # no longer a select) but NaN/missing (PF holding with no signals) is not.
        bits = []
        ns = rr.get("n_strategies")
        if ns is not None and pd.notna(ns):
            bits.append(f'⚡ {int(ns)} strat')
        f = fmap.get(u)
        if f is not None:
            px = f.get("close")
            if pd.notna(px):
                bits.append(f'₹{float(px):,.0f}')
            dh = f.get("dist_from_52w_high_pct")
            if pd.notna(dh):
                chip = _dh_chip(dh)
                if pd.notna(px) and (1 + float(dh) / 100) != 0:
                    hi = float(px) / (1 + float(dh) / 100)
                    bits.append(f'{chip} <span style="color:#666">52wH ₹{hi:,.0f}</span>')
                elif chip:
                    bits.append(chip)
            rets = []
            for lbl, rcol, kcol in [("1M", "return_1m_pct", "rs_rank_1m"),
                                    ("3M", "return_3m_pct", "rs_rank_3m"),
                                    ("6M", "return_6m_pct", "rs_rank_6m"),
                                    ("12M", "return_12m_pct", "rs_rank_12m")]:
                v = f.get(rcol)
                if pd.notna(v):
                    c = "#1a7a3a" if float(v) >= 0 else "#c0392b"
                    rk = f.get(kcol)
                    rks = (f' <span style="color:#999">#{float(rk):.0f}</span>'
                           if pd.notna(rk) else "")
                    rets.append(f'<span style="color:{c}">{lbl} {float(v):+.0f}%</span>{rks}')
            if rets:
                bits.append(" ".join(rets))
        m = mmap.get(u)
        if m is not None:
            dp = int(m.get("days_present", 0) or 0)
            tot = int(m.get("window_snapshots", 0) or 0)
            bits.append(f'★ came {dp}× of {tot}d' if tot else f'★ came {dp}×')
            if bool(m.get("dropped_last_7d")):
                bits.append('<span style="background:#c0392b;color:#fff;'
                            'padding:0 6px;border-radius:6px">📉 DROPPED</span>')
            else:
                b = _ADD_BADGE.get(str(m.get("add_tier", "") or ""))
                if b:
                    bits.append(f'<span style="background:{b[1]};color:#fff;'
                                f'padding:0 6px;border-radius:6px">{b[0]}</span>')
        # ---- signal memory: which engines, is it new, and what is at risk ----
        # Appended AFTER everything that was already on the card; nothing above
        # is removed or reordered.
        o = omap.get(u)
        if o:
            if o["families"]:
                bits.append('<span style="color:#555">🔧 '
                            + ", ".join(o["families"]) + '</span>')
            if o["times_seen"] == 1:
                bits.append('<span style="background:#0d7a35;color:#fff;'
                            'padding:0 6px;border-radius:6px">🆕 NEW TODAY</span>')
            elif o["times_seen"] > 1:
                bits.append(f'<span style="color:#666">held {o["times_seen"]}d'
                            f'</span>')
            if o["events"] > 0:
                bits.append('<span style="background:#b8860b;color:#fff;'
                            'padding:0 6px;border-radius:6px">⚡ EVENT TODAY'
                            '</span>')
            e, st = o["entry"], o["stop"]
            if pd.notna(e) and pd.notna(st) and float(e) > 0 and float(st) < float(e):
                risk = (float(e) - float(st)) / float(e) * 100
                # A list of names becomes a list of TRADES: what you risk, and a
                # 2R target measured off that risk rather than off a round number.
                tgt = float(e) + 2 * (float(e) - float(st))
                bits.append(f'<span style="color:#333">🎯 entry ₹{float(e):,.0f} '
                            f'· stop ₹{float(st):,.0f} '
                            f'<b>({risk:.1f}% risk)</b> · 2R ₹{tgt:,.0f}</span>')
                # "have I missed it?" — answerable only because entry is frozen
                f2 = fmap.get(u)
                px_now = f2.get("close") if f2 is not None else None
                if px_now is not None and pd.notna(px_now) and float(e) > 0:
                    move = (float(px_now) / float(e) - 1) * 100
                    c = "#1a7a3a" if move >= 0 else "#c0392b"
                    since = f' since {o["first_date"]}' if o.get("first_date") else ""
                    bits.append(f'<span style="color:{c}">{move:+.1f}%{since}</span>')

        annot[u] = (f'<div style="font-size:12px;color:#333;margin:2px 0;'
                    f'line-height:1.6">{" · ".join(bits)}</div>')
    return annot


# ───────────────────── guidance watchlist (--mode watchlist) ────────────────
# Rendered from company_repo/_index/guidance_watchlist.parquet, which
# build_guidance_watchlist.py writes. Cards come from the SHARED build_html, so
# a watchlist card is identical to a gallery.html card; only the summary table
# above the grid and the chip row are new.
_WL_VERDICT_CHIP = {
    "CONFIRMED": ("✓ confirmed", "#1a7a3a"),
    "CONSISTENT": ("≈ consistent", "#2e7d32"),
    "CONTRADICTED": ("✗ contradicted", "#c0392b"),
    "NO_EVIDENCE": ("? unverified", "#a66300"),
}


def _wl_pre_tracking(r) -> bool:
    """Is this row's PREVIOUS quarter before the watchlist started tracking?

    Without this a fresh table (floored at Q1FY27 because Q4FY26 measured 64%
    empty cells) would show every name as 'no concall last quarter', which reads
    as a coverage gap when it is really just our own start date.
    """
    tf = str(r.get("tracking_from") or "").strip()
    if not tf:
        return False
    m = re.match(r"\s*Q([1-4])\D*?(\d{2,4})", str(r.get("quarter") or ""))
    if not m:
        return False
    q, fy = int(m.group(1)), int(m.group(2)) % 100
    # _q_order is fy*100+q, so plain arithmetic on it is NOT quarter arithmetic:
    # the quarter before Q1FY27 is Q4FY26, not "2701 - 1".
    prev = f"Q4 FY{fy - 1:02d}" if q == 1 else f"Q{q - 1} FY{fy:02d}"
    return _q_order(prev) < _q_order(tf)


def _wl_pf_cell(v) -> str:
    """Do I already own this? None means no holdings file was on Drive, which is
    UNKNOWN -- it must not render as a confident No."""
    if v is True:
        return ('<span style="background:#1a7a3a;color:#fff;padding:1px 6px;'
                'border-radius:4px;font-weight:800" title="in your portfolio">'
                '◉ PF</span>')
    if v is False:
        return '<span style="color:#ccc" title="not held">–</span>'
    return '<span style="color:#ccc" title="no holdings file on Drive">?</span>'


def _wl_num(v, dp=0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if f != f else f"{f:,.{dp}f}"


def _wl_table(wl) -> str:
    """The 100-row summary above the cards.

    `Base ₹cr` and `Evidence` are load-bearing columns, not decoration: nothing
    is capped (2026-08-21 decision), so the base and the transcript verdict are
    what tell you a large CAGR is real.
    """
    if wl is None or wl.empty:
        return ""
    head = ["#", "PF", "Symbol", "Company", "ISIN", "BSE", "Qtr", "Added",
            "CAGR%", "Metric", "Src", "Horizon", "Base ₹cr", "Evidence",
            "#Stmts", "Prev-Q", "#Qtrs", "Streak", "Cred"]
    th = "".join(
        f'<th style="position:sticky;top:0;background:#1a3d6e;color:#fff;'
        f'font-size:11px;padding:4px 6px;text-align:left;white-space:nowrap">{h}</th>'
        for h in head)
    rows = []
    for j, (_, r) in enumerate(wl.iterrows()):
        sym = str(r.get("nse_symbol") or r.get("symbol") or "")
        cag = pd.to_numeric(r.get("cagr_pct"), errors="coerce")
        css = G.cell_css(G.grade_growth(cag)) if pd.notna(cag) else ""
        vlabel, vcol = _WL_VERDICT_CHIP.get(str(r.get("validation_verdict") or ""),
                                            ("", "#777"))
        if r.get("in_prev_quarter"):
            prev = '<span style="color:#1a7a3a;font-weight:700">✓</span>'
        elif _wl_pre_tracking(r):
            # the table simply does not go back that far — saying "gap" here
            # would blame the company for our own start date
            prev = '<span style="color:#999" title="the previous quarter is ' \
                   'before this watchlist started tracking">–</span>'
        elif r.get("prev_quarter_had_guidance"):
            prev = '<span style="color:#c0392b">✗</span>'
        else:
            prev = '<span style="color:#a66300" title="no concall extracted ' \
                   'last quarter — a coverage gap, not a churn-out">⚠</span>'
        base = _wl_num(r.get("base_ttm_cr"))
        nq = r.get("base_quarters")
        if base and pd.notna(nq) and int(nq) < 4:
            # an annualised partial window assumes the missing quarters look like
            # the present ones — seasonality is ignored, so say so
            base += (f'<span style="color:#a66300" title="built from {int(nq)} '
                     f'of 4 quarters, scaled x{r.get("base_scale")} — ignores '
                     f'seasonality"> ({int(nq)}q&times;{r.get("base_scale")})</span>')
        if r.get("base_suspect"):
            base = (f'<span style="color:#c0392b" title="tiny TTM base — any '
                    f'target looks explosive against it">{base} ⚠</span>')
        cells = [
            f"{j + 1}",
            _wl_pf_cell(r.get("in_pf")),
            f'<a href="#ch{j}" style="color:#1a3d6e;font-weight:700;'
            f'text-decoration:none">{sym}</a>',
            str(r.get("nse_name") or r.get("company_name") or "")[:34],
            str(r.get("isin") or ""),
            str(r.get("bse_code") or ""),
            str(r.get("quarter") or ""),
            str(r.get("date_added") or ""),
            f'<span style="{css};font-weight:800;padding:0 4px;border-radius:3px">'
            f'{_wl_num(cag, 0)}%</span>',
            str(r.get("score_metric") or ""),
            "deck" if str(r.get("guidance_source")) == "presentation" else "call",
            str(r.get("horizon_fy") or ""),
            base,
            f'<span style="color:{vcol};font-weight:700">{vlabel}</span>',
            # how many SEPARATE statements cleared the bar -- 3 consistent cells
            # is firmer than one borderline line
            (f'<b style="color:#1a7a3a">{int(r.get("n_rows_over_min") or 1)}</b>'
             if (r.get("n_rows_over_min") or 1) > 1
             else str(int(r.get("n_rows_over_min") or 1))),
            prev,
            str(int(r.get("n_quarters") or 0)),
            str(int(r.get("quarter_streak") or 0)),
            _wl_num(r.get("cred_score"), 1),
        ]
        tds = "".join(f'<td style="font-size:11.5px;padding:3px 6px;'
                      f'border-bottom:1px solid #eef1f5;white-space:nowrap">{c}</td>'
                      for c in cells)
        rows.append(f"<tr>{tds}</tr>")
    return ('<div style="max-width:1100px;margin:0 auto 16px">'
            '<div class="card" style="overflow-x:auto;max-height:70vh;'
            'overflow-y:auto">'
            f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}'
            '</tbody></table></div></div>')


def _wl_annot(wl) -> dict:
    """SYMBOL -> the two chip lines that sit under each card header."""
    out = {}
    if wl is None or wl.empty:
        return out
    for _, r in wl.iterrows():
        sym = str(r.get("nse_symbol") or r.get("symbol") or "").upper()
        if not sym:
            continue
        cag = pd.to_numeric(r.get("cagr_pct"), errors="coerce")
        detail = " · ".join(x for x in [
            str(r.get("score_metric") or "").title(),
            str(r.get("score_rule") or "").replace("_", " "),
        ] if x)
        line1 = (f'<div style="background:#0d2f5c;color:#fff;border-radius:6px;'
                 f'padding:4px 10px;font-size:13px;font-weight:700;margin:3px 0">'
                 f'🎯 Guided CAGR ~{cag:.0f}%'
                 f'<span style="font-weight:500;opacity:.85"> · {detail}'
                 f' · {r.get("value_type") or ""} · {r.get("horizon_fy") or ""}'
                 f'</span></div>') if pd.notna(cag) else ""

        bits = []
        if r.get("in_pf") is True:
            bits.append('<span style="background:#1a7a3a;color:#fff;padding:1px 7px;'
                        'border-radius:6px;font-weight:800">◉ IN PF</span>')
        bits += [f'📅 added {r.get("date_added") or ""}'
                 + (_fresh_badge(str(r.get("date_added") or "")) or ""),
                 f'<b>{r.get("quarter") or ""}</b>']
        if r.get("in_prev_quarter"):
            bits.append('<span style="color:#1a7a3a;font-weight:700">'
                        '↩ also last qtr</span>')
        elif _wl_pre_tracking(r):
            bits.append('<span style="color:#999">start of tracking</span>')
        elif r.get("prev_quarter_had_guidance"):
            bits.append('<span style="color:#777">first time</span>')
        else:
            # a coverage hole must never read as a company that stopped guiding
            bits.append('<span style="color:#a66300">no concall last qtr</span>')
        nst = int(r.get("n_rows_over_min") or 1)
        if nst > 1:
            bits.append(f'<span style="color:#1a7a3a;font-weight:700">'
                        f'{nst} statements agree</span>')
        if str(r.get("guidance_source")) == "presentation":
            bits.append('<span style="color:#7b4fa8">from deck</span>')
        nq = int(r.get("n_quarters") or 0)
        if nq:
            bits.append(f'★ {nq}× in filter')
        st = int(r.get("quarter_streak") or 0)
        if st >= 2:
            bits.append(f'🔥 {st}q streak')
        vlabel, vcol = _WL_VERDICT_CHIP.get(str(r.get("validation_verdict") or ""),
                                            ("", "#777"))
        if vlabel:
            bits.append(f'<span style="color:{vcol};font-weight:700">{vlabel}</span>')
        if r.get("base_suspect"):
            bits.append('<span style="color:#c0392b;font-weight:700">'
                        '⚠ tiny base</span>')
        cs = pd.to_numeric(r.get("cred_score"), errors="coerce")
        if pd.notna(cs):
            bits.append(f'cred {cs:.1f}')
        line2 = (f'<div style="font-size:12px;color:#333;margin:2px 0;'
                 f'line-height:1.6">{" · ".join(bits)}</div>')
        out[sym] = line1 + line2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["signals", "pf", "guidance", "watchlist"],
                    default="signals",
                    help="signals=ranked signal gallery; pf=portfolio holdings; "
                         "guidance=top companies by implied guidance CAGR; "
                         "watchlist=the running >50%%-CAGR guidance watchlist "
                         "(needs build_guidance_watchlist.py first), newest "
                         "idea first.")
    ap.add_argument("--quarter", default="auto",
                    help="watchlist mode: which quarter to show "
                         "(default auto = the newest present in the table)")
    ap.add_argument("--min-cagr", type=float, default=0.0,
                    help="watchlist mode: extra CAGR floor on top of whatever "
                         "the table was built with (0 = show everything in it)")
    ap.add_argument("--view", choices=["full", "additions", "drops", "ipo", "new"],
                    default="full",
                    help="signals mode: full=all ranked (existing limits); "
                         "additions=first seen last 14d (fresh/new/recent); "
                         "drops=fell off the select list in last 7d; "
                         "ipo=names listed in the last --ipo-days, ranked on "
                         "pure returns (no strategy-count gate).")
    ap.add_argument("--ipo-days", type=int, default=365,
                    help="IPO view: listing-date lookback window (default 365)")
    ap.add_argument("--min-strats", type=int, default=2)
    ap.add_argument("--zones", default="buy,add", help="comma list; '' = all")
    ap.add_argument("--timeframe-days", type=int, default=252)
    ap.add_argument("--resample", choices=["D", "W", "M"], default="D",
                    help="candle timeframe: D=daily (default), W=weekly, "
                         "M=monthly. EMAs are recomputed on the chosen bars.")
    ap.add_argument("--turnover", type=float, default=1.0, help="min Rs cr/day, 0=off")
    ap.add_argument("--max", type=int, default=0,
                    help="cap chart count (0=all; guidance mode defaults to 100)")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bse-last", action="store_true",
                    help="park BSE-only names below NSE regardless of rank "
                         "(off by default so the strategy/return sort is honoured)")
    ap.add_argument("--no-macro", action="store_true",
                    help="consumed by the .bat wrapper (skips the market_state "
                         "dashboard step); ignored here so pass-through never breaks")
    ap.add_argument("--no-news", action="store_true",
                    help="skip the Google-News headline pass (network-bound)")
    ap.add_argument("--news-days", type=int, default=30,
                    help="news lookback window in days (default 30)")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass the local parquet cache (always download)")
    ap.add_argument("--purge-cache", action="store_true",
                    help="wipe the local parquet cache first, then build "
                         "normally. gallery_all.bat passes this on its FIRST "
                         "gallery step so one batch shares one warm cache.")
    ap.add_argument("--max-missing-pct", type=float, default=5.0,
                    help="abort if more than this %% of the files that DO exist "
                         "on Drive fail to download (default 5). Guards against "
                         "writing a gallery full of silently blank charts.")
    args = ap.parse_args()

    cache = ParquetCache(enabled=not args.no_cache)
    if args.purge_cache:
        cache.purge()
        log(f"parquet cache purged ({cache.root})")

    drive = _drive()
    idx = _folder(drive, "company_repo/_index")
    fund = _folder(drive, "fundamentals")
    uni = _read_csv(drive, _folder(drive, "universe"), "master_list.csv")
    exch = dict(zip(uni["symbol"].astype(str), uni["exchange"].astype(str))) \
        if not uni.empty and {"symbol", "exchange"} <= set(uni.columns) else {}

    log("loading grades / guidance / summary…")
    grades = _read_parquet(drive, idx, "screener_grades.parquet")
    guidance = _read_parquet(drive, idx, "guidance_tracker.parquet")
    summ = _read_parquet(drive, fund, "summary.parquet")
    isin2sym = {}
    if not grades.empty and {"isin", "symbol"} <= set(grades.columns):
        isin2sym = {str(i): str(s) for i, s in zip(grades["isin"], grades["symbol"])}
    mcap_map, base_rev, base_pat = {}, {}, {}
    if not summ.empty and "symbol" in summ.columns:
        for _, r in summ.iterrows():
            s = str(r["symbol"]).upper()
            v = pd.to_numeric(r.get("market_cap_cr"), errors="coerce")
            if pd.notna(v) and v > 0:
                mcap_map[s] = v
            try:
                base_rev[s] = float(pd.Series(r.get("q_sales_last_4q")).astype(float).sum())
                base_pat[s] = float(pd.Series(r.get("q_netprofit_last_4q")).astype(float).sum())
            except Exception:
                pass

    title, annot, prelude = "📊 Signals gallery", {}, ""
    out_default = "gallery.html"

    # Meta-line inputs (52wH, returns+ranks, tenure) — used by signals AND pf modes
    mem = _read_parquet(drive, _folder(drive, "signals/aggregated"),
                        "membership.parquet")
    # The entry/stop memory. Loaded once so every view and both modes see the
    # same frame; absent until the new aggregator has run, and every consumer
    # degrades to the pre-existing card when it is.
    opens_df = _load_open_signals(drive)
    feats_g = _read_parquet(drive, _folder(drive, "features"), "latest.parquet")

    if args.mode == "pf":
        title, out_default = "💼 Portfolio (PF) charts", "gallery_pf.html"
        # auto-healed isin->symbol: master_list + grades + guidance_tracker
        # (fills SME symbols grades leaves blank), shared isin_symbol_map.
        isin2sym = isin_symbol_map(uni, grades, guidance)
        # RAW full list — every holding, including ones we can't chart (shown as
        # a name-only card so nothing is silently dropped). No filtering.
        hold = _load_pf_holdings(drive)            # DataFrame[isin, name] (both folders)
        if hold.empty:                             # last-ditch: ISIN set only
            isins = load_portfolio_isins(drive, os.environ["GDRIVE_FOLDER_ID"]) or set()
            hold = pd.DataFrame({"isin": sorted(isins), "name": ""})
        hold["symbol"] = hold["isin"].map(lambda i: isin2sym.get(str(i), ""))
        hold["_mc"] = hold["symbol"].map(lambda s: mcap_map.get(str(s).upper(), 0) if s else 0)
        # Sort: deepest % below 52w high FIRST (laggards on top for review);
        # no-data names fall to the end, tie-break by mcap.
        dh_map = {}
        if not feats_g.empty and {"symbol", "dist_from_52w_high_pct"} <= set(feats_g.columns):
            dh_map = dict(zip(feats_g["symbol"].astype(str),
                              pd.to_numeric(feats_g["dist_from_52w_high_pct"],
                                            errors="coerce")))
        hold["_dh"] = hold["symbol"].map(lambda s: dh_map.get(str(s)) if s else None)
        resolved = hold[hold["symbol"] != ""].sort_values(
            ["_dh", "_mc"], ascending=[True, False], na_position="last")
        unresolved = hold[hold["symbol"] == ""]
        ranked = pd.concat([resolved, unresolved], ignore_index=True)
        ranked = ranked.rename(columns={"name": "_pfname"})
        # Same enriched meta line as the signals gallery (52wH, returns+ranks,
        # tenure, badges). n_strategies joined from live signals so holdings that
        # are also selects show real coverage; non-select holdings omit the ⚡ bit.
        sig_pf = _load_signals(drive)
        if not sig_pf.empty:
            if args.zones.strip():
                sig_pf = sig_pf[sig_pf["zone_type"].isin(
                    [z.strip() for z in args.zones.split(",")])]
            nmap = sig_pf.groupby("symbol")["strategy_group"].nunique()
            ranked["n_strategies"] = ranked["symbol"].map(nmap)
        annot = _build_meta_annot(ranked, feats_g, mem, opens_df)
        log(f"  PF holdings: {len(ranked)} total "
            f"({len(resolved)} chartable, {len(unresolved)} name-only/not-in-universe)")

    elif args.mode == "guidance":
        title, out_default = "📈 Top guidance (implied CAGR)", "gallery_guidance.html"
        scores = guidance_scores(guidance, base_rev, base_pat)
        rows = sorted(scores.items(), key=lambda kv: -kv[1][0])
        top = args.max if args.max > 0 else 100
        rows = rows[:top]
        ranked = pd.DataFrame({"symbol": [k for k, _ in rows]})
        for sym, (sc, detail) in rows:
            annot[sym] = (f'<div style="background:#0d2f5c;color:#fff;border-radius:6px;'
                          f'padding:4px 10px;font-size:13px;font-weight:700;margin:3px 0">'
                          f'📈 Guidance CAGR ~{sc:.0f}%'
                          + (f' <span style="font-weight:500;opacity:.9">· {detail}</span>'
                             if detail else "") + "</div>")
        log(f"  scored {len(scores)} companies with guidance; top {len(ranked)}")

    elif args.mode == "watchlist":
        out_default = "gallery_guidance_watchlist.html"
        wl = _read_parquet(drive, idx, "guidance_watchlist.parquet")
        if wl.empty:
            log("No guidance_watchlist.parquet — run "
                "scripts/build_guidance_watchlist.py first.")
            return
        wl = wl.copy()
        wl["_qo"] = wl["quarter"].map(_q_order)
        q = (str(wl.sort_values("_qo")["quarter"].iloc[-1])
             if args.quarter.lower() == "auto" else args.quarter)
        # ONE quarter only, and that is structural rather than a preference: the
        # table holds one row per (isin, quarter), so a multi-quarter view would
        # render the same symbol — and the same chart — two or three times.
        wl = wl[wl["quarter"].astype(str).str.replace(" ", "")
                == str(q).replace(" ", "")]
        if args.min_cagr > 0 and "cagr_pct" in wl.columns:
            wl = wl[pd.to_numeric(wl["cagr_pct"], errors="coerce") >= args.min_cagr]
        top = args.max if args.max > 0 else 100
        # newest idea on top; CAGR breaks a same-day tie so the order is stable
        wl = wl.sort_values(["date_added", "cagr_pct"], ascending=[False, False],
                            na_position="last").head(top).reset_index(drop=True)
        if wl.empty:
            log(f"Nothing in the watchlist for {q}."); return
        ranked = wl.copy()
        ranked["symbol"] = [str(s or t or "") for s, t in
                            zip(wl.get("symbol", ""), wl.get("nse_symbol", ""))]
        # a name with no resolvable symbol still belongs on the list — _pfname
        # makes it render as a LABELLED card instead of "(unknown)"
        ranked["_pfname"] = wl.get("nse_name", wl.get("company_name", ""))
        annot = _wl_annot(wl)
        prelude = _wl_table(wl)
        floor = wl["min_cagr_used"].dropna()
        floor = float(floor.iloc[0]) if len(floor) else args.min_cagr
        n_pf = int(wl["in_pf"].fillna(False).sum()) if "in_pf" in wl.columns else 0
        title = (f"🎯 GUIDANCE WATCHLIST — {q} — management guided "
                 f"revenue or PAT above {floor:.0f}%/yr — transcript-checked "
                 f"— newest guidance first"
                 + (f" — {n_pf} already in PF" if n_pf else ""))
        log(f"  {len(wl)} names in the watchlist for {q}")

    else:  # signals
        if args.view == "drops":
            title, out_default = "📉 Recent drops (7d)", "gallery_drops.html"
            if mem.empty or "dropped_last_7d" not in mem.columns:
                log("No membership.parquet — run build_signal_membership.py first.")
                return
            dr = mem[mem["dropped_last_7d"] == True].sort_values(
                "days_present", ascending=False)
            ranked = pd.DataFrame({"symbol": dr["symbol"].astype(str).tolist()})
            ranked["n_strategies"] = 0
            ranked["_exch"] = ranked["symbol"].map(exch).fillna("NSE")
            log(f"  {len(ranked)} names dropped from the select list in last 7d")
        elif args.view == "ipo":
            title = f"🚀 Recent listings ({args.ipo_days}d) — ranked by return"
            out_default = "gallery_ipo.html"
            ranked = _select_ipos(drive, args, exch)
            if ranked.empty:
                log("Nothing to render."); return
        else:
            ranked = _select_signals(drive, args, exch)
            if args.view == "additions":
                title, out_default = "🆕 Recent additions (14d)", "gallery_additions.html"
                if not mem.empty and "add_tier" in mem.columns:
                    add = set(mem[mem["add_tier"].isin(["fresh", "new", "recent"])]
                              ["symbol"].astype(str).str.upper())
                    ranked = ranked[ranked["symbol"].astype(str).str.upper()
                                    .isin(add)].reset_index(drop=True)
                log(f"  {len(ranked)} recent additions (fresh/new/recent)")
            elif args.view == "new":
                # STRICTLY today's first-timers, from the signal memory rather
                # than a 14-day window. `additions` answers "what is recent";
                # this answers "what is new TODAY, and which engine found it".
                title = "🆕 New ideas today — by engine"
                out_default = "gallery_new_today.html"
                om = _open_map(opens_df)
                fresh = {k for k, v in om.items() if v["times_seen"] == 1}
                if not fresh:
                    log("  no first-time signals today — open_signals has no "
                        "times_seen == 1 rows (the memory may be one run old)")
                ranked = ranked[ranked["symbol"].astype(str).str.upper()
                                .isin(fresh)].reset_index(drop=True)
                ev = sum(1 for k in fresh if om[k]["events"] > 0)
                log(f"  {len(ranked)} NEW today ({ev} of them with an event "
                    f"firing, the rest are states that just qualified)")
        annot = _build_meta_annot(ranked, feats_g, mem, opens_df)
        if args.max > 0:
            ranked = ranked.head(args.max)

    if ranked.empty:
        log("Nothing to render."); return
    syms = [s for s in ranked["symbol"].tolist() if s]      # skip name-only entries
    log(f"  {len(ranked)} cards selected (mode={args.mode}); {len(syms)} chartable")

    if args.dry_run:
        log("DRY-RUN — top 10: " + ", ".join(syms[:10]))
        return

    # Timeframe: daily (default) / weekly / monthly — resample bars + EMAs, and
    # tag the title + output filename so the three files sit side by side.
    _rs_map = {"D": (None, "", ""),
               "W": ("W-FRI", " · Weekly", "_weekly"),
               "M": ("ME", " · Monthly", "_monthly")}
    resample, tf_title, tf_suffix = _rs_map[args.resample]
    title += tf_title
    if tf_suffix and not args.out:
        out_default = out_default.replace(".html", f"{tf_suffix}.html")

    out_path = args.out or os.path.join(os.path.dirname(_SCRIPTS_DIR), out_default)
    log("loading gf1 / announcements / research…")
    gf1 = _read_parquet(drive, idx, "gf1_guidance_statements.parquet")
    ann = _read_parquet(drive, idx, "announcement_ledger.parquet")
    res_idx = _read_parquet(drive, idx, "research_index.parquet")
    log(f"downloading OHLCV for {len(syms)} names…")
    ohlcv_failed = set()
    omap = _bulk_parquet(
        drive, _folder(drive, "data/ohlcv"), syms, cache=cache, what="OHLCV",
        max_missing_pct=args.max_missing_pct, failed_out=ohlcv_failed)
    log("downloading statements…")
    stmts = _bulk_parquet(
        drive, _folder(drive, "fundamentals/statements"), syms, cache=cache,
        what="statements", max_missing_pct=args.max_missing_pct)

    _name_by = {}
    if not grades.empty and {"symbol", "company_name"} <= set(grades.columns):
        _name_by = {str(r["symbol"]).upper(): str(r.get("company_name", "") or "")
                    for _, r in grades.iterrows()}
    news_map = ({} if args.no_news
                else _fetch_news_map(syms, _name_by, days=args.news_days))

    cards = Cards(grades, stmts, guidance, gf1, ann, research=res_idx, news=news_map)
    log("assembling HTML…")
    html = build_html(ranked, omap, cards, mcap_map, args.timeframe_days,
                      title=title, annot=annot, resample=resample,
                      prelude=prelude, failed_syms=ohlcv_failed)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"wrote {out_path}  ({len(html) / 1e6:.1f} MB, {len(syms)} charts)")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out_path))
        log("opened in default browser.")


if __name__ == "__main__":
    main()
