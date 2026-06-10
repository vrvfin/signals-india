"""
Phase 3 — T4.4 Stage 1: Investigative fraud grade (regulatory surveillance lists).

DISTINCT from build_fraud_risk.py (financial forensic rules): this detects KNOWN
external enforcement / watchlist signals. A company can pass every forensic rule
and still be on the exchange's surveillance radar. Writes:

  company_repo/_index/investigative_fraud.parquet   — one row per UNIVERSE company
  company_repo/_index/investigative_fraud.csv

Stage 1 sources (public CSV/JSON, no API key, ~5 HTTP calls total per run):
  NSE ASM  — Additional Surveillance Measure (long-term + short-term)
  NSE ESM  — Enhanced Surveillance Measure
  NSE GSM  — Graded Surveillance Measure (stages 1-6)
  NSE T2T  — Trade-to-Trade proxy: series BE/BZ in archives EQUITY_L.csv
  BSE      — group flags from ListofScripData (T/TS = trade-to-trade,
             Z/ZP/ZY = listing non-compliance), matched by ISIN

Grading rubric (locked 2026-06-10; higher = worse):
  0 CLEAN      no flag anywhere
  1 WATCH      T2T only
  2 CAUTION    GSM stage 1-2, OR ASM long-term stage 1, OR ASM short-term
  3 HIGH RISK  GSM stage 3-4, OR ASM long-term stage 2+, OR ESM any stage,
               OR BSE Z-group, OR SEBI/NFRA enforcement hit (Stage 2)
  4 AVOID      GSM stage 5-6, OR SFIO/NCLT proceedings (Stage 2)

Scorecard wiring (Option A — highlight, never filter): build_scorecard.py reads
this parquet as an 8th factor, score_investigative = (4 - grade) / 4 * 100,
weight 10%. NO hard cap on the composite.

Stage 2 (open items, columns reserved here but always 0/empty for now):
  sebi_actions (SEBI enforcement scrape), nfra_actions (NFRA orders),
  news_hits / news_snippets (Google News RSS keyword scan), MCA SFIO.

Usage:
    python scripts/build_investigative_fraud.py --dry-run        # fetch + grade, no writes
    python scripts/build_investigative_fraud.py --local          # write to .t4_local mirror
    python scripts/build_investigative_fraud.py                  # real Drive write
    python scripts/build_investigative_fraud.py --lists-from DIR # offline: read cached
        asm.json / esm.json / gsm.json / t2t.csv / bse.json snapshots instead of network
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Shared Drive layer (CLAUDE.md global rule #4 — reuse, never raw API calls).
from _extractor_base import (
    get_drive, get_or_create_subfolder, find_file, download_bytes, upload_bytes,
)

DATA_MISSING = "DATA_MISSING"

INVESTIGATIVE_COLS = [
    "isin", "symbol", "company_name",
    "asm_level", "esm_level", "gsm_stage", "t2t", "bse_group",
    "sebi_actions", "nfra_actions", "news_hits", "news_snippets",
    "investigative_grade", "grade_reason", "checked_at",
]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

NSE_API = {
    "asm": "https://www.nseindia.com/api/reportASM",
    "esm": "https://www.nseindia.com/api/reportESM",
    "gsm": "https://www.nseindia.com/api/reportGSM",
}
NSE_EQUITY_L = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_SCRIPS = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
              "?Group=&Scripcode=&industry=&segment=Equity&status=Active")

T2T_SERIES = {"BE", "BZ"}          # NSE trade-to-trade series
BSE_T2T_GROUPS = {"T", "TS", "XT"}  # BSE trade-to-trade groups
BSE_NONCOMPLIANT_GROUPS = {"Z", "ZP", "ZY"}  # listing-requirement non-compliance


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ------------------------------------------------------------------ #
#  Storage abstraction (mirrors build_valuation/build_fraud_risk)     #
# ------------------------------------------------------------------ #

class Store:
    def __init__(self, local: bool, local_dir: Path | None):
        self.local = local
        self.local_dir = local_dir
        self.drive = None
        if not local:
            self.drive = get_drive()
            self.root = os.environ["GDRIVE_FOLDER_ID"]

    def _folder(self, parts):
        fid = self.root
        for p in parts:
            fid = get_or_create_subfolder(self.drive, fid, p)
        return fid

    def read_csv(self, path_parts):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            return pd.read_csv(fp) if fp.exists() else None
        fid = find_file(self.drive, self._folder(folder), name)
        if not fid:
            return None
        return pd.read_csv(io.BytesIO(download_bytes(self.drive, fid)))

    def write_df(self, path_parts, df: pd.DataFrame):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            fp.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(fp, index=False) if name.endswith(".csv") else df.to_parquet(fp, index=False)
            return
        if name.endswith(".csv"):
            data = df.to_csv(index=False).encode("utf-8")
            mime = "text/csv"
        else:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            data = buf.getvalue()
            mime = "application/octet-stream"
        folder_id = self._folder(folder)
        existing = find_file(self.drive, folder_id, name)
        upload_bytes(self.drive, folder_id, name, data, mime, existing_id=existing)


# ------------------------------------------------------------------ #
#  Fetchers — defensive parsing (NSE field names drift over time)     #
# ------------------------------------------------------------------ #

def _nse_session() -> requests.Session:
    """NSE API needs a browser-like session warmed up with a homepage cookie."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        s.get("https://www.nseindia.com", timeout=15)
    except Exception as e:
        log(f"  NSE homepage warm-up failed ({str(e)[:60]}) — API calls may 401.")
    return s


def _extract_records(payload) -> list[dict]:
    """Recursively collect every dict that has a 'symbol' key from arbitrary
    NSE JSON shapes ({'data': [...]}, {'longterm': {'data': [...]}}, plain list)."""
    out = []
    if isinstance(payload, dict):
        if "symbol" in {str(k).lower() for k in payload}:
            out.append(payload)
        else:
            for v in payload.values():
                out.extend(_extract_records(v))
    elif isinstance(payload, list):
        for item in payload:
            out.extend(_extract_records(item))
    return out


_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


def _parse_stage(text) -> int | None:
    """Parse '1' / 'Stage I' / 'LTASM Stage IV' / 'II' into an int stage."""
    t = str(text).strip().upper()
    if not t or t in ("NONE", "NAN", "-"):
        return None
    m = re.search(r"\b(VI|IV|V|III|II|I)\b", t)
    if m:
        return _ROMAN[m.group(1)]
    m = re.search(r"(\d+)", t)
    return int(m.group(1)) if m else None


def _record_stage(rec: dict) -> int:
    """Best-effort stage from one NSE record; keys containing stage/surv/indicator
    first, then any value. Defaults to 1 (on the list = at least stage 1)."""
    keyed = [v for k, v in rec.items()
             if any(w in str(k).lower() for w in ("stage", "surv", "indicator"))]
    for v in keyed + list(rec.values()):
        st = _parse_stage(v)
        if st is not None and 1 <= st <= 6:
            return st
    return 1


def _record_symbol(rec: dict) -> str:
    for k, v in rec.items():
        if str(k).lower() == "symbol":
            return str(v).strip().upper()
    return ""


def fetch_nse_lists(lists_from: Path | None) -> dict[str, dict[str, int]]:
    """Returns {'asm': {symbol: stage}, 'esm': {...}, 'gsm': {...}}.
    Each fetch fails soft (warn + empty) so one broken endpoint never kills the run."""
    results: dict[str, dict[str, int]] = {"asm": {}, "esm": {}, "gsm": {}}
    session = None
    for name, url in NSE_API.items():
        payload = None
        if lists_from:
            fp = lists_from / f"{name}.json"
            if fp.exists():
                payload = json.loads(fp.read_text(encoding="utf-8"))
            else:
                log(f"  {name.upper()}: no cached {fp.name} — skipping")
                continue
        else:
            try:
                if session is None:
                    session = _nse_session()
                r = session.get(url, timeout=20)
                if r.status_code != 200:
                    log(f"  {name.upper()}: HTTP {r.status_code} — skipping")
                    continue
                payload = r.json()
            except Exception as e:
                log(f"  {name.upper()}: fetch failed ({str(e)[:80]}) — skipping")
                continue
        recs = _extract_records(payload)
        for rec in recs:
            sym = _record_symbol(rec)
            if sym:
                stage = _record_stage(rec)
                results[name][sym] = max(results[name].get(sym, 0), stage)
        log(f"  {name.upper()}: {len(results[name])} symbols flagged")
    return results


def fetch_nse_t2t(lists_from: Path | None) -> set[str]:
    """Symbols in NSE trade-to-trade series (BE/BZ) from the EQUITY_L master CSV."""
    try:
        if lists_from:
            fp = lists_from / "t2t.csv"
            if not fp.exists():
                log("  T2T: no cached t2t.csv — skipping")
                return set()
            df = pd.read_csv(fp)
        else:
            r = requests.get(NSE_EQUITY_L, headers={"User-Agent": UA}, timeout=20)
            if r.status_code != 200:
                log(f"  T2T: EQUITY_L HTTP {r.status_code} — skipping")
                return set()
            df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().upper() for c in df.columns]
        if "SERIES" not in df.columns or "SYMBOL" not in df.columns:
            log(f"  T2T: unexpected EQUITY_L columns {list(df.columns)[:5]} — skipping")
            return set()
        t2t = set(df[df["SERIES"].astype(str).str.strip().str.upper().isin(T2T_SERIES)]
                  ["SYMBOL"].astype(str).str.strip().str.upper())
        log(f"  T2T: {len(t2t)} symbols in BE/BZ series")
        return t2t
    except Exception as e:
        log(f"  T2T: fetch failed ({str(e)[:80]}) — skipping")
        return set()


def fetch_bse_groups(lists_from: Path | None) -> dict[str, str]:
    """Returns {ISIN: bse_group} for groups we care about (T2T + non-compliance)."""
    try:
        if lists_from:
            fp = lists_from / "bse.json"
            if not fp.exists():
                log("  BSE: no cached bse.json — skipping")
                return {}
            data = json.loads(fp.read_text(encoding="utf-8"))
        else:
            r = requests.get(BSE_SCRIPS,
                             headers={"User-Agent": UA, "Accept": "application/json",
                                      "Referer": "https://www.bseindia.com/"},
                             timeout=30)
            if r.status_code != 200:
                log(f"  BSE: HTTP {r.status_code} — skipping")
                return {}
            data = r.json()
        if isinstance(data, dict):
            data = data.get("Table") or data.get("data") or []
        watch = BSE_T2T_GROUPS | BSE_NONCOMPLIANT_GROUPS
        out = {}
        for row in data:
            grp = str(row.get("GROUP", row.get("Group", ""))).strip().upper()
            isin = str(row.get("ISIN_NUMBER", row.get("ISIN", ""))).strip().upper()
            if grp in watch and isin:
                out[isin] = grp
        log(f"  BSE: {len(out)} ISINs in watched groups (T2T/non-compliant)")
        return out
    except Exception as e:
        log(f"  BSE: fetch failed ({str(e)[:80]}) — skipping")
        return {}


# ------------------------------------------------------------------ #
#  Grading (pure — unit-testable offline)                             #
# ------------------------------------------------------------------ #

def grade_company(asm_lt_stage: int | None, asm_st_stage: int | None,
                  esm_stage: int | None, gsm_stage: int | None,
                  t2t: bool, bse_group: str = "",
                  sebi_actions: int = 0, nfra_actions: int = 0) -> tuple[int, str]:
    """Apply the locked rubric. Returns (grade 0-4, human-readable reason)."""
    grade, reasons = 0, []

    if t2t:
        grade = max(grade, 1)
        reasons.append("T2T segment")
    if bse_group in BSE_T2T_GROUPS:
        grade = max(grade, 1)
        reasons.append(f"BSE group {bse_group} (T2T)")

    if asm_st_stage:
        grade = max(grade, 2)
        reasons.append(f"ASM short-term stage {asm_st_stage}")
    if asm_lt_stage:
        grade = max(grade, 3 if asm_lt_stage >= 2 else 2)
        reasons.append(f"ASM long-term stage {asm_lt_stage}")

    if gsm_stage:
        grade = max(grade, 4 if gsm_stage >= 5 else 3 if gsm_stage >= 3 else 2)
        reasons.append(f"GSM stage {gsm_stage}")

    if esm_stage:
        grade = max(grade, 3)
        reasons.append(f"ESM stage {esm_stage}")

    if bse_group in BSE_NONCOMPLIANT_GROUPS:
        grade = max(grade, 3)
        reasons.append(f"BSE group {bse_group} (non-compliant)")

    # Stage 2 inputs — always 0 today, wired so the rubric is already complete.
    if sebi_actions > 0:
        grade = max(grade, 3)
        reasons.append(f"{sebi_actions} SEBI action(s)")
    if nfra_actions > 0:
        grade = max(grade, 3)
        reasons.append(f"{nfra_actions} NFRA order(s)")

    return grade, "; ".join(reasons) if reasons else "clean"


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--local-dir", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch the public lists and grade, but write NOTHING.")
    ap.add_argument("--lists-from", type=str, default=None,
                    help="Offline: dir with asm.json/esm.json/gsm.json/t2t.csv/bse.json "
                         "snapshots — no network calls at all.")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    local_dir = Path(args.local_dir) if args.local_dir else \
        Path(__file__).resolve().parent.parent / ".t4_local"
    store = Store(args.local, local_dir)
    lists_from = Path(args.lists_from) if args.lists_from else None
    log(f"build_investigative_fraud — mode={'LOCAL' if args.local else 'DRIVE'} "
        f"{'(dry-run)' if args.dry_run else ''}"
        f"{' lists<-' + str(lists_from) if lists_from else ''}")

    # ---- universe (every company gets a row, default grade 0) ----
    cu = store.read_csv(["company_repo", "_index", "company_universe.csv"])
    if cu is None or cu.empty:
        cu = store.read_csv(["universe", "master_list.csv"])
    if cu is None or cu.empty:
        log("No universe file (company_universe.csv / master_list.csv) — cannot grade.")
        return
    sym_col = "nse_symbol" if "nse_symbol" in cu.columns else "symbol"
    universe = []
    seen = set()
    for _, r in cu.iterrows():
        sym = str(r.get(sym_col, "")).strip().upper()
        isin = str(r.get("isin", "")).strip().upper()
        if not sym or sym == "NAN" or sym in seen:
            continue
        seen.add(sym)
        universe.append((isin, sym, str(r.get("name", "")).strip()))
    if args.names:
        wanted = {s.strip().upper() for s in args.names.split(",") if s.strip()}
        universe = [u for u in universe if u[1] in wanted]
    if args.limit:
        universe = universe[:args.limit]
    log(f"Universe: {len(universe)} companies to grade")
    if not universe:
        return

    # ---- fetch surveillance lists (each fails soft) ----
    log("Fetching surveillance lists…")
    nse = fetch_nse_lists(lists_from)
    t2t_set = fetch_nse_t2t(lists_from)
    bse_map = fetch_bse_groups(lists_from)

    n_signals = sum(len(v) for v in nse.values()) + len(t2t_set) + len(bse_map)
    if n_signals == 0:
        log("WARNING: every list came back empty — output would mark the whole "
            "universe Grade 0, which is indistinguishable from a fetch outage. "
            "NOT writing. (Use --lists-from with cached snapshots to test offline.)")
        return

    # NOTE: NSE reportASM mixes long-term and short-term blocks; _extract_records
    # flattens both, so a symbol's stage here is the max across blocks. We treat
    # stage>=2 as long-term-equivalent severity (grade 3 per rubric) — conservative.
    rows = []
    for isin, sym, cname in universe:
        asm_stage = nse["asm"].get(sym)
        esm_stage = nse["esm"].get(sym)
        gsm_stage = nse["gsm"].get(sym)
        t2t = sym in t2t_set
        bse_group = bse_map.get(isin, "")

        grade, reason = grade_company(
            asm_lt_stage=asm_stage, asm_st_stage=None,
            esm_stage=esm_stage, gsm_stage=gsm_stage,
            t2t=t2t, bse_group=bse_group,
        )
        rows.append({
            "isin": isin, "symbol": sym, "company_name": cname,
            "asm_level": f"ASM-{asm_stage}" if asm_stage else "none",
            "esm_level": f"ESM-{esm_stage}" if esm_stage else "none",
            "gsm_stage": gsm_stage or 0,
            "t2t": bool(t2t),
            "bse_group": bse_group,
            "sebi_actions": 0,      # Stage 2
            "nfra_actions": 0,      # Stage 2
            "news_hits": 0,         # Stage 2
            "news_snippets": "",    # Stage 2
            "investigative_grade": grade,
            "grade_reason": reason,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        })

    out = pd.DataFrame(rows, columns=INVESTIGATIVE_COLS)
    dist = out["investigative_grade"].value_counts().sort_index().to_dict()
    log(f"Graded {len(out)} companies. Distribution: {dist}")

    flagged = out[out["investigative_grade"] > 0]
    if args.dry_run:
        log("DRY-RUN — not writing. Flagged sample:")
        sample = (flagged if not flagged.empty else out).head(15)
        cols = ["symbol", "investigative_grade", "grade_reason",
                "asm_level", "esm_level", "gsm_stage", "t2t", "bse_group"]
        print(sample[cols].to_string(index=False))
        return

    store.write_df(["company_repo", "_index", "investigative_fraud.parquet"], out)
    store.write_df(["company_repo", "_index", "investigative_fraud.csv"], out)
    log("Wrote investigative_fraud.parquet + investigative_fraud.csv to _index/.")


if __name__ == "__main__":
    main()
