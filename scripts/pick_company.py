r"""
pick_company.py — turn what you typed into ONE confirmed company before a report runs or is
queued (2026-10-03).

WHY. "augoment" (a typo for Augmont) was queued as raw text in both report queues and would
only have failed at the 08:00 CI run, silently. Run-now paths quietly took the first partial
match. This step finds the candidates, shows them, and you confirm — reports and queues then
use the confirmed ISIN, so CI can never pick a different company.

WHERE CANDIDATES COME FROM (merged, one line per ISIN):
  1. universe exact   — ISIN / NSE symbol / BSE symbol / BSE code   (company_universe.csv)
  2. universe name    — every word you typed appears in the name
  3. universe typo    — fuzzy match on name and symbol ("augoment" -> Augmont)
  4. Screener search  — partial and brand names ("paytm" -> One 97 Communications)
  5. Yahoo search     — SYMBOL.NS / SYMBOL.BO
  Web hits are kept only when they map back to a company in the universe.

USAGE
  python scripts/pick_company.py --names "augoment, morepen"            # pick interactively
  python scripts/pick_company.py --names "augoment" --dry-run           # list only, no prompt
  python scripts/pick_company.py --names "TCS" --out picked.txt         # for run_report.bat
Names are separated by COMMAS (a space is part of the name: "Man Industries").
Prompt: Enter = 1, a number = that company, 0 = none of these.
--out gets one line per confirmed company: ISIN|SYMBOL|NAME. Exit 1 if any name is not
confirmed, so the caller stops instead of running or queuing the wrong thing.
Read-only: no Gemini call, no Drive write.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd

MAX_SHOWN = 8
TYPO_CUTOFF = 0.75          # difflib ratio; "augoment" vs "augmont" = 0.80
_ISIN = re.compile(r"^IN[A-Z0-9]{10}$", re.IGNORECASE)
_FILLER = re.compile(r"\b(limited|ltd|the|pvt|private)\b\.?", re.IGNORECASE)
_UA = {"User-Agent": "Mozilla/5.0"}


def _norm(s) -> str:
    s = _FILLER.sub(" ", str(s or "").lower())
    return " ".join(re.findall(r"[a-z0-9]+", s))


def _code(v) -> str:
    """BSE code as text: 544888.0 -> '544888'."""
    s = str(v or "").strip()
    return s[:-2] if s.endswith(".0") else ("" if s.lower() == "nan" else s)


def load_universe() -> pd.DataFrame:
    import company_deep_report as CDR
    from daily_research_summary import drive_service
    u = CDR._load_universe(drive_service(), os.environ["GDRIVE_FOLDER_ID"]).copy()
    for c in ("isin", "name", "nse_symbol", "bse_code", "bse_symbol"):
        if c not in u.columns:
            u[c] = ""
    u["bse_code"] = u["bse_code"].map(_code)
    for c in ("isin", "nse_symbol", "bse_symbol"):
        u[c] = u[c].fillna("").astype(str).str.strip().str.upper()
    u["norm"] = u["name"].map(_norm)
    return u.drop_duplicates("isin")


def _row(u: pd.DataFrame, mask, why: str) -> list[dict]:
    # r["isin"], never r.isin: on a row Series .isin is pandas' isin() METHOD, not the column.
    return [dict(isin=r["isin"], name=r["name"],
                 symbol=r["nse_symbol"] or r["bse_symbol"] or r["bse_code"],
                 bse=r["bse_code"], why=why) for _, r in u[mask].iterrows()]


def universe_candidates(u: pd.DataFrame, text: str) -> list[dict]:
    t = text.strip()
    up, q = t.upper(), _norm(t)
    out = _row(u, (u["isin"] == up) | (u["nse_symbol"] == up) | (u["bse_symbol"] == up)
               | (u["bse_code"] == t), "exact")
    if q:
        words = q.split()
        has_all = u["norm"].map(lambda n: all(w in n for w in words))
        hit = u[has_all].copy()
        hit["k"] = hit["norm"].map(lambda n: (not n.startswith(q), len(n)))   # starts-with first
        out += _row(hit.sort_values("k"), slice(None), "name")
        # typos: compare with the same number of leading words of each name, and with symbols
        n = len(words)
        lead = u["norm"].map(lambda s: " ".join(s.split()[:n]))
        score = lead.map(lambda s: difflib.SequenceMatcher(None, q, s).ratio())
        sym = u["nse_symbol"].str.lower().map(
            lambda s: difflib.SequenceMatcher(None, q.replace(" ", ""), s).ratio() if s else 0)
        best = pd.concat([score, sym], axis=1).max(axis=1)
        # A look-alike NAME ranks above a look-alike SYMBOL: "augoment" is 0.80 to the name
        # Augmont but 0.88 to the symbol AUSOMENT (Ausom Enterprise). Symbol-only typos
        # ("TCSS") still pass the cutoff and come after.
        typo = (u.assign(s=best, ns=score)[best >= TYPO_CUTOFF]
                .sort_values(["ns", "s"], ascending=False).head(5))
        out += [dict(r, why=f"typo match {s:.2f}") for r, s in
                zip(_row(typo, slice(None), ""), typo["s"])]
    return out


def web_candidates(u: pd.DataFrame, text: str) -> list[dict]:
    """Screener + Yahoo search, mapped back to the universe by symbol / BSE code."""
    import requests
    syms = []
    try:
        r = requests.get("https://www.screener.in/api/company/search/", params={"q": text},
                         headers=_UA, timeout=15)
        for hit in (r.json() if r.ok else [])[:5]:
            m = re.match(r"/company/([^/]+)/", str(hit.get("url", "")))
            if m:
                syms.append((m.group(1).upper(), "Screener search"))
    except Exception:
        pass
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                         params={"q": text, "quotesCount": 6, "newsCount": 0},
                         headers=_UA, timeout=15)
        for hit in (r.json().get("quotes", []) if r.ok else []):
            s = str(hit.get("symbol", ""))
            if s.endswith((".NS", ".BO")):
                syms.append((s.rsplit(".", 1)[0].upper(), "Yahoo search"))
    except Exception:
        pass
    out = []
    for s, why in syms:
        out += _row(u, (u["nse_symbol"] == s) | (u["bse_symbol"] == s) | (u["bse_code"] == s),
                    why)
    return out


def candidates(u: pd.DataFrame, text: str, web: bool = True) -> list[dict]:
    found = universe_candidates(u, text)
    if web and not any(c["why"] == "exact" for c in found):
        found += web_candidates(u, text)
    seen, out = set(), []
    for c in found:                               # one line per company, best reason first
        if c["isin"] and c["isin"] not in seen:
            seen.add(c["isin"])
            out.append(c)
    return out[:MAX_SHOWN]


def _line(i: int, c: dict) -> str:
    bse = f" · BSE {c['bse']}" if c["bse"] else ""
    return f"   {i}) {c['name']}  —  {c['symbol']}{bse} · {c['isin']}   [{c['why']}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True, help="comma-separated company names / symbols / ISINs")
    ap.add_argument("--out", default="", help="write ISIN|SYMBOL|NAME per confirmed company")
    ap.add_argument("--dry-run", action="store_true", help="list candidates only; no prompt, no file")
    ap.add_argument("--no-web", action="store_true", help="universe only (skip Screener / Yahoo)")
    a = ap.parse_args()

    names = [n.strip() for n in a.names.split(",") if n.strip()]
    u = load_universe()
    picked, ok = [], True
    for text in names:
        cands = candidates(u, text, web=not a.no_web)
        print(f"\n  '{text}':")
        if not cands:
            print("   NOT FOUND in the company universe, Screener or Yahoo — check the spelling.")
            ok = False
            continue
        for i, c in enumerate(cands, 1):
            print(_line(i, c))
        if a.dry_run:
            continue
        try:
            ans = input(f"   Pick 1-{len(cands)} (Enter = 1, 0 = none of these): ").strip()
        except EOFError:
            ans = "0"                             # no console: never guess
        n = 1 if ans == "" else int(ans) if ans.isdigit() else 0
        if not 1 <= n <= len(cands):
            print("   skipped — nothing will run or be queued for this name.")
            ok = False
            continue
        c = cands[n - 1]
        picked.append(c)
        print(f"   -> {c['name']} ({c['symbol']} / {c['isin']})")

    if a.out and not a.dry_run:
        Path(a.out).write_text("".join(f"{c['isin']}|{c['symbol']}|{c['name']}\n" for c in picked),
                               encoding="utf-8")
    return 0 if ok and (picked or a.dry_run) else 1


if __name__ == "__main__":
    sys.exit(main())
