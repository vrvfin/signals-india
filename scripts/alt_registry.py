r"""
alt_registry.py — N9. Sector-gated INDEPENDENT datasets for sections 16, 17 and 25.

Sections 16/17 exist to test a management claim against data the company does not
control. That only works if the dataset is genuinely independent AND genuinely reachable.

WHAT WAS ACTUALLY VERIFIED (probed 2026-07-30, not assumed):
    vahan.parivahan.gov.in            ConnectionReset   -> NOT reachable
    analytics.parivahan.gov.in        HTTP 403          -> NOT reachable
    cea.nic.in dashboard              HTTP 200 (HTML)   -> reachable, unstructured
    api.data.gov.in                   HTTP 400          -> reachable, needs an API key
    RBI notification/press RSS        working            -> already in alt_sources.py
    openFDA drug enforcement          working            -> already in alt_sources.py

So vehicle registrations — the single most useful series for an auto retailer, and the one
the source deck leaned on hardest — CANNOT be fetched programmatically. Note that the deck
itself says its Vahan figures are "the author's own extraction and aggregation from the raw
files": that was manual work, not an API.

The honest design is therefore a registry with THREE fetch modes, each labelled, and a
manual-intake path that makes the manual route first-class rather than a gap:

  "api"     fetched live (openFDA, RBI RSS via alt_sources)
  "keyed"   reachable but needs a credential this project does not hold
  "manual"  no machine access; reads a CSV you drop in the intake folder

A dataset in "manual" mode with no file present yields DATA_MISSING — never an estimate.

Manual intake layout (CSV, first row headers):
    <ALT_INTAKE_DIR>/<dataset_id>/*.csv     with columns: period,series,value
    e.g. D:\EMA_Screener\alt_intake\vahan_registrations\bydindia.csv
         period,series,value
         2026-01,BYD India,304

Usage:
  python scripts/alt_registry.py --list
  python scripts/alt_registry.py --sector auto
  python scripts/alt_registry.py --dataset vahan_registrations --series "BYD India"
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ALT_INTAKE_DIR = Path(os.environ.get("ALT_INTAKE_DIR",
                                     r"D:\EMA_Screener\alt_intake"))

# ---------------------------------------------------------------- registry ----
# Each entry: sector keywords -> dataset. `limits` is REQUIRED and feeds section 17's
# "what this does not tell you" — a scorecard without stated limits is a trap.
REGISTRY: list[dict] = [
    {
        "id": "vahan_registrations",
        "name": "Vahan vehicle registrations (Ministry of Road Transport & Highways)",
        "sector_kw": ("auto", "automobile", "vehicle", "dealership", "tyre", "auto anc",
                      "two wheeler", "four wheeler", "commercial vehicle"),
        "mode": "manual",
        "grain": "monthly, by maker entity, all-India",
        "why": "Registrations are recorded by the government, not the company, so they "
               "test a retailer's or OEM's volume narrative independently.",
        "limits": [
            "Measures REGISTRATIONS, not dealer sales — it captures the whole national "
            "market for an OEM, not any one retailer's share of it.",
            "Registration lags retail by days to weeks, so month boundaries are soft.",
            "Some maker entities bundle categories (e.g. two-wheelers with cars), and "
            "some manufacturers register under multiple entities.",
            "It says nothing about a dealer's own share, margin per unit, or after-sales.",
        ],
        "howto": "Vahan has no public API (connection reset / HTTP 403 as at "
                 "2026-07-30). Export the maker-wise monthly report from the Vahan "
                 "dashboard and save it as CSV under the intake folder.",
    },
    {
        "id": "fda_enforcement",
        "name": "US FDA drug enforcement / recalls (openFDA)",
        "sector_kw": ("pharma", "healthcare", "life scien", "biotech", "drug", "api "),
        "mode": "api",
        "grain": "per recall event, by recalling firm",
        "why": "Recalls are published by the regulator and are a direct, dated check on "
               "quality claims for exporters to the US.",
        "limits": [
            "US market only — silent on India, EU or RoW operations.",
            "Keyed on recalling-firm name, so subsidiary naming can miss matches.",
            "A recall is an event, not a rate: absence is not evidence of quality.",
        ],
        "howto": "Fetched live via alt_sources.fda_recalls(); no key required.",
    },
    {
        "id": "rbi_circulars",
        "name": "RBI notifications and press releases",
        "sector_kw": ("financ", "bank", "nbfc", "insur", "housing finance", "lending",
                      "amc", "broker", "capital market"),
        "mode": "api",
        "grain": "per circular, sector-wide",
        "why": "Regulatory change is exogenous to the lender and often explains a "
               "growth or margin break better than management commentary does.",
        "limits": [
            "Sector-wide, not company-specific — it never proves company impact.",
            "Headline/date only; the operative detail sits in the linked document.",
        ],
        "howto": "Fetched live via alt_sources.rbi_circulars(); no key required.",
    },
    {
        "id": "cea_generation",
        "name": "Central Electricity Authority generation and capacity dashboard",
        "sector_kw": ("power", "utility", "electricity", "renewable", "thermal",
                      "transmission"),
        "mode": "keyed",
        "grain": "monthly, by station and fuel",
        "why": "Plant-level generation is reported to the regulator, testing "
               "utilisation and PLF claims independently.",
        "limits": [
            "Published as HTML/PDF dashboards, not a structured feed — needs a parser "
            "per report format.",
            "Station naming does not map cleanly to listed entities.",
        ],
        "howto": "cea.nic.in returns HTTP 200 HTML but has no stable structured "
                 "endpoint; a per-report parser or manual CSV export is required.",
    },
    {
        "id": "data_gov_in",
        "name": "data.gov.in open datasets (IIP, trade, agriculture, and others)",
        "sector_kw": ("*",),
        "mode": "keyed",
        "grain": "varies by resource",
        "why": "A broad fallback when a sector has no dedicated independent series.",
        "limits": [
            "Requires an api.data.gov.in key, which this project does not hold.",
            "Dataset quality and update cadence vary sharply by resource.",
        ],
        "howto": "Set DATA_GOV_API_KEY and give a resource id. Verified reachable "
                 "(HTTP 400 without params), not yet wired.",
    },
]


def _blob(row) -> str:
    if isinstance(row, str):
        return row.lower()
    return " ".join(str(row.get(k, "")) for k in
                    ("macro_sector", "sector", "industry", "subsector", "peer_group",
                     "name")).lower()


def for_sector(row) -> list[dict]:
    """Datasets relevant to a company row (or a plain sector string). Generic
    ("*") datasets come last so a sector-specific series is always preferred."""
    text = _blob(row)
    specific = [d for d in REGISTRY
                if d["sector_kw"] != ("*",)
                and any(k in text for k in d["sector_kw"])]
    generic = [d for d in REGISTRY if d["sector_kw"] == ("*",)]
    return specific + generic


def get(dataset_id: str) -> dict | None:
    return next((d for d in REGISTRY if d["id"] == dataset_id), None)


# ------------------------------------------------------------ manual intake ---
_PERIOD_RE = re.compile(r"^(\d{4})[-/](\d{1,2})$")


def read_manual(dataset_id: str, intake: Path | None = None) -> dict:
    """Read every CSV under <intake>/<dataset_id>/ into
    {series_name: [(period, value)]} sorted by period.

    Returns `{}` when the folder is absent or empty — the caller must render
    DATA_MISSING rather than substituting anything.
    """
    base = (intake or ALT_INTAKE_DIR) / dataset_id
    if not base.exists():
        return {}
    series: dict[str, list[tuple[str, float]]] = {}
    for f in sorted(base.glob("*.csv")):
        try:
            with f.open(newline="", encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    keys = {k.strip().lower(): v for k, v in r.items() if k}
                    per = str(keys.get("period", "")).strip()
                    name = str(keys.get("series", "") or f.stem).strip()
                    raw = str(keys.get("value", "")).replace(",", "").strip()
                    if not _PERIOD_RE.match(per):
                        continue
                    try:
                        val = float(raw)
                    except ValueError:
                        continue
                    series.setdefault(name, []).append((per, val))
        except Exception:
            continue
    return {k: sorted(v) for k, v in series.items()}


def series_block(dataset_id: str, series_name: str | None = None,
                 intake: Path | None = None) -> dict:
    """A section-16/17-ready block: the series, its methodology, and its limits.
    `status` is always one of available / no_data / not_fetchable."""
    d = get(dataset_id)
    if d is None:
        return {"status": "unknown_dataset", "dataset": dataset_id}
    out = {"dataset": dataset_id, "name": d["name"], "mode": d["mode"],
           "grain": d["grain"], "why_independent": d["why"], "limits": d["limits"],
           "howto": d["howto"], "series": {}, "status": ""}
    if d["mode"] == "manual":
        data = read_manual(dataset_id, intake)
        if series_name:
            data = {k: v for k, v in data.items()
                    if k.lower() == series_name.lower()}
        out["series"] = {k: [{"period": p, "value": v} for p, v in rows]
                         for k, rows in data.items()}
        out["status"] = "available" if data else "no_data"
        if not data:
            out["missing_reason"] = (
                f"no CSV found under {(intake or ALT_INTAKE_DIR) / dataset_id}. "
                f"{d['howto']}")
    elif d["mode"] == "api":
        out["status"] = "available"
        out["note"] = "fetch at report time via alt_sources"
    else:
        out["status"] = "not_fetchable"
        out["missing_reason"] = d["howto"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sector", help="sector or company-name text to match")
    ap.add_argument("--dataset")
    ap.add_argument("--series")
    ap.add_argument("--intake", help="override the intake dir")
    a = ap.parse_args()
    intake = Path(a.intake) if a.intake else None

    if a.list or (not a.sector and not a.dataset):
        print(f"intake dir: {intake or ALT_INTAKE_DIR}"
              f"  ({'exists' if (intake or ALT_INTAKE_DIR).exists() else 'MISSING'})\n")
        for d in REGISTRY:
            print(f"  {d['id']:<22} [{d['mode']:<6}] {d['name']}")
            print(f"  {'':22}  grain: {d['grain']}")
            if d["mode"] == "manual":
                got = read_manual(d["id"], intake)
                print(f"  {'':22}  intake: "
                      + (f"{len(got)} series, "
                         f"{sum(len(v) for v in got.values())} rows" if got
                         else "NO DATA — " + d["howto"][:80]))
        print("\nmode: api = live fetch · keyed = needs a credential · "
              "manual = drop a CSV in the intake folder")
        return 0

    if a.sector:
        ds = for_sector(a.sector)
        print(f"datasets for '{a.sector}': "
              + (", ".join(d["id"] for d in ds) if ds else "none"))
        for d in ds:
            print(f"\n  {d['id']} [{d['mode']}] — {d['name']}")
            print(f"    why independent: {d['why']}")
            for lim in d["limits"]:
                print(f"    limit: {lim}")
        return 0

    import json
    print(json.dumps(series_block(a.dataset, a.series, intake), indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
