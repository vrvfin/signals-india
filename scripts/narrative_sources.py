r"""
narrative_sources.py — assembles the SOURCE BUNDLE {doc_id: text} that Gates 2 and 3 and
the quote spine all depend on.

Raw PDFs never survive on Drive (CLAUDE.md retention rule 1, RETAIN_DAYS=2), so documents
are re-fetched from `pdf_url` in the ONE global ledger
`company_repo/_index/processing_queue.parquet`, reusing the existing fetcher via
`company_deep_report._refetch_doc_bytes`. Nothing here writes to the queue, so the live
Phase-2 path is untouched.

Text is cached under the scratch dir, keyed by doc_id, so repeated report runs and the
auditor do not re-download the same transcript.

doc_id format: "<doc_type>_<YYYY-MM-DD>" (e.g. concall_2026-05-27) — stable, human
readable, and what a source footer cites.

Usage:
  python scripts/narrative_sources.py --names LANDMARK --outdir ./_src
  python scripts/narrative_sources.py --names LANDMARK --list        # no downloads
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from narrative_factpack import Store, resolve

# Newest-first per type. Enough history for the quote spine (needs >=4 calls) without
# pulling a decade of filings on every run.
DEFAULT_LIMITS = {"concall": 6, "annual_report": 3, "presentation": 3, "rating": 2,
                  "results": 2}
MAX_TEXT_CHARS = 400_000
NOT_AN_ANNUAL_REPORT = re.compile(
    r"newspaper|scrutini[sz]er|letter to (the )?members|notice|postal ballot|proceedings|"
    r"voting result|book closure|record date", re.I)


def _pdf_text(data: bytes) -> str:
    """PDF bytes -> text. HTML/text payloads (some rating rationales) pass through
    decoded, since not every 'pdf_url' actually serves a PDF."""
    if not data:
        return ""
    if not data[:5].startswith(b"%PDF"):
        try:
            txt = data.decode("utf-8", errors="ignore")
        except Exception:
            return ""
        return re.sub(r"<[^>]+>", " ", txt) if "<" in txt[:2000] else txt
    try:
        import fitz
    except ImportError:
        return ""
    try:
        with fitz.open(stream=data, filetype="pdf") as d:
            return "\n".join(p.get_text() for p in d)
    except Exception:
        return ""


def _clean(t: str, cap: int | None = MAX_TEXT_CHARS) -> str:
    """Collapse the artefacts PDF extraction leaves behind, WITHOUT destroying the exact
    wording — evidence_span matching happens against this text, so words and punctuation
    must survive verbatim. Only whitespace is normalised."""
    t = t.replace(" ", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()[:cap] if cap else t.strip()


def queue_rows(store: Store, isin: str, symbol: str,
               limits: dict[str, int] | None = None) -> pd.DataFrame:
    q = store.by_isin("processing_queue.parquet", isin, symbol)
    if q.empty:
        return q
    q = q[q["status"].astype(str).isin(["done", "superseded"])].copy()
    # Notices filed under the annual-report category are NOT the annual report (Morepen
    # 2026-10-18: a newspaper publication and the AGM scrutinizer's report were read as its
    # two latest "annual reports", so the real one never reached the writer). The queue
    # itself is left as it is — this only chooses what to read.
    if "title" in q.columns:
        notice = (q["doc_type"].astype(str) == "annual_report") & \
            q["title"].astype(str).str.contains(NOT_AN_ANNUAL_REPORT, na=False)
        q = q[~notice]
    q["_d"] = pd.to_datetime(q["announcement_date"], errors="coerce")
    q = q.sort_values("_d", ascending=False)
    lim = limits or DEFAULT_LIMITS
    keep = [g.head(lim.get(str(dt), 1)) for dt, g in q.groupby("doc_type", sort=False)]
    return pd.concat(keep).sort_values("_d", ascending=False) if keep else q.iloc[0:0]


def doc_id_for(row) -> str:
    d = pd.to_datetime(row.get("announcement_date"), errors="coerce")
    stamp = d.strftime("%Y-%m-%d") if pd.notna(d) else "undated"
    return f"{row.get('doc_type', 'doc')}_{stamp}"


def build(store: Store, token: str, cache_dir: Path | None = None,
          limits: dict[str, int] | None = None, log=print) -> tuple[dict, list[dict]]:
    """-> ({doc_id: text}, [manifest rows]). Manifest records every attempt, including
    failures, so a thin bundle is visible rather than silent."""
    r = resolve(store, token)
    if r is None:
        return {}, []
    isin, symbol, name = r
    rows = queue_rows(store, isin, symbol, limits)
    if rows.empty:
        log(f"  no processed documents in the queue for {name}")
        return {}, []

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    from company_deep_report import _refetch_doc_bytes

    sources, manifest = {}, []
    for _, row in rows.iterrows():
        did = doc_id_for(row)
        rec = {"doc_id": did, "doc_type": str(row.get("doc_type")),
               "date": str(row.get("announcement_date")),
               "title": str(row.get("title") or "")[:160],
               "url": str(row.get("pdf_url") or ""), "chars": 0, "status": ""}
        cached = (cache_dir / f"{did}.txt") if cache_dir else None
        if cached and cached.exists():
            txt = cached.read_text(encoding="utf-8", errors="ignore")
            rec["status"] = "cached"
        else:
            data = _refetch_doc_bytes(row)
            if not data:
                rec["status"] = "refetch_failed"
                manifest.append(rec)
                log(f"    {did}: re-fetch FAILED (url dead or unreachable)")
                continue
            raw = _pdf_text(data)
            txt = _clean(raw)
            rec["status"] = "fetched" if txt else "no_text_extracted"
            if cached and txt:
                cached.write_text(txt, encoding="utf-8")
                # Annual reports run past the cap: the auditor's report and the notes
                # to accounts sit in the second half. The FULL text is kept beside the
                # capped one for the company report's passages (2026-10-19); what
                # build() returns — and so what any extractor sends — is unchanged.
                if str(row.get("doc_type")) == "annual_report":
                    (cache_dir / f"{did}.full.txt").write_text(_clean(raw, cap=None),
                                                               encoding="utf-8")
        rec["chars"] = len(txt)
        if not txt:
            rec["status"] = rec["status"] or "empty"
            manifest.append(rec)
            log(f"    {did}: no text extracted (scanned PDF? needs OCR)")
            continue
        sources[did] = txt
        manifest.append(rec)
        log(f"    {did}: {len(txt):,} chars ({rec['status']})")
    return sources, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--outdir", help="write sources.json + per-doc text cache here")
    ap.add_argument("--list", action="store_true",
                    help="list the documents that WOULD be fetched; no downloads")
    a = ap.parse_args()

    store = Store()
    for token in a.names:
        r = resolve(store, token)
        if r is None:
            print(f"could not resolve '{token}'")
            continue
        isin, symbol, name = r
        print(f"\n{name} ({symbol} / {isin})")
        if a.list:
            rows = queue_rows(store, isin, symbol)
            if rows.empty:
                print("  no processed documents in the queue")
                continue
            for _, row in rows.iterrows():
                print(f"  {doc_id_for(row):<28} {str(row.get('title') or '')[:70]}")
            print(f"  {len(rows)} document(s); by type: "
                  f"{rows.groupby('doc_type').size().to_dict()}")
            continue
        out = Path(a.outdir or f"./_src_{symbol}")
        sources, manifest = build(store, token, cache_dir=out)
        (out / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
        ok = sum(1 for m in manifest if m["chars"] > 0)
        print(f"  {ok}/{len(manifest)} document(s) usable — wrote {out}/sources.json")
        for m in manifest:
            if not m["chars"]:
                print(f"    UNUSABLE {m['doc_id']}: {m['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
