r"""
company_narrative_report.py — the orchestrator. Runs the whole four-layer pipeline for
one company and emits the three-part artefact (narrative · forensic · audit) as both
markdown and an HTML deck.

    preflight   narrative_preflight   readiness + integrity; FAIL blocks by default
    Layer A     narrative_factpack    every number, computed, with provenance
    sources     narrative_sources     re-fetch documents for evidence spans
    Layer B     narrative_generate    Gemini writes prose; Gates 1-2 enforce grounding
    Part B      (existing deep dive)  attached via --forensic-md
    Layer C     report_auditor        Cerebras re-validates against source
    Layer D     render_narrative_deck md + html, audit-annotated

Part B is ATTACHED rather than invoked: `company_deep_report.py` writes to Drive and has
its own queue lifecycle, so calling it from here would duplicate side effects. Run it
separately and pass its markdown.

Usage:
  python scripts/company_narrative_report.py --names LANDMARK --dry-run
  python scripts/company_narrative_report.py --names LANDMARK --outdir ./out --open
  python scripts/company_narrative_report.py --names LANDMARK --sections 18 19 \
         --forensic-md company_deepdive_29Jul26.md
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from _extractor_base import find_file, download_bytes, upload_bytes, log
import narrative_factpack as FP
import narrative_generate as GEN
import narrative_preflight as PRE
import narrative_sources as SRC
import render_narrative_deck as RENDER

INDEX_FILE = "narrative_index.parquet"
INDEX_COLS = ["isin", "symbol", "company_name", "report_file", "as_of",
              "facts", "sections", "audit_model", "audit_verified",
              "audit_unsupported", "audit_contradicted", "gate_flagged",
              "preflight_fail", "generated_at"]
# Annual reports are ~400k chars each; sending them all to every section would blow the
# prompt and the quota. Concalls are the evidence base for management claims.
SOURCE_PRIORITY = ("concall", "presentation", "rating", "annual_report")
MAX_SOURCE_DOCS = 3


def _pick_sources(sources: dict[str, str], manifest: list[dict]) -> dict[str, str]:
    """Choose the documents worth sending: newest first within the priority order."""
    by_id = {m["doc_id"]: m for m in manifest}
    ranked = sorted(
        (d for d in sources),
        key=lambda d: (SOURCE_PRIORITY.index(by_id.get(d, {}).get("doc_type", "rating"))
                       if by_id.get(d, {}).get("doc_type") in SOURCE_PRIORITY else 9,
                       -len(by_id.get(d, {}).get("date", ""))),
    )
    return {d: sources[d] for d in ranked[:MAX_SOURCE_DOCS]}


def update_index(store: FP.Store, rec: dict) -> str:
    drive, folder = store.drive, store.folder(FP.IDX)
    existing = pd.DataFrame(columns=INDEX_COLS)
    fid = find_file(drive, folder, INDEX_FILE)
    if fid:
        try:
            existing = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
        except Exception as e:
            log(f"  WARNING: could not read {INDEX_FILE} ({str(e)[:70]})")
    for c in INDEX_COLS:
        if c not in existing.columns:
            existing[c] = None
    merged = pd.concat([existing, pd.DataFrame([rec])], ignore_index=True)[INDEX_COLS]
    merged = merged.drop_duplicates(subset=["isin", "report_file"], keep="last")
    buf = io.BytesIO()
    merged.to_parquet(buf, index=False)
    upload_bytes(drive, folder, INDEX_FILE, buf.getvalue(), fid)
    return f"{INDEX_FILE}: {len(merged)} rows"


def run_one(store: FP.Store, token: str, args) -> dict | None:
    t0 = time.time()
    log(f"\n{'=' * 74}\n{token}\n{'=' * 74}")

    # ---- preflight ---------------------------------------------------------
    log("[1/6] preflight")
    rep = PRE.run(store, token)
    if rep is None:
        log(f"  could not resolve '{token}'")
        return None
    co = rep["company"]
    rc, ic = rep["readiness_counts"], rep["integrity_counts"]
    log(f"  {co['name']} ({co['symbol']}) — sections {rc['READY']} ready / "
        f"{rc['FETCHABLE']} fetchable / {rc['BLOCKED']} blocked; "
        f"integrity {ic['PASS']} pass / {ic['WARN']} warn / {ic['FAIL']} fail")
    for ch in rep["integrity"]:
        if ch["status"] in ("FAIL", "WARN"):
            log(f"    [{ch['status']}] {ch['name']}: {ch['detail'][:110]}")
    if not rep["publishable"] and not args.ignore_preflight:
        log("  ABORT: integrity FAIL. Fix the data or pass --ignore-preflight to "
            "publish anyway (the failure is recorded on the report).")
        return None

    # ---- auto-fetch what Drive is missing ---------------------------------
    # The report should not simply REPORT a gap it can close. Preflight already knows
    # which sections are FETCHABLE (the pipeline can get the documents, this company
    # just has too few), so close those before building rather than rendering
    # DATA_MISSING and telling the user to run a command themselves.
    if args.fetch_missing:
        fetchable = [r for r in rep["readiness"] if r["state"] == "FETCHABLE"]
        if not fetchable:
            log("[1b] auto-fetch: nothing fetchable — Drive already has what it can")
        else:
            log(f"[1b] auto-fetch: {len(fetchable)} section(s) short of documents "
                f"— pulling from Screener/BSE/NSE")
            for r in fetchable[:1]:      # one backfill call covers all doc types
                log(f"     {r['remedy']}")
            try:
                import subprocess
                # backfill_company_docs takes --token (name / NSE / BSE / ISIN), NOT
                # --names. It already resolves ANY company through the universe, so
                # nothing here is company-specific.
                subprocess.run([sys.executable,
                                str(Path(_HERE) / "backfill_company_docs.py"),
                                "--token", token], check=False, timeout=1800)
                log("     backfill done — re-running preflight")
            except Exception as e:
                log(f"     backfill failed ({str(e)[:120]}) — continuing with what exists")
            # New documents are useless until they are extracted, so run the two
            # extractors that feed the document-backed sections.
            for mod, label in (("extract_structure", "structure (s1/3/4/6/9/23)"),
                               ("extract_mgmt_quotes", "quotes (s20)")):
                try:
                    import subprocess
                    log(f"     extracting {label}")
                    subprocess.run([sys.executable, str(Path(_HERE) / f"{mod}.py"),
                                    "--names", token, "--cache",
                                    str(Path(args.cache or (Path(args.outdir) /
                                        f"_src_{co['symbol']}")))],
                                   check=False, timeout=2400)
                except Exception as e:
                    log(f"     {mod} failed ({str(e)[:110]})")
            # Invalidate ONLY the tables the extractors just rewrote — clearing the whole
            # cache forced a re-read of company_facts too, and a transient failure of
            # that read cached an empty frame and broke resolve() for the rest of the run.
            for _p, _n in list(store._files.keys()):
                if _n in ("company_structure.parquet", "mgmt_quotes.parquet",
                          "processing_queue.parquet", "ratings.parquet"):
                    store._files.pop((_p, _n), None)
            rep = PRE.run(store, token) or rep
            rc = rep["readiness_counts"]
            log(f"     after fetch: {rc['READY']} ready / {rc['FETCHABLE']} fetchable / "
                f"{rc['BLOCKED']} blocked")

    # ---- Layer A ----------------------------------------------------------
    log("[2/6] fact pack")
    pack = FP.build(store, token)
    if pack is None:
        return None
    d = pack.to_dict()
    log(f"  {len(d['facts'])} facts · {len(d['tables'])} tables · "
        f"{len(d['coverage_gaps'])} gaps")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"factpack_{co['symbol']}.json").write_text(
        json.dumps(d, indent=2), encoding="utf-8")

    # ---- sources ----------------------------------------------------------
    log("[3/6] source documents")
    cache = Path(args.cache or (outdir / f"_src_{co['symbol']}"))
    sources, manifest = ({}, [])
    if not args.no_sources:
        sources, manifest = SRC.build(store, token, cache_dir=cache, log=log)
    chosen = _pick_sources(sources, manifest)
    log(f"  {len(sources)} fetched, {len(chosen)} sent to the model: {list(chosen)}")
    (outdir / f"sources_{co['symbol']}.json").write_text(
        json.dumps(chosen), encoding="utf-8")

    if args.dry_run:
        log("[4/6] DRY RUN — no generation, no audit, no upload")
        secs = sorted({f["section"] for f in d["facts"]})
        log(f"  would generate {len(secs)} section(s): {secs}")
        log(f"  would then audit and render to {outdir}")
        return {"dry_run": True, "company": co}

    # ---- Layer B ----------------------------------------------------------
    log("[4/6] narrative generation (Gemini)")
    try:
        nar = GEN.generate(d, chosen, args.sections, log=log)
    except Exception as e:
        log(f"  generation FAILED: {str(e)[:200]}")
        if args.debug:
            traceback.print_exc()
        return None
    flagged = nar.get("sections_with_unresolved_gate_failures", 0)
    log(f"  {len(nar['sections'])} section(s); {flagged} with unresolved gate failures")
    if args.forensic_md:
        p = Path(args.forensic_md)
        if p.exists():
            nar["forensic_report"] = p.read_text(encoding="utf-8")
            log(f"  attached forensic report: {p.name} "
                f"({len(nar['forensic_report']):,} chars)")
        else:
            log(f"  WARNING: --forensic-md {p} not found; Part B will be empty")
    (outdir / f"narrative_{co['symbol']}.json").write_text(
        json.dumps(nar, indent=2), encoding="utf-8")

    # ---- Layer C ----------------------------------------------------------
    audit = None
    if args.skip_audit:
        log("[5/6] audit SKIPPED (--skip-audit) — Part C will say so")
    else:
        log("[5/6] independent audit")
        try:
            from report_auditor import Adjudicator, audit_report
            adj = Adjudicator(prefer_alt=not args.force_gemini_audit)
            log(f"  adjudicator: {adj.model}"
                + ("  [DEGRADED — same family as the generator]" if adj.degraded
                   else "  [independent family]"))
            secs = [(str(s.get("id")), str(s.get("title", "")),
                     " ".join(str(s.get(k, "")) for k in ("takeaway", "body")))
                    for s in nar["sections"]
                    if (s.get("body") or s.get("takeaway"))]
            # The fact pack goes to the auditor as an evidence table. Without it every
            # computed figure comes back UNSUPPORTED, because those numbers live in
            # Screener statements rather than in any filing in the document bundle.
            audit = audit_report(adj, secs, chosen, factpack=d)
            s = audit["summary"]
            if not audit.get("ran"):
                log(f"  AUDIT DID NOT RUN — every section failed adjudication: "
                    f"{audit.get('failure_reason', '')[:150]}")
                log(f"  the report will be marked UNAUDITED")
            else:
                log(f"  {s['verified']}/{s['total']} verified · "
                    f"{s['unsupported']} unsupported · {s['contradicted']} contradicted"
                    + (f" · {s['sections_failed']} section(s) FAILED to audit"
                       if s.get("audit_failed") else ""))
            (outdir / f"audit_{co['symbol']}.json").write_text(
                json.dumps(audit, indent=2), encoding="utf-8")
        except Exception as e:
            log(f"  audit FAILED: {str(e)[:200]} — publishing WITHOUT an audit")
            if args.debug:
                traceback.print_exc()

    # ---- Layer D ----------------------------------------------------------
    log("[6/6] render")
    stamp = datetime.now().strftime("%d%b%y")
    md_p = outdir / f"company_narrative_{co['symbol']}_{stamp}.md"
    html_p = outdir / f"company_narrative_{co['symbol']}_{stamp}.html"
    md = RENDER.render_markdown(d, nar, audit)
    md_p.write_text(md, encoding="utf-8")
    html_doc = RENDER.render_html(d, nar, audit)
    html_p.write_text(html_doc, encoding="utf-8")
    log(f"  {md_p.name} ({len(md):,} chars)")
    log(f"  {html_p.name}")

    # ---- Drive ------------------------------------------------------------
    if args.upload:
        try:
            folder = store.folder(f"company_repo/{co['isin']}")
            fid = find_file(store.drive, folder, md_p.name)
            # (drive, folder_id, filename, data, mimetype, existing_id) — the id is the
            # SIXTH arg; passing it fifth silently lands it in `mimetype`.
            upload_bytes(store.drive, folder, md_p.name, md.encode("utf-8"),
                         "text/markdown", existing_id=fid)
            asum = (audit or {}).get("summary", {})
            log("  uploaded; " + update_index(store, {
                "isin": co["isin"], "symbol": co["symbol"],
                "company_name": co["name"], "report_file": md_p.name,
                "as_of": d["as_of_utc"], "facts": len(d["facts"]),
                "sections": len(nar["sections"]),
                "audit_model": (audit or {}).get("model", ""),
                "audit_verified": asum.get("verified", 0),
                "audit_unsupported": asum.get("unsupported", 0),
                "audit_contradicted": asum.get("contradicted", 0),
                "gate_flagged": flagged,
                "preflight_fail": ic["FAIL"],
                "generated_at": datetime.now().isoformat(timespec="seconds")}))
        except Exception as e:
            log(f"  upload FAILED: {str(e)[:160]} (local files are intact)")
    else:
        log("  --upload not set; nothing written to Drive")

    # ---- local copies (same destinations as run_deepdive.bat) --------------
    # CI has no Obsidian vault, so this is opt-in rather than automatic; the mail
    # below is what makes a CI run reach the user.
    if args.local_render:
        for env_key, default in (("OBSIDIAN_VAULT", r"D:\EMA_Screener\Obsidian"),
                                 ("REPORTS_DIR",
                                  r"D:\EMA_Screener\Reports\signals-india")):
            dest = Path(os.environ.get(env_key, default))
            try:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / md_p.name).write_text(md, encoding="utf-8")
                (dest / html_p.name).write_text(html_doc, encoding="utf-8")
                log(f"  local copy -> {dest / md_p.name}")
            except Exception as e:
                log(f"  local copy to {dest} failed: {str(e)[:110]}")

    # ---- mail --------------------------------------------------------------
    if args.mail:
        try:
            from mailer import send_email
            asum = (audit or {}).get("summary", {})
            ran = (audit or {}).get("ran", True)
            audit_line = ("<b style='color:#c33'>AUDIT DID NOT RUN</b> — no claim was "
                          "independently checked."
                          if audit and not ran else
                          f"Audit: <b>{asum.get('verified', 0)}/{asum.get('total', 0)}"
                          f"</b> claims verified · {asum.get('unsupported', 0)} "
                          f"unsupported · {asum.get('contradicted', 0)} contradicted"
                          if audit else "Audit: not run for this copy.")
            body = (
                f"<h2>{co['name']} — narrative report</h2>"
                f"<p>{co['symbol']} · {co['isin']} · data current to "
                f"{d['as_of_utc'][:10]}</p>"
                f"<p>{audit_line}</p>"
                f"<p>{len(d['facts'])} facts · {len(d['tables'])} tables · "
                f"{len(nar['sections'])} sections · {flagged} section(s) with "
                f"unresolved grounding flags</p>"
                f"<p>Full report attached (.md and .html). Open the .html for the "
                f"charts and source footers.</p>"
                f"<hr><pre style='white-space:pre-wrap;font-size:12px'>"
                f"{md[:4000].replace('<', '&lt;')}…</pre>")
            ok = send_email(
                f"Narrative report — {co['name']} ({co['symbol']})",
                body,
                attachments=[(md_p.name, md.encode("utf-8"), "octet-stream"),
                             (html_p.name, html_doc.encode("utf-8"), "octet-stream")])
            log("  mailed" if ok else "  mail SKIPPED (GMAIL_USER / "
                                      "GMAIL_APP_PASSWORD not set)")
        except Exception as e:
            log(f"  mail FAILED: {str(e)[:160]}")

    if args.open:
        webbrowser.open(html_p.resolve().as_uri())
    log(f"done in {time.time() - t0:.0f}s")
    return {"company": co, "md": str(md_p), "html": str(html_p),
            "facts": len(d["facts"]), "flagged": flagged,
            "audit": (audit or {}).get("summary")}


# ── narrative_queue: same principle as deep_dive_queue ─────────────────────────
# A scheduled run has no --names, so it DRAINS this queue — exactly how
# company_deep_report.py works with no args. The queue is a separate report-request
# ledger (like deep_dive_queue, the one allowed non-document queue), NOT the global
# document queue. Dedup-on-write is the correctness guarantee; a token already pending
# or done is never added twice.
NQUEUE = "company_repo/_index/narrative_queue.parquet"
NQUEUE_COLS = ["token", "status", "added_at", "done_at", "error"]


def _load_nqueue(store: FP.Store) -> pd.DataFrame:
    fid = find_file(store.drive, store.folder(FP.IDX), "narrative_queue.parquet")
    if not fid:
        return pd.DataFrame(columns=NQUEUE_COLS)
    try:
        df = pd.read_parquet(io.BytesIO(download_bytes(store.drive, fid)))
        for c in NQUEUE_COLS:
            if c not in df.columns:
                df[c] = None
        return df
    except Exception as e:
        log(f"  WARNING: narrative_queue unreadable ({str(e)[:70]}) — treating empty")
        return pd.DataFrame(columns=NQUEUE_COLS)


def _save_nqueue(store: FP.Store, df: pd.DataFrame):
    fid = find_file(store.drive, store.folder(FP.IDX), "narrative_queue.parquet")
    buf = io.BytesIO()
    df[NQUEUE_COLS].to_parquet(buf, index=False)
    upload_bytes(store.drive, store.folder(FP.IDX), "narrative_queue.parquet",
                 buf.getvalue(), "application/octet-stream", existing_id=fid)


def enqueue_narrative(store: FP.Store, tokens: list[str]) -> int:
    """Add pending rows, skipping any token already pending or done. Returns count added."""
    df = _load_nqueue(store)
    seen = (set(df[df["status"].astype(str).isin(["pending", "done"])]["token"].astype(str))
            if not df.empty else set())
    toks = [t.strip() for t in dict.fromkeys(tokens) if t.strip() and t.strip() not in seen]
    if not toks:
        return 0
    new = pd.DataFrame([{"token": t, "status": "pending",
                         "added_at": datetime.now().isoformat(timespec="seconds")}
                        for t in toks])
    _save_nqueue(store, pd.concat([df, new], ignore_index=True))
    return len(toks)


def _mark_nqueue(store: FP.Store, token: str, status: str, error: str = ""):
    df = _load_nqueue(store)
    if df.empty:
        return
    m = df["token"].astype(str) == str(token)
    if not m.any():
        return
    df.loc[m, "status"] = status
    df.loc[m, "done_at"] = datetime.now().isoformat(timespec="seconds")
    if error:
        df.loc[m, "error"] = error[:200]
    _save_nqueue(store, df)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not required: no --names DRAINS narrative_queue.parquet, mirroring how
    # company_deep_report.py runs with no args on a scheduled CI pass.
    ap.add_argument("--names", nargs="*", default=None,
                    help="ISIN / symbol / name fragment. Omit to drain the queue.")
    ap.add_argument("--add", nargs="+", default=None,
                    help="enqueue these tokens for the next scheduled run, then exit")
    ap.add_argument("--outdir", default="./_narrative")
    ap.add_argument("--cache", default="")
    ap.add_argument("--sections", nargs="*", type=int, default=None)
    ap.add_argument("--forensic-md", default="",
                    help="existing company_deepdive_*.md to attach as Part B")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight + fact pack + sources only; no LLM, no writes")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--force-gemini-audit", action="store_true",
                    help="audit on Gemini (DEGRADED: correlated with the generator)")
    ap.add_argument("--no-sources", action="store_true",
                    help="skip document re-fetch; qualitative claims become impossible")
    ap.add_argument("--ignore-preflight", action="store_true",
                    help="publish even when integrity checks FAIL")
    ap.add_argument("--upload", action="store_true", help="upload md + index to Drive")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--fetch-missing", action="store_true",
                    help="before building, pull any documents Drive is missing "
                         "(Screener/BSE/NSE via backfill_company_docs) and extract them")
    ap.add_argument("--mail", action="store_true",
                    help="email the report (HTML inline + .md and .html attached) to "
                         "NOTIFY_EMAIL — works identically local or in CI")
    ap.add_argument("--local-render", action="store_true",
                    help="also write the report to the Obsidian vault and Reports dir, "
                         "the same places run_deepdive.bat puts a deep dive")
    a = ap.parse_args()

    # Check delivery BEFORE a 20-30 minute run, not after it. mailer.send_email skips
    # silently when the credentials are absent, so --mail would otherwise appear to
    # work and quietly deliver nothing.
    if a.mail:
        missing = [k for k in ("GMAIL_USER", "GMAIL_APP_PASSWORD")
                   if not os.environ.get(k)]
        if missing:
            log(f"WARNING: --mail was requested but {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not set in this environment.")
            log("         The report will still be written locally, but NO EMAIL WILL "
                "BE SENT.")
            log("         Set them in .env for local runs; in CI they come from "
                "repository secrets.")
            if not os.environ.get("NOTIFY_EMAIL"):
                log("         NOTIFY_EMAIL is also unset — there is no recipient.")

    store = FP.Store()

    # --add: enqueue and exit (the scheduled run will pick these up).
    if a.add:
        n = enqueue_narrative(store, a.add)
        log(f"enqueued {n} token(s) to narrative_queue "
            f"({len(a.add) - n} already pending/done)")
        return 0

    # No --names -> DRAIN the queue, same as company_deep_report.py with no args.
    draining = not a.names
    if draining:
        q = _load_nqueue(store)
        pending = (q[q["status"].astype(str) == "pending"]["token"].astype(str).tolist()
                   if not q.empty else [])
        if not pending:
            log("narrative_queue is empty — nothing to do. Add companies with "
                "`--add TOKEN` or pass --names for an ad-hoc run.")
            return 0
        log(f"draining narrative_queue: {len(pending)} pending "
            f"({', '.join(pending[:8])}{'...' if len(pending) > 8 else ''})")
        tokens = pending
    else:
        tokens = a.names

    results, failures = [], 0
    for token in tokens:
        try:
            r = run_one(store, token, a)
        except Exception as e:
            log(f"  UNHANDLED for '{token}': {str(e)[:200]}")
            if a.debug:
                traceback.print_exc()
            r = None
        if r is None:
            failures += 1
            if draining:
                _mark_nqueue(store, token, "error", "run_one returned None or raised")
        else:
            results.append(r)
            if draining:
                _mark_nqueue(store, token, "done")

    log(f"\n{'=' * 74}")
    for r in results:
        if r.get("dry_run"):
            log(f"  {r['company']['symbol']}: dry run OK")
            continue
        au = r.get("audit") or {}
        log(f"  {r['company']['symbol']}: {r['facts']} facts, "
            f"{r['flagged']} gate-flagged section(s), audit "
            f"{au.get('verified', '-')}/{au.get('total', '-')} verified")
        log(f"    {r['md']}")
    if failures:
        log(f"  {failures} company/companies failed")
    return 1 if failures and not results else 0


if __name__ == "__main__":
    sys.exit(main())
