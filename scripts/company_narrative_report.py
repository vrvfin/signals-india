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
import format_deepdive_pdf as FMT     # PDF + file naming shared with the deep dive

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
    # (drive, folder, name, data, MIMETYPE, existing_id): fid used to sit in the mimetype
    # slot — None for a first write -> "'NoneType' object has no attribute 'split'", so
    # narrative_index.parquet was never created (checked on Drive 2026-10-03: absent).
    upload_bytes(drive, folder, INDEX_FILE, buf.getvalue(), "application/octet-stream",
                 existing_id=fid)
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
    # NOTE: no abort here. The user's standing rule is check -> if missing, FETCH ->
    # then judge. Aborting on the first preflight made the auto-fetch below unreachable
    # for exactly the companies that needed it. The publishable verdict is taken AFTER
    # the fetch pass re-runs preflight.

    # ---- auto-fetch what Drive is missing ---------------------------------
    # The report should not simply REPORT a gap it can close. Preflight already knows
    # which sections are FETCHABLE (the pipeline can get the documents, this company
    # just has too few), so close those before building rather than rendering
    # DATA_MISSING and telling the user to run a command themselves.
    if args.fetch_missing:
        # FINANCIALS FIRST. statements/<SYM>.parquet backs sections 5/8/18/19/22 — the
        # deterministic core. The nightly Screener sweep covers 5,381 companies, but a
        # name outside it had NO on-demand path and simply rendered those sections
        # empty. Pull it here so "any company" really means any company.
        # Also when the LATEST DUE QUARTER is missing (preflight quarter_status marks the
        # check `due`) — the same rule the fundamentals job's gap scan uses, applied to
        # this one company now instead of at the next scan. A company whose latest
        # quarter is stored is NOT refetched, however long ago it was downloaded.
        # --symbols is a partial run, so summary.parquet is upserted, not replaced.
        st_now = store.parquet(f"{FP.FUND}/statements", f"{co['symbol']}.parquet")
        due = [ch["id"].split(".", 1)[1] for ch in rep["integrity"] if ch.get("due")]
        refetched = False
        if st_now.empty or due:
            log(f"[1a] no fundamentals/statements/{co['symbol']}.parquet — "
                f"fetching financials from Screener" if st_now.empty else
                f"[1a] latest due quarter missing ({', '.join(due)}) — "
                f"pulling {co['symbol']} from Screener")
            try:
                import subprocess
                subprocess.run([sys.executable,
                                str(Path(_HERE) / "ingest_fundamentals.py"),
                                "--symbols", co["symbol"]], check=False, timeout=900)
                refetched = True
                store._files.pop((f"{FP.FUND}/statements",
                                  f"{co['symbol']}.parquet"), None)
                store._files.pop((FP.FUND, "summary.parquet"), None)
                got = store.parquet(f"{FP.FUND}/statements", f"{co['symbol']}.parquet")
                log(f"     statements now: {len(got)} row(s)"
                    if not got.empty else
                    "     still empty — Screener has no statements for this symbol")
            except Exception as e:
                log(f"     financials fetch failed ({str(e)[:110]})")

        fetchable = [r for r in rep["readiness"] if r["state"] == "FETCHABLE"]
        if not fetchable:
            log("[1b] auto-fetch: nothing fetchable — Drive already has what it can")
            if refetched:
                # The [1b] branch re-runs preflight; this one must too, or the verdict
                # below still sees the pre-refresh freshness FAIL.
                rep = PRE.run(store, token) or rep
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

    # ---- verdict, taken only AFTER fetching had its chance ------------------
    if not rep["publishable"] and not args.ignore_preflight:
        ic = rep["integrity_counts"]
        log(f"  ABORT: integrity still FAILs after the fetch pass "
            f"({ic['FAIL']} check(s)). Pass --ignore-preflight to publish anyway "
            f"(the failure is recorded on the report).")
        return None

    # ---- Layer A ----------------------------------------------------------
    log("[2/6] fact pack")
    pack = FP.build(store, token)
    if pack is None:
        return None
    d = pack.to_dict()
    # A due quarter Screener still does not have: build on the previous quarter and SAY
    # so at the top of the report (statements + summary carry the same sentence).
    d["data_notes"] = list(dict.fromkeys(ch["detail"] for ch in rep["integrity"]
                                         if ch.get("due")))
    for n in d["data_notes"]:
        log(f"  DATA NOTE on the report: {n}")
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
            # Gemini by default (user 2026-10-03): the Cerebras adjudicator failed every
            # call for 14 days (HTTP 402 payment required), so every story was UNAUDITED.
            # --alt-audit goes back to Cerebras/Groq once that account works again.
            adj = Adjudicator(prefer_alt=args.alt_audit and not args.force_gemini_audit)
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
            if getattr(adj, "_gem", None) is not None:    # Gemini audit: log its models
                from model_registry import record_usage
                record_usage(adj._gem, "narrative_audit", "narrative", log=log)
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
        store.refresh()      # fresh connection: the first one is stale after the LLM phases
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

    # ---- PDF: what the mail carries and what opens locally ------------------
    # Same formatter (and look) as the deep dive. Files a person receives are named
    # company _ report type _ date (FMT.report_basename); Drive keeps md_p.name.
    base = FMT.report_basename(co["name"], "Narrative")
    # The HTML a person receives (mail fallback, local copy) is the SAME styled page the PDF
    # is printed from — the render_html page (html_p, kept for the CI artefacts) read
    # poorly (user 2026-10-03: wide tables cut off, one figure card per line).
    styled_html = FMT._build_full_html(co["name"], co["symbol"], co["isin"], md,
                                       title="Narrative")
    pdf_p, pdf_err = None, ""
    if args.mail or args.local_render or args.open:
        try:
            pdf_p = outdir / f"{base}.pdf"
            pdf_p.write_bytes(FMT.md_to_pdf(md, co["name"], co["symbol"], co["isin"],
                                            title="Narrative"))
            log(f"  {pdf_p.name}")
        except Exception as e:
            pdf_p, pdf_err = None, str(e)[:120]
            log(f"  pdf FAILED ({pdf_err}) — mail and open fall back to the .html")

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
                (dest / f"{base}.md").write_text(md, encoding="utf-8")
                (dest / f"{base}.html").write_text(styled_html, encoding="utf-8")
                if pdf_p and env_key == "REPORTS_DIR":
                    (dest / pdf_p.name).write_bytes(pdf_p.read_bytes())
                    pdf_p = dest / pdf_p.name           # open the copy people keep
                log(f"  local copy -> {dest / base}.md")
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
            # PDF ONLY, same formatter (and look) as the deep dive mail (user 2026-10-02:
            # the .md / raw-markdown body arrived unformatted). The .md is on Drive and
            # the .html (with the chart) stays next to it locally / in the CI artefacts.
            # If the PDF cannot be made, say so and attach the .html instead.
            pdf_note = ""
            if pdf_p:
                att = (pdf_p.name, pdf_p.read_bytes(), "pdf")
            else:
                pdf_note = (f"<p style='color:#c33'><b>PDF could not be made</b> "
                            f"({pdf_err}) — the report is attached as .html.</p>")
                att = (f"{base}.html", styled_html.encode("utf-8"), "octet-stream")
            notes = "".join(f"<p style='color:#b45309'><b>Data note:</b> {n}</p>"
                            for n in d.get("data_notes") or [])
            part_b = ("Part B (forensic deep dive) is included."
                      if nar.get("forensic_report") else "No Part B (deep dive) in this copy.")
            body = (
                f"<h2>{co['name']} — narrative report</h2>"
                f"<p>{co['symbol']} · {co['isin']} · data current to "
                f"{d['as_of_utc'][:10]}</p>"
                f"{notes}"
                f"<p>{audit_line}</p>"
                f"<p>{len(d['facts'])} facts · {len(d['tables'])} tables · "
                f"{len(nar['sections'])} sections · {flagged} section(s) with "
                f"unresolved grounding flags</p>"
                f"<p>{part_b}</p>"
                f"<p>Attachment: <b>{att[0]}</b>. The .md is on Drive "
                f"(company_repo/{co['isin']}/).</p>"
                f"{pdf_note}")
            ok = send_email(
                f"Narrative report — {co['name']} ({co['symbol']})",
                body,
                attachments=[att])
            log("  mailed" if ok else "  mail SKIPPED (GMAIL_USER / "
                                      "GMAIL_APP_PASSWORD not set)")
        except Exception as e:
            log(f"  mail FAILED: {str(e)[:160]}")

    if args.open:
        # the PDF (same file the mail carries), as the deep dive does; the styled .html
        # (same page the PDF is printed from) if no PDF could be made
        target = pdf_p
        if target is None:
            target = outdir / f"{base}.html"
            target.write_text(styled_html, encoding="utf-8")
        webbrowser.open(target.resolve().as_uri())
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
# with_deepdive (added 2026-10-02, additive): True = BOTH reports were queued; the story
# waits until the company's deep dive is built (deepdive.yml, 08:00 IST) and attaches it
# as Part B. Older rows read it as None = story only, exactly as before.
NQUEUE_COLS = ["token", "status", "added_at", "done_at", "error", "with_deepdive"]


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


def enqueue_narrative(store: FP.Store, tokens: list[str],
                      with_deepdive: bool = False) -> int:
    """Queue tokens for the next drain. Returns how many are now pending.

    Only an ALREADY-PENDING token is skipped. A token whose last run is `done` or
    `error` is RE-QUEUED (its row resets to pending), because asking for a company
    again is a legitimate request — a report is a point-in-time artefact and the code,
    the filings and the data all move. Treating `done` as permanent meant a second
    `--add` reported "0 added (already pending/done)", the drain then found nothing,
    and the dispatch went green having built nothing.
    """
    df = _load_nqueue(store)
    toks = [t.strip() for t in dict.fromkeys(tokens) if t.strip()]
    if not toks:
        return 0
    if df.empty:
        df = pd.DataFrame(columns=NQUEUE_COLS)
    status = df["status"].astype(str) if not df.empty else pd.Series(dtype=str)
    already_pending = (set(df[status == "pending"]["token"].astype(str))
                       if not df.empty else set())

    now = datetime.now().isoformat(timespec="seconds")
    requeued, added, upgraded = [], [], []
    for t in toks:
        if t in already_pending:
            # Asking for BOTH on a company already queued story-only upgrades the row;
            # a plain --add never downgrades one that asked for both.
            if with_deepdive:
                m = (df["token"].astype(str) == t) & (status == "pending")
                df.loc[m, "with_deepdive"] = True
                upgraded.append(t)
            continue
        m = (df["token"].astype(str) == t) if not df.empty else None
        if m is not None and m.any():
            df.loc[m, ["status", "added_at", "done_at", "error"]] = \
                ["pending", now, None, None]
            df.loc[m, "with_deepdive"] = bool(with_deepdive)
            requeued.append(t)
        else:
            added.append(t)
    if added:
        df = pd.concat([df, pd.DataFrame([{"token": t, "status": "pending",
                                           "added_at": now,
                                           "with_deepdive": bool(with_deepdive)}
                                          for t in added])],
                       ignore_index=True)
    if added or requeued or upgraded:
        _save_nqueue(store, df)
    if upgraded:
        log(f"  already queued — now set to BOTH reports: {', '.join(upgraded)}")
    if requeued:
        log(f"  re-queued (previous run finished): {', '.join(requeued)}")
    return len(added) + len(requeued)


def _mark_nqueue(store: FP.Store, token: str, status: str, error: str = ""):
    store.refresh()          # runs after a long report build: fresh Drive connection
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


def _note_nqueue(store: FP.Store, token: str, note: str):
    """Record WHY a row is still pending (status unchanged) — visible in the queue file."""
    df = _load_nqueue(store)
    m = (df["token"].astype(str) == str(token)) & (df["status"].astype(str) == "pending")
    if m.any():
        df.loc[m, "error"] = note[:200]
        _save_nqueue(store, df)


def _queued_part_b(store: FP.Store, token: str, outdir: str) -> tuple[str | None, str]:
    """For a BOTH row: (path to the deep dive .md to attach, "") once it is built, or
    (None, why) while the story must keep waiting.

    Linked through deep_dive_queue STATUS, not timestamps: the two queues are stamped by
    different machines' clocks (IST locally, UTC in CI), so comparing times would mix
    zones. run_report.bat queues the same token in both queues and re-queues a finished
    deep dive, so "no pending deep-dive row" means this request's deep dive has run.
    """
    dq = store.parquet(FP.IDX, "deep_dive_queue.parquet")
    if not dq.empty and {"token", "status"} <= set(dq.columns):
        mine = dq[dq["token"].astype(str).str.strip().str.upper() == token.strip().upper()]
        if (mine["status"].astype(str) == "pending").any():
            return None, "deep dive still queued (deepdive.yml drains it at 08:00 IST)"
        errs = mine[mine["status"].astype(str) == "error"]
    else:
        errs = pd.DataFrame()
    r = FP.resolve(store, token)
    if r is None:
        return "", ""                    # unresolvable: run_one reports it as an error
    p, why = _drive_part_b(store, r[0], outdir)
    if p is None and not errs.empty:
        why += f" — deep_dive_queue says error: {str(errs.iloc[-1].get('error'))[:80]}"
    return p, why


def _drive_part_b(store: FP.Store, isin: str, outdir: str,
                  since: str = "") -> tuple[str | None, str]:
    """Download the company's latest deep dive from Drive (deep_dive_index ->
    company_repo/<ISIN>/company_deepdive_*.md) for Part B -> (local path, "") or (None, why).

    since (ISO time, the SAME machine's clock that ran the deep dive — it stamps
    last_update with datetime.now()): only a deep dive built at/after it counts, so a run
    whose deep dive just failed never silently attaches an older one. Used by run-now BOTH
    (local run_report.bat and CI narrative.yml alike) and by the queue drain (no since).
    """
    idx = store.parquet(FP.IDX, "deep_dive_index.parquet")
    row = idx[idx["isin"].astype(str) == isin] if not idx.empty and "isin" in idx else idx
    if row.empty or not str(row.iloc[-1].get("report_path") or ""):
        return None, "no deep dive on Drive yet"
    if since:
        built = pd.to_datetime(row.iloc[-1].get("last_update"), errors="coerce")
        if pd.isna(built) or built < pd.to_datetime(since):
            return None, (f"latest deep dive on Drive is from {str(row.iloc[-1].get('last_update'))[:16]}"
                          f", older than this run ({since[:16]}) — this run's deep dive "
                          f"did not finish")
    name = Path(str(row.iloc[-1]["report_path"])).name
    fid = find_file(store.drive, store.folder(f"company_repo/{isin}"), name)
    if not fid:
        return None, f"deep dive {name} listed in deep_dive_index but not found on Drive"
    p = Path(outdir) / f"_partb_{isin}_{name}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(download_bytes(store.drive, fid))
    log(f"  Part B from Drive: company_repo/{isin}/{name}")
    return str(p), ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Not required: no --names DRAINS narrative_queue.parquet, mirroring how
    # company_deep_report.py runs with no args on a scheduled CI pass.
    ap.add_argument("--names", nargs="*", default=None,
                    help="ISIN / symbol / name fragment. Omit to drain the queue.")
    ap.add_argument("--add", nargs="+", default=None,
                    help="enqueue these tokens for the next scheduled run, then exit")
    ap.add_argument("--with-deepdive", action="store_true",
                    help="with --add: BOTH reports — the queued story waits for the "
                         "company's deep dive (queue it too: company_deep_report.py --add "
                         "X --requeue) and attaches it from Drive as Part B")
    ap.add_argument("--allow-empty-queue", action="store_true",
                    help="an empty queue exits 0 instead of failing. For the SCHEDULED "
                         "run, where nothing queued is normal; a manual dispatch that "
                         "builds nothing should fail loudly instead.")
    ap.add_argument("--outdir", default="./_narrative")
    ap.add_argument("--cache", default="")
    ap.add_argument("--sections", nargs="*", type=int, default=None)
    ap.add_argument("--forensic-md", default="",
                    help="existing company_deepdive_*.md to attach as Part B")
    ap.add_argument("--part-b-drive", default="", metavar="SINCE",
                    help="BOTH run-now: attach the company's deep dive from Drive as Part B, "
                         "only if it was built at/after SINCE (ISO time, same clock as the "
                         "deep dive run). Used identically by run_report.bat and narrative.yml")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight + fact pack + sources only; no LLM, no writes")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--force-gemini-audit", action="store_true",
                    help="audit on Gemini — now the DEFAULT; kept so old command lines work")
    ap.add_argument("--alt-audit", action="store_true",
                    help="audit on Cerebras/Groq (independent family) instead of Gemini; "
                         "off by default since the Cerebras account returns HTTP 402")
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
                    help="email the report as a PDF (short summary inline; .html if the "
                         "PDF cannot be made) to NOTIFY_EMAIL — identical local or in CI")
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
        n = enqueue_narrative(store, a.add, with_deepdive=a.with_deepdive)
        log(f"enqueued {n} token(s) to narrative_queue "
            f"({len(a.add) - n} already pending/done)"
            + (" — BOTH reports: each waits for its deep dive" if a.with_deepdive else ""))
        return 0

    # No --names -> DRAIN the queue, same as company_deep_report.py with no args.
    draining = not a.names
    if draining:
        q = _load_nqueue(store)
        pending = (q[q["status"].astype(str) == "pending"]["token"].astype(str).tolist()
                   if not q.empty else [])
        if not pending:
            # A MANUAL run that produces nothing must not report success: dispatching
            # with a blank `names` field drains the queue, and an empty queue meant the
            # job went green having done no work — indistinguishable from a finished
            # report. A SCHEDULED run finding an empty queue is normal and stays green.
            if a.allow_empty_queue:
                log("narrative_queue is empty — nothing to do (scheduled run).")
                return 0
            log("ERROR: no company to build.")
            log("  This run was told to drain the queue (blank `names`), but")
            log("  narrative_queue.parquet has no pending companies.")
            log("  Either:")
            log("    - re-dispatch with names=<COMPANY>, or")
            log("    - queue one first:  python scripts/company_narrative_report.py "
                "--add \"<COMPANY>\"")
            return 1
        log(f"draining narrative_queue: {len(pending)} pending "
            f"({', '.join(pending[:8])}{'...' if len(pending) > 8 else ''})")
        tokens = pending
        wd = q["with_deepdive"] if "with_deepdive" in q.columns else pd.Series(dtype=object)
        both = set(q[(q["status"].astype(str) == "pending")
                     & wd.apply(lambda v: v is True or str(v).lower() == "true")
                     ]["token"].astype(str))
    else:
        tokens = a.names
        both = set()

    results, failures, waiting = [], 0, 0
    for token in tokens:
        a_tok = a
        if token in both:
            part_b, why = _queued_part_b(store, token, a.outdir)
            if part_b is None:
                log(f"  {token}: BOTH reports queued — waiting: {why}. Stays queued; "
                    f"the next run builds it once the deep dive exists.")
                _note_nqueue(store, token, f"waiting for deep dive: {why}")
                waiting += 1
                continue
            if part_b:
                a_tok = argparse.Namespace(**{**vars(a), "forensic_md": part_b})
        elif a.part_b_drive and not a.forensic_md:
            r0 = FP.resolve(store, token)
            part_b, why = (_drive_part_b(store, r0[0], a.outdir, since=a.part_b_drive)
                           if r0 else (None, "could not resolve the company"))
            if part_b:
                a_tok = argparse.Namespace(**{**vars(a), "forensic_md": part_b})
            else:
                log(f"  {token}: built WITHOUT Part B — {why}")
        try:
            r = run_one(store, token, a_tok)
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
    if waiting:
        log(f"  {waiting} company/companies waiting for their deep dive (still queued)")
    if failures:
        log(f"  {failures} company/companies failed")
    return 1 if failures and not results else 0


if __name__ == "__main__":
    sys.exit(main())
