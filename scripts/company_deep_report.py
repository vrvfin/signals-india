r"""
company_deep_report.py  —  signals-india / Workflow B (OT7)

On-demand company deep dive. Drains deep_dive_queue.parquet (populated by Excel /
bat / Streamlit), and for each pending company:

  resolve name/NSE/BSE/ISIN -> ISIN (via universe)
   -> PHASE 0 rule-based coverage check (read company_page.md; what exists vs missing)
   -> assemble context from RELIABLE sources only:
        Screener  (fundamentals/summary.parquet + company_repo/_index/results.parquet)
        Company Page brief (company_repo/<ISIN>/company_page.md)
        Research Index filtered to this ISIN / sector / promoter (research_index.parquet)
        BSE announcements (BSE Direct API, best-effort, no auth)
   -> comapnydeepdive_prompt.txt  (single call; inputs are summaries, so it fits free tier)
   -> company_repo/<ISIN>/company_deepdive_DDMMMYY.md
   -> update deep_dive_index.parquet (Streamlit dropdown + last_update) and mark queue done

Runs in CI (deepdive.yml) or locally. Reuses the Drive helpers from the daily script.

Run:   python scripts/company_deep_report.py                  # drain the queue
       python scripts/company_deep_report.py --names "TCS,INFY"   # ad-hoc, no queue
       python scripts/company_deep_report.py --add "INE467B01029" # enqueue only

Env:   GEMINI_API_KEY (comma-separated allowed), GDRIVE_FOLDER_ID, GDRIVE_OAUTH_TOKEN_JSON
Deps:  google-generativeai pandas pyarrow requests + the Drive stack
"""
from __future__ import annotations
import os, io, re, sys, json, argparse, datetime as dt, tempfile, webbrowser

# Ensure scripts/ is on sys.path whether run as `python scripts/foo.py` (CI/root)
# or as `python foo.py` (local, already in scripts/)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests

from daily_research_summary import (drive_service, drive_download, drive_upload, _folder_id)
from gemini_pool import BucketPool, AllBucketsExhausted, FatalCallError, load_keys

SCRIPTS_DIR = _SCRIPTS_DIR
INTER_CALL_SLEEP = 6.0

# Best model first; pool only downgrades when current model is dead on ALL keys.
# 5 models × N keys = N*5 independent daily buckets.
DEEPDIVE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
]

DRIVE = dict(
    queue        = "company_repo/_index/deep_dive_queue.parquet",
    index        = "company_repo/_index/deep_dive_index.parquet",
    research_idx = "company_repo/_index/research_index.parquet",
    fundamentals = "fundamentals/summary.parquet",
    results      = "company_repo/_index/results.parquet",
    universe     = "universe/master_list.csv",
    company_page = "company_repo",   # /<ISIN>/company_page.md  &  output report
)

# context-size caps so the single call stays well inside the free-tier budget
MAX_PAGE_CHARS   = 30_000
MAX_RESEARCH_ROWS = 25

# --------------------------------------------------------------------------
def _read_parquet(svc, path, root):
    b = drive_download(svc, path, root)
    return pd.read_parquet(io.BytesIO(b)) if b else pd.DataFrame()

def _read_csv(svc, path, root):
    b = drive_download(svc, path, root)
    return pd.read_csv(io.BytesIO(b)) if b else pd.DataFrame()

def resolve_isin(token, universe, interactive=False):
    """token may be ISIN / NSE symbol / BSE code / name -> (isin, symbol, name, bse_code).

    Falls back to fuzzy (contains) name match. If multiple hits and interactive=True,
    prompts user to pick; otherwise returns first hit.
    """
    if universe is None or universe.empty:
        return (token, token, token, None)
    cols = {c.lower(): c for c in universe.columns}
    isin_c = cols.get("isin"); sym_c = cols.get("symbol") or cols.get("nse_symbol")
    name_c = cols.get("name") or cols.get("company") or cols.get("company_name")
    bse_c  = cols.get("bse_code") or cols.get("scrip_code") or cols.get("bsecode")
    t = str(token).strip()
    def row_out(r):
        raw_bse = r[bse_c] if bse_c else None
        if bse_c and pd.notna(raw_bse):
            s = str(raw_bse)
            bse_out = str(int(float(s))) if s.replace(".", "").isdigit() else s
        else:
            bse_out = None
        return (str(r[isin_c]) if isin_c else t,
                str(r[sym_c]) if sym_c else t,
                str(r[name_c]) if name_c else t,
                bse_out)
    # exact matches first
    if isin_c and t.upper().startswith("INE"):
        hit = universe[universe[isin_c].astype(str) == t.upper()]
        if not hit.empty: return row_out(hit.iloc[0])
    if sym_c:
        hit = universe[universe[sym_c].astype(str).str.upper() == t.upper()]
        if not hit.empty: return row_out(hit.iloc[0])
    if bse_c and t.isdigit():
        # bse_code is numeric in CSV so pandas reads it as float -> "522101.0"
        # normalise by converting to Int64 string before comparing
        bse_norm = universe[bse_c].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace(".", "").isdigit() else str(x))
        hit = universe[bse_norm == t]
        if not hit.empty: return row_out(hit.iloc[0])
    if name_c:
        hit = universe[universe[name_c].astype(str).str.lower() == t.lower()]
        if not hit.empty: return row_out(hit.iloc[0])
    # fuzzy fallback — partial name contains
    if name_c:
        fuzzy = universe[universe[name_c].astype(str).str.contains(t, case=False, na=False)]
        if not fuzzy.empty:
            if len(fuzzy) == 1 or not interactive:
                return row_out(fuzzy.iloc[0])
            print(f"\nMultiple matches for '{t}':")
            for i, (_, r) in enumerate(fuzzy.head(10).iterrows(), 1):
                n = str(r[name_c]) if name_c else "?"
                s = str(r[sym_c]) if sym_c else "?"
                print(f"  {i}. {n} ({s})")
            while True:
                try:
                    pick = int(input("Pick number: ").strip()) - 1
                    if 0 <= pick < min(len(fuzzy), 10):
                        return row_out(fuzzy.iloc[pick])
                except (ValueError, KeyboardInterrupt):
                    pass
                print("Invalid — enter a number from the list.")
    return (t, t, t, None)

# ---- PHASE 0: rule-based coverage check on company_page.md ----------------
def coverage_check(page_md: str) -> dict:
    if not page_md:
        return dict(has_page=False, ar_years=[], n_concall=0, n_rating=0,
                    n_presentation=0, n_research=0)
    return dict(
        has_page=True,
        ar_years=sorted(set(re.findall(r"FY\s?(\d{2})", page_md))),
        n_concall=len(re.findall(r"(?i)concall", page_md)),
        n_rating=len(re.findall(r"(?i)rating", page_md)),
        n_presentation=len(re.findall(r"(?i)presentation", page_md)),
        n_research=len(re.findall(r"research_\d{4}", page_md)),
    )

# ---- Source assemblers ----------------------------------------------------
def screener_block(fund, results, isin, symbol):
    out = []
    def pick(df):
        if df is None or df.empty: return None
        for col in ("isin", "ISIN"):
            if col in df.columns:
                r = df[df[col].astype(str) == isin]
                if not r.empty: return r
        for col in ("symbol", "Symbol", "nse_symbol"):
            if col in df.columns:
                r = df[df[col].astype(str).str.upper() == symbol.upper()]
                if not r.empty: return r
        return None
    f = pick(fund)
    if f is not None:
        out.append("FUNDAMENTALS (Screener):")
        out.append(f.iloc[0].dropna().astype(str).to_string())
    r = pick(results)
    if r is not None:
        out.append("\nRECENT RESULTS (Screener):")
        out.append(r.head(8).to_string(index=False))
    return "\n".join(out) if out else "DATA_MISSING"

def research_block(ridx, isin, symbol, name):
    if ridx is None or ridx.empty:
        return "No external research context provided."
    def hit(row):
        blob = f"{row.get('isins','')}{row.get('companies','')}".lower()
        return isin.lower() in blob or symbol.lower() in blob or name.lower() in blob
    sel = ridx[ridx.apply(hit, axis=1)].tail(MAX_RESEARCH_ROWS)
    if sel.empty:
        return "No external research context provided."
    lines = []
    for _, r in sel.iterrows():
        lines.append(f"- [{r.get('doc_type','?')} | {r.get('source','?')} | "
                     f"{r.get('doc_date','NA')}] {str(r.get('file_name',''))}")
    return "\n".join(lines)

def bse_announcements(bse_code):
    if not bse_code:
        return "DATA_MISSING (no BSE scrip code resolved)."
    try:
        today = dt.date.today()
        prev = today - dt.timedelta(days=365 * 5)
        url = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
        params = {"strCat": "-1", "strPrevDate": prev.strftime("%Y%m%d"),
                  "strScrip": str(bse_code), "strSearch": "P",
                  "strToDate": today.strftime("%Y%m%d"), "strType": "C"}
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                   "Referer": "https://www.bseindia.com/",
                   "Origin": "https://www.bseindia.com"}
        rsp = requests.get(url, params=params, headers=headers, timeout=30)
        rsp.raise_for_status()
        rows = rsp.json().get("Table", [])[:40]
        if not rows:
            return "No announcements returned."
        return "\n".join(f"- {r.get('NEWS_DT','')[:10]} | {r.get('HEADLINE','')[:140]}"
                         for r in rows)
    except Exception as e:
        return f"DATA_MISSING (BSE fetch failed: {type(e).__name__})"

# ---- prompt assembly ------------------------------------------------------
def fill_section(tpl, tag, content):
    return re.sub(rf"\[{tag}\].*?\[/{tag}\]", f"[{tag}]\n{content}\n[/{tag}]",
                  tpl, flags=re.DOTALL)

def build_prompt(name, symbol, isin, screener, page, research, bse):
    tpl = open(os.path.join(SCRIPTS_DIR, "comapnydeepdive_prompt.txt"),
               encoding="utf-8").read()
    tpl = (tpl.replace("[COMPANY_NAME]", name)
              .replace("[NSE_SYMBOL]", symbol)
              .replace("[ISIN]", isin))
    tpl = fill_section(tpl, "SCREENER_FINANCIAL_DATA", screener)
    tpl = fill_section(tpl, "COMPANY_PAGE_BRIEF", page[:MAX_PAGE_CHARS] or "DATA_MISSING")
    tpl = fill_section(tpl, "RESEARCH_INDEX_CONTEXT", research)
    tpl = fill_section(tpl, "BSE_ANNOUNCEMENTS", bse)
    tpl = fill_section(tpl, "RAW_ANNUAL_REPORTS_FOR_GAPS",
                       "None supplied. Rely on Screener + Company Page.")
    return tpl

# --------------------------------------------------------------------------
def process_one(svc, root, pool, universe, fund, results, ridx, token, interactive=False):
    isin, symbol, name, bse_code = resolve_isin(token, universe, interactive=interactive)
    print(f"  deep dive: {token} -> {name} ({symbol} / {isin})")
    page_b = drive_download(svc, f"{DRIVE['company_page']}/{isin}/company_page.md", root)
    page = page_b.decode("utf-8") if page_b else ""
    cov = coverage_check(page)
    print(f"    coverage: page={cov['has_page']} ar={cov['ar_years']} "
          f"concall={cov['n_concall']} research={cov['n_research']}")

    prompt = build_prompt(name, symbol, isin,
                          screener_block(fund, results, isin, symbol),
                          page,
                          research_block(ridx, isin, symbol, name),
                          bse_announcements(bse_code))
    report, model_used = pool.call_text(prompt)

    stamp = dt.datetime.now().strftime("%d%b%y")
    out_path = f"{DRIVE['company_page']}/{isin}/company_deepdive_{stamp}.md"
    header = (f"# Deep Dive — {name} ({symbol} / {isin})\n"
              f"*Generated {dt.datetime.now():%Y-%m-%d %H:%M} · "
              f"coverage: AR{cov['ar_years']} concalls~{cov['n_concall']} "
              f"research~{cov['n_research']}*\n\n---\n\n")
    full_md = header + report
    drive_upload(svc, out_path, root, full_md.encode("utf-8"), "text/markdown")
    print(f"    wrote {out_path}")

    # generate Word + PPT and upload alongside the .md
    base_path = out_path.rsplit(".", 1)[0]
    try:
        from format_deepdive_docx import md_to_docx
        docx_bytes = md_to_docx(full_md, name, symbol, isin,
                                coverage=str(cov.get("ar_years", "")))
        drive_upload(svc, base_path + ".docx", root, docx_bytes,
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        print(f"    wrote {base_path}.docx")
    except Exception as e:
        print(f"    docx skipped: {e}")
    try:
        from format_deepdive_pptx import md_to_pptx
        pptx_bytes = md_to_pptx(full_md, name, symbol, isin,
                                 coverage=str(cov.get("ar_years", "")))
        drive_upload(svc, base_path + ".pptx", root, pptx_bytes,
                     "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        print(f"    wrote {base_path}.pptx")
    except Exception as e:
        print(f"    pptx skipped: {e}")
    return dict(isin=isin, symbol=symbol, name=name, report_path=out_path,
                last_update=dt.datetime.now().isoformat(),
                coverage=json.dumps(cov),
                _report_md=full_md,   # internal — not persisted to parquet
                _slug=f"{symbol.lower()}_{dt.datetime.now().strftime('%d%b%y').lower()}")

def open_report_local(report_md: str, slug: str,
                      name: str = "", symbol: str = "", isin: str = ""):
    """Open a markdown report locally — Obsidian if available, else HTML in browser.
    Also saves .docx and .pptx to the local reports folder.
    """
    obsidian_vault = os.path.join(
        os.environ.get("OBSIDIAN_VAULT", r"D:\EMA_Screener\Obsidian"),
        "signals-india", "deepdive")
    local_dir = os.path.join(
        os.environ.get("REPORTS_DIR", r"D:\EMA_Screener\Reports\signals-india"),
        "deepdive")
    os.makedirs(local_dir, exist_ok=True)

    # save .md
    md_path = os.path.join(local_dir, f"{slug}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # save .docx
    try:
        from format_deepdive_docx import md_to_docx
        docx_bytes = md_to_docx(report_md, name, symbol, isin)
        docx_path = os.path.join(local_dir, f"{slug}.docx")
        with open(docx_path, "wb") as f:
            f.write(docx_bytes)
        print(f"    saved docx: {docx_path}")
    except Exception as e:
        print(f"    docx local save skipped: {e}")

    # save .pptx
    try:
        from format_deepdive_pptx import md_to_pptx
        pptx_bytes = md_to_pptx(report_md, name, symbol, isin)
        pptx_path = os.path.join(local_dir, f"{slug}.pptx")
        with open(pptx_path, "wb") as f:
            f.write(pptx_bytes)
        print(f"    saved pptx: {pptx_path}")
    except Exception as e:
        print(f"    pptx local save skipped: {e}")

    # try Obsidian for .md
    try:
        os.makedirs(obsidian_vault, exist_ok=True)
        obs_path = os.path.join(obsidian_vault, f"{slug}.md")
        with open(obs_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"    saved to Obsidian: {obs_path}")
        return
    except Exception:
        pass

    # HTML fallback
    try:
        import markdown as md_lib
        html_body = md_lib.markdown(report_md, extensions=["tables", "fenced_code"])
    except ImportError:
        html_body = f"<pre>{report_md}</pre>"
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto;"
            f"line-height:1.6}}table{{border-collapse:collapse;width:100%}}"
            f"td,th{{border:1px solid #ccc;padding:6px 10px}}</style></head>"
            f"<body>{html_body}</body></html>")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                     prefix=f"deepdive_{slug}_", mode="w", encoding="utf-8")
    tmp.write(html); tmp.close()
    webbrowser.open(f"file://{tmp.name}")
    print(f"    opened in browser: {tmp.name}")


def update_index(svc, root, recs):
    idx = _read_parquet(svc, DRIVE["index"], root)
    new = pd.DataFrame(recs)
    if not idx.empty:
        idx = idx[~idx["isin"].isin(new["isin"])]      # keep latest per company
    idx = pd.concat([idx, new], ignore_index=True)
    buf = io.BytesIO(); idx.to_parquet(buf, index=False)
    drive_upload(svc, DRIVE["index"], root, buf.getvalue(), "application/octet-stream")
    print(f"  deep_dive_index updated ({len(idx)} companies).")

# --------------------------------------------------------------------------
def _strip_internal(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names",        help="comma-separated tokens, ad-hoc run (bypass queue)")
    ap.add_argument("--add",          help="comma-separated tokens, enqueue only (no processing)")
    ap.add_argument("--open",         action="store_true",
                    help="open report locally in Obsidian / browser after writing")
    ap.add_argument("--interactive",  action="store_true",
                    help="prompt to pick when fuzzy name match returns multiple results")
    ap.add_argument("--resolve-only", action="store_true",
                    help="resolve and print company name then exit (used by bat for confirmation)")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(SCRIPTS_DIR), ".env"))

    svc = drive_service(); root = os.environ["GDRIVE_FOLDER_ID"]

    if args.resolve_only and args.names:
        universe = _read_csv(svc, DRIVE["universe"], root)
        for t in [x.strip() for x in args.names.split(",") if x.strip()]:
            isin, symbol, name, _ = resolve_isin(t, universe, interactive=args.interactive)
            if isin == t and symbol == t:
                print(f"Could not resolve: {t}"); sys.exit(1)
            print(f"Resolved: {name} ({symbol} / {isin})")
        return

    if args.add:
        q = _read_parquet(svc, DRIVE["queue"], root)
        rows = [dict(token=t.strip(), status="pending",
                     added_at=dt.datetime.now().isoformat()) for t in args.add.split(",")]
        q = pd.concat([q, pd.DataFrame(rows)], ignore_index=True)
        buf = io.BytesIO(); q.to_parquet(buf, index=False)
        drive_upload(svc, DRIVE["queue"], root, buf.getvalue(), "application/octet-stream")
        print(f"Enqueued {len(rows)}."); return

    api_keys = load_keys(os.environ)
    if not api_keys:
        print("ERROR: no GEMINI_API_KEY or GEMINI_API_KEY_* found in .env")
        sys.exit(1)
    pool = BucketPool(api_keys, DEEPDIVE_MODELS, inter_call_s=INTER_CALL_SLEEP)
    print(f"Pool: {len(api_keys)} key(s) × {len(DEEPDIVE_MODELS)} model(s) "
          f"= {len(api_keys) * len(DEEPDIVE_MODELS)} daily buckets")

    universe = _read_csv(svc, DRIVE["universe"], root)
    fund     = _read_parquet(svc, DRIVE["fundamentals"], root)
    results  = _read_parquet(svc, DRIVE["results"], root)
    ridx     = _read_parquet(svc, DRIVE["research_idx"], root)

    if args.names:
        tokens = [t.strip() for t in args.names.split(",") if t.strip()]
        recs = []
        for t in tokens:
            try:
                recs.append(process_one(svc, root, pool, universe, fund, results, ridx, t,
                                        interactive=args.interactive))
            except AllBucketsExhausted as exc:
                print(f"  All Gemini buckets exhausted — stopping. ({exc})")
                break
            except FatalCallError as exc:
                print(f"  Fatal error for '{t}' (bad prompt/auth) — skipping. ({exc})")
        if recs:
            if args.open:
                for r in recs:
                    open_report_local(r["_report_md"], r["_slug"],
                                      r.get("name",""), r.get("symbol",""), r.get("isin",""))
            update_index(svc, root, [_strip_internal(r) for r in recs])
        return

    queue = _read_parquet(svc, DRIVE["queue"], root)
    if queue.empty or "status" not in queue:
        print("Queue empty. Nothing to do."); return
    pending = queue[queue["status"] == "pending"]
    if pending.empty:
        print("No pending companies."); return

    recs = []
    for i in pending.index:
        try:
            rec = process_one(svc, root, pool, universe, fund, results, ridx,
                              queue.at[i, "token"], interactive=args.interactive)
            recs.append(rec)
            queue.at[i, "status"] = "done"
            queue.at[i, "done_at"] = dt.datetime.now().isoformat()
        except AllBucketsExhausted as exc:
            # Quota exhausted — leave remaining rows pending for next run
            print(f"  All Gemini buckets exhausted — stopping queue drain. ({exc})")
            break
        except FatalCallError as exc:
            print(f"  FATAL (this company): {str(exc)[:120]}")
            queue.at[i, "status"] = "error"
            queue.at[i, "error"] = str(exc)[:300]
        except Exception as e:
            print(f"    FAILED {queue.at[i,'token']}: {e}")
            queue.at[i, "status"] = "error"
            queue.at[i, "error"] = str(e)[:300]

    buf = io.BytesIO(); queue.to_parquet(buf, index=False)
    drive_upload(svc, DRIVE["queue"], root, buf.getvalue(), "application/octet-stream")
    if recs:
        if args.open:
            for r in recs:
                open_report_local(r["_report_md"], r["_slug"],
                                  r.get("name",""), r.get("symbol",""), r.get("isin",""))
        update_index(svc, root, [_strip_internal(r) for r in recs])
    print(f"Done. {len(recs)} report(s) generated.")


if __name__ == "__main__":
    main()
