"""
synthesise_company_docs.py  —  find all docs mentioning a company, synthesise with Gemini

Reads summaries from research_ledger.parquet (local, has summary_md).
Uses the dedicated daily key pool (DAILY_GEMINI_KEY_1/2), same as Workflow A.

Usage:
  python scripts/synthesise_company_docs.py "venus remedies"
  python scripts/synthesise_company_docs.py INE680B01014
  python scripts/synthesise_company_docs.py "venus" --doc-type single_company_note
  python scripts/synthesise_company_docs.py "venus remedies" --upload
  python scripts/synthesise_company_docs.py "venus remedies" --upload --queue
  python scripts/synthesise_company_docs.py --all-new --upload --queue  (called by bat after daily run)

Flags:
  --upload     Upload dated + latest synthesis to Drive at company_repo/<ISIN>/
  --queue      Enqueue synthesised ISIN(s) in deep_dive_queue.parquet on Drive
  --all-new    Process all ISINs in _ledger/new_isins_latest.json (written by daily_research_summary.py)
  --outdir     Local save directory (default D:\\EMA_Screener\\Reports\\research_synthesis)

Invoked by synthesise_company.bat and run_daily_research.bat
"""
from __future__ import annotations
import os, sys, re, io, json, argparse, datetime as dt, time, subprocess, urllib.parse
from pathlib import Path
import pandas as pd

# load project .env (same pattern as all pipeline scripts)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

INTAKE_DIR    = Path(os.getenv("RESEARCH_INTAKE_DIR", r"D:\EMA_Screener\research_intake"))
MENTIONS_PATH = INTAKE_DIR / "_ledger" / "company_mentions.parquet"
LEDGER_PATH   = INTAKE_DIR / "_ledger" / "research_index.parquet"   # local copy, has summary_md
NEW_ISINS_PATH = INTAKE_DIR / "_ledger" / "new_isins_latest.json"
SCRIPTS_DIR   = Path(__file__).resolve().parent
PROMPT_FILE   = SCRIPTS_DIR / "company_synthesis_prompt.txt"

# Obsidian vault folder so syntheses are indexed + open in Obsidian
DEFAULT_OUTDIR   = Path(r"D:\EMA_Screener\Reports\signals-india\company_synthesis")

# Drive paths
DRIVE_COMPANY_DIR = "company_repo"
DRIVE_QUEUE       = "company_repo/_index/deep_dive_queue.parquet"

from _md_utils import fix_markdown_for_obsidian               # Obsidian table fixer
# shared BucketPool engine over the dedicated daily keys (same as the summariser)
from daily_research_summary import build_daily_pool


def open_in_obsidian(path: Path) -> None:
    uri = "obsidian://open?path=" + urllib.parse.quote(str(path).replace("\\", "/"), safe=":/")
    try:
        subprocess.run(["cmd", "/c", "start", "", uri], shell=False)
    except Exception as e:
        print(f"Could not open in Obsidian ({e}); open manually: {path}")


# ---------------------------------------------------------------------------
# Drive helpers (minimal; same pattern as daily_research_summary.py)
# ---------------------------------------------------------------------------
def _drive_service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    if os.environ.get("GDRIVE_OAUTH_TOKEN_JSON"):
        creds = Credentials.from_authorized_user_info(
            json.loads(os.environ["GDRIVE_OAUTH_TOKEN_JSON"]))
    else:
        token_path = os.environ.get("GDRIVE_OAUTH_TOKEN_PATH", "")
        if not token_path:
            raise RuntimeError("Set GDRIVE_OAUTH_TOKEN_JSON or GDRIVE_OAUTH_TOKEN_PATH")
        creds = Credentials.from_authorized_user_file(token_path)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _folder_id(svc, path: str, root: str, create=False) -> str | None:
    parent = root
    for part in path.strip("/").split("/"):
        q = (f"name='{part}' and '{parent}' in parents and "
             "mimeType='application/vnd.google-apps.folder' and trashed=false")
        res = svc.files().list(q=q, fields="files(id)", pageSize=1).execute().get("files", [])
        if res:
            parent = res[0]["id"]
        elif create:
            meta = {"name": part, "parents": [parent],
                    "mimeType": "application/vnd.google-apps.folder"}
            parent = svc.files().create(body=meta, fields="id").execute()["id"]
        else:
            return None
    return parent

def _drive_find(svc, full_path: str, root: str) -> str | None:
    *dirs, name = full_path.strip("/").split("/")
    fid = _folder_id(svc, "/".join(dirs), root) if dirs else root
    if fid is None:
        return None
    q = f"name='{name}' and '{fid}' in parents and trashed=false"
    res = svc.files().list(q=q, fields="files(id)", pageSize=1).execute().get("files", [])
    return res[0]["id"] if res else None

def _drive_download(svc, full_path: str, root: str) -> bytes | None:
    fid = _drive_find(svc, full_path, root)
    if not fid:
        return None
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=fid))
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

def _drive_upload(svc, full_path: str, root: str, data: bytes, mime: str):
    from googleapiclient.http import MediaIoBaseUpload
    *dirs, name = full_path.strip("/").split("/")
    fid_dir = _folder_id(svc, "/".join(dirs), root, create=True) if dirs else root
    existing = _drive_find(svc, full_path, root)
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)
    if existing:
        svc.files().update(fileId=existing, media_body=media).execute()
    else:
        svc.files().create(body={"name": name, "parents": [fid_dir]},
                           media_body=media, fields="id").execute()


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")


def search_mentions(query: str, doc_type: str | None) -> pd.DataFrame:
    if not MENTIONS_PATH.exists():
        sys.exit(f"No mentions index at {MENTIONS_PATH}. Run daily_research_summary.py first.")
    df = pd.read_parquet(MENTIONS_PATH)

    if doc_type:
        return df[df.doc_type == doc_type].copy()

    q = query.strip()
    if re.match(r"^INE[A-Z0-9]{9}$", q.upper()):
        return df[df.isin == q.upper()].copy()

    slug = _slug(q)
    words = [w for w in slug.split("_") if len(w) >= 4]
    mask = df.company_name_slug.str.contains(slug, na=False)
    for w in words:
        mask = mask | df.company_name_slug.str.contains(w, na=False)
    return df[mask].copy()


def get_summaries(research_ns: list[int]) -> dict[int, dict]:
    """Return {research_n: row_dict} from local ledger (has summary_md)."""
    if not LEDGER_PATH.exists():
        return {}
    idx = pd.read_parquet(LEDGER_PATH)
    if "summary_md" not in idx.columns:
        return {}
    subset = idx[idx.research_n.isin(research_ns)]
    out = {}
    for _, r in subset.iterrows():
        summary = str(r.get("summary_md", "")).strip()
        if not summary:
            continue
        out[int(r.research_n)] = {
            "summary_md": summary,
            "doc_date":   str(r.get("doc_date", "NA")),
            "source":     str(r.get("source", "NA")),
            "doc_type":   str(r.get("doc_type", "NA")),
            "file_name":  str(r.get("file_name", "NA")),
        }
    return out


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(label: str, summaries: dict[int, dict]) -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8")
    sorted_ns  = sorted(summaries, key=lambda n: summaries[n].get("doc_date", ""))
    sources    = sorted({summaries[n]["source"] for n in sorted_ns})
    dates      = [summaries[n]["doc_date"] for n in sorted_ns if summaries[n]["doc_date"] != "NA"]
    date_range = f"{min(dates)} to {max(dates)}" if dates else "unknown"

    doc_blocks = []
    for i, n in enumerate(sorted_ns, 1):
        s = summaries[n]
        header = (f"[doc_{i:02d}] research_{n:04d} | {s['doc_date']} | "
                  f"{s['source']} | {s['doc_type']} | {s['file_name']}")
        doc_blocks.append(f"--- {header} ---\n{s['summary_md']}")

    return (template
            .replace("{{N}}", str(len(sorted_ns)))
            .replace("{{SOURCES}}", ", ".join(sources))
            .replace("{{DATE_RANGE}}", date_range)
            .replace("{{DOCUMENTS}}", "\n\n".join(doc_blocks)))


# ---------------------------------------------------------------------------
# Drive: upload synthesis files for one ISIN
# ---------------------------------------------------------------------------
def upload_synthesis(svc, root: str, isin: str, slug_label: str,
                     header: str, body: str, stamp: str):
    """Upload dated archive + synthesis_latest.md to company_repo/<ISIN>/."""
    content = (header + body).encode("utf-8")
    dated_path  = f"{DRIVE_COMPANY_DIR}/{isin}/synthesis_{slug_label}_{stamp}.md"
    latest_path = f"{DRIVE_COMPANY_DIR}/{isin}/synthesis_latest.md"
    _drive_upload(svc, dated_path,  root, content, "text/markdown")
    _drive_upload(svc, latest_path, root, content, "text/markdown")
    print(f"   Drive: uploaded synthesis_latest.md and {dated_path.split('/')[-1]} -> {isin}/")


# ---------------------------------------------------------------------------
# Drive: enqueue ISINs in deep_dive_queue.parquet
# ---------------------------------------------------------------------------
def enqueue_deep_dive(svc, root: str, isins: list[str]):
    """Enqueue ISINs in deep_dive_queue via the ONE coordinated writer (lock + dedup),
    so this manual path can never clobber the queue or duplicate a name."""
    from company_deep_report import enqueue_tokens     # shared lock+dedup helper
    n = enqueue_tokens(svc, root, isins, owner="synthesise")
    print(f"   Queue: added {n} of {len(isins)} ISIN(s) (skipped already pending/done).")


# ---------------------------------------------------------------------------
# Core: synthesise one query label → result text + ISINs
# ---------------------------------------------------------------------------
def run_synthesis(label: str, doc_type: str | None,
                  pool, outdir: Path,
                  upload: bool, queue: bool,
                  svc=None, root: str = "") -> list[str]:
    """Run synthesis for one label. Returns list of ISINs synthesised."""
    matches = search_mentions(label, doc_type)
    if matches.empty:
        print(f"  No documents found for: {label!r}")
        return []

    research_ns = sorted(matches.research_n.dropna().astype(int).unique().tolist())
    summaries   = get_summaries(research_ns)
    if not summaries:
        print(f"  No summary_md available for {label!r} "
              "(data predates this feature — re-process PDFs to populate).")
        return []

    print(f"  [{label}] {len(summaries)} summaries found. Calling Gemini...")
    prompt = build_prompt(label, summaries)
    result = pool.call_text(prompt)[0]

    # Determine ISINs from matches (for Drive folder + queue)
    isins = sorted({str(i) for i in matches.isin.dropna() if str(i).startswith("INE")})

    stamp      = dt.datetime.now().strftime("%d%b%Y")
    slug_label = _slug(label)
    header     = (f"# Research Synthesis — {label}\n\n"
                  f"*Generated {dt.datetime.now():%d %b %Y %H:%M} IST · "
                  f"{len(summaries)} document(s)*\n\n---\n\n")

    # Local save
    outdir.mkdir(parents=True, exist_ok=True)
    fname   = f"synthesis_{slug_label}_{stamp}.md"
    outpath = outdir / fname
    outpath.write_text(fix_markdown_for_obsidian(header + result), encoding="utf-8")
    print(f"  Saved (Obsidian-fixed): {outpath}")

    # Drive upload
    if upload:
        if not isins:
            print(f"  WARN: no ISIN resolved for {label!r} — skipping Drive upload.")
        elif svc is None:
            print("  WARN: Drive upload requested but no Drive credentials (GDRIVE_OAUTH_TOKEN_JSON).")
        else:
            for isin in isins:
                upload_synthesis(svc, root, isin, slug_label, header, result, stamp)

    # Queue deep dive
    if queue:
        if not isins:
            print(f"  WARN: no ISIN resolved for {label!r} — nothing queued.")
        elif svc is None:
            print("  WARN: queue requested but no Drive credentials.")
        else:
            enqueue_deep_dive(svc, root, isins)

    return isins, outpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Synthesise all research docs mentioning a company via Gemini.")
    ap.add_argument("query", nargs="?", default="",
                    help="Company name, ISIN (INE...), or keyword")
    ap.add_argument("--doc-type",
                    help="Filter by doc_type instead of company")
    ap.add_argument("--all-new", action="store_true",
                    help=f"Process all ISINs in {NEW_ISINS_PATH} (written by daily_research_summary.py)")
    ap.add_argument("--upload", action="store_true",
                    help="Upload dated + synthesis_latest.md to Drive (company_repo/<ISIN>/)")
    ap.add_argument("--queue", action="store_true",
                    help="Add synthesised ISIN(s) to deep_dive_queue.parquet on Drive")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help=f"Local save directory (default: {DEFAULT_OUTDIR})")
    ap.add_argument("--no-open", action="store_true",
                    help="Do not open synthesis files in a browser (used for batch runs)")
    args = ap.parse_args()

    if not args.query and not args.doc_type and not args.all_new:
        ap.print_help(); sys.exit(1)

    # Load env for Drive if needed
    svc  = None
    root = os.environ.get("GDRIVE_FOLDER_ID", "")
    if (args.upload or args.queue) and root:
        try:
            svc = _drive_service()
        except Exception as e:
            print(f"WARN: Drive init failed ({e}) — upload/queue will be skipped.")

    pool   = build_daily_pool()
    outdir = Path(args.outdir)

    if args.all_new:
        if not NEW_ISINS_PATH.exists():
            print(f"No new_isins_latest.json found at {NEW_ISINS_PATH}. "
                  "Nothing to synthesise (run daily_research_summary.py first).")
            return
        new_isins = json.loads(NEW_ISINS_PATH.read_text(encoding="utf-8"))
        if not new_isins:
            print("new_isins_latest.json is empty — no new docs from last run."); return
        print(f"--all-new: processing {len(new_isins)} ISIN(s): {new_isins}")
        created = []
        for isin in new_isins:
            _, outpath = run_synthesis(isin, None, pool, outdir, args.upload, args.queue, svc, root)
            if outpath and outpath.exists():
                created.append(outpath)
        # open each synthesis file in Obsidian (same as other bat outputs)
        if created and not args.no_open:
            for p in created:
                open_in_obsidian(p)
        return

    # Single query
    label = args.query or args.doc_type
    _, outpath = run_synthesis(label, args.doc_type if not args.query else None,
                               pool, outdir, args.upload, args.queue, svc, root)

    # Open in Obsidian (same as daily digest and other outputs)
    if outpath and outpath.exists() and not args.no_open:
        open_in_obsidian(outpath)


if __name__ == "__main__":
    main()
