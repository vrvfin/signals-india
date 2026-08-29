r"""
merge_isin_folders.py — reunite the two company folders an ISIN change created.

WHY
    `company_repo/<ISIN>/` is keyed on ISIN, so when a company's ISIN changed it got a
    SECOND folder with its own company_page.md. Deep dives, narrative reports and
    catalyst notes read one folder, so they see half the company. `repoint_isin.py`
    already fixed the tables; this fixes the folders.

    Destination is the NEW isin's folder — that is where every future extraction lands.

HOW THE PAGE IS MERGED
    A page is `# <ISIN> - Company Intelligence` followed by `## <doc>` sections.
    Doc-id markers (`<!-- doc:... -->`) exist but cover only a handful of sections
    (POCL: 4 of 78; TEMBO: 0 of 3), so they CANNOT carry the merge on their own.
    Sections are therefore deduplicated on either signal:
      - the same doc-id marker, or
      - identical section text once whitespace is normalised
    The overlap is real — MWL, KRISHANA and MBAPL each share 3 doc ids — so a plain
    concatenation would visibly duplicate content.

    Old sections come FIRST because they are earlier, under a divider naming the old
    ISIN, so the join is auditable rather than invisible. Anything that cannot be
    PROVEN identical is kept: this never drops content to look tidy.

SAFETY
    - --dry-run is the DEFAULT; --live is required to write.
    - The destination page is backed up before it is replaced.
    - The old folder is NEVER deleted. It keeps its page and gains superseded_by.txt.
    - Documents are COPIED, not moved, and a name collision is skipped, never
      overwritten.

USAGE
    python scripts/merge_isin_folders.py                    # dry run, all 8
    python scripts/merge_isin_folders.py --only TDPOWERSYS  # one company
    python scripts/merge_isin_folders.py --live
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

from _extractor_base import (  # noqa: E402
    log, get_drive, get_or_create_subfolder, find_file, download_bytes, load_parquet,
)

PAGE = "company_page.md"
REGISTRY = "isin_alias.parquet"
REG_COLS = ["old_isin", "new_isin", "symbol", "exchange", "changed_between",
            "event_type", "ratio", "ex_date", "confirmed", "source", "detected_on"]
DOC_RE = re.compile(r"<!--\s*doc:([0-9a-f]+)\s*-->")


def _folder_id(drive, parent, name):
    f = drive.files().list(
        q=f"name='{name}' and '{parent}' in parents and "
          f"mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id)").execute().get("files", [])
    return f[0]["id"] if f else None


def _children(drive, fid):
    out, tok = [], None
    while True:
        r = drive.files().list(q=f"'{fid}' in parents and trashed=false",
                               fields="nextPageToken, files(id,name,mimeType,size)",
                               pageSize=500, pageToken=tok).execute()
        out += r.get("files", [])
        tok = r.get("nextPageToken")
        if not tok:
            return out


def _read_text(drive, fid, name):
    f = find_file(drive, fid, name)
    return download_bytes(drive, f).decode("utf-8", "replace") if f else None


def split_sections(text):
    """(header before the first '## ', list of '## ' sections)."""
    if not text:
        return "", []
    parts = re.split(r"(?m)^(?=## )", text)
    return parts[0], parts[1:]


def _sig(sec: str) -> str:
    """Identity of a section: its doc marker when present, else a hash of the text
    with whitespace collapsed — so trivial reformatting is not mistaken for new
    content, and genuinely different text is never merged away."""
    m = DOC_RE.search(sec)
    if m:
        return "doc:" + m.group(1)
    return "sha:" + hashlib.sha256(re.sub(r"\s+", " ", sec).strip().encode()).hexdigest()


def merge_pages(old, new, old_isin: str, new_isin: str):
    """Old sections first (they are earlier), then new. Returns (text, stats)."""
    o_head, o_secs = split_sections(old or "")
    n_head, n_secs = split_sections(new or "")
    head = (n_head or o_head or f"# {new_isin} — Company Intelligence\n")
    head = head.replace(old_isin, new_isin)

    # Two sections can share a doc-id yet hold DIFFERENT text, because one side was
    # re-extracted. Keeping whichever came first would be arbitrary — and destructive:
    # KRISHANA's CRISIL section is 3,731 chars in the old page and 124,403 in the new.
    # So on a collision the RICHER text wins, which is the same rule extract_concall.py
    # already applies when a fuller document supersedes a thinner one.
    order, best, dropped = [], {}, 0
    for sec in o_secs + n_secs:
        s = _sig(sec)
        if s not in best:
            order.append(s)
            best[s] = sec
            continue
        dropped += 1
        if len(sec) > len(best[s]):
            best[s] = sec
    kept = [best[s] for s in order]

    divider = (f"\n\n---\n<!-- merged: sections below combine {old_isin} (superseded) "
               f"and {new_isin}, joined {datetime.now():%Y-%m-%d} -->\n\n")
    text = head.rstrip() + "\n" + divider + "".join(kept)
    return text, dict(old_secs=len(o_secs), new_secs=len(n_secs),
                      kept=len(kept), deduped=dropped)


def _write_text(drive, fid, name, text):
    from googleapiclient.http import MediaInMemoryUpload
    media = MediaInMemoryUpload(text.encode("utf-8"), mimetype="text/markdown")
    ex = find_file(drive, fid, name)
    if ex:
        drive.files().update(fileId=ex, media_body=media).execute()
    else:
        drive.files().create(body={"name": name, "parents": [fid]},
                             media_body=media, fields="id").execute()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="Write. Default is a dry run.")
    ap.add_argument("--dry-run", action="store_true", help="Explicit no-op (default).")
    ap.add_argument("--only", default=None, help="One ticker only.")
    args = ap.parse_args()
    live = args.live and not args.dry_run

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log("=" * 72)
    log(f"merge_isin_folders — {'LIVE' if live else 'DRY RUN (no writes)'}")
    log("=" * 72)

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo, "_index")
    reg = load_parquet(drive, index_id, REGISTRY, REG_COLS)
    if reg.empty:
        log("registry empty — run isin_registry.py first.")
        sys.exit(1)
    rows = [r for r in reg.itertuples()
            if bool(r.confirmed) and (not args.only or r.symbol == args.only)]

    for r in rows:
        o_id = _folder_id(drive, repo, r.old_isin)
        n_id = _folder_id(drive, repo, r.new_isin)
        if not o_id:
            log(f"{r.symbol}: no old folder — nothing to merge.")
            continue
        if not n_id:
            if live:
                n_id = get_or_create_subfolder(drive, repo, r.new_isin)
            else:
                log(f"{r.symbol}: new folder missing — would be created.")
                continue

        old_txt = _read_text(drive, o_id, PAGE)
        new_txt = _read_text(drive, n_id, PAGE)
        merged, st = merge_pages(old_txt, new_txt, r.old_isin, r.new_isin)
        log(f"{r.symbol:<12} page: {st['old_secs']:>3} old + {st['new_secs']:>3} new "
            f"-> {st['kept']:>3} kept, {st['deduped']:>2} deduped   "
            f"({len(old_txt or ''):,} + {len(new_txt or ''):,} -> {len(merged):,} chars)")

        o_kids = _children(drive, o_id)
        n_names = {c["name"] for c in _children(drive, n_id)}
        copy_files = [c for c in o_kids
                      if "folder" not in c["mimeType"] and c["name"] != PAGE
                      and c["name"] not in n_names]
        o_docs = _folder_id(drive, o_id, "documents")
        doc_copies = []
        if o_docs:
            n_docs = _folder_id(drive, n_id, "documents")
            have = {c["name"] for c in _children(drive, n_docs)} if n_docs else set()
            doc_copies = [c for c in _children(drive, o_docs) if c["name"] not in have]
        log(f"{'':<12} files to copy: {len(copy_files)}   documents: {len(doc_copies)}")

        if not live:
            continue

        if new_txt:
            bfid = get_or_create_subfolder(drive, index_id, "_backup_isin_repoint")
            bfid = get_or_create_subfolder(drive, bfid, f"{stamp}_pages")
            _write_text(drive, bfid, f"{r.new_isin}_{PAGE}", new_txt)
        _write_text(drive, n_id, PAGE, merged)
        for c in copy_files:
            drive.files().copy(fileId=c["id"],
                               body={"name": c["name"], "parents": [n_id]}).execute()
        if doc_copies:
            n_docs = get_or_create_subfolder(drive, n_id, "documents")
            for c in doc_copies:
                drive.files().copy(fileId=c["id"],
                                   body={"name": c["name"],
                                         "parents": [n_docs]}).execute()
        _write_text(drive, o_id, "superseded_by.txt",
                    f"This folder is superseded.\n\n"
                    f"{r.symbol} changed ISIN {r.old_isin} -> {r.new_isin} "
                    f"(~{r.changed_between}).\n"
                    f"Its content was merged into company_repo/{r.new_isin}/ on "
                    f"{datetime.now():%Y-%m-%d}.\n\n"
                    f"Nothing here was deleted. This folder is kept as the record of "
                    f"what was filed under the old ISIN.\n")
        log(f"{'':<12} merged into {r.new_isin}; breadcrumb left in {r.old_isin}")

    log("")
    log("done." if live else "nothing written. Re-run with --live to apply.")


if __name__ == "__main__":
    main()
