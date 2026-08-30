r"""
isin_watch.py — daily ISIN-change watch: detect, apply to folders, track, mail.

WHAT RUNS EACH MORNING
    1. Pull the exchange's own security list (NSE main + SME, ~3,124 securities).
    2. Diff it against `universe/isin_state.csv` — ONE small file holding the
       ticker -> ISIN we last saw. It is overwritten each run, so nothing
       accumulates. (The earlier design kept a dated raw copy per day; that grew
       ~84 KB/day forever to answer a question one state file answers.)
    3. For every ticker whose ISIN changed, work out what the company's folder needs
       and DO it, one company at a time:
           only the OLD folder exists  -> RENAME it to the new ISIN
           BOTH folders exist          -> APPEND the old into the new (no rename)
           only the NEW folder exists  -> nothing to do, already correct
           neither exists              -> nothing to do, never ingested
    4. Append the change to `company_repo/_index/isin_changes.parquet`, recording the
       action taken and whether it succeeded. That file is the tracker: append-only,
       one row per change, never rewritten.
    5. Mail a table of what changed and whether the folder was already correct.
       Silent when nothing changed — a daily "nothing happened" mail trains you to
       ignore it, and this fires roughly once a week.

WHY A RENAME AND NOT ALWAYS A COPY
    If we spot the change before anything has been filed under the new ISIN, the old
    folder IS the company's whole history. Renaming keeps it in one piece and costs
    nothing. Only once both folders exist is there anything to reconcile — and then
    the old is appended into the new, because the new is where future extractions land.

APPEND FORMAT
    Page sections are joined by `merge_isin_folders.merge_pages`, which respects the
    existing `## <doc>` layout, keeps the `<!-- doc:... -->` markers, and on a
    collision keeps the RICHER text (the same rule extract_concall.py uses when a
    fuller document supersedes a thinner one). Documents are copied, never
    overwritten, and the old folder is never deleted.

USAGE
    python scripts/isin_watch.py --dry-run     # report only; touches nothing
    python scripts/isin_watch.py               # detect, apply, track, mail
    python scripts/isin_watch.py --seed        # first run: record state, apply nothing
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

from _extractor_base import (  # noqa: E402
    log, get_drive, get_or_create_subfolder, find_file, download_bytes, load_parquet,
    save_parquet,
)
from merge_isin_folders import (  # noqa: E402  — one page-merge implementation only
    merge_pages, _folder_id, _children, _read_text, _write_text,
)

NSE_MAIN = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_SME = "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
      "Accept": "text/csv,*/*"}

STATE_DIR, STATE_FILE = "universe", "isin_state.csv"
TRACKER = "isin_changes.parquet"
PAGE = "company_page.md"
TRACK_COLS = ["detected_on", "symbol", "exchange", "company_name",
              "old_isin", "new_isin", "folder_action", "folder_status",
              "detail", "applied_on"]


# ── Exchange list ────────────────────────────────────────────────────────────

def fetch_live() -> pd.DataFrame:
    """Current ticker -> ISIN, straight from the exchange."""
    frames = []
    for url, ex in ((NSE_MAIN, "NSE"), (NSE_SME, "NSE_SME")):
        r = requests.get(url, headers=UA, timeout=45)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        col = next((c for c in ("isin_number", "isin") if c in df.columns), None)
        if not col or "symbol" not in df.columns:
            raise ValueError(f"{ex}: unexpected columns {list(df.columns)}")
        name_col = next((c for c in ("name_of_company", "name") if c in df.columns), None)
        f = pd.DataFrame({
            "symbol": df["symbol"].astype(str).str.strip().str.upper(),
            "isin": df[col].astype(str).str.strip().str.upper(),
            "exchange": ex,
            "company_name": (df[name_col].astype(str).str.strip()
                             if name_col else ""),
        })
        f = f[f["isin"].str.match(r"^IN[A-Z0-9]{10}$", na=False)]
        log(f"  {ex}: {len(f)} securities")
        frames.append(f)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "exchange"])


def _csv_io(drive, folder_id, name):
    f = find_file(drive, folder_id, name)
    if not f:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(download_bytes(drive, f)))


def _csv_write(drive, folder_id, name, df):
    from googleapiclient.http import MediaInMemoryUpload
    media = MediaInMemoryUpload(df.to_csv(index=False).encode(), mimetype="text/csv")
    ex = find_file(drive, folder_id, name)
    if ex:
        drive.files().update(fileId=ex, media_body=media).execute()
    else:
        drive.files().create(body={"name": name, "parents": [folder_id]},
                             media_body=media, fields="id").execute()


# ── Folder actions ───────────────────────────────────────────────────────────

def apply_folder_change(drive, repo, old_isin, new_isin, symbol, dry):
    """Make the company's folder reflect the new ISIN. Returns (action, status, detail).

    Never deletes. A rename is reversible; an append copies and skips collisions."""
    o_id = _folder_id(drive, repo, old_isin)
    n_id = _folder_id(drive, repo, new_isin)

    if not o_id and not n_id:
        return "none", "ok", "no folder for this company yet"
    if not o_id and n_id:
        return "none", "ok", "already on the new ISIN"

    if o_id and not n_id:
        # Nothing has been filed under the new ISIN yet, so the old folder IS the
        # whole history. Renaming keeps it intact and copies nothing.
        if dry:
            return "rename", "would-do", f"{old_isin} -> {new_isin}"
        drive.files().update(fileId=o_id, body={"name": new_isin}).execute()
        return "rename", "done", f"renamed {old_isin} -> {new_isin}"

    # Both exist: reconcile into the new one, which is where future work lands.
    old_txt = _read_text(drive, o_id, PAGE)
    new_txt = _read_text(drive, n_id, PAGE)
    merged, st = merge_pages(old_txt, new_txt, old_isin, new_isin)
    o_kids = _children(drive, o_id)
    n_names = {c["name"] for c in _children(drive, n_id)}
    files = [c for c in o_kids if "folder" not in c["mimeType"]
             and c["name"] not in n_names and c["name"] != PAGE]
    o_docs = _folder_id(drive, o_id, "documents")
    docs = []
    if o_docs:
        n_docs = _folder_id(drive, n_id, "documents")
        have = {c["name"] for c in _children(drive, n_docs)} if n_docs else set()
        docs = [c for c in _children(drive, o_docs) if c["name"] not in have]
    detail = (f"page {st['old_secs']}+{st['new_secs']} -> {st['kept']} sections "
              f"({st['deduped']} deduped); {len(files)} file(s), {len(docs)} document(s)")
    if dry:
        return "append", "would-do", detail

    if new_txt:
        _write_text(drive, n_id, f".{PAGE}.bak", new_txt)
    _write_text(drive, n_id, PAGE, merged)
    for c in files:
        drive.files().copy(fileId=c["id"],
                           body={"name": c["name"], "parents": [n_id]}).execute()
    if docs:
        nd = get_or_create_subfolder(drive, n_id, "documents")
        for c in docs:
            drive.files().copy(fileId=c["id"],
                               body={"name": c["name"], "parents": [nd]}).execute()
    _write_text(drive, o_id, "superseded_by.txt",
                f"{symbol}: ISIN {old_isin} -> {new_isin}\n"
                f"Merged into company_repo/{new_isin}/ on {date.today()}.\n"
                f"Nothing deleted; kept as the record of what was filed here.\n")
    return "append", "done", detail


# ── Mail ─────────────────────────────────────────────────────────────────────

def send_report(rows, asof):
    from mailer import send_email, esc
    H = [f"<h2 style='margin:0'>ISIN change — {len(rows)} company(ies)</h2>",
         "<p style='color:#666;margin:4px 0 12px'>Detected by comparing the exchange's "
         "own security list against the ticker&rarr;ISIN we last saw.</p>",
         "<table cellpadding='6' cellspacing='0' style='border-collapse:collapse;"
         "font-size:13px'><tr><th align='left'>Ticker</th><th align='left'>Company</th>"
         "<th align='left'>Old ISIN</th><th align='left'>New ISIN</th>"
         "<th align='left'>Folder</th><th align='left'>What happened</th></tr>"]
    WORDS = {"rename": "renamed", "append": "old merged in",
             "none": "already correct"}
    for r in rows:
        colour = {"done": "#1a7a3c", "would-do": "#8a6d00",
                  "failed": "#c0392b"}.get(r["folder_status"], "#555")
        H.append(
            f"<tr><td><b>{esc(str(r['symbol']))}</b></td>"
            f"<td>{esc(str(r['company_name'])[:34])}</td>"
            f"<td><code>{esc(str(r['old_isin']))}</code></td>"
            f"<td><code>{esc(str(r['new_isin']))}</code></td>"
            f"<td style='color:{colour}'><b>{WORDS.get(r['folder_action'], r['folder_action'])}"
            f"</b></td><td style='color:#666'>{esc(str(r['detail'])[:70])}</td></tr>")
    H.append("</table><p style='color:#888;font-size:12px'>Folder column: "
             "<b>renamed</b> = only the old folder existed, so it now carries the new "
             "ISIN. <b>old merged in</b> = both existed, so the old was appended into "
             "the new (nothing deleted). <b>already correct</b> = nothing to do.<br>"
             "Data tables are NOT repointed by this job — run repoint_isin.py for that."
             "</p>")
    send_email(f"ISIN change — {len(rows)} company(ies) on {asof}", "".join(H))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    ap.add_argument("--seed", action="store_true",
                    help="Record today's state as the baseline; apply and mail nothing.")
    ap.add_argument("--no-mail", action="store_true", help="Skip the email.")
    args = ap.parse_args()
    asof = date.today()
    dry = args.dry_run

    log("=" * 72)
    log(f"isin_watch — {asof}{'  [DRY RUN]' if dry else ''}{'  [SEED]' if args.seed else ''}")
    log("=" * 72)

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    index_id = get_or_create_subfolder(drive, repo, "_index")
    uni = get_or_create_subfolder(drive, root, STATE_DIR)

    live = fetch_live()
    state = _csv_io(drive, uni, STATE_FILE)

    if state.empty or args.seed:
        log(f"  no prior state — seeding {len(live)} securities. "
            f"Detection starts on the next run.")
        if not dry:
            _csv_write(drive, uni, STATE_FILE,
                       live.assign(last_seen=str(asof))[
                           ["symbol", "exchange", "isin", "company_name", "last_seen"]])
        return

    prev = {(r.symbol, r.exchange): r.isin for r in state.itertuples()}
    changes = [dict(symbol=r.symbol, exchange=r.exchange, company_name=r.company_name,
                    old_isin=prev[(r.symbol, r.exchange)], new_isin=r.isin)
               for r in live.itertuples()
               if prev.get((r.symbol, r.exchange), r.isin) != r.isin]

    if not changes:
        log(f"  no ISIN changes ({len(live)} securities checked). State refreshed.")
        if not dry:
            _csv_write(drive, uni, STATE_FILE,
                       live.assign(last_seen=str(asof))[
                           ["symbol", "exchange", "isin", "company_name", "last_seen"]])
        return

    log(f"  {len(changes)} ISIN change(s) detected:")
    rows = []
    for c in changes:
        log(f"    {c['symbol']:<14} {c['old_isin']} -> {c['new_isin']}")
        try:
            action, status, detail = apply_folder_change(
                drive, repo, c["old_isin"], c["new_isin"], c["symbol"], dry)
        except Exception as e:                       # one failure must not stop the rest
            action, status, detail = "unknown", "failed", str(e)[:150]
        log(f"      folder: {action} — {status} ({detail})")
        rows.append({**c, "detected_on": str(asof), "folder_action": action,
                     "folder_status": status, "detail": detail,
                     "applied_on": ("" if dry else str(asof))})

    if not dry:
        track = load_parquet(drive, index_id, TRACKER, TRACK_COLS)
        track = pd.concat([track, pd.DataFrame(rows, columns=TRACK_COLS)],
                          ignore_index=True)
        save_parquet(drive, index_id, TRACKER, track)
        log(f"  tracker: {len(track)} row(s) in company_repo/_index/{TRACKER}")
        _csv_write(drive, uni, STATE_FILE,
                   live.assign(last_seen=str(asof))[
                       ["symbol", "exchange", "isin", "company_name", "last_seen"]])
        if not args.no_mail:
            send_report(rows, asof)
    else:
        log("  [DRY RUN] nothing written, nothing mailed.")

    log("")
    log("Data tables are NOT repointed here — run repoint_isin.py for that.")


if __name__ == "__main__":
    main()
