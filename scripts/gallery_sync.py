r"""
gallery_sync.py — move the chart galleries between CI and this PC via Drive.

  publish  upload the pages this run built to Drive company_repo/_galleries/ and MERGE
           their entries into manifest.json (per page: ok / failed, built_at, builder).
           --owner ci  (galleries.yml, after Phase 1): the pages CI builds
           --owner pc  (the PC's GalleriesRefresh task): the pages only this PC can build
           Each publisher touches only its own pages, so neither marks the other's failed.
  fetch    download the pages newer than the local copy (galleries_home.py / the bats)
  inputs-push / inputs-pull   the two small local inputs the market-state page needs
           (market breadth table + deep macro history): the PC pushes them to Drive
           _galleries/inputs/ when they change, CI pulls them into the same local paths
           before building market_state.html (the builder itself is unchanged)

Drive is the single source: CI builds every page whose data is on Drive; the PC builds
gallery_zanger.html (local library + Zanger images, by design) and
gallery_pf_history.html (local 25y price store) and uploads them too.

Privacy: the repo is public. The PF pages (holdings) are built in CI with their output
suppressed (galleries.yml) and go to Drive only — never to a GitHub artifact.

Usage:
    python scripts/gallery_sync.py publish --since <unix-ts> [--owner ci|pc] [--only P..] [--dry-run]
    python scripts/gallery_sync.py fetch --set all|pf [--dry-run] [--no-open] [--no-fallback]
    python scripts/gallery_sync.py inputs-push [--dry-run]      # PC
    python scripts/gallery_sync.py inputs-pull                  # CI
Plan: .claude/plans/galleries-ci-fetch.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SCRIPTS_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from dotenv import load_dotenv                      # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))

FOLDER = "_galleries"                               # under company_repo/
MANIFEST = "manifest.json"
IST = timezone(timedelta(hours=5, minutes=30))
MCAP = ["--min-mcap", "150"]                        # = gallery_all.bat's MCAP
BG = ["scripts/build_gallery.py"]

# page -> which set fetches it + its build command. builder: "ci" (default, galleries.yml)
# or "pc" (built + published by the PC's GalleriesRefresh task). "drive" overrides where
# the copy lives (Darvas is published by darvas_gallery.yml).
PAGES = {
    "market_state.html":               {"sets": ["all", "pf"], "cmd": ["scripts/build_market_state_html.py"]},
    "gallery.html":                    {"sets": ["all"], "cmd": BG + ["--view", "full"] + MCAP},
    "gallery_additions.html":          {"sets": ["all"], "cmd": BG + ["--view", "additions"] + MCAP},
    "gallery_drops.html":              {"sets": ["all"], "cmd": BG + ["--view", "drops"] + MCAP},
    "gallery_ipo.html":                {"sets": ["all"], "cmd": BG + ["--view", "ipo"] + MCAP},
    "gallery_guidance_watchlist.html": {"sets": ["all"], "cmd": BG + ["--mode", "watchlist"] + MCAP},
    "gallery_guru.html":               {"sets": ["all"], "cmd": ["guru/build_guru_gallery.py"]},
    "gallery_new_today.html":          {"sets": ["all"], "cmd": BG + ["--view", "new"] + MCAP},
    "gallery_tracker.html":            {"sets": ["all", "pf"], "cmd": ["scripts/build_tracker_html.py"]},
    "gallery_darvas.html":             {"sets": ["all"], "drive": ("_daily", "darvas_gallery.html"),
                                        "cmd": ["guru/darvas/darvas_gallery.py", "--sync", "--no-publish"],
                                        "local_out": "guru/darvas/darvas_gallery.html"},
    "gallery_pf.html":                 {"sets": ["pf"], "cmd": BG + ["--mode", "pf", "--resample", "D"]},
    "gallery_pf_weekly.html":          {"sets": ["pf"], "cmd": BG + ["--mode", "pf", "--resample", "W"]},
    "gallery_pf_monthly.html":         {"sets": ["pf"], "cmd": BG + ["--mode", "pf", "--resample", "M"]},
    "gallery_sold.html":               {"sets": ["pf"], "cmd": ["scripts/build_sold_gallery.py"]},
    "gallery_zanger.html":             {"sets": ["all"], "builder": "pc",
                                        "cmd": ["guru/patterns/build_pattern_gallery.py"]},
    "gallery_pf_history.html":         {"sets": ["pf"], "builder": "pc",
                                        "cmd": ["scripts/build_pf_history.py"]},
}

# market-state inputs: Drive name in _galleries/inputs/ <-> local path (relative to ROOT)
INPUTS_FOLDER = "inputs"
INPUT_DIRS = {"metrics__": "guru/data/metrics", "macro_hist__": "guru/data/macro_hist"}


def _input_files() -> dict:
    """local market-state inputs -> their Drive names."""
    out = {}
    p = os.path.join(ROOT, "guru", "data", "metrics", "market_breadth.parquet")
    if os.path.exists(p):
        out["metrics__market_breadth.parquet"] = p
    mh = os.path.join(ROOT, "guru", "data", "macro_hist")
    if os.path.isdir(mh):
        for f in sorted(os.listdir(mh)):
            if f.endswith(".parquet"):
                out[f"macro_hist__{f}"] = os.path.join(mh, f)
    return out


def _builder(spec: dict) -> str:
    return spec.get("builder", "ci")


def log(m):
    print(datetime.now().strftime("%H:%M:%S"), m, flush=True)


def _drive():
    from _extractor_base import get_drive, get_or_create_subfolder
    d = get_drive()
    repo = get_or_create_subfolder(d, os.environ["GDRIVE_FOLDER_ID"], "company_repo")
    return d, repo, get_or_create_subfolder


def _ist(iso_utc: str | None) -> str:
    if not iso_utc:
        return "-"
    t = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return t.astimezone(IST).strftime("%d %b %H:%M IST")


# ---------------------------------------------------------------- publish
def publish(since: float, dry_run: bool, only: list[str] | None = None,
            owner: str = "ci") -> int:
    """upload `owner`'s pages built after `since` (run start); its other pages are
    marked 'failed'. only: this run rebuilt just these pages (e.g. the guru page after
    the late guru screen). The manifest is always MERGED: pages of the other owner, and
    pages outside `only`, keep their existing entries."""
    from _extractor_base import find_file, download_bytes, upload_bytes
    from drive_io import drive_call
    manifest = {"built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "pages": {}}
    d = fid = None
    if not dry_run:
        d, repo, sub = _drive()
        fid = drive_call(lambda: sub(d, repo, FOLDER), label="galleries folder")
        mid = drive_call(lambda: find_file(d, fid, MANIFEST), label=MANIFEST)
        if mid:
            old = json.loads(drive_call(lambda: download_bytes(d, mid), label=MANIFEST))
            manifest["pages"] = old.get("pages", {})
    todo, mine = [], []
    for page, spec in PAGES.items():
        if "drive" in spec or _builder(spec) != owner or (only and page not in only):
            continue                                  # Darvas: own workflow
        mine.append(page)
        p = os.path.join(ROOT, page)
        fresh = os.path.exists(p) and os.path.getmtime(p) >= since
        manifest["pages"][page] = {
            "status": "ok" if fresh else "failed", "builder": owner,
            "built_at_utc": (datetime.fromtimestamp(os.path.getmtime(p), timezone.utc)
                             .isoformat(timespec="seconds") if fresh else None),
            "mb": round(os.path.getsize(p) / 1e6, 2) if fresh else None}
        if fresh:
            todo.append(page)
    n_bad = sum(manifest["pages"][p]["status"] != "ok" for p in mine)
    log(f"[{owner}] {len(todo)} pages built this run, {n_bad} failed/missing")
    for page in mine:
        v = manifest["pages"][page]
        print(f"  {page:34s} {v['status']:7s} {v['mb'] or '':>6} MB")
    if dry_run:
        log("DRY RUN: nothing uploaded"); return 0
    for page in todo:
        with open(os.path.join(ROOT, page), "rb") as fh:
            data = fh.read()
        drive_call(lambda: upload_bytes(d, fid, page, data, "text/html",
                                        existing_id=find_file(d, fid, page)), label=page)
    body = json.dumps(manifest, indent=1).encode("utf-8")
    drive_call(lambda: upload_bytes(d, fid, MANIFEST, body, "application/json",
                                    existing_id=find_file(d, fid, MANIFEST)), label=MANIFEST)
    log(f"published {len(todo)} pages + {MANIFEST} to Drive company_repo/{FOLDER}/")
    return 0


# ---------------------------------------------------------------- market-state inputs
def _inputs_folder():
    from drive_io import drive_call
    d, repo, sub = _drive()
    gid = drive_call(lambda: sub(d, repo, FOLDER), label="galleries folder")
    iid = drive_call(lambda: sub(d, gid, INPUTS_FOLDER), label="inputs folder")
    listing = drive_call(lambda: d.files().list(
        q=f"'{iid}' in parents and trashed=false",
        fields="files(id,name,modifiedTime)", pageSize=200).execute(), label="inputs list")
    return d, iid, {f["name"]: f for f in listing.get("files", [])}


def inputs_push(dry_run: bool) -> int:
    """PC: upload the market-state inputs that are newer than their Drive copy."""
    from _extractor_base import upload_bytes
    from drive_io import drive_call
    local = _input_files()
    if not local:
        log("no local market-state inputs found — nothing to push"); return 0
    d, iid, remote = _inputs_folder()
    todo = []
    for name, path in local.items():
        lt = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
        rt = remote.get(name, {}).get("modifiedTime")
        if not rt or lt > datetime.fromisoformat(rt.replace("Z", "+00:00")):
            todo.append((name, path))
    log(f"market-state inputs: {len(local)} local, {len(todo)} newer than Drive")
    for name, path in todo:
        log(f"  {'WOULD push' if dry_run else 'push'} {name}")
        if not dry_run:
            with open(path, "rb") as fh:
                data = fh.read()
            drive_call(lambda: upload_bytes(d, iid, name, data, "application/octet-stream",
                                            existing_id=remote.get(name, {}).get("id")), label=name)
    return 0


def inputs_pull() -> int:
    """CI: download the market-state inputs into the local paths the builder reads."""
    from _extractor_base import download_bytes
    from drive_io import drive_call
    d, iid, remote = _inputs_folder()
    n = 0
    for name, f in remote.items():
        pre = next((p for p in INPUT_DIRS if name.startswith(p)), None)
        if not pre:
            continue
        dest = os.path.join(ROOT, INPUT_DIRS[pre], name[len(pre):])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(drive_call(lambda: download_bytes(d, f["id"]), label=name))
        n += 1
    log(f"pulled {n} market-state input files from Drive")
    return 0


# ---------------------------------------------------------------- fetch
def _build_local(page: str, spec: dict) -> bool:
    py = sys.executable
    cmd = [py] + spec["cmd"] + (["--no-open"] if spec["cmd"][0] != "guru/darvas/darvas_gallery.py" else [])
    log(f"  building {page} locally: {' '.join(spec['cmd'])}")
    ok = subprocess.run(cmd, cwd=ROOT).returncode == 0
    if ok and spec.get("local_out"):
        src = os.path.join(ROOT, spec["local_out"])
        with open(src, "rb") as a, open(os.path.join(ROOT, page), "wb") as b:
            b.write(a.read())
    return ok


def fetch(which: str, dry_run: bool, no_open: bool, no_fallback: bool) -> int:
    from _extractor_base import find_file, download_bytes
    from drive_io import drive_call
    d, repo, sub = _drive()
    fid = drive_call(lambda: sub(d, repo, FOLDER), label="galleries folder")
    mid = drive_call(lambda: find_file(d, fid, MANIFEST), label=MANIFEST)
    man = (json.loads(drive_call(lambda: download_bytes(d, mid), label=MANIFEST))
           if mid else {"pages": {}})
    log(f"CI galleries last built {_ist(man.get('built_at_utc'))}")
    rows, opened, failed = [], [], []
    for page, spec in PAGES.items():
        if which not in spec["sets"]:
            continue
        local = os.path.join(ROOT, page)
        local_t = (datetime.fromtimestamp(os.path.getmtime(local), timezone.utc)
                   if os.path.exists(local) else None)
        if "drive" in spec:                       # Darvas: its own Drive file
            folder, name = spec["drive"]
            dfid = drive_call(lambda: sub(d, repo, folder), label=folder)
            xid = drive_call(lambda: find_file(d, dfid, name), label=name)
            meta = (drive_call(lambda: d.files().get(fileId=xid, fields="modifiedTime").execute(),
                               label=name) if xid else {})
            built, status = meta.get("modifiedTime"), ("ok" if xid else "missing")
        else:
            v = man["pages"].get(page, {})
            xid = drive_call(lambda: find_file(d, fid, page), label=page) if v.get("status") == "ok" else None
            built, status = v.get("built_at_utc"), v.get("status", "missing")
            if status == "ok" and not xid:
                status = "missing"
        if status == "ok":
            bt = datetime.fromisoformat(built.replace("Z", "+00:00"))
            if local_t and local_t >= bt:
                action = "local copy is newer"
            else:
                action = "download"
                if not dry_run:
                    data = drive_call(lambda: download_bytes(d, xid), label=page)
                    with open(local, "wb") as fh:
                        fh.write(data)
            opened.append(page)
        else:
            action = "BUILD LOCALLY" if not no_fallback else "skipped (no fallback)"
            failed.append(page)
        rows.append((page, status, _ist(built), action))
    print(f"\n  {'page':34s} {'CI status':9s} {'CI built':18s} action")
    for r in rows:
        print(f"  {r[0]:34s} {r[1]:9s} {r[2]:18s} {r[3]}")
    print()
    if dry_run:
        log("DRY RUN: nothing downloaded or built"); return 0
    bad = []
    if not no_fallback:
        for page in failed:
            if _build_local(page, PAGES[page]):
                opened.append(page)
            else:
                bad.append(page)
    if not no_open:
        for page in opened:
            webbrowser.open("file://" + os.path.abspath(os.path.join(ROOT, page)))
    if bad:
        log(f"FAILED (CI and local): {', '.join(bad)}")
        return 1
    log(f"done: {len(opened)} pages ready")
    return 0


def main():
    ap = argparse.ArgumentParser(description="publish / fetch chart galleries via Drive")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("publish")
    p.add_argument("--since", type=float, required=True, help="job start, unix seconds")
    p.add_argument("--only", nargs="+", choices=[k for k, v in PAGES.items() if "drive" not in v],
                   help="pages this run rebuilt; the rest keep their manifest entry")
    p.add_argument("--owner", choices=["ci", "pc"], default="ci",
                   help="which builder's pages this publish covers")
    p.add_argument("--dry-run", action="store_true")
    f = sp.add_parser("fetch")
    f.add_argument("--set", choices=["all", "pf"], default="all")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--no-open", action="store_true")
    f.add_argument("--no-fallback", action="store_true")
    ip = sp.add_parser("inputs-push")
    ip.add_argument("--dry-run", action="store_true")
    sp.add_parser("inputs-pull")
    a = ap.parse_args()
    if a.cmd == "publish":
        sys.exit(publish(a.since, a.dry_run, a.only, a.owner))
    if a.cmd == "inputs-push":
        sys.exit(inputs_push(a.dry_run))
    if a.cmd == "inputs-pull":
        sys.exit(inputs_pull())
    sys.exit(fetch(a.set, a.dry_run, a.no_open, a.no_fallback))


if __name__ == "__main__":
    main()
