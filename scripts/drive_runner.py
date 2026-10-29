"""
drive_runner.py — run a PRIVATE job whose code lives on Google Drive, from a public CI repo.

Why: the repo is public (free Actions minutes) but the strategy code and rules must stay
private. GitHub can only start jobs defined in the repo, so this generic runner is the one
piece that has to live here. It holds NO strategy code, rules, symbols or results.

What it does, for `--job NAME`:
  1. reads Drive  <root>/guru_config/jobs.json          -> {bundle, entry, args, timeout_min}
  2. downloads    <root>/guru_config/code/<bundle>.zip  + <bundle>.sha256 and checks they match
  3. unzips the bundle into the checkout; refuses any path that escapes it or that would
     OVERWRITE an existing file (private code may add files, never replace public ones)
  4. runs  python <entry> <args> <--extra ...>  with ALL output captured to a log file
  5. uploads the log to Drive <root>/guru_config/logs/<job>_<UTC>.log
  6. prints ONE line to the (public) console: job, bundle fingerprint, status, duration
  exit code = the job's exit code (a failed job still turns the CI run red).

    python scripts/drive_runner.py --job flows --dry-run     # read jobs.json, verify bundle,
                                                             # list files; run nothing
    python scripts/drive_runner.py --job flows [--extra "--days 400"]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "guru_config"


# ------------------------------------------------------------------ pure parts

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def safe_members(zf: zipfile.ZipFile, dest: Path) -> list[str]:
    """Bundle paths that are safe to extract under `dest`; raises on any path that is
    absolute, climbs out ('..'), or would overwrite an existing file."""
    out = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or ":" in name or "\\" in name:
            raise ValueError(f"unsafe path in bundle: {name!r}")
        target = (dest / Path(*p.parts)).resolve()
        if dest.resolve() not in target.parents:
            raise ValueError(f"path escapes the checkout: {name!r}")
        if target.exists():
            raise ValueError(f"bundle would overwrite an existing file: {name!r}")
        out.append(name)
    return out


def console_line(job: str, fp: str, status: str, secs: float, log_name: str) -> str:
    """The ONLY thing printed publicly. Nothing from the job's own output."""
    return f"drive_runner job={job} bundle={fp[:12]} status={status} duration={secs:.0f}s log={log_name}"


def run_job(cfg: dict, zip_bytes: bytes, expected_sha: str, dest: Path,
            extra: list[str] | None = None, dry: bool = False) -> tuple[int, bytes, str]:
    """Verify + extract + run. -> (exit code, captured log bytes, fingerprint). No Drive."""
    fp = sha256(zip_bytes)
    if expected_sha and fp != expected_sha.strip():
        return 97, f"bundle fingerprint mismatch: got {fp}, expected {expected_sha.strip()}\n".encode(), fp
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    members = safe_members(zf, dest)
    if cfg["entry"] not in members:
        return 98, f"entry {cfg['entry']} not in bundle\n".encode(), fp
    if dry:
        return 0, ("DRY RUN — bundle files:\n" + "\n".join(members) + "\n").encode(), fp
    for m in members:
        zf.extract(m, dest)
    cmd = [sys.executable, str(dest / cfg["entry"])] + list(cfg.get("args", [])) + list(extra or [])
    try:
        p = subprocess.run(cmd, cwd=dest, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=int(cfg.get("timeout_min", 60)) * 60)
        return p.returncode, p.stdout, fp
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or b"") + b"\nTIMEOUT\n", fp


# ------------------------------------------------------------------ Drive I/O

def main():
    ap = argparse.ArgumentParser(description="run a private Drive-hosted job")
    ap.add_argument("--job", required=True)
    ap.add_argument("--extra", default="", help="extra args appended to the job's own")
    ap.add_argument("--dry-run", action="store_true", help="verify + list bundle; run nothing")
    ap.add_argument("--dest", default=str(ROOT), help="checkout to extract into (default repo root)")
    a = ap.parse_args()
    t0 = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_name = f"{a.job}_{stamp}Z.log"
    sys.path.insert(0, str(ROOT / "scripts"))
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                                 download_bytes, upload_bytes)
    code, log, fp = 99, b"", "-" * 12
    drive = None
    try:
        drive = get_drive()
        cfg_id = get_or_create_subfolder(drive, os.environ["GDRIVE_FOLDER_ID"], FOLDER)
        jid = find_file(drive, cfg_id, "jobs.json")
        if not jid:
            raise RuntimeError("jobs.json missing on Drive")
        cfg = json.loads(download_bytes(drive, jid)).get(a.job)
        if not cfg:
            raise RuntimeError(f"job {a.job!r} not in jobs.json")
        code_id = get_or_create_subfolder(drive, cfg_id, "code")
        zid, sid = find_file(drive, code_id, f"{cfg['bundle']}.zip"), find_file(drive, code_id, f"{cfg['bundle']}.sha256")
        if not zid or not sid:
            raise RuntimeError(f"bundle {cfg['bundle']} (.zip/.sha256) missing on Drive")
        code, log, fp = run_job(cfg, download_bytes(drive, zid), download_bytes(drive, sid).decode(),
                                Path(a.dest), shlex.split(a.extra), a.dry_run)
    except Exception as e:                       # runner-level failure: detail goes to the log only
        code, log = 96, f"runner error: {e.__class__.__name__}: {e}\n".encode()
    status = "OK" if code == 0 else f"FAILED({code})"
    if a.dry_run:
        status = "DRY-OK" if code == 0 else status
    try:
        if drive is not None:
            cfg_id = get_or_create_subfolder(drive, os.environ["GDRIVE_FOLDER_ID"], FOLDER)
            logs_id = get_or_create_subfolder(drive, cfg_id, "logs")
            upload_bytes(drive, logs_id, log_name, log, "text/plain")
        else:
            log_name = "not-uploaded"
    except Exception:
        log_name = "upload-failed"
    print(console_line(a.job, fp, status, time.time() - t0, log_name), flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
