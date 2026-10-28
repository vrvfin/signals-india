r"""
test_drive_runner.py — offline checks for scripts/drive_runner.py (no Drive, no network).

  1. a bundle path that escapes the checkout ('..', absolute) is refused
  2. a bundle file that would OVERWRITE an existing file is refused
  3. a fingerprint mismatch stops the run before anything is extracted or executed
  4. a good bundle runs; the job's exit code and output are captured, NOT printed
  5. the public console line holds only job / fingerprint / status / duration / log name

Run:
    python scripts/tests/test_drive_runner.py
Exit code = number of failures.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import drive_runner as D  # noqa: E402

_FAILS: list[str] = []


def check(label, ok, extra=""):
    ok = bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {extra}" if extra else ""))
    if not ok:
        _FAILS.append(label)


def zipped(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, c in files.items():
            zf.writestr(n, c)
    return buf.getvalue()


def main():
    print("drive_runner")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        for bad in ("../evil.py", "/abs.py"):
            z = zipfile.ZipFile(io.BytesIO(zipped({bad: "x"})))
            try:
                D.safe_members(z, dest); ok = False
            except ValueError:
                ok = True
            check(f"refuses unsafe path {bad!r}", ok)
        (dest / "scripts").mkdir()
        (dest / "scripts" / "public.py").write_text("print('public')")
        z = zipfile.ZipFile(io.BytesIO(zipped({"scripts/public.py": "print('private')"})))
        try:
            D.safe_members(z, dest); ok = False
        except ValueError:
            ok = True
        check("refuses to overwrite an existing (public) file", ok)
        job = {"entry": "scripts/job.py", "args": ["--n", "3"], "timeout_min": 1}
        good = zipped({"scripts/job.py": "import sys\nprint('SECRET-RULE-OUTPUT', sys.argv[1:])\nsys.exit(5)\n"})
        code, log, fp = D.run_job(job, good, "0" * 64, dest)
        check("fingerprint mismatch -> nothing extracted or run", code == 97
              and not (dest / "scripts" / "job.py").exists())
        sha = hashlib.sha256(good).hexdigest()
        code, log, fp = D.run_job(job, good, sha, dest, ["--days", "400"], dry=True)
        check("dry run lists files, extracts nothing", code == 0 and b"scripts/job.py" in log
              and not (dest / "scripts" / "job.py").exists())
        code, log, fp = D.run_job(job, good, sha, dest, ["--days", "400"])
        check("job runs; exit code propagated (5)", code == 5, str(code))
        check("job output captured into the log (with args)", b"SECRET-RULE-OUTPUT" in log
              and b"--days" in log and b"--n" in log)
        line = D.console_line("flows", fp, "FAILED(5)", 12.3, "flows_x.log")
        check("console line carries no job output", "SECRET" not in line and line.startswith("drive_runner job=flows"),
              line)
    print(f"\n{'ALL PASS' if not _FAILS else 'FAILURES: ' + ', '.join(_FAILS)}")
    return len(_FAILS)


if __name__ == "__main__":
    sys.exit(main())
