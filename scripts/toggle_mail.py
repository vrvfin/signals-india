r"""
toggle_mail.py — flip the nightly mail toggles from the local machine.

Same company_repo/_index/mail_settings.json the Streamlit sidebar edits
("📧 Email toggles"); CI mailers read it before sending. Missing file/key = ON.

Usage:
    python scripts/toggle_mail.py                     # interactive (toggle_mail.bat)
    python scripts/toggle_mail.py --show              # print current, change nothing
    python scripts/toggle_mail.py --set fraud_scan=off catalyst=on
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes)
from mailer import MAIL_KEYS, SETTINGS_NAME, default_mail_settings


def _show(settings: dict) -> None:
    for i, (k, label) in enumerate(MAIL_KEYS.items(), 1):
        state = "ON " if settings.get(k, True) else "OFF"
        print(f"  {i}. [{state}] {k:14s} — {label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="Print current, change nothing.")
    ap.add_argument("--set", nargs="*", metavar="KEY=on|off",
                    help="Non-interactive, e.g. --set fraud_scan=off catalyst=on")
    args = ap.parse_args()

    drive = get_drive()
    repo_id = get_or_create_subfolder(drive, os.environ["GDRIVE_FOLDER_ID"],
                                      "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")
    fid = find_file(drive, index_id, SETTINGS_NAME)
    settings = default_mail_settings()
    if fid:
        try:
            data = json.loads(download_bytes(drive, fid).decode("utf-8"))
            settings.update({k: bool(v) for k, v in data.items() if k in settings})
        except Exception as e:
            print(f"  could not parse existing {SETTINGS_NAME} ({e}) — starting "
                  f"from all-ON defaults")

    if args.show:
        _show(settings)
        return

    changed = False
    if args.set:
        for kv in args.set:
            k, _, v = kv.partition("=")
            if k in settings and v.lower() in ("on", "off", "true", "false", "1", "0"):
                settings[k] = v.lower() in ("on", "true", "1")
                changed = True
            else:
                print(f"  ignored: {kv} (keys: {', '.join(MAIL_KEYS)}; values: on|off)")
    else:
        keys = list(MAIL_KEYS)
        while True:
            print()
            _show(settings)
            c = input("\n  Flip which number? (Enter=save & exit, q=quit without "
                      "saving): ").strip().lower()
            if c == "":
                break
            if c == "q":
                print("  No changes saved.")
                return
            if c.isdigit() and 1 <= int(c) <= len(keys):
                k = keys[int(c) - 1]
                settings[k] = not settings[k]
                changed = True
            else:
                print("  ?")

    if not changed:
        print("  Nothing changed.")
        return
    payload = json.dumps(
        {**settings, "updated_at": datetime.now().isoformat(timespec="seconds")},
        indent=2).encode("utf-8")
    upload_bytes(drive, index_id, SETTINGS_NAME, payload, "application/json",
                 existing_id=find_file(drive, index_id, SETTINGS_NAME))
    print("\n  Saved to Drive — applies from the next scheduled run.")
    _show(settings)


if __name__ == "__main__":
    main()
