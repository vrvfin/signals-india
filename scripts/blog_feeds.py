r"""
blog_feeds.py — manage the curated blog/Substack feed list (add_blog.bat).

The live list is company_repo/_index/blog_feeds.csv on Drive (name,url,enabled).
social_sources._load_feeds() merges it over the built-ins, so additions reach
the PF digest, catalyst notes and deep dive with NO code change.

Every add is VALIDATED live: the URL must return HTTP 200 and parse as RSS
with at least one <item>. Substack shorthand: any publication's feed is
https://<subdomain>.substack.com/feed.

Usage:
    python scripts/blog_feeds.py --list
    python scripts/blog_feeds.py --add "Invest Karo India" https://investkaroindia.substack.com/feed
    python scripts/blog_feeds.py --add-substack "Invest Karo India" investkaroindia
    python scripts/blog_feeds.py --remove "Invest Karo India"
    python scripts/blog_feeds.py --test-all
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)

CSV_NAME = "blog_feeds.csv"
COLS = ["name", "url", "enabled", "added_at"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _index_id(drive):
    root = os.environ["GDRIVE_FOLDER_ID"]
    return get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")


def load_df(drive, index_id) -> pd.DataFrame:
    fid = find_file(drive, index_id, CSV_NAME)
    if not fid:
        return pd.DataFrame(columns=COLS)
    try:
        return pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return pd.DataFrame(columns=COLS)


def save_df(drive, index_id, df: pd.DataFrame) -> None:
    fid = find_file(drive, index_id, CSV_NAME)
    upload_bytes(drive, index_id, CSV_NAME,
                 df.to_csv(index=False).encode("utf-8"), "text/csv",
                 existing_id=fid)


def validate(url: str) -> tuple[bool, str]:
    """(ok, detail) — feed must be 200 and contain at least one RSS <item>."""
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        import xml.etree.ElementTree as ET
        n = sum(1 for _ in ET.fromstring(r.content).iter("item"))
        return (n > 0), f"{n} item(s)"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:60]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--add", nargs=2, metavar=("NAME", "URL"))
    ap.add_argument("--add-substack", nargs=2, metavar=("NAME", "SUBDOMAIN"))
    ap.add_argument("--remove", type=str, metavar="NAME")
    ap.add_argument("--test-all", action="store_true")
    args = ap.parse_args()

    drive = get_drive()
    idx = _index_id(drive)
    df = load_df(drive, idx)

    if args.add or args.add_substack:
        if args.add:
            name, url = args.add
        else:
            name, sub = args.add_substack
            url = f"https://{sub.strip().strip('/')}.substack.com/feed"
        ok, detail = validate(url)
        print(f"  validate {url} -> {'OK' if ok else 'FAILED'} ({detail})")
        if not ok:
            print("  NOT added — fix the URL/subdomain and retry.")
            sys.exit(1)
        df = df[df["name"].astype(str).str.lower() != name.lower()]
        df = pd.concat([df, pd.DataFrame([{
            "name": name, "url": url, "enabled": 1,
            "added_at": datetime.now().strftime("%Y-%m-%d")}])],
            ignore_index=True)
        save_df(drive, idx, df)
        print(f"  ADDED '{name}' — live in tonight's digests.")
    elif args.remove:
        before = len(df)
        df = df[df["name"].astype(str).str.lower() != args.remove.lower()]
        if len(df) < before:
            save_df(drive, idx, df)
            print(f"  REMOVED '{args.remove}'.")
        else:
            print(f"  '{args.remove}' not found.")
    elif args.test_all:
        for _, r in df.iterrows():
            ok, detail = validate(str(r["url"]))
            print(f"  {'OK ' if ok else 'BAD'} {str(r['name']):<28} {detail}")

    if args.list or not (args.add or args.add_substack or args.remove
                         or args.test_all):
        df = load_df(drive, idx)
        print(f"\n  blog_feeds.csv on Drive: {len(df)} feed(s)")
        for _, r in df.iterrows():
            on = str(r.get("enabled", "1")).lower() in ("1", "true", "yes")
            print(f"  {'[on ]' if on else '[off]'} {str(r['name']):<28} {r['url']}")
        print("  (+3 built-ins: Dr Vijay Malik, Safal Niveshak, AlphaIdeas)")


if __name__ == "__main__":
    main()
