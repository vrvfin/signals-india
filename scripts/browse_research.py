"""
browse_research.py — Open past daily_focus + research digest files by date range.

Same flexible date parsing as fetch_concalls_range.py:
  DDmmmYYYY  (05Jun2026)
  DD-mmm-YYYY (05-Jun-2026)
  YYYY-MM-DD (2026-06-05)

Usage:
  python scripts/browse_research.py --from 03Jun2026
  python scripts/browse_research.py --from 03Jun2026 --to 05Jun2026
"""
from __future__ import annotations
import argparse, subprocess, urllib.parse, re, sys
from pathlib import Path
from datetime import datetime

FOCUS_DIR = Path(r"D:\EMA_Screener\Reports\signals-india\daily_focus")
DIGEST_DIR = Path(r"D:\EMA_Screener\Reports\signals-india")


def parse_date_flexible(date_str: str) -> str | None:
    """Parse flexible date formats to DD_MmmYYYY.

    Accepted:
      DDmmmYYYY    (05Jun2026)
      DD-mmm-YYYY  (05-Jun-2026)
      YYYY-MM-DD   (2026-06-05)

    Returns: DD_MmmYYYY or None if unparseable.
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    # Try YYYY-MM-DD
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d_%b%Y")
    except ValueError:
        pass

    # Try DD-mmm-YYYY (with any case)
    try:
        dt = datetime.strptime(date_str, "%d-%b-%Y")
        return dt.strftime("%d_%b%Y")
    except ValueError:
        pass

    # Try DDmmmYYYY (no separators, any case)
    try:
        dt = datetime.strptime(date_str, "%d%b%Y")
        return dt.strftime("%d_%b%Y")
    except ValueError:
        pass

    return None


def find_focus_files() -> list[tuple[str, Path]]:
    """Return list of (date_str_DD_MmmYYYY, path) sorted newest first."""
    files = []
    for f in FOCUS_DIR.glob("daily_focus_*.md"):
        match = re.search(r"daily_focus_([\d]{2}_[A-Za-z]{3}\d{4})\.md", f.name)
        if match:
            date_str = match.group(1)
            files.append((date_str, f))
    return sorted(files, key=lambda x: x[0], reverse=True)


def date_le(d1: str, d2: str) -> bool:
    """Compare DD_MmmYYYY dates. Return True if d1 <= d2."""
    try:
        dt1 = datetime.strptime(d1, "%d_%b%Y")
        dt2 = datetime.strptime(d2, "%d_%b%Y")
        return dt1 <= dt2
    except ValueError:
        return False


def find_digest_for_date(date_str: str) -> Path | None:
    """Try to find research_DD_MMMYYYY.md for the given date."""
    parts = date_str.split("_")
    if len(parts) == 2:
        # Search in multiple locations
        for p in [DIGEST_DIR / "_daily", DIGEST_DIR]:
            if not p.exists():
                continue
            for f in p.glob("research_*.md"):
                # Match DD and MMM from the filename
                if date_str.replace("_", "").lower() in f.name.replace("_", "").lower():
                    return f
    return None


def open_in_obsidian(path: Path) -> None:
    uri = "obsidian://open?path=" + urllib.parse.quote(str(path).replace("\\", "/"), safe=":/")
    try:
        subprocess.run(["cmd", "/c", "start", "", uri], shell=False)
    except Exception as e:
        print(f"  warn: could not open {path.name} ({e})")


def main():
    ap = argparse.ArgumentParser(
        description="Open past daily research focus + digest files by date range.")
    ap.add_argument("--from", dest="from_date",
                    help="Start date (DDmmmYYYY, DD-mmm-YYYY, or YYYY-MM-DD)")
    ap.add_argument("--to", dest="to_date",
                    help="End date (optional; defaults to today)")
    args = ap.parse_args()

    files = find_focus_files()
    if not files:
        print("ERROR: No daily_focus files found.")
        sys.exit(1)

    # Parse from_date (required)
    if not args.from_date:
        print("ERROR: --from date is required.")
        sys.exit(1)

    from_parsed = parse_date_flexible(args.from_date)
    if not from_parsed:
        print(f"ERROR: Could not parse from_date '{args.from_date}'")
        print(f"       Try: DDmmmYYYY (05Jun2026), DD-mmm-YYYY, or YYYY-MM-DD")
        sys.exit(1)

    # Parse to_date (optional, defaults to today)
    if args.to_date:
        to_parsed = parse_date_flexible(args.to_date)
        if not to_parsed:
            print(f"ERROR: Could not parse to_date '{args.to_date}'")
            sys.exit(1)
    else:
        to_parsed = datetime.now().strftime("%d_%b%Y")

    # Filter files in range
    in_range = [f for f in files if date_le(from_parsed, f[0]) and date_le(f[0], to_parsed)]

    if not in_range:
        print(f"No files found in range {from_parsed} to {to_parsed}")
        print(f"\nAvailable files:")
        for date_str, _ in files[:5]:
            print(f"  {date_str}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
        sys.exit(1)

    print(f"\n  Opening {len(in_range)} file(s):")
    for date_str, focus_path in in_range:
        print(f"  {date_str}...", end=" ", flush=True)
        open_in_obsidian(focus_path)

        digest = find_digest_for_date(date_str)
        if digest:
            open_in_obsidian(digest)
            print("(focus + digest)")
        else:
            print("(focus only)")

    print(f"\n  All files opened in Obsidian.\n")


if __name__ == "__main__":
    main()
