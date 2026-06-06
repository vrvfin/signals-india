r"""
fetch_latest_research.py  —  download the latest daily research digest from Drive
and open it locally (rendered like the concall digests).

Usage:  python scripts/fetch_latest_research.py
Saves to:  D:\EMA_Screener\claude\signals-india\research\
"""
import os, io, re, sys, subprocess, urllib.parse
from pathlib import Path

from daily_research_summary import drive_service, _folder_id  # reuse Drive helpers
from _md_utils import fix_markdown_for_obsidian               # Obsidian table fixer

# Obsidian vault folder (same vault as concall/intel tools) so the file is indexed + opens in Obsidian
OUT_DIR    = Path(os.getenv("RESEARCH_OUT_DIR",
                            r"D:\EMA_Screener\Reports\signals-india\research"))
DAILY_DIR  = "company_repo/_daily"


def open_in_obsidian(path: Path) -> None:
    uri = "obsidian://open?path=" + urllib.parse.quote(str(path).replace("\\", "/"), safe=":/")
    try:
        subprocess.run(["cmd", "/c", "start", "", uri], shell=False)
    except Exception as e:
        print(f"Could not open in Obsidian ({e}); open manually: {path}")


def main():
    svc = drive_service()
    root = os.environ["GDRIVE_FOLDER_ID"]
    fid = _folder_id(svc, DAILY_DIR, root)
    if not fid:
        sys.exit("No _daily folder on Drive yet — run daily_research_summary.py first.")
    q = (f"'{fid}' in parents and name contains 'research_' and "
         "name contains '.md' and trashed=false")
    files = svc.files().list(q=q, fields="files(id,name,modifiedTime)",
                             orderBy="modifiedTime desc", pageSize=1).execute().get("files", [])
    if not files:
        sys.exit("No research_*.md digests found on Drive.")

    f = files[0]
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO(); dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=f["id"]))
    done = False
    while not done:
        _, done = dl.next_chunk()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f["name"]
    fixed = fix_markdown_for_obsidian(buf.getvalue().decode("utf-8", errors="replace"))
    dest.write_text(fixed, encoding="utf-8")
    print(f"Saved (Obsidian-fixed): {dest}")
    open_in_obsidian(dest)


if __name__ == "__main__":
    main()
