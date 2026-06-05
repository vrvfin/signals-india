"""
fetch_deepdive.py — fetch the latest deep-dive report for a company from Drive.
Opens in Obsidian (primary) or browser HTML (fallback).

Usage:
    python scripts/fetch_deepdive.py "TCS"
    python scripts/fetch_deepdive.py "INE467B01029"
"""
from __future__ import annotations
import os, sys, io, re, tempfile, webbrowser, datetime as dt

import pandas as pd
from dotenv import load_dotenv

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(SCRIPTS_DIR), ".env"))

from daily_research_summary import drive_service, drive_download, _folder_id
from company_deep_report import resolve_isin, _read_csv, _read_parquet, DRIVE


OBSIDIAN_VAULT = os.path.join(
    os.environ.get("OBSIDIAN_VAULT", r"D:\EMA_Screener\Obsidian"),
    "signals-india", "deepdive")
LOCAL_DIR = os.path.join(
    os.environ.get("REPORTS_DIR", r"D:\EMA_Screener\Reports\signals-india"),
    "deepdive")


def _open_md(md_text: str, slug: str):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    md_path = os.path.join(LOCAL_DIR, f"{slug}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # Obsidian
    try:
        os.makedirs(OBSIDIAN_VAULT, exist_ok=True)
        obs_path = os.path.join(OBSIDIAN_VAULT, f"{slug}.md")
        with open(obs_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"Saved to Obsidian: {obs_path}")
        return
    except Exception:
        pass

    # HTML fallback
    try:
        import markdown as md_lib
        html_body = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        html_body = f"<pre>{md_text}</pre>"
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>body{{font-family:sans-serif;max-width:960px;margin:2em auto;"
            f"line-height:1.6}}table{{border-collapse:collapse;width:100%}}"
            f"td,th{{border:1px solid #ccc;padding:6px 10px}}"
            f"h1,h2,h3{{color:#1a1a2e}}</style></head>"
            f"<body>{html_body}</body></html>")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                     prefix=f"deepdive_{slug}_", mode="w", encoding="utf-8")
    tmp.write(html); tmp.close()
    webbrowser.open(f"file://{tmp.name}")
    print(f"Opened in browser: {tmp.name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_deepdive.py <name|symbol|ISIN>"); sys.exit(1)
    token = " ".join(sys.argv[1:]).strip()

    svc   = drive_service()
    root  = os.environ["GDRIVE_FOLDER_ID"]
    univ  = _read_csv(svc, DRIVE["universe"], root)
    isin, symbol, name, _ = resolve_isin(token, univ, interactive=True)

    if isin == token and symbol == token:
        print(f"Could not resolve '{token}' in universe."); sys.exit(1)

    print(f"Company: {name} ({symbol} / {isin})")

    # find the latest dated deepdive file in company_repo/<ISIN>/
    folder_path = f"{DRIVE['company_page']}/{isin}"
    idx = _read_parquet(svc, DRIVE["index"], root)
    report_path = None

    if not idx.empty and "isin" in idx.columns:
        row = idx[idx["isin"] == isin]
        if not row.empty:
            report_path = row.iloc[-1].get("report_path")

    if not report_path:
        # fallback: list files in company folder and pick latest deepdive
        try:
            from googleapiclient.discovery import build as _build  # noqa
            # use drive_service folder listing via _folder_id helper
            folder_id = _folder_id(svc, root, folder_path, create=False)
            if folder_id:
                resp = svc.files().list(
                    q=f"'{folder_id}' in parents and name contains 'company_deepdive'",
                    fields="files(id,name)", orderBy="name desc", pageSize=5
                ).execute()
                files = resp.get("files", [])
                if files:
                    report_path = f"{folder_path}/{files[0]['name']}"
        except Exception as e:
            print(f"Warning: folder listing failed ({e})")

    if not report_path:
        print(f"No deep-dive report found for {name}. Run run_deepdive.bat to generate one.")
        sys.exit(1)

    print(f"Fetching: {report_path}")
    raw = drive_download(svc, report_path, root)
    if not raw:
        print("File found in index but could not download."); sys.exit(1)

    md_text = raw.decode("utf-8")
    slug = f"{symbol.lower()}_{dt.date.today().strftime('%d%b%y').lower()}"
    _open_md(md_text, slug)


if __name__ == "__main__":
    main()
