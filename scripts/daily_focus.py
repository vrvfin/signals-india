r"""
daily_focus.py  —  signals-india / Workflow A companion (OT8)

Cross-report triage. Reads the document summaries already produced by
daily_research_summary.py (from research_ledger.parquet, LOCAL) and makes ONE Gemini
call that acts as a research-desk head: surfaces strong growth, guidance/order book,
valuation calls, new information, red flags / fraud risk, and macro->sector impact.

Two sections: TODAY (docs processed today) + ROLLING (last N days, today excluded).

Output:
  • local  D:\EMA_Screener\Reports\research_synthesis\daily_focus_DD_MMMYYYY.md  (opened in browser)
  • Drive  company_repo/_daily/daily_focus_DD_MMMYYYY.md

Run:
  python scripts/daily_focus.py                 # today + rolling 7d, upload + open
  python scripts/daily_focus.py --days 14       # wider rolling window
  python scripts/daily_focus.py --date 02_Jun2026   # re-run for a past date (by processed_at)
  python scripts/daily_focus.py --no-upload --no-open

Uses the dedicated DAILY_GEMINI_KEY_1/2 pool (same as the summariser).
"""
from __future__ import annotations
import os, sys, io, re, argparse, datetime as dt, subprocess, urllib.parse
from pathlib import Path
import pandas as pd

# load project .env (same pattern as all pipeline scripts)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# reuse Drive + Gemini helpers from the summariser (no duplication)
from daily_research_summary import drive_service, drive_upload, build_daily_pool, DRIVE
from _md_utils import fix_markdown_for_obsidian               # Obsidian table fixer

INTAKE_DIR   = Path(os.getenv("RESEARCH_INTAKE_DIR", r"D:\EMA_Screener\research_intake"))
INDEX_PATH   = INTAKE_DIR / "_ledger" / "research_index.parquet"   # local copy, has summary_md
SCRIPTS_DIR  = Path(__file__).resolve().parent
PROMPT_FILE  = SCRIPTS_DIR / "daily_focus_prompt.txt"
# Obsidian vault folder so the focus note is indexed + opens in Obsidian
OUT_DIR      = Path(os.getenv("FOCUS_OUT_DIR", r"D:\EMA_Screener\Reports\signals-india\daily_focus"))


def open_in_obsidian(path: Path) -> None:
    uri = "obsidian://open?path=" + urllib.parse.quote(str(path).replace("\\", "/"), safe=":/")
    try:
        subprocess.run(["cmd", "/c", "start", "", uri], shell=False)
    except Exception as e:
        print(f"Could not open in Obsidian ({e}); open manually: {path}")

ROLL_DAYS_DEFAULT = 7
CHAR_BUDGET       = 600_000   # ~150k tokens; trim ROLLING first if over


def _proc_date(val) -> dt.date | None:
    try:
        return dt.datetime.fromisoformat(str(val)).date()
    except Exception:
        return None


def _doc_label(i: int, r: pd.Series) -> str:
    return (f"[doc_{i:02d} | {r.get('doc_date','NA')} | {r.get('source','NA')} | "
            f"{r.get('doc_type','NA')} | {r.get('file_name','NA')}]")


def _blocks(df: pd.DataFrame, start_i: int) -> tuple[str, int]:
    """Render summaries to labelled blocks; returns (text, next_index)."""
    parts, i = [], start_i
    for _, r in df.iterrows():
        summ = str(r.get("summary_md", "")).strip()
        if not summ:
            continue
        parts.append(f"--- {_doc_label(i, r)} ---\n{summ}")
        i += 1
    return "\n\n".join(parts), i


def _trim(today_txt: str, roll_txt: str) -> str:
    """Keep TODAY whole; trim ROLLING from the end if total exceeds budget."""
    if len(today_txt) + len(roll_txt) <= CHAR_BUDGET:
        return roll_txt
    avail = max(0, CHAR_BUDGET - len(today_txt))
    if avail == 0:
        return "(rolling context omitted — today's batch already fills the budget)"
    return roll_txt[:avail] + "\n\n(... rolling context truncated to fit context budget ...)"


def main():
    ap = argparse.ArgumentParser(description="Daily cross-report focus / triage note.")
    ap.add_argument("--days", type=int, default=ROLL_DAYS_DEFAULT,
                    help=f"Rolling window in days (default {ROLL_DAYS_DEFAULT})")
    ap.add_argument("--date", help="Target date as DD_MmmYYYY (default: today, by processed_at)")
    ap.add_argument("--no-upload", action="store_true", help="Do not upload to Drive")
    ap.add_argument("--no-open", action="store_true", help="Do not open the result in a browser")
    args = ap.parse_args()

    if not INDEX_PATH.exists():
        print(f"No local index at {INDEX_PATH}. Run daily_research_summary.py first.")
        return
    led = pd.read_parquet(INDEX_PATH)
    if "summary_md" not in led.columns:
        print("Local index has no summary_md column (older data). Re-process docs to populate.")
        return

    # only docs that actually have a summary
    led = led[led["summary_md"].astype(str).str.len().gt(0)].copy()
    if led.empty:
        print("No summarised documents in the index yet."); return

    led["pdate"] = led["processed_at"].map(_proc_date)
    target = (dt.datetime.strptime(args.date, "%d_%b%Y").date()
              if args.date else dt.date.today())
    roll_start = target - dt.timedelta(days=args.days)

    today_df = led[led.pdate == target]
    roll_df  = led[(led.pdate >= roll_start) & (led.pdate < target)]

    if today_df.empty and roll_df.empty:
        print(f"No documents for {target} or the {args.days} days before it."); return

    today_txt, nxt = _blocks(today_df, 1)
    roll_txt, _    = _blocks(roll_df, nxt)
    roll_txt = _trim(today_txt, roll_txt)

    n_today = today_txt.count("--- [doc_")
    n_roll  = roll_txt.count("--- [doc_")
    print(f"Daily Focus for {target}: {n_today} today + {n_roll} rolling ({args.days}d). Calling Gemini...")

    prompt = (PROMPT_FILE.read_text(encoding="utf-8")
              .replace("{{TODAY_DATE}}", target.strftime("%d %b %Y"))
              .replace("{{ROLL_DAYS}}", str(args.days))
              .replace("{{N_TODAY}}", str(n_today))
              .replace("{{N_ROLL}}", str(n_roll))
              .replace("{{TODAY_DOCS}}", today_txt or "(none today)")
              .replace("{{ROLL_DOCS}}", roll_txt or "(none)"))

    result = build_daily_pool().call_text(prompt)[0]

    stamp   = target.strftime("%d_%b%Y")
    header  = (f"# Daily Focus — {target:%d %b %Y}\n\n"
               f"*Generated {dt.datetime.now():%d %b %Y %H:%M} IST · "
               f"{n_today} today + {n_roll} rolling ({args.days}d)*\n\n---\n\n")
    body    = header + result

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUT_DIR / f"daily_focus_{stamp}.md"
    outpath.write_text(fix_markdown_for_obsidian(body), encoding="utf-8")
    print(f"Saved (Obsidian-fixed): {outpath}")

    if not args.no_upload:
        try:
            svc  = drive_service()
            root = os.environ["GDRIVE_FOLDER_ID"]
            drive_upload(svc, f"{DRIVE['daily_dir']}/daily_focus_{stamp}.md",
                         root, body.encode("utf-8"), "text/markdown")
            print(f"Uploaded to Drive: {DRIVE['daily_dir']}/daily_focus_{stamp}.md")
        except Exception as e:
            print(f"WARN: Drive upload skipped ({e})")

    if not args.no_open:
        open_in_obsidian(outpath)


if __name__ == "__main__":
    main()
