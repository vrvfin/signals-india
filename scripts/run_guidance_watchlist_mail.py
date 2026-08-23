r"""
run_guidance_watchlist_mail.py — tell me when the >50% guidance watchlist CHANGES.

Silent by design. Sends only when a company ENTERS or LEAVES
company_repo/_index/guidance_watchlist.parquet since the last mail, so a quiet
off-season week produces no mail at all and a mail always means something moved.

    python scripts/run_guidance_watchlist_mail.py --dry-run   # preview, no send
    python scripts/run_guidance_watchlist_mail.py             # send if changed
    python scripts/run_guidance_watchlist_mail.py --force     # send even if not

Runs in t4_nightly.yml straight after the table build. Toggle key
'guidance_watchlist' in mail_settings.json (app sidebar / toggle_mail.bat).

WHAT COUNTS AS A CHANGE
-----------------------
The watchlist is keyed (isin, quarter). A ledger of the keys last mailed lives
beside it, so:
  ENTERED  a key present now that was not in the ledger — a company newly guiding
           above the bar, or the same company qualifying again in a NEW quarter
  LEFT     a key in the ledger that is gone now — only happens on a --rebuild
           (rules changed) or a purge, since the builder never deletes rows
Score moves on a key that is already known are deliberately NOT a trigger: a
company nudging 52% -> 55% is not news, and mailing it would train you to ignore
the mail. The current CAGR is still shown for every name in the "still on the
list" block.

NO NEW SCORING HAPPENS HERE. This script only reads what
build_guidance_watchlist.py published, so the mail can never disagree with the
page.
"""
from __future__ import annotations

import argparse
import html as html_mod
import os
import sys
from datetime import datetime, timezone

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import build_guidance_watchlist as BW
import quarterly_table as QT
from _extractor_base import (get_drive, get_or_create_subfolder, load_parquet,
                             log, save_parquet)
from mailer import load_mail_settings, send_email

MAIL_KEY = "guidance_watchlist"
LEDGER_NAME = "guidance_watchlist_mailed.parquet"
LEDGER_COLS = ["isin", "quarter", "nse_symbol", "cagr_pct", "mailed_at"]


def _esc(v) -> str:
    return html_mod.escape(str(v if v is not None else ""))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def diff_against_ledger(wl: pd.DataFrame, ledger: pd.DataFrame):
    """(entered_df, left_df, still_df) keyed on (isin, quarter)."""
    if wl is None:
        wl = pd.DataFrame(columns=BW.GW_COLS)
    now_keys = set(zip(wl["isin"].astype(str), wl["quarter"].astype(str)))
    old_keys = set()
    if ledger is not None and not ledger.empty:
        old_keys = set(zip(ledger["isin"].astype(str),
                           ledger["quarter"].astype(str)))
    ent = wl[[k not in old_keys for k in
              zip(wl["isin"].astype(str), wl["quarter"].astype(str))]]
    still = wl[[k in old_keys for k in
                zip(wl["isin"].astype(str), wl["quarter"].astype(str))]]
    left = pd.DataFrame(columns=LEDGER_COLS)
    if ledger is not None and not ledger.empty:
        left = ledger[[k not in now_keys for k in
                       zip(ledger["isin"].astype(str),
                           ledger["quarter"].astype(str))]]
    return ent, left, still


def _row_html(r) -> str:
    cag = pd.to_numeric(r.get("cagr_pct"), errors="coerce")
    nst = int(r.get("n_rows_over_min") or 1)
    base = r.get("base_ttm_cr")
    bits = [f'{r.get("score_metric") or ""}', f'{r.get("horizon_fy") or ""}']
    if pd.notna(base):
        bits.append(f'base ₹{float(base):,.0f}cr')
    if nst > 1:
        bits.append(f'<b style="color:#1a7a3a">{nst} statements agree</b>')
    if r.get("base_suspect"):
        bits.append('<b style="color:#c0392b">⚠ tiny base</b>')
    return (
        f'<tr>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eef1f5;white-space:nowrap">'
        f'<b style="font-size:14px;color:#1a3d6e">{_esc(r.get("nse_symbol") or r.get("symbol"))}</b>'
        f'<div style="font-size:11px;color:#777">{_esc(str(r.get("nse_name") or "")[:38])}</div></td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eef1f5;text-align:right;'
        f'font-size:17px;font-weight:800;color:#0d2f5c;white-space:nowrap">'
        f'{"" if pd.isna(cag) else f"{cag:,.0f}%"}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eef1f5;font-size:11.5px;color:#555">'
        f'{" · ".join(bits)}'
        f'<div style="color:#1565c0;font-style:italic;margin-top:3px">'
        f'"{_esc(str(r.get("evidence_stmt") or "")[:190])}"</div></td>'
        f'</tr>')


def build_html(ent: pd.DataFrame, left: pd.DataFrame, still: pd.DataFrame,
               quarter: str) -> str:
    p = [f'<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;'
         f'color:#222;max-width:860px">'
         f'<h2 style="margin:0 0 2px;font-size:18px">🎯 Guidance watchlist — {_esc(quarter)}</h2>'
         f'<div style="font-size:12px;color:#777;margin-bottom:14px">'
         f'revenue or PAT guided above 50% a year, each number checked against the '
         f'concall transcript · {len(ent) + len(still)} names on the list</div>']

    if len(ent):
        p.append('<h3 style="font-size:15px;color:#1a7a3a;margin:16px 0 4px">'
                 f'🆕 Entered ({len(ent)})</h3>'
                 '<table style="border-collapse:collapse;width:100%">')
        for _, r in ent.sort_values("cagr_pct", ascending=False).iterrows():
            p.append(_row_html(r))
        p.append("</table>")

    if len(left):
        p.append('<h3 style="font-size:15px;color:#c0392b;margin:18px 0 4px">'
                 f'↩ No longer on the list ({len(left)})</h3>'
                 '<div style="font-size:12.5px;color:#555">')
        p.append(" · ".join(
            f'<b>{_esc(r.get("nse_symbol"))}</b> ({_esc(r.get("quarter"))})'
            for _, r in left.iterrows()))
        p.append('<div style="font-size:11px;color:#999;margin-top:4px">'
                 'A name leaves when the cleaning rules change or the quarter '
                 'rolls — not because the company withdrew guidance.</div></div>')

    if len(still):
        p.append('<h3 style="font-size:15px;color:#555;margin:18px 0 4px">'
                 f'Still on the list ({len(still)})</h3>'
                 '<div style="font-size:12.5px;color:#555;line-height:1.8">')
        p.append(" · ".join(
            f'<b>{_esc(r.get("nse_symbol"))}</b> '
            f'{pd.to_numeric(r.get("cagr_pct"), errors="coerce"):,.0f}%'
            for _, r in still.sort_values("cagr_pct", ascending=False).iterrows()))
        p.append("</div>")

    p.append('<div style="font-size:11px;color:#999;margin-top:20px;'
             'border-top:1px solid #eee;padding-top:8px">'
             'Signals only — management said this, it is not a forecast that they '
             'will deliver it. Run <code>guidance_watchlist.bat</code> for the '
             'charts.</div></div>')
    return "".join(p)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="write a local preview; no mail, no ledger write")
    ap.add_argument("--force", action="store_true",
                    help="send even when nothing entered or left")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    drive = get_drive()
    idx = get_or_create_subfolder(
        drive, get_or_create_subfolder(
            drive, os.environ["GDRIVE_FOLDER_ID"], "company_repo"), "_index")

    wl = load_parquet(drive, idx, BW.GW_NAME, BW.GW_COLS)
    if wl.empty:
        log("guidance_watchlist.parquet is empty — run build_guidance_watchlist.py.")
        return 0
    wl = wl.copy()
    wl["_qo"] = wl["quarter"].map(QT.q_order)
    quarter = str(wl.sort_values("_qo")["quarter"].iloc[-1])
    wl = wl[wl["quarter"].astype(str) == quarter].sort_values(
        ["date_added", "cagr_pct"], ascending=[False, False])

    ledger = load_parquet(drive, idx, LEDGER_NAME, LEDGER_COLS)
    ent, left, still = diff_against_ledger(wl, ledger)
    log(f"{quarter}: {len(wl)} on the list · entered {len(ent)} · left {len(left)} "
        f"· unchanged {len(still)}")

    if not len(ent) and not len(left) and not args.force:
        log("nothing changed — no mail (this is the normal quiet path).")
        return 0

    html = build_html(ent, left, still, quarter)
    bits = []
    if len(ent):
        bits.append(f"{len(ent)} new")
    if len(left):
        bits.append(f"{len(left)} off")
    subject = (f"🎯 Guidance watchlist {quarter} — {', '.join(bits) or 'no change'}"
               f" · {len(wl)} names")

    if args.dry_run:
        path = os.path.join(args.out_dir, "guidance_watchlist_mail_preview.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"DRY RUN — preview {path} ({len(html.encode()):,} B); no mail, "
            f"no ledger write.")
        return 0

    if not load_mail_settings(drive, idx).get(MAIL_KEY, True):
        log(f"mail toggle '{MAIL_KEY}' is OFF — not sending.")
        return 0

    ok = send_email(subject, html)
    if not ok:
        log("mail NOT sent — ledger left untouched so the next run retries.")
        return 0

    # Stamp the ledger ONLY after a confirmed send, so a failed mail does not
    # mark today's arrivals as already announced.
    now = _now()
    fresh = pd.DataFrame({
        "isin": wl["isin"].astype(str), "quarter": wl["quarter"].astype(str),
        "nse_symbol": wl["nse_symbol"], "cagr_pct": wl["cagr_pct"],
        "mailed_at": now})
    save_parquet(drive, idx, LEDGER_NAME, fresh[LEDGER_COLS])
    log(f"sent · ledger now {len(fresh)} keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
