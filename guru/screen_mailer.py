r"""
SCREEN MAILER — screen_mailer.py  (Project Guru)

Sends the daily-screen results by email, and — the useful part — detects stocks
that have NEWLY appeared since the last run ("first come to limelight").

Reuses the repo's existing mail plumbing (scripts/mailer.py: GMAIL_USER /
GMAIL_APP_PASSWORD / NOTIFY_EMAIL). Nothing new to configure if the other
mails already work.

Modes:
  --mode new      send ONLY if new stocks appeared since the last snapshot
                  (this is the daily default — silent on a quiet day)
  --mode weekly   full digest: top conviction + coverage + new entrants
  --mode force    send the digest regardless (on demand)
  --dry-run       print what would be sent, send nothing

Snapshot lives at guru/backtest/screen_snapshot.parquet, so "new" means new
versus the previous run, however long ago that was.

Usage:
    python guru/screen_mailer.py --mode new
    python guru/screen_mailer.py --mode weekly
    python guru/screen_mailer.py --mode force --dry-run
"""
from __future__ import annotations
import argparse, glob, os, sys
from datetime import datetime
import pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
BT = os.path.join(GURU, "backtest")
SNAP = os.path.join(BT, "screen_snapshot.parquet")
SCRIPTS = os.path.join(os.path.dirname(GURU), "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(GURU).parent / ".env")

TOP_N = 25
NEW_MAX = 40


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def latest_screen() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    fs = sorted(glob.glob(os.path.join(GURU, "DAILY_SCREEN_*.xlsx")))
    if not fs:
        raise SystemExit("no DAILY_SCREEN_*.xlsx found — run daily_screen.py first")
    p = fs[-1]
    conv = pd.read_excel(p, "Conviction")
    try:
        cov = pd.read_excel(p, "Coverage")
    except Exception:
        cov = pd.DataFrame()
    try:
        stat = pd.read_excel(p, "Status")
    except Exception:
        stat = pd.DataFrame()
    return conv, cov, stat, os.path.basename(p)


def tbl(df: pd.DataFrame, cols: list[str], hdr: dict) -> str:
    if df.empty:
        return "<p style='color:#666'>(none)</p>"
    th = "".join(f"<th style='text-align:left;padding:6px 10px;border-bottom:2px solid #ddd;"
                 f"font-size:12px;color:#444'>{hdr.get(c,c)}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:,.1f}"
            tds.append(f"<td style='padding:6px 10px;border-bottom:1px solid #eee;"
                       f"font-size:13px'>{v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f"<table style='border-collapse:collapse;width:100%;font-family:"
            f"-apple-system,Segoe UI,Arial'><tr>{th}</tr>{''.join(rows)}</table>")


def build_html(conv, cov, stat, new_df, mode, fname) -> str:
    hdr = {"name": "Company", "conviction_score": "Score", "n_rules": "Rules",
           "market_cap_cr": "MCap (cr)", "turnover_20d_cr": "Turnover (cr/d)",
           "ret_12m_pct": "12M %", "rule_names": "Top rules",
           "rarest_rule_hits": "Rarest rule (n)"}
    cols = ["name", "conviction_score", "n_rules", "market_cap_cr",
            "turnover_20d_cr", "ret_12m_pct"]
    asof = ""
    if len(stat):
        d = dict(zip(stat.iloc[:, 0], stat.iloc[:, 1]))
        asof = (f"Price data: {d.get('price data as of','?')} &nbsp;|&nbsp; "
                f"Quarter: {d.get('latest quarter seen','?')} &nbsp;|&nbsp; "
                f"Stocks screened: {d.get('stocks with metrics','?')}")
    parts = [f"""<div style="font-family:-apple-system,Segoe UI,Arial;max-width:900px">
      <h2 style="margin-bottom:2px">Project Guru — stock screen</h2>
      <div style="color:#666;font-size:12px;margin-bottom:16px">{asof}<br>{fname}</div>"""]
    if len(new_df):
        parts.append(f"<h3 style='color:#0a7d33'>NEW since last run "
                     f"({len(new_df)})</h3>{tbl(new_df.head(NEW_MAX), cols, hdr)}")
    if mode in ("weekly", "force"):
        parts.append(f"<h3>Top {TOP_N} by conviction</h3>"
                     f"{tbl(conv.head(TOP_N), cols, hdr)}")
        if len(cov):
            c2 = [c for c in ["name", "rule_name", "turnover_20d_cr", "market_cap_cr"]
                  if c in cov.columns]
            parts.append(f"<h3>Coverage — {cov['symbol'].nunique()} unique stocks "
                         f"across rules</h3>{tbl(cov.head(60), c2, hdr)}")
    parts.append("""<p style="color:#888;font-size:11px;margin-top:20px">
      Conviction score = sum of (rule rarity x that rule's validated out-of-sample
      return) for every rule the stock currently satisfies. Filters: market cap
      &ge;100cr, traded value &ge;1cr/day. Backtest-derived, excludes costs and
      slippage — research output, not investment advice.</p></div>""")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["new", "weekly", "force"], default="new")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conv, cov, stat, fname = latest_screen()
    log(f"loaded {fname}: {len(conv):,} stocks")

    prev = set()
    if os.path.exists(SNAP):
        try:
            prev = set(pd.read_parquet(SNAP)["symbol"])
        except Exception:
            pass
    cur = set(conv["symbol"]) if "symbol" in conv else set()
    new_syms = cur - prev if prev else set()
    new_df = conv[conv["symbol"].isin(new_syms)] if new_syms else conv.head(0)
    log(f"previous snapshot: {len(prev):,} | now: {len(cur):,} | NEW: {len(new_df):,}")

    if args.mode == "new" and new_df.empty:
        log("no new stocks — nothing to send (quiet day)")
        if not args.dry_run:
            pd.DataFrame({"symbol": sorted(cur)}).to_parquet(SNAP, index=False)
        return

    html = build_html(conv, cov, stat, new_df, args.mode, fname)
    subj = {"new": f"Guru screen — {len(new_df)} NEW stock(s)",
            "weekly": "Guru screen — weekly digest",
            "force": "Guru screen — on-demand"}[args.mode]
    subj += f" ({datetime.now().date()})"

    if args.dry_run:
        out = os.path.join(GURU, "screen_mail_preview.html")
        open(out, "w", encoding="utf-8").write(html)
        log(f"DRY RUN — subject: {subj}")
        log(f"preview written: {out}")
        if len(new_df):
            print(new_df[["name", "conviction_score", "n_rules"]].head(15).to_string(index=False))
        return

    from mailer import send_email
    ok = send_email(subj, html)
    log(f"email sent: {ok}" if ok else "email FAILED — check GMAIL_USER / "
        "GMAIL_APP_PASSWORD / NOTIFY_EMAIL in .env")
    pd.DataFrame({"symbol": sorted(cur)}).to_parquet(SNAP, index=False)
    log(f"snapshot updated ({len(cur):,} symbols)")


if __name__ == "__main__":
    main()
