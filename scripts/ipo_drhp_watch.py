r"""
ipo_drhp_watch.py — auto-summarise newly-filed IPO prospectuses and email them.

WHY NOT SEBI: SEBI's site (sebi.gov.in) refuses automated/datacenter connections
(ECONNREFUSED from CI and dev). So we use the Chittorgarh IPO calendar purely to
DISCOVER new IPO company names; the prospectus itself is still the authoritative
SEBI/exchange-filed DRHP/RHP, found + verified via the deep-dive machinery.

FLOW (weekdays, once daily — ipo_drhp_watch.yml):
  Chittorgarh IPO calendar -> new IPO names (vs drhp_watch_ledger.parquet)
    -> for each NEW name: reuse company_deep_report's discover -> verify (guardrail)
       -> summarise (forensic drhp_prompt, text-chunked, RHP-primary)
    -> email the summary + save sidecar to company_repo/_ipo_drhp/ on Drive
    -> record in ledger (retry names with no prospectus yet; never re-email done ones)

Run:  python scripts/ipo_drhp_watch.py            (real)
      python scripts/ipo_drhp_watch.py --dry-run  (list new IPOs, no Gemini/email)
"""
from __future__ import annotations
import os, sys, io, re, smtplib, argparse, datetime as dt
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import company_deep_report as cdr          # reuse all DRHP discovery/verify/summarise
from gemini_pool import BucketPool, load_keys

LEDGER       = "company_repo/_index/drhp_watch_ledger.parquet"
SUMMARY_DIR  = "company_repo/_ipo_drhp"
IPO_SOURCES  = (
    "https://www.chittorgarh.com/ipo/ipo_dashboard.asp",
    "https://www.chittorgarh.com/report/mainboard-ipo-list-in-india/82/",
    "https://www.chittorgarh.com/report/sme-ipo-list-in-india/83/",
)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
GMAIL_USER   = os.environ.get("GMAIL_USER", "")
GMAIL_PASS   = os.environ.get("GMAIL_APP_PASSWORD", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", GMAIL_USER)
DONE_STATES  = ("emailed", "summarised", "seeded")
MAX_PER_RUN  = int(os.environ.get("IPO_WATCH_MAX_PER_RUN", "6"))   # cap cost/run


def fetch_ipo_list() -> list[tuple[str, str]]:
    """[(ipo_id, company_name)] of current/upcoming IPOs from Chittorgarh."""
    out: dict[str, str] = {}
    for url in IPO_SOURCES:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
            soup = BeautifulSoup(r.text, "lxml")
        except Exception as e:
            print(f"  source failed {url[:50]}: {type(e).__name__}")
            continue
        for a in soup.find_all("a", href=True):
            m = re.search(r"/ipo/([a-z0-9-]+-ipo)/(\d+)", a.get("href", ""))
            name = a.get_text(" ", strip=True)
            if not (m and name):
                continue
            ipo_id = m.group(2)
            nm = re.sub(r"\s*IPO$", "", name, flags=re.I).strip()
            if ipo_id not in out and nm:
                out[ipo_id] = nm
    return list(out.items())


def email_summary(name: str, typ: str, summ: str) -> bool:
    if not (GMAIL_USER and GMAIL_PASS):
        print("  (no Gmail creds — skipping email)")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"New IPO {typ.upper()}: {name}"
    msg["From"] = GMAIL_USER
    msg["To"] = NOTIFY_EMAIL
    body = (f"Forensic prospectus summary — {name} ({typ.upper()}).\n"
            f"Auto-generated from the SEBI/exchange-filed prospectus.\n"
            f"{'-'*60}\n\n{summ}")
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.send_message(msg)
        print(f"  emailed: {name}")
        return True
    except Exception as e:
        print(f"  email failed for {name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="list new IPOs only; no discovery, Gemini, or email")
    ap.add_argument("--seed", action="store_true",
                    help="mark all current IPOs as seen (no summarise/email) — run "
                         "ONCE on first deploy so you only get NEW filings afterwards")
    ap.add_argument("--max", type=int, default=MAX_PER_RUN,
                    help=f"max IPOs to summarise this run (default {MAX_PER_RUN})")
    args = ap.parse_args()

    svc = cdr.drive_service()
    root = os.environ["GDRIVE_FOLDER_ID"]
    ledger = cdr._read_parquet(svc, LEDGER, root)
    done = (set(ledger[ledger["status"].isin(DONE_STATES)]["ipo_id"].astype(str))
            if not ledger.empty and "status" in ledger else set())

    ipos = fetch_ipo_list()
    new = [(i, n) for i, n in ipos if i not in done]
    # newest first (Chittorgarh ids increase with recency) so the cap favours fresh filings
    new.sort(key=lambda x: int(x[0]), reverse=True)
    print(f"IPO watch: {len(ipos)} listed, {len(new)} not-yet-summarised")
    for i, n in new:
        print(f"   - {n} ({i})")
    if not new:
        return

    if args.seed:
        rows = [dict(ipo_id=i, name=n, doc="", status="seeded",
                     seen_at=dt.datetime.now().isoformat()) for i, n in new]
        led = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True
                        ).drop_duplicates("ipo_id", keep="last")
        buf = io.BytesIO(); led.to_parquet(buf, index=False)
        cdr.drive_upload(svc, LEDGER, root, buf.getvalue(), "application/octet-stream")
        print(f"IPO watch: SEEDED {len(rows)} current IPOs — future runs email only NEW filings.")
        return

    if args.dry_run:
        return

    new = new[:args.max]      # bound cost; remaining picked up over subsequent days

    pool = BucketPool(load_keys(os.environ), cdr.DEEPDIVE_MODELS,
                      inter_call_s=cdr.INTER_CALL_SLEEP)
    prompt = open(os.path.join(cdr.SCRIPTS_DIR, "drhp_prompt.txt"), encoding="utf-8").read()

    rows, emailed = [], 0
    for ipo_id, name in new:
        status, typ = "no_prospectus", None
        chosen = None
        try:
            for url in cdr._discover_prospectus_urls(name):
                data = cdr._download_prospectus(url)
                if not data:
                    continue
                t = cdr._verify_prospectus(data, name)
                if t == "rhp":
                    chosen = ("rhp", data); break        # final doc — stop
                if t == "drhp" and chosen is None:
                    chosen = ("drhp", data)              # hold draft, seek RHP
            if chosen:
                typ, data = chosen
                summ = cdr._summarise_pdf_chunked(pool, prompt, data,
                                                  f"{typ.upper()} {name}", prefer_text=True)
                if summ:
                    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
                    try:
                        cdr.drive_upload(svc, f"{SUMMARY_DIR}/{slug}_{typ}.md", root,
                                         summ.encode("utf-8"), "text/markdown")
                    except Exception:
                        pass
                    status = "emailed" if email_summary(name, typ, summ) else "summarised"
                    if status == "emailed":
                        emailed += 1
        except Exception as e:
            status = f"error:{type(e).__name__}"
            print(f"  {name}: {status}")
        rows.append(dict(ipo_id=ipo_id, name=name, doc=typ or "",
                         status=status, seen_at=dt.datetime.now().isoformat()))

    led = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
    led = led.drop_duplicates("ipo_id", keep="last")    # latest status per IPO
    buf = io.BytesIO(); led.to_parquet(buf, index=False)
    cdr.drive_upload(svc, LEDGER, root, buf.getvalue(), "application/octet-stream")
    print(f"IPO watch: processed {len(rows)} new, emailed {emailed}")


if __name__ == "__main__":
    main()
