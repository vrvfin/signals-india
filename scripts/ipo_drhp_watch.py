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
import os, sys, io, re, argparse, datetime as dt
from pathlib import Path

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
import drhp_seeds as ds                     # backfill -> DRHP inbox (Rule 7 hand-off)
from gemini_pool import BucketPool, load_keys
from mailer import send_email, esc         # shared Gmail sender (plain+html+attachments)

LEDGER       = "company_repo/_index/drhp_watch_ledger.parquet"
SUMMARY_DIR  = "company_repo/_ipo_drhp"
IPO_SOURCES  = (
    "https://www.chittorgarh.com/ipo/ipo_dashboard.asp",
    "https://www.chittorgarh.com/report/mainboard-ipo-list-in-india/82/",
    "https://www.chittorgarh.com/report/sme-ipo-list-in-india/83/",
)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
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


# ---- mail rendering (user 2026-07-09: short snapshot body + full summary as PDF;
# ---- raw markdown in a plain-text body was unreadable) -------------------------
_TOKEN_COLORS = {"[WARNING]": "#c0392b", "[COMFORT]": "#1e8449", "[INFO]": "#777777"}


def _md_inline(s: str, cap: int = 1200) -> str:
    """Escape + **bold** + colorized [WARNING]/[COMFORT]/[INFO] tokens."""
    h = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(s, cap))
    for tok, col in _TOKEN_COLORS.items():
        h = h.replace(tok, f"<span style='color:{col};font-weight:bold'>{tok}</span>")
    return h


def _md_to_html(md_text: str) -> str:
    """DRHP markdown -> simple HTML: headings, bullets, hr, pipe-tables, bold,
    colored risk tokens. Feeds the PyMuPDF Story below (same subset it supports)."""
    out: list[str] = []
    table: list[list[str]] = []

    def flush_table():
        if not table:
            return
        rows = []
        for i, cells in enumerate(table):
            tag = "th" if i == 0 else "td"
            rows.append("<tr>" + "".join(
                f"<{tag} style='border:1px solid #999;padding:2px 6px;"
                f"text-align:left'>{_md_inline(c, 300)}</{tag}>" for c in cells) + "</tr>")
        out.append("<table style='border-collapse:collapse;margin:6px 0'>"
                   + "".join(rows) + "</table>")
        table.clear()

    for ln in md_text.splitlines():
        s = ln.rstrip()
        st = s.strip()
        if st.startswith("|") and st.endswith("|") and st.count("|") >= 2:
            cells = [c.strip() for c in st.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue                                   # |---|---| separator row
            table.append(cells)
            continue
        flush_table()
        if s.startswith("### "):
            out.append(f"<h3>{_md_inline(s[4:], 300)}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{_md_inline(s[3:], 300)}</h2>")
        elif s.startswith("# "):
            out.append(f"<h1>{_md_inline(s[2:], 300)}</h1>")
        elif st.startswith(("- ", "* ")):
            out.append(f"<p>&bull; {_md_inline(st[2:], 600)}</p>")
        elif st == "---":
            out.append("<hr>")
        elif st:
            out.append(f"<p>{_md_inline(s)}</p>")
    flush_table()
    return "\n".join(out)


def _md_to_pdf(md_text: str) -> bytes | None:
    """Full summary -> A4 PDF via PyMuPDF Story (mirrors daily_ar_summary's
    fail-soft helper; this local variant adds tables + risk-token colors).
    None on failure — caller attaches the .md instead."""
    try:
        import fitz
        html = ("<body style='font-family:sans-serif;font-size:9pt;line-height:1.35'>"
                + _md_to_html(md_text) + "</body>")
        story = fitz.Story(html=html)
        buf = io.BytesIO()
        writer = fitz.DocumentWriter(buf)
        page = fitz.paper_rect("a4")
        where = page + (36, 36, -36, -36)
        more = 1
        while more:
            dev = writer.begin_page(page)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
        writer.close()
        return buf.getvalue()
    except Exception as e:
        print(f"  PDF render failed ({str(e)[:80]}) — attaching .md instead")
        return None


def _snapshot_html(name: str, typ: str, summ: str) -> str:
    """~10-line mail body: issue-snapshot lines + top red flags + token counts.
    Full detail lives in the attached PDF."""
    lines = summ.splitlines()

    def section(pattern: str, cap: int) -> list[str]:
        got: list[str] = []
        in_sec = False
        for ln in lines:
            st = ln.strip()
            if re.search(pattern, st, re.I) and (st.startswith("#") or re.match(r"\d+\.", st)):
                in_sec = True
                continue
            if in_sec:
                if st.startswith("#") or re.match(r"\d+\.\s+[A-Z]", st):
                    break                                   # next section heading
                if st and st != "---":
                    got.append(st.lstrip("-* ").strip())
            if len(got) >= cap:
                break
        return got

    snap = section(r"issue\s+snapshot", 6)
    flags = section(r"red.?flag\s+summary", 5)
    if not flags:                                           # fallback: gravest anywhere
        flags = [l.strip().lstrip("-* ") for l in lines if "[WARNING]" in l][:5]
    n_warn = summ.count("[WARNING]")
    n_comf = summ.count("[COMFORT]")

    def ul(items: list[str]) -> str:
        return "".join(f"<p style='margin:2px 0'>&bull; {_md_inline(i, 400)}</p>"
                       for i in items) or "<p style='color:#999'>DATA_MISSING</p>"

    return (f"<p><b>Forensic prospectus summary — {esc(name)} ({typ.upper()})</b></p>"
            f"<p style='font-size:12px;color:#666'>Auto-generated from the "
            f"SEBI/exchange-filed prospectus. <b>Full summary in the attached PDF</b>; "
            f"markdown copy on Drive: {SUMMARY_DIR}/</p>"
            f"<p><span style='color:#c0392b'><b>{n_warn} WARNING</b></span> · "
            f"<span style='color:#1e8449'>{n_comf} COMFORT</span> tokens in full summary</p>"
            f"<h3 style='margin:8px 0 2px'>Issue snapshot</h3>{ul(snap)}"
            f"<h3 style='margin:8px 0 2px'>Top red flags</h3>{ul(flags)}")


def email_summary(name: str, typ: str, summ: str) -> bool:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    pdf = _md_to_pdf(summ)
    attachments = ([(f"{slug}_{typ}.pdf", pdf, "pdf")] if pdf
                   else [(f"{slug}_{typ}.md", summ.encode("utf-8"), "octet-stream")])
    ok = send_email(f"New IPO {typ.upper()}: {name}",
                    _snapshot_html(name, typ, summ),
                    f"Forensic prospectus summary — {name} ({typ.upper()}). "
                    f"Full summary in the attached file.",
                    attachments=attachments)
    if ok:
        print(f"  emailed: {name}")
    return ok


def _seed_inbox_path() -> str:
    return f"company_repo/_index/{ds.SEED_FILE}"


def _load_seed_inbox(svc, root) -> pd.DataFrame:
    """The backfill DRHP inbox (drhp_seeds.parquet). Read via the cdr Drive client we
    already use for the ledger; run_backfill writes it via _extractor_base — same file."""
    try:
        return cdr._read_parquet(svc, _seed_inbox_path(), root)
    except Exception as e:
        print(f"  (could not read DRHP seed inbox: {type(e).__name__})")
        return pd.DataFrame(columns=ds.SEED_COLS)


def _mark_seeds_consumed(svc, root, consumed_ids: list[str]) -> None:
    """Flip processed seeds to status='consumed' so they're not re-opened. Re-reads
    before writing to avoid clobbering a concurrent backfill seed write."""
    if not consumed_ids:
        return
    df = _load_seed_inbox(svc, root)
    if df.empty or "seed_id" not in df.columns:
        return
    mask = df["seed_id"].astype(str).isin(set(consumed_ids))
    if not mask.any():
        return
    df.loc[mask, "status"] = "consumed"
    buf = io.BytesIO(); df.to_parquet(buf, index=False)
    cdr.drive_upload(svc, _seed_inbox_path(), root, buf.getvalue(),
                     "application/octet-stream")


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

    # DRHP seed inbox: prospectus links the company-page backfill surfaced under a
    # company's AR subsection (Rule 7 hand-off). Same discovery/summarise path; the
    # synthetic seed_id keys the ledger so a seed already summarised is skipped via `done`.
    seeds_df = _load_seed_inbox(svc, root)
    seed_items: list[tuple[str, str]] = []
    if not seeds_df.empty and "status" in seeds_df.columns:
        sn = seeds_df[seeds_df["status"].astype(str) == "new"]
        seed_items = [(str(r["seed_id"]), str(r["name"])) for _, r in sn.iterrows()
                      if str(r["seed_id"]) not in done and str(r.get("name") or "").strip()]

    print(f"IPO watch: {len(ipos)} listed, {len(new)} not-yet-summarised; "
          f"{len(seed_items)} backfill DRHP seed(s) pending")
    for i, n in new:
        print(f"   - {n} ({i})")
    for i, n in seed_items:
        print(f"   - [seed] {n} ({i})")
    if not new and not seed_items:
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

    # Fresh Chittorgarh filings first (time-sensitive), then backfill seeds; one cost
    # cap over the combined list — leftover seeds drain on subsequent days.
    targets = (new + seed_items)[:args.max]

    pool = BucketPool(load_keys(os.environ), cdr.DEEPDIVE_MODELS,
                      inter_call_s=cdr.INTER_CALL_SLEEP)
    prompt = open(os.path.join(cdr.SCRIPTS_DIR, "drhp_prompt.txt"), encoding="utf-8").read()

    rows, emailed, processed_seed_ids = [], 0, []
    for ipo_id, name in targets:
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
        if str(ipo_id).startswith("seed_"):
            processed_seed_ids.append(ipo_id)

    led = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
    led = led.drop_duplicates("ipo_id", keep="last")    # latest status per IPO
    buf = io.BytesIO(); led.to_parquet(buf, index=False)
    cdr.drive_upload(svc, LEDGER, root, buf.getvalue(), "application/octet-stream")
    # Flip backfill seeds we just handled to 'consumed' (ledger already records outcome).
    if processed_seed_ids:
        _mark_seeds_consumed(svc, root, processed_seed_ids)
    print(f"IPO watch: processed {len(rows)} new ({len(processed_seed_ids)} from "
          f"backfill seeds), emailed {emailed}")


if __name__ == "__main__":
    main()
