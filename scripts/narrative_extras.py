r"""
narrative_extras.py — inputs for the company report's added chapters (user 2026-10-18).

  recent_filings()   the company's latest NSE filings WITH their attached PDF text (orders, press
                     releases, business updates) — routine notices skipped. NSE works from the PC
                     and from CI; BSE's API is 403 from both (checked 2026-10-18).
  fraud_breakdown()  what the fraud score is made of, component by component, and which checks
                     came back clean — so the number can be read, not just quoted.
  guidance_facts()   management's latest explicit guidance, for the recent-updates chapter to test
                     against the newest filings.
  excerpts()         passages from the annual report / calls / presentations for one topic
                     (auditor, notes to accounts, board, geography and clients, supply chain,
                     right to win, competitive forces) — passed to that chapter as its SOURCE
                     DOCUMENTS under the original doc id, so every quote still verifies.
"""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

SEC_FRAUD, SEC_GUIDE, SEC_FILINGS = 36, 37, 38

# --------------------------------------------------------------- recent filings --
# Routine exchange notices with no business content.
_ROUTINE = re.compile(
    r"trading window|newspaper|shareholders? meeting|annual general meeting|\bagm\b|"
    r"scrutini[sz]er|postal ballot|record date|book closure|closure of register|"
    r"loss of share|duplicate share|certificate under regulation|reg(ulation)?\.? ?74|"
    r"esop|allotment of (equity )?shares|change in (registrar|rta)|investor (meet|call) "
    r"schedule|analysts?/institutional investor meet|intimation of board meeting|"
    r"compliance certificate|structured digital database|voting results", re.I)


def recent_filings(symbol: str, limit: int = 25, max_docs: int = 8,
                   text_chars: int = 12000, log=print) -> list[dict]:
    """[{doc_id, date, kind, subject, url, text}] newest first. Best-effort: [] on any error."""
    if not str(symbol or "").strip():
        return []
    try:
        import requests
        import fitz
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                          "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"})
        s.get("https://www.nseindia.com", timeout=12)
        r = s.get("https://www.nseindia.com/api/corporate-announcements",
                  params={"index": "equities", "symbol": symbol.upper()}, timeout=18)
        rows = r.json() if r.status_code == 200 else []
        rows = rows if isinstance(rows, list) else rows.get("data", [])
    except Exception as e:
        log(f"  recent filings: NSE unavailable ({str(e)[:70]})")
        return []
    out, seen = [], {}
    for a in rows[:limit]:
        kind, subj = str(a.get("desc") or "").strip(), str(a.get("attchmntText") or "").strip()
        if _ROUTINE.search(f"{kind} {subj}"):
            continue
        d = pd.to_datetime(str(a.get("an_dt") or a.get("sort_date") or "").split(" ")[0],
                           errors="coerce", dayfirst=True)
        if pd.isna(d):
            continue
        day = d.strftime("%Y-%m-%d")
        seen[day] = seen.get(day, 0) + 1
        did = f"announcement_{day}" + (f"_{seen[day]}" if seen[day] > 1 else "")
        url, text = str(a.get("attchmntFile") or ""), ""
        if url.lower().endswith(".pdf"):
            try:
                pr = s.get(url, timeout=30)
                if pr.status_code == 200 and pr.content[:4] == b"%PDF":
                    doc = fitz.open(stream=pr.content, filetype="pdf")
                    text = " ".join(" ".join(p.get_text() for p in doc).split())[:text_chars]
            except Exception:
                text = ""
        # NSE's own subject is often generic ("has informed the Exchange about General
        # Updates"); the letter's "Sub:" line says what it is (Morepen 28 Sep: "Press
        # Release - Morepen strengthens integrated CDMO capabilities with first U.S. ANDA
        # submission").
        m = re.search(r"\bSub(?:ject)?\s*[.:]+\s*(.{8,240}?)\s+(?:Dear|Ref\b|Respected|Sir|Madam)", text or "", re.I)
        if m and (not subj or "informed the exchange" in subj.lower()):
            subj = m.group(1).strip(" -–")
        out.append({"doc_id": did, "date": day, "kind": kind, "subject": subj, "url": url,
                    "text": text or subj})
        if len(out) >= max_docs:
            break
    log(f"  recent filings: {len(out)} non-routine NSE filing(s) read"
        + (f" ({sum(1 for f in out if len(f['text']) > len(f['subject']))} with PDF text)"
           if out else ""))
    return out


def add_filings(pack, filings: list[dict]) -> None:
    if not filings:
        return
    pack.table("tbl.recent_filings", "Latest filings with business content", SEC_FILINGS,
               ["date", "kind", "subject"],
               [{"date": f["date"], "kind": f["kind"], "subject": f["subject"][:180]}
                for f in filings],
               {"kind": "document", "doc_type": "exchange filing", "title": "NSE announcements"},
               note="Routine notices (AGM, trading window, newspaper, allotments) left out.")


# --------------------------------------------------------------- fraud breakdown --
def fraud_breakdown(pack, store) -> None:
    """Section 36 — the fraud score's parts. Every row says what it measures and what it
    found, so 'RED 75' can be read as e.g. 'exchange surveillance, not evidence of fraud'."""
    def row(name):
        try:
            df = store.by_isin(name, pack.isin, pack.symbol)
            return df.iloc[-1].to_dict() if not df.empty else {}
        except Exception:
            return {}
    ft, inv, fr = row("fraud_tracker.parquet"), row("investigative_fraud.parquet"), \
        row("fraud_risk.parquet")
    if not (ft or inv or fr):
        pack.gap(SEC_FRAUD, "no fraud tracker / surveillance / forensic rows for this company")
        return
    src = lambda t: {"kind": "parquet", "table": f"company_repo/_index/{t}", "note": ""}
    num = lambda v: pd.to_numeric(v, errors="coerce")
    rows = []

    def add(fid, label, value, unit, table, what, reading, status):
        v = num(value) if unit != "text" else value
        if unit != "text" and pd.notna(v):
            pack.add(fid, label, float(v), unit, "reported", SEC_FRAUD, src(table))
        rows.append({"check": label, "what_it_measures": what,
                     "reading": reading, "status": status})

    if ft:
        sc, band = num(ft.get("fraud_score")), str(ft.get("band") or "")
        add("fraud.score", "Fraud tracker score", sc, "score", "fraud_tracker.parquet",
            "Combined score: market surveillance + forensic accounting flags + regulator "
            "actions + news (higher = more concern)",
            f"{'' if pd.isna(sc) else int(sc)} ({band}); driven by "
            f"{ft.get('score_driver') or 'n/a'}", band or "n/a")
        reason = str(ft.get("reason") or "")
        if reason:
            rows.append({"check": "Why the tracker scored it so", "what_it_measures":
                         "The tracker's own breakdown", "reading": reason[:220], "status": ""})
        fs = num(ft.get("forensic_score"))
        add("fraud.forensic_score", "Forensic accounting score", fs, "score",
            "fraud_tracker.parquet", "Accounting red flags from the statements (cash vs "
            "profit, returns, receivables, etc.)",
            f"{'' if pd.isna(fs) else int(fs)}; flags: {ft.get('forensic_flags') or 'none'}",
            "flag" if (pd.notna(fs) and fs > 0) else "clean")
        for col, label, what in (("sebi_actions", "SEBI actions", "Orders / actions by the "
                                  "market regulator naming the company"),
                                 ("nfra_actions", "NFRA actions", "Audit-quality actions by "
                                  "the National Financial Reporting Authority"),
                                 ("news_hits", "Adverse news hits", "Fraud / probe / default "
                                  "keywords in recent news")):
            v = num(ft.get(col))
            add(f"fraud.{col}", label, v, "count", "fraud_tracker.parquet", what,
                "" if pd.isna(v) else str(int(v)), "clean" if (pd.notna(v) and v == 0) else
                ("flag" if pd.notna(v) else "n/a"))
        t7 = num(ft.get("score_7d_ago"))
        if pd.notna(t7):
            pack.add("fraud.score_7d_ago", "Fraud tracker score 7 days ago", float(t7),
                     "score", "reported", SEC_FRAUD, src("fraud_tracker.parquet"))
    if inv:
        asm, esm, gsm = (str(inv.get(k) or "none") for k in ("asm_level", "esm_level",
                                                            "gsm_stage"))
        t2t = str(inv.get("t2t")).lower() in ("true", "1", "yes")
        rows.append({"check": "Exchange surveillance (ASM / ESM / GSM)",
                     "what_it_measures": "Price-and-volume surveillance by NSE/BSE — about "
                     "trading behaviour, NOT a finding of fraud",
                     "reading": f"ASM: {asm} · ESM: {esm} · GSM stage: {gsm}",
                     "status": "flag" if (asm not in ("none", "nan", "0") or esm not in
                                          ("none", "nan") or gsm not in ("0", "none", "nan"))
                     else "clean"})
        rows.append({"check": "Trade-to-trade segment", "what_it_measures": "Shares moved to "
                     "compulsory delivery (no intraday) by the exchange",
                     "reading": "yes" if t2t else "no", "status": "flag" if t2t else "clean"})
    if fr and not ft:
        add("fraud.risk_score", "Fraud risk score", fr.get("fraud_risk_score"), "score",
            "fraud_risk.parquet", "Forensic flags + news", str(fr.get("forensic_flags") or ""),
            "")
    pack.table("tbl.fraud_breakdown", "What the fraud score is made of", SEC_FRAUD,
               ["check", "what_it_measures", "reading", "status"], rows,
               src("fraud_tracker.parquet"),
               note="'clean' = the check ran and found nothing. Surveillance measures are "
                    "about trading, not accounting.")


# --------------------------------------------------------------- guidance --
def guidance_facts(pack, store, quarters: int = 2) -> None:
    """Section 37 — management's latest explicit guidance (the last `quarters` calls)."""
    try:
        g = store.by_isin("guidance_tracker.parquet", pack.isin, pack.symbol)
    except Exception:
        return
    if g is None or g.empty or "quarter" not in g.columns:
        return

    def qk(q):
        m = re.match(r"Q([1-4])\s*FY\s*(\d{2,4})", str(q).strip(), re.I)
        if not m:
            return (-1, -1)
        y = int(m.group(2))
        return (y + 2000 if y < 100 else y, int(m.group(1)))
    g = g.assign(_k=g["quarter"].map(qk))
    g = g[(g["_k"] != (-1, -1)) & (g["guidance_type"].astype(str) == "explicit")]
    if g.empty:
        return
    keep = sorted(set(g["_k"]))[-quarters:]
    g = g[g["_k"].isin(keep)].sort_values(["_k", "metric"])
    rows = []
    for i, r in enumerate(g.itertuples(index=False)):
        v = pd.to_numeric(getattr(r, "value_num", None), errors="coerce")
        fid = f"guide.{str(r.metric).lower().replace(' ', '_')}.{str(r.quarter).replace(' ', '')}.{i}"
        if pd.notna(v):
            pack.add(fid, f"Guidance ({r.quarter}): {r.metric} {r.value} over {r.horizon_fy}",
                     float(v), str(getattr(r, "value_unit", "") or "%"), "reported", SEC_GUIDE,
                     {"kind": "parquet", "table": "company_repo/_index/guidance_tracker.parquet",
                      "note": f"call {r.quarter}"})
        rows.append({"quarter": r.quarter, "metric": r.metric, "guidance": r.value,
                     "horizon": r.horizon_fy})
    if rows:
        pack.table("tbl.guidance", "What management guided on the latest calls", SEC_GUIDE,
                   ["quarter", "metric", "guidance", "horizon"], rows,
                   {"kind": "parquet", "table": "company_repo/_index/guidance_tracker.parquet",
                    "note": ""})


# --------------------------------------------------------------- excerpts --
TOPICS: dict[str, list[str]] = {
    "auditor": [r"independent auditor", r"key audit matter", r"statutory auditor",
                r"basis for (qualified )?opinion", r"emphasis of matter", r"\bCARO\b",
                r"companies \(auditor'?s report\) order", r"auditors? (have|has) (resigned|been re-?appointed)",
                r"appointment of (statutory )?auditor"],
    "notes": [r"contingent liabilit", r"related party", r"change in accounting polic",
              r"exceptional item", r"impairment", r"going concern", r"subsequent event",
              r"commitments", r"guarantees? given", r"provision for doubtful", r"write.?off"],
    "board": [r"composition of the board", r"board of directors comprises", r"independent director",
              r"wom[ae]n director", r"board meetings? (were )?held", r"attendance",
              r"category of director", r"executive director", r"chairman"],
    "geo": [r"export", r"geograph", r"domestic market", r"overseas", r"countries",
            r"\bcustomers?\b", r"top (5|five|10|ten) customers", r"client", r"\bUSA?\b",
            r"europe", r"latam|latin america", r"segment"],
    "supply": [r"raw material", r"key starting material|\bKSM\b", r"supplier", r"import",
               r"china", r"procure", r"input cost", r"pass.?through", r"inventory",
               r"logistics", r"freight"],
    "moat": [r"USFDA|US FDA", r"EU.?GMP", r"\bWHO\b", r"\bANDA\b", r"\bDMF\b", r"CEP",
             r"approval", r"patent", r"technolog", r"R ?& ?D", r"complex (chemistry|molecule)",
             r"CDMO", r"capabilit", r"certif", r"accredit", r"market share", r"leader"],
    "porter": [r"competit", r"pricing pressure", r"new entrant", r"barrier", r"substitut",
               r"bargaining", r"customer concentration", r"supplier concentration",
               r"capacity addition", r"price war", r"market share"],
}
_KIND_ORDER = ("annual_report", "concall", "presentation", "announcement", "results", "rating")


def excerpts(sources: dict[str, str], topic: str, window: int = 700,
             per_doc: int = 8, cap: int = 24000) -> dict[str, str]:
    """{doc_id: passages joined by ' ... '} for one topic, annual reports first. Passages are
    exact slices of the document, so an evidence span taken from them is also a substring of
    the full document (the span check passes; the auditor can find it)."""
    pats = [re.compile(p, re.I) for p in TOPICS.get(topic, [])]
    if not pats:
        return {}
    # annual reports first, then calls, ...; NEWEST first within a kind
    def _rank(d):
        kind = next((i for i, k in enumerate(_KIND_ORDER) if d.startswith(k)), 9)
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", d)
        return (kind, -int("".join(m.groups())) if m else 0)
    order = sorted(sources, key=_rank)
    out, used = {}, 0
    for did in order:
        text = sources[did] or ""
        spans = []
        for p in pats:
            for m in p.finditer(text):
                spans.append((max(0, m.start() - window // 3), min(len(text), m.end() + window)))
        if not spans:
            continue
        spans.sort()
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        merged = sorted(merged, key=lambda ab: -sum(1 for p in pats
                                                    if p.search(text[ab[0]:ab[1]])))[:per_doc]
        merged.sort()
        chunk = " ... ".join(text[a:b] for a, b in merged)
        if used + len(chunk) > cap:
            chunk = chunk[:max(0, cap - used)]
        if chunk:
            out[did] = chunk
            used += len(chunk)
        if used >= cap:
            break
    return out
