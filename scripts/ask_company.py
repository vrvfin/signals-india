r"""
ask_company.py — chat with EVERYTHING on Drive about one company (v1).

Assembles the company's full knowledge base deterministically (no embeddings
needed — the data is already organised per company) and chats over it with a
lite Gemini model. Sources are cited inline; missing data is said plainly.

Context assembled per company:
  company_page.md (concall/AR/rating/results summaries, GF tables)
  + quant snapshot (scorecard, fraud tracker w/ reasons, valuation, current
    guidance, PEAD verdicts, catalysts w/ what-to-track, derived ratios,
    AR focus/defocus)
  + user research intake (research_index) + community (VP/blogs/X)

Key pool (user 2026-06-12): BACKFILL_GEMINI_KEY -> DAILY_GEMINI_KEY ->
GEMINI_API_KEY.

Usage (local CLI — immune to Streamlit Cloud crashes; ask.bat wraps this):
    python scripts/ask_company.py TCS
    python scripts/ask_company.py "Shilpa Medicare"
The app's "💬 Ask" page reuses answer()/SYSTEM from here.
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, log)

MAX_PAGE_CHARS = 60_000
SYSTEM = """You are the user's personal equity research assistant for Indian
listed companies. Answer ONLY from the CONTEXT below — it is the user's own
research system (concall/AR summaries, scorecard, fraud tracker, guidance vs
actuals, catalysts, community posts). Rules:
1. Cite the section every claim comes from, e.g. [company page / concall
   Q4FY26], [fraud tracker], [ValuePickr · user]. Never invent facts.
2. If the context does not contain the answer, say DATA_MISSING and name what
   would answer it (e.g. "no concall summarised yet").
3. Be concise: direct answer first, then the supporting points.
4. Numbers beat narrative. Flag when management's story and the quant data
   disagree.

=== CONTEXT for {company} ({symbol}) ===
{context}
=== END CONTEXT ==="""


def resolve(drive, root, token: str) -> tuple[str, str, str] | None:
    """token (symbol/ISIN/name fragment) -> (isin, symbol, name)."""
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")
    fid = find_file(drive, idx, "company_universe.csv")
    if not fid:
        return None
    uni = pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
    t = token.strip().upper()
    sym_col = "nse_symbol" if "nse_symbol" in uni.columns else "symbol"
    for cond in (uni[sym_col].astype(str).str.upper() == t,
                 uni["isin"].astype(str).str.upper() == t,
                 uni["name"].astype(str).str.upper().str.contains(t, na=False)):
        hit = uni[cond]
        if not hit.empty:
            r = hit.iloc[0]
            return (str(r["isin"]), str(r[sym_col]).upper(), str(r["name"]))
    return None


def assemble(drive, root, isin: str, sym: str, name: str) -> str:
    """One big sectioned context string. Reuses the deep dive's quant block."""
    parts = []
    repo = get_or_create_subfolder(drive, root, "company_repo")
    comp = find_file(drive, repo, isin)
    if comp:
        pf = find_file(drive, comp, "company_page.md")
        if pf:
            page = download_bytes(drive, pf).decode("utf-8", errors="replace")
            parts.append("## COMPANY PAGE (all summarised documents)\n"
                         + page[-MAX_PAGE_CHARS:])
    try:
        import company_deep_report as cdr
        parts.append("## QUANT SNAPSHOT (nightly pipelines)\n"
                     + cdr.phase3_block(drive, root, isin, sym))
    except Exception as e:
        log(f"quant snapshot failed ({str(e)[:60]})")
    try:
        idx = get_or_create_subfolder(drive, repo, "_index")
        fid = find_file(drive, idx, "research_index.parquet")
        if fid:
            r = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
            hit = r[r["isins"].astype(str).str.contains(isin, na=False)].tail(5)
            if not hit.empty:
                parts.append("## USER RESEARCH INTAKE\n" + "\n".join(
                    f"- [{x.get('doc_type')} {str(x.get('doc_date'))[:10]}] "
                    f"{str(x.get('summary_md'))[:400]}"
                    for _, x in hit.iterrows()))
    except Exception:
        pass
    try:
        import social_sources
        blk = social_sources.community_block(name, days=21)
        if blk != "DATA_MISSING":
            parts.append("## COMMUNITY (VP / blogs)\n" + blk)
    except Exception:
        pass
    return "\n\n".join(parts) if parts else "DATA_MISSING (no coverage yet)"


def build_pool():
    """BACKFILL -> DAILY -> GEMINI cascade (user 2026-06-12), lite models."""
    from gemini_pool import BucketPool, load_keys
    keys = load_keys(os.environ, prefix="BACKFILL_GEMINI_KEY")
    for p in ("DAILY_GEMINI_KEY", "GEMINI_API_KEY"):
        keys += [k for k in load_keys(os.environ, prefix=p) if k not in keys]
    if not keys:
        raise SystemExit("no Gemini keys in env")
    return BucketPool(keys, ["gemini-2.5-flash-lite", "gemini-2.5-flash",
                             "gemini-2.0-flash-lite"], inter_call_s=2.0,
                      logger=lambda m: None)


def answer(pool, context_prompt: str, history: list[tuple[str, str]],
           question: str) -> str:
    """history = [(user, assistant), ...]; returns the model reply."""
    convo = "\n".join(f"USER: {u}\nASSISTANT: {a}" for u, a in history[-6:])
    prompt = (context_prompt
              + ("\n\n=== CONVERSATION SO FAR ===\n" + convo if convo else "")
              + f"\n\nUSER: {question}\nASSISTANT:")
    text, _ = pool.call_text(prompt)
    return text.strip()


def main() -> None:
    token = " ".join(sys.argv[1:]).strip()
    if not token:
        token = input("Company (symbol / ISIN / name): ").strip()
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    hit = resolve(drive, root, token)
    if not hit:
        print(f"'{token}' not found in the universe.")
        return
    isin, sym, name = hit
    print(f"Assembling everything on Drive for {sym} ({name}) ...")
    ctx = assemble(drive, root, isin, sym, name)
    print(f"  context: {len(ctx):,} chars. Pool: BACKFILL->DAILY->GEMINI.")
    pool = build_pool()
    base = SYSTEM.format(company=name, symbol=sym, context=ctx)
    history: list[tuple[str, str]] = []
    print("Ask away (blank line or 'exit' to quit).\n")
    while True:
        try:
            q = input(f"[{sym}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        try:
            a = answer(pool, base, history, q)
        except Exception as e:
            print(f"  (call failed: {str(e)[:100]})")
            continue
        history.append((q, a))
        print("\n" + a.encode("ascii", "replace").decode() + "\n")


if __name__ == "__main__":
    main()
