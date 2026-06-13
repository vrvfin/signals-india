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
import re
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


# Generic words that must not, on their own, match a company name.
_NAME_STOP = {"LIMITED", "LTD", "INDIA", "INDIAN", "INDUSTRIES", "COMPANY", "CO",
              "CORP", "CORPORATION", "PVT", "PRIVATE", "ENTERPRISES", "GROUP",
              "PRODUCTS", "MEDICARE", "PHARMA", "TECHNOLOGIES", "TECH",
              "SERVICES", "FINANCE", "MOTORS", "STEEL", "POWER", "ENERGY"}
# Question words to ignore when scanning free text for a company.
_QUESTION_STOP = {"TELL", "ABOUT", "HOW", "WHAT", "WHY", "WHEN", "IS", "ARE",
                  "THE", "ME", "GOING", "DOING", "WITH", "AND", "FOR", "OF",
                  "ON", "IN", "TO", "A", "AN", "DOES", "DID", "HAS", "HAVE",
                  "THIS", "THAT", "ANY", "GIVE", "SHOW", "LATEST", "UPDATE",
                  "STATUS", "COMPANY", "STOCK", "SHARE", "PRICE", "NEWS"}


def _load_universe(drive, root):
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")
    fid = find_file(drive, idx, "company_universe.csv")
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid))) if fid else None


def resolve(drive, root, token: str, uni=None) -> tuple[str, str, str] | None:
    """Find the company a natural-language string is ABOUT.
    Tries, in order: exact symbol/ISIN; a symbol appearing as a word in the
    text; a distinctive company-name word appearing in the text (longest wins).
    So 'tell me about anondita how is it going?' -> ANONDITA."""
    if uni is None:
        uni = _load_universe(drive, root)
    if uni is None or uni.empty:
        return None
    sym_col = "nse_symbol" if "nse_symbol" in uni.columns else "symbol"
    raw = token.strip()
    up = raw.upper()

    def _ret(r):
        return (str(r["isin"]), str(r[sym_col]).upper(), str(r["name"]))

    # 1. exact symbol / ISIN
    for cond in (uni[sym_col].astype(str).str.upper() == up,
                 uni["isin"].astype(str).str.upper() == up):
        hit = uni[cond]
        if not hit.empty:
            return _ret(hit.iloc[0])

    # 2. whole-name substring (short inputs like "anondita medicare")
    if 4 <= len(up) <= 40:
        hit = uni[uni["name"].astype(str).str.upper().str.contains(
            up, na=False, regex=False)]
        if not hit.empty:
            return _ret(hit.iloc[0])

    # 3. scan free text for a symbol token or a distinctive name word
    words = {w for w in re.findall(r"[A-Z0-9&]{3,}", up)
             if w not in _QUESTION_STOP}
    if not words:
        return None
    syms = {str(s).upper(): i for i, s in uni[sym_col].items()}
    for w in words:
        if w in syms:                       # a symbol mentioned in the sentence
            return _ret(uni.loc[syms[w]])
    best = None                             # longest distinctive name-word match
    for i, nm in uni["name"].items():
        for nw in re.findall(r"[A-Z0-9&]{4,}", str(nm).upper()):
            if nw in _NAME_STOP:
                continue
            if nw in words and (best is None or len(nw) > best[0]):
                best = (len(nw), i)
    if best:
        return _ret(uni.loc[best[1]])
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
        import build_classification as bcl
        blk = bcl.classification_block(drive, root, isin, sym)
        if blk != "DATA_MISSING":
            parts.append("## CLASSIFICATION & PEERS\n" + blk)
    except Exception:
        pass
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
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    uni = _load_universe(drive, root)
    first = " ".join(sys.argv[1:]).strip()
    if not first:
        first = input("Ask about a company (mention its name, e.g. "
                      "'how is anondita doing?'): ").strip()

    pool = None
    sym = name = isin = None
    base = ""
    history: list[tuple[str, str]] = []
    pending = first                       # the first line may already be a question

    while True:
        if pending is None:
            try:
                pending = input(f"[{sym or '?'}] you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        msg, pending = pending, None
        if not msg or msg.lower() in ("exit", "quit"):
            break

        # (re)lock the company whenever the message names a different one
        hit = resolve(drive, root, msg, uni)
        if hit and hit[1] != sym:
            isin, sym, name = hit
            print(f"\n→ {sym} ({name}) — assembling Drive context…")
            ctx = assemble(drive, root, isin, sym, name)
            print(f"  {len(ctx):,} chars loaded.\n")
            base = SYSTEM.format(company=name, symbol=sym, context=ctx)
            history = []
        if not sym:
            print("  Which company? Mention its name or symbol.\n")
            continue

        # if the message was only the company name, wait for the actual question
        if msg.strip().upper() in (sym, name.upper()):
            continue
        if pool is None:
            pool = build_pool()
        try:
            a = answer(pool, base, history, msg)
        except Exception as e:
            print(f"  (call failed: {str(e)[:100]})")
            continue
        history.append((msg, a))
        print("\n" + a.encode("ascii", "replace").decode() + "\n")


if __name__ == "__main__":
    main()
