r"""
research_context.py — research-store context for the company deep dive (Workflow B).

Replaces the deep dive's old RESEARCH_INDEX_CONTEXT (up to 25 lines of "doc_type | source |
date | file name", no content) with the research itself, read from the store built by
research_rag.py (Drive company_repo/_research/{chunks,embeddings}.parquet; a local _rag/
copy is preferred when present). numpy only — no Chroma/LangChain — so it runs in CI.

  A. THIS COMPANY   its company cards + summaries / flag lines of documents about it
  B. SECTOR         chunks tagged with its industry/sector (peers included), ranked by cosine
                    similarity to the company's own research, last 12 months first
  C. MACRO/POLICY   macro-factor / policy chunks that touch its sector

    python scripts/research_context.py --isin INE0DYJ01015 [--symbol SYRMA] [--show]   # dry run

Never raises into the caller's happy path: context_block() returns (None, reason) when the
store is missing or the company has nothing, and the deep dive keeps its old behaviour.
"""
from __future__ import annotations
import io, os, re, sys, argparse, collections, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
LOCAL = Path(os.environ.get("RESEARCH_RAG_DIR", ROOT / "_rag"))
DRIVE_RESEARCH = "company_repo/_research"
DRIVE_INDEX = "company_repo/_index"

BUDGET = {"company": 40_000, "sector": 25_000, "macro": 8_000}   # characters per part
PER_DOC = 2                     # max chunks from one document per part
SECTOR_DAYS = 365               # sector / macro research older than this ranks last
MAX_FOCUSED_ISINS = 5           # a doc naming more companies is a sector doc, not "about" this one

_STORE: dict = {}


def _read(svc, root, name: str, drive_dir: str = DRIVE_RESEARCH) -> pd.DataFrame | None:
    p = (LOCAL / name) if drive_dir == DRIVE_RESEARCH else (ROOT / "scripts" / name)
    if p.exists():
        return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p, keep_default_na=False)
    if svc is None:
        return None
    from daily_research_summary import drive_download
    b = drive_download(svc, f"{drive_dir}/{name}", root)
    if not b:
        return None
    return pd.read_parquet(io.BytesIO(b)) if name.endswith(".parquet") else pd.read_csv(io.BytesIO(b), keep_default_na=False)

def load_store(svc, root):
    """chunks (+ unit vectors) and tag display names; cached for the whole process."""
    if "ch" in _STORE:
        return _STORE["ch"], _STORE["vec"], _STORE["names"]
    ch = _read(svc, root, "chunks.parquet")
    emb = _read(svc, root, "embeddings.parquet")
    if ch is None or emb is None:
        raise FileNotFoundError(f"research store not found (local {LOCAL} or Drive {DRIVE_RESEARCH})")
    emb = emb.drop_duplicates("text_sha")
    ch = ch.merge(emb[["text_sha", "vector"]], on="text_sha", how="left").reset_index(drop=True)
    dim = len(next(v for v in ch.vector if v is not None and not (isinstance(v, float))))
    vec = np.zeros((len(ch), dim), dtype=np.float32)
    for i, v in enumerate(ch.vector):
        if v is not None and not isinstance(v, float):
            vec[i] = v
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
    ch = ch.drop(columns=["vector"])
    tx = _read(svc, root, "tag_taxonomy.csv", DRIVE_INDEX)
    names = dict(zip(tx.slug, tx.display_name)) if tx is not None else {}
    _STORE.update(ch=ch, vec=vec, names=names)
    return ch, vec, names


def _body(t: str) -> str:
    return t.split("\n\n", 1)[-1].strip()

def _age_days(ts: int, today: dt.date) -> int:
    if not ts:
        return 10_000
    try:
        return (today - dt.datetime.strptime(str(int(ts)), "%Y%m%d").date()).days
    except ValueError:
        return 10_000

def _fmt(r) -> str:
    who = f" · {r.company}" if r.company else ""
    return (f"### [research_{int(r.research_n):04d} · {r.source_raw or r.source} · "
            f"{r.doc_date or 'undated'} · {r.doc_type}{who}]\n{_body(r.page_content)}")

def _take(rows, budget: int, per_doc: int) -> list:
    out, used, per = [], 0, collections.Counter()
    for r in rows:
        if per[r.research_n] >= per_doc:
            continue
        txt = _fmt(r)
        if used + len(txt) > budget:
            if used == 0:                                   # always give at least one, trimmed
                out.append(txt[:budget]); used = budget
            continue
        out.append(txt); used += len(txt); per[r.research_n] += 1
    return out

def _flag_lines(r, needles: list[str]) -> str | None:
    """Keep only the flag-table rows that mention this company."""
    lines = _body(r.page_content).splitlines()
    keep = [l for l in lines if any(n in l.lower() for n in needles)]
    return "\n".join(keep) if keep else None


def build(svc, root, isin: str, symbol: str = "", name: str = "", today: dt.date | None = None) -> dict:
    ch, vec, names = load_store(svc, root)
    today = today or dt.date.today()
    own = ch[ch["isin"] == isin]
    mention = ch[ch.doc_isins.str.contains(isin, regex=False) & (ch["isin"] != isin)]
    n_isins = mention.doc_isins.str.count(r"\|") + 1
    focused = mention[(n_isins <= MAX_FOCUSED_ISINS) & mention.chunk_type.isin(["doc_summary", "themes", "text"])]
    flags = mention[mention.chunk_type == "flags"]

    # ---- A. this company -------------------------------------------------------------
    needles = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", name or "")[:2]] + ([symbol.lower()] if symbol else [])
    needles = [n for n in needles if n not in ("limited", "india", "industries")] or [isin.lower()]
    flag_rows = []
    for r in flags.itertuples():
        kept = _flag_lines(r, needles)
        if kept:
            flag_rows.append(r._replace(page_content="x\n\n" + kept))
    a_rows = sorted(list(own.itertuples()) + list(focused.itertuples()) + flag_rows,
                    key=lambda r: (-int(r.doc_ts or 0), r.chunk_type != "company_card"))
    part_a = _take(a_rows, BUDGET["company"], PER_DOC + 1)
    a_docs = set(own.research_n) | set(focused.research_n) | {r.research_n for r in flag_rows}

    # ---- the company's industry / sector -------------------------------------------------
    ind = own.industry[own.industry != ""].mode()
    sec = own.sector[own.sector != ""].mode()
    grp = own.sector_group[own.sector_group != ""].mode()
    industry, sector, group = (ind.iloc[0] if len(ind) else ""), (sec.iloc[0] if len(sec) else ""), \
        (grp.iloc[0] if len(grp) else "")

    # query vector: centroid of the company's own research (what this business looks like)
    own_idx = own.index.to_numpy()
    q = vec[own_idx].mean(axis=0) if len(own_idx) else None

    def ranked(cands: pd.DataFrame, boost_tag: str) -> list:
        if cands.empty:
            return []
        idx = cands.index.to_numpy()
        base = vec[idx] @ (q / max(np.linalg.norm(q), 1e-9)) if q is not None else np.zeros(len(idx))
        age = np.array([_age_days(t, today) for t in cands.doc_ts])
        score = base + 0.05 * np.clip(1 - age / SECTOR_DAYS, 0, 1) \
            + (0.05 * cands.tag_sectors.str.contains(f"|{boost_tag}|", regex=False).to_numpy() if boost_tag else 0)
        score = score - 1.0 * (age > SECTOR_DAYS)                      # stale research last
        order = np.argsort(-score)
        return [cands.iloc[i] for i in order]

    # ---- B. sector & industry --------------------------------------------------------------
    part_b, b_label = [], ""
    target = industry or sector
    if target:
        pat = "|".join(re.escape(f"|{t}|") for t in (industry, sector) if t)
        cands = ch[ch.tag_sectors.str.contains(pat, regex=True) & (ch["isin"] != isin)
                   & ~ch.doc_isins.str.contains(isin, regex=False) & ~ch.research_n.isin(a_docs)
                   & ch.chunk_type.isin(["themes", "doc_summary", "company_card", "text"])]
        part_b = _take(ranked(cands, industry), BUDGET["sector"], PER_DOC)
        b_label = " · ".join(f"{k}: {names.get(v, v)}" for k, v in (("industry", industry), ("sector", sector)) if v)

    # ---- C. macro & policy touching the sector ----------------------------------------------
    part_c = []
    if sector or group:
        pat = "|".join(re.escape(f"|{t}|") for t in (sector, group) if t)
        cands = ch[((ch.tag_macro != "") | (ch.tag_policies != "")) & ch.tag_sectors.str.contains(pat, regex=True)
                   & ch.chunk_type.isin(["themes", "doc_summary", "text"]) & (ch["isin"] != isin)
                   & ~ch.research_n.isin(a_docs)]
        part_c = _take(ranked(cands, sector), BUDGET["macro"], 1)

    return dict(part_a=part_a, part_b=part_b, part_c=part_c, industry=industry, sector=sector,
                group=group, label=b_label, own_cards=int((own.chunk_type == "company_card").sum()))


def context_block(svc, root, isin: str, symbol: str = "", name: str = "") -> tuple[str | None, str]:
    """(text for the RESEARCH_INDEX_CONTEXT slot, one-line stats) — (None, reason) to fall back."""
    try:
        c = build(svc, root, isin, symbol, name)
    except Exception as e:
        return None, f"research store unavailable ({type(e).__name__}: {str(e)[:100]})"
    if not (c["part_a"] or c["part_b"] or c["part_c"]):
        return None, "research store has nothing for this company or its sector"
    text = (f"== A. RESEARCH ON THIS COMPANY — {len(c['part_a'])} extracts (broker company cards, "
            f"focused notes, watch-list flag rows) ==\n\n" + ("\n\n".join(c["part_a"]) or "None in the research store.")
            + f"\n\n== B. SECTOR & INDUSTRY RESEARCH — {c['label'] or 'sector not identified'} — "
            f"{len(c['part_b'])} extracts (peers included; most relevant + recent first) ==\n\n"
            + ("\n\n".join(c["part_b"]) or "None in the research store.")
            + f"\n\n== C. MACRO & POLICY TOUCHING THIS SECTOR — {len(c['part_c'])} extracts ==\n\n"
            + ("\n\n".join(c["part_c"]) or "None in the research store."))
    stats = (f"research store: company {len(c['part_a'])} · sector {len(c['part_b'])} "
             f"({c['label'] or 'n/a'}) · macro/policy {len(c['part_c'])} · {len(text):,} chars")
    return text, stats


def main():
    ap = argparse.ArgumentParser(description="dry run: print the research context a deep dive would get")
    ap.add_argument("--isin", required=True)
    ap.add_argument("--symbol", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--show", action="store_true", help="print the full text, not just headers")
    a = ap.parse_args()
    text, stats = context_block(None, None, a.isin, a.symbol, a.name)
    print(stats)
    if text:
        print(text if a.show else "\n".join(l for l in text.splitlines() if l.startswith(("==", "### ["))))

if __name__ == "__main__":
    main()
