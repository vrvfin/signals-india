r"""
research_rag.py — searchable research store: chunks + Gemini vectors (Chroma via LangChain)
+ keyword index (SQLite FTS5) + RAG answers with citations.

Source: the research summaries Workflow A already stores in research_index.parquet (no PDFs).

    python scripts/research_rag.py build  [--dry-run] [--cache DIR]     # chunk + tag
    python scripts/research_rag.py embed  [--dry-run] [--limit N] [--deadline-min M]
    python scripts/research_rag.py search "order book for pipe makers" [--isin/--sector/--theme/--doc-type/--since/-k]
    python scripts/research_rag.py ask    "What are brokers saying about EMS margins?" [-k 12]

Chunk types (one LangChain Document each — page_content + flat metadata):
  doc_summary   header line + executive summary              (one per document)
  company_card  one "--- Company (TICKER) ---" block          (resolved to ISIN where possible)
  themes        cross-company themes                          flags  watchlist-flag table
  text          paragraph chunks for summaries without the standard sections
Each chunk is tagged with the v2 taxonomy (tag_aliases): text tags from an n-gram scan, plus
for company cards the company's own sector / industry from company_classification.

Storage: Drive company_repo/_research/{chunks,embeddings}.parquet (source of truth).
Local _rag/ (gitignored): chroma/ vector store + fts.sqlite keyword index, rebuilt from them.
Embeddings: gemini-embedding-001, 768 dims, FREE_POOL keys (rotating; dead / rate-limited keys
skipped via gemini_pool.classify_error). Resumes: only chunks without a vector are embedded.
"""
from __future__ import annotations
import os, re, io, sys, json, time, hashlib, sqlite3, argparse, collections, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
ROOT_DIR = SCRIPTS_DIR.parent
LOCAL = ROOT_DIR / "_rag"
CHROMA_DIR = LOCAL / "chroma"
FTS_DB = LOCAL / "fts.sqlite"
LOCAL_CHUNKS = LOCAL / "chunks.parquet"
LOCAL_EMB = LOCAL / "embeddings.parquet"

DRIVE_RESEARCH = "company_repo/_research"
DRIVE_INDEX = "company_repo/_index"
COLLECTION = "research_chunks"

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_BATCH = 100            # API max texts per request
BATCH_TOKENS = 12_000        # real tokens per request; a key takes ~30k/min (see EmbedPool)

def est_tokens(text: str) -> int:
    """Measured 2026-09-26: count_tokens gave 4,170 where chars/4 gave 3,271 -> ~3.1 chars/token."""
    return len(text) * 10 // 31 + 1
EMBED_CHARS = 7000            # model input cap ~2k tokens; longer chunks are split before this
MAX_CHUNK = 6000
GENERIC_DF = 0.15             # a phrase found in >15% of chunks tags nothing (too generic)
# Single words that are fine as a SECTOR-FIELD alias but ambiguous in running text
# (measured 2026-09-27 on 4,000 chunks: "order book" -> Print & Publishing, "growth rate" ->
# bond yields, "lead to" -> zinc & lead, "reach" -> EU REACH, "order pipeline" -> gas
# pipelines). They never tag free text; multi-word phrases containing them still do.
TEXT_STOP = set("""book board investment valuation rate platform service material pipeline security reach
other lead trading distribution small generation trade equity debt credit manufacturing upgrade tech
technology digital consumer industrial finance retail global power energy infrastructure bulk block index
express windows commercial""".split())
PER_DOC = 2                   # max chunks from one document in a result list
TEXT_TAG_TYPES = ("sector_group", "sector", "subsector", "theme", "macro_factor", "policy")
KEY_PREFIXES = "FREE_POOL,BACKFILL_GEMINI_KEY,GEMINI_API_KEY"

from build_tag_aliases import norm, match_key, _singular   # the vocabulary's own matcher


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# =========================================================================== inputs
def _drive():
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    from daily_research_summary import drive_service
    return drive_service(), os.environ["GDRIVE_FOLDER_ID"]

def _dl(svc, root, path):
    from daily_research_summary import drive_download
    return drive_download(svc, path, root)

def _ul(svc, root, path, data, mime="application/octet-stream"):
    from daily_research_summary import drive_upload_with_retry
    drive_upload_with_retry(svc, path, root, data, mime)

def load_inputs(cache: str | None):
    """research_index, tag_aliases, taxonomy, universe, classification."""
    if cache:
        c = Path(cache)
        ri = pd.read_parquet(c / "research_index.parquet")
        cl = pd.read_csv(c / "classification.csv", keep_default_na=False)
        uni = pd.read_csv(c / "company_universe.csv", keep_default_na=False, dtype=str)
    else:
        svc, root = _drive()
        import build_classification as bcl
        ri = pd.read_parquet(io.BytesIO(_dl(svc, root, f"{DRIVE_INDEX}/research_index.parquet")))
        cl = bcl.load_classification(svc, root)
        uni = pd.read_csv(io.BytesIO(_dl(svc, root, f"{DRIVE_INDEX}/company_universe.csv")),
                          keep_default_na=False, dtype=str)
    al_path, tx_path = SCRIPTS_DIR / "tag_aliases.csv", SCRIPTS_DIR / "tag_taxonomy.csv"
    if not (al_path.exists() and tx_path.exists()):
        raise SystemExit("tag_aliases.csv / tag_taxonomy.csv missing — run build_tag_aliases.py first")
    al = pd.read_csv(al_path, keep_default_na=False)
    tx = pd.read_csv(tx_path, keep_default_na=False)
    return ri, al, tx, uni, cl


# =========================================================================== taxonomy helpers
class Taxonomy:
    def __init__(self, al: pd.DataFrame, tx: pd.DataFrame):
        self.parent = dict(zip(tx.slug, tx.parent_slug))
        self.ttype = dict(zip(tx.slug, tx.tag_type))
        self.display = dict(zip(tx.slug, tx.display_name))
        self.by_type = collections.defaultdict(dict)            # type -> match_key -> slug
        self.phrases = collections.defaultdict(set)             # match_key -> {slug} (text types)
        self.acronyms = collections.defaultdict(set)            # short upper-case terms, queries only
        for r in al.itertuples():
            k = match_key(r.alias)
            if not k:
                continue
            self.by_type[r.tag_type].setdefault(k, r.slug)
            if r.tag_type in TEXT_TAG_TYPES and not k.isdigit():
                if (len(k) >= 4 or " " in k) and k not in TEXT_STOP:
                    self.phrases[k].add(r.slug)
                elif len(k) >= 2 and " " not in k:
                    self.acronyms[k].add(r.slug)                 # "ems", "cgd", "pli": query-only
        self.first_tok = {k.split()[0] for k in self.phrases}
        self.max_n = max(len(k.split()) for k in self.phrases)

    def lookup(self, text: str, ttype: str, default: str = "") -> str:
        return self.by_type[ttype].get(match_key(text), default)

    def ancestors(self, slug: str) -> list[str]:
        out, s = [], slug
        while s:
            out.append(s)
            s = self.parent.get(s, "")
        return out

    def scan(self, text: str) -> set[str]:
        """Phrase keys (not slugs) found in text — slugs are resolved after the generic filter."""
        toks = [_singular(t) for t in norm(text).split()]
        found = set()
        for i, t in enumerate(toks):
            if t not in self.first_tok:
                continue
            for n in range(1, self.max_n + 1):
                if i + n > len(toks):
                    break
                k = " ".join(toks[i:i + n])
                if k in self.phrases:
                    found.add(k)
        return found


# =========================================================================== company resolution
_SFX = r"\b(limited|ltd|private|pvt|india|industries|corporation|corp|company|co|the|inc|plc)\b"

def _cname(s: str) -> str:
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(_SFX, " ", s)
    return re.sub(r"\s+", " ", s).strip()

class Companies:
    def __init__(self, uni: pd.DataFrame, cl: pd.DataFrame, tax: Taxonomy):
        self.sym, self.name, self.label, self.prefix = {}, {}, {}, {}
        for r in uni.itertuples():
            isin = str(getattr(r, "isin", "")).strip()
            if not isin.startswith("IN"):
                continue
            self.label[isin] = str(getattr(r, "name", ""))
            for f in ("nse_symbol", "bse_symbol", "bse_code"):
                v = str(getattr(r, f, "") or "").strip().upper()
                if v and v != "NAN":
                    self.sym.setdefault(v.split(".")[0], isin)
            k = _cname(getattr(r, "name", ""))
            if k:
                self.name.setdefault(k, isin)
                toks = k.split()                                # "syrma sgs technology" ->
                for n in range(2, len(toks)):                   # prefixes "syrma sgs" (unique only)
                    p = " ".join(toks[:n])
                    self.prefix[p] = isin if self.prefix.get(p, isin) == isin else None
        # company -> (group, sector, industry) from classification labels, most specific first
        self.sector = {}
        for r in cl.itertuples():
            isin = str(getattr(r, "isin", ""))
            sub = sec = ""
            for col in ("peer_group", "subsector", "industry"):
                v = str(getattr(r, col, "") or "")
                if not sub:
                    sub = tax.lookup(v, "subsector")
                if not sec:
                    sec = tax.lookup(v, "sector")
            if not sec:
                sec = tax.lookup(str(getattr(r, "sector", "")), "sector")
            if sub and not sec:
                sec = tax.parent.get(sub, "")
            grp = tax.parent.get(sec, "") if sec else tax.lookup(str(getattr(r, "macro_sector", "")), "sector_group")
            self.sector[isin] = (grp, sec, sub)

    def resolve(self, header: str, doc_isins: list[str]) -> tuple[str, str, str]:
        """'Man Industries (India) Ltd (MANINDS)' -> (name, ticker, isin)."""
        h = header.strip().strip("*[]# ").strip()
        m = re.search(r"\(([A-Za-z0-9&\-]{1,20})(?:[ .](?:IN|IS|IB|NS|BO|NSE|BSE))?\)\s*$", h)
        ticker = m.group(1).upper() if m else ""
        name = h[:m.start()].strip() if m else h
        if ticker.startswith("INE") and len(ticker) == 12:
            return name, ticker, ticker
        if ticker and ticker in self.sym:
            return name, ticker, self.sym[ticker]
        k = _cname(name)
        isin = self.name.get(k, "") or (self.prefix.get(k) or "")
        if not isin and k:
            parts = k.split()
            for n in range(len(parts) - 1, 1, -1):              # "tata motors cv" -> "tata motors"
                isin = self.name.get(" ".join(parts[:n]), "")
                if isin:
                    break
        if not isin:                                            # fall back to the doc's own ISINs
            for di in doc_isins:
                lab = _cname(self.label.get(di, ""))
                if lab and k and (k in lab or lab in k):
                    isin = di
                    break
        return name, ticker, isin


# =========================================================================== chunking
_SEC_RX = re.compile(
    r"(?im)^[ \t#*]*(?:OUTPUT\s+)?SECTION\s+([A-F])\b[^\n]*$"
    r"|^[ \t#*]*(DOCUMENT HEADER|EXECUTIVE SUMMARY|PER[ -]COMPANY INTELLIGENCE|CROSS[- ]COMPANY THEMES|"
    r"INVESTMENT WATCHLIST FLAGS|STRUCTURED TAGS)[^\n]*$")
_NAME2SEC = {"DOCUMENT": "A", "EXECUTIVE": "B", "PER": "C", "CROSS": "D", "INVESTMENT": "E", "STRUCTURED": "F"}
_CO_RX = re.compile(r"(?m)^[ \t]*-{2,}[ \t]*\[?(.{2,120}?)\]?[ \t]*-{2,}[ \t]*$")

def split_sections(md: str) -> dict[str, str]:
    md = re.sub(r"(?m)^[ \t]*={3,}[ \t]*$", "", md)
    marks = []
    for m in _SEC_RX.finditer(md):
        letter = m.group(1) or _NAME2SEC[re.split(r"[ -]", m.group(2).upper())[0]]
        marks.append((m.start(), m.end(), letter.upper()))
    out = collections.defaultdict(str)
    for i, (s, e, letter) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(md)
        out[letter] += md[e:end].strip() + "\n"
    return dict(out)

def split_long(text: str, limit: int = MAX_CHUNK) -> list[str]:
    """Paragraph-packed parts of <= limit chars; oversize paragraphs are cut on lines, then
    hard-cut; the last short paragraph of a part is repeated at the start of the next."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    units = []
    for p in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not p:
            continue
        if len(p) <= limit:
            units.append(p)
            continue
        buf = ""
        for ln in p.split("\n"):
            while len(ln) > limit:
                if buf:
                    units.append(buf.strip()); buf = ""
                units.append(ln[:limit]); ln = ln[limit:]
            if len(buf) + len(ln) + 1 > limit:
                units.append(buf.strip()); buf = ""
            buf += ln + "\n"
        if buf.strip():
            units.append(buf.strip())
    parts, cur = [], []
    for u in units:
        if cur and sum(len(x) + 2 for x in cur) + len(u) > limit:
            parts.append("\n\n".join(cur))
            cur = [cur[-1]] if len(cur[-1]) < 800 and len(cur[-1]) + len(u) + 2 <= limit else []
        cur.append(u)
    if cur:
        parts.append("\n\n".join(cur))
    return parts

def _substance(body: str) -> int:
    """Characters of real content once NA placeholders, bold labels and table skeleton go."""
    t = re.sub(r"(?m)^\s*\*\*[^*]+\*\*\s*$", "", body)          # **Positives** style labels
    t = re.sub(r"(?m)^\s*\|?[\s|:-]*\|[\s|:-]*$", "", t)           # |---|---| separators
    t = re.sub(r"\bN/?A\b|\(NA if[^)]*\)|[|•*\-]", " ", t)
    t = re.sub(r"(?i)target price|rating|upside/downside|key thesis|metric|period|value|yoy change", " ", t)
    return len(re.sub(r"\s+", " ", t).strip())

def _jl(s):
    try:
        return json.loads(s) if isinstance(s, str) else []
    except Exception:
        return []

def _date(s: str) -> tuple[str, int]:
    s = str(s or "").strip()[:10]
    return (s, int(s.replace("-", ""))) if re.match(r"^\d{4}-\d{2}-\d{2}$", s) else ("", 0)

def build_chunks(ri: pd.DataFrame, tax: Taxonomy, cos: Companies) -> pd.DataFrame:
    rows, seen_docs = [], set()
    for r in ri.itertuples():
        md = str(getattr(r, "summary_md", "") or "")
        dh = str(getattr(r, "doc_hash", "") or "")
        # research_n is a local counter and has collided (4 values reused, 2026-09-26: three
        # pairs of different docs + one doc indexed twice), so ids carry the content hash and
        # a doc_hash seen before is skipped.
        if len(md.strip()) < 50 or (dh and dh in seen_docs):
            continue
        seen_docs.add(dh)
        rn = int(getattr(r, "research_n", 0) or 0)
        src_raw = str(getattr(r, "source", "") or "")
        src = tax.lookup(src_raw, "source", "source_other")
        dtype = tax.lookup(str(getattr(r, "doc_type", "")), "doc_type", "other")
        ddate, dts = _date(getattr(r, "doc_date", ""))
        doc_isins = [i for i in _jl(getattr(r, "isins", "")) if str(i).startswith("IN")]
        comps = [c.get("name", "") for c in _jl(getattr(r, "companies", "")) if isinstance(c, dict)]
        head = (f"[research_{rn:04d} · {src_raw or 'NA'} · {ddate or 'undated'} · {dtype}]"
                + (f" Companies: {', '.join(comps[:12])}" if comps else ""))
        base = dict(research_n=rn, doc_hash=str(getattr(r, "doc_hash", "")), source=src, source_raw=src_raw,
                    doc_type=dtype, doc_date=ddate, doc_ts=dts, file_name=str(getattr(r, "file_name", "")),
                    daily_md_ref=str(getattr(r, "daily_md_ref", "")), doc_isins="|".join(doc_isins))
        secs = split_sections(md)
        pieces = []                                              # (chunk_type, company_header, text)
        if secs.get("B") or secs.get("C"):
            if secs.get("B"):
                pieces.append(("doc_summary", "", secs["B"]))
            c = secs.get("C", "")
            blocks = list(_CO_RX.finditer(c))
            if blocks:
                for i, m in enumerate(blocks):
                    end = blocks[i + 1].start() if i + 1 < len(blocks) else len(c)
                    body = c[m.end():end].strip()
                    if _substance(body) >= 80:
                        pieces.append(("company_card", m.group(1), body))
            elif c.strip():
                pieces.append(("company_card" if len(doc_isins) == 1 else "text", "", c))
            for letter, kind in (("D", "themes"), ("E", "flags")):
                if len(secs.get(letter, "").strip()) > 40:
                    pieces.append((kind, "", secs[letter]))
        else:
            pieces.append(("text", "", re.sub(r"```json.*?```", "", md, flags=re.S)))
        seq = 0
        for kind, cohead, text in pieces:
            name, ticker, isin = cos.resolve(cohead, doc_isins) if cohead else ("", "", "")
            if kind == "company_card" and not cohead and len(doc_isins) == 1:
                isin, name = doc_isins[0], cos.label.get(doc_isins[0], "")
            for part in split_long(text):
                seq += 1
                label = f"{head}\nCompany: {name} ({ticker or isin or 'unresolved'})" if name else head
                rows.append(dict(base, chunk_id=f"r{rn:05d}_{dh[:8]}_{kind}_{seq:02d}", chunk_type=kind, seq=seq,
                                 company=name, ticker=ticker, isin=isin,
                                 page_content=f"{label}\n\n{part}"))
    return pd.DataFrame(rows)

def tag_chunks(ch: pd.DataFrame, tax: Taxonomy, cos: Companies) -> pd.DataFrame:
    found = [tax.scan(t.split("\n\n", 1)[-1]) for t in ch.page_content]   # body only
    df = collections.Counter(k for f in found for k in f)
    generic = {k for k, n in df.items() if n > GENERIC_DF * len(ch)}
    cols = collections.defaultdict(list)
    for f, isin in zip(found, ch["isin"]):
        slugs = {s for k in f - generic for s in tax.phrases[k]}
        comp_sec = cos.sector.get(isin, ("", "", "")) if isin else ("", "", "")
        slugs |= {s for s in comp_sec if s}
        full = set()
        for s in slugs:
            full.update(tax.ancestors(s))
        by = collections.defaultdict(list)
        for s in sorted(full):
            by[tax.ttype.get(s, "")].append(s)
        sectors = by["sector_group"] + by["sector"] + by["subsector"]
        cols["sector_group"].append(comp_sec[0] or (by["sector_group"][0] if len(by["sector_group"]) == 1 else ""))
        cols["sector"].append(comp_sec[1] or (by["sector"][0] if len(by["sector"]) == 1 else ""))
        cols["industry"].append(comp_sec[2] or (by["subsector"][0] if len(by["subsector"]) == 1 else ""))
        cols["tag_sectors"].append("|" + "|".join(sectors) + "|" if sectors else "")
        cols["tag_themes"].append("|" + "|".join(by["theme"]) + "|" if by["theme"] else "")
        cols["tag_macro"].append("|" + "|".join(by["macro_factor"]) + "|" if by["macro_factor"] else "")
        cols["tag_policies"].append("|" + "|".join(by["policy"]) + "|" if by["policy"] else "")
    for k, v in cols.items():
        ch[k] = v
    ch["text_sha"] = [hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] for t in ch.page_content]
    log(f"tagging: {len(generic)} phrases dropped as generic (in >{GENERIC_DF:.0%} of chunks): "
        + ", ".join(sorted(generic)[:12]) + ("…" if len(generic) > 12 else ""))
    return ch


# =========================================================================== embeddings
class EmbedPool:
    """Round-robin over API keys for embed_content. Reuses gemini_pool's error typing:
    dead / PerDay keys are dropped for the run, PerMinute keys cool down, overloads retry."""

    # Measured 2026-09-26 (one key, ~4.2k-token requests every 3 s): 10 went through, the
    # 11th got 429 PerMinute at ~31k real tokens inside 47 s -> ~30k tokens/min per key.
    # 27 of 34 configured keys were live (7 key_dead), each with its own budget.
    TPM_BUDGET = 26_000

    def __init__(self, keys: list[str]):
        from google import genai
        self._genai = genai
        self.keys = list(keys)
        self.clients = {}
        self.cool = {}                      # key index -> monotonic time it may be used again
        self.dead = set()
        self.window = collections.defaultdict(collections.deque)   # key -> deque[(t, tokens)]
        self.i = 0
        self.calls = 0

    def _room(self, k: int, need: int, now: float) -> float:
        """0 if key k can take `need` tokens now, else seconds until it can."""
        w = self.window[k]
        while w and now - w[0][0] >= 60:
            w.popleft()
        used = sum(t for _, t in w)
        if used + need <= self.TPM_BUDGET:
            return 0.0
        return 60 - (now - w[0][0]) + 0.5 if w else 0.0

    def _client(self, k):
        if k not in self.clients:
            self.clients[k] = self._genai.Client(api_key=self.keys[k])
        return self.clients[k]

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        from google.genai import types
        from gemini_pool import classify_error, KEY_DEAD, PERDAY, PERMIN, FATAL, AllBucketsExhausted, FatalCallError
        cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM)
        need = sum(est_tokens(t) for t in texts)
        attempts = 0
        while True:
            live = [k for k in range(len(self.keys)) if k not in self.dead]
            if not live:
                raise AllBucketsExhausted("every embedding key is dead or out of daily quota")
            now = time.monotonic()
            waits = {k: max(self.cool.get(k, 0) - now, self._room(k, need, now)) for k in live}
            ready = [k for k in live if waits[k] <= 0]
            if not ready:
                time.sleep(max(0.5, min(waits.values())))
                continue
            k = ready[self.i % len(ready)]
            self.i += 1
            try:
                self.window[k].append((time.monotonic(), need))
                r = self._client(k).models.embed_content(model=EMBED_MODEL, contents=texts, config=cfg)
                self.calls += 1
                return [e.values for e in r.embeddings]
            except Exception as e:
                kind, retry = classify_error(e)
                attempts += 1
                if kind in (KEY_DEAD, PERDAY):
                    self.dead.add(k)
                    log(f"  key#{k + 1}: {kind} — dropped for this run")
                elif kind == PERMIN:
                    self.cool[k] = time.monotonic() + max(retry, 60)
                    self.window[k].clear()
                elif kind == FATAL:
                    raise FatalCallError(str(e)[:300])
                else:
                    self.cool[k] = time.monotonic() + max(retry, 3)
                if attempts > 8 * len(self.keys):
                    raise AllBucketsExhausted(f"embedding keeps failing: {str(e)[:120]}")

def _keys():
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    from gemini_pool import load_keys_multi
    return load_keys_multi(os.environ, KEY_PREFIXES)


# =========================================================================== stores
STORE_SCRIPT = SCRIPTS_DIR / "research_rag_store.py"
EXPORT_DIR = LOCAL / "chroma_export"

def _store(cmd: str, payload: dict | None = None) -> object:
    """Run a Chroma command in its own process: chromadb 1.5.9 crashes (access violation)
    when pyarrow is loaded alongside it on Windows — see research_rag_store.py."""
    import subprocess
    r = subprocess.run([sys.executable, str(STORE_SCRIPT), cmd], input=json.dumps(payload) if payload else None,
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"research_rag_store {cmd} failed (exit {r.returncode}): {r.stderr[-800:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])

META_COLS = ["research_n", "chunk_type", "company", "ticker", "isin", "source", "source_raw", "doc_type",
             "doc_date", "doc_ts", "sector_group", "sector", "industry", "tag_sectors", "tag_themes",
             "tag_macro", "tag_policies", "file_name", "daily_md_ref"]

def _meta(row) -> dict:
    out = {}
    for c in META_COLS:
        v = row[c]
        out[c] = int(v) if c in ("research_n", "doc_ts") else ("" if v is None else str(v))
    return out

def upsert_chroma(ch: pd.DataFrame, emb: pd.DataFrame) -> dict:
    """Export vectors + LangChain metadata to plain files, then load them into Chroma in the
    pyarrow-free store process (it skips ids already present)."""
    # vectors are content-addressed (text_sha): identical text -> identical vector, any chunk id
    m = ch.merge(emb.drop_duplicates("text_sha")[["text_sha", "vector"]], on="text_sha")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EXPORT_DIR / "vectors.npy", np.stack(m.vector.to_numpy()).astype(np.float32))
    with open(EXPORT_DIR / "rows.jsonl", "w", encoding="utf-8") as f:
        for _, r in m.iterrows():
            f.write(json.dumps({"id": r.chunk_id, "text": r.page_content, "meta": _meta(r)}, ensure_ascii=False) + "\n")
    return _store("load")

def build_fts(ch: pd.DataFrame) -> None:
    LOCAL.mkdir(exist_ok=True)
    con = sqlite3.connect(FTS_DB)
    con.execute("DROP TABLE IF EXISTS fts")
    con.execute("CREATE VIRTUAL TABLE fts USING fts5(chunk_id UNINDEXED, body, company, tags, "
                "tokenize='porter unicode61')")
    con.executemany("INSERT INTO fts VALUES (?,?,?,?)",
                    [(r.chunk_id, r.page_content, f"{r.company} {r.ticker}",
                      " ".join(filter(None, (r.tag_sectors, r.tag_themes, r.tag_macro, r.tag_policies))).replace("|", " "))
                     for r in ch.itertuples()])
    con.commit(); con.close()


# =========================================================================== search + RAG
_STOP = set("a an and are as at be by for from has have in is it of on or that the this to was were what which who "
            "with about how does did do say says said saying tell me their there these those on why when".split())

# words that describe WHO is talking in a question, not what it is about
_QUERY_NOISE = {"broker", "brokerage", "analyst", "research", "view", "outlook", "report", "note"}

def _fts_query(q: str) -> str:
    toks = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if t not in _STOP and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in toks)

def understand(q: str, tax: Taxonomy, cos: Companies) -> dict:
    """Auto-tag the question: taxonomy slugs + company ISINs mentioned in it."""
    keys = {k for k in tax.scan(q) if k not in _QUERY_NOISE}
    slugs = {s for k in keys for s in tax.phrases[k]}
    isins, expand = set(), []
    toks = re.findall(r"[A-Za-z0-9&]+", q)
    for t in toks:                                              # EMS / CGD / PLI / NBFC ...
        if t.isupper() and 2 <= len(t) <= 5 and match_key(t) in tax.acronyms:
            acr = tax.acronyms[match_key(t)]
            slugs |= acr
            expand += [tax.display.get(s, s) for s in sorted(acr)]
    for n in (4, 3, 2, 1):
        for i in range(len(toks) - n + 1):
            k = _cname(" ".join(toks[i:i + n]))
            # an upper-case word that is also a taxonomy term ("EMS", "IT", "CGD") is the
            # topic, not a ticker — EMS Ltd (water EPC) must not hijack an EMS-sector query
            if n == 1 and toks[i].isupper() and toks[i] in cos.sym \
                    and match_key(toks[i]) not in tax.phrases and match_key(toks[i]) not in tax.acronyms:
                isins.add(cos.sym[toks[i]])
            elif n > 1 and len(k.split()) >= 2 and k in cos.name:  # "EMS company" cleans to "ems": skip
                isins.add(cos.name[k])
    return {"slugs": slugs, "isins": isins, "expand": expand}

def search(q: str, k: int = 10, isin: str = "", sector: str = "", theme: str = "", doc_type: str = "",
           since: str = "", pool: EmbedPool | None = None, tax=None, cos=None, verbose=True):
    ch = pd.read_parquet(LOCAL_CHUNKS).set_index("chunk_id", drop=False)
    qa = understand(q, tax, cos) if tax is not None else {"slugs": set(), "isins": set(), "expand": []}
    q_full = q + (" (" + "; ".join(qa["expand"]) + ")" if qa["expand"] else "")   # EMS -> + Electronics Manufacturing
    where = [{"doc_ts": {"$gte": int(since.replace("-", ""))}}] if since else []
    if isin:
        where.append({"isin": isin})
    if doc_type:
        where.append({"doc_type": doc_type})
    flt = where[0] if len(where) == 1 else ({"$and": where} if where else None)
    # vector leg (LangChain + Chroma, in the store process)
    qv = pool.embed([q_full], task="RETRIEVAL_QUERY")[0]
    ids_v = [h["id"] for h in _store("query", {"vector": list(map(float, qv)), "k": 120, "filter": flt})]
    # keyword leg (SQLite FTS5 / BM25)
    ids_k = []
    fq = _fts_query(q_full)
    if fq:
        con = sqlite3.connect(FTS_DB)
        ids_k = [r[0] for r in con.execute("SELECT chunk_id FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT 120", (fq,))]
        con.close()
    # reciprocal-rank fusion + boosts from the question's own tags / companies
    score = collections.defaultdict(float)
    for rank, cid in enumerate(ids_v):
        if cid:
            score[cid] += 1.0 / (60 + rank)
    for rank, cid in enumerate(ids_k):
        score[cid] += 1.0 / (60 + rank)
    out = []
    for cid, s in score.items():
        if cid not in ch.index:
            continue
        r = ch.loc[cid]
        if isin and r["isin"] != isin: continue
        if doc_type and r.doc_type != doc_type: continue
        if since and r.doc_ts < int(since.replace("-", "")): continue
        if sector and f"|{sector}|" not in r.tag_sectors: continue
        if theme and f"|{theme}|" not in r.tag_themes: continue
        tags = set(filter(None, "|".join([r.tag_sectors, r.tag_themes, r.tag_macro, r.tag_policies]).split("|")))
        boost = 0.004 * len(tags & qa["slugs"]) + (0.02 if r["isin"] and r["isin"] in qa["isins"] else 0)
        out.append((s + boost, cid))
    out.sort(reverse=True)
    # at most PER_DOC chunks from one document, so one long report (e.g. KPMG's 9 masked EMS
    # players) cannot fill the whole answer context — unless the user filtered to one company
    per_doc, picked = collections.Counter(), []
    for s, cid in out:
        rn = ch.loc[cid, "research_n"]
        if per_doc[rn] >= PER_DOC and not isin:
            continue
        per_doc[rn] += 1
        picked.append((s, cid))
        if len(picked) == k:
            break
    hits = [ch.loc[cid].to_dict() | {"score": round(s, 4), "in_vector": cid in ids_v, "in_keyword": cid in ids_k}
            for s, cid in picked]
    return hits, qa

def answer(q: str, hits: list[dict]) -> tuple[str, str]:
    from gemini_pool import BucketPool
    from model_registry import resolve
    ctx = "\n\n".join(f"<<C{i + 1}>> research_{h['research_n']:04d} | {h['source_raw'] or h['source']} | "
                      f"{h['doc_date'] or 'undated'} | {h['doc_type']}"
                      + (f" | {h['company']}" if h['company'] else "") + f"\n{h['page_content'][:5000]}"
                      for i, h in enumerate(hits))
    prompt = ("You are an equity research assistant for Indian markets. Answer the QUESTION using ONLY the "
              "numbered research extracts below. Every factual sentence must end with its citation like "
              "[C3]. Quote numbers exactly as written. If extracts disagree, say so and cite both. If the "
              "extracts do not answer the question, say what is missing. Label broker opinions (ratings, "
              "targets) as the named broker's view. Keep it under 350 words, bullets where helpful.\n\n"
              f"QUESTION: {q}\n\nEXTRACTS:\n{ctx}")
    pool = BucketPool(_keys(), resolve("QUALITY"), inter_call_s=2.0)
    return pool.call_text(prompt, max_output_tokens=8192)   # thinking tokens count against this


# =========================================================================== commands
def cmd_build(a):
    ri, al, tx, uni, cl = load_inputs(a.cache)
    tax, cos = Taxonomy(al, tx), None
    cos = Companies(uni, cl, tax)
    log(f"inputs: {len(ri):,} research docs · {len(al):,} aliases · {len(uni):,} universe rows · {len(cl):,} classified")
    ch = build_chunks(ri, tax, cos)
    ch = tag_chunks(ch, tax, cos)
    cards = ch[ch.chunk_type == "company_card"]
    log(f"chunks: {len(ch):,} " + str(ch.chunk_type.value_counts().to_dict()))
    log(f"company cards resolved to an ISIN: {(cards['isin'] != '').sum():,}/{len(cards):,} "
        f"({(cards['isin'] != '').mean():.0%}) · distinct companies {cards['isin'][cards['isin'] != ''].nunique():,}")
    log(f"chunks with >=1 sector tag {(ch.tag_sectors != '').mean():.0%} · theme {(ch.tag_themes != '').mean():.0%} "
        f"· macro {(ch.tag_macro != '').mean():.0%} · policy {(ch.tag_policies != '').mean():.0%}")
    log(f"chunk chars: median {int(ch.page_content.str.len().median()):,} · max {ch.page_content.str.len().max():,} "
        f"· total {ch.page_content.str.len().sum():,} (~{ch.page_content.str.len().sum() // 4:,} tokens to embed)")
    if a.dry_run:
        ex = ch[ch.chunk_type == "company_card"].iloc[0]
        print("\nexample company_card:\n", ex.page_content[:700], "\n", {c: ex[c] for c in META_COLS[:14]})
        log("dry-run: nothing written"); return
    LOCAL.mkdir(exist_ok=True)
    ch.to_parquet(LOCAL_CHUNKS, index=False)
    build_fts(ch)
    log(f"wrote {LOCAL_CHUNKS} + keyword index {FTS_DB}")
    if not a.no_upload:
        svc, root = _drive()
        buf = io.BytesIO(); ch.to_parquet(buf, index=False)
        _ul(svc, root, f"{DRIVE_RESEARCH}/chunks.parquet", buf.getvalue())
        log(f"uploaded {DRIVE_RESEARCH}/chunks.parquet")

def cmd_embed(a):
    ch = pd.read_parquet(LOCAL_CHUNKS)
    emb = pd.read_parquet(LOCAL_EMB) if LOCAL_EMB.exists() else pd.DataFrame(columns=["chunk_id", "text_sha", "model", "vector"])
    done = set(emb.text_sha)                                    # content-addressed: ids may change
    todo = ch[~ch.text_sha.isin(done)].drop_duplicates("text_sha")
    if a.limit:
        todo = todo.head(a.limit)
    log(f"embed: {len(ch):,} chunks · {len(done):,} already embedded · {len(todo):,} to do "
        f"(~{sum(est_tokens(t) for t in todo.page_content):,} tokens)")
    if a.dry_run:
        log("dry-run: no Gemini calls"); return
    pool = EmbedPool(_keys())
    deadline = time.monotonic() + a.deadline_min * 60 if a.deadline_min else None
    new = []
    from gemini_pool import AllBucketsExhausted, FatalCallError

    def flush():
        nonlocal emb, new
        if new:
            emb = pd.concat([emb, pd.DataFrame(new)], ignore_index=True)
            emb.to_parquet(LOCAL_EMB, index=False)
            new = []

    # token-packed batches: <= EMBED_BATCH texts and <= BATCH_TOKENS estimated tokens each
    batches, cur, tok = [], [], 0
    for idx, t in enumerate(todo.page_content):
        n = est_tokens(t[:EMBED_CHARS])
        if cur and (len(cur) >= EMBED_BATCH or tok + n > BATCH_TOKENS):
            batches.append(cur); cur, tok = [], 0
        cur.append(idx); tok += n
    if cur:
        batches.append(cur)
    log(f"  {len(batches)} token-packed batches (<= {BATCH_TOKENS:,} tokens each)")
    stop, doneN = "", 0
    for bi, rows in enumerate(batches):
        if deadline and time.monotonic() > deadline:
            stop = "deadline"; break
        b = todo.iloc[rows]
        try:
            vecs = pool.embed([t[:EMBED_CHARS] for t in b.page_content])
        except AllBucketsExhausted as e:
            stop = f"quota: {e}"; break
        except FatalCallError as e:
            log(f"  batch {bi} fatal ({str(e)[:100]}) — skipped"); continue
        new += [dict(chunk_id=c, text_sha=s, model=EMBED_MODEL, vector=np.asarray(v, dtype=np.float32))
                for c, s, v in zip(b.chunk_id, b.text_sha, vecs)]
        doneN += len(b)
        if bi % 20 == 19:
            flush()
            log(f"  {doneN:,}/{len(todo):,} embedded · {pool.calls} calls · live keys {len(pool.keys) - len(pool.dead)}")
    flush()
    log(f"embeddings stored: {len(emb):,}/{len(ch):,}" + (f" — stopped early ({stop}); rerun to resume" if stop else ""))
    res = upsert_chroma(ch, emb)
    log(f"Chroma collection '{COLLECTION}': +{res['added']:,} vectors (total {res['total']:,}) at {CHROMA_DIR}")
    if not a.no_upload:
        svc, root = _drive()
        buf = io.BytesIO(); emb.to_parquet(buf, index=False)
        _ul(svc, root, f"{DRIVE_RESEARCH}/embeddings.parquet", buf.getvalue())
        log(f"uploaded {DRIVE_RESEARCH}/embeddings.parquet")

def _ctx(a):
    ri, al, tx, uni, cl = load_inputs(a.cache) if a.cache else (None,) + tuple(_light_inputs())
    tax = Taxonomy(al, tx)
    return tax, Companies(uni, cl, tax)

def _light_inputs():
    """search/ask need taxonomy + universe + classification only (no research_index)."""
    al = pd.read_csv(SCRIPTS_DIR / "tag_aliases.csv", keep_default_na=False)
    tx = pd.read_csv(SCRIPTS_DIR / "tag_taxonomy.csv", keep_default_na=False)
    uni_p, cl_p = LOCAL / "company_universe.csv", LOCAL / "classification.csv"
    if not (uni_p.exists() and cl_p.exists()):
        svc, root = _drive()
        import build_classification as bcl
        uni_p.write_bytes(_dl(svc, root, f"{DRIVE_INDEX}/company_universe.csv"))
        bcl.load_classification(svc, root).to_csv(cl_p, index=False)
    return al, tx, pd.read_csv(uni_p, keep_default_na=False, dtype=str), pd.read_csv(cl_p, keep_default_na=False)

def _print_hits(q, hits, qa, tax):
    tag_s = ", ".join(sorted(tax.display.get(s, s) for s in qa["slugs"])) or "—"
    print(f"\nQUERY: {q}\n  understood → tags: {tag_s} · companies: {', '.join(sorted(qa['isins'])) or '—'}\n")
    for i, h in enumerate(hits, 1):
        legs = "+".join(x for x, f in (("vector", h["in_vector"]), ("keyword", h["in_keyword"])) if f)
        body = re.sub(r"\s+", " ", h["page_content"].split("\n\n", 1)[-1])[:260]
        print(f"{i:>2}. [{h['chunk_type']}] research_{h['research_n']:04d} · {h['source_raw'] or h['source']} · "
              f"{h['doc_date'] or 'undated'}" + (f" · {h['company']} ({h['isin'] or '—'})" if h['company'] else "")
              + f"   score {h['score']} ({legs})")
        own = [s for s in (h['industry'], h['sector']) if s]    # the company's own sector first
        spec = [s for s in (h['tag_sectors'] + h['tag_themes']).split("|")
                if s and tax.ttype.get(s) in ("subsector", "theme") and s not in own]
        tags = [tax.display.get(s, s) for s in own + spec][:6]
        if tags:
            print(f"    tags: {', '.join(tags)}")
        print(f"    {body}…")

def cmd_search(a):
    tax, cos = _ctx(a)
    if a.dry_run:
        qa = understand(a.query, tax, cos)
        print("dry-run: query understood as", {k: sorted(v) for k, v in qa.items()}, "— no embedding call"); return
    hits, qa = search(a.query, a.k, a.isin, a.sector, a.theme, a.doc_type, a.since, EmbedPool(_keys()), tax, cos)
    _print_hits(a.query, hits, qa, tax)

def cmd_ask(a):
    tax, cos = _ctx(a)
    hits, qa = search(a.query, a.k, a.isin, a.sector, a.theme, a.doc_type, a.since, EmbedPool(_keys()), tax, cos)
    _print_hits(a.query, hits, qa, tax)
    if a.dry_run:
        print("\ndry-run: retrieval only, no answer call"); return
    text, model = answer(a.query, hits)
    print(f"\nANSWER ({model}):\n{text}\n\nSOURCES:")
    for i, h in enumerate(hits, 1):
        print(f"  [C{i}] research_{h['research_n']:04d} · {h['source_raw'] or h['source']} · {h['doc_date'] or 'undated'}"
              + (f" · {h['company']}" if h['company'] else "") + f" · {h['file_name']}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "embed", "search", "ask"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--cache", help="dir with research_index.parquet, classification.csv, company_universe.csv")
        if name in ("build", "embed"):
            p.add_argument("--no-upload", action="store_true", help="keep outputs local (skip Drive write)")
        if name == "embed":
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--deadline-min", type=float, default=0)
        if name in ("search", "ask"):
            p.add_argument("query")
            p.add_argument("-k", type=int, default=10)
            p.add_argument("--isin", default=""); p.add_argument("--sector", default="")
            p.add_argument("--theme", default=""); p.add_argument("--doc-type", default="")
            p.add_argument("--since", default="", help="YYYY-MM-DD")
    a = ap.parse_args()
    {"build": cmd_build, "embed": cmd_embed, "search": cmd_search, "ask": cmd_ask}[a.cmd](a)

if __name__ == "__main__":
    main()
