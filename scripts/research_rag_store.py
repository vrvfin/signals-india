r"""
research_rag_store.py — the Chroma (LangChain) side of research_rag.py, in its OWN process.

WHY A SEPARATE PROCESS. Measured 2026-09-26 on this PC: chromadb 1.5.9's Rust engine dies
with "Windows fatal exception: access violation" on upsert/count whenever pyarrow 17 is loaded
in the same process (import order irrelevant; numpy, google-genai, dotenv are fine). Every
pipeline here depends on pyarrow for parquet, so upgrading it is a separate decision. This
module therefore imports NO pandas / pyarrow: it reads plain .npy + .jsonl written by
research_rag.py and is called as a subprocess.

    python scripts/research_rag_store.py load   [--export DIR]      # upsert exported vectors
    python scripts/research_rag_store.py query  < {"vector": [...], "k": 120, "filter": {...}}
    python scripts/research_rag_store.py count
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "_rag" / "chroma"
EXPORT_DIR = ROOT_DIR / "_rag" / "chroma_export"
COLLECTION = "research_chunks"


def store():
    """LangChain Chroma vector store over the persisted collection (vectors supplied by us)."""
    from langchain_chroma import Chroma
    return Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR),
                  collection_metadata={"hnsw:space": "cosine"})

def cmd_load(export: Path) -> None:
    vecs = np.load(export / "vectors.npy")
    rows = [json.loads(l) for l in open(export / "rows.jsonl", encoding="utf-8")]
    assert len(rows) == len(vecs), f"{len(rows)} rows vs {len(vecs)} vectors"
    col = store()._collection
    have = set(col.get(include=[])["ids"]) if col.count() else set()
    todo = [i for i, r in enumerate(rows) if r["id"] not in have]
    for s in range(0, len(todo), 500):
        b = todo[s:s + 500]
        col.upsert(ids=[rows[i]["id"] for i in b], embeddings=vecs[b].tolist(),
                   documents=[rows[i]["text"] for i in b], metadatas=[rows[i]["meta"] for i in b])
    print(json.dumps({"added": len(todo), "total": col.count()}))

def cmd_query() -> None:
    q = json.load(sys.stdin)
    res = store().similarity_search_by_vector_with_relevance_scores(q["vector"], k=q.get("k", 50),
                                                                     filter=q.get("filter") or None)
    print(json.dumps([{"id": d.id, "score": float(s)} for d, s in res]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load", "query", "count"])
    ap.add_argument("--export", default=str(EXPORT_DIR))
    a = ap.parse_args()
    if a.cmd == "load":
        cmd_load(Path(a.export))
    elif a.cmd == "query":
        cmd_query()
    else:
        print(json.dumps({"total": store()._collection.count()}))

if __name__ == "__main__":
    main()
