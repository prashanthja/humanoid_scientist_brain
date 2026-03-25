import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#!/usr/bin/env python3
import argparse, json, re, os
from typing import List, Dict, Any

from data_pipeline.omni_retriever import OmniRetriever
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from knowledge_base.chunk_store import ChunkStore
from learning_module.trainer_online import OnlineTrainer
from learning_module.embedding_bridge import EmbeddingBridge
from retrieval.chunk_index import ChunkIndex

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140

def extract_keywords(text: str, n=5) -> str:
    words = re.findall(r"\b[a-zA-Z]{5,}\b", (text or "").lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return ", ".join([w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]])

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= size:
        return [t]
    out = []
    step = max(1, size - overlap)
    for i in range(0, len(t), step):
        c = t[i:i+size].strip()
        if len(c) > 80:
            out.append(c)
        if i + size >= len(t):
            break
    return out

def to_enriched_items(docs: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
    enriched = []
    for d in docs:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        enriched.append({
            "text": text,
            "paper_title": d.get("title", f"Unknown ({topic})"),
            "concepts": extract_keywords(text),
            "source": d.get("source", "unknown"),
        })
    return enriched

def to_chunks(enriched_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for it in enriched_docs:
        parts = chunk_text(it.get("text", ""))
        for p in parts:
            chunks.append({
                "text": p,
                "paper_title": it.get("paper_title", "unknown"),
                "source": it.get("source", "unknown"),
                "meta_json": json.dumps({"concepts": it.get("concepts","")}),
            })
    return chunks

def rebuild_index(chunk_store: ChunkStore):
    trainer = OnlineTrainer()
    bridge = EmbeddingBridge(trainer)
    idx = ChunkIndex(
        chunk_store=chunk_store,
        encoder=bridge,
        cache_dir="data",
        max_items=8000,
        chunk_batch=32,
    )
    idx.rebuild()
    print(f"✅ Index rebuilt. items={len(idx.items)} dim={idx.dim}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", type=str, nargs="?", default=None,
                    help="Topic to fetch. Omit if using --rebuild_only.")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--rebuild_index", action="store_true",
                    help="Rebuild index after fetching.")
    ap.add_argument("--rebuild_only", action="store_true",
                    help="Skip fetching. Just rebuild index from existing chunks.")
    args = ap.parse_args()

    chunk_store = ChunkStore()

    # ── Rebuild only — no fetch ──────────────
    if args.rebuild_only:
        print(f"📦 Chunks in store: {chunk_store.count()}")
        if chunk_store.count() == 0:
            print("⚠️  No chunks in store. Run ingestion first.")
            return
        print("🔧 Rebuilding index from existing chunks...")
        rebuild_index(chunk_store)
        return

    # ── Normal fetch + optional rebuild ──────
    if not args.topic:
        ap.error("topic is required unless --rebuild_only is set.")

    omni = OmniRetriever()
    safety = SafetyFilter()
    kb = KnowledgeBase()

    print(f"🔎 Retrieving docs for: {args.topic}")
    docs = omni.retrieve_universal(args.topic)[: args.limit]
    safe_docs = safety.filter(docs)
    print(f"✅ Retrieved={len(docs)} safe={len(safe_docs)}")

    enriched = to_enriched_items(safe_docs, args.topic)
    if not enriched:
        print("⚠️ No enriched docs. Nothing to store.")
        if args.rebuild_index:
            print("🔧 Rebuilding index from existing chunks anyway...")
            rebuild_index(chunk_store)
        return

    kb.store(enriched)
    chunks = to_chunks(enriched)
    res = chunk_store.upsert_chunks(chunks)
    print(f"🧩 ChunkStore added={res['added']} skipped={res['skipped']}")
    print(f"📦 chunks_in_store={chunk_store.count()}  kb_count={kb.count()}")

    if args.rebuild_index:
        rebuild_index(chunk_store)

if __name__ == "__main__":
    main()