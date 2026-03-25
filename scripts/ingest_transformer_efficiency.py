#!/usr/bin/env python3
"""
ingest_transformer_efficiency.py
---------------------------------
Wipes the chunk store and fills it with transformer efficiency papers.

Usage:
    python3 scripts/ingest_transformer_efficiency.py --dry_run
    python3 scripts/ingest_transformer_efficiency.py
"""

from __future__ import annotations
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_pipeline.omni_retriever import OmniRetriever
from data_pipeline.filter import SafetyFilter
from knowledge_base.chunk_store import ChunkStore
from knowledge_base.database import KnowledgeBase

CHUNK_SIZE = 900
CHUNK_OVERLAP = 140
DOCS_PER_QUERY = 20

# ─────────────────────────────────────────────
# Queries — specific enough to get real papers
# ─────────────────────────────────────────────
QUERIES = [
    # Core attention efficiency
    "FlashAttention memory efficient exact attention transformer",
    "sparse attention mechanism long sequence transformer",
    "linear attention efficient transformer approximation",
    "sliding window attention long context language model",

    # MoE
    "mixture of experts language model scaling efficiency",
    "sparse mixture experts routing transformer training",
    "MoE expert load balancing large language model",

    # KV Cache
    "KV cache optimization large language model inference",
    "paged attention memory management LLM serving",
    "KV cache compression eviction long context LLM",
    "grouped query attention multi query attention inference",

    # Quantization
    "post-training quantization large language model weights",
    "activation-aware weight quantization LLM inference",
    "int8 int4 quantization transformer inference efficiency",

    # LoRA / PEFT
    "low-rank adaptation parameter efficient fine-tuning language model",
    "parameter efficient fine-tuning large language model",

    # Pruning / Distillation
    "structured pruning large language model compression",
    "knowledge distillation language model compression efficiency",
    "sparse pruning transformer weights inference",

    # Inference systems
    "speculative decoding language model inference speedup",
    "continuous batching LLM serving throughput latency",
    "efficient LLM inference serving deployment",
    "tensor parallelism pipeline parallelism LLM inference",

    # Long context / position
    "rotary position embedding transformer length generalization",
    "long context language model attention efficiency benchmark",
    "ring attention sequence parallelism distributed training",

    # State space models (alternatives to attention)
    "Mamba state space model efficient sequence modeling",
    "RWKV recurrent language model linear complexity",
    "linear recurrent model transformer alternative efficiency",
]

# ─────────────────────────────────────────────
# Relevance gate
# ─────────────────────────────────────────────

# Must match at least ONE of these
MUST_MATCH = [
    "flashattention", "flash attention",
    "sparse attention",
    "mixture of experts", " moe ",
    "kv cache", "key-value cache",
    "paged attention", "pagedattention",
    "speculative decoding",
    "multi-query attention", "grouped query attention",
    "grouped-query attention", "multi query attention",
    "linear attention",
    "state space model", "selective state space",
    "rwkv",
    "lora ", "low-rank adaptation",
    "llm inference", "large language model inference",
    "transformer inference efficiency",
    "inference efficiency",
    "rotary position embedding", "rope embedding",
    "continuous batching",
    "token pruning",
    "llm quantization", "quantization llm",
    "transformer quantization",
    "post-training quantization",
    "weight quantization llm",
    "vllm",
    "ring attention",
    "sliding window attention",
    "expert routing",
    "knowledge distillation llm", "distillation language model",
    "llm compression", "language model compression",
    "parameter efficient fine-tuning",
    "pipeline parallelism llm", "tensor parallelism llm",
    "longformer", "bigbird",
    "mamba language", "mamba model",
    "efficient llm", "efficient language model",
    "transformer efficiency",
    "attention efficiency",
    "model compression transformer",
    "inference optimization",
]

# If any of these appear — reject regardless
MUST_NOT_MATCH = [
    # medicine / biology
    "alzheimer", "mri scan", "cryo-em",
    "single cell", "genomic", "proteomic",
    "medical diagnosis", "clinical trial",
    "drug discovery", "ecg signal",
    "eeg signal", "brain-computer",
    "cancer detection", "tumor",
    "bacteriophage", "hematopoietic",
    # physics
    "gravitational wave", "black hole merger",
    "quantum mechanics", "nuclear physics",
    "fluid dynamics", "thermodynamics",
    # unrelated ML
    "image segmentation", "object detection yolo",
    "3d mesh synthesis", "video diffusion",
    "music source separation",
    "automatic speech recognition",
    "person re-identification",
    "remote sensing satellite",
    "robotic arm manipulation",
    "hyperspectral imaging",
    # misc
    "fantasy landscape", "pembuatan lanskap",
    "geospatial disaster", "linkedin post",
    "data warehouse etl",
    "acoustic doa estimation",
    "point process sampling",
    "social media sentiment",
]

# Reject on title alone
TITLE_MUST_NOT = [
    "pembuatan", "lanskap",
    "alzheimer", "mri", "cryo-em",
    "eeg", "ecg", "genomic",
    "robotic manipulation",
    "music source",
    "hyperspectral",
    "linkedin",
    "geospatial",
    "disaster extraction",
]


def is_relevant(title: str, text: str) -> bool:
    t = title.lower()
    combined = t + " " + text.lower()

    for bad in TITLE_MUST_NOT:
        if bad in t:
            return False

    for bad in MUST_NOT_MATCH:
        if bad in combined:
            return False

    for good in MUST_MATCH:
        if good in combined:
            return True

    return False


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def chunk_text(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= CHUNK_SIZE:
        return [t]
    out = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for i in range(0, len(t), step):
        c = t[i: i + CHUNK_SIZE].strip()
        if len(c) > 120:
            out.append(c)
        if i + CHUNK_SIZE >= len(t):
            break
    return out


def to_chunks(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "text": p,
            "paper_title": doc.get("title", "unknown"),
            "source": doc.get("source", "unknown"),
            "meta_json": json.dumps({
                "url": doc.get("url", ""),
                "timestamp": doc.get("timestamp", ""),
                "domain": "transformer_efficiency",
            }),
        }
        for p in chunk_text(doc.get("text", ""))
    ]


def wipe_database(chunk_store: ChunkStore, kb: KnowledgeBase):
    print("Wiping chunk store...")
    chunk_store.cur.execute("DELETE FROM chunks")
    chunk_store.conn.commit()
    try:
        chunk_store.cur.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")
        chunk_store.conn.commit()
    except Exception:
        pass
    print(f"  Wiped. Count: {chunk_store.count()}")
    try:
        kb.conn.execute("DELETE FROM knowledge")
        kb.conn.commit()
        print("  KnowledgeBase wiped.")
    except Exception as e:
        print(f"  KnowledgeBase wipe skipped: {e}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_wipe", action="store_true")
    ap.add_argument("--limit", type=int, default=DOCS_PER_QUERY)
    args = ap.parse_args()

    retriever = OmniRetriever()
    safety = SafetyFilter()
    chunk_store = ChunkStore()
    kb = KnowledgeBase()

    if not args.skip_wipe and not args.dry_run:
        wipe_database(chunk_store, kb)

    seen: set = set()
    accepted: List[str] = []
    total_fetched = 0
    total_chunks_added = 0
    total_chunks_skipped = 0

    for i, query in enumerate(QUERIES):
        print(f"\n[{i+1}/{len(QUERIES)}] {query}")

        docs = retriever.retrieve_universal(query)[: args.limit]
        safe_docs = safety.filter(docs)
        total_fetched += len(safe_docs)

        relevant = []
        for doc in safe_docs:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            key = title.lower()

            if key in seen:
                continue
            seen.add(key)

            if is_relevant(title, text):
                relevant.append(doc)
                accepted.append(title)
                if args.dry_run:
                    print(f"  ACCEPT: {title[:80]}")
            else:
                if args.dry_run:
                    print(f"  REJECT: {title[:80]}")

        print(f"  Accepted: {len(relevant)}/{len(safe_docs)}")

        if not args.dry_run and relevant:
            kb.store(relevant)
            chunks = []
            for doc in relevant:
                chunks.extend(to_chunks(doc))
            if chunks:
                r = chunk_store.upsert_chunks(chunks)
                total_chunks_added += r["added"]
                total_chunks_skipped += r["skipped"]
                print(f"  Chunks: +{r['added']}")

        time.sleep(1.2)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Queries run    : {len(QUERIES)}")
    print(f"Docs fetched   : {total_fetched}")
    print(f"Docs accepted  : {len(accepted)}")
    print(f"Chunks added   : {total_chunks_added}")
    print(f"Total in store : {chunk_store.count()}")

    if args.dry_run:
        print(f"\nAccepted ({len(accepted)}):")
        for t in accepted:
            print(f"  * {t[:90]}")
        print("\nLooks good? Run without --dry_run.")
    elif chunk_store.count() < 50:
        print("\nWARNING: low chunk count. Check API connectivity.")
    else:
        print("\nReady. Next step:")
        print("  python3 scripts/ingest_chunks.py --rebuild_index")


if __name__ == "__main__":
    main()