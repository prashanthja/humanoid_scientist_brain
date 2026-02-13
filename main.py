#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Humanoid Scientist Brain — Universal Continuous Learning (Chunk-RAG + Judge Engine)

Key upgrades:
- Stores *chunks* (not whole docs) for retrieval
- Builds an embedding index over chunks (ChunkIndex)
- Adds ProposalEvaluator: evaluate any proposed "discovery" claim with sanity + evidence
- Keeps your full loop: retrieve → filter → store → train → reflect → hypotheses → benchmark → dashboard

NOTE:
- This is still "v1 production architecture" — scalable + debuggable.
"""

import os, time, json, gc, re
from typing import Dict, Any, List
import argparse

# Enable automatic CPU fallback for MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print("⚙️ Enabled MPS CPU fallback for unsupported ops.")

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from data_pipeline.omni_retriever import OmniRetriever
from data_pipeline.filter import SafetyFilter

from knowledge_base.database import KnowledgeBase
from knowledge_base.chunk_store import ChunkStore

from learning_module.trainer_online import OnlineTrainer, get_device
from learning_module.embedding_bridge import EmbeddingBridge
from learning_module.training_memory import TrainingMemory

from reflection_module.reflection import ReflectionEngine

from reasoning_module.evidence_evaluator import EvidenceEvaluator
from reasoning_module.hypothesis_generator import HypothesisGenerator
from reasoning_module.hypothesis_validator import HypothesisValidator
from reasoning_module.hypothesis_evolver import HypothesisEvolver

from testing_module.model_evaluator import ModelEvaluator  # your retrieval benchmark evaluator

from reasoning_module.proposal_evaluator import ProposalEvaluator
from reasoning_module.claim_schema import SourceTrace

from retrieval.chunk_index import ChunkIndex


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DASHBOARD_STATE_PATH = "data/dashboard_state.json"
os.makedirs("data", exist_ok=True)
os.makedirs("visualization/graphs", exist_ok=True)

MAX_CYCLES = 5
SLEEP_INTERVAL = 5  # DEBUG: 5 seconds

# retrieval/indexing
MAX_DOCS_PER_TOPIC = 60          # safety valve
CHUNK_SIZE = 900                 # characters
CHUNK_OVERLAP = 140              # characters
MAX_CHUNKS_TO_INDEX = 8000       # bigger as you scale
CHUNK_INDEX_BATCH = 64

# training
TRAIN_MIN_NEW = 6                # don't train on tiny batches
KB_INDEX_ITEMS_FOR_EVIDENCE = 1200


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def write_dashboard_state(payload: Dict[str, Any]):
    with open(DASHBOARD_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("📊 Dashboard updated → data/dashboard_state.json")


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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    device = get_device()
    print(f"🤖 Starting Humanoid Scientist Brain (Universal Mode) on {device}")

    # Core subsystems
    omni = OmniRetriever()
    safety = SafetyFilter()

    kb = KnowledgeBase()
    chunk_store = ChunkStore()

    trainer = OnlineTrainer()
    memory = TrainingMemory()
    bridge = EmbeddingBridge(trainer)

    reflection = ReflectionEngine(omni, safety, kb, trainer)

    # Evidence evaluator (kept) — uses KB embeddings cache; set higher later
    evaluator = EvidenceEvaluator(
        kb,
        bridge,
        reflection.kg,
        max_index_items=KB_INDEX_ITEMS_FOR_EVIDENCE,
        top_k=10
    )

    hypgen = HypothesisGenerator(reflection.kg, bridge, kb)
    validator = HypothesisValidator(kb, bridge, reflection.kg)
    evolver = HypothesisEvolver(kb, reflection.kg)

    # Benchmark evaluator (your retrieval-grounded benchmark)
    model_tester = ModelEvaluator(trainer, kb, bridge)

    # NEW: proposal judge engine
    proposal_engine = ProposalEvaluator(
    kb, bridge,
    top_k=10,
    evidence_threshold=0.60,
    max_kb_items=3000,
    require_evidence=True,          # ✅ force grounded behavior
)
    # NEW: chunk index (real retrieval)
    chunk_index = ChunkIndex(
        chunk_store=chunk_store,
        encoder=bridge,
        cache_dir="data",
        max_items=MAX_CHUNKS_TO_INDEX,
        chunk_batch=CHUNK_INDEX_BATCH,
    )

    # Topic seed set
    topics = [
        "physics", "mathematics", "quantum mechanics", "relativity",
        "thermodynamics", "electromagnetism", "nuclear physics", "fluid dynamics",
        "astrophysics", "cosmology", "string theory", "chaos theory",
        "solid state physics", "particle physics", "plasma physics",
        "optics", "statistical mechanics", "quantum field theory",
        "algebra", "geometry", "topology", "calculus", "number theory",
        "probability", "differential equations", "tensor calculus", "set theory"
    ]

    cycle = 1
    while cycle <= MAX_CYCLES:
        print(f"\n🌀 === Universal Learning Cycle {cycle} ===\n")

        all_docs_for_topic_mining = []
        total_new_for_training = 0
        total_chunks_added = 0

        # ----------------------------
        # Retrieve → Filter → Store → Chunk → Train
        # ----------------------------
        for topic in topics:
            docs = omni.retrieve_universal(topic)[:MAX_DOCS_PER_TOPIC]
            safe_docs = safety.filter(docs)

            # Track only unseen by TrainingMemory (so we don't retrain same doc)
            new_docs = memory.filter_new(safe_docs)

            if not new_docs:
                print(f"⚙️ Skipping '{topic}' (no unseen docs).")
                continue

            enriched = to_enriched_items(new_docs, topic)

            # Store into KB ONCE (enriched form)
            kb.store(enriched)

            # Chunk them and store chunks (this powers retrieval)
            chunks = to_chunks(enriched)
            res = chunk_store.upsert_chunks(chunks)
            total_chunks_added += res["added"]
            print(f"🧩 ChunkStore: +{res['added']} chunks (skipped {res['skipped']}) for '{topic}'")

            # Train if enough new material
            if len(enriched) >= TRAIN_MIN_NEW:
                trainer.incremental_train(enriched)
                total_new_for_training += len(enriched)
            else:
                print(f"⚙️ Not training on '{topic}' (only {len(enriched)} new docs)")

            memory.mark_trained(enriched)
            all_docs_for_topic_mining.extend(enriched)

        # After ingest, rebuild chunk index (simple + safe)
        if total_chunks_added > 0:
            print("\n🔧 Rebuilding ChunkIndex (new chunks ingested)...")
            chunk_index.rebuild()
            print(f"✅ ChunkIndex ready: dim={chunk_index.dim}, items={len(chunk_index.vecs)}")
        else:
            print("\nℹ️ No new chunks ingested; keeping existing ChunkIndex.")

        # ----------------------------
        # Reflection: update KG
        # ----------------------------
        print("\n🧠 Reflecting & updating Knowledge Graph...")
        reflection.review_knowledge()

        # ----------------------------
        # Hypothesis generation + validation + evidence eval
        # ----------------------------
        hyps = hypgen.generate(top_n=20)
        if hyps:
            validated = validator.validate(hyps, cycle=cycle)
            evolver.update(validated, cycle=cycle)
            reflection.promote_validated(validated)

            evaluated = evaluator.evaluate_batch(validated)
            print("🧩 Evidence check done (sample):")
            for e in evaluated[:5]:
                print(f"  → {e.get('verdict')}: {str(e.get('hypothesis',''))[:80]}")

        # ----------------------------
        # Expand topic space dynamically
        # ----------------------------
        new_topics = omni.extract_new_topics(all_docs_for_topic_mining)
        for t in new_topics:
            if t not in topics and len(t) > 4:
                topics.append(t)
                print(f"🌱 Discovered new topic: {t}")

        # ----------------------------
        # Benchmark evaluation
        # ----------------------------
        print("\n🧪 Evaluating model scientific understanding (benchmark)...")
        eval_report = model_tester.evaluate_on_benchmark()
        print(json.dumps(eval_report, indent=2))

        # ----------------------------
        # Demo: proposal judge (example)
        # ----------------------------
        demo_proposal = "F = m a assuming classical mechanics and low speeds."
        print("\n🧾 Proposal Judge Demo:")
        demo_hits = chunk_index.retrieve("Newton's second law", top_k=5, use_mmr=True)
        judged = proposal_engine.evaluate(
            demo_proposal,
            provenance=SourceTrace(source_type="user_proposal", source_name="main_demo"),
            evidence_chunks=demo_hits,
        )
        print(json.dumps(judged["verdict"], indent=2))

        # Also demo chunk retrieval:
        print("\n🔎 Chunk Retrieval Demo (Newton 2):")
        hits = chunk_index.retrieve("Newton's second law", top_k=5)
        print(json.dumps(hits, indent=2))

        # ----------------------------
        # Dashboard state
        # ----------------------------
        dash_payload = {
            "cycle": cycle,
            "device": str(device),
            "topics": len(topics),
            "kb_count": kb.count(),
            "chunk_count": chunk_store.count(),
            "chunk_index_items": int(chunk_index.vecs.shape[0]) if chunk_index.vecs is not None else 0,
            "last_loss": trainer.last_avg_loss,
            "last_similarity": trainer.last_similarity,
            "trained_docs_this_cycle": total_new_for_training,
            "chunks_added_this_cycle": total_chunks_added,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_dashboard_state(dash_payload)

        cycle += 1
        print(f"\n✅ Completed learning cycle {cycle-1}. Sleeping 1 week before next cycle...\n")
        gc.collect()
        time.sleep(SLEEP_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("run", help="Run universal learning cycles (current behavior).")

    p = sub.add_parser("discover", help="Generate a Discovery Report for a query.")
    p.add_argument("query", type=str, help="Discovery question / goal")

    args = parser.parse_args()

    if args.cmd == "discover":
        # --- Build subsystems once (same as your main, but no cycle loop) ---
        device = get_device()
        print(f"🤖 Starting Discovery Mode on {device}")

        omni = OmniRetriever()
        safety = SafetyFilter()

        kb = KnowledgeBase()
        chunk_store = ChunkStore()

        trainer = OnlineTrainer()
        memory = TrainingMemory()
        bridge = EmbeddingBridge(trainer)

        reflection = ReflectionEngine(omni, safety, kb, trainer)

        evaluator = EvidenceEvaluator(
            kb,
            bridge,
            reflection.kg,
            max_index_items=KB_INDEX_ITEMS_FOR_EVIDENCE,
            top_k=10
        )

        hypgen = HypothesisGenerator(reflection.kg, bridge, kb)
        validator = HypothesisValidator(kb, bridge, reflection.kg)
        proposal_engine = ProposalEvaluator(kb, bridge, top_k=10, evidence_threshold=0.60, max_kb_items=3000)
        print("ProposalEvaluator loaded from:", ProposalEvaluator.__module__)

        chunk_index = ChunkIndex(
            chunk_store=chunk_store,
            encoder=bridge,
            cache_dir="data",
            max_items=MAX_CHUNKS_TO_INDEX,
            chunk_batch=CHUNK_INDEX_BATCH,
        )

        from reasoning_module.discover import DiscoveryEngine, DiscoveryConfig
        discover = DiscoveryEngine(
            chunk_index=chunk_index,
            proposal_engine=proposal_engine,
            evidence_evaluator=evaluator,
            hypgen=hypgen,
            validator=validator,
            config=DiscoveryConfig(top_k_chunks=12, evidence_threshold=0.60, max_hypotheses=8),
        )

        report = discover.run(args.query, source_name="cli")
        print(json.dumps(report, indent=2))

        evidence = report.get("proposal_verdict", {}).get("evidence", []) or []
        print("\n===== TOP 3 PROPOSAL EVIDENCE =====")
        for e in evidence[:3]:
            print("\n---")
            print("chunk_id:", e.get("chunk_id"))
            print("source:", e.get("source"))
            print("similarity:", round(float(e.get("similarity_to_question", 0.0) or 0.0), 4))
            print("title:", e.get("paper_title"))
            print("preview:", (e.get("text","")[:200]).replace("\n"," "))

    else:
        main()
