#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Humanoid Scientist Brain — Universal Continuous Learning
--------------------------------------------------------
Now trains on *all* physics & mathematics topics (from history to present)
using OmniRetriever (multi-source aggregator) and evaluates after training.
"""

import os, time, json, gc, re
from typing import Dict, Any

# Enable automatic CPU fallback for MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print("⚙️ Enabled MPS CPU fallback for unsupported ops.")

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from data_pipeline.omni_retriever import OmniRetriever
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from learning_module.trainer_online import OnlineTrainer, get_device
from learning_module.embedding_bridge import EmbeddingBridge
from learning_module.training_memory import TrainingMemory
from reflection_module.reflection import ReflectionEngine
from reasoning_module.evidence_evaluator import EvidenceEvaluator
from reasoning_module.reasoning import GraphAugmentedReasoning
from reasoning_module.graph_reasoner import GraphReasoner
from reasoning_module.hypothesis_generator import HypothesisGenerator
from reasoning_module.hypothesis_validator import HypothesisValidator
from reasoning_module.hypothesis_evolver import HypothesisEvolver
from testing_module.model_evaluator import ModelEvaluator  # <-- new evaluator

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DASHBOARD_STATE_PATH = "data/dashboard_state.json"
os.makedirs("data", exist_ok=True)
os.makedirs("visualization/graphs", exist_ok=True)
MAX_CYCLES = 5
SLEEP_INTERVAL = 60 * 60 * 24 * 7  # 1 week

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def write_dashboard_state(payload: Dict[str, Any]):
    with open(DASHBOARD_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("📊 Dashboard updated → data/dashboard_state.json")


def extract_keywords(text: str, n=5):
    words = re.findall(r"\b[a-zA-Z]{5,}\b", (text or "").lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return ", ".join([w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:n]])

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
    trainer = OnlineTrainer()
    memory = TrainingMemory()
    bridge = EmbeddingBridge(trainer)
    reflection = ReflectionEngine(omni, safety, kb, trainer)
    evaluator = EvidenceEvaluator(
        kb,
        bridge,
        reflection.kg,
        max_index_items=800,   # start small; scale later
        top_k=10)
    reasoner = GraphReasoner(reflection.kg, online_trainer=trainer)
    hypgen = HypothesisGenerator(reflection.kg, bridge, kb)
    validator = HypothesisValidator(kb, bridge, reflection.kg)
    evolver = HypothesisEvolver(kb, reflection.kg)
    reasoning = GraphAugmentedReasoning(kb, bridge, reflection.kg, hypothesis_generator=hypgen)
    model_tester = ModelEvaluator(trainer, kb, bridge)

    # Start with full scientific coverage
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
        all_docs = []

        for topic in topics:
            docs = omni.retrieve_universal(topic)
            safe = safety.filter(docs)
            kb.store(safe)
            new_items = memory.filter_new(safe)
            if not new_items:
                print(f"⚙️ Skipping '{topic}' (no unseen papers).")
                continue

            # Enrich and train
            enriched = []
            for d in new_items:
                text = d.get("text", "")
                enriched.append({
                    "text": text,
                    "paper_title": d.get("title", "Unknown Paper"),
                    "concepts": extract_keywords(text),
                    "source": d.get("source", "")
                })
            kb.store(enriched)
            trainer.incremental_train(enriched)
            memory.mark_trained(enriched)
            all_docs.extend(enriched)

        # --- Reflection ---
        print("\n🧠 Reflecting & updating Knowledge Graph...")
        reflection.review_knowledge()

        # --- Hypothesis generation + validation ---
        hyps = hypgen.generate(top_n=20)
        if hyps:
            validated = validator.validate(hyps, cycle=cycle)
            evolver.update(validated, cycle=cycle)
            reflection.promote_validated(validated)
            evaluated = evaluator.evaluate_batch(validated)
            print("🧩 Evidence check done.")
            for e in evaluated[:5]:
                print(f"  → {e['verdict']}: {e['hypothesis'][:80]}")

        # --- Expand topic space dynamically ---
        new_topics = omni.extract_new_topics(all_docs)
        for t in new_topics:
            if t not in topics and len(t) > 4:
                topics.append(t)
                print(f"🌱 Discovered new topic: {t}")

        # --- Post-train model evaluation ---
        print("\n🧪 Evaluating model scientific understanding...")
        eval_report = model_tester.evaluate_on_benchmark()
        print(json.dumps(eval_report, indent=2))

        # --- Save dashboard state ---
        dash_payload = {
            "cycle": cycle,
            "device": str(device),
            "topics": len(topics),
            "last_loss": trainer.last_avg_loss,
            "last_similarity": trainer.last_similarity,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        write_dashboard_state(dash_payload)

        cycle += 1
        print(f"✅ Completed learning cycle {cycle-1}. Sleeping 1 week before next cycle...\n")
        gc.collect()
        time.sleep(SLEEP_INTERVAL)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
