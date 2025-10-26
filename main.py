#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Humanoid Scientist Brain — Continuous Learning & Hypothesis Evolution
---------------------------------------------------------------------
Phase F+: Semantic Reasoning Integration + Online Transformer Learning
Now includes:
  • Persistent TrainingMemory — prevents duplicate paper training
  • EvidenceEvaluator — checks if hypotheses are supported by known science
  • Full continual learning cycle with reflection, reasoning, and dashboard
"""

import time
import json
import os

# Enable automatic CPU fallback for unsupported MPS ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print("⚙️ Enabled MPS CPU fallback for unsupported ops.")

import gc
import re
from typing import List, Dict, Any

from data_pipeline.fetcher import DataFetcher
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from learning_module.trainer_online import OnlineTrainer, get_device
from learning_module.embedding_bridge import EmbeddingBridge
from learning_module.training_memory import TrainingMemory
from reasoning_module.reasoning import GraphAugmentedReasoning
from reasoning_module.graph_reasoner import GraphReasoner
from reasoning_module.hypothesis_generator import HypothesisGenerator
from reasoning_module.hypothesis_validator import HypothesisValidator
from reasoning_module.hypothesis_evolver import HypothesisEvolver
from reflection_module.reflection import ReflectionEngine
from data_pipeline.internet_retriever import InternetRetriever
from reasoning_module.evidence_evaluator import EvidenceEvaluator

# ----------------------------
# Setup
# ----------------------------
DASHBOARD_STATE_PATH = "data/dashboard_state.json"
os.makedirs("data", exist_ok=True)
os.makedirs("visualization/graphs", exist_ok=True)
MAX_CYCLES = 3  # switch to while True for autonomous daemon mode


# ----------------------------
# Helpers
# ----------------------------
def write_dashboard_state(payload: dict) -> None:
    """Write current model + KG state to dashboard JSON."""
    with open(DASHBOARD_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("📊 Dashboard state updated → data/dashboard_state.json")


def extract_concepts(text: str, top_n: int = 5) -> str:
    """Lightweight keyword extractor using regex frequency."""
    words = re.findall(r"\b[a-zA-Z]{5,}\b", (text or "").lower())
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return ", ".join([w for w, _ in top])


def _latest_visualization_path() -> str | None:
    vis_dir = "visualization/graphs"
    try:
        all_graphs = [
            f for f in os.listdir(vis_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
        ]
        if not all_graphs:
            return None
        all_graphs.sort(key=lambda f: os.path.getmtime(os.path.join(vis_dir, f)), reverse=True)
        return os.path.join(vis_dir, all_graphs[0])
    except Exception:
        return None


# ----------------------------
# Main
# ----------------------------
def main():
    # ---------- Device / startup ----------
    device = get_device()
    print(f"🤖 Starting Humanoid Scientist Brain (Continuous Learning Mode) on device: {device}")

    # ---------- Core components ----------
    fetcher = DataFetcher()
    safety = SafetyFilter()
    kb = KnowledgeBase()
    online_trainer = OnlineTrainer(None, batch_size=32, epochs=2, lr=1e-3)
    retriever = InternetRetriever()
    reflection = ReflectionEngine(fetcher, safety, kb, online_trainer)
    memory = TrainingMemory()
    bridge = EmbeddingBridge(online_trainer)

    # Reasoning + hypothesis modules
    hypgen = HypothesisGenerator(reflection.kg, bridge, kb)
    validator = HypothesisValidator(kb, bridge, reflection.kg)
    evolver = HypothesisEvolver(kb, reflection.kg)
    reasoning = GraphAugmentedReasoning(kb, bridge, reflection.kg, hypothesis_generator=hypgen)
    reasoner = GraphReasoner(reflection.kg, online_trainer=online_trainer)
    evaluator = EvidenceEvaluator(kb, bridge, reflection.kg)

    topic = "physics"
    cycle = 1

    try:
        while cycle <= MAX_CYCLES:
            print(f"\n🌀 Learning Cycle {cycle} starting...")

            # ---------- Local fetch + safety filtering ----------
            raw_items = fetcher.fetch(topic)
            safe_items = safety.filter(raw_items)
            safe_texts = [
                json.dumps(it, ensure_ascii=False) if isinstance(it, dict) else str(it)
                for it in safe_items
            ]
            kb.current_source = "fetcher"
            if safe_texts:
                kb.store(safe_texts)

            # ---------- Internet retrieval ----------
            print("\n🌐 [Internet Expansion] Retrieving scientific papers and concepts...")
            topics_to_search = [topic, "quantum mechanics", "AI in science", "theoretical physics"]
            try:
                web_items = retriever.retrieve_topics(topics_to_search, max_items=50, safety=safety)
            except Exception as e:
                print(f"⚠️ InternetRetriever error: {e}")
                web_items = []

            # ---------- Deduplicate & Train ----------
            if web_items:
                print(f"🌍 Retrieved {len(web_items)} new internet-based papers.")
                new_items = memory.filter_new(web_items)
                if not new_items:
                    print("⚙️ All retrieved papers were already trained — skipping training.")
                    enriched_items = []
                else:
                    print(f"🧠 {len(new_items)} new unseen papers will be trained.")
                    enriched_items = []
                    for item in new_items:
                        text = item.get("text", "") if isinstance(item, dict) else str(item)
                        title = item.get("title", "Unknown Paper") if isinstance(item, dict) else "Unknown Paper"
                        concepts = extract_concepts(text)
                        enriched_items.append({
                            "text": text,
                            "paper_title": title,
                            "concepts": concepts,
                            "source": "internet"
                        })
                    kb.store(enriched_items)
                    online_trainer.incremental_train(enriched_items)
                    memory.mark_trained(enriched_items)
            else:
                print("⚠️ No new web data retrieved this cycle.")
                enriched_items = []

            # ---------- Reflection + KG update ----------
            print("\n🧠 [Reflection] Updating Knowledge Graph...")
            reflection.review_knowledge()

            # ---------- Visualization snapshot ----------
            latest_vis_path = _latest_visualization_path()

            # ---------- Hypothesis Generation + Validation ----------
            hyps = hypgen.generate(top_n=20)
            if hyps:
                print("🧪 Generated hypotheses (top 5):")
                for h in hyps[:5]:
                    ht = h.get("type", "?")
                    hs = h.get("score", 0.0)
                    print(f"  • [{ht}] {h['hypothesis']} (score={round(hs, 3)})")
                validated = validator.validate(hyps, cycle=cycle)
                evolver.update(validated, cycle=cycle)
                reflection.promote_validated(validated)

                # ---------- Evidence Evaluation ----------
                evaluated = evaluator.evaluate_batch(validated)
                print("🧩 Evidence evaluation complete. Verdicts summary:")
                for e in evaluated[:5]:
                    print(
                        f"  - {e.get('hypothesis')[:70]}... "
                        f"→ {e.get('verdict', '?')} (conf={e.get('evidence_confidence')})"
                    )

            # ---------- Semantic Graph Reasoning ----------
            try:
                print("🔎 Graph reasoning demo:", reasoner.explain_relation("force", "motion"))
                print("🔎 Transitive (causes) from 'force':", reasoner.suggest_transitive("force", "causes"))
            except Exception as e:
                print(f"ℹ️ GraphReasoner demo skipped: {e}")

            # ---------- Reasoning Query ----------
            try:
                ans = reasoning.answer("Explain Newton’s third law in simple words.")
                print("🧠 Reasoning Output:\n", ans)
            except Exception as e:
                print(f"ℹ️ Reasoning demo skipped: {e}")

            # ---------- Dashboard ----------
            try:
                edge_count = reflection.kg.edge_count() if hasattr(reflection.kg, "edge_count") else 0
                node_count = len(getattr(reflection.kg, "graph", {}) or {})
            except Exception:
                edge_count, node_count = 0, 0

            dash_payload = {
                "cycle": cycle,
                "device": str(device),
                "kg": {"nodes": node_count, "edges": edge_count},
                "training": {
                    "online_loss": getattr(online_trainer, "last_avg_loss", None),
                    "online_sim": getattr(online_trainer, "last_similarity", None),
                    "vocab_size": getattr(getattr(online_trainer, "tokenizer", None), "vocab_size", None),
                },
                "recent_papers": [i.get("paper_title", "Unknown") for i in enriched_items[-5:]] if enriched_items else [],
                "concepts_learned": [i.get("concepts", "") for i in enriched_items[-5:]] if enriched_items else [],
                "visualization": latest_vis_path,
                "timestamp": int(time.time()),
            }
            write_dashboard_state(dash_payload)

            # ---------- Planning ----------
            print("\n🧭 [Planning] Choosing next topic...")
            try:
                reflection.plan_next_steps()
            except Exception as e:
                print(f"ℹ️ Planning skipped: {e}")

            # ---------- Cycle housekeeping ----------
            gc.collect()
            cycle += 1
            time.sleep(5)

        print("\n🧩 Maximum cycles reached — stopping continuous learning loop.")

    except KeyboardInterrupt:
        print("\n🧩 Learning loop interrupted by user.")
    finally:
        try:
            kb.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
