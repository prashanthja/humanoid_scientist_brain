# reflection_module/reflection.py
"""
Reflection Engine — Phase B Step 4 (Graph-Driven Expansion)
- Rebuilds KG from KB (each cycle)
- Merges + clusters concepts
- Visualizes KG and (every 3 cycles) animates snapshots
- Detects graph gaps (weak nodes, disconnected key pairs)
- Auto-expands knowledge by fetching + training on gap topics
- Plans next learning topic (fallback when gaps are minimal)
"""

import os
import random
from typing import List, Tuple

from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.concept_merger import ConceptMerger
from reasoning_module.graph_reasoner import GraphReasoner
from visualization.graph_progress import GraphProgressVisualizer

KG_PATH = "knowledge_graph/graph.json"

KEY_SCIENCE_PAIRS: List[Tuple[str, str]] = [
    ("force", "motion"),
    ("energy", "work"),
    ("temperature", "heat"),
    ("mass", "acceleration"),
    ("power", "energy"),
]


class ReflectionEngine:
    def __init__(self, fetcher, safety, kb, trainer):
        self.fetcher = fetcher
        self.safety = safety
        self.kb = kb
        self.trainer = trainer

        self.kg = KnowledgeGraph()
        self.visualizer = GraphProgressVisualizer()
        self.last_topic = "physics"
        self.cycle = 1

        if os.path.exists(KG_PATH):
            try:
                self.kg.load(KG_PATH)
                print("📥 Loaded existing Knowledge Graph from disk.")
            except Exception as e:
                print(f"⚠️ Could not load KG: {e}")

    # -----------------------------------------------------------
    def review_knowledge(self):
        print("\n🧠 [Reflection] Begin KG update and gap analysis...")
        print("[Reflection] Reviewing KB and updating Knowledge Graph...")

        items = self.kb.query("")
        texts = [item["text"] if isinstance(item, dict) else str(item) for item in items]
        if not texts:
            print("[Reflection] KB is empty; nothing to reflect on.")
            return

        # 1) Rebuild KG
        self.kg.build_from_corpus(texts)

        # 2) Merge + cluster
        merger = ConceptMerger(self.kg)
        merger.merge_similar_concepts(threshold=0.82)
        merger.cluster_topics()

        # 3) Reasoning probes
        reasoner = GraphReasoner(self.kg)
        for a, b in KEY_SCIENCE_PAIRS[:3]:
            insight = reasoner.explain_relation(a, b)
            print(f"🔎 Insight ({a}→{b}): {insight}")

        # 4) Save KG
        try:
            self.kg.save(KG_PATH)
            print("💾 KG saved.")
        except Exception as e:
            print(f"⚠️ Could not save KG: {e}")

        # 5) Visualize + periodic animation
        try:
            self.visualizer.plot(self.kg, cycle_num=self.cycle)
            if self.cycle % 3 == 0:
                self.visualizer.create_animation()
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")

        # 6) Auto expansion from graph gaps (new in Step 4)
        self.expand_graph_knowledge()

        self.cycle += 1

    # -----------------------------------------------------------
    def plan_next_steps(self):
        print("[Reflection] Evaluating next learning step...")

        # If gaps are few, proceed with breadth-first topic growth
        concepts = self.kg.all_concepts()
        if concepts:
            topic = random.choice(concepts)
            tries = 0
            while topic == self.last_topic and tries < 5:
                topic = random.choice(concepts)
                tries += 1
        else:
            print("[Reflection] KG empty — fallback topic selection.")
            topic = random.choice([
                "physics", "mathematics", "machine learning",
                "thermodynamics", "linear algebra"
            ])

        print(f"[Reflection] Planning next step: learn more about '{topic}'")
        self._fetch_filter_store_train(topic)

        self.last_topic = topic
        print(f"[Reflection] Finished updating with new knowledge on '{topic}' ✅")

    # -----------------------------------------------------------
    # Phase B — Step 4: Graph-Driven Knowledge Expansion
    def expand_graph_knowledge(self):
        """
        Identify weak nodes (low degree) and disconnected core pairs,
        then fetch+train on those topics to strengthen the graph.
        """
        weak_nodes = self.kg.find_weak_concepts(threshold=2)
        print(f"[Reflection] Weak concepts detected: {len(weak_nodes)}")

        # Probe a few canonical science pairs for missing paths
        reasoner = GraphReasoner(self.kg)
        missing_pairs = []
        for a, b in KEY_SCIENCE_PAIRS:
            relation = reasoner.explain_relation(a, b)
            if "No short path" in str(relation):
                missing_pairs.append((a, b))

        print(f"[Reflection] Disconnected science pairs: {len(missing_pairs)}")

        topics: List[str] = []

        # Prioritize bridging missing pairs
        for (a, b) in missing_pairs[:2]:
            topics.append(f"relationship between {a} and {b} in physics")

        # Then reinforce weak nodes
        for node in weak_nodes[:3]:
            topics.append(f"{node} basics in science")

        if not topics:
            print("[Reflection] No critical gaps found — skipping graph-expansion this cycle.")
            return

        print(f"[Reflection] Expanding knowledge for {len(topics)} targets...")
        for t in topics:
            self._fetch_filter_store_train(t)

        print("✅ [Reflection] Graph expansion complete.")

    # -----------------------------------------------------------
    # Helpers
    def _fetch_filter_store_train(self, topic: str):
        raw = self.fetcher.fetch(topic)
        safe = self.safety.filter(raw)
        self.kb.store(safe)
        # Train incrementally on the new batch (keeps cycles fast)
        try:
            self.trainer.run_training(safe)
        except Exception:
            # Fallback to train on entire KB if incremental fails
            self.trainer.run_training(self.kb.query(""))
