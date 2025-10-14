"""
Reflection Engine — Phase B Step 3
-----------------------------------
Enhancements:
• Automatically expands knowledge using gaps in the Knowledge Graph
• Detects missing or weak relations (e.g., "force → motion")
• Fetches, filters, stores, and trains on new data to fill conceptual gaps
"""

import os
import random
from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.concept_merger import ConceptMerger
from reasoning_module.graph_reasoner import GraphReasoner


KG_PATH = "knowledge_graph/graph.json"


class ReflectionEngine:
    def __init__(self, fetcher, safety, kb, trainer):
        self.fetcher = fetcher
        self.safety = safety
        self.kb = kb
        self.trainer = trainer
        self.kg = KnowledgeGraph()
        self.last_topic = "physics"

        if os.path.exists(KG_PATH):
            try:
                self.kg.load(KG_PATH)
                print("📥 Loaded existing Knowledge Graph from disk.")
            except Exception as e:
                print(f"⚠️ Could not load KG: {e}")

    # -------------------------------------------------------------------------
    def review_knowledge(self):
        """Rebuilds Knowledge Graph and identifies conceptual gaps."""
        print("[Reflection] Reviewing KB and updating Knowledge Graph...")
        items = self.kb.query("")
        texts = [item["text"] if isinstance(item, dict) else str(item) for item in items]

        self.kg.build_from_corpus(texts)

        # Merge & cluster similar concepts
        merger = ConceptMerger(self.kg)
        merger.merge_similar_concepts(threshold=0.82)
        merger.cluster_topics()

        # Display and persist
        self.kg.visualize()
        try:
            self.kg.save(KG_PATH)
            print("💾 KG saved.")
        except Exception as e:
            print(f"⚠️ Could not save KG: {e}")

        # Analyze weak connections
        self._expand_from_graph_gaps()

    # -------------------------------------------------------------------------
    def plan_next_steps(self):
        """Chooses the next high-value topic to study."""
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
        raw = self.fetcher.fetch(topic)
        safe = self.safety.filter(raw)
        self.kb.store(safe)
        self.trainer.run_training(self.kb.query(""))

        self.last_topic = topic
        print(f"[Reflection] Finished updating with new knowledge on '{topic}' ✅")

    # -------------------------------------------------------------------------
    def _expand_from_graph_gaps(self):
        """
        Detects weak or missing relations in the graph and learns about them.
        This is where the system autonomously expands its scientific knowledge.
        """
        print("🧩 [Auto-Expansion] Searching for missing or weak graph connections...")

        gr = GraphReasoner(self.kg)
        concepts = list(self.kg.graph.keys())

        if len(concepts) < 3:
            print("🟡 [Auto-Expansion] Not enough concepts to analyze relationships yet.")
            return

        # Randomly sample pairs to test for missing links
        possible_pairs = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                a, b = concepts[i], concepts[j]
                if not self.kg.graph.has_edge(a, b):
                    possible_pairs.append((a, b))

        if not possible_pairs:
            print("✅ [Auto-Expansion] All major concepts already connected.")
            return

        # Choose a small subset of interesting pairs
        to_explore = random.sample(possible_pairs, min(2, len(possible_pairs)))
        for a, b in to_explore:
            print(f"🔍 [Auto-Expansion] Exploring gap between '{a}' and '{b}'...")

            query = f"relationship between {a} and {b} in science"
            raw = self.fetcher.fetch(query)
            safe = self.safety.filter(raw)
            self.kb.store(safe)

            print(f"📘 [Auto-Expansion] Stored new data for '{a} ↔ {b}'")
            self.trainer.run_training(self.kb.query(""))

        print("✅ [Auto-Expansion] Graph gap learning complete.")
