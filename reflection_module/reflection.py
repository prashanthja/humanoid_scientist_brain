"""
Module: Reflection Engine
Phase B – Step 3
Analyzes the robot's knowledge base, updates and expands its Knowledge Graph,
merges overlapping concepts, and plans next learning steps.
"""

import os
import random
from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.concept_merger import ConceptMerger


KG_PATH = "knowledge_graph/graph.json"


class ReflectionEngine:
    """
    ReflectionEngine helps the humanoid robot:
    - Review current knowledge (Knowledge Base + Graph)
    - Expand its understanding
    - Merge overlapping or similar ideas
    - Plan what to learn next
    """

    def __init__(self, fetcher, safety, kb, trainer):
        self.fetcher = fetcher
        self.safety = safety
        self.kb = kb
        self.trainer = trainer
        self.kg = KnowledgeGraph()
        self.last_topic = "physics"

        # Load existing knowledge graph if available
        if os.path.exists(KG_PATH):
            try:
                self.kg.load(KG_PATH)
                print("📥 Loaded existing Knowledge Graph from disk.")
            except Exception as e:
                print(f"⚠️ Could not load KG: {e}")

    # ------------------------------------------------------------------
    # STEP 1: Review current knowledge
    # ------------------------------------------------------------------
    def review_knowledge(self):
        """Update the Knowledge Graph from the KB and show key insights."""
        print("[Reflection] Reviewing KB and updating Knowledge Graph...")

        items = self.kb.query("")
        texts = [item["text"] if isinstance(item, dict) else str(item) for item in items]

        # Rebuild KG
        self.kg.build_from_corpus(texts)

        # Merge and cluster
        merger = ConceptMerger(self.kg)
        merger.merge_similar_concepts(threshold=0.8)
        merger.cluster_topics()

        # Visualize and save
        self.kg.visualize()
        try:
            self.kg.save(KG_PATH)
            print("💾 KG saved.")
        except Exception as e:
            print(f"⚠️ Could not save KG: {e}")

    # ------------------------------------------------------------------
    # STEP 2: Plan next learning goal
    # ------------------------------------------------------------------
    def plan_next_steps(self):
        """Pick the next topic to learn and trigger data fetch + training."""
        concepts = self.kg.all_concepts()

        # Pick new topic intelligently
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

        # Fetch and filter new knowledge
        raw = self.fetcher.fetch(topic)
        safe = self.safety.filter(raw)

        # Store in KB
        self.kb.store(safe)

        # Retrain on entire KB
        all_data = self.kb.query("")
        self.trainer.run_training(all_data)

        self.last_topic = topic
        print(f"[Reflection] Finished updating with new knowledge on '{topic}' ✅")
