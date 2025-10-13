# reflection_module/reflection.py
"""
Reflection Engine — Phase B Step 2/3
- Rebuilds KG from KB each cycle
- Merges similar concepts and clusters them
- Plans next learning step using KG concepts
- Persists KG to disk
"""

import os
import random
from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.concept_merger import ConceptMerger

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

    def review_knowledge(self):
        print("[Reflection] Reviewing KB and updating Knowledge Graph...")
        items = self.kb.query("")
        texts = [item["text"] if isinstance(item, dict) else str(item) for item in items]

        self.kg.build_from_corpus(texts)

        # Merge & cluster
        merger = ConceptMerger(self.kg)
        merger.merge_similar_concepts(threshold=0.82)
        merger.cluster_topics()

        self.kg.visualize()
        try:
            self.kg.save(KG_PATH)
            print("💾 KG saved.")
        except Exception as e:
            print(f"⚠️ Could not save KG: {e}")

    def plan_next_steps(self):
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
