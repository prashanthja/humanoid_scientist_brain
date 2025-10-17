"""
Reflection Engine — Phase S Step 3
----------------------------------
• Rebuild/refresh the Knowledge Graph (KG) from the KB
• Merge similar concepts & (optionally) cluster topics
• Run lightweight graph reasoning to surface gaps
• Persist KG (JSON) and PNG snapshot per cycle
• Plan next topics (graph-guided + fallback)

This module is defensive: if optional helpers (visualizer/clusterer)
aren't available, it will continue without failing the run.
"""

from __future__ import annotations
import os
import random
from typing import List, Tuple

from data_pipeline.fetcher import DataFetcher
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from learning_module.trainer import Trainer

# KG core + tools
from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.concept_merger import ConceptMerger

# Optional visualizer (safe import)
try:
    from visualization.graph_progress import GraphProgressVisualizer
except Exception:  # pragma: no cover
    GraphProgressVisualizer = None  # type: ignore

# Optional graph reasoning (safe import)
try:
    from reasoning_module.graph_reasoner import GraphReasoner
except Exception:  # pragma: no cover
    GraphReasoner = None  # type: ignore


KG_PATH = "knowledge_graph/graph.json"
IMG_DIR = "visualization/graphs"


class ReflectionEngine:
    def __init__(
        self,
        fetcher: DataFetcher,
        safety: SafetyFilter,
        kb: KnowledgeBase,
        trainer: Trainer,
    ):
        self.fetcher = fetcher
        self.safety = safety
        self.kb = kb
        self.trainer = trainer

        self.kg = KnowledgeGraph()
        self.last_topic = "physics"
        self.cycle = 0

        # Optional progress plotter
        self.visualizer = GraphProgressVisualizer() if GraphProgressVisualizer else None

        # Try to load previous KG
        if os.path.exists(KG_PATH):
            try:
                self.kg.load(KG_PATH)
                print("📥 Loaded existing Knowledge Graph from disk.")
            except Exception as e:
                print(f"⚠️ Could not load KG: {e}")

        os.makedirs(IMG_DIR, exist_ok=True)

    # ---------------------------------------------------------------------
    # Public API called from main.py
    # ---------------------------------------------------------------------
    def review_knowledge(self) -> None:
        """
        Rebuild KG from the KB, merge similar, (optionally) cluster,
        run quick insights, and persist graph + PNG.
        """
        self.cycle += 1
        print("[Reflection] Reviewing KB and updating Knowledge Graph...")

        # 1) Rebuild KG from KB texts
        items = self.kb.query("")
        texts = [it["text"] if isinstance(it, dict) else str(it) for it in items]
        self.kg.build_from_corpus(texts)

        # 2) Merge similar concepts (safe)
        try:
            print("🧩 Merging similar concepts...")
            ConceptMerger(self.kg).merge_similar_concepts(threshold=0.82)
        except Exception as e:
            print(f"⚠️ Merge skipped: {e}")

        # 3) (Optional) topic clustering if available in your ConceptMerger
        try:
            if hasattr(ConceptMerger, "cluster_topics"):
                clusters = ConceptMerger(self.kg).cluster_topics()
                if clusters:
                    print(f"🧠 Identified {len(clusters)} topic clusters:")
                    for i, cl in enumerate(clusters, 1):
                        preview = ", ".join(list(cl)[:1]) + ("..." if len(cl) > 1 else "")
                        print(f"  Cluster {i}: {preview}")
        except Exception as e:
            # Non-fatal
            pass

        # 4) Quick graph insights (safe)
        try:
            if GraphReasoner:
                reasoner = GraphReasoner(self.kg)
                for a, b in [("force", "motion"), ("energy", "work"), ("temperature", "heat")]:
                    insight = reasoner.explain_relation(a, b)
                    print(f"🔎 Insight ({a}→{b}): {insight}")
        except Exception as e:
            print(f"⚠️ Graph reasoning preview skipped: {e}")

        # 5) Persist KG + PNG snapshot
        try:
            self.kg.save(KG_PATH)
            print("💾 KG saved.")
        except Exception as e:
            print(f"⚠️ Could not save KG: {e}")

        try:
            if self.visualizer:
                self.visualizer.plot(self.kg, cycle_num=self.cycle)
                print(f"🧩 Saved KG visualization: {self.visualizer.last_path}")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")

    def plan_next_steps(self) -> None:
        """
        Pick the next concept/topic to learn (KG-guided if possible),
        fetch → filter → store → train. Falls back gracefully.
        """
        print("[Reflection] Evaluating next learning step...")
        topic = self._choose_next_topic()

        print(f"[Reflection] Planning next step: learn more about '{topic}'")
        raw = self.fetcher.fetch(topic)
        safe = self.safety.filter(raw)
        self.kb.store(safe)
        # train on *just* this batch to keep step tight
        try:
            self.trainer.run_training(safe)
        except Exception:
            # fall back to full KB if trainer expects larger batch
            self.trainer.run_training([t for t in self.kb.query("")])

        print(f"[Reflection] Finished updating with new knowledge on '{topic}' ✅")

    def expand_from_graph_gaps(self, max_targets: int = 2) -> None:
        """
        Use the KG to find weak or disconnected concepts and acquire data
        to strengthen those areas (2 targets by default).
        """
        # 1) Weak concepts: low-degree nodes
        weak = self._weak_concepts(limit=max_targets)

        # 2) Disconnected/interesting pairs (optional in your KG impl)
        pairs = self._disconnected_science_pairs(limit=max_targets)

        targets = [*weak]
        for a, b in pairs:
            targets.append(f"{a} and {b} relationship")

        print(f"[Reflection] Weak concepts detected: {len(weak)}")
        print(f"[Reflection] Disconnected science pairs: {len(pairs)}")
        print(f"[Reflection] Expanding knowledge for {len(targets)} targets...")

        for t in targets:
            try:
                self._fetch_train_topic(f"{t} basics in science")
            except Exception:
                continue

        print("✅ [Reflection] Graph expansion complete.")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _choose_next_topic(self) -> str:
        # Prefer KG concepts
        try:
            concepts = list(self.kg.all_concepts())
            if concepts:
                topic = random.choice(concepts)
                # avoid repeating the same concept a few times
                tries = 0
                while topic == self.last_topic and tries < 5:
                    topic = random.choice(concepts)
                    tries += 1
                self.last_topic = topic
                return topic
        except Exception:
            pass

        # Fallback pool
        return random.choice([
            "physics",
            "mathematics",
            "machine learning",
            "thermodynamics",
            "linear algebra",
            "computational physics",
            "ethics in AI",
        ])

    def _weak_concepts(self, limit: int = 2) -> List[str]:
        try:
            degrees = [(c, len(self.kg.get_relations(c))) for c in self.kg.all_concepts()]
            degrees.sort(key=lambda x: x[1])
            return [c for c, _ in degrees[:limit]]
        except Exception:
            return []

    def _disconnected_science_pairs(self, limit: int = 2) -> List[Tuple[str, str]]:
        if not GraphReasoner:
            return []
        try:
            r = GraphReasoner(self.kg)
            concepts = list(self.kg.all_concepts())
            random.shuffle(concepts)
            pairs = []
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    a, b = concepts[i], concepts[j]
                    expl = r.explain_relation(a, b)
                    if "No" in str(expl):
                        pairs.append((a, b))
                        if len(pairs) >= limit:
                            return pairs
            return pairs[:limit]
        except Exception:
            return []

    def _fetch_train_topic(self, topic: str) -> None:
        print(f"[Fetcher] Fetching new data for topic: {topic}")
        raw = self.fetcher.fetch(topic)
        safe = self.safety.filter(raw)
        self.kb.store(safe)
        print(f"🧠 Training on {len(safe)} knowledge items...")
        self.trainer.run_training(safe)

    # --- Add this helper to ReflectionEngine ---
    def promote_validated(self, validated_list, save_path: str = "knowledge_graph/graph.json"):
        """
        Promote high-confidence hypotheses into the KG as edges.
        Only those with promote=True are added.
        """
        if not validated_list:
            return

        added = 0
        for v in validated_list:
            if not v.get("promote"):
                continue
            a = v["subject"].lower()
            b = v["object"].lower()
            rel = v["relation"].lower()
            try:
                self.kg.add_edge(a, rel, b)
                added += 1
            except Exception:
                pass

        if added > 0:
            try:
                self.kg.save(save_path)
                print(f"🌱 Promoted {added} validated hypotheses into KG and saved.")
            except Exception as e:
                print(f"⚠️ Could not save promoted KG: {e}")
