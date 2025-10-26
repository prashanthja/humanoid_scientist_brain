"""
Reflection Engine — Phase S Step 3
----------------------------------
• Rebuild/refresh the Knowledge Graph (KG) from the KB
• Merge similar concepts & (optionally) cluster topics
• Run lightweight graph reasoning to surface gaps
• Persist KG (JSON) and PNG snapshot per cycle
• Plan next topics (graph-guided + fallback)

Now uses OnlineTrainer for continuous incremental learning
instead of offline Trainer.
"""

from __future__ import annotations
import os
import random
from typing import List, Tuple

from data_pipeline.fetcher import DataFetcher
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from learning_module.trainer_online import OnlineTrainer  # ✅ switched to OnlineTrainer

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
        trainer: OnlineTrainer,  # ✅ changed type
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
        """Rebuild KG from KB and visualize."""
        self.cycle += 1
        print("[Reflection] Reviewing KB and updating Knowledge Graph...")

        # 1) Rebuild KG
        items = self.kb.query("")
        # after texts build
        try:
            paper_samples = [(getattr(it, "get", lambda k: None)("title") if isinstance(it, dict) else None,
                            getattr(it, "get", lambda k: None)("url") if isinstance(it, dict) else None)
                            for it in items[:10]]
            print("🧾 Sample sources:")
            for t,u in paper_samples:
                if t or u:
                    print(f"   • {t or '(no title)'} — {u or '(no url)'}")
        except Exception:
            pass
        texts = [it["text"] if isinstance(it, dict) else str(it) for it in items]
        self.kg.build_from_corpus(texts)

        # 2) Merge similar
        try:
            print("🧩 Merging similar concepts...")
            ConceptMerger(self.kg).merge_similar_concepts(threshold=0.82)
        except Exception as e:
            print(f"⚠️ Merge skipped: {e}")

        # 3) Optional clustering
        try:
            if hasattr(ConceptMerger, "cluster_topics"):
                clusters = ConceptMerger(self.kg).cluster_topics()
                if clusters:
                    # print(f"🧠 Identified {len(clusters)} topic clusters:")
                    for i, cl in enumerate(clusters, 1):
                        preview = ", ".join(list(cl)[:1]) + ("..." if len(cl) > 1 else "")
                       # print(f"  Cluster {i}: {preview}")
        except Exception:
            pass

        # 4) Insights
        try:
            if GraphReasoner:
                reasoner = GraphReasoner(self.kg)
                for a, b in [("force", "motion"), ("energy", "work"), ("temperature", "heat")]:
                    print(f"🔎 Insight ({a}→{b}):", reasoner.explain_relation(a, b))
        except Exception:
            pass

        # 5) Persist
        try:
            self.kg.save(KG_PATH)
            print("💾 KG saved.")
        except Exception as e:
            print(f"⚠️ Could not save KG: {e}")

        if self.visualizer:
            try:
                self.visualizer.plot(self.kg, cycle=self.cycle)
                print(f"🧩 Saved KG visualization: {self.visualizer.last_path}")
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")

    def plan_next_steps(self) -> None:
        """Use the KG to decide the next topic and train online."""
        print("[Reflection] Evaluating next learning step...")
        topic = self._choose_next_topic()

        print(f"[Reflection] Planning next step: learn more about '{topic}'")
        raw = self.fetcher.fetch(topic)
        safe = self.safety.filter(raw)
        self.kb.store(safe)

        try:
            # ✅ Use incremental online training
            self.trainer.incremental_train(safe)
        except Exception as e:
            print(f"⚠️ Online training failed, retrying with full KB: {e}")
            self.trainer.incremental_train(self.kb.query(""))

        print(f"[Reflection] Finished updating with new knowledge on '{topic}' ✅")

    # ---------------------------------------------------------------------
    # Helpers (unchanged except training replaced)
    # ---------------------------------------------------------------------
    def _fetch_train_topic(self, topic: str) -> None:
        print(f"[Fetcher] Fetching new data for topic: {topic}")
        raw = self.fetcher.fetch(topic)
        safe = self.safety.filter(raw)
        self.kb.store(safe)
        print(f"🧠 Training on {len(safe)} knowledge items...")
        self.trainer.incremental_train(safe)  # ✅ changed

    def promote_validated(self, validated_list, save_path: str = "knowledge_graph/graph.json"):
        """Promote validated hypotheses into KG."""
        if not validated_list:
            return

        added = 0
        for v in validated_list:
            if not v.get("promote"):
                continue
            a, b, rel = v["subject"].lower(), v["object"].lower(), v["relation"].lower()
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
