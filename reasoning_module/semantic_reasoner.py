# reasoning_module/semantic_reasoner.py
"""
Semantic Reasoner — Phase S Step 3
Bridges vector semantics (embeddings) with graph concepts.

Capabilities:
- Concept similarity using encoder embeddings
- Nearest-neighbor concept discovery inside the KG
- Soft relation plausibility scoring using semantic cues
"""

from typing import List, Tuple, Optional, Dict
import numpy as np


class SemanticReasoner:
    def __init__(self, encoder, kg=None):
        """
        encoder: your TextEncoder (must have get_vector(text) -> (64,) and encode_texts())
        kg: optional KnowledgeGraph (NetworkX G or dict graph)
        """
        self.encoder = encoder
        self.kg = kg

    # --- Graph plumbing --------------------------------------------------------
    def set_graph(self, kg):
        self.kg = kg

    def _all_nodes(self) -> List[str]:
        if self.kg is None:
            return []
        if hasattr(self.kg, "G") and self.kg.G is not None:
            return list(self.kg.G.nodes)
        if isinstance(self.kg.graph, dict):
            return list(self.kg.graph.keys())
        return []

    # --- Core semantics ---------------------------------------------------------
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-10
        nb = np.linalg.norm(b) + 1e-10
        return float(np.dot(a, b) / (na * nb))

    def similarity(self, a: str, b: str) -> float:
        va = self.encoder.get_vector(a)
        vb = self.encoder.get_vector(b)
        return self._cosine(va, vb)

    def nearest_concepts(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        nodes = self._all_nodes()
        if not nodes:
            return []
        v = self.encoder.get_vector(concept)
        sims = []
        for n in nodes:
            vn = self.encoder.get_vector(n)
            sims.append((n, self._cosine(v, vn)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def relation_plausibility(
        self, a: str, relation: str, b: str,
        relation_priors: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Lightweight plausibility:
        score = semantic_sim(a,b) * prior(relation) * context_boost
        """
        sim = self.similarity(a, b)
        pri = 1.0
        if relation_priors:
            pri = relation_priors.get(relation, 0.6)
        # simple context heuristic: if words overlap with relation tokens
        rel_words = set(relation.lower().split())
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        overlap = len((a_words | b_words) & rel_words)
        boost = 1.0 + 0.1 * overlap
        return float(sim * pri * boost)

    def best_relation(
        self, a: str, b: str, candidates: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        if candidates is None:
            candidates = ["causes", "part_of", "leads_to", "associated_with", "affects"]
        priors = {
            "causes": 0.9,
            "leads_to": 0.8,
            "affects": 0.75,
            "associated_with": 0.6,
            "part_of": 0.7,
        }
        scored = [
            (rel, self.relation_plausibility(a, rel, b, priors)) for rel in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0]

    def semantic_explain(self, a: str, b: str) -> str:
        """
        Human-readable semantic explanation suggestion.
        """
        rel, score = self.best_relation(a, b)
        nn_a = self.nearest_concepts(a, top_k=3)
        nn_b = self.nearest_concepts(b, top_k=3)
        return (
            f"Semantic similarity(a,b)={self.similarity(a,b):.3f}. "
            f"Most plausible relation: '{rel}' (score {score:.3f}). "
            f"Nearest to '{a}': {[n for n,_ in nn_a]}; "
            f"Nearest to '{b}': {[n for n,_ in nn_b]}."
        )
