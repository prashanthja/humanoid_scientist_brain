# reasoning_module/graph_reasoner.py
"""
GraphReasoner
--------------
Combines symbolic and semantic reasoning:
- Uses KnowledgeGraph edges and NetworkX paths for explicit reasoning.
- Uses transformer embeddings (via EmbeddingBridge) for semantic inference.
"""

from typing import List, Optional
import numpy as np
import networkx as nx
from learning_module.embedding_bridge import EmbeddingBridge


class GraphReasoner:
    def __init__(self, kg, online_trainer=None, semantic=None):
        """
        kg : KnowledgeGraph
        online_trainer : Optional[OnlineTrainer] → for semantic embeddings
        semantic : Optional SemanticReasoner (legacy)
        """
        self.kg = kg
        self.semantic = semantic
        self.bridge = EmbeddingBridge(online_trainer) if online_trainer else None

    # ---------------------------------------------------------------------
    #  Semantic reasoning core
    # ---------------------------------------------------------------------
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using transformer embeddings."""
        if not self.bridge or not text1 or not text2:
            return 0.0
        v1 = self.bridge.encode(text1)
        v2 = self.bridge.encode(text2)
        if np.all(v1 == 0) or np.all(v2 == 0):
            return 0.0
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return round(sim, 3)

    def semantic_explain(self, a: str, b: str) -> str:
        sim = self.semantic_similarity(a, b)
        if sim > 0.7:
            return f"{a} and {b} are semantically related (sim={sim})"
        elif sim > 0.4:
            return f"{a} and {b} are weakly related (sim={sim})"
        else:
            return f"No strong semantic link (sim={sim})"

    # ---------------------------------------------------------------------
    #  Explicit reasoning (from KG)
    # ---------------------------------------------------------------------
    def explain_relation(self, a, b):
        """Explain how 'a' and 'b' are connected in the KG or semantically."""
        rels = self.kg.get_relations(a)
        if not rels:
            # fallback to semantic reasoning
            return self.semantic_explain(a, b)

        # dict format
        if isinstance(rels, dict):
            for rel, objs in rels.items():
                if b in objs:
                    return f"'{a}' {rel} '{b}'"
        # list format: (src, rel, tgt)
        elif isinstance(rels, list):
            for (src, rel, tgt) in rels:
                if tgt == b:
                    return f"'{a}' {rel} '{b}'"

        # No direct link ⇒ try shortest path
        path = self.find_path(a, b)
        if path:
            return f"Indirect relation found: {' → '.join(path)}"

        # No explicit path ⇒ semantic reasoning
        return self.semantic_explain(a, b)

    def suggest_transitive(self, a, relation):
        """Suggest transitive or semantically related nodes."""
        results = []
        rels = self.kg.get_relations(a)

        # Collect first hop
        next_nodes = []
        if isinstance(rels, dict):
            next_nodes = rels.get(relation, [])
        elif isinstance(rels, list):
            next_nodes = [tgt for (src, rel, tgt) in rels if rel == relation]

        # Second hop
        for mid in next_nodes:
            mid_rels = self.kg.get_relations(mid)
            if isinstance(mid_rels, dict):
                results.extend(mid_rels.get(relation, []))
            elif isinstance(mid_rels, list):
                results.extend([tgt for (src, rel, tgt) in mid_rels if rel == relation])

        # If nothing found, semantic guesses
        if not results and self.bridge:
            guesses = []
            all_nodes = list(self.kg.graph.keys()) if hasattr(self.kg, "graph") else []
            for node in all_nodes:
                sim = self.semantic_similarity(a, node)
                if sim > 0.6:
                    guesses.append((node, sim))
            guesses.sort(key=lambda x: x[1], reverse=True)
            results = [n for n, _ in guesses[:5]]

        return list(set(results))

    def find_path(self, a, b, max_depth: int = 3) -> Optional[List[str]]:
        """Shortest path (≤ max_depth) if exists."""
        try:
            if hasattr(self.kg, "G") and self.kg.G is not None:
                path = nx.shortest_path(self.kg.G, source=a, target=b)
                if len(path) - 1 <= max_depth:
                    return path
        except Exception:
            pass
        return None
