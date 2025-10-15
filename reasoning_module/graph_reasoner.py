# reasoning_module/graph_reasoner.py
from typing import List, Optional
import networkx as nx

class GraphReasoner:
    def __init__(self, kg, semantic=None):
        """
        kg: KnowledgeGraph
        semantic: Optional SemanticReasoner (for fallbacks)
        """
        self.kg = kg
        self.semantic = semantic

    def explain_relation(self, a, b):
        rels = self.kg.get_relations(a)
        if not rels:
            return f"No known relations for '{a}'."

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

        # Semantic fallback (Phase S Step 3)
        if self.semantic:
            return "[Semantic] " + self.semantic.semantic_explain(a, b)

        return f"No short path found between '{a}' and '{b}'."

    def suggest_transitive(self, a, relation):
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

        # If nothing found, semantic guess (rank neighbors by similarity)
        if not results and self.semantic:
            # Propose nearest concepts to mid nodes
            guesses = []
            for mid in next_nodes:
                guesses.extend([n for n,_ in self.semantic.nearest_concepts(mid, top_k=3)])
            # Deduplicate
            results = list(dict.fromkeys(guesses))

        return list(set(results))

    def find_path(self, a, b, max_depth: int = 3) -> Optional[List[str]]:
        try:
            if hasattr(self.kg, "G") and self.kg.G is not None:
                path = nx.shortest_path(self.kg.G, source=a, target=b)
                if len(path) - 1 <= max_depth:
                    return path
        except Exception:
            pass
        return None
