# reasoning_module/graph_reasoner.py
class GraphReasoner:
    def __init__(self, kg):
        self.kg = kg

    def explain_relation(self, a, b):
        """
        Finds and explains the relationship between two concepts.
        Supports list-based relation outputs from KnowledgeGraph.get_relations().
        """
        rels = self.kg.get_relations(a)

        if not rels:
            return f"No known relations for '{a}'."

        # Handle both dict and list relation formats
        if isinstance(rels, dict):
            for rel, objs in rels.items():
                if b in objs:
                    return f"'{a}' {rel} '{b}'"
        elif isinstance(rels, list):
            for (src, rel, tgt) in rels:
                if tgt == b:
                    return f"'{a}' {rel} '{b}'"

        # If direct link not found, attempt indirect path
        path = self.find_path(a, b)
        if path:
            return f"Indirect relation found: {' → '.join(path)}"

        return f"No short path found between '{a}' and '{b}'."

    def suggest_transitive(self, a, relation):
        """
        Finds transitive reasoning chains like:
        if A causes B and B causes C → then A may cause C.
        Compatible with list-based get_relations().
        """
        results = []
        rels = self.kg.get_relations(a)

        # Handle both dict-based and list-based graph structures
        next_nodes = []
        if isinstance(rels, dict):
            next_nodes = rels.get(relation, [])
        elif isinstance(rels, list):
            next_nodes = [tgt for (src, rel, tgt) in rels if rel == relation]

        for mid in next_nodes:
            mid_rels = self.kg.get_relations(mid)
            if isinstance(mid_rels, dict):
                results.extend(mid_rels.get(relation, []))
            elif isinstance(mid_rels, list):
                results.extend([tgt for (src, rel, tgt) in mid_rels if rel == relation])

        return list(set(results))
