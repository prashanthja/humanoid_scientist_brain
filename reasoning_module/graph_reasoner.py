# reasoning_module/graph_reasoner.py
class GraphReasoner:
    def __init__(self, kg):
        self.kg = kg

    def explain_relation(self, a: str, b: str) -> str:
        a, b = a.lower().strip(), b.lower().strip()
        if a == b: return f"‘{a}’ and ‘{b}’ are the same concept."
        rels = self.kg.get_relations(a)
        for rel, objs in rels.items():
            if b in objs:
                return f"{a} —{rel}→ {b}"
        # 2-hop
        for rel1, xs in rels.items():
            for x in xs:
                rels2 = self.kg.get_relations(x)
                for rel2, ys in rels2.items():
                    if b in ys:
                        return f"{a} —{rel1}→ {x} —{rel2}→ {b}"
        return f"No short path found between '{a}' and '{b}'."

    def suggest_transitive(self, a: str, relation: str):
        a, relation = a.lower().strip(), relation.lower().strip()
        out = []
        xs = self.kg.get_relations(a).get(relation, [])
        for x in xs:
            ys = self.kg.get_relations(x).get(relation, [])
            for y in ys:
                if y != a and y not in out:
                    out.append(y)
        return out
