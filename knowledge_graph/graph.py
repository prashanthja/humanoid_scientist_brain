"""
KnowledgeGraph (final unified version)
-------------------------------------
Supports both old list format and new dict-based relation structure.
"""

import json, re, os
from collections import defaultdict
from typing import Dict, List, Tuple, Set


class KnowledgeGraph:
    def __init__(self):
        # Structure: {subject: {relation: [objects]}}
        self.graph: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    # ---------- Persistence ----------
    def save(self, path: str = "knowledge_graph/graph.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = {s: {r: list(objs) for r, objs in rels.items()} for s, rels in self.graph.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: str = "knowledge_graph/graph.json"):
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            data = json.load(f)

        # Convert both old and new formats
        new_graph = defaultdict(lambda: defaultdict(list))
        if isinstance(data, list):  # old format: list of tuples
            for s, rel, t in data:
                new_graph[s][rel].append(t)
        elif isinstance(data, dict):
            for s, rels in data.items():
                if isinstance(rels, list):  # old inner structure
                    for rel, t in rels:
                        new_graph[s][rel].append(t)
                elif isinstance(rels, dict):
                    for rel, objs in rels.items():
                        if isinstance(objs, list):
                            new_graph[s][rel].extend(objs)
                        else:
                            new_graph[s][rel].append(str(objs))
        self.graph = new_graph

    # ---------- Relation management ----------
    def add_relation(self, subj: str, relation: str, obj: str):
        """Add a relation (safe for both formats)."""
        if not subj or not obj or not relation:
            return

        subj, relation, obj = subj.strip(), relation.strip(), obj.strip()
        # ensure nested dicts exist
        if subj not in self.graph:
            self.graph[subj] = defaultdict(list)
        if relation not in self.graph[subj]:
            self.graph[subj][relation] = []

        # add object if not duplicate
        if obj not in self.graph[subj][relation]:
            self.graph[subj][relation].append(obj)
            # add symmetric link for bidirectional relations
            if relation == "related_to":
                if obj not in self.graph:
                    self.graph[obj] = defaultdict(list)
                if subj not in self.graph[obj][relation]:
                    self.graph[obj][relation].append(subj)

        # print(f"✅ Relation added to KG: {subj} -[{relation}]-> {obj}")
        self.save()

    # ---------- Core utilities ----------
    def build_from_corpus(self, texts: List[str]):
        """Extract simple relations from text patterns."""
        for text in texts:
            if not text:
                continue
            s = text.lower().strip()
            for pat, rel in [
                (r"([a-zA-Z\s]+)\s+is the study of\s+([a-zA-Z\s]+)", "study_of"),
                (r"([a-zA-Z\s]+)\s+is\s+([a-zA-Z\s]+)", "is"),
                (r"([a-zA-Z\s]+)\s+of\s+([a-zA-Z\s]+)", "of"),
                (r"([a-zA-Z\s]+)\s+causes\s+([a-zA-Z\s]+)", "causes"),
            ]:
                for a, b in re.findall(pat, s):
                    self.add_relation(a.strip(), rel, b.strip())

    def get_relations(self, concept: str) -> Dict[str, List[str]]:
        """Return relations for a given concept."""
        return self.graph.get(concept, {})

    def all_concepts(self) -> List[str]:
        nodes = set(self.graph.keys())
        for rels in self.graph.values():
            for objs in rels.values():
                nodes.update(objs)
        return sorted(nodes)

    def edge_count(self) -> int:
        return sum(len(objs) for rels in self.graph.values() for objs in rels.values())
