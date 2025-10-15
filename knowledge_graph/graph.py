# knowledge_graph/graph.py
"""
KnowledgeGraph
Phase B — lightweight concept graph built from corpus.
- Stores graph as: dict[str, list[tuple[str, str]]] == {source: [(relation, target), ...]}
- Can load/save JSON
- Builds from text with simple patterning (Phase A/B)
- Finds weak concepts (low degree) for graph-driven expansion
"""

from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any


class KnowledgeGraph:
    def __init__(self):
        # { src: [(relation, dst), ...] }
        self.graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    # ---------- Persistence ----------
    def save(self, path: str):
        serializable = {k: list(v) for k, v in self.graph.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.graph = defaultdict(list, {k: [tuple(t) for t in v] for k, v in data.items()})

    # ---------- Introspection ----------
    def all_concepts(self) -> List[str]:
        nodes: Set[str] = set(self.graph.keys())
        for _, edge_list in self.graph.items():
            for _, t in edge_list:
                nodes.add(t)
        return sorted(nodes)

    def degree(self) -> Dict[str, int]:
        """Undirected-like degree: out + in counts."""
        deg: Dict[str, int] = {}
        for s, edges in self.graph.items():
            deg[s] = deg.get(s, 0) + len(edges)
            for _, t in edges:
                deg[t] = deg.get(t, 0) + 1
        return deg

    def _has_any_edge(self, a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        for rel, t in self.graph.get(a, []):
            if t.lower() == b:
                return True
        for rel, t in self.graph.get(b, []):
            if t.lower() == a:
                return True
        return False

    def _undirected_adj(self) -> Dict[str, Set[str]]:
        adj: Dict[str, Set[str]] = {}
        for s, edges in self.graph.items():
            adj.setdefault(s, set())
            for _, t in edges:
                adj.setdefault(t, set())
                adj[s].add(t)
                adj[t].add(s)
        return adj

    def _path_exists(self, a: str, b: str) -> bool:
        a, b = a.lower().strip(), b.lower().strip()
        if a == b:
            return True
        adj = self._undirected_adj()
        if a not in adj or b not in adj:
            return False
        seen = {a}
        q = [a]
        while q:
            nxt = []
            for u in q:
                for v in adj.get(u, ()):
                    if v == b:
                        return True
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            q = nxt
        return False

    def find_weak_concepts(self, threshold: int = 2) -> List[str]:
        """
        Return concept names whose (undirected) degree < threshold.
        Used by ReflectionEngine to target expansion.
        """
        deg = self.degree()
        return [n for n, d in deg.items() if d < threshold]

    # ---------- Build from corpus (simple patterns; Phase A/B) ----------
    def build_from_corpus(self, texts: List[str]):
        """
        Very simple relation extraction (Phase B baseline):
        - "X is Y"  -> X --is--> Y
        - "X is the study of Y" -> X --study_of--> Y
        - "X of Y"  -> X --of--> Y
        - Also pick short 'X causes Y' patterns
        """
        self.graph = defaultdict(list)  # rebuild fresh each time

        for text in texts:
            if not text:
                continue
            s = " ".join(text.strip().split())
            lowered = s.lower()

            # pattern 1: "X is the study of Y"
            m = re.findall(r"([a-zA-Z][\w\s\-]{1,40})\s+is the study of\s+([a-zA-Z][\w\s\-\s,]{1,80})", lowered)
            for a, b in m:
                src = a.strip()
                tgt = b.strip().strip(".")
                self._add_edge(src, "study_of", tgt)

            # pattern 2: "X is Y"
            m2 = re.findall(r"([a-zA-Z][\w\s\-]{1,40})\s+is\s+([a-zA-Z][\w\s\-]{1,60})", lowered)
            for a, b in m2:
                src = a.strip()
                tgt = b.strip().strip(".")
                if src != tgt:
                    self._add_edge(src, "is", tgt)

            # pattern 3: "X of Y"
            m3 = re.findall(r"([a-zA-Z][\w\s\-]{1,30})\s+of\s+([a-zA-Z][\w\s\-\s,]{1,80})", lowered)
            for a, b in m3:
                src = a.strip()
                tgt = b.strip().strip(".")
                if src != tgt:
                    self._add_edge(src, "of", tgt)

            # pattern 4: "X causes Y"
            m4 = re.findall(r"([a-zA-Z][\w\s\-]{1,40})\s+causes\s+([a-zA-Z][\w\s\-]{1,60})", lowered)
            for a, b in m4:
                src = a.strip()
                tgt = b.strip().strip(".")
                if src != tgt:
                    self._add_edge(src, "causes", tgt)

    def _add_edge(self, src: str, rel: str, tgt: str):
        src = src.strip()
        rel = rel.strip()
        tgt = tgt.strip()
        if not src or not tgt or not rel:
            return
        # de-duplicate exact edge
        if (rel, tgt) not in self.graph[src]:
            self.graph[src].append((rel, tgt))

    # ---------- Optional: no-op viz hook for older calls ----------
    def visualize(self) -> None:
        # kept as a stub (visualization handled by visualization/graph_progress.py)
        return

    def get_relations(self, concept):
        """
        Returns all relationships (edges) connected to a concept node.
        Supports both dict-based graph and NetworkX-based graph structures.
        """
        if hasattr(self, "G") and self.G is not None:
            if concept not in self.G:
                return []
            relations = []
            for neighbor, attrs in self.G[concept].items():
                for key, val in attrs.items():
                    rel_label = val if isinstance(val, str) else val.get("label", "")
                    relations.append((concept, rel_label, neighbor))
            return relations

        elif isinstance(self.graph, dict):
            rels = self.graph.get(concept, {})
            if isinstance(rels, dict):
                return [(concept, rel, obj) for rel, objs in rels.items() for obj in (objs if isinstance(objs, list) else [objs])]
            elif isinstance(rels, list):
                return [(concept, "related_to", obj) for obj in rels]
            else:
                return []
        else:
            return []
