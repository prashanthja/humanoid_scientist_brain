# knowledge_graph/concept_merger.py
"""
Concept normalization & merging + simple topic clustering.
Helps reduce duplicates like "physics", "physics is the study".
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class ConceptMerger:
    def __init__(self, kg):
        self.kg = kg

    def merge_similar_concepts(self, threshold: float = 0.82):
        nodes = list(self.kg.graph.keys())
        merged = {}
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                if b in merged or a in merged:
                    continue
                aa = _norm(a)
                bb = _norm(b)
                sim = SequenceMatcher(None, aa, bb).ratio()
                if sim >= threshold or aa in bb or bb in aa:
                    merged[b] = a  # merge b into a

        # apply merges
        for b, a in merged.items():
            if b in self.kg.graph:
                for rel, objs in self.kg.graph[b].items():
                    self.kg.graph[a][rel].update(objs)
                del self.kg.graph[b]

        print(f"🧩 Merged {len(merged)} similar concepts in KG.")

    def cluster_topics(self):
        # very simple connectivity-based clustering for preview
        clusters = []
        seen = set()

        def dfs(node, bag):
            seen.add(node)
            bag.add(node)
            for rel, objs in self.kg.graph[node].items():
                for o in objs:
                    if o not in seen and o in self.kg.graph:
                        dfs(o, bag)

        for n in list(self.kg.graph.keys()):
            if n in seen: continue
            bag = set()
            dfs(n, bag)
            clusters.append(bag)

        print(f"🧠 Identified {len(clusters)} topic clusters:")
        for i, c in enumerate(clusters[:5], 1):
            preview = ", ".join(list(c)[:6])
            print(f"  Cluster {i}: {preview}...")
        return clusters
