# knowledge_graph/concept_merger.py
"""
Concept normalization and clustering for Knowledge Graph.
This helps the AI unify similar nodes and expand relationships.
"""

import re
import math
from collections import defaultdict, Counter
from difflib import SequenceMatcher

class ConceptMerger:
    def __init__(self, kg):
        self.kg = kg

    def normalize(self, text: str) -> str:
        """Normalize a concept (lowercase, remove stopwords, punctuation)."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def merge_similar_concepts(self, threshold: float = 0.8):
        """Merge similar nodes based on string similarity."""
        nodes = list(self.kg.graph.keys())
        merged = {}
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                if b in merged or a in merged:
                    continue
                sim = SequenceMatcher(None, a, b).ratio()
                if sim >= threshold:
                    merged[b] = a  # merge b into a

        # Apply merges
        for b, a in merged.items():
            if b in self.kg.graph:
                for rel, objs in self.kg.graph[b].items():
                    self.kg.graph[a][rel].update(objs)
                del self.kg.graph[b]

        print(f"🧩 Merged {len(merged)} similar concepts in Knowledge Graph.")

    def cluster_topics(self):
        """Cluster related nodes based on relation overlap."""
        clusters = defaultdict(set)
        for subj, rels in self.kg.graph.items():
            cluster_id = None
            for rel, objs in rels.items():
                for obj in objs:
                    for cid, members in clusters.items():
                        if subj in members or obj in members:
                            cluster_id = cid
                            break
                    if cluster_id:
                        break
                if cluster_id:
                    break
            if cluster_id is None:
                cluster_id = len(clusters) + 1
            clusters[cluster_id].add(subj)
            for rel, objs in rels.items():
                clusters[cluster_id].update(objs)

        print(f"🧠 Identified {len(clusters)} topic clusters:")
        for cid, members in clusters.items():
            print(f"  Cluster {cid}: {', '.join(list(members)[:6])}...")
        return clusters
