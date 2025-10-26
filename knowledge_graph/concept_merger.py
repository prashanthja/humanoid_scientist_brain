"""
Concept Merger — Phase B Step 2
---------------------------------
Merges semantically similar concepts in the Knowledge Graph
and clusters related topics for reflection & planning.
"""

import itertools
from collections import defaultdict


class ConceptMerger:
    def __init__(self, kg):
        self.kg = kg

    # ---------------------------------------------------------
    def merge_similar_concepts(self, threshold=0.8):
        """
        Merge concepts that are textually or semantically similar.
        Uses lexical overlap (or embedding similarity if available).
        """
        print("🧩 Merging similar concepts...")

        nodes = list(self.kg.graph.keys())
        merged = set()
        merge_map = {}

        for a, b in itertools.combinations(nodes, 2):
            if a in merged or b in merged:
                continue

            sim = self._similarity(a, b)
            if sim >= threshold:
                print(f"🔗 Merging '{a}' and '{b}' (similarity={sim:.2f})")
                self._merge_nodes(a, b)
                merged.add(b)
                merge_map[b] = a

        if merge_map:
            print(f"🧠 Merged {len(merge_map)} similar concepts in KG.")
        else:
            print("✅ No merges required at this cycle.")

    # ---------------------------------------------------------
    def cluster_topics(self, verbose=False):
        """
        Groups concepts into clusters based on textual overlap or shared subwords.
        Used by ReflectionEngine to decide next focus areas.
        """
        nodes = list(self.kg.graph.keys())
        clusters = []
        visited = set()

        for node in nodes:
            if node in visited:
                continue
            cluster = {node}
            for other in nodes:
                if other != node and self._similarity(node, other) > 0.5:
                    cluster.add(other)
                    visited.add(other)
            clusters.append(cluster)

        if clusters:
            if verbose:
                print(f"🧠 Identified {len(clusters)} topic clusters:")
                for i, cluster in enumerate(clusters, 1):
                    print(f"  Cluster {i}: {', '.join(cluster)}")
            else:
                print(f"🧠 {len(clusters)} topic clusters identified (details suppressed).")
        else:
            print("ℹ️ No distinct topic clusters formed this round.")

        self.kg.topic_clusters = clusters
        return clusters

    # ---------------------------------------------------------
    def _similarity(self, a, b):
        """Compute a lexical similarity between two concept names."""
        a_set, b_set = set(a.lower().split()), set(b.lower().split())
        overlap = len(a_set & b_set)
        union = len(a_set | b_set)
        return overlap / union if union else 0

    # ---------------------------------------------------------
    def _merge_nodes(self, a, b):
        """Merge all relationships and attributes of node b into node a."""
        G = getattr(self.kg, "G", None)

        if G is not None and hasattr(G, "neighbors"):
            if b not in G:
                return
            for nbr, attrs in list(G[b].items()):
                for rel, val in attrs.items():
                    G.add_edge(a, nbr, label=rel)
            G.remove_node(b)
        elif isinstance(self.kg.graph, dict):
            rels_b = self.kg.graph.get(b, {})
            if isinstance(rels_b, dict):
                for rel, objs in rels_b.items():
                    objs = objs if isinstance(objs, list) else [objs]
                    for obj in objs:
                        self.kg.add_relation(a, rel, obj)
            elif isinstance(rels_b, list):
                for obj in rels_b:
                    self.kg.add_relation(a, "related_to", obj)
            self.kg.graph.pop(b, None)
