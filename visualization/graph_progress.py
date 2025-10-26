"""
GraphProgressVisualizer
- Visualizes evolving Knowledge Graph state after each learning cycle.
- Auto-saves PNGs for dashboard display.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

class GraphProgressVisualizer:
    def __init__(self, output_dir="visualization/graphs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.last_path = None  # track last saved graph

    def plot(self, kg, cycle: int = 0):
        """
        Visualize the current state of the Knowledge Graph.
        Compatible with kg.graph in both list- and dict-based structures.
        """
        G = nx.DiGraph()

        # Handle both dict-of-dicts and dict-of-lists
        for src, rels in kg.graph.items():
            if isinstance(rels, list):  # format: [(relation, target), ...]
                for rel, tgt in rels:
                    G.add_edge(src, tgt, label=rel)
            elif isinstance(rels, dict):  # format: {relation: [targets]}
                for rel, tgts in rels.items():
                    for tgt in tgts:
                        G.add_edge(src, tgt, label=rel)

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=900)
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

        # edge labels (relation types)
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kg_cycle_{cycle}_{timestamp}.png"
        path = os.path.join(self.output_dir, filename)
        plt.title(f"Knowledge Graph — Cycle {cycle}")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        self.last_path = path
        print(f"🧩 Saved KG visualization: {path}")
        return path
