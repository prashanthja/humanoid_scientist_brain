# visualization/graph_progress.py
"""
GraphProgressVisualizer — generates static snapshots and an animated GIF
from the evolving Knowledge Graph.
"""

import os
import glob
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import imageio.v2 as imageio


class GraphProgressVisualizer:
    def __init__(self, output_dir="visualization/graphs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = []  # optional: store per-cycle metrics if you wish

    def plot(self, kg, cycle_num=None):
        """
        Visualize the Knowledge Graph.
        Handles dict-based graphs like: {src: [(rel, tgt), ...]}
        or a prebuilt networkx graph with 'label' edge attrs.
        """
        try:
            # Build a proper DiGraph
            if isinstance(kg.graph, dict):
                G = nx.DiGraph()
                for src, edge_list in kg.graph.items():
                    for edge in edge_list:
                        if len(edge) == 2:
                            rel, tgt = edge
                        elif len(edge) == 3:
                            _, rel, tgt = edge
                        else:
                            continue
                        G.add_edge(src, tgt, label=rel)
            else:
                G = kg.graph

            plt.figure(figsize=(9, 6))
            pos = nx.spring_layout(G, seed=42, k=0.5)
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=1800,
                font_size=9,
                font_weight="bold",
                arrows=True,
                edge_color="gray",
            )
            labels = nx.get_edge_attributes(G, "label")
            if labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color="red", font_size=8)

            title = f"Knowledge Graph — Cycle {cycle_num or len(self.history)+1}"
            plt.title(title)
            filename = f"kg_cycle_{cycle_num or len(self.history)+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            print(f"🧩 Saved KG visualization: {path}")

        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")

    def create_animation(self, output_path="visualization/kg_evolution.gif", fps=1):
        """Create a GIF from all PNG snapshots in chronological order."""
        try:
            pngs = sorted(glob.glob(os.path.join(self.output_dir, "*.png")))
            if not pngs:
                print("[Visualizer] ⚠️ No snapshots found for animation.")
                return
            frames = [imageio.imread(p) for p in pngs]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"🎞️ Saved KG evolution GIF: {output_path}")
        except Exception as e:
            print(f"⚠️ Could not build animation: {e}")
