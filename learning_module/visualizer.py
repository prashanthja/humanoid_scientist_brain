# learning_module/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import time

class TrainingVisualizer:
    """Handles live visualization and stats tracking for learning cycles."""

    def __init__(self):
        self.loss_history = []
        self.similarity_history = []

    def log_loss(self, losses):
        """Append loss values from each training epoch."""
        self.loss_history.extend(losses)

    def log_similarity(self, sim_score):
        """Track how similar new embeddings are to old ones."""
        self.similarity_history.append(sim_score)

    def plot_progress(self):
        """Plot training loss and embedding similarity over time."""
        plt.figure(figsize=(10, 4))

        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, color='orange')
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        # Similarity curve
        plt.subplot(1, 2, 2)
        plt.plot(self.similarity_history, color='blue')
        plt.title("Embedding Drift (Similarity)")
        plt.xlabel("Cycle")
        plt.ylabel("Cosine Similarity")

        plt.tight_layout()
        plt.show()

    def display_cycle_stats(self, cycle, avg_loss, sim):
        print(f"📊 Cycle {cycle}: Avg Loss={avg_loss:.4f}, Embedding Similarity={sim:.3f}")
        time.sleep(0.3)
