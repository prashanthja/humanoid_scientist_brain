"""
Embedding Bridge — unified interface between transformer and reasoning/evaluation layers.
Now ensures consistent [N, D] sentence-level embeddings via mean pooling.
"""

from __future__ import annotations
from typing import List
import numpy as np


class EmbeddingBridge:
    def __init__(self, trainer_online):
        """
        trainer_online: OnlineTrainer instance that has .embed(texts)
        """
        self.trainer = trainer_online

    # ---------- unified embedding ----------
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Returns numpy array [N, D] — guaranteed sentence-level mean embeddings.
        """
        if not self.trainer:
            raise RuntimeError("EmbeddingBridge: trainer not initialized.")
        arr = self.trainer.embed(texts)

        # Mean-pool if sequence dimension still exists
        if arr.ndim == 3:
            arr = arr.mean(axis=1)

        # If single vector, keep consistent shape
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype("float32")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Alias for embed() for compatibility with reasoning modules."""
        return self.embed(texts)

    def get_vector(self, text: str) -> np.ndarray:
        """Return a single 1D vector for a single text input."""
        vec = self.embed([text])
        return vec[0] if vec.ndim > 1 else vec

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.ndim > 1:
            a = a[0]
        if b.ndim > 1:
            b = b[0]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)
