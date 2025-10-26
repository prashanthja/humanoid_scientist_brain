"""
Embedding Bridge — unified interface between transformer and reasoning/evaluation layers
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

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Standard embedding interface expected by reasoning/evidence modules.
        Returns numpy array of shape [N, D].
        """
        if not self.trainer:
            raise RuntimeError("EmbeddingBridge: trainer not initialized.")
        return self.trainer.embed(texts)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Alias for embed() for compatibility with other modules.
        """
        return self.embed(texts)

    def get_vector(self, text: str) -> np.ndarray:
        """
        Single-text embedding shortcut.
        """
        return self.embed([text])[0]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        """
        if a.ndim > 1: a = a[0]
        if b.ndim > 1: b = b[0]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)
