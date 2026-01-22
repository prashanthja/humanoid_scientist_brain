"""
Embedding Bridge — unified interface between transformer and reasoning/evaluation layers.
Now supports eval knobs (batch_size/max_len/force_cpu) and guarantees [N, D] outputs.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np


class EmbeddingBridge:
    def __init__(self, trainer_online):
        """
        trainer_online: OnlineTrainer instance that has .embed(texts, max_len=..., batch_size=..., force_cpu=...)
        """
        self.trainer = trainer_online

    def embed(
        self,
        texts: List[str],
        *,
        batch_size: int = 16,
        max_len: int = 192,
        force_cpu: bool = False,
    ) -> np.ndarray:
        """
        Returns numpy array [N, D] — sentence-level embeddings.
        Passes through eval knobs to OnlineTrainer.embed().
        """
        if not self.trainer:
            raise RuntimeError("EmbeddingBridge: trainer not initialized.")

        arr = self.trainer.embed(
            texts,
            batch_size=int(batch_size),
            max_len=int(max_len),
            force_cpu=bool(force_cpu),
        )

        arr = np.asarray(arr, dtype=np.float32)

        # Mean-pool if sequence dimension exists (defensive)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return arr.astype("float32", copy=False)

    def encode_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """Alias for embed() for compatibility."""
        return self.embed(texts, **kwargs)

    def get_vector(self, text: str, **kwargs) -> np.ndarray:
        vec = self.embed([text], **kwargs)
        return vec[0]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.ndim > 1:
            a = a[0]
        if b.ndim > 1:
            b = b[0]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)
