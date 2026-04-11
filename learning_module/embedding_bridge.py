from __future__ import annotations
from typing import List
import numpy as np
import os

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_CACHE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models", "sentence_transformer")

class EmbeddingBridge:
    def __init__(self, trainer_online=None):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            os.makedirs(MODEL_CACHE, exist_ok=True)
            self._model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)
        return self._model

    def embed(self, texts, *, batch_size=32, max_len=256, force_cpu=False):
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vecs = self._get_model().encode(
            texts, batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        return arr

    def encode_texts(self, texts, **kwargs): return self.embed(texts, **kwargs)
    def get_vector(self, text, **kwargs): return self.embed([text], **kwargs)[0]
    def cosine_similarity(self, a, b):
        if a.ndim > 1: a = a[0]
        if b.ndim > 1: b = b[0]
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8))
