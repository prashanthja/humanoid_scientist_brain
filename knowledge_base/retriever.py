import numpy as np
from typing import List, Tuple
from embedding.encoder import TextEncoder
from .storage import KnowledgeStorage


class Retriever:
    def __init__(self, storage: KnowledgeStorage, encoder: TextEncoder):
        self.storage = storage
        self.encoder = encoder

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray):
        # a: (d,), b: (n, d) or (d,)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        scores = np.dot(b, a)  # (n,)
        return scores  # since vectors are normalized, dot = cosine

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Returns list of (id, content, score) sorted by descending score.
        Automatically handles string or list input.
        """
        # --- Fix: Ensure query is always a list ---
        if isinstance(query, str):
            query = [query]

        # encode query text(s)
        q_vec = self.encoder.encode_texts(query)[0]  # shape (d,)

        # fetch all stored embeddings
        all_items = self.storage.fetch_all_with_embeddings()  # list of (id, content, vec)
        if not all_items:
            return []

        ids, contents, vecs = [], [], []
        for k_id, content, vec in all_items:
            if vec is not None:
                ids.append(k_id)
                contents.append(content)
                vecs.append(vec)

        if not vecs:
            return []

        vecs = np.vstack(vecs)  # (n, d)
        # cosine similarity (assuming normalized)
        scores = np.dot(vecs, q_vec)  # (n,)
        # top k
        idx = np.argsort(-scores)[:top_k]
        results = [(ids[i], contents[i], float(scores[i])) for i in idx]

        return results
