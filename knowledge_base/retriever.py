import hashlib
import numpy as np
from typing import List, Tuple
from embedding.encoder import TextEncoder
from .storage import KnowledgeStorage


def _clean(t: str) -> str:
    t = (t or "").strip()
    return " ".join(t.split())


def _hash(t: str) -> str:
    return hashlib.sha1(_clean(t).lower().encode("utf-8", errors="ignore")).hexdigest()


class Retriever:
    def __init__(self, storage: KnowledgeStorage, encoder: TextEncoder):
        self.storage = storage
        self.encoder = encoder

    def semantic_search(self, query: str, top_k: int = 5, dedupe: bool = True) -> List[Tuple[int, str, float]]:
        """
        Returns list of (id, content, score) sorted by descending score.
        Dedupe prevents the same content appearing multiple times.
        """
        q = [query] if isinstance(query, str) else list(query)
        q_vec = self.encoder.encode_texts(q)[0]  # (d,)

        all_items = self.storage.fetch_all_with_embeddings()  # list of (id, content, vec)
        if not all_items:
            return []

        ids, contents, vecs = [], [], []
        seen = set()

        for k_id, content, vec in all_items:
            if vec is None:
                continue
            content = _clean(content)
            if not content:
                continue
            if dedupe:
                h = _hash(content)
                if h in seen:
                    continue
                seen.add(h)

            ids.append(k_id)
            contents.append(content)
            vecs.append(vec)

        if not vecs:
            return []

        vecs = np.vstack(vecs).astype(np.float32)  # (n, d)

        # If vectors are not normalized, normalize once here (safe)
        denom = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        vecs = vecs / denom

        q_norm = np.linalg.norm(q_vec) + 1e-8
        q_vec = q_vec / q_norm

        scores = np.dot(vecs, q_vec)  # (n,)
        k = min(int(top_k), int(scores.shape[0]))
        idx = np.argsort(-scores)[:k]
        return [(ids[i], contents[i], float(scores[i])) for i in idx]
