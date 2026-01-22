# retrieval/chunk_index.py
# ------------------------------------------------------------
# ChunkIndex — embed + retrieve chunk evidence
# - Saves {meta.json, vecs.npy}
# - Rebuild if dim mismatch
# - Uses EmbeddingBridge / trainer embeddings
# ------------------------------------------------------------

from __future__ import annotations
import os, json, time
from typing import List, Dict, Any
import numpy as np


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return (X / denom).astype(np.float32)


class ChunkIndex:
    def __init__(
        self,
        chunk_store,
        encoder,               # EmbeddingBridge or OnlineTrainer
        cache_dir: str = "data",
        max_items: int = 8000,
        chunk_batch: int = 64,
    ):
        self.chunk_store = chunk_store
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.max_items = int(max_items)
        self.chunk_batch = int(chunk_batch)

        os.makedirs(cache_dir, exist_ok=True)
        self.meta_path = os.path.join(cache_dir, "chunk_index_meta.json")
        self.vec_path = os.path.join(cache_dir, "chunk_index_vecs.npy")

        self.meta: List[Dict[str, Any]] = []
        self.vecs: np.ndarray = np.zeros((0, 256), dtype=np.float32)
        self.dim: int = 256

        self._load_or_rebuild()

    def _encode(self, texts: List[str]) -> np.ndarray:
        # EmbeddingBridge preferred
        if hasattr(self.encoder, "encode_texts"):
            arr = np.asarray(self.encoder.encode_texts(texts), dtype=np.float32)
            return arr
        if hasattr(self.encoder, "embed"):
            arr = np.asarray(self.encoder.embed(texts), dtype=np.float32)
            return arr
        raise RuntimeError("encoder must expose encode_texts() or embed()")

    def _load_or_rebuild(self):
        try:
            if os.path.exists(self.meta_path) and os.path.exists(self.vec_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f) or []
                self.vecs = np.load(self.vec_path)
                if self.vecs.ndim != 2:
                    raise ValueError("vecs not 2D")
                if len(self.meta) != int(self.vecs.shape[0]):
                    raise ValueError("meta/vec count mismatch")
                self.dim = int(self.vecs.shape[1])
                return
        except Exception:
            pass

        self.rebuild()

    def rebuild(self):
        items = self.chunk_store.fetch_recent(limit=self.max_items)
        texts = [x["text"] for x in items]
        meta = [{
            "chunk_id": x["chunk_id"],
            "paper_title": x.get("paper_title", ""),
            "source": x.get("source", ""),
            "text_preview": (x.get("text","")[:240]),
        } for x in items]

        vec_chunks = []
        for i in range(0, len(texts), self.chunk_batch):
            vec_chunks.append(self._encode(texts[i:i+self.chunk_batch]))

        vecs = np.concatenate(vec_chunks, axis=0) if vec_chunks else np.zeros((0, self.dim), dtype=np.float32)

        # fix dim
        if vecs.size > 0:
            self.dim = int(vecs.shape[1])
        else:
            vecs = np.zeros((0, self.dim), dtype=np.float32)

        self.meta = meta
        self.vecs = _normalize_rows(vecs)

        np.save(self.vec_path, self.vecs)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"built_at": _now(), "dim": self.dim, "items": self.meta}, f, indent=2, ensure_ascii=False)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.vecs is None or self.vecs.shape[0] == 0:
            return []

        qv = self._encode([query])
        if qv.size == 0:
            return []

        qv = _normalize_rows(qv)[0]

        # dim mismatch -> rebuild
        if qv.shape[0] != self.dim:
            self.rebuild()
            if self.vecs.shape[0] == 0:
                return []
            qv = _normalize_rows(self._encode([query]))[0]

        sims = self.vecs @ qv
        k = min(int(top_k), sims.shape[0])
        idx = np.argsort(-sims)[:k]

        out = []
        # meta file stored as {"items": [...]}; we loaded meta list directly for speed,
        # but if you loaded dict, handle it here:
        items_meta = self.meta
        for i in idx:
            m = items_meta[int(i)]
            out.append({
                "chunk_id": m["chunk_id"],
                "paper_title": m.get("paper_title",""),
                "source": m.get("source",""),
                "text_preview": m.get("text_preview",""),
                "similarity": float(sims[int(i)]),
            })
        return out
