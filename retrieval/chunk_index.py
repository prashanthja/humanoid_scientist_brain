# retrieval/chunk_index.py
# ------------------------------------------------------------
# ChunkIndex — embed + retrieve chunk evidence
# - Saves meta.json and vecs.npy
# - Meta format: {"built_at","dim","count","items":[...]}
# - Rebuild if dim mismatch or meta/vec mismatch
# - Uses EmbeddingBridge.encode_texts OR OnlineTrainer.embed
# ------------------------------------------------------------

from __future__ import annotations
import os
import json
import time
from typing import List, Dict, Any
import numpy as np


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32)
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
        embed_max_len: int = 256,   # keep consistent across index + query
    ):
        self.chunk_store = chunk_store
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.max_items = int(max_items)
        self.chunk_batch = int(chunk_batch)
        self.embed_max_len = int(embed_max_len)

        os.makedirs(cache_dir, exist_ok=True)
        self.meta_path = os.path.join(cache_dir, "chunk_index_meta.json")
        self.vec_path = os.path.join(cache_dir, "chunk_index_vecs.npy")

        self.items: List[Dict[str, Any]] = []
        self.vecs: np.ndarray = np.zeros((0, 256), dtype=np.float32)
        self.dim: int = 256

        self._load_or_rebuild()

    def _encode(self, texts: List[str]) -> np.ndarray:
        texts = [str(t or "") for t in texts]
        # EmbeddingBridge preferred
        if hasattr(self.encoder, "encode_texts"):
            arr = np.asarray(
                self.encoder.encode_texts(
                    texts,
                    max_len=self.embed_max_len,
                    batch_size=self.chunk_batch,
                    force_cpu=False,
                ),
                dtype=np.float32,
            )
            return arr

        if hasattr(self.encoder, "embed"):
            # OnlineTrainer.embed supports max_len kwarg in your code
            arr = np.asarray(self.encoder.embed(texts, max_len=self.embed_max_len), dtype=np.float32)
            return arr
        raise RuntimeError("encoder must expose encode_texts() or embed()")

    def _load_or_rebuild(self):
        try:
            if os.path.exists(self.meta_path) and os.path.exists(self.vec_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    payload = json.load(f) or {}

                vecs = np.load(self.vec_path)
                if vecs.ndim != 2:
                    raise ValueError("vecs not 2D")

                if isinstance(payload, dict) and "items" in payload:
                    items = payload.get("items", []) or []
                    dim = int(payload.get("dim", int(vecs.shape[1])))
                else:
                    # backward compatibility if older file saved list directly
                    items = payload if isinstance(payload, list) else []
                    dim = int(vecs.shape[1])

                if len(items) != int(vecs.shape[0]):
                    raise ValueError("meta/vec mismatch")

                self.items = items
                self.vecs = vecs.astype(np.float32)
                self.dim = dim
                return
        except Exception:
            pass

        self.rebuild()

    def rebuild(self):
        items = self.chunk_store.fetch_recent(limit=self.max_items)

        texts = [x["text"] for x in items]
        meta_items = [{
            "chunk_id": int(x["chunk_id"]),
            "paper_title": x.get("paper_title", "") or "",
            "source": x.get("source", "") or "",
            "text_preview": (x.get("text", "") or "")[:240],
        } for x in items]

        vec_chunks: List[np.ndarray] = []
        for i in range(0, len(texts), self.chunk_batch):
            vec_chunks.append(self._encode(texts[i:i + self.chunk_batch]))

        vecs = np.concatenate(vec_chunks, axis=0) if vec_chunks else np.zeros((0, self.dim), dtype=np.float32)

        if vecs.size > 0:
            self.dim = int(vecs.shape[1])
        else:
            vecs = np.zeros((0, self.dim), dtype=np.float32)

        self.items = meta_items
        self.vecs = _normalize_rows(vecs)

        np.save(self.vec_path, self.vecs)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "built_at": _now(),
                "dim": int(self.dim),
                "count": int(len(self.items)),
                "items": self.items,
            }, f, indent=2, ensure_ascii=False)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.vecs is None or self.vecs.shape[0] == 0:
            return []

        qv = self._encode([query])
        if qv.size == 0:
            return []

        qv = _normalize_rows(qv)[0]

        # dim mismatch -> rebuild
        if int(qv.shape[0]) != int(self.dim):
            self.rebuild()
            if self.vecs.shape[0] == 0:
                return []
            qv = _normalize_rows(self._encode([query]))[0]

        sims = self.vecs @ qv
        k = min(int(top_k), int(sims.shape[0]))
        idx = np.argsort(-sims)[:k]

        out: List[Dict[str, Any]] = []
        for i in idx:
            m = self.items[int(i)]
            out.append({
                "chunk_id": m["chunk_id"],
                "paper_title": m.get("paper_title", ""),
                "source": m.get("source", ""),
                "text_preview": m.get("text_preview", ""),
                "similarity": float(sims[int(i)]),
            })
        return out
