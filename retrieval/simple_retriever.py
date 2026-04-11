"""
Simple direct retriever — bypasses all gates in ChunkIndex.
Uses pure cosine similarity on the prebuilt numpy vectors.
"""
from __future__ import annotations
import json, os, numpy as np
from typing import List, Dict, Any

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_PATH  = os.path.join(ROOT, "data", "chunk_index_vecs.npy")
META_PATH = os.path.join(ROOT, "data", "chunk_index_meta.json")

class SimpleRetriever:
    def __init__(self, encoder):
        self.encoder = encoder
        self._vecs  = None
        self._items = None

    def _load(self):
        if self._vecs is None:
            self._vecs  = np.load(VEC_PATH).astype(np.float32)
            meta        = json.load(open(META_PATH))
            self._items = meta["items"]

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        self._load()
        qv   = self.encoder.embed([query])[0].astype(np.float32)
        norm = np.linalg.norm(qv)
        if norm > 0: qv = qv / norm
        sims = self._vecs @ qv
        top  = np.argsort(-sims)[:top_k]
        results = []
        for i in top:
            item = self._items[int(i)]
            results.append({
                "chunk_id":    item.get("chunk_id"),
                "paper_title": item.get("paper_title", ""),
                "source":      item.get("source", ""),
                "text":        item.get("text_preview", ""),
                "text_preview":item.get("text_preview", ""),
                "similarity":  float(sims[i]),
                "sim_embedding": float(sims[i]),
                "similarity_to_question": float(sims[i]),
            })
        return results

    def rebuild(self):
        """Rebuild index — called by background service."""
        from knowledge_base.chunk_store import ChunkStore
        cs   = ChunkStore()
        raw  = cs.fetch_recent(limit=50000)
        texts = [r["text"] for r in raw if r.get("text","").strip()]
        items = [{
            "chunk_id":    r["chunk_id"],
            "paper_title": r.get("paper_title",""),
            "source":      r.get("source",""),
            "text_preview": r.get("text","")[:500],
        } for r in raw if r.get("text","").strip()]

        # Encode in batches
        all_vecs = []
        batch = 64
        for i in range(0, len(texts), batch):
            vecs = self.encoder.embed(texts[i:i+batch])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            all_vecs.append(vecs / norms)

        vecs = np.concatenate(all_vecs, axis=0).astype(np.float32)
        np.save(VEC_PATH, vecs)
        with open(META_PATH, "w") as f:
            json.dump({"items": items, "dim": vecs.shape[1], "count": len(items)}, f)

        self._vecs  = vecs
        self._items = items
        print(f"Index rebuilt: {len(items)} chunks, dim={vecs.shape[1]}")
