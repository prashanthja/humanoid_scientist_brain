"""
SimpleRetriever — Pinecone-backed (production) with numpy fallback (local dev)
"""
from __future__ import annotations
import os, logging, numpy as np
log = logging.getLogger("tattva.retriever")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_PATH  = os.path.join(ROOT, "data", "chunk_index_vecs.npy")
META_PATH = os.path.join(ROOT, "data", "chunk_index_meta.json")

_pinecone_index = None

def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX")
    if api_key and index_name:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=api_key)
            _pinecone_index = pc.Index(index_name)
            log.info("SimpleRetriever: Pinecone connected")
        except Exception as e:
            log.warning(f"Pinecone init failed: {e}")
    return _pinecone_index

class SimpleRetriever:
    def __init__(self, encoder=None):
        self.encoder = encoder
        self._vecs = None
        self._meta = None

    def _load_local(self):
        if self._vecs is None and os.path.exists(VEC_PATH):
            self._vecs = np.load(VEC_PATH)
            import json
            raw = json.load(open(META_PATH))
            self._meta = raw.get("items", raw) if isinstance(raw, dict) else raw
        return self._vecs is not None

    def query(self, text: str, top_k: int = 10) -> list[dict]:
        if not self.encoder:
            return []
        try:
            # Support both EmbeddingBridge and raw SentenceTransformer
            if hasattr(self.encoder, 'get_vector'):
                vec = self.encoder.get_vector(text)
            elif hasattr(self.encoder, 'encode'):
                vec = self.encoder.encode(text)
            else:
                return []
            pinecone_idx = _get_pinecone_index()
            if pinecone_idx is not None:
                results = pinecone_idx.query(vector=vec.tolist(), top_k=top_k, include_metadata=True)
                chunks = []
                for match in results.matches:
                    m = match.metadata or {}
                    chunks.append({
                        "text": m.get("text",""),
                        "paper_title": m.get("paper_title",""),
                        "source": m.get("source",""),
                        "domain": m.get("domain","transformer_efficiency"),
                        "score": match.score,
                    })
                return chunks
            else:
                if not self._load_local():
                    return []
                sims = np.dot(self._vecs, vec)
                top_idx = np.argsort(sims)[::-1][:top_k]
                chunks = []
                for idx in top_idx:
                    m = self._meta[idx] if idx < len(self._meta) else {}
                    chunks.append({
                        "text": m.get("text_preview") or m.get("text",""),
                        "paper_title": m.get("paper_title",""),
                        "source": m.get("source",""),
                        "domain": m.get("domain","transformer_efficiency"),
                        "score": float(sims[idx]),
                    })
                return chunks
        except Exception as e:
            log.error(f"SimpleRetriever.query: {e}")
            return []

    def retrieve(self, text: str, top_k: int = 10, use_mmr: bool = False, **kwargs) -> list[dict]:
        """Retrieve with causal relevance filtering."""
        # Get more candidates than needed, then filter
        candidates = self.query(text, top_k=top_k * 2)
        if not candidates:
            return []
        filtered = self._causal_filter(text, candidates, top_k)
        return filtered

    def _causal_filter(self, query: str, chunks: list, top_k: int) -> list:
        """Filter chunks by causal relevance to query — removes off-topic retrievals."""
        import re
        q_low = query.lower()

        # Extract key entities from query
        # e.g. "Does LoRA reduce fine-tuning cost?" → ["lora", "fine-tuning", "cost"]
        stopwords = {"does","is","are","the","a","an","of","in","on","to","for",
                     "with","and","or","not","it","this","that","by","from","how",
                     "what","which","do","can","will","would","could","should","may"}
        q_tokens = set(w for w in re.findall(r"[a-z0-9\-]+", q_low) if w not in stopwords and len(w) > 2)

        # Key concepts to match — at least one must appear in chunk
        scored = []
        for chunk in chunks:
            text_low = (chunk.get("text","") or "").lower()
            title_low = (chunk.get("paper_title","") or "").lower()
            combined = text_low + " " + title_low

            # Count query token matches in chunk
            matches = sum(1 for t in q_tokens if t in combined)
            causal_score = matches / max(len(q_tokens), 1)

            # Boost if paper title directly mentions query concept
            title_boost = 1.5 if any(t in title_low for t in q_tokens if len(t) > 4) else 1.0

            # Penalize if chunk has zero query token matches (off-topic)
            if matches == 0:
                causal_score = 0.0

            final_score = (chunk.get("score", 0.5) * 0.6 + causal_score * 0.4) * title_boost
            scored.append((final_score, chunk))

        # Sort by combined score, take top_k
        scored.sort(key=lambda x: -x[0])

        # Always return at least some results even if low relevance
        result = [c for _, c in scored[:top_k]]
        if not result:
            return chunks[:top_k]
        return result

    def rebuild(self):
        """Rebuild index from ChunkStore — uploads to Pinecone or saves numpy locally."""
        from knowledge_base.chunk_store import ChunkStore
        cs = ChunkStore()
        # all_chunks may not exist on older instances — use direct SQLite
        if hasattr(cs, 'all_chunks'):
            chunks = cs.all_chunks(limit=50000)
        else:
            import sqlite3
            conn = sqlite3.connect(cs.db_path)
            conn.row_factory = sqlite3.Row
            chunks = [dict(r) for r in conn.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()]
            conn.close()
        if not chunks:
            log.warning("No chunks to index")
            return
        log.info(f"Encoding {len(chunks)} chunks...")
        texts = [c.get("text","") for c in chunks]
        if hasattr(self.encoder, 'get_vector'):
            vecs = np.array([self.encoder.get_vector(t) for t in texts])
        else:
            vecs = np.array([self.encoder.encode(t) for t in texts])

        if _get_pinecone_index() is not None:
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_vecs = vecs[i:i+batch_size]
                batch_chunks = chunks[i:i+batch_size]
                vectors = []
                for j, (vec, chunk) in enumerate(zip(batch_vecs, batch_chunks)):
                    vectors.append({
                        "id": str(chunk.get("id", i+j)),
                        "values": vec.tolist(),
                        "metadata": {
                            "text": (chunk.get("text",""))[:500],
                            "paper_title": chunk.get("paper_title",""),
                            "source": chunk.get("source",""),
                            "domain": chunk.get("domain","transformer_efficiency"),
                        }
                    })
                _get_pinecone_index().upsert(vectors=vectors)
            log.info(f"Pinecone index rebuilt with {len(chunks)} vectors")
        else:
            np.save(VEC_PATH, vecs)
            import json
            meta = [{"text_preview": c.get("text","")[:240],
                     "paper_title": c.get("paper_title",""),
                     "source": c.get("source",""),
                     "domain": c.get("domain","")} for c in chunks]
            json.dump({"items": meta}, open(META_PATH,"w"))
            self._vecs = vecs
            self._meta = meta
            log.info(f"Local numpy index rebuilt with {len(chunks)} vectors")
