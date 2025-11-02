# reasoning_module/evidence_evaluator.py
# ------------------------------------------------------------
# Advanced Evidence Evaluator with EmbeddingBridge Integration
# ------------------------------------------------------------

from __future__ import annotations
import os, re, json, time, hashlib
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno evidence\b", r"\bdoes\s+not\b",
    r"\bfails?\s+to\b", r"\binconsistent\s+with\b",
    r"\bcontradict(s|ed|ion)?\b", r"\bdispro(ve|ven|ves)\b",
    r"\brefute(s|d|ation)?\b",
]


def _now() -> str: return time.strftime("%Y-%m-%d %H:%M:%S")
def _safe_text(x: Any) -> str:
    if isinstance(x, dict):
        return str(x.get("text") or x.get("paper_title") or "")
    return str(x or "")
def _hash_key(t: str) -> str: return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()
def _polarity_is_contradiction(text: str) -> bool:
    t = text.lower(); return any(re.search(p, t) for p in NEGATION_PATTERNS)
def _cosine(a, b): return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-8)*(np.linalg.norm(b)+1e-8)))


class EvidenceEvaluator:
    def __init__(self, kb, encoder, kg=None, cache_dir="data", log_path="logs/evidence_reports.jsonl"):
        self.kb, self.encoder, self.kg = kb, encoder, kg
        self.cache_dir, self.log_path = cache_dir, log_path
        self.idx_meta_path = os.path.join(cache_dir, "kb_embed_index.json")
        self.idx_vec_path = os.path.join(cache_dir, "kb_embeddings.npy")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._meta, self._emb, self._dim = [], None, None
        self._ensure_cache()

    # ==========================================================
    # Cache Handling
    # ==========================================================
    def _ensure_cache(self):
        try:
            if os.path.exists(self.idx_meta_path) and os.path.exists(self.idx_vec_path):
                with open(self.idx_meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                self._emb = np.load(self.idx_vec_path)
                self._dim = self._emb.shape[1]
            else:
                self._rebuild_cache()
        except Exception:
            self._rebuild_cache()

    def _rebuild_cache(self):
        items = self.kb.query("") or []
        texts, meta = [], []
        for it in items[:1000]:
            text = _safe_text(it)
            if not text:
                continue
            key = it.get("id") if isinstance(it, dict) and "id" in it else _hash_key(text)
            meta.append({"key": key, "text": text[:1000], "source": it.get("source", ""), "title": it.get("paper_title", "")})
            texts.append(text)
        vecs = self._encode(texts)
        self._meta, self._emb = meta, vecs
        self._dim = self._emb.shape[1] if self._emb.size else 256
        np.save(self.idx_vec_path, self._emb.astype("float32"))
        with open(self.idx_meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2)

    # ==========================================================
    # Embedding Calls
    # ==========================================================
    def _encode(self, texts: List[str]) -> np.ndarray:
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return np.zeros((0, self._dim or 256), dtype=np.float32)
        if hasattr(self.encoder, "encode_texts"):
            return np.asarray(self.encoder.encode_texts(texts), dtype=np.float32)
        if hasattr(self.encoder, "embed"):
            return np.asarray(self.encoder.embed(texts), dtype=np.float32)
        if hasattr(self.encoder, "get_vector"):
            vecs = [np.asarray(self.encoder.get_vector(t), dtype=np.float32) for t in texts]
            return np.stack(vecs, axis=0)
        raise RuntimeError("Encoder does not expose encode_texts/embed/get_vector")

    # ==========================================================
    # Evaluation
    # ==========================================================
    def evaluate_batch(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for h in hypotheses:
            try:
                out.append(self.evaluate_one(h))
            except Exception as e:
                bad = dict(h)
                bad["evidence_error"] = str(e)
                out.append(bad)
        print("🧩 Evidence evaluation complete. Verdicts summary:")
        for x in out[:5]:
            print(f"  - {x['hypothesis'][:60]}... → {x['verdict']} (conf={x.get('evidence_confidence',0):.3f})")
        return out

    def evaluate_one(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        text = hypothesis.get("hypothesis", "")
        if not text.strip():
            return {"verdict": "invalid", "evidence_confidence": 0.0}
        vec = self._encode([text])[0]
        sims = self._emb @ (vec / (np.linalg.norm(vec) + 1e-8))
        top_idx = np.argsort(-sims)[:10]
        support, contra = 0.0, 0.0
        for i in top_idx:
            sim = sims[i]
            meta = self._meta[i]
            if _polarity_is_contradiction(meta["text"]):
                contra += sim
            else:
                support += sim
        total = support + contra
        if total == 0:
            conf = 0.0
        else:
            conf = support / total
        verdict = "supported" if conf > 0.6 else "contradicted" if conf < 0.4 else "inconclusive"
        return {"hypothesis": text, "verdict": verdict, "evidence_confidence": round(conf, 3), "evaluated_at": _now()}
