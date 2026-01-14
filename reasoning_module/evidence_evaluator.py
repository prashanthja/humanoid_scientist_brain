# reasoning_module/evidence_evaluator.py
# ------------------------------------------------------------
# Advanced Evidence Evaluator with EmbeddingBridge Integration
# - Stable output schema (never missing keys)
# - Safe cache rebuild + embedding dim handling
# ------------------------------------------------------------

from __future__ import annotations
import os, re, json, time, hashlib
from typing import List, Dict, Any, Optional
import numpy as np

NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno evidence\b", r"\bdoes\s+not\b",
    r"\bfails?\s+to\b", r"\binconsistent\s+with\b",
    r"\bcontradict(s|ed|ion)?\b", r"\bdispro(ve|ven|ves)\b",
    r"\brefute(s|d|ation)?\b",
]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_text(x: Any) -> str:
    if isinstance(x, dict):
        return str(x.get("text") or x.get("paper_title") or "")
    return str(x or "")


def _hash_key(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()


def _polarity_is_contradiction(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in NEGATION_PATTERNS)


class EvidenceEvaluator:
    def __init__(
        self,
        kb,
        encoder,
        kg=None,
        cache_dir="data",
        log_path="logs/evidence_reports.jsonl",
        max_index_items: int = 5000,
        top_k: int = 10,
    ):
        self.kb, self.encoder, self.kg = kb, encoder, kg
        self.cache_dir, self.log_path = cache_dir, log_path
        self.idx_meta_path = os.path.join(cache_dir, "kb_embed_index.json")
        self.idx_vec_path = os.path.join(cache_dir, "kb_embeddings.npy")

        self.max_index_items = int(max_index_items)
        self.top_k = int(top_k)

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self._meta: List[Dict[str, Any]] = []
        self._emb: np.ndarray = np.zeros((0, 256), dtype=np.float32)
        self._dim: int = 256

        self._ensure_cache()

    # ==========================================================
    # Cache Handling
    # ==========================================================
    def _ensure_cache(self):
        try:
            if os.path.exists(self.idx_meta_path) and os.path.exists(self.idx_vec_path):
                with open(self.idx_meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f) or []
                self._emb = np.load(self.idx_vec_path)
                if self._emb.ndim != 2:
                    raise ValueError("Embeddings file is not 2D.")
                self._dim = int(self._emb.shape[1])

                # If meta/emb mismatch, rebuild
                if len(self._meta) != int(self._emb.shape[0]):
                    raise ValueError("Meta and embedding row count mismatch.")
            else:
                self._rebuild_cache()
        except Exception:
            self._rebuild_cache()

    def _rebuild_cache(self):
        items = self.kb.query("") or []
        texts, meta = [], []

        for it in items[: self.max_index_items]:
            text = _safe_text(it)
            if not isinstance(text, str) or not text.strip():
                continue

            key = None
            if isinstance(it, dict):
                key = it.get("id")
            if not key:
                key = _hash_key(text)

            meta.append({
                "key": key,
                "text": text[:1000],
                "source": (it.get("source", "") if isinstance(it, dict) else ""),
                "title": (it.get("paper_title", "") if isinstance(it, dict) else ""),
            })
            texts.append(text)

        # ---- CHUNKED ENCODE to avoid OOM ----
        vec_chunks = []
        chunk = 64  # safe default; lower if needed
        for i in range(0, len(texts), chunk):
            vec_chunks.append(self._encode(texts[i:i+chunk]))

        vecs = np.concatenate(vec_chunks, axis=0) if vec_chunks else np.zeros((0, self._dim), dtype=np.float32)

        # If encode returned empty, keep safe shape
        if vecs.size == 0:
            vecs = np.zeros((0, self._dim), dtype=np.float32)

        self._meta, self._emb = meta, vecs.astype("float32", copy=False)
        self._dim = int(self._emb.shape[1]) if self._emb.ndim == 2 else self._dim

        np.save(self.idx_vec_path, self._emb)
        with open(self.idx_meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2, ensure_ascii=False)

    # ==========================================================
    # Embedding Calls
    # ==========================================================
    def _encode(self, texts: List[str]) -> np.ndarray:
        clean = [t for t in texts if isinstance(t, str) and t.strip()]
        if not clean:
            return np.zeros((0, self._dim), dtype=np.float32)

        # EmbeddingBridge style
        if hasattr(self.encoder, "encode_texts"):
            arr = np.asarray(self.encoder.encode_texts(clean), dtype=np.float32)
            return arr

        # OnlineTrainer style
        if hasattr(self.encoder, "embed"):
            arr = np.asarray(self.encoder.embed(clean), dtype=np.float32)
            return arr

        # custom encoders
        if hasattr(self.encoder, "get_vector"):
            vecs = [np.asarray(self.encoder.get_vector(t), dtype=np.float32) for t in clean]
            return np.stack(vecs, axis=0)

        raise RuntimeError("Encoder does not expose encode_texts/embed/get_vector")

    # ==========================================================
    # Evaluation
    # ==========================================================
    def evaluate_batch(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        # Ensure cache exists and is not stale shape-wise
        if self._emb is None or self._emb.ndim != 2:
            self._rebuild_cache()

        for h in (hypotheses or []):
            base_hyp = ""
            if isinstance(h, dict):
                base_hyp = str(h.get("hypothesis") or "")
            else:
                base_hyp = str(h or "")

            try:
                res = self.evaluate_one(h if isinstance(h, dict) else {"hypothesis": base_hyp})
            except Exception as e:
                res = {
                    "hypothesis": base_hyp,
                    "verdict": "error",
                    "evidence_confidence": 0.0,
                    "evaluated_at": _now(),
                    "evidence_error": str(e),
                }

            # enforce stable schema
            res.setdefault("hypothesis", base_hyp)
            res.setdefault("verdict", "unknown")
            res.setdefault("evidence_confidence", 0.0)
            res.setdefault("evaluated_at", _now())
            out.append(res)

            # log each result (optional, but useful)
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
            except Exception:
                pass

        print("🧩 Evidence evaluation complete. Verdicts summary:")
        for x in out[:5]:
            hyp = str(x.get("hypothesis", ""))[:60]
            verdict = str(x.get("verdict", "unknown"))
            conf = float(x.get("evidence_confidence", 0.0))
            print(f"  - {hyp}... → {verdict} (conf={conf:.3f})")

        return out

    def evaluate_one(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        text = str(hypothesis.get("hypothesis") or "")
        if not text.strip():
            return {
                "hypothesis": text,
                "verdict": "invalid",
                "evidence_confidence": 0.0,
                "evaluated_at": _now(),
            }

        # If KB index empty, can't evaluate
        if self._emb is None or self._emb.shape[0] == 0:
            return {
                "hypothesis": text,
                "verdict": "no_index",
                "evidence_confidence": 0.0,
                "evaluated_at": _now(),
            }

        q = self._encode([text])
        if q.size == 0:
            return {
                "hypothesis": text,
                "verdict": "encode_failed",
                "evidence_confidence": 0.0,
                "evaluated_at": _now(),
            }

        vec = q[0]
        # Dimension mismatch? Rebuild cache with current encoder
        if vec.shape[0] != self._dim:
            self._rebuild_cache()
            if self._emb.shape[0] == 0:
                return {
                    "hypothesis": text,
                    "verdict": "no_index",
                    "evidence_confidence": 0.0,
                    "evaluated_at": _now(),
                }
            self._dim = int(self._emb.shape[1])
            vec = self._encode([text])[0]

        # cosine sim: normalize query only; assume _emb already float32
        denom = (np.linalg.norm(vec) + 1e-8)
        sims = (self._emb @ (vec / denom)).astype(np.float32, copy=False)

        k = min(self.top_k, sims.shape[0])
        top_idx = np.argsort(-sims)[:k]

        support, contra = 0.0, 0.0
        for i in top_idx:
            sim = float(sims[i])
            meta = self._meta[i] if i < len(self._meta) else {"text": ""}
            if _polarity_is_contradiction(meta.get("text", "")):
                contra += sim
            else:
                support += sim

        total = support + contra
        conf = 0.0 if total <= 1e-12 else (support / total)

        verdict = (
            "supported" if conf > 0.60 else
            "contradicted" if conf < 0.40 else
            "inconclusive"
        )

        return {
            "hypothesis": text,
            "verdict": verdict,
            "evidence_confidence": round(float(conf), 3),
            "evaluated_at": _now(),
        }
