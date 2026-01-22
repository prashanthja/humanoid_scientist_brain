# reasoning_module/evidence_evaluator.py
# ------------------------------------------------------------
# Evidence Evaluator (Robust + Incremental Cache)
# - Stable output schema (never missing keys)
# - Hard reset on mismatch/corruption
# - Incremental embedding index updates (no full rebuild each run)
# - True cosine similarity (normalized index)
# - Chunked encoding with eval-safe settings
# ------------------------------------------------------------

from __future__ import annotations
import os, re, json, time, hashlib
from typing import List, Dict, Any, Optional, Tuple
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


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32, copy=False)


class EvidenceEvaluator:
    def __init__(
        self,
        kb,
        encoder,
        kg=None,
        cache_dir: str = "data",
        log_path: str = "logs/evidence_reports.jsonl",
        max_index_items: int = 5000,
        top_k: int = 10,
        rebuild_chunk: int = 32,
        eval_batch_size: int = 16,
        eval_max_len: int = 192,
        incremental: bool = True,
    ):
        self.kb, self.encoder, self.kg = kb, encoder, kg
        self.cache_dir, self.log_path = cache_dir, log_path

        self.idx_meta_path = os.path.join(cache_dir, "kb_embed_index.json")
        self.idx_vec_path = os.path.join(cache_dir, "kb_embeddings.npy")

        self.max_index_items = int(max_index_items)
        self.top_k = int(top_k)
        self.rebuild_chunk = int(rebuild_chunk)

        # eval-safe knobs (used by _encode if supported by your bridge/trainer)
        self.eval_batch_size = int(eval_batch_size)
        self.eval_max_len = int(eval_max_len)

        self.incremental = bool(incremental)

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self._meta: List[Dict[str, Any]] = []
        self._emb: np.ndarray = np.zeros((0, 256), dtype=np.float32)  # normalized rows
        self._dim: int = 256
        self._key_to_row: Dict[str, int] = {}

        self._ensure_cache()

        # Optional: keep cache fresh without full rebuild
        if self.incremental:
            self.refresh_index_incremental()

    # ==========================================================
    # Cache Handling
    # ==========================================================
    def _hard_reset_cache_files(self):
        """Delete both cache files. Used when mismatch/corruption occurs."""
        for p in (self.idx_meta_path, self.idx_vec_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def _ensure_cache(self):
        try:
            if os.path.exists(self.idx_meta_path) and os.path.exists(self.idx_vec_path):
                with open(self.idx_meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f) or []
                self._emb = np.load(self.idx_vec_path)

                if self._emb.ndim != 2:
                    raise ValueError("Embeddings file is not 2D.")
                self._dim = int(self._emb.shape[1])

                if len(self._meta) != int(self._emb.shape[0]):
                    raise ValueError("Meta and embedding row count mismatch.")

                # Build key->row map
                self._key_to_row = {}
                for i, m in enumerate(self._meta):
                    k = str(m.get("key") or "")
                    if k:
                        self._key_to_row[k] = i

                # Ensure normalized embeddings
                self._emb = _l2_normalize_rows(self._emb)

            else:
                self._rebuild_cache_full()

        except Exception as e:
            # If anything smells off, wipe and rebuild clean
            self._hard_reset_cache_files()
            self._rebuild_cache_full()

    def _save_cache(self):
        np.save(self.idx_vec_path, self._emb.astype(np.float32, copy=False))
        with open(self.idx_meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2, ensure_ascii=False)

    def _rebuild_cache_full(self):
        """Full rebuild from KB (normalized embedding index)."""
        items = self.kb.query("") or []
        items = items[: self.max_index_items]

        texts: List[str] = []
        meta: List[Dict[str, Any]] = []
        keys: List[str] = []

        for it in items:
            text = _safe_text(it)
            if not isinstance(text, str) or not text.strip():
                continue

            key = None
            if isinstance(it, dict):
                key = it.get("id")
            if not key:
                key = _hash_key(text)
            key = str(key)

            meta.append({
                "key": key,
                "text": text[:1000],
                "source": (it.get("source", "") if isinstance(it, dict) else ""),
                "title": (it.get("paper_title", "") if isinstance(it, dict) else ""),
            })
            keys.append(key)
            texts.append(text)

        vecs = self._encode_chunked(texts, chunk=self.rebuild_chunk)
        if vecs.size == 0:
            vecs = np.zeros((0, self._dim), dtype=np.float32)

        # set dim from actual vecs if possible
        if vecs.ndim == 2 and vecs.shape[0] > 0:
            self._dim = int(vecs.shape[1])
        else:
            self._dim = int(self._dim)

        self._meta = meta
        self._emb = _l2_normalize_rows(vecs)

        self._key_to_row = {k: i for i, k in enumerate(keys)}
        self._save_cache()

    # ==========================================================
    # Incremental Refresh
    # ==========================================================
    def refresh_index_incremental(self):
        """
        Add embeddings for KB items not yet in cache.
        This avoids a full rebuild every run.
        """
        items = self.kb.query("") or []
        items = items[: self.max_index_items]

        new_texts: List[str] = []
        new_meta: List[Dict[str, Any]] = []
        new_keys: List[str] = []

        for it in items:
            text = _safe_text(it)
            if not isinstance(text, str) or not text.strip():
                continue

            key = None
            if isinstance(it, dict):
                key = it.get("id")
            if not key:
                key = _hash_key(text)
            key = str(key)

            if key in self._key_to_row:
                continue  # already indexed

            new_meta.append({
                "key": key,
                "text": text[:1000],
                "source": (it.get("source", "") if isinstance(it, dict) else ""),
                "title": (it.get("paper_title", "") if isinstance(it, dict) else ""),
            })
            new_keys.append(key)
            new_texts.append(text)

        if not new_texts:
            return  # nothing to do

        vecs = self._encode_chunked(new_texts, chunk=self.rebuild_chunk)
        if vecs.size == 0:
            return

        # If dimension changed (model changed), do a clean rebuild
        if vecs.ndim != 2 or (self._emb.size > 0 and vecs.shape[1] != self._dim):
            self._hard_reset_cache_files()
            self._rebuild_cache_full()
            return

        vecs = _l2_normalize_rows(vecs)

        # Append
        start = self._emb.shape[0]
        self._emb = np.vstack([self._emb, vecs]).astype(np.float32, copy=False)
        self._meta.extend(new_meta)

        for j, k in enumerate(new_keys):
            self._key_to_row[k] = start + j

        self._save_cache()

    # ==========================================================
    # Embedding Calls (eval-safe + chunked)
    # ==========================================================
    def _encode(self, texts: List[str]) -> np.ndarray:
        clean = [t for t in texts if isinstance(t, str) and t.strip()]
        if not clean:
            return np.zeros((0, self._dim), dtype=np.float32)

        # EmbeddingBridge style
        if hasattr(self.encoder, "encode_texts"):
            try:
                # if bridge supports options, use them
                arr = self.encoder.embed(
                    clean,
                    batch_size=self.eval_batch_size,
                    max_len=self.eval_max_len,
                )
                return np.asarray(arr, dtype=np.float32)
            except Exception:
                arr = self.encoder.encode_texts(clean)
                return np.asarray(arr, dtype=np.float32)

        # OnlineTrainer style
        if hasattr(self.encoder, "embed"):
            try:
                arr = self.encoder.embed(clean, batch_size=self.eval_batch_size, max_len=self.eval_max_len)
            except TypeError:
                arr = self.encoder.embed(clean)
            return np.asarray(arr, dtype=np.float32)

        # custom encoders
        if hasattr(self.encoder, "get_vector"):
            vecs = [np.asarray(self.encoder.get_vector(t), dtype=np.float32) for t in clean]
            return np.stack(vecs, axis=0)

        raise RuntimeError("Encoder does not expose encode_texts/embed/get_vector")

    def _encode_chunked(self, texts: List[str], chunk: int = 32) -> np.ndarray:
        vec_chunks = []
        for i in range(0, len(texts), max(1, int(chunk))):
            vec_chunks.append(self._encode(texts[i : i + chunk]))
        return np.concatenate(vec_chunks, axis=0) if vec_chunks else np.zeros((0, self._dim), dtype=np.float32)

    # ==========================================================
    # Evaluation
    # ==========================================================
    def evaluate_batch(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        # ensure cache sane
        if self._emb is None or self._emb.ndim != 2:
            self._hard_reset_cache_files()
            self._rebuild_cache_full()

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

            # stable schema
            res.setdefault("hypothesis", base_hyp)
            res.setdefault("verdict", "unknown")
            res.setdefault("evidence_confidence", 0.0)
            res.setdefault("evaluated_at", _now())
            out.append(res)

            # log each
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
            return {"hypothesis": text, "verdict": "invalid", "evidence_confidence": 0.0, "evaluated_at": _now()}

        # If KB index empty, can't evaluate
        if self._emb is None or self._emb.shape[0] == 0:
            return {"hypothesis": text, "verdict": "no_index", "evidence_confidence": 0.0, "evaluated_at": _now()}

        q = self._encode([text])
        if q.size == 0:
            return {"hypothesis": text, "verdict": "encode_failed", "evidence_confidence": 0.0, "evaluated_at": _now()}

        vec = q[0].astype(np.float32, copy=False)

        # If dimension mismatch (model changed), full rebuild once
        if vec.shape[0] != self._dim:
            self._hard_reset_cache_files()
            self._rebuild_cache_full()
            if self._emb.shape[0] == 0:
                return {"hypothesis": text, "verdict": "no_index", "evidence_confidence": 0.0, "evaluated_at": _now()}
            vec = self._encode([text])[0].astype(np.float32, copy=False)

        # cosine similarity using normalized index
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        sims = (self._emb @ vec).astype(np.float32, copy=False)

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

        # Confidence logic: clamp into [0,1] and handle negative sims
        # If sims are negative overall, we still produce stable output.
        total = abs(support) + abs(contra)
        conf = 0.0 if total <= 1e-12 else (abs(support) / total)
        conf = float(max(0.0, min(1.0, conf)))

        verdict = "supported" if conf > 0.60 else "contradicted" if conf < 0.40 else "inconclusive"

        return {
            "hypothesis": text,
            "verdict": verdict,
            "evidence_confidence": round(conf, 3),
            "evaluated_at": _now(),
        }
