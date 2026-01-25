# reasoning_module/evidence_evaluator.py
# ------------------------------------------------------------
# Evidence Evaluator (Robust + Incremental Cache)
# - Stable output schema (never missing keys)
# - Hard reset on mismatch/corruption
# - Incremental embedding index updates (no full rebuild each run)
# - True cosine similarity (normalized index)
# - Returns top evidence rows (so "supported" means something)
# ------------------------------------------------------------

from __future__ import annotations
import os, re, json, time, hashlib
from typing import List, Dict, Any
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

        self.eval_batch_size = int(eval_batch_size)
        self.eval_max_len = int(eval_max_len)

        self.incremental = bool(incremental)

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self._items: List[Dict[str, Any]] = []
        self._emb: np.ndarray = np.zeros((0, 256), dtype=np.float32)  # normalized
        self._dim: int = 256
        self._key_to_row: Dict[str, int] = {}

        self._ensure_cache()
        if self.incremental:
            self.refresh_index_incremental()

    # ==========================================================
    # Cache IO
    # ==========================================================
    def _hard_reset_cache_files(self):
        for p in (self.idx_meta_path, self.idx_vec_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def _load_meta_file(self) -> Dict[str, Any]:
        with open(self.idx_meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Backwards compatibility: older files might be list directly
        if isinstance(data, list):
            return {"built_at": None, "dim": None, "items": data}
        if isinstance(data, dict) and "items" in data:
            return data
        # unknown format
        return {"built_at": None, "dim": None, "items": []}

    def _save_cache(self):
        np.save(self.idx_vec_path, self._emb.astype(np.float32, copy=False))
        meta = {"built_at": _now(), "dim": int(self._dim), "items": self._items}
        with open(self.idx_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _ensure_cache(self):
        try:
            if os.path.exists(self.idx_meta_path) and os.path.exists(self.idx_vec_path):
                meta = self._load_meta_file()
                self._items = meta.get("items") or []
                self._emb = np.load(self.idx_vec_path)

                if self._emb.ndim != 2:
                    raise ValueError("Embeddings file not 2D")
                if len(self._items) != int(self._emb.shape[0]):
                    raise ValueError("Meta/embedding row mismatch")

                self._dim = int(self._emb.shape[1])
                self._emb = _l2_normalize_rows(self._emb)

                self._key_to_row = {}
                for i, m in enumerate(self._items):
                    k = str(m.get("key") or "")
                    if k:
                        self._key_to_row[k] = i
            else:
                self._rebuild_cache_full()

        except Exception:
            self._hard_reset_cache_files()
            self._rebuild_cache_full()

    def _rebuild_cache_full(self):
        items = (self.kb.query("") or [])[: self.max_index_items]

        texts: List[str] = []
        meta_items: List[Dict[str, Any]] = []
        keys: List[str] = []

        for it in items:
            text = _safe_text(it).strip()
            if not text:
                continue

            # use KB id when present (stable)
            key = str(it.get("id")) if isinstance(it, dict) and it.get("id") is not None else _hash_key(text)
            keys.append(key)
            texts.append(text)

            meta_items.append({
                "key": key,
                "kb_id": it.get("id") if isinstance(it, dict) else None,
                "title": (it.get("paper_title") or "") if isinstance(it, dict) else "",
                "source": (it.get("source") or "") if isinstance(it, dict) else "",
                "text_preview": text[:500],
            })

        vecs = self._encode_chunked(texts, chunk=self.rebuild_chunk)
        if vecs.size > 0:
            self._dim = int(vecs.shape[1])
        else:
            vecs = np.zeros((0, self._dim), dtype=np.float32)

        self._items = meta_items
        self._emb = _l2_normalize_rows(vecs)
        self._key_to_row = {k: i for i, k in enumerate(keys)}
        self._save_cache()

    # ==========================================================
    # Incremental refresh
    # ==========================================================
    def refresh_index_incremental(self):
        items = (self.kb.query("") or [])[: self.max_index_items]

        new_texts: List[str] = []
        new_meta: List[Dict[str, Any]] = []
        new_keys: List[str] = []

        for it in items:
            text = _safe_text(it).strip()
            if not text:
                continue
            key = str(it.get("id")) if isinstance(it, dict) and it.get("id") is not None else _hash_key(text)
            if key in self._key_to_row:
                continue

            new_keys.append(key)
            new_texts.append(text)
            new_meta.append({
                "key": key,
                "kb_id": it.get("id") if isinstance(it, dict) else None,
                "title": (it.get("paper_title") or "") if isinstance(it, dict) else "",
                "source": (it.get("source") or "") if isinstance(it, dict) else "",
                "text_preview": text[:500],
            })

        if not new_texts:
            return

        vecs = self._encode_chunked(new_texts, chunk=self.rebuild_chunk)
        if vecs.size == 0:
            return

        if vecs.ndim != 2:
            self._hard_reset_cache_files()
            self._rebuild_cache_full()
            return

        if self._emb.size > 0 and vecs.shape[1] != self._dim:
            self._hard_reset_cache_files()
            self._rebuild_cache_full()
            return

        vecs = _l2_normalize_rows(vecs)

        start = self._emb.shape[0]
        self._emb = np.vstack([self._emb, vecs]).astype(np.float32, copy=False)
        self._items.extend(new_meta)
        for j, k in enumerate(new_keys):
            self._key_to_row[k] = start + j

        self._save_cache()

    # ==========================================================
    # Embedding calls
    # ==========================================================
    def _encode(self, texts: List[str]) -> np.ndarray:
        clean = []
        empty_mask = []
        for t in texts or []:
            s = t if isinstance(t, str) else str(t or "")
            s = s.strip()
            if not s:
                clean.append("[EMPTY]")
                empty_mask.append(True)
            else:
                clean.append(s)
                empty_mask.append(False)

        if not clean:
            return np.zeros((0, self._dim), dtype=np.float32)

        # Bridge path
        if hasattr(self.encoder, "embed"):
            try:
                arr = self.encoder.embed(clean, batch_size=self.eval_batch_size, max_len=self.eval_max_len)
            except TypeError:
                arr = self.encoder.embed(clean)
            arr = np.asarray(arr, dtype=np.float32)
        elif hasattr(self.encoder, "encode_texts"):
            arr = np.asarray(self.encoder.encode_texts(clean), dtype=np.float32)
        elif hasattr(self.encoder, "get_vector"):
            arr = np.stack([np.asarray(self.encoder.get_vector(t), dtype=np.float32) for t in clean], axis=0)
        else:
            raise RuntimeError("Encoder does not expose embed/encode_texts/get_vector")

        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        for i, is_empty in enumerate(empty_mask):
            if is_empty and i < arr.shape[0]:
                arr[i, :] = 0.0

        return arr.astype(np.float32, copy=False)

    def _encode_chunked(self, texts: List[str], chunk: int = 32) -> np.ndarray:
        vec_chunks = []
        step = max(1, int(chunk))
        for i in range(0, len(texts), step):
            vec_chunks.append(self._encode(texts[i:i + step]))
        return np.concatenate(vec_chunks, axis=0) if vec_chunks else np.zeros((0, self._dim), dtype=np.float32)

    # ==========================================================
    # Evaluation
    # ==========================================================
    def evaluate_batch(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        if self._emb is None or self._emb.ndim != 2:
            self._hard_reset_cache_files()
            self._rebuild_cache_full()

        for h in (hypotheses or []):
            base_hyp = str(h.get("hypothesis") or "") if isinstance(h, dict) else str(h or "")
            try:
                res = self.evaluate_one({"hypothesis": base_hyp} if not isinstance(h, dict) else h)
            except Exception as e:
                res = {
                    "hypothesis": base_hyp,
                    "verdict": "error",
                    "evidence_confidence": 0.0,
                    "top_evidence": [],
                    "evaluated_at": _now(),
                    "evidence_error": str(e),
                }

            res.setdefault("hypothesis", base_hyp)
            res.setdefault("verdict", "unknown")
            res.setdefault("evidence_confidence", 0.0)
            res.setdefault("top_evidence", [])
            res.setdefault("evaluated_at", _now())
            out.append(res)

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
        text = str(hypothesis.get("hypothesis") or "").strip()
        if not text:
            return {
                "hypothesis": text,
                "verdict": "invalid",
                "evidence_confidence": 0.0,
                "top_evidence": [],
                "evaluated_at": _now(),
            }

        if self._emb is None or self._emb.shape[0] == 0:
            return {
                "hypothesis": text,
                "verdict": "no_index",
                "evidence_confidence": 0.0,
                "top_evidence": [],
                "evaluated_at": _now(),
            }

        q = self._encode([text])
        if q.size == 0:
            return {
                "hypothesis": text,
                "verdict": "encode_failed",
                "evidence_confidence": 0.0,
                "top_evidence": [],
                "evaluated_at": _now(),
            }

        vec = q[0].astype(np.float32, copy=False)

        if vec.shape[0] != self._dim:
            self._hard_reset_cache_files()
            self._rebuild_cache_full()
            if self._emb.shape[0] == 0:
                return {
                    "hypothesis": text,
                    "verdict": "no_index",
                    "evidence_confidence": 0.0,
                    "top_evidence": [],
                    "evaluated_at": _now(),
                }
            vec = self._encode([text])[0].astype(np.float32, copy=False)

        vec = vec / (np.linalg.norm(vec) + 1e-8)
        sims = (self._emb @ vec).astype(np.float32, copy=False)

        k = min(self.top_k, sims.shape[0])
        top_idx = np.argsort(-sims)[:k]

        # Build evidence list
        top_evidence: List[Dict[str, Any]] = []
        support, contra = 0.0, 0.0

        for i in top_idx:
            sim = float(sims[int(i)])
            meta = self._items[int(i)] if int(i) < len(self._items) else {}
            preview = str(meta.get("text_preview") or "")[:240]

            is_contra = _polarity_is_contradiction(preview)
            if is_contra:
                contra += sim
            else:
                support += sim

            top_evidence.append({
                "kb_id": meta.get("kb_id"),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "similarity": round(sim, 4),
                "polarity": "contra" if is_contra else "support",
                "text_preview": preview,
            })

        total = abs(support) + abs(contra)
        conf = 0.0 if total <= 1e-12 else (abs(support) / total)
        conf = float(max(0.0, min(1.0, conf)))

        verdict = "supported" if conf > 0.60 else "contradicted" if conf < 0.40 else "inconclusive"

        return {
            "hypothesis": text,
            "verdict": verdict,
            "evidence_confidence": round(conf, 3),
            "top_evidence": top_evidence,
            "evaluated_at": _now(),
        }
