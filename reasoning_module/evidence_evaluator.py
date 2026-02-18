# reasoning_module/evidence_evaluator.py
# ------------------------------------------------------------
# Evidence Evaluator (Robust + Chunk-Grounded Option)
# - Stable output schema (never missing keys)
# - Hard reset on mismatch/corruption (KB cache)
# - Incremental KB embedding cache updates (optional fallback)
# - If chunk_index is provided and use_chunk_index=True:
#     * evaluate hypotheses using ChunkIndex retrieval
#     * evidence quality gating (methods + numbers + theory signals)
#     * require >=2 strong evidence chunks for "supported"
#     * confidence cap
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

_NUM_RE = re.compile(r"(\b\d+(\.\d+)?\b|%|σ|p<|p=|CI\b|\bconfidence\b)", re.IGNORECASE)
_METHOD_RE = re.compile(r"\b(we (find|show|measure|observe|evaluate)|experiment|dataset|method|results?|analysis|simulation)\b", re.IGNORECASE)
_DEFINITION_RE = re.compile(r"\b(is the study of|is a field of|refers to|defined as)\b", re.IGNORECASE)

_THEORY_RE = re.compile(
    r"\b(schr(ö|o)dinger|hamiltonian|operator|hilbert|eigen|unitary|"
    r"postulate|born rule|wavefunction|superposition|measurement|collapse|"
    r"time-dependent|time-independent|commutator|state vector)\b",
    re.IGNORECASE
)


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


def _evidence_quality(text: str) -> float:
    t = (text or "")
    score = 0.0

    # empirical signals
    if _NUM_RE.search(t):
        score += 0.45
    if _METHOD_RE.search(t):
        score += 0.45

    # theory signals
    if _THEORY_RE.search(t):
        score += 0.55

    # definition penalty should not nuke theory references
    if _DEFINITION_RE.search(t) and not _THEORY_RE.search(t):
        score -= 0.25

    if len(t) < 180:
        score -= 0.10

    return float(max(0.15, min(1.0, score)))


def _source_weight(src: str) -> float:
    s = (src or "").lower().strip()
    if not s:
        return 1.0
    if "wikipedia" in s:
        return 0.55
    if "crossref" in s:
        return 0.65
    return 1.0


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

        # chunk-grounded option
        chunk_index: Optional[Any] = None,
        use_chunk_index: bool = True,

        # verdict gating knobs
        strong_quality_threshold: float = 0.55,
        min_strong_evidence: int = 2,
        confidence_cap: float = 0.85,
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

        self.chunk_index = chunk_index
        self.use_chunk_index = bool(use_chunk_index)

        self.strong_quality_threshold = float(strong_quality_threshold)
        self.min_strong_evidence = int(min_strong_evidence)
        self.confidence_cap = float(confidence_cap)

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # KB fallback cache
        self._items: List[Dict[str, Any]] = []
        self._emb: np.ndarray = np.zeros((0, 256), dtype=np.float32)
        self._dim: int = 256
        self._key_to_row: Dict[str, int] = {}

        # Only build KB cache if we will ever use it
        if (self.chunk_index is None) or (not self.use_chunk_index):
            self._ensure_cache()
            if self.incremental:
                self.refresh_index_incremental()

    # ==========================================================
    # Cache IO (KB fallback)
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
        if isinstance(data, list):
            return {"built_at": None, "dim": None, "items": data}
        if isinstance(data, dict) and "items" in data:
            return data
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

        if (self.chunk_index is None) or (not self.use_chunk_index):
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

    # --------------------------
    # Chunk-grounded path
    # --------------------------

    def _evaluate_with_chunks(self, text: str) -> Dict[str, Any]:
        hits = self.chunk_index.retrieve(text, top_k=self.top_k, use_mmr=True) if self.chunk_index else []
        if not hits:
            return {
                "hypothesis": text,
                "verdict": "inconclusive",
                "evidence_confidence": 0.0,
                "top_evidence": [],
                "evaluated_at": _now(),
                "why": "no_chunk_evidence",
            }

        top_evidence: List[Dict[str, Any]] = []
        support_w, contra_w = 0.0, 0.0
        strong = 0

        # OPTIONAL: try to infer domain from the hypothesis text itself
        hyp_lower = (text or "").lower()
        domain_hint = None
        for d in ["quantum", "quantum mechanics", "relativity", "cosmology", "thermodynamics",
                "electromagnetism", "plasma", "fluid", "solid state", "topology", "algebra"]:
            if d in hyp_lower:
                domain_hint = d
                break

        for h in hits:
            # -----------------------
            # ✅ DOMAIN FILTER (inside loop)
            # -----------------------
            meta = h.get("meta_json", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            chunk_domain = (meta.get("domain") or "").lower().strip()

            # If we have a domain hint and chunk has a domain, enforce consistency
            if domain_hint and chunk_domain and (domain_hint not in chunk_domain) and (chunk_domain not in domain_hint):
                continue

            # -----------------------
            # Normal scoring logic
            # -----------------------
            src = h.get("source", "")
            snippet = (h.get("text") or h.get("text_preview") or "")[:400]
            sim = float(h.get("sim_embedding", h.get("similarity", 0.0)) or 0.0)

            qlty = _evidence_quality(snippet)
            sw = _source_weight(src)
            is_theory = bool(_THEORY_RE.search(snippet))

            # strong chunk rule: relevance + quality, OR theory-strong relevance
            if ((sim >= 0.70 and qlty >= self.strong_quality_threshold) or (is_theory and sim >= 0.75)):
                strong += 1

            w = sim * qlty * sw

            is_contra = _polarity_is_contradiction(snippet)
            if is_contra:
                contra_w += w
            else:
                support_w += w

            top_evidence.append({
                "chunk_id": h.get("chunk_id"),
                "title": h.get("paper_title", ""),
                "source": src,
                "similarity": round(sim, 4),
                "quality": round(float(qlty), 3),
                "weight": round(float(w), 4),
                "polarity": "contra" if is_contra else "support",
                "text_preview": snippet,
            })

        total_w = support_w + contra_w
        if total_w < 1e-6:
            return {
                "hypothesis": text,
                "verdict": "inconclusive",
                "evidence_confidence": 0.0,
                "top_evidence": top_evidence,
                "evaluated_at": _now(),
                "strong_chunks": int(strong),
                "why": "no_high_quality_evidence",
            }

        denom = abs(support_w) + abs(contra_w) + 1e-9
        conf_raw = abs(support_w) / denom
        conf = float(max(0.0, min(1.0, conf_raw)))

        if strong >= self.min_strong_evidence and conf >= 0.62:
            verdict = "supported"
            conf = min(conf, self.confidence_cap)
        elif conf <= 0.38:
            verdict = "contradicted"
            conf = min(conf, self.confidence_cap)
        else:
            verdict = "inconclusive"
            if strong < self.min_strong_evidence:
                conf = min(conf, 0.55)

        return {
            "hypothesis": text,
            "verdict": verdict,
            "evidence_confidence": round(float(conf), 3),
            "top_evidence": top_evidence,
            "evaluated_at": _now(),
            "strong_chunks": int(strong),
        }
    # --------------------------
    # KB fallback path
    # --------------------------
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

        if self.chunk_index is not None and self.use_chunk_index:
            return self._evaluate_with_chunks(text)

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

        top_evidence: List[Dict[str, Any]] = []
        support, contra = 0.0, 0.0
        strong = 0

        for i in top_idx:
            sim = float(sims[int(i)])
            meta = self._items[int(i)] if int(i) < len(self._items) else {}
            preview = str(meta.get("text_preview") or "")[:240]
            src = meta.get("source", "")

            qlty = _evidence_quality(preview)
            sw = _source_weight(src)
            w = sim * qlty * sw

            if (sim >= 0.75) and (qlty >= self.strong_quality_threshold):
                strong += 1

            is_contra = _polarity_is_contradiction(preview)
            if is_contra:
                contra += w
            else:
                support += w

            top_evidence.append({
                "kb_id": meta.get("kb_id"),
                "title": meta.get("title", ""),
                "source": src,
                "similarity": round(sim, 4),
                "quality": round(float(qlty), 3),
                "weight": round(float(w), 4),
                "polarity": "contra" if is_contra else "support",
                "text_preview": preview,
            })

        total_w = support + contra
        if total_w < 1e-6:
            return {
                "hypothesis": text,
                "verdict": "inconclusive",
                "evidence_confidence": 0.0,
                "top_evidence": top_evidence,
                "evaluated_at": _now(),
                "strong_chunks": int(strong),
                "why": "no_high_quality_evidence",
            }

        denom = abs(support) + abs(contra) + 1e-9
        conf = float(abs(support) / denom)
        conf = float(max(0.0, min(1.0, conf)))

        if strong >= self.min_strong_evidence and conf >= 0.62:
            verdict = "supported"
            conf = min(conf, self.confidence_cap)
        elif conf <= 0.38:
            verdict = "contradicted"
            conf = min(conf, self.confidence_cap)
        else:
            verdict = "inconclusive"
            if strong < self.min_strong_evidence:
                conf = min(conf, 0.55)

        return {
            "hypothesis": text,
            "verdict": verdict,
            "evidence_confidence": round(conf, 3),
            "top_evidence": top_evidence,
            "evaluated_at": _now(),
            "strong_chunks": int(strong),
        }
