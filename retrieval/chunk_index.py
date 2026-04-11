# retrieval/chunk_index.py
# ------------------------------------------------------------
# ChunkIndex — embed + retrieve chunk evidence (dedupe + Hybrid + MMR)
# MVP upgrade:
# - domain-aware retrieval for transformer-efficiency queries
# - stronger lexical gating
# - domain score bonus / off-domain penalty
# - wider candidate pool from both embedding and BM25
# ------------------------------------------------------------

from __future__ import annotations
import os
import json
import time
import hashlib
import pickle
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from retrieval.bm25 import BM25Index

BM25_PATH = "data/bm25_index.pkl"

_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return (X / denom).astype(np.float32)


def _clean_text(t: str) -> str:
    t = (t or "").strip()
    t = " ".join(t.split())
    return t


def _text_hash(t: str) -> str:
    t = _clean_text(t).lower()
    return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()


def _tokenize(x: str) -> set[str]:
    words = _WORD_RE.findall((x or "").lower())
    stop = {
        "this", "that", "with", "from", "into", "have", "been", "were", "their",
        "which", "also", "there", "these", "those", "about", "under", "between",
        "while", "where", "when", "what", "will", "would", "could", "should",
        "using", "used", "than", "then", "they", "them", "such", "much", "more",
        "less", "some", "many", "most", "very", "does", "doesnt", "over",
        "improve", "improves", "improved", "quality", "efficient", "efficiency",
        "reduce", "reduces", "reduced", "performance"
    }
    return {w for w in words if w not in stop and len(w) >= 3}


def _keyword_overlap(a: str, b: str) -> float:
    A = _tokenize(a)
    B = _tokenize(b)
    if not A or not B:
        return 0.0
    return float(len(A & B)) / float(len(A | B))


# ------------------------------------------------------------
# Domain profiles
# ------------------------------------------------------------

_TRANSFORMER_TERMS = {
    "transformer", "attention", "self-attention", "self attention",
    "flashattention", "flash-attention",
    "sparse", "sparse attention", "linear attention",
    "long-context", "long context", "context length",
    "mixture-of-experts", "mixture of experts", "moe",
    "router", "routing", "expert", "experts",
    "kv", "kv-cache", "kv cache", "cache",
    "inference", "latency", "throughput", "memory",
    "token", "tokens", "sequence", "context",
    "prefill", "decode", "decoding",
    "bandwidth", "quadratic", "subquadratic",
    "llm", "language model", "language models",
    "benchmark", "perplexity"
}

_SPACE_TERMS = {
    "space", "orbital", "orbit", "satellite", "propulsion", "rocket",
    "spacecraft", "astrodynamics", "planetary", "mars", "moon",
    "lunar", "deep space", "launch", "gravity assist", "payload",
    "thruster", "aerospace", "trajectory"
}


def _detect_domain_profile(query: str) -> str:
    q = (query or "").lower()

    if any(t in q for t in _TRANSFORMER_TERMS):
        return "transformer_efficiency"

    if any(t in q for t in _SPACE_TERMS):
        return "space"

    return "generic"


def _profile_terms(profile: str) -> set[str]:
    if profile == "transformer_efficiency":
        return _TRANSFORMER_TERMS
    if profile == "space":
        return _SPACE_TERMS
    return set()


def _count_profile_hits(text: str, profile_terms: set[str]) -> int:
    low = (text or "").lower()
    return sum(1 for t in profile_terms if t in low)


def _domain_score(query: str, title: str, preview: str, full_text: str = "") -> Tuple[float, int]:
    """
    Returns:
      (domain_score, hit_count)

    domain_score:
      + positive if query/chunk domain align
      - penalty if obviously off-domain for targeted MVP queries
    """
    profile = _detect_domain_profile(query)
    if profile == "generic":
        return 0.0, 0

    terms = _profile_terms(profile)
    hay = f"{title} {preview} {full_text}".lower()
    hits = _count_profile_hits(hay, terms)

    if profile == "transformer_efficiency":
        # strong preference for chunks that actually mention the domain
        if hits >= 4:
            return 0.22, hits
        if hits >= 2:
            return 0.12, hits
        if hits == 1:
            return 0.04, hits
        return -0.22, hits

    if profile == "space":
        if hits >= 4:
            return 0.22, hits
        if hits >= 2:
            return 0.12, hits
        if hits == 1:
            return 0.04, hits
        return -0.22, hits

    return 0.0, hits


def _query_anchor_terms(query: str) -> set[str]:
    """
    Extract a stronger set of must-care anchor terms from the query itself.
    """
    q = (query or "").lower()
    anchors = set()

    anchor_candidates = [
        "flashattention", "flash-attention",
        "mixture-of-experts", "mixture of experts", "moe",
        "sparse attention", "linear attention",
        "kv-cache", "kv cache",
        "long context", "long-context", "context length",
        "transformer", "attention", "latency", "throughput", "memory",
        "inference", "perplexity"
    ]
    for a in anchor_candidates:
        if a in q:
            anchors.add(a)

    # generic fallback for targeted profile
    if not anchors and _detect_domain_profile(query) == "transformer_efficiency":
        anchors.update({"transformer", "attention"})

    return anchors


def _mmr(
    rel: np.ndarray,
    vecs: np.ndarray,
    k: int,
    lambda_mult: float = 0.75,
) -> List[int]:
    N = int(rel.shape[0])
    if N == 0 or k <= 0:
        return []
    k = min(k, N)

    selected: List[int] = []
    remaining = set(range(N))

    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        sel_vecs = vecs[np.array(selected, dtype=int)]
        best_i = None
        best_score = -1e9

        for i in remaining:
            r = float(rel[i])
            div = float((sel_vecs @ vecs[i]).max()) if sel_vecs.size else 0.0
            score = lambda_mult * r - (1.0 - lambda_mult) * div
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break
        selected.append(int(best_i))
        remaining.remove(int(best_i))

    return selected


class ChunkIndex:
    def __init__(
        self,
        chunk_store,
        encoder,
        cache_dir: str = "data",
        max_items: int = 8000,
        chunk_batch: int = 64,
        embed_max_len: int = 256,
        dedupe: bool = True,
        mmr_lambda: float = 0.75,
        hybrid_alpha: float = 0.62,
        bm25_beta: float = 0.23,
        domain_gamma: float = 1.0,
        gate_keyword_overlap: float = 0.03,
    ):
        self.chunk_store = chunk_store
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.max_items = int(max_items)
        self.chunk_batch = int(chunk_batch)
        self.embed_max_len = int(embed_max_len)

        self.dedupe = bool(dedupe)
        self.mmr_lambda = float(mmr_lambda)

        self.hybrid_alpha = float(hybrid_alpha)
        self.bm25_beta = float(bm25_beta)
        self.domain_gamma = float(domain_gamma)
        self.gate_keyword_overlap = float(gate_keyword_overlap)

        os.makedirs(cache_dir, exist_ok=True)
        self.meta_path = os.path.join(cache_dir, "chunk_index_meta.json")
        self.vec_path = os.path.join(cache_dir, "chunk_index_vecs.npy")
        self.bm25_path = os.path.join(cache_dir, "bm25_index.pkl")

        self.items: List[Dict[str, Any]] = []
        self.vecs: np.ndarray = np.zeros((0, 256), dtype=np.float32)
        self.dim: int = 256
        self.bm25: Optional[BM25Index] = None

        self._load_or_rebuild()

    def _encode(self, texts: List[str]) -> np.ndarray:
        texts = [str(t or "") for t in texts]

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
                    items = payload if isinstance(payload, list) else []
                    dim = int(vecs.shape[1])

                if len(items) != int(vecs.shape[0]):
                    raise ValueError("meta/vec mismatch")

                self.items = items
                self.vecs = vecs.astype(np.float32)
                self.dim = dim

                if os.path.exists(self.bm25_path):
                    try:
                        with open(self.bm25_path, "rb") as f:
                            self.bm25 = pickle.load(f)
                    except Exception:
                        self.bm25 = None
                return
        except Exception:
            pass

        self.rebuild()

    def rebuild(self):
        raw = self.chunk_store.fetch_recent(limit=self.max_items)

        items: List[Dict[str, Any]] = []
        seen = set()
        texts: List[str] = []

        for x in raw:
            txt = _clean_text(x.get("text", ""))
            if not txt:
                continue
            if self.dedupe:
                h = _text_hash(txt)
                if h in seen:
                    continue
                seen.add(h)

            items.append(x)
            texts.append(txt)

        meta_items = [{
            "chunk_id": int(x["chunk_id"]),
            "paper_title": x.get("paper_title", "") or "",
            "source": x.get("source", "") or "",
            "text": (x.get("text", "") or ""),
            "text_preview": (x.get("text", "") or "")[:500],
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

        try:
            self.bm25 = BM25Index().build(texts)
            with open(self.bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)
        except Exception:
            self.bm25 = None

        np.save(self.vec_path, self.vecs)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "built_at": _now(),
                "dim": int(self.dim),
                "count": int(len(self.items)),
                "items": self.items,
                "dedupe": bool(self.dedupe),
                "mmr_lambda": float(self.mmr_lambda),
                "embed_max_len": int(self.embed_max_len),
                "hybrid_alpha": float(self.hybrid_alpha),
                "bm25_beta": float(self.bm25_beta),
                "domain_gamma": float(self.domain_gamma),
                "gate_keyword_overlap": float(self.gate_keyword_overlap),
                "bm25_enabled": bool(self.bm25 is not None),
            }, f, indent=2, ensure_ascii=False)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_mmr: bool = True,
        mmr_lambda: Optional[float] = None,
        gate: bool = True,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        if self.vecs is None or self.vecs.shape[0] == 0:
            return []

        q = (query or "").strip()
        if not q:
            return []

        qv = self._encode([q])
        if qv.size == 0:
            return []

        qv = _normalize_rows(qv)[0]

        if int(qv.shape[0]) != int(self.dim):
            self.rebuild()
            if self.vecs.shape[0] == 0:
                return []
            qv = _normalize_rows(self._encode([q]))[0]

        sim_emb_all = (self.vecs @ qv).astype(np.float32)

        bm25_scores_sparse: Dict[int, float] = {}
        if self.bm25 is not None:
            try:
                bm25_scores_sparse = self.bm25.score_query(q) or {}
            except Exception:
                bm25_scores_sparse = {}

        max_bm25 = max(bm25_scores_sparse.values()) if bm25_scores_sparse else 1.0
        max_bm25 = max(max_bm25, 1e-8)

        # use BOTH emb and bm25 to build candidate set
        emb_pre_k = min(int(top_k) * 80, int(sim_emb_all.shape[0]))
        emb_pre = set(np.argsort(-sim_emb_all)[:emb_pre_k].tolist())

        bm25_pre = set()
        if bm25_scores_sparse:
            bm25_sorted = sorted(bm25_scores_sparse.items(), key=lambda kv: kv[1], reverse=True)
            bm25_pre = {int(i) for i, _ in bm25_sorted[:emb_pre_k]}

        pre = list(emb_pre | bm25_pre)
        if not pre:
            return []

        anchors = _query_anchor_terms(q)
        profile = _detect_domain_profile(q)

        cand_idx = []
        sim_emb = []
        sim_bm25 = []
        rel_hyb = []
        domain_scores = []
        profile_hits_list = []

        for i in pre:
            emb_score = float(sim_emb_all[i])
            bm25_raw = float(bm25_scores_sparse.get(i, 0.0))
            bm25_norm = float(bm25_raw / max_bm25)

            preview = (self.items[i].get("text_preview") or self.items[i].get("text","")[:200])
            title = (self.items[i].get("paper_title") or "")
            hay = f"{title} {preview}".lower()

            # anchor gate disabled — using embedding similarity instead
            pass

            # lexical sanity gate disabled — using embedding similarity
            pass

            d_score, hit_count = _domain_score(q, title, preview, "")
            hyb = (
                self.hybrid_alpha * emb_score
                + self.bm25_beta * bm25_norm
                + self.domain_gamma * d_score
            )

            # hard off-domain rejection disabled
            pass

            cand_idx.append(int(i))
            sim_emb.append(emb_score)
            sim_bm25.append(bm25_norm)
            rel_hyb.append(hyb)
            domain_scores.append(float(d_score))
            profile_hits_list.append(int(hit_count))

        if not cand_idx:
            return []

        sim_emb = np.asarray(sim_emb, dtype=np.float32)
        sim_bm25 = np.asarray(sim_bm25, dtype=np.float32)
        rel_hyb = np.asarray(rel_hyb, dtype=np.float32)

        k = min(int(top_k), int(rel_hyb.shape[0]))

        if use_mmr:
            lam = float(self.mmr_lambda if mmr_lambda is None else mmr_lambda)
            cand_vecs = self.vecs[np.array(cand_idx, dtype=int)]
            chosen_local = _mmr(rel=rel_hyb, vecs=cand_vecs, k=k, lambda_mult=lam)

            chosen = [cand_idx[j] for j in chosen_local]
            chosen_emb = [float(sim_emb[j]) for j in chosen_local]
            chosen_bm25 = [float(sim_bm25[j]) for j in chosen_local]
            chosen_hyb = [float(rel_hyb[j]) for j in chosen_local]
            chosen_dom = [float(domain_scores[j]) for j in chosen_local]
            chosen_hits = [int(profile_hits_list[j]) for j in chosen_local]
        else:
            order = np.argsort(-rel_hyb)[:k].tolist()
            chosen = [cand_idx[j] for j in order]
            chosen_emb = [float(sim_emb[j]) for j in order]
            chosen_bm25 = [float(sim_bm25[j]) for j in order]
            chosen_hyb = [float(rel_hyb[j]) for j in order]
            chosen_dom = [float(domain_scores[j]) for j in order]
            chosen_hits = [int(profile_hits_list[j]) for j in order]

        chunk_ids = [int(self.items[i]["chunk_id"]) for i in chosen]
        full = {int(x["chunk_id"]): x for x in self.chunk_store.fetch_by_ids(chunk_ids)}

        out = []
        for pos, i in enumerate(chosen):
            m = self.items[int(i)]
            cid = int(m["chunk_id"])
            text = (full.get(cid, {}).get("text") or "").strip()

            row = {
                "chunk_id": cid,
                "paper_title": m.get("paper_title", ""),
                "source": m.get("source", ""),
                "text": text,
                "text_preview": m.get("text_preview", ""),
                "similarity": float(chosen_hyb[pos]),
                "sim_embedding": float(chosen_emb[pos]),
                "sim_bm25": float(chosen_bm25[pos]),
                "domain_score": float(chosen_dom[pos]),
                "profile_hits": int(chosen_hits[pos]),
            }

            if debug:
                row["debug"] = {
                    "profile": profile,
                    "keyword_overlap": _keyword_overlap(q, f"{row['paper_title']} {row['text_preview']}"),
                }

            out.append(row)

        return out