# retrieval/chunk_index.py
# ------------------------------------------------------------
# ChunkIndex — embed + retrieve chunk evidence (dedupe + Hybrid + MMR)
# - Saves meta.json and vecs.npy
# - NEW:
#   * Dedupe chunks by text hash
#   * BM25 lexical index with disk cache (data/bm25_index.pkl)
#   * Hybrid scoring: 0.75*embedding + 0.25*bm25_norm
#   * Lexical gate: drop items with bm25==0 AND keyword_overlap < threshold
#   * MMR on hybrid relevance (diversity penalty via embedding cosine)
# ------------------------------------------------------------

from __future__ import annotations
import os
import json
import time
import hashlib
import pickle
import re
from typing import List, Dict, Any, Optional

import numpy as np

from retrieval.bm25 import BM25Index

BM25_PATH = "data/bm25_index.pkl"

_WORD_RE = re.compile(r"\b[a-zA-Z]{4,}\b")


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


def _keyword_overlap(a: str, b: str) -> float:
    # Very small + fast overlap gate (NOT semantics, just sanity)
    def toks(x: str) -> set[str]:
        words = _WORD_RE.findall((x or "").lower())
        stop = {
            "this", "that", "with", "from", "into", "have", "been", "were", "their",
            "which", "also", "there", "these", "those", "about", "under", "between",
            "while", "where", "when", "what", "will", "would", "could", "should"
        }
        return {w for w in words if w not in stop}

    A = toks(a)
    B = toks(b)
    if not A or not B:
        return 0.0
    return float(len(A & B)) / float(len(A | B))

def _anchor_terms_from_query(q: str) -> set[str]:
    ql = (q or "").lower()

    anchors = {
        # quantum anchors
        "schrodinger", "schrödinger", "wavefunction", "born",
        "measurement", "superposition", "collapse",
        "postulate", "eigenstate", "eigenvalue",
        "operator", "hilbert", "commutator",

        # physics/math anchors (generic safety)
        "lagrangian", "hamiltonian", "tensor",
        "entropy", "thermodynamics", "relativity",
    }

    present = {a for a in anchors if a in ql}

    # if quantum mentioned, force quantum anchor
    if "quantum" in ql:
        present.add("quantum")

    return present

def _mmr(
    rel: np.ndarray,           # (N,) relevance scores (use hybrid)
    vecs: np.ndarray,          # (N, D) normalized embedding vecs
    k: int,
    lambda_mult: float = 0.75,  # higher = more relevance, lower = more diversity
) -> List[int]:
    """Maximal Marginal Relevance selection indices."""
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
        sel_vecs = vecs[np.array(selected, dtype=int)]  # (S, D)
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
        encoder,               # EmbeddingBridge or OnlineTrainer
        cache_dir: str = "data",
        max_items: int = 8000,
        chunk_batch: int = 64,
        embed_max_len: int = 256,   # keep consistent across index + query
        dedupe: bool = True,
        mmr_lambda: float = 0.75,

        # Hybrid retrieval knobs (Option A)
        hybrid_alpha: float = 0.75,         # weight for embedding similarity
        bm25_beta: float = 0.25,            # weight for normalized bm25
        gate_keyword_overlap: float = 0.02  # lexical gate threshold
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

    # -----------------------
    # Encoding
    # -----------------------
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

    # -----------------------
    # Load / rebuild
    # -----------------------
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

                # load bm25 if present
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
        raw = self.chunk_store.fetch_recent(limit=self.max_items)  # expects list[dict]

        # --- Dedupe by normalized text hash ---
        items: List[Dict[str, Any]] = []
        seen = set()
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

        texts = [x["text"] for x in items]
        meta_items = [{
            "chunk_id": int(x["chunk_id"]),
            "paper_title": x.get("paper_title", "") or "",
            "source": x.get("source", "") or "",
            "text_preview": (x.get("text", "") or "")[:240],
        } for x in items]

        # embed
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

        # build BM25
        try:
            bm25 = BM25Index().build(texts)
            self.bm25 = bm25
            with open(self.bm25_path, "wb") as f:
                pickle.dump(bm25, f)
        except Exception:
            self.bm25 = None

        # persist vecs + meta
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
                "gate_keyword_overlap": float(self.gate_keyword_overlap),
                "bm25_enabled": bool(self.bm25 is not None),
            }, f, indent=2, ensure_ascii=False)

    # -----------------------
    # Retrieval
    # -----------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_mmr: bool = True,
        mmr_lambda: Optional[float] = None,
        gate: bool = True,
    ) -> List[Dict[str, Any]]:

        if self.vecs is None or self.vecs.shape[0] == 0:
            return []

        q = (query or "").strip()
        if not q:
            return []

        anchors = _anchor_terms_from_query(q)

        qv = self._encode([q])
        if qv.size == 0:
            return []

        qv = _normalize_rows(qv)[0]

        if int(qv.shape[0]) != int(self.dim):
            self.rebuild()
            if self.vecs.shape[0] == 0:
                return []
            qv = _normalize_rows(self._encode([q]))[0]

        # ----------------------------
        # Embedding similarity
        # ----------------------------
        sim_emb_all = (self.vecs @ qv).astype(np.float32)

        # ----------------------------
        # BM25 lexical similarity
        # ----------------------------
        bm25_scores_sparse: Dict[int, float] = {}
        if self.bm25 is not None:
            try:
                bm25_scores_sparse = self.bm25.score_query(q) or {}
            except Exception:
                bm25_scores_sparse = {}

        if bm25_scores_sparse:
            max_bm25 = max(bm25_scores_sparse.values())
            max_bm25 = max(max_bm25, 1e-8)
        else:
            max_bm25 = 1.0

        # ----------------------------
        # Candidate selection
        # ----------------------------
        cand_idx = []
        sim_emb = []
        sim_bm25 = []
        rel_hyb = []

        pre_k = min(int(top_k) * 50, int(sim_emb_all.shape[0]))
        pre = np.argsort(-sim_emb_all)[:pre_k].tolist()

        for i in pre:
            emb_score = float(sim_emb_all[i])
            bm25_raw = float(bm25_scores_sparse.get(i, 0.0))
            bm25_norm = float(bm25_raw / max_bm25)

            hyb = self.hybrid_alpha * emb_score + self.bm25_beta * bm25_norm

            preview = (self.items[i].get("text_preview") or "")
            title = (self.items[i].get("paper_title") or "")
            hay = (title + " " + preview).lower()

            # ----------------------------
            # Anchor gate (NEW — critical)
            # ----------------------------
            if anchors:
                if not any(a in hay for a in anchors):
                    continue

            # ----------------------------
            # Lexical sanity gate
            # ----------------------------
            if gate and (bm25_raw <= 0.0):
                ov = _keyword_overlap(q, preview)
                if ov < self.gate_keyword_overlap:
                    continue

            cand_idx.append(int(i))
            sim_emb.append(emb_score)
            sim_bm25.append(bm25_norm)
            rel_hyb.append(hyb)

        if not cand_idx:
            return []

        sim_emb = np.asarray(sim_emb, dtype=np.float32)
        sim_bm25 = np.asarray(sim_bm25, dtype=np.float32)
        rel_hyb = np.asarray(rel_hyb, dtype=np.float32)

        # ----------------------------
        # MMR selection (diversity)
        # ----------------------------
        k = min(int(top_k), int(rel_hyb.shape[0]))

        if use_mmr:
            lam = float(self.mmr_lambda if mmr_lambda is None else mmr_lambda)
            cand_vecs = self.vecs[np.array(cand_idx, dtype=int)]
            chosen_local = _mmr(rel=rel_hyb, vecs=cand_vecs, k=k, lambda_mult=lam)

            chosen = [cand_idx[j] for j in chosen_local]
            chosen_emb = [float(sim_emb[j]) for j in chosen_local]
            chosen_bm25 = [float(sim_bm25[j]) for j in chosen_local]
            chosen_hyb = [float(rel_hyb[j]) for j in chosen_local]
        else:
            order = np.argsort(-rel_hyb)[:k].tolist()
            chosen = [cand_idx[j] for j in order]
            chosen_emb = [float(sim_emb[j]) for j in order]
            chosen_bm25 = [float(sim_bm25[j]) for j in order]
            chosen_hyb = [float(rel_hyb[j]) for j in order]

        # ----------------------------
        # Hydrate full chunk text
        # ----------------------------
        chunk_ids = [int(self.items[i]["chunk_id"]) for i in chosen]
        full = {int(x["chunk_id"]): x for x in self.chunk_store.fetch_by_ids(chunk_ids)}

        out = []
        for pos, i in enumerate(chosen):
            m = self.items[int(i)]
            cid = int(m["chunk_id"])
            text = (full.get(cid, {}).get("text") or "").strip()

            out.append({
                "chunk_id": cid,
                "paper_title": m.get("paper_title", ""),
                "source": m.get("source", ""),
                "text": text,
                "text_preview": m.get("text_preview", ""),
                "similarity": float(chosen_hyb[pos]),
                "sim_embedding": float(chosen_emb[pos]),
                "sim_bm25": float(chosen_bm25[pos]),
            })

        return out
