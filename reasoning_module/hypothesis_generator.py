"""
Hypothesis Generator — v2 (grounded + quality-gated)
---------------------------------------------------
Goals:
- Stop garbage hypotheses caused by polluted KG nodes ("beginning", "low", "closed")
- Use batched embeddings + caching (fast)
- Gate by domain buckets to prevent cross-domain hallucinated links
- Prefer meaningful relations and multi-word scientific concepts
"""

from __future__ import annotations

import os
import json
import time
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np


# -------------------------
# Concept quality gates
# -------------------------
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "when", "under", "for",
    "to", "of", "in", "on", "at", "by", "with", "without", "from", "into",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as",
    "we", "they", "you", "i",
    "low", "high", "small", "large", "rapidly", "major", "kind", "basis",
    "beginning", "closed", "open", "provide", "sense", "greater", "insight",
}

# Allow only relations that make sense for transitivity / inference
# (tune this to your KG schema)
_ALLOWED_RELATIONS = {
    "causes", "leads_to", "influences", "increases", "decreases",
    "part_of", "type_of", "subclass_of", "related_to",
    "associated_with", "correlates_with",
}

# Domain bucket keywords (cheap but effective)
_DOMAIN_KEYWORDS = {
    "water_microplastics": [
        "microplastic", "nanoplastic", "drinking water", "water treatment",
        "coagulation", "flocculation", "sedimentation", "filtration",
        "membrane", "ultrafiltration", "nanofiltration", "reverse osmosis",
        "activated carbon", "gac", "sand filter", "polyethylene", "pet",
    ],
    "quantum": [
        "quantum", "qubit", "decoherence", "entanglement", "hamiltonian",
        "superconduct", "ion trap", "measurement", "wavefunction",
    ],
    "physics": ["relativity", "entropy", "thermodynamics", "electromagnet", "mass", "force", "energy"],
    "math": ["theorem", "lemma", "proof", "tensor", "manifold", "topology", "algebra", "calculus"],
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s


def _looks_like_concept(s: str) -> bool:
    """
    Hard filter to kill KG junk:
    - too short
    - mostly stopwords
    - punctuation garbage
    - numeric-heavy
    - sentence-like long fragments
    """
    s = _clean(s)
    if not s:
        return False

    low = s.lower()

    # Too short / single token that is a stopword
    toks = re.findall(r"[a-zA-Z]+", low)
    if len(toks) == 0:
        return False
    if len(toks) == 1 and toks[0] in _STOPWORDS:
        return False

    # Minimum length: prefer real concepts
    if len(low) < 4:
        return False

    # Reject pure numbers / numeric-heavy
    num_chars = sum(ch.isdigit() for ch in low)
    if num_chars / max(1, len(low)) > 0.30:
        return False

    # Reject if too many stopwords
    stop = sum(1 for t in toks if t in _STOPWORDS)
    if stop / max(1, len(toks)) > 0.60:
        return False

    # Reject obvious sentence fragments (too long + ends with punctuation)
    if len(low) > 70 and low[-1] in {".", "!", "?"}:
        return False

    # Reject weird tokens (e.g., "--is-->")
    if "--" in low and "-->" in low:
        return False

    return True


def _guess_bucket(text: str) -> str:
    low = (text or "").lower()
    scores = {}
    for bucket, kws in _DOMAIN_KEYWORDS.items():
        scores[bucket] = sum(1 for w in kws if w in low)
    best_bucket, best_score = max(scores.items(), key=lambda x: x[1])
    return best_bucket if best_score > 0 else "unknown"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


class HypothesisGenerator:
    def __init__(
        self,
        knowledge_graph,
        encoder,
        kb=None,
        out_dir: str = "outputs",
        log_dir: str = "logs",
        embed_max_len: int = 256,
        max_nodes: int = 400,
        semantic_sim_thresh: float = 0.78,
        enforce_domain_gating: bool = True,
    ):
        """
        knowledge_graph: KnowledgeGraph instance
        encoder: EmbeddingBridge instance (should support encode_texts(texts, ...) -> (N,D))
        kb: optional KnowledgeBase
        """
        self.kg = knowledge_graph
        self.encoder = encoder
        self.kb = kb

        self.embed_max_len = int(embed_max_len)
        self.max_nodes = int(max_nodes)
        self.semantic_sim_thresh = float(semantic_sim_thresh)
        self.enforce_domain_gating = bool(enforce_domain_gating)

        self.out_dir = out_dir
        self.log_dir = log_dir
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.out_jsonl = os.path.join(self.out_dir, "hypotheses.jsonl")
        self.log_file = os.path.join(self.log_dir, "hypothesis_log.txt")

        # embedding cache: concept -> vec
        self._cache: Dict[str, np.ndarray] = {}

    # -------------------------
    # KG helpers
    # -------------------------
    def _neighbors(self, node: str) -> Dict[str, List[str]]:
        try:
            rels = self.kg.get_relations(node)
            if not isinstance(rels, dict):
                return {}
            out = {}
            for r, v in rels.items():
                if r not in _ALLOWED_RELATIONS:
                    continue
                if isinstance(v, list):
                    out[r] = v
                else:
                    out[r] = list(v)
            return out
        except Exception:
            return {}

    def _has_edge(self, a: str, relation: str, c: str) -> bool:
        rels = self._neighbors(a)
        return c in rels.get(relation, [])

    def _all_nodes(self) -> List[str]:
        # Pull candidates from KG
        try:
            nodes = list(self.kg.all_concepts())
        except Exception:
            try:
                nodes = list(getattr(self.kg, "graph", {}).keys())
            except Exception:
                nodes = []

        # Quality gate
        nodes = [_clean(n) for n in nodes]
        nodes = [n for n in nodes if _looks_like_concept(n)]

        # Deterministic cap
        nodes = nodes[: self.max_nodes]
        return nodes

    def _all_edges(self) -> List[Tuple[str, str, str]]:
        edges = []
        for a in self._all_nodes():
            for rel, objs in self._neighbors(a).items():
                for b in objs:
                    b = _clean(str(b))
                    if not _looks_like_concept(b):
                        continue
                    edges.append((a, rel, b))
        return edges

    # -------------------------
    # Embeddings (batched + cached)
    # -------------------------
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Must return (N,D) float32.
        EmbeddingBridge in your repo uses encode_texts(texts, max_len=..., batch_size=..., force_cpu=...)
        """
        texts = [str(t or "") for t in texts]
        if hasattr(self.encoder, "encode_texts"):
            arr = self.encoder.encode_texts(
                texts,
                max_len=self.embed_max_len,
                batch_size=64,
                force_cpu=False,
            )
            arr = np.asarray(arr, dtype=np.float32)
        elif hasattr(self.encoder, "embed"):
            arr = np.asarray(self.encoder.embed(texts, max_len=self.embed_max_len), dtype=np.float32)
        else:
            raise RuntimeError("encoder must expose encode_texts() or embed()")

        if arr.ndim == 3:
            # token embeddings -> mean pool
            arr = arr.mean(axis=1)

        # normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return (arr / norms).astype(np.float32)

    def _get_vecs(self, nodes: List[str]) -> np.ndarray:
        # encode only missing
        missing = [n for n in nodes if n not in self._cache]
        if missing:
            vecs = self._encode_texts(missing)
            for n, v in zip(missing, vecs):
                self._cache[n] = v
        return np.stack([self._cache[n] for n in nodes], axis=0).astype(np.float32)

    # -------------------------
    # Proposal builders
    # -------------------------
    def _propose_transitive(self, max_new: int = 30) -> List[Dict[str, Any]]:
        """
        If A -r1-> B and B -r2-> C and NOT A -r2-> C, propose A -r2-> C.
        Score = 0.5*(sim(A,B)+sim(B,C)), but gated by domain buckets.
        """
        proposals: List[Dict[str, Any]] = []
        edges_by_src = defaultdict(list)

        edges = self._all_edges()
        if not edges:
            return proposals

        for a, r, b in edges:
            edges_by_src[a].append((r, b))

        count = 0
        for a, out1 in edges_by_src.items():
            for (r1, b) in out1:
                out2 = self._neighbors(b)
                for r2, cs in out2.items():
                    if r2 not in _ALLOWED_RELATIONS:
                        continue
                    for c in cs:
                        c = _clean(str(c))
                        if not _looks_like_concept(c):
                            continue
                        if a == c:
                            continue
                        if self._has_edge(a, r2, c):
                            continue

                        # Domain gating
                        if self.enforce_domain_gating:
                            ba = _guess_bucket(a)
                            bc = _guess_bucket(c)
                            if ba != "unknown" and bc != "unknown" and ba != bc:
                                continue

                        # embedding score
                        s_ab = _cos(self._cache.get(a) or self._get_vecs([a])[0],
                                    self._cache.get(b) or self._get_vecs([b])[0])
                        s_bc = _cos(self._cache.get(b) or self._get_vecs([b])[0],
                                    self._cache.get(c) or self._get_vecs([c])[0])
                        score = 0.5 * (s_ab + s_bc)

                        proposals.append({
                            "type": "graph_transitivity",
                            "hypothesis": f"{a} --{r2}--> {c}",
                            "premises": [f"{a} --{r1}--> {b}", f"{b} --{r2}--> {c}"],
                            "score": round(float(score), 3),
                            "evidence": {"sim(a,b)": round(float(s_ab), 3), "sim(b,c)": round(float(s_bc), 3)},
                        })

                        count += 1
                        if count >= max_new:
                            return proposals

        return proposals

    def _propose_semantic_pairs(
        self,
        top_per_node: int = 3,
        max_pairs: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar nodes not directly connected.
        Uses batched embeddings + matrix cosine.
        """
        nodes = self._all_nodes()
        if len(nodes) < 8:
            return []

        X = self._get_vecs(nodes)  # (N,D)
        sims = X @ X.T             # (N,N)

        proposals: List[Dict[str, Any]] = []
        count = 0
        N = len(nodes)

        # For each node, pick top candidates
        for i in range(N):
            a = nodes[i]

            # exclude self
            sims[i, i] = -1.0

            # exclude neighbors both directions
            neigh_a = set(sum(self._neighbors(a).values(), []))
            # pick indices by similarity descending
            idx = np.argsort(-sims[i])[: min(50, N)]  # scan top-50, then filter
            chosen = 0
            for j in idx:
                b = nodes[int(j)]
                if chosen >= top_per_node:
                    break

                if b in neigh_a:
                    continue
                if a in set(sum(self._neighbors(b).values(), [])):
                    continue

                s = float(sims[i, j])
                if s < self.semantic_sim_thresh:
                    break

                # Domain gating
                if self.enforce_domain_gating:
                    ba = _guess_bucket(a)
                    bb = _guess_bucket(b)
                    if ba != "unknown" and bb != "unknown" and ba != bb:
                        continue

                proposals.append({
                    "type": "semantic_link",
                    "hypothesis": f"{a} --related_to--> {b}",
                    "premises": [f"embedding_sim({a},{b})={s:.3f} (no direct edge)"],
                    "score": round(s, 3),
                    "evidence": {"similarity": round(s, 3)},
                })
                chosen += 1
                count += 1
                if count >= max_pairs:
                    return proposals

        return proposals

    # -------------------------
    # Public API
    # -------------------------
    def generate(self, top_n: int = 25) -> List[Dict[str, Any]]:
        transitive = self._propose_transitive(max_new=max(10, top_n // 2))
        semantic = self._propose_semantic_pairs()

        all_props = transitive + semantic
        # sort by score, then keep unique hypotheses
        all_props.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        seen = set()
        deduped = []
        for h in all_props:
            key = h.get("hypothesis", "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(h)

        ts = _now()
        for h in deduped:
            h["timestamp"] = ts

        result = deduped[:top_n]
        self._save_jsonl(result)
        self._append_log(result)
        return result

    # -------------------------
    # Persistence
    # -------------------------
    def _save_jsonl(self, hyps: List[Dict[str, Any]]):
        with open(self.out_jsonl, "a", encoding="utf-8") as f:
            for h in hyps:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")

    def _append_log(self, hyps: List[Dict[str, Any]]):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== Hypothesis Batch @ {_now()} ===\n")
            for h in hyps:
                f.write(f"[{h['type']}] {h['hypothesis']} (score={h['score']})\n")
                for p in h.get("premises", []):
                    f.write(f"  · {p}\n")
