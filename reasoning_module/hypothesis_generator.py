"""
Hypothesis Generator — v3
--------------------------
Reads from KnowledgeGraph and generates hypotheses via:
1. Graph transitivity — A->B->C implies A->C
2. Semantic pairs — high similarity concepts with no direct edge

Changes from v2:
- Added transformer_efficiency domain to domain gating
- Expanded _ALLOWED_RELATIONS to include pipeline-generated relation types
- encoder is now optional (falls back to graph-only mode)
- Added generate_from_kg() as simple entry point
"""

from __future__ import annotations

import os
import json
import time
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np


# ─────────────────────────────────────────────
# Concept quality gates
# ─────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "when", "under", "for",
    "to", "of", "in", "on", "at", "by", "with", "without", "from", "into",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as",
    "we", "they", "you", "i",
    "low", "high", "small", "large", "rapidly", "major", "kind", "basis",
}

# All relation types — both old and new pipeline-generated

_ALLOWED_RELATIONS = {
    "causes", "leads_to", "influences", "increases", "decreases",
    "part_of", "type_of", "subclass_of", "related_to", "associated_with",
    "supports_efficiency", "partially_supports", "reduces", "improves",
    "has_tradeoff", "contradicts",
}

_DOMAIN_KEYWORDS = {
    "transformer_efficiency": [
        "transformer", "attention", "flashattention", "flash attention",
        "sparse attention", "mixture of experts", "moe",
        "kv cache", "kv-cache", "paged attention",
        "speculative decoding", "lora", "low-rank",
        "inference", "latency", "throughput", "memory overhead",
        "context length", "long context", "token", "llm",
        "quantization", "pruning", "distillation",
        "rotary", "rope", "state space", "mamba", "rwkv",
        "MemoryOverhead", "TransformerEfficiency", "ModelQuality",
        "InferenceSpeed", "KVCache", "FlashAttention",
        "MixtureOfExperts", "SparseAttention", "LoRA",
        "SpeculativeDecoding", "Latency", "Throughput",
    ],
    "water_microplastics": [
        "microplastic", "nanoplastic", "drinking water", "water treatment",
        "coagulation", "filtration", "membrane",
    ],
    "physics": [
        "relativity", "entropy", "thermodynamics", "electromagnet",
        "mass", "force", "energy", "gravity",
    ],
    "math": [
        "theorem", "lemma", "proof", "tensor", "manifold",
        "topology", "algebra", "calculus",
    ],
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _clean(s: str) -> str:
    return " ".join((s or "").strip().split())


def _looks_like_concept(s: str) -> bool:
    s = _clean(s)
    if not s or len(s) < 3:
        return False
    
    # Reject paper titles — colon in a long string signals "Title: Subtitle"
    if ":" in s and len(s) > 25:
        return False
    
    if len(s) > 25 and " " in s:
        return False

    # Reject long multi-word strings — concepts are short
    if len(s) > 50 and " " in s:
        return False

    low = s.lower()
    toks = re.findall(r"[a-zA-Z0-9\-]+", low)
    if not toks:
        return False

    num_chars = sum(ch.isdigit() for ch in low)
    if num_chars / max(1, len(low)) > 0.30:
        return False

    stop = sum(1 for t in toks if t in _STOPWORDS)
    if stop / max(1, len(toks)) > 0.60:
        return False

    content = [t for t in toks if t not in _STOPWORDS and len(t) > 2]
    if len(content) < 1:
        return False

    if len(low) > 70 and low[-1] in {".", "!", "?"}:
        return False

    if "-->" in low:
        return False

    bad = [
        "best practice", "a basic knowledge", "this paper",
        "the editorial", "reviewers are", "a concise",
        "generallimitation", "general limitation",
    ]
    if any(b in low for b in bad):
        return False

    return True


def _guess_bucket(text: str) -> str:
    low = (text or "").lower()
    scores = {}
    for bucket, kws in _DOMAIN_KEYWORDS.items():
        scores[bucket] = sum(1 for w in kws if w.lower() in low)
    best, score = max(scores.items(), key=lambda x: x[1])
    return best if score > 0 else "unknown"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ─────────────────────────────────────────────
# Human-readable hypothesis templates
# ─────────────────────────────────────────────

RELATION_TEMPLATES = {
    "supports_efficiency": "{a} supports efficiency improvements in {b}",
    "reduces": "{a} reduces {b}",
    "improves": "{a} improves {b}",
    "has_tradeoff": "{a} introduces tradeoffs with {b}",
    "related_to": "{a} and {b} may be related",
    "causes": "{a} causes {b}",
    "leads_to": "{a} leads to {b}",
    "increases": "{a} increases {b}",
    "decreases": "{a} decreases {b}",
    "associated_with": "{a} is associated with {b}",
}


def _make_readable(a: str, rel: str, c: str) -> str:
    tmpl = RELATION_TEMPLATES.get(rel, "{a} --{rel}--> {c}")
    return tmpl.format(a=a, b=c, c=c, rel=rel)


class HypothesisGenerator:

    BAD_PATTERNS = [
        r"\bunknown\b",
        r"\bchapter\b",
        r"\bcopyright\b",
        r"https?://",
        r"\bwww\.",
        r"\bdoi:\b",
    ]

    def __init__(
        self,
        kg,
        encoder=None,
        kb=None,
        out_dir: str = "outputs",
        log_dir: str = "logs",
        embed_max_len: int = 256,
        max_nodes: int = 400,
        semantic_sim_thresh: float = 0.78,
        enforce_domain_gating: bool = True,
    ):
        self.kg = kg
        self.encoder = encoder
        self.kb = kb
        self.embed_max_len = int(embed_max_len)
        self.max_nodes = int(max_nodes)
        self.semantic_sim_thresh = float(semantic_sim_thresh)
        self.enforce_domain_gating = bool(enforce_domain_gating)
        self.out_dir = out_dir
        self.log_dir = log_dir
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.out_jsonl = os.path.join(out_dir, "hypotheses.jsonl")
        self._cache: Dict[str, np.ndarray] = {}

    # ─────────────────────────────────────────
    # KG helpers
    # ─────────────────────────────────────────

    def _neighbors(self, node: str) -> Dict[str, List[str]]:
        try:
            rels = self.kg.get_relations(node)
            if not isinstance(rels, dict):
                return {}
            out = {}
            for r, v in rels.items():
                if r not in _ALLOWED_RELATIONS:
                    continue
                if isinstance(v, str):
                    out[r] = [v]
                elif isinstance(v, (list, tuple)):
                    out[r] = [str(x) for x in v]
            return out
        except Exception:
            return {}

    def _has_edge(self, a: str, rel: str, c: str) -> bool:
        return c in self._neighbors(a).get(rel, [])

    def _all_nodes(self) -> List[str]:
        try:
            nodes = list(self.kg.all_concepts())
        except Exception:
            nodes = list(getattr(self.kg, "graph", {}).keys())
        nodes = [_clean(n) for n in nodes if _looks_like_concept(_clean(n))]
        return nodes[: self.max_nodes]

    def _all_edges(self) -> List[Tuple[str, str, str]]:
        edges = []
        for a in self._all_nodes():
            for rel, objs in self._neighbors(a).items():
                for b in objs:
                    b = _clean(str(b))
                    if _looks_like_concept(b):
                        edges.append((a, rel, b))
        return edges

    @staticmethod
    def is_valid(text: str) -> bool:
        if not text or len(text) < 10:
            return False
        low = text.lower()
        for p in HypothesisGenerator.BAD_PATTERNS:
            if re.search(p, low):
                return False
        return True

    # ─────────────────────────────────────────
    # Embeddings (optional)
    # ─────────────────────────────────────────

    def _encode(self, texts: List[str]) -> Optional[np.ndarray]:
        if self.encoder is None:
            return None
        try:
            if hasattr(self.encoder, "encode_texts"):
                arr = np.asarray(
                    self.encoder.encode_texts(texts, max_len=self.embed_max_len),
                    dtype=np.float32,
                )
            elif hasattr(self.encoder, "embed"):
                arr = np.asarray(
                    self.encoder.embed(texts, max_len=self.embed_max_len),
                    dtype=np.float32,
                )
            else:
                return None
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            return (arr / norms).astype(np.float32)
        except Exception:
            return None

    def _get_vecs(self, nodes: List[str]) -> Optional[np.ndarray]:
        missing = [n for n in nodes if n not in self._cache]
        if missing:
            vecs = self._encode(missing)
            if vecs is None:
                return None
            for n, v in zip(missing, vecs):
                self._cache[n] = v
        try:
            return np.stack([self._cache[n] for n in nodes], axis=0)
        except Exception:
            return None

    # ─────────────────────────────────────────
    # Strategy 1: Graph transitivity
    # ─────────────────────────────────────────

    def _propose_transitive(self, max_new: int = 30) -> List[Dict[str, Any]]:
        proposals = []
        edges = self._all_edges()
        if not edges:
            return proposals

        edges_by_src = defaultdict(list)
        for a, r, b in edges:
            edges_by_src[a].append((r, b))

        nodes = self._all_nodes()
        vecs = self._get_vecs(nodes) if self.encoder else None

        count = 0
        for a, out1 in edges_by_src.items():
            for r1, b in out1:
                out2 = self._neighbors(b)
                for r2, cs in out2.items():
                    for c in cs:
                        c = _clean(str(c))
                        if not _looks_like_concept(c):
                            continue
                        if a == c:
                            continue
                        if self._has_edge(a, r2, c):
                            continue

                        if self.enforce_domain_gating:
                            buckets = [_guess_bucket(x) for x in (a, b, c)]
                            known = [x for x in buckets if x != "unknown"]
                            if known and len(set(known)) > 1:
                                continue

                        readable = _make_readable(a, r2, c)
                        if not self.is_valid(readable):
                            continue

                        score = 0.5
                        if vecs is not None and a in self._cache and c in self._cache:
                            score = _cos(self._cache[a], self._cache[c])

                        proposals.append({
                            "type": "graph_transitivity",
                            "hypothesis": readable,
                            "graph_triple": f"{a} --[{r2}]--> {c}",
                            "premises": [
                                f"{a} --[{r1}]--> {b}",
                                f"{b} --[{r2}]--> {c}",
                            ],
                            "score": round(float(score), 3),
                        })
                        count += 1
                        if count >= max_new:
                            return proposals
        return proposals

    # ─────────────────────────────────────────
    # Strategy 2: Semantic pairs (encoder only)
    # ─────────────────────────────────────────

    def _propose_semantic_pairs(self, max_pairs: int = 20) -> List[Dict[str, Any]]:
        if self.encoder is None:
            return []

        nodes = self._all_nodes()
        if len(nodes) < 4:
            return []

        X = self._get_vecs(nodes)
        if X is None:
            return []

        sims = X @ X.T
        proposals = []
        count = 0

        for i, a in enumerate(nodes):
            sims[i, i] = -1.0
            neigh_a = set(sum(self._neighbors(a).values(), []))
            top = np.argsort(-sims[i])[:20]

            for j in top:
                b = nodes[int(j)]
                s = float(sims[i, j])
                if s < self.semantic_sim_thresh:
                    break
                if b in neigh_a:
                    continue

                if self.enforce_domain_gating:
                    ba, bb = _guess_bucket(a), _guess_bucket(b)
                    if ba != "unknown" and bb != "unknown" and ba != bb:
                        continue

                readable = f"{a} and {b} may share an efficiency relationship"
                if not self.is_valid(readable):
                    continue

                proposals.append({
                    "type": "semantic_link",
                    "hypothesis": readable,
                    "graph_triple": f"{a} --[related_to]--> {b}",
                    "premises": [f"embedding_sim({a},{b})={s:.3f}"],
                    "score": round(s, 3),
                })
                count += 1
                if count >= max_pairs:
                    return proposals

        return proposals

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def generate(self, top_n: int = 25) -> List[Dict[str, Any]]:
        transitive = self._propose_transitive(max_new=max(10, top_n))
        semantic = self._propose_semantic_pairs(max_pairs=max(10, top_n // 2))

        all_props = transitive + semantic
        all_props.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        seen = set()
        result = []
        ts = _now()
        for h in all_props:
            key = h.get("hypothesis", "").strip().lower()
            if not key or key in seen:
                continue
            if not self.is_valid(h["hypothesis"]):
                continue
            seen.add(key)
            h["timestamp"] = ts
            result.append(h)
            if len(result) >= top_n:
                break

        self._save(result)
        return result

    def _save(self, hyps: List[Dict[str, Any]]):
        with open(self.out_jsonl, "a", encoding="utf-8") as f:
            for h in hyps:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")