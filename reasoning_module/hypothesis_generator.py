"""
Hypothesis Generator — Phase C · Step 1 (Transformer-compatible)
----------------------------------------------------------------
Generates new scientific hypotheses by combining:
• Graph transitivity patterns (A→B, B→C ⇒ maybe A→C)
• Semantic proximity in embedding space (high-similar-yet-unlinked pairs)

Now powered by the continual transformer via EmbeddingBridge.
Outputs JSONL to outputs/hypotheses.jsonl and an append-only text log.
"""

import os
import json
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np


class HypothesisGenerator:
    def __init__(self, knowledge_graph, encoder, kb=None,
                 out_dir: str = "outputs", log_dir: str = "logs"):
        """
        knowledge_graph: KnowledgeGraph instance
        encoder: EmbeddingBridge instance (must have encode(text) -> np.ndarray)
        kb: optional KnowledgeBase for additional context (optional)
        """
        self.kg = knowledge_graph
        self.encoder = encoder
        self.kb = kb

        self.out_dir = out_dir
        self.log_dir = log_dir
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.out_jsonl = os.path.join(self.out_dir, "hypotheses.jsonl")
        self.log_file = os.path.join(self.log_dir, "hypothesis_log.txt")

    # ---------- Embeddings ----------
    def _embed(self, text: str) -> np.ndarray:
        """Return normalized embedding using EmbeddingBridge (PyTorch transformer)."""
        if not text:
            return np.zeros(256, dtype=np.float32)
        try:
            vec = self.encoder.encode(text)
            if isinstance(vec, list):
                vec = np.array(vec)
            n = np.linalg.norm(vec) + 1e-9
            return vec / n
        except Exception:
            return np.zeros(256, dtype=np.float32)

    def _sim(self, a: str, b: str) -> float:
        """Cosine similarity between two text embeddings."""
        va, vb = self._embed(a), self._embed(b)
        if np.all(va == 0) or np.all(vb == 0):
            return 0.0
        return float(np.dot(va, vb))

    # ---------- Graph helpers ----------
    def _neighbors(self, node: str) -> Dict[str, List[str]]:
        """Return {relation: [neighbors...]} or empty dict."""
        try:
            rels = self.kg.get_relations(node)
            if not isinstance(rels, dict):
                return {}
            return {r: (v if isinstance(v, list) else list(v)) for r, v in rels.items()}
        except Exception:
            return {}

    def _has_edge(self, a: str, relation: str, c: str) -> bool:
        rels = self._neighbors(a)
        return c in rels.get(relation, [])

    def _all_nodes(self) -> List[str]:
        try:
            return list(self.kg.all_concepts())
        except Exception:
            try:
                return list(self.kg.graph.keys())
            except Exception:
                return []

    def _all_edges(self) -> List[Tuple[str, str, str]]:
        """Return list of (src, relation, dst)."""
        edges = []
        try:
            for a in self._all_nodes():
                for rel, objs in self._neighbors(a).items():
                    for b in objs:
                        edges.append((a, rel, b))
        except Exception:
            pass
        return edges

    # ---------- Pattern scanners ----------
    def _propose_transitive(self, max_new: int = 30) -> List[Dict[str, Any]]:
        """
        If A -r1-> B and B -r2-> C and NOT A -r2-> C, propose A -r2-> C.
        Score = 0.5*(sim(A,B)+sim(B,C)) with small penalty if A already strongly
        connected to C by any other relation.
        """
        proposals = []
        edges_by_src = defaultdict(list)
        for a, r1, b in self._all_edges():
            edges_by_src[a].append((r1, b))

        count = 0
        for a, out1 in edges_by_src.items():
            for (r1, b) in out1:
                out2 = self._neighbors(b)
                for r2, cs in out2.items():
                    for c in cs:
                        if a == c:
                            continue
                        if self._has_edge(a, r2, c):
                            continue  # already known
                        s_ab = self._sim(a, b)
                        s_bc = self._sim(b, c)
                        score = 0.5 * (s_ab + s_bc)
                        any_edge = any(c in lst for lst in self._neighbors(a).values())
                        if any_edge:
                            score *= 0.9
                        proposals.append({
                            "type": "graph_transitivity",
                            "hypothesis": f"{a} --{r2}--> {c}",
                            "premises": [f"{a} --{r1}--> {b}", f"{b} --{r2}--> {c}"],
                            "score": round(float(score), 3),
                            "evidence": {
                                "sim(a,b)": round(s_ab, 3),
                                "sim(b,c)": round(s_bc, 3),
                            }
                        })
                        count += 1
                        if count >= max_new:
                            return proposals
        return proposals

    def _propose_semantic_pairs(self, sim_thresh: float = 0.78, top_per_node: int = 3,
                                max_pairs: int = 40) -> List[Dict[str, Any]]:
        """
        For each node, find semantically similar nodes not directly connected;
        propose 'related_to' or 'influences' style hypotheses.
        Score = similarity.
        """
        nodes = self._all_nodes()
        if not nodes:
            return []

        sample_nodes = nodes if len(nodes) <= 300 else nodes[:300]

        proposals = []
        count = 0
        for i, a in enumerate(sample_nodes):
            va = self._embed(a)
            sims = []
            for b in sample_nodes:
                if a == b:
                    continue
                if b in sum(self._neighbors(a).values(), []):
                    continue
                if a in sum(self._neighbors(b).values(), []):
                    continue
                vb = self._embed(b)
                sim = float(np.dot(va, vb))
                sims.append((b, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            for b, s in sims[:top_per_node]:
                if s < sim_thresh:
                    continue
                proposals.append({
                    "type": "semantic_link",
                    "hypothesis": f"{a} --related_to--> {b}",
                    "premises": [f"embedding_sim({a},{b})={s:.3f} (no direct edge)"],
                    "score": round(s, 3),
                    "evidence": {"similarity": round(s, 3)}
                })
                count += 1
                if count >= max_pairs:
                    return proposals
        return proposals

    # ---------- Public API ----------
    def generate(self, top_n: int = 25) -> List[Dict[str, Any]]:
        """
        Generate a set of hypotheses ranked by score.
        Returns list of dicts with: type, hypothesis, premises, score, evidence, timestamp
        """
        transitive = self._propose_transitive(max_new=max(10, top_n // 2))
        semantic = self._propose_semantic_pairs()
        all_props = transitive + semantic
        all_props.sort(key=lambda x: x["score"], reverse=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for h in all_props:
            h["timestamp"] = ts
        result = all_props[:top_n]
        self._save_jsonl(result)
        self._append_log(result)
        return result

    # ---------- Persistence ----------
    def _save_jsonl(self, hyps: List[Dict[str, Any]]):
        with open(self.out_jsonl, "a", encoding="utf-8") as f:
            for h in hyps:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")

    def _append_log(self, hyps: List[Dict[str, Any]]):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n=== Hypothesis Batch @ {} ===\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            for h in hyps:
                f.write(f"[{h['type']}] {h['hypothesis']} (score={h['score']})\n")
                for p in h.get("premises", []):
                    f.write(f"  · {p}\n")
