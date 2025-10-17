# reasoning_module/hypothesis_validator.py
"""
Hypothesis Validator — Phase C Step 2
-------------------------------------
Validates hypotheses produced by HypothesisGenerator using:
  • KB support (retrieval evidence)
  • Semantic consistency (encoder similarity)
  • Persistence across cycles (rolling confidence)

If confidence passes a threshold, caller may promote the hypothesis into the KG.
Persists scores in data/validated_hypotheses.json
"""

import os
import json
import math
from typing import List, Dict, Any
import numpy as np


class HypothesisValidator:
    def __init__(
        self,
        kb,
        encoder,
        kg=None,
        store_path: str = "data/validated_hypotheses.json",
        kb_support_thresh: float = 0.55,
        semantic_thresh: float = 0.55,
        promote_conf_thresh: float = 0.85,
        max_kb_check: int = 200,
    ):
        self.kb = kb
        self.encoder = encoder
        self.kg = kg
        self.store_path = store_path
        self.kb_support_thresh = kb_support_thresh
        self.semantic_thresh = semantic_thresh
        self.promote_conf_thresh = promote_conf_thresh
        self.max_kb_check = max_kb_check

        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self.state = self._load_state()

    # ---------- persistence ----------
    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"hypotheses": {}}
        return {"hypotheses": {}}

    def _save_state(self) -> None:
        tmp = self.store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.store_path)

    # ---------- helpers ----------
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a) + 1e-8
        nb = np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / (na * nb))

    def _embed(self, text: str) -> np.ndarray:
        try:
            vec = self.encoder.get_vector(text)
            return np.asarray(vec, dtype=np.float32)
        except Exception:
            return np.zeros(64, dtype=np.float32)

    def _parse_hypothesis(self, h: Dict[str, Any]):
        """
        Expect formats from HypothesisGenerator:
          - {'type': 'edge', 'hypothesis': 'A --rel--> B', 'score': ...}
          - {'type': 'semantic_link', 'hypothesis': 'A --related_to--> B', 'score': ...}
        """
        s = h.get("hypothesis", "")
        # Very simple parse: "left --rel--> right"
        if "--" in s and "-->" in s:
            left, right = s.split("--", 1)
            rel, right = right.split("-->", 1)
            return left.strip(), rel.strip(), right.strip()
        # fallback: treat entire hypothesis as one string
        return s.strip(), "related_to", s.strip()

    def _kb_support(self, subj: str, obj: str, relation: str) -> float:
        """
        Retrieve KB items using subject/object as queries and measure:
        - how many items semantically align with 'subj rel obj'
        Returns ratio in [0,1].
        """
        hyp_text = f"{subj} {relation} {obj}"
        hyp_vec = self._embed(hyp_text)

        # collect candidate KB items (subject/object queries)
        cand_a = self.kb.query(subj) or []
        cand_b = self.kb.query(obj) or []
        cands = cand_a + cand_b
        if not cands:
            return 0.0

        # limit to first N for speed
        cands = cands[: self.max_kb_check]

        hits = 0
        total = 0
        for item in cands:
            text = item["text"] if isinstance(item, dict) else str(item)
            if not text:
                continue
            total += 1
            sim = self._cosine(hyp_vec, self._embed(text))
            if sim >= self.kb_support_thresh:
                hits += 1

        return (hits / total) if total > 0 else 0.0

    def _semantic_consistency(self, subj: str, obj: str) -> float:
        """
        Consistency between subject/object concepts by semantic similarity.
        """
        return self._cosine(self._embed(subj), self._embed(obj))

    def _update_persistence(self, key: str, conf: float) -> float:
        """
        Rolling persistence = exponential moving average over confidence.
        """
        rec = self.state["hypotheses"].get(key, {"ema": 0.0, "count": 0})
        count = rec["count"]
        # decay gets gentler with more observations
        alpha = 0.6 * math.exp(-0.05 * count) + 0.2
        ema = alpha * conf + (1 - alpha) * rec["ema"]
        self.state["hypotheses"][key] = {"ema": ema, "count": count + 1}
        return ema

    # ---------- main API ----------
    def validate(self, hypotheses: List[Dict[str, Any]], cycle: int) -> List[Dict[str, Any]]:
        """
        Returns list of validated entries:
          {'hypothesis': str, 'type': str, 'relation': str,
           'support': float, 'consistency': float, 'confidence': float,
           'persistence': float, 'promote': bool}
        """
        results = []
        if not hypotheses:
            return results

        for h in hypotheses:
            subj, rel, obj = self._parse_hypothesis(h)
            support = self._kb_support(subj, obj, rel)
            consistency = self._semantic_consistency(subj, obj)

            # combine
            confidence = 0.55 * support + 0.45 * consistency
            key = h.get("hypothesis", f"{subj} --{rel}--> {obj}")
            persistence = self._update_persistence(key, confidence)

            promote = (confidence >= self.promote_conf_thresh) and (persistence >= 0.7)

            results.append({
                "hypothesis": key,
                "type": h.get("type", "edge"),
                "relation": rel,
                "subject": subj,
                "object": obj,
                "support": round(support, 3),
                "consistency": round(consistency, 3),
                "confidence": round(confidence, 3),
                "persistence": round(persistence, 3),
                "promote": promote,
            })

        # persist state
        self._save_state()
        # sort by confidence desc
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
