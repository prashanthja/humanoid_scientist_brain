"""
Hypothesis Validator — Phase C Step 2 (Transformer Compatible)
--------------------------------------------------------------
Validates hypotheses produced by HypothesisGenerator using:
  • Knowledge Base (KB) evidence support
  • Semantic consistency using transformer embeddings (EmbeddingBridge)
  • Rolling confidence persistence across learning cycles

If a hypothesis remains consistently supported, it is promoted into the Knowledge Graph (KG).
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
        self.encoder = encoder  # ⚡ This is EmbeddingBridge (continual transformer)
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
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return float(np.dot(a, b) / (na * nb))

    def _embed(self, text: str) -> np.ndarray:
        """
        Use EmbeddingBridge.encode() → transformer-based embeddings.
        Fallback to zero-vector if not available.
        """
        try:
            emb = self.encoder.encode([text])  # EmbeddingBridge returns np.array
            if isinstance(emb, list):
                emb = np.array(emb)
            if emb.ndim == 2:
                emb = emb[0]
            return np.asarray(emb, dtype=np.float32)
        except Exception:
            return np.zeros(768, dtype=np.float32)

    def _parse_hypothesis(self, h: Dict[str, Any]):
        """
        Parse formats like:
          "A --rel--> B"
        """
        s = h.get("hypothesis", "")
        if "--" in s and "-->" in s:
            left, right = s.split("--", 1)
            rel, right = right.split("-->", 1)
            return left.strip(), rel.strip(), right.strip()
        return s.strip(), "related_to", s.strip()

    # ---------- validation logic ----------
    def _kb_support(self, subj: str, obj: str, relation: str) -> float:
        """
        Retrieve KB entries and check semantic alignment with hypothesis.
        """
        hyp_text = f"{subj} {relation} {obj}"
        hyp_vec = self._embed(hyp_text)

        cand_a = self.kb.query(subj) or []
        cand_b = self.kb.query(obj) or []
        cands = cand_a + cand_b
        if not cands:
            return 0.0

        cands = cands[: self.max_kb_check]
        sims = []
        for item in cands:
            text = item["text"] if isinstance(item, dict) else str(item)
            if not text:
                continue
            sim = self._cosine(hyp_vec, self._embed(text))
            sims.append(sim)

        if not sims:
            return 0.0
        return float(np.mean([s for s in sims if s >= self.kb_support_thresh]))

    def _semantic_consistency(self, subj: str, obj: str) -> float:
        """
        Measure semantic coherence between subject and object concepts.
        """
        return self._cosine(self._embed(subj), self._embed(obj))

    def _update_persistence(self, key: str, conf: float) -> float:
        """
        Smooth confidence history via exponential moving average.
        """
        rec = self.state["hypotheses"].get(key, {"ema": 0.0, "count": 0})
        count = rec["count"]
        alpha = 0.6 * math.exp(-0.05 * count) + 0.2
        ema = alpha * conf + (1 - alpha) * rec["ema"]
        self.state["hypotheses"][key] = {"ema": ema, "count": count + 1}
        return ema

    # ---------- main ----------
    def validate(self, hypotheses: List[Dict[str, Any]], cycle: int) -> List[Dict[str, Any]]:
        """
        Returns validated entries:
          {'hypothesis': str, 'support': float, 'consistency': float,
           'confidence': float, 'persistence': float, 'promote': bool}
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

        # persist
        self._save_state()
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
