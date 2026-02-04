"""
Hypothesis Validator — v2 (Bridge-consistent, batched, NaN-safe)
--------------------------------------------------------------
Validates hypotheses using:
  • KB evidence support (batched embedding similarity)
  • Semantic coherence (subject/object similarity)
  • Persistence EMA across cycles

Key fixes vs v1:
- Uses encoder.encode_texts() consistently
- Uses correct embedding dim (no 768 hardcode)
- NaN-safe support scoring (top-k mean)
- Batched KB embedding for speed
"""

from __future__ import annotations

import os
import json
import math
from typing import List, Dict, Any, Tuple
import numpy as np


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return (X / denom).astype(np.float32)


class HypothesisValidator:
    def __init__(
        self,
        kb,
        encoder,
        kg=None,
        store_path: str = "data/validated_hypotheses.json",
        kb_support_thresh: float = 0.55,
        semantic_thresh: float = 0.55,
        promote_conf_thresh: float = 0.72,      # lowered; persistence does the heavy lifting
        promote_support_min: float = 0.60,      # require real KB support
        max_kb_check: int = 200,
        topk_support: int = 8,                 # average of top-k sims
        embed_max_len: int = 256,
    ):
        self.kb = kb
        self.encoder = encoder
        self.kg = kg

        self.store_path = store_path
        self.kb_support_thresh = float(kb_support_thresh)
        self.semantic_thresh = float(semantic_thresh)

        self.promote_conf_thresh = float(promote_conf_thresh)
        self.promote_support_min = float(promote_support_min)

        self.max_kb_check = int(max_kb_check)
        self.topk_support = int(topk_support)
        self.embed_max_len = int(embed_max_len)

        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self.state = self._load_state()

        # Infer embedding dim once
        self.dim = self._infer_dim()

    # ---------- persistence ----------
    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    return obj if isinstance(obj, dict) else {"hypotheses": {}}
            except Exception:
                return {"hypotheses": {}}
        return {"hypotheses": {}}

    def _save_state(self) -> None:
        tmp = self.store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.store_path)

    # ---------- embeddings ----------
    def _infer_dim(self) -> int:
        v = self._embed_texts(["dim_probe"])
        if v.size == 0:
            return 256
        return int(v.shape[1])

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Returns (N,D) normalized float32.
        Supports EmbeddingBridge.encode_texts or encoder.embed fallback.
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

        # token-level -> mean pool
        if arr.ndim == 3:
            arr = arr.mean(axis=1)

        if arr.ndim != 2:
            # last-resort fallback
            arr = np.zeros((len(texts), 256), dtype=np.float32)

        return _normalize_rows(arr)

    def _embed(self, text: str) -> np.ndarray:
        v = self._embed_texts([text])
        if v.ndim == 2 and v.shape[0] == 1:
            return v[0]
        return np.zeros(self.dim, dtype=np.float32)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0.0
        # embeddings are normalized already
        return float(np.dot(a, b))

    # ---------- parsing ----------
    def _parse_hypothesis(self, h: Dict[str, Any]) -> Tuple[str, str, str]:
        s = str(h.get("hypothesis", "") or "").strip()
        if "--" in s and "-->" in s:
            left, right = s.split("--", 1)
            rel, right = right.split("-->", 1)
            return left.strip(), rel.strip(), right.strip()
        return s, "related_to", s

    # ---------- validation logic ----------
    def _kb_support(self, subj: str, obj: str, relation: str) -> float:
        """
        KB support = mean(top-k cosine(hyp, kb_text)) among best matches.
        NaN-safe. Batched embedding.
        """
        hyp_text = f"{subj} {relation} {obj}"
        hyp_vec = self._embed(hyp_text)

        cand_a = self.kb.query(subj) or []
        cand_b = self.kb.query(obj) or []
        cands = (cand_a + cand_b)[: self.max_kb_check]

        texts = []
        for item in cands:
            if isinstance(item, dict):
                t = str(item.get("text") or "").strip()
            else:
                t = str(item or "").strip()
            if t:
                texts.append(t)

        if not texts:
            return 0.0

        X = self._embed_texts(texts)        # (N,D) normalized
        sims = X @ hyp_vec                  # (N,)

        if sims.size == 0:
            return 0.0

        # Keep only sims above a floor, else support is 0
        sims_sorted = np.sort(sims)[::-1]
        sims_kept = sims_sorted[sims_sorted >= self.kb_support_thresh]

        if sims_kept.size == 0:
            return 0.0

        k = min(self.topk_support, int(sims_kept.size))
        return float(np.mean(sims_kept[:k]))

    def _semantic_consistency(self, subj: str, obj: str) -> float:
        return self._cosine(self._embed(subj), self._embed(obj))

    def _update_persistence(self, key: str, conf: float) -> float:
        rec = self.state["hypotheses"].get(key, {"ema": 0.0, "count": 0})
        count = int(rec.get("count", 0))
        prev = float(rec.get("ema", 0.0))

        # EMA that stabilizes as count grows
        alpha = 0.35 if count > 20 else (0.55 - 0.01 * count)
        alpha = float(max(0.20, min(0.55, alpha)))

        ema = alpha * float(conf) + (1.0 - alpha) * prev
        self.state["hypotheses"][key] = {"ema": ema, "count": count + 1}
        return ema

    # ---------- main ----------
    def validate(self, hypotheses: List[Dict[str, Any]], cycle: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not hypotheses:
            return results

        for h in hypotheses:
            subj, rel, obj = self._parse_hypothesis(h)

            support = self._kb_support(subj, obj, rel)
            consistency = self._semantic_consistency(subj, obj)

            confidence = 0.65 * support + 0.35 * consistency

            key = str(h.get("hypothesis") or f"{subj} --{rel}--> {obj}")
            persistence = self._update_persistence(key, confidence)

            promote = (
                confidence >= self.promote_conf_thresh
                and support >= self.promote_support_min
                and persistence >= 0.70
                and self.kg is not None
            )

            results.append({
                "hypothesis": key,
                "type": h.get("type", "edge"),
                "relation": rel,
                "subject": subj,
                "object": obj,
                "support": round(float(support), 3),
                "consistency": round(float(consistency), 3),
                "confidence": round(float(confidence), 3),
                "persistence": round(float(persistence), 3),
                "promote": bool(promote),
                "cycle": int(cycle),
            })

        self._save_state()
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
