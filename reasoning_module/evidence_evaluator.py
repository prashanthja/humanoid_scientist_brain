# reasoning_module/evidence_evaluator.py
# ------------------------------------------------------------
# Evidence Evaluator (Phase F)
# ------------------------------------------------------------
# Given hypotheses, retrieve supporting/contradicting evidence from the KB
# using the continual-transformer embeddings, compute evidence strength,
# and output a verdict with an evidence summary.
#
# Outputs append-only logs to: logs/evidence_reports.jsonl
# Optional on-disk embedding cache:
#   - data/kb_embed_index.json  (metadata)
#   - data/kb_embeddings.npy    (float32 matrix [N, D])
#
# The "encoder" argument must expose:
#   - encode_texts(list[str]) -> np.ndarray [N, D]   (preferred)
#   - or embed(list[str]) -> np.ndarray [N, D]
#   - or get_vector(str) -> np.ndarray [D]
#
# The KB is expected to support kb.query(query_string) -> list[dict]
# where dict may include: id, text, paper_title, source, timestamp, url
# ------------------------------------------------------------

from __future__ import annotations
import os
import re
import json
import time
import math
import hashlib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# ---------- helpers ----------

NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno evidence\b", r"\bdoes\s+not\b", r"\bfails?\s+to\b",
    r"\binconsistent\s+with\b", r"\bcontradict(s|ed|ion)?\b",
    r"\bdispro(ve|ven|ves)\b", r"\brefute(s|d|ation)?\b",
]

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _safe_text(x: Any) -> str:
    if isinstance(x, dict):
        t = x.get("text") or x.get("paper_title") or ""
        return str(t)
    return str(x or "")

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / (na * nb))

def _hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def _year_from_item(item: Dict[str, Any]) -> Optional[int]:
    ts = item.get("timestamp") or item.get("date") or ""
    m = re.search(r"\b(19\d{2}|20\d{2})\b", str(ts))
    return int(m.group(1)) if m else None

def _source_trust(source: str | None) -> float:
    if not source:
        return 0.8
    s = source.lower()
    if "arxiv" in s:
        return 0.9
    if "crossref" in s:
        return 0.95
    if "rss" in s or "web" in s:
        return 0.75
    if "fetcher" in s:
        return 0.7
    return 0.8

def _polarity_is_contradiction(text: str) -> bool:
    t = text.lower()
    return any(re.search(pat, t) for pat in NEGATION_PATTERNS)


class EvidenceEvaluator:
    """
    Evaluate hypotheses using KB evidence + continual transformer embeddings.

    Usage:
        evaluator = EvidenceEvaluator(kb, encoder, kg)
        enriched = evaluator.evaluate_batch(validated_hypotheses)
    """
    def __init__(
        self,
        kb,
        encoder,
        kg=None,
        cache_dir: str = "data",
        log_path: str = "logs/evidence_reports.jsonl",
        top_k_search: int = 1000,
        top_n_report: int = 10,
        support_sim_thresh: float = 0.80,
        contra_sim_thresh: float = 0.60,
    ):
        self.kb = kb
        self.encoder = encoder   # EmbeddingBridge or OnlineTrainer wrapper
        self.kg = kg

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.cache_dir = cache_dir
        self.log_path = log_path

        self.idx_meta_path = os.path.join(cache_dir, "kb_embed_index.json")
        self.idx_vec_path  = os.path.join(cache_dir, "kb_embeddings.npy")

        self.top_k_search = top_k_search
        self.top_n_report = top_n_report
        self.support_sim_thresh = support_sim_thresh
        self.contra_sim_thresh  = contra_sim_thresh

        # Lazy-load cache
        self._meta: List[Dict[str, Any]] = []
        self._emb: Optional[np.ndarray] = None
        self._dim: Optional[int] = None

        self._ensure_cache()

    # -------- cache building / update --------

    def _ensure_cache(self):
        """
        Build (or refresh) a compact embedding cache for KB items.
        Safe & robust even if KB schema varies; falls back to text-hash keys.
        """
        try:
            # Load prior cache if present
            if os.path.exists(self.idx_meta_path) and os.path.exists(self.idx_vec_path):
                with open(self.idx_meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                self._emb = np.load(self.idx_vec_path)
                self._dim = self._emb.shape[1] if self._emb.size else None
            else:
                self._rebuild_cache()
        except Exception:
            # If anything goes wrong, just rebuild
            self._rebuild_cache()

    def _rebuild_cache(self):
        items = self.kb.query("") or []
        # Keep at most top_k_search for speed on first build
        if len(items) > self.top_k_search:
            items = items[: self.top_k_search]

        texts = []
        meta  = []
        for it in items:
            text = _safe_text(it)
            if not text:
                continue
            key = it.get("id") if isinstance(it, dict) and ("id" in it) else _hash_key(text)
            meta.append({
                "key": key,
                "paper_title": it.get("paper_title", "") if isinstance(it, dict) else "",
                "source": it.get("source", "") if isinstance(it, dict) else "",
                "timestamp": it.get("timestamp", "") if isinstance(it, dict) else "",
                "url": it.get("url", "") if isinstance(it, dict) else "",
                "text": text[:1000],  # preview
            })
            texts.append(text)

        vecs = self._encode(texts)
        self._meta = meta
        self._emb = vecs.astype("float32")
        self._dim = self._emb.shape[1] if self._emb.size else None

        with open(self.idx_meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        np.save(self.idx_vec_path, self._emb)

    def refresh_cache_incremental(self, new_items: List[Dict[str, Any]]):
        """
        Append new embeddings for unseen KB items to the cache.
        """
        if not new_items:
            return
        # Build set of known keys
        known = set(m["key"] for m in self._meta)
        texts, metas = [], []
        for it in new_items:
            text = _safe_text(it)
            if not text:
                continue
            key = it.get("id") if isinstance(it, dict) and ("id" in it) else _hash_key(text)
            if key in known:
                continue
            metas.append({
                "key": key,
                "paper_title": it.get("paper_title", ""),
                "source": it.get("source", ""),
                "timestamp": it.get("timestamp", ""),
                "url": it.get("url", ""),
                "text": text[:1000],
            })
            texts.append(text)

        if not texts:
            return

        vecs = self._encode(texts).astype("float32")
        # Append
        self._meta.extend(metas)
        self._emb = vecs if self._emb is None or self._emb.size == 0 else np.vstack([self._emb, vecs])
        self._dim = self._emb.shape[1]

        # Save
        with open(self.idx_meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        np.save(self.idx_vec_path, self._emb)

    # -------- embedding calls --------

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Try encoder.encode_texts first; fall back to .embed or single .get_vector.
        """
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return np.zeros((0, self._dim or 256), dtype=np.float32)

        # Preferred
        if hasattr(self.encoder, "encode_texts"):
            arr = self.encoder.encode_texts(texts)
            return np.asarray(arr, dtype=np.float32)

        # Bridge/Trainer style
        if hasattr(self.encoder, "embed"):
            arr = self.encoder.embed(texts)
            return np.asarray(arr, dtype=np.float32)

        # Very last fallback
        if hasattr(self.encoder, "get_vector"):
            vecs = [np.asarray(self.encoder.get_vector(t), dtype=np.float32) for t in texts]
            return np.stack(vecs, axis=0)

        # Unknown encoder interface
        raise RuntimeError("Encoder does not expose encode_texts/embed/get_vector")

    # -------- claim extraction --------

    def _parse_hypothesis_text(self, h: Dict[str, Any]) -> str:
        """
        Accepts formats like "A --rel--> B" or arbitrary sentences.
        Returns a single claim string we can embed.
        """
        s = h.get("hypothesis", "")
        if "--" in s and "-->" in s:
            # "A --rel--> B"
            left, right = s.split("--", 1)
            rel, right = right.split("-->", 1)
            left, rel, right = left.strip(), rel.strip(), right.strip()
            return f"{left} {rel.replace('_',' ')} {right}"
        return s

    def _split_claims(self, claim: str) -> List[str]:
        """
        Break a complex claim into smaller clauses (very light heuristic).
        """
        if not claim:
            return []
        # Split by ';' or ' and ' or commas (lightweight)
        parts = re.split(r";|\band\b|,|\.\s+", claim)
        parts = [p.strip() for p in parts if p and len(p.strip()) > 3]
        # Ensure we always keep the original claim first
        if claim not in parts:
            parts = [claim] + parts
        # Deduplicate while keeping order
        seen, out = set(), []
        for p in parts:
            if p.lower() in seen:
                continue
            seen.add(p.lower())
            out.append(p)
        return out[:5]  # cap

    # -------- retrieval & scoring --------

    def _score_evidence(self, sim: float, item: Dict[str, Any]) -> float:
        # Time weighting (older papers get a mild age penalty)
        year = _year_from_item(item)
        now = time.gmtime().tm_year
        age = max(0, (now - year)) if year else 0
        age_factor = 1.0 + (age / 20.0)  # 20 years halves weight roughly

        trust = _source_trust(item.get("source"))
        # Final score balances similarity, trust, and recency
        score = (sim * trust) / age_factor
        return float(score)

    def _rank_candidates(
        self, claim_vec: np.ndarray, limit: int
    ) -> List[Tuple[int, float]]:
        """
        Brute-force cosine top-k over cached embeddings. Fast enough for N~10k.
        Returns list of (index, similarity) sorted desc.
        """
        if self._emb is None or self._emb.size == 0:
            return []
        sims = self._emb @ (claim_vec / (np.linalg.norm(claim_vec) + 1e-8))
        idx = np.argsort(-sims)[:limit]
        return [(int(i), float(sims[i])) for i in idx]

    # -------- public API --------

    def evaluate_one(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns hypothesis dict enriched with:
          - evidence: {supporting: [...], contradicting: [...], neutral: [...]}
          - evidence_confidence: float in [0,1]
          - verdict: "supported" | "inconclusive" | "contradicted"
        """
        claim_text = self._parse_hypothesis_text(hypothesis)
        subclaims = self._split_claims(claim_text)

        # Encode all subclaims together and search
        claim_vecs = self._encode(subclaims)
        supporting, contradicting, neutral = [], [], []

        for ci, (c_text, c_vec) in enumerate(zip(subclaims, claim_vecs)):
            # top-K retrieve
            top_hits = self._rank_candidates(c_vec, limit=self.top_k_search)
            # score & categorize top-N for report
            for j, (k, sim) in enumerate(top_hits[: self.top_n_report]):
                meta = self._meta[k]
                item = {
                    "title": meta.get("paper_title") or "(untitled)",
                    "source": meta.get("source", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "url": meta.get("url", ""),
                    "preview": meta.get("text", ""),
                    "similarity": round(sim, 3),
                    "score": 0.0,  # will be filled
                    "subclaim": c_text,
                }
                item["score"] = round(self._score_evidence(sim, meta), 3)

                # Polarity guess: high-sim + contains negation → contradiction
                if sim >= self.support_sim_thresh and not _polarity_is_contradiction(meta.get("text", "")):
                    supporting.append(item)
                elif sim >= self.contra_sim_thresh and _polarity_is_contradiction(meta.get("text", "")):
                    contradicting.append(item)
                else:
                    neutral.append(item)

        # Aggregate confidence
        def agg(items: List[Dict[str, Any]], top_m=5) -> float:
            if not items:
                return 0.0
            xs = sorted((it["score"] for it in items), reverse=True)[:top_m]
            return float(sum(xs) / max(1, len(xs)))

        conf_support = agg(supporting)
        conf_contra  = agg(contradicting)
        evidence_confidence = max(0.0, min(1.0, conf_support))  # clamp

        # Verdict
        if conf_support >= 0.6 and conf_contra < 0.35:
            verdict = "supported"
        elif conf_contra >= 0.5 and conf_contra > conf_support:
            verdict = "contradicted"
        else:
            verdict = "inconclusive"

        enriched = dict(hypothesis)
        enriched["evidence"] = {
            "supporting": supporting[: self.top_n_report],
            "contradicting": contradicting[: self.top_n_report],
            "neutral": neutral[: self.top_n_report],
        }
        enriched["evidence_confidence"] = round(evidence_confidence, 3)
        enriched["verdict"] = verdict
        enriched["evaluated_at"] = _now()

        # Append to log
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": enriched["evaluated_at"],
                    "hypothesis": hypothesis.get("hypothesis", ""),
                    "type": hypothesis.get("type", ""),
                    "verdict": verdict,
                    "evidence_confidence": enriched["evidence_confidence"],
                    "top_support_titles": [e["title"] for e in enriched["evidence"]["supporting"][:3]],
                    "top_contra_titles": [e["title"] for e in enriched["evidence"]["contradicting"][:2]],
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return enriched

    def evaluate_batch(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hypotheses:
            return []
        # Light incremental cache refresh with latest KB insertions
        try:
            latest_items = self.kb.query("") or []
            # Only try to append at most a small window to avoid recomputing all
            if len(latest_items) > 0:
                self.refresh_cache_incremental(latest_items[-500:])
        except Exception:
            pass

        out = []
        for h in hypotheses:
            try:
                out.append(self.evaluate_one(h))
            except Exception as e:
                bad = dict(h)
                bad["evidence_error"] = str(e)
                out.append(bad)
        return out
