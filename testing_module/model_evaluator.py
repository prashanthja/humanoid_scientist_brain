#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ModelEvaluator — Retrieval-grounded benchmark for Humanoid Scientist Brain.

Fixes:
- Keeps KB embedding rows aligned with kb_items (no silent dropping)
- Dedupe KB items + dedupe retrieved items
- Adds random-baseline sanity check + simple collapse diagnostics
- Adds quality gates to prevent title-only poison in benchmark retrieval
"""

from __future__ import annotations
import os, json, time, hashlib, random
from typing import List, Dict, Any, Tuple
import numpy as np


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_text(x: Any) -> str:
    if isinstance(x, dict):
        return str(x.get("text") or x.get("paper_title") or "")
    return str(x or "")


def _norm_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32)
    mat = mat.astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


def _stable_key(it: Dict[str, Any]) -> str:
    pid = it.get("paper_id") or it.get("id")
    if pid not in (None, "", "unknown"):
        return f"id:{pid}"

    title = (it.get("paper_title") or "").strip().lower()
    if title and title not in ("unknown",):
        return f"title:{title}"

    txt = (it.get("text") or "").strip()
    h = hashlib.sha1(txt[:500].encode("utf-8", errors="ignore")).hexdigest()
    return f"hash:{h}"


def _is_low_signal_row(text: str, title: str) -> bool:
    t = (text or "").strip()
    if len(t) < 180:
        return True
    if title and title.strip().lower() == t.lower():
        return True
    return False


class ModelEvaluator:
    def __init__(
        self,
        trainer,
        kb,
        bridge,
        top_k: int = 8,
        evidence_sim_threshold: float = 0.55,
        max_kb_items: int = 5000,
        report_path: str = "logs/model_benchmark.json",
        kb_embed_batch: int = 32,
        random_baseline_k: int = 64,
        seed: int = 1337,
        force_cpu_embeddings: bool = False,
        # quality gates
        drop_low_signal_kb_rows: bool = True,
    ):
        self.trainer = trainer
        self.kb = kb
        self.bridge = bridge
        self.top_k = int(top_k)
        self.th = float(evidence_sim_threshold)
        self.max_kb_items = int(max_kb_items)
        self.report_path = report_path
        self.kb_embed_batch = int(kb_embed_batch)
        self.random_baseline_k = int(random_baseline_k)
        self.seed = int(seed)
        self.force_cpu_embeddings = bool(force_cpu_embeddings)
        self.drop_low_signal_kb_rows = bool(drop_low_signal_kb_rows)

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.benchmarks = self._load_benchmarks()

    def _load_benchmarks(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "newton2",
                "question": "What is Newton's second law?",
                "expected": "Force equals mass times acceleration (F = m a).",
                "topic": "classical mechanics",
                "keywords": ["force", "mass", "acceleration", "f", "m", "a"],
            },
            {
                "id": "schrodinger",
                "question": "What does the Schrödinger equation describe?",
                "expected": "It describes how a quantum state (wavefunction) evolves over time.",
                "topic": "quantum mechanics",
                "keywords": ["wavefunction", "quantum", "evolves", "time", "state"],
            },
            {
                "id": "entropy",
                "question": "What is entropy in thermodynamics?",
                "expected": "A measure related to the number of microstates (often interpreted as disorder).",
                "topic": "thermodynamics",
                "keywords": ["entropy", "microstates", "disorder"],
            },
            {
                "id": "tensor",
                "question": "Define tensor.",
                "expected": "A mathematical object that generalizes scalars, vectors, and matrices.",
                "topic": "mathematics",
                "keywords": ["tensor", "scalar", "vector", "matrix"],
            },
            {
                "id": "maxwell",
                "question": "What are Maxwell’s equations about?",
                "expected": "They describe how electric and magnetic fields are generated and how they propagate.",
                "topic": "electromagnetism",
                "keywords": ["electric", "magnetic", "fields", "propagate", "generated"],
            },
        ]

    # ----------------------------
    # Encoding (alignment-safe)
    # ----------------------------
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        IMPORTANT: do NOT drop empty items silently; it breaks alignment.
        For empties, we embed a placeholder and then zero it out.
        """
        if not texts:
            return np.zeros((0, 256), dtype=np.float32)

        clean = []
        empty_mask = []
        for t in texts:
            s = t if isinstance(t, str) else str(t or "")
            s = s.strip()
            if not s:
                clean.append("[EMPTY]")
                empty_mask.append(True)
            else:
                clean.append(s)
                empty_mask.append(False)

        # Bridge preferred
        if hasattr(self.bridge, "encode_texts"):
            arr = np.asarray(self.bridge.encode_texts(clean), dtype=np.float32)
        elif hasattr(self.bridge, "embed"):
            arr = np.asarray(self.bridge.embed(clean), dtype=np.float32)
        elif self.trainer is not None and hasattr(self.trainer, "embed"):
            try:
                arr = np.asarray(
                    self.trainer.embed(clean, force_cpu=self.force_cpu_embeddings),
                    dtype=np.float32,
                )
            except TypeError:
                arr = np.asarray(self.trainer.embed(clean), dtype=np.float32)
        else:
            raise RuntimeError("No embedding interface found on bridge or trainer.")

        # Defensive: mean-pool if sequence
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Zero out embeddings for empties
        for i, is_empty in enumerate(empty_mask):
            if is_empty and i < arr.shape[0]:
                arr[i, :] = 0.0

        return arr.astype(np.float32, copy=False)

    # ----------------------------
    # KB loading + dedupe
    # ----------------------------
    def _load_kb_items(self) -> Tuple[List[Dict[str, Any]], float]:
        raw = self.kb.query("") or []
        seen = set()
        out = []
        dup = 0

        for it in raw[: self.max_kb_items]:
            if not isinstance(it, dict):
                it = {"text": str(it)}

            text = _safe_text(it).strip()
            title = (it.get("paper_title") or "") if isinstance(it, dict) else ""
            if not text:
                continue

            if self.drop_low_signal_kb_rows and _is_low_signal_row(text, title):
                continue

            row = {
                "id": it.get("id"),
                "paper_id": it.get("paper_id"),
                "paper_title": (title or ""),
                "source": (it.get("source") or "") if isinstance(it, dict) else "",
                "text": text,
            }
            k = _stable_key(row)
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            row["_dedupe_key"] = k
            out.append(row)

        denom = max(len(raw[: self.max_kb_items]), 1)
        dup_rate = dup / denom
        return out, float(dup_rate)

    # ----------------------------
    # Retrieval
    # ----------------------------
    def _retrieve_topk(
        self,
        query: str,
        kb_items: List[Dict[str, Any]],
        kb_emb_norm: np.ndarray,
    ) -> List[Dict[str, Any]]:
        qv = self._encode_texts([query])
        if qv.size == 0:
            return []
        qv = _norm_rows(qv)[0]

        sims = (kb_emb_norm @ qv).astype(np.float32)
        top_idx = np.argsort(-sims)[: self.top_k]

        retrieved = []
        used = set()
        for i in top_idx:
            row = kb_items[int(i)]
            k = row.get("_dedupe_key") or _stable_key(row)
            if k in used:
                continue
            used.add(k)
            retrieved.append({**row, "similarity_to_question": float(sims[int(i)])})

        return retrieved

    # ----------------------------
    # Scoring helpers
    # ----------------------------
    def _keyword_hit(self, evidence_text: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0
        t = (evidence_text or "").lower()
        hits = sum(1 for k in keywords if k.lower() in t)
        return float(hits) / float(len(keywords))

    def _best_sim_expected_vs_evidence(self, expected: str, evidence_texts: List[str]) -> float:
        if not evidence_texts:
            return 0.0
        ev = self._encode_texts(evidence_texts)
        if ev.size == 0:
            return 0.0
        ev = _norm_rows(ev)

        ex = self._encode_texts([expected])
        if ex.size == 0:
            return 0.0
        ex = _norm_rows(ex)[0]

        sims = ev @ ex
        return float(np.max(sims)) if sims.size else 0.0

    def _random_baseline_best_sim(self, expected: str, kb_items: List[Dict[str, Any]]) -> float:
        if not kb_items:
            return 0.0
        k = min(self.random_baseline_k, len(kb_items))
        sample = random.sample(kb_items, k=k)
        texts = [x["text"] for x in sample]
        return self._best_sim_expected_vs_evidence(expected, texts)

    def _collapse_diagnostics(self, kb_emb_norm: np.ndarray) -> Dict[str, Any]:
        """
        Quick sanity metrics to detect embedding collapse:
        - mean row norm (should be ~1 after norm, but if many zeros => low)
        - variance of embedding dims
        - mean pairwise cosine on a small sample (collapsed => very high)
        """
        if kb_emb_norm.size == 0:
            return {"ok": False, "reason": "empty_kb_emb"}

        norms = np.linalg.norm(kb_emb_norm, axis=1)
        mean_norm = float(np.mean(norms))

        var = float(np.mean(np.var(kb_emb_norm, axis=0)))

        n = kb_emb_norm.shape[0]
        m = min(200, n)
        idx = np.random.choice(n, size=m, replace=False) if n >= m else np.arange(n)
        sample = kb_emb_norm[idx]
        # approximate pairwise mean cosine by dotting vs mean vector
        mu = sample.mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-8)
        mean_cos_to_mu = float(np.mean(sample @ mu))

        return {
            "ok": True,
            "mean_norm": round(mean_norm, 4),
            "mean_dim_variance": round(var, 6),
            "mean_cosine_to_meanvec": round(mean_cos_to_mu, 4),
        }

    # ----------------------------
    # Main eval
    # ----------------------------
    def evaluate_on_benchmark(self) -> Dict[str, Any]:
        kb_items, dup_rate = self._load_kb_items()

        kb_texts = [x["text"] for x in kb_items]
        kb_emb_chunks = []
        for i in range(0, len(kb_texts), self.kb_embed_batch):
            kb_emb_chunks.append(self._encode_texts(kb_texts[i:i + self.kb_embed_batch]))
        kb_emb = np.concatenate(kb_emb_chunks, axis=0) if kb_emb_chunks else np.zeros((0, 256), dtype=np.float32)
        kb_emb_norm = _norm_rows(kb_emb) if kb_emb.size else kb_emb

        diag = self._collapse_diagnostics(kb_emb_norm)

        results = []
        recall_hits = 0
        best_sims = []
        kw_hits = []
        discr_gaps = []

        for b in self.benchmarks:
            q = b["question"]
            exp = b["expected"]

            retrieved = self._retrieve_topk(q, kb_items, kb_emb_norm)

            evidence_texts = [r["text"] for r in retrieved]
            best_sim = self._best_sim_expected_vs_evidence(exp, evidence_texts)

            hit = best_sim >= self.th
            recall_hits += int(hit)
            best_sims.append(best_sim)

            rand_best = self._random_baseline_best_sim(exp, kb_items)
            gap = best_sim - rand_best
            discr_gaps.append(gap)

            kh = 0.0
            if retrieved:
                ev = self._encode_texts(evidence_texts)
                ex = self._encode_texts([exp])
                if ev.size and ex.size:
                    ev = _norm_rows(ev)
                    ex = _norm_rows(ex)[0]
                    sims = ev @ ex
                    j = int(np.argmax(sims))
                    kh = self._keyword_hit(retrieved[j]["text"], b.get("keywords", []))
            kw_hits.append(kh)

            results.append({
                "id": b.get("id"),
                "topic": b.get("topic"),
                "question": q,
                "expected": exp,
                "top_k": retrieved,
                "best_evidence_similarity_to_expected": round(float(best_sim), 4),
                "random_baseline_best_sim": round(float(rand_best), 4),
                "discrimination_gap": round(float(gap), 4),
                "recall_hit": bool(hit),
                "keyword_hit_rate": round(float(kh), 3),
                "unique_in_top_k": len({r.get("_dedupe_key") for r in retrieved}),
            })

        n = max(len(self.benchmarks), 1)
        recall_at_k = recall_hits / n
        avg_best_sim = float(np.mean(best_sims)) if best_sims else 0.0
        avg_kw = float(np.mean(kw_hits)) if kw_hits else 0.0
        avg_gap = float(np.mean(discr_gaps)) if discr_gaps else 0.0

        if recall_at_k >= 0.8 and avg_best_sim >= 0.70 and avg_gap >= 0.05:
            verdict = "Strong retrieval-grounded understanding"
        elif recall_at_k >= 0.4 and avg_gap >= 0.02:
            verdict = "Moderate — retrieval somewhat discriminative, improve KB + embeddings"
        else:
            verdict = "Weak — retrieval not discriminative (likely duplicates or embedding collapse)"

        report = {
            "timestamp": _now(),
            "top_k": self.top_k,
            "evidence_sim_threshold": self.th,
            "kb_items_used": len(kb_items),
            "kb_duplicate_drop_rate": round(float(dup_rate), 4),
            "collapse_diagnostics": diag,
            "recall_at_k": round(float(recall_at_k), 4),
            "avg_best_evidence_similarity": round(float(avg_best_sim), 4),
            "avg_keyword_hit_rate": round(float(avg_kw), 4),
            "avg_discrimination_gap": round(float(avg_gap), 4),
            "verdict": verdict,
            "benchmarks": results,
        }

        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report
