# reasoning_module/discover.py
# ------------------------------------------------------------
# Discovery Orchestrator (Product Surface)
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

from reasoning_module.proposal_evaluator import ProposalEvaluator
from reasoning_module.claim_extractor import extract_claims
from reasoning_module.hypothesis_generator import HypothesisGenerator
from reasoning_module.hypothesis_validator import HypothesisValidator
from reasoning_module.evidence_evaluator import EvidenceEvaluator
from reasoning_module.claim_schema import SourceTrace


@dataclass
class DiscoveryConfig:
    top_k_chunks: int = 12
    evidence_threshold: float = 0.60
    max_hypotheses: int = 10
    max_claims: int = 12
    max_claims_per_chunk: int = 3
    use_mmr: bool = True   # ✅ default ON (your chunk_index supports it)


class DiscoveryEngine:
    def __init__(
        self,
        *,
        chunk_index,
        proposal_engine: ProposalEvaluator,
        evidence_evaluator: EvidenceEvaluator,
        hypgen: HypothesisGenerator,
        validator: HypothesisValidator,
        config: DiscoveryConfig = DiscoveryConfig(),
    ):
        self.chunk_index = chunk_index
        self.proposal_engine = proposal_engine
        self.evidence_evaluator = evidence_evaluator
        self.hypgen = hypgen
        self.validator = validator
        self.cfg = config

        # ✅ wire extractor
        self.claim_extractor = extract_claims

    # ---------------------------
    # Core
    # ---------------------------
    def run(self, query: str, *, source_name: str = "cli") -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {
                "query": query,
                "evidence_chunks": [],
                "extracted_claims": [],
                "proposal_verdict": {"verdict": "reject", "confidence": 0.0, "explanation": "Empty query."},
                "hypotheses": [],
                "next_actions": ["Provide a non-empty query."],
            }

        # 1) Retrieve evidence chunks
        chunks = self._retrieve_chunks(q)

        # 2) Extract claims from evidence chunks ✅ FIX: no trailing comma
        claims = self._extract_claims_from_chunks(chunks, source_name=source_name)

        # 3) Judge the query as a proposal (grounded to provided chunks)
        judged = self.proposal_engine.evaluate(
            q,
            provenance=SourceTrace(source_type="user_query", source_name=source_name),
            evidence_chunks=chunks,
        )
        verdict_obj = judged.get("verdict", judged) if isinstance(judged, dict) else judged

        # 4) Generate hypotheses from KG and validate
        hyps = self.hypgen.generate(top_n=max(self.cfg.max_hypotheses, 20)) or []
        validated = self.validator.validate(hyps[: self.cfg.max_hypotheses], cycle=0) if hyps else []

        # 5) Evidence evaluate hypotheses
        evaluated = self.evidence_evaluator.evaluate_batch(validated) if validated else []

        # 6) Rank hypotheses
        evaluated_sorted = sorted(evaluated, key=self._score_hypothesis, reverse=True)

        return {
            "query": q,
            "evidence_chunks": chunks,
            "extracted_claims": claims[: self.cfg.max_claims],
            "proposal_verdict": verdict_obj,
            "hypotheses": evaluated_sorted[: self.cfg.max_hypotheses],
            "next_actions": self._next_actions(chunks, claims, verdict_obj),
        }

    # ---------------------------
    # Retrieval
    # ---------------------------
    def _retrieve_chunks(self, query: str) -> List[Dict[str, Any]]:
        try:
            return self.chunk_index.retrieve(
                query,
                top_k=self.cfg.top_k_chunks,
                use_mmr=bool(self.cfg.use_mmr),
            )
        except TypeError:
            # If signature doesn't support use_mmr
            try:
                return self.chunk_index.retrieve(query, top_k=self.cfg.top_k_chunks)
            except Exception as e:
                return [{
                    "chunk_id": None,
                    "paper_title": "",
                    "source": "system",
                    "text_preview": f"Chunk retrieval failed: {e}",
                    "similarity": 0.0,
                }]
        except Exception as e:
            return [{
                "chunk_id": None,
                "paper_title": "",
                "source": "system",
                "text_preview": f"Chunk retrieval failed: {e}",
                "similarity": 0.0,
            }]

    # ---------------------------
    # Claim extraction
    # ---------------------------
    def _extract_claims_from_chunks(self, chunks: List[Dict[str, Any]], *, source_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not chunks:
            return out

        for c in chunks[: min(len(chunks), self.cfg.top_k_chunks)]:
            # Prefer full text if present; else preview
            text = (c.get("text") or c.get("chunk_text") or c.get("text_preview") or "").strip()
            if not text:
                continue

            prov = SourceTrace(
                source_type="chunk",
                source_name=source_name,
                extra={
                    "chunk_id": c.get("chunk_id"),
                    "paper_title": c.get("paper_title", ""),
                    "source": c.get("source", ""),
                    "similarity": c.get("similarity", None),
                },
            )

            extracted = self.claim_extractor(text, provenance=prov) or []
            for cl in extracted[: self.cfg.max_claims_per_chunk]:
                claim_text = getattr(cl, "claim_text", str(cl))
                claim_type = getattr(cl, "claim_type", "unknown")
                domain = getattr(cl, "domain", "unknown")

                out.append({
                    "claim": claim_text,
                    "claim_type": claim_type,
                    "domain": domain,
                    "chunk_id": c.get("chunk_id"),
                    "source": c.get("source", ""),
                    "paper_title": c.get("paper_title", ""),
                })

        # Dedup by claim text
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for x in out:
            k = (x.get("claim") or "").strip()
            if not k:
                continue
            if k in seen:
                continue
            seen.add(k)
            deduped.append(x)

        return deduped

    # ---------------------------
    # Scoring + Actions
    # ---------------------------
    def _score_hypothesis(self, h: Dict[str, Any]) -> float:
        conf = float(h.get("confidence", h.get("evidence_confidence", 0.0)) or 0.0)
        ev = float(h.get("evidence_score", 0.0) or 0.0)
        return 0.6 * ev + 0.4 * conf

    def _next_actions(self, chunks: List[Dict[str, Any]], claims: List[Dict[str, Any]], verdict_obj: Dict[str, Any]) -> List[str]:
        actions = [
            "Pick 1 concrete intervention and convert it into a falsifiable claim (inputs → mechanism → measurable outputs).",
            "Run targeted retrieval using 3–5 specific keywords and rebuild the ChunkIndex.",
            "Collect evidence that contains explicit method + result numbers (before/after, % removal, ppm, cost).",
            "Generate an experiment plan: variables, controls, measurement method, acceptance threshold.",
        ]

        if not chunks:
            actions.insert(0, "No evidence chunks were retrieved — ingest domain papers first and rebuild ChunkIndex.")

        if chunks and not claims:
            actions.insert(0, "Evidence retrieved but no claims extracted — confirm ChunkIndex returns full chunk text (not only previews).")

        v = str((verdict_obj or {}).get("verdict", "")).lower()
        if v in {"weak", "unsupported", "reject"}:
            actions.insert(0, "Current proposal is not well-supported — tighten query, retrieve higher-quality evidence, and retry.")

        return actions
