# reasoning_module/discover.py
# ------------------------------------------------------------
# Discovery Orchestrator (Product Surface)
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from reasoning_module.proposal_evaluator import ProposalEvaluator
from reasoning_module.claim_extractor import extract_claims
from reasoning_module.hypothesis_generator import HypothesisGenerator
from reasoning_module.hypothesis_validator import HypothesisValidator
from reasoning_module.evidence_evaluator import EvidenceEvaluator
from reasoning_module.claim_schema import SourceTrace
from reasoning_module.evidence_grounding_engine import EvidenceGroundingEngine
from reasoning_module.verdict_engine import VerdictEngine


@dataclass
class DiscoveryConfig:
    top_k_chunks: int = 12
    evidence_threshold: float = 0.60
    max_hypotheses: int = 10
    max_claims: int = 12
    max_claims_per_chunk: int = 3
    max_grounded_claims: int = 8
    use_mmr: bool = True


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

        self.claim_extractor = extract_claims
        self.grounder = EvidenceGroundingEngine()
        self.verdict_engine = VerdictEngine()

        # ------------------------------------------------------------
        # Hard wiring guard: force chunk-grounded evaluation
        # ------------------------------------------------------------
        if getattr(self.evidence_evaluator, "chunk_index", None) is None and self.chunk_index is not None:
            try:
                self.evidence_evaluator.chunk_index = self.chunk_index
            except Exception:
                pass

        try:
            if self.chunk_index is not None:
                self.evidence_evaluator.use_chunk_index = True
        except Exception:
            pass

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
                "grounded_claims": [],
                "proposal_verdict": {"verdict": "reject", "confidence": 0.0, "explanation": "Empty query."},
                "hypotheses": [],
                "next_actions": ["Provide a non-empty query."],
            }

        # 1) Retrieve evidence chunks
        chunks = self._retrieve_chunks(q)

        # 2) Extract claims from evidence chunks
        claims = self._extract_claims_from_chunks(chunks, source_name=source_name)

        # 3) Ground extracted claims against the same retrieved evidence
        grounded_claims = self._ground_claims(claims, chunks)

        # 4) Judge the original query as a proposal
        judged = self.proposal_engine.evaluate(
            q,
            provenance=SourceTrace(source_type="user_query", source_name=source_name),
            evidence_chunks=chunks,
        )
        verdict_obj = judged.get("verdict", judged) if isinstance(judged, dict) else judged

        # 5) Generate hypotheses from KG and validate
        hyps = self.hypgen.generate(top_n=max(self.cfg.max_hypotheses, 20)) or []
        validated = self.validator.validate(hyps[: self.cfg.max_hypotheses], cycle=0) if hyps else []

        # 6) Evidence-evaluate hypotheses
        evaluated = self.evidence_evaluator.evaluate_batch(validated) if validated else []

        # 7) Rank hypotheses
        evaluated_sorted = sorted(evaluated, key=self._score_hypothesis, reverse=True)

        return {
            "query": q,
            "evidence_chunks": chunks,
            "extracted_claims": claims[: self.cfg.max_claims],
            "grounded_claims": grounded_claims[: self.cfg.max_grounded_claims],
            "proposal_verdict": verdict_obj,
            "hypotheses": evaluated_sorted[: self.cfg.max_hypotheses],
            "next_actions": self._next_actions(chunks, claims, grounded_claims, verdict_obj),
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
            try:
                return self.chunk_index.retrieve(query, top_k=self.cfg.top_k_chunks)
            except Exception as e:
                return [{
                    "chunk_id": None,
                    "paper_title": "",
                    "source": "system",
                    "text": f"Chunk retrieval failed: {e}",
                    "text_preview": f"Chunk retrieval failed: {e}",
                    "similarity": 0.0,
                    "sim_embedding": 0.0,
                }]
        except Exception as e:
            return [{
                "chunk_id": None,
                "paper_title": "",
                "source": "system",
                "text": f"Chunk retrieval failed: {e}",
                "text_preview": f"Chunk retrieval failed: {e}",
                "similarity": 0.0,
                "sim_embedding": 0.0,
            }]

    # ---------------------------
    # Claim extraction
    # ---------------------------
    def _extract_claims_from_chunks(self, chunks: List[Dict[str, Any]], *, source_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not chunks:
            return out

        for c in chunks[: min(len(chunks), self.cfg.top_k_chunks)]:
            text = (c.get("text") or c.get("chunk_text") or c.get("text_preview") or "").strip()
            if not text:
                continue

            sim = c.get("sim_embedding") if c.get("sim_embedding") is not None else c.get("similarity")

            prov = SourceTrace(
                source_type="chunk",
                source_name=source_name,
                extra={
                    "chunk_id": c.get("chunk_id"),
                    "paper_title": c.get("paper_title", ""),
                    "source": c.get("source", ""),
                    "similarity": sim,
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
                    "similarity": sim,
                })

        # dedupe by claim text
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for x in out:
            k = (x.get("claim") or "").strip()
            if not k or k in seen:
                continue
            seen.add(k)
            deduped.append(x)

        return deduped

    # ---------------------------
    # Ground claims
    # ---------------------------
    def _ground_claims(self, claims: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grounded: List[Dict[str, Any]] = []
        if not claims or not chunks:
            return grounded

        for cl in claims[: self.cfg.max_grounded_claims]:
            claim_text = (cl.get("claim") or "").strip()
            if not claim_text:
                continue

            evidence_items = self._build_claim_evidence_items(claim_text, chunks)
            grounding = self.grounder.evaluate(claim_text, evidence_items).to_dict()
            verdict = self.verdict_engine.compute(grounding).to_dict()

            grounded.append({
                "claim": claim_text,
                "claim_type": cl.get("claim_type", "unknown"),
                "domain": cl.get("domain", "unknown"),
                "grounding": grounding,
                "verdict": verdict,
            })

        grounded.sort(
            key=lambda x: (
                float(x.get("verdict", {}).get("confidence", 0.0)),
                float(x.get("grounding", {}).get("grounding_score", 0.0)),
            ),
            reverse=True,
        )
        return grounded

    def _build_claim_evidence_items(self, claim_text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Lightweight MVP grounding:
        - use similarity + keyword overlap to classify chunk as support / neutral / contradict
        - later this can be replaced with stronger NLI / evaluator logic
        """
        claim_low = claim_text.lower()
        claim_tokens = set(t for t in claim_low.replace("-", " ").split() if len(t) > 2)

        contradict_markers = {
            "however", "but", "fails", "worse", "degrades", "hurts",
            "tradeoff", "overhead", "instability", "unstable", "limited",
            "does not", "no improvement", "regression"
        }
        support_markers = {
            "improves", "reduce", "reduces", "better", "faster", "efficient",
            "efficiency", "gain", "improved", "lower", "outperforms", "scales"
        }

        items: List[Dict[str, Any]] = []

        for c in chunks:
            text = (c.get("text") or c.get("chunk_text") or c.get("text_preview") or "").strip()
            if not text:
                continue

            low = text.lower()
            text_tokens = set(t for t in low.replace("-", " ").split() if len(t) > 2)
            overlap = len(claim_tokens & text_tokens) / max(1, len(claim_tokens))

            sim = c.get("similarity")
            if sim is None:
                sim = c.get("sim_embedding", 0.0)
            try:
                sim = float(sim or 0.0)
            except Exception:
                sim = 0.0

            support_hit = any(m in low for m in support_markers)
            contradict_hit = any(m in low for m in contradict_markers)

            if overlap >= 0.20 and support_hit and not contradict_hit:
                verdict = "support"
            elif overlap >= 0.20 and contradict_hit and not support_hit:
                verdict = "contradict"
            elif overlap >= 0.30 and support_hit and contradict_hit:
                verdict = "neutral"
            elif overlap >= 0.25:
                verdict = "neutral"
            else:
                continue

            score = max(0.0, min(1.0, (0.65 * sim) + (0.35 * overlap)))

            items.append({
                "text": text,
                "paper_title": c.get("paper_title", ""),
                "source": c.get("source", ""),
                "chunk_id": c.get("chunk_id"),
                "verdict": verdict,
                "score": score,
                "similarity": sim,
                "overlap": round(overlap, 4),
            })

        return items

    # ---------------------------
    # Scoring + Actions
    # ---------------------------
    def _score_hypothesis(self, h: Dict[str, Any]) -> float:
        conf = float(h.get("evidence_confidence", 0.0) or 0.0)
        strong = float(h.get("strong_chunks", 0) or 0)

        te = h.get("top_evidence") or []
        wsum = 0.0
        if isinstance(te, list) and te:
            for e in te[:10]:
                try:
                    wsum += float(e.get("weight", 0.0) or 0.0)
                except Exception:
                    pass

        return (0.60 * conf) + (0.25 * min(1.0, wsum)) + (0.15 * min(1.0, strong / 3.0))

    def _next_actions(
        self,
        chunks: List[Dict[str, Any]],
        claims: List[Dict[str, Any]],
        grounded_claims: List[Dict[str, Any]],
        verdict_obj: Dict[str, Any],
    ) -> List[str]:
        actions = [
            "Pick 1 concrete intervention and convert it into a falsifiable claim (inputs → mechanism → measurable outputs).",
            "Run targeted retrieval using 3–5 specific transformer-efficiency keywords and compare results.",
            "Prioritize evidence chunks with benchmark numbers (latency, throughput, memory, perplexity, context length).",
            "Generate an experiment plan: baseline, efficiency metric, quality metric, ablations, acceptance threshold.",
        ]

        if not chunks:
            actions.insert(0, "No evidence chunks were retrieved — ingest transformer-efficiency papers first and rebuild ChunkIndex.")

        if chunks and not claims:
            actions.insert(0, "Evidence retrieved but no claims extracted — confirm ChunkIndex returns full chunk text, not only previews.")

        if claims and not grounded_claims:
            actions.insert(0, "Claims extracted but not grounded — inspect overlap and evidence classification logic.")

        v = str((verdict_obj or {}).get("verdict", "")).lower()
        if v in {"weak", "unsupported", "reject"}:
            actions.insert(0, "Current proposal is not well-supported — tighten query, retrieve stronger benchmark evidence, and retry.")

        return actions