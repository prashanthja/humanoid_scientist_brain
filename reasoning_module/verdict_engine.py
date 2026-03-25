# reasoning_module/verdict_engine.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class VerdictResult:
    claim: str
    verdict: str
    confidence: float
    explanation: str
    grounding_score: float
    support_count: int
    contradict_count: int
    neutral_count: int
    total_evidence: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VerdictEngine:
    """
    Converts grounded evidence into a final verdict.

    Verdict labels:
    - supported
    - contradicted
    - partially_supported
    - inconclusive
    """

    def __init__(
        self,
        support_margin: int = 2,
        contradiction_margin: int = 2,
        min_grounding_supported: float = 0.55,
        min_grounding_partial: float = 0.40,
        max_confidence: float = 0.95,
    ):
        self.support_margin = int(support_margin)
        self.contradiction_margin = int(contradiction_margin)
        self.min_grounding_supported = float(min_grounding_supported)
        self.min_grounding_partial = float(min_grounding_partial)
        self.max_confidence = float(max_confidence)

    def compute(self, grounding: Dict[str, Any]) -> VerdictResult:
        claim = str(grounding.get("claim") or "").strip()
        support = int(grounding.get("support_count", 0))
        contradict = int(grounding.get("contradict_count", 0))
        neutral = int(grounding.get("neutral_count", 0))
        total = int(grounding.get("total_evidence", 0))
        grounding_score = float(grounding.get("grounding_score", 0.0))
        avg_support = float(grounding.get("avg_support_score", 0.0))
        avg_contradict = float(grounding.get("avg_contradict_score", 0.0))

        delta = support - contradict

        if total == 0:
            verdict = "inconclusive"
            confidence = 0.0
            explanation = "No evidence retrieved."
        elif (
            delta >= self.support_margin
            and grounding_score >= self.min_grounding_supported
            and avg_support >= avg_contradict
        ):
            verdict = "supported"
            confidence = min(self.max_confidence, grounding_score + 0.10)
            explanation = (
                f"Support outweighs contradiction (support={support}, contradict={contradict}) "
                f"with strong grounding={grounding_score:.2f}."
            )
        elif (
            -delta >= self.contradiction_margin
            and avg_contradict > avg_support
        ):
            verdict = "contradicted"
            confidence = min(self.max_confidence, max(0.35, 0.45 + abs(delta) * 0.05))
            explanation = (
                f"Contradiction outweighs support (support={support}, contradict={contradict})."
            )
        elif support > contradict and grounding_score >= self.min_grounding_partial:
            verdict = "partially_supported"
            confidence = min(self.max_confidence, grounding_score)
            explanation = (
                f"Evidence leans supportive but remains mixed "
                f"(support={support}, contradict={contradict}, neutral={neutral})."
            )
        else:
            verdict = "inconclusive"
            confidence = min(0.75, max(0.20, grounding_score))
            explanation = (
                f"Evidence is mixed or insufficient "
                f"(support={support}, contradict={contradict}, neutral={neutral})."
            )

        return VerdictResult(
            claim=claim,
            verdict=verdict,
            confidence=round(confidence, 4),
            explanation=explanation,
            grounding_score=round(grounding_score, 4),
            support_count=support,
            contradict_count=contradict,
            neutral_count=neutral,
            total_evidence=total,
        )