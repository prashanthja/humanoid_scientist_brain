# reasoning_module/evidence_grounding_engine.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GroundingResult:
    claim: str
    support_count: int
    contradict_count: int
    neutral_count: int
    total_evidence: int
    unique_sources: int
    unique_titles: int
    avg_support_score: float
    avg_contradict_score: float
    diversity_score: float
    contradiction_penalty: float
    coverage_score: float
    grounding_score: float
    top_support: List[Dict[str, Any]]
    top_contradict: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceGroundingEngine:
    """
    Aggregates evaluated evidence into a grounded score.

    Expected evidence item shape:
    {
        "text": "...",
        "paper_title": "...",   # or "title"
        "source": "...",
        "verdict": "support" | "contradict" | "neutral",   # or "polarity"
        "score": 0.72,          # optional
        "similarity": 0.72,     # optional fallback
        "weight": 0.72,         # optional fallback
    }
    """

    def __init__(
        self,
        min_sources: int = 2,
        min_titles: int = 2,
        support_weight: float = 1.0,
        contradict_weight: float = 1.15,
        diversity_bonus_weight: float = 0.20,
        coverage_weight: float = 0.25,
        max_items_per_side: int = 5,
    ):
        self.min_sources = int(min_sources)
        self.min_titles = int(min_titles)
        self.support_weight = float(support_weight)
        self.contradict_weight = float(contradict_weight)
        self.diversity_bonus_weight = float(diversity_bonus_weight)
        self.coverage_weight = float(coverage_weight)
        self.max_items_per_side = int(max_items_per_side)

    def _normalize_verdict(self, item: Dict[str, Any]) -> str:
        raw = str(item.get("verdict") or item.get("polarity") or "neutral").strip().lower()
        if raw in {"support", "supported", "pro"}:
            return "support"
        if raw in {"contradict", "contradicted", "contra", "oppose"}:
            return "contradict"
        return "neutral"

    def _score(self, item: Dict[str, Any]) -> float:
        for key in ("score", "weight", "similarity", "similarity_to_question"):
            if key in item:
                try:
                    return max(0.0, min(1.0, float(item[key])))
                except Exception:
                    pass
        return 0.5

    def _title(self, item: Dict[str, Any]) -> str:
        return str(item.get("paper_title") or item.get("title") or "unknown").strip()

    def _source(self, item: Dict[str, Any]) -> str:
        return str(item.get("source") or "unknown").strip()

    def _diversity_score(self, sources: int, titles: int) -> float:
        source_term = min(1.0, sources / max(1, self.min_sources))
        title_term = min(1.0, titles / max(1, self.min_titles))
        return 0.5 * (source_term + title_term)

    def _coverage_score(self, total_evidence: int) -> float:
        return min(1.0, total_evidence / 6.0)

    def evaluate(self, claim: str, evidence_items: Optional[List[Dict[str, Any]]]) -> GroundingResult:
        items = evidence_items or []

        support: List[Tuple[float, Dict[str, Any]]] = []
        contradict: List[Tuple[float, Dict[str, Any]]] = []
        neutral: List[Tuple[float, Dict[str, Any]]] = []

        sources = set()
        titles = set()

        for item in items:
            verdict = self._normalize_verdict(item)
            score = self._score(item)
            title = self._title(item)
            source = self._source(item)

            if source and source != "unknown":
                sources.add(source)
            if title and title != "unknown":
                titles.add(title)

            bucket_item = dict(item)
            bucket_item["verdict"] = verdict
            bucket_item["score"] = score

            if verdict == "support":
                support.append((score, bucket_item))
            elif verdict == "contradict":
                contradict.append((score, bucket_item))
            else:
                neutral.append((score, bucket_item))

        support.sort(key=lambda x: x[0], reverse=True)
        contradict.sort(key=lambda x: x[0], reverse=True)
        neutral.sort(key=lambda x: x[0], reverse=True)

        support_scores = [x[0] for x in support]
        contradict_scores = [x[0] for x in contradict]

        avg_support = sum(support_scores) / len(support_scores) if support_scores else 0.0
        avg_contradict = sum(contradict_scores) / len(contradict_scores) if contradict_scores else 0.0

        diversity_score = self._diversity_score(len(sources), len(titles))
        coverage_score = self._coverage_score(len(items))

        contradiction_penalty = avg_contradict * (len(contradict) / max(1, len(items)))

        raw = (
            self.support_weight * avg_support * (len(support) / max(1, len(items)))
            - self.contradict_weight * contradiction_penalty
            + self.diversity_bonus_weight * diversity_score
            + self.coverage_weight * coverage_score
        )

        grounding_score = max(0.0, min(1.0, raw))

        return GroundingResult(
            claim=str(claim or "").strip(),
            support_count=len(support),
            contradict_count=len(contradict),
            neutral_count=len(neutral),
            total_evidence=len(items),
            unique_sources=len(sources),
            unique_titles=len(titles),
            avg_support_score=round(avg_support, 4),
            avg_contradict_score=round(avg_contradict, 4),
            diversity_score=round(diversity_score, 4),
            contradiction_penalty=round(contradiction_penalty, 4),
            coverage_score=round(coverage_score, 4),
            grounding_score=round(grounding_score, 4),
            top_support=[x[1] for x in support[: self.max_items_per_side]],
            top_contradict=[x[1] for x in contradict[: self.max_items_per_side]],
        )