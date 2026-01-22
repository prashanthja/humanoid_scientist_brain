# reasoning_module/claim_schema.py
# ------------------------------------------------------------
# Claim Schema (v1)
# - Structured objects for: claims, equations, assumptions, evidence, verdict
# - Stable JSON export for dashboard / API / logs
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import time
import hashlib


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stable_id(text: str, prefix: str = "clm") -> str:
    h = hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{h}"


@dataclass
class SourceTrace:
    source_type: str = "unknown"   # paper, book, webpage, scripture, note, etc.
    source_name: str = "unknown"   # domain/site/book title
    url: str = ""
    retrieved_at: str = field(default_factory=now_ts)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Equation:
    raw: str
    lhs: str = ""
    rhs: str = ""
    symbols: List[str] = field(default_factory=list)


@dataclass
class Claim:
    claim_id: str
    claim_text: str
    domain: str = "unknown"  # physics, math, mixed
    claim_type: str = "statement"  # equation, definition, causal, prediction, invention
    equations: List[Equation] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    predicted_observables: List[str] = field(default_factory=list)
    provenance: SourceTrace = field(default_factory=SourceTrace)
    created_at: str = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SanityCheckResult:
    status: str = "ok"  # ok, warn, fail
    check_name: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    kb_id: Optional[int] = None
    text: str = ""
    paper_title: str = ""
    source: str = ""
    similarity_to_question: float = 0.0
    similarity_to_expected: float = 0.0


@dataclass
class ProposalVerdict:
    verdict: str  # supported, contradicted, inconclusive, needs_info, error
    confidence: float = 0.0
    explanation: str = ""
    required_to_convince: List[str] = field(default_factory=list)
    sanity: List[SanityCheckResult] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    evaluated_at: str = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
