# reasoning_module/discovery_report.py
# ------------------------------------------------------------
# Discovery Report Generator
# Converts pipeline output into a structured report saved to
# outputs/discovery_reports/
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import time
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "discovery_reports"
)


@dataclass
class DiscoveryReport:
    query: str
    timestamp: str
    domain: str

    # Verdict
    proposal_verdict: str
    proposal_confidence: float
    proposal_explanation: str

    # Evidence
    evidence_count: int
    top_papers: List[str]

    # Claims
    extracted_claim_count: int
    top_claims: List[Dict[str, Any]]

    # Grounded claims
    grounded_claim_count: int
    supported_count: int
    contradicted_count: int
    inconclusive_count: int
    top_grounded: List[Dict[str, Any]]

    # Knowledge gaps
    knowledge_gaps: List[str]

    # Next actions
    next_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = []

        lines.append(f"# Discovery Report")
        lines.append(f"")
        lines.append(f"**Query:** {self.query}")
        lines.append(f"**Domain:** {self.domain}")
        lines.append(f"**Generated:** {self.timestamp}")
        lines.append(f"")

        # Verdict section
        lines.append(f"---")
        lines.append(f"## Verdict")
        lines.append(f"")
        verdict_emoji = {
            "supported": "✅",
            "partially_supported": "🟡",
            "inconclusive": "⚪",
            "contradicted": "❌",
        }.get(self.proposal_verdict, "❓")
        lines.append(f"**{verdict_emoji} {self.proposal_verdict.upper()}**")
        lines.append(f"")
        lines.append(f"Confidence: `{self.proposal_confidence:.2f}`")
        lines.append(f"")
        lines.append(f"{self.proposal_explanation}")
        lines.append(f"")

        # Evidence section
        lines.append(f"---")
        lines.append(f"## Evidence Sources")
        lines.append(f"")
        lines.append(f"Retrieved **{self.evidence_count}** evidence chunks from:")
        lines.append(f"")
        for p in self.top_papers:
            lines.append(f"- {p}")
        lines.append(f"")

        # Top claims
        lines.append(f"---")
        lines.append(f"## Key Claims Extracted")
        lines.append(f"")
        lines.append(f"Extracted **{self.extracted_claim_count}** claims from evidence.")
        lines.append(f"")
        for i, c in enumerate(self.top_claims, 1):
            lines.append(f"**{i}.** {c.get('claim', '')}")
            lines.append(f"   - Source: *{c.get('paper_title', 'unknown')}*")
            lines.append(f"   - Domain: `{c.get('domain', 'unknown')}`")
            lines.append(f"")

        # Grounded claims
        lines.append(f"---")
        lines.append(f"## Evidence Evaluation")
        lines.append(f"")
        lines.append(
            f"| Verdict | Count |"
            f"\n|---|---|"
            f"\n| ✅ Supported | {self.supported_count} |"
            f"\n| 🟡 Partially supported | {self.inconclusive_count} |"
            f"\n| ❌ Contradicted | {self.contradicted_count} |"
        )
        lines.append(f"")
        lines.append(f"### Top Grounded Claims")
        lines.append(f"")
        for gc in self.top_grounded:
            v = gc.get("verdict", {})
            verdict = v.get("verdict", "unknown")
            conf = v.get("confidence", 0.0)
            claim = gc.get("claim", "")
            emoji = {
                "supported": "✅",
                "partially_supported": "🟡",
                "inconclusive": "⚪",
                "contradicted": "❌",
            }.get(verdict, "❓")
            lines.append(f"{emoji} **{verdict}** (confidence: `{conf:.2f}`)")
            lines.append(f"> {claim}")
            lines.append(f"")

        # Knowledge gaps
        if self.knowledge_gaps:
            lines.append(f"---")
            lines.append(f"## Knowledge Gaps")
            lines.append(f"")
            for g in self.knowledge_gaps:
                lines.append(f"- {g}")
            lines.append(f"")

        # Next actions
        lines.append(f"---")
        lines.append(f"## Recommended Next Actions")
        lines.append(f"")
        for i, a in enumerate(self.next_actions, 1):
            lines.append(f"{i}. {a}")
        lines.append(f"")

        return "\n".join(lines)


def _detect_domain(result: Dict[str, Any]) -> str:
    claims = result.get("extracted_claims", [])
    domain_counts: Dict[str, int] = {}
    for c in claims:
        d = c.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    if not domain_counts:
        return "unknown"
    return max(domain_counts, key=domain_counts.get)


def _detect_knowledge_gaps(
    result: Dict[str, Any],
    grounded: List[Dict[str, Any]],
) -> List[str]:
    gaps = []

    # Gap 1: low evidence coverage
    chunks = result.get("evidence_chunks", [])
    if len(chunks) < 5:
        gaps.append(
            "Limited evidence coverage — fewer than 5 chunks retrieved. "
            "Ingest more domain papers to improve coverage."
        )

    # Gap 2: high inconclusiveness
    inconclusive = [
        g for g in grounded
        if g.get("verdict", {}).get("verdict") in ("inconclusive", "partially_supported")
    ]
    if len(inconclusive) > len(grounded) // 2 and grounded:
        gaps.append(
            "Many claims are inconclusive — evidence exists but lacks "
            "benchmark numbers or experimental results."
        )

    # Gap 3: no contradictions found (suspicious if many claims)
    contradicted = [
        g for g in grounded
        if g.get("verdict", {}).get("verdict") == "contradicted"
    ]
    if not contradicted and len(grounded) >= 3:
        gaps.append(
            "No contradicting evidence found — this may indicate limited "
            "coverage of papers reporting failure cases or trade-offs."
        )

    # Gap 4: domain mismatch in retrieved chunks
    claims = result.get("extracted_claims", [])
    off_domain = [c for c in claims if c.get("domain") not in ("transformer_efficiency",)]
    if len(off_domain) > len(claims) // 3 and claims:
        gaps.append(
            "Some retrieved chunks appear off-domain — retrieval may benefit "
            "from stricter domain filtering."
        )

    return gaps


def build_report(query: str, result: Dict[str, Any]) -> DiscoveryReport:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    pv = result.get("proposal_verdict", {})
    chunks = result.get("evidence_chunks", [])
    claims = result.get("extracted_claims", [])
    grounded = result.get("grounded_claims", [])
    next_actions = result.get("next_actions", [])

    # Top papers — deduplicated
    seen_papers = set()
    top_papers = []
    for c in chunks:
        p = c.get("paper_title", "").strip()
        if p and p not in seen_papers:
            seen_papers.add(p)
            top_papers.append(p)
        if len(top_papers) >= 8:
            break

    # Verdict counts
    supported = sum(
        1 for g in grounded
        if g.get("verdict", {}).get("verdict") == "supported"
    )
    contradicted = sum(
        1 for g in grounded
        if g.get("verdict", {}).get("verdict") == "contradicted"
    )
    inconclusive = sum(
        1 for g in grounded
        if g.get("verdict", {}).get("verdict") in ("inconclusive", "partially_supported")
    )

    # Top claims — keep transformer_efficiency domain first
    sorted_claims = sorted(
        claims,
        key=lambda c: (
            0 if c.get("domain") == "transformer_efficiency" else 1,
            -float(c.get("similarity", 0) or 0),
        )
    )

    # Top grounded — sorted by confidence
    sorted_grounded = sorted(
        grounded,
        key=lambda g: float(g.get("verdict", {}).get("confidence", 0)),
        reverse=True,
    )

    gaps = _detect_knowledge_gaps(result, grounded)

    return DiscoveryReport(
        query=query,
        timestamp=timestamp,
        domain=_detect_domain(result),
        proposal_verdict=pv.get("verdict", "unknown"),
        proposal_confidence=float(pv.get("confidence", 0.0)),
        proposal_explanation=pv.get("explanation", ""),
        evidence_count=len(chunks),
        top_papers=top_papers,
        extracted_claim_count=len(claims),
        top_claims=[
            {
                "claim": c.get("claim", ""),
                "paper_title": c.get("paper_title", ""),
                "domain": c.get("domain", "unknown"),
            }
            for c in sorted_claims[:5]
        ],
        grounded_claim_count=len(grounded),
        supported_count=supported,
        contradicted_count=contradicted,
        inconclusive_count=inconclusive,
        top_grounded=sorted_grounded[:5],
        knowledge_gaps=gaps,
        next_actions=next_actions,
    )


def save_report(report: DiscoveryReport, fmt: str = "both") -> Dict[str, str]:
    """
    Save report to outputs/discovery_reports/
    fmt: "json", "markdown", or "both"
    Returns dict of saved paths.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    slug = report.query.lower()
    slug = "".join(c if c.isalnum() else "_" for c in slug)[:60]
    slug = slug.strip("_")
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{slug}_{ts}"

    saved = {}

    if fmt in ("json", "both"):
        path = os.path.join(REPORTS_DIR, f"{base}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        saved["json"] = path

    if fmt in ("markdown", "both"):
        path = os.path.join(REPORTS_DIR, f"{base}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        saved["markdown"] = path

    return saved


def generate_and_save(query: str, result: Dict[str, Any], fmt: str = "both") -> Dict[str, str]:
    """
    One-call helper: build report from pipeline result and save it.
    Returns paths of saved files.
    """
    report = build_report(query, result)
    paths = save_report(report, fmt=fmt)
    return paths