# reasoning_module/proposal_evaluator.py
# ------------------------------------------------------------
# Proposal Evaluator (v2 grounded)
# - Takes a proposal (any text)
# - Extracts structured claims
# - Runs sanity checks
# - Evaluates ONLY against provided evidence (chunks/texts)
#   (prevents "quantum paper supports microplastics" failures)
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, List, Optional, Sequence
import numpy as np
import re

from .claim_schema import (
    Claim, ProposalVerdict, SanityCheckResult, EvidenceItem, SourceTrace
)
from .claim_extractor import extract_claims
from .physics_sanity import run_sanity_checks


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)


# -------------------------
# Simple domain heuristics
# -------------------------
_DOMAIN_KEYWORDS = {
    "water_microplastics": [
        "microplastic", "nanoplastic", "plastic", "polyethylene", "polypropylene", "pet",
        "drinking water", "water treatment", "coagulation", "flocculation", "sedimentation",
        "filtration", "membrane", "ultrafiltration", "nanofiltration", "reverse osmosis",
        "activated carbon", "gac", "sand filter", "turbidity"
    ],
    "quantum": ["quantum", "qubit", "hamiltonian", "superconduct", "decoherence", "entanglement"],
    "math_physics_generic": ["tensor", "theorem", "lemma", "relativity", "entropy", "thermo", "electromagnet"],
}


def _guess_topic_bucket(text: str) -> str:
    low = (text or "").lower()
    scores = {}
    for k, kws in _DOMAIN_KEYWORDS.items():
        scores[k] = sum(1 for w in kws if w in low)
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] <= 0:
        return "unknown"
    return best[0]


def _jaccard_keywords(a: str, b: str) -> float:
    # crude but effective to penalize nonsense matches
    def toks(x: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", (x or "").lower())
        stop = {"this", "that", "with", "from", "into", "have", "been", "were", "their", "which", "also"}
        return {w for w in words if w not in stop}

    A = toks(a)
    B = toks(b)
    if not A or not B:
        return 0.0
    return float(len(A & B)) / float(len(A | B))


def _is_definitionish(text: str) -> bool:
    """
    For broad definition-style claims, keyword overlap can be low even when evidence is good.
    We relax overlap gating for these.
    """
    low = (text or "").lower().strip()
    if not low:
        return False
    patterns = [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bis defined as\b",
        r"\bdefined as\b",
        r"\brefers to\b",
        r"\bmeans\b",
        r"\bis the study of\b",
        r"\bis a\b",
        r"\bare a\b",
    ]
    return any(re.search(p, low) for p in patterns)


class ProposalEvaluator:
    """
    v2: Evidence must be supplied (chunks/texts). This stops false support.
    """

    def __init__(
        self,
        kb,
        bridge,
        *,
        top_k: int = 10,
        evidence_threshold: float = 0.60,
        max_kb_items: int = 3000,
        require_evidence: bool = True,   # default True (safer)
        domain_mismatch_penalty: float = 0.25,
        min_keyword_overlap: float = 0.02,
    ):
        self.kb = kb
        self.bridge = bridge
        self.top_k = int(top_k)
        self.th = float(evidence_threshold)
        self.max_kb_items = int(max_kb_items)

        self.require_evidence = bool(require_evidence)
        self.domain_mismatch_penalty = float(domain_mismatch_penalty)
        self.min_keyword_overlap = float(min_keyword_overlap)

    # -------------------------
    # Public API
    # -------------------------
    def evaluate(
        self,
        proposal_text: str,
        *,
        provenance: Optional[SourceTrace] = None,
        evidence_chunks: Optional[Sequence[Dict[str, Any]]] = None,
        evidence_texts: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        prov = provenance or SourceTrace(source_type="user_proposal", source_name="local")

        proposal = (proposal_text or "").strip()
        claims = extract_claims(proposal, provenance=prov)

        if not claims:
            verdict = ProposalVerdict(
                verdict="needs_info",
                confidence=0.0,
                explanation="No claims could be extracted. Provide a clearer statement or an equation.",
                required_to_convince=["Provide a precise claim, equation, or measurable prediction."],
            )
            return {"claims": [], "verdict": verdict.to_dict()}

        primary = claims[0]
        sanity = run_sanity_checks(primary)

        # Build evidence candidates ONLY from provided evidence
        candidates = self._evidence_candidates(evidence_chunks=evidence_chunks, evidence_texts=evidence_texts)

        # DEBUG: prove we are using provided chunks (chunk_id should appear here)
        print("DEBUG candidates_count =", len(candidates))
        print("DEBUG candidate_ids(sample) =", [c.get("id") for c in candidates[:5]])
        print("DEBUG candidate_sources(sample) =", [c.get("source") for c in candidates[:3]])

        if self.require_evidence and not candidates:
            verdict = ProposalVerdict(
                verdict="inconclusive",
                confidence=0.25,
                explanation="No evidence provided to ground the proposal. Retrieve/ingest domain chunks first, then re-evaluate.",
                sanity=sanity,
                evidence=[],
                required_to_convince=[
                    "Pass retrieved chunk evidence into ProposalEvaluator (evidence_chunks/evidence_texts).",
                    "Ingest domain sources relevant to this proposal and rebuild ChunkIndex.",
                ],
            )
            return {
                "claims": [c.to_dict() for c in claims],
                "primary_claim_id": primary.claim_id,
                "verdict": verdict.to_dict(),
            }

        evidence = self._rank_evidence(primary.claim_text, candidates)
        verdict = self._decide(primary, sanity, evidence)

        return {
            "claims": [c.to_dict() for c in claims],
            "primary_claim_id": primary.claim_id,
            "verdict": verdict.to_dict(),
        }

    # -------------------------
    # Embeddings
    # -------------------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        if hasattr(self.bridge, "encode_texts"):
            return np.asarray(self.bridge.encode_texts(texts), dtype=np.float32)
        if hasattr(self.bridge, "embed"):
            return np.asarray(self.bridge.embed(texts), dtype=np.float32)
        raise RuntimeError("Bridge has no encode_texts/embed")

    # -------------------------
    # Evidence handling (grounded)
    # -------------------------
    def _evidence_candidates(self, *, evidence_chunks=None, evidence_texts=None):
        candidates: List[Dict[str, Any]] = []

        # 1) Prefer explicit chunk evidence if provided
        if evidence_chunks:
            for c in evidence_chunks:
                text = (c.get("text") or c.get("chunk_text") or c.get("text_preview") or "").strip()
                if not text:
                    continue
                candidates.append({
                    "id": c.get("chunk_id"),  # chunk id
                    "text": text,
                    "paper_title": c.get("paper_title", ""),
                    "source": c.get("source", "chunk_index"),
                    "kind": "chunk",
                    "similarity_to_question": float(c.get("similarity", 0.0) or 0.0),
                })
            return candidates  # CRITICAL: stop here. Do NOT fall back to KB.

        # 2) Optional: if no chunks provided, use raw evidence texts
        if evidence_texts:
            for t in evidence_texts:
                if t and str(t).strip():
                    candidates.append({"id": None, "text": str(t).strip(), "kind": "text", "source": "text"})

        return candidates

    def _rank_evidence(self, query: str, candidates: List[Dict[str, Any]]) -> List[EvidenceItem]:
        if not candidates:
            return []

        qv = self._encode([query])
        if qv.size == 0:
            return []
        qv = _normalize_rows(qv)[0]

        texts = [x["text"] for x in candidates]
        X = self._encode(texts)
        if X.size == 0:
            return []
        Xn = _normalize_rows(X)

        sims = Xn @ qv
        top_idx = np.argsort(-sims)[: min(self.top_k, int(sims.shape[0]))]

        out: List[EvidenceItem] = []
        for i in top_idx:
            it = candidates[int(i)]
            kind = it.get("kind")

            out.append(EvidenceItem(
                kb_id=it.get("id") if kind != "chunk" else None,
                chunk_id=it.get("id") if kind == "chunk" else None,
                text=(it.get("text") or "")[:600],
                paper_title=it.get("paper_title", ""),
                source=it.get("source", ""),
                similarity_to_question=float(sims[int(i)]),
                similarity_to_expected=0.0,
            ))
        return out

    # -------------------------
    # Verdict logic (grounded)
    # -------------------------
    def _decide(self, claim: Claim, sanity: List[SanityCheckResult], evidence: List[EvidenceItem]) -> ProposalVerdict:
        fails = [s for s in sanity if s.status == "fail"]
        warns = [s for s in sanity if s.status == "warn"]

        best_ev_sim = max([e.similarity_to_question for e in evidence], default=0.0)

        # Domain mismatch penalty
        proposal_bucket = _guess_topic_bucket(claim.claim_text)
        evidence_bucket = _guess_topic_bucket(
            " ".join([(e.paper_title or "") + " " + (e.text or "") for e in evidence[:3]])
        )
        mismatch = (proposal_bucket != "unknown" and evidence_bucket != "unknown" and proposal_bucket != evidence_bucket)

        # Keyword overlap check (relaxed for definition-style claims)
        overlap = 0.0
        if evidence:
            overlap = max(_jaccard_keywords(claim.claim_text, e.text) for e in evidence[:5])

        definitionish = _is_definitionish(claim.claim_text)
        overlap_fail = (overlap < self.min_keyword_overlap) and (not definitionish)

        # 1) Failing sanity checks blocks
        if fails:
            conf = min(0.85, 0.60 + 0.10 * len(fails))
            if mismatch:
                conf = max(0.10, conf - self.domain_mismatch_penalty)

            return ProposalVerdict(
                verdict="contradicted",
                confidence=conf,
                explanation="Claim fails basic physics/math sanity checks (v1). Fix dimensional consistency or define symbols/assumptions.",
                sanity=sanity,
                evidence=evidence,
                required_to_convince=[
                    "Provide a corrected equation with dimensionally consistent terms.",
                    "Define every symbol and units.",
                    "State regime/assumptions (limits, approximations, conditions).",
                ],
            )

        # 2) Weak evidence -> inconclusive
        if (not evidence) or (best_ev_sim < self.th) or overlap_fail:
            base = 0.40 if not warns else 0.30
            if mismatch:
                base = max(0.10, base - self.domain_mismatch_penalty)

            why = []
            if not evidence:
                why.append("no evidence candidates")
            if best_ev_sim < self.th:
                why.append(f"best_sim {best_ev_sim:.3f} < {self.th:.3f}")
            if overlap_fail:
                why.append(f"keyword_overlap {overlap:.3f} < {self.min_keyword_overlap:.3f}")
            if mismatch:
                why.append(f"domain_mismatch ({proposal_bucket} vs {evidence_bucket})")

            return ProposalVerdict(
                verdict="inconclusive",
                confidence=base,
                explanation="Not enough strong grounded evidence to support/refute. " + ("; ".join(why) if why else ""),
                sanity=sanity,
                evidence=evidence,
                required_to_convince=[
                    "Retrieve higher-quality evidence chunks (methods + numeric results).",
                    "Tighten the proposal into a measurable claim (e.g., % removal under conditions).",
                    "Add sources explicitly about the same domain/mechanism.",
                ],
            )

        # 3) Evidence seems relevant
        base_conf = 0.70 if not warns else 0.55
        conf = min(0.90, base_conf + 0.10 * (best_ev_sim - self.th))
        if mismatch:
            conf = max(0.10, conf - self.domain_mismatch_penalty)

        return ProposalVerdict(
            verdict="supported",
            confidence=conf,
            explanation="Evidence is semantically aligned to the claim (grounded to provided chunks). This is retrieval-grounded support, not a proof.",
            sanity=sanity,
            evidence=evidence,
            required_to_convince=[
                "Provide a falsifiable prediction or quantitative threshold.",
                "Cite a source that explicitly reports the method + outcome.",
            ],
        )
