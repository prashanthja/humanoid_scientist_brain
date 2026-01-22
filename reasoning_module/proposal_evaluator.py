# reasoning_module/proposal_evaluator.py
# ------------------------------------------------------------
# Proposal Evaluator (v1)
# - Takes a proposal (any text)
# - Extracts structured claims
# - Runs sanity checks
# - Retrieves evidence from KB using embeddings
# - Produces a verdict object (stable schema)
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

from .claim_schema import (
    Claim, ProposalVerdict, SanityCheckResult, EvidenceItem, SourceTrace
)
from .claim_extractor import extract_claims
from .physics_sanity import run_sanity_checks


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)


class ProposalEvaluator:
    def __init__(
        self,
        kb,
        bridge,
        *,
        top_k: int = 10,
        evidence_threshold: float = 0.60,
        max_kb_items: int = 3000,
    ):
        self.kb = kb
        self.bridge = bridge
        self.top_k = int(top_k)
        self.th = float(evidence_threshold)
        self.max_kb_items = int(max_kb_items)

    def evaluate(self, proposal_text: str, *, provenance: Optional[SourceTrace] = None) -> Dict[str, Any]:
        prov = provenance or SourceTrace(source_type="user_proposal", source_name="local")

        claims = extract_claims(proposal_text, provenance=prov)
        if not claims:
            verdict = ProposalVerdict(
                verdict="needs_info",
                confidence=0.0,
                explanation="No claims could be extracted. Provide a clearer statement or an equation.",
                required_to_convince=["Provide a precise claim, equation, or measurable prediction."],
            )
            return {"claims": [], "verdict": verdict.to_dict()}

        # For v1, evaluate the "primary" claim = first extracted
        primary = claims[0]
        sanity = run_sanity_checks(primary)

        # Evidence retrieval
        evidence = self._retrieve_evidence(primary.claim_text)

        # Decide verdict
        verdict = self._decide(primary, sanity, evidence)

        return {
            "claims": [c.to_dict() for c in claims],
            "primary_claim_id": primary.claim_id,
            "verdict": verdict.to_dict(),
        }

    # -------------------------
    # Evidence retrieval
    # -------------------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        if hasattr(self.bridge, "encode_texts"):
            return np.asarray(self.bridge.encode_texts(texts), dtype=np.float32)
        if hasattr(self.bridge, "embed"):
            return np.asarray(self.bridge.embed(texts), dtype=np.float32)
        raise RuntimeError("Bridge has no encode_texts/embed")

    def _load_kb(self) -> List[Dict[str, Any]]:
        items = self.kb.query("") or []
        out = []
        for it in items[: self.max_kb_items]:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            out.append({
                "id": it.get("id"),
                "text": text,
                "paper_title": str(it.get("paper_title") or ""),
                "source": str(it.get("source") or ""),
            })
        return out

    def _retrieve_evidence(self, query: str) -> List[EvidenceItem]:
        kb_items = self._load_kb()
        if not kb_items:
            return []

        qv = self._encode([query])
        if qv.size == 0:
            return []
        qv = _normalize_rows(qv)[0]

        texts = [x["text"] for x in kb_items]
        X = self._encode(texts)
        if X.size == 0:
            return []
        Xn = _normalize_rows(X)

        sims = Xn @ qv
        top_idx = np.argsort(-sims)[: self.top_k]

        out: List[EvidenceItem] = []
        for i in top_idx:
            it = kb_items[int(i)]
            out.append(EvidenceItem(
                kb_id=it["id"],
                text=it["text"][:600],
                paper_title=it["paper_title"],
                source=it["source"],
                similarity_to_question=float(sims[int(i)]),
                similarity_to_expected=0.0,  # v1: no gold expected
            ))
        return out

    # -------------------------
    # Verdict logic (v1)
    # -------------------------
    def _decide(self, claim: Claim, sanity: List[SanityCheckResult], evidence: List[EvidenceItem]) -> ProposalVerdict:
        # Any FAIL sanity => mostly block
        fails = [s for s in sanity if s.status == "fail"]
        warns = [s for s in sanity if s.status == "warn"]

        best_ev_sim = max([e.similarity_to_question for e in evidence], default=0.0)

        if fails:
            return ProposalVerdict(
                verdict="contradicted",
                confidence=min(0.85, 0.60 + 0.10 * len(fails)),
                explanation="Claim fails basic physics/math sanity checks (v1). Fix dimensional consistency or define symbols/assumptions.",
                sanity=sanity,
                evidence=evidence,
                required_to_convince=[
                    "Provide a corrected equation with dimensionally consistent terms.",
                    "Define every symbol and units.",
                    "State regime/assumptions (limits, approximations, conditions).",
                ],
            )

        # If no evidence at all, or low similarity, we can't judge
        if not evidence or best_ev_sim < self.th:
            return ProposalVerdict(
                verdict="inconclusive",
                confidence=0.35 if warns else 0.45,
                explanation="Not enough strong matching evidence in KB to support or refute. Retrieval does not ground this claim yet.",
                sanity=sanity,
                evidence=evidence,
                required_to_convince=[
                    "Add high-quality sources relevant to this claim (textbooks/papers) into KB.",
                    "Provide measurable predictions or derivation steps.",
                    "Provide an experiment or calculation pathway that could falsify the claim.",
                ],
            )

        # Evidence seems relevant; if also warnings exist, reduce confidence
        base_conf = 0.70 if not warns else 0.55
        return ProposalVerdict(
            verdict="supported",
            confidence=min(0.90, base_conf + 0.10 * (best_ev_sim - self.th)),
            explanation="Evidence retrieved from KB is semantically close to the claim. v1 cannot guarantee truth—only retrieval-grounded support.",
            sanity=sanity,
            evidence=evidence,
            required_to_convince=[
                "Provide derivation steps or a falsifiable prediction.",
                "Add citations that explicitly state the relationship in a canonical source.",
            ],
        )
