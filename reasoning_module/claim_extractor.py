# reasoning_module/claim_extractor.py
# ------------------------------------------------------------
# Claim Extractor (v1)
# - Pure heuristic extraction (no pretrained models)
# - Finds equations, definitions, assumptions, conditions
# ------------------------------------------------------------

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
from .claim_schema import Claim, Equation, SourceTrace, stable_id

EQ_PATTERNS = [
    r"([A-Za-z0-9_\^\*\+\-/\(\)\s]+)\s*=\s*([A-Za-z0-9_\^\*\+\-/\(\)\s]+)",
]

DEF_PATTERNS = [
    r"\b([A-Za-z][A-Za-z0-9_\- ]{0,40})\s+is\s+defined\s+as\s+(.{3,200})",
    r"\bdefine\s+([A-Za-z][A-Za-z0-9_\- ]{0,40})\s*[:\-]\s*(.{3,200})",
]

ASSUME_WORDS = ["assume", "assuming", "assumption", "suppose", "given that"]
COND_WORDS = ["in the limit", "for small", "for large", "at high", "at low", "when", "under", "if"]


def _symbols_from_expr(expr: str) -> List[str]:
    # capture token-like symbols, exclude numbers
    toks = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expr or "")
    # remove common words
    blacklist = {"sin","cos","tan","log","ln","exp","sqrt","and","or","the","a","an","is","are"}
    out = []
    for t in toks:
        if t.lower() in blacklist:
            continue
        out.append(t)
    # unique preserve order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def extract_claims(text: str, *, provenance: SourceTrace | None = None) -> List[Claim]:
    t = (text or "").strip()
    if not t:
        return []

    prov = provenance or SourceTrace()

    claims: List[Claim] = []

    # 1) Equations
    eqs: List[Equation] = []
    for pat in EQ_PATTERNS:
        for m in re.finditer(pat, t):
            lhs = (m.group(1) or "").strip()
            rhs = (m.group(2) or "").strip()
            raw = f"{lhs} = {rhs}"
            syms = _symbols_from_expr(lhs) + [s for s in _symbols_from_expr(rhs) if s not in _symbols_from_expr(lhs)]
            eqs.append(Equation(raw=raw, lhs=lhs, rhs=rhs, symbols=syms))

    # 2) Definitions (make them claims too)
    defs: List[Tuple[str, str]] = []
    for pat in DEF_PATTERNS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            a = (m.group(1) or "").strip()
            b = (m.group(2) or "").strip()
            defs.append((a, b))

    # 3) Assumptions / conditions
    low = t.lower()
    assumptions = []
    conditions = []

    for w in ASSUME_WORDS:
        if w in low:
            # naive sentence grab
            assumptions.extend(_grab_sentences_containing(t, w))

    for w in COND_WORDS:
        if w in low:
            conditions.extend(_grab_sentences_containing(t, w))

    assumptions = _dedup(assumptions)
    conditions = _dedup(conditions)

    # Decide domain
    domain = _guess_domain(t)

    # Main claim object
    if eqs:
        ctext = t
        cid = stable_id(ctext)
        sym_all = []
        for e in eqs:
            for s in e.symbols:
                if s not in sym_all:
                    sym_all.append(s)
        claims.append(
            Claim(
                claim_id=cid,
                claim_text=ctext,
                domain=domain,
                claim_type="equation",
                equations=eqs,
                symbols=sym_all,
                assumptions=assumptions,
                conditions=conditions,
                provenance=prov,
            )
        )

    for (term, meaning) in defs:
        ctext = f"{term} is defined as {meaning}"
        cid = stable_id(ctext)
        syms = _symbols_from_expr(term) + [s for s in _symbols_from_expr(meaning) if s not in _symbols_from_expr(term)]
        claims.append(
            Claim(
                claim_id=cid,
                claim_text=ctext,
                domain=domain,
                claim_type="definition",
                equations=[],
                symbols=_dedup(syms),
                assumptions=assumptions,
                conditions=conditions,
                provenance=prov,
            )
        )

    # If nothing extracted, fallback: statement claim
    if not claims:
        cid = stable_id(t)
        claims.append(
            Claim(
                claim_id=cid,
                claim_text=t,
                domain=domain,
                claim_type="statement",
                symbols=_symbols_from_expr(t),
                assumptions=assumptions,
                conditions=conditions,
                provenance=prov,
            )
        )

    return claims


def _grab_sentences_containing(text: str, phrase: str) -> List[str]:
    # crude split; enough for v1
    sents = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    out = []
    p = phrase.lower()
    for s in sents:
        if p in s.lower():
            out.append(s.strip()[:240])
    return out


def _dedup(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = (x or "").strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _guess_domain(text: str) -> str:
    low = (text or "").lower()
    phys = ["force", "energy", "mass", "acceleration", "field", "quantum", "entropy", "thermo", "relativity"]
    math = ["theorem", "proof", "lemma", "group", "ring", "manifold", "topology", "tensor", "integral", "derivative"]
    p = sum(1 for w in phys if w in low)
    m = sum(1 for w in math if w in low)
    if p > m and p > 0:
        return "physics"
    if m > p and m > 0:
        return "math"
    if p == m and p > 0:
        return "mixed"
    return "unknown"

# ---- Compatibility wrapper (so imports like `from ... import ClaimExtractor` work) ----

class ClaimExtractor:
    """
    Thin wrapper over extract_claims() for compatibility with older imports.
    Usage:
        ce = ClaimExtractor()
        claims = ce.extract(text, provenance=SourceTrace(...))
    """
    def extract(self, text: str, *, provenance: SourceTrace | None = None) -> List[Claim]:
        return extract_claims(text, provenance=provenance)


__all__ = ["extract_claims", "ClaimExtractor"]
