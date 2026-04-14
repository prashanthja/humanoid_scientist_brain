# reasoning_module/claim_extractor.py
# ------------------------------------------------------------
# Claim Extractor (v2)
# - Heuristic scientific claim extraction without pretrained models
# - Extracts:
#     * equations
#     * definitions
#     * assumptions / conditions
#     * benchmark / result / method claims
# - IMPORTANT:
#     * avoids turning whole irrelevant chunks into claims
#     * extracts atomic sentence-level claims
#     * rejects boilerplate / URLs / metadata-style text
# ------------------------------------------------------------

from __future__ import annotations

import re
from typing import List, Tuple

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

# Sentences that look like scientific claims/results/method statements
CLAIM_CUES = [
    "we show", "we find", "we demonstrate", "we observe", "we report",
    "results show", "our results", "experiments show", "benchmark",
    "improves", "outperforms", "reduces", "increase", "decrease",
    "achieves", "preserves", "scales", "is more efficient",
    "memory", "latency", "throughput", "accuracy", "perplexity",
    "context length", "long-context", "sparse attention",
    "flashattention", "mixture-of-experts", "transformer",
]

BAD_PATTERNS = [
    r"https?://",
    r"\bgithub\.com\b",
    r"\bcode and models are publicly available\b",
    r"\bthis review\b",
    r"\bthe review concludes\b",
    r"\bcopyright\b",
    r"\ball rights reserved\b",
    r"\bet al\.\b",
    r"\barxiv\b",
    r"\bdoi\b",
    r"<jats:",
    r"</jats:",
]

NUMERIC_SIGNAL_RE = re.compile(r"\b\d+(\.\d+)?\b|%|x|×|ms|s|gb|mb|tb|tokens|params", re.IGNORECASE)
RESULT_SIGNAL_RE = re.compile(
    r"\b(improve|improves|improved|reduce|reduces|reduced|outperform|outperforms|"
    r"achieve|achieves|achieved|preserve|preserves|preserved|faster|lower|higher|"
    r"better|worse|efficient|efficiency|memory|latency|throughput|accuracy|perplexity)\b",
    re.IGNORECASE,
)
METHOD_SIGNAL_RE = re.compile(
    r"\b(method|approach|architecture|mechanism|attention|transformer|model|layer|routing|"
    r"sparse|dense|flashattention|mixture-of-experts|moe)\b",
    re.IGNORECASE,
)


def _symbols_from_expr(expr: str) -> List[str]:
    toks = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expr or "")
    blacklist = {
        "sin", "cos", "tan", "log", "ln", "exp", "sqrt",
        "and", "or", "the", "a", "an", "is", "are",
        "we", "this", "that"
    }
    out = []
    for t in toks:
        if t.lower() in blacklist:
            continue
        out.append(t)

    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[\.\!\?])\s+", (text or "").strip())
    out = []
    for s in sents:
        s = " ".join((s or "").strip().split())
        if s:
            out.append(s)
    return out


def _is_bad_sentence(s: str) -> bool:
    low = (s or "").lower().strip()
    if not low:
        return True
    if len(low) < 40:
        return True
    # Drop mid-sentence chunks (start with lowercase non-starter word)
    first_char = low[0] if low else ""
    good_starts = ("the","a ","an ","in ","on ","at ","by ","we ","our ","this","these","it ","its ","for ","with","from","that","they","when","here","both","all ","each","such","note","thus","also","as ","to ","furthermore","additionally","however","notably","overall","results","experiments","evaluation","performance")
    if first_char.islower() and not any(low.startswith(w) for w in good_starts):
        return True
    # Drop truncated sentences (don't end with proper punctuation)
    stripped = s.strip()
    if stripped and stripped[-1] not in ".!?\"')":
        return True

    if len(low) > 500:
        return True

    for p in BAD_PATTERNS:
        if re.search(p, low, flags=re.IGNORECASE):
            return True

    # kill metadata / citation junk
    if low.startswith(("abstract", "introduction", "keywords")):
        return True

    return False


def _looks_like_claim_sentence(s: str) -> bool:
    if _is_bad_sentence(s):
        return False

    low = s.lower()

    cue_hit = any(c in low for c in CLAIM_CUES)
    result_hit = bool(RESULT_SIGNAL_RE.search(s))
    method_hit = bool(METHOD_SIGNAL_RE.search(s))
    numeric_hit = bool(NUMERIC_SIGNAL_RE.search(s))

    # Good scientific claims usually have at least one of:
    # - result signal + method/topic
    # - numeric signal + result signal
    # - strong manual cue
    if cue_hit and (result_hit or method_hit or numeric_hit):
        return True

    if result_hit and method_hit:
        return True

    if numeric_hit and result_hit:
        return True

    return False


def _grab_sentences_containing(text: str, phrase: str) -> List[str]:
    sents = _split_sentences(text)
    out = []
    p = phrase.lower()
    for s in sents:
        if p in s.lower():
            out.append(s[:240])
    return out


def _dedup(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = " ".join((x or "").strip().split())
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _guess_domain(text: str) -> str:
    low = (text or "").lower()

    transformer = [
        "transformer", "attention", "flashattention", "sparse attention",
        "mixture-of-experts", "moe", "long-context", "context length",
        "kv cache", "latency", "throughput", "memory overhead", "tokens",
        "inference", "training", "perplexity"
    ]
    phys = [
        "force", "energy", "mass", "acceleration", "field",
        "quantum", "entropy", "thermo", "relativity"
    ]
    math = [
        "theorem", "proof", "lemma", "group", "ring",
        "manifold", "topology", "tensor", "integral", "derivative"
    ]

    tr = sum(1 for w in transformer if w in low)
    p = sum(1 for w in phys if w in low)
    m = sum(1 for w in math if w in low)

    if tr > 0 and tr >= max(p, m):
        return "transformer_efficiency"
    if p > m and p > 0:
        return "physics"
    if m > p and m > 0:
        return "math"
    if p == m and p > 0:
        return "mixed"
    return "unknown"


def extract_claims(text: str, *, provenance: SourceTrace | None = None) -> List[Claim]:
    t = " ".join((text or "").strip().split())
    if not t:
        return []

    prov = provenance or SourceTrace()
    claims: List[Claim] = []

    # ------------------------------------------------------------
    # 1) Equations
    # ------------------------------------------------------------
    eqs: List[Equation] = []
    for pat in EQ_PATTERNS:
        for m in re.finditer(pat, t):
            lhs = (m.group(1) or "").strip()
            rhs = (m.group(2) or "").strip()
            raw = f"{lhs} = {rhs}"
            lhs_syms = _symbols_from_expr(lhs)
            rhs_syms = _symbols_from_expr(rhs)
            syms = lhs_syms + [s for s in rhs_syms if s not in lhs_syms]
            eqs.append(Equation(raw=raw, lhs=lhs, rhs=rhs, symbols=syms))

    # ------------------------------------------------------------
    # 2) Definitions
    # ------------------------------------------------------------
    defs: List[Tuple[str, str]] = []
    for pat in DEF_PATTERNS:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            a = (m.group(1) or "").strip()
            b = (m.group(2) or "").strip()
            defs.append((a, b))

    # ------------------------------------------------------------
    # 3) Assumptions / conditions
    # ------------------------------------------------------------
    low = t.lower()
    assumptions = []
    conditions = []

    for w in ASSUME_WORDS:
        if w in low:
            assumptions.extend(_grab_sentences_containing(t, w))

    for w in COND_WORDS:
        if w in low:
            conditions.extend(_grab_sentences_containing(t, w))

    assumptions = _dedup(assumptions)
    conditions = _dedup(conditions)

    domain = _guess_domain(t)

    # ------------------------------------------------------------
    # 4) Equation claims
    # ------------------------------------------------------------
    if eqs:
        for e in eqs[:3]:
            ctext = e.raw
            cid = stable_id(ctext)
            claims.append(
                Claim(
                    claim_id=cid,
                    claim_text=ctext,
                    domain=domain,
                    claim_type="equation",
                    equations=[e],
                    symbols=e.symbols,
                    assumptions=assumptions,
                    conditions=conditions,
                    provenance=prov,
                )
            )

    # ------------------------------------------------------------
    # 5) Definition claims
    # ------------------------------------------------------------
    for (term, meaning) in defs[:3]:
        ctext = f"{term} is defined as {meaning}"
        cid = stable_id(ctext)
        syms_term = _symbols_from_expr(term)
        syms_meaning = _symbols_from_expr(meaning)
        syms = syms_term + [s for s in syms_meaning if s not in syms_term]

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

    # ------------------------------------------------------------
    # 6) Sentence-level scientific claim extraction
    # ------------------------------------------------------------
    sentences = _split_sentences(t)
    candidate_claims = []

    for s in sentences:
        if _looks_like_claim_sentence(s):
            candidate_claims.append(s)

    candidate_claims = _dedup(candidate_claims)

    for s in candidate_claims[:5]:
        cid = stable_id(s)
        claims.append(
            Claim(
                claim_id=cid,
                claim_text=s,
                domain=_guess_domain(s),
                claim_type="statement",
                equations=[],
                symbols=_symbols_from_expr(s),
                assumptions=assumptions,
                conditions=conditions,
                provenance=prov,
            )
        )

    # ------------------------------------------------------------
    # 7) Controlled fallback
    # IMPORTANT: never use full chunk as claim unless chunk itself
    # clearly looks like a focused claim-like sentence
    # ------------------------------------------------------------
    if not claims:
        best = None
        for s in sentences:
            if _is_bad_sentence(s):
                continue
            if RESULT_SIGNAL_RE.search(s) or METHOD_SIGNAL_RE.search(s):
                best = s
                break

        if best is not None:
            cid = stable_id(best)
            claims.append(
                Claim(
                    claim_id=cid,
                    claim_text=best,
                    domain=_guess_domain(best),
                    claim_type="statement",
                    symbols=_symbols_from_expr(best),
                    assumptions=assumptions,
                    conditions=conditions,
                    provenance=prov,
                )
            )

    return claims


class ClaimExtractor:
    """
    Compatibility wrapper.
    """
    def extract(self, text: str, *, provenance: SourceTrace | None = None) -> List[Claim]:
        return extract_claims(text, provenance=provenance)


__all__ = ["extract_claims", "ClaimExtractor"]