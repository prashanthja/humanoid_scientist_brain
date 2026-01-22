# reasoning_module/physics_sanity.py
# ------------------------------------------------------------
# Physics/Math Sanity Engine (v1)
# - Heuristic checks with a simple dimension algebra system
# - NO third-party libs
# ------------------------------------------------------------

from __future__ import annotations
import re
from typing import Dict, List, Tuple, Any
from .claim_schema import Claim, SanityCheckResult

# ------------------------------------------------------------
# Dimension algebra (base dims)
# L: length, M: mass, T: time, I: current, Θ: temperature
# Stored as dict[str,int] exponents
# ------------------------------------------------------------

Dim = Dict[str, int]


def dim_add(a: Dim, b: Dim) -> Dim:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
        if out[k] == 0:
            out.pop(k, None)
    return out


def dim_sub(a: Dim, b: Dim) -> Dim:
    return dim_add(a, {k: -v for k, v in b.items()})


def dim_mul(a: Dim, b: Dim) -> Dim:
    return dim_add(a, b)


def dim_pow(a: Dim, p: int) -> Dim:
    return {k: v * p for k, v in a.items()}


def dim_eq(a: Dim, b: Dim) -> bool:
    return a == b


# ------------------------------------------------------------
# A starter symbol->dimension map (you will expand this).
# This is where your engine becomes "physics aware".
# ------------------------------------------------------------

SYMBOL_DIMS: Dict[str, Dim] = {
    # mechanics
    "x": {"L": 1},
    "r": {"L": 1},
    "s": {"L": 1},
    "v": {"L": 1, "T": -1},
    "a": {"L": 1, "T": -2},
    "t": {"T": 1},
    "m": {"M": 1},
    "F": {"M": 1, "L": 1, "T": -2},  # Newton
    "E": {"M": 1, "L": 2, "T": -2},  # Joule
    "W": {"M": 1, "L": 2, "T": -2},
    "P": {"M": 1, "L": 2, "T": -3},  # power
    # thermo
    "T": {"Θ": 1},  # temperature (overloads time; context needed)
    "S": {"M": 1, "L": 2, "T": -2, "Θ": -1},  # entropy (J/K)
    # EM (very incomplete, but okay for v1 flags)
    "q": {"I": 1, "T": 1},  # coulomb = A*s
    "I": {"I": 1},
    "V": {"M": 1, "L": 2, "T": -3, "I": -1},  # volt
}

# tokens treated as scalars (dimensionless)
DIMLESS = {"pi", "e", "c0", "k", "hbar", "alpha"}


def run_sanity_checks(claim: Claim) -> List[SanityCheckResult]:
    out: List[SanityCheckResult] = []

    # 1) Basic: empty claim
    if not (claim.claim_text or "").strip():
        out.append(SanityCheckResult(status="fail", check_name="nonempty", message="Empty claim text."))
        return out

    # 2) Perpetual motion / free energy red flags
    out.extend(_perpetual_motion_flags(claim.claim_text))

    # 3) Equation checks (if any)
    if claim.claim_type == "equation" and claim.equations:
        for eq in claim.equations:
            out.append(_eq_structure_check(eq.raw))
            out.extend(_dimension_check(eq.lhs, eq.rhs, eq.raw))

    # 4) Assumption completeness (weak but useful)
    if claim.claim_type in ("equation", "prediction", "invention"):
        if len(claim.assumptions) == 0 and len(claim.conditions) == 0:
            out.append(SanityCheckResult(
                status="warn",
                check_name="missing_assumptions",
                message="No assumptions/conditions detected. Scientific claims usually require regime/limits.",
            ))

    # normalize: remove ok checks unless you want verbose
    return [x for x in out if x.check_name and x.status in ("warn", "fail")]


def _eq_structure_check(raw: str) -> SanityCheckResult:
    if "=" not in raw:
        return SanityCheckResult(status="fail", check_name="equation_format", message="Equation missing '='.")
    lhs, rhs = raw.split("=", 1)
    if not lhs.strip() or not rhs.strip():
        return SanityCheckResult(status="fail", check_name="equation_format", message="Equation has empty LHS/RHS.")
    return SanityCheckResult(status="ok", check_name="equation_format", message="ok")


def _perpetual_motion_flags(text: str) -> List[SanityCheckResult]:
    low = (text or "").lower()
    flags = []
    patterns = [
        ("free_energy", r"\bfree energy\b|\benergy from nothing\b|\bzero input\b|\bperpetual\b"),
        ("overunity", r"\boverunity\b|\bmore energy than input\b|\bout > in\b"),
    ]
    for name, pat in patterns:
        if re.search(pat, low):
            flags.append(SanityCheckResult(
                status="warn",
                check_name=name,
                message="Red-flag phrasing detected. Requires extremely strong evidence + explicit energy accounting.",
                details={"pattern": pat},
            ))
    return flags


# ------------------------------------------------------------
# Dimension parsing (very limited, but useful)
# Supports:
#   - multiplication: a*b or "m a" treated as multiply
#   - division: a/b
#   - powers: x^2
# Ignores:
#   - addition/subtraction (must be same dim; we check by splitting +,-)
# ------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\^|\*|/|\+|\-|\(|\)|\d+(\.\d+)?")


def _dimension_check(lhs: str, rhs: str, raw: str) -> List[SanityCheckResult]:
    out: List[SanityCheckResult] = []

    dl = _dim_of_expr(lhs)
    dr = _dim_of_expr(rhs)

    if dl is None or dr is None:
        out.append(SanityCheckResult(
            status="warn",
            check_name="dimension_unknown",
            message="Could not compute dimensions (unknown symbols). Add symbol dimensions to SYMBOL_DIMS.",
            details={"equation": raw},
        ))
        return out

    if not dim_eq(dl, dr):
        out.append(SanityCheckResult(
            status="fail",
            check_name="dimension_mismatch",
            message="Dimensional mismatch between LHS and RHS.",
            details={"equation": raw, "lhs_dim": dl, "rhs_dim": dr},
        ))

    return out


def _dim_of_expr(expr: str) -> Dim | None:
    # split by + or - first: all terms must match
    expr = (expr or "").strip()
    if not expr:
        return {}

    # Normalize implicit multiplication: "m a" -> "m*a"
    expr = re.sub(r"([A-Za-z0-9_])\s+([A-Za-z0-9_])", r"\1*\2", expr)

    parts = re.split(r"(?<!\^)[\+\-]", expr)  # rough: don't split inside powers
    dims = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        d = _dim_product_div(p)
        if d is None:
            return None
        dims.append(d)

    # All additive terms must match
    if not dims:
        return {}
    base = dims[0]
    for d in dims[1:]:
        if not dim_eq(base, d):
            return None
    return base


def _dim_product_div(expr: str) -> Dim | None:
    tokens = [t for t in TOKEN_RE.findall(expr)]
    # TOKEN_RE gives tuples for decimals; fix it:
    flat = []
    for t in tokens:
        if isinstance(t, tuple):
            continue
        flat.append(t)

    # Basic shunting-yard is overkill; we do left-to-right for * and /
    # Handle powers: x^2
    i = 0
    current: Dim = {}
    op = "*"

    def token_dim(tok: str) -> Dim | None:
        if _is_number(tok):
            return {}  # dimensionless scalar
        if tok in DIMLESS:
            return {}
        # Try exact and case variants
        if tok in SYMBOL_DIMS:
            return SYMBOL_DIMS[tok]
        if tok.upper() in SYMBOL_DIMS:
            return SYMBOL_DIMS[tok.upper()]
        if tok.lower() in SYMBOL_DIMS:
            return SYMBOL_DIMS[tok.lower()]
        return None

    while i < len(flat):
        tok = flat[i]
        if tok in ("*", "/"):
            op = tok
            i += 1
            continue
        if tok in ("(", ")"):
            # parentheses ignored in v1
            i += 1
            continue

        d = token_dim(tok)
        if d is None:
            return None

        # power?
        if i + 2 < len(flat) and flat[i + 1] == "^" and _is_int(flat[i + 2]):
            p = int(flat[i + 2])
            d = dim_pow(d, p)
            i += 3
        else:
            i += 1

        if op == "*":
            current = dim_mul(current, d)
        else:
            current = dim_sub(current, d)

    return current


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False
