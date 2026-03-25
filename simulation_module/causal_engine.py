# simulation_module/causal_engine.py
# ─────────────────────────────────────────────────────────────
# Causal/probabilistic transition rules from graph edges.
#
# Key fix: heavily dampened propagation so beliefs don't
# saturate to 1.0 within 2-3 steps.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import random
from typing import Dict, Any, List, Tuple

# How each relation type affects the target
# (direction, max_strength)
RELATION_EFFECTS: Dict[str, Tuple[str, float]] = {
    "supports_efficiency": ("positive", 0.55),
    "improves":            ("positive", 0.50),
    "reduces":             ("negative", 0.48),   # reduces a cost → beneficial
    "partially_supports":  ("positive", 0.30),
    "related_to":          ("neutral",  0.15),
    "has_tradeoff":        ("negative", 0.35),
    "contradicts":         ("negative", 0.45),
    "causes":              ("positive", 0.45),
    "leads_to":            ("positive", 0.42),
    "increases":           ("positive", 0.35),
    "decreases":           ("negative", 0.35),
    "associated_with":     ("neutral",  0.12),
}

COST_NODES    = {"MemoryOverhead", "Latency", "RoutingInstability", "GeneralLimitation"}
BENEFIT_NODES = {
    "TransformerEfficiency", "ModelQuality", "Throughput",
    "InferenceSpeed", "ParameterEfficiency", "Scalability",
    "ContextLength", "AdoptionMomentum",
}


class CausalEngine:
    def __init__(self, state: Dict[str, Any]):
        self.edges       = state.get("edges", [])
        self.nodes       = state.get("nodes", {})
        self.beliefs     = dict(state.get("beliefs", {}))
        # Per-node decay: prevents unbounded accumulation
        self.decay       = 0.96   # beliefs decay slightly each step toward 0.5
        self.step_scale  = 0.04   # max delta per step per edge (was implicitly ~0.12)

    def build_transitions(self) -> List[Dict[str, Any]]:
        transitions = []
        for edge in self.edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("relation", "")
            weight = float(edge.get("weight", 0.35))

            if not src or not tgt:
                continue

            effect_dir, max_strength = RELATION_EFFECTS.get(rel, ("neutral", 0.15))

            # Semantic adjustments
            if tgt in COST_NODES and rel in ("reduces", "decreases"):
                effect_dir = "positive"
            if tgt in BENEFIT_NODES and rel in ("improves", "supports_efficiency", "leads_to"):
                effect_dir = "positive"

            # Dampen: strength is weight × max_strength × step_scale
            strength = weight * max_strength * self.step_scale

            transitions.append({
                "source":   src,
                "target":   tgt,
                "relation": rel,
                "effect":   effect_dir,
                "strength": round(strength, 5),
                "noise":    0.005,   # very small noise (was 0.08 × 0.05 = 0.004, but now explicit)
            })
        return transitions

    def apply_step(
        self,
        beliefs: Dict[str, float],
        transitions: List[Dict[str, Any]],
        rng: random.Random,
    ) -> Dict[str, float]:
        new_beliefs = dict(beliefs)
        deltas: Dict[str, float] = {}

        for t in transitions:
            src = t["source"]
            tgt = t["target"]
            if src not in beliefs or tgt not in beliefs:
                continue

            src_b    = beliefs[src]
            strength = t["strength"]
            effect   = t["effect"]
            noise    = t["noise"]

            # Influence proportional to how far source belief is from neutral (0.5)
            # This prevents saturated nodes from over-driving their neighbors
            src_activation = abs(src_b - 0.5) * 2.0  # 0 at neutral, 1 at extremes

            if effect == "positive":
                delta = src_activation * strength
            elif effect == "negative":
                delta = -src_activation * strength
            else:
                delta = 0.0

            delta += rng.gauss(0, noise)
            deltas[tgt] = deltas.get(tgt, 0.0) + delta

        # Apply deltas with mean-reversion decay
        for node in new_beliefs:
            d = deltas.get(node, 0.0)
            # Mean reversion: nudge slightly toward 0.5 each step
            reversion = (0.5 - new_beliefs[node]) * (1.0 - self.decay)
            new_beliefs[node] = max(0.0, min(1.0, new_beliefs[node] + d + reversion))

        return new_beliefs

    def apply_step_with_focus(
        self,
        beliefs: Dict[str, float],
        transitions: List[Dict[str, Any]],
        rng: random.Random,
        focus_nodes: List[str],
        focus_boost: float = 0.15,
    ) -> Dict[str, float]:
        """
        Like apply_step but amplifies transitions involving focus_nodes.
        Used for query-focused simulation.
        """
        focused = set(focus_nodes)
        new_beliefs = dict(beliefs)
        deltas: Dict[str, float] = {}

        for t in transitions:
            src = t["source"]
            tgt = t["target"]
            if src not in beliefs or tgt not in beliefs:
                continue

            src_b    = beliefs[src]
            strength = t["strength"]
            effect   = t["effect"]
            noise    = t["noise"]

            # Boost if this edge involves a focus node
            if src in focused or tgt in focused:
                strength = min(strength * 2.5, 0.12)

            src_activation = abs(src_b - 0.5) * 2.0

            if effect == "positive":
                delta = src_activation * strength
            elif effect == "negative":
                delta = -src_activation * strength
            else:
                delta = 0.0

            delta += rng.gauss(0, noise)
            deltas[tgt] = deltas.get(tgt, 0.0) + delta

        for node in new_beliefs:
            d = deltas.get(node, 0.0)
            reversion = (0.5 - new_beliefs[node]) * (1.0 - self.decay)
            new_beliefs[node] = max(0.0, min(1.0, new_beliefs[node] + d + reversion))

        return new_beliefs

    def compute_field_score(self, beliefs: Dict[str, float]) -> float:
        b_vals = [beliefs[n] for n in BENEFIT_NODES if n in beliefs]
        c_vals = [beliefs[n] for n in COST_NODES    if n in beliefs]
        benefit = sum(b_vals) / len(b_vals) if b_vals else 0.5
        cost    = sum(c_vals) / len(c_vals) if c_vals else 0.5
        return round(max(0.0, min(1.0, benefit - cost * 0.4 + 0.3)), 4)