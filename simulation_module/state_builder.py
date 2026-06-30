# simulation_module/state_builder.py
# ------------------------------------------------------------
# Converts KnowledgeGraph + domain knowledge into a
# simulation-ready state dict.
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any, List, Optional
import time


# ─────────────────────────────────────────────
# Hardcoded domain model — transformer efficiency
# Used as fallback when KG is sparse, and to
# seed initial beliefs for known methods.
# ─────────────────────────────────────────────

DOMAIN_METHODS = {
    "FlashAttention": {
        "compute_efficiency": 0.82,
        "memory_reduction":   0.78,
        "quality_retention":  0.95,
        "adoption_momentum":  0.71,
        "implementation_complexity": 0.35,
        "inference_speed":    0.76,
    },
    "SparseAttention": {
        "compute_efficiency": 0.68,
        "memory_reduction":   0.62,
        "quality_retention":  0.74,
        "adoption_momentum":  0.48,
        "implementation_complexity": 0.55,
        "inference_speed":    0.61,
    },
    "MixtureOfExperts": {
        "compute_efficiency": 0.71,
        "memory_reduction":   0.45,
        "quality_retention":  0.82,
        "adoption_momentum":  0.65,
        "implementation_complexity": 0.72,
        "routing_instability": 0.58,
        "inference_speed":    0.55,
    },
    "KVCache": {
        "compute_efficiency": 0.74,
        "memory_reduction":   0.69,
        "quality_retention":  0.91,
        "adoption_momentum":  0.83,
        "implementation_complexity": 0.28,
        "inference_speed":    0.79,
    },
    "LoRA": {
        "compute_efficiency": 0.65,
        "memory_reduction":   0.72,
        "quality_retention":  0.86,
        "adoption_momentum":  0.88,
        "implementation_complexity": 0.22,
        "inference_speed":    0.58,
    },
    "SpeculativeDecoding": {
        "compute_efficiency": 0.69,
        "memory_reduction":   0.30,
        "quality_retention":  0.97,
        "adoption_momentum":  0.55,
        "implementation_complexity": 0.61,
        "inference_speed":    0.81,
    },
    "Quantization": {
        "compute_efficiency": 0.76,
        "memory_reduction":   0.80,
        "quality_retention":  0.78,
        "adoption_momentum":  0.72,
        "implementation_complexity": 0.40,
        "inference_speed":    0.74,
    },
    "LinearAttention": {
        "compute_efficiency": 0.72,
        "memory_reduction":   0.58,
        "quality_retention":  0.67,
        "adoption_momentum":  0.35,
        "implementation_complexity": 0.65,
        "inference_speed":    0.70,
    },
    "Mamba": {
        "compute_efficiency": 0.77,
        "memory_reduction":   0.65,
        "quality_retention":  0.79,
        "adoption_momentum":  0.42,
        "implementation_complexity": 0.58,
        "inference_speed":    0.72,
    },
}

# Causal edges from domain knowledge
# (source_method, relation, target_metric, strength)
DOMAIN_CAUSAL_EDGES = [
    ("FlashAttention",      "reduces",             "MemoryOverhead",        0.82),
    ("FlashAttention",      "improves",            "InferenceSpeed",        0.76),
    ("SparseAttention",     "reduces",             "ComputeCost",           0.68),
    ("SparseAttention",     "has_tradeoff",        "ModelQuality",          0.42),
    ("MixtureOfExperts",    "improves",            "ParameterEfficiency",   0.74),
    ("MixtureOfExperts",    "has_tradeoff",        "RoutingInstability",    0.58),
    ("MixtureOfExperts",    "improves",            "TransformerEfficiency", 0.65),
    ("KVCache",             "reduces",             "MemoryOverhead",        0.69),
    ("KVCache",             "improves",            "InferenceSpeed",        0.79),
    ("LoRA",                "reduces",             "TrainingCost",          0.72),
    ("LoRA",                "improves",            "ParameterEfficiency",   0.80),
    ("SpeculativeDecoding", "improves",            "InferenceSpeed",        0.81),
    ("Quantization",        "reduces",             "MemoryOverhead",        0.80),
    ("Quantization",        "has_tradeoff",        "ModelQuality",          0.28),
    ("LinearAttention",     "reduces",             "ComputeCost",           0.72),
    ("LinearAttention",     "has_tradeoff",        "ModelQuality",          0.35),
    ("Mamba",               "reduces",             "ComputeCost",           0.77),
    ("Mamba",               "improves",            "ContextLength",         0.68),
    ("RoutingInstability",  "reduces",             "TransformerEfficiency", 0.55),
    ("MemoryOverhead",      "limits",              "ContextLength",         0.71),
    ("ContextLength",       "requires",            "MemoryOverhead",        0.65),
]


class StateBuilder:
    """
    Converts KG + domain model into simulation state.

    Output format:
    {
      "nodes": { node_id: { state_vars } },
      "edges": [ { source, relation, target, weight } ],
      "beliefs": { node_id: confidence_float },
      "hypotheses": [ hypothesis_dicts ],
      "domain": "transformer_efficiency",
      "timestamp": "...",
      "source": "kg+domain" | "domain_only",
    }
    """

    def __init__(self, kg=None):
        self.kg = kg

    def build(
        self,
        hypotheses: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        beliefs: Dict[str, float] = {}
        source = "domain_only"

        # ── Seed from domain model ──────────────────
        for method, vars_ in DOMAIN_METHODS.items():
            nodes[method] = dict(vars_)
            beliefs[method] = vars_.get("adoption_momentum", 0.5)

        for s, rel, t, w in DOMAIN_CAUSAL_EDGES:
            edges.append({"source": s, "relation": rel, "target": t, "weight": float(w)})
            if t not in nodes:
                nodes[t] = {"value": 0.5}
                beliefs[t] = 0.5

        # ── Enrich from KG if available ─────────────
        if self.kg is not None:
            try:
                kg_edges = self._extract_kg_edges()
                if kg_edges:
                    source = "kg+domain"
                    for e in kg_edges:
                        # Add/update node beliefs from KG evidence
                        s, rel, t, w = e["source"], e["relation"], e["target"], e["weight"]

                        if s not in nodes:
                            nodes[s] = {"value": 0.5}
                        if t not in nodes:
                            nodes[t] = {"value": 0.5}

                        # Boost belief for nodes with KG evidence
                        beliefs[s] = min(1.0, beliefs.get(s, 0.5) + 0.05)
                        beliefs[t] = min(1.0, beliefs.get(t, 0.5) + 0.03)

                        # Add KG edge if not already present
                        existing = {(e2["source"], e2["relation"], e2["target"]) for e2 in edges}
                        if (s, rel, t) not in existing:
                            edges.append({"source": s, "relation": rel, "target": t, "weight": float(w)})
            except Exception:
                pass

        return {
            "nodes": nodes,
            "edges": edges,
            "beliefs": beliefs,
            "hypotheses": hypotheses or [],
            "domain": "transformer_efficiency",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
        }

    def _extract_kg_edges(self) -> List[Dict[str, Any]]:
        out = []
        rel_weights = {
            "supports_efficiency": 0.75,
            "reduces":             0.70,
            "improves":            0.68,
            "partially_supports":  0.50,
            "has_tradeoff":        0.45,
            "related_to":          0.35,
            "contradicts":         0.30,
        }
        try:
            concepts = self.kg.all_concepts()
            for concept in concepts:
                rels = self.kg.get_relations(concept)
                if not isinstance(rels, dict):
                    continue
                for rel, targets in rels.items():
                    w = rel_weights.get(rel, 0.30)
                    if not isinstance(targets, list):
                        targets = [targets]
                    for t in targets:
                        if t and str(t).strip():
                            out.append({
                                "source": concept,
                                "relation": rel,
                                "target": str(t).strip(),
                                "weight": w,
                            })
        except Exception:
            pass
        return out