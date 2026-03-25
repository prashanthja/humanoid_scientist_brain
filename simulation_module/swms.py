# simulation_module/swms.py
# ─────────────────────────────────────────────────────────────
# Scientific World Model Simulator — Orchestrator
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import json
import os
import time
from typing import Dict, Any, List, Optional

from .state_builder import StateBuilder
from .causal_engine import CausalEngine
from .rollout_simulator import RolloutSimulator
from .outcome_analyzer import OutcomeAnalyzer


# Query keywords → relevant KG nodes to focus on
QUERY_FOCUS_MAP = {
    "flashattention":        ["FlashAttention", "MemoryOverhead", "TransformerEfficiency"],
    "flash attention":       ["FlashAttention", "MemoryOverhead", "TransformerEfficiency"],
    "sparse attention":      ["SparseAttention", "ContextLength", "MemoryOverhead"],
    "mixture of experts":    ["MixtureOfExperts", "TransformerEfficiency", "Latency"],
    "moe":                   ["MixtureOfExperts", "TransformerEfficiency", "Latency"],
    "kv cache":              ["KVCache", "MemoryOverhead", "Latency", "ContextLength"],
    "lora":                  ["LoRA", "MemoryOverhead", "TransformerEfficiency"],
    "low-rank":              ["LoRA", "MemoryOverhead", "TransformerEfficiency"],
    "speculative decoding":  ["SpeculativeDecoding", "Throughput", "Latency"],
    "quantization":          ["MemoryOverhead", "ModelQuality", "TransformerEfficiency"],
    "context length":        ["ContextLength", "MemoryOverhead", "KVCache"],
    "long context":          ["ContextLength", "MemoryOverhead", "KVCache"],
    "mamba":                 ["Mamba", "TransformerEfficiency", "ContextLength"],
    "linear attention":      ["LinearAttention", "MemoryOverhead", "TransformerEfficiency"],
    "routing":               ["MixtureOfExperts", "Latency", "TransformerEfficiency"],
    "throughput":            ["Throughput", "SpeculativeDecoding", "KVCache"],
    "memory":                ["MemoryOverhead", "FlashAttention", "KVCache"],
    "latency":               ["Latency", "KVCache", "SpeculativeDecoding"],
}


def _extract_focus_nodes(query: str, known_nodes: List[str]) -> List[str]:
    """Extract which nodes to focus simulation on based on query keywords."""
    q = query.lower()
    focus = []
    for kw, nodes in QUERY_FOCUS_MAP.items():
        if kw in q:
            for n in nodes:
                if n in known_nodes and n not in focus:
                    focus.append(n)
    # Fallback: check if any known node name appears in query
    if not focus:
        for n in known_nodes:
            if n.lower() in q:
                focus.append(n)
    return focus[:5]


class SWMS:
    def __init__(
        self,
        kg=None,
        n_simulations: int = 500,
        n_steps: int = 8,
        seed: int = 42,
        out_dir: str = "outputs/simulation_reports",
    ):
        self.kg            = kg
        self.n_simulations = n_simulations
        self.n_steps       = n_steps
        self.seed          = seed
        self.out_dir       = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self._state_builder = StateBuilder(kg=kg)
        self._rollout       = RolloutSimulator(n_simulations, n_steps, seed)
        self._analyzer      = OutcomeAnalyzer()

    # ── Public API ────────────────────────────────────────────

    def simulate(
        self,
        hypotheses: Optional[List[Dict[str, Any]]] = None,
        query: str = "How will transformer efficiency research evolve?",
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Run field simulation focused on the given query.
        """
        state = self._state_builder.build(hypotheses=hypotheses)
        known_nodes = list(state["nodes"].keys())
        focus_nodes = _extract_focus_nodes(query, known_nodes)

        engine      = CausalEngine(state)
        transitions = engine.build_transitions()

        rollouts = self._rollout.run(
            initial_beliefs=state["beliefs"],
            transitions=transitions,
            causal_engine=engine,
            focus_nodes=focus_nodes if focus_nodes else None,
        )

        result = self._analyzer.analyze(rollouts, state, hypothesis_text=query)
        result["query"]        = query
        result["focus_nodes"]  = focus_nodes
        result["data_source"]  = "kg+domain" if self.kg else "domain"
        result["state_summary"] = {
            "nodes": len(state["nodes"]),
            "edges": len(state["edges"]),
            "transitions": len(transitions),
            "hypotheses_used": len(hypotheses or []),
        }

        if save:
            self._save(result, "field_simulation")

        return result

    def simulate_hypothesis(
        self,
        hypothesis_text: str,
        hypothesis_nodes: Optional[List[str]] = None,
        boost: float = 0.20,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Simulate: what happens if this hypothesis is true?
        Runs baseline vs hypothesis_true comparison.
        """
        state       = self._state_builder.build()
        known_nodes = list(state["nodes"].keys())
        engine      = CausalEngine(state)
        transitions = engine.build_transitions()

        if hypothesis_nodes is None:
            hypothesis_nodes = _extract_focus_nodes(hypothesis_text, known_nodes)
            if not hypothesis_nodes:
                hypothesis_nodes = ["FlashAttention", "MixtureOfExperts", "KVCache"]

        comparison = self._rollout.run_comparison(
            initial_beliefs=state["beliefs"],
            transitions=transitions,
            causal_engine=engine,
            focus_nodes=hypothesis_nodes,
            boost=boost,
        )

        result = self._analyzer.analyze_comparison(comparison, state, hypothesis_text)
        result["hypothesis_nodes"] = hypothesis_nodes
        result["boost_applied"]    = boost

        if save:
            self._save(result, "hypothesis_simulation")

        return result

    def format_report(self, result: Dict[str, Any]) -> str:
        """Format simulation result as readable terminal report."""
        lines = ["=" * 60,
                 "SIMULATION REPORT — Scientific World Model",
                 "=" * 60]

        # Handle comparison result
        if "hypothesis_impact" in result:
            impact  = result["hypothesis_impact"]
            r       = result.get("hypothesis_true", {})
            base    = result.get("baseline", {})
            lines.append(f"\nHypothesis: {impact.get('hypothesis','')}")
            lines.append(f"Impact    : {impact.get('impact','').upper()}")
            lines.append(f"{impact.get('interpretation','')}")
            lines.append(f"\nBaseline field score   : {base.get('avg_field_score',0):.3f}")
            lines.append(f"Hypothesis field score : {r.get('avg_field_score',0):.3f}")
            lines.append(f"Delta                  : {impact.get('delta_field_score',0):+.4f}")
        else:
            r = result
            lines.append(f"\nQuery     : {result.get('query','')}")
            lines.append(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Simulations: {r.get('n_simulations',0)} × {self.n_steps} steps")
            lines.append(f"Data source: {result.get('data_source','domain')}")
            if result.get('focus_nodes'):
                lines.append(f"Focus nodes: {', '.join(result['focus_nodes'])}")

        lines.append(f"\nField advancement score: {r.get('avg_field_score',0):.3f}")
        lines.append(f"{r.get('summary','')}")

        # Dominance
        lines.append("\n── Dominance Predictions ──────────────────────")
        for dm in r.get("dominant_methods", [])[:5]:
            p   = dm["probability"]
            bar = "█" * int(p * 20)
            lines.append(f"  {dm['method']:<28} {p*100:4.0f}%  {bar}")

        # Rising nodes
        lines.append("\n── Rising Methods ─────────────────────────────")
        rising = r.get("rising_nodes", [])
        for node, delta in (rising[:4] if isinstance(rising[0] if rising else None, tuple) else
                            [(x["node"], x["delta"]) for x in rising[:4]] if rising else []):
            lines.append(f"  ↑ {node} (+{delta:.3f})")

        # Contradiction risks
        lines.append("\n── Contradiction Emergence ─────────────────────")
        for risk in r.get("contradiction_risks", [])[:3]:
            lines.append(f"  ⚠ {risk}")

        # Roadmap
        lines.append("\n── Research Roadmap ────────────────────────────")
        for step in r.get("roadmap", []):
            yr    = step.get("year_offset", 0)
            label = step.get("label", "")
            score = step.get("avg_field_score", 0)
            lines.append(f"  ~{yr*12:2.0f} months [{score:.2f}]: {label}")

        # Experiments
        lines.append("\n── Best Next Experiments ───────────────────────")
        for i, exp in enumerate(r.get("best_experiments", [])[:5], 1):
            lines.append(f"  {i}. {exp}")

        # Hypothesis outcomes
        lines.append("\n── Hypothesis Outcomes ─────────────────────────")
        for h in r.get("hypothesis_outcomes", [])[:5]:
            p    = h.get("success_probability", 0)
            text = h.get("hypothesis", "")[:70]
            lines.append(f"  [{p*100:.0f}% success] {text}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _save(self, result: Dict[str, Any], prefix: str):
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.out_dir, f"{prefix}_{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SWMS] Save failed: {e}")