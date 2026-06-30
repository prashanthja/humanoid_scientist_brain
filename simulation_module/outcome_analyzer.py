# simulation_module/outcome_analyzer.py
# ─────────────────────────────────────────────────────────────
# Aggregates Monte Carlo rollout results into predictions.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import statistics
from typing import Dict, Any, List

HIGH_BELIEF   = 0.72
MEDIUM_BELIEF = 0.58
LOW_BELIEF    = 0.42


class OutcomeAnalyzer:

    def analyze(
        self,
        rollouts: List[Dict[str, Any]],
        state: Dict[str, Any],
        hypothesis_text: str = "",
    ) -> Dict[str, Any]:
        if not rollouts:
            return self._empty()

        n         = len(rollouts)
        nodes     = list(rollouts[0]["final_beliefs"].keys())
        n_steps   = len(rollouts[0]["trajectory"])

        # ── Collect final belief distributions ───────────────
        final_dist: Dict[str, List[float]] = {nd: [] for nd in nodes}
        init_dist:  Dict[str, List[float]] = {nd: [] for nd in nodes}
        field_scores: List[float] = []

        for r in rollouts:
            fb = r["final_beliefs"]
            ib = r["trajectory"][0]
            for nd in nodes:
                if nd in fb: final_dist[nd].append(fb[nd])
                if nd in ib: init_dist[nd].append(ib[nd])
            field_scores.append(r["field_score"])

        mean_final = {nd: statistics.mean(v) for nd, v in final_dist.items() if v}
        mean_init  = {nd: statistics.mean(v) for nd, v in init_dist.items()  if v}

        # ── Dominance: P(belief > HIGH at end) ───────────────
        dominance: Dict[str, float] = {}
        for nd, vals in final_dist.items():
            if vals:
                dominance[nd] = sum(1 for v in vals if v > HIGH_BELIEF) / len(vals)

        # ── Rising nodes: largest positive delta ─────────────
        deltas = []
        for nd in nodes:
            if nd in mean_final and nd in mean_init:
                d = mean_final[nd] - mean_init[nd]
                deltas.append({"node": nd, "delta": round(d, 4)})
        deltas.sort(key=lambda x: x["delta"], reverse=True)
        rising  = [x for x in deltas if x["delta"] > 0.005][:6]
        falling = [x for x in deltas if x["delta"] < -0.005][:4]

        # ── Node types ────────────────────────────────────────
        node_types = {
            nid: meta.get("type", "")
            for nid, meta in state.get("nodes", {}).items()
        }
        KNOWN_METHODS = {
            "FlashAttention", "SparseAttention", "MixtureOfExperts", "KVCache",
            "LoRA", "SpeculativeDecoding", "LinearAttention", "Mamba", "RWKV",
            "Quantization", "GroupedQueryAttention", "RoPE",
        }
        method_nodes = {
            n for n, t in node_types.items()
            if t in ("method", "architecture", "optimization")
            or n in KNOWN_METHODS
        }

        dominant_methods = sorted(
            [{"method": n, "probability": round(dominance.get(n, 0), 3)}
             for n in method_nodes if n in dominance],
            key=lambda x: x["probability"], reverse=True
        )[:5]

        dead_ends = [
            n for n in method_nodes
            if mean_final.get(n, 0.5) < LOW_BELIEF
        ]

        # ── Field trajectory ──────────────────────────────────
        trajectory = []
        for step in range(n_steps):
            step_scores = []
            for r in rollouts[:200]:   # sample for speed
                if step < len(r["trajectory"]):
                    b = r["trajectory"][step]
                    from .causal_engine import BENEFIT_NODES, COST_NODES
                    bv = [b[nd] for nd in BENEFIT_NODES if nd in b]
                    cv = [b[nd] for nd in COST_NODES    if nd in b]
                    sc = (sum(bv)/len(bv) if bv else 0.5) - (sum(cv)/len(cv) if cv else 0.5)*0.4 + 0.3
                    step_scores.append(max(0.0, min(1.0, sc)))
            trajectory.append({
                "step": step,
                "year_offset": round(step * 0.5, 1),
                "avg_field_score": round(statistics.mean(step_scores) if step_scores else 0.5, 4),
            })

        # ── Contradiction risks ───────────────────────────────
        risks = []
        for edge in state.get("edges", []):
            if edge.get("relation") in ("has_tradeoff", "contradicts"):
                src = edge["source"]
                tgt = edge["target"]
                sb  = mean_final.get(src, 0.5)
                if sb > MEDIUM_BELIEF:
                    risks.append(
                        f"{src} (belief={sb:.2f}) faces {edge['relation'].replace('_',' ')} "
                        f"with {tgt}"
                    )
        risks = risks[:4]

        # ── Hypothesis outcomes ───────────────────────────────
        hyp_outcomes = []
        for h in state.get("hypotheses", []):
            text  = h.get("text", "") or h.get("hypothesis", "")
            score = float(h.get("score", 0.5))
            # Success probability: blend hypothesis score with field score
            avg_fs = statistics.mean(field_scores)
            p_success = round(min(0.92, max(0.10, score * 0.6 + avg_fs * 0.4)), 3)
            hyp_outcomes.append({
                "hypothesis":          text,
                "success_probability": p_success,
                "type":                h.get("type", ""),
            })
        hyp_outcomes.sort(key=lambda x: x["success_probability"], reverse=True)

        # ── Experiments ───────────────────────────────────────
        experiments = self._suggest_experiments(
            mean_final, state, dominant_methods, dead_ends
        )

        # ── Roadmap ───────────────────────────────────────────
        roadmap = self._build_roadmap(trajectory, dominant_methods, risks)

        avg_fs  = statistics.mean(field_scores)
        summary = (
            f"Across {n} simulations, the field advances to score {avg_fs:.2f}. "
            f"{'Strong momentum detected.' if avg_fs > 0.65 else 'Moderate progress with significant uncertainty.'} "
            f"Top method: {dominant_methods[0]['method'] if dominant_methods else 'unknown'}."
        )

        return {
            "n_simulations":              n,
            "avg_field_score":            round(avg_fs, 4),
            "summary":                    summary,
            "dominance_probabilities":    {k: round(v, 3) for k, v in sorted(dominance.items(), key=lambda x:-x[1])[:8]},
            "mean_final_beliefs":         {k: round(v, 3) for k, v in sorted(mean_final.items(), key=lambda x:-x[1])[:8]},
            "dominant_methods":           dominant_methods,
            "rising_nodes":               rising,
            "falling_nodes":              falling,
            "dead_ends":                  dead_ends[:3],
            "contradiction_risks":        risks,
            "hypothesis_outcomes":        hyp_outcomes[:6],
            "field_trajectory":           trajectory,
            "roadmap":                    roadmap,
            "best_experiments":           experiments,
            "hypothesis":                 hypothesis_text,
        }

    def analyze_comparison(
        self,
        comparison: Dict[str, Any],
        state: Dict[str, Any],
        hypothesis_text: str = "",
    ) -> Dict[str, Any]:
        base_result = self.analyze(comparison["baseline"],       state, "baseline")
        hyp_result  = self.analyze(comparison["hypothesis_true"], state, hypothesis_text)

        delta  = round(hyp_result["avg_field_score"] - base_result["avg_field_score"], 4)
        impact = "positive" if delta > 0.01 else "negative" if delta < -0.01 else "neutral"

        return {
            "baseline":        base_result,
            "hypothesis_true": hyp_result,
            "hypothesis_impact": {
                "hypothesis":        hypothesis_text,
                "delta_field_score": delta,
                "impact":            impact,
                "interpretation": (
                    f"If '{hypothesis_text}' is true, the field advances "
                    f"{'faster' if impact=='positive' else 'slower' if impact=='negative' else 'at a similar rate'} "
                    f"(Δ={delta:+.4f})"
                ),
            },
        }

    def _suggest_experiments(self, mean_final, state, dominant, dead_ends):
        experiments = []
        if dominant:
            m = dominant[0]["method"]
            p = dominant[0]["probability"]
            experiments.append(
                f"Benchmark {m} on real inference cost (wall-clock latency, not just FLOPs) "
                f"— dominance probability {p*100:.0f}%"
            )
        if len(dominant) > 1:
            m2 = dominant[1]["method"]
            p2 = dominant[1]["probability"]
            experiments.append(
                f"Benchmark {m2} on real inference cost (wall-clock latency, not just FLOPs) "
                f"— dominance probability {p2*100:.0f}%"
            )
        for edge in state.get("edges", []):
            if edge.get("relation") == "has_tradeoff":
                src = edge["source"]
                sb  = mean_final.get(src, 0.5)
                if sb > 0.55:
                    experiments.append(
                        f"Investigate failure modes of {src} under production load "
                        f"— failure probability {(1-sb)*100:.0f}%"
                    )
                    break
        experiments.append(
            "Compare MoE vs dense on real inference cost, not just parameter count "
            "— routing instability remains a key bottleneck"
        )
        experiments.append(
            "Evaluate sparse attention methods at 128k+ context on reasoning-heavy tasks"
        )
        return experiments[:5]

    def _build_roadmap(self, trajectory, dominant, risks):
        roadmap = []
        for step in trajectory:
            score = step["avg_field_score"]
            yr    = step["year_offset"]
            if yr == 0:
                label = "Current state"
            elif score > 0.68:
                label = "Strong efficiency gains broadly adopted"
            elif score > 0.60:
                label = "Key methods mature, integration challenges emerge"
            elif score > 0.52:
                label = "Early adoption, research competition intensifies"
            else:
                label = "Fragmented landscape, no clear dominant method"
            roadmap.append({
                "year_offset":     yr,
                "label":           label,
                "avg_field_score": score,
            })
        return roadmap

    def _empty(self):
        return {
            "n_simulations": 0, "avg_field_score": 0.0, "summary": "No data.",
            "dominance_probabilities": {}, "mean_final_beliefs": {},
            "dominant_methods": [], "rising_nodes": [], "falling_nodes": [],
            "dead_ends": [], "contradiction_risks": [], "hypothesis_outcomes": [],
            "field_trajectory": [], "roadmap": [], "best_experiments": [],
            "hypothesis": "",
        }