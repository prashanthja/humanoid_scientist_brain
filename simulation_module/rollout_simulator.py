# simulation_module/rollout_simulator.py
# ─────────────────────────────────────────────────────────────
# Monte Carlo rollout engine.
# N simulations × T timesteps each.
# Each timestep ≈ 6 months of field evolution.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import random
import copy
from typing import Dict, Any, List, Optional


class RolloutSimulator:
    def __init__(
        self,
        n_simulations: int = 500,
        n_steps: int = 8,
        seed: int = 42,
    ):
        self.n_simulations = int(n_simulations)
        self.n_steps       = int(n_steps)
        self.seed          = int(seed)

    def run(
        self,
        initial_beliefs: Dict[str, float],
        transitions: List[Dict[str, Any]],
        causal_engine,
        focus_nodes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run all simulations.
        If focus_nodes provided, uses query-focused step logic.
        """
        rng = random.Random(self.seed)
        rollouts = []

        for sim_id in range(self.n_simulations):
            # Add per-simulation noise to initial beliefs (±5%)
            beliefs = {
                k: max(0.0, min(1.0, v + rng.gauss(0, 0.05)))
                for k, v in initial_beliefs.items()
            }
            trajectory = [copy.deepcopy(beliefs)]

            for step in range(self.n_steps):
                if focus_nodes:
                    beliefs = causal_engine.apply_step_with_focus(
                        beliefs, transitions, rng, focus_nodes
                    )
                else:
                    beliefs = causal_engine.apply_step(
                        beliefs, transitions, rng
                    )
                trajectory.append(copy.deepcopy(beliefs))

            field_score = causal_engine.compute_field_score(beliefs)

            rollouts.append({
                "sim_id":        sim_id,
                "trajectory":    trajectory,
                "final_beliefs": copy.deepcopy(beliefs),
                "field_score":   field_score,
            })

        return rollouts

    def run_comparison(
        self,
        initial_beliefs: Dict[str, float],
        transitions: List[Dict[str, Any]],
        causal_engine,
        focus_nodes: List[str],
        boost: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Run baseline vs hypothesis_true comparison.
        Hypothesis_true: focus_nodes get a belief boost at step 0.
        """
        # Baseline
        baseline = self.run(initial_beliefs, transitions, causal_engine)

        # Hypothesis true: boost focus nodes
        boosted_beliefs = {
            k: min(1.0, v + boost) if k in focus_nodes else v
            for k, v in initial_beliefs.items()
        }
        hyp_true = self.run(
            boosted_beliefs, transitions, causal_engine,
            focus_nodes=focus_nodes,
        )

        return {"baseline": baseline, "hypothesis_true": hyp_true}