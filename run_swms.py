#!/usr/bin/env python3
"""
run_swms.py
-----------
Runs the SWMS simulator and prints a formatted report.

Usage:
    python3 run_swms.py
    python3 run_swms.py --query "Does MoE improve transformer efficiency?"
"""

import sys
import os
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str,
                    default="How will transformer efficiency research evolve?")
    ap.add_argument("--n_sim", type=int, default=500)
    ap.add_argument("--n_steps", type=int, default=8)
    ap.add_argument("--use_kg", action="store_true", default=True)
    args = ap.parse_args()

    # Load KG if available
    kg = None
    if args.use_kg:
        try:
            from knowledge_graph.graph import KnowledgeGraph
            kg = KnowledgeGraph()
            kg.load()
            print(f"KG loaded: {kg.edge_count()} edges, {len(kg.all_concepts())} concepts")
        except Exception as e:
            print(f"KG not available ({e}), using domain model only")

    # Load hypotheses if available
    hypotheses = []
    try:
        from reasoning_module.hypothesis_generator import HypothesisGenerator
        gen = HypothesisGenerator(kg=kg)
        hypotheses = gen.generate(top_n=10)
        print(f"Hypotheses loaded: {len(hypotheses)}")
    except Exception as e:
        print(f"No hypotheses ({e})")

    # Run simulation
    from simulation_module.swms import SWMS
    swms = SWMS(kg=kg, n_simulations=args.n_sim, n_steps=args.n_steps)

    print(f"\nRunning {args.n_sim} simulations × {args.n_steps} steps...")
    report = swms.simulate(hypotheses=hypotheses, query=args.query)

    print(swms.format_report(report))
    print(f"\nReport saved to outputs/simulation_reports/")

if __name__ == "__main__":
    main()