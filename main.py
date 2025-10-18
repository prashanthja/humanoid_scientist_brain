"""
Humanoid Scientist Brain — Continuous Learning, Reasoning & Evolution
--------------------------------------------------------------------
Phase C Step 2–3:
- Hypothesis validation + evolution (reinforce/decay/retire/promotion)
- Automatic promotion of strong hypotheses into the KG
- Dashboard state updates (training, KG, hypotheses, evolution deltas)
"""

import time, json, os
from typing import Dict, Any, List

from data_pipeline.fetcher import DataFetcher
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from learning_module.trainer import Trainer
from embedding.encoder import TextEncoder

from reflection_module.reflection import ReflectionEngine
from reasoning_module.graph_reasoner import GraphReasoner
from reasoning_module.reasoning import GraphAugmentedReasoning
from reasoning_module.hypothesis_generator import HypothesisGenerator
from reasoning_module.hypothesis_validator import HypothesisValidator
from reasoning_module.hypothesis_evolver import HypothesisEvolver

# ------------------------------
# Dashboard helpers / constants
# ------------------------------
DASHBOARD_STATE_PATH = "data/dashboard_state.json"
HYP_STATE_PATH = "data/hypothesis_state.json"
os.makedirs("data", exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_dashboard_state(payload: Dict[str, Any]) -> None:
    write_json(DASHBOARD_STATE_PATH, payload)
    print("📊 Dashboard state updated → data/dashboard_state.json")


def _edge_count_fallback(kg) -> int:
    """Count edges from your KG’s internal dict structure if kg.edge_count() doesn’t exist."""
    g = getattr(kg, "graph", {})
    if not isinstance(g, dict):
        return 0
    total = 0
    for _, rels in g.items():
        if isinstance(rels, dict):
            for _, objs in rels.items():
                if isinstance(objs, list):
                    total += len(objs)
    return total


# ---------------
# Main program
# ---------------
def main():
    print("🤖 Starting Humanoid Scientist Brain (Continuous Learning Mode)...")

    # Core modules
    fetcher = DataFetcher()
    safety = SafetyFilter()
    kb = KnowledgeBase()
    encoder = TextEncoder()
    trainer = Trainer(encoder)
    reflection = ReflectionEngine(fetcher, safety, kb, trainer)

    # Reasoners
    hypgen = HypothesisGenerator(reflection.kg, encoder, kb)
    validator = HypothesisValidator(kb, encoder, reflection.kg)
    evolver = HypothesisEvolver(kb, reflection.kg)  # Phase C Step 3
    reasoning = GraphAugmentedReasoning(kb, encoder, reflection.kg, hypothesis_generator=hypgen)

    # Seed tokenizer (from KB if present)
    initial = kb.query("")
    seed_texts = [it["text"] if isinstance(it, dict) else str(it) for it in initial]
    encoder.fit_tokenizer(seed_texts or ["physics", "energy", "force", "science"])

    topic = "physics"
    cycle = 1

    try:
        while True:
            print(f"\n🌀 Learning Cycle {cycle} starting...")

            # 1) Fetch/store
            fresh = fetcher.fetch(topic)
            safe = safety.filter(fresh)
            kb.store(safe)

            # 2) Tokenizer refresh
            all_items = kb.query("")
            corpus = [it["text"] if isinstance(it, dict) else str(it) for it in all_items]
            if corpus:
                encoder.fit_tokenizer(corpus)

            # 3) Train on accumulated knowledge
            if all_items:
                trainer.run_training(all_items)

            # 4) Reflect / KG update
            print("\n🧠 [Reflection] Begin KG update and gap analysis...")
            reflection.review_knowledge()
            # keep reasoning in sync with the latest KG
            if hasattr(reasoning, "set_graph"):
                reasoning.set_graph(reflection.kg)

            # 5) Hypothesize → Validate → Evolve
            validated: List[Dict[str, Any]] = []
            hyps = hypgen.generate(top_n=20)
            if hyps:
                print("🧪 Generated hypotheses (top 5):")
                for h in hyps[:5]:
                    print(f"  • [{h.get('type','?')}] {h['hypothesis']} (gen_score={round(h.get('score',0.0), 3)})")

                validated = validator.validate(hyps, cycle=cycle)
                if validated:
                    print("\n🔍 Validated hypotheses (top 5):")
                    for v in validated[:5]:
                        mark = "✅" if v.get("promote") else "⚠️"
                        print(
                            f"  {mark} {v['hypothesis']} | "
                            f"support={v['support']}, cons={v['consistency']}, "
                            f"conf={v['confidence']}, persist={v['persistence']}"
                        )

            # Phase C Step 3: Evolution (reinforce / decay / retire)
            evo_summary = evolver.step(validated or [], cycle=cycle)
            # Persist evolutionary state (useful for restarts / dashboard)
            try:
                write_json(HYP_STATE_PATH, {"cycle": cycle, "state": evolver.state, "summary": evo_summary})
            except Exception:
                pass

            # Optional promotion of strong hypotheses to KG (robust against missing method)
            promote_list = [v for v in (validated or []) if v.get("promote")]
            if promote_list:
                if hasattr(reflection, "promote_validated"):
                    reflection.promote_validated(promote_list)
                else:
                    # Fallback: add as 'related_to' edges to KG
                    for v in promote_list:
                        hyp = v.get("hypothesis", "")
                        # naive parse: "A --rel--> B"
                        if "--" in hyp and "-->" in hyp:
                            try:
                                left, right = hyp.split("--", 1)
                                rel, dst = right.split("-->", 1)
                                a = left.strip()
                                relation = rel.strip()
                                b = dst.strip()
                                if hasattr(reflection.kg, "add_edge"):
                                    reflection.kg.add_edge(a, relation, b)
                            except Exception:
                                pass
                # Save KG after promotion to persist the change
                try:
                    if hasattr(reflection.kg, "save"):
                        reflection.kg.save("knowledge_graph/graph.json")
                except Exception:
                    pass

            # 6) Reasoning demos
            gr = GraphReasoner(reflection.kg)
            print("🔎 Graph reasoning demo:", gr.explain_relation("force", "motion"))
            print("🔎 Transitive (causes) from 'force':", gr.suggest_transitive("force", "causes"))

            ans = reasoning.answer("Explain Newton’s third law in simple words.")
            print("🧠 Reasoning Output:\n", ans)

            # 7) Dashboard: snapshot
            try:
                edge_count = getattr(reflection.kg, "edge_count", None)
                kg_edges = edge_count() if callable(edge_count) else _edge_count_fallback(reflection.kg)

                dash_payload = {
                    "cycle": cycle,
                    "kg": {
                        "nodes": len(getattr(reflection.kg, "graph", {})),
                        "edges": kg_edges,
                        "last_visual": getattr(getattr(reflection, "visualizer", None), "last_path", None),
                    },
                    "training": {
                        "last_avg_loss": getattr(trainer, "last_avg_loss", None),
                        "last_sim": getattr(trainer, "last_similarity", None),
                    },
                    "hypotheses": {
                        "active": len(getattr(evolver, "state", {})),
                        "promoted_total": sum(1 for r in evolver.state.values() if r.get("promoted")),
                        "retired_total": sum(1 for r in evolver.state.values() if r.get("retired")),
                        "evolution": evo_summary,  # counts: reinforced/decayed/retired/promoted
                    },
                }
                write_dashboard_state(dash_payload)
            except Exception:
                pass

            # 8) Plan next topic and loop
            print("\n🧭 [Planning] Deciding next topic...")
            reflection.plan_next_steps()

            print("⏳ Waiting before next learning cycle...\n")
            time.sleep(10)
            cycle += 1

    except KeyboardInterrupt:
        print("\n🧩 Learning loop interrupted by user.")
        try:
            trainer.visualizer.plot_progress()
        except Exception:
            pass
        try:
            reflection.progress.plot()
        except Exception:
            pass
        kb.close()


if __name__ == "__main__":
    main()
