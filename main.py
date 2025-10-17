"""
Humanoid Scientist Brain — Continuous Learning & Graph Reasoning
---------------------------------------------------------------
Phase C Step 2 (added):
- Hypothesis validation (KB support + semantic consistency + persistence)
- Optional promotion of strong hypotheses back into the KG
"""

import time
from data_pipeline.fetcher import DataFetcher
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from learning_module.trainer import Trainer
from reasoning_module.reasoning import GraphAugmentedReasoning
from reasoning_module.graph_reasoner import GraphReasoner
from reasoning_module.hypothesis_generator import HypothesisGenerator
from reasoning_module.hypothesis_validator import HypothesisValidator
from embedding.encoder import TextEncoder
from reflection_module.reflection import ReflectionEngine


def main():
    print("🤖 Starting Humanoid Scientist Brain (Continuous Learning Mode)...")

    # Core plumbling
    fetcher = DataFetcher()
    safety = SafetyFilter()
    kb = KnowledgeBase()
    encoder = TextEncoder()
    trainer = Trainer(encoder)
    reflection = ReflectionEngine(fetcher, safety, kb, trainer)

    # Reasoners
    hypgen = HypothesisGenerator(reflection.kg, encoder, kb)
    validator = HypothesisValidator(kb, encoder, reflection.kg)  # NEW
    reasoning = GraphAugmentedReasoning(kb, encoder, reflection.kg, hypothesis_generator=hypgen)

    # Initial tokenizer seed
    initial_knowledge = kb.query("")
    texts = [item["text"] if isinstance(item, dict) else str(item) for item in initial_knowledge]
    encoder.fit_tokenizer(texts or ["physics", "energy", "force", "science"])

    topic = "physics"
    cycle = 1

    try:
        while True:
            print(f"\n🌀 Learning Cycle {cycle} starting...")

            # 1) Pull & store
            raw = fetcher.fetch(topic)
            safe = safety.filter(raw)
            kb.store(safe)

            # 2) Tokenizer refresh
            all_items = kb.query("")
            all_texts = [it["text"] if isinstance(it, dict) else str(it) for it in all_items]
            if all_texts:
                encoder.fit_tokenizer(all_texts)

            # 3) Train model
            if all_items:
                trainer.run_training(all_items)

            # 4) Reflect & update KG
            print("\n🧠 [Reflection] Begin KG update and gap analysis...")
            reflection.review_knowledge()
            reasoning.set_graph(reflection.kg)  # keep in sync

            # 5) Hypothesize + Validate + Promote
            hyps = hypgen.generate(top_n=20)
            if hyps:
                print("🧪 Generated hypotheses (top 5):")
                for h in hyps[:5]:
                    print(f"  • [{h['type']}] {h['hypothesis']} (gen_score={round(h.get('score', 0.0), 3)})")

                validated = validator.validate(hyps, cycle=cycle)
                if validated:
                    print("\n🔍 Validated hypotheses (top 5):")
                    for v in validated[:5]:
                        mark = "✅" if v["promote"] else "⚠️"
                        print(
                            f"  {mark} {v['hypothesis']} | "
                            f"support={v['support']}, cons={v['consistency']}, "
                            f"conf={v['confidence']}, persist={v['persistence']}"
                        )

                    # Optional promotion into KG
                    reflection.promote_validated(validated)

            # 6) Reasoning demos
            gr = GraphReasoner(reflection.kg)
            print("🔎 Graph reasoning demo:", gr.explain_relation("force", "motion"))
            print("🔎 Transitive (causes) from 'force':", gr.suggest_transitive("force", "causes"))

            ans = reasoning.answer("Explain Newton’s third law in simple words.")
            print("🧠 Reasoning Output:\n", ans)

            # Loop wait
            print("⏳ Waiting before next learning cycle...\n")
            time.sleep(10)
            cycle += 1

    except KeyboardInterrupt:
        print("\n🧩 Learning loop interrupted by user.")
        print("📊 Displaying training progress (trainer)...")
        try:
            trainer.visualizer.plot_progress()
        except Exception:
            pass
        print("📈 Displaying KG growth...")
        try:
            reflection.progress.plot()
        except Exception:
            pass
        kb.close()


if __name__ == "__main__":
    main()
