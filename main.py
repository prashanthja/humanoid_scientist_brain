"""
Humanoid Scientist Brain — Continuous Learning & Graph Reasoning
---------------------------------------------------------------
Phase B Step 3–4:
- Continuous self-learning
- KG construction + reflection
- Graph-driven knowledge expansion
- Textual + Graph reasoning integration
- KG growth visualization on exit
"""

import time
from data_pipeline.fetcher import DataFetcher
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from learning_module.trainer import Trainer
from reasoning_module.reasoning import GraphAugmentedReasoning
from reasoning_module.graph_reasoner import GraphReasoner
from embedding.encoder import TextEncoder
from reflection_module.reflection import ReflectionEngine


def main():
    print("🤖 Starting Humanoid Scientist Brain (Continuous Learning Mode)...")

    # --- Initialize Core Modules ---
    fetcher = DataFetcher()
    safety = SafetyFilter()
    kb = KnowledgeBase()
    encoder = TextEncoder()
    trainer = Trainer(encoder)
    reflection = ReflectionEngine(fetcher, safety, kb, trainer)
    reasoning = GraphAugmentedReasoning(kb, encoder, reflection.kg)

    topic = "physics"

    # --- Initialize Tokenizer ---
    initial_knowledge = kb.query("")
    texts = [item["text"] if isinstance(item, dict) else str(item) for item in initial_knowledge]
    encoder.fit_tokenizer(texts or ["physics", "energy", "force", "science"])

    cycle = 1
    try:
        while True:
            print(f"\n🌀 Learning Cycle {cycle} starting...")

            # 1️⃣ Fetch and store new knowledge
            raw_data = fetcher.fetch(topic)
            safe_data = safety.filter(raw_data)
            kb.store(safe_data)

            # 2️⃣ Update tokenizer dynamically
            all_knowledge = kb.query("")
            new_texts = [item["text"] if isinstance(item, dict) else str(item) for item in all_knowledge]
            if new_texts:
                encoder.fit_tokenizer(new_texts)

            # 3️⃣ Train AI model on accumulated knowledge
            if all_knowledge:
                trainer.run_training(all_knowledge)

            # 4️⃣ Reflection and KG update
            print("\n🧠 [Reflection] Begin KG update and gap analysis...")
            reflection.review_knowledge()
            reasoning.set_graph(reflection.kg)

            # 5️⃣ Plan next learning goal
            print("\n🧭 [Planning] Deciding next topic...")
            reflection.plan_next_steps()

            # 6️⃣ Graph Reasoning Demonstration
            gr = GraphReasoner(reflection.kg)
            print("🔎 Graph reasoning demo:", gr.explain_relation("force", "motion"))
            print("🔎 Transitive (causes) from 'force':", gr.suggest_transitive("force", "causes"))

            # 7️⃣ Textual + Graph Reasoning Integration
            ans = reasoning.answer("Explain Newton’s third law in simple words.")
            print("🧠 Reasoning Output:\n", ans)

            # 8️⃣ Pause before next iteration
            print("⏳ Waiting before next learning cycle...\n")
            time.sleep(10)
            cycle += 1

    except KeyboardInterrupt:
        print("\n🧩 Learning loop interrupted by user.")
        print("📊 Displaying training progress (trainer)...")
        try:
            trainer.visualizer.plot_progress()
        except Exception:
            print("⚠️ Could not plot training progress.")

        print("📈 Displaying Knowledge Graph growth...")
        try:
            reflection.progress.plot()
        except Exception as e:
            print(f"⚠️ Could not plot KG growth: {e}")

        kb.close()


if __name__ == "__main__":
    main()
