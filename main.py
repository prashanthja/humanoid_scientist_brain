"""
Humanoid Scientist Brain — Continuous Learning & Graph Reasoning
---------------------------------------------------------------
Phase B Step 3–4:
- Continuous self-learning
- Knowledge Graph construction and reflection
- Graph-driven knowledge expansion (auto-filling conceptual gaps)
- Textual + Graph reasoning integration
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

    # --- Core modules ---
    fetcher = DataFetcher()
    safety = SafetyFilter()
    kb = KnowledgeBase()
    encoder = TextEncoder()
    trainer = Trainer(encoder)
    reasoning = GraphAugmentedReasoning(kb, encoder)
    reflection = ReflectionEngine(fetcher, safety, kb, trainer)

    # --- Initial topic ---
    topic = "physics"

    # --- Initialize tokenizer ---
    initial_knowledge = kb.query("")
    texts = [item["text"] if isinstance(item, dict) else str(item) for item in initial_knowledge]
    if texts:
        encoder.fit_tokenizer(texts)
    else:
        encoder.fit_tokenizer(["physics", "energy", "force", "science"])

    # --- Continuous Learning Loop ---
    cycle = 1
    try:
        while True:
            print(f"\n🌀 Learning Cycle {cycle} starting...")

            # 1️⃣ Fetch new knowledge
            raw_data = fetcher.fetch(topic)
            safe_data = safety.filter(raw_data)
            kb.store(safe_data)

            # 2️⃣ Update tokenizer
            all_knowledge = kb.query("")
            new_texts = [item["text"] if isinstance(item, dict) else str(item) for item in all_knowledge]
            if new_texts:
                encoder.fit_tokenizer(new_texts)

            # 3️⃣ Train model
            if all_knowledge:
                trainer.run_training(all_knowledge)

            # 4️⃣ Reflection — rebuild KG and detect gaps
            print("\n🧠 [Reflection] Beginning knowledge review and graph analysis...")
            reflection.review_knowledge()

            # --- Log when AI starts learning from graph gaps ---
            print("\n📈 [Graph Expansion Log] Checking for weak or missing relations...")
            print("   If new relationships are detected, the system will auto-fetch and train on them.")
            reflection._expand_from_graph_gaps()

            # --- Continue planned learning ---
            print("\n🧭 [Planning] Deciding next major topic...")
            reflection.plan_next_steps()

            # 5️⃣ Graph reasoning demo
            gr = GraphReasoner(reflection.kg)
            print("🔎 Graph reasoning demo:", gr.explain_relation("force", "motion"))
            print("🔎 Transitive (causes) from 'force':", gr.suggest_transitive("force", "causes"))

            # 6️⃣ Textual + Graph reasoning fusion
            ans = reasoning.answer("Explain Newton’s third law in simple words.")
            print("🧠 Reasoning Output:\n", ans)

            # 7️⃣ Cycle completion
            print("⏳ Waiting before next learning cycle...\n")
            time.sleep(10)
            cycle += 1

    except KeyboardInterrupt:
        print("\n🧩 Learning loop interrupted by user.")
        print("📊 Displaying training progress...")
        trainer.visualizer.plot_progress()
        kb.close()


if __name__ == "__main__":
    main()
