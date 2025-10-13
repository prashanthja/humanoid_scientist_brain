import time
from data_pipeline.fetcher import DataFetcher
from data_pipeline.filter import SafetyFilter
from knowledge_base.database import KnowledgeBase
from learning_module.trainer import Trainer
from reasoning_module.reasoning import ReasoningEngine
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
    reasoning = ReasoningEngine(kb, encoder)
    reflection = ReflectionEngine(fetcher, safety, kb, trainer)

    # --- Initial topic ---
    topic = "physics"

    # --- Fit tokenizer before starting the loop (on existing KB) ---
    initial_knowledge = kb.query("")
    texts = [item["text"] if isinstance(item, dict) else str(item) for item in initial_knowledge]
    if texts:
        encoder.fit_tokenizer(texts)
    else:
        encoder.fit_tokenizer(["physics", "energy", "force", "science"])

    # --- Continuous learning loop ---
    cycle = 1
    try:
        while True:
            print(f"\n🌀 Learning Cycle {cycle} starting...")

            # 1️⃣ Fetch new knowledge
            raw_data = fetcher.fetch(topic)
            safe_data = safety.filter(raw_data)
            kb.store(safe_data)

            # 2️⃣ Update tokenizer with new knowledge
            all_knowledge = kb.query("")
            new_texts = [item["text"] if isinstance(item, dict) else str(item) for item in all_knowledge]
            if new_texts:
                encoder.fit_tokenizer(new_texts)

            # 3️⃣ Train the model
            if all_knowledge:
                trainer.run_training(all_knowledge)

            # 4️⃣ Reflection: analyze and plan next steps
            reflection.review_knowledge()
            reflection.plan_next_steps()

            # 5️⃣ Reasoning example
            ans = reasoning.answer("Explain Newton’s third law in simple words.")
            print("🧠 Reasoning Output:", ans)

            # 6️⃣ Wait before next cycle (adjust as needed)
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
