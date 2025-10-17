"""
Humanoid Scientist Brain — Continuous Learning & Graph Reasoning
---------------------------------------------------------------
Phase C Step 2 (with dashboard updates):
- Hypothesis validation + promotion
- Dashboard JSON state logging for Flask visualization
"""

import time, json, os
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


# ---------- Dashboard State Helper ----------
def update_dashboard_state(cycle, reflection, hyps=None, validated=None, topic="physics"):
    """Store latest learning stats for dashboard"""
    state = {
        "cycle": cycle,
        "topic": topic,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "knowledge_stats": {
            "nodes": len(reflection.kg.graph),
            "edges": getattr(reflection.kg, "edge_count", lambda: 0)(),
        },
        "hypotheses": [],
        "validated": [],
    }

    if hyps:
        state["hypotheses"] = [
            {"text": h["hypothesis"], "score": round(h.get("score", 0.0), 3), "type": h.get("type", "")}
            for h in hyps[:10]
        ]
    if validated:
        state["validated"] = [
            {
                "text": v["hypothesis"],
                "support": v.get("support", 0),
                "consistency": v.get("consistency", 0),
                "confidence": v.get("confidence", 0),
                "persist": v.get("persistence", 0),
                "promote": v.get("promote", False),
            }
            for v in validated[:10]
        ]

    os.makedirs("data", exist_ok=True)
    with open("data/dashboard_state.json", "w") as f:
        json.dump(state, f, indent=2)
    print("📊 Dashboard state updated → data/dashboard_state.json")


# ---------- Main Loop ----------
def main():
    print("🤖 Starting Humanoid Scientist Brain (Continuous Learning Mode)...")

    fetcher = DataFetcher()
    safety = SafetyFilter()
    kb = KnowledgeBase()
    encoder = TextEncoder()
    trainer = Trainer(encoder)
    reflection = ReflectionEngine(fetcher, safety, kb, trainer)
    hypgen = HypothesisGenerator(reflection.kg, encoder, kb)
    validator = HypothesisValidator(kb, encoder, reflection.kg)
    reasoning = GraphAugmentedReasoning(kb, encoder, reflection.kg, hypothesis_generator=hypgen)

    # Tokenizer setup
    initial = kb.query("")
    texts = [i["text"] if isinstance(i, dict) else str(i) for i in initial]
    encoder.fit_tokenizer(texts or ["physics", "energy", "force", "science"])

    topic = "physics"
    cycle = 1

    try:
        while True:
            print(f"\n🌀 Learning Cycle {cycle} starting...")

            # 1️⃣ Fetch new data
            raw = fetcher.fetch(topic)
            safe = safety.filter(raw)
            kb.store(safe)

            # 2️⃣ Update tokenizer
            all_items = kb.query("")
            all_texts = [it["text"] if isinstance(it, dict) else str(it) for it in all_items]
            if all_texts:
                encoder.fit_tokenizer(all_texts)

            # 3️⃣ Train
            if all_items:
                trainer.run_training(all_items)

            # 4️⃣ Reflect
            print("\n🧠 [Reflection] Begin KG update and gap analysis...")
            reflection.review_knowledge()
            reasoning.set_graph(reflection.kg)

            # 5️⃣ Hypothesize + Validate
            hyps = hypgen.generate(top_n=20)
            validated = []
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
                    reflection.promote_validated(validated)

            # 6️⃣ Reasoning demo
            gr = GraphReasoner(reflection.kg)
            print("🔎 Graph reasoning demo:", gr.explain_relation("force", "motion"))
            print("🔎 Transitive (causes) from 'force':", gr.suggest_transitive("force", "causes"))

            ans = reasoning.answer("Explain Newton’s third law in simple words.")
            print("🧠 Reasoning Output:\n", ans)

            # 7️⃣ Update dashboard
            update_dashboard_state(cycle, reflection, hyps, validated, topic)

            # 8️⃣ Next
            print("⏳ Waiting before next learning cycle...\n")
            time.sleep(10)
            cycle += 1

    except KeyboardInterrupt:
        print("\n🧩 Learning loop interrupted by user.")
        kb.close()


if __name__ == "__main__":
    main()
