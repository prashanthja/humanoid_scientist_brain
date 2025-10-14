"""
Reasoning Engine — Graph-Augmented Reasoning
--------------------------------------------
Combines:
• Text-based retrieval from the Knowledge Base
• Graph reasoning through the Knowledge Graph
• Multi-hop causal inference & explanation synthesis
"""

from knowledge_base.retriever import Retriever
from reasoning_module.graph_reasoner import GraphReasoner


class GraphAugmentedReasoning:
    """
    Combines Knowledge Base + Knowledge Graph reasoning.
    """

    def __init__(self, kb, encoder, graph=None):
        self.kb = kb
        self.encoder = encoder
        self.retriever = Retriever(kb, encoder)
        self.graph_reasoner = GraphReasoner(graph) if graph else None

    # -----------------------------------------------------------
    def answer(self, query: str, top_k: int = 3) -> str:
        """
        1️⃣ Retrieve relevant knowledge from KB.
        2️⃣ Use KG for relational reasoning.
        3️⃣ Fuse both into a final, scientific explanation.
        """
        print(f"[ReasoningEngine] Processing query: {query}")

        # --- Step 1: Retrieve knowledge from KB ---
        kb_results = self.retriever.semantic_search(query, top_k=top_k)
        kb_summary = "\n".join([r[1] for r in kb_results]) if kb_results else "No relevant knowledge found."

        # --- Step 2: Use Knowledge Graph reasoning ---
        graph_insights = ""
        if self.graph_reasoner:
            try:
                key_terms = self._extract_keywords(query)
                if len(key_terms) >= 2:
                    a, b = key_terms[:2]
                    relation = self.graph_reasoner.explain_relation(a, b)
                    transitive = self.graph_reasoner.suggest_transitive(a, "causes")
                    graph_insights = f"Relation between '{a}' and '{b}': {relation}\nTransitive links: {transitive}"
                elif len(key_terms) == 1:
                    graph_insights = str(self.graph_reasoner.find_related_concepts(key_terms[0]))
                else:
                    graph_insights = "(No keywords to reason about.)"
            except Exception as e:
                graph_insights = f"(Graph reasoning error: {e})"
        else:
            graph_insights = "(No Knowledge Graph loaded)"

        # --- Step 3: Synthesize both results ---
        final_answer = (
            f"🧠 **Textual Knowledge Extracted:**\n{kb_summary}\n\n"
            f"🕸️ **Graph Reasoning Insights:**\n{graph_insights}\n\n"
            f"✅ **Scientific Synthesis:**\n"
            f"By integrating textual knowledge and conceptual graph structure, "
            f"the system concludes that *{query.lower()}* is derived from or related "
            f"to these observed scientific relationships."
        )

        return final_answer

    # -----------------------------------------------------------
    def _extract_keywords(self, text: str):
        """Lightweight keyword extractor (heuristic)."""
        words = [w.strip(".,?").lower() for w in text.split() if len(w) > 3]
        return list(dict.fromkeys(words))[:3]
