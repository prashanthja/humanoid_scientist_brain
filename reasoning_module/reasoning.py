# reasoning_module/reasoning.py
"""
Graph-Augmented + Semantic Reasoning (Phase S Step 3)
"""

from knowledge_base.retriever import Retriever
from reasoning_module.graph_reasoner import GraphReasoner
from reasoning_module.semantic_reasoner import SemanticReasoner


class GraphAugmentedReasoning:
    def __init__(self, kb, encoder, graph=None):
        self.kb = kb
        self.encoder = encoder
        self.retriever = Retriever(kb, encoder)
        self.semantic = SemanticReasoner(encoder, kg=graph)
        self.graph_reasoner = GraphReasoner(graph, semantic=self.semantic) if graph else None

    def set_graph(self, graph):
        """Call when KG is rebuilt to keep modules in sync."""
        self.semantic.set_graph(graph)
        self.graph_reasoner = GraphReasoner(graph, semantic=self.semantic)

    def answer(self, query: str, top_k: int = 3) -> str:
        print(f"[Reasoning] Answering (semantic+graph): {query}")

        # 1) KB retrieval
        kb_results = self.retriever.semantic_search(query, top_k=top_k)
        kb_summary = "\n".join([r[1] for r in kb_results]) if kb_results else "No direct text found."

        # 2) Graph reasoning (with semantic fallback)
        graph_insights = "(No Knowledge Graph loaded)"
        if self.graph_reasoner:
            try:
                key_terms = self._extract_keywords(query)
                if len(key_terms) >= 2:
                    a, b = key_terms[:2]
                    graph_insights = self.graph_reasoner.explain_relation(a, b)
                elif len(key_terms) == 1:
                    # Just show nearest concepts semantically
                    terms = self.semantic.nearest_concepts(key_terms[0], top_k=5)
                    graph_insights = f"Nearest concepts to '{key_terms[0]}': {[n for n,_ in terms]}"
            except Exception as e:
                graph_insights = f"(Graph reasoning unavailable: {e})"

        # 3) Semantic reasoning summary
        sem_block = ""
        try:
            k = self._extract_keywords(query)
            if len(k) >= 2:
                sem_block = self.semantic.semantic_explain(k[0], k[1])
            elif len(k) == 1:
                nn = self.semantic.nearest_concepts(k[0], top_k=5)
                sem_block = f"Nearest to '{k[0]}': {[(n, round(s,3)) for n,s in nn]}"
            else:
                sem_block = "No semantic extraction."
        except Exception as e:
            sem_block = f"(Semantic unavailable: {e})"

        # 4) Synthesis
        final = (
            f"🧠 **Textual Knowledge Extracted:**\n{kb_summary}\n\n"
            f"🕸️ **Graph Reasoning:**\n{graph_insights}\n\n"
            f"📐 **Semantic Reasoning:**\n{sem_block}\n\n"
            f"✅ **Scientific Synthesis:**\n"
            f"Combining retrieved facts, graph structure, and semantic proximity, "
            f"the system infers: {query}"
        )
        return final

    def _extract_keywords(self, text: str):
        words = [w.strip(".,?;:!()[]\"'").lower() for w in text.split() if len(w) > 3]
        unique = list(dict.fromkeys(words))
        return unique[:3]
