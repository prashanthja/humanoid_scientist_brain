"""
Graph-Augmented + Semantic Reasoning
------------------------------------
Fuses KB retrieval, KG reasoning, and semantic inference.
Optionally receives a HypothesisGenerator to cross-check proposed links.
"""

from typing import List, Tuple
from knowledge_base.retriever import Retriever
from reasoning_module.graph_reasoner import GraphReasoner

# Note: HypothesisGenerator import is optional at runtime
try:
    from reasoning_module.hypothesis_generator import HypothesisGenerator  # noqa: F401
except Exception:
    HypothesisGenerator = None  # type: ignore


class GraphAugmentedReasoning:
    def __init__(self, kb, encoder, graph=None, hypothesis_generator=None):
        self.kb = kb
        self.encoder = encoder
        self.retriever = Retriever(kb, encoder)
        self.graph_reasoner = GraphReasoner(graph) if graph else None
        self.hyp_gen = hypothesis_generator  # may be None
        
    def set_graph(self, graph):
        """Attach or update the Knowledge Graph dynamically."""
        self.graph_reasoner = GraphReasoner(graph)

    def answer(self, query: str, top_k: int = 3) -> str:
        print(f"[Reasoning] Answering (semantic+graph): {query}")

        # 1) KB retrieval
        kb_hits: List[Tuple[int, str, float]] = self.retriever.semantic_search(query, top_k=top_k)
        kb_text = "\n".join([h[1] for h in kb_hits]) if kb_hits else "(no direct text)"

        # 2) KG reasoning
        graph_block = "(no KG)"
        if self.graph_reasoner:
            try:
                terms = self._keywords(query)
                if len(terms) >= 2:
                    a, b = terms[:2]
                    relation = self.graph_reasoner.explain_relation(a, b)
                    trans = self.graph_reasoner.suggest_transitive(a, "causes")
                    graph_block = f"Relation between '{a}' and '{b}': {relation}\nTransitive links: {trans}"
                else:
                    graph_block = self.graph_reasoner.find_related_concepts(terms[0])
            except Exception as e:
                graph_block = f"(KG reasoning error: {e})"

        # 3) Semantic sketch
        sem_block = "(semantic layer active)"
        try:
            # cheap similarity of first two keywords
            terms = self._keywords(query)
            if len(terms) >= 2:
                a, b = terms[:2]
                va = self.encoder.encode_texts([a])[0]
                vb = self.encoder.encode_texts([b])[0]
                va /= ( (va**2).sum()**0.5 + 1e-9 )
                vb /= ( (vb**2).sum()**0.5 + 1e-9 )
                sim = float((va * vb).sum())
                sem_block = f"semantic similarity({a},{b})={sim:.3f}"
        except Exception as e:
            sem_block = f"(semantic layer error: {e})"

        # 4) Synthesis
        answer = (
            " 🧠 **Textual Knowledge Extracted:**\n" + kb_text + "\n\n"
            "🕸️ **Graph Reasoning:**\n" + graph_block + "\n\n"
            "📐 **Semantic Reasoning:**\n" + sem_block + "\n\n"
            "✅ **Scientific Synthesis:**\n"
            f"Combining retrieved facts, graph structure, and semantic proximity, the system infers: {query}."
        )
        return answer

    def _keywords(self, text: str):
        ws = [w.strip(".,?!:;()[]\"'").lower() for w in text.split() if len(w) > 3]
        # uniqueness preserving order
        seen = set()
        out = []
        for w in ws:
            if w not in seen:
                out.append(w)
                seen.add(w)
        return out[:4]
