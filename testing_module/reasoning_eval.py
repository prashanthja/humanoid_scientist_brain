# testing_module/reasoning_eval.py
from __future__ import annotations
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from reasoning_module.discover import DiscoveryEngine, DiscoveryConfig
from reasoning_module.evidence_evaluator import EvidenceEvaluator
from reasoning_module.proposal_evaluator import ProposalEvaluator
from reasoning_module.hypothesis_generator import HypothesisGenerator
from reasoning_module.hypothesis_validator import HypothesisValidator
from reasoning_module.discovery_report import generate_and_save
from reasoning_module.kg_builder import ingest_result_into_kg
from knowledge_graph.graph import KnowledgeGraph
from retrieval.simple_retriever import SimpleRetriever as ChunkIndex
from knowledge_base.chunk_store import ChunkStore
from learning_module.embedding_bridge import EmbeddingBridge
from learning_module.embedding_bridge import EmbeddingBridge


def build_engine(kg: KnowledgeGraph):
    chunk_store = ChunkStore()
    count = chunk_store.count()
    print(f"ChunkStore count: {count}")

    if count == 0:
        print("\nERROR: Chunk store is empty.")
        print("  python3 scripts/ingest_transformer_efficiency.py")
        print("  python3 scripts/ingest_chunks.py --rebuild_only")
        sys.exit(1)

    encoder = EmbeddingBridge()
    encoder = EmbeddingBridge(trainer)
    chunk_index = ChunkIndex(encoder=encoder, chunk_store=chunk_store)

    print("Rebuilding ChunkIndex...")
    chunk_index.rebuild()
    print(f"ChunkIndex ready: dim={chunk_index.dim}, items={len(chunk_index.items)}")

    proposal_evaluator = ProposalEvaluator(
        kb=chunk_store,
        bridge=encoder,
        top_k=10,
        evidence_threshold=0.55,
        require_evidence=True,
    )

    evidence_evaluator = EvidenceEvaluator(
        kb=chunk_store,
        encoder=encoder,
        kg=None,
        chunk_index=chunk_index,
        use_chunk_index=True,
    )

    try:
        hypgen = HypothesisGenerator(kg=kg)
        validator = HypothesisValidator(kg=kg)
    except Exception:
        class _NullGen:
            def generate(self, top_n=10): return []
        class _NullVal:
            def validate(self, hyps, cycle=0): return []
        hypgen = _NullGen()
        validator = _NullVal()

    return DiscoveryEngine(
        chunk_index=chunk_index,
        proposal_engine=proposal_evaluator,
        evidence_evaluator=evidence_evaluator,
        hypgen=hypgen,
        validator=validator,
        config=DiscoveryConfig(
            top_k_chunks=10,
            max_claims=10,
            max_hypotheses=5,
            max_grounded_claims=5,
            use_mmr=True,
        ),
    )


def print_section(label: str):
    print(f"\n{'─' * 35} {label} {'─' * 35}")


def main():
    queries = [
        # Original 10
        "Do mixture-of-experts improve transformer efficiency?",
        "Does sparse attention preserve long-context quality?",
        "Does FlashAttention reduce memory overhead?",
        "Does KV cache compression reduce LLM inference latency?",
        "Does LoRA reduce fine-tuning memory cost?",
        "Does speculative decoding improve LLM inference throughput?",
        "Does quantization reduce transformer memory without quality loss?",
        "Does MoE routing instability hurt model quality?",
        "Does rotary position embedding improve long-context performance?",
        "Does continuous batching improve LLM serving throughput?",
        # New queries — different topics = new papers = new KG edges
        "Does grouped query attention reduce KV cache memory?",
        "Does PagedAttention improve GPU memory utilization?",
        "Does pruning transformer weights reduce inference cost?",
        "Does knowledge distillation preserve model quality?",
        "Does sliding window attention scale to long sequences?",
        "Does Mamba outperform transformers on long sequences?",
        "Does tensor parallelism improve LLM training throughput?",
        "Does pipeline parallelism reduce training time?",
        "Does RWKV match transformer quality with linear complexity?",
        "Does model quantization affect downstream task performance?",
    ]

    kg = KnowledgeGraph()
    kg.load()
    print(f"KnowledgeGraph loaded: {kg.edge_count()} existing edges")

    engine = build_engine(kg)

    for q in queries:
        print("\n" + "=" * 80)
        print("QUERY:", q)

        result = engine.run(q, source_name="reasoning_eval")

        print_section("Evidence Chunks")
        chunks = result.get("evidence_chunks", [])
        print(f"Retrieved: {len(chunks)}")
        for c in chunks[:3]:
            print(f"  [{c.get('profile_hits',0)} hits | sim={c.get('similarity',0):.3f}] "
                  f"{c.get('paper_title','')[:65]}")

        print_section("Proposal Verdict")
        pv = result.get("proposal_verdict", {})
        print(f"  verdict     : {pv.get('verdict')}")
        print(f"  confidence  : {pv.get('confidence', 0):.3f}")
        print(f"  explanation : {pv.get('explanation','')[:120]}")

        print_section("Grounded Claims")
        for gc in result.get("grounded_claims", [])[:3]:
            v = gc.get("verdict", {})
            print(f"  {v.get('verdict')} | conf={v.get('confidence',0):.3f} | "
                  f"{gc.get('claim','')[:90]}")

        print_section("Saving Report")
        try:
            paths = generate_and_save(q, result, fmt="both")
            for fmt, path in paths.items():
                print(f"  {fmt}: {os.path.basename(path)}")
        except Exception as e:
            print(f"  Report save failed: {e}")

        print_section("Knowledge Graph Update")
        try:
            added = ingest_result_into_kg(kg, q, result, min_confidence=0.45)
            print(f"  Relations added: {added}")
            print(f"  Total KG edges : {kg.edge_count()}")
            print(f"  KG concepts    : {len(kg.all_concepts())}")
        except Exception as e:
            print(f"  KG update failed: {e}")

    print("\n" + "=" * 80)
    print("KNOWLEDGE GRAPH SUMMARY")
    print(f"  Total edges    : {kg.edge_count()}")
    print(f"  Total concepts : {len(kg.all_concepts())}")
    print("\n  Relations by concept:")
    for concept in sorted(kg.all_concepts())[:15]:
        rels = kg.get_relations(concept)
        if rels:
            for rel, targets in rels.items():
                for t in targets[:2]:
                    print(f"    {concept} --[{rel}]--> {t}")


def test_hypotheses():
    kg = KnowledgeGraph()
    kg.load()
    print(f"KG edges: {kg.edge_count()}, concepts: {len(kg.all_concepts())}")

    try:
        encoder = EmbeddingBridge()
        encoder = EmbeddingBridge(trainer)
    except Exception:
        encoder = None
        print("No encoder — graph-only mode")

    gen = HypothesisGenerator(kg=kg, encoder=encoder)
    hyps = gen.generate(top_n=10)

    print(f"\nGenerated {len(hyps)} hypotheses:\n")
    for i, h in enumerate(hyps, 1):
        print(f"{i}. [{h['type']}] score={h['score']}")
        print(f"   {h['hypothesis']}")
        print(f"   Premises:")
        for p in h.get("premises", []):
            print(f"     - {p}")
        print()


# ── Single entry point ───────────────────────
if __name__ == "__main__":
    if "--hyp_only" in sys.argv:
        test_hypotheses()
    else:
        main()