# reasoning_module/kg_builder.py
# ------------------------------------------------------------
# Builds Knowledge Graph entries from discovery pipeline output.
# Called after each query run to persist claims as structured
# relations in the KnowledgeGraph.
#
# Relation types used:
#   supports_efficiency  — claim supports efficiency improvement
#   reduces              — method reduces a cost/overhead
#   improves             — method improves a metric
#   contradicts          — claim contradicts another
#   related_to           — general relationship
#   has_tradeoff         — known limitation or tradeoff
# ------------------------------------------------------------

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

# ─────────────────────────────────────────────
# Keyword maps for relation extraction
# ─────────────────────────────────────────────

# Method/algorithm names to normalize
METHOD_ALIASES = {
    "flashattention": "FlashAttention",
    "flash attention": "FlashAttention",
    "flash-attention": "FlashAttention",
    "sparse attention": "SparseAttention",
    "sparse transformer": "SparseAttention",
    "mixture of experts": "MixtureOfExperts",
    "mixture-of-experts": "MixtureOfExperts",
    "moe": "MixtureOfExperts",
    "kv cache": "KVCache",
    "kv-cache": "KVCache",
    "key-value cache": "KVCache",
    "paged attention": "PagedAttention",
    "pagedattention": "PagedAttention",
    "lora": "LoRA",
    "low-rank adaptation": "LoRA",
    "speculative decoding": "SpeculativeDecoding",
    "rotary position embedding": "RoPE",
    "rope": "RoPE",
    "grouped query attention": "GroupedQueryAttention",
    "multi-query attention": "MultiQueryAttention",
    "linear attention": "LinearAttention",
    "mamba": "Mamba",
    "rwkv": "RWKV",
    "gptq": "GPTQ",
    "awq": "AWQ",
    "sliding window attention": "SlidingWindowAttention",
    "ring attention": "RingAttention",
    "distilbert": "DistilBERT",
    "sparsegpt": "SparseGPT",
    "longformer": "Longformer",
    "bigbird": "BigBird",
}

# Metrics/properties
METRIC_ALIASES = {
    "memory": "MemoryOverhead",
    "memory overhead": "MemoryOverhead",
    "memory cost": "MemoryOverhead",
    "latency": "Latency",
    "throughput": "Throughput",
    "compute": "ComputeCost",
    "compute cost": "ComputeCost",
    "flops": "ComputeCost",
    "perplexity": "Perplexity",
    "accuracy": "ModelAccuracy",
    "quality": "ModelQuality",
    "context length": "ContextLength",
    "long context": "ContextLength",
    "sequence length": "ContextLength",
    "training stability": "TrainingStability",
    "routing instability": "RoutingInstability",
    "inference speed": "InferenceSpeed",
    "inference efficiency": "InferenceEfficiency",
    "parameter efficiency": "ParameterEfficiency",
    "scalability": "Scalability",
    "efficiency": "TransformerEfficiency",
}

# Verbs that indicate a positive relation
IMPROVE_VERBS = [
    "improves", "improve", "reduces", "reduce", "decreases", "decrease",
    "accelerates", "accelerate", "enhances", "enhance", "outperforms",
    "increases throughput", "lowers latency", "lowers memory",
    "faster", "more efficient", "better", "higher", "lower overhead",
]

# Verbs that indicate a negative/tradeoff relation
TRADEOFF_VERBS = [
    "instability", "unstable", "degrades", "degrade", "hurts", "hurts quality",
    "tradeoff", "trade-off", "overhead", "does not", "fails", "worse",
    "limited", "limitation", "however", "but requires",
]


def _normalize_method(text: str) -> Optional[str]:
    low = text.lower()
    for alias, canonical in METHOD_ALIASES.items():
        if alias in low:
            return canonical
    return None


def _normalize_metric(text: str) -> Optional[str]:
    low = text.lower()
    for alias, canonical in METRIC_ALIASES.items():
        if alias in low:
            return canonical
    return None


def _extract_relation_from_claim(claim_text: str):
    """
    Try to extract (subject, relation, object) from a claim sentence.
    Returns list of (subj, rel, obj) tuples.
    """
    triples = []
    low = claim_text.lower()

    method = _normalize_method(low)
    metric = _normalize_metric(low)

    if not method:
        return triples

    # Check for improvement relations
    for verb in IMPROVE_VERBS:
        if verb in low:
            if metric:
                triples.append((method, "reduces" if "reduc" in verb or "lower" in verb else "improves", metric))
            else:
                triples.append((method, "improves", "TransformerEfficiency"))
            break

    # Check for tradeoff/contradiction relations
    for verb in TRADEOFF_VERBS:
        if verb in low:
            if metric:
                triples.append((method, "has_tradeoff", metric))
            else:
                triples.append((method, "has_tradeoff", "GeneralLimitation"))
            break

    # If no verb matched but method + metric found — add general relation
    if not triples and metric:
        triples.append((method, "related_to", metric))

    return triples


def _verdict_to_relation(verdict: str) -> Optional[str]:
    mapping = {
        "supported": "supports_efficiency",
        "partially_supported": "partially_supports",
        "contradicted": "contradicts",
        "inconclusive": None,  # don't store inconclusive relations
    }
    return mapping.get(verdict)


def ingest_result_into_kg(
    kg,
    query: str,
    result: Dict[str, Any],
    min_confidence: float = 0.45,
) -> int:
    """
    Takes a discovery pipeline result and writes structured
    relations into the KnowledgeGraph.

    Returns number of relations added.
    """
    added = 0
    grounded = result.get("grounded_claims", [])

    for gc in grounded:
        claim_text = (gc.get("claim") or "").strip()
        if not claim_text:
            continue

        verdict_obj = gc.get("verdict", {})
        verdict = verdict_obj.get("verdict", "inconclusive")
        confidence = float(verdict_obj.get("confidence", 0.0))

        # Skip low-confidence or inconclusive
        if confidence < min_confidence:
            continue
        if verdict == "inconclusive":
            continue

        # 1) Extract structural triples from claim text
        triples = _extract_relation_from_claim(claim_text)
        for subj, rel, obj in triples:
            kg.add_relation(subj, rel, obj)
            added += 1

        # 2) Link claim verdict to query topic
        query_method = _normalize_method(query)
        query_metric = _normalize_metric(query)
        rel = _verdict_to_relation(verdict)

        if rel and query_method:
            target = query_metric or "TransformerEfficiency"
            kg.add_relation(query_method, rel, target)
            added += 1

        # 3) Store paper as source of evidence
        paper_title = gc.get("grounding", {})
        # grounded claims don't carry paper directly — use claims
        claims = result.get("extracted_claims", [])
        for cl in claims:
            if cl.get("claim", "").strip() == claim_text:
                paper = cl.get("paper_title", "").strip()
                method = _normalize_method(claim_text)
                if paper and method:
                    # Shorten paper title to key identifier
                    short = paper[:50].strip()
                    kg.add_relation(method, "evidenced_by", short)
                    added += 1
                break

    return added


def build_kg_from_reports(kg, reports: List[Dict[str, Any]]) -> int:
    """
    Process a list of saved report dicts into the KG.
    Useful for batch rebuilding KG from saved JSON reports.
    """
    total = 0
    for r in reports:
        query = r.get("query", "")
        total += ingest_result_into_kg(kg, query, r)
    return total