"""
KnowledgeGraph (final unified version)
-------------------------------------
Supports both old list format and new dict-based relation structure.
"""

import json, re, os
from collections import defaultdict
from typing import Dict, List, Tuple, Set


class KnowledgeGraph:
    def __init__(self):
        # Structure: {subject: {relation: [objects]}}
        self.graph: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    # ---------- Persistence ----------
    def save(self, path: str = "knowledge_graph/graph.json"):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        serializable = {s: {r: list(objs) for r, objs in rels.items()} 
                        for s, rels in self.graph.items() if rels}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: str = "knowledge_graph/graph.json"):
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            data = json.load(f)

        # Convert both old and new formats
        new_graph = defaultdict(lambda: defaultdict(list))
        if isinstance(data, list):  # old format: list of tuples
            for s, rel, t in data:
                new_graph[s][rel].append(t)
        elif isinstance(data, dict):
            for s, rels in data.items():
                if isinstance(rels, list):  # old inner structure
                    for rel, t in rels:
                        new_graph[s][rel].append(t)
                elif isinstance(rels, dict):
                    for rel, objs in rels.items():
                        if isinstance(objs, list):
                            new_graph[s][rel].extend(objs)
                        else:
                            new_graph[s][rel].append(str(objs))
        self.graph = new_graph

    # ---------- Relation management ----------
    def add_relation(self, subj: str, relation: str, obj: str):
        """Add a relation (safe for both formats)."""
        if not subj or not obj or not relation:
            return

        subj, relation, obj = subj.strip(), relation.strip(), obj.strip()
        # ensure nested dicts exist
        if subj not in self.graph:
            self.graph[subj] = defaultdict(list)
        if relation not in self.graph[subj]:
            self.graph[subj][relation] = []

        # add object if not duplicate
        if obj not in self.graph[subj][relation]:
            self.graph[subj][relation].append(obj)
            # add symmetric link for bidirectional relations
            if relation == "related_to":
                if obj not in self.graph:
                    self.graph[obj] = defaultdict(list)
                if subj not in self.graph[obj][relation]:
                    self.graph[obj][relation].append(subj)

        # print(f"✅ Relation added to KG: {subj} -[{relation}]-> {obj}")
        self.save()

    # ---------- Core utilities ----------
    def build_from_corpus(self, texts: List[str]):
        """Extract ML-specific causal relations from text patterns."""
        import re as _re

        CONCEPT_MAP = {
            "kv cache": "KVCache", "key-value cache": "KVCache", "key value cache": "KVCache",
            "flash attention": "FlashAttention", "flashattention": "FlashAttention",
            "flashattention-2": "FlashAttention", "flashattention-3": "FlashAttention",
            "lora": "LoRA", "low-rank adaptation": "LoRA", "low rank adaptation": "LoRA",
            "qlora": "QLoRA", "quantized lora": "QLoRA",
            "mixture of experts": "MixtureOfExperts", "moe": "MixtureOfExperts",
            "speculative decoding": "SpeculativeDecoding",
            "paged attention": "PagedAttention", "pagedattention": "PagedAttention",
            "sparse attention": "SparseAttention",
            "sliding window attention": "SlidingWindowAttention",
            "mamba": "Mamba", "state space model": "Mamba", "ssm": "Mamba",
            "rwkv": "RWKV",
            "linear attention": "LinearAttention",
            "grouped query attention": "GroupedQueryAttention", "gqa": "GroupedQueryAttention",
            "multi-head latent attention": "MultiHeadLatentAttention", "mla": "MultiHeadLatentAttention",
            "prefix caching": "PrefixCaching", "prefix cache": "PrefixCaching",
            "continuous batching": "ContinuousBatching",
            "tensor parallelism": "TensorParallelism",
            "pipeline parallelism": "PipelineParallelism",
            "quantization": "Quantization", "quantisation": "Quantization",
            "pruning": "Pruning", "model pruning": "Pruning",
            "knowledge distillation": "KnowledgeDistillation",
            "latency": "Latency", "throughput": "Throughput",
            "memory": "MemoryOverhead", "memory overhead": "MemoryOverhead", "memory usage": "MemoryOverhead",
            "accuracy": "ModelAccuracy", "model accuracy": "ModelAccuracy", "model quality": "ModelQuality",
            "compute": "ComputeCost", "flops": "ComputeCost", "compute cost": "ComputeCost",
            "context length": "ContextLength", "sequence length": "ContextLength", "long context": "ContextLength",
            "rope": "RoPE", "rotary position": "RoPE", "rotary embedding": "RoPE",
            "routing instability": "RoutingInstability", "load balancing": "LoadBalancing",
            "hardware compatibility": "HardwareCompatibility",
            "activation checkpointing": "ActivationCheckpointing",
            "gradient checkpointing": "ActivationCheckpointing",
            "weight sharing": "WeightSharing",
            "early exit": "EarlyExit", "adaptive computation": "EarlyExit",
            "token merging": "TokenMerging", "token pruning": "TokenMerging",
            "retrieval augmented": "RAG", "rag": "RAG",
            "transformer efficiency": "TransformerEfficiency",
            "inference optimization": "InferenceOptimization",
            "training efficiency": "TrainingEfficiency",
        }

        PATTERNS = [
            (_re.compile(r'([\w\s\-]+?)\s+reduces?\s+(latency|memory|compute|overhead|flops|cost)', _re.I), 'reduces'),
            (_re.compile(r'([\w\s\-]+?)\s+improves?\s+(throughput|accuracy|quality|efficiency|performance)', _re.I), 'improves'),
            (_re.compile(r'([\w\s\-]+?)\s+outperforms?\s+(transformer|attention|baseline|dense)', _re.I), 'supports_efficiency'),
            (_re.compile(r'([\w\s\-]+?)\s+(?:has|introduces?)\s+(?:a\s+)?tradeoff\s+(?:with|between)\s+([\w\s]+)', _re.I), 'has_tradeoff'),
            (_re.compile(r'([\w\s\-]+?)\s+(?:does not|doesn\'t|fails to)\s+(?:reduce|improve|outperform)\s+([\w\s]+)', _re.I), 'contradicts'),
            (_re.compile(r'([\w\s\-]+?)\s+(?:enables?|supports?)\s+(?:efficient|faster|better)\s+([\w\s]+)', _re.I), 'supports_efficiency'),
            (_re.compile(r'([\w\s\-]+?)\s+partially\s+(?:supports?|addresses?)\s+([\w\s]+)', _re.I), 'partially_supports'),
        ]

        def _norm(text):
            t = text.lower().strip()
            return CONCEPT_MAP.get(t, None)

        new_relations = 0
        for text in texts:
            if not text: continue
            tl = text.lower()
            for pattern, rel in PATTERNS:
                for m in pattern.finditer(tl):
                    subj_raw = m.group(1).strip()
                    obj_raw = m.group(2).strip() if len(m.groups()) > 1 else ""
                    subj = _norm(subj_raw)
                    obj = _norm(obj_raw)
                    if subj and obj and subj != obj:
                        self.add_relation(subj, rel, obj)
                        new_relations += 1

        # Seed known important relations if not present
        SEED_RELATIONS = [
            ("FlashAttention", "reduces", "MemoryOverhead"),
            ("FlashAttention", "improves", "Throughput"),
            ("FlashAttention", "has_tradeoff", "MemoryOverhead"),
            ("LoRA", "reduces", "ComputeCost"),
            ("LoRA", "supports_efficiency", "TrainingEfficiency"),
            ("LoRA", "has_tradeoff", "ModelAccuracy"),
            ("QLoRA", "reduces", "MemoryOverhead"),
            ("QLoRA", "supports_efficiency", "TrainingEfficiency"),
            ("MixtureOfExperts", "improves", "Throughput"),
            ("MixtureOfExperts", "has_tradeoff", "RoutingInstability"),
            ("MixtureOfExperts", "partially_supports", "ModelQuality"),
            ("Mamba", "reduces", "Latency"),
            ("Mamba", "supports_efficiency", "Throughput"),
            ("Mamba", "partially_supports", "ModelQuality"),
            ("KVCache", "reduces", "Latency"),
            ("KVCache", "has_tradeoff", "MemoryOverhead"),
            ("SpeculativeDecoding", "improves", "Throughput"),
            ("SpeculativeDecoding", "reduces", "Latency"),
            ("Quantization", "reduces", "MemoryOverhead"),
            ("Quantization", "has_tradeoff", "ModelAccuracy"),
            ("SparseAttention", "reduces", "ComputeCost"),
            ("SparseAttention", "has_tradeoff", "ModelAccuracy"),
            ("GroupedQueryAttention", "reduces", "MemoryOverhead"),
            ("GroupedQueryAttention", "supports_efficiency", "Throughput"),
            ("PrefixCaching", "reduces", "Latency"),
            ("PagedAttention", "reduces", "MemoryOverhead"),
            ("PagedAttention", "improves", "Throughput"),
            ("RoPE", "supports_efficiency", "ContextLength"),
            ("ContextLength", "has_tradeoff", "MemoryOverhead"),
            ("ContextLength", "has_tradeoff", "Latency"),
            ("TokenMerging", "reduces", "ComputeCost"),
            ("TokenMerging", "has_tradeoff", "ModelAccuracy"),
            ("ActivationCheckpointing", "reduces", "MemoryOverhead"),
            ("ActivationCheckpointing", "has_tradeoff", "Throughput"),
            ("ContinuousBatching", "improves", "Throughput"),
            ("TensorParallelism", "supports_efficiency", "TrainingEfficiency"),
            ("Pruning", "reduces", "ComputeCost"),
            ("Pruning", "has_tradeoff", "ModelAccuracy"),
            ("KnowledgeDistillation", "reduces", "ComputeCost"),
            ("KnowledgeDistillation", "partially_supports", "ModelAccuracy"),
            ("RAG", "improves", "ModelAccuracy"),
            ("RAG", "has_tradeoff", "Latency"),
            ("EarlyExit", "reduces", "Latency"),
            ("EarlyExit", "has_tradeoff", "ModelAccuracy"),
            ("RWKV", "supports_efficiency", "ModelQuality"),
            ("RWKV", "related_to", "Mamba"),
            ("LinearAttention", "reduces", "ComputeCost"),
            ("LinearAttention", "has_tradeoff", "ModelAccuracy"),
            ("MultiHeadLatentAttention", "reduces", "MemoryOverhead"),
            ("MultiHeadLatentAttention", "supports_efficiency", "Throughput"),
        ]
        for subj, rel, obj in SEED_RELATIONS:
            self.add_relation(subj, rel, obj)

        return new_relations


    def get_relations(self, concept: str) -> Dict[str, List[str]]:
        """Return relations for a given concept."""
        return self.graph.get(concept, {})

    def all_concepts(self) -> List[str]:
        nodes = set(self.graph.keys())
        for rels in self.graph.values():
            for objs in rels.values():
                nodes.update(objs)
        return sorted(nodes)

    def edge_count(self) -> int:
        return sum(len(objs) for rels in self.graph.values() for objs in rels.values())
