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
        # Concept aliases — map variations to canonical names
        CONCEPT_MAP = {
            "kv cache": "KVCache", "key-value cache": "KVCache", "key value cache": "KVCache",
            "flash attention": "FlashAttention", "flashattention": "FlashAttention",
            "lora": "LoRA", "low-rank adaptation": "LoRA", "low rank adaptation": "LoRA",
            "mixture of experts": "MixtureOfExperts", "moe": "MixtureOfExperts",
            "speculative decoding": "SpeculativeDecoding",
            "paged attention": "PagedAttention", "pagedattention": "PagedAttention",
            "sparse attention": "SparseAttention",
            "sliding window attention": "SlidingWindowAttention",
            "mamba": "Mamba", "state space model": "Mamba", "ssm": "Mamba",
            "rwkv": "RWKV",
            "linear attention": "LinearAttention",
            "quantization": "Quantization", "quantised": "Quantization",
            "pruning": "Pruning",
            "knowledge distillation": "KnowledgeDistillation",
            "latency": "Latency", "throughput": "Throughput",
            "memory": "MemoryOverhead", "memory overhead": "MemoryOverhead",
            "accuracy": "ModelAccuracy", "model accuracy": "ModelAccuracy",
            "compute": "ComputeCost", "flops": "ComputeCost", "compute cost": "ComputeCost",
            "context length": "ContextLength", "sequence length": "ContextLength",
            "rope": "RoPE", "rotary": "RoPE",
        }

        EFFICIENCY_PATTERNS = [
            (r'([\w\s\-]+?)\s+reduces?\s+(latency|memory|compute|overhead|flops)', 'reduces'),
            (r'([\w\s\-]+?)\s+improves?\s+(throughput|accuracy|quality|efficiency|performance)', 'improves'),
            (r'([\w\s\-]+?)\s+outperforms?\s+(transformer|attention|baseline)', 'supports_efficiency'),
            (r'([\w\s\-]+?)\s+(?:has|introduces?)\s+(?:a\s+)?tradeoff\s+(?:with|between)\s+([\w\s]+)', 'has_tradeoff'),
            (r'([\w\s\-]+?)\s+(?:does not|doesn\'t|fails to)\s+(?:reduce|improve|outperform)\s+([\w\s]+)', 'contradicts'),
        ]

        # Auto-discover new concepts from chunks
        import re as _re
        # Known concept patterns in ML papers
        AUTO_CONCEPTS = {
            r'(mamba)': 'Mamba',
            r'(rwkv)': 'RWKV',
            r'(flash\s*attention\d*)': 'FlashAttention',
            r'(lora|low.rank adaptation)': 'LoRA',
            r'(kv.cache|key.value cache)': 'KVCache',
            r'(speculative decoding)': 'SpeculativeDecoding',
            r'(mixture of experts|moe)': 'MixtureOfExperts',
            r'(paged\s*attention)': 'PagedAttention',
            r'(sparse attention)': 'SparseAttention',
            r'(sliding window attention)': 'SlidingWindowAttention',
            r'(quantization|quantisation)': 'Quantization',
            r'(pruning)': 'Pruning',
            r'(knowledge distillation)': 'KnowledgeDistillation',
            r'(linear attention)': 'LinearAttention',
            r'(grouped query attention|gqa)': 'GroupedQueryAttention',
            r'(multi.head latent attention|mla)': 'MultiHeadLatentAttention',
            r'(prefix caching)': 'PrefixCaching',
            r'(continuous batching)': 'ContinuousBatching',
            r'(tensor parallelism)': 'TensorParallelism',
            r'(pipeline parallelism)': 'PipelineParallelism',
        }

        # Auto-seed new concepts not yet in KG
        # Use a fresh set based on current saved keys to avoid defaultdict pollution
        _saved_keys = set()
        try:
            import json as _j
            import glob as _g
            _files = _g.glob('knowledge_graph/graph.json') or _g.glob('*/knowledge_graph/graph.json')
            if _files:
                _saved_keys = set(_j.load(open(_files[0])).keys())
        except Exception:
            pass
        if not _saved_keys:
            _saved_keys = set(k for k, v in self.graph.items() if len(v) > 0)
        
        all_text = ' '.join(t for t in texts[:1000] if t).lower()
        _newly_added = []
        for pattern, concept in AUTO_CONCEPTS.items():
            if concept not in _saved_keys:
                if _re.search(pattern, all_text):
                    self.add_relation(concept, 'related_to', 'TransformerEfficiency')
                    _saved_keys.add(concept)
                    _newly_added.append(concept)
        if _newly_added:
            new_relations += len(_newly_added)

        new_relations = 0
        for text in texts:
            if not text or len(text) < 20:
                continue
            s = text.lower()

            # Normalize concepts
            def find_concept(phrase):
                phrase = phrase.strip().lower()
                for alias, canonical in CONCEPT_MAP.items():
                    if alias in phrase or phrase in alias:
                        return canonical
                return None

            for pat, rel in EFFICIENCY_PATTERNS:
                for match in re.finditer(pat, s):
                    groups = match.groups()
                    if len(groups) >= 2:
                        subj = find_concept(groups[0])
                        obj = find_concept(groups[1])
                        if subj and obj and subj != obj:
                            self.add_relation(subj, rel, obj)
                            new_relations += 1
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
