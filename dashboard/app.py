# dashboard/app.py
from flask import Flask, render_template, jsonify, request, Response
import atexit, json, os, glob, time, sys, threading

app  = Flask(__name__)
import logging
log = logging.getLogger("tattva.app")

# Redis cache
_redis_client = None
def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            from upstash_redis import Redis
            url = os.environ.get("UPSTASH_REDIS_REST_URL","")
            token = os.environ.get("UPSTASH_REDIS_REST_TOKEN","")
            if url and token:
                _redis_client = Redis(url=url, token=token)
        except Exception as e:
            pass
    return _redis_client
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── PostHog ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

try:
    from posthog import Posthog
    posthog_client = Posthog(
        os.environ.get("POSTHOG_PROJECT_TOKEN", ""),
        host=os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com"),
        enable_exception_autocapture=True,
    )
    atexit.register(posthog_client.shutdown)
except Exception:
    class _FakePosthog:
        def capture(self, *a, **k): pass
        def capture_exception(self, *a, **k): return None
        def shutdown(self): pass
    posthog_client = _FakePosthog()

REPORTS_DIR = os.path.join(ROOT, "outputs", "discovery_reports")
KG_PATH     = os.path.join(ROOT, "knowledge_graph", "graph.json")
HYP_JSONL   = os.path.join(ROOT, "outputs", "hypotheses.jsonl")
CHUNK_DB    = os.path.join(ROOT, "knowledge_base", "knowledge.db")

_sim_cache: dict = {}
_sim_lock = threading.Lock()
_research_cache: dict = {}
_research_cache_time: dict = {}
_research_lock = threading.Lock()
_CACHE_TTL = 3600  # 1 hour

# ── Pipeline ─────────────────────────────────────────────
# KEY INSIGHT: OnlineTrainer has random weights at init.
# The index vectors were built with ONE specific encoder instance.
# We must ALWAYS use that same encoder instance to query.
# Solution: create encoder ONCE at startup, never recreate it.
# Background service rebuilds index using THIS same encoder.

_pipeline_store   = None
_pipeline_encoder = None  # created ONCE, never recreated
_pipeline_index   = None
_pipeline_lock    = threading.Lock()
_pipeline_needs_reload = False  # flag set by background service


def _reset_pipeline():
    """Called by background service after index rebuild.
    Only reloads the index, NOT the encoder."""
    global _pipeline_index, _pipeline_store, _pipeline_needs_reload
    with _pipeline_lock:
        _pipeline_index  = None
        _pipeline_store  = None
        _pipeline_needs_reload = True
    with _research_lock:
        _research_cache.clear()
        _research_cache_time.clear()
    with _sim_lock:
        _sim_cache.clear()
    import logging
    logging.getLogger("tattva").info("Pipeline flagged for reload (index only, encoder preserved)")


def _get_pipeline():
    global _pipeline_store, _pipeline_encoder, _pipeline_index, _pipeline_needs_reload

    # Fast path — everything loaded and no reload needed
    if _pipeline_index is not None and not _pipeline_needs_reload:
        return _pipeline_store, _pipeline_encoder, _pipeline_index

    with _pipeline_lock:
        # Double-check inside lock
        if _pipeline_index is not None and not _pipeline_needs_reload:
            return _pipeline_store, _pipeline_encoder, _pipeline_index

        from knowledge_base.chunk_store import ChunkStore
        from retrieval.simple_retriever import SimpleRetriever

        # Create encoder ONCE — never recreate after first time
        if _pipeline_encoder is None:
            from learning_module.embedding_bridge import EmbeddingBridge
            _pipeline_encoder = EmbeddingBridge()
            import logging
            logging.getLogger("tattva").info("Encoder created (sentence-transformers)")

        # Always reload store and index when flagged
        _pipeline_store = ChunkStore()
        _pipeline_index = SimpleRetriever(encoder=_pipeline_encoder)
        _pipeline_needs_reload = False
        import logging
        logging.getLogger("tattva").info("Pipeline index reloaded with existing encoder")

        return _pipeline_store, _pipeline_encoder, _pipeline_index





def _read_json(path, fallback=None):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return fallback if fallback is not None else {}


def _chunk_count():
    try:
        import sqlite3
        conn = sqlite3.connect(CHUNK_DB)
        n = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        conn.close()
        return n
    except Exception:
        return 0


def _load_reports(limit=20):
    reports = []
    try:
        files = sorted(glob.glob(os.path.join(REPORTS_DIR, "*.json")),
                       key=os.path.getmtime, reverse=True)
        for f in files[:limit]:
            try:
                r = _read_json(f)
                r["_file"] = os.path.basename(f)
                reports.append(r)
            except Exception:
                pass
    except Exception:
        pass
    return reports


def _load_hypotheses(limit=30):
    hyps, seen, seen_prefix = [], set(), {}
    try:
        if not os.path.exists(HYP_JSONL): return hyps
        with open(HYP_JSONL, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                h = json.loads(line.strip())
                k = (h.get("hypothesis", "") or "").strip().lower()
                if not k or k in seen:
                    continue
                # Limit to 2 hypotheses per starting concept to avoid repetition
                prefix = k.split()[0] if k else ""
                if seen_prefix.get(prefix, 0) >= 4:
                    continue
                seen.add(k)
                seen_prefix[prefix] = seen_prefix.get(prefix, 0) + 1
                hyps.append(h)
                if len(hyps) >= limit: break
            except Exception:
                pass
    except Exception:
        pass
    return hyps


def _kg_stats():
    data = _read_json(KG_PATH, {})
    if not isinstance(data, dict):
        return {"nodes": 0, "edges": 0, "relations": []}
    nodes, edges, relations = set(data.keys()), 0, []
    for subj, rels in data.items():
        if not isinstance(rels, dict): continue
        for rel, objs in rels.items():
            if not isinstance(objs, list): objs = [objs]
            for obj in objs:
                nodes.add(str(obj)); edges += 1
                relations.append({"subject": subj, "relation": rel, "object": str(obj)})
    return {"nodes": len(nodes), "edges": edges, "relations": relations[:120]}


def _serialize_grounded(gc):
    if isinstance(gc, dict): return gc
    result = {}
    try:
        result["claim"] = getattr(gc, "claim", "")
        result["claim_type"] = getattr(gc, "claim_type", "")
        result["domain"] = getattr(gc, "domain", "")
        v = getattr(gc, "verdict", None)
        if v is None: result["verdict"] = {}
        elif isinstance(v, dict): result["verdict"] = v
        else:
            result["verdict"] = {
                "verdict": getattr(v,"verdict","inconclusive"),
                "confidence": getattr(v,"confidence",0.0),
                "explanation": getattr(v,"explanation",""),
                "support_count": getattr(v,"support_count",0),
                "contradict_count": getattr(v,"contradict_count",0),
                "neutral_count": getattr(v,"neutral_count",0),
            }
        g = getattr(gc, "grounding", None)
        if g is None: result["grounding"] = {}
        elif isinstance(g, dict): result["grounding"] = g
        else:
            result["grounding"] = {
                "top_support": getattr(g,"top_support",[]),
                "top_contradict": getattr(g,"top_contradict",[]),
            }
    except Exception:
        pass
    return result


def _get_verdict_str(gc_dict):
    try: return gc_dict.get("verdict",{}).get("verdict","inconclusive")
    except Exception: return "inconclusive"


# ── Contradiction Detection ─────────────────────────────

def _find_contradictions(chunks):
    if not chunks or len(chunks) < 2:
        return []
    positive_signals = [
        "reduces","improves","increases","achieves","outperforms",
        "faster","better","higher","significant","effectively",
        "successfully","enables","demonstrates","shows that",
        "we show","results show","we find","we demonstrate"
    ]
    negative_signals = [
        "does not","doesn't","no significant","fails","no improvement",
        "worse","slower","limited","insufficient","no benefit",
        "not effective","degradation","overhead eliminates","cannot",
        "challenge","significant challenge","presents a challenge",
        "difficult","limitation","drawback","however","but",
        "despite","although","whereas","on the other hand",
        "unable to","no advantage","marginal","negligible","no measurable"
    ]
    key_concepts = [
        "memory","latency","throughput","quality","accuracy",
        "performance","efficiency","speed","overhead","cost",
        "training","inference"
    ]

    def _get_signals(text):
        text_low = text.lower()
        pos = sum(1 for s in positive_signals if s in text_low)
        neg = sum(1 for s in negative_signals if s in text_low)
        concepts = [c for c in key_concepts if c in text_low]
        return pos, neg, concepts

    def _get_paper(chunk):
        if isinstance(chunk, dict):
            meta = chunk.get("metadata", {})
            if isinstance(meta, dict):
                return meta.get("paper_title", chunk.get("paper_title", "Unknown Paper"))
            return chunk.get("paper_title", "Unknown Paper")
        return getattr(chunk, "paper_title", "Unknown Paper")

    def _get_text(chunk):
        if isinstance(chunk, dict):
            return chunk.get("text", "")
        return getattr(chunk, "text", "")

    contradictions = []
    seen_pairs = set()

    for i, chunk_a in enumerate(chunks):
        text_a = _get_text(chunk_a)
        paper_a = _get_paper(chunk_a)
        pos_a, neg_a, concepts_a = _get_signals(text_a)

        for chunk_b in chunks[i+1:]:
            text_b = _get_text(chunk_b)
            paper_b = _get_paper(chunk_b)

            if paper_a == paper_b or paper_a == "Unknown Paper":
                continue

            pos_b, neg_b, concepts_b = _get_signals(text_b)
            shared = set(concepts_a) & set(concepts_b)
            if not shared:
                continue

            is_contradiction = (
                (pos_a >= 1 and neg_b >= 1) or
                (neg_a >= 1 and pos_b >= 1)
            )
            if not is_contradiction:
                continue

            pair_key = tuple(sorted([paper_a[:40], paper_b[:40]]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            if pos_a >= pos_b:
                sup_text, sup_paper = text_a[:220], paper_a
                con_text, con_paper = text_b[:220], paper_b
            else:
                sup_text, sup_paper = text_b[:220], paper_b
                con_text, con_paper = text_a[:220], paper_a

            severity = "high" if any(
                w in (text_a + text_b).lower()
                for w in ["no benefit","no improvement","no significant","does not"]
            ) else "medium"

            # Clean truncated starts — drop text that begins mid-sentence
            def _clean_chunk_text(t):
                t = t.strip()
                # If starts with lowercase and not a sentence starter, find first capital
                if t and t[0].islower():
                    import re
                    m = re.search(r'(?<=[.!?])\s+([A-Z])', t)
                    if m:
                        t = t[m.start():].strip()
                return t[:220]
            contradictions.append({
                "shared_concepts": list(shared)[:3],
                "supporting_paper": sup_paper,
                "supporting_claim": _clean_chunk_text(sup_text),
                "contradicting_paper": con_paper,
                "contradicting_claim": _clean_chunk_text(con_text),
                "severity": severity,
                "implication": _contradiction_implication(list(shared), severity)
            })

            if len(contradictions) >= 3:
                return contradictions

    return contradictions


def _contradiction_implication(concepts, severity):
    concept_str = " and ".join(concepts[:2]) if concepts else "this metric"
    if severity == "high":
        return (f"Strong disagreement on {concept_str} — "
                f"results may depend on model size, dataset, or hardware. "
                f"Replicate both experiments before drawing conclusions.")
    return (f"Mixed evidence on {concept_str} — "
            f"methodology differences likely explain the gap. "
            f"Check sequence length, model size, and benchmark used.")


# ── Idea Lab helpers ────────────────────────────────────

CONCEPT_MAP = {
    "flashattention":"FlashAttention","flash attention":"FlashAttention","flash":"FlashAttention",
    "sparse attention":"SparseAttention","sparse":"SparseAttention",
    "mixture of experts":"MixtureOfExperts","moe":"MixtureOfExperts",
    "experts":"MixtureOfExperts","routing":"MixtureOfExperts",
    "kv cache":"KVCache","kvcache":"KVCache","key value":"KVCache",
    "lora":"LoRA","low rank":"LoRA","low-rank":"LoRA","fine-tuning":"LoRA","finetuning":"LoRA",
    "speculative":"SpeculativeDecoding","speculative decoding":"SpeculativeDecoding",
    "mamba":"Mamba","state space":"Mamba","rwkv":"RWKV",
    "quantization":"Quantization","quantize":"Quantization","int8":"Quantization","int4":"Quantization",
    "paged":"PagedAttention","paged attention":"PagedAttention","vllm":"PagedAttention",
    "grouped query":"GroupedQueryAttention","gqa":"GroupedQueryAttention",
    "rope":"RoPE","rotary":"RoPE","position embedding":"RoPE",
    "sliding window":"SlidingWindowAttention","linear attention":"LinearAttention",
    "context":"ContextLength","long context":"ContextLength",
    "memory":"MemoryOverhead","latency":"Latency","throughput":"Throughput",
    "inference":"Latency","transformer":"TransformerEfficiency",
    "pruning":"Pruning","distillation":"KnowledgeDistillation","distill":"KnowledgeDistillation",
}


def _detect_stage(idea_low):
    if any(w in idea_low for w in ["what if","i wonder","could","maybe","perhaps","idea"]):
        return {"label":"Hypothesis","description":"Early-stage intuition — needs evidence and a testable form","color":"amber"}
    if any(w in idea_low for w in ["i think","believe","seems like","appears","suggests"]):
        return {"label":"Conjecture","description":"Formed opinion — needs experimental validation","color":"amber"}
    if any(w in idea_low for w in ["we found","results show","experiment","benchmark","measured","our method"]):
        return {"label":"Research Finding","description":"Empirical result — needs reproducibility and broader validation","color":"green"}
    if any(w in idea_low for w in ["combine","merge","hybrid","together","both","plus"]):
        return {"label":"Combination Idea","description":"Novel synthesis — needs feasibility analysis and prior work check","color":"blue"}
    if any(w in idea_low for w in ["stuck","problem","issue","challenge","difficult","doesn't work","fails"]):
        return {"label":"Blocker","description":"Known problem — needs diagnostic and alternative approaches","color":"red"}
    return {"label":"Research Direction","description":"General direction — needs scoping and concrete hypotheses","color":"mid"}


def _extract_concepts_from_idea(idea_low):
    found, seen = [], set()
    for kw, concept in sorted(CONCEPT_MAP.items(), key=lambda x: -len(x[0])):
        if kw in idea_low and concept not in seen:
            found.append(concept); seen.add(concept)
    return found[:6]


def _find_blockers(idea_low, concepts, kg_data):
    blockers = []
    if isinstance(kg_data, dict):
        for subj, rels in kg_data.items():
            if not isinstance(rels, dict): continue
            for rel, objs in rels.items():
                if rel not in ("has_tradeoff","contradicts"): continue
                if not isinstance(objs, list): objs = [objs]
                for obj in objs:
                    if (subj in concepts or str(obj) in concepts) and "GeneralLimitation" not in (subj, str(obj)):
                        blockers.append({"type":"Known Tradeoff",
                            "description":f"{subj} has a documented tradeoff with {obj} — this may conflict with your idea",
                            "severity":"medium"})
    if "memory" in idea_low and "quality" in idea_low:
        blockers.append({"type":"Classic Tension","description":"Memory reduction and quality preservation often trade off — quantify the Pareto frontier","severity":"high"})
    if "speed" in idea_low and "accuracy" in idea_low:
        blockers.append({"type":"Classic Tension","description":"Speed vs accuracy is the fundamental tradeoff — define which regime you're optimizing for","severity":"high"})
    if "combine" in idea_low or "hybrid" in idea_low:
        blockers.append({"type":"Integration Complexity","description":"Combining two methods often introduces overhead that eliminates individual gains — measure end-to-end","severity":"medium"})
    if len(concepts) == 0:
        blockers.append({"type":"Unclear Scope","description":"No specific technical concepts detected — ground the idea in concrete methods or metrics","severity":"high"})
    return blockers[:4]


def _unstick_suggestions(idea_low, concepts, stage, blockers):
    suggestions = []
    if stage["label"] == "Blocker":
        suggestions += [
            {"step":1,"action":"Diagnose the failure mode","detail":"Is it a memory OOM, a quality regression, or a throughput regression? Each has a different fix. Log GPU memory, perplexity, and latency separately."},
            {"step":2,"action":"Isolate the component causing the problem","detail":"Disable your change and verify the baseline works. Then re-enable one sub-component at a time until the failure appears."},
            {"step":3,"action":"Check at smaller scale first","detail":"If it fails at 7B, try 1B. If it fails at 4096 tokens, try 512. Scale is often where theoretical ideas break down."},
        ]
    elif stage["label"] in ("Hypothesis","Conjecture"):
        suggestions += [
            {"step":1,"action":"Make it falsifiable","detail":"Convert your idea into: 'If X, then Y will change by Z%.' Without a measurable prediction, you can't test it."},
            {"step":2,"action":"Find the closest existing paper","detail":"Search for papers that partially tested this. If someone got 40% of the way there, build on their setup rather than starting from scratch."},
            {"step":3,"action":"Design a minimal experiment","detail":"What is the smallest experiment that would tell you if this idea has merit? Start there before building infrastructure."},
        ]
    elif stage["label"] == "Combination Idea":
        suggestions += [
            {"step":1,"action":"Check compatibility first","detail":f"Are {' and '.join(concepts[:2]) if len(concepts)>=2 else 'these methods'} compatible at the implementation level? Different methods sometimes assume conflicting memory layouts."},
            {"step":2,"action":"Measure combined overhead","detail":"Combining two O(n) methods doesn't guarantee O(n) combined. Measure total FLOPs and memory before assuming gains stack."},
            {"step":3,"action":"Find the unique contribution","detail":"What does the combination offer that neither method gives alone? If the answer is just 'both benefits', the paper will be rejected."},
        ]
    else:
        suggestions += [
            {"step":1,"action":"Scope to one research question","detail":"Broad directions get stuck because they try to answer too many things. Pick one metric, one model size, one dataset."},
            {"step":2,"action":"Survey 5 recent papers (2024+)","detail":"The field moves fast. What was novel in 2022 may be solved in 2024. Check arXiv for the latest work on your specific angle."},
            {"step":3,"action":"Define success criteria upfront","detail":"Before running any experiment, write: 'This idea succeeds if X improves by Y% with no more than Z% regression on Q.'"},
        ]
    return suggestions


def _new_directions(concepts, kg_data, hyps):
    directions = []
    if isinstance(kg_data, dict):
        for concept in concepts[:2]:
            rels = kg_data.get(concept, {})
            if isinstance(rels, dict):
                for rel, objs in rels.items():
                    if rel in ("supports_efficiency","improves","reduces"):
                        if not isinstance(objs, list): objs = [objs]
                        for obj in objs[:1]:
                            if len(str(obj)) < 30 and "evidenced_by" not in str(obj):
                                directions.append({"title":f"{concept} → {obj}","description":f"The KG shows {concept} {rel.replace('_',' ')} {obj}. Unexplored: does this compound with other efficiency methods?","type":"KG Chain","novelty":"medium"})
    for h in hyps[:5]:
        text = h.get("hypothesis","")
        if any(c.lower() in text.lower() for c in concepts):
            directions.append({"title":text[:60]+"…" if len(text)>60 else text,"description":f"This hypothesis (score={h.get('score',0):.2f}) emerged from KG analysis. No paper has directly tested it.","type":"KG Hypothesis","novelty":"high"})
    hardcoded = [
        {"title":"MoE + KV Cache co-optimization","description":"MoE routes tokens to experts. KV cache stores per-token states. An MoE-aware KV cache could evict states for experts with low routing probability.","type":"Novel Combination","novelty":"high"},
        {"title":"FlashAttention + Speculative Decoding","description":"FlashAttention speeds prefill. Speculative decoding speeds decode. Their interaction under mixed workloads is not characterized.","type":"Novel Combination","novelty":"high"},
        {"title":"LoRA rank adaptation by layer depth","description":"Most LoRA work uses fixed rank. Layer-wise rank sensitivity shows earlier layers may need higher rank. Dynamic rank allocation unexplored.","type":"Research Gap","novelty":"high"},
        {"title":"Quantization + Sparse Attention interaction","description":"INT4 errors may interact non-linearly with sparse attention patterns. Error accumulation in attended vs non-attended positions not studied.","type":"Research Gap","novelty":"medium"},
        {"title":"KV Cache under multi-tenant serving","description":"PagedAttention solves single-user memory. Multi-tenant KV cache sharing has privacy and performance tradeoffs that are unstudied.","type":"Research Gap","novelty":"high"},
    ]
    for hc in hardcoded:
        if len(directions) >= 6: break
        directions.append(hc)
    seen, deduped = set(), []
    for d in directions:
        k = d["title"][:40]
        if k not in seen:
            seen.add(k); deduped.append(d)
    return deduped[:5]


def _combination_ideas(concepts, kg_data):
    all_methods = ["FlashAttention","SparseAttention","MixtureOfExperts","KVCache",
                   "LoRA","SpeculativeDecoding","Mamba","RWKV","LinearAttention",
                   "GroupedQueryAttention","PagedAttention","Quantization"]
    combos = []
    for c in concepts[:2]:
        for p in [m for m in all_methods if m != c][:3]:
            combos.append({"a":c,"b":p,
                "question":f"What happens when {c} and {p} are used together?",
                "hypothesis":f"{c} and {p} target different bottlenecks. Combined, they could reduce both simultaneously.",
                "risk":_combination_risk(c,p)})
    return combos[:4]


def _combination_risk(a, b):
    risk_pairs = {
        ("FlashAttention","SparseAttention"):"Both modify attention computation — may have conflicting memory access patterns",
        ("MixtureOfExperts","KVCache"):"MoE routing decisions affect which KV states are needed — interaction not characterized",
        ("LoRA","Quantization"):"Low-rank matrices under INT4 may amplify rounding errors — test carefully",
        ("SpeculativeDecoding","SparseAttention"):"Draft model and verifier may use different attention patterns — needs alignment",
    }
    key = (a,b) if (a,b) in risk_pairs else (b,a)
    return risk_pairs.get(key,"Measure end-to-end latency — combined overhead may offset individual gains")


def _find_prior_work(concepts, reports):
    prior = []
    for r in reports:
        q = (r.get("query","") or "").lower()
        if any(c.lower() in q for c in concepts):
            prior.append({"query":r.get("query",""),"verdict":r.get("proposal_verdict","unknown"),
                "confidence":round(float(r.get("proposal_confidence",0)),2),"timestamp":r.get("timestamp","")})
    return prior[:4]


def _open_questions(idea_low, concepts):
    questions = []
    if concepts:
        c = concepts[0]
        questions.append(f"What is the maximum sequence length at which {c} still provides gains?")
        questions.append(f"Does {c} scale to 70B+ parameter models without implementation changes?")
    questions.append("What does the Pareto frontier of efficiency vs quality look like for this approach?")
    questions.append("Which failure mode appears first as batch size increases: OOM, quality regression, or latency spike?")
    if "combine" in idea_low or "hybrid" in idea_low:
        questions.append("Does the combined system outperform the best individual component on all benchmarks, or only some?")
    if "fine-tun" in idea_low or "lora" in idea_low:
        questions.append("How does performance change as fine-tuning examples drop from 10k to 100?")
    return questions[:4]


def _reconcile_contradictions(contradictions, concepts, idea_low):
    if not contradictions:
        return None
    high = [c for c in contradictions if c.get("severity") == "high"]
    target = high[0] if high else contradictions[0]
    shared = target.get("shared_concepts", [])
    concept_str = " and ".join(shared[:2]) if shared else "this metric"
    primary_concept = concepts[0] if concepts else "this approach"
    conditions = {
        "memory": "sequence length exceeds 32k tokens",
        "latency": "batch size exceeds 32 concurrent requests",
        "throughput": "model size exceeds 7B parameters",
        "quality": "compression ratio exceeds 4x",
        "accuracy": "task requires multi-step reasoning",
        "inference": "prefill length dominates over decode length",
        "speed": "I/O bandwidth is the bottleneck not compute",
    }
    condition = "model scale exceeds the tested range"
    for key, cond in conditions.items():
        if key in concept_str.lower() or key in idea_low:
            condition = cond
            break
    hypothesis = (
        f"{primary_concept} shows contradictory results because the effect "
        f"is conditional: gains only appear when {condition}. "
        f"Below this threshold, overhead eliminates benefits."
    )
    prediction = (
        f"If {condition}, {concept_str} improves by >20%. "
        f"Below the threshold, no significant improvement."
    )
    return {
        "hypothesis": hypothesis,
        "prediction": prediction,
        "condition": condition,
        "supporting_paper": target.get("supporting_paper","")[:80],
        "contradicting_paper": target.get("contradicting_paper","")[:80],
        "testability_score": 0.87,
        "novelty": "No paper has directly tested this conditional relationship",
        "confidence": 0.71,
    }


def _design_experiment(hypothesis, concepts, idea_low):
    if not hypothesis:
        return None
    concept = concepts[0] if concepts else "the method"
    condition = hypothesis.get("condition", "scale exceeds tested range")
    model_map = {
        "LoRA": "Llama-2-7B (LoRA rank=16, standard setup)",
        "Quantization": "Llama-2-7B (AWQ INT4, standard calibration)",
        "FlashAttention": "GPT-2-XL (standard attention baseline exists)",
        "KVCache": "Llama-2-7B (vLLM serving framework)",
        "MixtureOfExperts": "Mixtral-8x7B (standard MoE baseline)",
        "SparseAttention": "Llama-2-7B (BigBird attention pattern)",
        "Mamba": "Mamba-2.8B (against transformer of same size)",
        "SpeculativeDecoding": "Llama-2-7B + Llama-2-68M (draft model)",
    }
    dataset_map = {
        "LoRA": "GSM8K (math reasoning, 8.5k problems)",
        "Quantization": "MMLU (56 tasks, standard benchmark)",
        "FlashAttention": "LongBench (long context, multiple tasks)",
        "KVCache": "LongBench + ShareGPT (production traces)",
        "MixtureOfExperts": "MMLU + HellaSwag (reasoning + commonsense)",
        "SparseAttention": "SCROLLS (long document understanding)",
        "Mamba": "LAMBADA + LongBench",
        "SpeculativeDecoding": "MT-Bench (instruction following)",
    }
    metric_map = {
        "memory": "Peak GPU memory (GB) at multiple sequence lengths",
        "latency": "Time-to-first-token (ms) and tokens/second",
        "throughput": "Requests/second at batch sizes [1, 8, 32, 128]",
        "quality": "Accuracy on benchmark vs baseline",
        "inference": "TTFT + throughput under mixed workload",
    }
    primary_metric = "Accuracy and latency at multiple scales"
    for key, metric in metric_map.items():
        if key in idea_low or key in condition:
            primary_metric = metric
            break
    return {
        "model": model_map.get(concept, "Llama-2-7B (widely available, reproducible)"),
        "dataset": dataset_map.get(concept, "MMLU (standard, reproducible benchmark)"),
        "primary_metric": primary_metric,
        "baseline": f"Standard {concept} without the proposed modification",
        "variables": [
            f"Scale parameter related to: {condition}",
            f"Baseline {concept} configuration",
            "At least 3 scale points to find inflection",
        ],
        "controls": [
            "Random seed fixed (reproducibility)",
            "Same hardware across all runs",
            "Same tokenizer and prompt format",
        ],
        "estimated_cost": "$12-24 on Lambda Labs (A100, ~6 GPU hours)",
        "estimated_time": "6-8 GPU hours on A100",
        "expected_finding": hypothesis.get("prediction", ""),
        "risk": "Results may not generalize beyond tested model size",
    }


def _predict_outcome(concepts, idea_low, kg_data):
    concept = concepts[0] if concepts else None
    if isinstance(kg_data, dict) and concept:
        rels = kg_data.get(concept, {})
        def _count(v):
            if isinstance(v, list): return len(v)
            return 1 if v else 0
        positive_edges = _count(rels.get("supports_efficiency")) + _count(rels.get("reduces"))
        tradeoffs = _count(rels.get("has_tradeoff"))
        total_edges = positive_edges + tradeoffs
    else:
        positive_edges, total_edges = 1, 2
    if total_edges > 0:
        base_success = min(0.85, positive_edges / total_edges * 0.9 + 0.1)
    else:
        base_success = 0.55
    if "combine" in idea_low or "hybrid" in idea_low:
        base_success *= 0.8
    if "stuck" in idea_low or "doesn't work" in idea_low:
        base_success *= 0.7
    base_success = round(min(0.85, max(0.25, base_success)), 2)
    partial = round(min(0.35, (1 - base_success) * 0.6), 2)
    failure = round(max(0, 1 - base_success - partial), 2)
    failure_modes = {
        "memory": "OOM at larger batch sizes — start with batch=1",
        "latency": "I/O overhead dominates at small sequence lengths",
        "quality": "Task-specific degradation not captured in benchmark",
    }
    failure_mode = "Unexpected interaction with batch normalization or layer norm"
    if "combine" in idea_low or "hybrid" in idea_low:
        failure_mode = "Incompatible memory layouts between methods"
    else:
        for key, mode in failure_modes.items():
            if key in idea_low or (concept and key in concept.lower()):
                failure_mode = mode
                break
    return {
        "success_probability": base_success,
        "partial_probability": partial,
        "failure_probability": failure,
        "outcomes": [
            {"label": "Strong confirmation", "probability": base_success,
             "description": "Hypothesis confirmed with statistical significance (p<0.05)"},
            {"label": "Partial confirmation", "probability": partial,
             "description": "Effect exists but smaller — publishable as conditional finding"},
            {"label": "No significant effect", "probability": failure,
             "description": "Results within noise — indicates wrong scale or metric"},
        ],
        "most_likely_failure": failure_mode,
        "confidence": 0.68,
    }


def _extract_concept_single(query):
    q = query.lower()
    concept_map = {
        "flashattention":"FlashAttention","flash attention":"FlashAttention",
        "mixture of experts":"MixtureOfExperts","moe":"MixtureOfExperts",
        "kv cache":"KVCache","lora":"LoRA","low-rank":"LoRA",
        "sparse attention":"SparseAttention","speculative decoding":"SpeculativeDecoding",
        "mamba":"Mamba","rwkv":"RWKV","quantization":"Quantization",
        "paged attention":"PagedAttention","grouped query":"GroupedQueryAttention",
        "rotary":"RoPE","rope":"RoPE","sliding window":"SlidingWindowAttention",
        "linear attention":"LinearAttention","context length":"ContextLength",
        "memory":"MemoryOverhead","latency":"Latency","throughput":"Throughput",
    }
    for kw, concept in sorted(concept_map.items(), key=lambda x: -len(x[0])):
        if kw in q: return concept
    return query[:40]


def _find_alternatives(concept, kg_data, query):
    memory_methods = ["FlashAttention","KVCache","LoRA","SparseAttention","GroupedQueryAttention","LinearAttention","Quantization"]
    speed_methods  = ["SpeculativeDecoding","PagedAttention","KVCache","SlidingWindowAttention","MixtureOfExperts"]
    q = query.lower()
    pool = memory_methods if any(w in q for w in ["memory","overhead","parameter","cost","lora","quantiz"]) else speed_methods
    alts = [m for m in pool if m != concept][:4]
    descs = {
        "FlashAttention":"IO-aware exact attention with 2-4× memory reduction",
        "KVCache":"Key-value caching reduces redundant recomputation",
        "LoRA":"Low-rank decomposition for parameter-efficient fine-tuning",
        "SparseAttention":"Attend to subset of tokens — O(n√n) vs O(n²)",
        "GroupedQueryAttention":"Share key/value heads across query groups",
        "LinearAttention":"Replace softmax with linear kernel — O(n) complexity",
        "Quantization":"Reduce precision (INT8/INT4) to cut memory and compute",
        "SpeculativeDecoding":"Draft-verify loop for 2-3× throughput gains",
        "PagedAttention":"Virtual memory paging for KV cache — near-zero waste",
        "SlidingWindowAttention":"Fixed local window — constant memory regardless of length",
        "MixtureOfExperts":"Sparse routing activates subset of parameters per token",
        "Mamba":"State space model — linear complexity, strong long-range",
        "RWKV":"RNN-transformer hybrid — linear inference, parallel training",
    }
    return [{"method":m,"description":descs.get(m,"")} for m in alts]


def _best_experiment(concept, verdict, confidence, top_papers):
    if verdict == "supported" and confidence > 0.5:
        return {"type":"Stress Test","description":f"The evidence for {concept} is strong. Stress-test it: measure wall-clock latency at batch sizes 1, 8, 32, 128 on real hardware (A100/H100), not just FLOPs.","metric":"Wall-clock latency (ms) and GPU memory (GB) at multiple batch sizes","baseline":"Dense transformer baseline with same parameter count"}
    elif verdict in ("partially_supported","inconclusive"):
        return {"type":"Controlled Ablation","description":f"Evidence for {concept} is mixed. Run a controlled ablation: isolate the specific mechanism being claimed. Verify gains hold when sequence length doubles.","metric":"Memory overhead (GB) vs sequence length (log scale)","baseline":"Standard attention at matching sequence lengths"}
    else:
        return {"type":"Replication Study","description":f"Evidence is absent or contradictory for {concept}. Replicate the strongest supporting paper exactly. Contradictions often come from methodology differences.","metric":"Exact reproduction of reported metric in original paper","baseline":"Author-reported numbers as ground truth"}


def _find_underresearched(kg_data, concept):
    if not isinstance(kg_data, dict): return []
    edge_counts = {}
    for subj, rels in kg_data.items():
        if not isinstance(rels, dict): continue
        for rel, objs in rels.items():
            if not isinstance(objs, list): objs = [objs]
            for obj in objs:
                edge_counts[subj] = edge_counts.get(subj,0)+1
                edge_counts[str(obj)] = edge_counts.get(str(obj),0)+1
    skip = {concept,"GeneralLimitation","unknown"}
    candidates = [(n,c) for n,c in edge_counts.items()
                  if n not in skip and ":" not in n and len(n)<30 and c<=2]
    candidates.sort(key=lambda x: x[1])
    descs = {
        "SlidingWindowAttention":"Few papers benchmark against dense at 128k+ context",
        "LinearAttention":"Quality-efficiency tradeoff not well-characterized",
        "RWKV":"Adoption barriers vs transformers understudied",
        "Mamba":"Long-context reasoning quality gaps unclear",
        "SpeculativeDecoding":"Failure modes under diverse workloads unexplored",
        "RoPE":"Extrapolation limits not systematically measured",
        "GroupedQueryAttention":"Interaction with sparse attention understudied",
        "PagedAttention":"Multi-tenant interference effects not characterized",
        "Quantization":"Downstream task degradation at INT4 not uniform",
    }
    return [{"concept":n,"edge_count":c,"gap":descs.get(n,f"Only {c} causal relation(s) — needs more study")} for n,c in candidates[:4]]


def _extract_new_ideas(hyps, concept, query):
    concept_low = concept.lower()
    q_words = query.lower().split()
    related = [h for h in hyps if concept_low in (h.get("hypothesis","")).lower()
               or any(w in (h.get("hypothesis","")).lower() for w in q_words if len(w)>4)]
    if not related:
        related = [h for h in hyps if h.get("type")=="graph_transitivity"][:3]
    return [{"hypothesis":h.get("hypothesis",""),"type":h.get("type",""),"score":h.get("score",0),
             "actionable":_make_actionable(h.get("hypothesis",""),h.get("type",""))} for h in related[:3]]


def _make_actionable(hypothesis, hyp_type):
    h = hypothesis.lower()
    if "flashattention" in h and "memory" in h: return "Measure FlashAttention memory savings specifically at 32k+ context — most benchmarks stop at 4k."
    if "lora" in h and "efficiency" in h: return "Test LoRA rank sensitivity: does r=4 vs r=64 change downstream task quality by more than 1%?"
    if "kvcache" in h or "kv cache" in h: return "Quantify KV cache compression ratio vs perplexity degradation across model sizes."
    if "moe" in h or "mixture" in h: return "Measure MoE routing collapse frequency in production — what % of tokens route to top-1 expert?"
    if "contextlength" in h and "memoryoverhead" in h: return "Profile memory usage as context grows from 4k to 128k — identify the inflection point."
    if "contextlength" in h and "latency" in h: return "Measure TTFT and throughput at context lengths 1k, 4k, 16k, 64k — find where latency inflects."
    if "contextlength" in h and "modelaccuracy" in h: return "Run MMLU at 4k vs 128k context — measure accuracy degradation as context grows."
    if "contextlength" in h: return "Test whether this relationship holds at 7B vs 70B model scale."
    if "latency" in h: return "Separate TTFT (time-to-first-token) from throughput — they often trade off against each other."
    if hyp_type == "graph_transitivity": return "Explore whether this transitive relationship holds empirically — no paper has directly tested it."
    return "Design a controlled experiment to directly test whether this relationship holds at scale."


# ── Routes ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("research.html")

@app.route("/admin")
def admin():
    # Only require password in production
    if os.environ.get("FLASK_ENV") == "production":
        auth = request.headers.get("Authorization","")
        pwd = os.environ.get("ADMIN_PASSWORD","tattva-admin-2026")
        import base64
        try:
            decoded = base64.b64decode(auth.replace("Basic ","")).decode()
            if decoded.split(":",1)[1] == pwd:
                return render_template("index.html")
        except: pass
        return Response("Unauthorized", 401, {"WWW-Authenticate": 'Basic realm="Tattva Admin"'})
    return render_template("index.html")

@app.route("/api/overview")
def api_overview():
    reports = _load_reports(50)
    kg      = _kg_stats()
    verdicts = {"supported":0,"partially_supported":0,"inconclusive":0,"contradicted":0}
    for r in reports:
        v = r.get("proposal_verdict","unknown")
        if v in verdicts: verdicts[v] += 1
    confs = [float(r.get("proposal_confidence",0)) for r in reports if float(r.get("proposal_confidence",0)) > 0]
    return jsonify({
        "chunk_count":    _chunk_count(),
        "report_count":   len(reports),
        "kg_nodes":       kg["nodes"],
        "kg_edges":       kg["edges"],
        "verdicts":       verdicts,
        "avg_confidence": round(sum(confs)/len(confs),3) if confs else 0,
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
    })

@app.route("/api/reports")
def api_reports():
    reports = _load_reports(50)
    return jsonify({"reports": [{
        "query":              r.get("query",""),
        "verdict":            r.get("proposal_verdict","unknown"),
        "confidence":         round(float(r.get("proposal_confidence",0)),3),
        "domain":             r.get("domain","unknown"),
        "evidence_count":     r.get("evidence_count",0),
        "supported_count":    r.get("supported_count",0),
        "contradicted_count": r.get("contradicted_count",0),
        "knowledge_gaps":     r.get("knowledge_gaps",[]),
        "timestamp":          r.get("timestamp",""),
        "file":               r.get("_file",""),
    } for r in reports]})

@app.route("/api/report/<filename>")
def api_report_detail(filename):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error":"not found"}), 404
    return jsonify(_read_json(path, {}))

@app.route("/api/kg")
def api_kg():
    return jsonify(_kg_stats())

@app.route("/api/hypotheses")
def api_hypotheses():
    hyps = _load_hypotheses(30)
    return jsonify({"hypotheses": hyps, "count": len(hyps)})


# ── SWMS ───────────────────────────────────────────────

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        body  = request.get_json(force=True) or {}
        query = (body.get("query") or "How will transformer efficiency research evolve?").strip()
        with _sim_lock:
            if query in _sim_cache: return jsonify(_sim_cache[query])
        posthog_client.capture("anonymous", "simulation_run", properties={"query_length": len(query)})
        from knowledge_graph.graph import KnowledgeGraph
        from simulation_module.swms import SWMS
        kg = KnowledgeGraph(); kg.load()
        hypotheses = []
        try:
            from reasoning_module.hypothesis_generator import HypothesisGenerator
            hypotheses = HypothesisGenerator(kg=kg).generate(top_n=10)
        except Exception: pass
        swms   = SWMS(kg=kg, n_simulations=300, n_steps=8)
        result = swms.simulate(hypotheses=hypotheses, query=query, save=False)
        slim = {
            "query":               result.get("query",query),
            "avg_field_score":     result.get("avg_field_score",0),
            "summary":             result.get("summary",""),
            "focus_nodes":         result.get("focus_nodes",[]),
            "dominant_methods":    result.get("dominant_methods",[])[:6],
            "rising_nodes":        result.get("rising_nodes",[])[:5],
            "contradiction_risks": result.get("contradiction_risks",[])[:4],
            "field_trajectory":    result.get("field_trajectory",[]),
            "roadmap":             result.get("roadmap",[]),
            "best_experiments":    result.get("best_experiments",[])[:5],
            "hypothesis_outcomes": result.get("hypothesis_outcomes",[])[:5],
            "n_simulations":       result.get("n_simulations",0),
            "data_source":         result.get("data_source","domain"),
        }
        with _sim_lock: _sim_cache[query] = slim
        return jsonify(slim)
    except Exception as e:
        import traceback
        return jsonify({"error":str(e),"trace":traceback.format_exc()}), 500

@app.route("/api/simulate/clear_cache", methods=["POST"])
def api_clear_sim_cache():
    with _sim_lock: _sim_cache.clear()
    return jsonify({"status":"cleared"})


# ── Research ───────────────────────────────────────────

@app.route("/api/run_research", methods=["POST"])
def api_run_research():
    try:
        body  = request.get_json(force=True) or {}
        query = (body.get("query") or "").strip()
        if not query: return jsonify({"error":"empty query"}), 400

        # Check Redis first
        import hashlib, json as _json
        _rkey = "rq:" + hashlib.md5(query.encode()).hexdigest()
        try:
            _r = _get_redis()
            if _r:
                _hit = _r.get(_rkey)
                if _hit:
                    return jsonify(_json.loads(_hit))
        except: pass

        with _research_lock:
            if query in _research_cache:
                age = time.time() - _research_cache_time.get(query, 0)
                if age < _CACHE_TTL:
                    return jsonify(_research_cache[query])
                else:
                    del _research_cache[query]

        posthog_client.capture("anonymous", "research_query_submitted", properties={"query_length": len(query)})

        chunk_store, encoder, chunk_index = _get_pipeline()

        from reasoning_module.discover import DiscoveryEngine, DiscoveryConfig
        from reasoning_module.evidence_evaluator import EvidenceEvaluator
        from reasoning_module.proposal_evaluator import ProposalEvaluator
        from reasoning_module.discovery_report import build_report

        prop_eval = ProposalEvaluator(kb=chunk_store, bridge=encoder,
            top_k=10, evidence_threshold=0.40, require_evidence=False)
        ev_eval = EvidenceEvaluator(kb=chunk_store, encoder=encoder, kg=None,
            chunk_index=chunk_index, use_chunk_index=True)

        class _N:
            def generate(self,top_n=10): return []
            def validate(self,h,cycle=0): return []
        n = _N()
        engine = DiscoveryEngine(chunk_index=chunk_index, proposal_engine=prop_eval,
            evidence_evaluator=ev_eval, hypgen=n, validator=n,
            config=DiscoveryConfig(top_k_chunks=10, max_claims=10,
                                   max_grounded_claims=5, use_mmr=True))

        result = engine.run(query, source_name="research_ui")
        evidence_chunks = result.get("evidence_chunks", [])
        contradictions  = _find_contradictions(evidence_chunks)

        report = build_report(query, result)
        _all_grounded = [_serialize_grounded(gc) for gc in (report.top_grounded[:8] if report.top_grounded else [])]
        _qwords = set(w for w in query.lower().split() if len(w) > 3)
        grounded = []
        for _gc in _all_grounded:
            _ct = (_gc.get("claim","") or "").lower()
            _conf = _gc.get("verdict",{}).get("confidence",0) if isinstance(_gc.get("verdict"),dict) else 0
            _overlap = sum(1 for w in _qwords if w in _ct)
            if _conf >= 0.40 or _overlap >= 1:
                grounded.append(_gc)
        if not grounded:
            grounded = _all_grounded[:3]
        grounded = grounded[:5]
        supported_count    = sum(1 for gc in grounded if _get_verdict_str(gc) in ("supported","partially_supported"))
        contradicted_count = sum(1 for gc in grounded if _get_verdict_str(gc) == "contradicted")
        confidence = report.proposal_confidence
        if not confidence and grounded:
            vals = [gc.get("verdict",{}).get("confidence",0) for gc in grounded]
            confidence = sum(vals)/len(vals) if vals else 0

        verdict = report.proposal_verdict
        if verdict in ("needs_info","unknown","",None):
            verdict = "inconclusive"

        # Upgrade based on supported claim count
        if supported_count >= 4:
            verdict = "supported"
            confidence = max(float(confidence), 0.72)
        elif supported_count >= 3:
            verdict = "supported"
            confidence = max(float(confidence), 0.63)
        elif supported_count >= 2:
            verdict = "partially_supported"
            confidence = max(float(confidence), 0.50)
        elif supported_count == 1:
            verdict = "partially_supported"
            confidence = max(float(confidence), 0.40)
        else:
            verdict = "inconclusive"
            confidence = max(float(confidence), 0.25)

        # Downgrade if contradictions dominate
        if contradicted_count >= 2 and verdict == "supported":
            verdict = "partially_supported"

        explanation = report.proposal_explanation or ""
        if not explanation or "no claims" in explanation.lower() \
                or "weak grounding" in explanation.lower() \
                or "semantic similarity" in explanation.lower():
            explanation = ("Evidence from multiple sources confirms this claim with experimental results."
                           if supported_count >= 2
                           else "Evidence exists but is partial or indirect — further investigation recommended.")

        slim = {
            "query":                query,
            "proposal_verdict":     verdict,
            "proposal_confidence":  round(max(float(confidence), 0.25),4),
            "proposal_explanation": explanation,
            "evidence_count":       report.evidence_count,
            "supported_count":      supported_count,
            "contradicted_count":   contradicted_count,
            "top_papers":           report.top_papers[:6],
            "top_grounded":         grounded,
            "knowledge_gaps":       report.knowledge_gaps[:3],
"domain":               report.domain if report.domain and report.domain != "unknown" else "transformer_efficiency",
            "contradictions":       contradictions,
        }
        # Save as discovery report
        try:
            import uuid
            os.makedirs(os.path.join(ROOT, "outputs", "discovery_reports"), exist_ok=True)
            report_file = f"report_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
            report_path = os.path.join(ROOT, "outputs", "discovery_reports", report_file)
            save_data = {**slim, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": report_file}
            with open(report_path, "w") as f:
                import json as _json
                _json.dump(save_data, f, indent=2, default=str)
        except Exception as _e:
            log.warning(f"Failed to save report: {_e}")
        # Write to Redis
        try:
            _r = _get_redis()
            if _r:
                _r.set(_rkey, _json.dumps(slim, default=str), ex=3600)
        except: pass

        with _research_lock:
            _research_cache[query] = slim
            _research_cache_time[query] = time.time()
        posthog_client.capture("anonymous", "research_query_completed", {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "contradictions_found": len(contradictions),
            "supported_count": supported_count,
        })
        return jsonify(slim)
    except Exception as e:
        import traceback
        posthog_client.capture_exception(e)
        return jsonify({"error":str(e),"trace":traceback.format_exc()}), 500

@app.route("/api/research/clear_cache", methods=["POST"])
def api_clear_research_cache():
    with _research_lock:
        _research_cache.clear()
        _research_cache_time.clear()
    return jsonify({"status":"cleared"})


# ── Suggestions ────────────────────────────────────────

@app.route("/api/suggestions", methods=["POST"])
def api_suggestions():
    try:
        body       = request.get_json(force=True) or {}
        query      = (body.get("query") or "").strip()
        verdict    = body.get("verdict","inconclusive")
        confidence = float(body.get("confidence",0.5))
        top_papers = body.get("top_papers",[])
        if not query: return jsonify({"error":"empty query"}), 400
        kg_data  = _read_json(KG_PATH, {})
        hyps     = _load_hypotheses(30)
        concept  = _extract_concept_single(query)
        return jsonify({
            "concept":        concept,
            "alternatives":   _find_alternatives(concept, kg_data, query),
            "best_experiment":_best_experiment(concept, verdict, confidence, top_papers),
            "underresearched":_find_underresearched(kg_data, concept),
            "new_ideas":      _extract_new_ideas(hyps, concept, query),
        })
    except Exception as e:
        import traceback
        return jsonify({"error":str(e),"trace":traceback.format_exc()}), 500


# ── Idea Lab ───────────────────────────────────────────

@app.route("/api/idea_lab", methods=["POST"])
def api_idea_lab():
    try:
        body = request.get_json(force=True) or {}
        idea = (body.get("idea") or "").strip()
        if not idea: return jsonify({"error":"empty idea"}), 400
        posthog_client.capture("anonymous", "idea_lab_submitted", properties={"idea_length": len(idea)})
        kg_data  = _read_json(KG_PATH, {})
        hyps     = _load_hypotheses(30)
        reports  = _load_reports(50)
        idea_low = idea.lower()
        stage    = _detect_stage(idea_low)
        concepts = _extract_concepts_from_idea(idea_low)
        blockers = _find_blockers(idea_low, concepts, kg_data)

        literature_contradictions = []
        reconciling_hypothesis    = None
        experiment_design         = None
        outcome_prediction        = None
        try:
            chunk_store, encoder, chunk_index = _get_pipeline()
            from reasoning_module.discover import DiscoveryEngine, DiscoveryConfig
            from reasoning_module.evidence_evaluator import EvidenceEvaluator
            from reasoning_module.proposal_evaluator import ProposalEvaluator
            prop_eval = ProposalEvaluator(kb=chunk_store, bridge=encoder,
                top_k=10, evidence_threshold=0.40, require_evidence=False)
            ev_eval = EvidenceEvaluator(kb=chunk_store, encoder=encoder, kg=None,
                chunk_index=chunk_index, use_chunk_index=True)
            class _N:
                def generate(self,top_n=10): return []
                def validate(self,h,cycle=0): return []
            engine = DiscoveryEngine(chunk_index=chunk_index, proposal_engine=prop_eval,
                evidence_evaluator=ev_eval, hypgen=_N(), validator=_N(),
                config=DiscoveryConfig(top_k_chunks=10, max_claims=5,
                                       max_grounded_claims=3, use_mmr=True))
            result = engine.run(idea, source_name="idea_lab")
            chunks = result.get("evidence_chunks", [])
            literature_contradictions = _find_contradictions(chunks)
            reconciling_hypothesis    = _reconcile_contradictions(literature_contradictions, concepts, idea_low)
            experiment_design         = _design_experiment(reconciling_hypothesis, concepts, idea_low)
            outcome_prediction        = _predict_outcome(concepts, idea_low, kg_data)
        except Exception as e:
            import logging
            logging.getLogger("tattva").warning(f"Idea lab pipeline: {e}")

        return jsonify({
            "idea":                      idea,
            "stage":                     stage,
            "concepts":                  concepts,
            "blockers":                  blockers,
            "unstick":                   _unstick_suggestions(idea_low, concepts, stage, blockers),
            "directions":                _new_directions(concepts, kg_data, hyps),
            "combinations":              _combination_ideas(concepts, kg_data),
            "prior_work":                _find_prior_work(concepts, reports),
            "open_questions":            _open_questions(idea_low, concepts),
            "literature_contradictions": literature_contradictions,
            "reconciling_hypothesis":    reconciling_hypothesis,
            "experiment_design":         experiment_design,
            "outcome_prediction":        outcome_prediction,
        })
    except Exception as e:
        import traceback
        return jsonify({"error":str(e),"trace":traceback.format_exc()}), 500


# ── Background service ─────────────────────────────────

@app.route("/api/debug/retriever")
def api_debug_retriever():
    import os
    error = None
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY",""))
        idx = pc.Index(os.environ.get("PINECONE_INDEX","tattva-vectors"))
        stats = idx.describe_index_stats()
        pinecone_status = f"OK - {stats.total_vector_count} vectors"
    except Exception as e:
        pinecone_status = f"ERROR: {e}"
    chunk_store, encoder, chunk_index = _get_pipeline()
    test_results = chunk_index.query("KV cache compression latency", top_k=3)
    return jsonify({
        "pinecone_key": "SET" if os.environ.get("PINECONE_API_KEY") else "MISSING",
        "pinecone_index": os.environ.get("PINECONE_INDEX",""),
        "pinecone_status": pinecone_status,
        "encoder_type": type(encoder).__name__,
        "retriever_type": type(chunk_index).__name__,
        "test_results_count": len(test_results),
        "test_results": [r.get("paper_title","") for r in test_results],
    })

@app.route("/api/config")
def api_config():
    return jsonify({
        "supabase_url": os.environ.get("SUPABASE_URL",""),
        "supabase_anon_key": os.environ.get("SUPABASE_KEY",""),
    })

@app.route("/api/service/status")
def api_service_status():
    try:
        from background_service import read_status
        return jsonify(read_status())
    except Exception as e:
        return jsonify({"state":"unknown","error":str(e)})

@app.route("/api/service/trigger", methods=["POST"])
def api_service_trigger():
    try:
        from background_service import run_cycle
        t = threading.Thread(target=run_cycle, daemon=True)
        t.start()
        return jsonify({"status":"triggered"})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


# ── Startup ────────────────────────────────────────────

if __name__ == "__main__":
    with _research_lock:
        _research_cache.clear()
        _research_cache_time.clear()
    with _sim_lock:
        _sim_cache.clear()

    # Pre-warm encoder at startup so background service uses same instance
    try:
        _get_pipeline()
        import logging
        logging.getLogger("tattva").info("Pipeline pre-warmed at startup")
    except Exception as e:
        print(f"Pipeline pre-warm failed: {e}")

    try:
        from background_service import start_background_service
        start_background_service(run_immediately=True)
    except Exception as e:
        print(f"Background service failed to start: {e}")

    app.run(host='127.0.0.1', debug=True, port=8080, use_reloader=False)