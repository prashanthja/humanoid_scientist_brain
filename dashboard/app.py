# dashboard/app.py
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
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

# Research dimension detection for context bucketing
_DIMENSIONS = {
    "training_cost":  ["training","fine-tuning","fine tuning","gradient","trainable","adapter","lora","finetuning","pretrain"],
    "inference_cost": ["inference","serving","decoding","generation","throughput","tokens per second","batch","vllm","tgi","tpot"],
    "memory_usage":   ["memory","vram","gpu memory","kv cache","kv-cache","memory footprint","memory overhead","oom","memory bandwidth"],
    "latency":        ["latency","speed","faster","slower","millisecond","wall-clock","time-to-first-token","ttft","response time"],
    "accuracy":       ["accuracy","perplexity","benchmark","mmlu","quality","f1","bleu","rouge","performance","degradation","loss"],
    "scalability":    ["scale","scaling","billion","70b","13b","7b","model size","distributed","parallelism","long context","context length"],
}

def _score_paper_quality(paper_title: str, source: str = "") -> float:
    """Score paper quality 0-1 based on title signals and source."""
    title_low = (paper_title or "").lower()
    score = 0.5  # baseline

    # Top venue signals boost score
    top_venues = ["neurips","icml","iclr","acl","emnlp","cvpr","iccv","eccv",
                  "nature","science","arxiv","openreview"]
    source_low = (source or "").lower()
    if any(v in source_low or v in title_low for v in top_venues):
        score += 0.2

    # Well-known paper signals
    landmark = ["flash","lora","mamba","attention is all you need","gpt","llama",
                "transformer","bert","t5","palm","gemini","claude","mixtral",
                "speculative","paged","vllm","deepspeed"]
    if any(l in title_low for l in landmark):
        score += 0.15

    # Penalty for very specific/niche titles
    if len(title_low) > 100:
        score -= 0.05

    return min(1.0, max(0.1, score))

def _extract_benchmark_context(text: str) -> dict:
    """Extract hardware, dataset, task context from chunk text."""
    import re as _re
    text_low = (text or "").lower()
    ctx = {}
    # Hardware
    hw = []
    for hw_kw in ["a100","h100","v100","a6000","rtx 3090","rtx 4090","t4","tpu","hopper","ampere","ada lovelace"]:
        if hw_kw in text_low:
            hw.append(hw_kw.upper().replace(" ","-"))
    if hw: ctx["hardware"] = hw[0]
    # Sequence length
    seq = _re.findall(r'(\d+[k])\s*(?:token|sequence|context|seq|len)', text_low)
    if not seq:
        seq = _re.findall(r'(?:sequence|context)\s+length[s]?\s+(?:of\s+)?(\d+[k]?)', text_low)
    if seq: ctx["sequence"] = seq[0].strip()
    # Task type
    for task in ["fine-tuning","inference","pretraining","language modeling","classification","generation","training"]:
        if task in text_low:
            ctx["task"] = task
            break
    # Model size
    sizes = _re.findall(r'(\d+(?:\.\d+)?[bm])\s*(?:parameter|param)', text_low)
    if sizes: ctx["model_size"] = sizes[0].upper() + " params"
    return ctx

def _detect_dimension(text: str) -> str:
    text_low = (text or "").lower()
    scores = {}
    for dim, keywords in _DIMENSIONS.items():
        score = sum(1 for kw in keywords if kw in text_low)
        if score > 0:
            scores[dim] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)

def _bucket_claims(grounded: list) -> dict:
    """Group grounded claims by research dimension."""
    buckets = {}
    for gc in grounded:
        claim_text = gc.get("claim","")
        dim = _detect_dimension(claim_text)
        gc["dimension"] = dim
        if dim not in buckets:
            buckets[dim] = []
        buckets[dim].append(gc)
    return buckets

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
        "does not reduce","does not improve","does not outperform",
        "doesn't reduce","doesn't improve","no significant improvement",
        "no significant reduction","fails to","no improvement",
        "worse than","slower than","no benefit","not effective",
        "degradation","overhead eliminates","cannot achieve",
        "no advantage","no measurable","underperforms","inferior to",
        "no statistically significant","null result"
    ]
    key_concepts = [
        "memory","latency","throughput","quality","accuracy",
        "performance","efficiency","speed","overhead","cost",
        "training","inference"
    ]

    # Context markers — detect conditional contradictions
    CONTEXT_MARKERS = {
        "architecture": ["moe","mixture of experts","multi-adapter","multi-lora",
                        "transformer","decoder","encoder","dense","sparse"],
        "scale": ["7b","13b","70b","small","large","billion","parameter"],
        "task": ["fine-tuning","pretraining","inference","serving","generation",
                "reasoning","math","code","nlp","vision"],
        "hardware": ["gpu","cpu","a100","h100","memory","vram","bandwidth"],
        "dataset": ["benchmark","dataset","task","evaluation","downstream"],
    }

    def _get_context(text):
        """Extract context conditions from text."""
        text_low = text.lower()
        found = {}
        for ctx_type, markers in CONTEXT_MARKERS.items():
            matched = [m for m in markers if m in text_low]
            if matched:
                found[ctx_type] = matched[:2]
        return found

    def _contexts_differ(ctx_a, ctx_b):
        """Check if two chunks have different contexts."""
        for ctx_type in CONTEXT_MARKERS:
            if ctx_type in ctx_a and ctx_type in ctx_b:
                if set(ctx_a[ctx_type]) != set(ctx_b[ctx_type]):
                    return True, ctx_type
        return False, None

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

            def _clean_chunk_text(t):
                t = t.strip()
                if t and t[0].islower():
                    import re
                    m = re.search(r'(?<=[.!?])\s+([A-Z])', t)
                    if m:
                        t = t[m.start():].strip()
                return t[:220]

            # Determine contradiction type
            def _contradiction_type(ta, tb):
                ta_low, tb_low = ta.lower(), tb.lower()
                combined = ta_low + " " + tb_low
                # Tradeoff signals — not a true contradiction
                tradeoff_words = ["tradeoff","trade-off","however","but","although","despite",
                                  "limitation","caveat","drawback","at the cost","sacrifice",
                                  "may not","not always","in some cases","can still"]
                # Hardware/setup variance — context-dependent
                hardware_words = ["gpu","cpu","a100","h100","v100","hopper","hardware",
                                  "device","memory bandwidth","serving","production"]
                # Direct contradiction signals
                direct_words = ["no benefit","no improvement","does not reduce","does not improve",
                                "no significant","fails to","worse than","inferior","outperformed by"]
                # Same metric opposite outcome
                metric_words = ["accuracy","perplexity","throughput","latency","memory","flops","speed"]
                has_metric = any(w in combined for w in metric_words)

                if any(w in combined for w in tradeoff_words):
                    return "tradeoff"
                elif any(w in combined for w in hardware_words):
                    return "hardware_variance"
                elif any(w in combined for w in direct_words) and has_metric:
                    return "direct"
                else:
                    return "methodological"

            # Extract the specific triggering signals
            def _triggering_signals(ta, tb):
                ta_low, tb_low = ta.lower(), tb.lower()
                found_pos = [s for s in positive_signals if s in ta_low or s in tb_low]
                found_neg = [s for s in negative_signals if s in ta_low or s in tb_low]
                return found_pos[:3], found_neg[:3]

            c_type = _contradiction_type(sup_text, con_text)
            trig_pos, trig_neg = _triggering_signals(sup_text, con_text)

            # Context-aware analysis
            ctx_a = _get_context(sup_text)
            ctx_b = _get_context(con_text)
            ctx_differs, ctx_type = _contexts_differ(ctx_a, ctx_b)

            if ctx_differs:
                c_type = "contextual"
                ctx_a_vals = ctx_a.get(ctx_type, [])
                ctx_b_vals = ctx_b.get(ctx_type, [])
                detection_explanation = (
                    f"Conditional contradiction detected on [{', '.join(list(shared)[:2])}]. "
                    f"Paper A reports positive results in context: {ctx_a_vals}. "
                    f"Paper B reports negative/different results in context: {ctx_b_vals}. "
                    f"Effect likely depends on {ctx_type}."
                )
                implication = (
                    f"Results are context-dependent — effect on {', '.join(list(shared)[:2])} "
                    f"varies by {ctx_type}. "
                    f"Verify which context matches your use case before drawing conclusions."
                )
            else:
                detection_explanation = (
                    f"Paper A contains positive signals {trig_pos} on [{', '.join(list(shared)[:2])}]. "
                    f"Paper B contains negative signals {trig_neg} on the same concepts. "
                    f"This suggests a {c_type} contradiction."
                )
                implication = _contradiction_implication(list(shared), severity, c_type)

            contradictions.append({
                "shared_concepts": list(shared)[:3],
                "supporting_paper": sup_paper,
                "supporting_claim": _clean_chunk_text(sup_text),
                "contradicting_paper": con_paper,
                "contradicting_claim": _clean_chunk_text(con_text),
                "severity": severity,
                "contradiction_type": c_type,
                "detection_method": "context_aware_signal_analysis",
                "triggering_positive_signals": trig_pos,
                "triggering_negative_signals": trig_neg,
                "context_a": ctx_a,
                "context_b": ctx_b,
                "context_differs": ctx_differs,
                "context_dimension": ctx_type,
                "detection_explanation": detection_explanation,
                "implication": implication,
            })

            if len(contradictions) >= 3:
                return contradictions

    return contradictions


def _contradiction_implication(concepts, severity, c_type=""):
    topic = " and ".join(concepts[:2]) if concepts else "this metric"
    if c_type == "tradeoff":
        return f"Known tradeoff on {topic} — gains in one dimension may come at cost in another. Consider which metric matters most for your use case."
    elif c_type == "hardware_variance":
        return f"Results on {topic} vary by hardware — verify which GPU/setup matches your deployment before concluding."
    elif c_type == "direct":
        return f"Papers directly disagree on {topic}. Examine sample size, dataset, and baseline carefully before drawing conclusions."
    elif severity == "high":
        return f"Strong disagreement on {topic} — results may depend on model size, dataset, or hardware. Replicate both experiments before drawing conclusions."
    return f"Partial disagreement on {topic} — methodology differences likely explain the gap. Check sequence length, model size, and benchmark used."
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
    if verdict == "strong_evidence":
        return {"type":"Ablation Study","description":f"Strong evidence exists for {concept}. Isolate the mechanism: run ablations varying model size, sequence length, and batch size to find boundary conditions where gains disappear.","metric":"Performance vs each variable independently (log scale)","baseline":"Same model without the technique"}
    elif verdict in ("moderate_evidence","supported") and confidence > 0.5:
        return {"type":"Stress Test","description":f"Moderate evidence for {concept}. Stress-test it: measure wall-clock latency at batch sizes 1, 8, 32, 128 on real hardware (A100/H100), not just FLOPs.","metric":"Wall-clock latency (ms) and GPU memory (GB) at multiple batch sizes","baseline":"Dense transformer baseline with same parameter count"}
    elif verdict == "context_dependent":
        return {"type":"Comparative Study","description":f"Results for {concept} vary by context. Run head-to-head comparison across different settings (model size, architecture, task type) to map exactly when gains appear and disappear.","metric":"Performance delta across settings","baseline":"Standard baseline in each setting"}
    elif verdict in ("mixed_evidence","partially_supported"):
        return {"type":"Controlled Ablation","description":f"Evidence for {concept} is mixed. Run a controlled ablation: isolate the specific mechanism being claimed. Verify gains hold when sequence length doubles.","metric":"Memory overhead (GB) vs sequence length (log scale)","baseline":"Standard attention at matching sequence lengths"}
    else:
        return {"type":"Replication Study","description":f"Limited evidence for {concept}. Replicate the strongest supporting paper exactly — contradictions often come from methodology differences.","metric":"Exact reproduction of reported metric in original paper","baseline":"Author-reported numbers as ground truth"}


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
    q = query.lower()
    q_words = set(w for w in q.split() if len(w) > 3)
    # Map query to specific KG concept names
    CONCEPT_MAP = [
        (["flashattention","flash attention"], "FlashAttention"),
        (["kv cache","kvcache","key-value cache"], "KVCache"),
        (["lora","low-rank","low rank","qlora","peft","fine-tuning","finetuning"], "LoRA"),
        (["mixture of experts","moe"], "MixtureOfExperts"),
        (["mamba","state space"], "Mamba"),
        (["speculative decoding"], "SpeculativeDecoding"),
        (["paged attention","pagedattention"], "PagedAttention"),
        (["sparse attention"], "SparseAttention"),
        (["sliding window"], "SlidingWindowAttention"),
        (["quantization","quantisation"], "Quantization"),
    ]
    # Find matching concept
    target_concepts = set()
    for aliases, kg_concept in CONCEPT_MAP:
        if any(a in q for a in aliases):
            target_concepts.add(kg_concept.lower())
    if not target_concepts:
        target_concepts = {concept_low}
    # Filter hypotheses by target concept
    related = [h for h in hyps if
               any(tc in h.get("hypothesis","").lower() for tc in target_concepts)]
    if not related:
        related = [h for h in hyps if
                   sum(1 for w in q_words if len(w)>4 and w in h.get("hypothesis","").lower()) >= 1]
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
    return render_template("landing.html")

@app.route("/app")
def app_page():
    return render_template("research.html")

# ── Private Corpus API ─────────────────────────────────

@app.route("/api/company/upload", methods=["POST"])
def api_company_upload():
    try:
        from private_corpus import ingest_document, verify_company
        company_id = request.form.get("company_id","")
        password   = request.form.get("password","")
        if not verify_company(company_id, password):
            return jsonify({"error": "Invalid credentials"}), 401
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files["file"]
        if not f.filename.endswith(".pdf"):
            return jsonify({"error": "Only PDF files supported"}), 400
        title = request.form.get("title", "")
        pdf_bytes = f.read()
        result = ingest_document(company_id, pdf_bytes, f.filename, title)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/company/search", methods=["POST"])
def api_company_search():
    try:
        from private_corpus import search_private, verify_company
        body = request.get_json(force=True) or {}
        company_id = body.get("company_id","")
        password   = body.get("password","")
        query      = body.get("query","")
        if not verify_company(company_id, password):
            return jsonify({"error": "Invalid credentials"}), 401
        chunks = search_private(company_id, query, top_k=10)
        return jsonify({"chunks": chunks, "count": len(chunks)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/company/stats", methods=["POST"])
def api_company_stats():
    try:
        from private_corpus import get_company_stats, verify_company
        body = request.get_json(force=True) or {}
        company_id = body.get("company_id","")
        password   = body.get("password","")
        if not verify_company(company_id, password):
            return jsonify({"error": "Invalid credentials"}), 401
        stats = get_company_stats(company_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/company/research", methods=["POST"])
def api_company_research():
    """Research query against BOTH public + private corpus."""
    try:
        from private_corpus import search_private, verify_company
        body = request.get_json(force=True) or {}
        company_id = body.get("company_id","")
        password   = body.get("password","")
        query      = body.get("query","")
        if not verify_company(company_id, password):
            return jsonify({"error": "Invalid credentials"}), 401

        # Get public results
        import hashlib, json as _json
        _rkey = "rq:" + hashlib.md5(query.encode()).hexdigest()
        chunk_store, encoder, chunk_index = _get_pipeline()
        public_chunks = chunk_index.retrieve(query, top_k=10)

        # Get private results
        private_chunks = search_private(company_id, query, top_k=6)

        # Combine and run contradiction detection
        all_chunks = public_chunks + private_chunks
        contradictions = _find_contradictions(all_chunks)

        # Check for cross-corpus contradictions (most valuable)
        cross_contradictions = []
        for c in contradictions:
            sup_is_private = any(
                pc.get("paper_title") == c.get("supporting_paper")
                for pc in private_chunks
            )
            con_is_private = any(
                pc.get("paper_title") == c.get("contradicting_paper")
                for pc in private_chunks
            )
            if sup_is_private or con_is_private:
                c["cross_corpus"] = True
                c["alert"] = "⚠ Your internal research contradicts/supports public literature"
                cross_contradictions.append(c)

        return jsonify({
            "query": query,
            "public_chunks": len(public_chunks),
            "private_chunks": len(private_chunks),
            "contradictions": contradictions,
            "cross_corpus_contradictions": cross_contradictions,
            "has_cross_contradiction": len(cross_contradictions) > 0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/company/change_password", methods=["POST"])
def api_company_change_password():
    try:
        from private_corpus import verify_company
        import sqlite3
        body = request.get_json(force=True) or {}
        company_id   = body.get("company_id","")
        old_password = body.get("old_password","")
        new_password = body.get("new_password","")
        if not verify_company(company_id, old_password):
            return jsonify({"error": "Invalid credentials"}), 401
        if len(new_password) < 8:
            return jsonify({"error": "Password too short"}), 400
        conn = sqlite3.connect("knowledge_base/knowledge.db")
        conn.execute("UPDATE private_companies SET password=? WHERE company_id=?",
                     (new_password, company_id))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/company/create", methods=["POST"])
def api_company_create():
    """Admin only - create new enterprise client."""
    try:
        admin_key = request.headers.get("X-Admin-Key","")
        if admin_key != os.environ.get("ADMIN_KEY","tattva-admin-2026"):
            return jsonify({"error": "Unauthorized"}), 401
        from private_corpus import create_company
        body = request.get_json(force=True) or {}
        name = body.get("name","")
        password = body.get("password","")
        if not name or not password:
            return jsonify({"error": "name and password required"}), 400
        domain = body.get("domain","ml")
        result = create_company(name, password, domain)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/company")
@app.route("/company/login")
def company_login():
    """Public login page for enterprise clients."""
    return render_template("company_login.html")

@app.route("/company/dashboard")
def company_dashboard():
    """Dashboard — served after login, auth via session."""
    return render_template("company.html")

@app.route("/api/company/login", methods=["POST"])
def api_company_login():
    """Verify company credentials and return session info."""
    try:
        from private_corpus import verify_company, get_company_stats
        import sqlite3
        body = request.get_json(force=True) or {}
        company_id = body.get("company_id","").strip()
        password   = body.get("password","").strip()

        if not verify_company(company_id, password):
            return jsonify({"error": "Invalid Company ID or password"}), 401

        # Get company info including domain
        conn = sqlite3.connect("knowledge_base/knowledge.db")
        row = conn.execute(
            "SELECT company_name, domain, chunk_count FROM private_companies WHERE company_id=?",
            (company_id,)
        ).fetchone()
        conn.close()

        if not row:
            return jsonify({"error": "Company not found"}), 404

        return jsonify({
            "status": "ok",
            "company_id": company_id,
            "company_name": row[0],
            "domain": row[1] or "ml",
            "chunk_count": row[2] or 0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/company/old")
def company_old():
    import base64
    auth = request.headers.get("Authorization","")
    master_pwd = os.environ.get("COMPANY_PASSWORD","tattva-enterprise-2026")
    try:
        decoded = base64.b64decode(auth.replace("Basic ","")).decode()
        parts = decoded.split(":",1)
        username = parts[0]
        password = parts[1] if len(parts) > 1 else ""

        # Master password — full access
        if password == master_pwd:
            return render_template("company.html")

        # Company-specific login: username=company_id, password=corpus_password
        import sqlite3
        conn = sqlite3.connect("knowledge_base/knowledge.db")
        row = conn.execute(
            "SELECT company_id FROM private_companies WHERE company_id=? AND password=?",
            (username, password)
        ).fetchone()
        conn.close()
        if row:
            return render_template("company.html",
                                   company_id=username,
                                   auto_login=True)
    except: pass
    return Response(
        "Tattva AI Enterprise — Contact sales@tattvaai.org for access",
        401,
        {"WWW-Authenticate": 'Basic realm="Tattva Enterprise"'}
    )

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
    verdicts = {"strong_evidence":0,"moderate_evidence":0,"limited_evidence":0,
                    "mixed_evidence":0,"context_dependent":0,"inconclusive":0}
    for r in reports:
        v = r.get("proposal_verdict","inconclusive")
        if v in verdicts:
            verdicts[v] += 1
        else:
            verdicts["inconclusive"] += 1
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
    reports = _load_reports(100)
    # Deduplicate by query — keep most recent
    seen = {}
    for r in reports:
        q = (r.get("query","")).lower().strip()
        if q not in seen:
            seen[q] = r
    unique = list(seen.values())[:50]
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
    } for r in unique]})

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
    # Clean up hypothesis text for display
    import re as _re
    cleaned = []
    seen_hyp = set()
    for h in hyps:
        text = h.get("hypothesis","")
        # Clean raw graph notation
        text = _re.sub(r"--\[?[\w_]+\]?-->", "→", text)
        text = _re.sub(r"\s*-->\s*", " → ", text)
        text = text.strip()
        # Deduplicate
        if text.lower() in seen_hyp:
            continue
        seen_hyp.add(text.lower())
        h["hypothesis"] = text
        # Clean type label
        h_type = h.get("type","")
        if h_type == "graph_transitivity":
            h["type"] = "Research Direction"
        cleaned.append(h)
    return jsonify({"hypotheses": cleaned[:30], "count": len(cleaned)})


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

def _extract_scope(grounded_claims, contradictions):
    hardware, tasks, model_sizes, seq_lengths = set(), set(), set(), set()
    for gc in grounded_claims:
        bc = gc.get("benchmark_context", {}) or {}
        if bc.get("hardware"): hardware.add(bc["hardware"])
        if bc.get("task"): tasks.add(bc["task"])
        if bc.get("model_size"): model_sizes.add(bc["model_size"])
    conflict_hw = set()
    for c in contradictions:
        if c.get("contradiction_type") == "hardware_variance":
            for p in [c.get("supporting_paper",""), c.get("contradicting_paper","")]:
                if "H100" in p: conflict_hw.add("H100")
                if "A100" in p: conflict_hw.add("A100")
    within, untested = [], []
    if hardware: within.append("🖥 " + " · ".join(sorted(hardware)))
    if tasks: within.append("📋 " + " · ".join(sorted(tasks)))
    if model_sizes: within.append("⚙ " + " · ".join(sorted(model_sizes)))
    all_hw = hardware | conflict_hw
    if "A100" in all_hw and "H100" not in all_hw: untested.append("🖥 H100 Hopper (untested)")
    if "H100" in all_hw and "A100" not in all_hw: untested.append("🖥 A100 (untested)")
    if model_sizes and not any("70" in m or "405" in m for m in model_sizes): untested.append("⚙ 70B+ models (untested)")
    if not seq_lengths: untested.append("📏 Long context >32k (untested)")
    return {"within": within, "untested": untested, "has_scope": len(within) > 0}


@app.route("/api/domain_stats", methods=["GET"])
def api_domain_stats():
    """Return chunk counts per domain."""
    try:
        import sqlite3
        conn = sqlite3.connect("knowledge_base/knowledge.db")
        rows = conn.execute(
            "SELECT domain, COUNT(*) as n FROM chunks GROUP BY domain ORDER BY n DESC"
        ).fetchall()
        conn.close()
        return jsonify({
            "domains": [{"domain": r[0], "chunks": r[1]} for r in rows],
            "total": sum(r[1] for r in rows)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
                    cached = _json.loads(_hit)
                    try:
                        from claim_drift_tracker import make_claim_id as _mcid, update_belief as _ub, get_belief as _gb
                        qcid = _mcid(query)
                        v = cached.get("proposal_verdict","inconclusive")
                        _ub(qcid, query, "supports" if v in ("strong_evidence","moderate_evidence") else "contradicts", float(cached.get("proposal_confidence",0.5)), "", "cache")
                        di = _gb(qcid)
                        cached["drift"] = {"belief": di.get("belief",0.5), "direction": di.get("drift","stable"), "cusum_neg": round(di.get("cusum_neg",0),2)}
                    except: pass
                    return jsonify(cached)
        except: pass

        with _research_lock:
            if query in _research_cache:
                age = time.time() - _research_cache_time.get(query, 0)
                if age < _CACHE_TTL:
                    cached = dict(_research_cache[query])
                    try:
                        from claim_drift_tracker import make_claim_id as _mcid, update_belief as _ub, get_belief as _gb
                        qcid = _mcid(query)
                        v = cached.get("proposal_verdict","inconclusive")
                        _ub(qcid, query, "supports" if v in ("strong_evidence","moderate_evidence") else "contradicts", float(cached.get("proposal_confidence",0.5)), "", "cache")
                        di = _gb(qcid)
                        cached["drift"] = {"belief": di.get("belief",0.5), "direction": di.get("drift","stable"), "cusum_neg": round(di.get("cusum_neg",0),2)}
                    except: pass
                    return jsonify(cached)
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
            config=DiscoveryConfig(top_k_chunks=10, max_claims=20,
                                   max_grounded_claims=8, use_mmr=True))

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
            # Extract benchmark context from supporting evidence
            _grounding = _gc.get("grounding", {})
            _top_support = _grounding.get("top_support", []) if isinstance(_grounding, dict) else []
            _bench_ctx = {}
            for _ev in _top_support[:3]:
                _ev_text = _ev.get("text","") if isinstance(_ev, dict) else ""
                _ctx = _extract_benchmark_context(_ev_text)
                _bench_ctx.update(_ctx)
            _gc["benchmark_context"] = _bench_ctx
            # Score paper quality from top supporting papers
            _paper_scores = []
            for _ev in _top_support[:3]:
                if isinstance(_ev, dict):
                    _pt = _ev.get("paper_title","")
                    _src = _ev.get("source","")
                    _paper_scores.append(_score_paper_quality(_pt, _src))
            _gc["paper_quality"] = round(sum(_paper_scores)/len(_paper_scores), 2) if _paper_scores else 0.5
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

        # Rich verdict system — based on evidence strength
        if supported_count >= 4 and contradicted_count == 0:
            verdict = "strong_evidence"
            confidence = max(float(confidence), 0.78)
        elif supported_count >= 3 and contradicted_count <= 1:
            verdict = "moderate_evidence"
            confidence = max(float(confidence), 0.63)
        elif supported_count >= 2 and contradicted_count <= 1:
            verdict = "moderate_evidence"
            confidence = max(float(confidence), 0.52)
        elif supported_count >= 1 and contradicted_count == 0:
            # Boost to moderate if confidence is already high
            if float(confidence) >= 0.65:
                verdict = "moderate_evidence"
            else:
                verdict = "limited_evidence"
            confidence = max(float(confidence), 0.40)
        elif supported_count >= 1 and contradicted_count >= 1:
            verdict = "mixed_evidence"
            confidence = max(float(confidence), 0.45)
        elif contradicted_count >= 2:
            verdict = "mixed_evidence"
            confidence = max(float(confidence), 0.40)
        else:
            verdict = "inconclusive"
            confidence = max(float(confidence), 0.25)

        # Downgrade if contradictions dominate
        if contradicted_count >= 3:
            verdict = "mixed_evidence"



        # Check for contextual contradictions
        contextual_contradictions = [c for c in contradictions if c.get("context_differs")]
        if contextual_contradictions:
            ctx_dim = contextual_contradictions[0].get("context_dimension","context")
            verdict = "context_dependent"
            confidence = max(float(confidence), 0.45)

        # Fallback: engine verdict wins when count-based gives inconclusive
        if verdict == "inconclusive" and report.proposal_verdict in ("supported","partially_supported") and float(report.proposal_confidence or 0) >= 0.6:
            verdict = "moderate_evidence"
            confidence = max(float(confidence), float(report.proposal_confidence))

        # Generate rich explanation for every verdict
        contextual_contradictions = [c for c in contradictions if c.get("context_differs")]
        ctx_dim = contextual_contradictions[0].get("context_dimension","context") if contextual_contradictions else None

        VERDICT_EXPLANATIONS = {
            "strong_evidence": f"{supported_count} independent studies directly support this claim with no contradictions. High confidence.",
            "moderate_evidence": f"{supported_count} studies support this claim" + (f", {contradicted_count} show tradeoffs." if contradicted_count else ".") + " Moderate confidence.",
            "limited_evidence": f"Only {supported_count} study found direct evidence. More replication needed before drawing conclusions.",
            "mixed_evidence": f"{supported_count} studies support, {contradicted_count} contradict. Results are genuinely mixed — likely depends on experimental setup.",
            "context_dependent": f"Results vary by {ctx_dim or 'context'}. {supported_count} studies support in some settings, {contradicted_count} show opposite results in others.",
            "inconclusive": f"No direct supporting evidence found in {report.evidence_count} retrieved chunks. The question may be too specific or evidence may be sparse.",
        }
        explanation = VERDICT_EXPLANATIONS.get(verdict, "")
        if not explanation:
            explanation = report.proposal_explanation or "Evidence exists but is partial or indirect."

        scope = _extract_scope(grounded, contradictions)
        slim = {
            "query":                query,
            "proposal_verdict":     verdict,
            "proposal_confidence":  round(max(float(confidence), 0.25),4),
            "proposal_explanation": explanation,
            "scope":                scope,
            "evidence_count":       report.evidence_count,
            "supported_count":      supported_count,
            "contradicted_count":   contradicted_count,
            "top_papers":           report.top_papers[:6],
            "top_grounded":         grounded,
            "knowledge_gaps":       report.knowledge_gaps[:3],
            "domain":               report.domain if report.domain and report.domain != "unknown" else "transformer_efficiency",
            "is_conditional":        len([c for c in contradictions if c.get("context_differs")]) > 0,
            "conditional_dimension":  next((c.get("context_dimension") for c in contradictions if c.get("context_differs")), None),
            "context_buckets":       _bucket_claims(grounded),
            "cross_domain_warning":  len(set(
                (gc.get("domain","") or "transformer_efficiency") for gc in grounded 
                if (gc.get("domain","") or "transformer_efficiency") not in ("unknown","transformer_efficiency","")
            )) > 1,
            "domains_found":         list(set(
                (gc.get("domain","") or "transformer_efficiency") for gc in grounded
                if (gc.get("domain","") or "") not in ("unknown","")
            ))[:5] or ["transformer_efficiency"],
            "contradictions":       contradictions,
            "best_experiments":     [],
        }
        # Generate suggested experiments
        try:
            exps = []
            if contradictions:
                c = contradictions[0]
                exps.append(
                    f"Resolve conflict: run identical benchmark on {(c.get('supporting_paper','Paper A'))[:35]} "
                    f"vs {(c.get('contradicting_paper','Paper B'))[:35]} — use same hardware, dataset, metrics"
                )
            if verdict in ("inconclusive","limited_evidence"):
                exps.append(
                    f"First proper study of '{query[:50]}': measure on A100 + H100, "
                    f"compare 7B vs 70B, report wall-clock time and peak memory"
                )
            if slim["top_papers"]:
                exps.append(
                    f"Reproduce '{slim['top_papers'][0][:45]}' with your hardware "
                    f"to verify generalizability beyond original lab conditions"
                )
            exps.append(
                f"Ablation: vary the key parameter across 3 settings, "
                f"measure accuracy vs efficiency tradeoff on standard benchmark"
            )
            slim["best_experiments"] = exps[:4]
        except: pass
        # Save to Supabase research history if user is logged in
        try:
            user_id = request.headers.get("X-User-ID")
            log.info(f"Research history: user_id={user_id}")
            user_email = request.headers.get("X-User-Email")
            if user_id:
                from supabase import create_client as _sb_create
                _sb = _sb_create(os.environ.get("SUPABASE_URL",""), os.environ.get("SUPABASE_KEY",""))
                _sb.table("research_history").insert({
                    "user_id": user_id,
                    "query": query,
                    "verdict": verdict,
                    "confidence": round(float(confidence), 4),
                    "evidence_count": report.evidence_count,
                    "report_json": slim,
                }).execute()
        except Exception as _he:
            log.warning(f"History save failed: {_he}")

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
        # Drift tracker
        try:
            from claim_drift_tracker import process_research_result as _pdr, make_claim_id as _mcid, update_belief as _ub, get_belief as _gb
            _pdr(query, slim)
            qcid = _mcid(query)
            v = slim.get("proposal_verdict","inconclusive")
            direction = "supports" if v in ("strong_evidence","moderate_evidence") else "contradicts"
            _ub(qcid, query, direction, float(slim.get("proposal_confidence",0.5)), "", "pipeline")
            drift_info = _gb(qcid)
            slim["drift"] = {"belief": drift_info.get("belief",0.5), "direction": drift_info.get("drift","stable"), "cusum_neg": round(drift_info.get("cusum_neg",0),2)}
        except Exception as _de:
            print(f"[drift] error: {_de}")
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
                config=DiscoveryConfig(top_k_chunks=10, max_claims=15,
                                       max_grounded_claims=6, use_mmr=True))
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

@app.route("/api/user/history")
def api_user_history():
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"history": []})
    try:
        from supabase import create_client
        sb = create_client(os.environ.get("SUPABASE_URL",""), os.environ.get("SUPABASE_KEY",""))
        r = sb.table("research_history").select("id,query,verdict,confidence,evidence_count,created_at")             .eq("user_id", user_id).order("created_at", desc=True).limit(20).execute()
        return jsonify({"history": r.data or []})
    except Exception as e:
        return jsonify({"history": [], "error": str(e)})

@app.route("/landing")
def landing():
    return send_from_directory("templates", "landing.html")

@app.route("/api/waitlist", methods=["POST"])
def api_waitlist():
    try:
        email = request.json.get("email","").strip()
        if not email or "@" not in email:
            return jsonify({"error": "invalid email"}), 400
        from supabase import create_client
        sb = create_client(os.environ.get("SUPABASE_URL",""), os.environ.get("SUPABASE_KEY",""))
        sb.table("waitlist").insert({"email": email}).execute()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "ok"})  # Fail silently

@app.route("/api/share", methods=["POST"])
def api_share():
    """Save a research result and return a shareable link."""
    try:
        import hashlib, json as _json
        body = request.get_json(force=True) or {}
        query = body.get("query","").strip()
        result = body.get("result",{})
        if not query or not result:
            return jsonify({"error": "missing query or result"}), 400
        # Generate hash from query
        share_id = hashlib.md5(query.lower().encode()).hexdigest()[:10]
        # Save to Supabase
        try:
            from supabase import create_client
            sb = create_client(os.environ.get("SUPABASE_URL",""), os.environ.get("SUPABASE_KEY",""))
            sb.table("shared_results").upsert({
                "share_id": share_id,
                "query": query,
                "result_json": result,
            }).execute()
        except Exception as e:
            log.warning(f"Share save failed: {e}")
        share_url = f"{request.host_url}r/{share_id}"
        return jsonify({"share_id": share_id, "share_url": share_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/r/<share_id>")
def shared_result(share_id):
    """Display a shared research result."""
    try:
        from supabase import create_client
        sb = create_client(os.environ.get("SUPABASE_URL",""), os.environ.get("SUPABASE_KEY",""))
        r = sb.table("shared_results").select("*").eq("share_id", share_id).execute()
        if not r.data:
            return "Result not found", 404
        data = r.data[0]
        # Render research page with pre-loaded result
        import json as _json
        return render_template("research.html", 
                             shared_query=data.get("query",""),
                             shared_result=_json.dumps(data.get("result_json",{})))
    except Exception as e:
        return render_template("research.html")

@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Compare two research methods side by side."""
    try:
        body = request.get_json(force=True) or {}
        method_a = body.get("method_a","").strip()
        method_b = body.get("method_b","").strip()
        if not method_a or not method_b:
            return jsonify({"error": "Need two methods to compare"}), 400

        kg_data = _read_json(KG_PATH, {})
        hyps = _load_hypotheses(50)

        def get_method_info(method):
            m_low = method.lower()
            # Get KG edges
            edges = {}
            for concept, rels in kg_data.items():
                if concept.lower() == m_low or m_low in concept.lower():
                    edges = rels
                    break
            # Get related hypotheses
            m_hyps = [h for h in hyps if m_low in h.get('hypothesis','').lower()]
            # Determine strengths from KG
            strengths = []
            weaknesses = []
            if isinstance(edges, dict):
                for rel, targets in edges.items():
                    if rel in ('reduces','improves','supports_efficiency'):
                        for t in (targets if isinstance(targets,list) else [targets]):
                            strengths.append(f"{rel.replace('_',' ')} {t}")
                    elif rel in ('has_tradeoff','contradicts'):
                        for t in (targets if isinstance(targets,list) else [targets]):
                            weaknesses.append(f"tradeoff with {t}")
            return {
                "method": method,
                "kg_edges": len(edges),
                "strengths": strengths[:4],
                "weaknesses": weaknesses[:3],
                "hypotheses": [h.get('hypothesis','') for h in m_hyps[:3]],
            }

        info_a = get_method_info(method_a)
        info_b = get_method_info(method_b)

        # Find shared concepts
        shared = set(info_a['strengths']) & set(info_b['strengths'])

        return jsonify({
            "method_a": info_a,
            "method_b": info_b,
            "shared_strengths": list(shared),
            "recommendation": f"{method_a} focuses on {', '.join(info_a['strengths'][:2]) or 'general efficiency'}. {method_b} focuses on {', '.join(info_b['strengths'][:2]) or 'general efficiency'}. Choose based on your primary bottleneck.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/gaps", methods=["GET"])
def api_gaps():
    """Return research gaps from KG."""
    try:
        kg_data = _read_json(KG_PATH, {})
        gaps = _find_underresearched(kg_data, "")
        return jsonify({"gaps": gaps})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.json or {}
        query = (data.get("query") or "").strip()
        history = data.get("history") or []
        show_comparison = data.get("show_comparison", False)
        
        if not query:
            return jsonify({"error": "No query provided"})
        
        # Import scientist
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from scientist import chat, get_llm_comparison
        except ImportError:
            from dashboard.scientist import chat, get_llm_comparison
        
        # Retrieve evidence from your corpus
        chunks = []
        retriever_obj = None
        try:
            store, encoder, retriever = _get_pipeline()
            retrieval_query = query
            if history:
                original = next((h['content'] for h in history if h['role']=='user'), '')
                if original and original != query:
                    retrieval_query = original + '. ' + query
            # Use fallback-aware retrieval
            from retrieval.simple_retriever import SimpleRetriever
            retriever_obj = SimpleRetriever()
            chunks = retriever_obj.retrieve_with_fallback(retrieval_query, top_k=10)
            print(f"[chat] Retrieved {len(chunks)} chunks for: {query[:50]}")
        except Exception as e:
            print(f"[chat] Retrieval error: {e}")
            try:
                from retrieval.simple_retriever import SimpleRetriever
                retriever_obj = SimpleRetriever()
                chunks = retriever_obj.retrieve_sqlite(query, top_k=10)
                print(f"[chat] SQLite fallback: {len(chunks)} chunks")
            except Exception as e2:
                print(f"[chat] SQLite fallback error: {e2}")
                chunks = []

        # Get researcher_id from session if available
        researcher_id = request.json.get('researcher_id', 'anonymous')

        # Chain-of-thought with agentic retriever
        result = chat(query, history, chunks,
                     retriever=retriever_obj,
                     researcher_id=researcher_id)
        
        # Get generic LLM comparison if requested
        if show_comparison:
            result["llm_comparison"] = get_llm_comparison(query)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/map")
def contradiction_map():
    return render_template("map.html")

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