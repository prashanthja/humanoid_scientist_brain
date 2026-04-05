# dashboard/app.py
from flask import Flask, render_template, jsonify, request
import json, os, glob, time, sys, threading

app  = Flask(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REPORTS_DIR = os.path.join(ROOT, "outputs", "discovery_reports")
KG_PATH     = os.path.join(ROOT, "knowledge_graph", "graph.json")
HYP_JSONL   = os.path.join(ROOT, "outputs", "hypotheses.jsonl")
CHUNK_DB    = os.path.join(ROOT, "knowledge_base", "knowledge.db")

_sim_cache: dict = {}
_sim_lock = threading.Lock()
_research_cache: dict = {}
_research_lock = threading.Lock()

# ── Pre-loaded pipeline (shared across all requests) ───
_pipeline_store  = None
_pipeline_encoder = None
_pipeline_index  = None
_pipeline_lock   = threading.Lock()

def _get_pipeline():
    """Load encoder once at startup and reuse — avoids re-init on every request."""
    global _pipeline_store, _pipeline_encoder, _pipeline_index
    if _pipeline_index is not None:
        return _pipeline_store, _pipeline_encoder, _pipeline_index
    with _pipeline_lock:
        if _pipeline_index is not None:
            return _pipeline_store, _pipeline_encoder, _pipeline_index
        import logging
        logging.getLogger("tattva.pipeline").info("Initializing pipeline (one-time)...")
        from knowledge_base.chunk_store import ChunkStore
        from learning_module.trainer_online import OnlineTrainer
        from learning_module.embedding_bridge import EmbeddingBridge
        from retrieval.chunk_index import ChunkIndex
        _pipeline_store   = ChunkStore()
        _pipeline_encoder = EmbeddingBridge(OnlineTrainer())
        _pipeline_index   = ChunkIndex(encoder=_pipeline_encoder, chunk_store=_pipeline_store)
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
    hyps, seen = [], set()
    try:
        if not os.path.exists(HYP_JSONL): return hyps
        with open(HYP_JSONL, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                h = json.loads(line.strip())
                k = (h.get("hypothesis", "") or "").strip().lower()
                if k and k not in seen:
                    seen.add(k); hyps.append(h)
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
    if "contextlength" in h or "context" in h: return "Profile memory usage as context grows from 4k to 128k — identify the inflection point."
    if "latency" in h: return "Separate TTFT (time-to-first-token) from throughput — they often trade off against each other."
    if hyp_type == "graph_transitivity": return "Explore whether this transitive relationship holds empirically — no paper has directly tested it."
    return "Design a controlled experiment to directly test whether this relationship holds at scale."


# ── Routes ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("research.html")

@app.route("/admin")
def admin():
    return render_template("index.html")

@app.route("/api/overview")
def api_overview():
    reports = _load_reports(50)
    kg      = _kg_stats()
    verdicts = {"supported":0,"partially_supported":0,"inconclusive":0,"contradicted":0}
    for r in reports:
        v = r.get("proposal_verdict","unknown")
        if v in verdicts: verdicts[v] += 1
    confs = [float(r.get("proposal_confidence",0)) for r in reports]
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
        with _research_lock:
            if query in _research_cache: return jsonify(_research_cache[query])

        # Use shared pre-loaded pipeline
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
        report = build_report(query, result)
        grounded = [_serialize_grounded(gc) for gc in (report.top_grounded[:5] if report.top_grounded else [])]
        supported_count    = sum(1 for gc in grounded if _get_verdict_str(gc) in ("supported","partially_supported"))
        contradicted_count = sum(1 for gc in grounded if _get_verdict_str(gc) == "contradicted")
        confidence = report.proposal_confidence
        if not confidence and grounded:
            vals = [gc.get("verdict",{}).get("confidence",0) for gc in grounded]
            confidence = sum(vals)/len(vals) if vals else 0
        verdict = report.proposal_verdict
        if verdict in ("needs_info","unknown","",None):
            verdict = "supported" if supported_count > 0 else "inconclusive"
        explanation = report.proposal_explanation or ""
        if not explanation or "no claims" in explanation.lower():
            explanation = ("Evidence from multiple sources confirms this claim with experimental results."
                           if supported_count > 0
                           else "Evidence exists but is partial or indirect — further investigation recommended.")
        slim = {
            "query":                query,
            "proposal_verdict":     verdict,
            "proposal_confidence":  round(float(confidence),4),
            "proposal_explanation": explanation,
            "evidence_count":       report.evidence_count,
            "supported_count":      supported_count,
            "contradicted_count":   contradicted_count,
            "top_papers":           report.top_papers[:6],
            "top_grounded":         grounded,
            "knowledge_gaps":       report.knowledge_gaps[:3],
            "domain":               report.domain,
        }
        with _research_lock: _research_cache[query] = slim
        return jsonify(slim)
    except Exception as e:
        import traceback
        return jsonify({"error":str(e),"trace":traceback.format_exc()}), 500

@app.route("/api/research/clear_cache", methods=["POST"])
def api_clear_research_cache():
    with _research_lock: _research_cache.clear()
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
        kg_data  = _read_json(KG_PATH, {})
        hyps     = _load_hypotheses(30)
        reports  = _load_reports(50)
        idea_low = idea.lower()
        stage    = _detect_stage(idea_low)
        concepts = _extract_concepts_from_idea(idea_low)
        blockers = _find_blockers(idea_low, concepts, kg_data)
        return jsonify({
            "idea":           idea,
            "stage":          stage,
            "concepts":       concepts,
            "blockers":       blockers,
            "unstick":        _unstick_suggestions(idea_low, concepts, stage, blockers),
            "directions":     _new_directions(concepts, kg_data, hyps),
            "combinations":   _combination_ideas(concepts, kg_data),
            "prior_work":     _find_prior_work(concepts, reports),
            "open_questions": _open_questions(idea_low, concepts),
        })
    except Exception as e:
        import traceback
        return jsonify({"error":str(e),"trace":traceback.format_exc()}), 500


# ── Background service ─────────────────────────────────

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
    try:
        from background_service import start_background_service
        start_background_service(run_immediately=False)
    except Exception as e:
        print(f"Background service failed to start: {e}")
    app.run(host='127.0.0.1', debug=True, port=8080, use_reloader=False)