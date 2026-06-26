"""
TATTVA-1 Training Data Generator — Multi-model rotation
Rotates across 4 Groq models to maximize daily token budget:
  llama-3.1-8b-instant:    500K tokens/day
  llama3-8b-8192:          500K tokens/day  
  gemma2-9b-it:            500K tokens/day
  mixtral-8x7b-32768:      500K tokens/day
Total: ~2M tokens/day = ~800 training examples/day
"""
import os, sys, json, time, sqlite3
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from datetime import datetime
from groq import Groq

TRAINING_DB = 'knowledge_base/training_data.db'

# Rotate through models to avoid per-model rate limits
MODELS = [
    'llama-3.1-8b-instant',
    'gemma2-9b-it',
    'mixtral-8x7b-32768',
]
model_idx = 0
model_errors = {m: 0 for m in MODELS}

def get_model():
    """Get next available model."""
    global model_idx
    # Skip models with too many errors
    for _ in range(len(MODELS)):
        m = MODELS[model_idx % len(MODELS)]
        if model_errors[m] < 5:
            return m
        model_idx += 1
    return MODELS[0]  # fallback

def rotate_model():
    global model_idx
    model_idx += 1
    print(f"  → switching to {MODELS[model_idx % len(MODELS)]}")

def init_db():
    conn = sqlite3.connect(TRAINING_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            domain TEXT,
            chunks_json TEXT,
            response TEXT,
            verdict TEXT,
            confidence REAL,
            followup_searches TEXT,
            iterations INTEGER,
            model_used TEXT,
            created_at TEXT,
            quality_score REAL DEFAULT 0.0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON training_examples(domain)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_verdict ON training_examples(verdict)")
    conn.commit()
    conn.close()

QUERIES = {
    "ml_ai": [
        "Does LoRA reduce memory usage during fine-tuning?",
        "Does FlashAttention improve training speed on long sequences?",
        "Does chain-of-thought prompting improve LLM reasoning?",
        "Does RLHF improve model alignment with human preferences?",
        "Does mixture of experts improve LLM efficiency?",
        "Does quantization reduce inference cost without quality loss?",
        "Does RAG reduce hallucination in LLMs?",
        "Does self-consistency improve LLM math accuracy?",
        "Does instruction tuning improve zero-shot performance?",
        "Does speculative decoding speed up LLM inference?",
        "Does KV cache compression reduce memory without accuracy loss?",
        "Does sparse attention scale better than dense attention?",
        "Does PEFT match full fine-tuning on domain tasks?",
        "Does DPO outperform PPO for LLM alignment?",
        "Does context length affect LLM reasoning quality?",
        "Does model size correlate with reasoning ability?",
        "Does synthetic data improve LLM training quality?",
        "Does in-context learning generalize to unseen tasks?",
        "Does constitutional AI reduce harmful outputs?",
        "Does continual learning cause catastrophic forgetting?",
        "Does model merging preserve individual capabilities?",
        "Does beam search outperform sampling for factual tasks?",
        "Does prompt engineering match fine-tuning for specialist tasks?",
        "Does LoRA rank affect fine-tuning quality?",
        "Does reinforcement learning from AI feedback work?",
    ],
    "neuroscience": [
        "Does sleep consolidate declarative memory?",
        "Does dopamine drive reward prediction error?",
        "Does prefrontal cortex control working memory?",
        "Does synaptic plasticity underlie long-term memory?",
        "Does the amygdala modulate emotional memory?",
        "Does neuroplasticity decline with age?",
        "Does theta oscillation coordinate hippocampal communication?",
        "Does attention modulate sensory cortex activity?",
        "Does the cerebellum contribute to motor learning?",
        "Does REM sleep affect emotional memory consolidation?",
        "Does stress impair hippocampal neurogenesis?",
        "Does exercise promote brain-derived neurotrophic factor?",
        "Does serotonin regulate mood and depression?",
        "Does meditation change brain structure measurably?",
        "Does the default mode network activate during mind-wandering?",
        "Does adult neurogenesis occur in the human hippocampus?",
        "Does glial activation contribute to neuroinflammation?",
        "Does vagus nerve stimulation reduce seizure frequency?",
        "Does the gut microbiome influence brain function?",
        "Does deep brain stimulation improve Parkinson's symptoms?",
        "Does mirror neuron activity support social cognition?",
        "Does myelination affect neural signal transmission speed?",
        "Does the blood-brain barrier prevent drug delivery?",
        "Does optogenetics allow precise neural circuit control?",
        "Does transcranial magnetic stimulation affect cortical excitability?",
    ],
    "biology": [
        "Does CRISPR-Cas9 cause off-target edits in human cells?",
        "Does AlphaFold accurately predict protein structure?",
        "Does single-cell sequencing reveal cell-type heterogeneity?",
        "Does DNA methylation regulate gene expression?",
        "Does the gut microbiome affect immune function?",
        "Does epigenetic inheritance pass traits to offspring?",
        "Does stem cell therapy regenerate damaged tissue?",
        "Does telomere length predict cellular aging?",
        "Does mRNA stability affect protein expression?",
        "Does cell senescence drive aging phenotypes?",
        "Does autophagy protect against neurodegeneration?",
        "Does liquid-liquid phase separation organize the nucleus?",
        "Does mitochondrial dysfunction cause aging?",
        "Does chromatin remodeling regulate stem cell differentiation?",
        "Does horizontal gene transfer drive bacterial evolution?",
        "Does synthetic biology enable production of complex molecules?",
        "Does microbiome diversity correlate with health outcomes?",
        "Does CRISPR prime editing outperform base editing?",
        "Does cell-free DNA enable liquid biopsy for cancer?",
        "Does organoid modeling recapitulate organ development?",
        "Does RNA splicing diversity expand the proteome?",
        "Does the exposome influence disease risk beyond genetics?",
        "Does CRISPR base editing reduce off-target effects?",
        "Does single-molecule sequencing outperform short-read methods?",
        "Does microRNA regulate post-transcriptional gene expression?",
    ],
    "medicine": [
        "Does immunotherapy improve survival in lung cancer?",
        "Does early antibiotic treatment worsen antibiotic resistance?",
        "Do statins reduce cardiovascular mortality in low-risk patients?",
        "Does exercise reduce depression symptoms as effectively as medication?",
        "Does metformin extend lifespan beyond diabetes treatment?",
        "Does aspirin prevent colorectal cancer?",
        "Does bariatric surgery resolve type 2 diabetes?",
        "Do antidepressants outperform placebo for mild depression?",
        "Does GLP-1 agonist therapy reduce cardiovascular events?",
        "Does CAR-T cell therapy achieve durable remission in lymphoma?",
        "Does PD-1 checkpoint inhibition improve melanoma survival?",
        "Does continuous glucose monitoring improve type 1 diabetes?",
        "Does cognitive behavioral therapy treat insomnia effectively?",
        "Does mindfulness-based stress reduction reduce chronic pain?",
        "Does high-intensity interval training outperform moderate exercise?",
        "Does the Mediterranean diet reduce all-cause mortality?",
        "Does social isolation increase dementia risk?",
        "Does air pollution increase childhood asthma?",
        "Does vitamin D supplementation reduce COVID-19 severity?",
        "Does maternal diet affect offspring metabolic health?",
        "Does proton pump inhibitor use increase dementia risk?",
        "Does aspirin therapy prevent preeclampsia?",
        "Does telomere shortening predict cardiovascular disease?",
        "Does fecal microbiota transplant treat C. difficile?",
        "Does low-dose CT screening reduce lung cancer mortality?",
    ],
    "physics": [
        "Does quantum error correction make fault-tolerant computing feasible?",
        "Do topological insulators conduct electricity without resistance?",
        "Does dark matter interact with ordinary matter gravitationally only?",
        "Does quantum annealing outperform classical computers for optimization?",
        "Does the Casimir effect demonstrate quantum vacuum energy?",
        "Does gravitational wave detection confirm general relativity?",
        "Does quantum decoherence explain the quantum-classical boundary?",
        "Does nuclear fusion energy production become net positive?",
        "Does dark energy cause accelerating cosmic expansion?",
        "Does quantum supremacy demonstrate practical advantage?",
        "Does antimatter fall upward in a gravitational field?",
        "Does quantum chaos affect thermalization in isolated systems?",
        "Does quasicrystal structure violate crystallographic symmetry?",
        "Does quantum biology explain photosynthesis efficiency?",
        "Does Bose-Einstein condensation occur in magnons?",
        "Does high-temperature superconductivity occur at room temperature?",
        "Does string theory make testable predictions?",
        "Does loop quantum gravity resolve the singularity problem?",
        "Does quantum tunneling explain enzyme catalysis rates?",
        "Does the holographic principle apply beyond black holes?",
        "Does quantum entanglement enable faster-than-light communication?",
        "Does the Higgs field give mass to all particles?",
        "Does the Many-Worlds interpretation make different predictions?",
        "Does gravitoelectromagnetism produce measurable effects?",
        "Does the measurement problem in quantum mechanics have a solution?",
    ],
    "chemistry": [
        "Does heterogeneous catalysis reduce activation energy?",
        "Does lithium-sulfur battery outperform lithium-ion energy density?",
        "Does photocatalysis enable efficient water splitting for hydrogen?",
        "Does metal-organic framework outperform zeolite for CO2 capture?",
        "Does click chemistry enable bioorthogonal labeling in cells?",
        "Does mechanochemistry enable solvent-free reactions?",
        "Does single-atom catalysis maximize atomic efficiency?",
        "Does electrochemical CO2 reduction produce useful fuels?",
        "Does chirality of drug molecules affect biological activity?",
        "Does perovskite solar cell efficiency match silicon?",
        "Does solid electrolyte improve battery safety?",
        "Does radical polymerization control molecular weight distribution?",
        "Does graphene oxide enable efficient water filtration?",
        "Does green solvent replace toxic solvents in synthesis?",
        "Does coordination chemistry enable anticancer drug design?",
        "Does self-assembly create functional nanoscale structures?",
        "Does hydrogen bonding stabilize protein secondary structure?",
        "Does photoredox catalysis enable new synthetic transformations?",
        "Does ionic liquid improve electrochemical device performance?",
        "Does polymer brush coating prevent protein adsorption?",
        "Does flow chemistry improve pharmaceutical synthesis?",
        "Does enzymatic catalysis outperform chemical catalysis for specificity?",
        "Does microplastic contamination affect aquatic ecosystems?",
        "Does supramolecular chemistry enable drug delivery capsules?",
        "Does OLED outperform LCD for energy efficiency?",
    ],
    "climate": [
        "Does direct air capture of CO2 become economically viable?",
        "Does solar geoengineering reduce global temperature effectively?",
        "Does permafrost thaw create a positive feedback loop?",
        "Does ocean acidification harm coral reef ecosystems?",
        "Does climate change increase hurricane intensity?",
        "Does reforestation significantly offset carbon emissions?",
        "Does sea level rise threaten coastal megacities by 2100?",
        "Does Arctic sea ice loss accelerate global warming?",
        "Does carbon pricing reduce greenhouse gas emissions?",
        "Does climate change increase wildfire frequency?",
        "Does ocean warming disrupt thermohaline circulation?",
        "Does biodiversity loss accelerate ecosystem collapse?",
        "Does climate change increase food insecurity globally?",
        "Does urban heat island effect worsen climate impacts?",
        "Does clean energy transition create more jobs than it destroys?",
        "Does soil carbon sequestration scale as a climate solution?",
        "Does nuclear power reduce carbon emissions effectively?",
        "Does climate change increase vector-borne disease spread?",
        "Does precision agriculture reduce agricultural emissions?",
        "Does plant-based diet reduce individual carbon footprint?",
        "Does green hydrogen become cost-competitive with fossil fuels?",
        "Does building electrification significantly reduce emissions?",
        "Does methane from livestock contribute substantially to warming?",
        "Does renewable energy replace fossil fuels in electricity grids?",
        "Does climate migration exceed current international capacity?",
    ],
    "economics": [
        "Does minimum wage increase reduce employment?",
        "Does quantitative easing cause inflation?",
        "Does income inequality reduce economic growth?",
        "Does free trade increase aggregate welfare?",
        "Does universal basic income reduce work incentives?",
        "Does carbon tax effectively reduce emissions?",
        "Does automation increase structural unemployment?",
        "Does healthcare spending improve population health outcomes?",
        "Does education spending increase economic mobility?",
        "Does immigration increase native worker wages?",
        "Does financial deregulation increase economic growth?",
        "Does austerity during recession prolong downturns?",
        "Does foreign direct investment benefit developing economies?",
        "Does patent protection incentivize innovation?",
        "Does market concentration reduce consumer welfare?",
        "Does behavioral nudge increase retirement savings?",
        "Does public infrastructure spending have high economic multiplier?",
        "Does inequality of opportunity persist across generations?",
        "Does gig economy work reduce worker welfare?",
        "Does central bank independence reduce inflation?",
        "Does corporate tax cut increase business investment?",
        "Does land value tax improve urban housing affordability?",
        "Does trade deficit indicate economic weakness?",
        "Does economic growth reduce poverty in developing nations?",
        "Does monetary policy affect real economic output long-term?",
    ],
    "psychology": [
        "Does cognitive behavioral therapy outperform medication for depression?",
        "Does mindfulness reduce anxiety in clinical populations?",
        "Does growth mindset improve academic performance?",
        "Does early childhood trauma predict adult mental health?",
        "Does social media use increase adolescent depression?",
        "Does implicit bias training reduce discriminatory behavior?",
        "Does ego depletion reduce self-control capacity?",
        "Does money buy happiness beyond a threshold income?",
        "Does personality remain stable across the lifespan?",
        "Does the bystander effect reduce helping in emergencies?",
        "Does placebo effect produce measurable neurological changes?",
        "Does sleep deprivation impair decision-making?",
        "Does positive reinforcement outperform punishment in learning?",
        "Does nature exposure reduce stress measurably?",
        "Does social support buffer against stress-related illness?",
        "Does loneliness increase mortality risk?",
        "Does trauma-focused CBT treat PTSD effectively?",
        "Does gratitude practice improve psychological wellbeing?",
        "Does perfectionism predict burnout?",
        "Does emotional intelligence predict leadership effectiveness?",
        "Does procrastination correlate with anxiety?",
        "Does music therapy reduce pain perception?",
        "Does expressive writing improve physical health outcomes?",
        "Does group therapy match individual therapy for most conditions?",
        "Does self-compassion improve mental health outcomes?",
    ],
    "mathematics": [
        "Does optimal transport provide better generative model training?",
        "Does persistent homology detect meaningful structure in data?",
        "Does stochastic gradient descent converge to global minima for neural nets?",
        "Does random matrix theory predict neural network generalization?",
        "Does differential privacy provide meaningful privacy guarantees?",
        "Does compressed sensing enable accurate recovery from few measurements?",
        "Does spectral clustering outperform k-means for non-convex clusters?",
        "Does the lottery ticket hypothesis explain neural network pruning?",
        "Does convex relaxation solve NP-hard combinatorial problems?",
        "Does mean field theory approximate particle interactions accurately?",
        "Does chaos theory limit long-term weather prediction?",
        "Does graph theory model social networks accurately?",
        "Does wavelet transform outperform Fourier for non-stationary signals?",
        "Does topological data analysis outperform statistical methods?",
        "Does information geometry improve optimization algorithms?",
        "Does algebraic topology have machine learning applications?",
        "Does number theory underlie modern cryptography?",
        "Does ergodic theory explain statistical mechanics?",
        "Does Riemannian optimization outperform Euclidean for constrained problems?",
        "Does measure theory provide foundations for probability?",
        "Does the P vs NP problem affect practical computation?",
        "Does tropical geometry have optimization applications?",
        "Does category theory unify mathematical structures?",
        "Does functional analysis underlie quantum mechanics?",
        "Does the Riemann hypothesis affect prime number distribution?",
    ],
    "computer_systems": [
        "Does CPU cache prefetching reduce memory latency?",
        "Does RDMA improve distributed machine learning performance?",
        "Does Rust prevent memory safety bugs in systems programming?",
        "Does containerization improve application deployment reliability?",
        "Does NVMe SSD outperform SATA for database workloads?",
        "Does FPGA acceleration outperform GPU for inference?",
        "Does log-structured merge tree outperform B-tree for write-heavy workloads?",
        "Does speculative execution create security vulnerabilities?",
        "Does kernel bypass improve network throughput?",
        "Does disaggregated memory improve datacenter efficiency?",
        "Does compiler auto-vectorization match manual SIMD optimization?",
        "Does microservice architecture improve system reliability?",
        "Does garbage collection introduce unacceptable latency for real-time systems?",
        "Does hardware transactional memory simplify concurrent programming?",
        "Does approximate computing reduce energy consumption acceptably?",
        "Does software-defined networking improve network management?",
        "Does tail latency optimization require different techniques than average?",
        "Does serverless computing reduce operational costs?",
        "Does eBPF enable safe kernel extensibility?",
        "Does confidential computing protect data in use?",
        "Does energy-proportional computing reduce datacenter power?",
        "Does NUMA-aware scheduling improve multiprocessor performance?",
        "Does CXL interconnect enable memory pooling across servers?",
        "Does persistent memory change storage system design?",
        "Does network function virtualization reduce latency?",
    ],
}

def run_query(query, chunks, model):
    """Run a single query with the given model."""
    client = Groq(api_key=os.environ.get('GROQ_API_KEY',''))
    
    evidence = "\n".join([
        f"[{i+1}] {c.get('paper_title','Unknown')}: {c.get('text','')[:300]}..."
        for i, c in enumerate(chunks[:6])
    ]) if chunks else "No papers retrieved."
    
    prompt = f"""You are a research scientist evaluating scientific evidence.

EVIDENCE ({len(chunks)} papers):
{evidence}

QUESTION: {query}

Think step by step:
STEP 1 - SUPPORTING EVIDENCE: What do papers confirm?
STEP 2 - CONTRADICTIONS: Do papers disagree? Why exactly?
STEP 3 - SCOPE: TESTED IN: [conditions] / NOT TESTED IN: [conditions]
STEP 4 - CONFIDENCE: Rate 0.0-1.0 and explain why.
STEP 5 - NEXT STEP: What should a researcher do next?

Be specific. Cite paper titles. Max 200 words."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content

def extract_verdict(text):
    text_lower = text.lower()
    confidence = 0.5
    import re
    nums = re.findall(r'confidence[:\s]+([0-9.]+)', text_lower)
    if nums:
        try: confidence = float(nums[0])
        except: pass
    if any(w in text_lower for w in ['strong support','well established','consensus','clearly']):
        verdict = 'STRONG SUPPORT'
    elif any(w in text_lower for w in ['mixed','contradict','disagree','conflict']):
        verdict = 'MIXED EVIDENCE'
    elif any(w in text_lower for w in ['insufficient','limited','few papers','no evidence']):
        verdict = 'INSUFFICIENT EVIDENCE'
    elif any(w in text_lower for w in ['refuted','disproven','false','incorrect']):
        verdict = 'REFUTED'
    else:
        verdict = 'MODERATE SUPPORT'
    return verdict, confidence

def quality_score(response, verdict, confidence, chunks):
    score = 0.0
    if verdict not in ['ERROR', None]: score += 0.2
    if confidence > 0 and confidence != 0.5: score += 0.2
    if 'TESTED' in response.upper(): score += 0.2
    if len(response) > 150: score += 0.2
    if len(chunks) >= 3: score += 0.2
    return score

def generate(target=2000):
    from retrieval.simple_retriever import SimpleRetriever
    retriever = SimpleRetriever()
    init_db()
    
    conn = sqlite3.connect(TRAINING_DB)
    existing = conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
    # Get already-done queries
    done_queries = {r[0] for r in conn.execute("SELECT query FROM training_examples WHERE verdict != 'ERROR'").fetchall()}
    conn.close()
    
    print(f"Existing examples: {existing}")
    print(f"Already completed queries: {len(done_queries)}")
    
    all_queries = []
    for domain, queries in QUERIES.items():
        for q in queries:
            if q not in done_queries:  # skip already done
                all_queries.append((domain, q))
    
    print(f"Queries remaining: {len(all_queries)}")
    print(f"Target: {target} new examples")
    print(f"Models: {MODELS}")
    print(f"Starting...\n")
    
    total = 0
    errors = 0
    rate_limit_wait = 15  # seconds to wait on rate limit
    
    for i, (domain, query) in enumerate(all_queries):
        if total >= target:
            break
        
        model = get_model()
        
        try:
            # Retrieve chunks
            chunks = retriever.retrieve_with_fallback(query, top_k=8)
            
            # Run inference
            response = run_query(query, chunks, model)
            verdict, confidence = extract_verdict(response)
            qscore = quality_score(response, verdict, confidence, chunks)
            
            # Store
            conn = sqlite3.connect(TRAINING_DB)
            conn.execute("""
                INSERT INTO training_examples
                (query,domain,chunks_json,response,verdict,
                 confidence,model_used,created_at,quality_score)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                query, domain,
                json.dumps([{'text':c.get('text','')[:400],
                            'title':c.get('paper_title',''),
                            'domain':c.get('domain','')} 
                           for c in chunks[:6]]),
                response, verdict, confidence,
                model, datetime.now().isoformat(), qscore
            ))
            conn.commit()
            conn.close()
            
            total += 1
            model_errors[model] = max(0, model_errors[model]-1)
            
            if total % 10 == 0:
                print(f"[{total}/{target}] {domain[:12]} | {verdict[:15]} | "
                      f"conf={confidence:.2f} | q={qscore:.1f} | model={model[:15]}")
            
            time.sleep(1.0)  # conservative delay
            
        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'rate' in err_str.lower():
                model_errors[model] = model_errors.get(model,0) + 1
                print(f"  Rate limit on {model} → rotating. Waiting {rate_limit_wait}s...")
                rotate_model()
                time.sleep(rate_limit_wait)
            else:
                errors += 1
                print(f"  Error: {err_str[:80]}")
                time.sleep(2)
    
    # Final stats
    conn = sqlite3.connect(TRAINING_DB)
    total_stored = conn.execute("SELECT COUNT(*) FROM training_examples WHERE verdict != 'ERROR'").fetchone()[0]
    by_domain = conn.execute("""
        SELECT domain, COUNT(*) n, AVG(quality_score) q
        FROM training_examples WHERE verdict != 'ERROR'
        GROUP BY domain ORDER BY n DESC
    """).fetchall()
    by_verdict = conn.execute("""
        SELECT verdict, COUNT(*) n FROM training_examples
        WHERE verdict != 'ERROR'
        GROUP BY verdict ORDER BY n DESC
    """).fetchall()
    conn.close()
    
    print(f"\n{'='*55}")
    print(f"COMPLETE — Total good examples: {total_stored}")
    print(f"New this run: {total} | Errors: {errors}")
    print(f"\nBy domain:")
    for d in by_domain:
        bar = '█' * int(d[2]*5)
        print(f"  {d[0]:20} {d[1]:4} examples  quality:{d[2]:.2f} {bar}")
    print(f"\nBy verdict:")
    for v in by_verdict:
        print(f"  {v[0]:25} {v[1]:4}")
    print(f"{'='*55}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--queries', type=int, default=275)
    args = p.parse_args()
    generate(args.queries)
