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
    "ml_ai_2": [
        "Does gradient checkpointing reduce memory during LLM training?",
        "Does weight tying improve language model performance?",
        "Does dropout improve generalization in transformers?",
        "Does layer normalization stabilize transformer training?",
        "Does positional encoding affect long-context performance?",
        "Does temperature scaling improve LLM calibration?",
        "Does prefix tuning match LoRA for parameter efficiency?",
        "Does adapter tuning preserve pretrained knowledge?",
        "Does curriculum learning improve LLM training efficiency?",
        "Does data augmentation help LLM fine-tuning?",
        "Does gradient clipping prevent training instability?",
        "Does sparse fine-tuning outperform dense fine-tuning?",
        "Does knowledge distillation preserve model capabilities?",
        "Does reinforcement learning improve code generation?",
        "Does chain of thought emerge from scale alone?",
        "Does few-shot prompting match fine-tuning on reasoning?",
        "Does tool use improve LLM problem solving?",
        "Does activation sparsity improve inference efficiency?",
        "Does early exit reduce LLM inference latency?",
        "Does token pruning maintain generation quality?",
        "Does continuous pretraining improve domain adaptation?",
        "Does scaling laws hold across different architectures?",
        "Does model merging reduce catastrophic forgetting?",
        "Does dynamic batching improve GPU utilization?",
        "Does synthetic data match real data for LLM training?",
    ],
    "neuroscience_2": [
        "Does cortical oscillation frequency predict working memory?",
        "Does dopaminergic signaling modulate decision making?",
        "Does sleep spindle density correlate with memory consolidation?",
        "Does prefrontal-hippocampal interaction support episodic memory?",
        "Does neuroinflammation accelerate cognitive decline?",
        "Does adult hippocampal neurogenesis support pattern separation?",
        "Does the cerebellum contribute to cognitive functions?",
        "Does thalamic gating control information flow in cortex?",
        "Does astrocyte signaling modulate synaptic transmission?",
        "Does transcranial direct current stimulation improve memory?",
        "Does deep sleep improve emotional memory consolidation?",
        "Does the default mode network support self-referential processing?",
        "Does dopamine depletion impair spatial navigation?",
        "Does neuroplasticity support recovery after stroke?",
        "Does the hippocampus encode time in episodic memory?",
        "Does serotonin modulate social behavior?",
        "Does the basal ganglia support habit formation?",
        "Does the amygdala encode threat prediction?",
        "Does acetylcholine support attentional modulation?",
        "Does cortical feedback improve perceptual discrimination?",
        "Does neurogenesis decline with chronic stress?",
        "Does inhibitory interneuron activity shape cortical rhythms?",
        "Does the entorhinal cortex support spatial memory?",
        "Does mirror neuron activity correlate with empathy?",
        "Does cortical thickness predict cognitive performance?",
    ],
    "biology_2": [
        "Does telomerase activity extend cellular lifespan?",
        "Does DNA repair efficiency predict cancer susceptibility?",
        "Does the circadian clock regulate metabolism?",
        "Does microRNA dysregulation contribute to cancer?",
        "Does protein aggregation cause neurodegeneration?",
        "Does epigenetic reprogramming reverse aging markers?",
        "Does chromatin accessibility predict gene expression?",
        "Does alternative splicing expand protein diversity?",
        "Does mitochondrial dysfunction trigger apoptosis?",
        "Does autophagy protect against metabolic disease?",
        "Does stem cell niche control differentiation?",
        "Does lateral inhibition pattern biological development?",
        "Does protein folding predict disease risk?",
        "Does RNA editing expand transcriptome diversity?",
        "Does phase separation organize nuclear compartments?",
        "Does cell polarity guide tissue organization?",
        "Does membrane tension regulate cell migration?",
        "Does metabolic flux predict cell fate?",
        "Does cell-cell communication coordinate tissue repair?",
        "Does synthetic gene circuit design enable precise control?",
        "Does gene regulatory network topology predict robustness?",
        "Does the microbiome influence drug metabolism?",
        "Does horizontal gene transfer accelerate bacterial adaptation?",
        "Does CRISPR efficiency vary by cell type?",
        "Does gut microbiome composition predict autoimmune disease?",
    ],
    "medicine_2": [
        "Does statin therapy reduce dementia risk?",
        "Does intermittent fasting improve metabolic health?",
        "Does gut microbiome transplant treat inflammatory bowel disease?",
        "Does checkpoint immunotherapy cause autoimmune side effects?",
        "Does CRISPR therapy correct sickle cell disease?",
        "Does mRNA vaccine technology prevent cancer?",
        "Does physical activity prevent Alzheimer disease?",
        "Does precision medicine improve treatment outcomes?",
        "Does telemedicine improve access to care?",
        "Does artificial intelligence improve diagnostic accuracy?",
        "Does antibiotic cycling prevent resistance?",
        "Does immunotherapy work for triple negative breast cancer?",
        "Does stem cell therapy repair spinal cord injury?",
        "Does gene therapy correct hemophilia?",
        "Does liquid biopsy detect early cancer?",
        "Does microbiome modulation improve mental health?",
        "Does ketogenic diet reduce seizure frequency?",
        "Does bariatric surgery reduce cancer risk?",
        "Does PCSK9 inhibition reduce cardiovascular events?",
        "Does aspirin prevent pancreatic cancer?",
        "Does metformin extend healthspan beyond diabetes?",
        "Does CAR-NK cell therapy overcome CAR-T limitations?",
        "Does continuous positive airway pressure prevent cardiovascular events?",
        "Does early cancer screening improve survival?",
        "Does tPA improve stroke outcomes beyond 4.5 hours?",
    ],
    "physics_2": [
        "Does quantum error correction threshold scale favorably?",
        "Does topological protection enable fault-tolerant quantum gates?",
        "Does room temperature superconductivity exist in hydrogen compounds?",
        "Does quantum advantage hold for practical optimization?",
        "Does dark matter self-interact at galactic scales?",
        "Does the cosmological constant vary with cosmic time?",
        "Does quantum entanglement enable secure communication?",
        "Does spin liquid state exist in frustrated magnets?",
        "Does phonon engineering reduce thermal conductivity?",
        "Does metamaterial design overcome diffraction limits?",
        "Does quantum chaos thermalize isolated quantum systems?",
        "Does plasma confinement improve with magnetic field geometry?",
        "Does quantum sensing outperform classical sensors?",
        "Does photonic crystal slow light propagation?",
        "Does cold atom simulation predict condensed matter behavior?",
        "Does spin-orbit coupling enable spintronics applications?",
        "Does cavity quantum electrodynamics enable quantum networking?",
        "Does nonlinear optics enable efficient frequency conversion?",
        "Does Anderson localization prevent conductance in disordered systems?",
        "Does Bose-Einstein condensate exhibit superfluidity?",
        "Does quantum phase transition exhibit universal scaling?",
        "Does gravitational wave astronomy constrain neutron star equations of state?",
        "Does topological insulator surface state conduct robustly?",
        "Does quantum walk algorithm outperform classical random walk?",
        "Does quantum gravity produce observable effects at CERN?",
    ],
    "chemistry_2": [
        "Does photocatalyst bandgap determine hydrogen evolution efficiency?",
        "Does solvent polarity affect reaction selectivity?",
        "Does catalyst loading affect product yield linearly?",
        "Does temperature control stereoselectivity in asymmetric synthesis?",
        "Does molecular weight distribution affect polymer mechanical properties?",
        "Does surface functionalization improve electrode performance?",
        "Does reaction pressure affect gas-phase selectivity?",
        "Does Lewis acid strength predict catalytic activity?",
        "Does nanoparticle size affect catalytic turnover frequency?",
        "Does electrolyte composition affect battery cycle life?",
        "Does membrane selectivity limit electrochemical efficiency?",
        "Does porosity affect mass transfer in heterogeneous catalysis?",
        "Does chirality transfer operate across multiple bond formations?",
        "Does organocatalysis match metal catalysis for enantioselectivity?",
        "Does solvent-free synthesis improve atom economy?",
        "Does continuous flow chemistry reduce reaction time?",
        "Does machine learning predict reaction yield accurately?",
        "Does fluorination improve drug bioavailability?",
        "Does supercritical CO2 replace organic solvents effectively?",
        "Does electrochemical oxidation replace stoichiometric reagents?",
        "Does computational screening predict catalyst performance?",
        "Does biocatalysis achieve pharmaceutical synthesis at scale?",
        "Does ring strain drive cycloaddition reactivity?",
        "Does hyperconjugation stabilize carbocation intermediates?",
        "Does aromaticity predict reaction pathway preference?",
    ],
    "climate_2": [
        "Does permafrost carbon feedback accelerate warming nonlinearly?",
        "Does marine cloud brightening cool regional temperatures?",
        "Does enhanced weathering sequester carbon at scale?",
        "Does tropical deforestation alter regional precipitation?",
        "Does Arctic amplification destabilize mid-latitude weather?",
        "Does ocean heat content predict hurricane intensification?",
        "Does stratospheric aerosol injection reduce monsoon rainfall?",
        "Does blue carbon sequestration offset coastal emissions?",
        "Does urban greening reduce heat island intensity?",
        "Does climate change shift disease vector ranges?",
        "Does glacier retreat affect downstream water availability?",
        "Does ocean deoxygenation threaten marine biodiversity?",
        "Does peatland drainage increase greenhouse gas emissions?",
        "Does renewable energy intermittency require storage solutions?",
        "Does climate adaptation reduce economic damages?",
        "Does carbon pricing accelerate clean energy transition?",
        "Does climate change increase conflict risk?",
        "Does extreme heat reduce labor productivity?",
        "Does sea level rise threaten small island nations this century?",
        "Does biodiversity loss reduce ecosystem carbon storage?",
        "Does wetland restoration improve water quality?",
        "Does sustainable agriculture reduce food system emissions?",
        "Does aviation contribute disproportionately to warming?",
        "Does climate change increase wildfire smoke health impacts?",
        "Does green hydrogen production become cost competitive?",
    ],
    "economics_2": [
        "Does automation reduce wage inequality?",
        "Does trade liberalization increase productivity?",
        "Does financial inclusion reduce poverty?",
        "Does minimum wage reduce income inequality?",
        "Does intellectual property protection stimulate innovation?",
        "Does inequality reduce intergenerational mobility?",
        "Does public debt crowd out private investment?",
        "Does corruption reduce economic growth?",
        "Does foreign aid promote development?",
        "Does urbanization increase productivity?",
        "Does inflation targeting improve macroeconomic stability?",
        "Does financial development predict economic growth?",
        "Does social capital improve economic outcomes?",
        "Does technological progress reduce employment long-term?",
        "Does market power reduce consumer welfare?",
        "Does health investment increase economic productivity?",
        "Does education spending reduce inequality?",
        "Does environmental regulation reduce competitiveness?",
        "Does central bank credibility reduce inflation expectations?",
        "Does natural resource abundance cause economic stagnation?",
        "Does globalization increase within-country inequality?",
        "Does competition policy improve innovation incentives?",
        "Does behavioral economics improve policy effectiveness?",
        "Does housing affordability affect labor mobility?",
        "Does digital economy increase productivity growth?",
    ],
    "psychology_2": [
        "Does meditation reduce cortisol levels measurably?",
        "Does cognitive training transfer to real-world tasks?",
        "Does social media comparison increase body dissatisfaction?",
        "Does childhood adversity predict adult psychopathology?",
        "Does emotional regulation training reduce anxiety?",
        "Does nature therapy reduce depression symptoms?",
        "Does chronic pain affect cognitive functioning?",
        "Does sleep quality predict emotional reactivity?",
        "Does creative expression improve psychological wellbeing?",
        "Does peer support reduce mental health stigma?",
        "Does positive psychology intervention improve resilience?",
        "Does trauma-informed care improve treatment outcomes?",
        "Does attachment style predict relationship satisfaction?",
        "Does self-efficacy predict academic achievement?",
        "Does rumination prolong depressive episodes?",
        "Does social connection buffer against burnout?",
        "Does exercise intensity affect mood improvement?",
        "Does cognitive restructuring reduce catastrophizing?",
        "Does acceptance-based therapy outperform avoidance?",
        "Does early intervention prevent chronic mental illness?",
        "Does psychological safety improve team performance?",
        "Does implicit association test predict discriminatory behavior?",
        "Does loss aversion affect financial decision making?",
        "Does cognitive load affect moral judgment?",
        "Does scarcity mindset impair executive function?",
    ],
    "mathematics_2": [
        "Does compressed sensing recover sparse signals exactly?",
        "Does stochastic optimization converge to global minima?",
        "Does optimal control theory apply to neural networks?",
        "Does game theory predict human strategic behavior?",
        "Does information theory bound machine learning generalization?",
        "Does algebraic geometry have cryptographic applications?",
        "Does harmonic analysis improve signal processing?",
        "Does combinatorics underlie algorithm complexity?",
        "Does number theory have quantum computing applications?",
        "Does differential geometry describe spacetime accurately?",
        "Does probability theory underpin statistical inference?",
        "Does linear algebra constrain neural network expressivity?",
        "Does convex optimization guarantee global solutions?",
        "Does discrete mathematics model network phenomena?",
        "Does dynamical systems theory explain chaos?",
        "Does measure theory formalize probability rigorously?",
        "Does category theory unify mathematical structures?",
        "Does topology classify manifolds completely?",
        "Does group theory underlie particle physics symmetries?",
        "Does set theory resolve mathematical foundations?",
        "Does numerical analysis bound computational errors?",
        "Does approximation theory limit function representation?",
        "Does graph theory model epidemic spreading?",
        "Does knot theory have biological applications?",
        "Does matroid theory generalize linear independence?",
    ],
    "computer_systems_2": [
        "Does cache coherence protocol affect multicore scalability?",
        "Does branch prediction accuracy affect CPU performance?",
        "Does out-of-order execution improve throughput?",
        "Does memory bandwidth limit deep learning training?",
        "Does interconnect topology affect distributed training speed?",
        "Does storage hierarchy design affect database performance?",
        "Does network virtualization add measurable latency?",
        "Does containerization overhead affect high-performance computing?",
        "Does memory disaggregation improve datacenter efficiency?",
        "Does hardware prefetching reduce memory stall cycles?",
        "Does compiler optimization match hand-tuned assembly?",
        "Does cache partitioning improve quality of service?",
        "Does NUMA topology affect parallel application performance?",
        "Does power capping affect computational throughput?",
        "Does persistent memory change checkpoint-restart overhead?",
        "Does workload co-location degrade performance predictability?",
        "Does network compression reduce distributed training time?",
        "Does job scheduling policy affect cluster utilization?",
        "Does hardware acceleration outperform CPU for inference?",
        "Does memory compression increase effective capacity?",
        "Does instruction-level parallelism limit CPU performance gains?",
        "Does operating system scheduling affect tail latency?",
        "Does data locality optimization improve Spark performance?",
        "Does kernel bypass reduce network stack overhead?",
        "Does disaggregated storage affect application latency?",
    ],
    "ml_ai_3": [
        "Does flash attention 2 outperform flash attention on H100?",
        "Does grouped query attention reduce KV cache size?",
        "Does sliding window attention handle infinite context?",
        "Does speculative decoding work with quantized models?",
        "Does LoRA rank affect downstream task performance?",
        "Does instruction tuning improve reasoning benchmarks?",
        "Does model size predict emergent capabilities?",
        "Does RLHF improve factual accuracy?",
        "Does prompt length affect LLM reasoning quality?",
        "Does beam search produce more factual outputs than sampling?",
        "Does fine-tuning on code improve math reasoning?",
        "Does multimodal training improve language understanding?",
        "Does self-play improve LLM capabilities?",
        "Does constitutional AI scale to frontier models?",
        "Does debate between models improve truthfulness?",
        "Does activation patching identify causal mechanisms?",
        "Does superposition explain polysemanticity in LLMs?",
        "Does chain of thought improve multi-step reasoning?",
        "Does scratchpad improve LLM arithmetic accuracy?",
        "Does tree of thought outperform chain of thought?",
        "Does graph of thought improve complex reasoning?",
        "Does self-consistency improve factual QA accuracy?",
        "Does retrieval augmentation reduce hallucination rates?",
        "Does calibration training improve LLM uncertainty estimates?",
        "Does long context fine-tuning improve summarization?",
    ],
    "neuroscience_3": [
        "Does gamma oscillation support working memory binding?",
        "Does theta-gamma coupling coordinate hippocampal sequences?",
        "Does sharp wave ripple replay consolidate spatial memory?",
        "Does pattern completion rely on CA3 recurrent connectivity?",
        "Does pattern separation depend on dentate gyrus neurogenesis?",
        "Does prefrontal cortex maintain task rules across delays?",
        "Does anterior cingulate cortex signal prediction errors?",
        "Does orbitofrontal cortex encode expected value?",
        "Does nucleus accumbens gate motivational salience?",
        "Does ventral tegmental area encode reward prediction errors?",
        "Does locus coeruleus modulate global brain state?",
        "Does raphe nucleus serotonin control patience?",
        "Does habenula encode negative prediction errors?",
        "Does superior colliculus prioritize salient stimuli?",
        "Does lateral geniculate nucleus gate visual attention?",
        "Does primary visual cortex encode spatial frequency?",
        "Does inferotemporal cortex represent object identity invariantly?",
        "Does retrosplenial cortex support spatial reference frames?",
        "Does parahippocampal cortex encode scene context?",
        "Does perirhinal cortex support familiarity recognition?",
        "Does the claustrum coordinate cortical synchronization?",
        "Does the insula integrate interoceptive signals?",
        "Does the cerebellum predict sensory consequences of movement?",
        "Does the striatum learn stimulus-response associations?",
        "Does the subthalamic nucleus implement response inhibition?",
    ],
    "biology_3": [
        "Does mechanical force regulate stem cell differentiation?",
        "Does cell geometry affect division orientation?",
        "Does tissue stiffness influence cancer cell migration?",
        "Does extracellular matrix composition affect cell fate?",
        "Does cytoskeletal tension regulate gene expression?",
        "Does nutrient sensing coordinate growth and division?",
        "Does reactive oxygen species signaling regulate lifespan?",
        "Does NAD+ supplementation reverse aging phenotypes?",
        "Does caloric restriction extend lifespan across species?",
        "Does mTOR inhibition mimic caloric restriction benefits?",
        "Does senolytics improve healthspan in aged animals?",
        "Does gut microbiome diversity decline with age?",
        "Does microbiome transfer rejuvenate aged immune systems?",
        "Does epigenetic clock accurately measure biological age?",
        "Does DNA methylation predict age-related disease risk?",
        "Does proteostasis decline drive aging phenotypes?",
        "Does lysosomal function affect protein aggregate clearance?",
        "Does stem cell exhaustion limit tissue regeneration?",
        "Does inflammaging drive chronic disease in old age?",
        "Does telomere shortening cause cellular senescence?",
        "Does mitochondrial fission affect metabolic health?",
        "Does circadian disruption accelerate aging?",
        "Does heat shock response protect against proteotoxic stress?",
        "Does autophagy flux predict longevity?",
        "Does cellular reprogramming reset epigenetic age?",
    ],
    "medicine_3": [
        "Does tumor mutational burden predict immunotherapy response?",
        "Does microsatellite instability predict checkpoint inhibitor benefit?",
        "Does PD-L1 expression predict immunotherapy response?",
        "Does neoadjuvant chemotherapy improve surgical outcomes?",
        "Does adjuvant immunotherapy prevent cancer recurrence?",
        "Does targeted therapy overcome immunotherapy resistance?",
        "Does radiotherapy enhance immunotherapy efficacy?",
        "Does gut microbiome predict immunotherapy outcomes?",
        "Does biomarker-guided therapy improve precision medicine?",
        "Does liquid biopsy monitor treatment response?",
        "Does minimal residual disease predict relapse?",
        "Does chimeric antigen receptor therapy cure leukemia?",
        "Does bispecific antibody therapy improve lymphoma outcomes?",
        "Does antibody drug conjugate improve cancer survival?",
        "Does PARP inhibitor extend ovarian cancer progression-free survival?",
        "Does CDK4/6 inhibitor improve breast cancer outcomes?",
        "Does BCL2 inhibitor treat chronic lymphocytic leukemia?",
        "Does IDH inhibitor treat acute myeloid leukemia?",
        "Does KRAS inhibitor overcome decades of failed attempts?",
        "Does epigenetic therapy sensitize tumors to immunotherapy?",
        "Does metabolic reprogramming affect cancer immunotherapy?",
        "Does tumor microenvironment predict treatment response?",
        "Does angiogenesis inhibition improve cancer outcomes?",
        "Does cancer vaccine induce durable immune responses?",
        "Does adoptive cell therapy treat solid tumors effectively?",
    ],
    "cross_domain": [
        "Does sparse coding in visual cortex inspire sparse attention in transformers?",
        "Does winner-take-all inhibition in neocortex prevent expert collapse in MoE?",
        "Does predictive coding in the brain improve transformer pretraining?",
        "Does Hebbian learning inspire better optimizer design for neural networks?",
        "Does dendritic computation inspire multi-head attention mechanisms?",
        "Does neural oscillation synchrony inspire positional encoding in LLMs?",
        "Does cortical hierarchy inspire deep learning architecture depth?",
        "Does memory reconsolidation inspire continual learning algorithms?",
        "Does synaptic homeostasis inspire weight normalization techniques?",
        "Does neural pruning during development inspire network pruning methods?",
        "Does competitive learning in the brain inspire mixture of experts routing?",
        "Does error backpropagation have biological analogues in the brain?",
        "Does the binding problem in neuroscience inform multimodal AI fusion?",
        "Does working memory capacity limit inspire context window design?",
        "Does attention blink phenomenon inform transformer attention bottlenecks?",
        "Does DNA compression inspire neural network weight compression?",
        "Does immune system memory inspire few-shot learning algorithms?",
        "Does evolutionary selection pressure inspire neural architecture search?",
        "Does protein folding energy minimization inspire loss landscape analysis?",
        "Does cellular signaling cascade inspire deep network information flow?",
        "Does ecosystem diversity inspire ensemble learning methods?",
        "Does natural language evolution inspire LLM training data curation?",
        "Does economic market equilibrium inspire Nash equilibrium in GANs?",
        "Does physical entropy inspire information theoretic regularization?",
        "Does quantum superposition inspire probabilistic neural networks?",
    ],
    "applications": [
        "Does LLM-based code generation reduce software bugs?",
        "Does AI-assisted drug discovery accelerate clinical trials?",
        "Does machine learning improve weather forecasting accuracy?",
        "Does deep learning match radiologist accuracy for cancer detection?",
        "Does NLP improve clinical documentation efficiency?",
        "Does reinforcement learning solve real-world robotics tasks?",
        "Does AI improve supply chain optimization?",
        "Does machine translation match human quality for medical documents?",
        "Does AI-powered tutoring improve student learning outcomes?",
        "Does predictive maintenance reduce industrial downtime?",
        "Does AI fraud detection outperform rule-based systems?",
        "Does autonomous vehicle AI reduce accident rates?",
        "Does AI improve protein structure prediction beyond AlphaFold?",
        "Does neural machine translation preserve medical nuance?",
        "Does AI clinical decision support reduce diagnostic errors?",
        "Does deep learning improve satellite image analysis?",
        "Does AI accelerate materials discovery?",
        "Does NLP improve systematic review efficiency?",
        "Does AI improve financial risk assessment?",
        "Does machine learning predict treatment response in psychiatry?",
        "Does AI improve legal document review accuracy?",
        "Does computer vision match pathologist accuracy?",
        "Does AI improve crop yield prediction?",
        "Does deep learning improve earthquake early warning?",
        "Does AI accelerate vaccine candidate identification?",
    ],
    "safety_alignment": [
        "Does RLHF reliably align LLM behavior with human values?",
        "Does constitutional AI reduce harmful outputs at scale?",
        "Does red teaming effectively identify LLM safety failures?",
        "Does model scale increase or decrease alignment difficulty?",
        "Does instruction following generalize to novel harmful requests?",
        "Does chain of thought improve or worsen sycophancy?",
        "Does debate between AI systems improve truthfulness?",
        "Does interpretability research enable better alignment?",
        "Does activation steering reliably control LLM behavior?",
        "Does fine-tuning on safety data generalize to new situations?",
        "Does reward hacking undermine RLHF alignment?",
        "Does refusal training cause excessive over-refusal?",
        "Does model honesty training improve factual accuracy?",
        "Does AI watermarking reliably detect AI-generated content?",
        "Does adversarial training improve LLM robustness?",
        "Does capability evaluation predict dangerous AI behavior?",
        "Does sandbagging occur in AI safety evaluations?",
        "Does deceptive alignment emerge in advanced AI systems?",
        "Does model collapse occur when training on AI-generated data?",
        "Does RLHF improve or worsen model calibration?",
        "Does safety training reduce model capability?",
        "Does representation engineering control model behavior?",
        "Does model size affect susceptibility to jailbreaking?",
        "Does few-shot prompting bypass safety training?",
        "Does model merging preserve safety properties?",
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


def retrieve_domain_chunks(query, domain, top_k=8):
    """Retrieve chunks filtered by domain."""
    import sqlite3
    conn = sqlite3.connect('knowledge_base/knowledge.db')
    stopwords = {'does','what','when','how','the','and','for','with',
                'that','this','are','was','were','have','from','will',
                'would','could','should','than','then','very','also'}
    keywords = [w.strip('?.,!') for w in query.lower().split()
               if len(w.strip('?.,!')) > 3 and w.strip('?.,!') not in stopwords][:4]
    try:
        if keywords:
            conditions = ' OR '.join([f"text LIKE ?" for _ in keywords])
            params = [f'%{k}%' for k in keywords] + [domain]
            rows = conn.execute(
                f"SELECT text, paper_title, domain, source FROM chunks WHERE ({conditions}) AND domain=? ORDER BY RANDOM() LIMIT ?",
                params + [top_k]
            ).fetchall()
            if len(rows) >= 2:
                conn.close()
                return [{'text':r[0],'paper_title':r[1],'domain':r[2],'source':r[3]} for r in rows]
        # Fallback: just domain
        rows = conn.execute(
            "SELECT text, paper_title, domain, source FROM chunks WHERE domain=? ORDER BY RANDOM() LIMIT ?",
            (domain, top_k)
        ).fetchall()
        conn.close()
        return [{'text':r[0],'paper_title':r[1],'domain':r[2],'source':r[3]} for r in rows]
    except Exception as e:
        try: conn.close()
        except: pass
        return []

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
