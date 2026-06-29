# background_service.py
# ─────────────────────────────────────────────────────────────
# Background ingestion service — pulls from ALL free research
# sources on the internet, every 6 hours.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import os, sys, json, time, threading, logging, hashlib, requests, subprocess, sqlite3, shutil
from datetime import datetime
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
import urllib.parse

log = logging.getLogger("tattva.background")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

ROOT        = os.path.dirname(os.path.abspath(__file__))
STATUS_PATH = os.path.join(ROOT, "data", "service_status.json")
CHUNK_DB    = os.path.join(ROOT, "knowledge_base", "knowledge.db")
INTERVAL_SEC = 6 * 60 * 60  # 6 hours

# ── Domain Configuration ─────────────────────────────────
DOMAIN_QUERIES = {
    "ml_ai": [
        "transformer attention efficiency memory",
        "mixture of experts language model routing",
        "KV cache compression LLM inference",
        "FlashAttention IO aware memory efficient attention",
        "LoRA low rank adaptation fine-tuning",
        "speculative decoding LLM throughput",
        "sparse attention long context transformer",
        "quantization LLM inference INT8 INT4",
        "MoE routing instability training sparse",
        "continuous batching LLM serving vLLM",
        "grouped query attention KV cache memory",
        "PagedAttention virtual memory GPU LLM",
        "pruning transformer inference cost",
        "knowledge distillation model compression",
        "sliding window attention sequence length",
        "Mamba state space model transformers linear",
        "tensor parallelism distributed LLM training",
        "RWKV linear complexity recurrent transformer",
        "prefix caching KV reuse inference",
        "flash decoding parallel LLM inference",
    ],
    "neuroscience": [
        "neural attention mechanism selective perception brain",
        "memory consolidation hippocampus sleep",
        "synaptic plasticity hebbian learning",
        "sparse coding visual cortex energy efficiency",
        "predictive coding brain information processing",
        "dopamine reward learning basal ganglia",
        "prefrontal cortex decision making working memory",
        "neural oscillations cognition gamma theta",
        "cortical hierarchy sensory processing",
        "neuroplasticity learning experience brain",
        "default mode network resting state fMRI",
        "optogenetics neural circuit causality",
        "connectome neural wiring brain mapping",
        "glial cells neuron support function",
        "neurodegeneration Alzheimer Parkinson mechanism",
    ],
    "biology": [
        "CRISPR Cas9 gene editing off-target specificity",
        "protein folding structure prediction AlphaFold",
        "gene regulatory network boolean transcription",
        "cell signaling pathway receptor kinase",
        "evolutionary biology natural selection fitness",
        "microbiome gut bacteria health disease",
        "stem cell differentiation pluripotent",
        "epigenetics DNA methylation gene expression",
        "CRISPR base editing prime editing precision",
        "single cell RNA sequencing transcriptomics",
        "synthetic biology genetic circuit design",
        "metabolic engineering biosynthesis pathway",
        "protein protein interaction network disease",
        "mRNA vaccine delivery lipid nanoparticle",
        "antibiotic resistance mechanism evolution",
    ],
    "medicine": [
        "clinical trial randomized controlled efficacy",
        "cancer immunotherapy checkpoint inhibitor",
        "drug resistance tumor microenvironment",
        "biomarker disease diagnosis prognosis",
        "precision medicine genomics personalized treatment",
        "vaccine efficacy immunogenicity safety",
        "mental health depression anxiety treatment",
        "cardiovascular disease risk factor prevention",
        "diabetes insulin resistance metabolic syndrome",
        "COVID long covid mechanism treatment",
        "CAR T cell therapy cancer immunology",
        "CRISPR gene therapy clinical trial",
        "microbiome disease causality treatment",
        "aging longevity mechanism intervention",
        "drug repurposing computational discovery",
    ],
    "physics": [
        "quantum computing error correction fault tolerant",
        "quantum entanglement measurement Bell inequality",
        "superconductor high temperature mechanism BCS",
        "dark matter detection direct experiment",
        "quantum field theory standard model particle",
        "thermodynamics entropy information theory",
        "condensed matter phase transition critical phenomena",
        "quantum annealing optimization classical comparison",
        "topological insulator quantum material property",
        "gravitational wave detection LIGO black hole",
        "photonic quantum computing optical qubit",
        "nuclear fusion plasma confinement tokamak",
        "quantum sensing metrology precision measurement",
        "many body quantum simulation lattice",
        "semiconductor material property doping",
    ],
    "chemistry": [
        "catalyst reaction mechanism selectivity",
        "organic synthesis drug molecule",
        "electrochemical energy storage battery",
        "photocatalysis solar energy conversion",
        "nanoparticle synthesis optical electronic",
        "polymer material mechanical thermal property",
        "green chemistry sustainable solvent",
        "reaction kinetics transition state theory",
        "computational chemistry DFT molecular dynamics",
        "metal organic framework porous material",
        "electrocatalysis hydrogen evolution oxygen",
        "surface chemistry heterogeneous catalysis",
        "supramolecular chemistry self assembly",
        "flow chemistry continuous synthesis",
        "analytical chemistry spectroscopy detection",
    ],
    "climate": [
        "carbon capture sequestration storage efficiency",
        "climate model temperature projection scenario",
        "renewable energy solar wind efficiency cost",
        "ocean acidification marine ecosystem impact",
        "greenhouse gas emission reduction pathway",
        "sea level rise ice sheet melting prediction",
        "biodiversity loss climate change species",
        "carbon cycle feedback methane permafrost",
        "climate tipping point threshold irreversible",
        "direct air capture carbon dioxide removal",
        "electric vehicle battery grid decarbonization",
        "deforestation land use carbon stock",
        "climate adaptation resilience vulnerability",
        "geoengineering solar radiation management",
        "extreme weather event attribution climate",
    ],
    "economics": [
        "minimum wage employment effect labor market",
        "monetary policy inflation central bank",
        "income inequality growth redistribution",
        "trade policy welfare gains tariff",
        "behavioral economics bias decision making",
        "market efficiency hypothesis asset pricing",
        "fiscal policy multiplier government spending",
        "automation labor displacement inequality",
        "universal basic income poverty experiment",
        "healthcare economics insurance market failure",
        "environmental economics carbon tax externality",
        "development economics poverty growth institution",
        "financial crisis contagion systemic risk",
        "game theory mechanism design auction",
        "economic growth productivity innovation",
    ],
    "psychology": [
        "cognitive behavioral therapy efficacy depression",
        "mindfulness meditation stress anxiety reduction",
        "cognitive bias heuristic judgment decision",
        "trauma PTSD treatment exposure therapy",
        "personality trait heritability twin study",
        "social conformity influence group behavior",
        "memory false recall reconstruction",
        "attention deficit ADHD treatment intervention",
        "motivation intrinsic extrinsic reward",
        "emotion regulation strategy mental health",
        "sleep quality mental health performance",
        "placebo effect expectation treatment",
        "childhood adversity adult mental health",
        "positive psychology wellbeing intervention",
        "psychedelic therapy psilocybin depression",
    ],
    "computer_systems": [
        "cache memory hierarchy optimization prefetch",
        "distributed systems consensus fault tolerance",
        "compiler optimization loop vectorization",
        "operating system scheduler fairness latency",
        "network congestion control TCP throughput",
        "storage system log structured write amplification",
        "database query optimization index",
        "security vulnerability side channel attack",
        "cloud computing resource allocation scheduling",
        "edge computing inference latency mobile",
        "Byzantine fault tolerance blockchain consensus",
        "memory safety Rust systems programming",
        "kernel bypass DPDK network performance",
        "containerization microservice orchestration",
        "formal verification program correctness proof",
    ],
    "mathematics": [
        "optimal transport Wasserstein distance application",
        "random matrix theory neural network spectrum",
        "information geometry Fisher information manifold",
        "topology persistent homology data analysis",
        "stochastic differential equation sampling",
        "graph theory network analysis spectral",
        "compressed sensing sparse recovery measurement",
        "convex optimization convergence rate algorithm",
        "differential geometry Riemannian manifold learning",
        "number theory cryptography prime factorization",
        "probability theory concentration inequality",
        "numerical analysis stability convergence",
        "algebraic geometry machine learning application",
        "combinatorics counting probabilistic method",
        "functional analysis operator Hilbert space",
    ],
}

# Domain detection — tag every chunk with its domain
def detect_domain(text: str, title: str = "", source_query: str = "") -> str:
    """
    Detect domain using STRICT signals.
    Priority: source_query (most reliable) > title signals > text signals
    """
    combined = (text + " " + title).lower()
    title_lower = title.lower()

    # If we know which query fetched this paper, use that directly
    if source_query:
        for domain, queries in DOMAIN_QUERIES.items():
            if source_query in queries:
                return domain

    # STRICT title-based detection — titles are reliable
    # These are domain-exclusive terms unlikely to appear in ML papers
    title_signals = {
        "neuroscience": ["hippocampus","synapse","cortex","neuron","fmri","eeg","dopamine","serotonin","axon","dendrite","glial","amygdala","cerebellum","neural circuit","prefrontal","optogenetics"],
        "biology":      ["crispr","gene editing","protein folding","genomics","transcriptomics","dna methylation","rna sequencing","cell biology","microbiome","epigenetics","stem cell","metabolomics"],
        "medicine":     ["clinical trial","randomized controlled","patient outcome","drug efficacy","disease treatment","cancer therapy","vaccine efficacy","surgical outcome","epidemiology","cohort study"],
        "physics":      ["quantum entanglement","superconductor","dark matter","particle physics","quantum computing","thermodynamics","condensed matter","gravitational wave","nuclear physics","plasma physics"],
        "chemistry":    ["organic synthesis","catalyst","chemical reaction","polymer","electrochemistry","spectroscopy","crystallography","nanoparticle synthesis","photocatalysis","reaction mechanism"],
        "climate":      ["climate change","carbon emission","greenhouse gas","sea level","ocean acidification","biodiversity loss","renewable energy","carbon capture","global warming","fossil fuel"],
        "economics":    ["gdp growth","monetary policy","labor market","fiscal policy","income inequality","trade policy","unemployment rate","inflation rate","economic growth","market equilibrium"],
        "psychology":   ["cognitive behavioral","mental health","anxiety disorder","depression treatment","psychological","ptsd","personality disorder","behavioral therapy","mindfulness","cognitive bias"],
        "computer_systems": ["operating system","distributed system","network protocol","compiler optimization","database system","file system","cache replacement","cpu scheduling","memory management"],
        "mathematics":  ["mathematical proof","theorem","algebraic","topological","stochastic process","differential equation","linear algebra","number theory","combinatorics","functional analysis"],
    }

    for domain, terms in title_signals.items():
        if any(term in title_lower for term in terms):
            return domain

    # ML/AI is the default for our current corpus
    # Only assign other domains if VERY strong signals in text
    strong_text_signals = {
        "neuroscience": ["hippocampal","synaptic plasticity","neural firing","cortical","cerebellar","dopaminergic"],
        "biology":      ["crispr-cas","gene expression","rna-seq","whole genome","protein structure","cell line","in vitro","in vivo"],
        "medicine":     ["patients with","clinical outcomes","randomized","placebo-controlled","hazard ratio","odds ratio"],
        "physics":      ["quantum circuit","qubit","superconducting","entanglement","hamiltonian","fermion","boson"],
        "chemistry":    ["synthesis of","reaction yield","catalytic activity","spectroscopic","crystalline"],
        "climate":      ["co2 concentration","sea surface temperature","carbon cycle","climate model","ipcc"],
        "economics":    ["gross domestic","central bank","elasticity","supply and demand","econometric"],
        "psychology":   ["participants reported","clinical sample","psychometric","behavioral intervention"],
        "computer_systems": ["page fault","cache miss","context switch","throughput latency","packet loss"],
        "mathematics":  ["we prove","lemma","corollary","let g be","suppose x"],
    }

    for domain, terms in strong_text_signals.items():
        if sum(1 for t in terms if t in combined) >= 2:
            return domain

    return "ml_ai"

# Flatten all queries for backward compatibility
SEARCH_QUERIES = [q for queries in DOMAIN_QUERIES.values() for q in queries]

_LEGACY_SEARCH_QUERIES = []  # moved to DOMAIN_QUERIES above

DOMAIN_KEYWORDS = [
    "attention", "transformer", "language model", "llm", "inference",
    "fine-tuning", "finetuning", "training", "efficiency", "memory",
    "latency", "throughput", "quantization", "pruning", "distillation",
    "sparse", "mixture of experts", "moe", "kv cache", "flashattention",
    "lora", "speculative", "mamba", "rwkv", "paged", "rotary", "rope",
    "neural network", "deep learning", "bert", "gpt", "llama", "mistral",
    "parameter efficient", "model compression", "sequence length",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "TattvaAI/1.0 (research aggregator)"})


# ── Helpers ───────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text: return ""
    return " ".join(text.strip().split())

def _strip_xml(text: str) -> str:
    import re
    return re.sub(r'<[^>]+>', ' ', text).strip()

def _is_relevant(paper: Dict) -> bool:
    text = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
    return sum(1 for kw in DOMAIN_KEYWORDS if kw in text) >= 2

def _chunk_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def _get_existing_hashes() -> set:
    try:
        conn = sqlite3.connect(CHUNK_DB)
        rows = conn.execute("SELECT text FROM chunks").fetchall()
        conn.close()
        return {_chunk_hash(r[0]) for r in rows}
    except Exception:
        return set()

def _get_chunk_count() -> int:
    try:
        conn = sqlite3.connect(CHUNK_DB)
        n = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        conn.close()
        return n
    except Exception:
        return 0


# ── Status ────────────────────────────────────────────────

def _write_status(status: Dict):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2, default=str)

def read_status() -> Dict:
    try:
        if os.path.exists(STATUS_PATH):
            with open(STATUS_PATH) as f: return json.load(f)
    except Exception: pass
    return {"state":"idle","last_run":None,"next_run":None,"total_chunks":0,"cycles":0}


# ── PDF extraction ────────────────────────────────────────

def _fetch_pdf_text(pdf_url: str, max_chars: int = 8000) -> Optional[str]:
    if not pdf_url or not pdf_url.startswith("http"): return None
    try:
        r = SESSION.get(pdf_url, timeout=30, stream=True)
        if r.status_code != 200: return None
        if "pdf" not in r.headers.get("content-type","").lower(): return None
        content = b""
        for chunk in r.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 2_000_000: break
        import io
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = " ".join(page.extract_text() or "" for page in reader.pages[:20])
            return _clean(text)[:max_chars] if text.strip() else None
        except ImportError: pass
        try:
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            output = io.StringIO()
            extract_text_to_fp(io.BytesIO(content), output, laparams=LAParams())
            text = output.getvalue()
            return _clean(text)[:max_chars] if text.strip() else None
        except ImportError: pass
        return None
    except Exception:
        return None


# ── Source fetchers ───────────────────────────────────────

def fetch_semantic_scholar(query: str, limit: int = 25) -> List[Dict]:
    try:
        r = SESSION.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": limit,
                    "fields": "title,abstract,year,openAccessPdf"},
            timeout=20)
        if r.status_code == 429: time.sleep(5); return []
        if r.status_code != 200: return []
        papers = []
        for p in r.json().get("data", []):
            title    = _clean(p.get("title",""))
            abstract = _clean(p.get("abstract","") or "")
            full_text = abstract
            pdf_url = (p.get("openAccessPdf") or {}).get("url")
            if pdf_url: full_text = _fetch_pdf_text(pdf_url) or abstract
            if title and abstract and len(abstract) > 80:
                papers.append({"title":title,"abstract":abstract,
                                "full_text":full_text,"source":"semantic_scholar"})
        return papers
    except Exception as e: log.debug(f"S2: {e}"); return []


def fetch_arxiv(query: str, limit: int = 25) -> List[Dict]:
    try:
        params = (f"search_query=all:{urllib.parse.quote(query)}"
                  f"&max_results={limit}&sortBy=submittedDate&sortOrder=descending")
        r = SESSION.get(f"http://export.arxiv.org/api/query?{params}", timeout=20)
        if r.status_code != 200: return []
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            title    = _clean((entry.find("atom:title",ns) or {}).text or "").replace("\n"," ")
            abstract = _clean((entry.find("atom:summary",ns) or {}).text or "").replace("\n"," ")
            pdf_url  = None
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href","")
            full_text = abstract
            if pdf_url: full_text = _fetch_pdf_text(pdf_url) or abstract
            if title and abstract and len(abstract) > 80:
                papers.append({"title":title,"abstract":abstract,
                                "full_text":full_text,"source":"arxiv"})
        return papers
    except Exception as e: log.debug(f"ArXiv: {e}"); return []


def fetch_openalex(query: str, limit: int = 25) -> List[Dict]:
    try:
        r = SESSION.get("https://api.openalex.org/works",
            params={"search":query,"per_page":limit,
                    "sort":"publication_date:desc",
                    "select":"title,abstract_inverted_index,open_access"},
            timeout=20)
        if r.status_code != 200: return []
        papers = []
        for work in r.json().get("results", []):
            title = _clean(work.get("title","") or "")
            inv   = work.get("abstract_inverted_index") or {}
            abstract = _reconstruct_abstract(inv)
            if not abstract or len(abstract) <= 80: continue
            full_text = abstract
            oa = work.get("open_access",{})
            pdf_url = oa.get("oa_url") if oa.get("is_oa") else None
            if pdf_url and pdf_url.endswith(".pdf"):
                full_text = _fetch_pdf_text(pdf_url) or abstract
            if title:
                papers.append({"title":title,"abstract":abstract,
                                "full_text":full_text,"source":"openalex"})
        return papers
    except Exception as e: log.debug(f"OpenAlex: {e}"); return []

def _reconstruct_abstract(inv: Dict) -> str:
    if not inv: return ""
    try:
        max_pos = max(pos for positions in inv.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inv.items():
            for pos in positions: words[pos] = word
        return " ".join(w for w in words if w)
    except Exception: return ""


def fetch_core(query: str, limit: int = 20) -> List[Dict]:
    """Fetch from Unpaywall/DOAJ as CORE replacement."""
    try:
        r = SESSION.get("https://doaj.org/api/search/articles/" + urllib.parse.quote(query),
            params={"pageSize":min(limit,10)}, timeout=15)
        if r.status_code != 200: return []
        papers = []
        for p in r.json().get("results",[]):
            bib = p.get("bibjson",{})
            title = _clean(bib.get("title","") or "")
            abstract = _clean(bib.get("abstract","") or "")
            if title and abstract and len(abstract) > 80:
                papers.append({"title":title,"abstract":abstract,
                                "full_text":abstract,"source":"doaj"})
        return papers
    except Exception as e: log.debug(f"DOAJ: {e}"); return []


def fetch_papers_with_code(query: str, limit: int = 20) -> List[Dict]:
    try:
        r = SESSION.get("https://paperswithcode.com/api/v1/papers/",
            params={"q":query,"page_size":limit}, timeout=20)
        if r.status_code != 200: return []
        data = r.json()
        if not isinstance(data, dict): return []
        papers = []
        for p in data.get("results", []):
            title    = _clean(p.get("title","") or "")
            abstract = _clean(p.get("abstract","") or "")
            if title and abstract and len(abstract) > 80:
                papers.append({"title":title,"abstract":abstract,
                                "full_text":abstract,"source":"papers_with_code"})
        return papers
    except Exception as e: log.debug(f"PWC: {e}"); return []


def fetch_huggingface(query: str, limit: int = 20) -> List[Dict]:
    try:
        r = SESSION.get("https://huggingface.co/api/papers",
            params={"q":query}, timeout=20)
        if r.status_code != 200: return []
        data = r.json() if isinstance(r.json(), list) else r.json().get("papers", [])
        papers = []
        for p in data[:limit]:
            title    = _clean(p.get("title","") or "")
            abstract = _clean(p.get("abstract","") or p.get("summary","") or "")
            if title and abstract and len(abstract) > 80:
                papers.append({"title":title,"abstract":abstract,
                                "full_text":abstract,"source":"huggingface"})
        return papers
    except Exception as e: log.debug(f"HF: {e}"); return []


def fetch_europe_pmc(query: str, limit: int = 20) -> List[Dict]:
    try:
        r = SESSION.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query":query,"resultType":"core","pageSize":limit,
                    "format":"json","sort":"P_PDATE_D desc"}, timeout=20)
        if r.status_code != 200: return []
        papers = []
        for p in r.json().get("resultList",{}).get("result",[]):
            title    = _clean(p.get("title","") or "")
            abstract = _clean(p.get("abstractText","") or "")
            if title and abstract and len(abstract) > 80:
                papers.append({"title":title,"abstract":abstract,
                                "full_text":abstract,"source":"europe_pmc"})
        return papers
    except Exception as e: log.debug(f"EPMC: {e}"); return []


def fetch_crossref(query: str, limit: int = 20) -> List[Dict]:
    try:
        r = SESSION.get("https://api.crossref.org/works",
            params={"query":query,"rows":limit,
                    "select":"title,abstract","sort":"published","order":"desc"},
            timeout=20)
        if r.status_code != 200: return []
        papers = []
        for p in r.json().get("message",{}).get("items",[]):
            titles   = p.get("title",[])
            title    = _clean(titles[0] if titles else "")
            abstract = _clean(_strip_xml(p.get("abstract","") or ""))
            if title and abstract and len(abstract) > 80:
                papers.append({"title":title,"abstract":abstract,
                                "full_text":abstract,"source":"crossref"})
        return papers
    except Exception as e: log.debug(f"CrossRef: {e}"); return []


def fetch_all_sources(query: str) -> List[Dict]:
    results = []
    for name, fetcher in [
        ("Semantic Scholar", fetch_semantic_scholar),
        ("ArXiv",            fetch_arxiv),
        ("OpenAlex",         fetch_openalex),
        ("CORE",             fetch_core),
        ("PapersWithCode",   fetch_papers_with_code),
        ("HuggingFace",      fetch_huggingface),
        ("EuropePMC",        fetch_europe_pmc),
        ("CrossRef",         fetch_crossref),
    ]:
        try:
            papers = fetcher(query, limit=30)
            results.extend(papers)
            log.debug(f"  {name}: {len(papers)} papers")
        except Exception as e:
            log.debug(f"  {name} failed: {e}")
        time.sleep(0.5)
    return results


# ── Direct SQLite ingestion ───────────────────────────────

def _ingest_directly(papers: List[Dict]) -> int:
    existing_hashes = _get_existing_hashes()
    log.info(f"  Existing chunk hashes loaded: {len(existing_hashes)}")

    new_chunks = []
    for paper in papers:
        text_to_chunk = paper.get("full_text","") or paper["abstract"]
        if len(text_to_chunk) < len(paper["abstract"]):
            text_to_chunk = paper["abstract"]
        words = text_to_chunk.split()
        chunk_size = 200
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            if len(chunk_text) < 50: continue
            h = _chunk_hash(chunk_text)
            if h not in existing_hashes:
                existing_hashes.add(h)
                new_chunks.append({
                    "text":        chunk_text,
                    "paper_title": paper["title"],
                    "source":      paper["source"],
                    "domain":      detect_domain(chunk_text, paper.get("title","")),
                    "chunk_idx":   i // chunk_size,
                })

    if not new_chunks:
        log.info("  No new chunks to add (all duplicates)")
        return 0

    try:
        conn = sqlite3.connect(CHUNK_DB)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
        added = 0
        for chunk in new_chunks:
            try:
                data = {"text": chunk["text"]}
                if "paper_title" in cols: data["paper_title"] = chunk["paper_title"]
                if "source"      in cols: data["source"]      = chunk["source"]
                if "domain"      in cols: data["domain"]      = chunk["domain"]
                if "chunk_idx"   in cols: data["chunk_idx"]   = chunk["chunk_idx"]
                if "metadata"    in cols:
                    data["metadata"] = json.dumps({
                        "paper_title": chunk["paper_title"],
                        "source":      chunk["source"],
                        "domain":      chunk["domain"],
                    })
                cols_str = ", ".join(data.keys())
                vals_str = ", ".join("?" * len(data))
                conn.execute(f"INSERT INTO chunks ({cols_str}) VALUES ({vals_str})",
                             list(data.values()))
                added += 1
            except Exception as e:
                log.debug(f"  Chunk insert failed: {e}")
        conn.commit()
        conn.close()
        log.info(f"  Directly inserted {added} new chunks into SQLite")

        # Sync new chunks to Supabase automatically
        if added > 0:
            try:
                from supabase import create_client
                import os
                sb_url = os.environ.get("SUPABASE_URL")
                sb_key = os.environ.get("SUPABASE_KEY")
                if sb_url and sb_key:
                    sb = create_client(sb_url, sb_key)
                    # Get chunks not yet in Supabase
                    r = sb.table("chunks").select("id", count="exact").execute()
                    sb_count = r.count or 0
                    conn2 = sqlite3.connect(DB_PATH)
                    conn2.row_factory = sqlite3.Row
                    new_rows = [dict(r) for r in conn2.execute(
                        "SELECT * FROM chunks ORDER BY rowid LIMIT -1 OFFSET ?", (sb_count,)
                    ).fetchall()]
                    conn2.close()
                    if new_rows:
                        for i in range(0, len(new_rows), 100):
                            batch = new_rows[i:i+100]
                            data = [{"text": r.get("text",""), "paper_title": r.get("paper_title",""),
                                     "source": r.get("source",""), "domain": detect_domain(r.get("text",""), r.get("paper_title","")),
                                     "chunk_idx": 0} for r in batch]
                            sb.table("chunks").insert(data).execute()
                        log.info(f"  Synced {len(new_rows)} new chunks to Supabase")
            except Exception as e:
                log.warning(f"  Supabase sync failed: {e}")

        return added
    except Exception as e:
        log.error(f"  Direct SQLite ingestion failed: {e}")
        return 0


# ── Main cycle ────────────────────────────────────────────

def run_cycle():
    start = time.time()
    prev  = read_status()
    log.info("=" * 60)
    log.info(f"Background cycle starting — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    _write_status({**prev, "state":"running", "phase":"fetching papers",
                   "started_at": datetime.now().isoformat()})

    # ── Phase 1: Fetch ────────────────────────────────────
    log.info(f"Phase 1: Fetching from all sources ({len(SEARCH_QUERIES)} queries across {len(DOMAIN_QUERIES)} domains)")
    log.info("Domains: " + ", ".join(f"{d}({len(q)}q)" for d,q in DOMAIN_QUERIES.items()))
    all_papers: Dict[str, Dict] = {}

    for i, query in enumerate(SEARCH_QUERIES):
        log.info(f"  [{i+1}/{len(SEARCH_QUERIES)}] {query}")
        batch = fetch_all_sources(query)
        added = 0
        for paper in batch:
            if not _is_relevant(paper): continue
            key = hashlib.md5(paper["title"].lower().strip().encode()).hexdigest()
            if key not in all_papers:
                all_papers[key] = paper
                added += 1
        log.info(f"    → {added} new unique relevant papers")
        time.sleep(1)

    new_papers = list(all_papers.values())
    log.info(f"Total unique relevant papers: {len(new_papers)}")

    # ── Phase 2: Ingest directly into SQLite ─────────────
    log.info(f"Phase 2: Ingesting {len(new_papers)} papers")
    _write_status({**read_status(), "phase": f"ingesting {len(new_papers)} papers"})
    chunks_added = _ingest_directly(new_papers)
    log.info(f"  {chunks_added} new chunks added to database")

    # ── Phase 3: Rebuild index ────────────────────────────
    log.info("Phase 3: Rebuilding chunk index")
    _write_status({**read_status(), "phase": "rebuilding chunk index"})
    try:
        # First sync ALL local chunks to Supabase
        sb_url = os.environ.get("SUPABASE_URL","")
        sb_key = os.environ.get("SUPABASE_KEY","")
        if sb_url and sb_key:
            try:
                from supabase import create_client
                sb = create_client(sb_url, sb_key)
                r = sb.table("chunks").select("id", count="exact").execute()
                sb_count = r.count or 0
                conn = sqlite3.connect(DB_PATH)
                conn.row_factory = sqlite3.Row
                local_rows = [dict(r) for r in conn.execute(
                    "SELECT * FROM chunks ORDER BY rowid LIMIT -1 OFFSET ?", (sb_count,)
                ).fetchall()]
                conn.close()
                if local_rows:
                    for i in range(0, len(local_rows), 100):
                        batch = local_rows[i:i+100]
                        data = [{"text": r.get("text",""), "paper_title": r.get("paper_title",""),
                                 "source": r.get("source",""), "domain": detect_domain(r.get("text",""), r.get("paper_title","")),
                                 "chunk_idx": 0} for r in batch]
                        sb.table("chunks").insert(data).execute()
                    log.info(f"Supabase synced: +{len(local_rows)} chunks")
            except Exception as e:
                log.warning(f"Supabase sync failed: {e}")
        from retrieval.simple_retriever import SimpleRetriever
        from learning_module.embedding_bridge import EmbeddingBridge
        encoder = EmbeddingBridge()
        _idx = SimpleRetriever(encoder=encoder)
        _idx.rebuild()
        log.info("Index rebuilt successfully")
        # Reset Flask pipeline so it reloads the new index
        try:
            from dashboard.app import _reset_pipeline
            _reset_pipeline()
            log.info("Flask pipeline reset after index rebuild")
        except Exception as e:
            log.warning(f"Pipeline reset failed: {e}")
    except Exception as e:
        log.error(f"Index rebuild failed: {e}")

    # ── Phase 3.5: Auto-update Knowledge Graph ──────────────
    log.info("Phase 3.5: Auto-updating knowledge graph from new chunks")
    try:
        from knowledge_graph.graph import KnowledgeGraph
        import sqlite3, json as _json
        _kg = KnowledgeGraph()
        _kg.load(os.path.join(ROOT, "knowledge_graph", "graph.json"))
        before = _kg.edge_count()
        # Get recent chunks
        conn = sqlite3.connect(DB_PATH)
        rows = [r[0] for r in conn.execute("SELECT text FROM chunks ORDER BY rowid DESC LIMIT 2000").fetchall()]
        conn.close()
        new_rels = _kg.build_from_corpus(rows)
        after = _kg.edge_count()
        _kg.save(os.path.join(ROOT, "knowledge_graph", "graph.json"))
        log.info(f"KG updated: {before} → {after} edges (+{after-before} new)")
    except Exception as e:
        log.error(f"KG auto-update failed: {e}")

    # ── Phase 4: Reasoning eval ───────────────────────────
    log.info("Phase 4: Running reasoning_eval.py")
    _write_status({**read_status(), "phase": "running reasoning pipeline"})
    result = None
    try:
        result = subprocess.run(
            [sys.executable, "testing_module/reasoning_eval.py"],
            cwd=ROOT, capture_output=True, text=True, timeout=1200)
        if result.stdout: log.info(result.stdout[-500:])
        if result.returncode != 0 and result.stderr:
            log.warning(f"Reasoning eval stderr: {result.stderr[-300:]}")
    except Exception as e:
        log.error(f"Reasoning eval failed: {e}")

    # ── Phase 5: Hypotheses ───────────────────────────────
    log.info("Phase 5: Regenerating hypotheses")
    _write_status({**read_status(), "phase": "regenerating hypotheses"})
    try:
        from knowledge_graph.graph import KnowledgeGraph
        from learning_module.embedding_bridge import EmbeddingBridge
        from reasoning_module.hypothesis_generator import HypothesisGenerator
        from knowledge_base.chunk_store import ChunkStore
        import json as _json
        _kg = KnowledgeGraph()
        _kg.load(os.path.join(ROOT, "knowledge_graph", "graph.json"))
        _enc = EmbeddingBridge()
        _cs = ChunkStore()
        _gen = HypothesisGenerator(kg=_kg, kb=_cs, encoder=_enc)
        _hyps = _gen.generate(top_n=30)
        with open(HYP_JSONL, "w") as _f:
            for _h in _hyps:
                _f.write(_json.dumps(_h) + "\n")
        log.info(f"Hypotheses regenerated: {len(_hyps)}")
    except Exception as e:
        log.error(f"Hypothesis generation failed: {e}")

    # ── Done ──────────────────────────────────────────────
    total_chunks = _get_chunk_count()
    elapsed      = round(time.time() - start, 1)
    log.info(f"Cycle complete in {elapsed}s — {total_chunks} total chunks (+{chunks_added} new)")
    log.info("=" * 60)

    _write_status({
        "state":         "idle",
        "phase":         "complete",
        "last_run":      datetime.now().isoformat(),
        "next_run":      datetime.fromtimestamp(time.time() + INTERVAL_SEC).isoformat(),
        "total_chunks":  total_chunks,
        "chunks_added":  chunks_added,
        "papers_fetched":len(new_papers),
        "elapsed_sec":   elapsed,
        "cycles":        prev.get("cycles", 0) + 1,
        "sources":       ["semantic_scholar","arxiv","openalex","core",
                          "papers_with_code","huggingface","europe_pmc","crossref"],
    })


# ── Daemon thread ─────────────────────────────────────────

def start_background_service(run_immediately: bool = False):
    def loop():
        if run_immediately:
            log.info("Background service: running initial cycle now")
            try: run_cycle()
            except Exception as e: log.error(f"Initial cycle: {e}")
        else:
            log.info("Background service: first auto-run in 6h")
            _write_status({
                "state":    "idle",
                "phase":    "waiting — first run in 6h",
                "last_run": None,
                "next_run": datetime.fromtimestamp(time.time() + INTERVAL_SEC).isoformat(),
                "cycles":   0,
            })
            time.sleep(INTERVAL_SEC)
        while True:
            try: run_cycle()
            except Exception as e:
                log.error(f"Cycle failed: {e}")
                _write_status({**read_status(), "state":"error", "error":str(e)})
            time.sleep(INTERVAL_SEC)

    t = threading.Thread(target=loop, daemon=True, name="tattva-bg")
    t.start()
    log.info("Background service thread started (6h interval)")
    return t
if __name__ == "__main__":
    import sys
    run_now = "--now" in sys.argv
    print(f"Starting Tattva background service (run_now={run_now})")
    t = start_background_service(run_immediately=run_now)
    try:
        while t.is_alive():
            time.sleep(60)
    except KeyboardInterrupt:
        print("Background service stopped.")

