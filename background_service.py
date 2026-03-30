# background_service.py
# ─────────────────────────────────────────────────────────────
# Background ingestion service — pulls from ALL free research
# sources on the internet, every 24 hours.
#
# Sources (all free, no API key required):
#   1. Semantic Scholar  — 200M+ papers
#   2. ArXiv             — all CS/ML papers, full text
#   3. Papers With Code  — ML papers with code
#   4. Hugging Face      — daily ML papers
#   5. OpenAlex          — 250M+ works, fully open
#   6. CORE              — 200M+ open access, full text
#   7. CrossRef          — DOI metadata
#   8. Europe PMC        — life sciences + CS
#   9. Unpaywall         — fetches full PDF text
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import os, sys, json, time, threading, logging, hashlib, requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
import urllib.parse

log = logging.getLogger("tattva.background")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

ROOT        = os.path.dirname(os.path.abspath(__file__))
SEEN_PATH   = os.path.join(ROOT, "data", "seen_papers.json")
STATUS_PATH = os.path.join(ROOT, "data", "service_status.json")
INTERVAL_SEC = 24 * 60 * 60

# ── Search queries ────────────────────────────────────────
SEARCH_QUERIES = [
    "transformer attention efficiency memory",
    "mixture of experts language model routing",
    "KV cache compression LLM inference",
    "FlashAttention IO aware memory efficient attention",
    "LoRA low rank adaptation fine-tuning",
    "speculative decoding LLM throughput",
    "sparse attention long context transformer",
    "quantization LLM inference INT8 INT4",
    "MoE routing instability training sparse",
    "rotary position embedding long context RoPE",
    "continuous batching LLM serving vLLM",
    "grouped query attention KV cache memory",
    "PagedAttention virtual memory GPU LLM",
    "pruning transformer inference cost",
    "knowledge distillation model compression",
    "sliding window attention sequence length",
    "Mamba state space model transformers linear",
    "tensor parallelism distributed LLM training",
    "RWKV linear complexity recurrent transformer",
    "model quantization downstream performance",
    "prefix caching KV reuse inference",
    "weight sharing parameter efficient transformers",
    "linear attention kernel approximation",
    "multi-head latent attention MLA DeepSeek",
    "pipeline parallelism LLM disaggregation",
]

# ── Discovery eval queries ────────────────────────────────
EVAL_QUERIES = [
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
    "Does grouped query attention reduce KV cache memory?",
    "Does PagedAttention improve GPU memory utilization?",
    "Does pruning transformer weights reduce inference cost?",
    "Does knowledge distillation preserve model quality?",
    "Does sliding window attention scale to long sequences?",
    "Does Mamba outperform transformers on long sequences?",
    "Does tensor parallelism improve LLM training throughput?",
    "Does RWKV match transformer quality with linear complexity?",
    "Does model quantization affect downstream task performance?",
    "Does pipeline parallelism reduce training time?",
]

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
SESSION.headers.update({"User-Agent": "TattvaAI/1.0 (research aggregator; contact@tattva.ai)"})


# ── Seen paper dedup ──────────────────────────────────────

def _paper_id(title: str, abstract: str) -> str:
    return hashlib.md5(f"{title.lower().strip()}{abstract[:80]}".encode()).hexdigest()

def _load_seen() -> set:
    try:
        if os.path.exists(SEEN_PATH):
            with open(SEEN_PATH) as f: return set(json.load(f))
    except Exception: pass
    return set()

def _save_seen(seen: set):
    os.makedirs(os.path.dirname(SEEN_PATH), exist_ok=True)
    with open(SEEN_PATH, "w") as f: json.dump(list(seen), f)

def _is_relevant(paper: Dict) -> bool:
    text = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
    return sum(1 for kw in DOMAIN_KEYWORDS if kw in text) >= 2

def _clean(text: str) -> str:
    if not text: return ""
    return " ".join(text.strip().split())


# ── Source 1: Semantic Scholar ────────────────────────────

def fetch_semantic_scholar(query: str, limit: int = 25) -> List[Dict]:
    try:
        r = SESSION.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": limit,
                    "fields": "title,abstract,year,openAccessPdf"},
            timeout=20
        )
        if r.status_code == 429:
            time.sleep(5); return []
        if r.status_code != 200: return []
        papers = []
        for p in r.json().get("data", []):
            title    = _clean(p.get("title",""))
            abstract = _clean(p.get("abstract","") or "")
            full_text = abstract
            # Try fetching full PDF text if open access
            pdf_url = (p.get("openAccessPdf") or {}).get("url")
            if pdf_url:
                full_text = _fetch_pdf_text(pdf_url) or abstract
            if title and abstract and len(abstract) > 80:
                papers.append({"title": title, "abstract": abstract,
                                "full_text": full_text, "source": "semantic_scholar",
                                "year": p.get("year")})
        return papers
    except Exception as e:
        log.debug(f"Semantic Scholar: {e}"); return []


# ── Source 2: ArXiv ───────────────────────────────────────

def fetch_arxiv(query: str, limit: int = 25) -> List[Dict]:
    try:
        params = (f"search_query=all:{urllib.parse.quote(query)}"
                  f"&max_results={limit}&sortBy=submittedDate&sortOrder=descending")
        r = SESSION.get(f"http://export.arxiv.org/api/query?{params}", timeout=20)
        if r.status_code != 200: return []
        root = ET.fromstring(r.text)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            title    = _clean((entry.find("atom:title",ns) or {}).text or "").replace("\n"," ")
            abstract = _clean((entry.find("atom:summary",ns) or {}).text or "").replace("\n"," ")
            # ArXiv PDF URL
            arxiv_id = None
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "application/pdf":
                    arxiv_id = link.get("href","")
            full_text = abstract
            if arxiv_id:
                full_text = _fetch_pdf_text(arxiv_id) or abstract
            if title and abstract and len(abstract) > 80:
                papers.append({"title": title, "abstract": abstract,
                                "full_text": full_text, "source": "arxiv"})
        return papers
    except Exception as e:
        log.debug(f"ArXiv: {e}"); return []


# ── Source 3: OpenAlex (250M+ works, fully open) ─────────

def fetch_openalex(query: str, limit: int = 25) -> List[Dict]:
    try:
        r = SESSION.get(
            "https://api.openalex.org/works",
            params={"search": query, "per_page": limit,
                    "filter": "concepts.display_name:Computer Science",
                    "sort": "publication_date:desc",
                    "select": "title,abstract_inverted_index,open_access,doi"},
            timeout=20
        )
        if r.status_code != 200: return []
        papers = []
        for work in r.json().get("results", []):
            title = _clean(work.get("title","") or "")
            # OpenAlex stores abstracts as inverted index — reconstruct
            inv = work.get("abstract_inverted_index") or {}
            abstract = _reconstruct_abstract(inv)
            if not abstract: continue
            full_text = abstract
            # Try open access PDF
            oa = work.get("open_access",{})
            pdf_url = oa.get("oa_url") if oa.get("is_oa") else None
            if pdf_url and pdf_url.endswith(".pdf"):
                full_text = _fetch_pdf_text(pdf_url) or abstract
            if title and len(abstract) > 80:
                papers.append({"title": title, "abstract": abstract,
                                "full_text": full_text, "source": "openalex"})
        return papers
    except Exception as e:
        log.debug(f"OpenAlex: {e}"); return []


def _reconstruct_abstract(inv: Dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inv: return ""
    try:
        max_pos = max(pos for positions in inv.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inv.items():
            for pos in positions:
                words[pos] = word
        return " ".join(w for w in words if w)
    except Exception:
        return ""


# ── Source 4: CORE (200M+ open access papers) ─────────────

def fetch_core(query: str, limit: int = 20) -> List[Dict]:
    try:
        r = SESSION.get(
            "https://core.ac.uk/api-v2/search",
            params={"q": query, "pageSize": limit,
                    "apiKey": ""},  # CORE has a free tier, no key needed for basic
            timeout=20
        )
        # Try without key first
        r = SESSION.get(
            f"https://core.ac.uk/api/search/works",
            params={"q": query, "limit": limit},
            headers={"Authorization": "Bearer "},
            timeout=20
        )
        if r.status_code != 200:
            # Fallback: CORE OAI endpoint
            return _fetch_core_oai(query, limit)
        papers = []
        for p in r.json().get("results", []):
            title    = _clean(p.get("title","") or "")
            abstract = _clean(p.get("abstract","") or "")
            full_text = _clean(p.get("fullText","") or "") or abstract
            if title and abstract and len(abstract) > 80:
                papers.append({"title": title, "abstract": abstract,
                                "full_text": full_text, "source": "core"})
        return papers
    except Exception as e:
        log.debug(f"CORE: {e}"); return []


def _fetch_core_oai(query: str, limit: int) -> List[Dict]:
    """CORE OAI-PMH fallback."""
    try:
        r = SESSION.get(
            "https://core.ac.uk/api-v2/search/",
            params={"q": query, "pageSize": min(limit,10)},
            timeout=15
        )
        if r.status_code != 200: return []
        papers = []
        for p in r.json().get("data",[]):
            title    = _clean(p.get("title","") or "")
            abstract = _clean(p.get("description","") or "")
            full_text= _clean(p.get("fullText","") or "") or abstract
            if title and abstract and len(abstract) > 80:
                papers.append({"title":title,"abstract":abstract,
                                "full_text":full_text,"source":"core"})
        return papers
    except Exception:
        return []


# ── Source 5: Papers With Code ────────────────────────────

def fetch_papers_with_code(query: str, limit: int = 20) -> List[Dict]:
    try:
        r = SESSION.get(
            "https://paperswithcode.com/api/v1/papers/",
            params={"q": query, "page_size": limit},
            timeout=20
        )
        if r.status_code != 200: return []
        papers = []
        for p in r.json().get("results", []):
            title    = _clean(p.get("title","") or "")
            abstract = _clean(p.get("abstract","") or "")
            if title and abstract and len(abstract) > 80:
                papers.append({"title": title, "abstract": abstract,
                                "full_text": abstract, "source": "papers_with_code"})
        return papers
    except Exception as e:
        log.debug(f"PapersWithCode: {e}"); return []


# ── Source 6: Hugging Face Papers ────────────────────────

def fetch_huggingface(query: str, limit: int = 20) -> List[Dict]:
    try:
        r = SESSION.get(
            "https://huggingface.co/api/papers",
            params={"q": query},
            timeout=20
        )
        if r.status_code != 200: return []
        data = r.json() if isinstance(r.json(), list) else r.json().get("papers", [])
        papers = []
        for p in data[:limit]:
            title    = _clean(p.get("title","") or "")
            abstract = _clean(p.get("abstract","") or p.get("summary","") or "")
            if title and abstract and len(abstract) > 80:
                papers.append({"title": title, "abstract": abstract,
                                "full_text": abstract, "source": "huggingface"})
        return papers
    except Exception as e:
        log.debug(f"HuggingFace: {e}"); return []


# ── Source 7: Europe PMC ──────────────────────────────────

def fetch_europe_pmc(query: str, limit: int = 20) -> List[Dict]:
    try:
        r = SESSION.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "resultType": "core",
                    "pageSize": limit, "format": "json",
                    "sort": "P_PDATE_D desc"},
            timeout=20
        )
        if r.status_code != 200: return []
        papers = []
        for p in r.json().get("resultList",{}).get("result",[]):
            title    = _clean(p.get("title","") or "")
            abstract = _clean(p.get("abstractText","") or "")
            if title and abstract and len(abstract) > 80:
                papers.append({"title": title, "abstract": abstract,
                                "full_text": abstract, "source": "europe_pmc"})
        return papers
    except Exception as e:
        log.debug(f"EuropePMC: {e}"); return []


# ── Source 8: CrossRef ────────────────────────────────────

def fetch_crossref(query: str, limit: int = 20) -> List[Dict]:
    try:
        r = SESSION.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": limit,
                    "select": "title,abstract,published-print,DOI",
                    "sort": "published", "order": "desc"},
            timeout=20
        )
        if r.status_code != 200: return []
        papers = []
        for p in r.json().get("message",{}).get("items",[]):
            titles   = p.get("title",[])
            title    = _clean(titles[0] if titles else "")
            abstract = _clean(p.get("abstract","") or "")
            # Strip JATS XML tags from CrossRef abstracts
            abstract = _strip_xml(abstract)
            if title and abstract and len(abstract) > 80:
                papers.append({"title": title, "abstract": abstract,
                                "full_text": abstract, "source": "crossref"})
        return papers
    except Exception as e:
        log.debug(f"CrossRef: {e}"); return []


def _strip_xml(text: str) -> str:
    """Remove XML/JATS tags from CrossRef abstracts."""
    try:
        import re
        return re.sub(r'<[^>]+>', ' ', text).strip()
    except Exception:
        return text


# ── Source 9: Unpaywall full text ─────────────────────────

def _fetch_pdf_text(pdf_url: str, max_chars: int = 8000) -> Optional[str]:
    """Download and extract text from a PDF URL."""
    if not pdf_url or not pdf_url.startswith("http"):
        return None
    try:
        r = SESSION.get(pdf_url, timeout=30, stream=True)
        if r.status_code != 200: return None
        if "pdf" not in r.headers.get("content-type","").lower(): return None
        content = b""
        for chunk in r.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 2_000_000: break  # 2MB max
        # Try PyPDF2 or pdfminer if available
        try:
            import io
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(content))
                text = " ".join(page.extract_text() or "" for page in reader.pages[:20])
                return _clean(text)[:max_chars] if text.strip() else None
            except ImportError:
                pass
            try:
                from pdfminer.high_level import extract_text_to_fp
                from pdfminer.layout import LAParams
                import io as _io
                output = _io.StringIO()
                extract_text_to_fp(io.BytesIO(content), output, laparams=LAParams())
                text = output.getvalue()
                return _clean(text)[:max_chars] if text.strip() else None
            except ImportError:
                pass
        except Exception:
            pass
        return None
    except Exception:
        return None


# ── Aggregate all sources ─────────────────────────────────

def fetch_all_sources(query: str) -> List[Dict]:
    """Fetch from all 8+ sources for a single query."""
    results = []
    fetchers = [
        ("Semantic Scholar", fetch_semantic_scholar),
        ("ArXiv",            fetch_arxiv),
        ("OpenAlex",         fetch_openalex),
        ("CORE",             fetch_core),
        ("PapersWithCode",   fetch_papers_with_code),
        ("HuggingFace",      fetch_huggingface),
        ("EuropePMC",        fetch_europe_pmc),
        ("CrossRef",         fetch_crossref),
    ]
    for name, fetcher in fetchers:
        try:
            papers = fetcher(query, limit=20)
            results.extend(papers)
            log.debug(f"  {name}: {len(papers)} papers")
        except Exception as e:
            log.debug(f"  {name} failed: {e}")
        time.sleep(0.5)  # polite rate limiting
    return results


# ── Status tracking ───────────────────────────────────────

def _write_status(status: Dict):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2, default=str)

def read_status() -> Dict:
    try:
        if os.path.exists(STATUS_PATH):
            with open(STATUS_PATH) as f: return json.load(f)
    except Exception: pass
    return {"state":"idle","last_run":None,"next_run":None,
            "papers_added":0,"total_chunks":0,"cycles":0}


# ── Main cycle ────────────────────────────────────────────

def run_cycle():
    start = time.time()
    log.info("=" * 60)
    log.info(f"Background cycle starting — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Sources: Semantic Scholar, ArXiv, OpenAlex, CORE, "
             f"PapersWithCode, HuggingFace, EuropePMC, CrossRef")
    log.info("=" * 60)

    prev = read_status()
    _write_status({**prev, "state":"running", "phase":"fetching papers",
                   "started_at": datetime.now().isoformat()})

    seen       = _load_seen()
    new_papers = []

    # ── Phase 1: Fetch ────────────────────────────────────
    log.info(f"Phase 1: Fetching from all sources ({len(SEARCH_QUERIES)} queries)")
    for i, query in enumerate(SEARCH_QUERIES):
        log.info(f"  [{i+1}/{len(SEARCH_QUERIES)}] {query}")
        batch = fetch_all_sources(query)
        added = 0
        for paper in batch:
            pid = _paper_id(paper["title"], paper["abstract"])
            if pid not in seen and _is_relevant(paper):
                seen.add(pid)
                new_papers.append(paper)
                added += 1
        log.info(f"    → {added} new relevant papers")
        time.sleep(1)

    log.info(f"Total new papers: {len(new_papers)}")
    _save_seen(seen)

    if not new_papers:
        log.info("No new papers — skipping ingestion phases")
        _write_status({
            "state":"idle","phase":"complete — no new papers",
            "last_run":datetime.now().isoformat(),
            "next_run":datetime.fromtimestamp(time.time()+INTERVAL_SEC).isoformat(),
            "papers_added":0,"cycles":prev.get("cycles",0)+1,
            "total_chunks":prev.get("total_chunks",0),
        })
        return

    # ── Phase 2: Ingest ───────────────────────────────────
    log.info(f"Phase 2: Ingesting {len(new_papers)} papers")
    _write_status({**read_status(),"phase":f"ingesting {len(new_papers)} papers"})
    papers_added = 0
    try:
        from knowledge_base.chunk_store import ChunkStore
        chunk_store = ChunkStore()
        for paper in new_papers:
            # Use full text if available and longer than abstract
            text_to_chunk = paper.get("full_text","") or paper["abstract"]
            if len(text_to_chunk) < len(paper["abstract"]):
                text_to_chunk = paper["abstract"]
            words = text_to_chunk.split()
            chunk_size = 200
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i:i+chunk_size])
                if len(chunk_text) > 50:
                    chunk_store.add(
                        text=chunk_text,
                        metadata={
                            "paper_title": paper["title"],
                            "source":      paper["source"],
                            "domain":      "transformer_efficiency",
                            "chunk_idx":   i // chunk_size,
                        }
                    )
            papers_added += 1
        log.info(f"  Ingested {papers_added} papers")
    except Exception as e:
        log.error(f"Ingestion failed: {e}")

    # ── Phase 3: Rebuild index ────────────────────────────
    log.info("Phase 3: Rebuilding chunk index")
    _write_status({**read_status(),"phase":"rebuilding chunk index"})
    total_chunks = 0
    try:
        from knowledge_base.chunk_store import ChunkStore
        from learning_module.trainer_online import OnlineTrainer
        from learning_module.embedding_bridge import EmbeddingBridge
        from retrieval.chunk_index import ChunkIndex
        chunk_store = ChunkStore()
        trainer     = OnlineTrainer()
        encoder     = EmbeddingBridge(trainer)
        chunk_index = ChunkIndex(encoder=encoder, chunk_store=chunk_store)
        chunk_index.rebuild()
        total_chunks = chunk_store.count()
        log.info(f"  Index rebuilt: {total_chunks} chunks")
    except Exception as e:
        log.error(f"Index rebuild failed: {e}")

    # ── Phase 4: Discovery pipeline ───────────────────────
    log.info(f"Phase 4: Discovery pipeline ({len(EVAL_QUERIES)} queries)")
    _write_status({**read_status(),"phase":"running discovery pipeline"})
    try:
        from knowledge_base.chunk_store import ChunkStore
        from learning_module.trainer_online import OnlineTrainer
        from learning_module.embedding_bridge import EmbeddingBridge
        from retrieval.chunk_index import ChunkIndex
        from reasoning_module.discover import DiscoveryEngine, DiscoveryConfig
        from reasoning_module.evidence_evaluator import EvidenceEvaluator
        from reasoning_module.proposal_evaluator import ProposalEvaluator
        from reasoning_module.discovery_report import build_report, save_report
        from knowledge_graph.graph import KnowledgeGraph
        from reasoning_module.kg_builder import KGBuilder

        chunk_store = ChunkStore()
        trainer     = OnlineTrainer()
        encoder     = EmbeddingBridge(trainer)
        chunk_index = ChunkIndex(encoder=encoder, chunk_store=chunk_store)
        kg          = KnowledgeGraph(); kg.load()
        kg_builder  = KGBuilder(kg)

        prop_eval = ProposalEvaluator(kb=chunk_store, bridge=encoder,
            top_k=10, evidence_threshold=0.55, require_evidence=True)
        ev_eval   = EvidenceEvaluator(kb=chunk_store, encoder=encoder, kg=None,
            chunk_index=chunk_index, use_chunk_index=True)

        class _Null:
            def generate(self,top_n=10): return []
            def validate(self,h,cycle=0): return []

        null = _Null()
        engine = DiscoveryEngine(
            chunk_index=chunk_index, proposal_engine=prop_eval,
            evidence_evaluator=ev_eval, hypgen=null, validator=null,
            config=DiscoveryConfig(top_k_chunks=10, max_claims=10,
                                   max_grounded_claims=5, use_mmr=True),
        )

        for i, query in enumerate(EVAL_QUERIES):
            try:
                log.info(f"  [{i+1}/{len(EVAL_QUERIES)}] {query[:55]}")
                result = engine.run(query, source_name="background_service")
                report = build_report(query, result)
                save_report(report)
                kg_builder.ingest(result)
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"  Query failed: {e}")

        kg.save()
        log.info(f"  KG: {kg.edge_count()} edges, {len(kg.all_concepts())} concepts")
    except Exception as e:
        log.error(f"Discovery pipeline failed: {e}")

    # ── Phase 5: Hypotheses ───────────────────────────────
    log.info("Phase 5: Regenerating hypotheses")
    _write_status({**read_status(),"phase":"regenerating hypotheses"})
    try:
        from knowledge_graph.graph import KnowledgeGraph
        from reasoning_module.hypothesis_generator import HypothesisGenerator
        from learning_module.trainer_online import OnlineTrainer
        from learning_module.embedding_bridge import EmbeddingBridge
        kg      = KnowledgeGraph(); kg.load()
        trainer = OnlineTrainer()
        encoder = EmbeddingBridge(trainer)
        gen     = HypothesisGenerator(kg=kg, encoder=encoder)
        hyps    = gen.generate(top_n=30)
        log.info(f"  Generated {len(hyps)} hypotheses")
    except Exception as e:
        log.error(f"Hypothesis generation failed: {e}")

    elapsed = round(time.time()-start, 1)
    log.info(f"Cycle complete in {elapsed}s — {papers_added} papers added, {total_chunks} total chunks")
    log.info("=" * 60)

    _write_status({
        "state":        "idle",
        "phase":        "complete",
        "last_run":     datetime.now().isoformat(),
        "next_run":     datetime.fromtimestamp(time.time()+INTERVAL_SEC).isoformat(),
        "papers_added": papers_added,
        "total_chunks": total_chunks,
        "elapsed_sec":  elapsed,
        "cycles":       prev.get("cycles",0)+1,
        "sources":      ["semantic_scholar","arxiv","openalex","core",
                         "papers_with_code","huggingface","europe_pmc","crossref"],
    })


# ── Daemon thread ─────────────────────────────────────────

def start_background_service(run_immediately: bool = False):
    """Start as a daemon thread inside Flask."""
    def loop():
        if run_immediately:
            log.info("Background service: running initial cycle now")
            try: run_cycle()
            except Exception as e: log.error(f"Initial cycle: {e}")
        else:
            log.info(f"Background service: first auto-run in 24h")
            _write_status({
                "state":"idle","phase":"waiting — first run in 24h",
                "last_run":None,
                "next_run":datetime.fromtimestamp(time.time()+INTERVAL_SEC).isoformat(),
                "cycles":0,
            })
            time.sleep(INTERVAL_SEC)
        while True:
            try: run_cycle()
            except Exception as e:
                log.error(f"Cycle failed: {e}")
                _write_status({**read_status(),"state":"error","error":str(e)})
            time.sleep(INTERVAL_SEC)

    t = threading.Thread(target=loop, daemon=True, name="tattva-bg")
    t.start()
    log.info("Background service thread started (24h interval)")
    return t