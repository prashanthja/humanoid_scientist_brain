# Discovery Report

**Query:** Does continuous batching improve LLM serving throughput?
**Domain:** transformer_efficiency
**Generated:** 2026-03-19 10:57:56

---
## Verdict

**✅ SUPPORTED**

Confidence: `0.75`

Evidence from multiple sources confirms this claim with experimental results.

---
## Evidence Sources

Retrieved **10** evidence chunks from:

- Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction
- SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving
- BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving
- MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts
- Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching
- Trinity: Disaggregating Vector Search from Prefill-Decode Disaggregation in LLM Serving
- ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching
- Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

---
## Key Claims Extracted

Extracted **10** claims from evidence.

**1.** erve achieves up to 9.26x lower mean time-to-first-token (TTFT) latency and up to 3.14x higher token generation throughput compared to state-of-the-art LLM serving systems.
   - Source: *SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving*
   - Domain: `transformer_efficiency`

**2.** The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training.
   - Source: *MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts*
   - Domain: `transformer_efficiency`

**3.** latency feedback mechanism that optimizes decoding processes under SLA constraints.
   - Source: *Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching*
   - Domain: `transformer_efficiency`

**4.** The numerical experiments demonstrate throughput gains of 8% to 28% and capacity improvements of 22% compared to traditional static batching methods, while maintaining full compatibility with existing inference infrastructure.
   - Source: *Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching*
   - Domain: `transformer_efficiency`

**5.** However, the inference of LLMs is resource-intensive or latencysensitive, posing significant challenges for serving systems.
   - Source: *BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving*
   - Domain: `transformer_efficiency`

---
## Evidence Evaluation

| Verdict | Count |
|---|---|
| ✅ Supported | 2 |
| 🟡 Partially supported | 3 |
| ❌ Contradicted | 0 |

### Top Grounded Claims

✅ **supported** (confidence: `0.75`)
> erve achieves up to 9.26x lower mean time-to-first-token (TTFT) latency and up to 3.14x higher token generation throughput compared to state-of-the-art LLM serving systems.

✅ **supported** (confidence: `0.68`)
> However, the inference of LLMs is resource-intensive or latencysensitive, posing significant challenges for serving systems.

🟡 **partially_supported** (confidence: `0.60`)
> Evaluations on real-world datasets and production workload traces show that SSJF reduces average job completion times by 30.5-39.6% and increases throughput

⚪ **inconclusive** (confidence: `0.28`)
> To address the non-deterministic nature of LLMs and enable efficient interactive LLM serving, we present a speculative shortest-job-first (SSJF) scheduler that uses a light proxy model to predict LLM output sequence lengths.

⚪ **inconclusive** (confidence: `0.20`)
> Our open-source SSJF implementation does not require changes to memory management or batching strategies.

---
## Knowledge Gaps

- Many claims are inconclusive — evidence exists but lacks benchmark numbers or experimental results.
- No contradicting evidence found — this may indicate limited coverage of papers reporting failure cases or trade-offs.

---
## Recommended Next Actions

1. Pick 1 concrete intervention and convert it into a falsifiable claim (inputs → mechanism → measurable outputs).
2. Run targeted retrieval using 3–5 specific transformer-efficiency keywords and compare results.
3. Prioritize evidence chunks with benchmark numbers (latency, throughput, memory, perplexity, context length).
4. Generate an experiment plan: baseline, efficiency metric, quality metric, ablations, acceptance threshold.
