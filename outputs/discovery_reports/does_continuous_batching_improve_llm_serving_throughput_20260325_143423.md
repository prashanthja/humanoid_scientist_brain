# Discovery Report

**Query:** Does continuous batching improve LLM serving throughput?
**Domain:** transformer_efficiency
**Generated:** 2026-03-25 14:34:23

---
## Verdict

**⚪ INCONCLUSIVE**

Confidence: `0.35`

Semantic similarity exists but evidence lacks strength (strong_chunks=0/2 best_quality=0.45)

---
## Evidence Sources

Retrieved **10** evidence chunks from:

- BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving
- SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving
- Seesaw: High-throughput LLM Inference via Model Re-sharding
- Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction
- Trinity: Disaggregating Vector Search from Prefill-Decode Disaggregation in LLM Serving
- Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching
- ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching
- Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

---
## Key Claims Extracted

Extracted **10** claims from evidence.

**1.** erve achieves up to 9.26x lower mean time-to-first-token (TTFT) latency and up to 3.14x higher token generation throughput compared to state-of-the-art LLM serving systems.
   - Source: *SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving*
   - Domain: `transformer_efficiency`

**2.** tional efficiency, we employ tiered KV cache buffering and transition-minimizing scheduling.
   - Source: *Seesaw: High-throughput LLM Inference via Model Re-sharding*
   - Domain: `transformer_efficiency`

**3.** Our evaluation demonstrates that Seesaw achieves a throughput increase of up to 1.78x (1.36x on average) compared to vLLM, the most widely used state-of-the-art LLM inference engine.
   - Source: *Seesaw: High-throughput LLM Inference via Model Re-sharding*
   - Domain: `transformer_efficiency`

**4.** However, the inference of LLMs is resource-intensive or latencysensitive, posing significant challenges for serving systems.
   - Source: *BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving*
   - Domain: `transformer_efficiency`

**5.** Existing LLM serving systems often use static or continuous batching strategies, which can lead to inefficient GPU memory utilization and increased latency, especially under heterogeneous workloads.
   - Source: *BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving*
   - Domain: `transformer_efficiency`

---
## Evidence Evaluation

| Verdict | Count |
|---|---|
| ✅ Supported | 3 |
| 🟡 Partially supported | 2 |
| ❌ Contradicted | 0 |

### Top Grounded Claims

✅ **supported** (confidence: `0.78`)
> These methods may also struggle to adapt to dynamic workload fluctuations, resulting in suboptimal throughput and potential service level objective (SLO) violations.

✅ **supported** (confidence: `0.72`)
> erve achieves up to 9.26x lower mean time-to-first-token (TTFT) latency and up to 3.14x higher token generation throughput compared to state-of-the-art LLM serving systems.

✅ **supported** (confidence: `0.68`)
> However, the inference of LLMs is resource-intensive or latencysensitive, posing significant challenges for serving systems.

🟡 **partially_supported** (confidence: `0.50`)
> Existing LLM serving systems often use static or continuous batching strategies, which can lead to inefficient GPU memory utilization and increased latency, especially under heterogeneous workloads.

⚪ **inconclusive** (confidence: `0.20`)
> tional efficiency, we employ tiered KV cache buffering and transition-minimizing scheduling.

---
## Knowledge Gaps

- No contradicting evidence found — this may indicate limited coverage of papers reporting failure cases or trade-offs.

---
## Recommended Next Actions

1. Pick 1 concrete intervention and convert it into a falsifiable claim (inputs → mechanism → measurable outputs).
2. Run targeted retrieval using 3–5 specific transformer-efficiency keywords and compare results.
3. Prioritize evidence chunks with benchmark numbers (latency, throughput, memory, perplexity, context length).
4. Generate an experiment plan: baseline, efficiency metric, quality metric, ablations, acceptance threshold.
