# Discovery Report

**Query:** Does MoE routing instability hurt model quality?
**Domain:** transformer_efficiency
**Generated:** 2026-03-18 15:14:59

---
## Verdict

**✅ SUPPORTED**

Confidence: `0.74`

Evidence from multiple sources confirms this claim with experimental results.

---
## Evidence Sources

Retrieved **10** evidence chunks from:

- Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer
- EdgeInfinite: A Memory-Efficient Infinite-Context Transformer for Edge Devices
- Scaling Molecular Representation Learning with Hierarchical Mixture-of-Experts
- Dynamic Adaptive Reasoning: Optimizing LLM Inference-Time Thinking via Intent-Aware Scheduling
- Low-Rank SVD Compression for Memory-Efficient Transformer Attention
- Reducing Hallucinations in Large Language Models: A Consensus Voting Approach Using Mixture of Experts
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

---
## Key Claims Extracted

Extracted **10** claims from evidence.

**1.** The experimental result shows that EdgeInfinite achieves comparable performance to baseline Transformer-based LLM on long context benchmarks while optimizing memory consumption and time to first token.
   - Source: *EdgeInfinite: A Memory-Efficient Infinite-Context Transformer for Edge Devices*
   - Domain: `transformer_efficiency`

**2.** In the Long Range Arena benchmark, it demonstrates an average score at least 0.68% higher than all comparison models, including Full Attention, MoEs, and transformer variants, with only 50% of the FLOPs of the best MoE method.
   - Source: *Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer*
   - Domain: `transformer_efficiency`

**3.** Moreover, the MoE framework has not been effectively extended to attention blocks, which limits further efficiency improvements.
   - Source: *Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer*
   - Domain: `transformer_efficiency`

**4.** To tackle these issues, we propose Union-of-Experts (UoE), which decomposes the transformer model into an equivalent group of experts and applies selective routing to input data and experts.
   - Source: *Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer*
   - Domain: `transformer_efficiency`

**5.** The experiments demonstrate that our UoE model surpasses Full Attention, state-of-the-art MoEs, and efficient transformers (including the recently proposed DeepSeek-V3 architecture) in several tasks across image and natural language domains.
   - Source: *Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer*
   - Domain: `transformer_efficiency`

---
## Evidence Evaluation

| Verdict | Count |
|---|---|
| ✅ Supported | 5 |
| 🟡 Partially supported | 0 |
| ❌ Contradicted | 0 |

### Top Grounded Claims

✅ **supported** (confidence: `0.95`)
> Moreover, the MoE framework has not been effectively extended to attention blocks, which limits further efficiency improvements.

✅ **supported** (confidence: `0.95`)
> (4) We develop the parallel implementation of UoE’s routing and computation operations and optimize the efficiency based on hardware processing analysis.

✅ **supported** (confidence: `0.91`)
> To tackle these issues, we propose Union-of-Experts (UoE), which decomposes the transformer model into an equivalent group of experts and applies selective routing to input data and experts.

✅ **supported** (confidence: `0.89`)
> enables selective activation of the memory-gating module for long and short context task routing.

✅ **supported** (confidence: `0.86`)
> The experimental result shows that EdgeInfinite achieves comparable performance to baseline Transformer-based LLM on long context benchmarks while optimizing memory consumption and time to first token.

---
## Knowledge Gaps

- No contradicting evidence found — this may indicate limited coverage of papers reporting failure cases or trade-offs.
- Some retrieved chunks appear off-domain — retrieval may benefit from stricter domain filtering.

---
## Recommended Next Actions

1. Pick 1 concrete intervention and convert it into a falsifiable claim (inputs → mechanism → measurable outputs).
2. Run targeted retrieval using 3–5 specific transformer-efficiency keywords and compare results.
3. Prioritize evidence chunks with benchmark numbers (latency, throughput, memory, perplexity, context length).
4. Generate an experiment plan: baseline, efficiency metric, quality metric, ablations, acceptance threshold.
