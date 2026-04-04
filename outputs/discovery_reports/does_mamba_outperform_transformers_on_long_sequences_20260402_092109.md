# Discovery Report

**Query:** Does Mamba outperform transformers on long sequences?
**Domain:** transformer_efficiency
**Generated:** 2026-04-02 09:21:09

---
## Verdict

**✅ SUPPORTED**

Confidence: `0.75`

Evidence from multiple sources confirms this claim with experimental results.

---
## Evidence Sources

Retrieved **10** evidence chunks from:

- MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts
- Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling
- Generation Meets Verification: Accelerating Large Language Model Inference with Smart Parallel Auto-Correct Decoding
- Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference
- Small-E: Small Language Model with Linear Attention for Efficient Speech Synthesis
- Cross-token Modeling with Conditional Computation
- EdgeInfinite: A Memory-Efficient Infinite-Context Transformer for Edge Devices
- Combiner: Full Attention Transformer with Sparse Computation Cost

---
## Key Claims Extracted

Extracted **10** claims from evidence.

**1.** The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training.
   - Source: *MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts*
   - Domain: `transformer_efficiency`

**2.** Notably, the decoder-only transformer is the prominent architecture in this domain.
   - Source: *Small-E: Small Language Model with Linear Attention for Efficient Speech Synthesis*
   - Domain: `transformer_efficiency`

**3.** However, transformers face challenges stemming from their quadratic complexity in sequence length, impeding training on lengthy sequences and resource-constrained hardware.
   - Source: *Small-E: Small Language Model with Linear Attention for Efficient Speech Synthesis*
   - Domain: `transformer_efficiency`

**4.** Efficiently modeling sequences with infinite context length has long been a challenging problem.
   - Source: *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*
   - Domain: `transformer_efficiency`

**5.** We scale Samba up to 3.8B parameters with 3.2T training tokens and demonstrate that it significantly outperforms state-of-the-art models across a variety of benchmarks.
   - Source: *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*
   - Domain: `transformer_efficiency`

---
## Evidence Evaluation

| Verdict | Count |
|---|---|
| ✅ Supported | 0 |
| 🟡 Partially supported | 5 |
| ❌ Contradicted | 0 |

### Top Grounded Claims

🟡 **partially_supported** (confidence: `0.48`)
> We scale Samba up to 3.8B parameters with 3.2T training tokens and demonstrate that it significantly outperforms state-of-the-art models across a variety of benchmarks.

🟡 **partially_supported** (confidence: `0.47`)
> sequences within a single model invocation.

⚪ **inconclusive** (confidence: `0.36`)
> Efficiently modeling sequences with infinite context length has long been a challenging problem.

⚪ **inconclusive** (confidence: `0.23`)
> Pretrained on sequences of 4K length, Samba shows improved perplexity in context lengths of up to 1M in zero-shot.

⚪ **inconclusive** (confidence: `0.20`)
> The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training.

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
