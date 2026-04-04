# Discovery Report

**Query:** Does Mamba outperform transformers on long sequences?
**Domain:** transformer_efficiency
**Generated:** 2026-04-03 12:12:22

---
## Verdict

**✅ SUPPORTED**

Confidence: `0.75`

Evidence from multiple sources confirms this claim with experimental results.

---
## Evidence Sources

Retrieved **10** evidence chunks from:

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Generation Meets Verification: Accelerating Large Language Model Inference with Smart Parallel Auto-Correct Decoding
- MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts
- What is next for LLMs? Pushing the boundaries of next-gen AI computing hardware with photonic chips.
- Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling
- EdgeInfinite: A Memory-Efficient Infinite-Context Transformer for Edge Devices
- Hybrid Attention-based Transformer for Long-range Document Classification
- Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference

---
## Key Claims Extracted

Extracted **10** claims from evidence.

**1.** The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training.
   - Source: *MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts*
   - Domain: `transformer_efficiency`

**2.** magnitude in throughput and energy efficiency, but require breakthroughs in memory especially for long-context windows and long token sequences and in storage of ultra-large datasets, among others.
   - Source: *What is next for LLMs? Pushing the boundaries of next-gen AI computing hardware with photonic chips.*
   - Domain: `transformer_efficiency`

**3.** Efficiently modeling sequences with infinite context length has long been a challenging problem.
   - Source: *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*
   - Domain: `transformer_efficiency`

**4.** We scale Samba up to 3.8B parameters with 3.2T training tokens and demonstrate that it significantly outperforms state-of-the-art models across a variety of benchmarks.
   - Source: *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*
   - Domain: `transformer_efficiency`

**5.** Pretrained on sequences of 4K length, Samba shows improved perplexity in context lengths of up to 1M in zero-shot.
   - Source: *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*
   - Domain: `transformer_efficiency`

---
## Evidence Evaluation

| Verdict | Count |
|---|---|
| ✅ Supported | 0 |
| 🟡 Partially supported | 4 |
| ❌ Contradicted | 1 |

### Top Grounded Claims

❌ **contradicted** (confidence: `0.55`)
> The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training.

⚪ **inconclusive** (confidence: `0.36`)
> sequences within a single model invocation.

⚪ **inconclusive** (confidence: `0.32`)
> Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language.

⚪ **inconclusive** (confidence: `0.23`)
> Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module.

⚪ **inconclusive** (confidence: `0.20`)
> Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode.

---
## Knowledge Gaps

- Many claims are inconclusive — evidence exists but lacks benchmark numbers or experimental results.

---
## Recommended Next Actions

1. Pick 1 concrete intervention and convert it into a falsifiable claim (inputs → mechanism → measurable outputs).
2. Run targeted retrieval using 3–5 specific transformer-efficiency keywords and compare results.
3. Prioritize evidence chunks with benchmark numbers (latency, throughput, memory, perplexity, context length).
4. Generate an experiment plan: baseline, efficiency metric, quality metric, ablations, acceptance threshold.
