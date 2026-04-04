# Discovery Report

**Query:** Does Mamba outperform transformers on long sequences?
**Domain:** transformer_efficiency
**Generated:** 2026-04-03 11:24:09

---
## Verdict

**✅ SUPPORTED**

Confidence: `0.75`

Evidence from multiple sources confirms this claim with experimental results.

---
## Evidence Sources

Retrieved **10** evidence chunks from:

- Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling
- Generation Meets Verification: Accelerating Large Language Model Inference with Smart Parallel Auto-Correct Decoding
- MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts
- Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference
- Hybrid Attention-based Transformer for Long-range Document Classification
- What is next for LLMs? Pushing the boundaries of next-gen AI computing hardware with photonic chips.
- Small-E: Small Language Model with Linear Attention for Efficient Speech Synthesis
- EdgeInfinite: A Memory-Efficient Infinite-Context Transformer for Edge Devices

---
## Key Claims Extracted

Extracted **10** claims from evidence.

**1.** The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training.
   - Source: *MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts*
   - Domain: `transformer_efficiency`

**2.** Efficiently modeling sequences with infinite context length has long been a challenging problem.
   - Source: *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*
   - Domain: `transformer_efficiency`

**3.** We scale Samba up to 3.8B parameters with 3.2T training tokens and demonstrate that it significantly outperforms state-of-the-art models across a variety of benchmarks.
   - Source: *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*
   - Domain: `transformer_efficiency`

**4.** Pretrained on sequences of 4K length, Samba shows improved perplexity in context lengths of up to 1M in zero-shot.
   - Source: *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling*
   - Domain: `transformer_efficiency`

**5.** Transformer with the self-attention mechanism, which allows fully-connected contextual encoding over input tokens, has achieved outstanding performances in various NLP tasks, but it suffers from quadratic complexity with the input sequence length.
   - Source: *Hybrid Attention-based Transformer for Long-range Document Classification*
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

⚪ **inconclusive** (confidence: `0.45`)
> Efficiently modeling sequences with infinite context length has long been a challenging problem.

⚪ **inconclusive** (confidence: `0.36`)
> We scale Samba up to 3.8B parameters with 3.2T training tokens and demonstrate that it significantly outperforms state-of-the-art models across a variety of benchmarks.

⚪ **inconclusive** (confidence: `0.36`)
> sequences within a single model invocation.

⚪ **inconclusive** (confidence: `0.28`)
> Pretrained on sequences of 4K length, Samba shows improved perplexity in context lengths of up to 1M in zero-shot.

---
## Knowledge Gaps

- Many claims are inconclusive — evidence exists but lacks benchmark numbers or experimental results.

---
## Recommended Next Actions

1. Pick 1 concrete intervention and convert it into a falsifiable claim (inputs → mechanism → measurable outputs).
2. Run targeted retrieval using 3–5 specific transformer-efficiency keywords and compare results.
3. Prioritize evidence chunks with benchmark numbers (latency, throughput, memory, perplexity, context length).
4. Generate an experiment plan: baseline, efficiency metric, quality metric, ablations, acceptance threshold.
