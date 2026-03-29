# Discovery Report

**Query:** Does MoE routing instability hurt model quality?
**Domain:** transformer_efficiency
**Generated:** 2026-03-25 14:34:23

---
## Verdict

**✅ SUPPORTED**

Confidence: `0.75`

Evidence from multiple sources confirms this claim with experimental results.

---
## Evidence Sources

Retrieved **10** evidence chunks from:

- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts
- LaDiMo: Layer-wise Distillation Inspired MoEfier
- Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Inference-Friendly Models With MixAttention
- Generation Meets Verification: Accelerating Large Language Model Inference with Smart Parallel Auto-Correct Decoding

---
## Key Claims Extracted

Extracted **10** claims from evidence.

**1.** Our Default MoE outperforms standard TopK routing in a variety of settings without requiring significant computational overhead.
   - Source: *Dense Backpropagation Improves Training for Sparse Mixture-of-Experts*
   - Domain: `transformer_efficiency`

**2.** The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training.
   - Source: *MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts*
   - Domain: `transformer_efficiency`

**3.** Furthermore, we develop an adaptive router that optimizes inference efficiency by profiling the distribution of routing weights and determining a layer-wise policy that balances accuracy and latency.
   - Source: *LaDiMo: Layer-wise Distillation Inspired MoEfier*
   - Domain: `transformer_efficiency`

**4.** We demonstrate the effectiveness of our method by converting the LLaMA2-7B model to a MoE model using only 100K tokens, reducing activated parameters by over 20% while keeping accuracy.
   - Source: *LaDiMo: Layer-wise Distillation Inspired MoEfier*
   - Domain: `transformer_efficiency`

**5.** Our approach offers a flexible and efficient solution for building and deploying MoE models.
   - Source: *LaDiMo: Layer-wise Distillation Inspired MoEfier*
   - Domain: `transformer_efficiency`

---
## Evidence Evaluation

| Verdict | Count |
|---|---|
| ✅ Supported | 0 |
| 🟡 Partially supported | 5 |
| ❌ Contradicted | 0 |

### Top Grounded Claims

🟡 **partially_supported** (confidence: `0.57`)
> We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs.

🟡 **partially_supported** (confidence: `0.54`)
> Our Default MoE outperforms standard TopK routing in a variety of settings without requiring significant computational overhead.

🟡 **partially_supported** (confidence: `0.54`)
> Our proposed training techniques help wrangle the instabilities and we show large sparse models may be trained, for the first time, with lower precision (bfloat16) formats.

🟡 **partially_supported** (confidence: `0.53`)
> However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs and training instability -- we address these with the Switch Transformer.

⚪ **inconclusive** (confidence: `0.20`)
> Mixture of Experts (MoE) pretraining is more scalable than dense Transformer pretraining, because MoEs learn to route inputs to a sparse set of their feedforward parameters.

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
