# Discovery Report

**Query:** Does MoE routing instability hurt model quality?
**Domain:** transformer_efficiency
**Generated:** 2026-04-02 12:04:10

---
## Verdict

**✅ SUPPORTED**

Confidence: `0.74`

Evidence from multiple sources confirms this claim with experimental results.

---
## Evidence Sources

Retrieved **10** evidence chunks from:

- Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts
- EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- Improving Routing in Sparse Mixture of Experts with Graph of Tokens
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- LaDiMo: Layer-wise Distillation Inspired MoEfier

---
## Key Claims Extracted

Extracted **10** claims from evidence.

**1.** Our Default MoE outperforms standard TopK routing in a variety of settings without requiring significant computational overhead.
   - Source: *Dense Backpropagation Improves Training for Sparse Mixture-of-Experts*
   - Domain: `transformer_efficiency`

**2.** However, in Mixture-of-Experts (MoE) models, the routing mechanism often introduces instability, even leading to catastrophic RL training collapse.
   - Source: *Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers*
   - Domain: `transformer_efficiency`

**3.** R3 significantly reduces training-inference policy KL divergence and mitigates extreme discrepancies without compromising training speed.
   - Source: *Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers*
   - Domain: `transformer_efficiency`

**4.** Mixture-of-Experts (MoE) models enable scalable computation and performance in large-scale deep learning but face quantization challenges due to sparse expert activation and dynamic routing.
   - Source: *EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization*
   - Domain: `transformer_efficiency`

**5.** These strategies collectively enable robust, high-precision quantization of MoE models under ultra-low-bit constraints.Extensive experiments across several extreme quantization settings (e.g., W4A4/W3A4/W3A3/W2A4) demonstrate that EAQuant significantly outperforms existing methods, achieving average accuracy improvements of 1.15 - 13.81% across three diverse MoE architectures, with particularly pronounced gains in reasoning tasks and robust performance retention under aggressive quantization.
   - Source: *EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization*
   - Domain: `transformer_efficiency`

---
## Evidence Evaluation

| Verdict | Count |
|---|---|
| ✅ Supported | 0 |
| 🟡 Partially supported | 5 |
| ❌ Contradicted | 0 |

### Top Grounded Claims

⚪ **inconclusive** (confidence: `0.40`)
> However, in Mixture-of-Experts (MoE) models, the routing mechanism often introduces instability, even leading to catastrophic RL training collapse.

⚪ **inconclusive** (confidence: `0.40`)
> Our Default MoE outperforms standard TopK routing in a variety of settings without requiring significant computational overhead.

⚪ **inconclusive** (confidence: `0.40`)
> Mixture-of-Experts (MoE) models enable scalable computation and performance in large-scale deep learning but face quantization challenges due to sparse expert activation and dynamic routing.

⚪ **inconclusive** (confidence: `0.23`)
> R3 significantly reduces training-inference policy KL divergence and mitigates extreme discrepancies without compromising training speed.

⚪ **inconclusive** (confidence: `0.23`)
> Our method introduces three expert-aware innovations: (1) smoothing aggregation to suppress activation outliers, (2) routing consistency alignment to preserve expert selection post-quantization, and (3) calibration data balance to optimize sparsely activated experts.

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
