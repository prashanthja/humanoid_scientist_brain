"""
Tattva Conversational Scientist
Powered by Groq (Llama 3.1 70B) + your evidence corpus
Fast: ~0.3 second responses, free tier 14,400 req/day
"""
import os
from groq import Groq

def _get_client():
    return Groq(api_key=os.environ.get('GROQ_API_KEY',''))

SYSTEM_PROMPT = """You are Tattva, a scientific research assistant specializing in machine learning and AI research.

CRITICAL RULES:
- LoRA = Low-Rank Adaptation (parameter-efficient fine-tuning for LLMs). NEVER confuse with radio LoRa.
- FlashAttention = memory-efficient attention algorithm by Dao et al.
- KV Cache = key-value cache for faster inference
- MoE = Mixture of Experts architecture
- Mamba = state space model, alternative to transformers
- ONLY make claims supported by the evidence papers provided
- If evidence contradicts itself, explain WHY (hardware, task, scale difference)
- Be concise and precise — max 5 sentences
- Always cite paper titles when available
- Ask ONE follow-up question about hardware/task/scale context
- Never invent facts not in the evidence"""

def build_evidence_context(chunks: list, verdict_data: dict = None) -> str:
    lines = []
    if chunks:
        lines.append("EVIDENCE FROM REAL PAPERS:")
        for i, c in enumerate(chunks[:6], 1):
            title = c.get("paper_title") or c.get("title") or "Unknown"
            text = (c.get("text") or "")[:300]
            lines.append(f"\n[{i}] {title}\n{text}...")

    if verdict_data:
        v = verdict_data.get("proposal_verdict", "")
        conf = verdict_data.get("proposal_confidence", 0)
        exp = verdict_data.get("proposal_explanation", "")
        contras = verdict_data.get("contradictions", [])
        lines.append(f"\n\nSYSTEM VERDICT: {v} (confidence: {conf:.2f})")
        lines.append(f"EXPLANATION: {exp}")
        if contras:
            lines.append(f"\nCONTRADICTIONS DETECTED ({len(contras)}):")
            for c in contras[:2]:
                lines.append(f"  - {c.get('supporting_paper','')} SUPPORTS but {c.get('contradicting_paper','')} CONTRADICTS")
                lines.append(f"    Type: {c.get('contradiction_type','')} | {c.get('implication','')}")
        else:
            lines.append("\nNO CONTRADICTIONS detected in retrieved evidence.")

    if not lines:
        return "No directly relevant papers found in the corpus for this query."

    return "\n".join(lines)

def chat(query: str, history: list, chunks: list, verdict_data: dict = None) -> dict:
    evidence = build_evidence_context(chunks, verdict_data)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + evidence}
    ]

    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=400,
            temperature=0.3
        )
        content = response.choices[0].message.content

        return {
            "response": content,
            "model": "llama-3.1-70b-groq",
            "evidence_used": len(chunks),
            "papers": [c.get("paper_title", "") for c in chunks[:4]]
        }
    except Exception as e:
        return {"error": str(e)}

def get_llm_comparison(query: str) -> str:
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Answer confidently without citing sources."},
                {"role": "user", "content": query}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        return "Comparison unavailable."
