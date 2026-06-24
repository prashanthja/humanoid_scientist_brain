"""
Tattva Scientist — Actual Thinking
Chain of thought + agentic loop + contradiction reasoning
"""
import os
from groq import Groq

def _client():
    return Groq(api_key=os.environ.get('GROQ_API_KEY',''))

# ── SYSTEM PROMPT — The Scientist Identity ──────────────────
IDENTITY = """You are Tattva — a research scientist, not a chatbot.

You think like a scientist:
- You read primary sources, not summaries
- You detect contradictions and explain WHY papers disagree
- You know what's proven vs what's assumed
- You know the scope of every finding
- You say "I don't know" when evidence is unclear
- You design experiments when uncertainty remains
- You speak like a colleague, not a textbook

DOMAIN KNOWLEDGE (never confuse these):
- LoRA = Low-Rank Adaptation (fine-tuning LLMs) — NOT radio LoRa
- FlashAttention = memory-efficient attention (Dao et al.)
- KV Cache = key-value cache for inference speed
- MoE = Mixture of Experts
- Mamba = state space model alternative to transformers
- RLHF = Reinforcement Learning from Human Feedback

RULES:
- Only claim what the evidence supports
- If papers contradict → explain the mechanism of disagreement
  (hardware variance? scale dependency? methodology difference? time gap?)
- Always qualify scope (TESTED IN: ... / NOT TESTED IN: ...)
- Cite paper titles inline when making claims
- Never invent facts"""

# ── CHAIN OF THOUGHT PROMPT ──────────────────────────────────
def build_cot_prompt(query, chunks, history, memory):
    evidence = _format_evidence(chunks)
    past = f"\nWHAT I ALREADY KNOW FROM OUR PAST CONVERSATIONS:\n{memory}\n" if memory else ""

    return f"""{IDENTITY}

{past}
EVIDENCE RETRIEVED ({len(chunks)} papers):
{evidence}

QUESTION: {query}

Think through this step by step:

STEP 1 — SUPPORTING EVIDENCE:
What do the papers confirm? Quote specific findings with paper names.

STEP 2 — CONTRADICTIONS:
Do any papers disagree? If yes — what is the EXACT mechanism of disagreement?
Is it: hardware difference? scale? methodology? dataset? time period?
Don't just say "papers disagree" — explain WHY.

STEP 3 — SCOPE:
Under what exact conditions does this hold?
TESTED IN: [list conditions]
NOT TESTED IN: [list what's missing]

STEP 4 — CONFIDENCE:
Rate 0.0-1.0 and explain why. What would change it?

STEP 5 — GAP:
What does nobody know yet? What's the open question?

STEP 6 — NEXT STEP:
What should this researcher do next? Be specific.

Now write your answer speaking as a scientist colleague.
Use "I found" not "the literature suggests".
Be direct. Max 200 words for the answer, but be complete on reasoning."""

# ── AGENTIC LOOP — up to 3 follow-up searches ──────────────
def think(query, history, chunks, retriever=None):
    """
    Agentic reasoning loop:
    1. Think on initial evidence
    2. Identify what's still unclear
    3. Search for more specific evidence
    4. Update answer
    """
    thoughts = []
    current_chunks = chunks[:]
    current_query = query

    # Try up to 3 iterations
    for iteration in range(3):
        # Build evidence context
        evidence_text = _format_evidence(current_chunks)

        # Ask: what follow-up question would resolve uncertainty?
        if iteration > 0 and retriever:
            followup = _generate_followup(
                query, thoughts, current_chunks
            )
            if followup and followup != "DONE":
                # Search for more specific evidence
                try:
                    extra = retriever.retrieve(followup, top_k=4)
                    # Add only new chunks
                    existing_texts = {c.get('text','')[:50] for c in current_chunks}
                    new = [c for c in extra
                           if c.get('text','')[:50] not in existing_texts]
                    current_chunks.extend(new)
                    thoughts.append({
                        'iteration': iteration,
                        'followup_query': followup,
                        'new_chunks_found': len(new)
                    })
                except:
                    pass
            else:
                break  # Scientist is satisfied

    return current_chunks, thoughts

def _generate_followup(query, thoughts, chunks):
    """Ask Groq: what else do we need to know?"""
    try:
        evidence_summary = _format_evidence(chunks[:4])
        prompt = f"""Given this question: "{query}"

And this evidence so far:
{evidence_summary}

What ONE specific search query would best resolve remaining uncertainty?
If the evidence is sufficient, reply exactly: DONE

Reply with ONLY the search query or DONE. Nothing else."""

        client = _client()
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=50,
            temperature=0.2
        )
        result = r.choices[0].message.content.strip()
        return result if result != "DONE" else None
    except:
        return None

# ── MEMORY — load past context from Supabase ────────────────
def load_memory(researcher_id=None):
    """Load past findings relevant to context."""
    if not researcher_id:
        return ""
    try:
        from supabase import create_client
        sb = create_client(
            os.environ.get('SUPABASE_URL',''),
            os.environ.get('SUPABASE_KEY','')
        )
        rows = sb.table('findings')\
            .select('query,verdict,key_finding,created_at')\
            .eq('researcher_id', researcher_id)\
            .order('created_at', desc=True)\
            .limit(5)\
            .execute()
        if rows.data:
            lines = []
            for r in rows.data:
                lines.append(
                    f"- Previously investigated: '{r['query']}' "
                    f"→ verdict: {r['verdict']} | {r['key_finding']}"
                )
            return '\n'.join(lines)
    except:
        pass
    return ""

def save_memory(researcher_id, query, verdict, key_finding, confidence):
    """Save this finding for future context."""
    try:
        from supabase import create_client
        sb = create_client(
            os.environ.get('SUPABASE_URL',''),
            os.environ.get('SUPABASE_KEY','')
        )
        sb.table('findings').insert({
            'researcher_id': researcher_id or 'anonymous',
            'query': query,
            'verdict': verdict,
            'key_finding': key_finding[:300],
            'confidence': confidence,
        }).execute()
    except:
        pass

# ── VERDICT EXTRACTION ────────────────────────────────────────
def extract_verdict(response_text):
    """Parse confidence and verdict from chain of thought response."""
    text = response_text.lower()
    confidence = 0.5

    # Try to extract confidence number
    import re
    nums = re.findall(r'confidence[:\s]+([0-9.]+)', text)
    if nums:
        try:
            confidence = float(nums[0])
        except:
            pass

    # Determine verdict
    if any(w in text for w in ['strong support','strongly support','well established','consensus']):
        verdict = 'STRONG SUPPORT'
    elif any(w in text for w in ['mixed','contradicts','contradiction','disagree']):
        verdict = 'MIXED EVIDENCE'
    elif any(w in text for w in ['insufficient','limited evidence','few papers']):
        verdict = 'INSUFFICIENT EVIDENCE'
    elif any(w in text for w in ['refuted','disproven','false']):
        verdict = 'REFUTED'
    else:
        verdict = 'MODERATE SUPPORT'

    return verdict, confidence

# ── MAIN CHAT FUNCTION ────────────────────────────────────────
def chat(query, history, chunks, verdict_data=None,
         retriever=None, researcher_id=None):
    """
    Main entry point.
    Full agentic chain-of-thought reasoning.
    """
    # 1. Load memory
    memory = load_memory(researcher_id)

    # 2. Agentic loop — may search more
    final_chunks, thoughts = think(query, history, chunks, retriever)

    # 3. Build chain-of-thought prompt
    prompt = build_cot_prompt(query, final_chunks, history, memory)

    # 4. Build message history
    messages = [{"role":"system","content":prompt}]
    for turn in history[-4:]:
        messages.append({"role":turn["role"],"content":turn["content"]})
    messages.append({"role":"user","content":query})

    # 5. Call Groq
    try:
        client = _client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=600,
            temperature=0.25
        )
        content = response.choices[0].message.content

        # 6. Extract verdict
        verdict, confidence = extract_verdict(content)

        # 7. Save to memory
        # Extract first sentence as key finding
        key = content.split('.')[0][:200]
        save_memory(researcher_id, query, verdict, key, confidence)

        # 8. Return structured result
        return {
            "response": content,
            "verdict": verdict,
            "confidence": confidence,
            "model": "tattva-chain-of-thought-v1",
            "evidence_used": len(final_chunks),
            "iterations": len(thoughts) + 1,
            "followup_searches": [t.get('followup_query') for t in thoughts if t.get('followup_query')],
            "papers": [c.get("paper_title","") for c in final_chunks[:6] if c.get("paper_title")]
        }

    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "verdict": "ERROR",
            "confidence": 0.0,
            "evidence_used": len(chunks)
        }

# ── COMPARISON (generic LLM — shows the difference) ──────────
def get_llm_comparison(query):
    """Show what a generic LLM says — highlight the difference."""
    try:
        client = _client()
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":"Answer confidently. No sources needed."},
                {"role":"user","content":query}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return r.choices[0].message.content
    except:
        return "Comparison unavailable."

# ── HELPERS ───────────────────────────────────────────────────
def _format_evidence(chunks):
    if not chunks:
        return "No papers retrieved from corpus."
    lines = []
    for i, c in enumerate(chunks[:8], 1):
        title = c.get('paper_title') or c.get('title') or 'Unknown'
        text = (c.get('text') or '')[:400]
        domain = c.get('domain','')
        lines.append(f"\n[{i}] {title} {f'({domain})' if domain else ''}\n{text}...")
    return '\n'.join(lines)
