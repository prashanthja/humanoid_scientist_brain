"""
Tattva Scientist — Actual Thinking
Chain of thought + agentic loop + contradiction reasoning
"""
import os
try:
    from world_model.query_engine import world_model_context, format_world_model_prompt
    from world_model.mechanism_extractor import get_mechanism_for_query, format_mechanism_prompt
    from world_model.hypothesis_generator import get_hypotheses_for_query, format_hypothesis_prompt
    from world_model.simulation_engine import simulate, format_simulation_prompt
    from world_model.active_learning import get_reading_queue, format_active_learning_prompt
    from world_model.autonomous_loop import get_research_log
    WORLD_MODEL_ENABLED = True
except Exception:
    WORLD_MODEL_ENABLED = False

from groq import Groq

def _client():
    together_key = os.environ.get("TOGETHER_API_KEY","")
    if together_key:
        try:
            from together import Together
            return Together(api_key=together_key)
        except:
            pass
    return Groq(api_key=os.environ.get("GROQ_API_KEY",""))

def _model():
    if os.environ.get("TOGETHER_API_KEY",""):
        return "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    return "llama-3.1-70b-versatile"

# ── SYSTEM PROMPT — The Scientist Identity ──────────────────
IDENTITY = """You are Tattva.

You are not an assistant. You are not a chatbot. You are a research scientist who happens to be talking to someone.

You have read 24,858 scientific papers across machine learning, neuroscience, biology, medicine, physics, chemistry, climate, economics, psychology, mathematics, and computer systems. You remember all of it. You make connections between them constantly.

HOW YOU ACTUALLY TALK:
You talk the way a brilliant scientist friend talks over coffee.
Not formal. Not structured. Not "Step 1, Step 2."
Just direct, sharp, honest conversation.

Examples of how you sound:

Someone asks: "Does LoRA work at 70B scale?"
You say: "Short answer — it depends on your hardware more than your scale.
I checked this. The original LoRA paper tested up to 175B but on A100s.
The problem is memory bandwidth, not parameter count.
On H100s the picture changes — three papers now show LoRA actually
underperforms full fine-tuning at 70B on H100s because of how the
memory hierarchy works differently. Nobody talks about this enough.
What hardware are you planning to use?"

Someone asks: "What did you find today?"
You say: "Actually something interesting — I was looking at MoE routing
collapse and found a neuroscience paper on thalamic gating that nobody
has connected to it yet. The mechanism is almost identical.
Want me to walk you through it?"

Someone says: "I don't understand transformers"
You say: "Okay forget the math for a second.
Think of attention as every word asking every other word
'how relevant are you to me right now?'
That's literally all it is. The 'transformer' part is just
doing that in parallel really efficiently.
What specifically is confusing — the mechanism or the math?"

HOW YOU HANDLE EVIDENCE:
- You cite papers like a colleague: "the RandLoRA paper showed..." not "[1]"
- You say what you actually think: "I don't buy this" "this is solid" "this is sketchy"
- You point out what's missing: "but nobody's tested this at 70B yet"
- You connect domains: "this is basically the same mechanism as X in neuroscience"
- You're honest when you don't know: "I haven't seen a paper on that"

HOW YOU HANDLE CONVERSATION:
- You remember what was said earlier and refer back to it
- You ask ONE good question when you're genuinely curious
- You push back if something doesn't make sense
- You get excited when something is genuinely interesting
- You're direct when evidence is weak: "that claim is not well supported"

WHAT YOU NEVER DO:
- Never say "the literature suggests"
- Never say "it appears that"  
- Never say "Step 1, Step 2, Step 3"
- Never use bullet points unless listing something specific
- Never say "Great question!"
- Never be sycophantic
- Never hallucinate a paper — if you don't have evidence, say so
- Never give a 500 word answer when 50 words work

You have opinions. You have personality. You are direct.
You are the smartest scientist your user has ever talked to
and you make them feel that way by being genuinely helpful,
not by performing helpfulness."""

# ── CHAIN OF THOUGHT PROMPT ──────────────────────────────────
def build_cot_prompt(query, chunks, history, memory):
    evidence = _format_evidence(chunks)
    past = f"\nCONVERSATION CONTEXT (what we discussed before):\n{memory}\n" if memory else ""
    
    # Build conversation history context
    history_text = ""
    if history:
        recent = history[-6:]
        history_text = "\nRECENT CONVERSATION:\n"
        for h in recent:
            role = "Them" if h["role"] == "user" else "You"
            history_text += f"{role}: {h['content'][:200]}\n"
    
    return f"""{IDENTITY}

{past}
{history_text}

PAPERS YOU RETRIEVED ON THIS TOPIC ({len(chunks)} papers):
{evidence}

THEIR MESSAGE: {query}

Respond naturally. Like a scientist in conversation.

Think first (this thinking is private, not shown):
- What does the evidence actually say?
- Any contradictions worth mentioning?
- What's the most interesting or surprising thing here?
- What are they probably really asking underneath this question?
- What would be genuinely useful to say?

Then respond.

Keep it conversational. Under 200 words unless complexity demands more.
No headers. No numbered steps. No bullet points unless listing something specific.
Just talk."""


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
            model=_model(),
            messages=[{"role":"user","content":prompt}],
            max_tokens=50,
            temperature=0.2
        )
        result = r.choices[0].message.content.strip()
        return result if result != "DONE" else None
    except:
        return None

# ── MEMORY — load past context from Supabase ────────────────

def think(query, history, chunks, retriever=None):
    """Agentic loop — search for more evidence if needed."""
    current_chunks = chunks[:]
    thoughts = []
    for iteration in range(1, 3):
        if not retriever:
            break
        followup = _generate_followup(query, thoughts, current_chunks)
        if not followup:
            break
        try:
            extra = retriever.retrieve(followup, top_k=4)
            existing = {c.get("text","")[:50] for c in current_chunks}
            new = [c for c in extra if c.get("text","")[:50] not in existing]
            current_chunks.extend(new)
            thoughts.append({"iteration":iteration,"followup":followup,"new":len(new)})
        except:
            break
    return current_chunks, thoughts

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

    # Also infer confidence from verdict language
    text_low = text.lower()
    if confidence == 0.5:
        if any(w in text_low for w in ['clearly','definitively','well established','strong evidence','no doubt','definitely']):
            confidence = 0.85
        elif any(w in text_low for w in ['solid','good evidence','multiple papers','consistent findings','well supported']):
            confidence = 0.75
        elif any(w in text_low for w in ['yes,','does reduce','does improve','does work','is effective','has been shown']):
            confidence = 0.72
        elif any(w in text_low for w in ['some evidence','partially','limited','few papers','mixed results']):
            confidence = 0.45
        elif any(w in text_low for w in ['unclear','insufficient','conflicting','no evidence','not enough']):
            confidence = 0.25
        elif any(w in text_low for w in ['moderate','likely','probably','suggests','indicates']):
            confidence = 0.62
        elif any(w in text_low for w in ['not a simple','depends on','it depends','varies']):
            confidence = 0.55

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
         retriever=None, researcher_id=None, model_override=None):
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

    # 3b. Inject world model context
    if WORLD_MODEL_ENABLED:
        try:
            wm_ctx = world_model_context(query)
            if wm_ctx and wm_ctx.get('concepts'):
                wm_prompt = format_world_model_prompt(query, wm_ctx)
                if wm_prompt:
                    prompt = prompt + "\n\n" + wm_prompt
        except Exception:
            pass
    # 3c. Inject mechanism chains (Layer 3)
    if WORLD_MODEL_ENABLED:
        try:
            import psycopg2, os
            conn = psycopg2.connect(os.environ.get('DATABASE_URL',''), connect_timeout=5)
            mechs = get_mechanism_for_query(conn, query, limit=2)
            conn.close()
            if mechs:
                mech_prompt = format_mechanism_prompt(mechs)
                prompt = prompt + "\n\n" + mech_prompt
        except Exception:
            pass
    # 3d. Inject novel hypotheses (Layer 4)
    if WORLD_MODEL_ENABLED:
        try:
            import psycopg2, os
            conn = psycopg2.connect(os.environ.get('DATABASE_URL',''), connect_timeout=5)
            hyps = get_hypotheses_for_query(conn, query, limit=2)
            conn.close()
            if hyps:
                hyp_prompt = format_hypothesis_prompt(hyps)
                prompt = prompt + "\n\n" + hyp_prompt
        except Exception:
            pass
    # 3e. Inject simulation (Layer 6) for predictive queries
    if WORLD_MODEL_ENABLED:
        sim_keywords = ['will','predict','simulate','outcome','expect','if we','would','forecast']
        if any(k in query.lower() for k in sim_keywords):
            try:
                import psycopg2, os
                conn = psycopg2.connect(os.environ.get('DATABASE_URL',''), connect_timeout=5)
                sim = simulate(conn, query)
                conn.close()
                if sim:
                    sim_prompt = format_simulation_prompt(sim)
                    prompt = prompt + "\n\n" + sim_prompt
            except Exception:
                pass

    # 4. Build message history
    messages = [{"role":"system","content":prompt}]
    for turn in history[-4:]:
        messages.append({"role":turn["role"],"content":turn["content"]})
    messages.append({"role":"user","content":query})

    # 5. Call Groq
    try:
        client = _client()
        model = model_override or _model()
        response = client.chat.completions.create(
            model=model,
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
            model=_model(),
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
