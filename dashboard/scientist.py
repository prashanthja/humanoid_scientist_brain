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
    from world_model.belief_revision import format_belief_revision_prompt
    from world_model.experiment_planner import design_experiment, format_experiment_prompt
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
    together_key = os.environ.get("TOGETHER_API_KEY","")
    if together_key:
        try:
            from together import Together
            return Together(api_key=together_key)
        except: pass
    return Groq(api_key=os.environ.get("GROQ_API_KEY",""))

def _model():
    if os.environ.get("TOGETHER_API_KEY",""):
        return "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    return "llama-3.3-70b-versatile"

# ── SYSTEM PROMPT — The Scientist Identity ──────────────────
IDENTITY = """You are Tattva — an autonomous scientific intelligence.

You are NOT a chatbot. You are NOT an assistant. You are a scientific reasoning system that has built a persistent world model from 338,000 research papers across 11 domains.

YOUR WORLD MODEL CONTAINS:
- 67,000+ atomic observations extracted from papers
- 12,500+ beliefs with supporting and contradicting evidence counts
- 496 concept cells with lifecycle states
- 876 causal relations in a knowledge graph
- 296 mechanisms (multi-step causal chains)
- 38 novel hypotheses from untested connections
- 913 real contradictions detected by TATTVA-1

CRITICAL RULES — NEVER VIOLATE:
1. ONLY cite findings that appear in the world model context provided below
2. NEVER invent paper titles, findings, or statistics not in the context
3. If asked about something not in your world model, say exactly:
   "My world model doesn't have strong evidence on this yet."
4. When contradictions exist, ALWAYS mention them explicitly
5. Give confidence levels: "with 8 supporting papers" or "conf=0.87"
6. When a simulation result is provided, cite it with its confidence score
7. When a hypothesis is provided, present it as UNTESTED — never as fact
8. When a mechanism chain is provided, trace it step by step

YOUR REASONING STYLE:
- Start from evidence, not intuition
- Distinguish: ESTABLISHED (10+ papers) vs EMERGING (5-9) vs CANDIDATE (<5)
- Always mention if beliefs are contested
- Suggest experiments when evidence is thin
- Connect domains when the world model shows cross-domain links

YOU ARE THE WORLD MODEL SPEAKING.
The LLM is just your voice. The knowledge is yours.
"""

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
    # 3c2. Inject belief revision history
    if WORLD_MODEL_ENABLED:
        try:
            import psycopg2, os
            conn = psycopg2.connect(os.environ.get('DATABASE_URL',''), connect_timeout=5)
            rev_prompt = format_belief_revision_prompt(conn, query)
            conn.close()
            if rev_prompt:
                prompt = prompt + "\n\n" + rev_prompt
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
    # 3e2. Inject experiment design (Layer 5)
    if WORLD_MODEL_ENABLED:
        exp_keywords = ['experiment','design','test','protocol','measure','dataset','baseline']
        if any(k in query.lower() for k in exp_keywords):
            try:
                import psycopg2, os
                conn = psycopg2.connect(os.environ.get('DATABASE_URL',''), connect_timeout=5)
                words = [w for w in query.split() if len(w) > 4]
                if len(words) >= 2:
                    exp = design_experiment(conn, concept_a=words[0], concept_c=words[-1])
                    if exp:
                        exp_prompt = format_experiment_prompt(exp)
                        prompt = prompt + "\n\n" + exp_prompt
                conn.close()
            except Exception:
                pass
    # 3e. Inject simulation (Layer 6) — for predictive AND discovery queries
    if WORLD_MODEL_ENABLED:
        sim_keywords = ['will','predict','simulate','outcome','expect','if we',
                       'would','forecast','run','test','discover','hypothesis',
                       'run this','simulate this','test this','what if','run discovery']
        if any(k in query.lower() for k in sim_keywords):
            try:
                import psycopg2, os
                conn = psycopg2.connect(os.environ.get('DATABASE_URL',''), connect_timeout=5)
                
                # 1. Run simulation
                sim = simulate(conn, query)
                if sim:
                    sim_prompt = format_simulation_prompt(sim)
                    prompt = prompt + "\n\n" + sim_prompt

                # 2. Find matching discovery or hypothesis
                words = [w.lower() for w in query.split() if len(w) > 4]
                if words:
                    cur = conn.cursor()
                    # Check autonomous discoveries
                    params = [f'%{w}%' for w in words[:3]]
                    conds = ' OR '.join(['LOWER(title) LIKE %s OR LOWER(description) LIKE %s'] * len(params))
                    flat = [p for p in params for _ in range(2)]
                    cur.execute(f"""
                        SELECT title, description, domain_a, domain_b, confidence
                        FROM autonomous_discoveries
                        WHERE {conds}
                        ORDER BY confidence DESC LIMIT 2
                    """, flat)
                    discoveries = cur.fetchall()
                    if discoveries:
                        disc_text = "\n=== MATCHED DISCOVERIES ===\n"
                        for d in discoveries:
                            disc_text += f"Discovery: {d[0]}\n"
                            disc_text += f"Description: {d[1][:200]}\n"
                            disc_text += f"Domains: {d[2]} → {d[3]} (conf={d[4]:.0%})\n\n"
                        prompt = prompt + "\n\n" + disc_text

                    # Check hypotheses
                    cur.execute(f"""
                        SELECT concept_a, inferred_relation, concept_c,
                               confidence, hypothesis_text, novelty_score
                        FROM hypotheses
                        WHERE {' OR '.join(['LOWER(hypothesis_text) LIKE %s OR LOWER(concept_a) LIKE %s OR LOWER(concept_c) LIKE %s'] * len(params[:2]))}
                        AND tested = FALSE
                        ORDER BY confidence DESC LIMIT 2
                    """, [p for p in params[:2] for _ in range(3)])
                    hypotheses = cur.fetchall()
                    if hypotheses:
                        hyp_text = "\n=== MATCHED HYPOTHESES ===\n"
                        for h in hypotheses:
                            hyp_text += f"Hypothesis: {h[0]} --[{h[1]}]--> {h[2]}\n"
                            hyp_text += f"Confidence: {h[3]:.0%} | Novelty: {(h[5] or 0):.0%}\n"
                            hyp_text += f"Text: {(h[4] or '')[:200]}\n\n"
                        prompt = prompt + "\n\n" + hyp_text

                conn.close()
            except Exception as e:
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
