"""
Stage 10: World Model Query Engine
Replaces pure RAG with concept activation + belief lookup + causal traversal.
LLM only writes English at the end.
"""
import os, sys, json, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')

def activate_concepts(conn, query, top_k=5):
    """Find concepts relevant to query using text matching."""
    cur = conn.cursor()
    words = [w.strip().lower() for w in query.split() if len(w) > 3]
    if not words:
        return []
    
    # Use parameterized queries — no f-string injection
    params = []
    conditions = []
    for w in words[:6]:
        conditions.append("LOWER(canonical_name) LIKE %s")
        params.append(f"%{w}%")
    
    where = " OR ".join(conditions)
    params.append(top_k)
    
    cur.execute(f"""
        SELECT canonical_name, domain, evidence_count,
               confidence_score, lifecycle_state, cell_data
        FROM concept_cells
        WHERE {where}
        ORDER BY evidence_count DESC
        LIMIT %s
    """, params)
    return cur.fetchall()

def get_beliefs_for_concept(conn, concept_name, limit=10):
    """Get beliefs about a concept with evidence counts."""
    cur = conn.cursor()
    cur.execute("""
        SELECT belief_text, supporting_count, contradicting_count,
               confidence, domain
        FROM beliefs
        WHERE LOWER(concept_name) LIKE LOWER(%s)
        ORDER BY supporting_count DESC, contradicting_count DESC
        LIMIT %s
    """, (f'%{concept_name[:30]}%', limit))
    return cur.fetchall()

def get_causal_relations(conn, concept_name, limit=5):
    """Get causal relations for a concept."""
    cur = conn.cursor()
    cur.execute("""
        SELECT relation_type, target_concept, confidence, evidence_count
        FROM causal_relations
        WHERE LOWER(source_concept) LIKE LOWER(%s)
        ORDER BY confidence DESC, evidence_count DESC
        LIMIT %s
    """, (f'%{concept_name[:30]}%', limit))
    return cur.fetchall()

def get_contradictions_for_concept(conn, concept_name, limit=5):
    """Get contradicted beliefs for a concept."""
    cur = conn.cursor()
    cur.execute("""
        SELECT belief_text, supporting_count, contradicting_count, confidence
        FROM beliefs
        WHERE LOWER(concept_name) LIKE LOWER(%s)
        AND contradicting_count > 0
        ORDER BY contradicting_count DESC
        LIMIT %s
    """, (f'%{concept_name[:30]}%', limit))
    return cur.fetchall()

def world_model_context(query):
    """
    Build structured context from world model for a query.
    Returns dict with concepts, beliefs, causal chains, contradictions.
    This replaces chunk retrieval as the primary knowledge source.
    """
    try:
        conn = psycopg2.connect(PG_URL, connect_timeout=5)

        # Step 1: Activate relevant concepts
        concepts = activate_concepts(conn, query, top_k=5)
        if not concepts:
            conn.close()
            return None

        context = {
            'concepts': [],
            'beliefs': [],
            'causal_chains': [],
            'contradictions': [],
            'world_model_summary': ''
        }

        all_beliefs = []
        all_causal = []
        all_contradictions = []

        for concept_row in concepts:
            name, domain, ev_count, conf, state, cell_data = concept_row
            concept_info = {
                'name': name,
                'domain': domain,
                'evidence': ev_count,
                'confidence': conf,
                'state': state
            }
            if cell_data and isinstance(cell_data, dict):
                concept_info['mechanism'] = cell_data.get('mechanism', [])[:3]
                concept_info['predictions'] = cell_data.get('predictions', [])[:2]
            context['concepts'].append(concept_info)

            # Get beliefs
            beliefs = get_beliefs_for_concept(conn, name, limit=8)
            for b_text, sup, con, b_conf, b_domain in beliefs:
                all_beliefs.append({
                    'concept': name,
                    'belief': b_text,
                    'supporting': sup,
                    'contradicting': con,
                    'confidence': b_conf
                })

            # Get causal relations
            causal = get_causal_relations(conn, name, limit=4)
            for rel_type, target, c_conf, ev in causal:
                all_causal.append({
                    'from': name,
                    'relation': rel_type,
                    'to': target,
                    'confidence': c_conf
                })

            # Get contradictions
            contras = get_contradictions_for_concept(conn, name, limit=3)
            for b_text, sup, con, b_conf in contras:
                all_contradictions.append({
                    'concept': name,
                    'belief': b_text,
                    'supporting': sup,
                    'contradicting': con
                })

        context['beliefs'] = all_beliefs[:15]
        context['causal_chains'] = all_causal[:10]
        context['contradictions'] = all_contradictions[:5]

        # Build summary
        concept_names = [c['name'] for c in context['concepts']]
        n_beliefs = len(context['beliefs'])
        n_contra = len(context['contradictions'])
        n_causal = len(context['causal_chains'])

        context['world_model_summary'] = (
            f"World model activated {len(concept_names)} concepts: {', '.join(concept_names[:3])}. "
            f"Found {n_beliefs} relevant beliefs, {n_causal} causal relations, "
            f"{n_contra} contradicted beliefs."
        )

        conn.close()
        return context

    except Exception as e:
        return None

def format_world_model_prompt(query, wm_context, chunks_context=""):
    """
    Format world model context into LLM prompt.
    LLM writes English only — all reasoning already done.
    """
    if not wm_context:
        return None

    lines = ["=== TATTVA WORLD MODEL ==="]
    lines.append(f"Query: {query}")
    lines.append("")

    if wm_context['concepts']:
        lines.append("ACTIVATED CONCEPTS:")
        for c in wm_context['concepts']:
            lines.append(f"  • {c['name']} [{c['domain']}] evidence={c['evidence']} state={c['state']}")
        lines.append("")

    if wm_context['beliefs']:
        lines.append("BELIEFS WITH EVIDENCE:")
        for b in wm_context['beliefs'][:8]:
            sup = b['supporting']
            con = b['contradicting']
            conf = b['confidence']
            marker = "⚠ CONTESTED" if con > 0 else "✓"
            lines.append(f"  {marker} [{conf:.2f}] {b['belief'][:120]}")
            if con > 0:
                lines.append(f"     sup={sup} con={con}")
        lines.append("")

    if wm_context['causal_chains']:
        lines.append("CAUSAL RELATIONS:")
        for c in wm_context['causal_chains'][:6]:
            lines.append(f"  {c['from']} --[{c['relation']}]--> {c['to']} (conf={c['confidence']:.2f})")
        lines.append("")

    if wm_context['contradictions']:
        lines.append("CONTRADICTED BELIEFS (review carefully):")
        for c in wm_context['contradictions']:
            lines.append(f"  ⚡ {c['belief'][:100]} (sup={c['supporting']} con={c['contradicting']})")
        lines.append("")

    lines.append("=== END WORLD MODEL ===")
    lines.append("")
    lines.append("Using the world model above, answer the query.")
    lines.append("Cite specific beliefs and their confidence levels.")
    lines.append("Flag any contradictions and explain their significance.")
    lines.append("Be precise about scope and conditions.")

    return "\n".join(lines)

if __name__ == "__main__":
    print("Testing world model query engine...")
    ctx = world_model_context("Does FlashAttention reduce memory?")
    if ctx:
        print(f"Concepts: {[c['name'] for c in ctx['concepts']]}")
        print(f"Beliefs: {len(ctx['beliefs'])}")
        print(f"Causal: {len(ctx['causal_chains'])}")
        print(f"Contradictions: {len(ctx['contradictions'])}")
        print(f"\nSummary: {ctx['world_model_summary']}")
        print("\nFormatted prompt preview:")
        prompt = format_world_model_prompt("Does FlashAttention reduce memory?", ctx)
        print(prompt[:500])
    else:
        print("No world model context found")
