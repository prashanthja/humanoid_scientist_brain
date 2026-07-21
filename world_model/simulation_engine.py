"""
Layer 6: Simulation Engine
Uses causal graph + beliefs + mechanisms to predict outcomes
WITHOUT calling the LLM for reasoning.
LLM only writes the final English summary.
"""
import os, sys, psycopg2, json
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')

def get_conn():
    return psycopg2.connect(PG_URL, connect_timeout=10)

def setup_simulations_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS simulations (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            predicted_outcome TEXT,
            confidence REAL DEFAULT 0.5,
            causal_path JSONB,
            supporting_beliefs JSONB,
            contradicting_beliefs JSONB,
            assumptions TEXT[],
            risks TEXT[],
            recommendation TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    print("✅ simulations table ready")

def get_relevant_beliefs(conn, query, limit=10):
    """Get beliefs most relevant to query."""
    words = [w.lower() for w in query.split() if len(w) > 3]
    if not words:
        return []
    params = [f'%{w}%' for w in words[:5]]
    conditions = ' OR '.join(['LOWER(belief_text) LIKE %s'] * len(params))
    cur = conn.cursor()
    cur.execute(f"""
        SELECT belief_text, supporting_count, contradicting_count,
               confidence, domain, concept_name
        FROM beliefs
        WHERE {conditions}
        ORDER BY supporting_count DESC, confidence DESC
        LIMIT %s
    """, params + [limit])
    return cur.fetchall()

def get_causal_path(conn, query):
    """Find causal path relevant to query."""
    words = [w.lower() for w in query.split() if len(w) > 3]
    if not words:
        return []
    params = [f'%{w}%' for w in words[:4]]
    conditions = ' OR '.join(['LOWER(source_concept) LIKE %s'] * len(params))
    cur = conn.cursor()
    cur.execute(f"""
        SELECT source_concept, relation_type, target_concept, confidence
        FROM causal_relations
        WHERE {conditions}
        ORDER BY confidence DESC
        LIMIT 5
    """, params)
    return cur.fetchall()

def get_mechanisms(conn, query):
    """Get mechanism chains for query."""
    words = [w.lower() for w in query.split() if len(w) > 3]
    if not words:
        return []
    params = [f'%{w}%' for w in words[:4]]
    conditions = ' OR '.join(['LOWER(root_concept) LIKE %s'] * len(params))
    cur = conn.cursor()
    cur.execute(f"""
        SELECT root_concept, summary, chain_length, min_confidence
        FROM mechanisms
        WHERE {conditions}
        ORDER BY chain_length DESC, min_confidence DESC
        LIMIT 3
    """, params)
    return cur.fetchall()

def simulate(conn, query):
    """
    Simulate outcome of a query using world model.
    Returns structured prediction WITHOUT LLM reasoning.
    """
    beliefs = get_relevant_beliefs(conn, query, limit=10)
    causal = get_causal_path(conn, query)
    mechs = get_mechanisms(conn, query)

    if not beliefs and not causal:
        return None

    # Separate supporting and contradicting
    supporting = [b for b in beliefs if b[1] > b[2]]  # sup > con
    contradicting = [b for b in beliefs if b[2] > 0]

    # Calculate overall confidence
    if supporting:
        avg_conf = sum(b[3] for b in supporting) / len(supporting)
        support_ratio = len(supporting) / len(beliefs) if beliefs else 0
        confidence = round(avg_conf * support_ratio * 0.9, 2)
    else:
        confidence = 0.3

    # Identify assumptions from mechanisms
    assumptions = []
    for mech in mechs:
        assumptions.append(f"Assumes {mech[0]} mechanism holds: {mech[1][:80]}")

    # Identify risks from contradicting beliefs
    risks = []
    for b in contradicting[:3]:
        risks.append(f"Contradicted by {b[2]} papers: {b[0][:80]}")

    # Build causal path summary
    causal_path = []
    for src, rel, tgt, conf in causal:
        causal_path.append({
            'from': src, 'relation': rel,
            'to': tgt, 'confidence': round(conf, 2)
        })

    # Determine recommendation
    if confidence >= 0.75:
        recommendation = "HIGH CONFIDENCE — worth pursuing"
    elif confidence >= 0.55:
        recommendation = "MODERATE CONFIDENCE — run small pilot first"
    elif confidence >= 0.35:
        recommendation = "LOW CONFIDENCE — needs more evidence before investing"
    else:
        recommendation = "INSUFFICIENT EVIDENCE — do not invest resources yet"

    # Build predicted outcome from top belief
    if supporting:
        top_belief = supporting[0][0]
        predicted = f"Based on {len(supporting)} supporting beliefs: {top_belief[:200]}"
    else:
        predicted = "Insufficient evidence to predict outcome confidently."

    simulation = {
        'query': query,
        'predicted_outcome': predicted,
        'confidence': confidence,
        'causal_path': causal_path,
        'supporting_count': len(supporting),
        'contradicting_count': len(contradicting),
        'supporting_beliefs': [b[0][:150] for b in supporting[:3]],
        'contradicting_beliefs': [b[0][:150] for b in contradicting[:3]],
        'assumptions': assumptions[:3],
        'risks': risks[:3],
        'recommendation': recommendation,
        'mechanisms': [m[1][:150] for m in mechs[:2]]
    }

    return simulation

def save_simulation(conn, sim):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO simulations
        (query, predicted_outcome, confidence, causal_path,
         supporting_beliefs, contradicting_beliefs,
         assumptions, risks, recommendation)
        VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s)
        RETURNING id
    """, (
        sim['query'], sim['predicted_outcome'], sim['confidence'],
        json.dumps(sim['causal_path']),
        json.dumps(sim['supporting_beliefs']),
        json.dumps(sim['contradicting_beliefs']),
        sim['assumptions'], sim['risks'], sim['recommendation']
    ))
    conn.commit()
    return cur.fetchone()[0]

def format_simulation_prompt(sim):
    """Format simulation for LLM — LLM writes English only."""
    if not sim:
        return ""
    lines = ["=== TATTVA SIMULATION RESULT ==="]
    lines.append(f"Query: {sim['query']}")
    lines.append(f"Predicted outcome: {sim['predicted_outcome']}")
    lines.append(f"Confidence: {sim['confidence']:.0%}")
    lines.append(f"Recommendation: {sim['recommendation']}")
    lines.append(f"Supporting evidence: {sim['supporting_count']} beliefs")
    lines.append(f"Contradicting evidence: {sim['contradicting_count']} beliefs")
    if sim['causal_path']:
        lines.append("\nCausal path:")
        for step in sim['causal_path'][:3]:
            lines.append(f"  {step['from']} --[{step['relation']}]--> {step['to']} ({step['confidence']:.0%})")
    if sim['assumptions']:
        lines.append("\nKey assumptions:")
        for a in sim['assumptions'][:2]:
            lines.append(f"  - {a}")
    if sim['risks']:
        lines.append("\nRisks:")
        for r in sim['risks'][:2]:
            lines.append(f"  - {r}")
    lines.append("\nWrite a clear simulation report explaining the predicted outcome,")
    lines.append("confidence level, and whether this is worth pursuing.")
    return '\n'.join(lines)

if __name__ == "__main__":
    conn = get_conn()
    setup_simulations_table(conn)

    test_queries = [
        "Will LoRA reduce memory requirements for fine-tuning?",
        "Does climate change increase extreme weather events?",
        "Will attention mechanism improve transformer performance?"
    ]

    for q in test_queries:
        print(f"\nSimulating: {q}")
        sim = simulate(conn, q)
        if sim:
            print(f"  Confidence: {sim['confidence']:.0%}")
            print(f"  Supporting: {sim['supporting_count']} beliefs")
            print(f"  Contradicting: {sim['contradicting_count']} beliefs")
            print(f"  Recommendation: {sim['recommendation']}")
            print(f"  Causal path: {len(sim['causal_path'])} steps")
            save_simulation(conn, sim)
        else:
            print("  No simulation data available")

    conn.close()
