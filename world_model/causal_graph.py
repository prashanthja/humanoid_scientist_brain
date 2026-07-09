"""
Stage 9: Causal Graph Builder
Connects concept cells through executable causal relations.
Tattva traverses this graph without LLM to answer WHY questions.

Relations: causes, enables, prevents, requires, inhibits,
           analogous_to, contradicts, supports, shares_mechanism
"""
import os, sys, json, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')

CAUSAL_RELATIONS = [
    'causes', 'enables', 'prevents', 'requires', 'inhibits',
    'accelerates', 'stabilizes', 'depends_on', 'analogous_to',
    'contradicts', 'supports', 'shares_mechanism', 'generalizes',
    'specializes', 'reduces', 'increases', 'improves', 'worsens'
]

def setup_causal_graph_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS causal_relations (
            id SERIAL PRIMARY KEY,
            source_concept TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            target_concept TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            evidence_count INTEGER DEFAULT 1,
            conditions TEXT,
            source_beliefs TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_causal_source ON causal_relations(source_concept)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_causal_target ON causal_relations(target_concept)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_causal_type ON causal_relations(relation_type)")
    conn.commit()
    print("✅ causal_relations table ready")

def extract_causal_from_beliefs(conn):
    """Extract causal relations from existing beliefs using predicate matching."""
    cur = conn.cursor()
    cur.execute("""
        SELECT concept_name, belief_text, supporting_count, confidence, domain
        FROM beliefs
        WHERE supporting_count >= 1
        ORDER BY supporting_count DESC
        LIMIT 2000
    """)
    beliefs = cur.fetchall()
    print(f"Processing {len(beliefs)} beliefs for causal relations...")

    # Causal predicate patterns
    CAUSAL_PATTERNS = {
        'causes':      ['causes', 'leads to', 'results in', 'produces', 'triggers'],
        'enables':     ['enables', 'allows', 'facilitates', 'supports', 'promotes'],
        'prevents':    ['prevents', 'inhibits', 'blocks', 'reduces risk', 'protects'],
        'requires':    ['requires', 'needs', 'depends on', 'necessitates'],
        'reduces':     ['reduces', 'decreases', 'lowers', 'diminishes', 'minimizes'],
        'increases':   ['increases', 'enhances', 'improves', 'boosts', 'amplifies'],
        'contradicts': ['contradicts', 'opposes', 'conflicts with', 'contrary to'],
        'analogous_to':['similar to', 'analogous to', 'like', 'resembles'],
    }

    relations_found = []
    for concept, belief_text, sup_count, conf, domain in beliefs:
        belief_lower = belief_text.lower()
        for rel_type, patterns in CAUSAL_PATTERNS.items():
            for pattern in patterns:
                if pattern in belief_lower:
                    # Extract target from belief text
                    idx = belief_lower.find(pattern)
                    target_raw = belief_text[idx + len(pattern):].strip()
                    # Clean target
                    target = target_raw.split('.')[0].split(',')[0].strip()[:100]
                    if len(target) > 3 and len(target) < 100:
                        relations_found.append({
                            'source': concept,
                            'relation': rel_type,
                            'target': target,
                            'confidence': min(0.95, conf + 0.1),
                            'evidence': sup_count,
                            'belief': belief_text[:200]
                        })
                    break

    return relations_found

def save_causal_relations(conn, relations):
    """Save causal relations to database."""
    cur = conn.cursor()
    saved = updated = 0
    for rel in relations:
        cur.execute("""
            SELECT id, evidence_count FROM causal_relations
            WHERE LOWER(source_concept) = LOWER(%s)
            AND relation_type = %s
            AND LOWER(target_concept) LIKE LOWER(%s)
        """, (rel['source'], rel['relation'], f"{rel['target'][:30]}%"))
        existing = cur.fetchone()
        if existing:
            cur.execute("""
                UPDATE causal_relations
                SET evidence_count = evidence_count + 1,
                    confidence = LEAST(0.95, confidence + 0.02),
                    updated_at = NOW()
                WHERE id = %s
            """, (existing[0],))
            updated += 1
        else:
            cur.execute("""
                INSERT INTO causal_relations
                (source_concept, relation_type, target_concept,
                 confidence, evidence_count, source_beliefs)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (
                rel['source'][:200], rel['relation'],
                rel['target'][:200], rel['confidence'],
                rel['evidence'], rel['belief'][:300]
            ))
            saved += 1
    conn.commit()
    return saved, updated

def traverse_causal_chain(conn, concept, max_depth=3, visited=None):
    """Traverse causal graph from a concept — no LLM needed."""
    if visited is None:
        visited = set()
    if concept in visited or max_depth == 0:
        return []
    visited.add(concept)
    cur = conn.cursor()
    cur.execute("""
        SELECT relation_type, target_concept, confidence, evidence_count
        FROM causal_relations
        WHERE LOWER(source_concept) LIKE LOWER(%s)
        ORDER BY confidence DESC, evidence_count DESC
        LIMIT 5
    """, (f'%{concept[:30]}%',))
    relations = cur.fetchall()
    chain = []
    for rel_type, target, conf, ev in relations:
        chain.append({
            'from': concept,
            'relation': rel_type,
            'to': target,
            'confidence': conf,
            'evidence': ev
        })
        # Recurse
        sub_chain = traverse_causal_chain(conn, target, max_depth-1, visited)
        chain.extend(sub_chain)
    return chain

def explain_without_llm(conn, concept):
    """Generate structured explanation from causal graph — no LLM."""
    chain = traverse_causal_chain(conn, concept, max_depth=3)
    if not chain:
        return None
    explanation = {
        'concept': concept,
        'causal_chain': chain,
        'summary': f"{concept} has {len(chain)} causal connections in the world model.",
        'top_effects': [r for r in chain if r['relation'] in ['causes','enables','increases','reduces']],
        'requirements': [r for r in chain if r['relation'] in ['requires','depends_on']],
        'contradictions': [r for r in chain if r['relation'] == 'contradicts']
    }
    return explanation

def get_graph_stats(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM causal_relations")
    total = cur.fetchone()[0]
    cur.execute("SELECT relation_type, COUNT(*) FROM causal_relations GROUP BY relation_type ORDER BY COUNT(*) DESC")
    by_type = cur.fetchall()
    cur.execute("SELECT COUNT(DISTINCT source_concept) FROM causal_relations")
    unique_sources = cur.fetchone()[0]
    return {'total': total, 'by_type': by_type, 'unique_sources': unique_sources}

if __name__ == "__main__":
    conn = psycopg2.connect(PG_URL)
    setup_causal_graph_table(conn)

    print("\nExtracting causal relations from beliefs...")
    relations = extract_causal_from_beliefs(conn)
    print(f"Found {len(relations)} potential causal relations")

    print("\nSaving to causal graph...")
    saved, updated = save_causal_relations(conn, relations)
    print(f"Saved: {saved} | Updated: {updated}")

    stats = get_graph_stats(conn)
    print(f"\nCausal graph stats:")
    print(f"  Total relations: {stats['total']}")
    print(f"  Unique source concepts: {stats['unique_sources']}")
    print(f"  By type:")
    for rel_type, count in stats['by_type']:
        print(f"    {rel_type}: {count}")

    print("\nTest: traverse causal chain from 'Climate change'")
    chain = traverse_causal_chain(conn, 'Climate change', max_depth=2)
    for link in chain[:5]:
        print(f"  {link['from']} --[{link['relation']}]--> {link['to']} (conf={link['confidence']:.2f})")

    conn.close()
