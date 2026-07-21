"""
Layer 7: Active Learning
System decides what to read next based on:
- Concepts with low evidence (needs more papers)
- Beliefs with high uncertainty (needs verification)
- Hypothesis gaps (areas with no causal chains)
- Contradictions needing resolution
"""
import os, sys, psycopg2, json
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')

def get_conn():
    return psycopg2.connect(PG_URL, connect_timeout=10)

def setup_reading_queue_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reading_queue (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            reason TEXT,
            priority REAL DEFAULT 0.5,
            category TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    print("✅ reading_queue table ready")

def find_weak_concepts(conn, limit=10):
    """Find concepts that need more evidence."""
    cur = conn.cursor()
    cur.execute("""
        SELECT canonical_name, domain, evidence_count, lifecycle_state
        FROM concept_cells
        WHERE lifecycle_state IN ('candidate','emerging')
        AND evidence_count < 5
        ORDER BY evidence_count ASC
        LIMIT %s
    """, (limit,))
    return cur.fetchall()

def find_uncertain_beliefs(conn, limit=10):
    """Find beliefs with low confidence or high contradiction."""
    cur = conn.cursor()
    cur.execute("""
        SELECT belief_text, confidence, supporting_count,
               contradicting_count, concept_name, domain
        FROM beliefs
        WHERE contradicting_count > 0
        AND supporting_count < 5
        ORDER BY contradicting_count DESC
        LIMIT %s
    """, (limit,))
    return cur.fetchall()

def find_hypothesis_gaps(conn, limit=5):
    """Find hypotheses that need testing."""
    cur = conn.cursor()
    cur.execute("""
        SELECT concept_a, inferred_relation, concept_c,
               confidence, hypothesis_text
        FROM hypotheses
        WHERE tested = FALSE
        ORDER BY confidence DESC
        LIMIT %s
    """, (limit,))
    return cur.fetchall()

def generate_reading_queue(conn):
    """Generate prioritized reading list for what to learn next."""
    queue = []

    # 1. Weak concepts needing more papers
    weak = find_weak_concepts(conn)
    for name, domain, ev, state in weak:
        queue.append({
            'query': f"{name} {domain} recent advances",
            'reason': f"Concept '{name}' only has {ev} observations — needs more evidence",
            'priority': 0.9 - (ev * 0.1),
            'category': 'strengthen_concept'
        })

    # 2. Contradicted beliefs needing resolution
    uncertain = find_uncertain_beliefs(conn)
    for belief, conf, sup, con, concept, domain in uncertain:
        queue.append({
            'query': f"{concept} {belief[:50]} evidence",
            'reason': f"Belief contradicted by {con} papers — needs resolution",
            'priority': 0.8,
            'category': 'resolve_contradiction'
        })

    # 3. Hypothesis gaps needing testing
    gaps = find_hypothesis_gaps(conn)
    for a, rel, c, conf, text in gaps:
        queue.append({
            'query': f"{a} {rel} {c} experimental evidence",
            'reason': f"Untested hypothesis: {a} {rel} {c}",
            'priority': conf,
            'category': 'test_hypothesis'
        })

    # Sort by priority
    queue.sort(key=lambda x: x['priority'], reverse=True)
    return queue[:20]

def save_reading_queue(conn, queue):
    cur = conn.cursor()
    cur.execute("DELETE FROM reading_queue WHERE status = 'pending'")
    for item in queue:
        cur.execute("""
            INSERT INTO reading_queue (query, reason, priority, category)
            VALUES (%s, %s, %s, %s)
        """, (item['query'], item['reason'], item['priority'], item['category']))
    conn.commit()
    return len(queue)

def get_reading_queue(conn, limit=5):
    cur = conn.cursor()
    cur.execute("""
        SELECT query, reason, priority, category
        FROM reading_queue
        WHERE status = 'pending'
        ORDER BY priority DESC
        LIMIT %s
    """, (limit,))
    return [{'query': r[0], 'reason': r[1],
             'priority': r[2], 'category': r[3]}
            for r in cur.fetchall()]

def format_active_learning_prompt(queue):
    if not queue:
        return ""
    lines = ["=== TATTVA ACTIVE LEARNING — KNOWLEDGE GAPS ==="]
    lines.append("Areas where Tattva needs more evidence:")
    for i, item in enumerate(queue[:3], 1):
        lines.append(f"\n{i}. [{item['category']}] {item['reason']}")
        lines.append(f"   Searching for: {item['query']}")
    lines.append("\nMention these knowledge gaps in your response where relevant.")
    return '\n'.join(lines)

if __name__ == "__main__":
    conn = get_conn()
    setup_reading_queue_table(conn)

    print("Generating reading queue...")
    queue = generate_reading_queue(conn)
    print(f"Found {len(queue)} items to read next")

    saved = save_reading_queue(conn, queue)
    print(f"Saved {saved} items to reading queue")

    print("\nTop 5 priority reads:")
    for item in queue[:5]:
        print(f"\n  [{item['priority']:.0%}] {item['category']}")
        print(f"  Reason: {item['reason'][:80]}")
        print(f"  Query: {item['query'][:60]}")

    conn.close()
