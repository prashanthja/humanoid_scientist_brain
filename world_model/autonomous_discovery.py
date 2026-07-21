"""
Layer 8: Autonomous Discovery Loop
Finds novel cross-domain connections automatically.
Runs every cycle: read → contradict → hypothesize → report
"""
import os, sys, psycopg2, json, time
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')

def get_conn():
    return psycopg2.connect(PG_URL, connect_timeout=10)

def setup_discoveries_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS autonomous_discoveries (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            domain_a TEXT,
            domain_b TEXT,
            concept_a TEXT,
            concept_b TEXT,
            connection_type TEXT,
            confidence REAL DEFAULT 0.5,
            evidence_count INTEGER DEFAULT 1,
            hypothesis_id INTEGER,
            status TEXT DEFAULT 'new',
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_disc_confidence ON autonomous_discoveries(confidence DESC)")
    conn.commit()
    print("✅ autonomous_discoveries table ready")

def find_cross_domain_connections(conn):
    """Find causal connections that cross domain boundaries."""
    cur = conn.cursor()
    cur.execute("""
        SELECT cr.source_concept, cr.relation_type, cr.target_concept,
               cr.confidence, b1.domain as domain_a, b2.domain as domain_b
        FROM causal_relations cr
        JOIN beliefs b1 ON LOWER(b1.concept_name) LIKE LOWER('%' || LEFT(cr.source_concept,20) || '%')
        JOIN beliefs b2 ON LOWER(b2.concept_name) LIKE LOWER('%' || LEFT(cr.target_concept,20) || '%')
        WHERE b1.domain != b2.domain
        AND cr.confidence >= 0.7
        AND b1.domain NOT IN ('unknown','other','general')
        AND b2.domain NOT IN ('unknown','other','general')
        ORDER BY cr.confidence DESC
        LIMIT 20
    """)
    return cur.fetchall()

def find_analogies(conn):
    """Find analogous_to relations — these are the richest discoveries."""
    cur = conn.cursor()
    cur.execute("""
        SELECT source_concept, target_concept, confidence, evidence_count
        FROM causal_relations
        WHERE relation_type IN ('analogous_to', 'shares_mechanism', 'generalizes')
        AND confidence >= 0.6
        ORDER BY confidence DESC, evidence_count DESC
        LIMIT 10
    """)
    return cur.fetchall()

def generate_discoveries(conn):
    """Generate autonomous discoveries from world model."""
    discoveries = []

    # Cross-domain causal connections
    cross_domain = find_cross_domain_connections(conn)
    for src, rel, tgt, conf, dom_a, dom_b in cross_domain:
        title = f"{src} [{rel}] {tgt}"
        desc = (f"Tattva found that '{src}' (from {dom_a}) {rel} '{tgt}' (from {dom_b}). "
                f"This cross-domain connection has not been explicitly tested. "
                f"Confidence: {conf:.0%}.")
        discoveries.append({
            'title': title[:200],
            'description': desc[:500],
            'domain_a': dom_a,
            'domain_b': dom_b,
            'concept_a': src[:200],
            'concept_b': tgt[:200],
            'connection_type': rel,
            'confidence': conf
        })

    # Analogies
    analogies = find_analogies(conn)
    for src, tgt, conf, ev in analogies:
        title = f"{src} is analogous to {tgt}"
        desc = (f"Tattva identified structural similarity between '{src}' and '{tgt}'. "
                f"This analogy suggests methods from one domain may apply to the other. "
                f"Evidence count: {ev}, Confidence: {conf:.0%}.")
        discoveries.append({
            'title': title[:200],
            'description': desc[:500],
            'domain_a': 'cross-domain',
            'domain_b': 'cross-domain',
            'concept_a': src[:200],
            'concept_b': tgt[:200],
            'connection_type': 'analogous_to',
            'confidence': conf
        })

    discoveries.sort(key=lambda x: x['confidence'], reverse=True)
    return discoveries[:10]

def save_discoveries(conn, discoveries):
    cur = conn.cursor()
    cur.execute("DELETE FROM autonomous_discoveries WHERE status = 'new'")
    saved = 0
    for d in discoveries:
        cur.execute("""
            INSERT INTO autonomous_discoveries
            (title, description, domain_a, domain_b, concept_a, concept_b,
             connection_type, confidence)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (d['title'], d['description'], d['domain_a'], d['domain_b'],
              d['concept_a'], d['concept_b'], d['connection_type'], d['confidence']))
        saved += 1
    conn.commit()
    return saved

def get_latest_discoveries(conn, limit=5):
    cur = conn.cursor()
    cur.execute("""
        SELECT title, description, domain_a, domain_b,
               connection_type, confidence, created_at
        FROM autonomous_discoveries
        ORDER BY confidence DESC, created_at DESC
        LIMIT %s
    """, (limit,))
    return [{'title': r[0], 'description': r[1], 'domain_a': r[2],
             'domain_b': r[3], 'connection_type': r[4],
             'confidence': r[5], 'created_at': str(r[6])[:10]}
            for r in cur.fetchall()]

def run_discovery_cycle(conn):
    """Run one full discovery cycle."""
    print("Running autonomous discovery cycle...", flush=True)
    discoveries = generate_discoveries(conn)
    saved = save_discoveries(conn, discoveries)
    print(f"Discoveries found: {len(discoveries)}, saved: {saved}", flush=True)
    return discoveries

if __name__ == "__main__":
    conn = get_conn()
    setup_discoveries_table(conn)

    discoveries = run_discovery_cycle(conn)

    print(f"\nTop discoveries:")
    for d in discoveries[:5]:
        print(f"\n  [{d['confidence']:.0%}] {d['title'][:80]}")
        print(f"  {d['domain_a']} → {d['domain_b']}")
        print(f"  {d['description'][:120]}")

    conn.close()
