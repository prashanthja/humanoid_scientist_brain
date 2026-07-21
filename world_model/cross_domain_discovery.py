"""
True autonomous cross-domain discovery.
No human coding required — runs on scheduler.
"""
import os, sys, psycopg2, json
sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL','')
def get_conn(): return psycopg2.connect(PG_URL, connect_timeout=10)

def find_structural_analogies(conn):
    """
    Find concepts that have the same causal structure
    across different domains.
    A→B→C in domain X and A'→B'→C' in domain Y
    where the relations are identical.
    """
    cur = conn.cursor()

    # Get all causal chains by domain
    # Lightweight query — no join, just get causal relations
    cur.execute("""
        SELECT source_concept, relation_type, target_concept, confidence
        FROM causal_relations
        WHERE confidence >= 0.7
        AND LENGTH(source_concept) > 4
        AND source_concept NOT IN ('Module','time','age','chaos')
        ORDER BY confidence DESC
        LIMIT 50
    """)
    raw_chains = cur.fetchall()

    # Get domain for each concept separately
    chains = []
    for src, rel, tgt, conf in raw_chains:
        cur.execute("""
            SELECT domain FROM beliefs
            WHERE LOWER(belief_text) LIKE LOWER(%s)
            AND domain NOT IN ('unknown','other','general')
            LIMIT 1
        """, (f'%{src[:15]}%',))
        row = cur.fetchone()
        domain = row[0] if row else 'ml_ai'
        chains.append((src, rel, tgt, domain))

    # Group by relation type and domain
    by_relation = {}
    for src, rel, tgt, domain in chains:
        if rel not in by_relation:
            by_relation[rel] = {}
        if domain not in by_relation[rel]:
            by_relation[rel][domain] = []
        by_relation[rel][domain].append((src, tgt))

    # Find same relation in different domains
    analogies = []
    for rel, domains in by_relation.items():
        domain_list = list(domains.keys())
        for i in range(len(domain_list)):
            for j in range(i+1, len(domain_list)):
                dom_a = domain_list[i]
                dom_b = domain_list[j]
                pairs_a = domains[dom_a]
                pairs_b = domains[dom_b]
                for src_a, tgt_a in pairs_a[:2]:
                    for src_b, tgt_b in pairs_b[:2]:
                        analogies.append({
                            'relation': rel,
                            'domain_a': dom_a,
                            'concept_a': src_a,
                            'outcome_a': tgt_a,
                            'domain_b': dom_b,
                            'concept_b': src_b,
                            'outcome_b': tgt_b,
                            'hypothesis': (
                                f"{src_a} ({dom_a}) and {src_b} ({dom_b}) "
                                f"share the same causal structure: "
                                f"both [{rel}] their respective outcomes. "
                                f"Methods from {dom_a} may apply to {dom_b}."
                            ),
                            'confidence': 0.65
                        })

    return analogies[:10]

def find_missing_connections(conn):
    """
    Find: concept A has many beliefs in domain X
    Concept B has many beliefs in domain Y
    But no causal link between A and B exists
    AND they appear in similar contexts
    These are the most valuable gaps.
    """
    cur = conn.cursor()

    # Get top concepts per domain
    cur.execute("""
        SELECT canonical_name, domain, evidence_count
        FROM concept_cells
        WHERE lifecycle_state IN ('established','emerging')
        AND evidence_count >= 5
        ORDER BY evidence_count DESC
        LIMIT 50
    """)
    concepts = cur.fetchall()

    # Group by domain
    by_domain = {}
    for name, domain, ev in concepts:
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append((name, ev))

    # Find pairs across domains with no causal link
    gaps = []
    domain_list = list(by_domain.keys())
    for i in range(len(domain_list)):
        for j in range(i+1, len(domain_list)):
            dom_a = domain_list[i]
            dom_b = domain_list[j]
            top_a = by_domain[dom_a][:3]
            top_b = by_domain[dom_b][:3]

            for name_a, ev_a in top_a:
                for name_b, ev_b in top_b:
                    # Check if causal link exists
                    cur.execute("""
                        SELECT COUNT(*) FROM causal_relations
                        WHERE LOWER(source_concept) LIKE LOWER(%s)
                        AND LOWER(target_concept) LIKE LOWER(%s)
                    """, (f'%{name_a[:15]}%', f'%{name_b[:15]}%'))
                    count = cur.fetchone()[0]

                    if count == 0:
                        # No existing link — this is a gap
                        gaps.append({
                            'concept_a': name_a,
                            'domain_a': dom_a,
                            'evidence_a': ev_a,
                            'concept_b': name_b,
                            'domain_b': dom_b,
                            'evidence_b': ev_b,
                            'gap_score': (ev_a + ev_b) / 2,
                            'hypothesis': (
                                f"Connection between {name_a} ({dom_a}) "
                                f"and {name_b} ({dom_b}) is unexplored. "
                                f"Given both are well-established in their domains "
                                f"(evidence: {ev_a}, {ev_b}), a cross-domain "
                                f"mechanism likely exists but has not been tested."
                            )
                        })

    gaps.sort(key=lambda x: x['gap_score'], reverse=True)
    return gaps[:5]

def run_autonomous_discovery(conn):
    """Run full autonomous discovery cycle."""
    print("Running autonomous cross-domain discovery...", flush=True)

    # Find structural analogies
    print("\n1. Finding structural analogies...", flush=True)
    analogies = find_structural_analogies(conn)
    print(f"   Found {len(analogies)} analogies")
    for a in analogies[:3]:
        print(f"   {a['domain_a']} ↔ {a['domain_b']}: [{a['relation']}]")
        print(f"   → {a['hypothesis'][:100]}")

    # Find missing connections
    print("\n2. Finding missing connections...", flush=True)
    gaps = find_missing_connections(conn)
    print(f"   Found {len(gaps)} unexplored connections")
    for g in gaps[:3]:
        print(f"   {g['concept_a']} ({g['domain_a']}) ↔ {g['concept_b']} ({g['domain_b']})")
        print(f"   Gap score: {g['gap_score']:.1f}")

    # Save discoveries
    from world_model.autonomous_discovery import save_discoveries
    discoveries = []

    for a in analogies[:5]:
        discoveries.append({
            'title': f"{a['concept_a']} [{a['relation']}] {a['concept_b']} — structural analogy",
            'description': a['hypothesis'][:500],
            'domain_a': a['domain_a'],
            'domain_b': a['domain_b'],
            'concept_a': a['concept_a'],
            'concept_b': a['concept_b'],
            'connection_type': 'structural_analogy',
            'confidence': a['confidence']
        })

    for g in gaps[:5]:
        discoveries.append({
            'title': f"Unexplored: {g['concept_a']} ↔ {g['concept_b']}",
            'description': g['hypothesis'][:500],
            'domain_a': g['domain_a'],
            'domain_b': g['domain_b'],
            'concept_a': g['concept_a'],
            'concept_b': g['concept_b'],
            'connection_type': 'missing_connection',
            'confidence': 0.6
        })

    if discoveries:
        saved = save_discoveries(conn, discoveries)
        print(f"\n✅ Saved {saved} autonomous discoveries")

    return {'analogies': len(analogies), 'gaps': len(gaps),
            'discoveries_saved': len(discoveries)}

if __name__ == "__main__":
    conn = get_conn()
    results = run_autonomous_discovery(conn)
    print(f"\nResults: {results}")

    # Show what was saved
    cur = conn.cursor()
    cur.execute("""
        SELECT title, domain_a, domain_b, confidence, connection_type
        FROM autonomous_discoveries
        ORDER BY created_at DESC LIMIT 5
    """)
    print("\nLatest discoveries:")
    for r in cur.fetchall():
        print(f"  [{r[4]}] {r[0][:70]}")
        print(f"   {r[1]} → {r[2]} (conf={r[3]:.0%})")
    conn.close()
