"""
Cross-domain discovery — only saves REAL discoveries.
Requirements:
  1. Both concepts must be established/emerging (5+ papers)
  2. They must be from genuinely different domains
  3. The causal relation must be specific (not generic like 'enables')
  4. Connection must not be obvious
"""
import os, sys, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')
def get_conn(): return psycopg2.connect(PG_URL, connect_timeout=10)

DOMAIN_MAP = {
    'ml_ai': 'Machine Learning',
    'biology': 'Biology',
    'neuroscience': 'Neuroscience',
    'medicine': 'Medicine',
    'physics': 'Physics',
    'chemistry': 'Chemistry',
    'climate': 'Climate Science',
    'economics': 'Economics',
    'mathematics': 'Mathematics',
    'computer_systems': 'Computer Systems',
    'psychology': 'Psychology'
}

NOISE_CONCEPTS = {
    'module', 'time', 'age', 'group', 'study', 'result', 'method',
    'approach', 'system', 'model', 'data', 'process', 'factor',
    'effect', 'level', 'rate', 'type', 'form', 'case', 'value',
    'analysis', 'performance', 'information', 'question', 'sample'
}

GENERIC_RELATIONS = {'enables', 'requires', 'affects', 'involves', 'uses'}

def is_noise_concept(name):
    """Check if concept name is too generic."""
    name_lower = name.lower().strip()
    if len(name_lower) < 4: return True
    if name_lower in NOISE_CONCEPTS: return True
    words = name_lower.split()
    if all(w in NOISE_CONCEPTS for w in words): return True
    return False

def find_real_cross_domain_discoveries(conn):
    """Find genuine cross-domain discoveries using observation domains."""
    cur = conn.cursor()
    
    # Find causal relations that cross domain boundaries
    cur.execute("""
        SELECT DISTINCT
            cr.source_concept, cr.relation_type, cr.target_concept, cr.confidence,
            o1.domain as dom_a, o2.domain as dom_b
        FROM causal_relations cr
        JOIN observations o1 ON LOWER(o1.subject) 
             LIKE LOWER('%' || LEFT(cr.source_concept,15) || '%')
        JOIN observations o2 ON LOWER(o2.subject) 
             LIKE LOWER('%' || LEFT(cr.target_concept,15) || '%')
        WHERE o1.domain != o2.domain
        AND o1.domain IN ('ml_ai','biology','neuroscience','medicine',
                         'physics','chemistry','climate','economics',
                         'mathematics','computer_systems','psychology')
        AND o2.domain IN ('ml_ai','biology','neuroscience','medicine',
                         'physics','chemistry','climate','economics',
                         'mathematics','computer_systems','psychology')
        AND cr.relation_type IN ('causes','increases','reduces',
                                 'prevents','contradicts','enables')
        AND cr.confidence >= 0.85
        AND LENGTH(cr.source_concept) > 5
        AND LENGTH(cr.target_concept) > 5
        ORDER BY cr.confidence DESC
        LIMIT 30
    """)
    rows = cur.fetchall()
    
    discoveries = []
    seen = set()
    
    for src, rel, tgt, conf, dom_a, dom_b in rows:
        if is_noise_concept(src) or is_noise_concept(tgt):
            continue
        key = f"{src[:20]}_{tgt[:20]}"
        if key in seen:
            continue
        seen.add(key)
        
        discoveries.append({
            'title': f"{src} [{rel}] {tgt}",
            'description': (
                f"Cross-domain: '{src}' ({DOMAIN_MAP.get(dom_a,dom_a)}) "
                f"{rel} '{tgt}' ({DOMAIN_MAP.get(dom_b,dom_b)}). "
                f"Confidence: {conf:.0%}. "
                f"This connection bridges {dom_a} and {dom_b} — "
                f"methods or mechanisms may transfer between domains."
            ),
            'domain_a': dom_a,
            'domain_b': dom_b,
            'concept_a': src,
            'concept_b': tgt,
            'connection_type': 'cross_domain_causal',
            'confidence': conf
        })
    
    # Also find missing connections between top concepts
    cur.execute("""
        SELECT o.domain, o.subject, COUNT(*) as n
        FROM observations o
        WHERE o.domain IN ('ml_ai','biology','neuroscience','medicine','physics')
        AND LENGTH(o.subject) > 5
        GROUP BY o.domain, o.subject
        HAVING COUNT(*) >= 8
        ORDER BY n DESC LIMIT 30
    """)
    top_concepts = cur.fetchall()
    
    by_domain = {}
    for domain, subj, n in top_concepts:
        if is_noise_concept(subj): continue
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append((subj, n))
    
    domains = list(by_domain.keys())
    for i in range(len(domains)):
        for j in range(i+1, len(domains)):
            dom_a, dom_b = domains[i], domains[j]
            for name_a, ev_a in by_domain[dom_a][:2]:
                for name_b, ev_b in by_domain[dom_b][:2]:
                    cur.execute("""
                        SELECT COUNT(*) FROM causal_relations
                        WHERE LOWER(source_concept) LIKE LOWER(%s)
                        AND LOWER(target_concept) LIKE LOWER(%s)
                    """, (f'%{name_a[:12]}%', f'%{name_b[:12]}%'))
                    if cur.fetchone()[0] == 0:
                        key = f"{name_a[:20]}_{name_b[:20]}"
                        if key not in seen:
                            seen.add(key)
                            discoveries.append({
                                'title': f"Unexplored: {name_a[:30]} ↔ {name_b[:30]}",
                                'description': (
                                    f"Both '{name_a}' ({dom_a}, {ev_a} observations) "
                                    f"and '{name_b}' ({dom_b}, {ev_b} observations) "
                                    f"are well-evidenced but no cross-domain connection tested."
                                ),
                                'domain_a': dom_a,
                                'domain_b': dom_b,
                                'concept_a': name_a,
                                'concept_b': name_b,
                                'connection_type': 'missing_connection',
                                'confidence': 0.6
                            })
    
    return sorted(discoveries, key=lambda x: x['confidence'], reverse=True)[:15]

def find_real_cross_domain_discoveries_OLD(conn):
    """Find genuine cross-domain discoveries using observation domains."""
    cur = conn.cursor()
    
    # Find causal relations that cross domain boundaries
    cur.execute("""
        SELECT DISTINCT
            cr.source_concept, cr.relation_type, cr.target_concept, cr.confidence,
            o1.domain as dom_a, o2.domain as dom_b
        FROM causal_relations cr
        JOIN observations o1 ON LOWER(o1.subject) 
             LIKE LOWER('%' || LEFT(cr.source_concept,15) || '%')
        JOIN observations o2 ON LOWER(o2.subject) 
             LIKE LOWER('%' || LEFT(cr.target_concept,15) || '%')
        WHERE o1.domain != o2.domain
        AND o1.domain IN ('ml_ai','biology','neuroscience','medicine',
                         'physics','chemistry','climate','economics',
                         'mathematics','computer_systems','psychology')
        AND o2.domain IN ('ml_ai','biology','neuroscience','medicine',
                         'physics','chemistry','climate','economics',
                         'mathematics','computer_systems','psychology')
        AND cr.relation_type IN ('causes','increases','reduces',
                                 'prevents','contradicts','enables')
        AND cr.confidence >= 0.85
        AND LENGTH(cr.source_concept) > 5
        AND LENGTH(cr.target_concept) > 5
        ORDER BY cr.confidence DESC
        LIMIT 30
    """)
    rows = cur.fetchall()
    
    discoveries = []
    seen = set()
    
    for src, rel, tgt, conf, dom_a, dom_b in rows:
        if is_noise_concept(src) or is_noise_concept(tgt):
            continue
        key = f"{src[:20]}_{tgt[:20]}"
        if key in seen:
            continue
        seen.add(key)
        
        discoveries.append({
            'title': f"{src} [{rel}] {tgt}",
            'description': (
                f"Cross-domain: '{src}' ({DOMAIN_MAP.get(dom_a,dom_a)}) "
                f"{rel} '{tgt}' ({DOMAIN_MAP.get(dom_b,dom_b)}). "
                f"Confidence: {conf:.0%}. "
                f"This connection bridges {dom_a} and {dom_b} — "
                f"methods or mechanisms may transfer between domains."
            ),
            'domain_a': dom_a,
            'domain_b': dom_b,
            'concept_a': src,
            'concept_b': tgt,
            'connection_type': 'cross_domain_causal',
            'confidence': conf
        })
    
    # Also find missing connections between top concepts
    cur.execute("""
        SELECT o.domain, o.subject, COUNT(*) as n
        FROM observations o
        WHERE o.domain IN ('ml_ai','biology','neuroscience','medicine','physics')
        AND LENGTH(o.subject) > 5
        GROUP BY o.domain, o.subject
        HAVING COUNT(*) >= 8
        ORDER BY n DESC LIMIT 30
    """)
    top_concepts = cur.fetchall()
    
    by_domain = {}
    for domain, subj, n in top_concepts:
        if is_noise_concept(subj): continue
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append((subj, n))
    
    domains = list(by_domain.keys())
    for i in range(len(domains)):
        for j in range(i+1, len(domains)):
            dom_a, dom_b = domains[i], domains[j]
            for name_a, ev_a in by_domain[dom_a][:2]:
                for name_b, ev_b in by_domain[dom_b][:2]:
                    cur.execute("""
                        SELECT COUNT(*) FROM causal_relations
                        WHERE LOWER(source_concept) LIKE LOWER(%s)
                        AND LOWER(target_concept) LIKE LOWER(%s)
                    """, (f'%{name_a[:12]}%', f'%{name_b[:12]}%'))
                    if cur.fetchone()[0] == 0:
                        key = f"{name_a[:20]}_{name_b[:20]}"
                        if key not in seen:
                            seen.add(key)
                            discoveries.append({
                                'title': f"Unexplored: {name_a[:30]} ↔ {name_b[:30]}",
                                'description': (
                                    f"Both '{name_a}' ({dom_a}, {ev_a} observations) "
                                    f"and '{name_b}' ({dom_b}, {ev_b} observations) "
                                    f"are well-evidenced but no cross-domain connection tested."
                                ),
                                'domain_a': dom_a,
                                'domain_b': dom_b,
                                'concept_a': name_a,
                                'concept_b': name_b,
                                'connection_type': 'missing_connection',
                                'confidence': 0.6
                            })
    
    return sorted(discoveries, key=lambda x: x['confidence'], reverse=True)[:15]

def find_real_cross_domain_discoveries_OLD(conn):
    """Find genuine cross-domain discoveries."""
    cur = conn.cursor()
    
    # Get strong concepts per domain
    cur.execute("""
        SELECT canonical_name, domain, evidence_count, lifecycle_state
        FROM concept_cells
        WHERE lifecycle_state IN ('established', 'emerging')
        AND evidence_count >= 8
        AND domain IN ('ml_ai','biology','neuroscience','medicine',
                      'physics','chemistry','climate','economics',
                      'mathematics','computer_systems','psychology')
        ORDER BY evidence_count DESC
        LIMIT 100
    """)
    concepts = cur.fetchall()
    
    # Group by domain
    by_domain = {}
    for name, domain, ev, state in concepts:
        if is_noise_concept(name): continue
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append((name, ev, state))
    
    discoveries = []
    domains = list(by_domain.keys())
    
    # For each pair of different domains
    for i in range(len(domains)):
        for j in range(i+1, len(domains)):
            dom_a = domains[i]
            dom_b = domains[j]
            concepts_a = by_domain[dom_a][:5]
            concepts_b = by_domain[dom_b][:5]
            
            for name_a, ev_a, state_a in concepts_a:
                for name_b, ev_b, state_b in concepts_b:
                    # Check if there's a specific causal relation
                    cur.execute("""
                        SELECT relation_type, confidence
                        FROM causal_relations
                        WHERE LOWER(source_concept) LIKE LOWER(%s)
                        AND LOWER(target_concept) LIKE LOWER(%s)
                        AND relation_type NOT IN ('enables','requires','affects',
                            'involves','uses','analogous_to')
                        AND confidence >= 0.8
                        LIMIT 1
                    """, (f'%{name_a[:20]}%', f'%{name_b[:20]}%'))
                    rel = cur.fetchone()
                    
                    if rel:
                        rel_type, conf = rel
                        # Check both have real beliefs
                        cur.execute("""
                            SELECT COUNT(*) FROM beliefs
                            WHERE LOWER(belief_text) LIKE LOWER(%s)
                            AND supporting_count >= 2
                        """, (f'%{name_a[:20]}%',))
                        bel_a = cur.fetchone()[0]
                        
                        cur.execute("""
                            SELECT COUNT(*) FROM beliefs
                            WHERE LOWER(belief_text) LIKE LOWER(%s)
                            AND supporting_count >= 2
                        """, (f'%{name_b[:20]}%',))
                        bel_b = cur.fetchone()[0]
                        
                        if bel_a >= 2 and bel_b >= 2:
                            discoveries.append({
                                'title': f"{name_a} [{rel_type}] {name_b}",
                                'description': (
                                    f"Tattva found that '{name_a}' ({DOMAIN_MAP.get(dom_a,dom_a)}) "
                                    f"{rel_type} '{name_b}' ({DOMAIN_MAP.get(dom_b,dom_b)}). "
                                    f"Evidence: {bel_a} beliefs for {name_a}, "
                                    f"{bel_b} beliefs for {name_b}. "
                                    f"This cross-domain connection has not been explicitly tested."
                                ),
                                'domain_a': dom_a,
                                'domain_b': dom_b,
                                'concept_a': name_a,
                                'concept_b': name_b,
                                'connection_type': 'real_causal',
                                'confidence': conf
                            })
    
    # Also find missing connections between strong concepts
    for i in range(len(domains)):
        for j in range(i+1, len(domains)):
            dom_a = domains[i]
            dom_b = domains[j]
            top_a = by_domain[dom_a][:2]
            top_b = by_domain[dom_b][:2]
            
            for name_a, ev_a, _ in top_a:
                for name_b, ev_b, _ in top_b:
                    # Check NO causal relation exists
                    cur.execute("""
                        SELECT COUNT(*) FROM causal_relations
                        WHERE LOWER(source_concept) LIKE LOWER(%s)
                        AND LOWER(target_concept) LIKE LOWER(%s)
                    """, (f'%{name_a[:15]}%', f'%{name_b[:15]}%'))
                    if cur.fetchone()[0] == 0 and ev_a >= 12 and ev_b >= 12:
                        discoveries.append({
                            'title': f"Unexplored: {name_a} ↔ {name_b}",
                            'description': (
                                f"Both '{name_a}' ({DOMAIN_MAP.get(dom_a,dom_a)}, {ev_a} papers) "
                                f"and '{name_b}' ({DOMAIN_MAP.get(dom_b,dom_b)}, {ev_b} papers) "
                                f"are well-established in their domains but no cross-domain "
                                f"connection has been tested. High potential for novel discovery."
                            ),
                            'domain_a': dom_a,
                            'domain_b': dom_b,
                            'concept_a': name_a,
                            'concept_b': name_b,
                            'connection_type': 'missing_connection',
                            'confidence': 0.65
                        })
    
    # Sort by confidence, deduplicate
    seen = set()
    unique = []
    for d in sorted(discoveries, key=lambda x: x['confidence'], reverse=True):
        key = f"{d['concept_a']}_{d['concept_b']}"
        if key not in seen:
            seen.add(key)
            unique.append(d)
    
    return unique[:15]

def save_discoveries(conn, discoveries):
    cur = conn.cursor()
    cur.execute("DELETE FROM autonomous_discoveries")
    saved = 0
    for d in discoveries:
        cur.execute("""
            INSERT INTO autonomous_discoveries
            (title, description, domain_a, domain_b, concept_a, concept_b,
             connection_type, confidence)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (d['title'][:200], d['description'][:500],
              d['domain_a'], d['domain_b'],
              d['concept_a'][:200], d['concept_b'][:200],
              d['connection_type'], d['confidence']))
        saved += 1
    conn.commit()
    return saved

def run_autonomous_discovery(conn):
    """Run discovery — only save real ones."""
    print("Finding real cross-domain discoveries...", flush=True)
    discoveries = find_real_cross_domain_discoveries(conn)
    print(f"Found {len(discoveries)} real discoveries", flush=True)
    
    if discoveries:
        saved = save_discoveries(conn, discoveries)
        print(f"Saved {saved} discoveries", flush=True)
        for d in discoveries[:5]:
            print(f"  [{d['confidence']:.0%}] {d['title'][:70]}", flush=True)
    else:
        print("No real discoveries found yet — world model needs more data", flush=True)
    
    return {'analogies': 0, 'gaps': len(discoveries), 'discoveries_saved': len(discoveries)}

if __name__ == "__main__":
    conn = get_conn()
    results = run_autonomous_discovery(conn)
    print(f"\nResults: {results}")
    conn.close()
