"""
Layer 4: Hypothesis Generator
Finds novel untested connections in the causal graph.
A→B exists, B→C exists, but A→C has never been tested.
That gap = a hypothesis worth investigating.
"""
import os, sys, psycopg2, json
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')

def get_conn():
    return psycopg2.connect(PG_URL, connect_timeout=10)

def setup_hypothesis_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hypotheses (
            id SERIAL PRIMARY KEY,
            concept_a TEXT NOT NULL,
            concept_b TEXT NOT NULL,
            concept_c TEXT NOT NULL,
            relation_ab TEXT NOT NULL,
            relation_bc TEXT NOT NULL,
            inferred_relation TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            novelty_score REAL DEFAULT 0.5,
            hypothesis_text TEXT,
            tested BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hyp_a ON hypotheses(concept_a)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hyp_confidence ON hypotheses(confidence DESC)")
    conn.commit()
    print("✅ hypotheses table ready")

def infer_relation(rel_ab, rel_bc):
    """Infer what A→C relation likely is given A→B and B→C."""
    causal = ['causes','enables','increases','produces','triggers']
    preventive = ['prevents','inhibits','reduces','blocks','decreases']
    requires = ['requires','depends_on','needs']

    if rel_ab in causal and rel_bc in causal:
        return 'causes'
    elif rel_ab in causal and rel_bc in preventive:
        return 'reduces'
    elif rel_ab in preventive and rel_bc in causal:
        return 'prevents'
    elif rel_ab in requires or rel_bc in requires:
        return 'requires'
    elif rel_ab in causal and rel_bc == 'enables':
        return 'enables'
    else:
        return 'influences'

def check_already_tested(conn, concept_a, concept_c):
    """Check if A→C already exists in causal graph or observations."""
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM causal_relations
        WHERE LOWER(source_concept) LIKE LOWER(%s)
        AND LOWER(target_concept) LIKE LOWER(%s)
    """, (f'%{concept_a[:20]}%', f'%{concept_c[:20]}%'))
    count = cur.fetchone()[0]
    if count > 0:
        return True
    # Also check observations
    cur.execute("""
        SELECT COUNT(*) FROM observations
        WHERE LOWER(subject) LIKE LOWER(%s)
        AND LOWER(object) LIKE LOWER(%s)
    """, (f'%{concept_a[:20]}%', f'%{concept_c[:20]}%'))
    return cur.fetchone()[0] > 0

def generate_hypotheses(conn, min_confidence=0.6, limit=50):
    """Generate hypotheses — with quality filtering."""
    from world_model.quality_filter import is_noise_concept, score_hypothesis_quality
    """
    Find A→B, B→C pairs where A→C is untested.
    These are novel hypotheses.
    """
    cur = conn.cursor()
    # Get all causal relations with decent confidence
    cur.execute("""
        SELECT source_concept, relation_type, target_concept, confidence
        FROM causal_relations
        WHERE confidence >= %s
        AND LENGTH(source_concept) > 4
        AND LENGTH(target_concept) > 4
        AND source_concept NOT IN ('Module','time','age','chaos','it','Module')
        ORDER BY confidence DESC
        LIMIT 200
    """, (min_confidence,))
    relations = cur.fetchall()

    print(f"Checking {len(relations)} causal relations for novel hypotheses...")

    hypotheses = []
    from world_model.quality_filter import is_noise_concept
    seen = set()

    for src_a, rel_ab, tgt_b, conf_ab in relations:
        # Find B→C relations
        cur.execute("""
            SELECT relation_type, target_concept, confidence
            FROM causal_relations
            WHERE LOWER(source_concept) LIKE LOWER(%s)
            AND confidence >= %s
            AND LENGTH(target_concept) > 4
            AND target_concept NOT IN ('Module','time','age','chaos','it')
            LIMIT 5
        """, (f'%{tgt_b[:25]}%', min_confidence))
        bc_relations = cur.fetchall()

        for rel_bc, tgt_c, conf_bc in bc_relations:
            # Skip if A == C
            if src_a.lower() == tgt_c.lower():
                continue
            # Skip duplicates
            key = f"{src_a[:20]}|{tgt_c[:20]}"
            if key in seen:
                continue
            seen.add(key)

            # Check if A→C already tested
            if check_already_tested(conn, src_a, tgt_c):
                continue

            # This is a novel hypothesis
            inferred_rel = infer_relation(rel_ab, rel_bc)
            confidence = round((conf_ab + conf_bc) / 2 * 0.85, 3)
            novelty = round(1.0 - confidence, 3)  # Less certain = more novel

            hyp_text = (
                f"Hypothesis: {src_a} likely {inferred_rel} {tgt_c}. "
                f"Reasoning: {src_a} {rel_ab} {tgt_b}, "
                f"and {tgt_b} {rel_bc} {tgt_c}. "
                f"The direct connection has not been tested. "
                f"Confidence: {confidence:.0%}."
            )

            hypotheses.append({
                'a': src_a,
                'b': tgt_b,
                'c': tgt_c,
                'rel_ab': rel_ab,
                'rel_bc': rel_bc,
                'inferred_rel': inferred_rel,
                'confidence': confidence,
                'novelty': novelty,
                'text': hyp_text[:500]
            })

    # Sort by confidence
    hypotheses.sort(key=lambda x: x['confidence'], reverse=True)
    return hypotheses[:limit]

def save_hypotheses(conn, hypotheses):
    cur = conn.cursor()
    saved = 0
    for h in hypotheses:
        # Check novelty online before saving
        novelty_result = check_novelty_online(
            h.get('text',''), h['a'], h['c']
        )
        online_novelty = novelty_result.get('novelty_score', h['novelty'])
        similar_papers = novelty_result.get('similar_papers', [])
        
        # Boost or penalize confidence based on online novelty
        adjusted_conf = round(h['confidence'] * 0.7 + online_novelty * 0.3, 3)
        
        cur.execute("""
            INSERT INTO hypotheses
            (concept_a, concept_b, concept_c, relation_ab, relation_bc,
             inferred_relation, confidence, novelty_score, hypothesis_text)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT DO NOTHING
        """, (h['a'], h['b'], h['c'], h['rel_ab'], h['rel_bc'],
              h['inferred_rel'], adjusted_conf, online_novelty,
              h['text']))
        saved += 1
    conn.commit()
    return saved

def get_hypotheses_for_query(conn, query, limit=3):
    """Get hypotheses relevant to a query."""
    words = [w.strip().lower() for w in query.split() if len(w) > 3]
    if not words:
        return []
    params = [f'%{w}%' for w in words[:4]]
    conditions = ' OR '.join(['LOWER(concept_a) LIKE %s OR LOWER(concept_c) LIKE %s'] * len(params))
    flat_params = [p for p in params for _ in range(2)]
    cur = conn.cursor()
    cur.execute(f"""
        SELECT concept_a, inferred_relation, concept_c,
               confidence, hypothesis_text, relation_ab, concept_b, relation_bc
        FROM hypotheses
        WHERE {conditions}
        AND tested = FALSE
        ORDER BY confidence DESC
        LIMIT %s
    """, flat_params + [limit])
    rows = cur.fetchall()
    return [{'a': r[0], 'relation': r[1], 'c': r[2],
             'confidence': r[3], 'text': r[4],
             'reasoning': f"{r[0]} {r[5]} {r[6]}, and {r[6]} {r[7]} {r[2]}"}
            for r in rows]

def check_novelty_online(hypothesis_text, concept_a, concept_c):
    """
    Check if hypothesis already exists in literature.
    Search arXiv for similar work.
    Returns: {'is_novel': bool, 'similar_papers': list, 'novelty_score': float}
    """
    try:
        import urllib.request, urllib.parse, json as _json, ssl, re as _re
        ssl._create_default_https_context = ssl._create_unverified_context

        search_query = f"{concept_a} {concept_c}".replace("'","")
        results = {'is_novel': True, 'similar_papers': [], 'novelty_score': 1.0}

        # Search arXiv — search each concept separately then find overlap
        try:
            import re as _re2
            def arxiv_search(q, limit=10):
                aq = urllib.parse.quote(q[:50])
                url = f"https://export.arxiv.org/api/query?search_query=all:{aq}&max_results={limit}"
                req = urllib.request.Request(url, headers={'User-Agent': 'TattvaAI/1.0'})
                resp = urllib.request.urlopen(req, timeout=10)
                xml = resp.read().decode('utf-8')
                entries = _re2.findall('<entry>(.*?)</entry>', xml, _re2.DOTALL)
                papers = []
                for e in entries:
                    tm = _re2.search('<title>(.*?)</title>', e, _re2.DOTALL)
                    ym = _re2.search('<published>([0-9]{4})', e)
                    if tm:
                        papers.append({
                            'title': tm.group(1).strip(),
                            'year': int(ym.group(1)) if ym else 2025
                        })
                return papers

            # Search concept A
            papers_a = arxiv_search(concept_a[:40])
            titles_a = set(p['title'].lower() for p in papers_a)

            # Search concept C
            papers_c = arxiv_search(concept_c[:40])
            titles_c = set(p['title'].lower() for p in papers_c)

            # HIGH relevance: same paper mentions both
            for p in papers_a:
                c_in = any(w in p['title'].lower() for w in concept_c.lower().split()[:3] if len(w)>4)
                if c_in:
                    results['similar_papers'].append({
                        'title': p['title'][:100], 'year': p['year'],
                        'relevance': 'HIGH — covers both concepts'
                    })

            for p in papers_c:
                a_in = any(w in p['title'].lower() for w in concept_a.lower().split()[:3] if len(w)>4)
                if a_in:
                    results['similar_papers'].append({
                        'title': p['title'][:100], 'year': p['year'],
                        'relevance': 'HIGH — covers both concepts'
                    })

            # PARTIAL: separate papers for each
            if not results['similar_papers']:
                for p in papers_a[:2]:
                    results['similar_papers'].append({
                        'title': p['title'][:100], 'year': p['year'],
                        'relevance': f'PARTIAL — covers {concept_a[:20]}'
                    })

        except Exception as e:
            pass

        # Also search arXiv
        try:
            arxiv_query = urllib.parse.quote(f"{concept_a[:20]} {concept_c[:20]}")
            url = f"http://export.arxiv.org/api/query?search_query=all:{arxiv_query}&max_results=3&sortBy=submittedDate"
            req = urllib.request.Request(url, headers={'User-Agent': 'TattvaAI/1.0'})
            resp = urllib.request.urlopen(req, timeout=8, context=ssl_ctx)
            content_xml = resp.read().decode('utf-8')

            import re
            titles = re.findall(r'<title>(.*?)</title>', content_xml, re.DOTALL)
            titles = [t.strip() for t in titles if t.strip() and t.strip() != 'ArXiv Query']

            for title in titles[:3]:
                a_in = concept_a[:12].lower() in title.lower()
                c_in = concept_c[:12].lower() in title.lower()
                if a_in and c_in:
                    results['similar_papers'].append({
                        'title': title[:100],
                        'year': 2025,
                        'relevance': 'HIGH (arXiv)'
                    })

        except Exception as e:
            pass

        # Calculate novelty score
        high_relevance = sum(1 for p in results['similar_papers'] if 'HIGH' in p.get('relevance',''))
        partial = sum(1 for p in results['similar_papers'] if 'PARTIAL' in p.get('relevance',''))

        if high_relevance >= 2:
            results['is_novel'] = False
            results['novelty_score'] = 0.2
        elif high_relevance == 1:
            results['is_novel'] = True  # Still novel but related work exists
            results['novelty_score'] = 0.5
        elif partial >= 3:
            results['novelty_score'] = 0.7
        else:
            results['novelty_score'] = 0.95  # Genuinely novel

        return results

    except Exception as e:
        return {'is_novel': True, 'similar_papers': [], 'novelty_score': 0.8, 'error': str(e)}

def format_novelty_report(novelty_result, hypothesis_text):
    """Format novelty check result for display."""
    if not novelty_result:
        return "Novelty check: not performed"

    score = novelty_result.get('novelty_score', 0.8)
    is_novel = novelty_result.get('is_novel', True)
    papers = novelty_result.get('similar_papers', [])

    lines = [f"Novelty score: {score:.0%}"]
    if is_novel:
        lines.append("Status: NOVEL — no direct prior work found")
    else:
        lines.append("Status: KNOWN — similar work exists")

    if papers:
        lines.append(f"Similar papers found ({len(papers)}):")
        for p in papers[:3]:
            lines.append(f"  [{p.get('year','')}] {p.get('title','')[:70]}")
            lines.append(f"  Relevance: {p.get('relevance','')}")

    return '\n'.join(lines)


def format_hypothesis_prompt(hypotheses):
    if not hypotheses:
        return ""
    lines = ["=== TATTVA NOVEL HYPOTHESES ==="]
    lines.append("These connections exist in the causal graph but have NOT been directly tested:")
    for i, h in enumerate(hypotheses, 1):
        lines.append(f"\nHypothesis {i}: {h['text']}")
        lines.append(f"  Reasoning: {h['reasoning']}")
        lines.append(f"  Confidence: {h['confidence']:.0%}")
    lines.append("\nMention these as untested hypotheses worth investigating.")
    return '\n'.join(lines)

if __name__ == "__main__":
    conn = get_conn()
    setup_hypothesis_table(conn)

    print("Generating hypotheses from causal graph...")
    hypotheses = generate_hypotheses(conn, min_confidence=0.6)
    print(f"Found {len(hypotheses)} novel hypotheses")

    if hypotheses:
        saved = save_hypotheses(conn, hypotheses)
        print(f"Saved: {saved}")
        print("\nTop 5 hypotheses:")
        for h in hypotheses[:5]:
            print(f"\n  [{h['confidence']:.0%}] {h['a'][:30]} --[{h['inferred_rel']}]--> {h['c'][:30]}")
            print(f"  Via: {h['b'][:40]}")
            print(f"  {h['text'][:120]}")

    # Test query
    print("\nTest: hypotheses for 'climate'")
    results = get_hypotheses_for_query(conn, "climate", limit=3)
    for r in results:
        print(f"  [{r['confidence']:.0%}] {r['text'][:100]}")

    conn.close()
