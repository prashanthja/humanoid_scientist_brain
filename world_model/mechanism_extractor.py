"""
Layer 3: Mechanism Extractor
Builds multi-step causal chains from beliefs and causal relations.
"""
import os, sys, psycopg2, json
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv(".env")

PG_URL = os.environ.get("DATABASE_URL", "")

def get_conn():
    return psycopg2.connect(PG_URL, connect_timeout=10)

def setup_mechanisms_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mechanisms (
            id SERIAL PRIMARY KEY,
            root_concept TEXT NOT NULL,
            chain JSONB NOT NULL,
            chain_length INTEGER DEFAULT 1,
            min_confidence REAL DEFAULT 0.5,
            domains TEXT[],
            summary TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mech_root ON mechanisms(root_concept)")
    conn.commit()
    print("✅ mechanisms table ready")

def get_causal_for_concept(conn, concept):
    """Get causal relations where source matches concept keywords."""
    cur = conn.cursor()
    # Use keywords from concept name
    words = [w for w in concept.lower().split() if len(w) > 3]
    if not words:
        return []
    conditions = " OR ".join(["LOWER(source_concept) LIKE %s"] * len(words))
    params = [f"%{w}%" for w in words]
    cur.execute(f"""
        SELECT source_concept, relation_type, target_concept, confidence
        FROM causal_relations
        WHERE ({conditions})
        AND LENGTH(target_concept) > 5
        AND source_concept NOT IN ('Module','time','age','chaos','it')
        ORDER BY confidence DESC LIMIT 5
    """, params)
    return cur.fetchall()

def build_chain(conn, concept, depth=0, max_depth=3, visited=None):
    """Build multi-step causal chain recursively."""
    if visited is None:
        visited = set()
    if concept.lower() in visited or depth >= max_depth:
        return []
    visited.add(concept.lower())

    relations = get_causal_for_concept(conn, concept)
    chains = []
    for src, rel, tgt, conf in relations:
        step = {"from": src, "relation": rel, "to": tgt, "confidence": round(conf, 2)}
        sub = build_chain(conn, tgt, depth+1, max_depth, visited.copy())
        if sub:
            for s in sub:
                chains.append([step] + s)
        else:
            chains.append([step])
    return chains

def extract_mechanisms(conn, concept, max_depth=3):
    """Extract mechanism chains for a concept."""
    chains = build_chain(conn, concept, max_depth=max_depth)
    mechanisms = []
    for chain in chains:
        if len(chain) < 1:
            continue
        steps = [chain[0]["from"]]
        for s in chain:
            steps.append(f"--[{s['relation']}]-->")
            steps.append(s["to"])
        summary = " ".join(steps)
        min_conf = min(s["confidence"] for s in chain)
        mechanisms.append({
            "root": concept,
            "chain": chain,
            "length": len(chain),
            "min_confidence": min_conf,
            "summary": summary[:500]
        })
    mechanisms.sort(key=lambda x: x["length"], reverse=True)
    return mechanisms[:3]

def save_mechanisms(conn, mechanisms):
    cur = conn.cursor()
    saved = 0
    for m in mechanisms:
        cur.execute("""
            INSERT INTO mechanisms (root_concept, chain, chain_length, min_confidence, summary)
            VALUES (%s, %s::jsonb, %s, %s, %s)
        """, (m["root"], json.dumps(m["chain"]), m["length"], m["min_confidence"], m["summary"]))
        saved += 1
    conn.commit()
    return saved

def get_mechanism_for_query(conn, query, limit=3):
    """Get mechanisms relevant to a query."""
    words = [w.strip().lower() for w in query.split() if len(w) > 3]
    if not words:
        return []
    cur = conn.cursor()
    params = [f"%{w}%" for w in words[:4]]
    conditions = " OR ".join(["LOWER(root_concept) LIKE %s"] * len(params))
    cur.execute(f"""
        SELECT root_concept, chain, chain_length, min_confidence, summary
        FROM mechanisms
        WHERE {conditions}
        ORDER BY chain_length DESC, min_confidence DESC
        LIMIT %s
    """, params + [limit])
    rows = cur.fetchall()
    return [{"root": r[0], "chain": r[1], "length": r[2],
             "confidence": r[3], "summary": r[4]} for r in rows]

def format_mechanism_prompt(mechanisms):
    if not mechanisms:
        return ""
    lines = ["=== TATTVA MECHANISMS ==="]
    for m in mechanisms:
        lines.append(f"\nMechanism for '{m['root']}':")
        lines.append(f"  {m['summary']}")
    lines.append("\nUse these mechanisms to explain WHY and HOW, not just WHAT.")
    return "\n".join(lines)

if __name__ == "__main__":
    conn = get_conn()
    setup_mechanisms_table(conn)

    cur = conn.cursor()
    cur.execute("""
        SELECT canonical_name FROM concept_cells
        WHERE lifecycle_state IN ('established','emerging')
        AND evidence_count >= 5
        ORDER BY evidence_count DESC LIMIT 20
    """)
    concepts = [r[0] for r in cur.fetchall()]
    print(f"Extracting mechanisms for {len(concepts)} concepts...")

    total = 0
    for concept in concepts:
        mechs = extract_mechanisms(conn, concept, max_depth=3)
        if mechs:
            saved = save_mechanisms(conn, mechs)
            total += saved
            print(f"  {concept}: {len(mechs)} mechanisms")
            print(f"    Best: {mechs[0]['summary'][:100]}")

    print(f"\nTotal saved: {total}")

    print("\nTest query: climate change")
    results = get_mechanism_for_query(conn, "climate change")
    for r in results:
        print(f"  [{r['length']} steps] {r['summary'][:100]}")

    conn.close()
