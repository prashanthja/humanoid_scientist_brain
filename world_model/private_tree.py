"""
Private Company Tree
Each company gets isolated world model.
Can query public tree + private tree together.
"""
import os, sys, psycopg2, json
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL','')
def get_conn(): return psycopg2.connect(PG_URL, connect_timeout=10)

def setup_private_tree_tables(conn):
    cur = conn.cursor()
    # Add company_id to key tables
    cur.execute("""
        ALTER TABLE observations 
        ADD COLUMN IF NOT EXISTS company_id TEXT DEFAULT NULL
    """)
    cur.execute("""
        ALTER TABLE beliefs
        ADD COLUMN IF NOT EXISTS company_id TEXT DEFAULT NULL
    """)
    cur.execute("""
        ALTER TABLE concept_cells
        ADD COLUMN IF NOT EXISTS company_id TEXT DEFAULT NULL
    """)
    # Company profiles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS company_profiles (
            id SERIAL PRIMARY KEY,
            company_id TEXT UNIQUE NOT NULL,
            company_name TEXT,
            research_domains TEXT[],
            paper_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            last_active TIMESTAMP DEFAULT NOW()
        )
    """)
    # Private paper uploads
    cur.execute("""
        CREATE TABLE IF NOT EXISTS private_papers (
            id SERIAL PRIMARY KEY,
            company_id TEXT NOT NULL,
            title TEXT,
            text TEXT,
            source TEXT DEFAULT 'upload',
            processed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    print("✅ private tree tables ready")

def create_company_profile(conn, company_id, company_name, domains=None):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO company_profiles (company_id, company_name, research_domains)
        VALUES (%s, %s, %s)
        ON CONFLICT (company_id) DO UPDATE
        SET company_name=%s, last_active=NOW()
        RETURNING id
    """, (company_id, company_name, domains or [], company_name))
    conn.commit()
    return cur.fetchone()[0]

def add_private_paper(conn, company_id, title, text):
    """Add a paper to company's private tree."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO private_papers (company_id, title, text)
        VALUES (%s, %s, %s) RETURNING id
    """, (company_id, title[:200], text[:50000]))
    paper_id = cur.fetchone()[0]
    conn.commit()
    return paper_id

def process_private_paper(conn, paper_id):
    """Extract observations from private paper into company tree."""
    from world_model.observation_store import extract_observations, save_observations
    from world_model.belief_engine import extract_belief_simple, upsert_belief

    cur = conn.cursor()
    cur.execute("SELECT company_id, title, text FROM private_papers WHERE id=%s", (paper_id,))
    row = cur.fetchone()
    if not row: return 0
    company_id, title, text = row

    # Extract observations
    obs = extract_observations(text[:3000], f"private_{paper_id}", 'private', title)
    if not obs: return 0

    # Save with company_id tag
    pg2 = get_conn()
    cur2 = pg2.cursor()
    saved = 0
    for o in obs:
        cur2.execute("""
            INSERT INTO observations
            (subject, predicate, object, conditions, confidence,
             negated, domain, chunk_id, paper_title, company_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (o.get('subject',''), o.get('predicate',''), o.get('object',''),
              o.get('conditions',''), o.get('confidence',0.75),
              o.get('negated',False), 'private',
              paper_id, title[:200], company_id))
        saved += 1

    # Build beliefs for this company
    for o in obs[:20]:
        od = {'subject':o.get('subject',''),'predicate':o.get('predicate',''),
              'object':o.get('object',''),'conditions':o.get('conditions',''),
              'confidence':o.get('confidence',0.75),'negated':o.get('negated',False),
              'domain':'private'}
        b = extract_belief_simple(od)
        if b:
            try:
                upsert_belief(pg2, b['belief'], b['concept'],
                            b['observation_type'], 'private',
                            b.get('confidence',0.75), company_id=company_id)
            except: pass

    # Mark processed
    cur2.execute("UPDATE private_papers SET processed=TRUE WHERE id=%s", (paper_id,))
    pg2.commit(); pg2.close()
    return saved

def query_combined_tree(conn, query, company_id, limit=10):
    """
    Query both public world tree AND company private tree.
    Returns merged results with source labels.
    """
    from world_model.query_engine import world_model_context
    
    cur = conn.cursor()
    
    # Public tree results
    public_ctx = world_model_context(query)
    
    # Private tree results
    words = [w.lower() for w in query.split() if len(w) > 3]
    params = [f'%{w}%' for w in words[:3]]
    conditions = ' OR '.join(['LOWER(belief_text) LIKE %s'] * len(params))
    
    private_beliefs = []
    if params:
        cur.execute(f"""
            SELECT belief_text, confidence, supporting_count, 'private' as source
            FROM beliefs
            WHERE company_id = %s AND ({conditions})
            ORDER BY supporting_count DESC LIMIT %s
        """, [company_id] + params + [limit])
        private_beliefs = cur.fetchall()

    return {
        'public': public_ctx,
        'private_beliefs': private_beliefs,
        'company_id': company_id
    }

def get_company_stats(conn, company_id):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM observations WHERE company_id=%s", (company_id,))
    obs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM beliefs WHERE company_id=%s", (company_id,))
    bel = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM private_papers WHERE company_id=%s", (company_id,))
    papers = cur.fetchone()[0]
    return {'observations': obs, 'beliefs': bel, 'papers': papers}

if __name__ == "__main__":
    conn = get_conn()
    setup_private_tree_tables(conn)
    
    # Test with demo company
    cid = "demo_biotech"
    create_company_profile(conn, cid, "Demo Biotech", ['biology','medicine'])
    print(f"✅ Company profile created: {cid}")
    
    # Add test paper
    paper_id = add_private_paper(conn, cid,
        "Internal Study: Drug X reduces inflammation",
        "Drug X was administered to 50 patients. Results show Drug X reduces IL-6 levels by 40%. Drug X inhibits NF-kB pathway. Drug X causes reduction in CRP levels. Side effects include mild nausea in 10% of patients."
    )
    print(f"✅ Private paper added: ID={paper_id}")
    
    # Process it
    saved = process_private_paper(conn, paper_id)
    print(f"✅ Extracted {saved} private observations")
    
    # Stats
    stats = get_company_stats(conn, cid)
    print(f"✅ Company stats: {stats}")
    
    conn.close()
