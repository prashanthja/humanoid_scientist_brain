"""
Algorithm 4: Concept Life Cycle
Every concept has a state:
  candidate → emerging → established → contested → deprecated → archived

State transitions are driven by evidence count and contradictions.
"""
import os, sys, psycopg2, json
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')

STATES = ['candidate', 'emerging', 'established', 'contested', 'deprecated', 'archived']

def compute_state(evidence_count, confidence_score, contradiction_count=0):
    """Compute concept state from evidence and contradictions."""
    if evidence_count < 3:
        return 'candidate'
    elif evidence_count < 6 and contradiction_count == 0:
        return 'emerging'
    elif evidence_count >= 6 and confidence_score >= 0.75 and contradiction_count == 0:
        return 'established'
    elif contradiction_count >= 2 or confidence_score < 0.5:
        return 'contested'
    elif confidence_score < 0.3:
        return 'deprecated'
    else:
        return 'emerging'

def add_lifecycle_column(conn):
    """Add lifecycle state column to concept_cells."""
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE concept_cells ADD COLUMN IF NOT EXISTS lifecycle_state TEXT DEFAULT 'candidate'")
        cur.execute("ALTER TABLE concept_cells ADD COLUMN IF NOT EXISTS contradiction_count INTEGER DEFAULT 0")
        conn.commit()
        print("✅ lifecycle columns added")
    except Exception as e:
        conn.rollback()
        print(f"Columns may already exist: {e}")

def update_all_lifecycles(conn):
    """Recompute lifecycle state for all concepts."""
    cur = conn.cursor()
    cur.execute("SELECT id, evidence_count, confidence_score, contradiction_count FROM concept_cells")
    rows = cur.fetchall()
    
    state_counts = {s: 0 for s in STATES}
    for cid, ev_count, conf, contra_count in rows:
        state = compute_state(ev_count or 1, conf or 0.5, contra_count or 0)
        cur.execute("UPDATE concept_cells SET lifecycle_state=%s WHERE id=%s", (state, cid))
        state_counts[state] += 1
    
    conn.commit()
    return state_counts

def get_lifecycle_summary(conn):
    """Get count of concepts in each lifecycle state."""
    cur = conn.cursor()
    cur.execute("""
        SELECT lifecycle_state, COUNT(*), AVG(confidence_score), AVG(evidence_count)
        FROM concept_cells
        GROUP BY lifecycle_state
        ORDER BY lifecycle_state
    """)
    return cur.fetchall()

if __name__ == "__main__":
    conn = psycopg2.connect(PG_URL)
    add_lifecycle_column(conn)
    state_counts = update_all_lifecycles(conn)
    
    print("\nConcept Life Cycle Distribution:")
    for state, count in state_counts.items():
        bar = "█" * count
        print(f"  {state:12} {bar} ({count})")
    
    print("\nDetailed summary:")
    for state, count, avg_conf, avg_ev in get_lifecycle_summary(conn):
        print(f"  {state:12} count={count} avg_conf={avg_conf:.2f} avg_evidence={avg_ev:.1f}")
    
    conn.close()
