"""
Full world model extraction across all 11 domains.
Handles connection drops, runs overnight.
"""
import sys, time, psycopg2, os
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from world_model.observation_store import extract_observations, save_observations, setup_observations_table
from world_model.concept_formation import run_formation
from world_model.belief_engine import build_beliefs_from_observations, extract_belief_simple, upsert_belief
from world_model.concept_lifecycle import add_lifecycle_column, compute_state
from world_model.causal_graph import setup_causal_graph_table, extract_causal_from_beliefs, save_causal_relations

PG = os.environ.get('DATABASE_URL', '')

DOMAINS = [
    'ml_ai', 'biology', 'medicine', 'neuroscience',
    'physics', 'chemistry', 'climate', 'economics',
    'psychology', 'mathematics', 'computer_systems'
]

def get_conn():
    return psycopg2.connect(PG, connect_timeout=10)

def extract_domain(domain, limit=200):
    print(f"\n{'='*50}", flush=True)
    print(f"DOMAIN: {domain}", flush=True)

    # Get chunks
    conn = get_conn()
    setup_observations_table(conn)
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_id, text, domain, paper_title
        FROM chunks
        WHERE domain=%s AND LENGTH(text)>100
        ORDER BY RANDOM() LIMIT %s
    """, (domain, limit))
    chunks = cur.fetchall()
    conn.close()
    print(f"  {len(chunks)} chunks to process", flush=True)

    # Extract observations — new connection per chunk
    total_obs = 0
    for i, (cid, text, dom, title) in enumerate(chunks):
        obs = extract_observations(text, cid, dom, title or '')
        if obs:
            try:
                conn2 = get_conn()
                saved = save_observations(conn2, obs, cid, dom, title or '')
                conn2.close()
                total_obs += saved
            except Exception as e:
                print(f"  obs save error: {e}", flush=True)
        time.sleep(0.25)
        if i % 40 == 0:
            print(f"  {i}/{len(chunks)} obs={total_obs}", flush=True)

    print(f"  observations extracted: {total_obs}", flush=True)

    # Form concepts
    try:
        run_formation(domain=domain)
    except Exception as e:
        print(f"  concept formation error: {e}", flush=True)

    # Build beliefs — new connection per belief
    try:
        conn3 = get_conn()
        cur3 = conn3.cursor()
        cur3.execute("""
            SELECT subject, predicate, object, conditions,
                   confidence, negated, domain
            FROM observations WHERE domain=%s LIMIT 500
        """, (domain,))
        obs_rows = cur3.fetchall()
        conn3.close()

        created = updated = 0
        for row in obs_rows:
            obs = {'subject':row[0],'predicate':row[1],'object':row[2],
                   'conditions':row[3],'confidence':row[4],'negated':row[5],'domain':row[6]}
            belief = extract_belief_simple(obs)
            if not belief: continue
            try:
                conn4 = get_conn()
                status, _ = upsert_belief(conn4, belief['belief'], belief['concept'],
                                         belief['observation_type'], domain,
                                         belief.get('confidence', 0.75))
                conn4.close()
                if status == 'created': created += 1
                else: updated += 1
            except: pass
        print(f"  beliefs: created={created} updated={updated}", flush=True)
    except Exception as e:
        print(f"  belief error: {e}", flush=True)

    # Update lifecycle
    try:
        conn5 = get_conn()
        add_lifecycle_column(conn5)
        cur5 = conn5.cursor()
        cur5.execute("SELECT id, evidence_count, confidence_score, contradiction_count FROM concept_cells")
        for cid, ev, conf, contra in cur5.fetchall():
            state = compute_state(ev or 1, conf or 0.5, contra or 0)
            cur5.execute("UPDATE concept_cells SET lifecycle_state=%s WHERE id=%s", (state, cid))
        conn5.commit()
        conn5.close()
    except Exception as e:
        print(f"  lifecycle error: {e}", flush=True)

    print(f"  {domain} DONE", flush=True)

def print_stats():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM observations'); obs = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM beliefs'); bel = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM concept_cells'); con = cur.fetchone()[0]
        cur.execute('SELECT SUM(supporting_count), SUM(contradicting_count) FROM beliefs')
        sup, contra = cur.fetchone()
        conn.close()
        print(f"\n{'='*50}", flush=True)
        print(f"WORLD MODEL STATE:", flush=True)
        print(f"  Observations: {obs}", flush=True)
        print(f"  Beliefs:      {bel}", flush=True)
        print(f"  Concepts:     {con}", flush=True)
        print(f"  Supporting:   {sup}", flush=True)
        print(f"  Contradicting:{contra}", flush=True)
    except Exception as e:
        print(f"Stats error: {e}", flush=True)

if __name__ == "__main__":
    print("Starting full world model extraction...", flush=True)
    print_stats()

    for domain in DOMAINS:
        try:
            extract_domain(domain, limit=200)
            time.sleep(2)
        except Exception as e:
            print(f"Domain {domain} failed: {e}", flush=True)
            continue

    # Final causal graph update
    print("\nBuilding causal graph...", flush=True)
    try:
        conn = get_conn()
        setup_causal_graph_table(conn)
        relations = extract_causal_from_beliefs(conn)
        saved, updated = save_causal_relations(conn, relations)
        conn.close()
        print(f"Causal relations: saved={saved} updated={updated}", flush=True)
    except Exception as e:
        print(f"Causal error: {e}", flush=True)

    print_stats()
    print("\nALL DONE", flush=True)
