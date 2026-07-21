"""
Bulk extraction — simple and fast.
"""
import sys, os, time, sqlite3, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG = os.environ.get('DATABASE_URL', '')
SQLITE = 'knowledge_base/knowledge.db'

def get_pg():
    return psycopg2.connect(PG, connect_timeout=10)

print("=== BULK WORLD MODEL EXTRACTION ===", flush=True)
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

from world_model.observation_store import extract_observations, save_observations, setup_observations_table
from world_model.belief_engine import extract_belief_simple, upsert_belief

pg = get_pg()
setup_observations_table(pg)
pg.close()
print("✅ setup done", flush=True)

# Get already processed chunk_ids from PostgreSQL
print("Loading processed chunks...", flush=True)
pg = get_pg()
cur = pg.cursor()
cur.execute("SELECT DISTINCT chunk_id FROM observations WHERE chunk_id IS NOT NULL")
processed = set(str(r[0]) for r in cur.fetchall())
pg.close()
print(f"Already processed: {len(processed):,}", flush=True)

# Get all chunks from SQLite
print("Loading chunks...", flush=True)
sc = sqlite3.connect(SQLITE)
sc_cur = sc.cursor()
sc_cur.execute("SELECT chunk_id, text, domain, paper_title FROM chunks WHERE LENGTH(text) > 100 ORDER BY rowid")
all_chunks = sc_cur.fetchall()
sc.close()
print(f"Total chunks: {len(all_chunks):,}", flush=True)

# Filter unprocessed
chunks = [(str(r[0]), str(r[1])[:3000], str(r[2] or 'ml_ai'), str(r[3] or ''))
          for r in all_chunks if str(r[0]) not in processed]
print(f"To process: {len(chunks):,}", flush=True)

total_obs = 0
total_beliefs = 0
start = time.time()

for i, (cid, text, domain, title) in enumerate(chunks):
    try:
        obs = extract_observations(text, cid, domain, title)
        if obs:
            pg = get_pg()
            saved = save_observations(pg, obs, cid, domain, title)
            pg.close()
            total_obs += saved

            # Build beliefs every 100 chunks
            if i % 100 == 0 and i > 0:
                pg = get_pg()
                cur = pg.cursor()
                cur.execute("SELECT subject,predicate,object,conditions,confidence,negated,domain FROM observations ORDER BY id DESC LIMIT 300")
                obs_rows = cur.fetchall()
                pg.close()
                created = 0
                for row in obs_rows:
                    obs_d = {'subject':row[0],'predicate':row[1],'object':row[2],
                             'conditions':row[3],'confidence':row[4],'negated':row[5],'domain':row[6]}
                    belief = extract_belief_simple(obs_d)
                    if not belief: continue
                    try:
                        pg2 = get_pg()
                        status, _ = upsert_belief(pg2, belief['belief'], belief['concept'],
                            belief['observation_type'], row[6], belief.get('confidence',0.75))
                        pg2.close()
                        if status == 'created': created += 1
                    except: pass
                total_beliefs += created

        time.sleep(0.25)

    except Exception as e:
        pass

    if i % 100 == 0:
        elapsed = time.time() - start
        rate = (i+1) / elapsed * 3600
        remaining = len(chunks) - i
        eta_hours = remaining / rate if rate > 0 else 0
        print(f"[{time.strftime('%H:%M:%S')}] {i+1:,}/{len(chunks):,} chunks | "
              f"{total_obs:,} obs | {total_beliefs} beliefs | "
              f"ETA: {eta_hours:.1f}h", flush=True)

print(f"\n=== DONE ===", flush=True)
print(f"Total observations: {total_obs:,}", flush=True)
print(f"Total beliefs: {total_beliefs:,}", flush=True)

pg = get_pg()
cur = pg.cursor()
cur.execute('SELECT COUNT(*) FROM observations'); print(f"DB observations: {cur.fetchone()[0]:,}", flush=True)
cur.execute('SELECT COUNT(*) FROM beliefs'); print(f"DB beliefs: {cur.fetchone()[0]:,}", flush=True)
pg.close()
