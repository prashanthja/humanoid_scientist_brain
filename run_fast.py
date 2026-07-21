import sys, os, time, sqlite3, psycopg2
sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv('.env')
from world_model.observation_store import extract_observations, save_observations
from world_model.belief_engine import extract_belief_simple, upsert_belief

PG = os.environ.get('DATABASE_URL','')

def get_pg(): return psycopg2.connect(PG, connect_timeout=10)

print(f"=== FAST EXTRACTION === {time.strftime('%H:%M:%S')}", flush=True)

pg = get_pg(); cur = pg.cursor()
cur.execute("SELECT DISTINCT chunk_id FROM observations WHERE chunk_id IS NOT NULL")
processed = set(r[0] for r in cur.fetchall())
pg.close()
print(f"Already processed: {len(processed):,}", flush=True)

sc = sqlite3.connect('knowledge_base/knowledge.db')
sc_cur = sc.cursor()
sc_cur.execute("SELECT chunk_id, text, domain, paper_title FROM chunks WHERE LENGTH(text)>100 ORDER BY rowid")
all_rows = sc_cur.fetchall()
sc.close()

chunks = [(r[0], str(r[1])[:3000], str(r[2] or 'ml_ai'), str(r[3] or ''))
          for r in all_rows if r[0] not in processed]
print(f"To process: {len(chunks):,}", flush=True)

total_obs = 0; total_new = 0; start = time.time()
belief_buffer = []

for i, (cid, text, domain, title) in enumerate(chunks):
    try:
        obs = extract_observations(text, cid, domain, title)
        if obs:
            pg = get_pg()
            saved = save_observations(pg, obs, cid, domain, title)
            pg.close()
            total_obs += saved
            total_new += 1
    except: pass

    if i % 500 == 0 and i > 0:
        elapsed = time.time()-start
        rate = i/elapsed*3600
        remaining = len(chunks)-i
        eta = remaining/rate if rate>0 else 0
        print(f"[{time.strftime('%H:%M')}] {i:,}/{len(chunks):,} | obs={total_obs:,} | {rate:.0f}/hr | ETA {eta:.1f}h", flush=True)

    if i % 2000 == 0 and i > 0:
        try:
            pg = get_pg(); cur = pg.cursor()
            cur.execute("SELECT subject,predicate,object,conditions,confidence,negated,domain FROM observations ORDER BY id DESC LIMIT 300")
            rows = cur.fetchall(); pg.close()
            created = 0
            for row in rows:
                od = {'subject':row[0],'predicate':row[1],'object':row[2],'conditions':row[3],'confidence':row[4],'negated':row[5],'domain':row[6]}
                b = extract_belief_simple(od)
                if not b: continue
                try:
                    pg2 = get_pg()
                    status, _ = upsert_belief(pg2, b['belief'], b['concept'], b['observation_type'], row[6], b.get('confidence',0.75))
                    pg2.close()
                    if status == 'created': created += 1
                except: pass
            if created: print(f"  +{created} beliefs", flush=True)
        except: pass

print(f"DONE: {total_new:,} chunks | {total_obs:,} obs", flush=True)
pg = get_pg(); cur = pg.cursor()
cur.execute('SELECT COUNT(*) FROM observations'); print(f"Total obs: {cur.fetchone()[0]:,}", flush=True)
cur.execute('SELECT COUNT(*) FROM beliefs'); print(f"Total beliefs: {cur.fetchone()[0]:,}", flush=True)
pg.close()
