"""
Smart extraction - processes in small batches, never loads all chunks.
"""
import sys, os, time, sqlite3, psycopg2
sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv('.env')
from world_model.observation_store import extract_observations, save_observations
from world_model.belief_engine import extract_belief_simple, upsert_belief

PG = os.environ.get('DATABASE_URL','')

def get_pg(): return psycopg2.connect(PG, connect_timeout=10)

print(f"=== SMART EXTRACTION === {time.strftime('%H:%M:%S')}", flush=True)

# Get max processed ID from PG
pg = get_pg(); cur = pg.cursor()
cur.execute("SELECT DISTINCT chunk_id FROM observations WHERE chunk_id IS NOT NULL")
processed = set(r[0] for r in cur.fetchall())
pg.close()
print(f"Processed: {len(processed):,}", flush=True)

# Get total
sc = sqlite3.connect('knowledge_base/knowledge.db')
cur2 = sc.cursor()
cur2.execute("SELECT COUNT(*) FROM chunks WHERE LENGTH(text)>100")
total = cur2.fetchone()[0]
sc.close()
print(f"Total chunks: {total:,}", flush=True)
print(f"To process: ~{total-len(processed):,}", flush=True)

total_obs = 0; total_new = 0; start = time.time()
BATCH = 100; offset = 0

while True:
    sc = sqlite3.connect('knowledge_base/knowledge.db')
    cur2 = sc.cursor()
    cur2.execute("""
        SELECT chunk_id, text, domain, paper_title
        FROM chunks WHERE LENGTH(text)>100
        ORDER BY rowid LIMIT ? OFFSET ?
    """, (BATCH, offset))
    rows = cur2.fetchall()
    sc.close()

    if not rows:
        print("✅ ALL DONE!", flush=True)
        break

    new_chunks = [(r[0], str(r[1])[:3000], str(r[2] or 'ml_ai'), str(r[3] or ''))
                  for r in rows if r[0] not in processed]

    for cid, text, domain, title in new_chunks:
        try:
            obs = extract_observations(text, cid, domain, title)
            if obs:
                pg = get_pg()
                saved = save_observations(pg, obs, cid, domain, title)
                pg.close()
                total_obs += saved
            processed.add(cid)
            total_new += 1
        except:
            processed.add(cid)

    offset += BATCH

    if offset % 2000 == 0:
        elapsed = time.time()-start
        rate = offset/elapsed*3600 if elapsed>0 else 1
        remaining = total - offset
        eta = remaining/rate
        print(f"[{time.strftime('%H:%M')}] {offset:,}/{total:,} | new={total_new:,} | obs={total_obs:,} | {rate:.0f}/hr | ETA {eta:.1f}h", flush=True)

    if total_new % 1000 == 0 and total_new > 0:
        try:
            pg = get_pg(); c = pg.cursor()
            c.execute("SELECT subject,predicate,object,conditions,confidence,negated,domain FROM observations ORDER BY id DESC LIMIT 200")
            created = 0
            for row in c.fetchall():
                od = {'subject':row[0],'predicate':row[1],'object':row[2],'conditions':row[3],'confidence':row[4],'negated':row[5],'domain':row[6]}
                b = extract_belief_simple(od)
                if not b: continue
                try:
                    pg2 = get_pg()
                    status, _ = upsert_belief(pg2, b['belief'], b['concept'], b['observation_type'], row[6], b.get('confidence',0.75))
                    pg2.close()
                    if status == 'created': created += 1
                except: pass
            pg.close()
            if created: print(f"  +{created} new beliefs", flush=True)
        except: pass

pg = get_pg(); cur = pg.cursor()
cur.execute('SELECT COUNT(*) FROM observations'); print(f"Final obs: {cur.fetchone()[0]:,}", flush=True)
cur.execute('SELECT COUNT(*) FROM beliefs'); print(f"Final beliefs: {cur.fetchone()[0]:,}", flush=True)
pg.close()
