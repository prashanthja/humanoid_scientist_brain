"""
Fast extraction - batched DB writes, minimal overhead.
"""
import sys, os, time, sqlite3, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from world_model.observation_store import extract_observations, save_observations

PG = os.environ.get('DATABASE_URL', '')
def get_pg(): return psycopg2.connect(PG, connect_timeout=10)

print(f"=== FAST2 EXTRACTION === {time.strftime('%H:%M:%S')}", flush=True)

# Load processed set
pg = get_pg(); cur = pg.cursor()
cur.execute("SELECT DISTINCT chunk_id FROM observations WHERE chunk_id IS NOT NULL")
processed = set(r[0] for r in cur.fetchall())
pg.close()
print(f"Processed: {len(processed):,}", flush=True)

sc = sqlite3.connect('knowledge_base/knowledge.db')
cur2 = sc.cursor()
cur2.execute("SELECT COUNT(*) FROM chunks WHERE LENGTH(text)>100")
total = cur2.fetchone()[0]
sc.close()
print(f"Total: {total:,} | Remaining: ~{total-len(processed):,}", flush=True)

total_obs = 0; total_new = 0; start = time.time()
BATCH = 200; offset = 0
obs_buffer = []

def flush_buffer(buf):
    if not buf: return 0
    pg = get_pg(); cur = pg.cursor()
    saved = 0
    for obs_list, cid, domain, title in buf:
        try:
            s = save_observations(pg, obs_list, cid, domain, title)
            saved += s
        except: pass
    pg.commit(); pg.close()
    return saved

while True:
    sc = sqlite3.connect('knowledge_base/knowledge.db')
    cur2 = sc.cursor()
    cur2.execute("SELECT chunk_id, text, domain, paper_title FROM chunks WHERE LENGTH(text)>100 ORDER BY rowid LIMIT ? OFFSET ?", (BATCH, offset))
    rows = cur2.fetchall()
    sc.close()
    if not rows: print("✅ DONE!", flush=True); break

    new = [(r[0], str(r[1])[:3000], str(r[2] or 'ml_ai'), str(r[3] or ''))
           for r in rows if r[0] not in processed]

    for cid, text, domain, title in new:
        try:
            obs = extract_observations(text, cid, domain, title)
            if obs:
                obs_buffer.append((obs, cid, domain, title))
                total_obs += len(obs)
            processed.add(cid)
            total_new += 1
        except:
            processed.add(cid)

        # Flush every 20 chunks
        if len(obs_buffer) >= 20:
            flush_buffer(obs_buffer)
            obs_buffer = []

    offset += BATCH

    if offset % 2000 == 0:
        flush_buffer(obs_buffer); obs_buffer = []
        elapsed = time.time()-start
        rate = offset/elapsed*3600 if elapsed>0 else 1
        remaining = total - offset
        print(f"[{time.strftime('%H:%M')}] {offset:,}/{total:,} | new={total_new:,} | obs={total_obs:,} | {rate:.0f}/hr | ETA {remaining/rate:.1f}h", flush=True)

flush_buffer(obs_buffer)
pg = get_pg(); cur = pg.cursor()
cur.execute('SELECT COUNT(*) FROM observations'); print(f"Final: {cur.fetchone()[0]:,}", flush=True)
pg.close()
