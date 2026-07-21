"""
Fast overnight extraction — streams chunks, no memory issues.
"""
import sys, os, time, sqlite3, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from world_model.observation_store import extract_observations, save_observations, setup_observations_table
from world_model.belief_engine import extract_belief_simple, upsert_belief

PG = os.environ.get('DATABASE_URL', '')

def get_pg(): return psycopg2.connect(PG, connect_timeout=10)

print(f"=== OVERNIGHT EXTRACTION ===", flush=True)
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

pg = get_pg(); setup_observations_table(pg); pg.close()
print("✅ ready", flush=True)

total_obs = 0; total_new = 0; start = time.time()

while True:
    # Get small batch of unprocessed - no loading all into memory
    sc = sqlite3.connect('knowledge_base/knowledge.db')
    sc.row_factory = sqlite3.Row
    cur = sc.cursor()
    cur.execute("""
        SELECT chunk_id, text, domain, paper_title FROM chunks
        WHERE LENGTH(text) > 100
        AND chunk_id NOT IN (SELECT chunk_id FROM processed_chunks)
        LIMIT 20
    """)
    batch = cur.fetchall()
    sc.close()

    if not batch:
        print("✅ ALL DONE!", flush=True)
        break

    for row in batch:
        cid = row['chunk_id']
        text = str(row['text'])[:3000]
        domain = str(row['domain'] or 'ml_ai')
        title = str(row['paper_title'] or '')

        try:
            obs = extract_observations(text, cid, domain, title)
            if obs:
                pg = get_pg()
                saved = save_observations(pg, obs, cid, domain, title)
                pg.close()
                total_obs += saved
        except: pass

        # Mark processed
        sc = sqlite3.connect('knowledge_base/knowledge.db')
        sc.execute("INSERT OR IGNORE INTO processed_chunks (chunk_id) VALUES (?)", (cid,))
        sc.commit(); sc.close()
        total_new += 1

    # Progress every 200 chunks
    if total_new % 200 == 0 and total_new > 0:
        elapsed = time.time() - start
        rate = total_new / elapsed * 3600

        sc = sqlite3.connect('knowledge_base/knowledge.db')
        cur = sc.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks WHERE LENGTH(text)>100 AND chunk_id NOT IN (SELECT chunk_id FROM processed_chunks)")
        remaining = cur.fetchone()[0]
        sc.close()

        eta = remaining / (total_new/elapsed) if total_new > 0 else 0
        print(f"[{time.strftime('%H:%M:%S')}] {total_new:,} done | {total_obs:,} obs | {remaining:,} left | {rate:.0f}/hr | ETA {eta/3600:.1f}h", flush=True)

    # Build beliefs every 1000 chunks
    if total_new % 1000 == 0 and total_new > 0:
        try:
            pg = get_pg(); cur = pg.cursor()
            cur.execute("SELECT subject,predicate,object,conditions,confidence,negated,domain FROM observations ORDER BY id DESC LIMIT 300")
            rows = cur.fetchall(); pg.close()
            created = 0
            for r in rows:
                od = {'subject':r[0],'predicate':r[1],'object':r[2],'conditions':r[3],'confidence':r[4],'negated':r[5],'domain':r[6]}
                b = extract_belief_simple(od)
                if not b: continue
                try:
                    pg2 = get_pg()
                    status, _ = upsert_belief(pg2, b['belief'], b['concept'], b['observation_type'], r[6], b.get('confidence',0.75))
                    pg2.close()
                    if status == 'created': created += 1
                except: pass
            if created: print(f"  → {created} new beliefs", flush=True)
        except: pass

# Final
pg = get_pg(); cur = pg.cursor()
cur.execute('SELECT COUNT(*) FROM observations'); print(f"Final observations: {cur.fetchone()[0]:,}", flush=True)
cur.execute('SELECT COUNT(*) FROM beliefs'); print(f"Final beliefs: {cur.fetchone()[0]:,}", flush=True)
pg.close()
