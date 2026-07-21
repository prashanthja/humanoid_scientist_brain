"""
Fast bulk extraction - no sleep, batched DB writes.
"""
import sys, os, time, sqlite3, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG = os.environ.get('DATABASE_URL', '')
SQLITE = 'knowledge_base/knowledge.db'

def get_pg():
    return psycopg2.connect(PG, connect_timeout=10)

print("=== FAST BULK EXTRACTION ===", flush=True)
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

from world_model.observation_store import extract_observations, save_observations, setup_observations_table
from world_model.belief_engine import extract_belief_simple, upsert_belief
from world_model.concept_formation import run_formation

pg = get_pg()
setup_observations_table(pg)
pg.close()

# Get processed
pg = get_pg()
cur = pg.cursor()
cur.execute("SELECT DISTINCT chunk_id FROM observations WHERE chunk_id IS NOT NULL")
processed = set(r[0] for r in cur.fetchall())
pg.close()
print(f"Already processed: {len(processed):,}", flush=True)

# Get chunks in batches from SQLite
sc = sqlite3.connect(SQLITE)
sc_cur = sc.cursor()
sc_cur.execute("SELECT COUNT(*) FROM chunks WHERE LENGTH(text) > 100")
total = sc_cur.fetchone()[0]
sc.close()
print(f"Total chunks: {total:,}", flush=True)

BATCH = 200
offset = 0
total_obs = 0
total_new = 0
start = time.time()
domain_batches = {}

while True:
    sc = sqlite3.connect(SQLITE)
    sc_cur = sc.cursor()
    sc_cur.execute("""
        SELECT chunk_id, text, domain, paper_title
        FROM chunks WHERE LENGTH(text) > 100
        ORDER BY rowid LIMIT ? OFFSET ?
    """, (BATCH, offset))
    rows = sc_cur.fetchall()
    sc.close()

    if not rows:
        break

    # Filter unprocessed
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
                processed.add(r[0])
                total_new += 1
                domain_batches[domain] = domain_batches.get(domain, 0) + 1
        except Exception as e:
            processed.add(cid[0] if isinstance(cid, tuple) else cid)

    offset += BATCH

    elapsed = time.time() - start
    rate = total_new / elapsed * 3600 if elapsed > 0 else 0
    remaining = total - offset
    eta = remaining / (total_new / elapsed) if total_new > 0 and elapsed > 0 else 0

    print(f"[{time.strftime('%H:%M:%S')}] {offset:,}/{total:,} | "
          f"{total_new:,} new | {total_obs:,} obs | "
          f"rate={rate:.0f}/hr | ETA={eta/3600:.1f}h", flush=True)

    # Build beliefs every 1000 new chunks
    if total_new % 1000 == 0 and total_new > 0:
        for domain in list(domain_batches.keys()):
            try:
                run_formation(domain=domain)
            except: pass
        domain_batches = {}

        pg = get_pg()
        cur = pg.cursor()
        cur.execute("SELECT subject,predicate,object,conditions,confidence,negated,domain FROM observations ORDER BY id DESC LIMIT 500")
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
        if created:
            print(f"  → {created} new beliefs", flush=True)

print(f"\n=== DONE ===", flush=True)
print(f"New chunks: {total_new:,} | Observations: {total_obs:,}", flush=True)
