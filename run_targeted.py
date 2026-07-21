"""
Targeted extraction: 500 chunks per domain.
Covers all 11 domains properly in ~3 hours.
"""
import sys, os, time, sqlite3, psycopg2
sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv('.env')
from world_model.observation_store import extract_observations, save_observations
from world_model.belief_engine import extract_belief_simple, upsert_belief
from world_model.concept_formation import run_formation

PG = os.environ.get('DATABASE_URL','')
def get_pg(): return psycopg2.connect(PG, connect_timeout=10)

DOMAINS = ['biology','medicine','neuroscience','physics','psychology',
           'chemistry','economics','climate','mathematics','computer_systems','ml_ai']
PER_DOMAIN = 500

print(f"=== TARGETED EXTRACTION === {time.strftime('%H:%M:%S')}", flush=True)
print(f"Plan: {PER_DOMAIN} chunks × {len(DOMAINS)} domains = {PER_DOMAIN*len(DOMAINS):,} total", flush=True)

pg = get_pg(); cur = pg.cursor()
cur.execute("SELECT DISTINCT chunk_id FROM observations WHERE chunk_id IS NOT NULL")
processed = set(r[0] for r in cur.fetchall())
pg.close()
print(f"Already processed: {len(processed):,}", flush=True)

total_obs = 0; total_new = 0; start = time.time()

for domain in DOMAINS:
    print(f"\n--- {domain.upper()} ---", flush=True)
    
    sc = sqlite3.connect('knowledge_base/knowledge.db')
    cur2 = sc.cursor()
    cur2.execute("""
        SELECT chunk_id, text, domain, paper_title 
        FROM chunks 
        WHERE domain=? AND LENGTH(text)>100
        AND chunk_id NOT IN ({})
        ORDER BY RANDOM() LIMIT ?
    """.format(','.join(str(x) for x in list(processed)[:5000]) if processed else '0'),
    (domain, PER_DOMAIN))
    chunks = cur2.fetchall()
    sc.close()
    
    print(f"  Processing {len(chunks)} chunks...", flush=True)
    domain_obs = 0
    
    for i, (cid, text, dom, title) in enumerate(chunks):
        if cid in processed:
            continue
        try:
            obs = extract_observations(str(text)[:2000], cid, str(dom or domain), str(title or ''))
            if obs:
                pg = get_pg()
                saved = save_observations(pg, obs, cid, str(dom or domain), str(title or ''))
                pg.close()
                total_obs += saved
                domain_obs += saved
            processed.add(cid)
            total_new += 1
        except:
            processed.add(cid)
        
        if (i+1) % 100 == 0:
            elapsed = time.time()-start
            print(f"  [{domain}] {i+1}/{len(chunks)} | obs={domain_obs} | total={total_obs}", flush=True)
    
    print(f"  {domain}: {domain_obs} observations extracted", flush=True)
    
    # Form concepts for this domain
    try:
        run_formation(domain=domain)
        print(f"  Concepts formed for {domain}", flush=True)
    except Exception as e:
        print(f"  Concept formation error: {e}", flush=True)
    
    # Build beliefs
    try:
        pg = get_pg(); c = pg.cursor()
        c.execute("SELECT subject,predicate,object,conditions,confidence,negated,domain FROM observations WHERE domain=%s ORDER BY id DESC LIMIT 300", (domain,))
        created = 0
        for row in c.fetchall():
            od = {'subject':row[0],'predicate':row[1],'object':row[2],'conditions':row[3],'confidence':row[4],'negated':row[5],'domain':row[6]}
            b = extract_belief_simple(od)
            if not b: continue
            try:
                pg2 = get_pg()
                status, _ = upsert_belief(pg2, b['belief'], b['concept'], b['observation_type'], domain, b.get('confidence',0.75))
                pg2.close()
                if status == 'created': created += 1
            except: pass
        pg.close()
        print(f"  +{created} new beliefs for {domain}", flush=True)
    except Exception as e:
        print(f"  Belief error: {e}", flush=True)

print(f"\n=== DONE === {time.strftime('%H:%M:%S')}", flush=True)
print(f"New chunks: {total_new:,} | New observations: {total_obs:,}", flush=True)

pg = get_pg(); cur = pg.cursor()
cur.execute('SELECT COUNT(*) FROM observations'); print(f"Total obs: {cur.fetchone()[0]:,}", flush=True)
cur.execute('SELECT COUNT(*) FROM beliefs'); print(f"Total beliefs: {cur.fetchone()[0]:,}", flush=True)
cur.execute('SELECT COUNT(*) FROM concept_cells'); print(f"Total concepts: {cur.fetchone()[0]:,}", flush=True)
pg.close()
