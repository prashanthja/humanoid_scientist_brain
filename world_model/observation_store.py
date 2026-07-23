"""
Observation Store — Layer 1 of the World Model

Stores raw atomic scientific observations extracted from papers.
These are IMMUTABLE — the ground truth that everything builds on.

Format: Subject → Predicate → Object + conditions + source
Example: "FlashAttention" → "reduces" → "memory_usage"
         conditions: "on transformer models"
         source: chunk_id 12345, paper "FlashAttention 2024"
"""
import os, json, psycopg2
from dotenv import load_dotenv; load_dotenv('.env')
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY',''))
PG_URL = os.environ.get('DATABASE_URL','')

OBSERVE_PROMPT = """You are a scientific observation extractor.
Extract ALL atomic scientific observations from this text.
Each observation = one Subject-Predicate-Object triple.

Rules:
- Subject: the thing being studied (specific, not vague)
- Predicate: the relationship verb (reduces/increases/causes/enables/prevents/contradicts/requires)
- Object: what is affected
- Keep each observation atomic (one fact per observation)
- Extract 3-7 observations per text

Return ONLY valid JSON array. No markdown.

[
  {
    "subject": "FlashAttention",
    "predicate": "reduces",
    "object": "memory_usage",
    "magnitude": "quadratic to linear",
    "conditions": "on transformer models with long sequences",
    "confidence": 0.90,
    "domain": "ml_ai",
    "negated": false
  }
]

Text:
"""

def llm_complete(prompt, max_tokens=800):
    """Call Together AI if available, else Groq."""
    import os
    together_key = os.environ.get('TOGETHER_API_KEY', '')
    if together_key:
        try:
            import time as _time
            from together import Together
            client = Together(api_key=together_key)
            r = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if 'Connection' in str(e) or 'timeout' in str(e).lower():
                import time as _time
                _time.sleep(2)  # Brief pause on connection error
            print(f"Together AI error: {e}, falling back to Groq")
    
    # Groq fallback
    from groq import Groq
    groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY', ''))
    r = groq_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1
    )
    return r.choices[0].message.content.strip()

def setup_observations_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id SERIAL PRIMARY KEY,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            magnitude TEXT,
            conditions TEXT,
            confidence REAL DEFAULT 0.75,
            domain TEXT,
            negated BOOLEAN DEFAULT FALSE,
            chunk_id INTEGER,
            paper_title TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_subject ON observations(subject)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_predicate ON observations(predicate)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_domain ON observations(domain)")
    conn.commit()
    print("✅ observations table ready")

def extract_observations(text, chunk_id, domain, paper_title=""):
    """Extract atomic observations from a paper chunk."""
    try:
        raw = llm_complete(OBSERVE_PROMPT + text[:800], max_tokens=800)
        if raw.startswith('```'):
            lines = raw.split('\n')
            raw = '\n'.join(lines[1:-1] if lines[-1]=='```' else lines[1:])
        obs_list = json.loads(raw)
        if not isinstance(obs_list, list):
            return []
        return obs_list
    except Exception as e:
        return []

# Noise patterns to filter out
OBS_NOISE = [
    'is not mentioned', 'not mentioned', 'is_not_mentioned',
    'no relationship', 'not applicable', 'not relevant',
    'not present', 'not discussed', 'not described',
    'not provided', 'not specified', 'n/a', 'none',
    'in the text', 'in this text', 'in the provided',
    'in the given', 'in this context', 'in_this_context'
]

def is_noise_observation(obs):
    """Return True if observation is an extraction artifact."""
    subj = (obs.get('subject','') or '').lower()
    pred = (obs.get('predicate','') or '').lower()
    obj = (obs.get('object','') or '').lower()
    combined = f"{subj} {pred} {obj}"
    return any(noise in combined for noise in OBS_NOISE)

def save_observations(conn, obs_list, chunk_id, domain, paper_title):
    """Save observations to the store."""
    cur = conn.cursor()
    saved = 0
    for obs in obs_list:
        if not obs.get('subject') or not obs.get('predicate') or not obs.get('object'):
            continue
        if is_noise_observation(obs):
            continue
        try:
            cur.execute("""
                INSERT INTO observations
                (subject, predicate, object, magnitude, conditions,
                 confidence, domain, negated, chunk_id, paper_title)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                obs.get('subject','')[:200],
                obs.get('predicate','')[:100],
                obs.get('object','')[:200],
                obs.get('magnitude','')[:200],
                obs.get('conditions','')[:500],
                obs.get('confidence', 0.75),
                obs.get('domain', domain)[:50],
                obs.get('negated', False),
                chunk_id,
                paper_title[:200]
            ))
            saved += 1
        except Exception as e:
            pass
    conn.commit()
    return saved

def get_observations_for_subject(conn, subject, limit=50):
    """Get all observations about a concept."""
    cur = conn.cursor()
    cur.execute("""
        SELECT subject, predicate, object, magnitude, conditions,
               confidence, domain, negated, paper_title
        FROM observations
        WHERE LOWER(subject) LIKE LOWER(%s)
        ORDER BY confidence DESC LIMIT %s
    """, (f'%{subject}%', limit))
    return cur.fetchall()

def get_top_subjects(conn, domain=None, min_count=3):
    """Get subjects with most observations — these become concepts."""
    cur = conn.cursor()
    if domain:
        cur.execute("""
            SELECT subject, COUNT(*) as cnt, AVG(confidence) as avg_conf
            FROM observations WHERE domain=%s
            GROUP BY subject HAVING COUNT(*) >= %s
            ORDER BY cnt DESC LIMIT 100
        """, (domain, min_count))
    else:
        cur.execute("""
            SELECT subject, COUNT(*) as cnt, AVG(confidence) as avg_conf
            FROM observations
            GROUP BY subject HAVING COUNT(*) >= %s
            ORDER BY cnt DESC LIMIT 100
        """, (min_count,))
    return cur.fetchall()

if __name__ == "__main__":
    import sys
    domain = sys.argv[1] if len(sys.argv) > 1 else 'ml_ai'
    max_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    conn = psycopg2.connect(PG_URL)
    setup_observations_table(conn)

    import psycopg2 as pg2
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_id, text, domain, paper_title FROM chunks
        WHERE domain=%s AND LENGTH(text) > 100
        ORDER BY chunk_id DESC LIMIT %s
    """, (domain, max_chunks))
    chunks = cur.fetchall()
    print(f"Processing {len(chunks)} chunks for observations...")

    import time
    total_obs = 0
    for i, (cid, text, dom, title) in enumerate(chunks):
        obs_list = extract_observations(text, cid, dom, title or '')
        saved = save_observations(conn, obs_list, cid, dom, title or '')
        total_obs += saved
        if i % 5 == 0:
            print(f"  {i}/{len(chunks)} | observations={total_obs}", flush=True)
        time.sleep(0.8)

    print(f"\nDone. Total observations: {total_obs}")

    print("\nTop subjects (candidate concepts):")
    top = get_top_subjects(conn, domain=domain, min_count=2)
    for subj, cnt, conf in top[:20]:
        print(f"  [{cnt} obs, conf={conf:.2f}] {subj}")

    conn.close()
