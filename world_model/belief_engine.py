"""
Belief Engine — Stage 7

Beliefs sit between Observations and Concepts.
A belief is a claim with evidence attached.

Example:
  Belief: "LoRA reduces GPU memory requirements"
  Supporting: 118 observations
  Contradicting: 9 observations
  Neutral: 22 observations
  Confidence: 0.82

A concept owns many beliefs.
Beliefs are updated as new observations arrive.
This is how scientific understanding evolves.
"""
import os, sys, json, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY', ''))
PG_URL = os.environ.get('DATABASE_URL', '')

BELIEF_EXTRACT_PROMPT = """Given a scientific observation, extract the core BELIEF it implies.
A belief is a general claim that can be supported or contradicted.

Observation: {subject} {predicate} {object}
Conditions: {conditions}
Negated: {negated}

Return ONLY valid JSON:
{{
  "belief": "the general claim as a declarative sentence",
  "concept": "the primary concept this belief is about",
  "observation_type": "supports/contradicts/refines/narrows/neutral",
  "confidence": 0.75
}}"""

def setup_beliefs_table(conn):
    """Create beliefs table."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS beliefs (
            id SERIAL PRIMARY KEY,
            belief_text TEXT NOT NULL,
            concept_name TEXT NOT NULL,
            supporting_count INTEGER DEFAULT 0,
            contradicting_count INTEGER DEFAULT 0,
            refining_count INTEGER DEFAULT 0,
            narrowing_count INTEGER DEFAULT 0,
            neutral_count INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.5,
            domain TEXT,
            prediction_count INTEGER DEFAULT 0,
            prediction_correct INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_concept ON beliefs(concept_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_domain ON beliefs(domain)")
    conn.commit()
    print("✅ beliefs table ready")

def compute_confidence(supporting, contradicting, refining, narrowing, neutral):
    """Compute belief confidence from evidence counts."""
    total = supporting + contradicting + refining + narrowing + neutral
    if total == 0: return 0.5
    
    # Supporting and refining increase confidence
    # Contradicting decreases it
    # Narrowing slightly decreases (scope limitation)
    # Neutral has minimal effect
    
    positive = supporting + (refining * 0.7)
    negative = contradicting + (narrowing * 0.3)
    
    raw = positive / (positive + negative + 0.1)
    
    # Scale to 0.3-0.95 range (never fully certain or impossible)
    confidence = 0.3 + (raw * 0.65)
    
    # More evidence = more confident (up to a point)
    evidence_bonus = min(0.1, total * 0.002)
    
    return min(0.95, confidence + evidence_bonus)

def clean_json(raw):
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                try:
                    import json as _j
                    return _j.loads(part)
                except:
                    continue
    try:
        import json as _j
        return _j.loads(raw)
    except:
        return None

def extract_belief_simple(obs):
    """Extract belief directly from SPO without LLM — saves tokens."""
    subject = obs.get('subject','').strip()
    predicate = obs.get('predicate','').strip()
    obj = obs.get('object','').strip()
    negated = obs.get('negated', False)
    conditions = obs.get('conditions','').strip()
    
    if not subject or not predicate or not obj:
        return None
    
    # Build belief text directly
    if negated:
        belief = f"{subject} does not {predicate} {obj}"
        obs_type = "contradicts"
    else:
        belief = f"{subject} {predicate} {obj}"
        obs_type = "supports"
    
    if conditions:
        belief += f" when {conditions}"
    
    return {
        "belief": belief[:300],
        "concept": subject,
        "observation_type": obs_type,
        "confidence": float(obs.get('confidence', 0.75))
    }

def extract_belief_from_observation(obs):
    """Convert a raw observation into a belief."""
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": BELIEF_EXTRACT_PROMPT.format(
                subject=obs.get("subject",""),
                predicate=obs.get("predicate",""),
                object=obs.get("object",""),
                conditions=obs.get("conditions",""),
                negated=obs.get("negated", False)
            )}],
            max_tokens=150,
            temperature=0.1
        )
        raw = r.choices[0].message.content.strip()
        return clean_json(raw)
    except Exception as e:
        return None
def upsert_belief(conn, belief_text, concept_name, obs_type, domain, confidence):
    """Insert or update a belief based on new observation."""
    cur = conn.cursor()
    
    # Check if belief exists
    cur.execute("""
        SELECT id, supporting_count, contradicting_count, 
               refining_count, narrowing_count, neutral_count
        FROM beliefs 
        WHERE LOWER(belief_text) = LOWER(%s) AND LOWER(concept_name) = LOWER(%s)
    """, (belief_text[:500], concept_name))
    existing = cur.fetchone()
    
    if existing:
        bid, sup, con, ref, nar, neu = existing
        # Update counts
        col_map = {
            'supports': 'supporting_count',
            'contradicts': 'contradicting_count', 
            'refines': 'refining_count',
            'narrows': 'narrowing_count',
            'neutral': 'neutral_count'
        }
        col = col_map.get(obs_type, 'neutral_count')
        new_sup = sup + (1 if obs_type == 'supports' else 0)
        new_con = con + (1 if obs_type == 'contradicts' else 0)
        new_ref = ref + (1 if obs_type == 'refines' else 0)
        new_nar = nar + (1 if obs_type == 'narrows' else 0)
        new_neu = neu + (1 if obs_type == 'neutral' else 0)
        new_conf = compute_confidence(new_sup, new_con, new_ref, new_nar, new_neu)
        
        cur.execute("""
            UPDATE beliefs SET
                supporting_count=%s, contradicting_count=%s,
                refining_count=%s, narrowing_count=%s, neutral_count=%s,
                confidence=%s, updated_at=NOW()
            WHERE id=%s
        """, (new_sup, new_con, new_ref, new_nar, new_neu, new_conf, bid))
        conn.commit()
        return 'updated', bid
    else:
        s = 1 if obs_type == 'supports' else 0
        c = 1 if obs_type == 'contradicts' else 0
        conf = compute_confidence(s, c, 0, 0, 0)
        
        cur.execute("""
            INSERT INTO beliefs 
            (belief_text, concept_name, supporting_count, contradicting_count,
             confidence, domain)
            VALUES (%s,%s,%s,%s,%s,%s) RETURNING id
        """, (belief_text[:500], concept_name, s, c, conf, domain))
        bid = cur.fetchone()[0]
        conn.commit()
        return 'created', bid

def build_beliefs_from_observations(domain=None, limit=100):
    """Process observations and build belief layer."""
    import time
    conn = psycopg2.connect(PG_URL)
    setup_beliefs_table(conn)
    
    cur = conn.cursor()
    if domain:
        cur.execute("""
            SELECT subject, predicate, object, conditions, confidence, negated, domain
            FROM observations WHERE domain=%s LIMIT %s
        """, (domain, limit))
    else:
        cur.execute("""
            SELECT subject, predicate, object, conditions, confidence, negated, domain
            FROM observations LIMIT %s
        """, (limit,))
    
    obs_rows = cur.fetchall()
    print(f"Processing {len(obs_rows)} observations into beliefs...")
    
    created = updated = errors = 0
    
    for i, row in enumerate(obs_rows):
        obs = {
            'subject': row[0], 'predicate': row[1], 'object': row[2],
            'conditions': row[3], 'confidence': row[4],
            'negated': row[5], 'domain': row[6]
        }
        
        belief_data = extract_belief_simple(obs)  # no LLM needed
        if not belief_data or not belief_data.get('belief'):
            errors += 1
            continue
        
        status, _ = upsert_belief(
            conn,
            belief_data['belief'],
            belief_data.get('concept', obs['subject']),
            belief_data.get('observation_type', 'neutral'),
            obs['domain'],
            belief_data.get('confidence', 0.5)
        )
        
        if status == 'created': created += 1
        else: updated += 1
        
        if i % 10 == 0:
            print(f"  {i}/{len(obs_rows)} | created={created} updated={updated}", flush=True)
        
        time.sleep(0.8)
    
    # Show belief summary
    cur.execute("""
        SELECT concept_name, COUNT(*) as belief_count,
               SUM(supporting_count) as total_support,
               SUM(contradicting_count) as total_contra,
               AVG(confidence) as avg_conf
        FROM beliefs
        GROUP BY concept_name
        ORDER BY belief_count DESC LIMIT 15
    """)
    beliefs = cur.fetchall()
    
    print(f"\nDone. Created={created} Updated={updated} Errors={errors}")
    print(f"\nTop concepts by belief count:")
    for concept, bc, sup, con, conf in beliefs:
        print(f"  {concept[:35]:35} beliefs={bc} sup={sup} con={con} conf={conf:.2f}")
    
    conn.close()

if __name__ == "__main__":
    import sys
    domain = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    build_beliefs_from_observations(domain=domain, limit=limit)
