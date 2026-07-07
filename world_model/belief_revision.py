"""
Algorithm 3: Belief Revision
A concept changes qualitatively, not just quantitatively.
A new paper can: support / contradict / refine / narrow scope

This is richer than just incrementing evidence_count.
"""
import os, sys, psycopg2, json, time
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY', ''))
PG_URL = os.environ.get('DATABASE_URL', '')

REVISION_PROMPT = """Given a scientific concept and a new observation about it,
determine how the observation affects the concept's belief.

Concept: {concept_name}
Current confidence: {confidence}
Current mechanism: {mechanism}

New observation: {subject} {predicate} {object}
Conditions: {conditions}
Negated: {negated}

Return ONLY valid JSON:
{{
  "revision_type": "supports/contradicts/refines/narrows_scope/unrelated",
  "confidence_delta": 0.05,
  "reason": "brief explanation",
  "scope_update": "new scope condition if narrowing, else null",
  "mechanism_update": "updated mechanism step if refining, else null"
}}"""

def revise_belief(conn, concept_id, observation):
    """Apply a new observation to update a concept's belief."""
    cur = conn.cursor()
    cur.execute("""
        SELECT canonical_name, confidence_score, cell_data, contradiction_count
        FROM concept_cells WHERE id=%s
    """, (concept_id,))
    row = cur.fetchone()
    if not row: return None
    
    name, conf, cell_data, contra_count = row
    mechanism = cell_data.get('mechanism', [])[:2] if cell_data else []
    
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": REVISION_PROMPT.format(
                concept_name=name,
                confidence=conf,
                mechanism=json.dumps(mechanism),
                subject=observation.get('subject',''),
                predicate=observation.get('predicate',''),
                object=observation.get('object',''),
                conditions=observation.get('conditions',''),
                negated=observation.get('negated', False)
            )}],
            max_tokens=200,
            temperature=0.1
        )
        raw = r.choices[0].message.content.strip()
        if raw.startswith('```'):
            lines = raw.split('\n')
            raw = '\n'.join(lines[1:-1] if lines[-1]=='```' else lines[1:])
        revision = json.loads(raw)
        
        # Apply revision
        revision_type = revision.get('revision_type', 'unrelated')
        delta = revision.get('confidence_delta', 0.0)
        
        new_conf = max(0.1, min(0.99, conf + delta))
        new_contra = contra_count + (1 if revision_type == 'contradicts' else 0)
        
        # Update concept
        cur.execute("""
            UPDATE concept_cells 
            SET confidence_score=%s,
                contradiction_count=%s,
                evidence_count=evidence_count+1,
                updated_at=NOW()
            WHERE id=%s
        """, (new_conf, new_contra, concept_id))
        conn.commit()
        
        return {
            'concept': name,
            'revision_type': revision_type,
            'old_confidence': conf,
            'new_confidence': new_conf,
            'reason': revision.get('reason','')
        }
    except Exception as e:
        return None

def run_belief_revision_for_new_observations(conn, limit=20):
    """Find recent observations and apply belief revision to matching concepts."""
    cur = conn.cursor()
    # Get recent observations
    cur.execute("""
        SELECT id, subject, predicate, object, conditions, confidence, negated
        FROM observations
        ORDER BY created_at DESC LIMIT %s
    """, (limit,))
    observations = cur.fetchall()
    
    revised = 0
    for obs_row in observations:
        obs_id, subject, predicate, obj, conditions, conf, negated = obs_row
        observation = {
            'subject': subject, 'predicate': predicate,
            'object': obj, 'conditions': conditions, 'negated': negated
        }
        
        # Find matching concept
        cur.execute("""
            SELECT id FROM concept_cells
            WHERE LOWER(canonical_name) LIKE LOWER(%s)
            LIMIT 1
        """, (f'%{subject[:30]}%',))
        concept = cur.fetchone()
        if not concept: continue
        
        result = revise_belief(conn, concept[0], observation)
        if result:
            print(f"  [{result['revision_type']}] {result['concept'][:40]}: "
                  f"{result['old_confidence']:.2f} → {result['new_confidence']:.2f}")
            revised += 1
        time.sleep(0.5)
    
    return revised

if __name__ == "__main__":
    conn = psycopg2.connect(PG_URL)
    print("Running belief revision on recent observations...")
    revised = run_belief_revision_for_new_observations(conn, limit=20)
    print(f"\nRevised {revised} concept beliefs")
    conn.close()
