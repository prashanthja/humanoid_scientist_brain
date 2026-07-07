"""
Concept Formation Engine — Layer 2 of the World Model

Takes raw observations and forms Concept Cells.
Unlike concept_extractor.py (LLM guesses the concept),
this builds concepts from accumulated evidence.

Tattva forms the concept. LLM only helps structure it.
"""
import os, json, psycopg2, time
from dotenv import load_dotenv; load_dotenv('.env')
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY',''))
PG_URL = os.environ.get('DATABASE_URL','')

FORM_PROMPT = """You are a scientific knowledge structurer.
Given a set of raw observations about a scientific concept,
form a structured Concept Cell. The concept name and content
must come FROM the observations — do not invent.

Observations provided:
{observations}

Return ONLY valid JSON. No markdown.

{{
  "canonical_name": "exact name from observations",
  "aliases": ["other names seen in observations"],
  "domain": "detected from observations",
  "type": "method/mechanism/finding/law/hypothesis",
  "is_a": ["what category this belongs to — from observations"],
  "is_not": ["what this is NOT — inferred from observation context"],
  "mechanism": [
    {{"step": 1, "action": "from observation predicates", "effect": "from observation objects"}}
  ],
  "conditions_for_validity": [
    {{"requires": "from conditions field in observations", "confidence": 0.8}}
  ],
  "fails_when": [
    {{"condition": "from negated observations", "confidence": 0.7}}
  ],
  "predictions": [
    {{
      "if": "input condition from observations",
      "then": [{{"target": "object from observation", "direction": "up/down"}}],
      "confidence": 0.75,
      "testable": true
    }}
  ],
  "goals": [
    {{"type": "unknown_mechanism", "description": "what we still don't know", "priority": "high"}}
  ],
  "relations": [
    {{"type": "predicate from observation", "target": "object from observation", "confidence": 0.8}}
  ],
  "confidence_score": 0.75,
  "evidence_count": 0,
  "formed_from_observations": true
}}
"""

def form_concept_from_observations(subject, obs_rows):
    """Form a Concept Cell from accumulated observations."""
    obs_text = "\n".join([
        f"- {r[0]} {r[1]} {r[2]}"
        + (f" (magnitude: {r[3]})" if r[3] else "")
        + (f" [conditions: {r[4]}]" if r[4] else "")
        + (f" [confidence: {r[5]:.2f}]" if r[5] else "")
        + (f" [NEGATED]" if r[7] else "")
        for r in obs_rows
    ])

    prompt = FORM_PROMPT.format(observations=obs_text)

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        raw = r.choices[0].message.content.strip()
        if raw.startswith('```'):
            lines = raw.split('\n')
            raw = '\n'.join(lines[1:-1] if lines[-1]=='```' else lines[1:])
        cell = json.loads(raw)
        cell['evidence_count'] = len(obs_rows)
        cell['formed_from_observations'] = True
        return cell
    except Exception as e:
        return None

def save_concept_cell(conn, cell, subject):
    """Save formed concept cell."""
    cur = conn.cursor()
    # Check if exists
    cur.execute(
        "SELECT id FROM concept_cells WHERE LOWER(canonical_name)=LOWER(%s)",
        (cell.get('canonical_name', subject),)
    )
    existing = cur.fetchone()

    if existing:
        # Update with new evidence
        cur.execute("""
            UPDATE concept_cells
            SET cell_data=%s, evidence_count=%s,
                confidence_score=%s, updated_at=NOW()
            WHERE id=%s
        """, (
            json.dumps(cell),
            cell.get('evidence_count', 1),
            cell.get('confidence_score', 0.75),
            existing[0]
        ))
        conn.commit()
        return 'updated', existing[0]
    else:
        cur.execute("""
            INSERT INTO concept_cells
            (canonical_name, aliases, domain, type, cell_data,
             evidence_count, confidence_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (
            cell.get('canonical_name', subject)[:200],
            json.dumps(cell.get('aliases',[])),
            cell.get('domain','unknown')[:50],
            cell.get('type','method')[:50],
            json.dumps(cell),
            cell.get('evidence_count', 1),
            cell.get('confidence_score', 0.75)
        ))
        conn.commit()
        return 'created', None

def run_formation(domain=None, min_observations=3):
    """Form concept cells from all subjects with enough observations."""
    import sys; sys.path.insert(0,"."); from world_model.observation_store import get_top_subjects, get_observations_for_subject

    conn = psycopg2.connect(PG_URL)
    subjects = get_top_subjects(conn, domain=domain, min_count=min_observations)
    print(f"Forming concepts from {len(subjects)} subjects with {min_observations}+ observations...")

    created = 0
    updated = 0

    for subject, count, avg_conf in subjects:
        obs_rows = get_observations_for_subject(conn, subject, limit=20)
        if not obs_rows:
            continue

        cell = form_concept_from_observations(subject, obs_rows)
        if not cell:
            continue

        status, _ = save_concept_cell(conn, cell, subject)
        if status == 'created':
            created += 1
        else:
            updated += 1

        print(f"  [{status}] {cell.get('canonical_name',subject)[:50]} ({count} obs)", flush=True)
        time.sleep(0.8)

    conn.close()
    print(f"\nDone. Created: {created} | Updated: {updated}")

if __name__ == "__main__":
    import sys
    domain = sys.argv[1] if len(sys.argv) > 1 else None
    run_formation(domain=domain)
