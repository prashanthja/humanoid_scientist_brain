"""
Automated Concept Cell Extractor
Reads paper chunks from PostgreSQL, extracts Concept Cells using Groq,
stores them back to PostgreSQL.
Grows automatically as new papers arrive.
"""
import os, sys, json, time, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY', ''))
PG_URL = os.environ.get('DATABASE_URL', '')

EXTRACT_PROMPT = """You are a scientific knowledge extractor.
Given a scientific text, extract the PRIMARY concept as a rich Concept Cell.
Be specific. Extract mechanisms with multiple steps. Generate real predictions.
Identify genuine open questions as goals.

Return ONLY valid JSON. No markdown. No explanation.

{
  "canonical_name": "precise short name (2-4 words max)",
  "aliases": ["2-3 alternative names used in literature"],
  "domain": "detect from text: ml_ai/biology/medicine/neuroscience/physics/chemistry/climate/economics/psychology/mathematics/computer_systems",
  "type": "method/mechanism/finding/law/hypothesis",
  "is_a": ["2-3 parent concepts this belongs to"],
  "is_not": ["2-3 things people confuse this with — prevents wrong activation"],
  "mechanism": [
    {"step": 1, "action": "first thing that happens", "effect": "direct result"},
    {"step": 2, "action": "second thing", "effect": "result"},
    {"step": 3, "action": "third thing", "effect": "final outcome"}
  ],
  "conditions_for_validity": [
    {"requires": "specific condition needed", "confidence": 0.85},
    {"requires": "another condition", "confidence": 0.70}
  ],
  "fails_when": [
    {"condition": "specific failure condition", "confidence": 0.75}
  ],
  "applies_to": ["3-4 specific things it applies to"],
  "does_not_apply_to": ["2-3 specific things it does NOT apply to"],
  "predictions": [
    {
      "if": "specific input condition",
      "then": [
        {"target": "measurable output", "direction": "up/down/changes"},
        {"target": "another output", "direction": "up/down/changes"}
      ],
      "confidence": 0.75,
      "testable": true
    }
  ],
  "goals": [
    {"type": "unknown_mechanism", "description": "specific open question about HOW", "priority": "high"},
    {"type": "weak_confidence", "description": "where evidence is thin", "priority": "medium"}
  ],
  "relations": [
    {"type": "causes", "target": "specific effect", "confidence": 0.80},
    {"type": "enables", "target": "what this makes possible", "confidence": 0.75},
    {"type": "analogous_to", "target": "similar concept in another domain", "confidence": 0.60},
    {"type": "contradicts", "target": "conflicting concept if any", "confidence": 0.70}
  ],
  "confidence_score": 0.75,
  "confidence_explanation": "specific reason for this confidence level"
}

Text to extract from:
"""

def setup_concepts_table(conn):
    """Create concepts table if not exists."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS concept_cells (
            id SERIAL PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            aliases TEXT,
            domain TEXT,
            type TEXT,
            cell_data JSONB,
            source_chunk_ids TEXT,
            evidence_count INTEGER DEFAULT 1,
            confidence_score REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_concepts_name ON concept_cells(canonical_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concept_cells(domain)")
    conn.commit()
    print("✅ concept_cells table ready")

def concept_exists(conn, name):
    """Check if concept already extracted."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM concept_cells WHERE LOWER(canonical_name) = LOWER(%s)",
        (name,)
    )
    return cur.fetchone()

def strengthen_concept(conn, existing_id, new_chunk_id, new_confidence):
    """Update existing concept with new evidence."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE concept_cells
        SET evidence_count = evidence_count + 1,
            confidence_score = (confidence_score + %s) / 2,
            updated_at = NOW(),
            source_chunk_ids = source_chunk_ids || ',' || %s
        WHERE id = %s
    """, (new_confidence, str(new_chunk_id), existing_id))
    conn.commit()

def save_concept(conn, cell, chunk_id):
    """Save new concept cell to database."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO concept_cells
        (canonical_name, aliases, domain, type, cell_data,
         source_chunk_ids, confidence_score, evidence_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
    """, (
        cell.get('canonical_name', 'unknown'),
        json.dumps(cell.get('aliases', [])),
        cell.get('domain', 'ml_ai'),
        cell.get('type', 'method'),
        json.dumps(cell),
        str(chunk_id),
        cell.get('confidence_score', 0.5)
    ))
    conn.commit()

def extract_concepts_from_chunk(text, chunk_id, domain):
    """Call Groq to extract concept from chunk text."""
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": EXTRACT_PROMPT + text[:800]
            }],
            max_tokens=600,
            temperature=0.1
        )
        raw = r.choices[0].message.content.strip()
        # Clean markdown if present
        if raw.startswith('```'):
            lines = raw.split('\n')
            raw = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
        cell = json.loads(raw)
        # Use LLM-detected domain if more specific, else keep chunk domain
        if not cell.get('domain') or cell.get('domain') == 'ml_ai':
            cell['domain'] = domain
        return cell
    except Exception as e:
        return None

def run_extraction(batch_size=50, domain_filter=None, max_chunks=500):
    """Main extraction loop."""
    conn = psycopg2.connect(PG_URL)
    setup_concepts_table(conn)

    # Get chunks not yet processed
    cur = conn.cursor()
    if domain_filter:
        cur.execute("""
            SELECT chunk_id, text, domain FROM chunks
            WHERE domain = %s AND LENGTH(text) > 100
            ORDER BY chunk_id DESC LIMIT %s
        """, (domain_filter, max_chunks))
    else:
        cur.execute("""
            SELECT chunk_id, text, domain FROM chunks
            WHERE LENGTH(text) > 100
            ORDER BY chunk_id DESC LIMIT %s
        """, (max_chunks,))

    chunks = cur.fetchall()
    print(f"Processing {len(chunks)} chunks...")

    new_concepts = 0
    strengthened = 0
    errors = 0

    for i, (chunk_id, text, domain) in enumerate(chunks):
        if i % 10 == 0:
            print(f"  {i}/{len(chunks)} | new={new_concepts} strengthened={strengthened}", flush=True)

        cell = extract_concepts_from_chunk(text, chunk_id, domain)
        if not cell or not cell.get('canonical_name'):
            errors += 1
            continue

        name = cell.get('canonical_name', '')
        existing = concept_exists(conn, name)

        if existing:
            strengthen_concept(conn, existing[0], chunk_id, cell.get('confidence_score', 0.5))
            strengthened += 1
        else:
            save_concept(conn, cell, chunk_id)
            new_concepts += 1

        # Rate limit: Groq free tier
        time.sleep(0.8)

    conn.close()
    print(f"\nDone. New: {new_concepts} | Strengthened: {strengthened} | Errors: {errors}")
    return new_concepts, strengthened

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default=None, help='Filter by domain')
    parser.add_argument('--max', type=int, default=100, help='Max chunks to process')
    args = parser.parse_args()
    
    print(f"Extracting concepts from papers (domain={args.domain}, max={args.max})")
    run_extraction(domain_filter=args.domain, max_chunks=args.max)
