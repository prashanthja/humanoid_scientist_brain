"""
Algorithm 1: Concept Identity
Determines if two concept names refer to the same concept.
Outputs: same / parent-child / duplicate / different

Also cleans noise concepts like "The phenomenon", "Module", "new attribution method"
"""
import os, sys, json, psycopg2
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY', ''))
PG_URL = os.environ.get('DATABASE_URL', '')

# Noise patterns — these are language artifacts not real concepts
NOISE_PATTERNS = [
    'the phenomenon', 'new method', 'new approach', 'new attribution',
    'the method', 'this method', 'the technique', 'the approach',
    'module', 'the system', 'the model', 'the framework',
    'our method', 'our approach', 'proposed method', 'inclusion of',
    'distinguishing between', 'memory-level formalization of obs',
    'rustsem\'s formalization', 'formalization of obs'
]

def is_noise(name):
    """Return True if concept name is a language artifact not a real concept."""
    name_low = name.lower().strip()
    if len(name_low) < 4: return True
    if name_low.startswith('the '): return True
    if name_low.startswith('this '): return True
    if name_low.startswith('our '): return True
    for pattern in NOISE_PATTERNS:
        if pattern in name_low: return True
    return False

IDENTITY_PROMPT = """Given two scientific concept names, determine their relationship.

Concept A: {a}
Concept B: {b}

Return ONLY one of these exact words:
- "duplicate" if they refer to exactly the same concept
- "parent-child" if one is a specific instance of the other
- "same-family" if they are closely related aspects of one concept
- "different" if they are genuinely different concepts

Return ONLY the single word. Nothing else."""

def check_identity(name_a, name_b):
    """Check relationship between two concept names."""
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": 
                IDENTITY_PROMPT.format(a=name_a, b=name_b)}],
            max_tokens=10,
            temperature=0.0
        )
        result = r.choices[0].message.content.strip().lower()
        if result not in ['duplicate', 'parent-child', 'same-family', 'different']:
            return 'different'
        return result
    except:
        return 'different'

def clean_noise_concepts(conn):
    """Remove noise concept cells."""
    cur = conn.cursor()
    cur.execute("SELECT id, canonical_name FROM concept_cells")
    rows = cur.fetchall()
    removed = 0
    for cid, name in rows:
        if is_noise(name):
            cur.execute("DELETE FROM concept_cells WHERE id=%s", (cid,))
            print(f"  ❌ removed noise: {name}")
            removed += 1
    conn.commit()
    return removed

def merge_duplicates(conn):
    """Find and merge duplicate/same-family concepts."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, canonical_name, evidence_count, cell_data 
        FROM concept_cells ORDER BY evidence_count DESC
    """)
    rows = cur.fetchall()
    names = [(r[0], r[1]) for r in rows]
    
    merged = 0
    checked = set()
    
    for i, (id_a, name_a) in enumerate(names):
        for j, (id_b, name_b) in enumerate(names):
            if i >= j: continue
            pair = tuple(sorted([id_a, id_b]))
            if pair in checked: continue
            checked.add(pair)
            
            # Quick heuristic first (avoid API calls)
            a_words = set(name_a.lower().split())
            b_words = set(name_b.lower().split())
            overlap = len(a_words & b_words) / max(len(a_words | b_words), 1)
            
            if overlap < 0.3:
                continue  # clearly different, skip API call
            
            relation = check_identity(name_a, name_b)
            
            if relation in ['duplicate', 'same-family']:
                # Merge B into A (keep A, the one with more evidence)
                # Update B's observations to point to A's name
                cur.execute("""
                    UPDATE concept_cells 
                    SET evidence_count = evidence_count + (
                        SELECT evidence_count FROM concept_cells WHERE id=%s
                    )
                    WHERE id=%s
                """, (id_b, id_a))
                cur.execute("DELETE FROM concept_cells WHERE id=%s", (id_b,))
                conn.commit()
                print(f"  🔗 merged: '{name_b}' → '{name_a}' ({relation})")
                merged += 1
                break
            
            elif relation == 'parent-child':
                print(f"  👶 parent-child: '{name_a}' ↔ '{name_b}'")
    
    return merged

def run_concept_identity():
    """Clean noise and merge duplicates."""
    conn = psycopg2.connect(PG_URL)
    
    print("Step 1: Removing noise concepts...")
    removed = clean_noise_concepts(conn)
    print(f"Removed {removed} noise concepts")
    
    print("\nStep 2: Merging duplicates...")
    merged = merge_duplicates(conn)
    print(f"Merged {merged} duplicate concepts")
    
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM concept_cells")
    total = cur.fetchone()[0]
    print(f"\nConcept cells remaining: {total}")
    
    conn.close()

if __name__ == "__main__":
    run_concept_identity()
