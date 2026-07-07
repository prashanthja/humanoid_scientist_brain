"""
Concept Identity Engine v2
Solves 4 problems:
1. Case duplicates: "Non-Neuronal Cells" = "non-neuronal cells"
2. Synonym variants: "FlashAttention" = "Flash Attention" = "Flash-Attention"
3. Parent-child: "Zinc Ion" → "Zinc Ion Binding" → "Hydrogen Bond Stabilization"
4. Noise: "Numerical Engine", "The phenomenon" → remove

Architecture:
  Step 1: Case/punctuation normalization (no LLM needed)
  Step 2: Semantic similarity clustering (embedding-based)
  Step 3: Hierarchy detection (LLM only for ambiguous cases)
  Step 4: Noise detection (heuristic + LLM)
"""
import os, sys, json, psycopg2
import numpy as np
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from sentence_transformers import SentenceTransformer
from groq import Groq

encoder = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.environ.get('GROQ_API_KEY', ''))
PG_URL = os.environ.get('DATABASE_URL', '')

# ── STEP 1: Normalization ────────────────────────────────
import re

def normalize_name(name):
    """Normalize concept name for comparison."""
    n = name.lower().strip()
    n = re.sub(r'[-_/]', ' ', n)
    n = re.sub(r'\s+', ' ', n)
    return n

def exact_duplicates(concepts):
    """Find exact duplicates after normalization."""
    seen = {}
    duplicates = []
    for cid, name, ev in concepts:
        norm = normalize_name(name)
        if norm in seen:
            duplicates.append((cid, name, seen[norm][0], seen[norm][1]))
        else:
            seen[norm] = (cid, name)
    return duplicates

# ── STEP 2: Semantic Clustering ──────────────────────────
def semantic_duplicates(concepts, threshold=0.82):
    """Find semantically similar concepts using embeddings."""
    if len(concepts) < 2:
        return []
    
    names = [c[1] for c in concepts]
    ids = [c[0] for c in concepts]
    
    vecs = encoder.encode(names)
    
    similar_pairs = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            cos_sim = np.dot(vecs[i], vecs[j]) / (
                np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j])
            )
            if cos_sim > threshold:
                similar_pairs.append((ids[i], names[i], ids[j], names[j], float(cos_sim)))
    
    return similar_pairs

# ── STEP 3: Hierarchy Detection ──────────────────────────
HIERARCHY_PROMPT = """Given two scientific concepts, determine their relationship.

Concept A: {a}
Concept B: {b}

Options:
- "duplicate": same concept, different wording
- "a_parent_of_b": A is broader, B is a specific type of A
- "b_parent_of_a": B is broader, A is a specific type of B  
- "associated": related but distinct concepts
- "different": unrelated concepts

Return ONLY the option word. Nothing else."""

def detect_hierarchy(name_a, name_b):
    """Detect if two concepts have a hierarchy relationship."""
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": 
                HIERARCHY_PROMPT.format(a=name_a, b=name_b)}],
            max_tokens=10,
            temperature=0.0
        )
        result = r.choices[0].message.content.strip().lower()
        valid = ['duplicate', 'a_parent_of_b', 'b_parent_of_a', 'associated', 'different']
        return result if result in valid else 'different'
    except:
        return 'different'

# ── STEP 4: Noise Detection ──────────────────────────────
NOISE_WORDS = [
    'the phenomenon', 'new method', 'new approach', 'the method',
    'this method', 'the technique', 'the approach', 'module',
    'the system', 'the model', 'the framework', 'our method',
    'proposed method', 'numerical engine', 'spectral governor',
    'open-access', 'community resource', 'formalization of obs',
    'incorporation of', 'disruption of', 'discrepancies between',
    'decrease in', 'increase in', 'modulating'
]

def is_noise(name):
    """Detect paper-specific phrases that aren't real concepts."""
    n = name.lower().strip()
    if len(n) < 4: return True
    if n.startswith(('the ', 'this ', 'our ', 'a ', 'an ')): return True
    if any(noise in n for noise in NOISE_WORDS): return True
    # Too long = likely a sentence fragment
    if len(n.split()) > 8: return True
    return False

# ── MAIN ENGINE ──────────────────────────────────────────
def run_identity_engine():
    """Full concept identity resolution pipeline."""
    conn = psycopg2.connect(PG_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, canonical_name, evidence_count 
        FROM concept_cells ORDER BY evidence_count DESC
    """)
    concepts = cur.fetchall()
    print(f"Starting with {len(concepts)} concept cells\n")

    # Step 1: Remove noise
    print("STEP 1: Noise removal")
    removed = 0
    for cid, name, ev in concepts:
        if is_noise(name):
            cur.execute("DELETE FROM concept_cells WHERE id=%s", (cid,))
            print(f"  ❌ noise: '{name}'")
            removed += 1
    conn.commit()
    print(f"Removed {removed} noise concepts\n")

    # Reload after noise removal
    cur.execute("SELECT id, canonical_name, evidence_count FROM concept_cells ORDER BY evidence_count DESC")
    concepts = cur.fetchall()

    # Step 2: Exact duplicates (no LLM needed)
    print("STEP 2: Exact duplicate merging")
    dupes = exact_duplicates(concepts)
    merged_exact = 0
    for keep_id, keep_name, remove_id, remove_name in [(d[2],d[3],d[0],d[1]) for d in dupes]:
        print(f"  🔗 exact: '{remove_name}' → '{keep_name}'")
        cur.execute("UPDATE concept_cells SET evidence_count=evidence_count+1 WHERE id=%s", (keep_id,))
        cur.execute("DELETE FROM concept_cells WHERE id=%s", (remove_id,))
        merged_exact += 1
    conn.commit()
    print(f"Merged {merged_exact} exact duplicates\n")

    # Step 3: Semantic similarity
    print("STEP 3: Semantic similarity clustering")
    cur.execute("SELECT id, canonical_name, evidence_count FROM concept_cells")
    concepts = cur.fetchall()
    similar = semantic_duplicates(concepts, threshold=0.92)
    
    merged_semantic = 0
    processed = set()
    for id_a, name_a, id_b, name_b, sim in similar:
        if id_a in processed or id_b in processed:
            continue
        relation = detect_hierarchy(name_a, name_b)
        
        if relation == 'duplicate':
            # Keep the one with more evidence
            cur.execute("SELECT evidence_count FROM concept_cells WHERE id=%s", (id_a,))
            ev_a = cur.fetchone()[0]
            cur.execute("SELECT evidence_count FROM concept_cells WHERE id=%s", (id_b,))
            ev_b = cur.fetchone()[0]
            keep, remove = (id_a, id_b) if ev_a >= ev_b else (id_b, id_a)
            keep_name = name_a if keep == id_a else name_b
            remove_name = name_b if keep == id_a else name_a
            cur.execute("UPDATE concept_cells SET evidence_count=evidence_count+1 WHERE id=%s", (keep,))
            cur.execute("DELETE FROM concept_cells WHERE id=%s", (remove,))
            conn.commit()
            print(f"  🔗 merged: '{remove_name}' → '{keep_name}' (sim={sim:.2f})")
            processed.add(remove)
            merged_semantic += 1
        
        elif relation in ['a_parent_of_b', 'b_parent_of_a']:
            parent = name_a if relation == 'a_parent_of_b' else name_b
            child = name_b if relation == 'a_parent_of_b' else name_a
            parent_id = id_a if relation == 'a_parent_of_b' else id_b
            child_id = id_b if relation == 'a_parent_of_b' else id_a
            # Add parent_id to child's cell_data
            cur.execute("SELECT cell_data FROM concept_cells WHERE id=%s", (child_id,))
            row = cur.fetchone()
            if row:
                cell = row[0] or {}
                cell['parent_concept'] = parent
                cur.execute("UPDATE concept_cells SET cell_data=%s WHERE id=%s", 
                           (json.dumps(cell), child_id))
                conn.commit()
            print(f"  👶 hierarchy: '{parent}' → '{child}'")

    print(f"Merged {merged_semantic} semantic duplicates\n")

    # Final state
    cur.execute("SELECT COUNT(*) FROM concept_cells")
    total = cur.fetchone()[0]
    cur.execute("""
        SELECT lifecycle_state, domain, canonical_name, evidence_count
        FROM concept_cells ORDER BY evidence_count DESC
    """)
    final = cur.fetchall()
    
    print(f"FINAL: {total} concept cells")
    for state, dom, name, ev in final:
        print(f"  [{state:12}] [{dom:15}] {name[:40]} (ev={ev})")
    
    conn.close()

if __name__ == "__main__":
    run_identity_engine()
