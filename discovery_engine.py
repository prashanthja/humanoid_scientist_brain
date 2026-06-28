"""
Tattva Autonomous Discovery Engine
Runs automatically when called from /api/chat
No manual code needed
"""
import sqlite3, os, json, subprocess, sys, tempfile
from datetime import datetime
from groq import Groq

DB = 'knowledge_base/knowledge.db'
OUT = 'outputs/discoveries'
os.makedirs(OUT, exist_ok=True)

MECHANISMS = [
    ('oscillation','neuroscience','ml_ai'),
    ('plasticity','neuroscience','ml_ai'),
    ('compression','biology','ml_ai'),
    ('feedback','neuroscience','ml_ai'),
    ('pruning','neuroscience','ml_ai'),
    ('clustering','biology','ml_ai'),
    ('entropy','physics','ml_ai'),
    ('routing','neuroscience','ml_ai'),
]

def get_chunks(domain, mechanism, n=3):
    conn = sqlite3.connect(DB)
    rows = conn.execute(f"""
        SELECT text, paper_title FROM chunks
        WHERE domain=? AND LOWER(text) LIKE ?
        ORDER BY RANDOM() LIMIT ?
    """, (domain, f'%{mechanism}%', n)).fetchall()
    conn.close()
    return rows

def count_bridge(mech, d1, d2):
    conn = sqlite3.connect(DB)
    n = conn.execute("""
        SELECT COUNT(*) FROM chunks
        WHERE LOWER(text) LIKE ?
        AND (LOWER(text) LIKE ? OR LOWER(text) LIKE ?)
    """, (f'%{mech}%', f'%{d1[:4]}%', f'%{d2[:4]}%')).fetchone()[0]
    conn.close()
    return n

def find_best_discovery():
    """Find the cross-domain connection with lowest bridge count = most novel"""
    client = Groq(api_key=os.environ.get('GROQ_API_KEY',''))
    best = None
    best_novelty = 0

    for mech, src, tgt in MECHANISMS:
        src_chunks = get_chunks(src, mech)
        tgt_chunks = get_chunks(tgt, mech)
        if not src_chunks or not tgt_chunks:
            continue
        bridge = count_bridge(mech, src, tgt)
        novelty = 1.0 / (1 + bridge/10)

        if novelty > best_novelty:
            best_novelty = novelty
            best = {
                'mechanism': mech,
                'from': src,
                'to': tgt,
                'bridge_papers': bridge,
                'novelty': round(novelty, 2),
                'src_chunks': src_chunks,
                'tgt_chunks': tgt_chunks,
            }

    if not best:
        return None

    # Generate hypothesis
    src_text = '\n'.join([f"{r[1]}: {r[0][:150]}" for r in best['src_chunks']])
    tgt_text = '\n'.join([f"{r[1]}: {r[0][:150]}" for r in best['tgt_chunks']])

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":f"""
You are a scientific discovery AI.

MECHANISM: {best['mechanism']}
FROM DOMAIN: {best['from']}
{src_text}

TO DOMAIN: {best['to']}
{tgt_text}

BRIDGE PAPERS CONNECTING THESE: {best['bridge_papers']} (fewer = more novel)

Generate:
HYPOTHESIS: One specific testable hypothesis (1 sentence)
PREDICTION: What we expect to find (1 sentence with numbers)
EXPERIMENT: How to test it (2 sentences, specific)
IMPACT: Why this matters (1 sentence)

Be specific. No fluff."""}],
        max_tokens=300,
        temperature=0.3
    )

    best['hypothesis_text'] = r.choices[0].message.content
    return best

def generate_experiment_code(discovery):
    """Generate runnable Python experiment code from a discovery"""
    client = Groq(api_key=os.environ.get('GROQ_API_KEY',''))

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":f"""
Write a complete, runnable Python experiment to test this hypothesis:

{discovery['hypothesis_text']}

Mechanism: {discovery['mechanism']} from {discovery['from']} applied to {discovery['to']}

Requirements:
- Use only: torch, numpy, json (already installed)
- Use MPS device if available: torch.backends.mps.is_available()
- Small scale experiment (runs in under 2 minutes on MacBook)
- Compare baseline vs hypothesis method
- Print clear results: improvement percentage
- Save results to outputs/auto_experiment.json
- End with: print("RESULT:", improvement_pct, "% improvement")

Write ONLY the Python code. No explanation. No markdown backticks."""}],
        max_tokens=800,
        temperature=0.2
    )
    return r.choices[0].message.content

def run_experiment(code):
    """Run generated experiment code and return results"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                     delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=120,
            cwd=os.getcwd()
        )
        output = result.stdout + result.stderr
        # Extract improvement
        for line in output.split('\n'):
            if 'RESULT:' in line:
                return output, line
        return output, None
    except subprocess.TimeoutExpired:
        return "Experiment timed out after 2 minutes", None
    finally:
        os.unlink(tmp)

def run_full_discovery():
    """Full autonomous discovery pipeline"""
    print("[Tattva] Starting autonomous discovery...")

    discovery = find_best_discovery()
    if not discovery:
        return {"error": "No discovery found"}

    print(f"[Tattva] Found: {discovery['mechanism']} {discovery['from']}→{discovery['to']}")
    print(f"[Tattva] Novelty: {discovery['novelty']} | Bridge papers: {discovery['bridge_papers']}")

    # Generate experiment
    print("[Tattva] Generating experiment code...")
    code = generate_experiment_code(discovery)

    # Run experiment
    print("[Tattva] Running experiment...")
    output, result_line = run_experiment(code)

    # Save discovery
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{OUT}/discovery_{ts}.json"
    record = {
        'timestamp': datetime.now().isoformat(),
        'mechanism': discovery['mechanism'],
        'from_domain': discovery['from'],
        'to_domain': discovery['to'],
        'bridge_papers': discovery['bridge_papers'],
        'novelty_score': discovery['novelty'],
        'hypothesis': discovery['hypothesis_text'],
        'experiment_output': output[:2000],
        'result_line': result_line,
    }
    with open(fname, 'w') as f:
        json.dump(record, f, indent=2)

    print(f"[Tattva] Discovery saved: {fname}")
    return record

if __name__ == '__main__':
    result = run_full_discovery()
    print("\n" + "="*50)
    print("DISCOVERY:")
    print(result.get('hypothesis',''))
    print("\nEXPERIMENT RESULT:")
    print(result.get('result_line','No result captured'))
