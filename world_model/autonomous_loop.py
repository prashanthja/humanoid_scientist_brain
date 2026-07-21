"""
The Autonomous Research Loop.
Scheduler wakes up → finds weakest beliefs → generates hypotheses →
designs experiments → simulates → updates beliefs → repeat.
"""
import os, sys, psycopg2, json, time
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')

def get_conn():
    return psycopg2.connect(PG_URL, connect_timeout=10)

def setup_research_log_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS research_log (
            id SERIAL PRIMARY KEY,
            cycle INTEGER DEFAULT 1,
            hypothesis_id INTEGER,
            hypothesis_text TEXT,
            experiment_design TEXT,
            simulation_result TEXT,
            simulation_confidence REAL,
            belief_update TEXT,
            outcome TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hypothesis_memory (
            id SERIAL PRIMARY KEY,
            concept_a TEXT,
            concept_b TEXT,
            hypothesis_text TEXT,
            status TEXT DEFAULT 'untested',
            confidence_before REAL,
            confidence_after REAL,
            test_result TEXT,
            failure_reason TEXT,
            retry_condition TEXT,
            tested_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    print("✅ research_log and hypothesis_memory tables ready")

def find_weakest_beliefs(conn, limit=5):
    """Find beliefs with low confidence or high contradiction — need testing."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, belief_text, confidence, supporting_count,
               contradicting_count, concept_name, domain
        FROM beliefs
        WHERE supporting_count < 5
        AND LENGTH(belief_text) > 20
        AND belief_text NOT LIKE %(noise)s
        AND concept_name NOT IN ('Module','time','age')
        ORDER BY contradicting_count DESC, confidence ASC
        LIMIT %(lim)s
    """, {'noise': '%does not does not%', 'lim': limit})
    return cur.fetchall()

def find_untested_hypotheses(conn, limit=5):
    """Get highest-confidence untested hypotheses."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, concept_a, inferred_relation, concept_c,
               confidence, hypothesis_text
        FROM hypotheses
        WHERE tested = FALSE
        AND hypothesis_text IS NOT NULL
        ORDER BY confidence DESC
        LIMIT %s
    """, (limit,))
    return cur.fetchall()

def estimate_information_gain(hypothesis, beliefs):
    """
    Estimate expected information gain from testing a hypothesis.
    Higher = more worth testing.
    """
    # Factors that increase information gain:
    # 1. Low current confidence (0.5 = maximum uncertainty)
    # 2. Contradicting evidence exists (contested = worth resolving)
    # 3. Concept appears in many beliefs (high-impact concept)
    conf = hypothesis[4]  # confidence
    uncertainty = 1.0 - abs(conf - 0.5) * 2  # max at 0.5
    
    # Check if concept appears in contested beliefs
    concept_a = hypothesis[1].lower()
    concept_c = hypothesis[3].lower()
    contested = sum(1 for b in beliefs if
                   (concept_a in b[1].lower() or concept_c in b[1].lower())
                   and b[4] > 0)
    
    info_gain = uncertainty * 0.6 + min(contested * 0.1, 0.4)
    return round(info_gain, 3)

def design_experiment(hypothesis_text, concept_a, concept_c):
    """Design experiment to test hypothesis — no LLM needed for structure."""
    return {
        'hypothesis': hypothesis_text,
        'type': 'observational',
        'components': {
            'independent_variable': concept_a,
            'dependent_variable': concept_c,
            'control': f"baseline without {concept_a}",
            'metric': f"change in {concept_c}",
            'n_samples': 10,
            'statistical_test': 'paired t-test',
            'significance_level': 0.05
        },
        'expected_outcome': f"If hypothesis correct: {concept_a} change → {concept_c} change",
        'failure_criteria': f"No significant change in {concept_c} when {concept_a} varies"
    }

def run_simulation(conn, hypothesis_id, concept_a, concept_c, inferred_rel):
    """
    Simulate experiment using world model.
    Returns predicted outcome and confidence.
    """
    from world_model.simulation_engine import simulate
    
    query = f"Does {concept_a} {inferred_rel} {concept_c}?"
    sim = simulate(conn, query)
    
    if not sim:
        return {
            'predicted_outcome': 'insufficient_data',
            'confidence': 0.3,
            'supporting': 0,
            'contradicting': 0,
            'recommendation': 'INSUFFICIENT EVIDENCE'
        }
    return sim

def update_belief_from_simulation(conn, hypothesis_id, sim_result):
    """Update belief confidence based on simulation result."""
    if not sim_result:
        return None
    
    conf = sim_result.get('confidence', 0.5)
    
    if conf >= 0.75:
        outcome = 'SUPPORTED'
        update = f"Simulation supports hypothesis (conf={conf:.0%}). Belief strengthened."
    elif conf >= 0.45:
        outcome = 'INCONCLUSIVE'
        update = f"Simulation inconclusive (conf={conf:.0%}). Need more evidence."
    else:
        outcome = 'REJECTED'
        update = f"Simulation rejects hypothesis (conf={conf:.0%}). Belief weakened."

    # Mark hypothesis as tested
    cur = conn.cursor()
    cur.execute("""
        UPDATE hypotheses SET tested = TRUE WHERE id = %s
    """, (hypothesis_id,))
    conn.commit()

    return {'outcome': outcome, 'update': update, 'confidence': conf}

def record_to_memory(conn, hyp_id, concept_a, concept_c, hyp_text,
                     conf_before, result):
    """Remember what we tested so we don't repeat failures."""
    cur = conn.cursor()
    outcome = result.get('outcome', 'UNKNOWN')
    conf_after = result.get('confidence', conf_before)
    
    failure_reason = None
    retry_condition = None
    if outcome == 'REJECTED':
        failure_reason = f"Simulation confidence only {conf_after:.0%}"
        retry_condition = f"Retry if new papers add evidence for {concept_a} → {concept_c}"

    cur.execute("""
        INSERT INTO hypothesis_memory
        (concept_a, concept_b, hypothesis_text, status,
         confidence_before, confidence_after, test_result,
         failure_reason, retry_condition, tested_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
    """, (concept_a, concept_c, hyp_text, outcome.lower(),
          conf_before, conf_after, outcome,
          failure_reason, retry_condition))
    conn.commit()

def run_research_cycle(conn, cycle_num=1, verbose=True):
    """
    Run one complete autonomous research cycle.
    Returns summary of what was done.
    """
    if verbose:
        print(f"\n{'='*50}", flush=True)
        print(f"RESEARCH CYCLE {cycle_num}", flush=True)
        print(f"{'='*50}", flush=True)

    # Step 1: Find weakest beliefs
    weak = find_weakest_beliefs(conn, limit=10)
    if verbose:
        print(f"Step 1: Found {len(weak)} weak beliefs", flush=True)

    # Step 2: Find untested hypotheses
    hypotheses = find_untested_hypotheses(conn, limit=10)
    if verbose:
        print(f"Step 2: Found {len(hypotheses)} untested hypotheses", flush=True)

    if not hypotheses:
        if verbose:
            print("No hypotheses to test — generating new ones...", flush=True)
        from world_model.hypothesis_generator import generate_hypotheses, save_hypotheses
        new_hyps = generate_hypotheses(conn, min_confidence=0.5, limit=20)
        if new_hyps:
            save_hypotheses(conn, new_hyps)
            hypotheses = find_untested_hypotheses(conn, limit=10)

    # Step 3: Estimate information gain and select top 3
    scored = []
    for hyp in hypotheses:
        ig = estimate_information_gain(hyp, weak)
        scored.append((ig, hyp))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_hypotheses = [h for _, h in scored[:3]]

    if verbose:
        print(f"Step 3: Selected top {len(top_hypotheses)} hypotheses by information gain", flush=True)

    results = []
    for hyp in top_hypotheses:
        hyp_id, concept_a, rel, concept_c, conf, hyp_text = hyp

        if verbose:
            print(f"\nTesting: {concept_a} --[{rel}]--> {concept_c} (conf={conf:.0%})", flush=True)

        # Step 4: Design experiment
        experiment = design_experiment(hyp_text or '', concept_a, concept_c)
        if verbose:
            print(f"  Experiment designed: {experiment['type']}", flush=True)

        # Step 5: Run simulation
        sim = run_simulation(conn, hyp_id, concept_a, concept_c, rel)
        if verbose:
            print(f"  Simulation: {sim.get('recommendation','unknown')} (conf={sim.get('confidence',0):.0%})", flush=True)

        # Step 6: Update beliefs
        belief_update = update_belief_from_simulation(conn, hyp_id, sim)
        if verbose and belief_update:
            print(f"  Belief update: {belief_update['outcome']}", flush=True)

        # Step 7: Record to memory
        if belief_update:
            record_to_memory(conn, hyp_id, concept_a, concept_c,
                           hyp_text or '', conf, belief_update)

        # Step 8: Log to research_log
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO research_log
            (cycle, hypothesis_id, hypothesis_text, experiment_design,
             simulation_result, simulation_confidence, belief_update, outcome)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            cycle_num, hyp_id, hyp_text,
            json.dumps(experiment),
            json.dumps(sim) if sim else None,
            sim.get('confidence', 0) if sim else 0,
            belief_update.get('update','') if belief_update else '',
            belief_update.get('outcome','UNKNOWN') if belief_update else 'UNKNOWN'
        ))
        conn.commit()

        results.append({
            'hypothesis': f"{concept_a} {rel} {concept_c}",
            'outcome': belief_update.get('outcome','UNKNOWN') if belief_update else 'UNKNOWN',
            'confidence': sim.get('confidence', 0) if sim else 0
        })

    summary = {
        'cycle': cycle_num,
        'hypotheses_tested': len(results),
        'supported': sum(1 for r in results if r['outcome'] == 'SUPPORTED'),
        'rejected': sum(1 for r in results if r['outcome'] == 'REJECTED'),
        'inconclusive': sum(1 for r in results if r['outcome'] == 'INCONCLUSIVE'),
        'results': results
    }

    if verbose:
        print(f"\nCycle {cycle_num} complete:", flush=True)
        print(f"  Tested: {summary['hypotheses_tested']}", flush=True)
        print(f"  Supported: {summary['supported']}", flush=True)
        print(f"  Rejected: {summary['rejected']}", flush=True)
        print(f"  Inconclusive: {summary['inconclusive']}", flush=True)

    return summary

def get_research_log(conn, limit=10):
    """Get recent research cycle results."""
    cur = conn.cursor()
    cur.execute("""
        SELECT cycle, hypothesis_text, outcome, simulation_confidence, created_at
        FROM research_log
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))
    return cur.fetchall()

def get_hypothesis_memory(conn, limit=10):
    """Get memory of tested hypotheses."""
    cur = conn.cursor()
    cur.execute("""
        SELECT concept_a, concept_b, status, confidence_after,
               failure_reason, retry_condition, tested_at
        FROM hypothesis_memory
        ORDER BY tested_at DESC
        LIMIT %s
    """, (limit,))
    return cur.fetchall()

if __name__ == "__main__":
    conn = get_conn()
    setup_research_log_table(conn)
    
    print("Running autonomous research cycle...", flush=True)
    summary = run_research_cycle(conn, cycle_num=1)
    
    print(f"\nResearch log:", flush=True)
    log = get_research_log(conn, limit=5)
    for r in log:
        print(f"  Cycle {r[0]}: {(r[1] or '')[:60]} → {r[2]} ({r[3]:.0%})")
    
    print(f"\nHypothesis memory:", flush=True)
    mem = get_hypothesis_memory(conn, limit=5)
    for m in mem:
        print(f"  {m[0][:20]} → {m[1][:20]}: {m[2]} (conf={m[3]:.0%})")
        if m[4]:
            print(f"    Failure: {m[4][:60]}")
        if m[5]:
            print(f"    Retry when: {m[5][:60]}")
    
    conn.close()
