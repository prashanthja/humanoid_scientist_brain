"""
Layer 2 Complete: Automatic Belief Revision
Beliefs strengthen, weaken, split, or retire
as new evidence arrives. No human intervention.
"""
import os, sys, psycopg2, json
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')
def get_conn(): return psycopg2.connect(PG_URL, connect_timeout=10)

def setup_belief_history_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS belief_history (
            id SERIAL PRIMARY KEY,
            belief_id INTEGER NOT NULL,
            belief_text TEXT,
            confidence_before REAL,
            confidence_after REAL,
            event_type TEXT,
            evidence TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    print("✅ belief_history table ready")

def revise_belief_confidence(conn, belief_id, new_supporting, new_contradicting):
    """
    Revise belief confidence based on new evidence.
    Uses Bayesian-inspired update:
    - More support → higher confidence
    - More contradiction → lower confidence
    - Balance → uncertainty
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT belief_text, confidence, supporting_count, 
               contradicting_count, domain, concept_name
        FROM beliefs WHERE id = %s
    """, (belief_id,))
    row = cur.fetchone()
    if not row:
        return None

    belief_text, old_conf, old_sup, old_con, domain, concept = row
    
    total_sup = old_sup + new_supporting
    total_con = old_con + new_contradicting
    total_evidence = total_sup + total_con

    if total_evidence == 0:
        return None

    # Bayesian confidence update
    # Prior: old_conf
    # Likelihood: supporting vs contradicting ratio
    support_ratio = total_sup / total_evidence
    
    # Weight recent evidence more
    new_conf = round(
        old_conf * 0.4 +           # Prior (40% weight)
        support_ratio * 0.6,        # New evidence (60% weight)
        3
    )
    new_conf = max(0.05, min(0.99, new_conf))

    # Determine event type
    if new_conf > old_conf + 0.05:
        event_type = 'strengthened'
    elif new_conf < old_conf - 0.05:
        event_type = 'weakened'
    else:
        event_type = 'stable'

    # Retire belief if confidence drops below threshold
    if new_conf < 0.15 and total_con > total_sup * 3:
        event_type = 'retired'

    # Update belief
    cur.execute("""
        UPDATE beliefs 
        SET confidence = %s,
            supporting_count = %s,
            contradicting_count = %s
        WHERE id = %s
    """, (new_conf, total_sup, total_con, belief_id))

    # Log to history
    cur.execute("""
        INSERT INTO belief_history
        (belief_id, belief_text, confidence_before, confidence_after, 
         event_type, evidence)
        VALUES (%s,%s,%s,%s,%s,%s)
    """, (
        belief_id, belief_text[:200], old_conf, new_conf,
        event_type,
        f"sup+{new_supporting} con+{new_contradicting}"
    ))
    conn.commit()

    return {
        'belief': belief_text[:100],
        'confidence_before': old_conf,
        'confidence_after': new_conf,
        'event_type': event_type,
        'total_sup': total_sup,
        'total_con': total_con
    }

def check_for_belief_splits(conn, belief_id):
    """
    Detect if a belief should be split into regional/conditional variants.
    Example: "X causes Y" → "X causes Y in condition A" + "X causes Y in condition B"
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT belief_text, supporting_count, contradicting_count, 
               confidence, domain
        FROM beliefs WHERE id = %s
    """, (belief_id,))
    row = cur.fetchone()
    if not row:
        return None

    belief_text, sup, con, conf, domain = row

    # Split candidates: high contradiction with moderate support
    if con >= 3 and sup >= 3 and 0.3 <= conf <= 0.7:
        # Look for conditional variants in observations
        cur.execute("""
            SELECT conditions, COUNT(*) as n
            FROM observations
            WHERE LOWER(subject || ' ' || predicate || ' ' || object) 
                  LIKE LOWER(%s)
            AND conditions IS NOT NULL
            AND LENGTH(conditions) > 5
            GROUP BY conditions
            ORDER BY n DESC LIMIT 3
        """, (f'%{belief_text[:30]}%',))
        conditions = cur.fetchall()

        if len(conditions) >= 2:
            return {
                'should_split': True,
                'belief': belief_text[:100],
                'conditions': [c[0] for c in conditions],
                'reason': f"Contested belief (sup={sup}, con={con}) with {len(conditions)} distinct conditions"
            }

    return {'should_split': False}

def run_belief_revision_cycle(conn, verbose=True):
    """
    Run one complete belief revision cycle.
    Check all contested beliefs and revise.
    """
    cur = conn.cursor()

    # Get contested beliefs
    cur.execute("""
        SELECT id, belief_text, confidence, supporting_count, 
               contradicting_count, domain
        FROM beliefs
        WHERE contradicting_count > 0
        OR supporting_count >= 10
        ORDER BY (supporting_count + contradicting_count) DESC
        LIMIT 50
    """)
    contested = cur.fetchall()

    if verbose:
        print(f"Revising {len(contested)} beliefs...", flush=True)

    revised = weakened = strengthened = retired = 0

    for belief_id, text, conf, sup, con, domain in contested:
        # Simulate new evidence based on current state
        # In production this would come from newly ingested papers
        new_sup = 0
        new_con = 0

        # Check for new supporting observations
        cur.execute("""
            SELECT COUNT(*) FROM observations
            WHERE LOWER(subject || ' ' || predicate || ' ' || object)
                  LIKE LOWER(%s)
            AND negated = FALSE
            AND id > (SELECT COALESCE(MAX(id),0) FROM observations) - 1000
        """, (f'%{text[:25]}%',))
        new_sup = min(cur.fetchone()[0], 5)

        # Check for new contradicting observations
        cur.execute("""
            SELECT COUNT(*) FROM observations
            WHERE LOWER(subject || ' ' || predicate || ' ' || object)
                  LIKE LOWER(%s)
            AND negated = TRUE
            AND id > (SELECT COALESCE(MAX(id),0) FROM observations) - 1000
        """, (f'%{text[:25]}%',))
        new_con = min(cur.fetchone()[0], 3)

        if new_sup == 0 and new_con == 0:
            continue

        result = revise_belief_confidence(conn, belief_id, new_sup, new_con)
        if result:
            revised += 1
            if result['event_type'] == 'strengthened':
                strengthened += 1
            elif result['event_type'] == 'weakened':
                weakened += 1
            elif result['event_type'] == 'retired':
                retired += 1
                if verbose:
                    print(f"  RETIRED: {result['belief'][:60]}", flush=True)

        # Check for splits
        split_check = check_for_belief_splits(conn, belief_id)
        if split_check and split_check.get('should_split'):
            if verbose:
                print(f"  SPLIT candidate: {text[:60]}", flush=True)
                print(f"    Conditions: {split_check['conditions'][:2]}", flush=True)

    if verbose:
        print(f"\nRevision complete:", flush=True)
        print(f"  Revised: {revised}", flush=True)
        print(f"  Strengthened: {strengthened}", flush=True)
        print(f"  Weakened: {weakened}", flush=True)
        print(f"  Retired: {retired}", flush=True)

    return {
        'revised': revised,
        'strengthened': strengthened,
        'weakened': weakened,
        'retired': retired
    }

def get_belief_history(conn, limit=10):
    """Get recent belief revision history."""
    cur = conn.cursor()
    cur.execute("""
        SELECT belief_text, confidence_before, confidence_after,
               event_type, evidence, created_at
        FROM belief_history
        ORDER BY created_at DESC LIMIT %s
    """, (limit,))
    return cur.fetchall()

def format_belief_revision_prompt(conn, query):
    """Format belief revision context for LLM prompt."""
    cur = conn.cursor()
    words = [w for w in query.lower().split() if len(w) > 3]
    if not words:
        return ""
    
    params = [f'%{w}%' for w in words[:3]]
    conditions = ' OR '.join(['LOWER(belief_text) LIKE %s'] * len(params))
    
    cur.execute(f"""
        SELECT bh.belief_text, bh.confidence_before, bh.confidence_after,
               bh.event_type, bh.created_at
        FROM belief_history bh
        WHERE {conditions}
        ORDER BY bh.created_at DESC LIMIT 3
    """, params)
    history = cur.fetchall()
    
    if not history:
        return ""
    
    lines = ["=== BELIEF REVISION HISTORY ==="]
    for text, before, after, event, dt in history:
        lines.append(f"\n{event.upper()}: {text[:80]}")
        lines.append(f"  Confidence: {before:.0%} → {after:.0%}")
    lines.append("\nMention how beliefs have changed if relevant.")
    return '\n'.join(lines)

if __name__ == "__main__":
    conn = get_conn()
    setup_belief_history_table(conn)

    print("Running belief revision cycle...", flush=True)
    results = run_belief_revision_cycle(conn, verbose=True)

    print("\nBelief history (last 5):")
    history = get_belief_history(conn, limit=5)
    for h in history:
        arrow = "↑" if h[2] > h[1] else "↓"
        print(f"  {h[3]:12} {h[1]:.2f} {arrow} {h[2]:.2f} | {h[0][:60]}")

    conn.close()
