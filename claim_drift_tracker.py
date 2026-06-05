"""
Bayesian Claim Drift Tracker
-----------------------------
Novel method: Models scientific consensus as a Beta distribution
that updates as new papers arrive. Detects when consensus shifts
using CUSUM algorithm adapted from industrial quality control.

Innovations:
1. Evidence weighting by paper quality + recency
2. CUSUM drift detection for scientific consensus
3. Contradiction velocity metric
4. Temporal decay of older evidence
"""

import sqlite3
import json
import math
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = "knowledge_base/knowledge.db"

# ── Setup ────────────────────────────────────────────────────────────

def init_drift_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claim_beliefs (
            claim_id    TEXT PRIMARY KEY,
            claim_text  TEXT,
            alpha       REAL DEFAULT 1.0,
            beta        REAL DEFAULT 1.0,
            cusum_pos   REAL DEFAULT 0.0,
            cusum_neg   REAL DEFAULT 0.0,
            history_json TEXT DEFAULT '[]',
            created_at  TEXT,
            updated_at  TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claim_evidence (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id    TEXT,
            direction   TEXT,
            weight      REAL,
            paper_title TEXT,
            source      TEXT,
            recorded_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("Drift DB initialized")

# ── Core Algorithm ───────────────────────────────────────────────────

def compute_weight(source: str, recorded_at: str) -> float:
    """
    Innovation 1+2: Weight evidence by quality × recency.
    Landmark papers count more. Recent papers count more.
    """
    # Quality weight
    source_lower = (source or "").lower()
    quality = 1.0
    if any(v in source_lower for v in ["nature","science","cell"]):
        quality = 3.0
    elif any(v in source_lower for v in ["neurips","icml","iclr","acl","emnlp"]):
        quality = 2.5
    elif any(v in source_lower for v in ["arxiv","semantic_scholar"]):
        quality = 1.0
    elif any(v in source_lower for v in ["workshop","preprint"]):
        quality = 0.5

    # Temporal decay — half-life = 1 year
    try:
        dt = datetime.fromisoformat(recorded_at)
        days_old = (datetime.now() - dt).days
        recency = math.exp(-days_old / 365.0)
    except:
        recency = 1.0

    return quality * recency


def update_belief(claim_id: str, claim_text: str,
                  direction: str, weight: float,
                  paper_title: str = "", source: str = "") -> Optional[dict]:
    """
    Update Beta distribution for a claim and run CUSUM drift detection.
    Returns drift alert if consensus is shifting.
    """
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        "SELECT alpha, beta, cusum_pos, cusum_neg, history_json "
        "FROM claim_beliefs WHERE claim_id=?", (claim_id,)
    ).fetchone()

    if row:
        alpha, beta, cusum_pos, cusum_neg = row[0], row[1], row[2], row[3]
        history = json.loads(row[4])
    else:
        alpha, beta, cusum_pos, cusum_neg = 1.0, 1.0, 0.0, 0.0
        history = []

    # Update Beta distribution
    if direction == "supports":
        alpha += weight
    elif direction == "contradicts":
        beta += weight

    # Current belief (mean of Beta distribution)
    belief = alpha / (alpha + beta)
    uncertainty = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
    now = datetime.now().isoformat()
    history.append({"t": now, "belief": round(belief, 4), "alpha": round(alpha,3), "beta": round(beta,3)})
    history = history[-100:]  # keep last 100 updates

    # Innovation 3: CUSUM drift detection
    # Adapted from Page (1954) industrial quality control
    target = 0.75   # expected stable belief for strong evidence
    slack  = 0.02   # sensitivity (lower = more sensitive)

    deviation = belief - target
    cusum_pos = max(0.0, cusum_pos + deviation - slack)
    cusum_neg = max(0.0, cusum_neg + (-deviation) - slack)

    CUSUM_THRESHOLD = 1.5
    drift_alert = None
    if cusum_pos > CUSUM_THRESHOLD:
        drift_alert = {
            "direction": "strengthening",
            "magnitude": round(cusum_pos, 2),
            "belief": round(belief, 3),
            "claim": claim_text
        }
    elif cusum_neg > CUSUM_THRESHOLD:
        drift_alert = {
            "direction": "weakening",
            "magnitude": round(cusum_neg, 2),
            "belief": round(belief, 3),
            "claim": claim_text
        }

    # Save to DB
    conn.execute("""
        INSERT INTO claim_beliefs
            (claim_id, claim_text, alpha, beta, cusum_pos, cusum_neg,
             history_json, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(claim_id) DO UPDATE SET
            alpha=excluded.alpha, beta=excluded.beta,
            cusum_pos=excluded.cusum_pos, cusum_neg=excluded.cusum_neg,
            history_json=excluded.history_json, updated_at=excluded.updated_at
    """, (claim_id, claim_text, alpha, beta, cusum_pos, cusum_neg,
          json.dumps(history), now, now))

    conn.execute("""
        INSERT INTO claim_evidence
            (claim_id, direction, weight, paper_title, source, recorded_at)
        VALUES (?,?,?,?,?,?)
    """, (claim_id, direction, weight, paper_title, source, now))

    conn.commit()
    conn.close()

    return drift_alert


def get_belief(claim_id: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT alpha, beta, cusum_pos, cusum_neg, history_json, claim_text "
        "FROM claim_beliefs WHERE claim_id=?", (claim_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {"belief": 0.5, "uncertainty": 0.25, "history": [], "drift": "unknown"}
    alpha, beta = row[0], row[1]
    cusum_pos, cusum_neg = row[2], row[3]
    history = json.loads(row[4])
    belief = alpha / (alpha + beta)
    uncertainty = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

    # Drift status — matches CUSUM_THRESHOLD = 1.5
    if cusum_pos > 1.5:   drift = "strengthening"
    elif cusum_neg > 1.5: drift = "weakening"
    elif cusum_neg > 0.8: drift = "slight_weakening"
    elif cusum_pos > 0.8: drift = "slight_strengthening"
    else:                 drift = "stable"

    return {
        "claim": row[5],
        "belief": round(belief, 3),
        "uncertainty": round(uncertainty, 4),
        "alpha": round(alpha, 2),
        "beta": round(beta, 2),
        "drift": drift,
        "cusum_pos": round(cusum_pos, 2),
        "cusum_neg": round(cusum_neg, 2),
        "history": history[-10:]
    }


def get_contradiction_velocity(claim_id: str, window_days: int = 30) -> float:
    """
    Innovation 4: How fast are contradictions arriving?
    High velocity = claim under attack right now.
    """
    since = (datetime.now() - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT recorded_at, weight FROM claim_evidence
        WHERE claim_id=? AND direction='contradicts' AND recorded_at > ?
        ORDER BY recorded_at
    """, (claim_id, since)).fetchall()
    conn.close()

    if len(rows) < 2:
        return 0.0

    # Fit linear slope to contradiction weights over time
    n = len(rows)
    x = list(range(n))
    y = [r[1] for r in rows]
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean)**2 for i in range(n))
    slope = num / den if den != 0 else 0.0
    return round(slope, 4)


def get_all_drifting_claims(min_cusum: float = 1.0) -> list:
    """Get all claims showing drift signals."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT claim_id, claim_text, alpha, beta, cusum_pos, cusum_neg
        FROM claim_beliefs
        WHERE cusum_pos > ? OR cusum_neg > ?
        ORDER BY cusum_neg + cusum_pos DESC
    """, (min_cusum, min_cusum)).fetchall()
    conn.close()

    results = []
    for r in rows:
        alpha, beta = r[2], r[3]
        belief = alpha / (alpha + beta)
        direction = "strengthening" if r[4] > r[5] else "weakening"
        results.append({
            "claim_id": r[0],
            "claim": r[1],
            "belief": round(belief, 3),
            "direction": direction,
            "magnitude": round(max(r[4], r[5]), 2)
        })
    return results


def make_claim_id(text: str) -> str:
    """Stable ID from claim text."""
    import hashlib
    return hashlib.md5(text.lower().strip()[:100].encode()).hexdigest()[:12]


# ── Wire into pipeline ───────────────────────────────────────────────

def process_research_result(query: str, result: dict):
    """
    Call this after every run_research result.
    Updates beliefs for all grounded claims.
    """
    init_drift_db()
    alerts = []

    for gc in result.get("top_grounded", []):
        claim_text = gc.get("claim", "")
        if not claim_text or len(claim_text) < 20:
            continue

        v = gc.get("verdict", {})
        direction = "supports" if v.get("verdict") in (
            "supported", "partially_supported"
        ) else "contradicts" if v.get("verdict") == "contradicted" else None

        if not direction:
            continue

        # Use paper source for quality weighting
        paper = gc.get("paper_title", "")
        source = gc.get("source", "arxiv")
        weight = compute_weight(source, datetime.now().isoformat())

        claim_id = make_claim_id(claim_text)
        alert = update_belief(claim_id, claim_text, direction, weight, paper, source)
        if alert:
            alerts.append(alert)

    # Also process contradictions
    for c in result.get("contradictions", []):
        # The contradicting claim gets a "contradicts" signal
        contra_text = c.get("contradicting_claim", "")
        if contra_text and len(contra_text) > 20:
            claim_id = make_claim_id(contra_text)
            weight = compute_weight("arxiv", datetime.now().isoformat())
            alert = update_belief(claim_id, contra_text, "contradicts",
                                  weight, c.get("contradicting_paper",""), "arxiv")
            if alert:
                alerts.append(alert)

    return alerts


if __name__ == "__main__":
    # Test the tracker
    init_drift_db()

    # Simulate 10 supporting papers then 5 contradicting ones
    test_claim = "LoRA reduces fine-tuning cost for large language models"
    cid = make_claim_id(test_claim)

    print("Simulating evidence accumulation...")
    for i in range(10):
        update_belief(cid, test_claim, "supports", 1.0, f"Paper {i}", "arxiv")

    b = get_belief(cid)
    print(f"After 10 supports: belief={b['belief']}, drift={b['drift']}")

    for i in range(8):
        alert = update_belief(cid, test_claim, "contradicts", 1.2, f"Counter {i}", "neurips")
        if alert:
            print(f"DRIFT ALERT: {alert}")

    b = get_belief(cid)
    print(f"After 8 contradictions: belief={b['belief']}, drift={b['drift']}")
    print(f"Contradiction velocity: {get_contradiction_velocity(cid)}")
    print(f"Drifting claims: {get_all_drifting_claims()}")
