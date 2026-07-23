"""
Layer 5 Complete: Experiment Planner from World Model
Uses causal graph + beliefs + mechanisms to design
full executable experiment protocols.
No LLM needed for structure — LLM only writes prose.
"""
import os, sys, psycopg2, json
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

PG_URL = os.environ.get('DATABASE_URL', '')
def get_conn(): return psycopg2.connect(PG_URL, connect_timeout=10)

def setup_experiments_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id SERIAL PRIMARY KEY,
            hypothesis_id INTEGER,
            title TEXT,
            hypothesis_text TEXT,
            independent_variable TEXT,
            dependent_variable TEXT,
            control_condition TEXT,
            dataset TEXT,
            metrics JSONB,
            baselines JSONB,
            statistical_test TEXT,
            sample_size INTEGER,
            expected_effect_size REAL,
            confidence_threshold REAL DEFAULT 0.05,
            estimated_compute TEXT,
            risk_factors JSONB,
            protocol JSONB,
            status TEXT DEFAULT 'designed',
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    print("✅ experiments table ready")

DOMAIN_DATASETS = {
    'ml_ai': [
        'GLUE/SuperGLUE (NLP benchmarks)',
        'ImageNet (vision)',
        'HumanEval (code generation)',
        'MMLU (knowledge)',
        'Split-CIFAR-10 (continual learning)'
    ],
    'biology': [
        'UniProt (protein sequences)',
        'PubChem (molecules)',
        'NCBI Gene Expression Omnibus',
        'Human Cell Atlas'
    ],
    'medicine': [
        'MIMIC-III (clinical records)',
        'UK Biobank (genomics)',
        'PhysioNet (physiological signals)',
        'TCGA (cancer genomics)'
    ],
    'neuroscience': [
        'Allen Brain Atlas',
        'Human Connectome Project',
        'OpenNeuro (fMRI datasets)',
        'CRCNS (neural recordings)'
    ],
    'climate': [
        'ERA5 (reanalysis data)',
        'CMIP6 (climate models)',
        'NOAA global temperature records'
    ],
    'chemistry': [
        'QM9 (molecular properties)',
        'ChEMBL (bioactive molecules)',
        'CSD (crystal structures)'
    ],
    'physics': [
        'Materials Project (material properties)',
        'Open Quantum Materials Database',
        'PhysioNet (physical signals)',
        'NIST datasets (physical constants)'
    ],
    'economics': [
        'World Bank Open Data',
        'FRED (Federal Reserve Economic Data)',
        'IMF datasets'
    ],
    'mathematics': [
        'MATH benchmark',
        'GSM8K (grade school math)',
        'MMLU mathematics subset'
    ],
    'computer_systems': [
        'SPEC CPU benchmarks',
        'MLPerf benchmarks',
        'Linux kernel traces'
    ],
    'psychology': [
        'OSF psychology datasets',
        'UK Biobank (cognitive measures)',
        'ABCD Study (brain development)'
    ]
}

DOMAIN_METRICS = {
    'ml_ai': ['accuracy', 'F1', 'perplexity', 'BLEU', 'backward_transfer', 'forward_transfer'],
    'biology': ['AUC-ROC', 'precision@k', 'RMSE', 'pearson_r'],
    'medicine': ['sensitivity', 'specificity', 'PPV', 'NNT', 'hazard_ratio'],
    'neuroscience': ['firing_rate', 'coherence', 'decoding_accuracy', 'mutual_information'],
    'climate': ['RMSE', 'bias', 'correlation', 'skill_score'],
    'chemistry': ['MAE', 'RMSE', 'validity', 'novelty', 'diversity'],
    'physics': ['MAE', 'RMSE', 'R²', 'energy_error', 'force_error'],
    'economics': ['RMSE', 'R²', 'MAE', 'AIC', 'BIC'],
    'mathematics': ['accuracy', 'exact_match', 'pass@k'],
    'computer_systems': ['throughput', 'latency', 'memory_usage'],
    'psychology': ['Cohen_d', 'AUC-ROC', 'sensitivity', 'specificity']
}

STATISTICAL_TESTS = {
    'continuous': 'paired t-test (n≥30) or Wilcoxon signed-rank (n<30)',
    'categorical': 'chi-squared test or Fisher exact',
    'survival': 'log-rank test + Cox proportional hazards',
    'correlation': 'Pearson r or Spearman rho',
    'default': 'paired t-test, p<0.05, Bonferroni correction for multiple comparisons'
}

def get_relevant_mechanisms(conn, concept_a, concept_c):
    """Get mechanisms connecting concept A to concept C."""
    cur = conn.cursor()
    cur.execute("""
        SELECT root_concept, summary, chain_length, min_confidence
        FROM mechanisms
        WHERE LOWER(root_concept) LIKE LOWER(%s)
        OR LOWER(summary) LIKE LOWER(%s)
        ORDER BY chain_length DESC, min_confidence DESC
        LIMIT 3
    """, (f'%{concept_a[:20]}%', f'%{concept_a[:20]}%'))
    return cur.fetchall()

def get_supporting_beliefs(conn, concept_a, concept_c):
    """Get beliefs supporting the hypothesis."""
    cur = conn.cursor()
    cur.execute("""
        SELECT belief_text, supporting_count, confidence, domain
        FROM beliefs
        WHERE (LOWER(belief_text) LIKE LOWER(%s)
            OR LOWER(belief_text) LIKE LOWER(%s))
        AND supporting_count >= 2
        ORDER BY supporting_count DESC LIMIT 5
    """, (f'%{concept_a[:20]}%', f'%{concept_c[:20]}%'))
    return cur.fetchall()

def estimate_effect_size(supporting_beliefs, mechanisms):
    """Estimate expected effect size from world model evidence."""
    if not supporting_beliefs:
        return 0.2  # Small effect — uncertain

    avg_conf = sum(b[2] for b in supporting_beliefs) / len(supporting_beliefs)
    mech_depth = max((m[2] for m in mechanisms), default=1)

    # Higher confidence + deeper mechanism = larger expected effect
    if avg_conf >= 0.85 and mech_depth >= 3:
        return 0.8  # Large effect
    elif avg_conf >= 0.7 and mech_depth >= 2:
        return 0.5  # Medium effect
    else:
        return 0.3  # Small effect

def calculate_sample_size(effect_size, alpha=0.05, power=0.8):
    """Calculate required sample size for given effect size."""
    # Simplified power analysis
    if effect_size >= 0.8:
        return 26   # Large effect — fewer samples needed
    elif effect_size >= 0.5:
        return 52   # Medium effect
    elif effect_size >= 0.3:
        return 128  # Small effect
    else:
        return 256  # Very small effect

def design_experiment(conn, hypothesis_id=None, concept_a=None,
                      concept_c=None, inferred_rel=None, domain=None):
    """
    Design complete experiment protocol from world model.
    """
    if hypothesis_id:
        cur = conn.cursor()
        cur.execute("""
            SELECT concept_a, inferred_relation, concept_c,
                   confidence, hypothesis_text, concept_b
            FROM hypotheses WHERE id = %s
        """, (hypothesis_id,))
        row = cur.fetchone()
        if row:
            concept_a, inferred_rel, concept_c, conf, hyp_text, concept_b = row
            domain = domain or 'ml_ai'

    if not concept_a or not concept_c:
        return None

    # Get world model evidence
    mechanisms = get_relevant_mechanisms(conn, concept_a, concept_c)
    beliefs = get_supporting_beliefs(conn, concept_a, concept_c)

    # Determine domain
    if not domain and beliefs:
        domain = beliefs[0][3] if beliefs else 'ml_ai'

    # Get domain-specific resources
    datasets = DOMAIN_DATASETS.get(domain, DOMAIN_DATASETS['ml_ai'])
    metrics = DOMAIN_METRICS.get(domain, DOMAIN_METRICS['ml_ai'])

    # Estimate effect size and sample size
    effect_size = estimate_effect_size(beliefs, mechanisms)
    sample_size = calculate_sample_size(effect_size)

    # Build risk factors from contradicting beliefs
    cur = conn.cursor()
    cur.execute("""
        SELECT belief_text, contradicting_count
        FROM beliefs
        WHERE LOWER(belief_text) LIKE LOWER(%s)
        AND contradicting_count > 0
        ORDER BY contradicting_count DESC LIMIT 3
    """, (f'%{concept_a[:20]}%',))
    contested = cur.fetchall()
    risks = [f"Contested belief: {r[0][:80]} ({r[1]} contradicting papers)"
             for r in contested]

    # Build protocol steps
    protocol = [
        f"1. Setup: Establish baseline using {datasets[0]}",
        f"2. Intervention: Apply {concept_a} to test {inferred_rel or 'effect on'} {concept_c}",
        f"3. Control: Run identical setup without {concept_a}",
        f"4. Measure: Track {', '.join(metrics[:3])}",
        f"5. Replicate: n={sample_size} independent runs",
        f"6. Analyze: {STATISTICAL_TESTS['default']}",
        f"7. Validate: Test on held-out {datasets[1] if len(datasets)>1 else datasets[0]}"
    ]

    experiment = {
        'title': f"Testing: {concept_a} → {concept_c}",
        'hypothesis': hyp_text if hypothesis_id else f"{concept_a} {inferred_rel or 'influences'} {concept_c}",
        'independent_variable': concept_a,
        'dependent_variable': concept_c,
        'control_condition': f"Standard approach without {concept_a}",
        'dataset': datasets[0],
        'metrics': metrics[:4],
        'baselines': [
            f"No {concept_a} (negative control)",
            f"Best existing method (positive control)",
            f"Random baseline"
        ],
        'statistical_test': STATISTICAL_TESTS['default'],
        'sample_size': sample_size,
        'expected_effect_size': effect_size,
        'confidence_threshold': 0.05,
        'estimated_compute': '4-8 GPU hours' if domain == 'ml_ai' else '1-2 days compute',
        'risk_factors': risks or ['Insufficient prior evidence'],
        'protocol': protocol,
        'supporting_beliefs': len(beliefs),
        'mechanism_depth': max((m[2] for m in mechanisms), default=0),
        'domain': domain
    }

    return experiment

def save_experiment(conn, experiment, hypothesis_id=None):
    """Save experiment to database."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO experiments
        (hypothesis_id, title, hypothesis_text, independent_variable,
         dependent_variable, control_condition, dataset, metrics,
         baselines, statistical_test, sample_size, expected_effect_size,
         estimated_compute, risk_factors, protocol)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
        RETURNING id
    """, (
        hypothesis_id,
        experiment['title'][:200],
        experiment['hypothesis'][:500],
        experiment['independent_variable'][:200],
        experiment['dependent_variable'][:200],
        experiment['control_condition'][:200],
        experiment['dataset'][:200],
        json.dumps(experiment['metrics']),
        json.dumps(experiment['baselines']),
        experiment['statistical_test'][:200],
        experiment['sample_size'],
        experiment['expected_effect_size'],
        experiment['estimated_compute'][:100],
        json.dumps(experiment['risk_factors']),
        json.dumps(experiment['protocol'])
    ))
    exp_id = cur.fetchone()[0]
    conn.commit()
    return exp_id

def format_experiment_prompt(experiment):
    """Format experiment for LLM — LLM writes prose only."""
    if not experiment:
        return ""
    lines = ["=== TATTVA EXPERIMENT PROTOCOL ==="]
    lines.append(f"Title: {experiment['title']}")
    lines.append(f"Hypothesis: {experiment['hypothesis'][:200]}")
    lines.append(f"Dataset: {experiment['dataset']}")
    lines.append(f"Metrics: {', '.join(experiment['metrics'][:3])}")
    lines.append(f"Sample size: n={experiment['sample_size']} (effect size={experiment['expected_effect_size']:.1f})")
    lines.append(f"Statistical test: {experiment['statistical_test']}")
    lines.append(f"Supporting evidence: {experiment['supporting_beliefs']} beliefs, {experiment['mechanism_depth']}-step mechanism")
    lines.append(f"Estimated compute: {experiment['estimated_compute']}")
    if experiment['risk_factors']:
        lines.append(f"Key risks: {experiment['risk_factors'][0][:100]}")
    lines.append("\nWrite a clear experiment description using this protocol.")
    return '\n'.join(lines)

def get_all_experiments(conn, limit=10):
    cur = conn.cursor()
    cur.execute("""
        SELECT title, hypothesis_text, dataset, sample_size,
               expected_effect_size, status, created_at
        FROM experiments ORDER BY created_at DESC LIMIT %s
    """, (limit,))
    return cur.fetchall()

if __name__ == "__main__":
    conn = get_conn()
    setup_experiments_table(conn)

    # Design experiments for top hypotheses
    cur = conn.cursor()
    cur.execute("""
        SELECT id, concept_a, inferred_relation, concept_c, confidence
        FROM hypotheses WHERE tested=FALSE
        ORDER BY confidence DESC LIMIT 5
    """)
    hypotheses = cur.fetchall()
    print(f"Designing experiments for {len(hypotheses)} hypotheses...")

    for hyp_id, a, rel, c, conf in hypotheses:
        exp = design_experiment(conn, hypothesis_id=hyp_id)
        if exp:
            eid = save_experiment(conn, exp, hyp_id)
            print(f"\n[{conf:.0%}] {a[:30]} → {c[:30]}")
            print(f"  Dataset: {exp['dataset']}")
            print(f"  n={exp['sample_size']} | effect={exp['expected_effect_size']:.1f}")
            print(f"  Mechanism depth: {exp['mechanism_depth']} steps")
            print(f"  Compute: {exp['estimated_compute']}")
            print(f"  Risks: {len(exp['risk_factors'])}")

    # Test with Sleep-LoRA
    print("\n--- Sleep-LoRA experiment ---")
    exp = design_experiment(conn,
        concept_a="sleep consolidation",
        concept_c="catastrophic forgetting",
        inferred_rel="reduces",
        domain="ml_ai"
    )
    if exp:
        print(f"Dataset: {exp['dataset']}")
        print(f"Metrics: {exp['metrics'][:3]}")
        print(f"n={exp['sample_size']} | effect={exp['expected_effect_size']:.1f}")
        print(f"Protocol:")
        for step in exp['protocol']:
            print(f"  {step}")
        save_experiment(conn, exp)

    conn.close()
