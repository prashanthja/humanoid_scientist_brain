"""
Quality filter for world model entities.
Prevents noise from entering beliefs, concepts, hypotheses.
"""

# Scientific entity patterns - what good concepts look like
SCIENTIFIC_ENTITIES = {
    'ml_ai': ['LoRA', 'Transformer', 'Attention', 'BERT', 'GPT', 
               'Mamba', 'Quantization', 'Fine-tuning', 'KV Cache',
               'Speculative Decoding', 'Flash Attention', 'MoE'],
    'biology': ['protein', 'gene', 'DNA', 'RNA', 'cell', 'enzyme',
                'bacteria', 'virus', 'antibody', 'receptor', 'pathway'],
    'medicine': ['drug', 'patient', 'clinical', 'treatment', 'disease',
                 'cancer', 'tumor', 'therapy', 'diagnosis', 'symptom'],
    'neuroscience': ['neuron', 'synapse', 'cortex', 'hippocampus',
                     'dopamine', 'serotonin', 'plasticity', 'memory'],
    'chemistry': ['molecule', 'reaction', 'compound', 'synthesis',
                  'polymer', 'catalyst', 'bond', 'element'],
    'physics': ['quantum', 'particle', 'energy', 'force', 'field',
                'momentum', 'wave', 'photon', 'electron'],
}

# Generic noise words to reject
NOISE_WORDS = {
    'parameter', 'extraction', 'evaluation', 'grade', 'higher',
    'accurate', 'integration', 'acquired', 'proposed', 'model',
    'method', 'approach', 'system', 'process', 'result', 'study',
    'analysis', 'effect', 'level', 'rate', 'type', 'form', 'case',
    'value', 'stage', 'phase', 'step', 'factor', 'sample', 'data',
    'output', 'input', 'feature', 'weight', 'gradient', 'loss',
    'score', 'accuracy', 'error', 'task', 'dataset', 'baseline',
    'experiment', 'observation', 'measurement', 'condition', 'group',
    'subject', 'population', 'exposure', 'treatment', 'intervention',
    'time', 'period', 'duration', 'frequency', 'rate', 'level',
}

def is_noise_concept(name: str) -> bool:
    """Return True if concept is too generic to be useful."""
    name_lower = name.lower().strip()
    if len(name_lower) < 4:
        return True
    words = name_lower.split()
    # All words are noise words
    if all(w in NOISE_WORDS for w in words):
        return True
    # Starts with generic adjective
    if words[0] in {'accurate', 'higher', 'lower', 'proposed', 
                    'new', 'novel', 'improved', 'enhanced', 'better'}:
        return True
    return False

def is_noise_relation(subject: str, obj: str) -> bool:
    """Return True if subject-object pair is too generic."""
    return is_noise_concept(subject) or is_noise_concept(obj)

def score_hypothesis_quality(concept_a: str, concept_c: str, 
                              domain_a: str, domain_c: str,
                              supporting_a: int, supporting_c: int) -> float:
    """
    Score hypothesis quality 0-1.
    High score = worth testing.
    """
    score = 0.5
    
    # Cross-domain bonus
    if domain_a != domain_c:
        score += 0.2
    
    # Evidence bonus
    if supporting_a >= 10 and supporting_c >= 10:
        score += 0.2
    elif supporting_a >= 5 and supporting_c >= 5:
        score += 0.1
    
    # Noise penalty
    if is_noise_concept(concept_a) or is_noise_concept(concept_c):
        score -= 0.4
    
    # Scientific entity bonus
    a_lower = concept_a.lower()
    c_lower = concept_c.lower()
    for domain, entities in SCIENTIFIC_ENTITIES.items():
        if any(e.lower() in a_lower for e in entities):
            score += 0.1
        if any(e.lower() in c_lower for e in entities):
            score += 0.1
    
    return max(0.0, min(1.0, score))

def filter_observations(observations: list) -> list:
    """Filter extracted observations to remove noise."""
    filtered = []
    for obs in observations:
        subj = obs.get('subject', '')
        obj = obs.get('object', '')
        pred = obs.get('predicate', '')
        
        # Skip if subject or object is noise
        if is_noise_concept(subj) or is_noise_concept(obj):
            continue
        
        # Skip if predicate is too vague
        if pred in {'is', 'are', 'was', 'has', 'have', 'do', 'does'}:
            continue
            
        filtered.append(obs)
    
    return filtered
