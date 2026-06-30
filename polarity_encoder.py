"""
TATTVA-1 Polarity Encoder
Detects SUPPORTS / NEUTRAL / CONTRADICTS between claim pairs.
Trained on 73,596 examples (MultiNLI + SciTail), SciBERT base.
Val accuracy: ~87%
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'polarity_encoder')
_tokenizer = None
_model = None
_device = None

LABELS = ['SUPPORTS', 'NEUTRAL', 'CONTRADICTS']

def _load():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_PATH)
    _device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    _model = _model.to(_device)
    _model.eval()

def check_polarity(premise: str, hypothesis: str) -> dict:
    """Returns {'label': str, 'confidence': float, 'probs': dict}"""
    _load()
    enc = _tokenizer(premise, hypothesis, return_tensors='pt', max_length=256, truncation=True).to(_device)
    with torch.no_grad():
        logits = _model(**enc).logits
    probs = torch.softmax(logits, -1)[0].tolist()
    pred_idx = logits.argmax(-1).item()
    return {
        'label': LABELS[pred_idx],
        'confidence': round(probs[pred_idx], 3),
        'probs': {l: round(p, 3) for l, p in zip(LABELS, probs)}
    }

def find_contradictions(chunks: list) -> list:
    """
    Given a list of chunk dicts (each with 'text' key),
    find pairs that contradict each other.
    Returns list of {'chunk_a', 'chunk_b', 'confidence'}
    """
    _load()
    contradictions = []
    texts = [c.get('text', '')[:300] for c in chunks]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if not texts[i] or not texts[j]:
                continue
            result = check_polarity(texts[i], texts[j])
            if result['label'] == 'CONTRADICTS' and result['confidence'] > 0.55:
                contradictions.append({
                    'chunk_a': chunks[i].get('paper_title', ''),
                    'chunk_b': chunks[j].get('paper_title', ''),
                    'text_a': texts[i][:150],
                    'text_b': texts[j][:150],
                    'confidence': result['confidence']
                })
    return contradictions
