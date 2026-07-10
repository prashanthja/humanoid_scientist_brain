"""
TATTVA-1 Polarity Encoder — ONNX Quantized Version
Detects SUPPORTS / NEUTRAL / CONTRADICTS between claim pairs.
Trained on 73,596 examples (MultiNLI + SciTail), SciBERT base.
Val accuracy: ~87% (quantized: ~84%)
Size: 107MB (was 420MB) — fits on Render free tier
"""
import os
import torch

_LOCAL_ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'polarity_encoder_quantized')
_LOCAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'polarity_encoder')
_HF_MODEL_ID = "prashanthja/tattva-1-polarity-encoder-quantized"

LABELS = ['SUPPORTS', 'NEUTRAL', 'CONTRADICTS']

_model = None
_tokenizer = None

def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        # Try local quantized first
        if os.path.exists(os.path.join(_LOCAL_ONNX, 'model_quantized.onnx')):
            _model = ORTModelForSequenceClassification.from_pretrained(
                _LOCAL_ONNX, file_name='model_quantized.onnx'
            )
            _tokenizer = AutoTokenizer.from_pretrained(_LOCAL_ONNX)
            print("TATTVA-1 loaded from local quantized ONNX")
        else:
            # Download from HuggingFace
            from huggingface_hub import snapshot_download
            import tempfile
            cache = os.path.join(tempfile.gettempdir(), 'tattva1_onnx')
            path = snapshot_download(
                repo_id=_HF_MODEL_ID,
                cache_dir=cache,
                token=os.environ.get('HF_TOKEN', None)
            )
            _model = ORTModelForSequenceClassification.from_pretrained(
                path, file_name='model_quantized.onnx'
            )
            _tokenizer = AutoTokenizer.from_pretrained(path)
            print("TATTVA-1 loaded from HuggingFace")
    except Exception as e:
        print(f"TATTVA-1 load failed: {e}")
        _model = None

def check_polarity(premise: str, hypothesis: str) -> dict:
    """Returns {label, confidence, probs}"""
    _load()
    if _model is None:
        return _check_polarity_together(premise, hypothesis)
    try:
        enc = _tokenizer(premise, hypothesis, return_tensors='pt', max_length=256, truncation=True)
        out = _model(**enc)
        probs = torch.softmax(torch.tensor(out.logits), -1)[0].tolist()
        pred_idx = int(torch.tensor(probs).argmax())
        return {
            'label': LABELS[pred_idx],
            'confidence': round(probs[pred_idx], 3),
            'probs': {l: round(p, 3) for l, p in zip(LABELS, probs)}
        }
    except Exception as e:
        return _check_polarity_together(premise, hypothesis)

def _check_polarity_together(premise: str, hypothesis: str) -> dict:
    """Fallback: use Together AI when TATTVA-1 not available."""
    try:
        from together import Together
        client = Together(api_key=os.environ.get('TOGETHER_API_KEY',''))
        prompt = f"""Classify relationship between two scientific statements.
A: {premise[:200]}
B: {hypothesis[:200]}
Answer with exactly one word: SUPPORTS, NEUTRAL, or CONTRADICTS"""
        r = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=10, temperature=0.0
        )
        label = r.choices[0].message.content.strip().upper()
        if label not in LABELS: label = 'NEUTRAL'
        return {'label': label, 'confidence': 0.75, 'probs': {}}
    except:
        return {'label': 'NEUTRAL', 'confidence': 0.5, 'probs': {}}

def find_contradictions(chunks: list) -> list:
    """Find contradicting pairs in a list of chunks."""
    _load()
    contradictions = []
    texts = [c.get('text','')[:300] for c in chunks]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if not texts[i] or not texts[j]: continue
            result = check_polarity(texts[i], texts[j])
            if result['label'] == 'CONTRADICTS' and result['confidence'] > 0.55:
                contradictions.append({
                    'chunk_a': chunks[i].get('paper_title',''),
                    'chunk_b': chunks[j].get('paper_title',''),
                    'text_a': texts[i][:150],
                    'text_b': texts[j][:150],
                    'confidence': result['confidence']
                })
    return contradictions
