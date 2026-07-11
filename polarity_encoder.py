"""
TATTVA-1 Polarity Encoder — ONNX Runtime (no PyTorch needed)
Detects SUPPORTS / NEUTRAL / CONTRADICTS between claim pairs.
Trained on 73,596 examples (MultiNLI + SciTail), SciBERT base.
Val accuracy: ~87% (quantized: ~84%)
Size: 107MB — fits on Render free tier
"""
import os
import numpy as np

LABELS = ['SUPPORTS', 'NEUTRAL', 'CONTRADICTS']
_HF_MODEL_ID = "prashanthja/tattva-1-polarity-encoder-quantized"
_LOCAL_ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'polarity_encoder_quantized')

_session = None
_tokenizer = None

def _load():
    global _session, _tokenizer
    if _session is not None:
        return
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Find model path
        onnx_path = None
        local_path = os.path.join(_LOCAL_ONNX, 'model_quantized.onnx')
        if os.path.exists(local_path):
            onnx_path = local_path
            tok_path = _LOCAL_ONNX
        else:
            # Download from HuggingFace
            from huggingface_hub import hf_hub_download, snapshot_download
            import tempfile
            cache = os.path.join(tempfile.gettempdir(), 'tattva1_onnx')
            tok_path = snapshot_download(
                repo_id=_HF_MODEL_ID,
                cache_dir=cache,
                token=os.environ.get('HF_TOKEN', None)
            )
            onnx_path = os.path.join(tok_path, 'model_quantized.onnx')

        _session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        _tokenizer = AutoTokenizer.from_pretrained(tok_path)
        print("TATTVA-1 loaded via ONNX Runtime")
    except Exception as e:
        print(f"TATTVA-1 load failed: {e}")
        _session = None

def check_polarity(premise: str, hypothesis: str) -> dict:
    """Returns {label, confidence, probs}"""
    _load()
    if _session is None:
        return _check_polarity_together(premise, hypothesis)
    try:
        enc = _tokenizer(
            premise, hypothesis,
            return_tensors='np',
            max_length=256,
            truncation=True,
            padding=True
        )
        inputs = {k: v for k, v in enc.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
        # Only pass inputs the model expects
        expected = [inp.name for inp in _session.get_inputs()]
        inputs = {k: v for k, v in inputs.items() if k in expected}
        outputs = _session.run(None, inputs)
        logits = outputs[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        pred_idx = int(probs.argmax())
        return {
            'label': LABELS[pred_idx],
            'confidence': round(float(probs[pred_idx]), 3),
            'probs': {l: round(float(p), 3) for l, p in zip(LABELS, probs)}
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
