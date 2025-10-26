# embedding/encoder.py
"""
Lightweight TextEncoder
-----------------------
Simple tokenizer + frequency-based embedding stub.
No TensorFlow/Keras, no training.  Used only for
tokenization and fixed-size numeric representations
for reasoning modules.
"""

import re
import numpy as np
from collections import Counter


class TextEncoder:
    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 64, max_len: int = 50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.word_index = {}
        self.fitted = False
        # deterministic seed so encodings stay stable
        rng = np.random.default_rng(1337)
        self.random_matrix = rng.normal(0, 1, (vocab_size + 1, embedding_dim)).astype(np.float32)

    # ---------- tokenizer ----------
    def fit_tokenizer(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        words = re.findall(r"\b[a-zA-Z]{2,}\b", " ".join(texts).lower())
        counts = Counter(words).most_common(self.vocab_size)
        self.word_index = {w: i + 1 for i, (w, _) in enumerate(counts)}
        self.fitted = True

    def texts_to_sequences(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for t in texts:
            tokens = re.findall(r"\b[a-zA-Z]{2,}\b", (t or "").lower())
            seq = [self.word_index.get(tok, 0) for tok in tokens[: self.max_len]]
            seqs.append(seq)
        return seqs

    # ---------- embedding ----------
    def encode_texts(self, texts):
        if not self.fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit_tokenizer first.")
        seqs = self.texts_to_sequences(texts)
        if not seqs:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

        embeds = []
        for seq in seqs:
            if not seq:
                embeds.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue
            vecs = self.random_matrix[seq]
            embeds.append(vecs.mean(axis=0))
        return np.vstack(embeds)

    def get_vector(self, text):
        return self.encode_texts(text)[0]
