# embedding/encoder.py
"""
Deterministic Semantic-ish Text Encoder
---------------------------------------
- TF-IDF weighted word vectors
- Position-aware hashing
- L2 normalized output
- No DL, no Keras, no Torch
"""

import re
import numpy as np
from collections import Counter, defaultdict
import math


class TextEncoder:
    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 256,
        max_len: int = 256,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        self.word_index = {}
        self.idf = {}
        self.fitted = False

        rng = np.random.default_rng(1337)
        self.word_matrix = rng.normal(0, 1, (vocab_size + 1, embedding_dim)).astype(np.float32)

    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------
    def _tokenize(self, text: str):
        return re.findall(r"\b[a-zA-Z]{2,}\b", (text or "").lower())

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    def fit_tokenizer(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        df = Counter()
        total_docs = 0

        for t in texts:
            toks = set(self._tokenize(t))
            if not toks:
                continue
            df.update(toks)
            total_docs += 1

        # build vocab
        most_common = df.most_common(self.vocab_size)
        self.word_index = {w: i + 1 for i, (w, _) in enumerate(most_common)}

        # compute IDF
        self.idf = {}
        for w, c in most_common:
            self.idf[w] = math.log((1 + total_docs) / (1 + c)) + 1.0

        self.fitted = True

    # --------------------------------------------------
    # Encode
    # --------------------------------------------------
    def encode_texts(self, texts):
        if not self.fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit_tokenizer first.")

        if isinstance(texts, str):
            texts = [texts]

        out = []

        for text in texts:
            toks = self._tokenize(text)[: self.max_len]

            if not toks:
                out.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue

            counts = Counter(toks)

            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            total_weight = 0.0

            for i, (w, tf) in enumerate(counts.items()):
                idx = self.word_index.get(w)
                if not idx:
                    continue

                idf = self.idf.get(w, 1.0)

                # TF-IDF weight
                weight = tf * idf

                # slight position bias to break prefix collapse
                pos_scale = 1.0 + 0.01 * i

                vec += self.word_matrix[idx] * weight * pos_scale
                total_weight += abs(weight)

            if total_weight > 1e-8:
                vec /= total_weight

            # L2 normalize
            norm = np.linalg.norm(vec) + 1e-8
            vec = vec / norm

            out.append(vec.astype(np.float32))

        return np.vstack(out)

    def get_vector(self, text):
        return self.encode_texts([text])[0]
