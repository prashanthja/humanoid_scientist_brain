import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

class TextEncoder:
    def __init__(self, vocab_size=5000, embedding_dim=64, max_len=50):
        self.vocab_size = vocab_size
        self.embedding_dim = 64  # fixed embedding dimension to 64
        self.max_len = max_len
        self.fitted = False

        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")

        # Build embedding model (trainable)
        self.encoder = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim),
            LSTM(64),
            Dense(64, activation="relu")  # output dimension = 64
        ])

    def fit_tokenizer(self, texts):
        """Fit tokenizer on a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
        self.tokenizer.fit_on_texts(texts)
        self.fitted = True

    def texts_to_sequences(self, texts):
        """Convert texts to sequences of token indices."""
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for text in texts:
            if text is None:
                continue
            seq = self.tokenizer.texts_to_sequences([text])
            seqs.append(seq[0] if seq and len(seq[0]) > 0 else [])
        return seqs

    def encode_texts(self, texts):
        """
        Convert input text(s) into vector embeddings.
        Automatically handles single string or list input.
        """
        if not self.fitted:
            raise RuntimeError("Tokenizer is not fitted. Call fit_tokenizer() before encoding texts.")

        if isinstance(texts, str):
            texts = [texts]

        seqs = self.texts_to_sequences(texts)
        seqs = [s for s in seqs if s is not None and len(s) > 0]

        if not seqs:
            # Return a zero vector if input is invalid or all tokens are OOV
            return np.zeros((1, self.embedding_dim))

        padded = pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post")

        embeddings = self.encoder.predict(padded, verbose=0)
        return embeddings

    def get_vector(self, text):
        """Return the embedding vector for a single text."""
        vecs = self.encode_texts(text)
        return vecs[0] if len(vecs) > 0 else np.zeros(self.embedding_dim)
