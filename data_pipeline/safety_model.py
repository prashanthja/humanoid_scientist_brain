"""
Module: Safety Model
A simple trainable classifier to detect harmful content.
Phase A: basic LSTM model
Later: Transformer model.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 

class SafetyModel:
    def __init__(self, vocab_size=1000, max_len=50):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.model = None

    def build_model(self):
        model = Sequential([
            Embedding(self.vocab_size, 16, input_length=self.max_len),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        self.model = model

    def train(self, texts, labels, epochs=5):
        # Fit tokenizer
        self.tokenizer.fit_on_texts(texts)
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.max_len, padding="post")

        labels = np.array(labels)   # ✅ FIX: ensure labels are numpy array

        if self.model is None:
            self.build_model()

        self.model.fit(padded, labels, epochs=epochs, verbose=1)

    def predict(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post")
        pred = self.model.predict(padded, verbose=0)[0][0]
        return pred  # value between 0 (safe) and 1 (unsafe)
