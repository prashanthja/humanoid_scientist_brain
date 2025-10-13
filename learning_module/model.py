"""
Module: Learning Model
Phase A: Simple text embedding + classification model.
Later: Custom transformer.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class LearningModel:
    def __init__(self, vocab_size=1000, max_len=50):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.model = None

    def build_model(self):
        model = Sequential([
            Embedding(self.vocab_size, 16, input_length=self.max_len),
            GlobalAveragePooling1D(),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        self.model = model

    def train(self, texts, labels, epochs=5):
        self.tokenizer.fit_on_texts(texts)
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.max_len, padding="post")

        labels = np.array(labels)

        if self.model is None:
            self.build_model()

        self.model.fit(padded, labels, epochs=epochs, verbose=1)

    def predict(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post")
        pred = self.model.predict(padded, verbose=0)[0][0]
        return pred
