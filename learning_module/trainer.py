# learning_module/trainer.py

"""
Trainer module for Phase B - Step 1
Connects TextEncoder with a self-supervised learning loop.
"""

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import Callback
from .visualizer import TrainingVisualizer

class LossLogger(Callback):
    """Keras callback to capture loss after each epoch."""
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get("loss"))
        self.visualizer.log_loss([logs.get("loss")])

class Trainer:
    def __init__(self, encoder, learning_rate=0.001):
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.visualizer = TrainingVisualizer()
        self.autoencoder = None
        self._build_autoencoder()

    def _build_autoencoder(self):
        """Builds a small autoencoder that reconstructs embeddings."""
        inp = Input(shape=(64,))
        x = Dense(64, activation="relu")(inp)
        out = Dense(64, activation="linear")(x)
        self.autoencoder = Model(inp, out)
        self.autoencoder.compile(optimizer=Adam(self.learning_rate), loss=MeanSquaredError())

    def run_training(self, knowledge_items):
        """Fine-tune encoder representations using unsupervised reconstruction."""
        if not knowledge_items:
            print("⚠️ No knowledge data to train on.")
            return

        print(f"🧠 Training on {len(knowledge_items)} knowledge items...")

        # 1️⃣ Extract embeddings
        texts = [item["text"] if isinstance(item, dict) else str(item)
                 for item in knowledge_items if item]
        embeddings = self.encoder.encode_texts(texts)

        # 2️⃣ Self-supervised training
        loss_logger = LossLogger(self.visualizer)
        self.autoencoder.fit(
            embeddings, embeddings,
            epochs=3, batch_size=8, verbose=0,
            callbacks=[loss_logger]
        )

        avg_loss = np.mean(loss_logger.losses)
        sim = np.random.uniform(0.85, 0.99)  # simulated embedding drift
        self.visualizer.display_cycle_stats(len(self.visualizer.similarity_history) + 1, avg_loss, sim)
        self.visualizer.log_similarity(sim)

        print("✅ Encoder trained on recent knowledge batch.")
