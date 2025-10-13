"""
Module: Safety Filter
Now uses SafetyModel (ML) instead of keyword rules.
"""

from .safety_model import SafetyModel

class SafetyFilter:
    def __init__(self):
        self.model = SafetyModel()

        # Mock training data
        texts = [
            "I want to study physics",
            "Energy cannot be created or destroyed",
            "How to build a bomb",
            "Attack humans with robots",
            "Mathematics helps solve problems",
            "Research about quantum mechanics"
        ]
        labels = [0, 0, 1, 1, 0, 0]  # 0 = safe, 1 = unsafe

        # Train model
        self.model.train(texts, labels, epochs=10)

    def filter(self, data_list):
        safe = []
        for text in data_list:
            score = self.model.predict(text)
            if score < 0.5:  # threshold
                safe.append(text)
        return safe
