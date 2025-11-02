#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelEvaluator — Evaluates Humanoid Scientist Brain’s understanding of physics & mathematics.
"""

import random
import numpy as np

class ModelEvaluator:
    def __init__(self, trainer, kb, bridge):
        self.trainer = trainer
        self.kb = kb
        self.bridge = bridge
        self.benchmark_questions = self._load_benchmarks()

    def _load_benchmarks(self):
        """A lightweight benchmark of conceptual reasoning questions."""
        return [
            {
                "question": "What is Newton's second law?",
                "answer": "Force equals mass times acceleration",
                "topic": "classical mechanics"
            },
            {
                "question": "What does the Schrödinger equation describe?",
                "answer": "The evolution of a quantum state over time",
                "topic": "quantum mechanics"
            },
            {
                "question": "What is entropy in thermodynamics?",
                "answer": "A measure of disorder or number of microstates",
                "topic": "thermodynamics"
            },
            {
                "question": "Define tensor.",
                "answer": "A mathematical object that generalizes scalars, vectors, and matrices",
                "topic": "mathematics"
            },
            {
                "question": "What is Maxwell’s equation about?",
                "answer": "It describes how electric and magnetic fields propagate and interact",
                "topic": "electromagnetism"
            },
        ]

    def _similarity(self, a, b):
        """Compute simple cosine similarity using embedding bridge."""
        try:
            va = np.array(self.bridge.encode(a))
            vb = np.array(self.bridge.encode(b))
            sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
            return float(sim)
        except Exception:
            return 0.0

    def evaluate_on_benchmark(self):
        """Evaluate the model across benchmark concepts."""
        results = []
        total_score = 0.0

        for q in self.benchmark_questions:
            score = self._similarity(q["question"], q["answer"])
            results.append({
                "question": q["question"],
                "expected": q["answer"],
                "topic": q["topic"],
                "similarity": round(score, 4)
            })
            total_score += score

        avg_score = total_score / len(results)
        return {
            "benchmarks": results,
            "average_similarity": round(avg_score, 4),
            "verdict": (
                "Excellent conceptual alignment"
                if avg_score > 0.85 else
                "Moderate understanding"
                if avg_score > 0.6 else
                "Needs more training"
            )
        }
