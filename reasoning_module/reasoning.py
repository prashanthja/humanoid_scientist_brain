"""
Reasoning engine that uses KB + Physics Engine for simple scientific Q&A.
"""

from knowledge_base.retriever import Retriever
from .physics_engine import PhysicsEngine

class ReasoningEngine:
    def __init__(self, kb, encoder):
        self.kb = kb
        self.retriever = Retriever(kb, encoder)
        self.physics = PhysicsEngine()

    def answer(self, query):
        # Simple hard-coded parser (Phase A)
        if "force" in query.lower():
            return f"Force = mass * acceleration. Example: 2kg * 3m/s² = {self.physics.calculate_force(2,3)} N"
        elif "kinetic" in query.lower():
            return f"Kinetic Energy = 0.5 * m * v². Example: 2kg, 4m/s = {self.physics.kinetic_energy(2,4)} J"
        else:
            # fallback: just return KB info
            results = self.retriever.semantic_search(query)
            return results if results else "I don't know yet."
