"""
Basic Physics Engine for simple reasoning tasks.
Phase A: Only handles basic formulas.
Later: will expand into symbolic math + real simulations.
"""

class PhysicsEngine:
    def __init__(self):
        pass

    def calculate_force(self, mass, acceleration):
        # F = m * a
        return mass * acceleration

    def kinetic_energy(self, mass, velocity):
        # KE = 0.5 * m * v^2
        return 0.5 * mass * (velocity ** 2)

    def potential_energy(self, mass, height, g=9.81):
        # PE = m * g * h
        return mass * g * height
