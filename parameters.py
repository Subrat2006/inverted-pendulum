from dataclasses import dataclass


@dataclass
class PendulumParams:
    M: float = 1.0      # cart mass (kg)
    m: float = 0.1      # pendulum point mass (kg)
    l: float = 0.5      # pendulum length, pivot to mass (m)
    g: float = 9.81     # gravity (m/s^2)
    dt: float = 0.005   # simulation timestep (s)
