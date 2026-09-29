import numpy as np
from parameters import PendulumParams


def nonlinear_dynamics(state, u, p: PendulumParams):
    """Ground-truth plant. state = [x, xdot, theta, thetadot], theta=0 is upright."""
    x, xdot, theta, thetadot = state
    s, c = np.sin(theta), np.cos(theta)
    denom = p.M + p.m * s ** 2
    xddot = (u + p.m * s * (p.l * thetadot ** 2 - p.g * c)) / denom
    thetaddot = (p.g * s - xddot * c) / p.l
    return np.array([xdot, xddot, thetadot, thetaddot])


def linear_matrices(p: PendulumParams):
    """A, B for the linearization about the upright equilibrium (theta=0)."""
    M, m, l, g = p.M, p.m, p.l, p.g
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -m * g / M, 0],
        [0, 0, 0, 1],
        [0, 0, (M + m) * g / (M * l), 0],
    ])
    B = np.array([[0], [1 / M], [0], [-1 / (M * l)]])
    return A, B


def linear_dynamics(state, u, p: PendulumParams):
    A, B = linear_matrices(p)
    return A @ state + B.flatten() * u


def rk4_step(f, state, u, p: PendulumParams):
    dt = p.dt
    k1 = f(state, u, p)
    k2 = f(state + dt / 2 * k1, u, p)
    k3 = f(state + dt / 2 * k2, u, p)
    k4 = f(state + dt * k3, u, p)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
