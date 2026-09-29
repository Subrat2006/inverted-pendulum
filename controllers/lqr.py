import numpy as np
from scipy.linalg import solve_continuous_are
from dynamics import linear_matrices


def compute_lqr_gain(p, Q, R):
    """Solve the continuous-time algebraic Riccati equation for the linearized plant."""
    A, B = linear_matrices(p)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K, A, B


class LQRController:
    """Full-state feedback: u = -K @ (state - target), target = [x_ref, 0, 0, 0]."""

    def __init__(self, p, Q, R):
        self.K, self.A, self.B = compute_lqr_gain(p, Q, R)

    def reset(self):
        pass  # LQR is stateless (no integrator), nothing to reset

    def compute(self, state, x_ref, dt):
        target = np.array([x_ref, 0.0, 0.0, 0.0])
        return float((-self.K @ (state - target)).item())
