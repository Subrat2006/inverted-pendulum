"""
Verifies that the nonlinear plant and the linearized plant agree closely for small
initial tilt angles, and diverge for larger ones - confirming the linearization
(used for LQR design) is valid in its intended small-angle regime before it's
trusted for controller design.
"""

import numpy as np
from parameters import PendulumParams
from dynamics import nonlinear_dynamics, linear_dynamics, rk4_step


def instantaneous_check():
    """Compares theta-acceleration from both models AT the initial instant (before
    the unstable dynamics have had time to run away). This isolates the modeling
    error introduced by the small-angle approximation itself."""
    p = PendulumParams()
    print("Instantaneous check (u=0, thetadot=0): theta-acceleration mismatch")
    print(f"{'theta0 (deg)':>14} | {'nonlinear thetaddot':>20} | {'linear thetaddot':>18} | {'% error':>8}")
    print("-" * 70)
    for theta0_deg in [1, 2, 5, 10, 20, 40]:
        theta0 = np.radians(theta0_deg)
        state = np.array([0.0, 0.0, theta0, 0.0])
        nl_acc = nonlinear_dynamics(state, 0.0, p)[3]
        lin_acc = linear_dynamics(state, 0.0, p)[3]
        pct_err = abs(nl_acc - lin_acc) / abs(nl_acc) * 100
        print(f"{theta0_deg:>14} | {nl_acc:>20.5f} | {lin_acc:>18.5f} | {pct_err:>7.2f}%")


def short_horizon_check():
    """Compares full trajectories over a short horizon (before large-angle behavior
    dominates), which is the regime the linearization is actually meant to cover."""
    p = PendulumParams()
    t_final = 0.05  # short horizon: instability hasn't had time to blow the models apart
    n_steps = int(t_final / p.dt)

    print("\nShort-horizon trajectory check (0.05 s, free response, u=0)")
    print(f"{'theta0 (deg)':>14} | {'max |theta_nl - theta_lin| (deg)':>34}")
    print("-" * 52)
    for theta0_deg in [1, 2, 5, 10, 20, 40]:
        theta0 = np.radians(theta0_deg)
        state_nl = np.array([0.0, 0.0, theta0, 0.0])
        state_lin = np.array([0.0, 0.0, theta0, 0.0])
        max_err = 0.0
        for _ in range(n_steps):
            u = 0.0
            state_nl = rk4_step(nonlinear_dynamics, state_nl, u, p)
            state_lin = rk4_step(linear_dynamics, state_lin, u, p)
            max_err = max(max_err, abs(state_nl[2] - state_lin[2]))
        print(f"{theta0_deg:>14} | {np.degrees(max_err):>34.4f}")


def run_validation():
    instantaneous_check()
    short_horizon_check()


if __name__ == "__main__":
    run_validation()
