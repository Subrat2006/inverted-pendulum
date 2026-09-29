"""
Systematic PID tuning: optimizes the cascaded PID gains against a fixed,
predefined objective (settling time, overshoot, steady-state error, control
effort) using differential evolution - not hand-picked by eye. This gives a
defensible answer to "how did you choose your PID gains?".

Usage:
    python3 tune_pid.py
"""

import numpy as np
from scipy.optimize import differential_evolution

from parameters import PendulumParams
from controllers.pid import CascadedPID
from simulation import simulate
from metrics import compute_metrics

THETA0_DEG = 8.5
T_TUNE = 6.0    # matches T_FINAL exactly, so the optimizer is scored on the
                # same window the final report will actually use (an earlier
                # version used a shorter tuning horizon and got a gain set that
                # only barely settled by the very end of the real 6s window -
                # tuning and reporting must use the same horizon)
T_FINAL = 6.0   # full horizon used for final verification (matches run_benchmark.py)

# Search bounds: [pos_kp, pos_ki, pos_kd, angle_kp, angle_ki, angle_kd].
# angle_kp's lower bound (15.0) is set ABOVE (M+m)*g (~10.79 for nominal
# parameters) deliberately - see controllers/pid.py's docstring on why the
# proportional gain must exceed this for the restoring-force sign to even be
# correct. This isn't an arbitrary search range, it's constrained by the
# plant physics, so every candidate the optimizer tries is at least
# structurally capable of stabilizing.
BOUNDS = [
    (0.0, 0.5),     # pos_kp
    (0.0, 0.2),     # pos_ki
    (0.0, 0.5),     # pos_kd
    (15.0, 300.0),  # angle_kp
    (0.0, 20.0),    # angle_ki
    (0.0, 60.0),    # angle_kd
]

# Objective weights - chosen to prioritize actually settling first, then
# steady-state accuracy, then overshoot, then control effort. State this
# weighting explicitly in the report; it IS the tuning methodology.
W_SETTLE = 1.0
W_OVERSHOOT = 0.05
W_SSE = 2.0
W_EFFORT = 0.0005
FAIL_PENALTY = 50.0  # added on top of steady-state error when the run never settles


def cost(gains, p, theta0, t_final):
    pos_gains = tuple(gains[0:3])
    angle_gains = tuple(gains[3:6])
    ctrl = CascadedPID(pos_gains=pos_gains, angle_gains=angle_gains, max_tilt_deg=6.0)
    log, forces = simulate(ctrl, p, theta0=theta0, x_ref=0.0, t_final=t_final)
    m = compute_metrics(log, forces, theta0)

    if np.isnan(m["settling_time_s"]):
        settle_term = FAIL_PENALTY + m["steady_state_error_deg"]
    else:
        settle_term = m["settling_time_s"]

    return (
        W_SETTLE * settle_term
        + W_OVERSHOOT * m["overshoot_pct"]
        + W_SSE * m["steady_state_error_deg"]
        + W_EFFORT * m["control_effort"]
    )


def tune(seed=0, maxiter=40, popsize=15):
    p = PendulumParams()
    theta0 = np.radians(THETA0_DEG)

    result = differential_evolution(
        lambda g: cost(g, p, theta0, T_TUNE),
        BOUNDS,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=True,
        disp=True,
    )

    pos_gains = tuple(float(v) for v in result.x[0:3])
    angle_gains = tuple(float(v) for v in result.x[3:6])

    # Verify on the FULL horizon (T_FINAL), not just the shorter tuning
    # horizon used during the search, to make sure the result generalizes.
    ctrl = CascadedPID(pos_gains=pos_gains, angle_gains=angle_gains, max_tilt_deg=6.0)
    log, forces = simulate(ctrl, p, theta0=theta0, x_ref=0.0, t_final=T_FINAL)
    m = compute_metrics(log, forces, theta0)

    print("\n=== Tuned PID gains ===")
    print(f"pos_gains   = {tuple(round(v, 4) for v in pos_gains)}")
    print(f"angle_gains = {tuple(round(v, 4) for v in angle_gains)}")
    print(f"Final cost (tuning horizon) = {result.fun:.4f}")
    print(f"\nVerification on full {T_FINAL}s horizon (theta0={THETA0_DEG} deg):")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

    return pos_gains, angle_gains, m


if __name__ == "__main__":
    # seed=1, maxiter=70, popsize=18 is the setting that actually converged well
    # (an earlier seed/budget got stuck in a local optimum around cost~6.5-8.9,
    # left in git history/report as a legitimate "first attempt" data point -
    # global optimizers like differential_evolution are not guaranteed to find
    # the global optimum in one run, which is itself worth a line in the report).
    tune(seed=1, maxiter=70, popsize=18)
