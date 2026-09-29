import numpy as np


def compute_metrics(log, forces, theta0):
    t, theta = log[:, 0], log[:, 3]
    theta_deg = np.degrees(theta)
    theta0_deg = np.degrees(theta0)
    dt = t[1] - t[0]

    # Settling time: the earliest time after which the response stays inside the
    # tolerance band for the rest of the run (NOT just the first time it dips in -
    # a response that re-exits the band later must not count as settled early).
    settle_band = max(0.02 * abs(theta0_deg), 0.5)
    out_of_band = np.where(np.abs(theta_deg) >= settle_band)[0]
    if len(out_of_band) == 0:
        settling_time = 0.0
    elif out_of_band[-1] == len(theta_deg) - 1:
        settling_time = np.nan  # never actually settles within the simulated horizon
    else:
        settling_time = t[out_of_band[-1] + 1]

    # "decay_time_90_to_10": time from first crossing 90% of the initial offset down
    # to first crossing 10% of it. This is NOT the conventional step-response rise
    # time (which is defined for a response rising toward a nonzero setpoint) - it's
    # a response-speed proxy for a regulation/decay problem, named accordingly so it
    # isn't mistaken for the textbook definition. Define it this way explicitly in
    # the report.
    try:
        idx_90 = np.where(np.abs(theta_deg) <= 0.9 * abs(theta0_deg))[0][0]
        idx_10 = np.where(np.abs(theta_deg) <= 0.1 * abs(theta0_deg))[0][0]
        decay_time_90_to_10 = t[idx_10] - t[idx_90]
    except IndexError:
        decay_time_90_to_10 = np.nan

    overshoot = -np.min(theta_deg) / abs(theta0_deg) * 100 if np.min(theta_deg) < 0 else 0.0
    steady_state_error = np.mean(np.abs(theta_deg[-200:]))
    control_effort = float(np.sum(forces ** 2) * dt)
    max_control_force = float(np.max(np.abs(forces)))

    return {
        "settling_time_s": settling_time,
        "decay_time_90_to_10_s": decay_time_90_to_10,
        "overshoot_pct": overshoot,
        "steady_state_error_deg": steady_state_error,
        "control_effort": control_effort,
        "max_control_force_N": max_control_force,
    }
