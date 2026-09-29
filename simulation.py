import numpy as np
from dynamics import nonlinear_dynamics, rk4_step


def simulate(controller, p, theta0=0.15, x_ref=0.0, t_final=6.0,
             disturbance_start=None, disturbance_duration=0.0, disturbance_force=0.0,
             force_limit=20.0):
    """
    Runs one controller against the nonlinear plant. Every controller passed in
    here sees the SAME plant, integrator, initial condition, disturbance and
    actuator limits, which is what makes the benchmark fair.

    disturbance_start / disturbance_duration / disturbance_force: an extra force
    is added to the controller's output for t in [disturbance_start,
    disturbance_start + disturbance_duration). duration=0 gives a single-timestep
    impulse; a positive duration gives a sustained disturbance.
    """
    controller.reset()
    n_steps = int(t_final / p.dt)
    state = np.array([0.0, 0.0, theta0, 0.0])
    log = np.zeros((n_steps, 5))   # columns: t, x, xdot, theta, thetadot
    forces = np.zeros(n_steps)

    for i in range(n_steps):
        t = i * p.dt
        F = float(np.clip(controller.compute(state, x_ref, p.dt), -force_limit, force_limit))
        if disturbance_start is not None and disturbance_start <= t < disturbance_start + max(disturbance_duration, p.dt):
            F += disturbance_force
        forces[i] = F
        log[i, 0] = t
        log[i, 1:] = state
        state = rk4_step(nonlinear_dynamics, state, F, p)

    return log, forces
