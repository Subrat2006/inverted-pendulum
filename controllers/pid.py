import numpy as np


class PID:
    """Generic PID: output = kp*error + ki*integral(error) + kd*d(error)/dt."""

    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class CascadedPID:
    """
    Outer loop: cart position error -> desired tilt angle.
    Inner loop: tilt-angle error -> force command.

    IMPORTANT sign convention (specific to this plant): with the equations of
    motion used here, thetaddot = a*theta - b*u with a, b > 0 (theta=0 upright).
    To stabilize, u must feed back POSITIVE theta (not the usual setpoint-minus-
    measurement sign), i.e. the inner-loop error is (theta - desired_theta), and
    the proportional gain must satisfy kp > (M+m)*g for the sign of the restoring
    force to even be correct - verify this against your own linearization before
    trusting these numbers, exactly as urged in the "own the derivation" step.
    """

    def __init__(self, pos_gains, angle_gains, max_tilt_deg=6.0):
        self.pos_pid = PID(*pos_gains)
        self.angle_pid = PID(*angle_gains)
        self.max_tilt = np.radians(max_tilt_deg)

    def reset(self):
        self.pos_pid.reset()
        self.angle_pid.reset()

    def compute(self, state, x_ref, dt):
        x, xdot, theta, thetadot = state
        desired_theta = np.clip(
            self.pos_pid.compute(x_ref - x, dt), -self.max_tilt, self.max_tilt
        )
        return self.angle_pid.compute(theta - desired_theta, dt)
