import numpy as np
import gymnasium as gym
from gymnasium import spaces
from parameters import PendulumParams
from dynamics import nonlinear_dynamics, rk4_step


class CartPoleEnv(gym.Env):
    """
    Wraps the exact same nonlinear plant used by the PID and LQR controllers
    (dynamics.py) as a Gymnasium environment, so the RL agent is trained and
    evaluated against an identical plant - required for a fair 3-way benchmark.
    """

    metadata = {"render_modes": []}

    def __init__(self, force_limit=20.0, t_final=6.0, theta_fail=np.radians(24), x_fail=2.4):
        super().__init__()
        self.p = PendulumParams()
        self.force_limit = force_limit
        self.max_steps = int(t_final / self.p.dt)
        self.theta_fail = theta_fail
        self.x_fail = x_fail

        # Action is normalized to [-1, 1] and scaled to +-force_limit N inside step();
        # this is standard practice for PPO/SAC and keeps gradients well-scaled.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        high = np.array([x_fail * 1.5, 10.0, theta_fail * 1.5, 15.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Matches the +-8.5 deg nominal tilt used in the PID/LQR benchmark scenarios.
        theta0 = self.np_random.uniform(-0.15, 0.15)
        x0 = self.np_random.uniform(-0.05, 0.05)
        # Randomize the target position per episode so the policy learns to be
        # reference-aware (matches PID/LQR, which both take x_ref explicitly) rather
        # than only ever having seen x_ref = 0.
        self.x_ref = self.np_random.uniform(-0.2, 0.2)
        self.state = np.array([x0, 0.0, theta0, 0.0], dtype=np.float64)
        self.steps = 0
        return self._obs(), {}

    def _obs(self):
        x, xdot, theta, thetadot = self.state
        return np.array([x - self.x_ref, xdot, theta, thetadot], dtype=np.float32)

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        u = a * self.force_limit
        self.state = rk4_step(nonlinear_dynamics, self.state, u, self.p)
        self.steps += 1

        x, xdot, theta, thetadot = self.state
        failed = abs(theta) > self.theta_fail or abs(x) > self.x_fail
        truncated = self.steps >= self.max_steps

        # v3 reward - fixed BEFORE training, not tuned against PID/LQR results.
        # v2's reward used raw force u (Newtons) for the effort term: with
        # theta in radians (max ~0.42 near failure) and u up to 20N, the
        # effort term (0.001*u^2, max 0.4) could numerically OUTWEIGH the
        # angle term (theta^2, max ~0.175) - a real scale bug, not just "the
        # wrong coefficient", and it explains why v2 learned "keep force
        # small" over "fix the angle" (see README for the v2 diagnosis).
        # v3 fixes this by penalizing the normalized action `a` in [-1, 1]
        # instead of raw Newtons (decouples the penalty from force_limit's
        # scale), adds explicit velocity-damping terms so "stability" isn't
        # only about position, and sets an explicit priority BEFORE looking
        # at any results: theta stability > x tracking > control effort.
        W_THETA, W_THETADOT = 10.0, 0.1
        W_X, W_XDOT = 0.5, 0.05
        W_U = 0.01
        FAIL_PENALTY = 100.0  # raised from v2's 20.0 - v2's sustained-disturbance
                               # collapse showed a weak failure penalty doesn't
                               # discourage losing control badly enough

        reward = -(
            W_THETA * theta ** 2 + W_THETADOT * thetadot ** 2
            + W_X * (x - self.x_ref) ** 2 + W_XDOT * xdot ** 2
            + W_U * a ** 2
        )
        if failed:
            reward -= FAIL_PENALTY

        return self._obs(), float(reward), bool(failed), bool(truncated), {}
