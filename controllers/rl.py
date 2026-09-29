import os
import numpy as np


class RLController:
    """
    Wraps a trained Stable-Baselines3 model behind the same .compute(state,
    x_ref, dt) interface as PID and LQR, so it drops straight into the same
    simulate()/compute_metrics() benchmark harness.

    Requires stable-baselines3 to be installed AND a trained model file
    (produced by train_rl.py) to exist at model_path. If vecnorm_path also
    exists (train_rl.py's v2 saves it alongside the model), observations are
    normalized the same way they were during training before being passed to
    the policy - skipping this for a model trained WITH VecNormalize will
    silently produce a badly-mismatched, poorly-performing policy at
    inference time even though the model loads without error.
    """

    def __init__(self, model_path="ppo_cartpole.zip", vecnorm_path="vecnormalize.pkl", force_limit=20.0):
        from stable_baselines3 import PPO  # imported lazily: PID/LQR-only runs don't need SB3 installed
        self.model = PPO.load(model_path)
        self.force_limit = force_limit

        self.vecnorm = None
        if os.path.exists(vecnorm_path):
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from rl_env import CartPoleEnv
            # A throwaway env, never stepped - only used to hold/apply the
            # saved running mean/variance for observation normalization.
            dummy = DummyVecEnv([lambda: CartPoleEnv()])
            self.vecnorm = VecNormalize.load(vecnorm_path, dummy)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False

    def reset(self):
        pass  # policy is stateless at inference time - nothing to reset

    def compute(self, state, x_ref, dt):
        obs = np.array([state[0] - x_ref, state[1], state[2], state[3]], dtype=np.float32)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return float(np.clip(action[0], -1.0, 1.0)) * self.force_limit
