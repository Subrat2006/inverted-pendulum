"""
Trains a PPO agent on the same nonlinear cart-pole plant used by the PID/LQR
benchmark (see rl_env.py). Requires gymnasium, stable-baselines3 and torch.

Version lineage (documented, not hidden, since the failures were diagnostic):
  v1: 200k timesteps, no observation normalization, raw-force reward.
      -> never settled reliably even in the nominal case.
  v2: 500k timesteps, VecNormalize added.
      -> settled but slowly/inaccurately (nominal SSE 0.36 deg vs PID's
         0.014 deg), used very little control effort, and collapsed under
         the sustained disturbance and most of the IC sweep. Diagnosed cause:
         a reward-scaling bug where the effort penalty could numerically
         outweigh the angle penalty (see rl_env.py's step() comment).
  v3 (this run): same training distribution as v2 (nominal-only IC
      randomization, no disturbance/parameter randomization) so this is a
      SINGLE-VARIABLE test of the reward fix in isolation, not a mix of the
      "reward" question and the "does it generalize to disturbances"
      question - those are kept as separate experiments, per the reward
      priority theta-stability > x-tracking > control-effort, fixed BEFORE
      this run, not tuned to beat PID/LQR after the fact.

If v3 converges well on the nominal/IC-sweep tests, the next separate
experiment (not this script) is training a "robust" variant with disturbance
and parameter randomization added to the training loop - do that only after
v3's baseline stabilization is solid, and only if time allows.

Usage:
    pip install gymnasium stable-baselines3 torch
    python3 train_rl.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_env import CartPoleEnv


def make_env():
    return CartPoleEnv()


if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", env, verbose=1, seed=0)
    model.learn(total_timesteps=500_000)

    model.save("ppo_cartpole")
    env.save("vecnormalize.pkl")
    print("Saved model to ppo_cartpole.zip and normalization stats to vecnormalize.pkl")
    print("This is v3 (reward-fixed). Compare its run_benchmark.py numbers against the")
    print("v2 numbers already recorded in README.md before drawing conclusions.")
