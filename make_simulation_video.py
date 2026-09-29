"""
Renders a simulation video (animated cart-pole) for every available
controller running the same scenario side by side. This is the "simulation
video" deliverable the problem statement asks for, alongside the reproducible
code. Uses the exact same plant, dynamics and controllers as
run_benchmark.py - not a separate/simplified rendering model.

Usage:
    python3 make_simulation_video.py

Requires ffmpeg to be installed and on PATH (used by matplotlib's
FFMpegWriter). If RL's ppo_cartpole.zip/vecnormalize.pkl and
stable-baselines3/torch are all available, RL is included automatically as a
third panel - otherwise the video renders with PID and LQR only.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle

from parameters import PendulumParams
from controllers.pid import CascadedPID
from controllers.lqr import LQRController
from simulation import simulate

RESULTS_DIR = "results"


def build_controllers(p):
    controllers = {
        "PID": CascadedPID(
            pos_gains=(0.0166, 0.0, 0.0299),
            angle_gains=(290.2944, 1.1885, 18.0269),
            max_tilt_deg=6.0,
        ),
        "LQR": LQRController(p, Q=np.diag([1.0, 1.0, 10.0, 1.0]), R=np.array([[0.05]])),
    }
    try:
        from controllers.rl import RLController
        controllers["RL"] = RLController()
    except Exception as e:
        print(f"[make_simulation_video] Skipping RL panel: {e}")
    return controllers


def render_video(theta0_deg=8.5, disturbance_start=None, disturbance_duration=0.0,
                  disturbance_force=0.0, t_final=6.0, fps=30, out_path="results/simulation_video.mp4"):
    p = PendulumParams()
    theta0 = np.radians(theta0_deg)
    controllers = build_controllers(p)
    names = list(controllers.keys())

    logs = {}
    for name, ctrl in controllers.items():
        log, _ = simulate(
            ctrl, p, theta0=theta0, x_ref=0.0, t_final=t_final,
            disturbance_start=disturbance_start, disturbance_duration=disturbance_duration,
            disturbance_force=disturbance_force,
        )
        logs[name] = log

    stride = max(1, round(1.0 / (p.dt * fps)))
    n_frames = len(logs[names[0]][::stride])

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    if len(names) == 1:
        axes = [axes]

    cart_w, cart_h = 0.3, 0.15
    pole_l_vis = 0.5  # visual pole length, matches p.l
    track_half = 1.6

    elements = {}
    for ax, name in zip(axes, names):
        ax.set_xlim(-track_half, track_half)
        ax.set_ylim(-0.3, 1.0)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.axhline(0, color="k", lw=1)
        cart_patch = Rectangle((-cart_w / 2, -cart_h / 2), cart_w, cart_h, fc="steelblue", ec="k")
        ax.add_patch(cart_patch)
        pole_line, = ax.plot([], [], "r-", lw=3)
        mass_dot, = ax.plot([], [], "o", color="firebrick", markersize=10)
        info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", fontsize=9, family="monospace")
        elements[name] = dict(cart=cart_patch, pole=pole_line, mass=mass_dot, text=info_text)

    def update(frame_idx):
        artists = []
        for name in names:
            row = logs[name][::stride][frame_idx]
            t, x, xdot, theta, thetadot = row
            el = elements[name]
            el["cart"].set_xy((x - cart_w / 2, -cart_h / 2))
            pole_x = [x, x + pole_l_vis * np.sin(theta)]
            pole_y = [0, pole_l_vis * np.cos(theta)]
            el["pole"].set_data(pole_x, pole_y)
            el["mass"].set_data([pole_x[1]], [pole_y[1]])
            el["text"].set_text(f"t={t:5.2f}s\ntheta={np.degrees(theta):6.2f} deg\nx={x:6.2f} m")
            artists += [el["cart"], el["pole"], el["mass"], el["text"]]
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000 / fps)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved video to {out_path} ({n_frames} frames at {fps} fps)")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    render_video(
        theta0_deg=8.5, t_final=6.0,
        out_path=os.path.join(RESULTS_DIR, "simulation_video_nominal.mp4"),
    )
    render_video(
        theta0_deg=8.5, disturbance_start=3.0, disturbance_duration=0.5, disturbance_force=3.0,
        t_final=6.0, out_path=os.path.join(RESULTS_DIR, "simulation_video_sustained_disturbance.mp4"),
    )
