"""Run a scripted controller and save a readable trace, summary, and diagnostic figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path

import numpy as np

from o2s.reference import contract
from o2s.tracking.env import TrackingEnv


def run_episode(env: TrackingEnv, controller: str = "reference") -> tuple[dict, list[dict]]:
    if controller not in ("reference", "hold"):
        raise ValueError("controller must be reference or hold")
    obs, initial = env.reset()
    initial["observation"] = obs
    records = []
    while True:
        # Deliberately simple: current reference joint offsets, no lookahead or torque input.
        action = obs["reference_joint_position"] if controller == "reference" else np.zeros(env.model.nu)
        obs, reward, terminated, truncated, info = env.step(action)
        info.update(reward=reward, terminated=terminated, truncated=truncated, observation=obs)
        records.append(info)
        if terminated or truncated:
            break
    return initial, records


def write_outputs(out: Path, env: TrackingEnv, initial: dict, records: list[dict], controller: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(records)
    states = np.stack([initial["qpos"], *[r["qpos"] for r in records]])
    velocities = np.stack([initial["qvel"], *[r["qvel"] for r in records]])
    time = np.array([0., *[r["time"] for r in records]])
    arrays = {key: np.stack([r[key] for r in records]) for key in (
        "action", "requested_target", "applied_target", "target_clipped", "substep_start_time",
        "substep_torque", "substep_requested_torque", "substep_contact_count",
        "substep_contact_normal", "mean_torque", "reference_torque")}
    arrays.update(time=time, qpos=states, qvel=velocities,
                  reference_qpos=env.ref["qpos"][:count + 1], reference_qvel=env.ref["qvel"][:count + 1],
                  joint_names=np.array(env.cfg["joints"]))
    for key, value in initial["observation"].items():
        arrays["observation_" + key] = np.stack([value, *[r["observation"][key] for r in records]])
    np.savez_compressed(out / "trace.npz", **arrays)
    scalar_keys = ("interval_index", "state_index", "time", "reward", "pelvis_error", "joint_rms_error",
                   "torque_rms_error", "peak_effort_ratio", "saturated_fraction", "min_pelvis_height",
                   "terminated", "truncated", "reason")
    reward_names = list(records[0]["reward_components"])
    with (out / "steps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[*scalar_keys, *["reward_" + k for k in reward_names]])
        writer.writeheader()
        for r in records:
            writer.writerow({**{k: r[k] for k in scalar_keys},
                             **{"reward_" + k: v for k, v in r["reward_components"].items()}})
    summary = {
        "controller": controller, "steps": count, "reference_intervals": env.length,
        "reference_file": str(env.reference_path),
        "reference_sha256": hashlib.sha256(env.reference_path.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(),
                     **{name: importlib.metadata.version(name) for name in ("mujoco", "numpy")}},
        "observation_shapes": {k: list(v.shape) for k, v in initial["observation"].items()},
        "simulated_seconds": time[-1], "reason": records[-1]["reason"],
        "max_pelvis_error_m": max(r["pelvis_error"] for r in records),
        "max_effort_ratio": max(r["peak_effort_ratio"] for r in records),
        "mean_reward": float(np.mean([r["reward"] for r in records])),
        "saturated_fraction": float(np.mean([r["saturated_fraction"] for r in records])),
        "clipped_target_count": int(arrays["target_clipped"].sum()),
        "control_dt": contract.DT, "physics_dt": env.model.opt.timestep,
        "action_units": "radian offsets from home; clipped to joint/control ranges",
        "gains": "unmodified model position actuators; no external feedforward",
        "kp": env.model.actuator_gainprm[:, 0].tolist(),
        "effort_limits": env.effort_limits.tolist(), "reference_metadata": env.meta,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"G1 tracking: {controller} | {summary['reason']} at {time[-1]:.2f} s")
    axes[0, 0].plot(time, states[:, 2], label="actual")
    axes[0, 0].plot(time, env.ref["qpos"][:count + 1, 2], "--", label="reference")
    axes[0, 0].axhline(.4, color="r", linestyle=":", label="fall threshold")
    axes[0, 0].set_ylabel("Pelvis height [m]")
    knee = env.cfg["joints"].index("left_knee_joint")
    axes[0, 1].plot(time, states[:, 7 + knee], label="actual")
    axes[0, 1].plot(time, env.ref["qpos"][:count + 1, 7 + knee], "--", label="reference")
    axes[0, 1].step(time[:-1], arrays["applied_target"][:, knee], where="post", label="command")
    axes[0, 1].set_ylabel("Left knee angle [rad]")
    st = arrays["substep_start_time"].ravel()
    axes[1, 0].plot(st, arrays["substep_torque"][:, :, knee].ravel(), label="actual")
    axes[1, 0].step(time[:-1], arrays["reference_torque"][:, knee], where="post", label="reference")
    axes[1, 0].set_ylabel("Left knee torque [N m]")
    axes[1, 1].plot(time[1:], [r["peak_effort_ratio"] for r in records], label="applied peak / limit")
    axes[1, 1].plot(time[1:], [r["saturated_fraction"] for r in records], label="saturated fraction")
    axes[1, 1].axhline(1, color="r", linestyle=":")
    axes[1, 1].set_ylabel("Actuator utilization")
    for side, name in enumerate(("left", "right")):
        axes[2, 0].plot(st, arrays["substep_contact_normal"][:, :, side].ravel(), label=f"{name} actual normal")
        axes[2, 0].step(time[:-1], env.ref[f"foot_wrench_{name}"][:count, 2], where="post", linestyle="--", label=f"{name} ref world Fz")
    axes[2, 0].set_ylabel("Foot contact force [N]")
    for name in reward_names:
        axes[2, 1].plot(time[1:], [r["reward_components"][name] for r in records], label=name)
    axes[2, 1].set_ylabel("Reward components [0, 1]")
    for ax in axes.flat:
        ax.set_xlabel("Simulation time [s]")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(out / "diagnostics.png", dpi=140)
    plt.close(fig)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--controller", choices=("reference", "hold"), default="reference")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        ap.error("--out must be a new or empty directory")
    env = TrackingEnv(args.reference)
    initial, records = run_episode(env, args.controller)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = write_outputs(args.out, env, initial, records, args.controller)
    print(json.dumps({k: summary[k] for k in ("controller", "steps", "reason", "simulated_seconds",
                     "max_pelvis_error_m", "max_effort_ratio", "saturated_fraction", "mean_reward")}, indent=2))
    print(f"Wrote {args.out}: summary.json, steps.csv, trace.npz, diagnostics.png")
    return 0  # A recorded fall is an experiment result, not a CLI failure.


if __name__ == "__main__":
    raise SystemExit(main())
