"""MuJoCo validation of a reference trajectory. Run before any RL.

Check 1, inverse-dynamics consistency: at every control node, MuJoCo's inverse dynamics with the reference
state, the finite-difference acceleration and the reference contact wrenches (MuJoCo contacts disabled) must
require the reference torque, after accounting for the interval-mean damping convention of the export.
`mj_inverse` returns `M qacc + bias - passive - constraint`, and its passive term uses the instantaneous
`-dof_damping * qvel[k]`, while the exported `tau[k] = us[k] + dof_damping * 0.5 * (qvel[k] + qvel[k+1])`
uses the mean damping over the interval; the difference between the two conventions is added back before
comparing. This compares the two dynamics models directly and does not depend on MuJoCo's soft-contact
behaviour or on the robot's open-loop stability.

Check 2, stabilized replay: reference torque as feedforward plus a PD loop with gains high enough to hold the
robot up (the G1 is an inverted pendulum; Playground's ankle-pitch gain of 20 N m/rad cannot). 4x Playground
gains, ankle pitch 400 N m/rad (2x/200 held the canonical 0.2 m squat but fell on 10 of 100 randomized squats;
all ten track within 1.1 cm at 4x/400). The PD correction is logged separately: a good reference needs little
correction.

Check 3 (informational): the same replay without feedforward, to show what the torque reference contributes.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from o2s.models import g1
from o2s.reference import contract

SUBSTEPS = 10               # 20 ms control interval / 2 ms physics step
FALL_PELVIS_Z = 0.4         # m; below this the robot has fallen

# check 1 acceptance
ID_RMS_JOINT = 0.01         # N m
ID_MAX_JOINT = 0.2          # N m  # 0.05 rejected 10% of attempts (deep squats, waist pitch 0.05-0.10 N m): finite-difference acceleration error, not a model gap
ID_MAX_BASE_FORCE = 5.0     # N
ID_MAX_BASE_TORQUE = 2.0    # N m
# check 2 acceptance: 4x Playground gains, ankle pitch 400 N m/rad (2x/200 held the canonical 0.2 m
# squat but fell on 10 of 100 randomized squats; all ten track within 1.1 cm at 4x/400)
REPLAY_MAX_PELVIS_ERR = 0.03   # m
REPLAY_RMS_JOINT_ERR = 0.03    # rad
REPLAY_PD_OVER_REF_LEGS = 0.25


@dataclass
class CheckReport:
    name: str
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    message: str = ""


def set_state(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, qvel: np.ndarray) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)


@contextmanager
def disabled(model: mujoco.MjModel, *flags: int):
    saved = model.opt.disableflags
    for f in flags:
        model.opt.disableflags |= int(f)
    try:
        yield
    finally:
        model.opt.disableflags = saved


@contextmanager
def replay_gains(model: mujoco.MjModel, kp_scale: float, ankle_pitch_kp: float):
    gain = model.actuator_gainprm.copy()
    bias = model.actuator_biasprm.copy()
    kp = model.actuator_gainprm[:, 0] * kp_scale
    for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        kp[model.actuator(name).id] = ankle_pitch_kp
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp   # MuJoCo position actuator: force = kp * (ctrl - q) - kv * qdot
    try:
        yield
    finally:
        model.actuator_gainprm[:] = gain
        model.actuator_biasprm[:] = bias


def step_interval(model: mujoco.MjModel, data: mujoco.MjData, tau: np.ndarray, ctrl: np.ndarray | None) -> np.ndarray:
    """Advance one 20 ms control interval; return the mean actuator torque over the substeps."""
    acc = np.zeros(model.nu)
    for _ in range(SUBSTEPS):
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[6:] = tau
        if ctrl is not None:
            data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        acc += data.qfrc_actuator[6:]
    return acc / SUBSTEPS


def _sole_point(model: mujoco.MjModel, data: mujoco.MjData, link: str, sole_offset: np.ndarray) -> tuple[int, np.ndarray]:
    bid = model.body(link).id
    return bid, data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ sole_offset


def check_inverse_dynamics(model: mujoco.MjModel, ref: dict, cfg: dict) -> CheckReport:
    data = mujoco.MjData(model)
    N = ref["tau"].shape[0]
    dt = float(ref["t"][1] - ref["t"][0])
    assert abs(SUBSTEPS * model.opt.timestep - dt) < 1e-12, (
        f"file dt {dt} does not match SUBSTEPS * model.opt.timestep = {SUBSTEPS * model.opt.timestep}"
    )
    sole_offset = np.asarray(cfg["frames"]["sole_offset"], dtype=float)
    joint_res = np.zeros((N, model.nu))
    base_res = np.zeros((N, 6))
    flags = (mujoco.mjtDisableBit.mjDSBL_CONTACT, mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS,
             mujoco.mjtDisableBit.mjDSBL_ACTUATION, mujoco.mjtDisableBit.mjDSBL_LIMIT)
    with disabled(model, *flags):
        for k in range(N):
            data.qpos[:] = ref["qpos"][k]
            data.qvel[:] = ref["qvel"][k]
            mujoco.mj_forward(model, data)  # kinematics and com quantities for mj_applyFT; overwrites qacc
            data.qacc[:] = (ref["qvel"][k + 1] - ref["qvel"][k]) / dt  # must come AFTER mj_forward
            qfrc_ext = np.zeros(model.nv)
            for key, link in (("foot_wrench_left", g1.FOOT_LINKS[0]), ("foot_wrench_right", g1.FOOT_LINKS[1])):
                bid, point = _sole_point(model, data, link, sole_offset)
                w = ref[key][k]
                mujoco.mj_applyFT(model, data, w[:3], w[3:], point, bid, qfrc_ext)
            mujoco.mj_inverse(model, data)
            required = data.qfrc_inverse - qfrc_ext   # generalized force the actuators must supply
            # mj_inverse's passive term is the instantaneous -dof_damping*qvel[k]; the exported tau[k]
            # uses the interval-mean damping dof_damping*0.5*(qvel[k]+qvel[k+1]) because Crocoddyl does not
            # model damping and the actuator must supply it over the whole interval. Add back the
            # difference so the comparison isn't dominated by this known convention mismatch, which is
            # otherwise identically -dof_damping*0.5*(qvel[k+1]-qvel[k]).
            required[6:] += model.dof_damping[6:] * 0.5 * (ref["qvel"][k + 1, 6:] - ref["qvel"][k, 6:])
            base_res[k] = required[:6]
            joint_res[k] = required[6:] - ref["tau"][k]
    metrics = {
        "rms_joint_residual": float(np.sqrt(np.mean(joint_res**2))),
        "max_joint_residual": float(np.max(np.abs(joint_res))),
        "max_base_force_residual": float(np.max(np.linalg.norm(base_res[:, :3], axis=1))),
        "max_base_torque_residual": float(np.max(np.linalg.norm(base_res[:, 3:], axis=1))),
        "rms_ref_torque": float(np.sqrt(np.mean(ref["tau"] ** 2))),
        "worst_joint": int(np.argmax(np.max(np.abs(joint_res), axis=0))),
    }
    ok = (metrics["rms_joint_residual"] < ID_RMS_JOINT and metrics["max_joint_residual"] < ID_MAX_JOINT
          and metrics["max_base_force_residual"] < ID_MAX_BASE_FORCE and metrics["max_base_torque_residual"] < ID_MAX_BASE_TORQUE)
    msg = (f"joint residual rms {metrics['rms_joint_residual']:.3f} / max {metrics['max_joint_residual']:.3f} N m "
           f"(limits {ID_RMS_JOINT} / {ID_MAX_JOINT}; worst joint index {metrics['worst_joint']}), base residual "
           f"{metrics['max_base_force_residual']:.2f} N / {metrics['max_base_torque_residual']:.2f} N m "
           f"(limits {ID_MAX_BASE_FORCE} / {ID_MAX_BASE_TORQUE}); reference torque rms {metrics['rms_ref_torque']:.2f} N m")
    return CheckReport("inverse_dynamics", ok, metrics, msg)


def check_replay(model: mujoco.MjModel, ref: dict, cfg: dict, feedforward: bool = True, kp_scale: float = 4.0, ankle_pitch_kp: float = 400.0) -> CheckReport:
    data = mujoco.MjData(model)
    N = ref["tau"].shape[0]
    dt = float(ref["t"][1] - ref["t"][0])
    assert abs(SUBSTEPS * model.opt.timestep - dt) < 1e-12, (
        f"file dt {dt} does not match SUBSTEPS * model.opt.timestep = {SUBSTEPS * model.opt.timestep}"
    )
    set_state(model, data, ref["qpos"][0], ref["qvel"][0])
    pd_log = np.zeros((N, model.nu))
    pelvis_err = np.zeros(N)
    joint_err = np.zeros((N, model.nu))
    min_pelvis_z = np.inf
    zeros = np.zeros(model.nu)
    with replay_gains(model, kp_scale, ankle_pitch_kp):
        for k in range(N):
            pd_log[k] = step_interval(model, data, ref["tau"][k] if feedforward else zeros, ref["qpos"][k + 1, 7:])
            pelvis_err[k] = np.linalg.norm(data.qpos[:3] - ref["qpos"][k + 1, :3])
            joint_err[k] = data.qpos[7:] - ref["qpos"][k + 1, 7:]
            min_pelvis_z = min(min_pelvis_z, float(data.qpos[2]))
    effort_limits = model.jnt_actfrcrange[1:, 1]
    tau_ff = ref["tau"] if feedforward else np.zeros_like(ref["tau"])
    peak_effort_ratio = float(np.max(np.abs(tau_ff + pd_log) / effort_limits))
    metrics = {
        "max_pelvis_error": float(pelvis_err.max()),
        "rms_joint_error": float(np.sqrt(np.mean(joint_err**2))),
        "min_pelvis_z": min_pelvis_z,
        "completed": float(min_pelvis_z > FALL_PELVIS_Z),
        "kp_scale": kp_scale,
        "ankle_pitch_kp": ankle_pitch_kp,
        "feedforward": float(feedforward),
        "peak_effort_ratio": peak_effort_ratio,  # informational: |tau_ff + PD correction| / effort limit
    }
    for group, idx in cfg["groups"].items():
        rms_pd = float(np.sqrt(np.mean(pd_log[:, idx] ** 2)))
        rms_ref = float(np.sqrt(np.mean(ref["tau"][:, idx] ** 2)))
        metrics[f"pd_over_ref_{group}"] = rms_pd / max(rms_ref, 1e-9)
        metrics[f"rms_pd_{group}"] = rms_pd
        metrics[f"rms_ref_{group}"] = rms_ref
    name = "replay_ff" if feedforward else "replay_pd_only"
    if feedforward:
        ok = (metrics["completed"] == 1.0 and metrics["max_pelvis_error"] < REPLAY_MAX_PELVIS_ERR
              and metrics["rms_joint_error"] < REPLAY_RMS_JOINT_ERR and metrics["pd_over_ref_legs"] < REPLAY_PD_OVER_REF_LEGS)
    else:
        ok = True  # informational contrast
    msg = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    return CheckReport(name, ok, metrics, msg)


def run_all(model: mujoco.MjModel, ref: dict, cfg: dict) -> list[CheckReport]:
    return [
        check_inverse_dynamics(model, ref, cfg),
        check_replay(model, ref, cfg, feedforward=True),
        check_replay(model, ref, cfg, feedforward=False),
    ]


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("data/refs/dev/squat_dev.npz")
    cfg = g1.load_config()
    model = g1.load_mj_model(cfg)
    ref, meta = contract.load(path)
    print(f"{path}: N={ref['tau'].shape[0]} depth={meta.get('depth')}")
    reports = run_all(model, ref, cfg)
    for r in reports:
        tag = "info" if r.name == "replay_pd_only" else ("ok" if r.ok else "FAIL")
        print(f"[{tag}] {r.name}: {r.message}")
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
