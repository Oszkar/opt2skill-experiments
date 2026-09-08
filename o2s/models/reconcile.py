"""Checks that the MuJoCo (Playground MJCF) and Pinocchio (Unitree URDF) G1 models agree."""
from __future__ import annotations

import sys
from dataclasses import dataclass

import mujoco
import numpy as np
import pinocchio as pin

from o2s.models import g1


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def body_index_map(mj_model: mujoco.MjModel, pin_model: pin.Model) -> dict[str, int]:
    """MuJoCo body name -> Pinocchio joint id carrying that body's inertia.

    Every moving MuJoCo body carries exactly one joint (`body_jntadr`), the URDF has a joint of the same
    name, and Pinocchio stores the child link's inertia on that joint. Names do not follow a `_link`/`_joint`
    pattern everywhere (torso_link is driven by waist_pitch_joint), so the joint is read from the model.
    The pelvis carries the free joint, which is Pinocchio's root joint (id 1).
    """
    out = {}
    for i in range(1, mj_model.nbody):
        name = mj_model.body(i).name
        if mj_model.body_jntnum[i] != 1:
            raise ValueError(f"body {name} has {mj_model.body_jntnum[i]} joints, expected exactly 1")
        jadr = int(mj_model.body_jntadr[i])
        if mj_model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
            out[name] = 1
            continue
        jname = mj_model.joint(jadr).name
        jid = pin_model.getJointId(jname)
        if jid >= pin_model.njoints:
            raise KeyError(f"no Pinocchio joint named {jname} for MuJoCo body {name}")
        out[name] = jid
    return out


def random_state(pin_model: pin.Model, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    q = pin.neutral(pin_model)
    q[:3] = rng.uniform(-1, 1, 3)
    quat = rng.normal(size=4)
    q[3:7] = quat / np.linalg.norm(quat)  # Pinocchio order xyzw
    lo, hi = pin_model.lowerPositionLimit[7:], pin_model.upperPositionLimit[7:]
    q[7:] = rng.uniform(lo, hi)
    v = rng.normal(size=pin_model.nv)
    return q, v


def run_checks(mj_model: mujoco.MjModel, pin_model: pin.Model, cfg: dict, n_poses: int = 20, seed: int = 0) -> list[CheckResult]:
    results: list[CheckResult] = []
    names_mj = [mj_model.joint(i).name for i in range(1, mj_model.njnt)]
    names_pin = list(pin_model.names)[2:]
    results.append(CheckResult("joint_names", names_mj == names_pin == g1.joint_names(cfg), f"mj={names_mj[:2]}..., pin={names_pin[:2]}..."))

    lim_err = np.max(np.abs(mj_model.jnt_range[1:, 0] - pin_model.lowerPositionLimit[7:]))
    lim_err = max(lim_err, np.max(np.abs(mj_model.jnt_range[1:, 1] - pin_model.upperPositionLimit[7:])))
    results.append(CheckResult("joint_limits", lim_err < 1e-4, f"max limit difference {lim_err:.2e} rad"))

    m_mj = float(mj_model.body_subtreemass[0])
    m_pin = float(pin.computeTotalMass(pin_model))
    results.append(CheckResult("total_mass", abs(m_mj - m_pin) / m_pin < 0.01, f"mj {m_mj:.4f} kg, pin {m_pin:.4f} kg"))

    bmap = body_index_map(mj_model, pin_model)
    worst_mass, worst_com, worst_inertia = 0.0, 0.0, 0.0
    for name, jid in bmap.items():
        bid = mj_model.body(name).id
        inertia = pin_model.inertias[jid]
        worst_mass = max(worst_mass, abs(mj_model.body_mass[bid] - inertia.mass) / max(inertia.mass, 1e-6))
        worst_com = max(worst_com, float(np.max(np.abs(mj_model.body_ipos[bid] - inertia.lever))))
        # MuJoCo stores principal moments in the body_iquat frame; compare rotation-invariant eigenvalues.
        eig_mj = np.sort(mj_model.body_inertia[bid])
        eig_pin = np.sort(np.linalg.eigvalsh(inertia.inertia))
        worst_inertia = max(worst_inertia, float(np.max(np.abs(eig_mj - eig_pin) / np.maximum(eig_pin, 1e-6))))
    results.append(CheckResult("body_mass", worst_mass < 0.01, f"worst relative body-mass error {worst_mass:.2e}"))
    results.append(CheckResult("body_com", worst_com < 1e-3, f"worst body-CoM error {worst_com:.2e} m"))
    results.append(CheckResult("body_inertia", worst_inertia < 0.02, f"worst relative principal-inertia error {worst_inertia:.2e}"))

    from o2s.reference import convert  # local import: convert depends on nothing in this module

    rng = np.random.default_rng(seed)
    data_mj = mujoco.MjData(mj_model)
    data_pin = pin_model.createData()
    frames = [g1.PELVIS, *g1.FOOT_LINKS, *g1.HAND_LINKS]
    worst_fk = 0.0
    for _ in range(n_poses):
        q, v = random_state(pin_model, rng)
        qpos, qvel = convert.pin_to_mj(q, v)
        data_mj.qpos[:] = qpos
        data_mj.qvel[:] = qvel
        mujoco.mj_forward(mj_model, data_mj)
        pin.forwardKinematics(pin_model, data_pin, q)
        pin.updateFramePlacements(pin_model, data_pin)
        for f in frames:
            p_mj = data_mj.xpos[mj_model.body(f).id]
            p_pin = data_pin.oMf[pin_model.getFrameId(f)].translation
            worst_fk = max(worst_fk, float(np.max(np.abs(p_mj - p_pin))))
    results.append(CheckResult("forward_kinematics", worst_fk < 1e-4, f"worst frame-position error {worst_fk:.2e} m over {n_poses} poses"))

    trn_ok = bool(np.all(mj_model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)) and bool(np.all(mj_model.actuator_gear[:, 0] == 1.0))
    trn_ok &= sorted(mj_model.actuator_trnid[:, 0].tolist()) == list(range(1, mj_model.njnt))
    results.append(CheckResult("transmission", trn_ok, "all actuators are gear-1 joint transmissions covering joints 1..29"))

    sole_offset = np.asarray(cfg["frames"]["sole_offset"], dtype=float)
    foot_half_size = np.asarray(cfg["frames"]["foot_half_size"], dtype=float)
    foot_err = 0.0
    for geom_name in ("left_foot", "right_foot"):
        gid = mj_model.geom(geom_name).id
        sole_from_geom = mj_model.geom_pos[gid] - np.array([0.0, 0.0, mj_model.geom_size[gid, 2]])
        foot_err = max(foot_err, float(np.max(np.abs(sole_from_geom - sole_offset))))
        foot_err = max(foot_err, float(np.max(np.abs(mj_model.geom_size[gid, :2] - foot_half_size))))
    results.append(CheckResult("foot_geometry", foot_err < 1e-6, f"worst foot geometry error {foot_err:.2e} m vs config sole_offset/foot_half_size"))

    gain_ok = bool(np.all(mj_model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED))
    bias_ok = bool(np.all(mj_model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE))
    kp_err = float(np.max(np.abs(mj_model.actuator_biasprm[:, 1] + mj_model.actuator_gainprm[:, 0])))
    act_ok = gain_ok and bias_ok and kp_err < 1e-9
    results.append(CheckResult("actuator_type", act_ok, f"all actuators FIXED gain / AFFINE bias position servos (biasprm[:,1] == -gainprm[:,0], worst error {kp_err:.2e})"))
    return results


def main() -> int:
    cfg = g1.load_config()
    results = run_checks(g1.load_mj_model(cfg), g1.load_pin_model(cfg), cfg)
    for r in results:
        print(f"[{'ok' if r.ok else 'FAIL'}] {r.name}: {r.detail}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
