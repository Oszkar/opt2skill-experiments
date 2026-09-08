"""Convert a SquatSolution into the reference contract arrays (MuJoCo conventions)."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from o2s.models import g1
from o2s.reference import contract, convert


def _pelvis_local(p_world: np.ndarray, pelvis_pos: np.ndarray, rot_pelvis: np.ndarray) -> np.ndarray:
    return rot_pelvis.T @ (p_world - pelvis_pos)


def solution_to_reference(sol, pin_model: pin.Model, mj_model, cfg: dict) -> dict:
    N = sol.us.shape[0]
    nq = pin_model.nq
    data = pin_model.createData()
    ref = {k: np.zeros((N + 1, *s[1:])) if s[0] == "N+1" else np.zeros((N, *s[1:])) for k, s in contract.ARRAY_SPEC.items()}
    ref["t"] = np.arange(N + 1) * sol.dt
    fids = {
        "foot_pos_left": pin_model.getFrameId(g1.FOOT_LINKS[0]),
        "foot_pos_right": pin_model.getFrameId(g1.FOOT_LINKS[1]),
        "hand_pos_left": pin_model.getFrameId(g1.PALM_FRAMES[0]),
        "hand_pos_right": pin_model.getFrameId(g1.PALM_FRAMES[1]),
    }
    for k in range(N + 1):
        x = sol.xs[k]
        q, v = x[:nq], x[nq:]
        qpos, qvel = convert.pin_to_mj(q, v)
        ref["qpos"][k] = qpos
        ref["qvel"][k] = qvel
        pin.forwardKinematics(pin_model, data, q)
        pin.updateFramePlacements(pin_model, data)
        ref["com"][k] = pin.centerOfMass(pin_model, data, q)
        ref["pelvis_pos"][k] = qpos[:3]
        ref["pelvis_quat"][k] = qpos[3:7]
        ref["pelvis_linvel"][k] = v[:3]     # Pinocchio base linear velocity is already pelvis-local
        ref["pelvis_angvel"][k] = v[3:6]
        rot = convert.quat_wxyz_to_mat(qpos[3:7])
        for key, fid in fids.items():
            ref[key][k] = _pelvis_local(data.oMf[fid].translation, qpos[:3], rot)

    # MuJoCo applies passive joint damping -b*qdot; the actuator must supply it on top of the model torque.
    damping = np.asarray(mj_model.dof_damping[6:], dtype=float)
    qd = sol.xs[:, nq + 6:]
    ref["tau"] = sol.us + damping * 0.5 * (qd[:-1] + qd[1:])
    ref["foot_wrench_left"] = sol.forces["left"].copy()
    ref["foot_wrench_right"] = sol.forces["right"].copy()
    contract.validate(ref)
    return ref


def solution_meta(sol, filter_result, cfg: dict) -> dict:
    p = sol.params
    return {
        "depth": p.depth, "t_stand0": p.t_stand0, "t_down": p.t_down, "t_hold": p.t_hold, "t_up": p.t_up,
        "t_stand1": p.t_stand1, "com_shift_x": p.com_shift_x, "dt": p.dt,
        "solver": {"converged": sol.converged, "iters": sol.iters, "cost": sol.cost, "solve_time": sol.solve_time},
        "filter": {"ok": filter_result.ok, "failures": filter_result.failures, "details": filter_result.details},
        "assets": {k: v["commit"] for k, v in cfg["assets"].items() if isinstance(v, dict)},
        "torque_includes_mujoco_damping": True,
        "wrench_frame": "world-aligned axes at sole frame origin",
    }
