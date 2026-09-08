"""Shared simulator settings and reversible replay controller configuration."""
from contextlib import contextmanager
import hashlib
import json

import mujoco
import numpy as np


def apply_config(model, cfg):
    physics = cfg["simulation"]
    for key, choices in {
        "integrator": {"Euler": mujoco.mjtIntegrator.mjINT_EULER},
        "solver": {"Newton": mujoco.mjtSolver.mjSOL_NEWTON},
        "cone": {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL},
    }.items():
        setattr(model.opt, key, choices[physics[key]])
    for key in ("timestep", "iterations", "ls_iterations", "tolerance", "ls_tolerance", "noslip_iterations", "impratio"):
        value = physics[key]
        if not np.isfinite(value) or value < 0 or (key != "noslip_iterations" and value == 0):
            raise ValueError(f"invalid simulation {key}: {value}")
        if key.endswith("iterations") and int(value) != value:
            raise ValueError(f"simulation {key} must be an integer")
        setattr(model.opt, key, value)
    gravity = np.asarray(physics["gravity"], dtype=float)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must contain three finite values")
    model.opt.gravity[:] = gravity
    position = cfg["controllers"]["position"]
    def vector(key):
        value = np.broadcast_to(np.asarray(position[key], dtype=float), (model.nu,))
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"invalid controller {key}")
        return value
    kp, kd, damping = (vector(key) for key in ("kp", "kd", "joint_damping"))
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 0] = 0
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kd
    model.dof_damping[:6] = 0
    model.dof_damping[6:] = damping


def replay_parameters(cfg=None, kp_scale=None, ankle_pitch_kp=None):
    if cfg is None:
        from o2s.models.g1 import load_config
        cfg = load_config()
    preset = cfg["controllers"]["stabilized"]
    scale = preset["kp_scale"] if kp_scale is None else kp_scale
    ankle = preset["ankle_pitch_kp"] if ankle_pitch_kp is None else ankle_pitch_kp
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(ankle) or ankle <= 0:
        raise ValueError("replay gains must be finite and positive")
    return float(scale), float(ankle)


@contextmanager
def replay_gains(model, kp_scale=None, ankle_pitch_kp=None, *, cfg=None):
    scale, ankle = replay_parameters(cfg, kp_scale, ankle_pitch_kp)
    gain, bias = model.actuator_gainprm.copy(), model.actuator_biasprm.copy()
    kp = gain[:, 0] * scale
    for name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        kp[model.actuator(name).id] = ankle
    try:
        model.actuator_gainprm[:, 0] = kp
        model.actuator_biasprm[:, 1] = -kp
        yield
    finally:
        model.actuator_gainprm[:] = gain
        model.actuator_biasprm[:] = bias


def snapshot(model, cfg):
    """Actual loaded values, including temporary gain overrides, not just defaults."""
    keys = ("timestep", "integrator", "solver", "cone", "iterations", "ls_iterations",
            "tolerance", "ls_tolerance", "noslip_iterations", "noslip_tolerance",
            "impratio", "disableflags", "enableflags")
    return dict(physics={**{key: getattr(model.opt, key) for key in keys},
                         "gravity": model.opt.gravity.tolist()},
                kp=model.actuator_gainprm[:, 0].tolist(),
                kd=(-model.actuator_biasprm[:, 2]).tolist(),
                joint_damping=model.dof_damping[6:].tolist(),
                joint_frictionloss=model.dof_frictionloss[6:].tolist(),
                config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
