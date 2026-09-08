"""Deterministic, motion-only reference tracking with an explicit reset/step API.

Actions are 29 joint-position offsets in radians from home. They are clipped to
joint/control ranges before being sent through the model's position actuators.
By default no external feedforward is applied. An explicit diagnostic-only
feedforward keyword enables comparison with reference validation. Observations contain
proprioception and current motion references; privileged state, reference torques,
contact forces and reward diagnostics live in info only.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from o2s.models import g1
from o2s.reference import contract, convert


def tracking_rewards(qpos: np.ndarray, qvel: np.ndarray, ref: dict, k: int) -> dict[str, float]:
    """Four equally weighted diagnostic rewards, not a frozen paper replication.

    Each term is exp(-coefficient * SUM squared error); orientation uses squared
    shortest quaternion angle in radians. Torque is deliberately not rewarded.
    """
    dot = abs(float(np.dot(qpos[3:7], ref["qpos"][k, 3:7])))
    angle = 2 * np.arccos(np.clip(dot, 0.0, 1.0))
    return {
        "joint_position": float(np.exp(-5 * np.sum((qpos[7:] - ref["qpos"][k, 7:]) ** 2))),
        "joint_velocity": float(np.exp(-0.1 * np.sum((qvel[6:] - ref["qvel"][k, 6:]) ** 2))),
        "pelvis_position": float(np.exp(-20 * np.sum((qpos[:3] - ref["qpos"][k, :3]) ** 2))),
        "pelvis_orientation": float(np.exp(-10 * angle ** 2)),
    }


class TrackingEnv:
    """One reference, one private MuJoCo model, deterministic reset; no auto-reset.

    reset() -> (observation, info)
    step(action) -> (observation, reward, terminated, truncated, info)
    A step consumes interval k and returns state/reference k+1. A fall terminates;
    completing N intervals truncates. A fall takes precedence at the last interval.
    """

    def __init__(self, reference: str | Path, *, equations: bool = False):
        self.reference_path = Path(reference)
        self.ref, self.meta = contract.load(self.reference_path)
        self.length = contract.validate(self.ref)
        self.cfg = g1.load_config()
        self.model = g1.load_mj_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        self.substeps = int(round(contract.DT / self.model.opt.timestep))
        if self.substeps < 1 or not np.isclose(self.substeps * self.model.opt.timestep,
                                               contract.DT, rtol=0, atol=1e-12):
            raise ValueError("physics timestep must divide the 20 ms reference interval")
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("tracking diagnostics require Euler integration")
        # Direct array mapping and the unsaturated force formula below depend on these properties.
        if not (self.model.nu == g1.NJ
                and np.all(self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                and np.array_equal(self.model.actuator_trnid[:, 0], np.arange(1, g1.NJ + 1))
                and np.all(self.model.actuator_gear[:, 0] == 1)
                and np.all(self.model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
                and np.all(self.model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
                and np.all(self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)):
            raise ValueError("requires ordered gear-1 stateless position actuators")
        self.home = g1.home_qpos(self.model, self.cfg)[7:]
        self.target_low = np.maximum(self.model.jnt_range[1:, 0], self.model.actuator_ctrlrange[:, 0])
        self.target_high = np.minimum(self.model.jnt_range[1:, 1], self.model.actuator_ctrlrange[:, 1])
        self.effort_limits = g1.effort_limits(self.cfg)
        self.foot_geoms = [self.model.geom(name).id for name in ("left_foot", "right_foot")]
        self.index = 0
        self.previous_action = np.zeros(g1.NJ)
        self._done = True
        self.equations = None
        if equations:
            from o2s.tracking.equations import EquationMonitor
            self.equations = EquationMonitor(self)

    def observation(self) -> dict[str, np.ndarray]:
        k = self.index
        rot = convert.quat_wxyz_to_mat(self.data.qpos[3:7])
        ref_rot = convert.quat_wxyz_to_mat(self.ref["qpos"][k, 3:7])
        return {
            "joint_position": self.data.qpos[7:].copy() - self.home,
            "joint_velocity": self.data.qvel[6:].copy(),
            "base_linear_velocity": rot.T @ self.data.qvel[:3],
            "base_angular_velocity": self.data.qvel[3:6].copy(),
            "projected_gravity": rot.T @ np.array([0., 0., -1.]),
            "previous_action": self.previous_action.copy(),
            "reference_joint_position": self.ref["qpos"][k, 7:].copy() - self.home,
            "reference_joint_velocity": self.ref["qvel"][k, 6:].copy(),
            "reference_base_linear_velocity": self.ref["pelvis_linvel"][k].copy(),
            "reference_base_angular_velocity": self.ref["pelvis_angvel"][k].copy(),
            "reference_projected_gravity": ref_rot.T @ np.array([0., 0., -1.]),
            "phase": np.array([k / self.length]),
        }

    def reset(self) -> tuple[dict, dict]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.ref["qpos"][0]
        self.data.qvel[:] = self.ref["qvel"][0]
        self.data.ctrl[:] = np.clip(self.ref["qpos"][0, 7:], self.target_low, self.target_high)
        mujoco.mj_forward(self.model, self.data)
        self.index = 0
        self.previous_action[:] = 0
        self._done = False
        return self.observation(), {"state_index": 0, "time": 0.,
                                    "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}

    def _contacts(self) -> tuple[np.ndarray, np.ndarray]:
        counts = np.zeros(2, dtype=int)
        normal = np.zeros(2)
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for side, geom in enumerate(self.foot_geoms):
                if geom in (contact.geom1, contact.geom2):
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    counts[side] += 1
                    normal[side] += force[0]
        return counts, normal

    def step(self, action: np.ndarray, *, feedforward: np.ndarray | None = None) -> tuple[dict, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("call reset() before stepping or after episode end")
        action = np.asarray(action, dtype=float)
        if action.shape != (g1.NJ,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain 29 finite joint offsets in radians")
        ff = np.zeros(g1.NJ) if feedforward is None else np.asarray(feedforward, dtype=float)
        if ff.shape != (g1.NJ,) or not np.all(np.isfinite(ff)):
            raise ValueError("feedforward must contain 29 finite torques")
        k = self.index
        requested = self.home + action
        target = np.clip(requested, self.target_low, self.target_high)
        self.data.ctrl[:] = target
        torque = np.empty((self.substeps, g1.NJ))
        requested_torque = np.empty_like(torque)
        contact_counts = np.empty((self.substeps, 2), dtype=int)
        contact_normal = np.empty((self.substeps, 2))
        times = np.empty(self.substeps)
        min_height = float(self.data.qpos[2])
        equation_samples = []
        for j in range(self.substeps):
            times[j] = self.data.time
            # G1 gear-1 affine position servos; evaluate at the substep's pre-integration state.
            requested_torque[j] = (self.model.actuator_gainprm[:, 0] * target
                                   + self.model.actuator_biasprm[:, 0]
                                   + self.model.actuator_biasprm[:, 1] * self.data.qpos[7:]
                                   + self.model.actuator_biasprm[:, 2] * self.data.qvel[6:])
            self.data.qfrc_applied[:] = 0
            self.data.qfrc_applied[6:] = ff
            self.data.xfrc_applied[:] = 0
            if self.equations is not None:
                equation_samples.append(self.equations.sample(self.data))
            mujoco.mj_step(self.model, self.data)
            torque[j] = self.data.qfrc_actuator[6:]
            contact_counts[j], contact_normal[j] = self._contacts()
            if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))
                    and np.all(np.isfinite(torque[j]))
                    and np.isclose(self.data.time, times[j] + self.model.opt.timestep,
                                   rtol=0, atol=1e-10)):
                self._done = True
                raise RuntimeError("MuJoCo became non-finite or reset during simulation")
            min_height = min(min_height, float(self.data.qpos[2]))
        self.index += 1
        self.previous_action = (target - self.home).copy()
        # Refresh kinematics for the returned state; substep forces were saved before this call.
        mujoco.mj_forward(self.model, self.data)
        components = tracking_rewards(self.data.qpos, self.data.qvel, self.ref, self.index)
        terminated = min_height < 0.4
        truncated = self.index == self.length and not terminated
        self._done = terminated or truncated
        total_torque = torque + ff
        mean_torque = torque.mean(axis=0)
        info = {
            "interval_index": k, "state_index": self.index, "time": float(self.data.time),
            "reason": "fall" if terminated else ("reference_end" if truncated else "running"),
            "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
            "action": action.copy(), "requested_target": requested, "applied_target": target,
            "target_clipped": requested != target,
            "substep_start_time": times, "substep_torque": torque,
            "substep_requested_torque": requested_torque,
            "feedforward_torque": ff.copy(), "substep_total_torque": total_torque,
            "mean_total_torque": total_torque.mean(axis=0),
            "effort_ok": bool(np.all(np.abs(total_torque) / self.effort_limits <= 1.0 + 1e-6)),
            "substep_contact_count": contact_counts, "substep_contact_normal": contact_normal,
            "mean_torque": mean_torque, "reference_torque": self.ref["tau"][k].copy(),
            "peak_effort_ratio": float(np.max(np.abs(total_torque) / self.effort_limits)),
            "saturated_fraction": float(np.mean(np.abs(requested_torque - torque) > 1e-6)),
            "pelvis_error": float(np.linalg.norm(self.data.qpos[:3] - self.ref["qpos"][self.index, :3])),
            "joint_rms_error": float(np.sqrt(np.mean((self.data.qpos[7:] - self.ref["qpos"][self.index, 7:]) ** 2))),
            "torque_rms_error": float(np.sqrt(np.mean((total_torque.mean(axis=0) - self.ref["tau"][k]) ** 2))),
            "min_pelvis_height": min_height, "reward_components": components,
        }
        if equation_samples:
            info["equations"] = {key: np.stack([sample[key] for sample in equation_samples])
                                 for key in equation_samples[0]}
        return self.observation(), float(np.mean(list(components.values()))), terminated, truncated, info
