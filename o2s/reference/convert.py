"""State conversions between Pinocchio and MuJoCo conventions (pure numpy).

Pinocchio free-flyer: q = [pos(3), quat xyzw(4), joints], v = [lin vel LOCAL(3), ang vel LOCAL(3), qdot].
MuJoCo free joint:    qpos = [pos(3), quat wxyz(4), joints], qvel = [lin vel WORLD(3), ang vel LOCAL(3), qdot].
"""
from __future__ import annotations

import numpy as np


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float)[[1, 2, 3, 0]]


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=float)[[3, 0, 1, 2]]


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pin_to_mj(q_pin: np.ndarray, v_pin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_pin = np.asarray(q_pin, dtype=float)
    v_pin = np.asarray(v_pin, dtype=float)
    quat_wxyz = quat_xyzw_to_wxyz(q_pin[3:7])
    rot = quat_wxyz_to_mat(quat_wxyz)
    qpos = np.concatenate([q_pin[:3], quat_wxyz, q_pin[7:]])
    qvel = np.concatenate([rot @ v_pin[:3], v_pin[3:6], v_pin[6:]])
    return qpos, qvel


def mj_to_pin(qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=float)
    qvel = np.asarray(qvel, dtype=float)
    rot = quat_wxyz_to_mat(qpos[3:7])
    q_pin = np.concatenate([qpos[:3], quat_wxyz_to_xyzw(qpos[3:7]), qpos[7:]])
    v_pin = np.concatenate([rot.T @ qvel[:3], qvel[3:6], qvel[6:]])
    return q_pin, v_pin


def pin_x_to_mj(x: np.ndarray, nq: int = 36) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    return pin_to_mj(x[:nq], x[nq:])
