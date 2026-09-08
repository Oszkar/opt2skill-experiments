"""Squat task parameters and the smooth CoM reference profile the optimizer tracks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SquatParams:
    depth: float                 # crouch depth in metres (CoM drop)
    t_stand0: float = 0.5        # standing before the squat
    t_down: float = 1.0
    t_hold: float = 0.4
    t_up: float = 1.0
    t_stand1: float = 0.5        # standing after the squat
    com_shift_x: float = 0.0     # fore-aft CoM shift at the bottom (m)
    dt: float = 0.02

    @property
    def total_time(self) -> float:
        return self.t_stand0 + self.t_down + self.t_hold + self.t_up + self.t_stand1

    def num_nodes(self) -> int:
        return int(round(self.total_time / self.dt))


def min_jerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def time_grid(params: SquatParams) -> np.ndarray:
    return np.arange(params.num_nodes() + 1) * params.dt


def bump(params: SquatParams, t: np.ndarray) -> np.ndarray:
    """0 while standing, rises to 1 over t_down, holds, falls back to 0 over t_up."""
    t = np.asarray(t, dtype=float)
    t1 = params.t_stand0
    t2 = t1 + params.t_down
    t3 = t2 + params.t_hold
    t4 = t3 + params.t_up
    out = np.zeros_like(t)
    down = (t >= t1) & (t < t2)
    hold = (t >= t2) & (t < t3)
    up = (t >= t3) & (t < t4)
    out[down] = min_jerk((t[down] - t1) / params.t_down)
    out[hold] = 1.0
    out[up] = 1.0 - min_jerk((t[up] - t3) / params.t_up)
    return out


def com_reference(params: SquatParams, com0: np.ndarray) -> np.ndarray:
    b = bump(params, time_grid(params))
    ref = np.tile(np.asarray(com0, dtype=float), (b.size, 1))
    ref[:, 0] += params.com_shift_x * b
    ref[:, 2] -= params.depth * b
    return ref
