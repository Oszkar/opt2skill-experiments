"""The reference trajectory contract: which arrays an npz holds, their shapes, frames, and timing.

A trajectory has N control intervals of DT seconds, N+1 state samples and N control samples.
  qpos (N+1, 36): MuJoCo layout, base pos world, quat wxyz, 29 joints in MuJoCo order.      time t[k]
  qvel (N+1, 35): base lin vel WORLD, base ang vel pelvis-LOCAL, 29 joint velocities.        time t[k]
  tau  (N, 29):   torque MuJoCo's actuators must output, zero-order hold over [t[k], t[k+1]).
  foot_wrench_{left,right} (N, 6): [fx fy fz tx ty tz], world-aligned axes at the sole frame origin, held over the interval.
  com, pelvis_pos (N+1, 3) world; pelvis_quat (N+1, 4) wxyz; pelvis_linvel, pelvis_angvel (N+1, 3) pelvis-LOCAL.
  foot_pos_*, hand_pos_* (N+1, 3): ankle-roll-link origins and palm points, pelvis-LOCAL.

Timing rule: at control step k the policy observes state k and reference k, and acts over
[t[k], t[k+1]); the torque reward compares the substep-mean applied joint torque over that
interval with tau[k]. State N is truncation-only -- never index tau or foot_wrench_* at N,
they have only N rows.

Two asymmetries worth knowing before using this data:
- foot_pos_{left,right} is the ankle-roll-link *origin*, not the sole frame where
  foot_wrench_{left,right} acts; the sole is origin + (0.04, 0, -0.037) in that link
  (`configs/g1_reconcile.yaml`'s frames.sole_offset).
- pelvis_angvel duplicates qvel[3:6] (both are the base angular velocity, pelvis-local), but
  pelvis_linvel is pelvis-local while qvel[0:3] is the base linear velocity in world axes --
  qvel follows MuJoCo's own free-joint layout, pelvis_linvel follows the policy observation
  convention.

tau includes MuJoCo's joint damping (dof_damping * qvel, interval-mean; see export/validate for
the exact convention); it does not include `frictionloss` (0.1 N m per joint), which stays
unmodeled by the trajectory optimizer and is not compensated at export.

foot_wrench_* is documented as world-aligned axes at the sole frame origin (Crocoddyl's
LOCAL_WORLD_ALIGNED). That convention is asserted by construction -- the contact model is created
with LOCAL_WORLD_ALIGNED, not measured after the fact -- and is untested for a foot that is not
flat on the ground: in every squat in this dataset the sole stays within 1e-6 rad of
world-aligned, so a rotated foot (uneven terrain, foot roll during a step) has not exercised this
frame conversion and its correctness there is unverified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DT = 0.02
NJ = 29

ARRAY_SPEC: dict[str, tuple] = {
    "t": ("N+1",),
    "qpos": ("N+1", 36),
    "qvel": ("N+1", 35),
    "tau": ("N", NJ),
    "foot_wrench_left": ("N", 6),
    "foot_wrench_right": ("N", 6),
    "com": ("N+1", 3),
    "pelvis_pos": ("N+1", 3),
    "pelvis_quat": ("N+1", 4),
    "pelvis_linvel": ("N+1", 3),
    "pelvis_angvel": ("N+1", 3),
    "foot_pos_left": ("N+1", 3),
    "foot_pos_right": ("N+1", 3),
    "hand_pos_left": ("N+1", 3),
    "hand_pos_right": ("N+1", 3),
}
STATE_KEYS = tuple(k for k, s in ARRAY_SPEC.items() if s[0] == "N+1")
CONTROL_KEYS = tuple(k for k, s in ARRAY_SPEC.items() if s[0] == "N")


def validate(ref: dict) -> int:
    missing = [k for k in ARRAY_SPEC if k not in ref]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    N = int(ref["tau"].shape[0])
    for key, shape in ARRAY_SPEC.items():
        arr = np.asarray(ref[key])
        rows = N + 1 if shape[0] == "N+1" else N
        expected = (rows, *shape[1:])
        if arr.shape != expected:
            raise ValueError(f"{key}: shape {arr.shape} != {expected}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key}: contains non-finite values")
    return N


def save(path: str | Path, ref: dict, meta: dict) -> None:
    validate(ref)
    np.savez_compressed(Path(path), meta=np.array(json.dumps(meta)), **{k: np.asarray(ref[k], dtype=np.float64) for k in ARRAY_SPEC})


def load(path: str | Path) -> tuple[dict, dict]:
    with np.load(Path(path)) as z:
        missing = set(ARRAY_SPEC) - set(z.files)
        if missing:
            k = sorted(missing)[0]
            raise ValueError(f"{path}: missing key {k}")
        ref = {k: z[k] for k in ARRAY_SPEC}
        meta = json.loads(str(z["meta"]))
    validate(ref)
    return ref, meta


@dataclass
class ReferenceSet:
    names: list[str]
    lengths: np.ndarray            # N per trajectory
    arrays: dict[str, np.ndarray]  # padded: (M, max_N+1, ...) for state keys, (M, max_N, ...) for control keys
    metas: list[dict]


def _pad(arr: np.ndarray, rows: int, key: str | None = None) -> np.ndarray:
    """Pad `arr` up to `rows` by repeating its last row, except for `t`, which is extrapolated at a
    constant step of `DT` so that `t[k+1] - t[k]` never becomes 0 in the padded region (padded arrays
    still line up index-for-index with the un-padded ones, but a consumer that takes finite differences
    of `t` across the pad boundary would otherwise divide by zero)."""
    if arr.shape[0] == rows:
        return arr
    n_pad = rows - arr.shape[0]
    if key == "t":
        pad = arr[-1] + DT * np.arange(1, n_pad + 1)
    else:
        pad = np.repeat(arr[-1:], n_pad, axis=0)
    return np.concatenate([arr, pad], axis=0)


def load_dir(dir_path: str | Path, names: list[str] | None = None) -> ReferenceSet:
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob("*.npz"))
    if names is not None:
        wanted = set(names)
        files = [f for f in files if f.stem in wanted]
        missing = wanted - {f.stem for f in files}
        if missing:
            raise FileNotFoundError(f"reference names not found in {dir_path}: {sorted(missing)}")
    if not files:
        raise FileNotFoundError(f"no reference npz files in {dir_path}")
    refs, metas = zip(*(load(f) for f in files))
    lengths = np.array([validate(r) for r in refs], dtype=int)
    max_n = int(lengths.max())
    arrays = {}
    for key, shape in ARRAY_SPEC.items():
        rows = max_n + 1 if shape[0] == "N+1" else max_n
        arrays[key] = np.stack([_pad(r[key], rows, key) for r in refs])
    return ReferenceSet(names=[f.stem for f in files], lengths=lengths, arrays=arrays, metas=list(metas))
