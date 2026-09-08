"""Replay a reference in the MuJoCo viewer with reference keypoints drawn as a ghost (WSLg window)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from o2s.models import g1
from o2s.reference import contract, convert, validate

GHOST_KEYS = ["com", "pelvis_pos", "foot_pos_left", "foot_pos_right", "hand_pos_left", "hand_pos_right"]
COLORS = {"com": (1, 0.2, 0.2, 0.6), "pelvis_pos": (0.2, 0.2, 1, 0.5), "foot_pos_left": (0.2, 1, 0.2, 0.5),
          "foot_pos_right": (0.2, 1, 0.2, 0.5), "hand_pos_left": (1, 1, 0.2, 0.5), "hand_pos_right": (1, 1, 0.2, 0.5)}


def keypoints_world(ref: dict, k: int) -> dict[str, np.ndarray]:
    p = ref["pelvis_pos"][k]
    rot = convert.quat_wxyz_to_mat(ref["pelvis_quat"][k])
    out = {"com": ref["com"][k], "pelvis_pos": p}
    for key in GHOST_KEYS[2:]:
        out[key] = p + rot @ ref[key][k]
    return out


def draw_ghost(viewer, ref: dict, k: int) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0
    for key, pos in keypoints_world(ref, k).items():
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]), pos, np.eye(3).flatten(), np.array(COLORS[key], dtype=np.float32))
        scn.ngeom += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, nargs="?", default=Path("data/refs/dev/squat_dev.npz"))
    ap.add_argument("--mode", choices=["kinematic", "replay"], default="kinematic")
    ap.add_argument("--kp-scale", type=float, default=4.0)
    ap.add_argument("--ankle-pitch-kp", type=float, default=400.0)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    cfg = g1.load_config()
    model = g1.load_mj_model(cfg)
    data = mujoco.MjData(model)
    ref, _ = contract.load(args.path)
    N = ref["tau"].shape[0]
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            validate.set_state(model, data, ref["qpos"][0], ref["qvel"][0])
            for k in range(N + 1 if args.mode == "kinematic" else N):
                if not viewer.is_running():
                    break
                t0 = time.time()
                if args.mode == "kinematic":
                    data.qpos[:] = ref["qpos"][k]
                    data.qvel[:] = ref["qvel"][k]
                    mujoco.mj_forward(model, data)
                else:
                    with validate.replay_gains(model, args.kp_scale, args.ankle_pitch_kp):
                        validate.step_interval(model, data, ref["tau"][k], ref["qpos"][k + 1, 7:])
                # Replay has advanced to state k+1; kinematic mode displays state k.
                state_index = k if args.mode == "kinematic" else k + 1
                draw_ghost(viewer, ref, state_index)
                viewer.sync()
                time.sleep(max(0.0, contract.DT - (time.time() - t0)))
            if not args.loop:
                time.sleep(1.0)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
