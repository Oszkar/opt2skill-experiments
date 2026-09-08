"""Unitree G1 model loading for both simulators, driven by configs/g1_reconcile.yaml."""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "g1_reconcile.yaml"

NJ = 29   # actuated joints
NQ = 36   # 3 pos + 4 quat + 29
NV = 35   # 6 + 29
PELVIS = "pelvis"
SOLE_FRAMES = ("left_sole", "right_sole")
PALM_FRAMES = ("left_palm", "right_palm")
FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
HAND_LINKS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def third_party_dir(cfg: dict) -> Path:
    env = cfg["assets"]["third_party_dir_env"]
    return Path(os.environ.get(env, Path.home() / "o2s_third_party")).expanduser()


def joint_names(cfg: dict) -> list[str]:
    return list(cfg["joints"])


def effort_limits(cfg: dict) -> np.ndarray:
    return np.asarray(cfg["effort_limits"], dtype=float)


def _mjcf_assets(scene_path: Path, cfg: dict) -> dict[str, bytes]:
    """Mirror MuJoCo Playground's get_assets(): Playground xmls + Menagerie unitree_g1 files."""
    xml_dir = scene_path.parent
    menagerie = third_party_dir(cfg) / "mujoco_menagerie" / cfg["assets"]["menagerie"]["subdir"]
    assets: dict[str, bytes] = {}
    for d in (xml_dir, xml_dir / "assets", menagerie, menagerie / "assets"):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file():
                assets[f.name] = f.read_bytes()
    return assets


def load_mj_model(cfg: dict | None = None) -> mujoco.MjModel:
    cfg = cfg or load_config()
    scene = third_party_dir(cfg) / cfg["mjcf_scene"]
    model = mujoco.MjModel.from_xml_string(scene.read_text(encoding="utf-8"), _mjcf_assets(scene, cfg))
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if names != joint_names(cfg):
        raise ValueError(f"MJCF joint order differs from config: {names}")
    lim = effort_limits(cfg)
    model.jnt_actfrclimited[1:] = 1
    model.jnt_actfrcrange[1:, 0] = -lim
    model.jnt_actfrcrange[1:, 1] = lim
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:, 0] = -lim
    model.actuator_forcerange[:, 1] = lim
    for jname, (lo, hi) in cfg.get("joint_range_overrides", {}).items():
        model.jnt_range[model.joint(jname).id] = (lo, hi)
        model.actuator_ctrlrange[model.actuator(jname).id] = (lo, hi)  # position actuators inherit the joint range
    return model


def _add_offset_frame(model: pin.Model, name: str, parent_link: str, offset: np.ndarray) -> int:
    parent_id = model.getFrameId(parent_link)
    parent = model.frames[parent_id]
    placement = parent.placement * pin.SE3(np.eye(3), np.asarray(offset, dtype=float))
    return model.addFrame(pin.Frame(name, parent.parentJoint, parent_id, placement, pin.OP_FRAME))


def load_pin_model(cfg: dict | None = None) -> pin.Model:
    cfg = cfg or load_config()
    urdf = third_party_dir(cfg) / cfg["urdf"]
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    names = list(model.names)[2:]
    if names != joint_names(cfg):
        raise ValueError(f"URDF joint order differs from config: {names}")
    fr = cfg["frames"]
    for sole, link in zip(SOLE_FRAMES, fr["foot_links"]):
        _add_offset_frame(model, sole, link, fr["sole_offset"])
    for palm, link in zip(PALM_FRAMES, fr["hand_links"]):
        _add_offset_frame(model, palm, link, fr["palm_offset"])
    model.effortLimit[6:] = effort_limits(cfg)
    return model


def home_qpos(mj_model: mujoco.MjModel, cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or load_config()
    key = mj_model.key(cfg["default_pose_keyframe"])
    return np.array(mj_model.key_qpos[key.id], dtype=float)
