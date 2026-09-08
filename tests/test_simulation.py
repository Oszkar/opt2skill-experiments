import copy
import json

import numpy as np
import pytest

from o2s.models import g1
from o2s.models.simulation import replay_gains, snapshot
from tests.conftest import requires_assets

pytestmark = requires_assets


def test_configuration_controls_loaded_model_and_replay(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["simulation"].update(iterations=31, tolerance=1e-9)
    cfg["controllers"]["position"]["kp"][0] = 81
    cfg["controllers"]["position"]["kd"] = .25
    cfg["controllers"]["stabilized"].update(kp_scale=2, ankle_pitch_kp=180)
    model = g1.load_mj_model(cfg)
    assert model.opt.iterations == 31 and model.opt.tolerance == 1e-9
    assert model.actuator_gainprm[0, 0] == 81
    np.testing.assert_array_equal(-model.actuator_biasprm[:, 2], np.full(29, .25))
    original = snapshot(model, cfg)
    with pytest.raises(RuntimeError):
        with replay_gains(model, cfg=cfg):
            snap = snapshot(model, cfg)
            assert snap["kp"][0] == 162
            assert snap["kp"][4] == 180
            assert snap["kd"] == original["kd"]
            assert snap["joint_damping"] == original["joint_damping"]
            json.dumps(snap, allow_nan=False)
            raise RuntimeError("restore after failure")
    assert snapshot(model, cfg) == original


@pytest.mark.parametrize("key,value", [("iterations", 0), ("iterations", 1.5), ("timestep", -1), ("tolerance", float("nan"))])
def test_invalid_physics_config_is_rejected(cfg, key, value):
    cfg = copy.deepcopy(cfg)
    cfg["simulation"][key] = value
    with pytest.raises(ValueError):
        g1.load_mj_model(cfg)


def test_invalid_replay_gain_leaves_model_unchanged(mj_model, cfg):
    before = snapshot(mj_model, cfg)
    with pytest.raises(ValueError):
        with replay_gains(mj_model, kp_scale=float("nan"), cfg=cfg):
            pass
    assert snapshot(mj_model, cfg) == before
