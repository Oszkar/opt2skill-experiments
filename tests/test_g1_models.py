import numpy as np
import pinocchio as pin

from o2s.models import g1
from tests.conftest import requires_assets


def test_config_lists_29_joints_and_limits(cfg):
    names = g1.joint_names(cfg)
    assert len(names) == 29
    assert names[0] == "left_hip_pitch_joint"
    assert names[-1] == "right_wrist_yaw_joint"
    lim = g1.effort_limits(cfg)
    assert lim.shape == (29,)
    # ankle pitch/roll and waist roll/pitch are the min(URDF, MJCF) = 35
    assert lim[4] == 35.0 and lim[5] == 35.0 and lim[13] == 35.0 and lim[14] == 35.0
    assert lim[1] == 139.0 and lim[15] == 25.0 and lim[20] == 5.0


@requires_assets
def test_mj_model_dimensions_and_limits(mj_model, cfg):
    assert (mj_model.nq, mj_model.nv, mj_model.nu) == (36, 35, 29)
    assert [mj_model.joint(i).name for i in range(1, mj_model.njnt)] == g1.joint_names(cfg)
    lim = g1.effort_limits(cfg)
    np.testing.assert_allclose(mj_model.jnt_actfrcrange[1:, 1], lim)
    np.testing.assert_allclose(mj_model.jnt_actfrcrange[1:, 0], -lim)
    np.testing.assert_allclose(mj_model.actuator_forcerange[:, 1], lim)
    assert mj_model.actuator_forcelimited.all()
    assert mj_model.key("home").id >= 0
    # Playground's shared hip_roll class bug is corrected from the config (URDF governs)
    jid = mj_model.joint("right_hip_roll_joint").id
    np.testing.assert_allclose(mj_model.jnt_range[jid], [-2.9671, 0.5236])
    np.testing.assert_allclose(mj_model.actuator_ctrlrange[mj_model.actuator("right_hip_roll_joint").id], [-2.9671, 0.5236])


@requires_assets
def test_pin_model_dimensions_frames_and_limits(pin_model, cfg):
    assert (pin_model.nq, pin_model.nv) == (36, 35)
    assert list(pin_model.names)[2:] == g1.joint_names(cfg)
    for name in g1.SOLE_FRAMES + g1.PALM_FRAMES:
        assert pin_model.existFrame(name), name
    np.testing.assert_allclose(pin_model.effortLimit[6:], g1.effort_limits(cfg))
    # sole frame sits 0.037 m below and 0.04 m ahead of the ankle-roll link origin
    data = pin_model.createData()
    q = pin.neutral(pin_model)
    pin.forwardKinematics(pin_model, data, q)
    pin.updateFramePlacements(pin_model, data)
    ankle = data.oMf[pin_model.getFrameId("left_ankle_roll_link")]
    sole = data.oMf[pin_model.getFrameId("left_sole")]
    np.testing.assert_allclose(ankle.actInv(sole).translation, [0.04, 0.0, -0.037], atol=1e-9)


@requires_assets
def test_home_qpos(mj_model):
    q = g1.home_qpos(mj_model)
    assert q.shape == (36,)
    assert abs(q[2] - 0.785) < 1e-9
    np.testing.assert_allclose(q[3:7], [1, 0, 0, 0])
