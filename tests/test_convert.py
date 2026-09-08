import mujoco
import numpy as np
import pinocchio as pin

from o2s.models import g1
from o2s.reference import convert
from tests.conftest import requires_assets


def test_pin_x_to_mj_default_nq_matches_g1():
    # convert must not import o2s.models.g1 (layering: models depends on nothing in reference,
    # not the other way around), so its default nq is a plain literal that has to be kept in
    # sync with g1.NQ by hand; this pins that agreement down as a test instead.
    assert convert.pin_x_to_mj.__defaults__[0] == g1.NQ


def test_quat_order_roundtrip():
    q = np.array([0.1, 0.2, 0.3, 0.4])
    q = q / np.linalg.norm(q)
    np.testing.assert_allclose(convert.quat_xyzw_to_wxyz(convert.quat_wxyz_to_xyzw(q)), q)
    np.testing.assert_allclose(convert.quat_wxyz_to_xyzw(np.array([1.0, 0, 0, 0])), [0, 0, 0, 1.0])


def test_quat_to_mat_matches_mujoco():
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        ref = np.zeros(9)
        mujoco.mju_quat2Mat(ref, q)
        np.testing.assert_allclose(convert.quat_wxyz_to_mat(q), ref.reshape(3, 3), atol=1e-12)


def test_state_roundtrip():
    rng = np.random.default_rng(2)
    q = rng.normal(size=36)
    q[3:7] /= np.linalg.norm(q[3:7])
    v = rng.normal(size=35)
    qpos, qvel = convert.pin_to_mj(q, v)
    q2, v2 = convert.mj_to_pin(qpos, qvel)
    np.testing.assert_allclose(q2, q, atol=1e-12)
    np.testing.assert_allclose(v2, v, atol=1e-12)
    np.testing.assert_allclose(qpos[7:], q[7:])
    np.testing.assert_allclose(qvel[6:], v[6:])


@requires_assets
def test_check0_kinematics_and_velocities_agree(mj_model, pin_model):
    """Spec validation check 0: rotated base + nonzero velocities, both libraries agree."""
    from o2s.models import reconcile

    rng = np.random.default_rng(3)
    data_mj = mujoco.MjData(mj_model)
    data_pin = pin_model.createData()
    frames = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    for _ in range(10):
        q, v = reconcile.random_state(pin_model, rng)
        qpos, qvel = convert.pin_to_mj(q, v)
        data_mj.qpos[:] = qpos
        data_mj.qvel[:] = qvel
        mujoco.mj_forward(mj_model, data_mj)
        pin.forwardKinematics(pin_model, data_pin, q, v)
        pin.updateFramePlacements(pin_model, data_pin)
        for f in frames:
            bid = mj_model.body(f).id
            fid = pin_model.getFrameId(f)
            np.testing.assert_allclose(data_mj.xpos[bid], data_pin.oMf[fid].translation, atol=1e-4)
            # mjOBJ_XBODY = velocity of the body frame origin (xpos); mjOBJ_BODY would give the inertial frame (xipos).
            vel6 = np.zeros(6)  # [ang(3), lin(3)], world axes
            mujoco.mj_objectVelocity(mj_model, data_mj, mujoco.mjtObj.mjOBJ_XBODY, bid, vel6, 0)
            m = pin.getFrameVelocity(pin_model, data_pin, fid, pin.LOCAL_WORLD_ALIGNED)
            np.testing.assert_allclose(vel6[3:], m.linear, atol=1e-3)
            np.testing.assert_allclose(vel6[:3], m.angular, atol=1e-3)
