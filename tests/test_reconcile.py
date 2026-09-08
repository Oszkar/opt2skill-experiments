import numpy as np

from o2s.models import reconcile
from tests.conftest import requires_assets


@requires_assets
def test_body_index_map_covers_all_moving_bodies(mj_model, pin_model):
    m = reconcile.body_index_map(mj_model, pin_model)
    assert len(m) == 30  # pelvis + 29 links
    assert m["pelvis"] == 1
    assert m["left_ankle_roll_link"] == pin_model.getJointId("left_ankle_roll_joint")
    assert m["torso_link"] == pin_model.getJointId("waist_pitch_joint")  # names do not follow _link/_joint here


@requires_assets
def test_random_state_is_valid(pin_model):
    rng = np.random.default_rng(0)
    q, v = reconcile.random_state(pin_model, rng)
    assert q.shape == (36,) and v.shape == (35,)
    assert abs(np.linalg.norm(q[3:7]) - 1.0) < 1e-12
    assert np.all(q[7:] >= pin_model.lowerPositionLimit[7:]) and np.all(q[7:] <= pin_model.upperPositionLimit[7:])


@requires_assets
def test_all_reconciliation_checks_pass(mj_model, pin_model, cfg):
    results = reconcile.run_checks(mj_model, pin_model, cfg)
    failed = [r for r in results if not r.ok]
    assert not failed, "\n".join(f"{r.name}: {r.detail}" for r in failed)
    assert {r.name for r in results} == {
        "joint_names", "joint_limits", "total_mass", "body_mass", "body_com", "body_inertia",
        "forward_kinematics", "transmission", "foot_geometry", "actuator_type",
    }
