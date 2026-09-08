import numpy as np
import pytest

from o2s.reference import contract, convert
from o2s.trajopt import export, filter as flt
from o2s.trajopt.profile import SquatParams
from o2s.trajopt import squat_problem as sp
from tests.conftest import requires_assets


@pytest.fixture(scope="module")
def solution(pin_model, mj_model, cfg):
    params = SquatParams(depth=0.15, t_stand0=0.4, t_down=0.8, t_hold=0.3, t_up=0.8, t_stand1=0.4)
    sol = sp.solve_squat(pin_model, mj_model, cfg, params, max_iter=300)
    assert sol.converged
    return sol


@requires_assets
def test_export_matches_contract_and_conventions(solution, pin_model, mj_model, cfg):
    ref = export.solution_to_reference(solution, pin_model, mj_model, cfg)
    N = contract.validate(ref)
    assert N == solution.params.num_nodes()
    # qpos/qvel come from the Pinocchio state through the documented conversion
    qpos0, qvel0 = convert.pin_x_to_mj(solution.xs[0])
    np.testing.assert_allclose(ref["qpos"][0], qpos0)
    np.testing.assert_allclose(ref["qvel"][0], qvel0)
    np.testing.assert_allclose(ref["pelvis_quat"][0], [1, 0, 0, 0], atol=1e-6)
    # tau is the Crocoddyl torque plus MuJoCo passive damping over the interval
    qd = solution.xs[:, 42:]  # joint velocities (36 + 6 = 42)
    expected = solution.us + mj_model.dof_damping[6:] * 0.5 * (qd[:-1] + qd[1:])
    np.testing.assert_allclose(ref["tau"], expected, atol=1e-12)
    # wrenches copied through
    np.testing.assert_allclose(ref["foot_wrench_left"], solution.forces["left"])
    # pelvis-local foot positions: feet are below the pelvis (negative z) and left is +y
    assert np.all(ref["foot_pos_left"][:, 2] < -0.5) and np.all(ref["foot_pos_left"][:, 1] > 0.05)
    assert np.all(ref["foot_pos_right"][:, 1] < -0.05)
    # time grid
    np.testing.assert_allclose(ref["t"], np.arange(N + 1) * contract.DT)


@requires_assets
def test_filter_accepts_good_solution(solution, pin_model, mj_model, cfg):
    ref = export.solution_to_reference(solution, pin_model, mj_model, cfg)
    res = flt.check_solution(solution, ref, pin_model, cfg)
    assert res.ok, res.failures
    assert res.details["max_torque_ratio"] <= 1.0 + 1e-6
    assert res.details["min_normal_force"] > 0
    assert res.details["depth_error"] < 0.01


@requires_assets
def test_filter_rejects_tampered_solution(solution, pin_model, mj_model, cfg):
    import copy

    ref = export.solution_to_reference(solution, pin_model, mj_model, cfg)
    bad = copy.deepcopy(solution)
    bad.forces["left"][:, 0] = 10 * bad.forces["left"][:, 2]  # violates friction cone
    res = flt.check_solution(bad, ref, pin_model, cfg)
    assert not res.ok and any("friction" in f for f in res.failures)
    bad = copy.deepcopy(solution)
    bad.converged = False
    res = flt.check_solution(bad, ref, pin_model, cfg)
    assert not res.ok and any("converged" in f for f in res.failures)


@requires_assets
@pytest.mark.parametrize("sign", [-1, 1])
def test_filter_rejects_exported_torque_above_limit(solution, pin_model, mj_model, cfg, sign):
    ref = export.solution_to_reference(solution, pin_model, mj_model, cfg)
    assert np.max(np.abs(solution.us) / np.asarray(cfg["effort_limits"])) < 1
    ref["tau"][0, 0] = sign * cfg["effort_limits"][0] * 1.01
    res = flt.check_solution(solution, ref, pin_model, cfg)
    assert not res.ok
    assert res.details["max_exported_torque_ratio"] == pytest.approx(1.01)
    assert any("exported torque" in failure for failure in res.failures)


@requires_assets
def test_damping_compensation_can_push_bounded_control_over_limit(solution, pin_model, mj_model, cfg):
    import copy
    bad = copy.deepcopy(solution)
    limit = cfg["effort_limits"][0]
    bad.us[0, 0] = 0.99 * limit
    bad.xs[:2, pin_model.nq + 6] = 0.02 * limit / mj_model.dof_damping[6]
    ref = export.solution_to_reference(bad, pin_model, mj_model, cfg)
    assert ref["tau"][0, 0] == pytest.approx(1.01 * limit)
    res = flt.check_solution(bad, ref, pin_model, cfg)
    assert res.details["max_torque_ratio"] < 1
    assert not res.ok and any("exported torque" in f for f in res.failures)
