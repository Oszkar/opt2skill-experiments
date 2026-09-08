import numpy as np
import pinocchio as pin

from o2s.trajopt.profile import SquatParams
from o2s.trajopt import squat_problem as sp
from tests.conftest import requires_assets


@requires_assets
def test_standing_state_is_home_pose_with_soles_on_floor(pin_model, mj_model):
    x0 = sp.standing_state(pin_model, mj_model)
    assert x0.shape == (71,)
    np.testing.assert_allclose(x0[3:7], [0, 0, 0, 1])  # xyzw identity
    np.testing.assert_allclose(x0[7:36], mj_model.key("home").qpos[7:])
    assert np.all(x0[36:] == 0)
    assert 0.785 < x0[2] < 0.79  # lifted by the 1.2 mm keyframe penetration
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, x0[:36])
    pin.updateFramePlacements(pin_model, data)
    for name in sp.g1.SOLE_FRAMES:
        assert abs(data.oMf[pin_model.getFrameId(name)].translation[2]) < 1e-9


@requires_assets
def test_short_squat_solves_and_is_physically_consistent(pin_model, mj_model, cfg):
    params = SquatParams(depth=0.12, t_stand0=0.3, t_down=0.6, t_hold=0.2, t_up=0.6, t_stand1=0.3)
    sol = sp.solve_squat(pin_model, mj_model, cfg, params, max_iter=200)
    N = params.num_nodes()
    assert sol.converged, f"not converged after {sol.iters} iterations"
    assert sol.xs.shape == (N + 1, 71) and sol.us.shape == (N, 29)
    assert sol.forces["left"].shape == (N, 6) and sol.forces["right"].shape == (N, 6)
    # torque limits are hard box constraints
    lim = np.asarray(cfg["effort_limits"], dtype=float)
    assert np.max(np.abs(sol.us) / lim) <= 1.0 + 1e-6
    # at the first node the robot is (nearly) static: total normal force ~ weight, forces point up
    mg = pin.computeTotalMass(pin_model) * 9.81
    fz0 = sol.forces["left"][0, 2] + sol.forces["right"][0, 2]
    assert abs(fz0 - mg) / mg < 0.05, f"fz0={fz0:.1f} mg={mg:.1f}"
    assert np.all(sol.forces["left"][:, 2] > 0) and np.all(sol.forces["right"][:, 2] > 0)
    # reached the bottom: min CoM height within 1.5 cm of the target
    data = pin_model.createData()
    z = np.array([pin.centerOfMass(pin_model, data, x[:36])[2] for x in sol.xs])
    assert abs(z.min() - (sol.com_ref[0, 2] - params.depth)) < 0.015
    assert abs(z[-1] - sol.com_ref[0, 2]) < 0.015

    print(f"iters={sol.iters} solve_time={sol.solve_time:.3f}s cost={sol.cost:.3f}")
    print(f"max_torque_ratio={np.max(np.abs(sol.us) / lim):.4f}")
    print(f"fz0={fz0:.2f} mg={mg:.2f}")
    print(f"com_z_min={z.min():.4f} com_z_final={z[-1]:.4f} com_ref0_z={sol.com_ref[0, 2]:.4f}")
