from pathlib import Path

import mujoco
import numpy as np
import pytest

from o2s.reference import contract, validate
from tests.conftest import requires_assets

DEV = Path(__file__).resolve().parents[1] / "data" / "refs" / "dev" / "squat_dev.npz"


@pytest.fixture(scope="module")
def dev_ref(mj_model, pin_model, cfg):
    if DEV.exists():
        ref, _ = contract.load(DEV)
        return ref
    # No dev file on this machine: solve a short squat and validate it the same way the CLI would,
    # so the fixture never depends on an artifact that isn't checked in (data/ is git-ignored).
    from o2s.trajopt import export
    from o2s.trajopt import filter as flt
    from o2s.trajopt.profile import SquatParams
    from o2s.trajopt.squat_problem import solve_squat

    params = SquatParams(depth=0.15, t_stand0=0.4, t_down=0.8, t_hold=0.3, t_up=0.8, t_stand1=0.4)
    sol = solve_squat(pin_model, mj_model, cfg, params)
    ref = export.solution_to_reference(sol, pin_model, mj_model, cfg)
    res = flt.check_solution(sol, ref, pin_model, cfg)
    assert res.ok, res.failures
    return ref


@requires_assets
def test_set_state_puts_soles_on_floor(mj_model, dev_ref):
    data = mujoco.MjData(mj_model)
    validate.set_state(mj_model, data, dev_ref["qpos"][0], dev_ref["qvel"][0])
    np.testing.assert_allclose(data.qpos, dev_ref["qpos"][0])
    assert data.ncon >= 4  # both feet's contact boxes are on the floor: at least one corner per foot in flat contact
    # feet are on, not in, the floor: any contact reported has |dist| below 0.1 mm
    assert all(abs(c.dist) < 1e-4 for c in data.contact[: data.ncon])


@requires_assets
def test_step_interval_returns_mean_actuator_torque(mj_model, dev_ref):
    data = mujoco.MjData(mj_model)
    validate.set_state(mj_model, data, dev_ref["qpos"][0], dev_ref["qvel"][0])
    with validate.disabled(mj_model, mujoco.mjtDisableBit.mjDSBL_ACTUATION):
        mean_tau = validate.step_interval(mj_model, data, dev_ref["tau"][0], None)
    assert mean_tau.shape == (29,)
    np.testing.assert_allclose(mean_tau, 0.0)  # actuation disabled -> no actuator torque
    assert abs(data.time - 0.02) < 1e-12


@requires_assets
def test_replay_gains_scales_and_overrides_ankle_pitch(mj_model):
    kp_before = mj_model.actuator_gainprm[:, 0].copy()
    ankle = [mj_model.actuator("left_ankle_pitch_joint").id, mj_model.actuator("right_ankle_pitch_joint").id]
    with validate.replay_gains(mj_model, kp_scale=2.0, ankle_pitch_kp=200.0):
        kp = mj_model.actuator_gainprm[:, 0]
        assert np.all(kp[ankle] == 200.0)
        others = [i for i in range(mj_model.nu) if i not in ankle]
        np.testing.assert_allclose(kp[others], 2.0 * kp_before[others])
        np.testing.assert_allclose(mj_model.actuator_biasprm[:, 1], -kp)
    np.testing.assert_allclose(mj_model.actuator_gainprm[:, 0], kp_before)


@requires_assets
def test_check1_inverse_dynamics_consistency(mj_model, dev_ref, cfg):
    rep = validate.check_inverse_dynamics(mj_model, dev_ref, cfg)
    assert rep.ok, rep.message


@requires_assets
def test_check2_stabilized_replay_with_feedforward(mj_model, dev_ref, cfg):
    rep = validate.check_replay(mj_model, dev_ref, cfg, feedforward=True)
    assert rep.ok, rep.message


@requires_assets
def test_check3_pd_only_replay_is_informational(mj_model, dev_ref, cfg):
    rep = validate.check_replay(mj_model, dev_ref, cfg, feedforward=False)
    assert rep.ok  # informational: never gates
    assert "max_pelvis_error" in rep.metrics and "rms_joint_error" in rep.metrics


@requires_assets
def test_run_all_reports_three_checks(mj_model, dev_ref, cfg):
    reps = validate.run_all(mj_model, dev_ref, cfg)
    assert [r.name for r in reps] == ["inverse_dynamics", "replay_ff", "replay_pd_only"]


@requires_assets
@pytest.mark.parametrize("sign", [-1, 1])
def test_substep_torque_spike_cannot_hide_in_interval_mean(mj_model, dev_ref, cfg, monkeypatch, sign):
    # Keep real state integration and tracking, but inject one actuator-torque spike
    # into the sampled MuJoCo output. Its interval mean stays below the motor limit.
    original = mujoco.mj_step
    count = 0
    def spike(model, data):
        nonlocal count
        original(model, data)
        if count == 0:
            data.qfrc_actuator[6] = sign * 1.1 * model.jnt_actfrcrange[1, 1] - data.qfrc_applied[6]
        count += 1
    monkeypatch.setattr(mujoco, "mj_step", spike)
    report = validate.check_replay(mj_model, dev_ref, cfg)
    assert report.metrics["peak_interval_mean_effort_ratio"] < 1
    assert report.metrics["peak_effort_ratio"] > 1
    assert report.metrics["effort_violation_substeps"] == 1
    assert report.metrics["effort_ok"] == 0
    assert not report.ok


@requires_assets
def test_substep_log_records_feedforward_plus_actuation(mj_model, dev_ref):
    data = mujoco.MjData(mj_model)
    validate.set_state(mj_model, data, dev_ref["qpos"][0], dev_ref["qvel"][0])
    log = np.empty((validate.SUBSTEPS, mj_model.nu))
    tau = dev_ref["tau"][0]
    with validate.disabled(mj_model, mujoco.mjtDisableBit.mjDSBL_ACTUATION):
        mean_pd = validate.step_interval(mj_model, data, tau, None, substep_torques=log)
    np.testing.assert_allclose(mean_pd, 0)
    np.testing.assert_allclose(log, np.tile(tau, (validate.SUBSTEPS, 1)))


@requires_assets
def test_combined_feedforward_and_pd_can_exceed_limit(mj_model, dev_ref, monkeypatch):
    data = mujoco.MjData(mj_model)
    validate.set_state(mj_model, data, dev_ref["qpos"][0], dev_ref["qvel"][0])
    limits = mj_model.jnt_actfrcrange[1:, 1]
    original = mujoco.mj_step
    def pd_within_limit(model, data):
        original(model, data)
        data.qfrc_actuator[6:] = 0.6 * limits
    monkeypatch.setattr(mujoco, "mj_step", pd_within_limit)
    log = np.empty((validate.SUBSTEPS, mj_model.nu))
    validate.step_interval(mj_model, data, 0.6 * limits, None, substep_torques=log)
    np.testing.assert_allclose(log / limits, 1.2)


@requires_assets
def test_valid_time_roundoff_uses_contract_step(mj_model, dev_ref, cfg):
    ref = dict(dev_ref)
    ref["t"] = dev_ref["t"].copy()
    ref["t"][1] += 1e-10
    assert validate.check_inverse_dynamics(mj_model, ref, cfg).ok
    assert validate.check_replay(mj_model, ref, cfg).ok
