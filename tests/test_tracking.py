from pathlib import Path
import csv
import json
import sys

import mujoco
import numpy as np
import pytest

from o2s.reference import contract
from o2s.tracking.env import TrackingEnv, tracking_rewards
from o2s.tracking.inspect import run_episode, write_outputs
from tests.conftest import requires_assets

pytestmark = requires_assets


@pytest.fixture(scope="module")
def reference_path(tmp_path_factory, mj_model, pin_model, cfg):
    from o2s.trajopt import export
    from o2s.trajopt.profile import SquatParams
    from o2s.trajopt.squat_problem import solve_squat
    sol = solve_squat(pin_model, mj_model, cfg, SquatParams(depth=.15))
    assert sol.converged
    ref = export.solution_to_reference(sol, pin_model, mj_model, cfg)
    path = tmp_path_factory.mktemp("tracking") / "reference.npz"
    from o2s.trajopt.filter import check_solution
    contract.save(path, ref, export.solution_meta(sol, check_solution(sol, ref, pin_model, cfg), cfg))
    return path


@pytest.fixture
def env(reference_path):
    return TrackingEnv(reference_path)


def test_reset_and_step_are_deterministic_and_observations_are_copies(env):
    first, initial = env.reset()
    assert initial["time"] == 0 and initial["state_index"] == 0
    assert not any("torque" in key or "contact" in key for key in first)
    action = first["reference_joint_position"].copy()
    first["joint_position"][:] = 99
    first_result = env.step(action)
    env.reset()
    second_result = env.step(action)
    np.testing.assert_array_equal(first_result[4]["qpos"], second_result[4]["qpos"])
    np.testing.assert_array_equal(first_result[4]["substep_torque"], second_result[4]["substep_torque"])
    np.testing.assert_allclose(env.data.qfrc_applied, 0)
    assert second_result[4]["interval_index"] == 0
    assert second_result[4]["state_index"] == 1
    assert second_result[4]["time"] == pytest.approx(.02)
    np.testing.assert_allclose(second_result[4]["substep_start_time"], np.arange(10) * .002)
    np.testing.assert_array_equal(second_result[4]["reference_torque"], env.ref["tau"][0])
    np.testing.assert_array_equal(second_result[0]["reference_joint_velocity"], env.ref["qvel"][1, 6:])


def test_reward_perfect_match_and_end_state_index(env):
    for k in (0, env.length):
        values = tracking_rewards(env.ref["qpos"][k], env.ref["qvel"][k], env.ref, k)
        assert list(values.values()) == pytest.approx([1, 1, 1, 1])
    q = env.ref["qpos"][0].copy()
    q[3:7] *= -1
    assert tracking_rewards(q, env.ref["qvel"][0], env.ref, 0)["pelvis_orientation"] == pytest.approx(1)
    q[7] += .2
    assert tracking_rewards(q, env.ref["qvel"][0], env.ref, 0)["joint_position"] == pytest.approx(np.exp(-5 * .2**2))
    env.reset()
    _, reward, _, _, info = env.step(np.zeros(29))
    expected = tracking_rewards(info["qpos"], info["qvel"], env.ref, 1)
    assert info["reward_components"] == expected
    assert reward == pytest.approx(np.mean(list(expected.values())))


def test_action_clipping_and_real_actuator_saturation(env):
    env.reset()
    _, _, _, _, info = env.step(np.full(29, 100.))
    np.testing.assert_array_equal(info["applied_target"], env.target_high)
    assert info["target_clipped"].all()
    assert info["saturated_fraction"] > 0
    assert info["peak_effort_ratio"] <= 1 + 1e-6
    assert np.max(np.abs(info["substep_requested_torque"]) / env.effort_limits) > 1
    np.testing.assert_allclose(info["substep_torque"],
                               np.clip(info["substep_requested_torque"], -env.effort_limits, env.effort_limits), atol=1e-9)


def test_contact_diagnostics_and_physics_not_pose_animation(env):
    obs, _ = env.reset()
    _, _, _, _, info = env.step(obs["reference_joint_position"])
    assert info["substep_contact_count"].shape == (10, 2)
    assert np.all(info["substep_contact_count"][0] > 0)
    assert np.all(info["substep_contact_normal"][0] > 0)
    assert not np.array_equal(info["qpos"], env.ref["qpos"][1])


@pytest.mark.parametrize("action", [np.zeros(28), np.full(29, np.nan), np.full(29, np.inf)])
def test_invalid_action_does_not_advance(env, action):
    env.reset()
    with pytest.raises(ValueError, match="29 finite"):
        env.step(action)
    assert env.index == 0 and env.data.time == 0


def test_terminal_state_has_no_torque_indexing(reference_path, tmp_path):
    ref, meta = contract.load(reference_path)
    short = {k: v[:2 if k in contract.STATE_KEYS else 1] for k, v in ref.items()}
    path = tmp_path / "short.npz"
    contract.save(path, short, meta)
    env = TrackingEnv(path)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(29))
    obs, _ = env.reset()
    obs, _, terminated, truncated, info = env.step(obs["reference_joint_position"])
    assert not terminated and truncated and info["reason"] == "reference_end"
    assert obs["phase"][0] == 1
    assert info["reference_torque"].shape == (29,)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(29))


def test_fall_terminates_and_can_reset(env):
    env.reset()
    env.data.qpos[2] = .3
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, truncated, info = env.step(np.zeros(29))
    assert terminated and not truncated and info["reason"] == "fall"
    with pytest.raises(RuntimeError):
        env.step(np.zeros(29))
    env.reset()
    assert env.index == 0 and env.data.time == 0


def test_trace_roundtrip_and_plot(env, tmp_path):
    initial, records = run_episode(env, "reference")
    summary = write_outputs(tmp_path, env, initial, records, "reference")
    n = summary["steps"]
    assert summary["reason"] in ("fall", "reference_end")
    with np.load(tmp_path / "trace.npz") as trace:
        assert trace["qpos"].shape == (n + 1, 36)
        assert trace["substep_torque"].shape == (n, 10, 29)
        assert trace["reference_torque"].shape == (n, 29)
        assert trace["observation_joint_position"].shape == (n + 1, 29)
        assert trace["observation_phase"][-1, 0] == pytest.approx(n / env.length)
        np.testing.assert_array_equal(trace["reference_torque"], env.ref["tau"][:n])
        np.testing.assert_allclose(trace["time"], np.arange(n + 1) * .02)
    assert len(list(csv.DictReader((tmp_path / "steps.csv").open()))) == n
    assert json.loads((tmp_path / "summary.json").read_text())["reason"] == summary["reason"]
    assert (tmp_path / "diagnostics.png").read_bytes().startswith(b"\x89PNG")



def test_cli_refuses_existing_run_before_loading(tmp_path, monkeypatch):
    from o2s.tracking import inspect
    marker = tmp_path / "summary.json"
    marker.write_text("preserve")
    monkeypatch.setattr(sys, "argv", ["inspect", "missing.npz", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        inspect.main()
    assert exc.value.code == 2
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("controller", ["stabilized-pd", "stabilized-ff"])
def test_stabilized_controller_matches_validator(env, controller):
    from o2s.reference.validate import check_replay, replay_gains, set_state, step_interval
    gain = env.model.actuator_gainprm.copy()
    initial, records = run_episode(env, controller)
    np.testing.assert_array_equal(env.model.actuator_gainprm, gain)
    ff_enabled = controller == "stabilized-ff"
    data = mujoco.MjData(env.model)
    set_state(env.model, data, env.ref["qpos"][0], env.ref["qvel"][0])
    with replay_gains(env.model, 4., 400.):
        np.testing.assert_array_equal(initial["kp"], env.model.actuator_gainprm[:, 0])
        for k, record in enumerate(records):
            ff = env.ref["tau"][k] if ff_enabled else np.zeros(29)
            total = np.empty((env.substeps, 29))
            step_interval(env.model, data, ff, env.ref["qpos"][k + 1, 7:], substep_torques=total)
            np.testing.assert_allclose(record["qpos"], data.qpos, atol=1e-7)
            np.testing.assert_allclose(record["substep_total_torque"], total, atol=1e-5)
            np.testing.assert_allclose(record["substep_total_torque"], record["substep_torque"] + ff)
    if ff_enabled:
        report = check_replay(env.model, env.ref, env.cfg)
        assert report.ok and records[-1]["reason"] == "reference_end"
        assert max(r["peak_effort_ratio"] for r in records) == pytest.approx(report.metrics["peak_effort_ratio"], abs=1e-6)
    else:
        assert records[-1]["reason"] == "fall"


def test_external_feedforward_limit_is_observed_not_hidden(env):
    obs, _ = env.reset()
    _, _, _, _, info = env.step(obs["reference_joint_position"], feedforward=env.effort_limits * 3)
    assert not info["effort_ok"]
    assert info["peak_effort_ratio"] > 1
    env.reset()
    _, _, _, _, info = env.step(obs["reference_joint_position"])
    np.testing.assert_array_equal(info["feedforward_torque"], np.zeros(29))


@pytest.mark.parametrize("ff", [np.zeros(28), np.full(29, np.nan)])
def test_invalid_feedforward_does_not_advance(env, ff):
    obs, _ = env.reset()
    with pytest.raises(ValueError, match="feedforward"):
        env.step(obs["reference_joint_position"], feedforward=ff)
    assert env.data.time == 0 and env.index == 0



def test_equations_preserve_simulation_and_export(reference_path, tmp_path, capsys):
    plain = TrackingEnv(reference_path)
    observed = TrackingEnv(reference_path, equations=True)
    _, expected = run_episode(plain, "stabilized-ff")
    initial, records = run_episode(observed, "stabilized-ff")
    np.testing.assert_array_equal([r["qpos"] for r in records], [r["qpos"] for r in expected])
    np.testing.assert_array_equal([r["substep_total_torque"] for r in records],
                                  [r["substep_total_torque"] for r in expected])
    obj = observed.equations.objective
    assert obj["total"] == pytest.approx(observed.meta["solver"]["cost"], rel=1e-10)
    np.testing.assert_allclose(obj["raw_u"] + observed.model.dof_damping[6:] * .5 *
                               (observed.ref["qvel"][:-1, 6:] + observed.ref["qvel"][1:, 6:]), observed.ref["tau"])
    write_outputs(tmp_path, observed, initial, records, "stabilized-ff")
    assert (tmp_path / "equations.png").read_bytes().startswith(b"\x89PNG")
    with np.load(tmp_path / "equations.npz") as a:
        assert a["sole_velocity"].shape == (observed.length, 10, 2, 6)
        assert a["inertia"].shape == (observed.length, 10, 35)
        assert a["running_cost"].sum() + a["terminal_cost"].sum() == pytest.approx(obj["total"])
    log = capsys.readouterr().out
    assert "2a planned" in log and "2b at" in log and "2c sole speed" in log


def test_equation_force_projection_and_contact_derivative(reference_path):
    env = TrackingEnv(reference_path, equations=True)
    env.reset()
    env.data.qpos[:] = env.ref["qpos"][45]
    env.data.qvel[:] = env.ref["qvel"][45]
    # Accurate scratch dynamics makes the balance test independent of iteration truncation.
    env.model.opt.iterations = 100
    env.model.opt.ls_iterations = 100
    r = env.equations.sample(env.data)
    m, d = env.model, env.equations.data
    assert np.max(np.abs(r["balance_residual"])) < 1e-7
    contact_forces = d.efc_force.copy()
    contact_forces[d.efc_type < int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0
    projected = np.zeros(m.nv)
    mujoco.mj_mulJacTVec(m, d, projected, contact_forces)
    np.testing.assert_allclose(r["contact"], projected, atol=1e-8)
    # Independently approximate dJ/dt by perturbing configuration along v.
    eps = 1e-6
    jacobians = []
    body = env.equations.soles[0]
    for sign in (-1, 1):
        perturbed = mujoco.MjData(m)
        mujoco.mj_copyData(perturbed, m, d)
        mujoco.mj_integratePos(m, perturbed.qpos, d.qvel, sign * eps)
        mujoco.mj_forward(m, perturbed)
        point = perturbed.xpos[body] + perturbed.xmat[body].reshape(3, 3) @ env.equations.offset
        jac = np.zeros((6, m.nv))
        mujoco.mj_jac(m, perturbed, jac[:3], jac[3:], point, body)
        jacobians.append(jac)
    derivative = (jacobians[1] - jacobians[0]) / (2 * eps)
    np.testing.assert_allclose(r["sole_jdot_v"][0], derivative @ d.qvel, atol=1e-7)


def test_equation_cost_rejects_incompatible_provenance(reference_path):
    from o2s.trajopt.diagnostics import objective_components
    env = TrackingEnv(reference_path)
    meta = dict(env.meta, weights={**env.meta["weights"], "control_reg": 100})
    with pytest.raises(ValueError, match="differs from saved cost"):
        objective_components(env.ref, meta, env.model, env.cfg)
