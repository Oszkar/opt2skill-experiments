"""Whole-body squat trajectory optimization for the G1 with Crocoddyl (BoxFDDP, two-foot 6D contact)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import crocoddyl as croc
import mujoco
import numpy as np
import pinocchio as pin

from o2s.models import g1
from o2s.reference import convert
from o2s.trajopt.profile import SquatParams, com_reference


@dataclass
class SquatWeights:
    com: float = 1e5              # CoM tracking (running and terminal); 1e4 lets the solver smooth the standing phase (fz0 12% below weight)
    wrench_cone: float = 1e1      # friction + CoP barrier per foot
    state_reg: float = 1e0        # posture regularization, running
    state_reg_terminal: float = 1e1
    control_reg: float = 1e-3
    joint_limits: float = 1e5     # quadratic barrier on joint positions; 1e3 never binds against com=1e5 (0.044 rad excursion costs ~1.9)
    base_rotation: float = 100.0  # inside state_reg: keep pelvis upright
    legs: float = 1.0             # inside state_reg: joint position weights per group
    waist: float = 10.0
    arms: float = 10.0
    velocity: float = 1.0         # inside state_reg: all velocities
    velocity_terminal: float = 10.0
    baumgarte: tuple[float, float] = (100.0, 20.0)  # position gain 100, velocity gain 20 (tuned for foot drift and solver convergence)
    min_normal_force: float = 20.0
    max_normal_force: float = 1000.0
    joint_limit_margin: float = 0.04  # > the filter's 0.02 rad: the quadratic barrier always admits some penetration


@dataclass
class SquatSolution:
    params: SquatParams
    dt: float
    xs: np.ndarray
    us: np.ndarray
    forces: dict[str, np.ndarray]
    converged: bool
    iters: int
    cost: float
    solve_time: float
    com_ref: np.ndarray
    weights: SquatWeights = field(default_factory=SquatWeights)
    simulation: dict = field(default_factory=dict)


def armature_vector(mj_model: mujoco.MjModel) -> np.ndarray:
    return np.concatenate([np.zeros(6), np.asarray(mj_model.dof_armature[6:], dtype=float)])


def standing_state(pin_model: pin.Model, mj_model: mujoco.MjModel) -> np.ndarray:
    """`home` keyframe as a Pinocchio state, lifted so the sole frames sit exactly at z = 0.

    In the raw keyframe both soles are 1.2 mm below the floor; starting a MuJoCo replay from there makes the
    soft contact push the robot up. The TO does not care where the floor is (the contact frames are fixed),
    so the lift makes the exported reference start with the feet on the floor.
    """
    qpos = g1.home_qpos(mj_model)
    q, v = convert.mj_to_pin(qpos, np.zeros(mj_model.nv))
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, q)
    pin.updateFramePlacements(pin_model, data)
    sole_z = min(data.oMf[pin_model.getFrameId(name)].translation[2] for name in g1.SOLE_FRAMES)
    q[2] -= sole_z
    return np.concatenate([q, v])


def _state_weights(cfg: dict, w: SquatWeights, terminal: bool) -> np.ndarray:
    nj = g1.NJ
    joint_w = np.zeros(nj)
    joint_w[cfg["groups"]["legs"]] = w.legs
    joint_w[cfg["groups"]["waist"]] = w.waist
    joint_w[cfg["groups"]["arms"]] = w.arms
    vel_w = w.velocity_terminal if terminal else w.velocity
    return np.concatenate([np.zeros(3), np.full(3, w.base_rotation), joint_w, np.full(6 + nj, vel_w)])


def _node_model(pin_model, state, actuation, x0, com_target, sole_placements, cfg, w, effort, armature, dt, terminal, terminal_wrench_cost=False):
    nu = actuation.nu
    contacts = croc.ContactModelMultiple(state, nu)
    costs = croc.CostModelSum(state, nu)
    mu = float(cfg["contact"]["friction_mu"])
    box = 2.0 * np.asarray(cfg["frames"]["foot_half_size"], dtype=float)  # (length, width)
    for name in g1.SOLE_FRAMES:
        fid = pin_model.getFrameId(name)
        contacts.addContact(
            f"{name}_contact",
            croc.ContactModel6D(state, fid, sole_placements[name], pin.LOCAL_WORLD_ALIGNED, nu, np.asarray(w.baumgarte, dtype=float)),
        )
        cone = croc.WrenchCone(np.eye(3), mu, box, 4, True, w.min_normal_force, w.max_normal_force)
        residual = croc.ResidualModelContactWrenchCone(state, fid, cone, nu, True)
        activation = croc.ActivationModelQuadraticBarrier(croc.ActivationBounds(cone.lb, cone.ub))
        # No terminal control/contact wrench is evaluated. A wrench barrier here
        # would penalize a zero placeholder, adding a constant 4000 at default weights.
        if not terminal or terminal_wrench_cost:  # legacy objective reconstruction only
            costs.addCost(f"{name}_wrench", croc.CostModelResidual(state, activation, residual), w.wrench_cone)

    costs.addCost("com", croc.CostModelResidual(state, croc.ResidualModelCoMPosition(state, com_target, nu)), w.com)

    sw = _state_weights(cfg, w, terminal)
    costs.addCost(
        "state_reg",
        croc.CostModelResidual(state, croc.ActivationModelWeightedQuad(sw**2), croc.ResidualModelState(state, x0, nu)),
        w.state_reg_terminal if terminal else w.state_reg,
    )

    # Joint-limit barrier: residual = x (-) x_zero has plain joint positions in entries 6:35.
    x_zero = x0.copy()
    x_zero[7:36] = 0.0
    x_zero[36:] = 0.0
    inf = np.inf
    lb = np.concatenate([np.full(6, -inf), pin_model.lowerPositionLimit[7:] + w.joint_limit_margin, np.full(35, -inf)])
    ub = np.concatenate([np.full(6, inf), pin_model.upperPositionLimit[7:] - w.joint_limit_margin, np.full(35, inf)])
    costs.addCost(
        "joint_limits",
        croc.CostModelResidual(state, croc.ActivationModelQuadraticBarrier(croc.ActivationBounds(lb, ub)), croc.ResidualModelState(state, x_zero, nu)),
        w.joint_limits,
    )

    if not terminal:
        costs.addCost("control_reg", croc.CostModelResidual(state, croc.ResidualModelControl(state, nu)), w.control_reg)

    dam = croc.DifferentialActionModelContactFwdDynamics(state, actuation, contacts, costs, 0.0, True)
    dam.armature = armature
    dam.u_lb = -effort
    dam.u_ub = effort
    return croc.IntegratedActionModelEuler(dam, 0.0 if terminal else dt)


def solve_squat(pin_model, mj_model, cfg, params: SquatParams, weights: SquatWeights | None = None, max_iter: int = 300, verbose: bool = False) -> SquatSolution:
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    w = weights or SquatWeights()
    x0 = standing_state(pin_model, mj_model)
    data = pin_model.createData()
    pin.forwardKinematics(pin_model, data, x0[:36])
    pin.updateFramePlacements(pin_model, data)
    sole_placements = {name: pin.SE3(data.oMf[pin_model.getFrameId(name)]) for name in g1.SOLE_FRAMES}
    com0 = pin.centerOfMass(pin_model, data, x0[:36]).copy()
    com_ref = com_reference(params, com0)
    N = params.num_nodes()

    state = croc.StateMultibody(pin_model)
    actuation = croc.ActuationModelFloatingBase(state)
    effort = g1.effort_limits(cfg)
    armature = armature_vector(mj_model)
    running = [
        _node_model(pin_model, state, actuation, x0, com_ref[k], sole_placements, cfg, w, effort, armature, params.dt, False)
        for k in range(N)
    ]
    terminal = _node_model(pin_model, state, actuation, x0, com_ref[N], sole_placements, cfg, w, effort, armature, params.dt, True)
    problem = croc.ShootingProblem(x0, running, terminal)
    solver = croc.SolverBoxFDDP(problem)
    if verbose:
        solver.setCallbacks([croc.CallbackVerbose()])
    xs_init = [x0] * (N + 1)
    us_init = problem.quasiStatic([x0] * N)
    t0 = time.time()
    converged = solver.solve(xs_init, us_init, max_iter, False)
    solve_time = time.time() - t0

    # runningDatas reflect whatever (xs, us) problem.calc was last called with, which is an
    # intermediate line-search iterate, not necessarily solver.xs/solver.us; recompute at the
    # solver's final solution before harvesting contact forces from it.
    problem.calc(solver.xs, solver.us)
    forces = {"left": np.zeros((N, 6)), "right": np.zeros((N, 6))}
    for k in range(N):
        cd = problem.runningDatas[k].differential.multibody.contacts.contacts
        for side, name in zip(("left", "right"), g1.SOLE_FRAMES):
            f = cd[f"{name}_contact"].f
            forces[side][k, :3] = f.linear
            forces[side][k, 3:] = f.angular
    from o2s.models.simulation import snapshot
    return SquatSolution(
        params=params, dt=params.dt, xs=np.array(solver.xs), us=np.array(solver.us), forces=forces,
        converged=bool(converged), iters=int(solver.iter), cost=float(solver.cost), solve_time=solve_time,
        com_ref=com_ref, weights=w, simulation=snapshot(mj_model, cfg),
    )
