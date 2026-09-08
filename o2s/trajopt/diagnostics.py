"""Re-evaluate a saved squat's objective without running an optimization."""
from dataclasses import fields

import crocoddyl as croc
import numpy as np
import pinocchio as pin

from o2s.models import g1
from o2s.reference import convert
from o2s.trajopt.profile import SquatParams, com_reference
from o2s.trajopt.squat_problem import SquatWeights, _node_model, armature_vector, standing_state


def objective_components(ref, meta, mj_model, cfg):
    if "depth" not in meta:
        raise ValueError("equation cost diagnostics require a squat reference with task metadata")
    params = SquatParams(**{f.name: meta[f.name] for f in fields(SquatParams) if f.name in meta})
    weights = SquatWeights(**meta.get("weights", {}))
    version = meta.get("objective_version", 1)
    if version not in (1, 2):
        raise ValueError(f"unsupported objective_version {version}")
    p = g1.load_pin_model(cfg)
    x0 = standing_state(p, mj_model)
    data = p.createData()
    pin.forwardKinematics(p, data, x0[:p.nq])
    pin.updateFramePlacements(p, data)
    soles = {name: pin.SE3(data.oMf[p.getFrameId(name)]) for name in g1.SOLE_FRAMES}
    target = com_reference(params, pin.centerOfMass(p, data, x0[:p.nq]).copy())
    n = len(ref["tau"])
    if len(target) != n + 1:
        raise ValueError("squat task metadata does not match reference duration")
    state = croc.StateMultibody(p)
    actuation = croc.ActuationModelFloatingBase(state)
    # Undo export's interval-mean MuJoCo damping compensation to recover optimizer u.
    raw_u = ref["tau"] - mj_model.dof_damping[6:] * .5 * (ref["qvel"][:-1, 6:] + ref["qvel"][1:, 6:])
    rows = []
    for k in range(n + 1):
        terminal = k == n
        node = _node_model(p, state, actuation, x0, target[k], soles, cfg, weights,
                           g1.effort_limits(cfg), armature_vector(mj_model), params.dt, terminal, terminal_wrench_cost=version == 1)
        nd = node.createData()
        q, v = convert.mj_to_pin(ref["qpos"][k], ref["qvel"][k])
        x = np.concatenate([q, v])
        if terminal:
            node.calc(nd, x)
        else:
            node.calc(nd, x, raw_u[k])
        costs = nd.differential.costs.costs.todict()
        # Crocoddyl includes 1/2 in quadratic residual costs; running costs also integrate dt.
        rows.append({name: float(cd.cost * node.differential.costs.costs[name].weight
                                  * (1. if terminal else params.dt)) for name, cd in costs.items()})
    names = sorted(set().union(*rows))
    values = np.array([[row.get(name, 0.) for name in names] for row in rows])
    total = float(values.sum())
    saved_cost = meta.get("solver", {}).get("cost")
    if saved_cost is not None and not np.isclose(total, saved_cost, rtol=1e-5, atol=1e-5):
        raise ValueError(f"reconstructed objective {total:.6g} differs from saved cost {saved_cost:.6g}; regenerate the reference with saved weights")
    return dict(names=names, running=values[:-1], terminal=values[-1], total=total,
                target_com=target, raw_u=raw_u,
                provenance="saved weights" if "weights" in meta else "legacy reference: default weights assumed",
                saved_cost=saved_cost, version=version)
