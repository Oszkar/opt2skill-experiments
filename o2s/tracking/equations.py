"""Equation diagnostics, isolated from the live simulation and actor observations.

Samples are continuous forward-dynamics evaluations at substep START states,
not finite differences of Euler-integrated velocities. Spatial rows are linear
xyz then angular xyz in world axes at the configured sole point.
"""
import json

import mujoco
import numpy as np

from o2s.models import g1


class EquationMonitor:
    def __init__(self, env):
        from o2s.trajopt.diagnostics import objective_components
        self.model = env.model
        self.data = mujoco.MjData(env.model)
        self.soles = [env.model.body(name).id for name in g1.FOOT_LINKS]
        self.offset = np.array(env.cfg["frames"]["sole_offset"])
        self.objective = objective_components(env.ref, env.meta, env.model, env.cfg)

    def sample(self, source):
        m, d = self.model, self.data
        mujoco.mj_copyData(d, m, source)
        mujoco.mj_forward(m, d)
        inertia = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, inertia, d.qacc)
        contact = np.zeros(m.nv)
        force = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, force)
            rotation = c.frame.reshape(3, 3).T
            # MuJoCo reports force on geom2; apply the equal/opposite force to geom1.
            for geom, sign in ((c.geom1, -1.), (c.geom2, 1.)):
                body = m.geom_bodyid[geom]
                if body:
                    mujoco.mj_applyFT(m, d, sign * (rotation @ force[:3]),
                                     sign * (rotation @ force[3:]), c.pos, body, contact)
        drive = d.qfrc_actuator + d.qfrc_applied
        other = d.qfrc_constraint - contact
        residual = inertia + d.qfrc_bias - drive - contact - other - d.qfrc_passive
        velocity, ja, jdotv, position = [], [], [], []
        for body in self.soles:
            point = d.xpos[body] + d.xmat[body].reshape(3, 3) @ self.offset
            jac = np.zeros((6, m.nv))
            jacdot = np.zeros_like(jac)
            mujoco.mj_jac(m, d, jac[:3], jac[3:], point, body)
            mujoco.mj_jacDot(m, d, jacdot[:3], jacdot[3:], point, body)
            position.append(point.copy())
            velocity.append(jac @ d.qvel)
            ja.append(jac @ d.qacc)
            jdotv.append(jacdot @ d.qvel)
        return dict(inertia=inertia, bias=d.qfrc_bias.copy(), drive=drive.copy(),
                    contact=contact, other_constraint=other, passive=d.qfrc_passive.copy(),
                    balance_residual=residual, sole_velocity=np.array(velocity),
                    sole_j_a=np.array(ja), sole_jdot_v=np.array(jdotv),
                    sole_acceleration=np.array(ja) + np.array(jdotv), sole_position=np.array(position),
                    solver_iterations=np.array(d.solver_niter).copy(),
                    com=d.subtree_com[1].copy())


def live_values(env, info):
    e = info["equations"]
    k = info["interval_index"]
    objective = env.equations.objective
    terms = ", ".join(f"{name}={value:.3g}" for name, value in zip(objective["names"], objective["running"][k]))
    j = env.cfg["joints"].index("left_knee_joint") + 6
    last = lambda key: e[key][-1, j]
    speed = np.linalg.norm(e["sole_velocity"][-1, :, :3], axis=1)
    ja, jd = e["sole_j_a"][-1, 0, 2], e["sole_jdot_v"][-1, 0, 2]
    return (f"  2a planned interval cost: {terms} (terminal cost={sum(objective['terminal']):.3g})\n"
            f"  2b at {info['substep_start_time'][-1]:.3f}s, left knee [Nm]: Ma={last('inertia'):+.3f}, C={last('bias'):+.3f}; "
            f"Bu={last('drive'):+.3f}, JTF={last('contact'):+.3f}, passive={last('passive'):+.3f}, "
            f"other={last('other_constraint'):+.3f}; residual={last('balance_residual'):+.2e}\n"
            f"  2c sole speed L/R={speed[0]:.4f}/{speed[1]:.4f} m/s; left z [m/s2]: "
            f"Ja={ja:+.3f} + Jdotv={jd:+.3f} = {ja+jd:+.3f} (ideal 0); impacts: not modeled")


def write_equations(out, env, records, controller):
    import matplotlib.pyplot as plt
    objective = env.equations.objective
    arrays = {key: np.stack([r["equations"][key] for r in records]) for key in records[0]["equations"]}
    times = np.stack([r["substep_start_time"] for r in records])
    np.savez_compressed(out / "equations.npz", time=times, **arrays,
                        objective_time=env.ref["t"], cost_names=np.array(objective["names"]),
                        running_cost=objective["running"], terminal_cost=objective["terminal"],
                        target_com=objective["target_com"], optimizer_u=objective["raw_u"])
    residual = arrays["balance_residual"]
    metadata = dict(objective_total=objective["total"], saved_objective=objective["saved_cost"],
                    objective_provenance=objective["provenance"],
                    cost_convention="Planned trajectory only; running weighted costs include dt and quadratic 1/2; terminal separate",
                    solver_iterations=env.model.opt.iterations, solver_linesearch_iterations=env.model.opt.ls_iterations,
                    residual_note="Finite solver convergence can leave a nonzero balance residual; this is not the reference inverse-dynamics validation residual",
                    objective_version=objective["version"],
                    terminal_note="Legacy terminal wrench penalties included" if objective["version"] == 1 else "Terminal wrench penalties excluded; no terminal wrench control exists",
                    sampling="Independent continuous forward dynamics at substep start; no mutation of live data",
                    dynamics="M a + C = drive + contact + passive + other_constraint; all 35 generalized coordinates",
                    contact="All simulated contacts; other_constraint includes friction loss and joint limits",
                    sole_convention="World axes at configured sole origin; [linear xyz, angular xyz]; measured even when airborne",
                    impacts="Not modeled by this fixed-contact optimizer; no impulse or pre/post-impact diagnostics",
                    max_base_force_residual_N=float(np.max(np.abs(residual[:, :, :3]))),
                    max_base_torque_residual_Nm=float(np.max(np.abs(residual[:, :, 3:6]))),
                    max_joint_residual_Nm=float(np.max(np.abs(residual[:, :, 6:]))))
    (out / "equations.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    t = times.ravel()
    fig, axes = plt.subplots(5, 2, figsize=(15, 19), constrained_layout=True)
    fig.suptitle(f"Equations along the squat | {controller} | {records[-1]['reason']}\n"
                 "2a: planned optimizer objective   |   2b/2c: measured simulation   |   impact branches: not modeled", fontsize=15)
    ax = axes[0, 0]
    for i, name in enumerate(objective["names"]):
        ax.plot(env.ref["t"][:-1], objective["running"][:, i], label=name)
    ax.set_title("2a: running objective contributions (include dt)")
    ax.set_ylabel("Weighted cost per interval (symlog)")
    ax.set_yscale("symlog", linthresh=.01)
    ax = axes[0, 1]
    ax.barh(objective["names"], objective["terminal"])
    ax.set_title(f"2a: terminal contributions at {env.ref['t'][-1]:.2f}s (planned)")
    terminal_log = np.max(objective["terminal"]) >= 1
    ax.set_xlabel("Weighted terminal cost" + (" (symlog scale)" if terminal_log else ""))
    ax.text(.02, .02, ("Legacy terminal wrench artifact:\nzero terminal wrenches vs minimum normal force."
                      if objective["version"] == 1 else "Terminal state/CoM costs only; no terminal wrench penalty."),
            transform=ax.transAxes, fontsize=8, va="bottom", bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    if terminal_log:
        ax.set_xscale("symlog", linthresh=.01)
    else:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    for ax, idx, label, unit in ((axes[1, 0], 2, "floating base vertical", "N"),
                                (axes[1, 1], 6 + env.cfg["joints"].index("left_knee_joint"), "left knee", "N m")):
        for key, name in (("inertia", "M a"), ("bias", "C"), ("drive", "B u (PD + FF)"),
                          ("contact", "J^T F (contact)"), ("passive", "passive"), ("other_constraint", "other constraints")):
            ax.plot(t, arrays[key][:, :, idx].ravel(), label=name, linewidth=.8)
        ax.plot(t, arrays["balance_residual"][:, :, idx].ravel(), 'k--', label="LHS - RHS", linewidth=1)
        ax.set_title(f"2b: {label} | Ma + C = Bu + JTF + passive + other")
        ax.set_ylabel(unit)
        ax.text(.02, .97, "Residual includes finite contact-solver convergence", transform=ax.transAxes,
                va="top", fontsize=8, bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
    for column, sl, label, unit in ((0, slice(0, 3), "linear", "m/s"), (1, slice(3, 6), "angular", "rad/s")):
        ax = axes[2, column]
        for side, name in enumerate(("left", "right")):
            values = np.linalg.norm(arrays["sole_velocity"][:, :, side, sl], axis=-1)
            ax.plot(t, values.ravel(), label=name)
        ax.axhline(0, color="k", linestyle=":", label="ideal planted contact")
        ax.set_title(f"2c: norm of J v | sole {label} velocity")
        ax.set_ylabel(unit)
    for side, name in enumerate(("left", "right")):
        ax = axes[3, side]
        for key, label in (("sole_j_a", "J a"), ("sole_jdot_v", "Jdot v"), ("sole_acceleration", "sum (ideal 0)")):
            ax.plot(t, arrays[key][:, :, side, 2].ravel(), label=label, linewidth=.9)
        ax.set_title(f"2c: {name} sole vertical acceleration | Ja + Jdot v")
        ax.set_ylabel("m/s²")
        ax.axhline(0, color="k", linestyle=":")
    ax = axes[4, 0]
    ax.plot(env.ref["t"], objective["target_com"][:, 2], 'k--', label="optimizer target")
    ax.plot(env.ref["t"], env.ref["com"][:, 2], label="optimized reference")
    ax.plot(t, arrays["com"][:, :, 2].ravel(), label="simulation")
    ax.set_title("2a context: desired, optimized and simulated CoM height")
    ax.set_ylabel("m")
    ax = axes[4, 1]
    for side, name in enumerate(("left", "right")):
        positions = arrays["sole_position"][:, :, side]
        ax.plot(t, np.linalg.norm(positions - positions[0, 0], axis=-1).ravel(), label=name)
    ax.set_title("2c context: sole displacement from initial world position")
    ax.set_ylabel("m")
    for i, ax in enumerate(axes.flat):
        ax.grid(alpha=.2)
        if i != 1:
            ax.set_xlabel("Simulation / reference time [s]")
            ax.legend(fontsize=8, ncol=2)
    fig.savefig(out / "equations.png", dpi=120)
    plt.close(fig)
    return metadata
