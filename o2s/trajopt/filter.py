"""Post-solve feasibility checks that gate export."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from o2s.models import g1

JOINT_LIMIT_MARGIN = 0.02        # rad; the optimizer's own barrier uses a wider 0.04 rad margin
FRICTION_UTILIZATION_MAX = 0.9   # |f_t| / (mu * f_n) must stay under this fraction of the friction cone
COP_MARGIN = 0.005               # m; center of pressure must stay this far inside the sole rectangle edge
FOOT_DRIFT_POS_MAX = 3e-3        # m; 20 ms Euler integration with Baumgarte (100, 20) leaves ~1.7 mm
FOOT_DRIFT_ROT_MAX = 3e-3        # rad
DEPTH_ERROR_MAX = 0.01           # m
FINAL_COM_ERROR_MAX = 0.01       # m
FINAL_JOINT_SPEED_MAX = 0.05     # rad/s


@dataclass
class FilterResult:
    ok: bool
    failures: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


def check_solution(sol, ref: dict, pin_model: pin.Model, cfg: dict) -> FilterResult:
    failures: list[str] = []
    details: dict[str, float] = {}
    lim = np.asarray(cfg["effort_limits"], dtype=float)
    mu = float(cfg["contact"]["friction_mu"])
    half = np.asarray(cfg["frames"]["foot_half_size"], dtype=float)
    N = sol.us.shape[0]
    nq = pin_model.nq

    if not sol.converged:
        failures.append("solver not converged")

    ratio = float(np.max(np.abs(sol.us) / lim))
    details["max_torque_ratio"] = ratio
    if ratio > 1.0 + 1e-6:
        failures.append(f"torque ratio {ratio:.3f} > 1")

    qj = sol.xs[:, 7:nq]
    lo, hi = pin_model.lowerPositionLimit[7:], pin_model.upperPositionLimit[7:]
    margin = float(min(np.min(qj - lo), np.min(hi - qj)))
    details["joint_limit_margin"] = margin
    if margin < JOINT_LIMIT_MARGIN:
        failures.append(f"joint limit margin {margin:.3f} < {JOINT_LIMIT_MARGIN} rad")

    min_fz, worst_friction, worst_cop = np.inf, 0.0, -np.inf
    for side in ("left", "right"):
        w = sol.forces[side]
        fz = w[:, 2]
        min_fz = min(min_fz, float(fz.min()))
        ft = np.hypot(w[:, 0], w[:, 1])
        worst_friction = max(worst_friction, float(np.max(ft / np.maximum(fz, 1e-6))))
        # wrench at the sole center, world-aligned: tau = d x f with d = (dx, dy, 0) -> dx = -ty/fz, dy = tx/fz
        cop_x = -w[:, 4] / np.maximum(fz, 1e-6)
        cop_y = w[:, 3] / np.maximum(fz, 1e-6)
        worst_cop = max(worst_cop, float(np.max(np.abs(cop_x) - (half[0] - COP_MARGIN))), float(np.max(np.abs(cop_y) - (half[1] - COP_MARGIN))))
    details["min_normal_force"] = min_fz
    details["friction_utilization"] = worst_friction / mu
    details["cop_margin_violation"] = worst_cop   # negative = inside the margin (slack), positive = violation
    if min_fz <= 0:
        failures.append(f"normal force {min_fz:.1f} N <= 0")
    if worst_friction > FRICTION_UTILIZATION_MAX * mu:
        failures.append(f"friction |ft|/fz {worst_friction:.3f} > {FRICTION_UTILIZATION_MAX} mu ({FRICTION_UTILIZATION_MAX * mu:.3f})")
    if worst_cop > 0:
        failures.append(f"center of pressure outside sole minus {COP_MARGIN * 1000:.0f} mm margin by {worst_cop * 1000:.1f} mm")

    data = pin_model.createData()
    drift_pos, drift_rot = 0.0, 0.0
    sole0 = {}
    for k in range(N + 1):
        pin.forwardKinematics(pin_model, data, sol.xs[k, :nq])
        pin.updateFramePlacements(pin_model, data)
        for name in g1.SOLE_FRAMES:
            M = pin.SE3(data.oMf[pin_model.getFrameId(name)])
            if k == 0:
                sole0[name] = M
            else:
                dM = sole0[name].actInv(M)
                drift_pos = max(drift_pos, float(np.linalg.norm(dM.translation)))
                drift_rot = max(drift_rot, float(np.linalg.norm(pin.log3(dM.rotation))))
    details["foot_drift_pos"] = drift_pos
    details["foot_drift_rot"] = drift_rot
    # 20 ms Euler integration leaves about 1.7 mm of constraint drift with Baumgarte (100, 20).
    if drift_pos > FOOT_DRIFT_POS_MAX or drift_rot > FOOT_DRIFT_ROT_MAX:
        failures.append(f"foot drift {drift_pos * 1000:.2f} mm / {drift_rot * 1000:.2f} mrad > {FOOT_DRIFT_POS_MAX * 1000:.0f} mm / {FOOT_DRIFT_ROT_MAX * 1000:.0f} mrad")

    z = ref["com"][:, 2]
    z0 = sol.com_ref[0, 2]
    depth_err = abs((z0 - z.min()) - sol.params.depth)
    details["depth_error"] = float(depth_err)
    if depth_err > DEPTH_ERROR_MAX:
        failures.append(f"achieved depth off by {depth_err * 100:.1f} cm")
    final_z_err = abs(z[-1] - z0)
    final_qd = float(np.max(np.abs(sol.xs[-1, nq + 6:])))
    details["final_com_height_error"] = float(final_z_err)
    details["final_joint_speed"] = final_qd
    if final_z_err > FINAL_COM_ERROR_MAX:
        failures.append(f"final CoM height off by {final_z_err * 100:.1f} cm")
    if final_qd > FINAL_JOINT_SPEED_MAX:
        failures.append(f"final joint speed {final_qd:.3f} rad/s > {FINAL_JOINT_SPEED_MAX}")

    return FilterResult(ok=not failures, failures=failures, details=details)
