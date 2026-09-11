"""Solve one squat, filter it, export it. Usage: python -m o2s.trajopt.solve_one --depth 0.2 --out data/refs/dev/squat_dev.npz"""
from __future__ import annotations

import argparse
from pathlib import Path

from o2s.models import g1
from o2s.reference import contract
from o2s.trajopt import export
from o2s.trajopt import filter as flt
from o2s.trajopt.profile import SquatParams
from o2s.trajopt.squat_problem import solve_squat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=float, default=0.2)
    ap.add_argument("--t-down", type=float, default=1.0)
    ap.add_argument("--t-hold", type=float, default=0.4)
    ap.add_argument("--t-up", type=float, default=1.0)
    ap.add_argument("--com-shift-x", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=Path("data/refs/dev/squat_dev.npz"))
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--force", action="store_true", help="write even if the filter fails")
    args = ap.parse_args()

    try:
        params = SquatParams(depth=args.depth, t_down=args.t_down, t_hold=args.t_hold, t_up=args.t_up, com_shift_x=args.com_shift_x)
        if args.max_iter <= 0:
            raise ValueError("--max-iter must be positive")
    except ValueError as exc:
        ap.error(str(exc))
    cfg = g1.load_config()
    mj_model = g1.load_mj_model(cfg)
    pin_model = g1.load_pin_model(cfg)
    sol = solve_squat(pin_model, mj_model, cfg, params, max_iter=args.max_iter, verbose=args.verbose)
    print(f"converged={sol.converged} iters={sol.iters} cost={sol.cost:.3f} time={sol.solve_time:.2f}s N={sol.us.shape[0]}")
    ref = export.solution_to_reference(sol, pin_model, mj_model, cfg)
    res = flt.check_solution(sol, ref, pin_model, cfg)
    for k, v in res.details.items():
        print(f"  {k:26s} {v: .4f}")
    for f in res.failures:
        print(f"  FAIL {f}")
    if res.ok or args.force:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        contract.save(args.out, ref, export.solution_meta(sol, res, cfg))
        print(f"wrote {args.out}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
