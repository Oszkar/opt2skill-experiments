"""Generate a randomized, filtered squat reference dataset with a train/val/test split.

After a solve passes the trajectory-optimization filter (`filter.check_solution`), the exported
reference is additionally gated on MuJoCo inverse-dynamics consistency and replay torque
limits. Every exported reference torque and every simulated substep's total feedforward-plus-PD
torque must stay within the configured joint effort limits. Other closed-loop tracking metrics
are recorded under meta["replay_ff"] but do not gate acceptance. The output must be new or empty;
existing data is never resumed or overwritten by this CLI.

An exception raised anywhere in solving, exporting, or filtering a sampled solution (e.g. NaNs
from a badly-conditioned solve) is caught and rejected with stage "solve" rather than propagating,
so one bad sample cannot abort the whole run; a solver that ran but did not converge is rejected
with stage "solver" before export is attempted, since export's own validation can raise on NaNs
before the filter gets a chance to report "solver not converged" as an ordinary failure.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from o2s.models import g1
from o2s.reference import contract, validate
from o2s.trajopt import export
from o2s.trajopt import filter as flt
from o2s.trajopt.profile import SquatParams
from o2s.trajopt.squat_problem import solve_squat


@dataclass
class SquatRanges:
    depth: tuple[float, float] = (0.10, 0.30)
    t_down: tuple[float, float] = (0.8, 1.5)
    t_hold: tuple[float, float] = (0.2, 0.6)
    t_up: tuple[float, float] = (0.8, 1.5)
    com_shift_x: tuple[float, float] = (-0.03, 0.03)


def sample_params(rng: np.random.Generator, r: SquatRanges) -> SquatParams:
    return SquatParams(
        depth=float(rng.uniform(*r.depth)),
        t_stand0=0.5,
        t_down=float(rng.uniform(*r.t_down)),
        t_hold=float(rng.uniform(*r.t_hold)),
        t_up=float(rng.uniform(*r.t_up)),
        t_stand1=0.5,
        com_shift_x=float(rng.uniform(*r.com_shift_x)),
    )


def make_split(names: list[str], rng: np.random.Generator, fractions=(0.7, 0.1, 0.2)) -> dict[str, list[str]]:
    order = list(np.array(names)[rng.permutation(len(names))])
    n = len(order)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    n_test = n - n_train - n_val
    return {"train": sorted(order[:n_train]), "val": sorted(order[n_train:n_train + n_val]), "test": sorted(order[n_train + n_val:n_train + n_val + n_test])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of accepted trajectories to produce")
    ap.add_argument("--out", type=Path, default=Path("data/refs/squat"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=300)
    args = ap.parse_args()

    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        ap.error(f"output must be a new or empty directory: {args.out}; choose a different --out")
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = g1.load_config()
    mj_model = g1.load_mj_model(cfg)
    pin_model = g1.load_pin_model(cfg)
    rng = np.random.default_rng(args.seed)
    ranges = SquatRanges()
    accepted, attempts, times, depths = [], 0, [], []
    replay_oks, replay_falls = [], []
    t_start = time.time()
    with open(args.out / "rejected.jsonl", "x", encoding="utf-8") as rejected:
        while len(accepted) < args.n and attempts < args.max_attempts:
            attempts += 1
            params = sample_params(rng, ranges)
            try:
                sol = solve_squat(pin_model, mj_model, cfg, params)
                times.append(sol.solve_time)
                if not sol.converged:
                    rejected.write(json.dumps({"params": params.__dict__, "failures": ["solver not converged"], "iters": sol.iters, "stage": "solver"}) + "\n")
                    print(f"[{attempts:3d}] reject depth={params.depth:.3f}: solver not converged")
                    continue
                ref = export.solution_to_reference(sol, pin_model, mj_model, cfg)
                res = flt.check_solution(sol, ref, pin_model, cfg)
            except Exception as e:
                rejected.write(json.dumps({"params": params.__dict__, "failures": [repr(e)], "iters": -1, "stage": "solve"}) + "\n")
                print(f"[{attempts:3d}] reject depth={params.depth:.3f}: {e!r}")
                continue
            if not res.ok:
                rejected.write(json.dumps({"params": params.__dict__, "failures": res.failures, "iters": sol.iters, "stage": "filter"}) + "\n")
                print(f"[{attempts:3d}] reject depth={params.depth:.3f}: {res.failures[0]}")
                continue
            id_check = validate.check_inverse_dynamics(mj_model, ref, cfg)
            if not id_check.ok:
                rejected.write(json.dumps({"params": params.__dict__, "failures": [id_check.message], "iters": sol.iters, "stage": "check_inverse_dynamics"}) + "\n")
                print(f"[{attempts:3d}] reject depth={params.depth:.3f}: {id_check.message}")
                continue
            replay = validate.check_replay(mj_model, ref, cfg, feedforward=True)
            if not replay.metrics["effort_ok"]:
                rejected.write(json.dumps({"params": params.__dict__, "failures": [replay.message], "iters": sol.iters, "stage": "replay_effort"}) + "\n")
                print(f"[{attempts:3d}] reject depth={params.depth:.3f}: replay torque limit exceeded")
                continue
            name = f"squat_{len(accepted):04d}"
            meta = export.solution_meta(sol, res, cfg)
            meta["replay_ff"] = {"ok": replay.ok, **replay.metrics}
            contract.save(args.out / f"{name}.npz", ref, meta)
            accepted.append(name)
            depths.append(params.depth)
            replay_oks.append(replay.ok)
            replay_falls.append(replay.metrics["completed"] == 0.0)
            print(f"[{attempts:3d}] accept {name} depth={params.depth:.3f} iters={sol.iters} {sol.solve_time:.2f}s replay_ff_ok={replay.ok}")
    split = make_split(accepted, np.random.default_rng(args.seed + 1))
    (args.out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    summary = {
        "accepted": len(accepted), "attempts": attempts, "accept_rate": len(accepted) / max(attempts, 1),
        "solve_time_mean": float(np.mean(times)) if times else 0.0,
        "solve_time_max": float(np.max(times)) if times else 0.0,
        "depth_hist": np.histogram(depths, bins=np.linspace(ranges.depth[0], ranges.depth[1], 5))[0].tolist(),
        "wall_time": time.time() - t_start, "seed": args.seed, "ranges": ranges.__dict__,
        "split_sizes": {k: len(v) for k, v in split.items()},
        "replay_ff_pass_rate": (sum(replay_oks) / len(replay_oks)) if replay_oks else None,
        "replay_ff_fall_count": sum(replay_falls),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if len(accepted) == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
