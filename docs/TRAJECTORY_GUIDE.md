# Trajectory generation and validation

This guide describes the implemented G1 squat workflow. See [README](../README.md)
for environment setup, [diagrams](TRAJECTORY_DIAGRAMS.md) for explanations, and
[next steps](NEXT_STEPS.md) for work that is not implemented yet.

## Follow the code

| Module | Responsibility |
|---|---|
| `o2s/models/g1.py` | Load the URDF and MJCF using `configs/g1_reconcile.yaml` |
| `o2s/models/reconcile.py` | Compare joint layout, limits, inertias, frames, geometry, and actuators |
| `o2s/trajopt/profile.py` | Build a smooth CoM descent, hold, and rise from `SquatParams` |
| `o2s/trajopt/squat_problem.py` | Solve floating-base dynamics with two fixed sole contacts using BoxFDDP |
| `o2s/trajopt/export.py` | Convert frames, compute keypoints, and add damping compensation to torque |
| `o2s/trajopt/filter.py` | Check convergence, optimizer torque, joint limits, contact feasibility, and task completion |
| `o2s/trajopt/generate.py` | Sample parameters, filter trajectories, and write data plus train/validation/test splits |
| `o2s/reference/contract.py` | Save, load, and pad reference arrays |
| `o2s/reference/validate.py` | Compare inverse dynamics and run two controller replays |
| `o2s/reference/view.py` | Display the reference or physics replay with matching reference keypoints |

Start reading with `profile.py`, then `squat_problem.py`, `export.py`, and
`validate.py`. The default squat lasts 3.4 seconds: 170 control intervals at 20 ms.
MuJoCo takes ten 2 ms physics steps per control interval.

## Run and inspect a squat

All commands below run from the project root inside the activated WSL environment.

```bash
python -m o2s.models.reconcile
python -m pytest -q
python -m o2s.trajopt.solve_one --depth 0.2 --out data/refs/learning/squat.npz
python -m o2s.reference.validate data/refs/learning/squat.npz
python -m o2s.reference.view data/refs/learning/squat.npz --mode kinematic --loop
python -m o2s.reference.view data/refs/learning/squat.npz --mode replay --loop
```

Close each viewer before starting the next command. Spheres show reference CoM
(red), pelvis (blue), ankle-roll-link origins (green), and palms (yellow).
Kinematic mode displays every state including the terminal state. Replay displays
the state and reference markers at the end of each simulated control interval.

`solve_one` writes only when the feasibility filter passes, unless `--force` is
set. It does not run MuJoCo validation automatically. A rejection exits with code
1; inspect the printed failures before trying to load the output file. Use a new
filename for each experiment so an older successful output cannot be mistaken
for a rejected attempt.

## Learn by changing one parameter

```bash
python -m o2s.trajopt.solve_one --depth 0.15 --out data/refs/learning/shallow.npz
python -m o2s.trajopt.solve_one --depth 0.2 --t-down 1.4 --t-up 1.4 --out data/refs/learning/slow.npz
python -m o2s.trajopt.solve_one --depth 0.2 --com-shift-x 0.03 --out data/refs/learning/lean.npz
```

Validate and replay each successful output. Compare `max_torque_ratio`,
`foot_drift_pos`, `depth_error`, `max_pelvis_error`, and `pd_over_ref_legs`.
Try a deeper or faster motion and inspect which feasibility checks become tight;
exact failure modes depend on the parameters and solver result. `--verbose`
prints solver iterations. `--force` exports a failed filter result for inspection
and still returns failure; keep those files separate from training data.

For a controller experiment, replay the default squat with lower gains:

```bash
python -m o2s.reference.view data/refs/learning/squat.npz --mode replay --kp-scale 2 --ankle-pitch-kp 200
```

Changing gains changes the controller, not the reference. Passing inverse dynamics
does not guarantee a particular feedback controller will track the motion.
For optimization experiments, `SquatWeights` controls CoM tracking, posture,
contact-wrench penalties, joint-limit margins, and contact stabilization.

## Generate a dataset

Use a new output directory for each run:

```bash
python -m o2s.trajopt.generate --n 10 --out data/refs/learning_batch_01 --seed 1
```

The generator samples squat depth, timing, and CoM shift. It writes numbered
`.npz` files, rejection reasons in `rejected.jsonl`, `summary.json`, and a
70/10/20 split in `split.json` (rounded for small datasets). It exits with failure
if `--max-attempts` is reached before the requested count is accepted.

**Nonempty output directories are rejected before model loading or file writes.**
Use a new or empty directory. Existing runs are not resumed or cleared.

```bash
python - <<'PY'
import json
from pathlib import Path
from o2s.reference import contract
root = Path("data/refs/learning_batch_01")
ref, meta = contract.load(root / "squat_0000.npz")
print({key: value.shape for key, value in ref.items()})
print("depth:", meta["depth"], "replay:", meta["replay_ff"])
split = json.loads((root / "split.json").read_text())
training = contract.load_dir(root, names=split["train"])
print("padded states:", training.arrays["qpos"].shape)
print("valid control lengths:", training.lengths)
PY
```

## What the checks establish

| Check | Single-file `validate` | Dataset generator |
|---|---|---|
| Optimization feasibility | Not rerun; inspect exported metadata | Required before acceptance |
| Inverse-dynamics consistency | Must pass | Must pass |
| Exported and per-substep total torque limits | Must pass in feedforward replay | Must pass |
| Feedforward-plus-PD tracking | Must pass | Recorded in metadata; tracking failure does not reject the file |
| PD-only replay | Informational | Not run |

Inverse dynamics compares MuJoCo's required generalized forces against the
reference torques and sole wrenches with contacts, friction loss, joint limits,
and actuation disabled. It checks model agreement, independently of feedback
stability. Joint residual limits are 0.01 N m RMS and 0.2 N m maximum.

Stabilized replay uses 4x the model's position gains with ankle pitch set to
400 N m/rad. It requires pelvis height above 0.4 m, pelvis error below 3 cm,
joint error below 0.03 rad RMS, and leg PD correction below 25% of reference RMS
torque. Exported reference torque and total drive torque must also remain within
configured limits (ratio tolerance 1e-6). `peak_effort_ratio` is now the maximum
across physics substeps; `peak_interval_mean_effort_ratio` retains the average-based
comparison, and `effort_violation_substeps` counts offending substeps. Fresh replay
metadata carries `effort_checked_per_substep=1`; older saved metadata must be
recomputed to obtain these guarantees. The PD-only comparison falls on the tested
default squat. This is a controller comparison, not a learned Pos versus Pos+T
policy ablation.

## Reference conventions and limitations

The authoritative array layout is in [contract.py](../o2s/reference/contract.py).
For N intervals, states have N+1 rows; torques and wrenches have N rows.
At step k, use state k and act over `[t[k], t[k+1])`; compare the resulting
state against reference k+1 and interval-mean actuator torque against `tau[k]`.
State N has no torque or wrench sample. Padded rows must be masked using lengths.

- `qpos`: world base position, **wxyz** quaternion, 29 joints in configured order.
- `qvel`: world base linear velocity, pelvis-local angular velocity, joint velocities.
  Both `pelvis_linvel` and `pelvis_angvel` are pelvis-local.
- `tau`: exported actuator torque includes interval-mean joint damping compensation.
  Do not add damping again. Joint friction loss is not modeled by the optimizer.
- Foot and hand keypoints are pelvis-local. Foot positions are ankle-link origins;
  wrenches act at sole origins offset by `(0.04, 0, -0.037)` in the ankle-link frame.
- Wrenches use world-aligned axes at each sole. Rotated-foot cases remain unverified;
  these trajectories keep both feet flat and planted.
- The optimizer bounds its raw torque; the filter also checks exported torque after
  damping compensation. Replay still applies feedforward externally, outside actuator
  clipping, but rejects total drive torque violations at any Euler physics substep.
  Passive damping and contact forces are not counted as motor torque.
- The loader checks shapes, finiteness, at least one interval, and time starting at
  zero with uniform 20 ms steps (1e-9 s absolute tolerance). Base quaternions in
  `qpos` and `pelvis_quat` must be unit length and agree up to sign (1e-6 tolerance).

The [knowledge base](../KNOWLEDGEBASE.md) retains tuning history and detailed
measurements. Current defaults and acceptance thresholds live in the code.
