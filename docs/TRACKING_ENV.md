# Inspecting a single-reference tracking environment

`o2s.tracking` is a deterministic MuJoCo environment with a plain Python reset/step
interface. It makes actions, timing, physics, and rewards inspectable before RL
training. There is no Gymnasium/MJX integration, PPO implementation, randomization,
or trained policy yet. It runs with the existing `trajopt` and `dev` dependencies.

## Run an experiment

From the repository root with the Python environment activated:

```bash
python -m o2s.trajopt.solve_one --depth 0.15 --out data/refs/learning/shallow.npz
python -m o2s.tracking.inspect data/refs/learning/shallow.npz --controller reference --out runs/shallow_reference_01
python -m o2s.tracking.inspect data/refs/learning/shallow.npz --controller hold --out runs/shallow_hold_01
```

Skip the solve if that reference already exists. Use a new or empty output
directory: existing runs are rejected. This command is headless and writes:

| File | Contents |
|---|---|
| `diagnostics.png` | Six panels: pelvis height, knee angle/command, knee torque, actuator utilization, foot contact forces, reward components |
| `steps.csv` | One row per control interval: errors, rewards, saturation, termination reason |
| `trace.npz` | Full states, returned observations, commands, and every substep's torques/contacts |
| `summary.json` | Episode outcome, gains/limits, reference metadata and SHA-256, library versions, observation shapes |

Open the PNG to start. Inspect the CSV when a metric changes sharply; use the NPZ
for individual joints or physics substeps. `runs/` is ignored by Git.

The `reference` script sends the current reference joint positions as targets;
`hold` sends the fixed home joint positions. Both use unmodified model gains,
including ankle pitch 20 N m/rad, and **no external feedforward torque**. Their
falls are expected diagnostic outcomes. They do not reproduce `reference.validate`'s
4x/400 feedforward controller. A successfully recorded fall exits with code zero;
invalid arguments and simulation failures exit with an error.

In the initial shallow-squat experiment, reference targets fell at 1.20 s and
home targets at 1.24 s. Neither saturated actuators. Low torque utilization does
not imply stable balance: the controller may simply fail to request the right
joint torques. These measurements describe the scripted controllers, not reference
feasibility or the potential performance of a future policy.

## Compare the validation controller

Both stabilized modes use 4x position stiffness, ankle pitch stiffness of
400 N m/rad, unchanged velocity damping gains, and target `q_ref[k+1]` during
interval k. Only `stabilized-ff` adds `tau_ref[k]` as feedforward. This isolates
what the optimized torque contributes. These names describe controller presets;
"stabilized" does not guarantee a successful episode.

```bash
python -m o2s.tracking.inspect data/refs/learning/shallow.npz --controller stabilized-pd --out runs/shallow_pd_01
python -m o2s.tracking.inspect data/refs/learning/shallow.npz --controller stabilized-ff --out runs/shallow_ff_01
```

Add `--view` to either command to watch the actual simulation at real time with
reference keypoint markers and live terminal readouts of time, pelvis error,
effort and reward. The same trace and plots are saved after the run. The viewer
closes at episode end; closing it early saves a partial run as `viewer_closed`.
A graphical desktop is required (WSLg on WSL2); on macOS use `mjpython` instead of
`python` for `--view`. The six-panel PNG is an after-run plot, not a live dashboard.

On the saved 15 cm shallow squat, stabilized PD fell at 2.22 s (maximum pelvis
position error 0.5575 m, peak effort ratio 0.4497). With feedforward it completed
all 3.40 s (maximum pelvis error 0.00769 m, peak effort ratio 0.3155). Neither run
saturated the PD actuators or exceeded total effort limits. The PD trace ends at
the fall; validation continues over the full reference, so its post-fall metrics
are not directly comparable. Tests compare both controllers' states and substep
torques with validation over the intervals they share.

The knee torque panel separates PD actuator torque, applied feedforward, total
drive torque, and optimized reference torque. In the successful run, feedforward
supplies most of the planned load while PD corrects tracking errors. This is a
validated replay baseline, not a learned policy.

For paper-equation costs, force balance and contact-motion plots, add
`--equations`. See [Watching the equations](TRACKING_EQUATIONS.md) for the
CLI readouts, units, solver residuals and current terminal-cost artifact.

## The interface

```python
from o2s.tracking.env import TrackingEnv

env = TrackingEnv("data/refs/learning/shallow.npz")
obs, info = env.reset()
while True:
    action = obs["reference_joint_position"]  # replace with a policy later
    obs, reward, terminated, truncated, info = env.step(action)
    print(info["state_index"], info["pelvis_error"], info["reward_components"])
    if terminated or truncated:
        print(info["reason"])
        break
```

Each environment owns its model and data. Reset restores the reference's initial
state, simulation clock, episode index, and previous-action observation. Reset is
required before the first step and after every episode; there is no auto-reset.

**Actions:** exactly 29 finite joint-position offsets in radians from `home`, in
configured joint order. Action scale is 1 rad/rad, not an assumed normalized
`[-1, 1]` range. Targets are clipped to the intersection of joint and actuator
control ranges. Position actuators then enforce configured torque limits.
`previous_action` records the actually applied target offset after clipping.
Requested offsets, requested targets, applied targets, and clipping flags are all
logged separately. A later policy adapter can introduce normalized action scaling
explicitly. By default no torque reference enters the actuator path. The diagnostic
`step(action, feedforward=...)` keyword adds an explicit 29-joint torque vector
through external generalized forces; it does not change actor observations.

**Timing:** observation/reference k leads to action k and ten 2 ms physics steps.
The returned observation and motion rewards use state/reference k+1. Mean total drive
torque over the interval is compared with reference `tau[k]` for diagnostics only.
Substep timestamps are **start times**: torque and contact solver outputs correspond
to that substep, whereas returned states correspond to the interval end. Kinematics
are refreshed after the interval before producing the observation.

**Episode end:** a pelvis height below 0.4 m at any sampled substep terminates the
episode as `fall`; the current 20 ms interval is completed. Reaching N intervals
truncates as `reference_end`. Fall takes precedence if both occur together. State N
is a valid terminal observation with phase 1; torque references are never indexed
at N. Non-finite simulation states or a MuJoCo time reset raise an error.

## Observations versus diagnostics

All observation arrays are copies. The intended motion-only actor inputs are:

| Key | Shape | Convention |
|---|---|---|
| `joint_position`, `joint_velocity` | 29 each | Offsets from home [rad], velocities [rad/s] |
| `base_linear_velocity`, `base_angular_velocity` | 3 each | Pelvis-local [m/s], [rad/s] |
| `projected_gravity` | 3 | Unit downward world vector expressed in pelvis axes |
| `previous_action` | 29 | Applied target offset from the previous interval; zero at reset |
| `reference_joint_position`, `reference_joint_velocity` | 29 each | Current reference offsets from home and joint velocities |
| `reference_base_linear_velocity`, `reference_base_angular_velocity` | 3 each | Current pelvis-local reference velocities |
| `reference_projected_gravity` | 3 | Reference orientation expressed as projected gravity |
| `phase` | 1 | Current index divided by reference length |

World base position, full state, torque references, measured contacts, and tracking
errors are debug/reward information in `info`, not actor observations. There is no
critic interface yet. This distinction avoids silently giving a motion-only actor
torque supervision. The trace stores observations under `observation_*` keys.

## Rewards and diagnostics

The scalar reward is the arithmetic mean of four terms. Each is in [0, 1] and
has maximum 1 at zero error:

| Term | Formula |
|---|---|
| `joint_position` | exp(-5 * sum squared joint-angle error) |
| `joint_velocity` | exp(-0.1 * sum squared joint-velocity error) |
| `pelvis_position` | exp(-20 * sum squared world pelvis-position error) |
| `pelvis_orientation` | exp(-10 * squared shortest quaternion angle error) |

These transparent diagnostic rewards are not a frozen reproduction of the paper's
reward table. There is no torque reward, action penalty, or terminal bonus/penalty.
Inspect components separately: a robot can maintain good joint tracking while its
base tips, so a high joint-position reward alone is misleading.

`substep_requested_torque` is the affine servo force before force clipping;
`substep_torque` is actual generalized actuator torque after clipping, excluding
passive damping, gravity, and contact forces. The code checks the ordered, stateless,
gear-1 actuator assumptions and Euler integration needed for this measurement.
`feedforward_torque` records the applied external torque for each interval;
`substep_total_torque` is actuator plus feedforward torque, and `mean_total_torque`
is its interval mean. `peak_effort_ratio` uses **total drive torque at every physics
substep**. `effort_ok` reports whether those totals respect limits (1e-6 tolerance).
External feedforward bypasses actuator clipping: a violation is recorded, not
silently clipped or used to terminate an inspection. Use `reference.validate` for
reference acceptance, including the separate exported-torque check.
`saturated_fraction` is the fraction of joint/substep samples where requested and
actual servo torque differ by more than 1e-6 N m. It is separate from target clipping.

Contact diagnostics count contact points involving each foot and sum their normal
force magnitudes from `mj_contactForce`. These are contact-frame normals, not full
world-aligned sole wrenches. The plot also shows reference world Fz for orientation
on the flat-floor task; do not interpret it as a general wrench-frame comparison.

## Read a saved trace

```python
import numpy as np
with np.load("runs/shallow_reference_01/trace.npz") as trace:
    print(trace["qpos"].shape)             # (executed intervals + 1, 36)
    print(trace["substep_torque"].shape)   # (executed intervals, 10, 29)
    print(trace["observation_phase"][-1])
    print(trace["joint_names"])           # array order for all joint channels
```

Tests cover deterministic reset, no external feedforward, observation copies,
substep timestamps, reference alignment, reward values, real actuator saturation,
contact logging, malformed actions, fall/end handling, and trace export. The next
step is choosing the learning adapter and training one policy on one reference;
dataset sampling and torque-informed comparisons can follow a working baseline.
