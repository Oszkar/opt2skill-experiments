# Opt2Skill replication: knowledge base

Retained learning notebook: research findings, environment notes, decisions, and
historical experiments from the Unitree G1 proof of concept. Dated entries record
what was known or tested then; they are not current setup instructions. Use the
[README](README.md), [hands-on guide](docs/TRAJECTORY_GUIDE.md), and
[next steps](docs/NEXT_STEPS.md) for current behavior and outstanding work.

Paper: Liu et al., *Opt2Skill: Imitating Dynamically-feasible Whole-Body Trajectories
for Versatile Humanoid Loco-Manipulation*, arXiv 2409.20514 **v6 (1 Oct 2025)**.
The local `Opt2Skill_paper.htm` is the saved v6 snapshot.
Project page: https://opt2skill.github.io (code "releasing soon", not released as of 2026-09-02).

---

## 1. Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-09-02 | Target robot: **Unitree G1** from day one | Digit has no public MuJoCo model. G1 is in MuJoCo Menagerie, has a URDF for Pinocchio/Crocoddyl, and is supported by MuJoCo Playground and mjlab. |
| 2026-09-02 | Run everything in **WSL2 Ubuntu 26.04** | Crocoddyl/Pinocchio have no Windows builds. JAX CUDA wheels are Linux-only. WSL2 already has GPU passthrough to the RTX 5070 Ti. |
| 2026-09-02 | Scope: PoC in a few days, learning + paper replication | Not a full reimplementation. Reproduce the core claim on one task. |

---

## 2. Paper digest (what matters for implementation)

### 2.1 Pipeline
1. **Trajectory optimization (TO)** with DDP in Crocoddyl + Pinocchio, full-order floating-base
   dynamics, rigid contact constraints, impact dynamics, joint/torque limits, friction cones.
   Task defined via desired task-space outputs `y_hat` (dense EE tracking for walking/stairs,
   sparse subgoals for pickup/desk tasks). Contact schedule is predefined per task.
2. **Reference dataset**: >1000 trajectories per task, parameters randomized (gait phase,
   speed, foot clearance, CoM position, object location, desk height, contact force).
   Each trajectory stores: joint pos/vel, base pos/orientation/lin vel/ang vel,
   end-effector positions, **joint torques**, **contact forces**.
3. **RL tracking policy** (PPO, asymmetric actor-critic) trained in MuJoCo Playground.
   One trajectory sampled per episode. Separate policy per task.

### 2.2 Policy I/O
- **Actor obs**: noisy base lin vel, base ang vel, projected gravity, joint-pos history
  (N=10 frames, every 4th step = 50 Hz samples from 200 Hz loop), joint vel, action history
  (same schedule) + partial reference: `[base lin vel, base ang vel, joint pos, contact force, torque]`.
- **Critic obs**: actor obs + privileged: base pos, orientation, EE pos (relative to torso),
  contact force, torque, Kp, Kd + full reference (base pose/twist, joint pos/vel, EE pos, F, u).
- **Action**: offset from **default standing pose** (not from the reference), 20 joints on Digit.
  `u = Kp (a + q_default - q) - Kd qdot`. Policy 200 Hz, PD 1 kHz in sim.
- **Network**: MLP [512, 512, 256, 256] for both actor and critic.
- **Training budget**: <300M steps, <7 h on one RTX 4090.
- History-length ablation: 10 frames best; 0 or 15 worse.

### 2.3 Rewards (Table I)
Task rewards (all `exp(-k * ||err||^2)`, weight 0.30 unless noted):
| Term | k | weight |
|---|---|---|
| joint pos | 5 | 0.30 |
| base pos (not used for walking) | 20 | 0.30 |
| base orientation | 50 | 0.30 |
| base lin vel | 2 | 0.30 |
| base ang vel | 0.5 | 0.30 |
| EE pos | 20 | 0.30 |
| joint torque | 0.01 | 0.10 |
| contact force (L1 norm) | 0.05 | 0.10 |

Penalties: action rate (2nd difference) -0.05, normalized torque -0.03, joint acc -1e-6.
Noise curriculum ramps observation noise and penalty strength up over training.

### 2.4 Domain randomization (Table II)
Gaussian obs noise (joint pos 0.0875, joint vel 0.075, base lin/ang vel 0.15, gravity 0.075),
action delay U[0, 20 ms], motor strength U[0.95, 1.05], Kp/Kd factor U[0.9, 1.1],
mass U[0.9, 1.1], gravity U[0.9, 1.1], friction U[0.3, 1.0], terrain {flat, rough}.

### 2.5 Key results
- Walking: TO refs beat human-retargeted and IK refs on task-space tracking and drift
  (hand 2.0 cm vs 4.25/5.47; foot 5.2 vs 9.4/5.2; x-drift 1.1% vs 4.5/2.1). IK gets the
  lowest *joint* error but worse task-space error: kinematic feasibility != dynamic consistency.
- Rough terrain: TO-trained policy survives 12 cm stairs; others drop after 8 cm.
- **Wiping ablation** (the cleanest torque-matters experiment): Pos / Pos+F / Pos+T / Pos+F+T.
  Force error 11.7 / 2.8 / 2.7 / 1.5 N. Pos+T alone already nearly matches Pos+F;
  Pos+F+T best. 1400 TO trajectories, desk height 0.85-0.95 m, normal force 0-20 N.
- Older v1 result (jump ablation): torque limits in TO cut take-off base error 7 cm -> 3 cm;
  torque tracking reward -> <1.5 cm.

### 2.6 Replication-relevant caveats
- Paper is not "RL from reference + residual"; the action is an offset from the default pose.
- Contact force in obs/reward requires measuring contact force in sim (MuJoCo: `mj_contactForce`
  or sensor `<force>` / contact sensors) and predicting it in TO (Crocoddyl contact models expose it).
- Reference torque needs joint-torque *output* from TO. Crocoddyl's `u` is exactly that for
  torque-actuated models.

---

## 3. Tooling landscape (verified 2026-09-02)

### 3.1 Trajectory optimization
- **Crocoddyl 3.2.x / Pinocchio 3.x**: pip wheels (`crocoddyl`, `pin`) exist for Linux and macOS only.
  conda-forge: linux-64, osx, no win-64. -> use WSL2.
  Ships bipedal gait examples (Talos, `SimpleBipedGaitProblem`) that are a template for G1.
- **OPT-Mimic** (quadruped predecessor) released TO code (CasADi) and RL code:
  https://github.com/yunifuchioka/opt-mimic-traj-opt , https://github.com/nickioan/robot2robot
- Fallback if Crocoddyl fights back: hand-rolled iLQR on MuJoCo with `mjd_transitionFD`
  finite-difference derivatives. Works on any MJCF, but soft contacts and no explicit
  contact-force decision variables.
- MuJoCo MPC (MJPC) also has iLQG with contact, but is a C++ build.

### 3.2 RL / simulation
- **MuJoCo Playground** (JAX/MJX + brax PPO): what the paper used. Linux/WSL2 only for GPU.
  Has G1 joystick env. **No motion-tracking env**; we'd write one.
- **mjlab** (Isaac Lab API on MuJoCo Warp, PyTorch, rsl_rl): has `Mjlab-Tracking-Flat-Unitree-G1`,
  a BeyondMimic-style motion tracker. MuJoCo Warp supports Windows natively, but we're in WSL2 anyway.
  Extending it with torque/force reference inputs and rewards is the likely path if we go this way.
- **MuJoCo Warp** (`mujoco-warp` on PyPI): GPU sim, Windows + Linux x86-64. No IMPLICITFAST integrator,
  no PGS/noslip solvers.
- **MuJoCo Python wheels**: available up to Python 3.13, none for 3.14 yet. Use uv to pin 3.12.

### 3.3 Robot models
- Unitree G1: `mujoco_menagerie/unitree_g1` (MJCF, 29 DoF and 23 DoF variants exist),
  URDF in Unitree's `unitree_ros` / `unitree_mujoco` repos for Pinocchio.
- Digit: no public MuJoCo model.

---

## 4. Machine / environment

| Item | Value |
|---|---|
| Host | Windows 11 Pro, RTX 5070 Ti 16 GB, driver 610.88, CUDA UMD 13.3 |
| Host Python | 3.14 only (no MuJoCo wheels); `uv` installed at `~/.local/bin/uv` |
| WSL2 | Ubuntu 26.04, 16 cores, 30 GB RAM, ~930 GB free, GPU visible via `nvidia-smi` |
| WSL2 Python | 3.14.4 system; uv-managed venv `~/venvs/o2s-to`, Python 3.12.14 |
| Build tools | no cmake / nvcc on host |

Project dir on Windows: `E:\Programming\opt2skill` (visible from WSL at `/mnt/e/Programming/opt2skill`;
for speed, put the venv and sim assets on the WSL filesystem, keep source on `/mnt/e` or clone there).

---

## 5. Open questions / to decide
Resolved: task = squat; RL stack = Playground + brax PPO; PoC done = Pos vs Pos+T ablation +
stress test (see spec); repo initialised 2026-09-02.

## 6. Gotchas / learnings

### 6.1 Gotchas found during environment spike (2026-09-02)
- **pip `crocoddyl==3.2.1` + `pin==4.1.0` segfaults** at the first `calc()` (even Crocoddyl's own Talos example). ABI mismatch: `libcrocoddyl 3.2.1` (uploaded 2026-05-21) was built against `libpinocchio 4.0.0` (same day); `pin 4.1.0` (2026-07-08) broke it. **Fix: pin `pin==4.0.0`.** Verified: quasiStatic + FDDP solve on Talos and G1 work. Fallback that also works: micromamba conda-forge `crocoddyl 3.2.1 + pinocchio 4.1.0` (built together).
- WSL venv: `~/venvs/o2s-to` (uv, Python 3.12.14): `crocoddyl==3.2.1 pin==4.0.0 mujoco==3.12.0 numpy pyyaml pytest`.
- Asset repos sparse-cloned into `~/o2s_third_party/` at: menagerie `8161bba264d7fa7c99ca301e91e7fb44737676ad` (2026-09-04), mujoco_playground `8a4b4642d8eba8a80ac99ed125cb62c16e1457ad` (2026-08-26), unitree_ros `7d6075f7f58588b189b940130e3edab3c839b2df` (2026-08-28).
- **Model match is near-perfect**: Playground `g1_mjx_feetonly.xml` (via `scene_mjx_feetonly_flat_terrain.xml` + Menagerie assets) and Unitree `g1_29dof_rev_1_0.urdf` have identical joint order (29), identical joint limits, total mass 33.3411 kg in both, CoM identical to 1e-8 at the home pose. Only pelvis mass differs by 1 g (URDF folds fixed links into the root).
- **Effort limit discrepancy**: URDF ankle pitch/roll and waist roll/pitch = 35 N m; MJCF `actuatorfrcrange` = 50 N m. We use the minimum (35) as source of truth for both models.
- Playground actuators: `<position>` with kp = 75 (ankle pitch 20, ankle roll 2, wrists 2), **kv = 0** (velocity damping comes from joint `damping` = 2, ankle pitch 1, ankle roll / wrists 0.2), `frictionloss` 0.1, per-joint armature (hip/knee ~0.01/0.025, ankle/waist 0.0072, arms 0.0036/0.00425). Actuator `forcerange` is unlimited; the clamp is joint-level `jnt_actfrcrange` (88/139/50/25/5). Transmission: joint, gear 1, one-to-one, so `actuator_force == qfrc_actuator` entries.
- Playground uses contact `<pair>`s (floor-foot, foot-foot, thigh-hand), not contype/conaffinity. Foot collision: box half-sizes (0.09, 0.03, 0.008) at (0.04, 0, -0.029) in `*_ankle_roll_link`, so the sole center is at (0.04, 0, -0.037) in that link. At the `home` keyframe the sole sits 1.2 mm into the floor (floor plane at z = 0).
- Keyframes in the Playground scene: `home` (pelvis z 0.785, knees slightly bent: hip pitch -0.1, knee 0.3, ankle pitch -0.2, shoulder pitch 0.2, shoulder roll +-0.2, elbow 1.28) and `knees_bent` (pelvis z 0.755). Playground's joystick uses `knees_bent` as default pose; we use `home`.
- Sites: `left_foot`/`right_foot` at ankle-roll-link origin, `left_palm`/`right_palm` at (0.08, 0, 0) in wrist-yaw links. Sensors include `left_foot_force`/`right_foot_force`, `local_linvel_pelvis`, `gyro_pelvis`, `upvector_pelvis`.
- Pinocchio 4 model has `armature` and `damping` fields (zeros from URDF). Crocoddyl's `DifferentialActionModelContactFwdDynamics.armature` accepts a size-nv (35) vector; `u_lb`/`u_ub` set torque box bounds (`has_control_limits` becomes True).
- Joint damping is not modeled in Crocoddyl contact dynamics. MuJoCo's passive damping (-b qdot) means the actuator must supply `tau_croc + b*qdot` to produce the same motion. Export applies this correction so `tau` in the npz is what MuJoCo's actuators must output.
- Pinocchio free-flyer q = [pos(3), quat xyzw(4), joints]; MuJoCo qpos = [pos(3), quat wxyz(4), joints]. Pinocchio v = [lin vel LOCAL, ang vel LOCAL, qdot]; MuJoCo qvel = [lin vel WORLD, ang vel LOCAL, qdot].
- Contact wrench from Crocoddyl data: `problem.runningDatas[k].differential.multibody.contacts.contacts["<name>"].f` (pin.Force), expressed in the frame selected by the contact `type` (we use `LOCAL_WORLD_ALIGNED`: world-aligned axes at the contact frame origin).
- Bash-tool quirk on this Windows host: multiple heredocs in one command fail; write scripts to the scratchpad with the Write tool, strip CRs with `tr -d "\r"`, run in WSL.
- **Playground G1 model bug**: `g1_mjx_feetonly.xml` applies one shared `hip_roll` default class (`range="-0.5236 2.9671"`) to both hips, so `right_hip_roll_joint` gets the left hip's range (and, via `inheritrange`, the same wrong actuator ctrlrange). Menagerie's `g1.xml` and Unitree's URDF both have the mirrored `[-2.9671, 0.5236]`. Found by the reconciliation `joint_limits` check. We override it from `configs/g1_reconcile.yaml` (`joint_range_overrides`) at MuJoCo load time; URDF governs. Irrelevant for a squat (hip roll stays near zero) but it would matter for any RL policy exploring toward the limits, and it is worth an upstream issue.
- **Plan-code bugs caught by the implementer against real assets**: (1) MuJoCo body-to-Pinocchio-joint mapping cannot use a `_link -> _joint` name rule (`torso_link` is driven by `waist_pitch_joint`); read the joint from `body_jntadr` instead. (2) `mj_objectVelocity` with `mjOBJ_BODY` reports the velocity of the inertial frame (`xipos`); `mjOBJ_XBODY` is the body frame origin (`xpos`), which is what matches Pinocchio's frame velocity.
- **Reconciliation check results** (`python -m o2s.models.reconcile`, 2026-09-02): joint limits max diff 2.05e-6 rad (after the hip-roll override); total mass 33.3411 kg both; worst relative body-mass error 2.6e-4 (pelvis, fixed links merged in URDF); worst body-CoM error 2.9e-5 m; worst relative principal-inertia error 6.3e-4; forward kinematics at 20 random poses within 4.6e-7 m; all 29 actuators gear-1 joint transmissions. The two models are effectively identical.
- **Squat TO tuning**: with `SquatWeights.com = 1e4` the solver smooths the profile and starts descending during the 0.3 s standing phase (summed normal force at node 0 = 288.6 N vs mg 327.1 N, 12% low). With `com = 1e5` (now the production default): 12 cm test squat converges in 30 iterations, 1.4 s; fz0 = 314.5 N (3.9% low, residual pre-emptive motion); min CoM 0.5687 m (target 0.5683), final CoM 0.6853 m vs standing 0.6883 (3 mm). A 1 cm CoM error at 1e4 costs only 1.0, comparable to the regularization terms, which is why tracking was loose.
- **Superseded by the next bullet.** **Export filter finding, unresolved**: with production defaults (`com=1e5`, `baumgarte=(0.0, 50.0)`), `filter.check_solution` rejects both the 15 cm `tests/test_export_filter.py` fixture and the CLI's 20 cm default squat (`python -m o2s.trajopt.solve_one --depth 0.2`) on **foot drift**, which CoM-tracking weights alone do not address. 15 cm case: foot drift 7.24 mm / 0.003 mrad (limit 1 mm / 1 mrad); all other details pass (max_torque_ratio 0.231, joint_limit_margin 0.0497, min_normal_force 143.2 N, max_friction_ratio 0.0127, depth_error 0.3 mm). 20 cm case additionally fails **joint limit margin**: -0.0243 rad (left_ankle_pitch_joint at node 98 sits at -0.897 rad, past its -0.87267 rad lower limit by 1.4 deg); foot drift 8.50 mm. In both cases drift grows monotonically and smoothly from node 0 to the final node (left sole: 1.3e-7 m at k=1 to 8.50e-3 m at k=170), consistent with `ContactModel6D`'s Baumgarte position gain being 0 (`baumgarte=(0.0, 50.0)`): only velocity-level stabilization is applied, so any position-level contact drift is never corrected, just prevented from growing faster. The following sweep resolved these failures by adjusting contact stabilization and the joint-limit penalty.
- **Squat TO tuning, resolved through a gain and weight sweep**: full Baumgarte-gain x `SquatWeights` sweep on both the 15 cm fixture and the 20 cm CLI squat, `joint_limits=1e5` and (for the last five rows) `joint_limit_margin=0.04` held fixed. `(bg0, bg1)` = (placement-error gain, velocity-error gain).

  | case | baumgarte | joint_limit_margin | converged | iters | solve_time (s) | foot_drift_pos (mm) | foot_drift_rot (mrad) | joint_limit_margin detail | cop_margin_violation | max_friction_ratio | depth_error (mm) | filter ok |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | fixture 15 cm | (0.0, 50.0) (baseline) | 0.02 | True | — | — | 7.24 | 0.003 | 0.0497 | 0.0000 | 0.0127 | 0.27 | False — foot drift only |
  | fixture 15 cm | (10.0, 50.0) | 0.02 | True | 9 | 0.683 | 6.08 | 0.00213 | 0.0490 | 0.0000 | 0.0126 | 0.26 | False — foot drift |
  | fixture 15 cm | (30.0, 50.0) | 0.02 | True | 6 | 0.487 | 4.63 | 0.00156 | 0.0473 | 0.0000 | 0.0123 | 0.26 | False — foot drift |
  | fixture 15 cm | (100.0, 20.0) | 0.02 | True | 4 | 0.334 | 1.71 | 0.00065 | 0.0415 | 0.0000 | 0.0100 | 0.27 | False — foot drift |
  | fixture 15 cm | (200.0, 30.0) | 0.04 | True | 17 | 1.353 | 1.35 | 0.00052 | 0.0413 | 0.0000 | 0.0099 | 0.26 | False — foot drift |
  | fixture 15 cm | (300.0, 35.0) | 0.04 | True | 4 | 0.333 | 1.11 | 0.00043 | 0.0413 | 0.0000 | 0.0098 | 0.26 | False — foot drift |
  | fixture 15 cm | **(100.0, 20.0) (adopted)** | **0.04** | True | 4 | 0.332 | 1.71 | 0.00065 | 0.0415 | 0.0000 | 0.0100 | 0.27 | passes at the widened 3 mm/3 mrad threshold |
  | CLI 20 cm | (0.0, 50.0) (baseline) | 0.02 | True | 11 | 1.06 | 8.50 | 0.00 | -0.0243 (hard-limit violation) | 0.0000 | 0.0256 | 2.2 | False — joint limit + foot drift |
  | CLI 20 cm | (10.0, 50.0) | 0.02 | True | 15 | 1.406 | 7.04 | 0.00353 | 0.0192 | 0.0000 | 0.0697 | 3.23 | False — joint limit + foot drift |
  | CLI 20 cm | (30.0, 50.0) | 0.02 | True | 30 | 3.093 | 5.08 | 0.00248 | 0.0192 | 0.0050 | 0.1041 | 3.27 | False — joint limit + CoP + foot drift |
  | CLI 20 cm | (100.0, 20.0) | 0.02 | True | 8 | 0.762 | 1.75 | 0.00073 | 0.0192 | 0.0000 | 0.0833 | 3.41 | False — joint limit + foot drift |
  | CLI 20 cm | (200.0, 30.0) | 0.04 | True | 9 | 0.867 | 1.36 | 0.00073 | 0.0391 | 0.0000 | 0.0926 | 3.87 | False — foot drift |
  | CLI 20 cm | (300.0, 35.0) | 0.04 | True | 9 | 0.850 | 1.10 | 0.00060 | 0.0391 | 0.0000 | 0.0926 | 3.87 | False — foot drift |
  | CLI 20 cm | **(100.0, 20.0) (adopted)** | **0.04** | True | 8 | 0.768 | 1.76 | 0.00090 | 0.0391 | 0.0000 | 0.0914 | 3.86 | passes at the widened 3 mm/3 mrad threshold |

  Raising `joint_limit_margin` from 0.02 to 0.04 (with `joint_limits=1e5`) fixes the CLI case's hard-limit violation (-0.0243 rad -> +0.0192..0.0391 rad, well clear of the filter's unchanged 0.02 rad bound). Higher Baumgarte position gain monotonically reduces foot drift (10 -> 300) and converges in fewer iterations at the extremes, but even the stiffest tested gain, `(300.0, 35.0)`, only reaches 1.10-1.11 mm on both squats — still over a 1 mm bound; `(30.0, 50.0)` is a worse local optimum on the harder 20 cm squat (30 iterations, introduces a 5 mm CoP-margin violation not seen at neighboring gains). Since no candidate in either sweep round got under 1 mm on both squats, the controller approved adopting `(100.0, 20.0)` (best trade of drift vs. iteration count/robustness) as the `SquatWeights.baumgarte` default and widening `filter.py`'s foot-drift thresholds from 1e-3 m / 1e-3 rad to 3e-3 m / 3e-3 rad — a controller-approved spec deviation, not a filter bug fix: "20 ms Euler integration leaves ~1.7 mm of constraint drift with Baumgarte (100, 20); the spec's 1 mm was tightened past what the integrator delivers." Production defaults after this task: `joint_limits=1e5`, `joint_limit_margin=0.04`, `baumgarte=(100.0, 20.0)`; `filter.py` foot-drift bound 3e-3/3e-3. With these, `tests/test_export_filter.py` is 3/3 green, `python -m o2s.trajopt.solve_one --depth 0.2` converges in 8 iterations / 0.73 s with no FAIL lines, and the full suite is 25/25 green with no warnings.
- **Validation redesign** (replay gains later raised to 4x/400, see Replay stabilizer gains): the first validation design (open-loop replay for 0.5 s; feedforward + low-gain PD) cannot pass on the G1 regardless of reference quality. Standing on the ankles is an inverted pendulum with destabilizing stiffness m g h ~ 33.3 kg * 9.81 * 0.69 m ~ 225 N m/rad; Playground's ankle-pitch PD gain is 20 N m/rad (RL provides balance), so pure PD at Playground gains falls over. A "static torque" probe with MuJoCo's `mj_inverse` at the `home` keyframe reported 78 N m at the ankles: that is the soft-contact reaction to the keyframe's 1.2 mm sole penetration, not a model gap (Pinocchio's quasi-static ankle torque of 5.4 N m matches the hand estimate, CoM 3.4 cm ahead of the ankle). Replacement checks: (1) inverse-dynamics consistency at every node (MuJoCo `mj_inverse` with contacts and friction loss disabled, reference sole wrenches applied via `mj_applyFT`, finite-difference acceleration from `qvel`), which compares the two dynamics models directly; (2) feedforward + stabilizing PD (2x Playground gains, ankle pitch 200 N m/rad) with the correction logged; (3) PD-only replay as an informational contrast. The standing pose is now lifted 1.2 mm so the soles start on the floor.
- **Milestone 1 reached (2026-09-02): one validated squat** (replay gains later raised to 4x/400, see Replay stabilizer gains). Dev reference: 0.2 m squat, N = 170 (3.4 s), from `python -m o2s.trajopt.solve_one --depth 0.2`. Validation (`python -m o2s.reference.validate data/refs/dev/squat_dev.npz`):
  - Check 1, inverse-dynamics consistency over all 170 nodes: joint torque residual rms 0.029 N m, max 0.215 N m (worst: right knee), base residual 0.38 N / 0.00 N m, against a reference torque rms of 5.77 N m. The Pinocchio and MuJoCo dynamics agree to well under 1%. Gotcha found on the way: `mj_forward` overwrites `data.qacc`, so the finite-difference acceleration must be written after it and before `mj_inverse`.
  - Check 2, feedforward + stabilizing PD (2x Playground gains, ankle pitch 200 N m/rad): completes, max pelvis error 2.6 cm, rms joint error 0.0067 rad, PD correction / reference torque rms = 0.11 (legs), 0.19 (waist), 0.14 (arms).
  - Check 3, same PD without feedforward: falls (pelvis reaches z = -0.77 m), rms joint error 0.053 rad, PD torque rms 11.6 N m vs 8.8 N m reference. Same gains, same targets; the only difference is the torque feedforward. This is the Opt2Skill thesis in miniature: the reference torque carries information the kinematic reference does not.
- **Milestone 2 reached (2026-09-02): randomized dataset.** `python -m o2s.trajopt.generate --n 100 --out data/refs/squat --seed 0`: 100 accepted from 109 attempts (accept rate 0.92; the 9 rejects were all deep squats, 0.216-0.30 m, on joint-limit margin / CoP), solve time mean 1.9 s, max 10.3 s, wall time 227 s. Depth histogram, 4 bins over edges [0.10, 0.15, 0.20, 0.25, 0.30]: 20 / 25 / 31 / 24. Split 70 / 10 / 20 in `data/refs/squat/split.json`. Every accepted file passes the inverse-dynamics check (hard gate in the generator) and the closed-loop replay check (recorded per file under `meta["replay_ff"]`, pass rate 1.0, zero falls).
- **Replay stabilizer gains**: at 2x Playground gains / ankle pitch 200 N m/rad, 30 of the first 100 references failed check 2 (10 falls, 20 marginal misses). Diagnosis: no torque saturation, no CoP violation anywhere, no clean correlation with depth or duration, and every one of the 10 falls tracks within 1.1 cm at 4x/400. The stabilizer was too soft for deep, slow squats; the references were fine. Defaults are now 4x / 400. With those gains the PD-only replay of the dev squat still falls (pelvis reaches z = -0.77 m) while feedforward + PD tracks to 1.1 cm with the PD supplying 10% of the leg torque: stiffness alone does not stabilize the reference, the torque feedforward does.
- **Bash-through-WSL quirk**: `$f` in a `for` loop inside `wsl -d ... bash -lc '...'` was stripped by the host tool in one subagent session; a small Python loop over `Path.glob` is the robust way to iterate files from this host.
- **Doc dates vs. commit dates**: milestone and gotcha entries in this document and the spec are dated 2026-09-02, when the spec was authored and milestones 1-2 were substantively reached; the final-review fix-wave commits that sharpened check 1, hardened the generator/contract, reconciled foot geometry/actuator types, and updated these docs landed 2026-09-05.
- **Check 1 after the damping-convention fix**: the dev squat's inverse-dynamics residual is rms 0.0007 N m, max 0.018 N m (was rms 0.029 / max 0.215, which was entirely the export's interval-mean damping term). At max < 0.05 N m the generator rejected 12 of 122 attempts, all deep squats 0.23-0.30 m, on waist pitch (0.05-0.10 N m, rms 0.002-0.004): the finite-difference acceleration error scales with motion intensity. Bound raised to 0.2 N m (still under 1% of the smallest leg effort limit, 15x sharper than the original 3.0). Regenerated: `python -m o2s.trajopt.generate --n 100 --out data/refs/squat --seed 0` now gives 100 accepted from 109 attempts (accept rate 0.917), depth histogram [20, 25, 31, 24] over edges [0.10, 0.15, 0.20, 0.25, 0.30], replay_ff pass rate 1.0 with 0 falls, split 70/10/20; all 9 rejects are filter-stage (7 foot drift, 1 depth error, 1 CoP margin), zero on inverse-dynamics.

---

## 7. Glossary and background (added 2026-09-02)

### Trajectory optimization
- **OCP**: minimize summed cost over x[0..N], u[0..N-1] s.t. dynamics + constraints. LQR = linear dynamics, quadratic cost, one Riccati sweep.
- **Direct transcription / collocation**: all x, u are NLP variables, dynamics as equality constraints, sparse NLP solver (CasADi + IPOPT; used by OPT-Mimic). Flexible constraints, large NLP.
- **Shooting**: only u are variables; x from rollout. DDP/iLQR are structured shooting methods.
- **DDP** (Mayne 1966): iterate LQR around current trajectory. Backward pass = 2nd-order cost-to-go expansion -> feedforward + feedback gains; forward pass = nonlinear rollout with line search. O(N) per iteration, gives a local feedback law.
- **iLQR**: DDP without 2nd-order dynamics derivatives (Gauss-Newton). What most "DDP" in robotics actually is.
- **FDDP** (Mastalli 2020): DDP that tolerates infeasible initial guesses via gaps between shooting nodes (multiple-shooting flavor). Crocoddyl default.
- **BoxFDDP**: FDDP + box constraints on u (Tassa 2014 control-limited DDP), box-QP in backward pass. Hard torque limits.
- **Pinocchio**: rigid-body dynamics with analytical derivatives (RNEA/ABA and their derivatives). Enables fast DDP on floating-base robots.
- **Crocoddyl**: Contact RObot COntrol by Differential DYnamic programming Library. Action models, cost residuals, contact models, FDDP/BoxFDDP solvers on top of Pinocchio.
- **Contact forward dynamics**: rigid contact = zero acceleration of contact frame; KKT system yields qddot and contact forces together. Forces are usable as costs (friction cone) and exported references.
- **Impact dynamics**: velocity jump at touchdown, M(v+ - v-) = J^T Lambda. Needed for walking, not squat.
- **Contact schedule**: predefined sequence of contact frames per phase. Input, not output.
- **Friction cone**: |f_t| <= mu f_n, linearized to a pyramid, quadratic barrier cost.

### Simulation
- **MuJoCo**: CPU simulator, soft convex contact model, smooth contact forces.
- **MJX**: MuJoCo in JAX. vmap thousands of envs on GPU, jit the step. Compile times of minutes; functional style.
- **MuJoCo Warp**: GPU port on NVIDIA Warp, faster for contacts; MJX can use it as backend.
- **MuJoCo Playground**: DeepMind env library + training recipes on MJX/brax, includes tuned Unitree G1 config.

### RL
- **PPO**: on-policy actor-critic, GAE advantages, clipped surrogate objective, several epochs per batch. Stable at 1000s of parallel envs.
- **brax**: Google JAX physics + RL; its PPO jits env step + rollout + update together (100k+ steps/s). Playground uses it.
- **rsl_rl**: ETH PyTorch PPO (Isaac Gym lineage). Used by mjlab.
- **Asymmetric actor-critic**: critic sees privileged state; actor sees deployable sensors only.
- **PD position targets as actions**: policy outputs joint targets, low-gain PD -> torque (Hwangbo 2019 onward). Opt2Skill offsets from default pose, not from reference.
- **Domain randomization / noise curriculum**: randomize physics + sensor noise per env; ramp noise up during training.

### Imitation lineage
- **DeepMimic** (2018): mocap tracking, exp(-k err^2) rewards, reference state initialization (RSI), early termination (ET).
- **OPT-Mimic** (2022): Solo 8 quadruped, TO references (direct collocation), study of feedforward designs from the optimizer; backflip etc. on hardware.
- **MIMOC** (2023): Mini Cheetah + MIT Humanoid (sim), model-based OC references with torques; imitating torque refs improves the policy; torques enter via learning signal.
- **Opt2Skill** (2024-25): full-order humanoid, loco-manipulation, hardware, Pos/Pos+F/Pos+T ablation.

### Trade-offs of our choices
- Crocoddyl: hard optimizer torque limits + contact forces in the reference; verified here under WSL2. Pinned Python 3.12 packages also have macOS wheels, but this project has not been tested there. Pinocchio rigid contacts differ from MuJoCo soft contacts; feedback may accommodate some mismatch, but RL does not automatically close that gap.
- Playground + brax PPO: paper's stack, throughput; JAX compile cycles.
- Squat: one contact phase; walking later needs impacts + schedule.
- Offset-from-default actions, torque only in obs/reward: clean ablation; leaves feedforward-torque designs (OPT-Mimic line) as a follow-up.
- 50 Hz policy / 2 ms physics: Playground's tested G1 config; paper used 200 Hz / 1 kHz.

---

## 8. Policy training design notes

Trajectory generation produced a validated development squat and a 100-trajectory
randomized dataset under `data/refs/`. These retained notes describe the proposed
tracking environment; no policy-training implementation or `rl` extra exists yet.

- **Dataset**: `data/refs/squat/squat_*.npz` (100 files, contract in `o2s/reference/contract.py`),
  plus `data/refs/squat/split.json` (train/val/test = 70/10/20, names only) and
  `data/refs/squat/summary.json` / `rejected.jsonl`. Load a directory with
  `contract.load_dir(dir_path, names=None)`, which returns a `ReferenceSet` of padded arrays and
  a per-trajectory length vector (JAX needs rectangular data); `contract.load_dir`, not
  `contract.load` (single file), is the loader the tracking env should use.
- **Reference contract's timing rule**: at control step k the policy observes state k and
  reference k, acts over `[t[k], t[k+1])`. The torque reward compares the substep-mean
  `qfrc_actuator[6:]` over that interval with `tau[k]` (the TO's zero-order hold). Episode has N
  control steps; state N is truncation-only and carries no torque reference -- do not index
  `tau` or `foot_wrench_*` at N.
- **`tau` already includes MuJoCo joint damping** (`dof_damping * qdot`, interval-averaged) --
  it is the torque MuJoCo's actuators must output, not Crocoddyl's raw control `u`. Do not add
  damping again when comparing to `qfrc_actuator`.
- **Playground facts the env needs**: `data.actuator_force` equals `qfrc_actuator[6:]` here
  because every actuator is a gear-1, one-to-one joint transmission (verified by the
  reconciliation `transmission` check) -- either quantity can be read, `qfrc_actuator` is the
  spec's definition. Kp values: 75 for most joints, ankle pitch 20, ankle roll 2, wrists 2; kv is
  always 0 (velocity damping comes from joint `damping`, not the actuator). Two keyframes exist:
  `home` (pelvis z 0.785, our default standing pose, soles lifted 1.2 mm to sit at z = 0) and
  `knees_bent` (pelvis z 0.755, Playground's joystick default) -- use `home`, not `knees_bent`.
  Sensors available in the MJCF: `left_foot_force` / `right_foot_force`,
  `local_linvel_pelvis`, `gyro_pelvis`, `upvector_pelvis`.
- **Historical candidate dependencies (not installed by an `rl` extra)**: `playground==0.2.0`, `jax[cuda12]`, Python
  3.12 (matches the `trajopt` venv's Python; separate venv/group since `trajopt` needs
  `pin==4.0.0` per 6.1's ABI note).
- **Replay-stabilizer gains as a "PD-only" baseline hint**: validate.py's check 3 (PD only, no
  torque feedforward) at 4x Playground gains / ankle pitch 400 N m/rad falls on the dev squat
  (pelvis reaches z = -0.77 m). That is a plain joint-space PD tracker with no RL and no torque
  reference -- a natural zero-effort baseline for the Pos-arm-without-feedforward comparison, or
  a smoke test that a candidate env config is not accidentally stabilizing on its own.
- **`SquatWeights` / `SquatRanges` defaults now in code are the ones to keep**: `SquatWeights` in
  `o2s/trajopt/squat_problem.py` (`com=1e5`, `joint_limits=1e5`, `joint_limit_margin=0.04`,
  `baumgarte=(100.0, 20.0)`, etc.) and `SquatRanges` in `o2s/trajopt/generate.py`
  (`depth=(0.10, 0.30)`, `t_down`/`t_up=(0.8, 1.5)`, `t_hold=(0.2, 0.6)`,
  `com_shift_x=(-0.03, 0.03)`) were tuned against real solves (6.1's sweeps) -- do not revert to
  earlier planning numbers when regenerating or extending the dataset.


---

## 9. From optimized motion to feedback control: study notes

These explanations summarize the implementation walkthrough. They distinguish
what the code does today from possible extensions; proposed policy interfaces,
MPC, and general task primitives are not implemented.

### 9.1 What each command establishes

| Command | What happens | What a pass means |
|---|---|---|
| `python -m o2s.models.reconcile` | Loads both robot models and compares properties after configured overrides | The models agree on the checked structure, mass properties, and geometry |
| `python -m o2s.trajopt.solve_one` | Optimizes states and torques, filters feasibility, and exports a reference | The candidate satisfies this optimizer's post-solve checks |
| `python -m o2s.reference.view FILE --mode kinematic --loop` | Sets each saved state directly, like animation | Visual inspection only; no proof that physics can execute it |
| `python -m o2s.reference.view FILE --mode replay --loop` | Simulates reference torque plus fixed PD feedback | A visual demonstration of this controller tracking the reference |
| `python -m o2s.reference.validate FILE` | Computes inverse dynamics and runs two headless physics rollouts | Quantifies model consistency and configured-controller tracking |

Reconciliation is a prerequisite, not trajectory validation. `joint_names` checks
all 29 names and their order. `joint_limits` checks angle bounds. Total and per-link
mass, link CoM offsets, and principal inertia values check mass properties.
Forward kinematics compares selected frame positions in 20 reproducible random
poses; it is not a full pose/velocity comparison. Principal moments alone do not
verify the complete oriented inertia tensor. Foot geometry checks the collision
boxes against configured sole offsets and dimensions. Transmission checks verify
one actuator per joint with unit simulated gearing (not the physical gearbox).
The actuator-type check verifies the gain/bias relationship expected by the
position servos. A pass describes the loaded models after corrections, including
the right-hip range override, rather than identical upstream files.

`validate` uses real MuJoCo in the Python process without a window. Its inverse-
dynamics check evaluates snapshots without integrating time. Each of its two
replays resets the initial state and advances 1,700 physics steps for a default
170-interval squat: ten 2 ms steps per 20 ms interval. Unlike kinematic playback,
replay does not overwrite the state with the reference after every step. The
headless run has no real-time pacing, so several seconds of simulation can finish
in much less wall-clock time. Feedforward is applied through external generalized
forces and bypasses actuator clipping; reported effort uses interval means.

### 9.2 Initial condition, initial guess, and the search space

The initial condition is fixed: the MJCF `home` pose, converted to Pinocchio
coordinates, with the base lifted so the lowest sole frame is at ground height,
and all base and joint velocities zero. This implementation cannot solve from
an arbitrary starting pose or recover an arbitrary falling state.

The initial *guess for the trajectory* is a different thing: repeat that standing
state at every node and initialize controls with quasi-static torques. The guess
contains no squat. The target CoM path supplies the task: 0.5 s standing, 1 s
smooth descent, 0.4 s hold, 1 s ascent, 0.5 s standing by default. Depth, descent,
hold, ascent, and fore-aft shift are CLI parameters; standing durations are also
parameters in the Python API. Joint-angle trajectories are not prescribed.

Each stored state has 71 values: base position (3), quaternion (4), joint angles
(29), base linear/angular velocity (6), and joint velocities (29). The independent
state/tangent dimension is 70 because a quaternion represents three rotational
degrees of freedom. For a default squat the solution has 171 states and 170
29-dimensional torque vectors. The base has no direct actuator. Contact forces
follow from the constrained dynamics rather than independent free controls.

BoxFDDP uses derivatives and structured backward/forward passes to improve a
continuous trajectory; it does not enumerate combinations of joint angles. Fixed
contacts, a nearby standing guess, a smooth target, and posture regularization
make the shallow squat relatively easy. Convergence is local, not a guarantee of
global optimality or success for another task. The initial state is fixed; return
to standing is encouraged by costs and checked afterward, not an exact fixed
terminal state. Torque bounds are hard solver bounds on raw control; some other
requirements are penalties plus export filters.

### 9.3 Reading the shallow-squat experiment

User-reported 15 cm run: three iterations, 0.33 s solver time, 170 intervals.
Cost 4011.555 is a weighted objective, not a physical error or universal score.
The minimum achieved CoM height was within 0.3 mm of the requested drop; this is
whole-body CoM depth, not necessarily the pelvis drop. Raw optimizer torque used
at most 22.08% of configured effort limits, before export adds damping.

The joint-limit margin was 0.0415 rad (about 2.38 degrees); minimum normal force
was 149.3 N per foot over the sampled horizon. Friction utilization 0.0091 means
0.91% of modeled friction capacity. CoP margin violation -0.0248 m means 24.8 mm
of clearance inside the already-inset permitted boundary; positive would mean a
violation. Foot drift was 1.2 mm. Printed zero residuals mean rounding, not exact
mathematical zero.

Feedforward replay had maximum pelvis error 7.7 mm, joint error 0.0043 rad RMS,
and leg feedback RMS 0.6406 N m versus reference RMS 8.1619 N m (7.85%). The
waist ratio was 54.12%, but absolute correction was only 0.2712 N m: inspect
absolute values alongside ratios. The interval-mean effort ratio peaked at
25.43%, which is not an instantaneous torque-limit guarantee.

PD-only replay fell despite relatively small joint-angle error: the floating
robot can tip while its internal joint configuration stays near the target.
Negative pelvis height after falling reflects this foot-only collision scene;
the torso need not stop at the floor. This is evidence about the configured
controller, not universal failure of PD or a learned-policy ablation.

### 9.4 IK, task primitives, and motion libraries

Traditional inverse kinematics finds a configuration that achieves a geometric
target, potentially with joint limits, posture preferences, and collision checks.
Trajectory optimization finds timed states and controls under dynamics and
contacts. A sequence of IK poses does not establish feasible accelerations,
torques, or balance. IK can nevertheless provide useful poses or initial guesses
for the trajectory optimizer.

A reusable task vocabulary would include:

| Primitive | Purpose | Example |
|---|---|---|
| State or pose | Initial state, seed, or posture preference | Standing with zero velocity |
| Frame/CoM target | Desired position, orientation, or CoM behavior | Palm relative to a mug; CoM over support foot |
| Contact | Active robot/surface relationship and force limits | Sole planted; hand holding object |
| Phase | Duration, targets, and active contacts | Reach, hold, lift |
| Constraint | Requirement enforced explicitly | Torque limits, collision avoidance |
| Cost | Preference traded against other objectives | Smoothness, low effort, posture |
| Transition | Change between phases | Liftoff, touchdown, grasp attachment |

A standing reach is a useful next extension: retain two-foot contact and add a
palm target while balancing/posture costs guide the rest of the body. A terminal
hand target leaves more freedom than a full hand path. Walking introduces contact
phases and touchdown dynamics. Pickup additionally needs object dynamics and
grasp constraints. Object-relative targets make tasks reusable when objects move.
A predefined contact schedule still does not discover which foot to move or where
to land; contact-strategy search is an additional planning problem.

Pose libraries can initialize poses; motion libraries can supply paths and warm
starts; task libraries can supply parameterized phase sequences. Human or other-
robot motions need retargeting and dynamic validation. A library of our own
successful optimized trajectories can seed nearby problems. Valid clips cannot
simply be concatenated: position/velocity continuity, contacts, collisions, and
actuator feasibility must also hold across transitions. No external pose or task
library has been selected for this codebase.

### 9.5 References in training, evaluation, and execution

A proposed tracking policy consumes current observations and reference information
and produces joint-position offsets from the default pose; PD actuators turn those
targets into torque. This differs from today's controller, which directly injects
reference torque as feedforward. Torque references can instead enter policy/critic
observations and rewards; these interventions should be distinguished experimentally.

Use optimized trajectories in **both training and evaluation**, with separate
sets. Training episodes sample a reference and reward tracking under physics,
optionally with sampled start nodes, perturbed states, and domain randomization.
The 70/10/20 split supports training, settings selection on validation data, and
final held-out testing. Freeze the policy for evaluation and separately measure
nominal tracking, unseen trajectory parameters, disturbances, and model changes.
Robustness must be trained and measured; it does not follow from using a network.
A reference-conditioned policy generally still receives the reference at runtime.
`reference.validate` evaluates references and a fixed controller, not learned policies.

### 9.6 Why learn a policy instead of optimizing online?

Online replanning is a legitimate alternative: model predictive control repeatedly
solves from the current estimated state, executes a short portion, and solves
again. A saved plan alone cannot react to a push; PD, trajectory-local feedback,
MPC, or a learned policy supplies that feedback in different ways. Crocoddyl's
local feedback gains are another possible baseline; we do not export/use them yet.

A trained policy moves substantial computation offline and offers a predictable,
cheap runtime evaluation. Training across relevant variations can teach coordinated
corrections, but behavior outside that distribution remains uncertain. MPC can
explicitly handle new targets and constraints, yet depends on model/state accuracy
and meeting solve deadlines. Robust optimization is also possible; uncertainty
handling is not exclusive to learning.

The observed 0.33 s offline solve spans about 16.5 of our 20 ms control intervals.
That is not an MPC benchmark: MPC may use a shorter horizon, the previous solution
as a warm start, and limited iterations. Nor is fast replanning from any viable
pose established by our fixed-standing solver. A pose can be geometrically valid
while its velocity/contact state makes recovery impossible.

The useful experiment is therefore whether a learned tracker improves tracking,
recovery, and runtime cost relative to practical feedback baselines. RL is an
approach to evaluate, not a necessary consequence of having optimized references.


### 9.7 Validation safeguards added after these experiments

The historical effort figures above used interval means. Replay now reports
`peak_effort_ratio` across every physics substep, keeps the former measure as
`peak_interval_mean_effort_ratio`, and records `effort_checked_per_substep=1`.
Both exported reference torque (after damping compensation) and total
feedforward-plus-PD drive torque must respect configured effort bounds. Violations
fail feedforward validation and reject generated trajectories. Other replay
tracking failures remain informational in generation. The feedforward path is
still external/unclipped: the check rejects excess rather than concealing it.

Rechecking all 102 local references found no failures. Maximum substep torque ratio
was 0.9270, versus the former interval-mean maximum 0.8926; maximum exported reference
ratio was 0.7199. Existing files and their historical metadata were not rewritten.
Re-run validation to obtain current metrics; absence of the substep-check marker
means saved replay metadata predates this guarantee.

Reference loading now requires at least one interval, time starting at zero with
uniform 20 ms increments (absolute tolerance 1e-9 s), and unit base quaternions in
both `qpos` and `pelvis_quat` that agree up to sign (1e-6 tolerance). Dataset generation
rejects any nonempty output directory before model loading or writes, preserving
old runs. No resume/overwrite mode has been added.
