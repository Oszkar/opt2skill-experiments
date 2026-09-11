# Next steps

Trajectory generation, model reconciliation, reference export, numerical
validation, visualization, and a [single-reference tracking environment](TRACKING_ENV.md)
with diagnostic traces are implemented. Policy training and hardware
execution are not. See the [hands-on guide](TRAJECTORY_GUIDE.md) to use what exists.

## Make reference data a reliable training input

Implemented safeguards now reject nonempty output directories, enforce positive
reference lengths, valid 20 ms timing, consistent duplicate base states and
metadata, and gate exported plus per-physics-substep replay torque limits. Squat
parameters are checked before solving. A tested dependency lock and full dataset
provenance/file manifests are implemented; see [reproducibility](REPRODUCIBILITY.md).
Remaining work:

1. Verify manifests, enforce split selection, and respect padding masks at the
   future training boundary. Require an explicit policy for legacy/missing provenance.
2. Decide whether replay tracking failures should also gate training data; they
   remain informational during generation, unlike torque violations.
3. Keep the eventual training dependency lock separate from trajectory optimization
   and revalidate the numerical baseline when either environment changes.

## Build a tracking policy

The detailed design notes are retained in [knowledge base section 8](../KNOWLEDGEBASE.md#8-policy-training-design-notes).
They are proposals, not an implemented training interface.

1. Verify a compatible MuJoCo Playground, MJX, JAX, and GPU environment separately
   from the working trajectory-optimization environment. There is no `rl` package
   extra or training command yet; select and test dependencies before adding one.
2. Adapt the tested single-reference reset/step environment to the chosen learning
   framework. Preserve timing, actuator limits, and diagnostic coverage; settle
   policy action scaling and rewards before treating them as paper comparisons.
   Add dataset sampling only after the single-reference learner works.
3. Train on one short trajectory and establish stable tracking before introducing
   dataset sampling or domain randomization.
4. Compare position-based tracking with torque-informed tracking using matched
   settings, multiple seeds, and held-out trajectories. Specify whether torque
   enters actor observations, critic inputs, rewards, or feedforward control;
   those are different experimental interventions.
5. Evaluate tracking error, falls, torque utilization, and recovery under controlled
   disturbances. Keep the existing feedforward-plus-PD replay as a separate
   controller baseline, not a substitute for the learned-policy comparison.

Walking, changing contacts, uneven terrain, manipulation, and hardware deployment
need additional modeling and validation. Flat-foot squat results do not establish
those capabilities.

## Numerical baseline

The terminal-wrench cost artifact is removed for new references, and simulator
settings and controller gains are explicit in `configs/g1_reconcile.yaml`.
The shallow squat was regenerated and validated; legacy references retain their
original objective version. [Equation diagnostics](TRACKING_EQUATIONS.md) expose
remaining solver residuals. Recheck numerical convergence when changing contact
models or tasks; the squat calibration is not a guarantee for other motions.
