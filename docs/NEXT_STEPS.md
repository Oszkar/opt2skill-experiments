# Next steps

Trajectory generation, model reconciliation, reference export, numerical
validation, and visualization are implemented. Policy training and hardware
execution are not. See the [hands-on guide](TRAJECTORY_GUIDE.md) to use what exists.

## Make reference data a reliable training input

1. Reject nonempty dataset output directories by default, or implement an explicit
   resume/overwrite policy. Verify file membership agrees with the split and summary.
2. Strengthen the reference contract: positive interval count, monotonic uniform
   20 ms timing, unit quaternions, metadata consistency, and padding masks.
   Validate squat parameters before invoking the solver.
3. Check exported torque after damping compensation and total applied torque at
   every replay substep. Decide whether torque saturation and replay failures should
   gate training data acceptance; record the chosen acceptance policy with the data.
4. Capture a reproducible dependency lock and dataset provenance, including code,
   config, solver weights, and library versions. Asset commits are already pinned.

## Build a tracking policy

The detailed design notes are retained in [knowledge base section 8](../KNOWLEDGEBASE.md#8-policy-training-design-notes).
They are proposals, not an implemented training interface.

1. Verify a compatible MuJoCo Playground, MJX, JAX, and GPU environment separately
   from the working trajectory-optimization environment. There is no `rl` package
   extra or training command yet; select and test dependencies before adding one.
2. Implement a tracking environment that samples references, respects valid lengths,
   and handles reset, terminal state, observations, action scaling, and rewards.
   Start with 50 Hz control against the existing 20 ms references.
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
