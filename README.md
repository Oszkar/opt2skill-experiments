# opt2skill

A learning-focused Opt2Skill proof of concept for the Unitree G1. Specify a squat,
optimize it with Crocoddyl and Pinocchio, then validate and replay it in MuJoCo.

![Unitree G1 squatting in MuJoCo with colored reference keypoints](img/mujoco_screen.png)

*G1 squat visualization in MuJoCo, with reference keypoints shown as colored spheres.*

## Current state

- **Robot model reconciliation:** shared joint order, limits, frames, and ten checks
  comparing the Pinocchio and MuJoCo models.
- **Trajectory generation:** whole-body squats with both feet planted, feasibility
  filtering, torque and contact-wrench export, and randomized datasets.
- **Validation and visualization:** inverse-dynamics checks, feedforward-plus-PD
  replay, a PD-only comparison, and a viewer with reference keypoints.
- **Policy training:** not implemented. There is no RL environment, training CLI,
  trained policy, or hardware deployment in this package.

The local review on 2026-09-08 passed all 40 tests and all ten model checks. All
100 saved dataset trajectories passed inverse dynamics and stabilized replay;
maximum pelvis error was 1.37 cm. These are measured results for the local data
and configured controller, not guarantees for other motions or learned policies.

## Run on this machine

Open WSL from PowerShell:

```powershell
wsl -d Ubuntu-26.04
```

Then run inside WSL:

```bash
cd /mnt/e/Programming/opt2skill
source ~/venvs/o2s-to/bin/activate
python -m pytest -q
python -m o2s.models.reconcile
python -m o2s.reference.validate data/refs/dev/squat_dev.npz
python -m o2s.reference.view data/refs/dev/squat_dev.npz --mode kinematic --loop
```

Close the viewer, then try physics replay:

```bash
python -m o2s.reference.view data/refs/dev/squat_dev.npz --mode replay --loop
```

Kinematic mode poses the robot directly. Replay simulates reference torque plus
PD feedback. The viewer uses WSLg; the numerical checks do not require a window.

## Fresh environment

From the project root inside WSL2 Ubuntu, with `uv` and Git installed:

```bash
uv venv ~/venvs/o2s-to --python 3.12
source ~/venvs/o2s-to/bin/activate
uv pip install -e ".[trajopt,dev]"
bash scripts/fetch_assets.sh
python -m pytest -q
python -m o2s.trajopt.solve_one --depth 0.2 --out data/refs/dev/squat_dev.npz
python -m o2s.reference.validate data/refs/dev/squat_dev.npz
```

The supported workflow uses Linux under WSL. Assets are fetched at pinned commits
into `~/o2s_third_party` (override with `O2S_THIRD_PARTY`). Install both extras to
run the full suite: asset-dependent tests skip when assets are missing, but test
collection still imports the simulation packages. Generated `data/` is ignored
by Git and must be regenerated on a fresh checkout.

## Documentation

- [Hands-on guide](docs/TRAJECTORY_GUIDE.md): code map, commands, experiments,
  reference conventions, and what validation means.
- [Visual guide](docs/TRAJECTORY_DIAGRAMS.md): diagrams of optimization, export,
  simulation, and the proposed policy interface.
- [Next steps](docs/NEXT_STEPS.md): remaining reliability work and policy training.
- [Knowledge base](KNOWLEDGEBASE.md): retained research notes, historical
  experiments, environment gotchas, and detailed policy design notes.
- [Study booklet](<Opt2Skill Study Booklet.html>) and [saved paper](Opt2Skill_paper.htm):
  retained learning materials; open the HTML files in a browser.

Use the README and hands-on guide for current behavior. Historical learning
notes may describe earlier experiments or proposed features.
