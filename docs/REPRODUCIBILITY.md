# Reproducible trajectory experiments

## Install the tested environment

`requirements/trajectory.lock` pins the complete trajectory-optimization and test
dependency graph, including wheel/source hashes, for **Linux x86_64 / Python 3.12**.
It was initialized from the working Python **3.12.14** environment. Keep the
ABI-sensitive Crocoddyl 3.2.1 / Pinocchio 4.0.0 pair together.

From Ubuntu or WSL, in the repository root:

```bash
uv venv .venv --python 3.12.14
source .venv/bin/activate
uv pip sync --require-hashes requirements/trajectory.lock
uv pip install --no-deps -e .
bash scripts/fetch_assets.sh
python -m pytest -q
```

The final install registers this checkout as an editable package. The lock covers
runtime/test dependencies; setuptools' isolated editable-build environment is
separate. The dependency lock does not freeze drivers, the operating system, or
floating-point behavior across machines. Record those differences when comparing
experiments. Future GPU training dependencies should have a separate environment
and lock once that stack has been selected and verified.

To deliberately update the numerical dependency graph:

```bash
uv pip compile pyproject.toml --extra trajopt --extra dev \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --output-file requirements/trajectory.lock --upgrade
```

Review the lock diff, install it in a fresh environment, run the full suite and
model reconciliation, and revalidate references before using it for experiments.
Omit `--upgrade` to retain existing pins while incorporating dependency changes.

## Generate and freeze a dataset

Commit source changes first when creating a baseline, then choose a new output
directory. Existing datasets are never overwritten or silently migrated.

```bash
python -m o2s.trajopt.generate --n 100 --seed 0 --out data/refs/squat_v2
python -m o2s.provenance data/refs/squat_v2
```

Each new reference includes:

- Task parameters, actual interval count/duration, solver results and weights,
  objective version, reference conventions, and feasibility results.
- Actual simulation settings, gains, damping, and configuration hash.
- Full configuration; source revision, dirty status and working-file hashes;
  installed package versions; Python/platform; dependency-lock hash; and actual
  asset revisions, dirty status and working-file hashes alongside expected pins.

Git information is explicitly unknown when a checkout or Git is unavailable.
`matches_pin` is true only when an asset's actual revision matches its configured
pin and its working tree is clean. Provenance records deviations; it does not
prevent exploratory runs with modified assets. For a frozen baseline, require a
clean source checkout, all assets matching their pins, and installation from the
recorded lock. A dirty revision/hash identifies local differences but does not
archive those differences; preserve them separately or commit them.

`manifest.json` records the shared provenance, generation seed/ranges/weights,
requested count, attempt/iteration budgets, split seed/fractions/membership, and
SHA-256 hashes of every reference, `split.json`, `summary.json`, and
`rejected.jsonl`. Verification rejects changed/missing/extra reference files,
overlapping splits, and partial generation. A run that exhausts its attempt
budget gets a `partial` manifest and a nonzero exit code. An interrupted run may
have no manifest and is not a frozen dataset. The manifest is written only after
all output files are finished.

Run manifest verification before consuming a frozen dataset. Then load the
explicit names from its chosen split using `contract.load_dir(..., names=...)`;
the loader's default loads every reference, not just training data. The future
training adapter must respect the returned lengths when consuming padded arrays.

References with historical metadata remain readable. Missing provenance is not
invented from today's configuration. Regenerate legacy data to establish a new
baseline and keep the originals for historical comparisons. Generated datasets
remain ignored by Git; committing source does not publish local `.npz` files.

## Input conventions

Loading and saving reject inconsistent duplicate pelvis positions, quaternions,
and velocities. Base linear velocity in `qvel` is world-aligned; `pelvis_linvel`
must match it after rotation into pelvis coordinates. Duplicate state fields use
an absolute tolerance of 1e-6, with no relative tolerance. Declared metadata must
be a finite JSON object and agree with reference timing and supported torque and
wrench conventions. Older metadata may omit newer fields.

Squat parameters must be finite: depth and optional standing/hold durations may
be zero but not negative; descent/ascent durations must be positive. Zero depth
is a supported standing experiment. The only supported reference interval is
20 ms. Continuous sampled durations are retained in metadata while their total
is rounded to the nearest control interval, using Python's ties-to-even rounding;
the actual endpoint can differ by up to 10 ms. Very short profiles that round to
zero intervals are rejected before solving.
