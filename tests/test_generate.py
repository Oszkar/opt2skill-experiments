import json
import sys

import numpy as np
import pytest

from o2s.reference import contract
from o2s.trajopt import generate
from tests.conftest import requires_assets


def test_sample_params_within_ranges():
    rng = np.random.default_rng(0)
    r = generate.SquatRanges()
    for _ in range(50):
        p = generate.sample_params(rng, r)
        assert r.depth[0] <= p.depth <= r.depth[1]
        assert r.t_down[0] <= p.t_down <= r.t_down[1]
        assert r.t_hold[0] <= p.t_hold <= r.t_hold[1]
        assert r.t_up[0] <= p.t_up <= r.t_up[1]
        assert r.com_shift_x[0] <= p.com_shift_x <= r.com_shift_x[1]
        assert p.t_stand0 == 0.5 and p.t_stand1 == 0.5 and p.dt == 0.02


def test_make_split_is_deterministic_and_partitions():
    names = [f"squat_{i:04d}" for i in range(97)]
    a = generate.make_split(names, np.random.default_rng(3))
    b = generate.make_split(names, np.random.default_rng(3))
    assert a == b
    assert set(a) == {"train", "val", "test"}
    allx = a["train"] + a["val"] + a["test"]
    assert sorted(allx) == names and len(allx) == 97
    assert len(a["train"]) == 68 and len(a["val"]) == 10 and len(a["test"]) == 19


@requires_assets
def test_main_with_zero_n_and_zero_attempts_writes_empty_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["generate.py", "--n", "0", "--max-attempts", "0", "--out", str(tmp_path)]
    )
    rc = generate.main()
    assert rc == 0

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted"] == 0
    assert summary["solve_time_mean"] == 0.0
    assert summary["solve_time_max"] == 0.0
    assert summary["depth_hist"] == [0, 0, 0, 0]

    assert (tmp_path / "rejected.jsonl").read_text(encoding="utf-8") == ""

    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    assert split == {"train": [], "val": [], "test": []}


@requires_assets
def test_main_accepts_one_trajectory_and_records_full_replay_metrics(tmp_path, monkeypatch):
    # seed 1 accepts on the first attempt (checked against this codebase); max-attempts leaves a
    # small margin without turning this into a slow, flaky, multi-solve test.
    monkeypatch.setattr(
        sys, "argv",
        ["generate.py", "--n", "1", "--max-attempts", "3", "--out", str(tmp_path), "--seed", "1"],
    )
    rc = generate.main()
    assert rc == 0

    npz_files = sorted(tmp_path.glob("squat_*.npz"))
    assert len(npz_files) == 1

    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    assert split["train"] == [npz_files[0].stem]
    assert split["val"] == [] and split["test"] == []

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted"] == 1
    assert "replay_ff_pass_rate" in summary

    _, meta = contract.load(npz_files[0])
    assert meta["replay_ff"]["ok"] is True
    # the full check_replay metrics dict is stored, not just the three original fields
    for key in ("max_pelvis_error", "completed", "rms_joint_error", "peak_effort_ratio", "pd_over_ref_legs"):
        assert key in meta["replay_ff"]


@pytest.mark.parametrize("entry", ["squat_0000.npz", "summary.json", ".hidden", "subdirectory"])
def test_rejects_nonempty_output_before_loading_models(tmp_path, monkeypatch, entry):
    marker = tmp_path / entry
    if entry == "subdirectory":
        marker.mkdir()
    else:
        marker.write_bytes(b"preserve me")
    monkeypatch.setattr(sys, "argv", ["generate", "--out", str(tmp_path)])
    def unexpected_load():
        pytest.fail("must reject output before model loading")
    monkeypatch.setattr(generate.g1, "load_config", unexpected_load)
    with pytest.raises(SystemExit) as exc:
        generate.main()
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == [marker]
    if marker.is_file():
        assert marker.read_bytes() == b"preserve me"


@requires_assets
def test_rejects_replay_effort_violation_before_saving(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate", "--n", "1", "--max-attempts", "1",
                                      "--seed", "1", "--out", str(tmp_path / "new")])
    original = generate.validate.check_replay
    def over_limit(*args, **kwargs):
        report = original(*args, **kwargs)
        report.metrics["effort_ok"] = 0.0
        report.ok = False
        return report
    monkeypatch.setattr(generate.validate, "check_replay", over_limit)
    assert generate.main() == 1
    assert not list((tmp_path / "new").glob("*.npz"))
    record = json.loads((tmp_path / "new" / "rejected.jsonl").read_text())
    assert record["stage"] == "replay_effort"



def test_rejects_output_file_without_modifying_it(tmp_path, monkeypatch):
    path = tmp_path / "not-a-directory"
    path.write_bytes(b"keep")
    monkeypatch.setattr(sys, "argv", ["generate", "--out", str(path)])
    with pytest.raises(SystemExit) as exc:
        generate.main()
    assert exc.value.code == 2 and path.read_bytes() == b"keep"
