import json
import subprocess

import pytest

from o2s import provenance
from o2s.reference import contract
from tests.test_contract import make_ref


def init_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "model.txt").write_text("original\n")
    subprocess.run(["git", "-C", str(path), "add", "model.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"], check=True)


def test_git_state_distinguishes_revision_working_edits_and_unknown_checkout(tmp_path):
    root = tmp_path / "repo"
    init_repo(root)
    clean = provenance.git_state(root)
    assert clean["dirty"] is False and len(clean["commit"]) == 40
    (root / "model.txt").write_text("changed\n")
    (root / "new.txt").write_text("untracked\n")
    dirty = provenance.git_state(root)
    assert dirty["commit"] == clean["commit"] and dirty["dirty"] is True
    assert dirty["content_sha256"] != clean["content_sha256"]
    assert "new.txt" in dirty["files_sha256"]
    unknown = provenance.git_state(tmp_path)
    assert unknown["commit"] is None and unknown["dirty"] is None


def test_capture_records_actual_asset_state_config_and_lock(tmp_path, monkeypatch):
    root = tmp_path / "project"
    init_repo(root)
    lock = root / provenance.LOCK_PATH
    lock.parent.mkdir()
    lock.write_text("numpy==2.5.2\n")
    asset = tmp_path / "robot"
    init_repo(asset)
    actual = provenance.git_state(asset)["commit"]
    cfg = {"assets": {"third_party_dir_env": "O2S_TEST_ASSETS",
                      "robot": {"repo": "example/robot", "commit": "0" * 40}},
           "simulation": {"timestep": .002}}
    monkeypatch.setenv("O2S_TEST_ASSETS", str(tmp_path))
    monkeypatch.setattr(provenance, "ROOT", root)
    snap = provenance.capture(cfg)
    assert snap["assets"]["robot"]["commit"] == actual
    assert snap["assets"]["robot"]["matches_pin"] is False
    assert snap["dependency_lock"]["sha256"] == provenance.file_sha256(lock)
    assert "numpy" in snap["packages"] and snap["python"]
    cfg["simulation"]["timestep"] = .001
    assert snap["config"]["simulation"]["timestep"] == .002
    cfg["assets"]["robot"]["commit"] = actual
    assert provenance.capture(cfg)["assets"]["robot"]["matches_pin"] is True
    (asset / "model.txt").write_text("dirty")
    assert provenance.capture(cfg)["assets"]["robot"]["matches_pin"] is False


@pytest.fixture
def dataset(tmp_path):
    contract.save(tmp_path / "a.npz", make_ref(2), {})
    split = {"train": ["a"], "val": [], "test": []}
    (tmp_path / "split.json").write_text(json.dumps(split))
    (tmp_path / "summary.json").write_text('{"accepted": 1}')
    (tmp_path / "rejected.jsonl").write_text("")
    provenance.write_manifest(tmp_path, {}, {"requested": 1}, split, complete=True)
    return tmp_path


@pytest.mark.parametrize("change", ["reference", "missing", "extra", "split", "partial", "overlap"])
def test_manifest_detects_corruption_and_incomplete_or_overlapping_splits(dataset, change):
    assert provenance.verify_dataset(dataset)["status"] == "complete"
    path = dataset / "a.npz"
    if change == "reference":
        with path.open("ab") as stream:
            stream.write(b"corruption")
    elif change == "missing":
        path.unlink()
    elif change == "extra":
        (dataset / "unexpected.npz").write_bytes(path.read_bytes())
    elif change == "split":
        (dataset / "split.json").write_text('{"train": [], "val": ["a"], "test": []}')
    else:
        manifest_path = dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if change == "partial":
            manifest["status"] = "partial"
        else:
            manifest["split"]["val"] = ["a"]
            split_path = dataset / "split.json"
            split_path.write_text(json.dumps(manifest["split"]))
            manifest["files"]["split.json"] = provenance.file_sha256(split_path)
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        provenance.verify_dataset(dataset)


def test_manifest_rejects_paths_outside_dataset(dataset):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filename"):
        provenance.verify_dataset(dataset)
