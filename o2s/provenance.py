"""Capture experiment inputs and verify a generated dataset's file manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path("requirements/trajectory.lock")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_state(root: Path) -> dict:
    """Unknown Git state is explicit; never infer a revision from the config."""
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None, "error": "Git checkout not available"}
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, timeout=30).stdout
    try:
        revision = git("rev-parse", "HEAD").decode().strip()
        status = git("status", "--porcelain", "--untracked-files=all")
        files = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        # Hash working files as well as the revision so local asset/source edits are identifiable.
        hashes = {name: file_sha256(root / name) for name in
                  sorted(set(files.decode().strip("\0").split("\0")))
                  if name and (root / name).is_file()}
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        return {"commit": revision, "dirty": bool(status.strip()), "files_sha256": hashes,
                "content_sha256": digest}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "dirty": None, "error": type(exc).__name__}


def capture(cfg: dict) -> dict:
    assets_root = Path(os.environ.get(cfg["assets"]["third_party_dir_env"],
                                    Path.home() / "o2s_third_party")).expanduser()
    assets = {}
    for key, spec in cfg["assets"].items():
        if not isinstance(spec, dict):
            continue
        state = git_state(assets_root / spec["repo"].rsplit("/", 1)[-1])
        assets[key] = {**state, "repo": spec["repo"], "expected_commit": spec["commit"],
                       "matches_pin": state["commit"] == spec["commit"] and state["dirty"] is False}
    config = json.loads(json.dumps(cfg, allow_nan=False))
    lock = ROOT / LOCK_PATH
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": git_state(ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted((dist.metadata["Name"].lower().replace("_", "-"), dist.version)
                                for dist in metadata.distributions() if dist.metadata["Name"])),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "dependency_lock": {"path": LOCK_PATH.as_posix(),
                            "sha256": file_sha256(lock) if lock.is_file() else None},
        "assets": assets,
    }


def write_manifest(directory: Path, provenance: dict, generation: dict,
                   split: dict, *, complete: bool) -> None:
    names = [f"{name}.npz" for members in split.values() for name in members]
    names += ["split.json", "summary.json", "rejected.jsonl"]
    manifest = {"schema_version": 1, "status": "complete" if complete else "partial",
                "provenance": provenance, "generation": generation, "split": split,
                "files": {name: file_sha256(directory / name) for name in sorted(names)}}
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_dataset(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("dataset manifest is unsupported or generation is incomplete")
    files = manifest["files"]
    for name, expected in files.items():
        if not name or name in (".", "..") or "/" in name or "\\" in name or ":" in name:
            raise ValueError(f"invalid manifest filename: {name}")
        path = directory / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"dataset file missing or modified: {name}")
    split = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    if set(split) != {"train", "val", "test"} or split != manifest["split"]:
        raise ValueError("dataset split differs from manifest")
    names = [name for members in split.values() for name in members]
    expected_files = {f"{name}.npz" for name in names}
    if (len(set(names)) != len(names)
            or expected_files != {p.name for p in directory.glob("*.npz")}
            or set(files) != expected_files | {"split.json", "summary.json", "rejected.jsonl"}):
        raise ValueError("dataset references do not form the recorded disjoint split")
    if len(names) != manifest["generation"]["requested"]:
        raise ValueError("dataset reference count differs from the requested count")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        manifest = verify_dataset(args.directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"[ok] verified {manifest['generation']['requested']} references and train/val/test split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
