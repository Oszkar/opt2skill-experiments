import json

import numpy as np
import pytest

from o2s.reference import contract


def make_ref(N: int) -> dict:
    rng = np.random.default_rng(N)
    ref = {}
    for key, shape in contract.ARRAY_SPEC.items():
        rows = N + 1 if shape[0] == "N+1" else N
        ref[key] = rng.normal(size=(rows, *shape[1:])) if len(shape) > 1 else np.arange(rows) * contract.DT
    ref["qpos"][:, 3:7] /= np.linalg.norm(ref["qpos"][:, 3:7], axis=1, keepdims=True)
    return ref


def test_spec_has_expected_keys_and_shapes():
    assert contract.ARRAY_SPEC["qpos"] == ("N+1", 36)
    assert contract.ARRAY_SPEC["qvel"] == ("N+1", 35)
    assert contract.ARRAY_SPEC["tau"] == ("N", 29)
    assert contract.ARRAY_SPEC["foot_wrench_left"] == ("N", 6)
    assert contract.ARRAY_SPEC["t"] == ("N+1",)
    assert set(contract.STATE_KEYS) | set(contract.CONTROL_KEYS) == set(contract.ARRAY_SPEC)


def test_validate_accepts_good_and_rejects_bad():
    ref = make_ref(50)
    assert contract.validate(ref) == 50
    # N is derived from tau, so corrupt a key that does not define N
    bad = dict(ref)
    bad["qvel"] = ref["qvel"][:-1]
    with pytest.raises(ValueError, match="qvel"):
        contract.validate(bad)
    bad = dict(ref)
    del bad["com"]
    with pytest.raises(ValueError, match="com"):
        contract.validate(bad)
    bad = dict(ref)
    bad["qvel"] = ref["qvel"].copy()
    bad["qvel"][3, 0] = np.nan
    with pytest.raises(ValueError, match="qvel"):
        contract.validate(bad)


def test_save_load_roundtrip(tmp_path):
    ref = make_ref(40)
    meta = {"depth": 0.2, "solver_iters": 12, "filter": {"ok": True}}
    p = tmp_path / "a.npz"
    contract.save(p, ref, meta)
    ref2, meta2 = contract.load(p)
    assert meta2 == meta
    for k in ref:
        np.testing.assert_array_equal(ref2[k], ref[k])


def test_load_dir_pads_and_records_lengths(tmp_path):
    for i, N in enumerate([30, 45, 20]):
        contract.save(tmp_path / f"squat_{i:04d}.npz", make_ref(N), {"i": i})
    rs = contract.load_dir(tmp_path)
    assert rs.names == ["squat_0000", "squat_0001", "squat_0002"]
    np.testing.assert_array_equal(rs.lengths, [30, 45, 20])
    assert rs.arrays["qpos"].shape == (3, 46, 36)
    assert rs.arrays["tau"].shape == (3, 45, 29)
    # padding repeats the last valid row
    np.testing.assert_array_equal(rs.arrays["qpos"][2, 20], rs.arrays["qpos"][2, 45])
    np.testing.assert_array_equal(rs.arrays["tau"][2, 19], rs.arrays["tau"][2, 44])
    sub = contract.load_dir(tmp_path, names=["squat_0001"])
    assert sub.arrays["qpos"].shape == (1, 46, 36)
    # padding extrapolates t at a constant DT step instead of repeating the last value
    np.testing.assert_allclose(np.diff(rs.arrays["t"][2, 19:]), contract.DT)


def test_load_dir_raises_on_missing_names(tmp_path):
    contract.save(tmp_path / "squat_0000.npz", make_ref(10), {})
    with pytest.raises(FileNotFoundError, match="squat_0099"):
        contract.load_dir(tmp_path, names=["squat_0000", "squat_0099"])


def test_load_raises_value_error_on_missing_key(tmp_path):
    ref = make_ref(10)
    p = tmp_path / "a.npz"
    contract.save(p, ref, {"depth": 0.2})
    # rewrite the file without one of the required arrays
    with np.load(p) as z:
        data = {k: z[k] for k in z.files if k != "com"}
    np.savez_compressed(p, **data)
    with pytest.raises(ValueError, match="com"):
        contract.load(p)
