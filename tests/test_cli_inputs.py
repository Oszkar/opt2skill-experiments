import sys

import pytest

from o2s.models import g1
from o2s.trajopt import generate, solve_one
from o2s.trajopt.profile import SquatParams
from o2s.trajopt.squat_problem import solve_squat


@pytest.mark.parametrize("module,args", [
    (solve_one, ["--depth", "nan"]), (solve_one, ["--t-down", "-1"]),
    (solve_one, ["--t-up", "0"]), (solve_one, ["--max-iter", "0"]),
    (generate, ["--n", "-1"]), (generate, ["--max-attempts", "-1"]),
    (generate, ["--seed", "-1"]),
])
def test_cli_rejects_bad_inputs_before_loading_or_writing(module, args, tmp_path, monkeypatch):
    def unexpected_load():
        pytest.fail("invalid inputs must be rejected before loading models")
    monkeypatch.setattr(g1, "load_config", unexpected_load)
    out = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["command", *args, "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert not out.exists()


@pytest.mark.parametrize("max_iter", [0, -1, 1.5, True])
def test_solver_rejects_invalid_iteration_budget_before_using_models(max_iter):
    with pytest.raises(ValueError, match="max_iter"):
        solve_squat(None, None, {}, SquatParams(depth=.1), max_iter=max_iter)
