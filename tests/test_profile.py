import numpy as np
import pytest

from o2s.trajopt.profile import SquatParams, com_reference, min_jerk, time_grid


def test_min_jerk_endpoints_and_monotone():
    s = np.linspace(0, 1, 101)
    y = min_jerk(s)
    assert y[0] == 0.0 and abs(y[-1] - 1.0) < 1e-12
    assert np.all(np.diff(y) >= -1e-12)
    # zero slope at both ends
    assert abs((y[1] - y[0]) / 0.01) < 1e-3 and abs((y[-1] - y[-2]) / 0.01) < 1e-3


def test_num_nodes_and_time_grid():
    p = SquatParams(depth=0.2, t_stand0=0.5, t_down=1.0, t_hold=0.4, t_up=1.0, t_stand1=0.5, dt=0.02)
    assert abs(p.total_time - 3.4) < 1e-12
    assert p.num_nodes() == 170
    t = time_grid(p)
    assert t.shape == (171,) and t[0] == 0.0 and abs(t[-1] - 3.4) < 1e-9


def test_com_reference_shape_depth_and_return():
    p = SquatParams(depth=0.2, com_shift_x=0.03)
    com0 = np.array([0.01, 0.0, 0.69])
    ref = com_reference(p, com0)
    N = p.num_nodes()
    assert ref.shape == (N + 1, 3)
    np.testing.assert_allclose(ref[0], com0)
    np.testing.assert_allclose(ref[-1], com0, atol=1e-9)
    assert abs(ref[:, 2].min() - (com0[2] - 0.2)) < 1e-9
    assert abs(ref[:, 0].max() - (com0[0] + 0.03)) < 1e-9
    np.testing.assert_allclose(ref[:, 1], com0[1])
    # hold plateau: at least t_hold/dt consecutive nodes at the bottom
    at_bottom = np.isclose(ref[:, 2], com0[2] - 0.2, atol=1e-9).sum()
    assert at_bottom >= int(round(p.t_hold / p.dt))
    # the profile never has a step larger than a physically plausible per-node change
    assert np.max(np.abs(np.diff(ref[:, 2]))) < 0.02


@pytest.mark.parametrize("key,value", [("depth", -.1), ("depth", float("nan")),
                                      ("com_shift_x", float("inf")), ("depth", True),
                                      ("t_down", 0), ("t_down", -1), ("t_up", 0),
                                      ("t_hold", -.1), ("t_stand0", -.1), ("t_stand1", -.1),
                                      ("dt", 0), ("dt", .01)])
def test_rejects_invalid_parameters(key, value):
    with pytest.raises(ValueError, match=key):
        SquatParams(**{"depth": .2, key: value})


def test_standing_and_zero_optional_phases_are_supported():
    params = SquatParams(depth=0, t_stand0=0, t_hold=0, t_stand1=0)
    np.testing.assert_allclose(com_reference(params, [0, 0, .7]),
                               np.tile([0, 0, .7], (params.num_nodes() + 1, 1)))


def test_duration_rounding_and_too_short_profile():
    params = SquatParams(depth=.1, t_down=.813)
    assert abs(time_grid(params)[-1] - params.total_time) <= params.dt / 2
    with pytest.raises(ValueError, match="duration"):
        SquatParams(depth=.1, t_down=.001, t_up=.001, t_hold=0, t_stand0=0, t_stand1=0)
