import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml.physics import cooling, kinetics, quench  # noqa: E402


def _series_theta_centre(Fo, Bi, n_terms=40):
    z0 = np.concatenate([[0.0], jn_zeros(1, n_terms - 1)])
    zj0 = jn_zeros(0, n_terms)
    total = np.zeros_like(Fo)
    for n in range(n_terms):
        a, b = z0[n] + 1e-9, zj0[n] - 1e-9
        z = brentq(lambda x: x * j1(x) - Bi * j0(x), a, b)
        C = 2.0 * j1(z) / (z * (j0(z) ** 2 + j1(z) ** 2))
        total += C * np.exp(-z * z * Fo)
    return total


@pytest.mark.parametrize("Bi", [0.5, 5.0])
def test_constant_property_series_solution(Bi):
    k, cp, D, T0, Tq = 30.0, 500.0, 0.05, 1000.0, 20.0
    R = D / 2
    h = Bi * k / R
    grid = cooling.T_GRID.astype(float)
    times = quench._solve(T0, h, Tq, 0, D, 41, grid, 30.0, 1e6, k, cp, cooling.T_AMB, cooling.EPS)
    t = times[0]
    sel = (grid >= 60.0) & (grid <= 950.0)
    theta_num = (grid[sel] - Tq) / (T0 - Tq)
    Fo = k / (cooling.RHO * cp) * t[sel] / R ** 2
    theta_an = _series_theta_centre(Fo, Bi)
    assert np.max(np.abs(theta_an - theta_num)) < 0.02


def _t_to(path, T_target):
    t, T = path
    return float(np.interp(T_target, T[::-1], t[::-1]))


def test_centre_slower_than_surface_and_size_effect():
    c, m, s = quench.quench_paths(850.0, 25.0, medium="oil", positions=(0.0, 0.5, 1.0))
    assert _t_to(c, 500.0) > _t_to(m, 500.0) > _t_to(s, 500.0)
    small = quench.quench_paths(850.0, 13.0, medium="oil", positions=(0.0,))[0]
    big = quench.quench_paths(850.0, 50.0, medium="oil", positions=(0.0,))[0]
    assert _t_to(small, 500.0) < _t_to(c, 500.0) < _t_to(big, 500.0)
    assert quench.rate_at_700([small])[0] > quench.rate_at_700([big])[0]


def test_medium_ordering():
    kw = dict(positions=(0.0,))
    water = quench.quench_paths(850.0, 25.0, H=1.0, **kw)[0]
    oil = quench.quench_paths(850.0, 25.0, H=0.35, T_q=25.0, **kw)[0]
    air = quench.quench_paths(850.0, 25.0, medium="air", **kw)[0]
    assert _t_to(water, 500.0) < _t_to(oil, 500.0) < _t_to(air, 500.0)
    r = quench.rate_at_700([water, oil, air])
    assert r[0] > r[1] > r[2]


def test_severity_and_conversion():
    s = quench.severity("oil", "mild")
    assert s["H_nom"] == pytest.approx(0.325)
    assert quench.H_from_h(quench.h_from_H(0.7)) == pytest.approx(0.7)
    assert quench.severity("water", "violent")["H_nom"] == 4.0
    with pytest.raises(KeyError):
        quench.severity("oil", "nonsense")


def test_thin_bar_in_air_matches_lumped_model():
    T0 = 850.0
    c = quench.quench_paths(T0, 10.0, medium="air", positions=(0.0,))[0]
    ref = cooling.air_cooling_path(T0, 10.0)
    r, r_ref = quench.rate_at_700([c])[0], cooling.rate_at(*ref, 700.0)
    assert abs(r / r_ref - 1.0) < 0.05


def test_paths_run_through_kinetics():
    """Paths are valid kinetic input; martensite falls with bar size and rises with severity.

    Note: with a constant Grossmann h the surface does NOT always give more
    martensite than the centre. The centre lags, then cools faster through the
    bainite/pearlite window (transient-conduction effect: the surface log-rate
    sits below the fundamental mode, the centre above it), so the surface spends
    longer in the nose. That ordering is therefore not asserted.
    """
    comp = pd.DataFrame([dict(C=0.4, Mn=0.8, Cr=1.0, Mo=0.2, Si=0.25)] * 4)
    paths = (quench.quench_paths(850.0, 50.0, medium="oil", positions=(0.0, 1.0))
             + quench.quench_paths(850.0, 25.0, medium="oil", positions=(0.0,))
             + quench.quench_paths(850.0, 50.0, medium="water", positions=(0.0,)))
    assert len(paths[0][0]) == len(cooling.T_GRID)
    kin = kinetics.simulate_batch(comp, np.full(4, 7.0), paths)
    m = kin.f_martensite.to_numpy()
    assert m[2] > m[0]          # 25 mm centre beats 50 mm centre (oil)
    assert m[3] > m[0]          # water beats oil at 50 mm centre
    assert np.all(np.isfinite(m)) and np.all(m >= 0)
