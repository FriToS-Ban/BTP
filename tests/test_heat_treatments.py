"""Tests of the v3 heat-treatment physics: tempering, stress relief,
precipitation, toughness flags, composition flags and the hardening default."""

import numpy as np
import pandas as pd
import pytest

from steelml.models.flags import composition_flags
from steelml.physics import precipitation, stress_relief, tempering, thermo, toughness


def test_hollomon_jaffe_values_and_equivalence():
    assert tempering.hollomon_jaffe(540.0, 1.0) == pytest.approx(813.15 * 20.0)
    P = tempering.hollomon_jaffe(600.0, 1.0)
    t_eq = tempering.equivalent_time_h(P, 550.0)
    assert tempering.hollomon_jaffe(550.0, t_eq) == pytest.approx(P)
    assert t_eq > 1.0          # a lower temperature needs longer for the same effect


def test_tempered_hardness_falls_with_tempering_and_alloy_delays_it():
    comp_c = {"C": 0.45, "Mn": 0.75, "Si": 0.25, "Cr": 0.1, "Mo": 0.0, "V": 0.0}
    comp_mo = dict(comp_c, Cr=1.0, Mo=0.25)
    P = tempering.hollomon_jaffe(np.array([200.0, 400.0, 550.0, 650.0]))
    hv_c = tempering.tempered_martensite_hv(650.0, P, comp_c)
    hv_mo = tempering.tempered_martensite_hv(650.0, P, comp_mo)
    assert np.all(np.diff(hv_c) < 0) and np.all(np.diff(hv_mo) < 0)
    assert hv_mo[-1] > hv_c[-1]            # Cr-Mo resists softening at 650 C
    assert hv_c[0] <= 650.0 + 1e-6         # never harder than as-quenched


def test_bainite_softens_only_below_tempered_martensite():
    comp = {"C": 0.4, "Mn": 0.8, "Si": 0.25, "Cr": 1.0, "Mo": 0.2, "V": 0.0}
    P = tempering.hollomon_jaffe(np.array([200.0, 650.0]))
    hv = tempering.tempered_hv(np.array([620.0, 620.0]), np.array([380.0, 380.0]), np.array([200.0, 200.0]),
                               np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0]), P, comp)
    assert hv[0] == pytest.approx(380.0)    # low temper: bainite unchanged
    assert hv[1] < 380.0                    # high temper: follows tempered martensite


def test_combine_is_monotone_and_lower_soak_adds_little():
    P1 = tempering.hollomon_jaffe(600.0, 1.0)
    low = stress_relief.tempering_after_relief(P1, 500.0, 1.0)
    high = stress_relief.tempering_after_relief(P1, 650.0, 1.0)
    assert P1 <= low < P1 + 150.0
    assert high > P1 + 500.0


def test_stress_relief_bound_and_subcritical_check():
    assert stress_relief.ky(400.0) == pytest.approx(1.0)
    assert stress_relief.ky(600.0) == pytest.approx(0.47)
    assert stress_relief.residual_stress_bound(400.0, 600.0) == pytest.approx(188.0)
    comp = {"C": 0.45, "Mn": 0.75, "Si": 0.25, "Ni": 0.1, "Cr": 0.1, "W": 0.0}
    assert bool(stress_relief.is_subcritical(comp, 600.0))
    assert not bool(stress_relief.is_subcritical(comp, 720.0))


def test_precipitation_needs_microalloy_and_air_cooling_gives_more():
    base = {"C": 0.18, "Mn": 1.3, "Si": 0.3, "N": 0.010, "V": 0.0, "Nb": 0.0}
    assert float(precipitation.strengthening(base, 920.0, "AC")) == 0.0
    v = dict(base, V=0.10)
    ac = float(precipitation.strengthening(v, 920.0, "AC"))
    fc = float(precipitation.strengthening(v, 920.0, "FC"))
    assert 40.0 < ac < 300.0 and fc < ac
    # niobium is less soluble: little dissolves at a low austenitising temperature
    nb = dict(base, Nb=0.04)
    assert precipitation.dissolved_at(nb, 850.0)[1] < precipitation.dissolved_at(nb, 1250.0)[1]


def test_toughness_flags():
    comp = {"Mn": 0.8, "Si": 0.25, "P": 0.02, "Ni": 1.8, "Cr": 0.8}
    f = toughness.embrittlement_flags(comp, np.array([300.0, 500.0, 650.0]), np.array([0.9, 0.9, 0.9]))
    assert list(f["TME_risk"]) == [True, False, False]
    assert list(f["TE_risk"]) == [False, True, False]


def test_composition_flags():
    df = pd.DataFrame([{"C": 0.08, "Mn": 1.5, "Nb": 0.03, "V": 0.0, "Ti": 0.0, "B": 0.0},
                       {"C": 0.40, "Mn": 0.8, "Nb": 0.0, "V": 0.0, "Ti": 0.0, "B": 0.002},
                       {"C": 0.45, "Mn": 0.75, "Nb": 0.0, "V": 0.0, "Ti": 0.0, "B": 0.0}])
    f = composition_flags(df)
    assert list(f["tmcp_like"]) == [True, False, False]
    assert list(f["boron"]) == [False, True, False]
    assert f["composition_notes"][2] == ""


def test_hardening_temperature_default():
    comp = pd.DataFrame([{"C": 0.40, "Mn": 0.85, "Si": 0.25, "Cr": 1.0, "Ni": 0.0, "Mo": 0.2}])
    harden = float(np.ravel(thermo.default_austenitising_T(comp, np.array(["Q"])))[0])
    anneal = float(np.ravel(thermo.default_austenitising_T(comp, np.array(["FC"])))[0])
    assert 800.0 < harden < 900.0 and harden > anneal
