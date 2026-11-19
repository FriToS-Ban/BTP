"""Physics sanity tests: each encodes a textbook fact the models rely on."""

import numpy as np
import pytest

from steelml.physics import cooling, kinetics, microstructure, thermo
from steelml.physics.properties import hb_from_hv, uniform_elongation_estimate

PLAIN = {"C": 0.40, "Mn": 0.75, "Si": 0.25}


def test_eutectoid_of_plain_carbon_steel_is_near_077():
    ce, te = thermo.eutectoid_point({"C": 0.4, "Mn": 0.0, "Si": 0.0})
    assert 0.70 < float(ce) < 0.82
    assert 700 < float(te) < 735


def test_alloying_lowers_eutectoid_carbon():
    ce_plain, _ = thermo.eutectoid_point({"C": 0.4, "Mn": 0.75})
    ce_cr, _ = thermo.eutectoid_point({"C": 0.4, "Mn": 0.75, "Cr": 1.0})
    assert ce_cr < ce_plain


def test_lever_rule_pearlite_fraction():
    lr = thermo.lever_rule({"C": 0.40, "Mn": 0.75, "Si": 0.25})
    assert 0.45 < float(lr["f_pearlite_eq"]) < 0.56          # textbook ~50 % for 0.4C
    assert float(lr["f_ferrite_eq"] + lr["f_pearlite_eq"] + lr["f_cementite_eq"]) == pytest.approx(1.0)


def test_hypereutectoid_has_proeutectoid_cementite():
    lr = thermo.lever_rule({"C": 0.97, "Mn": 0.4, "Si": 0.25})
    assert float(lr["f_cementite_eq"]) == pytest.approx((0.97 - float(lr["C_eutectoid"])) /
                                                        (6.67 - float(lr["C_eutectoid"])), rel=1e-6)
    assert str(thermo.eutectoid_class({"C": 0.97, "Mn": 0.4})) == "hypereutectoid"


def test_ae3_and_ms_fall_with_carbon():
    lo, hi = {"C": 0.2, "Mn": 0.7}, {"C": 0.6, "Mn": 0.7}
    assert thermo.ae3(hi) < thermo.ae3(lo)
    assert thermo.martensite_start(hi) < thermo.martensite_start(lo)


def _run(comp, rate, G=7.5):
    t, T = cooling.constant_rate_path(float(thermo.ae3(comp)) + 50, rate)
    return kinetics.simulate(comp, G, t, T)


@pytest.mark.parametrize("comp", [PLAIN, {"C": 0.40, "Mn": 0.7, "Ni": 1.8, "Cr": 0.8, "Mo": 0.25},
                                  {"C": 0.01, "Mn": 0.3}, {"C": 0.95, "Mn": 0.4}])
@pytest.mark.parametrize("rate", [0.005, 0.5, 5.0, 100.0])
def test_fractions_conserve_volume(comp, rate):
    d = _run(comp, rate)
    total = sum(d[k] for k in ("f_ferrite", "f_pearlite", "f_bainite", "f_martensite", "f_cementite_pro",
                               "f_austenite_ret"))
    assert total == pytest.approx(1.0, abs=1e-6)
    assert all(d[k] >= -1e-9 for k in ("f_ferrite", "f_pearlite", "f_bainite", "f_martensite"))


def test_slow_cooling_approaches_lever_rule():
    d = _run(PLAIN, 0.005)
    lr = thermo.lever_rule(PLAIN)
    assert abs(d["f_pearlite"] - float(lr["f_pearlite_eq"])) < 0.08


def test_faster_cooling_suppresses_ferrite_pearlite_in_a_hardenable_steel():
    comp = {"C": 0.40, "Mn": 0.7, "Si": 0.25, "Ni": 1.8, "Cr": 0.8, "Mo": 0.25}
    fp = [(_run(comp, r)["f_ferrite"] + _run(comp, r)["f_pearlite"]) for r in (0.003, 0.03, 0.3, 30.0)]
    assert fp[0] > 0.95 and fp[-1] < 0.01
    assert all(a >= b - 1e-9 for a, b in zip(fp, fp[1:]))
    assert _run(comp, 30.0)["f_martensite"] > 0.9


def test_plain_carbon_steel_stays_ferrite_pearlite_on_air_cooling():
    d = _run(PLAIN, 1.0)
    assert d["f_bainite"] + d["f_martensite"] < 0.02


def test_air_cooling_rate_falls_with_section_size():
    rates = [cooling.rate_at(*cooling.air_cooling_path(900, D)) for D in (10, 25, 50, 100)]
    assert all(a > b for a, b in zip(rates, rates[1:]))
    assert 0.5 < rates[1] < 2.0          # 25 mm bar: ~1 K/s, typical normalising


def test_furnace_path_hits_requested_rate():
    t, T = cooling.furnace_cooling_path(880, 0.01)
    assert cooling.rate_at(t, T) == pytest.approx(0.01, rel=0.02)


def test_pearlite_spacing_falls_with_undercooling():
    s = microstructure.pearlite_spacing_um(np.array([700.0, 650.0, 600.0]), 720.0)
    assert s[0] > s[1] > s[2] > 0


def test_astm_grain_size_round_trip():
    assert float(microstructure.um_to_astm(microstructure.astm_to_um(7.0))) == pytest.approx(7.0)


def test_uniform_elongation_estimate():
    n, eu, lo, hi = uniform_elongation_estimate(36.0, 32.0, 40.0)
    assert lo < eu < hi < 36.0
    assert float(n) == pytest.approx(np.log1p(float(eu) / 100.0))


def test_hardness_conversion_is_monotone():
    assert hb_from_hv(200) < hb_from_hv(300)


def test_normalising_is_done_hotter_than_annealing():
    """Default austenitising temperatures follow the practice of the training
    data: normalising well above annealing (1040: 899 C in the handbook)."""
    from steelml.physics import thermo
    for comp in ({"C": 0.45, "Mn": 0.75, "Si": 0.25}, {"C": 0.95, "Mn": 0.40, "Si": 0.25}):
        fc = float(thermo.default_austenitising_T(comp, "FC"))
        ac = float(thermo.default_austenitising_T(comp, "AC"))
        assert ac > fc + 50.0
    t_1040 = float(thermo.default_austenitising_T({"C": 0.405, "Mn": 0.75, "Si": 0.225}, "AC"))
    assert abs(t_1040 - 899.0) < 25.0
