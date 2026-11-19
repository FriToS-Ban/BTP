"""Tests for the hardenability tools, the Jominy model, the AISI/SAE decision
list and the microstructure fixes made after the metallurgical review. Each
test encodes a textbook fact."""

import numpy as np
import pandas as pd
import pytest

from steelml import config
from steelml.models.rules import rule_category
from steelml.physics import jominy, kinetics, microstructure
from steelml.physics.hardenability import ideal_critical_diameter


def _comp(**kw):
    d = {el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS}
    d.update(kw)
    return d


STEELS = {
    "1020": _comp(C=0.20, Mn=0.45, Si=0.22),
    "1045": _comp(C=0.46, Mn=0.75, Si=0.22),
    "1095": _comp(C=0.96, Mn=0.40, Si=0.22),
    "1137": _comp(C=0.36, Mn=1.50, S=0.10),
    "1340": _comp(C=0.40, Mn=1.75, Si=0.25),
    "4140": _comp(C=0.40, Mn=0.875, Si=0.25, Cr=0.975, Mo=0.20),
    "4340": _comp(C=0.40, Mn=0.70, Si=0.25, Ni=1.80, Cr=0.80, Mo=0.25),
    "5140": _comp(C=0.40, Mn=0.80, Si=0.25, Cr=0.80),
    "8620": _comp(C=0.20, Mn=0.80, Si=0.25, Ni=0.55, Cr=0.50, Mo=0.20),
    "9255": _comp(C=0.55, Mn=0.80, Si=2.00),
    "304": _comp(C=0.05, Mn=1.5, Si=0.5, Cr=18.5, Ni=9.0),
    "430": _comp(C=0.06, Mn=0.6, Si=0.5, Cr=17.0),
    "440C": _comp(C=1.05, Mn=0.6, Si=0.5, Cr=17.0, Mo=0.5),
    "D2": _comp(C=1.50, Mn=0.4, Si=0.3, Cr=12.0, Mo=0.9, V=0.9),
    "M300 maraging": _comp(C=0.01, Ni=18.5, Co=9.0, Mo=4.8, Ti=0.7),
}
EXPECTED = {"1020": "low_carbon", "1045": "medium_carbon", "1095": "high_carbon", "1137": "free_machining",
            "1340": "carbon_manganese", "4140": "cr_mo", "4340": "ni_cr_mo", "5140": "chromium",
            "8620": "ni_cr_mo", "9255": "si_mn", "304": "stainless_austenitic", "430": "stainless_ferritic",
            "440C": "stainless_martensitic", "D2": "tool"}


def test_aisi_decision_list_classifies_textbook_grades():
    X = pd.DataFrame(STEELS).T.astype(float)
    label, unknown = rule_category(X)
    got = dict(zip(X.index, label))
    for g, want in EXPECTED.items():
        assert got[g] == want, (g, got[g])
    assert unknown[list(X.index).index("M300 maraging")]


def test_ideal_critical_diameter_orders_the_classic_grades():
    X = pd.DataFrame(STEELS).T.astype(float).loc[["1045", "5140", "4140", "4340"]]
    di_in, di_mm = ideal_critical_diameter(X)
    d = dict(zip(X.index, di_in))
    assert 0.7 < d["1045"] < 1.6          # plain medium carbon: about 1 in.
    assert 4.0 < d["4140"] < 6.5          # 4140: about 5 in.
    assert d["1045"] < d["5140"] < d["4140"] < d["4340"]
    assert di_mm == pytest.approx(25.4 * di_in)


def test_jominy_cooling_rate_falls_along_the_bar():
    paths = jominy.jominy_paths(870.0, jominy.STANDARD_J)
    r = jominy.rate_at_700(paths)
    assert np.all(np.diff(r) < 0)
    assert r[0] > 100.0                    # quenched end: hundreds of K/s
    assert 1.0 < r[-1] < 6.0               # J32: a few K/s, like still air on a ~15-25 mm bar


def test_jominy_hardness_falls_with_distance_and_rises_with_alloying():
    X = pd.DataFrame([STEELS["1045"], STEELS["4340"]], index=["1045", "4340"])
    prof = jominy.simulate_profiles(X, 845.0, J=(1, 4, 8, 16, 24))
    h = prof.pivot(index="J", columns="row", values="HV")
    assert np.all(np.diff(h["1045"].to_numpy()) <= 1e-6)
    assert h.loc[16, "4340"] > h.loc[16, "1045"] + 100.0
    assert jominy.hrc_from_hv(513.0) == pytest.approx(50.0)


def test_medium_carbon_ferrite_grain_is_not_collapsed_and_refines_on_air_cooling():
    d_fc = microstructure.ferrite_grain_size_um(0.45, 0.75, 0.005, 25.0)
    d_ac = microstructure.ferrite_grain_size_um(0.45, 0.75, 1.0, 25.0)
    assert d_fc > 8.0 and d_ac > 5.0       # not at the 3 um floor
    assert d_ac < d_fc                      # normalising refines the grain


def test_transformation_finishes_below_its_start():
    t, T = kinetics.linear_path(0.1, T0=860.0)
    d = kinetics.simulate(STEELS["1045"], 7.5, t, T)
    assert d["T_finish"] <= d["T_pearlite_start"] + 1e-9
    assert d["f_ferrite"] + d["f_pearlite"] > 0.99
