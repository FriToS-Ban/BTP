"""Data-parsing, taxonomy and end-to-end tests."""

import numpy as np
import pandas as pd
import pytest

from steelml import categories as cat
from steelml import config
from steelml.data.parsers import parse_range


@pytest.mark.parametrize("text, expected", [
    (".06 max", (0.0, 0.06)), (".25/.40", (0.25, 0.40)), ("0.10/0.16", (0.10, 0.16)),
    (".27-.33", (0.27, 0.33)), ("16–18", (16.0, 18.0)), ("0.035 (max)", (0.0, 0.035)),
    (".15Min.V", (0.15, None)), ("", (None, None)),
])
def test_parse_range(text, expected):
    lo, hi = parse_range(text)
    for got, exp in ((lo, expected[0]), (hi, expected[1])):
        if exp is None:
            assert np.isnan(got)
        else:
            assert got == pytest.approx(exp)


@pytest.mark.parametrize("grade, c, expected", [
    ("AISI 1045", 0.465, "medium_carbon"), ("AISI 1018", 0.175, "low_carbon"), ("AISI 1095", 0.965, "high_carbon"),
    ("AISI 1117", 0.17, "free_machining"), ("AISI 1340H", 0.40, "carbon_manganese"), ("AISI 4140H", 0.40, "cr_mo"),
    ("AISI 4340H", 0.40, "ni_cr_mo"), ("AISI 8620H", 0.20, "ni_cr_mo"), ("AISI 5160H", 0.60, "chromium"),
    ("AISI 6150H", 0.50, "chromium"), ("AISI 9255", 0.55, "si_mn"), ("AISI 15B41H", 0.40, "carbon_manganese"),
])
def test_aisi_series_category(grade, c, expected):
    assert cat.aisi_series_category(grade, c) == expected


processed = config.PROCESSED_DIR / "properties_handbook.csv"


@pytest.mark.skipif(not processed.exists(), reason="run the pipeline first")
def test_handbook_table_integrity():
    p = pd.read_csv(processed)
    assert len(p) == 80 and p.grade.nunique() == 33
    row = p[(p.grade == "AISI 4140") & (p.condition == "normalized")].iloc[0]
    assert row.UTS_MPa == pytest.approx(148000 * config.PSI_TO_MPA, abs=0.1)   # 1020.4 MPa
    # every grade's normalised strength is at least its annealed strength (measured data)
    w = p.pivot(index="grade", columns="condition", values="UTS_MPa")
    assert (w.normalized >= w.annealed).all()


bundle = config.MODELS_DIR / "steel_predictor.joblib"


@pytest.mark.skipif(not bundle.exists(), reason="run the pipeline first")
def test_predictor_what_if_is_physically_consistent():
    from steelml.predictor import SteelPredictor
    sp = SteelPredictor.load()
    out = sp.what_if({"C": 0.45, "Mn": 0.75, "Si": 0.25})
    fc, ac = out.iloc[0], out.iloc[1]
    assert ac.UTS_MPa_pred > fc.UTS_MPa_pred and ac.HB_pred > fc.HB_pred
    assert ac.El_pct_pred < fc.El_pct_pred + 1.0
    for r in (fc, ac):
        assert r.UTS_MPa_lo90 < r.UTS_MPa_pred < r.UTS_MPa_hi90
        s = r.F_best + r.P_best + r.B_best + r.M_best + r.Fe3C_best
        assert s == pytest.approx(1.0, abs=0.02)
    assert bool(fc.in_domain) and fc.category == "medium_carbon"


@pytest.mark.skipif(not bundle.exists(), reason="run the pipeline first")
def test_predictor_flags_out_of_scope_steel():
    from steelml.predictor import SteelPredictor
    sp = SteelPredictor.load()
    out = sp.predict(pd.DataFrame([{"C": 0.02, "Mn": 0.1, "Si": 0.1, "Ni": 18.0, "Co": 9.0, "Mo": 5.0,
                                    "Ti": 0.6, "route": "AC", "section_mm": 25.0, "treat_T_C": 820.0}]))
    assert bool(out.category_unknown.iloc[0]) and not bool(out.in_domain.iloc[0])


def test_domain_ignores_low_residual_alloying_elements():
    """Less Cr/Ni/Mo/V than the assumed residual level is not an extrapolation,
    but carbon or manganese outside the training range is."""
    from steelml.models.domain import Domain
    train = pd.DataFrame({"C": [0.2, 0.4, 0.6], "Mn": [0.5, 0.8, 1.0], "Si": [0.2, 0.25, 0.3],
                          "Cr": [0.1, 0.1, 1.0], "Ni": [0.1, 0.1, 0.5], "Mo": [0.03, 0.03, 0.2],
                          "V": [0.005, 0.005, 0.005]})
    dom = Domain().fit(train)
    clean = pd.DataFrame([{"C": 0.4, "Mn": 0.8, "Si": 0.25, "Cr": 0.0, "Ni": 0.0, "Mo": 0.0, "V": 0.0}])
    low_c = clean.assign(C=0.02)
    high_cr = clean.assign(Cr=2.0)
    assert bool(dom.check(clean).in_range.iloc[0])
    assert not bool(dom.check(low_c).in_range.iloc[0])
    assert not bool(dom.check(high_cr).in_range.iloc[0])


def _air_cooled_bars(comp, sections):
    from steelml.predictor import default_austenitising_T
    rows = [{**{el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS}, **comp,
             "route": "AC", "section_mm": float(D), "fc_rate": config.FC_RATE_700C} for D in sections]
    X = pd.DataFrame(rows)
    X["treat_T_C"] = default_austenitising_T(X, X.route)
    return X


AISI_4340 = {"C": 0.405, "Mn": 0.725, "Si": 0.25, "Ni": 1.775, "Cr": 0.80, "Mo": 0.25}


def test_model_descriptors_vanish_smoothly_without_ferrite_pearlite():
    """A trace of ferrite in a bainitic 4340 bar must not flip the descriptors
    the GPs see (the raw ratio fa_third jumps from 0 to 1)."""
    from steelml.features import featurize
    F = featurize(_air_cooled_bars(AISI_4340, [75, 85, 120]))
    assert (F.f_fp < 0.05).all()
    assert F.ferrite_term.max() < 0.05
    assert F.pearlite_term.max() < 5.0   # a pearlitic steel is ~50-70


@pytest.mark.skipif(not bundle.exists(), reason="run the pipeline first")
def test_strength_falls_steadily_with_air_cooled_bar_size():
    from steelml.predictor import SteelPredictor
    sp = SteelPredictor.load()
    sections = [10, 15, 25, 35, 50, 65, 75, 85, 100, 120, 150]
    for comp in (AISI_4340, {"C": 0.465, "Mn": 0.75, "Si": 0.22}, {"C": 0.205, "Mn": 0.45, "Si": 0.22}):
        uts = sp.predict(_air_cooled_bars(comp, sections)).UTS_MPa_pred.to_numpy()
        assert np.all(np.diff(uts) <= 10.0), uts.round(0)
