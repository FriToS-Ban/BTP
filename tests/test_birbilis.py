import numpy as np
import pandas as pd
import pytest

from steelml.data import birbilis as bb

ELS = ["Al", "Cu", "Mn", "N", "Ni", "Ti", "S", "Fe", "Zr", "P", "Si", "V",
       "Mo", "Co", "C", "Nb", "B", "Cr"]


def _df(rows):
    recs = []
    for i, (name, proc) in enumerate(rows, 1):
        r = {"Entry": i, "Name": name, "Processing condition": proc,
             "(Ultimate) Tensile strength (MPa)": 800 + i,
             "Yield strength (MPa)": 500, "Ductility (%)": 15,
             "Cluster Label": "x"}
        r.update({e: np.nan for e in ELS})
        r.update({"C": 0.4, "Mn": 0.8, "Cr": 1.0, "Mo": 0.2})
        recs.append(r)
    return bb.parse(pd.DataFrame(recs))


def one(proc, name="AISI 4140 Steel"):
    return _df([(name, proc)]).iloc[0]


@pytest.mark.parametrize("proc,medium,aust,temper,sec", [
    ("Oil Quenched 845C (1550F), 540C (1000F) Temper, 50 mm (2 In.) Round",
     "oil", 845, 540, 50),
    ("Oil Quenched From 830C (1525F), 540C (1000F) Temper, 25 mm (1 In.) Round",
     "oil", 830, 540, 25),
    ("Oil Quenched 845C (1550F), Tempered 540C (1000F), 50 mm (2 In.) Round",
     "oil", 845, 540, 50),
    ("Oil Quenched, 25 mm (1 In.) Round (845C (1550F) Quench, 595C (1100F) Temper)",
     "oil", 845, 595, 25),
    ("Oil Quenched, 50 mm (2 In.) Round [845C (1550F) Quench, 540C (1000F) Temper]",
     "oil", 845, 540, 50),
    ("Oil Quenched, Tempered 425C (800F), 25 mm (1 In.) Round",
     "oil", np.nan, 425, 25),
    ("Water Quenched 865C (1590F), 480C (900F) Temper, 50 mm (2 In.) Round",
     "water", 865, 480, 50),
    ("Water Quenched 865C (1590F), 540C (1000F) Temper, 14 mm (0.55 In.) Round",
     "water", 865, 540, 14),
    ("Oil Quenched 800C (1470F), 650C (1200F) Temper, 50 mm Round",
     "oil", 800, 650, 50),
    ("Oil Quenched 845C (1550F), 455C (850F) Temper",
     "oil", 845, 455, np.nan),
])
def test_qt(proc, medium, aust, temper, sec):
    r = one(proc)
    assert r.route == "QT"
    assert r.quench_medium == medium
    assert np.isclose(r.austenitize_T_C, aust, equal_nan=True)
    assert r.temper_T_C == temper
    assert np.isclose(r.section_mm, sec, equal_nan=True)


def test_normalized():
    r = one("Normalized At 870C (1600F), Air Cooled, 13 mm (0.5 In.) Round")
    assert r.route == "NO" and r.normalize_T_C == 870
    assert r.austenitize_T_C == 870 and r.section_mm == 13
    assert np.isnan(r.temper_T_C)
    r = one("Normalized, 50 mm (2 In.) Round")
    assert r.route == "NO" and np.isnan(r.normalize_T_C) and r.section_mm == 50


def test_anneal():
    r = one("Annealed At 870C (1600F)")
    assert r.route == "AN" and r.austenitize_T_C == 870
    r = one("Annealed 795C (1460F)")
    assert r.route == "AN" and r.austenitize_T_C == 795
    r = one("Annealed At 870C (1600F), Furnace Cooled 28C (50F) Per Hour To "
            "650C (1200F), Air Cooled, 25 mm (1 In.) Round")
    assert r.route == "AN" and r.austenitize_T_C == 870
    assert one("Spheroidized Annealed").route == "SP"


def test_cd_hr():
    r = one("Cold Drawn, 19-32 mm (0.75-1.25 In) Round")
    assert r.route == "CD" and np.isnan(r.section_mm)
    assert (r.section_lo_mm, r.section_hi_mm) == (19, 32)
    assert one("Cold Drawn, Spheroidized, Annealed, 19-32 mm (0.75-1.25 In) Round"
               ).route == "CD"
    r = one("Hot Rolled, 19-32 mm (0.75-1.25 In) Round")
    assert r.route == "HR" and (r.section_lo_mm, r.section_hi_mm) == (19, 32)
    assert one("As Rolled").route == "HR"
    r = one("Cold Drawn, High Temperature, Stress Relieved, 19-32 mm Round")
    assert r.route == "CD" and r.stress_relieved and r.cold_worked


def test_double_quench_and_carb():
    r = one("Double Quenched, 150C (300F) Temper")
    assert r.route == "QT" and r.double_quench and r.temper_T_C == 150
    r = one("Normalized At 925C (1700F), Machined To 25 mm (1 In.) Rounds, "
            "Pseudocarburized At 925C (1700F), Box Cooled To Rt, Reheated To "
            "800C (1470F) And Oil Quenched, 150C (300F) Temper")
    assert r.route == "CARB" and r.carburizing
    assert r.austenitize_T_C == 800 and r.temper_T_C == 150
    assert one("Direct Quenched From Pot, Tempered At 150C (300F), "
               "1 mm Case Depth").route == "CARB"
    assert one("Mock Carburized, 150C (300F) Temper, 25 mm (1 In.) Round"
               ).route == "CARB"


def test_normalize_then_quench():
    r = one("Normalized At 870C (1600F), Oil Quenched 845C (1550F), "
            "540C (1000F) Temper, 25 mm Diam.")
    assert r.route == "QT" and r.normalize_T_C == 870
    assert r.austenitize_T_C == 845 and r.temper_T_C == 540


def test_misc():
    r = one("Water Quenched At 845C (1550F), Tempered 540C (1000F)")
    assert r.quench_medium == "water" and r.temper_T_C == 540
    assert one("Brine Quenched 845C (1550F), 540C (1000F) Temper"
               ).quench_medium == "brine"
    r = one("Quenched 845C (1550F), Tempered 540C (1000F)")
    assert r.route == "QT" and np.isnan(r.quench_medium)
    r = one("Oil Quenched 845C (1550F), 540C (1000F) Temper For 2 Hrs")
    assert r.temper_time_h == 2
    assert one("Grade B").route == "OTHER"
    assert one("Astm A743").route == "OTHER"


def test_grades_and_scope():
    for name, grade, base, fam in [
        ("AISI 4140H Steel", "4140H", "4140", "alloy"),
        ("AISI E4340 Steel", "E4340", "4340", "alloy"),
        ("AISI E4340 H Steel", "E4340H", "4340", "alloy"),
        ("AISI 86B30H Steel", "86B30H", "86B30", "alloy"),
        ("AISI 1045 Steel", "1045", "1045", "carbon"),
        ("AISI 12L14 Steel", "12L14", "12L14", "carbon"),
        ("Cast Stainless Steel CH20", "Cast Stainless CH20",
         "Cast Stainless CH20", "cast"),
    ]:
        r = one("Normalized, 25 mm (1 In.) Round", name)
        assert (r.grade, r.base_grade, r.family) == (grade, base, fam) or \
            (fam == "cast" and r.family == "cast")
    assert one("Annealed", "AISI 4140 Steel").in_scope
    assert not one("Annealed", "Cast Stainless Steel CH20").in_scope
    assert not one("Annealed", "ASTM A36 Steel").in_scope


def test_dup_of():
    df = pd.DataFrame({
        "Entry": [10, 11, 12], "Name": ["AISI 4140 Steel", "AISI 4140H Steel",
                                         "AISI 4140 Steel"],
        "Processing condition": ["Normalized, 25 mm (1 In.) Round"] * 3,
        "(Ultimate) Tensile strength (MPa)": [800, 800, 801],
        "Yield strength (MPa)": [500] * 3, "Ductility (%)": [15] * 3,
        "C": [0.4] * 3})
    p = bb.parse(df)
    assert np.isnan(p.dup_of[0]) and p.dup_of[1] == 10 and np.isnan(p.dup_of[2])


@pytest.fixture(scope="module")
def real():
    if not bb.RAW_FILE.exists():
        pytest.skip("raw file missing")
    return bb.parse(bb.load_raw())


def test_real_file(real):
    assert len(real) == 3234
    ins = real[real.in_scope]
    qt = ins[(ins.route == "QT") & ins.temper_T_C.notna()
             & ins.quench_medium.notna()]
    assert len(qt) >= 800
    assert (ins.route == "NO").sum() >= 300
    qts = ins[ins.route == "QT"]
    sec = qts.section_mm.dropna()
    assert sec.between(5, 300).all()
    assert {13, 14, 25, 50, 100} <= set(sec)
    assert qts.temper_T_C.dropna().between(100, 750).all()
    assert qts.austenitize_T_C.dropna().between(700, 1000).all()
    assert set(real.route) <= set(bb.ROUTES)
    assert bb.summary(real).loc[:, "total"].sum() == 3234
