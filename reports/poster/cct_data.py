"""Compute the CCT diagrams of AISI 1045 and 4140 with their furnace and air cooling paths
for the poster, using the project's own transformation model, and write the
curves to reports/poster/data/ so that make_figures.py can draw them without
numba.

    .venv/Scripts/python reports/poster/cct_data.py      (needs the project environment)

The curves are computed as in `steelml.cct.draw_cct` (figure 9b), for the nominal AISI 1045 and
AISI 4140H compositions that stage 9 uses for its five-treatment comparison.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from steelml import config  # noqa: E402
from steelml.cct import cct_starts, full_comp, route_microstructures  # noqa: E402
from steelml.physics import cooling, microstructure, thermo  # noqa: E402
from steelml.predictor import SteelPredictor  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
# the compositions stage 9 uses for its "1045" and "4140" rows (s09_five_treatments.csv)
STEELS = {"1045": "AISI 1045", "4140": "AISI 4140H"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, name in STEELS.items():
        one(tag, name)


def predictor():
    try:
        sp = SteelPredictor.load()
    except Exception as e:  # the curves and phase fractions do not need the property models
        print(f"predictor not loaded ({type(e).__name__}: {e}); using the physics-only route table")
        sp = None
    return sp


def one(tag, NAME):
    sp = predictor()
    cat = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv").set_index("grade").loc[NAME]
    comp = full_comp({el: float(cat[f"{el}_nom"]) for el in config.ELEMENTS
                      if f"{el}_nom" in cat and pd.notna(cat[f"{el}_nom"])})
    T0 = float(thermo.default_austenitising_T(comp, "AC"))
    G = float(microstructure.austenite_astm(T0))
    df = cct_starts(comp, G, T0=T0)
    rows = []
    for key, frac, ph in (("T_ferrite_start", "f_ferrite", "ferrite"),
                          ("T_pearlite_start", "f_pearlite", "pearlite"),
                          ("T_bainite_start", "f_bainite", "bainite"),
                          ("T_martensite_start", "f_martensite", "martensite")):
        ok = df[key].notna() & (df[frac] >= 0.01)
        t = (T0 - df.loc[ok, key]) / df.loc[ok, "rate"]
        rows += [{"curve": ph, "t_s": a, "T_C": b} for a, b in zip(t, df.loc[ok, key])]
    last_start = np.fmin(df["T_pearlite_start"].fillna(np.inf), df["T_bainite_start"].fillna(np.inf))
    fin = df["T_finish"].notna() & (df["T_finish"] < last_start - 3.0)
    t = (T0 - df.loc[fin, "T_finish"]) / df.loc[fin, "rate"]
    rows += [{"curve": "finish", "t_s": a, "T_C": b} for a, b in zip(t, df.loc[fin, "T_finish"])]

    rm = route_microstructures(comp, sp)
    for route in ("FC", "AC"):
        T0r = float(rm.loc[route, "T0"])
        tt, TT = (cooling.furnace_cooling_path(T0r, config.FC_RATE_700C) if route == "FC"
                  else cooling.air_cooling_path(T0r, config.AC_DEFAULT_DIAMETER_MM))
        ok = tt > 0
        rows += [{"curve": f"path_{route}", "t_s": a, "T_C": b} for a, b in zip(tt[ok], TT[ok])]
    pd.DataFrame(rows).to_csv(OUT / f"cct_{tag}.csv", index=False)

    meta = {"steel": NAME, "composition": {k: float(v) for k, v in comp.items() if v},
            "T_aust_C": T0, "Ae1_C": float(thermo.ae1(comp)), "Ae3_C": float(thermo.ae3(comp)),
            "fc_rate_K_s": float(config.FC_RATE_700C), "ac_bar_mm": float(config.AC_DEFAULT_DIAMETER_MM),
            "routes": {r: {k: float(rm.loc[r, k]) for k in ("F", "P", "B", "M", "H", "H_lo", "H_hi", "rate700")}
                       for r in ("FC", "AC")}, "hardness_unit": rm.attrs["unit"]}
    (OUT / f"cct_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(tag, {r: {k: round(v, 3) for k, v in meta["routes"][r].items() if k in "FPBM"} for r in ("FC", "AC")})


if __name__ == "__main__":
    main()
