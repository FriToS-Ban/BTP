"""Physical-consistency tests for property models.

Accuracy on held-out grades is necessary but not sufficient: the models are
later queried at cooling rates and compositions that the handbook data do
not contain (other furnace rates, other bar diameters, a carbon sweep). Two
pieces of textbook metallurgy must hold there:

1. Cooling-rate monotonicity. For a fixed steel and austenitising
   temperature, faster cooling lowers the transformation temperature and
   refines the product, so strength and hardness must not fall, and
   elongation must not rise, as the cooling rate increases from slow furnace
   cooling to air cooling of thin sections.
2. Carbon monotonicity. For plain-carbon steels cooled the same way,
   strength and hardness rise and elongation falls with carbon up to the
   eutectoid composition.
3. Silicon sensitivity. Across the usual 0.10-0.40 % Si of carbon and
   low-alloy steels, silicon adds roughly 80 MPa per wt% of solid-solution
   strength, so strength should change by a few tens of MPa at most. Every
   training grade has the same nominal Si (0.225 or 0.25 %, apart from the
   2 % Si spring steel 9255), so leave-one-grade-out cannot detect a model
   that has learned a spurious silicon effect; this sweep can. It is
   reported as the span (max - min) of the prediction over the sweep.

A violation is counted when consecutive points move the wrong way by more
than ``tol`` (as a fraction of the property's data range).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..features import featurize
from ..physics import thermo

FC_SWEEP = [0.003, 0.005, 0.01, 0.02, 0.03]
AC_SWEEP_MM = [150, 100, 75, 50, 35, 25, 15, 10]


def rate_sweep_frame(grades: pd.DataFrame):
    """One row per (grade, cooling condition), ordered by increasing rate."""
    rows = []
    for _, g in grades.iterrows():
        base = {el: g[el] for el in config.ELEMENTS}
        T = float(g["T_aust"])
        for r in FC_SWEEP:
            rows.append({**base, "grade": g["grade"], "route": "FC", "treat_T_C": T, "fc_rate": r,
                         "section_mm": 25.0})
        for D in AC_SWEEP_MM:
            rows.append({**base, "grade": g["grade"], "route": "AC", "treat_T_C": T,
                         "fc_rate": config.FC_RATE_700C, "section_mm": float(D)})
    return pd.DataFrame(rows)


def carbon_sweep_frame(C_values=None, Mn=0.75, Si=0.25):
    if C_values is None:
        C_values = np.round(np.arange(0.05, 0.86, 0.05), 2)
    rows = []
    for route in ("FC", "AC"):
        for C in C_values:
            comp = {el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS}
            comp.update({"C": C, "Mn": Mn, "Si": Si})
            # usual austenitising temperature for the route (as in the handbook data)
            T = float(thermo.default_austenitising_T(comp, route))
            rows.append({**comp, "grade": f"C{C:.2f}", "route": route, "treat_T_C": T,
                         "section_mm": 25.0, "fc_rate": config.FC_RATE_700C})
    return pd.DataFrame(rows)


def _violations(values, increasing, tol):
    v = np.asarray(values, float)
    dv = np.diff(v)
    bad = dv < -tol if increasing else dv > tol
    return int(bad.sum()), len(dv)


def consistency_report(models: dict, F_rate: pd.DataFrame, F_carbon: pd.DataFrame,
                       target_ranges: dict, tol_frac=0.02, F_si: pd.DataFrame | None = None):
    """models: {(model_name, target): fitted model}. Returns a tidy DataFrame."""
    out = []
    for (mname, target), m in models.items():
        inc = target != "El_pct"
        tol = tol_frac * target_ranges[target]
        mu, _ = m.predict(F_rate)
        F_rate = F_rate.assign(_p=mu)
        nb = nt = 0
        for _, grp in F_rate.groupby("grade", sort=False):
            grp = grp.sort_values("cool_rate_700")
            b, n = _violations(grp._p, inc, tol)
            nb += b
            nt += n
        mu_c, _ = m.predict(F_carbon)
        Fc = F_carbon.assign(_p=mu_c)
        cb = ct = 0
        for _, grp in Fc.groupby("route"):
            grp = grp.sort_values("C")
            b, n = _violations(grp._p, inc, tol)
            cb += b
            ct += n
        rec = {"model": mname, "target": target,
               "rate_violation_frac": nb / max(nt, 1), "carbon_violation_frac": cb / max(ct, 1)}
        if F_si is not None:
            rec["si_0.10_0.40_span"] = silicon_span(m, F_si)
        out.append(rec)
    return pd.DataFrame(out)


SI_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
SI_STEELS = {"0.20C": {"C": 0.20, "Mn": 0.45}, "0.45C": {"C": 0.45, "Mn": 0.75},
             "4140-type": {"C": 0.40, "Mn": 0.875, "Cr": 0.95, "Mo": 0.20}}


def silicon_sweep_frame():
    rows = []
    for name, base in SI_STEELS.items():
        for route in ("FC", "AC"):
            for si in SI_SWEEP:
                comp = {el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS}
                comp.update(base, Si=si)
                T = float(thermo.default_austenitising_T(comp, route))
                rows.append({**comp, "grade": f"{name}|{route}", "route": route, "treat_T_C": T,
                             "section_mm": 25.0, "fc_rate": config.FC_RATE_700C})
    return pd.DataFrame(rows)


def silicon_span(m, F_si):
    """Mean span (max - min) of the prediction over the silicon sweep."""
    mu, _ = m.predict(F_si)
    return float(F_si.assign(_p=mu).groupby("grade", sort=False)._p.agg(lambda s: s.max() - s.min()).mean())


def featurized_sweeps(primary: pd.DataFrame):
    """Featurise the rate sweep (all handbook grades, normalising temperature),
    the plain-carbon sweep and the silicon sweep."""
    norm = primary[primary.condition == "normalized"].drop_duplicates("grade")
    grades = norm[["grade"] + config.ELEMENTS].copy()
    grades["T_aust"] = norm["treat_T_C"].to_numpy()
    Fr = featurize(rate_sweep_frame(grades)).reset_index(drop=True)
    Fc = featurize(carbon_sweep_frame()).reset_index(drop=True)
    Fs = featurize(silicon_sweep_frame()).reset_index(drop=True)
    return Fr, Fc, Fs
