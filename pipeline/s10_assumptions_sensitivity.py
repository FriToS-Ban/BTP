"""Stage 10 - every assumption made explicit, and how much each one matters.

1. Assumptions register (results/s10_assumptions.csv): each condition the
   models rely on, whether it is documented, fitted, taken from the
   literature or inferred, the evidence, the range tested, and where it is used.
2. The one inferred condition that could be checked against data: the bar
   size of the handbook's normalised rows. Matching them to the Birbilis et
   al. (2025) rows that state the bar size shows they are 25 mm (1 in.)
   rounds (s10_handbook_section_match.csv), not the 50 mm that versions 1-2
   inferred.
3. One-at-a-time retraining: the annealing/normalising production model is
   retrained and re-scored (leave one grade out) with each assumption at the
   ends of its range, and the furnace-vs-air answer (UTS gain on air
   cooling) is recomputed for representative grades. For the
   quench-and-temper chain the fitted physics (grey box) is re-scored the
   same way.
4. Global sensitivity (Morris elementary effects) of the physics chain to all
   the assumptions at once.

Outputs: results/s10_*.csv, figures fig38-fig39.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import config, viz  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt
RES = config.RESULTS_DIR
REP = {"1020": "AISI 1020", "1045": "AISI 1045", "1080": "AISI 1080", "4140": "AISI 4140H", "4340": "AISI 4340H"}

ASSUMPTIONS = [
    # name, value, status, evidence, range tested, used in
    ("Bar size of the handbook normalised rows", "25 mm (1 in.) round", "documented",
     "same values as the 25 mm rows of Birbilis et al. 2025 (21 of 28 grades within 3 MPa)", "13, 25, 50 mm",
     "property models (stage 4)"),
    ("Furnace cooling rate of the handbook annealed rows", "0.005 K/s at 700 C (18 K/h)", "inferred",
     "usual full-annealing practice (10-50 K/h); physics-only error flat below 0.01 K/s", "0.002-0.02 K/s",
     "property models (stage 4)"),
    ("Austenitising temperatures of handbook / Birbilis rows", "as stated in the sources", "documented",
     "given for 100 % of handbook rows, 72 % of normalised and 91 % of Q+T bars", "-", "all stages"),
    ("Default austenitising rules (anneal / normalise / harden)", "Ae3+50 / upper+110 (>=870) / Ae3+70 C", "fitted",
     "MAE 14 / 15 / 16 C against the handbook and Birbilis practice", "-", "predictor defaults"),
    ("Still-air heat transfer", "laminar natural convection + radiation, emissivity 0.8", "literature",
     "agrees with measured 25 and 45-50 mm bar cooling rates (stage 3)", "emissivity 0.6-0.9; convection x0.7-1.3",
     "normalising paths"),
    ("Prior-austenite grain size", "ASTM 8 at 850 C, one number coarser per 100 C", "assumed",
     "fine-grained Al-killed practice (ASTM E112 charts)", "-1 to +1 ASTM number", "kinetics, Hall-Petch"),
    ("Quench severity", "Grossmann H, middle of the 'mild agitation' class", "literature",
     "Grossmann et al. 1939 table; oil 0.30-0.35, water 1.0-1.1 per inch", "low and high end of the class",
     "hardening paths"),
    ("Conductivity used to turn H into h", "30 W/m/K", "assumed", "hot austenite 25-35 W/m/K (EN 1993-1-2)",
     "25-40 W/m/K", "hardening paths"),
    ("Specific heat during a quench", "Eurocode polynomial capped at 650 J/kg/K (no 735 C peak)", "assumed",
     "the peak is the heat of slow transformation; with it an oil-quenched surface came out softer than the centre",
     "Eurocode with peak", "hardening paths"),
    ("Tensile specimen position", "centre up to 38 mm, mid-radius above (ASTM A370)", "documented standard",
     "ASTM A370 bar products", "centre for every bar", "Q+T models"),
    ("Tempering time when not given", "1 h", "assumed", "time is not stated in the Birbilis rows",
     "0.5 and 2 h", "Q+T models"),
    ("Hollomon-Jaffe constant", "C = 20", "literature", "Hollomon & Jaffe 1945", "18 and 22", "Q+T, stress relief"),
    ("Kinetic constants (bainite)", "Bs - 107 C, rate x 79", "fitted (Bayesian)",
     "SAE H-bands, cross-validated by grade (stage 8)", "published Li constants", "all transformation paths"),
    ("Precipitation particle size", "8 nm air cooled, 30 nm furnace cooled", "assumed",
     "interphase vs coarsened carbonitrides (Gladman 1997)", "+/-50 % on the strengthening", "microalloyed steels"),
    ("Uniform / total elongation", "0.508 +/- 0.051", "fitted", "10 tests, 4 sources (stage 6)", "its SD",
     "uniform elongation estimate"),
]


def catalogue_comp(name):
    cat = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv").set_index("grade")
    r = cat.loc[name]
    return {el: float(r[f"{el}_nom"]) for el in config.ELEMENTS if f"{el}_nom" in r and pd.notna(r[f"{el}_nom"])}


# ---------------------------------------------------------------------------
# 2 handbook bar size
# ---------------------------------------------------------------------------

def handbook_section_match():
    hb = pd.read_csv(config.RAW_DIR / "asm_bethlehem_properties.csv")
    hb["grade"] = hb.grade.str.replace("AISI ", "", regex=False)
    nb = pd.read_csv(config.PROCESSED_DIR / "normalized_bars.csv")
    nb = nb[(nb.route == "AC") & nb.section_mm.notna()]
    m = hb[hb.condition == "normalized"][["grade", "UTS_MPa", "YS_MPa"]].merge(
        nb[["grade", "section_mm", "UTS_MPa", "YS_MPa"]], on="grade", suffixes=("_handbook", "_bar"))
    m["dUTS"] = m.UTS_MPa_bar - m.UTS_MPa_handbook
    m["dYS"] = m.YS_MPa_bar - m.YS_MPa_handbook
    s = m.groupby("section_mm").agg(grades=("grade", "nunique"), mean_dUTS=("dUTS", "mean"),
                                    rms_dUTS=("dUTS", lambda x: float(np.sqrt(np.mean(x ** 2)))),
                                    within_3MPa=("dUTS", lambda x: int((x.abs() <= 3).sum())),
                                    mean_dYS=("dYS", "mean")).reset_index()
    return m, s


# ---------------------------------------------------------------------------
# 3 one-at-a-time retraining
# ---------------------------------------------------------------------------

FCAC_SETTINGS = [
    ("baseline", {}),
    ("furnace cooling 0.002 K/s", {"fc_rate": 0.002}), ("furnace cooling 0.01 K/s", {"fc_rate": 0.01}),
    ("furnace cooling 0.02 K/s", {"fc_rate": 0.02}),
    ("normalised bars 50 mm (v2 inference)", {"section": 50.0}), ("normalised bars 13 mm", {"section": 13.0}),
    ("emissivity 0.6", {"eps": 0.6}), ("emissivity 0.9", {"eps": 0.9}),
    ("convection x0.7", {"hconv": 0.7}), ("convection x1.3", {"hconv": 1.3}),
    ("grain size ASTM -1 (coarser)", {"astm": -1.0}), ("grain size ASTM +1 (finer)", {"astm": 1.0}),
]


def _apply(opts):
    from steelml.physics import cooling, microstructure, quench, tempering
    cooling.EPS = opts.get("eps", 0.8)
    cooling.H_CONV_FACTOR = opts.get("hconv", 1.0)
    microstructure.ASTM_OFFSET = opts.get("astm", 0.0)
    quench.K_REF = opts.get("kref", 30.0)
    quench.SMOOTH_CP_LIQUID = opts.get("smooth_cp", True)
    tempering.C_HJ = opts.get("chj", 20.0)


def run_fcac_setting(label, opts):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from steelml.features import featurize
    from steelml.models.evaluation import logo_oof
    from steelml.models.property_models import PRODUCTION_MODEL, make_model
    _apply(opts)
    p = pd.read_csv(config.PROCESSED_DIR / "properties_handbook.csv")
    p.loc[p.condition == "normalized", "section_mm"] = opts.get("section", config.HANDBOOK_AC_SECTION_MM)
    p["fc_rate"] = opts.get("fc_rate", config.HANDBOOK_FC_RATE)
    F = featurize(p).reset_index(drop=True)
    fcac = (F.route != "AR").to_numpy()
    row = {"setting": label}
    for t in ("UTS_MPa", "HB"):
        y = F[t].to_numpy(float)
        mu, _ = logo_oof(F, y, F.grade.to_numpy(), PRODUCTION_MODEL, t)
        row[f"{t}_RMSE"] = float(np.sqrt(np.mean((mu[fcac] - y[fcac]) ** 2)))
    m = make_model(PRODUCTION_MODEL, "UTS_MPa").fit(F, F.UTS_MPa.to_numpy(float))
    rows = []
    for g, name in REP.items():
        c = catalogue_comp(name)
        for route in ("FC", "AC"):
            rows.append({**c, "route": route, "section_mm": 25.0, "fc_rate": config.FC_RATE_700C, "g": g})
    d = pd.DataFrame(rows)
    from steelml.physics import thermo
    d["treat_T_C"] = np.asarray(thermo.default_austenitising_T(d, d.route), float)
    Fd = featurize(d)
    mu, _ = m.predict(Fd)
    Fd["UTS"] = mu
    w = Fd.pivot_table(index="g", columns="route", values="UTS")
    for g in REP:
        row[f"dUTS_{g}"] = float(w.loc[g, "AC"] - w.loc[g, "FC"])
    _apply({})
    return row


QT_SETTINGS = [
    ("baseline", {}),
    ("specimen at the centre of every bar", {"pos": "centre"}),
    ("tempering time 0.5 h", {"t": 0.5}), ("tempering time 2 h", {"t": 2.0}),
    ("H at the low end of the class", {"H": "low"}), ("H at the high end of the class", {"H": "high"}),
    ("conductivity for H->h 25 W/m/K", {"kref": 25.0}), ("conductivity for H->h 40 W/m/K", {"kref": 40.0}),
    ("Hollomon-Jaffe C = 18", {"chj": 18.0}), ("Hollomon-Jaffe C = 22", {"chj": 22.0}),
    ("Eurocode specific heat with peak", {"smooth_cp": False}),
]


def run_qt_setting(label, opts):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from steelml.features import featurize_quench
    from steelml.models import qt_models as qm
    from steelml.physics import quench
    _apply(opts)
    qt = pd.read_csv(config.PROCESSED_DIR / "qt_bars.csv")
    if "t" in opts:
        qt["temper_time_h"] = opts["t"]
    if opts.get("pos") == "centre":
        qt["pos_rR"] = 0.0
    if "H" in opts:
        key = {"low": "H_low", "high": "H_high"}[opts["H"]]
        qt["H"] = [quench.severity(m, "mild")[key] for m in qt.quench_medium]
    F = featurize_quench(qt)
    y = F.UTS_MPa.to_numpy(float)
    mu = np.full(len(F), np.nan)
    for g in F.grade.unique():
        te = (F.grade == g).to_numpy()
        m = qm.make_qt_model("qt_greybox", "UTS_MPa").fit(F[~te], y[~te])
        mu[te] = m.predict(F[te])[0]
    e = mu - y
    row = {"setting": label, "greybox_UTS_RMSE": float(np.sqrt(np.mean(e ** 2))),
           "mean_f_martensite": float(F.f_M.mean())}
    for D in (13, 25, 50, 100):
        row[f"bias_{D}mm"] = float(np.mean(e[F.section_mm.round().to_numpy() == D]))
    _apply({})
    return row


# ---------------------------------------------------------------------------
# 4 Morris elementary effects of the physics chain
# ---------------------------------------------------------------------------

MORRIS_FACTORS = [  # name, low, high, log-scale
    ("fc_rate", 0.002, 0.02, True), ("eps", 0.6, 0.9, False), ("hconv", 0.7, 1.3, False),
    ("astm", -1.0, 1.0, False), ("H_oil", 0.25, 0.40, False), ("kref", 25.0, 40.0, False),
    ("t_temper", 0.5, 2.0, True), ("chj", 18.0, 22.0, False),
]


def _physics_outputs(x, comps, theta):
    from steelml.features import add_tempered_physics, featurize, featurize_quench
    from steelml.physics import thermo
    opts = dict(zip([f[0] for f in MORRIS_FACTORS], x))
    _apply(opts)
    rows = []
    for g in ("1045", "4140"):
        for route in ("FC", "AC"):
            rows.append({**comps[g], "route": route, "section_mm": 25.0, "fc_rate": opts["fc_rate"], "g": g})
    d = pd.DataFrame(rows)
    d["treat_T_C"] = np.asarray(thermo.default_austenitising_T(d, d.route), float)
    Fd = featurize(d)
    w = Fd.pivot_table(index="g", columns="route", values="UTS_phys")
    q = pd.DataFrame([{**comps["4140"], "quench_medium": "oil", "section_mm": 50.0, "temper_T_C": 550.0,
                       "temper_time_h": opts["t_temper"], "H": opts["H_oil"]}])
    Fq = featurize_quench(q, theta)
    add_tempered_physics(Fq, theta)
    _apply({})
    return np.array([w.loc["1045", "AC"] - w.loc["1045", "FC"], w.loc["4140", "AC"] - w.loc["4140", "FC"],
                     float(Fq.UTS_qt_phys.iloc[0]), float(Fq.f_M.iloc[0])])


def morris(comps, theta, r=12, levels=4, seed=0):
    rng = np.random.default_rng(seed)
    k = len(MORRIS_FACTORS)
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    effects = [[[] for _ in range(4)] for _ in range(k)]

    def to_x(u):
        x = []
        for ui, (_, lo, hi, lg) in zip(u, MORRIS_FACTORS):
            x.append(float(np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo)))) if lg else lo + ui * (hi - lo))
        return x
    for _ in range(r):
        u = rng.choice(grid[grid <= 1 - delta + 1e-9], k)
        y0 = _physics_outputs(to_x(u), comps, theta)
        for i in rng.permutation(k):
            u2 = u.copy()
            u2[i] = u[i] + delta
            y1 = _physics_outputs(to_x(u2), comps, theta)
            for j in range(4):
                effects[i][j].append((y1[j] - y0[j]) / delta)
            u, y0 = u2, y1
    out = []
    names = ["dUTS_AC_FC_1045", "dUTS_AC_FC_4140", "UTS_QT_4140_50mm", "f_martensite_4140_50mm_oil"]
    for i, (f, *_rest) in enumerate(MORRIS_FACTORS):
        for j, n in enumerate(names):
            e = np.array(effects[i][j])
            out.append({"factor": f, "output": n, "mu_star": float(np.mean(np.abs(e))), "sigma": float(np.std(e))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------

def fig_tornado(oat, qto, match):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1.0, 1.25, 1.1]})
    ax = axes[0]
    ax.bar(match.section_mm.astype(int).astype(str), match.mean_dUTS,
           color=[viz.BLUE if s == 25 else viz.NEUTRAL for s in match.section_mm])
    for i, (v, n, w) in enumerate(zip(match.mean_dUTS, match.grades, match.within_3MPa)):
        ax.text(i, v + (4 if v >= 0 else -10), f"{w}/{n} within 3 MPa", ha="center", fontsize=8, color=viz.INK2)
    ax.axhline(0, color=viz.AXIS, lw=1)
    ax.set_xlabel("bar size of the Birbilis normalised rows (mm)")
    ax.set_ylabel("UTS: bar minus handbook (MPa), mean of grades")
    ax.set_title("Handbook normalised rows are 25 mm bars", loc="left")
    ax = axes[1]
    base = oat[oat.setting == "baseline"].iloc[0]
    s = oat[oat.setting != "baseline"]
    y = np.arange(len(s))
    ax.barh(y - 0.2, s.UTS_MPa_RMSE - base.UTS_MPa_RMSE, height=0.4, color=viz.BLUE, label="UTS RMSE change (MPa)")
    ax.barh(y + 0.2, s.dUTS_1045 - base.dUTS_1045, height=0.4, color=viz.ORANGE,
            label="1045 air-cooling gain change (MPa)")
    ax.set_yticks(y); ax.set_yticklabels(s.setting, fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color=viz.AXIS, lw=1)
    ax.set_title("Anneal / normalise: retrained per assumption", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax = axes[2]
    b = qto[qto.setting == "baseline"].iloc[0]
    s = qto[qto.setting != "baseline"]
    y = np.arange(len(s))
    ax.barh(y, s.greybox_UTS_RMSE - b.greybox_UTS_RMSE, color=viz.AQUA)
    ax.set_yticks(y); ax.set_yticklabels(s.setting, fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color=viz.AXIS, lw=1)
    ax.set_xlabel("change in Q+T physics RMSE (MPa)")
    ax.set_title("Quench + temper: refitted per assumption", loc="left")
    return viz.savefig(fig, "fig38_assumption_sensitivity")


def fig_morris(mo):
    outs = mo.output.unique()
    fig, axes = plt.subplots(1, len(outs), figsize=(3.4 * len(outs), 3.6))
    labels = {"dUTS_AC_FC_1045": "1045: air-cooling gain (MPa)", "dUTS_AC_FC_4140": "4140: air-cooling gain (MPa)",
              "UTS_QT_4140_50mm": "4140 Q+T, 50 mm: UTS (MPa)", "f_martensite_4140_50mm_oil": "4140 50 mm oil: martensite"}
    for ax, o in zip(axes, outs):
        s = mo[mo.output == o].sort_values("mu_star")
        ax.barh(s.factor, s.mu_star, color=viz.BLUE)
        ax.set_title(labels[o], loc="left", fontsize=9)
        ax.set_xlabel("Morris mu* (effect over the range)")
    return viz.savefig(fig, "fig39_morris_sensitivity")


def main():
    viz.apply_style()
    t0 = time.time()
    pd.DataFrame(ASSUMPTIONS, columns=["assumption", "value", "status", "evidence", "range_tested", "used_in"]).to_csv(
        RES / "s10_assumptions.csv", index=False)
    m, s = handbook_section_match()
    m.to_csv(RES / "s10_handbook_section_match_rows.csv", index=False)
    s.to_csv(RES / "s10_handbook_section_match.csv", index=False)
    print(s.round(1).to_string(index=False))

    oat = pd.DataFrame(Parallel(n_jobs=4)(delayed(run_fcac_setting)(lab, o) for lab, o in FCAC_SETTINGS))
    oat.to_csv(RES / "s10_oat_anneal_normalise.csv", index=False)
    print(f"  anneal/normalise OAT: {time.time() - t0:.0f} s")
    qto = pd.DataFrame(Parallel(n_jobs=4)(delayed(run_qt_setting)(lab, o) for lab, o in QT_SETTINGS))
    qto.to_csv(RES / "s10_oat_quench_temper.csv", index=False)
    print(f"  quench/temper OAT: {time.time() - t0:.0f} s")

    import joblib
    theta = joblib.load(config.MODELS_DIR / "qt_models.joblib").theta
    comps = {g: catalogue_comp(REP[g]) for g in ("1045", "4140")}
    mo = morris(comps, theta)
    mo.to_csv(RES / "s10_morris.csv", index=False)
    fig_tornado(oat, qto, s)
    fig_morris(mo)
    print(f"stage 10 done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
