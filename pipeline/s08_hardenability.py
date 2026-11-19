"""Stage 8 - hardenability: why some steels change so much on air cooling.

1. Grossmann's ideal critical diameter D_I (ASTM A255 factors) for every
   in-scope catalogue grade, set against the microstructure that the model
   predicts for a 25 mm bar cooled in still air. D_I is independent of the
   kinetic model, so their agreement is a check on the model and the
   relation is the metallurgical explanation of the category results.
2. Jominy end-quench simulation of the SAE H-steels with the same kinetic
   and hardness chain, compared with the published H-bands (if the band
   data are present in data/raw/jominy_h_bands.csv).

Outputs: results/s08_*.csv, figures fig25 and fig26.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import categories as cat  # noqa: E402
from steelml import config, viz  # noqa: E402
from steelml.physics import calibration as kcal  # noqa: E402
from steelml.physics import jominy  # noqa: E402
from steelml.physics.hardenability import ideal_critical_diameter  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt

CATEGORY_ORDER = ["low_carbon", "medium_carbon", "high_carbon", "free_machining", "carbon_manganese",
                  "chromium", "cr_mo", "ni_cr_mo", "si_mn"]


def di_table():
    """D_I of every in-scope grade with its predicted furnace / air-cooled state."""
    o = pd.read_csv(config.RESULTS_DIR / "s05_catalogue_fc_vs_ac.csv")
    catalog = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv")
    nom = catalog.set_index("grade")[[f"{e}_nom" for e in config.ELEMENTS]]
    nom.columns = config.ELEMENTS
    di_in, di_mm = ideal_critical_diameter(nom)
    w = o.pivot_table(index="grade", columns="case",
                      values=["UTS_MPa_pred", "YS_MPa_pred", "HB_pred", "B_best", "M_best"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.join(pd.DataFrame({"DI_in": di_in, "DI_mm": di_mm}, index=nom.index))
    w = w.join(nom[["C", "Mn", "Si", "Cr", "Ni", "Mo"]])
    w = w.join(o.drop_duplicates("grade").set_index("grade")[["category_true"]])
    for case in ("AC25", "AC50"):
        w[f"hard_{case}"] = w[f"B_best_{case}"] + w[f"M_best_{case}"]
        w[f"dUTS_{case}"] = w[f"UTS_MPa_pred_{case}"] - w["UTS_MPa_pred_FC"]
    return w.reset_index()


def di_summary(w):
    rows = []
    bins = [(0, 40), (40, 60), (60, 80), (80, 100), (100, 130), (130, 1000)]
    for lo, hi in bins:
        s = w[(w.DI_mm >= lo) & (w.DI_mm < hi)]
        rows.append({"DI_mm_from": lo, "DI_mm_to": hi, "grades": len(s),
                     "median_hard_phase_AC25": s.hard_AC25.median(),
                     "share_over_50pct_B_M_AC25": float(np.mean(s.hard_AC25 > 0.5)) if len(s) else np.nan,
                     "median_dUTS_AC25_MPa": s.dUTS_AC25.median()})
    out = pd.DataFrame(rows)
    rho = spearmanr(w.DI_mm, w.hard_AC25).correlation
    rho_d = spearmanr(w.DI_mm, w.dUTS_AC25).correlation
    return out, {"spearman_DI_vs_hard_phase_AC25": rho, "spearman_DI_vs_dUTS_AC25": rho_d}


def fig_hardenability(w, stats):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for i, c in enumerate(CATEGORY_ORDER):
        s = w[w.category_true == c]
        if s.empty:
            continue
        kw = dict(ms=5.5, mec=viz.SURFACE, mew=0.6, color=viz.SLOTS[i % len(viz.SLOTS)] if i < 8 else viz.INK2,
                  alpha=0.9, label=cat.CATEGORY_LABELS[c])
        axes[0].plot(s.DI_mm, 100 * s.hard_AC25, "o", **kw)
        axes[1].plot(s.DI_mm, s.dUTS_AC25, "o", **kw)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"Ideal critical diameter $D_I$ (mm; ASTM A255 factors, grain size 7)")
    axes[0].set_ylabel("Bainite + martensite in a 25 mm air-cooled bar (%)")
    axes[1].set_ylabel("Gain in UTS on air cooling a 25 mm bar (MPa)")
    axes[0].set_title(f"Hardenability decides what air cooling forms (Spearman {stats['spearman_DI_vs_hard_phase_AC25']:.2f})")
    axes[1].set_title(f"and how much strength it adds (Spearman {stats['spearman_DI_vs_dUTS_AC25']:.2f})")
    axes[1].legend(loc="upper left", fontsize=8, ncol=1)
    fig.tight_layout()
    return viz.savefig(fig, "fig25_hardenability_vs_air_cooling")


# ---------------------------------------------------------------------------
# Jominy end-quench: H-bands
# ---------------------------------------------------------------------------

H_BANDS = config.RAW_DIR / "jominy_h_bands.csv"
J_RATES = config.RAW_DIR / "jominy_cooling_rates.csv"
JOMINY_G = 7.0      # fine-grained steels; the ASTM A255 D_I basis


def jominy_austenitising_T(C):
    """Jominy test temperature by carbon content (ASTM A255 / SAE J406 practice)."""
    C = np.asarray(C, float)
    return np.where(C <= 0.25, 925.0, np.where(C <= 0.36, 870.0, 845.0))


def calibrate_quench(rates):
    """Effective heat-transfer coefficient of the water jet, set so the computed
    cooling rates at 700 C match the standard Jominy table (log-rate error)."""
    from scipy.optimize import minimize_scalar
    J = rates.J.to_numpy(float)
    meas = rates.cooling_rate_C_per_s_at_700C.to_numpy(float)

    def err(log_h):
        r = jominy.rate_at_700(jominy.jominy_paths(870.0, J, h_q=10 ** log_h))
        return float(np.mean(np.log(r / meas) ** 2))
    res = minimize_scalar(err, bounds=(3.5, 5.5), method="bounded", options={"xatol": 0.01})
    h = 10 ** res.x
    r = jominy.rate_at_700(jominy.jominy_paths(870.0, J, h_q=h))
    table = rates.assign(model_rate=r, ratio=r / meas)
    return h, table


def h_band_frame(catalog):
    """Band points joined to the nominal, leanest and richest compositions."""
    b = pd.read_csv(H_BANDS)
    b["grade"] = "AISI " + b.grade.astype(str).str.upper().str.replace("SAE", "").str.replace("AISI", "").str.strip()
    b = b.dropna(subset=["HRC_max", "HRC_min"])
    b = b.rename(columns={"J_sixteenths": "J"})
    cat_h = catalog.set_index("grade")
    b = b[b.grade.isin(cat_h.index)].copy()
    comps = {}
    for kind in ("nom", "min", "max"):
        cols = {}
        for el in config.ELEMENTS:
            col = f"{el}_{kind}"
            nom = cat_h[f"{el}_nom"]
            v = cat_h[col] if col in cat_h else nom
            cols[el] = v.fillna(nom).fillna(config.RESIDUAL_TYPICAL.get(el, 0.0))
        comps[kind] = pd.DataFrame(cols).loc[sorted(b.grade.unique())]
    boron = comps["nom"].index.str.contains(r"\dB\d", regex=True)
    return b, comps, pd.Series(boron, index=comps["nom"].index)


def jominy_metrics(pts, pred_col):
    r = pts[pred_col] - pts.mid
    inside = (pts[pred_col] >= pts.HRC_min - 1e-9) & (pts[pred_col] <= pts.HRC_max + 1e-9)
    rows = []
    for name, sel in (("all distances", pts.J > 0), ("J1-J4 (quenched end)", pts.J <= 4),
                      ("J5-J16", (pts.J > 4) & (pts.J <= 16)), ("J18-J32 (air-cooling range)", pts.J > 16)):
        rr = r[sel]
        rows.append({"distances": name, "n": int(sel.sum()), "RMSE_HRC": float(np.sqrt(np.mean(rr ** 2))),
                     "bias_HRC": float(rr.mean()), "inside_band": float(inside[sel].mean())})
    return pd.DataFrame(rows)


PUBLISHED = np.array([0.0, 0.0, 0.0, 0.0, np.log(3.0)])


def theta_in_config():
    """The kinetic factors currently set in config, as a theta vector."""
    return np.array([np.log(getattr(config, "FERRITE_RATE_FACTOR", 1.0)),
                     np.log(getattr(config, "PEARLITE_RATE_FACTOR", 1.0)),
                     np.log(getattr(config, "BAINITE_RATE_FACTOR", 1.0)),
                     getattr(config, "BAINITE_START_SHIFT_C", 0.0), np.log(3.0)])


def transfer_checks(theta):
    """Independent tests of a set of kinetic factors: data that the Jominy
    calibration does not see."""
    import s03_transformation_model as s03
    import s04_property_models as s04
    with kcal.kinetic_params(theta):
        _, summ = s03.validate_kinetics()
        _, fs = s03.literature_fractions()
        cal = s03.calibrate()
        Fx = s04.collins_frame()
        ca = s03.cct_atlas_summary(s03.cct_atlas_predictions()).set_index("subset")
    k = summ[summ.model == "this work"].set_index("start")
    fk = fs[fs.estimate == "kinetic"].set_index("subset")
    norm = cal[cal.condition == "normalized"].set_index("value")
    sel = (Fx.cooling_rate <= 5).to_numpy()
    return {"ferrite_start_MAE_C": k.loc["ferrite", "MAE_C"], "pearlite_start_MAE_C": k.loc["pearlite", "MAE_C"],
            "bainite_start_bias_C": k.loc["bainite", "bias_C"], "martensite_start_MAE_C": k.loc["martensite", "MAE_C"],
            "dilatometry_HB_RMSE_le_5Ks": float(np.sqrt(np.mean((Fx.HB_phys[sel] - Fx.HB_meas[sel]) ** 2))),
            "pearlite_fraction_MAE_furnace": fk.loc["furnace cooled", "MAE_pct_points"],
            "pearlite_fraction_MAE_air": fk.loc["air cooled", "MAE_pct_points"],
            "handbook_HB_RMSE_best_bar": float(norm.HB_RMSE.min()),
            "handbook_best_bar_mm": float(norm.HB_RMSE.idxmin()),
            "handbook_HB_RMSE_25mm": float(norm.loc[25.0, "HB_RMSE"]),
            "cct_atlas_MAE_FP": ca.loc["all points", "MAE_ferrite_pearlite"],
            "cct_atlas_MAE_bainite": ca.loc["all points", "MAE_bainite"],
            "cct_atlas_HV_RMSE": ca.loc["all points", "HV_RMSE"]}


def jominy_study(catalog):
    """Jominy simulation of the H-steels against their bands; Bayesian
    calibration of the kinetic factors on the non-boron grades (Jominy only,
    and jointly with the dilatometry start temperatures), grade-grouped
    cross-validation, and the transfer of each calibration to independent data."""
    out = {}
    h_q = 2.0e4
    if J_RATES.exists():
        h_q, rt = calibrate_quench(pd.read_csv(J_RATES))
        out["rates"] = rt
        print(f"water-jet heat-transfer coefficient {h_q:.0f} W/m2K; model/standard rate:", rt.ratio.round(2).tolist())
    out["h_q"] = h_q
    bands, comps, boron = h_band_frame(catalog)
    T_aust = pd.Series(jominy_austenitising_T(comps["nom"].C), index=comps["nom"].index)
    pts = bands[["grade", "J", "HRC_min", "HRC_max"]].copy()
    nb_idx = comps["nom"].index[~boron.to_numpy()]
    D = kcal.JominyData(pts, comps["nom"], T_aust, G=JOMINY_G, h_q=h_q)
    Dn = kcal.JominyData(pts[pts.grade.isin(nb_idx)], comps["nom"], T_aust, G=JOMINY_G, h_q=h_q)
    # calibrations
    th_j, _ = kcal.fit_map_multistart(lambda th: -(kcal.log_likelihood(th, Dn) + kcal.log_prior(th)), kcal.STARTS)
    sd_j = np.sqrt(np.diag(kcal.laplace_cov(Dn, th_j)))
    dil = kcal.DilatometryData()
    th_d, _ = kcal.fit_map_multistart(lambda th: -(kcal.log_likelihood(th, Dn) + dil.log_likelihood(th)
                                                     + kcal.log_prior(th)), kcal.STARTS)
    thetas = {"published Li et al. constants": PUBLISHED, "calibrated on the H-bands": th_j,
              "calibrated on the H-bands + dilatometry starts": th_d}
    post = pd.DataFrame({"parameter": kcal.PARAM_NAMES, "H-bands MAP": th_j, "H-bands Laplace sd": sd_j,
                         "H-bands + dilatometry MAP": th_d})
    post["H-bands factor"] = np.where(post.parameter.str.startswith("ln_"), np.exp(th_j), th_j)
    post["joint factor"] = np.where(post.parameter.str.startswith("ln_"), np.exp(th_d), th_d)
    out["posterior"] = post
    # predictions on every band point
    res = D.points.copy()
    res["mid"] = D.mid
    res["boron"] = res.grade.map(boron).to_numpy()
    cols = {"published Li et al. constants": "HRC_published", "calibrated on the H-bands": "HRC_jominy_cal",
            "calibrated on the H-bands + dilatometry starts": "HRC_joint_cal"}
    for name, th in thetas.items():
        res[cols[name]], _ = D.predict(th)
    rows = []
    for name, col in cols.items():
        for sub, sel in (("non-boron grades", ~res.boron), ("boron grades", res.boron)):
            rows.append(jominy_metrics(res[sel], col).assign(kinetics=name, grades=sub))
    out["metrics"] = pd.concat(rows, ignore_index=True)
    out["points"] = res
    # grade-grouped 5-fold cross-validation of the H-band calibration
    gr = np.array(sorted(Dn.points.grade.unique()))
    folds = np.array_split(np.random.default_rng(config.RANDOM_SEED).permutation(gr), 5)
    pred_0, _ = Dn.predict(PUBLISHED)
    cv = []
    for k, test in enumerate(folds):
        tr_idx = np.where(~Dn.points.grade.isin(test).to_numpy())[0]
        th_k, _ = kcal.fit_map(Dn, idx=tr_idx, x0=th_j, maxiter=300)
        pred_k, _ = Dn.predict(th_k)
        for i in np.where(Dn.points.grade.isin(test).to_numpy())[0]:
            cv.append({"fold": k, "grade": Dn.points.grade.iat[i], "J": Dn.points.J.iat[i], "mid": Dn.mid[i],
                       "HRC_min": Dn.points.HRC_min.iat[i], "HRC_max": Dn.points.HRC_max.iat[i],
                       "HRC_published": pred_0[i], "HRC_calibrated_cv": pred_k[i]})
    out["cv"] = pd.DataFrame(cv)
    # transfer to independent data
    tr = []
    for name, th in thetas.items():
        rec = {"kinetics": name}
        r = res[~res.boron]
        rec["H-band_RMSE_HRC_non_boron"] = float(np.sqrt(np.mean((r[cols[name]] - r.mid) ** 2)))
        rec.update(transfer_checks(th))
        tr.append(rec)
    out["transfer"] = pd.DataFrame(tr)
    return out


def fig_jominy(points, h_q):
    grades = ["AISI 1045H", "AISI 1340H", "AISI 5140H", "AISI 4140H", "AISI 4340H", "AISI 8620H", "AISI 4620H",
              "AISI 6150H", "AISI 9310H"]
    grades = [g for g in grades if g in set(points.grade)]
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.2), sharex=True, sharey=True)
    for ax, g in zip(axes.ravel(), grades):
        d = points[points.grade == g].sort_values("J")
        x = d.J * jominy.SIXTEENTH_MM
        ax.fill_between(x, d.HRC_min, d.HRC_max, color=viz.GRID, alpha=0.9, lw=0, label="SAE H-band")
        ax.plot(x, d.HRC_published, "-", color=viz.ORANGE, lw=2, label="published Li kinetics")
        ax.plot(x, d.HRC_jominy_cal, "--", color=viz.BLUE, lw=1.6, label="calibrated on the H-bands")
        ax.set_title(g.replace("AISI ", ""))
        ax.set_ylim(15, 68)
    for ax in axes[-1]:
        ax.set_xlabel("Distance from the quenched end (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("HRC")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Jominy end-quench: model against the SAE hardenability bands (1-D conduction, water jet "
                 f"{h_q:.0f} W/m²K)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return viz.savefig(fig, "fig29_jominy_hbands")


def main():
    config.ensure_dirs()
    w = di_table()
    w.to_csv(config.RESULTS_DIR / "s08_ideal_diameter_catalogue.csv", index=False)
    summ, stats = di_summary(w)
    summ.to_csv(config.RESULTS_DIR / "s08_ideal_diameter_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))
    print({k: round(v, 3) for k, v in stats.items()})
    fig_hardenability(w, stats)
    if H_BANDS.exists():
        catalog = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv")
        js = jominy_study(catalog)
        if "rates" in js:
            js["rates"].assign(h_q_W_m2K=js["h_q"]).to_csv(config.RESULTS_DIR / "s08_jominy_cooling_rates.csv", index=False)
        js["metrics"].to_csv(config.RESULTS_DIR / "s08_jominy_metrics.csv", index=False)
        js["points"].to_csv(config.RESULTS_DIR / "s08_jominy_predictions.csv", index=False)
        js["posterior"].to_csv(config.RESULTS_DIR / "s08_kinetic_calibration_posterior.csv", index=False)
        js["cv"].to_csv(config.RESULTS_DIR / "s08_kinetic_calibration_cv.csv", index=False)
        js["transfer"].to_csv(config.RESULTS_DIR / "s08_kinetic_calibration_transfer.csv", index=False)
        print(js["metrics"].round(2).to_string(index=False))
        print(js["posterior"].round(3).to_string(index=False))
        cvm = pd.concat([jominy_metrics(js["cv"], c).assign(kinetics=c)
                         for c in ("HRC_published", "HRC_calibrated_cv")])
        cvm.to_csv(config.RESULTS_DIR / "s08_kinetic_calibration_cv_metrics.csv", index=False)
        print(cvm.round(2).to_string(index=False))
        print(js["transfer"].round(2).T.to_string())
        fig_jominy(js["points"], js["h_q"])


if __name__ == "__main__":
    main()
