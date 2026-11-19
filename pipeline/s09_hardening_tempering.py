"""Stage 9 - hardening, tempering and stress relieving: the full heat-treatment route.

1. Quench heat flow (physics.quench): centre / mid-radius / surface cooling of
   round bars in oil and water (Grossmann H) and in an ideal quench.
2. Hardenability check: the bar diameter whose centre reaches 50 % martensite
   in an ideal quench, from the heat flow + kinetics, against Grossmann's
   D_I from the ASTM A255 factors (independent of the kinetics), for both
   the published and the recalibrated kinetic constants.
3. Quenched-and-tempered bars of Birbilis et al. (2025): 324 bars, 28
   grades, oil or water quenched, tempered at 150-705 C, 13-100 mm.
   * tempering constants fitted on all bars, with a grade bootstrap;
   * model ladder with leave-one-grade-out (physics only -> data only ->
     grey box -> physics + GP), for UTS, YS and elongation;
   * the same chain with the published kinetic constants: an independent test
     of the Jominy recalibration of stage 8 on data it never saw;
   * normalised split-conformal 90 % intervals, with nested coverage;
   * mass effect (13-100 mm) and tempering curves.
4. Production models for quenched and quenched-and-tempered bars
   (results/models/qt_models.joblib).
5. The five classic treatments side by side for representative grades
   (anneal, normalise, harden, harden + temper, stress relieve), stress
   relief and precipitation-strengthening tables.

Outputs: results/s09_*.csv, figures fig31-fig37.
"""

import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import config, viz  # noqa: E402
from steelml.features import featurize_quench  # noqa: E402
from steelml.models import qt_models as qm  # noqa: E402
from steelml.physics import calibration as kcal  # noqa: E402
from steelml.physics import cooling, precipitation, quench, stress_relief, tempering  # noqa: E402
from steelml.physics.hardenability import ideal_critical_diameter, model_critical_diameter  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt
RES = config.RESULTS_DIR
PUBLISHED = np.array([0.0, 0.0, 0.0, 0.0, 0.0])   # ln factors 0, Bs shift 0: Li et al. (1998) as published
REP_GRADES = {"1020": "AISI 1020", "1045": "AISI 1045", "1080": "AISI 1080", "4140": "AISI 4140H",
              "4340": "AISI 4340H", "5160": "AISI 5160H"}


def catalogue_comp(name):
    cat = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv").set_index("grade")
    r = cat.loc[name]
    return {el: float(r[f"{el}_nom"]) for el in config.ELEMENTS if f"{el}_nom" in r and pd.notna(r[f"{el}_nom"])}


# ---------------------------------------------------------------------------
# 1-2 quench heat flow and hardenability
# ---------------------------------------------------------------------------

def quench_rate_table(T0=850.0):
    rows = []
    for D in (13, 25, 50, 100):
        for med in ("oil", "water", "ideal"):
            H = np.inf if med == "ideal" else quench.severity(med, "mild")["H_nom"]
            ps = quench.quench_paths(T0, D, H=H, T_q=quench.T_QUENCHANT.get(med, 25.0),
                                     positions=(0.0, 0.5, 0.75))
            for pos, p in zip((0.0, 0.5, 0.75), ps):
                rows.append({"D_mm": D, "medium": med, "H_per_in": H, "r_over_R": pos,
                             "rate700_K_s": cooling.rate_at(*p, 700.0), "t85_s": cooling.t85(*p)})
    return pd.DataFrame(rows)


def hardenability_check(qt):
    comps = qt.drop_duplicates("grade").set_index("grade")[config.ELEMENTS]
    rows = []
    for g, c in comps.iterrows():
        d = c.to_dict()
        di_a255 = float(np.ravel(ideal_critical_diameter(d)[1])[0])
        d_cal = model_critical_diameter(d)
        with kcal.kinetic_params(PUBLISHED):
            d_pub = model_critical_diameter(d)
        rows.append({"grade": g, "C": d["C"], "Mn": d["Mn"], "Cr": d["Cr"], "Ni": d["Ni"], "Mo": d["Mo"],
                     "DI_A255_mm": di_a255, "DI_model_calibrated_mm": d_cal, "DI_model_published_mm": d_pub})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3 quenched-and-tempered bars
# ---------------------------------------------------------------------------

def _fold(F, name, target, test_grade, theta=None):
    tr = F.grade != test_grade
    m = qm.make_qt_model(name, target, theta).fit(F[tr], F.loc[tr, target].to_numpy(float))
    mu, sd = m.predict(F[~tr])
    return F.index[~tr], mu, sd


def logo(F, name, target, n_jobs=4):
    res = Parallel(n_jobs=n_jobs)(delayed(_fold)(F, name, target, g) for g in F.grade.unique())
    mu = pd.Series(np.nan, index=F.index)
    sd = pd.Series(np.nan, index=F.index)
    for idx, m, s in res:
        mu.loc[idx], sd.loc[idx] = m, s
    return mu.to_numpy(), sd.to_numpy()


def metrics(y, p):
    e = p - y
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


def conformal(y, mu, sd, groups, alpha=0.1):
    """Normalised split-conformal quantile on all out-of-fold residuals, and its
    nested coverage (the quantile recomputed without each grade, applied to it)."""
    s = np.abs(y - mu) / np.maximum(sd, 1e-9)
    n = len(s)
    q = float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
    cov = []
    for g in np.unique(groups):
        te = groups == g
        sn = s[~te]
        qn = np.quantile(sn, min(1.0, np.ceil((len(sn) + 1) * (1 - alpha)) / len(sn)))
        cov.append(s[te] <= qn)
    cov = np.concatenate(cov)
    return q, float(np.mean(cov)), float(np.median(2 * q * sd))


def bootstrap_theta(F, n=30, seed=0):
    rng = np.random.default_rng(seed)
    grades = F.grade.unique()
    out = []
    for _ in range(n):
        pick = rng.choice(grades, len(grades), replace=True)
        B = pd.concat([F[F.grade == g] for g in pick], ignore_index=True)
        out.append(tempering.fit(B, B.UTS_MPa))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_quench(rt, hd):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for med, col in (("oil", viz.ORANGE), ("water", viz.BLUE), ("ideal", viz.INK2)):
        for pos, ls in ((0.0, "-"), (0.75, "--")):
            s = rt[(rt.medium == med) & (rt.r_over_R == pos)]
            ax.plot(s.D_mm, s.rate700_K_s, ls, color=col, marker="o", ms=4,
                    label=f"{med}, {'centre' if pos == 0 else '3/4 radius'}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Bar diameter (mm)"); ax.set_ylabel("Cooling rate at 700 °C (K/s)")
    ax.set_title("Quench heat flow: Grossmann H, radial conduction", loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax = axes[1]
    lim = [10, 450]
    ax.plot(lim, lim, color=viz.GRID, lw=1)
    ax.scatter(hd.DI_A255_mm, hd.DI_model_published_mm, s=22, color=viz.NEUTRAL, label="published Li constants")
    ax.scatter(hd.DI_A255_mm, hd.DI_model_calibrated_mm, s=22, color=viz.BLUE, label="recalibrated (stage 8)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("D_I from ASTM A255 factors (mm)"); ax.set_ylabel("D_I from heat flow + kinetics (mm)")
    ax.set_title("Hardenability: model vs Grossmann", loc="left")
    ax.legend(fontsize=8, frameon=False)
    return viz.savefig(fig, "fig31_quench_and_hardenability")


def fig_ladder(bench):
    order = qm.QT_MODEL_ORDER
    labels = {"qt_physics": "Textbook physics", "qt_gbm": "Gradient boosting", "qt_gp": "Gaussian process",
              "qt_greybox": "Fitted physics (grey box)", "qt_anchored": "Physics + GP (production)",
              "qt_ensemble": "Physics + GP / GP mixture"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, t in zip(axes, qm.QT_TARGETS):
        s = bench[bench.target == t].set_index("model").reindex(order)
        cols = [viz.BLUE if m == qm.QT_PRODUCTION else viz.NEUTRAL for m in order]
        ax.barh(range(len(order)), s.RMSE, color=cols)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels[m] for m in order] if t == "UTS_MPa" else [""] * len(order), fontsize=9)
        ax.invert_yaxis()
        for i, v in enumerate(s.RMSE):
            ax.text(v, i, f" {v:.0f}" if t != "El_pct" else f" {v:.1f}", va="center", fontsize=8, color=viz.INK2)
        ax.set_title({"UTS_MPa": "UTS (MPa)", "YS_MPa": "Yield strength (MPa)", "El_pct": "Elongation (%)"}[t],
                     loc="left")
        ax.set_xlabel("RMSE, leave one grade out")
    return viz.savefig(fig, "fig32_qt_model_ladder")


def fig_parity(F, oof):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, t in zip(axes, qm.QT_TARGETS):
        y, p = F[t].to_numpy(float), oof[f"{t}_pred"].to_numpy(float)
        lo, hi = oof[f"{t}_lo90"].to_numpy(float), oof[f"{t}_hi90"].to_numpy(float)
        lim = [min(y.min(), p.min()) * 0.9, max(y.max(), p.max()) * 1.05]
        ax.plot(lim, lim, color=viz.GRID, lw=1)
        ax.errorbar(y, p, yerr=[p - lo, hi - p], fmt="none", ecolor=viz.GRID, lw=0.6)
        sc = ax.scatter(y, p, c=F.section_mm, s=12, cmap="viridis", norm=viz.matplotlib.colors.LogNorm(10, 110))
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("measured"); ax.set_ylabel("predicted (grade held out)")
        ax.set_title({"UTS_MPa": "UTS (MPa)", "YS_MPa": "Yield strength (MPa)", "El_pct": "Elongation (%)"}[t],
                     loc="left")
    cb = fig.colorbar(sc, ax=axes, shrink=0.8)
    cb.set_label("bar diameter (mm)")
    return viz.savefig(fig, "fig33_qt_parity")


def fig_mass_effect(me):
    grades = [g for g in ("1050", "4140", "4340", "5160") if g in set(me.grade)]
    fig, axes = plt.subplots(1, len(grades), figsize=(3.2 * len(grades), 3.8), sharey=False)
    for ax, g in zip(np.atleast_1d(axes), grades):
        s = me[me.grade == g]
        for (med, T), q in s.groupby(["quench_medium", "temper_T_C"]):
            col = {480: viz.VIOLET, 540: viz.BLUE, 595: viz.AQUA, 650: viz.ORANGE}.get(int(T), viz.NEUTRAL)
            q = q.sort_values("section_mm")
            ax.plot(q.section_mm, q.UTS_MPa_pred, "-", color=col, lw=1.5)
            ax.fill_between(q.section_mm, q.UTS_MPa_lo90, q.UTS_MPa_hi90, color=col, alpha=0.12, lw=0)
            ax.plot(q.section_mm, q.UTS_MPa, "o", color=col, ms=4, label=f"{med}, {T:.0f} °C")
        ax.set_xscale("log"); ax.set_xticks([13, 25, 50, 100]); ax.set_xticklabels(["13", "25", "50", "100"])
        ax.xaxis.set_minor_formatter(viz.matplotlib.ticker.NullFormatter())
        ax.set_title(f"AISI {g}", loc="left"); ax.set_xlabel("bar diameter (mm)")
        ax.legend(fontsize=7, frameon=False)
    np.atleast_1d(axes)[0].set_ylabel("UTS (MPa): measured (dots), predicted, grade held out")
    return viz.savefig(fig, "fig34_qt_mass_effect")


def fig_tempering(tc, qtF):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for (g, col) in zip(tc.grade.unique(), viz.SLOTS):
        s = tc[tc.grade == g]
        axes[0].plot(s.temper_T_C, s.UTS_MPa_pred, color=col, lw=1.6, label=f"{g} ({s.quench_medium.iloc[0]})")
        axes[0].fill_between(s.temper_T_C, s.UTS_MPa_lo90, s.UTS_MPa_hi90, color=col, alpha=0.12, lw=0)
        m = qtF[(qtF.grade == g) & (qtF.section_mm == 25) & (qtF.quench_medium == s.quench_medium.iloc[0])]
        axes[0].plot(m.temper_T_C, m.UTS_MPa, "o", color=col, ms=4)
        axes[1].plot(s.temper_T_C, s.El_pct_pred, color=col, lw=1.6)
        axes[1].plot(m.temper_T_C, m.El_pct, "o", color=col, ms=4)
    axes[0].set_ylabel("UTS (MPa)"); axes[1].set_ylabel("Elongation (%)")
    for ax in axes:
        ax.set_xlabel("tempering temperature (°C), 1 h")
    axes[0].set_title("Tempering curves, 25 mm bars (dots: measured)", loc="left")
    axes[0].legend(fontsize=8, frameon=False)
    for T, lab in ((325, "TME"), (475, "TE")):
        axes[1].axvspan(T - 75 if lab == "TME" else 375, T + 75 if lab == "TME" else 575, color=viz.GRID, alpha=0.35)
        axes[1].text(T, axes[1].get_ylim()[1] * 0.97, lab, ha="center", fontsize=8, color=viz.INK2)
    return viz.savefig(fig, "fig35_tempering_curves")


def fig_five(ft):
    """The five treatments of each grade: hardness, strength, constituents."""
    grades = list(dict.fromkeys(ft.grade))
    tr_order = ["anneal", "normalise", "harden", "quench_temper", "stress_relieve"]
    tr_lab = ["anneal", "normalise", "harden", "Q + T", "stress rel."]
    cols = [viz.BLUE, viz.ORANGE, viz.YELLOW, viz.AQUA, viz.NEUTRAL]
    fig, axes = plt.subplots(3, len(grades), figsize=(2.55 * len(grades), 8.2), sharey="row")
    x = np.arange(len(tr_order))
    for j, g in enumerate(grades):
        s = ft[ft.grade == g].set_index("treatment").reindex(tr_order)
        ax = axes[0, j]
        ax.bar(x, s.HV, color=cols)
        ax.set_title(f"AISI {g}", loc="left"); ax.set_xticks(x); ax.set_xticklabels([])
        ax = axes[1, j]
        uts = s.UTS_MPa_pred.to_numpy(float)
        ax.bar(x, uts, color=cols)
        if "UTS_equiv_MPa" in s and np.isfinite(s.UTS_equiv_MPa.get("harden", np.nan)):
            ax.bar([2], [s.UTS_equiv_MPa["harden"]], color="none", edgecolor=viz.YELLOW, hatch="///", lw=1.0)
        err = np.vstack([uts - s.UTS_MPa_lo90.to_numpy(float), s.UTS_MPa_hi90.to_numpy(float) - uts])
        ax.errorbar(x, uts, yerr=np.nan_to_num(err), fmt="none", ecolor=viz.INK2, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([])
        ax = axes[2, j]
        bottom = np.zeros(len(x))
        for col, key in (("ferrite", "F"), ("pearlite", "P"), ("bainite", "B"), ("martensite", "M")):
            v = s[key].to_numpy(float)
            ax.bar(x, v, bottom=bottom, color=viz.PHASE_COLORS[col], label=col if j == 0 else None)
            bottom += v
        ax.set_xticks(x); ax.set_xticklabels(tr_lab, fontsize=7, rotation=45, ha="right")
    axes[0, 0].set_ylabel("hardness (HV)")
    axes[1, 0].set_ylabel("UTS (MPa), 90 % interval\n(hatched: as-quenched, indicative)")
    axes[2, 0].set_ylabel("constituent fraction")
    axes[2, 0].legend(fontsize=7, frameon=False, loc="upper left")
    return viz.savefig(fig, "fig36_five_heat_treatments")


def fig_sr_ppt(sr, ppt):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    for (lab, s), col in zip(sr.groupby("state"), viz.SLOTS):
        ax.plot(s.T_C, s.residual_stress_bound_MPa, color=col, lw=1.6, label=lab)
    ax.set_xlabel("stress-relief temperature (°C)"); ax.set_ylabel("upper bound of residual stress (MPa)")
    ax.set_title("Stress relief: residual stress <= k_y(T) x YS", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    for (route, el), s in ppt.groupby(["route", "element"]):
        ax.plot(s.wt_pct, s.dsigma_MPa, lw=1.6, ls="-" if route == "AC" else "--",
                color=viz.BLUE if el == "V" else viz.ORANGE, label=f"{el}, {'air cooled' if route == 'AC' else 'furnace cooled'}")
    ax.set_xlabel("microalloy content (wt %)"); ax.set_ylabel("precipitation strengthening (MPa)")
    ax.set_title("Microalloy precipitation (Ashby-Orowan estimate)", loc="left")
    ax.legend(fontsize=8, frameon=False)
    return viz.savefig(fig, "fig37_stress_relief_and_precipitation")


# ---------------------------------------------------------------------------

def main():
    viz.apply_style()
    t0 = time.time()
    qt = pd.read_csv(config.PROCESSED_DIR / "qt_bars.csv")
    summ = (qt.groupby(["quench_medium", "section_mm"]).size().rename("bars").reset_index())
    summ.to_csv(RES / "s09_qt_dataset_summary.csv", index=False)
    print(f"QT bars: {len(qt)} rows, {qt.grade.nunique()} grades")

    rt = quench_rate_table()
    rt.to_csv(RES / "s09_quench_rates.csv", index=False)
    hd = hardenability_check(qt)
    hd.to_csv(RES / "s09_hardenability_check.csv", index=False)
    fig_quench(rt, hd)
    print(f"  quench + hardenability: {time.time() - t0:.0f} s")

    F = featurize_quench(qt)
    theta = tempering.fit(F, F.UTS_MPa)
    boot = bootstrap_theta(F)
    tc_rows = [{"constant": k, "prior_mean": tempering.PRIOR[k][0], "prior_sd": tempering.PRIOR[k][1],
                "fitted": theta[k], "bootstrap_sd": float(boot[k].std()),
                "at_bound": bool(np.isclose(theta[k], tempering.PRIOR[k][2]) or np.isclose(theta[k], tempering.PRIOR[k][3]))}
               for k in tempering.PARAMS]
    pd.DataFrame(tc_rows).to_csv(RES / "s09_tempering_constants.csv", index=False)
    print(f"  tempering constants: {time.time() - t0:.0f} s")

    # ---- ladder
    bench, oof = [], pd.DataFrame(index=F.index)
    for t in qm.QT_TARGETS:
        y = F[t].to_numpy(float)
        for name in qm.QT_MODEL_ORDER:
            mu, sd = logo(F, name, t)
            bench.append({"model": name, "target": t, **metrics(y, mu)})
            if name == qm.QT_PRODUCTION:
                q, cov, width = conformal(y, mu, sd, F.grade.to_numpy())
                oof[f"{t}_pred"], oof[f"{t}_sd"] = mu, sd
                oof[f"{t}_lo90"], oof[f"{t}_hi90"] = mu - q * sd, mu + q * sd
                bench[-1].update(q90=q, coverage_nested=cov, median_width=width)
        print(f"  ladder {t}: {time.time() - t0:.0f} s")
    bench = pd.DataFrame(bench)
    bench.to_csv(RES / "s09_qt_benchmark.csv", index=False)
    out = pd.concat([F[["sample_id", "grade", "quench_medium", "treat_T_C", "temper_T_C", "section_mm",
                        "f_M", "f_B", "f_FP", "cool_rate_700"] + qm.QT_TARGETS], oof], axis=1)
    out.to_csv(RES / "s09_qt_oof.csv", index=False)
    conf = bench[bench.model == qm.QT_PRODUCTION][["target", "q90", "coverage_nested", "median_width"]]
    conf.to_csv(RES / "s09_qt_conformal.csv", index=False)
    fig_ladder(bench)
    fig_parity(F, oof)

    # ---- kinetics: an independent test of the stage-8 recalibration
    kc = []
    with kcal.kinetic_params(PUBLISHED):
        Fp = featurize_quench(qt)
    for label, FF in (("recalibrated (production)", F), ("published Li et al.", Fp)):
        for name in ("qt_greybox", "qt_anchored"):
            mu, _ = logo(FF, name, "UTS_MPa")
            e = mu - FF.UTS_MPa.to_numpy()
            row = {"kinetics": label, "model": name, "UTS_RMSE": float(np.sqrt(np.mean(e ** 2))),
                   "mean_f_martensite": float(FF.f_M.mean()), "mean_f_bainite": float(FF.f_B.mean())}
            for D in (13, 25, 50, 100):
                row[f"bias_{D}mm"] = float(np.mean(e[FF.section_mm.round() == D]))
            kc.append(row)
    pd.DataFrame(kc).to_csv(RES / "s09_qt_kinetics_check.csv", index=False)
    print(f"  kinetics check: {time.time() - t0:.0f} s")

    # ---- mass effect table (out-of-fold, production)
    me = pd.concat([F[["grade", "quench_medium", "temper_T_C", "section_mm", "UTS_MPa"]],
                    oof[["UTS_MPa_pred", "UTS_MPa_lo90", "UTS_MPa_hi90"]]], axis=1)
    me.to_csv(RES / "s09_mass_effect.csv", index=False)
    fig_mass_effect(me)

    # ---- production bundle
    q90 = dict(zip(conf.target, conf.q90))
    models = {t: qm.make_qt_model(qm.QT_PRODUCTION, t, theta).fit(F, F[t].to_numpy(float)) for t in qm.QT_TARGETS}
    bundle = qm.QTBundle(models, q90, theta, meta={"n_bars": len(F), "grades": sorted(F.grade.unique()),
                                                   "kinetics": "production (config)"})
    joblib.dump(bundle, config.MODELS_DIR / "qt_models.joblib")

    # ---- tempering curves for representative grades (25 mm)
    tc = []
    for g, med in (("1050", "water"), ("4140", "oil"), ("4340", "oil"), ("5160", "oil")):
        base = qt[qt.grade == g].iloc[0]
        d = pd.DataFrame([{**{el: base[el] for el in config.ELEMENTS}, "quench_medium": med, "section_mm": 25.0,
                           "treat_T_C": base.treat_T_C, "temper_T_C": T, "temper_time_h": 1.0}
                          for T in np.arange(150, 706, 15)])
        o = bundle.predict(d)
        o["grade"] = g
        tc.append(o[["grade", "quench_medium", "temper_T_C", "UTS_MPa_pred", "UTS_MPa_lo90", "UTS_MPa_hi90",
                     "YS_MPa_pred", "El_pct_pred", "HRC_est", "TME_risk", "TE_risk"]])
    tc = pd.concat(tc, ignore_index=True)
    tc.to_csv(RES / "s09_tempering_curves.csv", index=False)
    fig_tempering(tc, F)

    # ---- the five treatments side by side
    from steelml.predictor import SteelPredictor
    sp = SteelPredictor.load()
    sp._qt_bundle = bundle
    ft = []
    for g, name in REP_GRADES.items():
        comp = catalogue_comp(name)
        r = sp.compare_treatments(comp, section_mm=25.0, temper_T=600.0, sr_T=600.0)
        r.insert(0, "grade", g)
        ft.append(r)
    ft = pd.concat(ft, ignore_index=True)
    ft.to_csv(RES / "s09_five_treatments.csv", index=False)
    fig_five(ft)

    # ---- stress relief and precipitation
    sr = []
    for (lab, ys) in (("normalised 1045 (YS %.0f MPa)" % ft[(ft.grade == "1045") & (ft.treatment == "normalise")].YS_MPa_pred.iloc[0],
                       ft[(ft.grade == "1045") & (ft.treatment == "normalise")].YS_MPa_pred.iloc[0]),
                      ("Q+T 4140 (YS %.0f MPa)" % ft[(ft.grade == "4140") & (ft.treatment == "quench_temper")].YS_MPa_pred.iloc[0],
                       ft[(ft.grade == "4140") & (ft.treatment == "quench_temper")].YS_MPa_pred.iloc[0])):
        for T in np.arange(300, 701, 10):
            sr.append({"state": lab, "T_C": T, "k_y": float(stress_relief.ky(T)),
                       "residual_stress_bound_MPa": float(stress_relief.residual_stress_bound(ys, T))})
    sr = pd.DataFrame(sr)
    sr.to_csv(RES / "s09_stress_relief.csv", index=False)
    # Hollomon-Jaffe additivity: 4140 tempered 600 C / 1 h, then stress relieved
    c4140 = catalogue_comp(REP_GRADES["4140"])
    add = []
    P1 = float(tempering.hollomon_jaffe(600.0, 1.0))
    for Tsr in (500, 550, 575, 600, 625, 650):
        P2 = float(stress_relief.tempering_after_relief(P1, Tsr, 1.0))
        r = sp.heat_treat(c4140, "stress_relieve", 25.0, "oil", 600.0, 1.0, Tsr, 1.0, sr_base="quench_temper").iloc[0]
        add.append({"grade": "4140", "temper": "600 C / 1 h", "sr_T_C": Tsr, "sr_h": 1.0, "P_before": P1,
                    "P_after": P2, "UTS_after_MPa": r.UTS_MPa_pred, "subcritical": r.subcritical,
                    "residual_stress_bound_MPa": r.residual_stress_bound_MPa})
    pd.DataFrame(add).to_csv(RES / "s09_stress_relief_after_temper.csv", index=False)
    ppt = []
    for el in ("V", "Nb"):
        for route in ("AC", "FC"):
            for x in np.linspace(0, 0.15 if el == "V" else 0.06, 31):
                comp = {"C": 0.18, "Mn": 1.3, "Si": 0.3, "N": 0.010, "V": 0.0, "Nb": 0.0}
                comp[el] = x
                ppt.append({"element": el, "route": route, "wt_pct": x, "T_aust_C": 920.0,
                            "dsigma_MPa": float(precipitation.strengthening(comp, 920.0, route))})
    ppt = pd.DataFrame(ppt)
    ppt.to_csv(RES / "s09_precipitation.csv", index=False)
    fig_sr_ppt(sr, ppt)
    print(f"stage 9 done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
