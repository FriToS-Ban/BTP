"""Stage 11 - how uncertain is the transformation model, and does that
uncertainty explain its errors on data it never saw?

1. Posterior sampling. Stage 8 found the maximum a posteriori point of the
   kinetic calibration on the SAE H-bands and summarised it with a Laplace
   (Gaussian) approximation. The likelihood is not smooth (a reaction either
   starts within a time step or it does not), so the numerical Hessian is
   unreliable. In particular, it gave posterior standard deviations larger
   than the prior's for the two factors the bands do not constrain. Here the
   posterior is sampled with four adaptive Metropolis chains (Haario et al.
   2001) started from overdispersed points, and checked with the split
   R-hat and the effective sample size (Gelman et al. 2013).
2. Propagation. 200 posterior draws of the kinetic constants are pushed
   through the kinetics to predictions: the 423 measured points of the CCT
   atlas, the dilatometry start temperatures, representative furnace / air /
   oil-quench states, and the quench-and-temper strength.
3. Independent validation. If the parameter uncertainty explained the
   errors, about 90 % of the measured points of the CCT atlas would fall in
   the 90 % credible band. The coverage measures how much of the error is
   parametric and how much is structural (equation form). A band that adds a
   structural error term, estimated on the atlas itself, is reported
   alongside.

Outputs: results/s11_*.csv, figures fig40-fig41.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "src"))

from steelml import config, viz  # noqa: E402
from steelml.physics import calibration as kcal  # noqa: E402
from steelml.physics import cooling, kinetics  # noqa: E402
from steelml.physics import properties as pp  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt
RES = config.RESULTS_DIR
NAMES = list(kcal.PARAM_NAMES)


def jominy_data():
    import s08_hardenability as s08
    catalog = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv")
    bands, comps, boron = s08.h_band_frame(catalog)
    T_aust = pd.Series(s08.jominy_austenitising_T(comps["nom"].C), index=comps["nom"].index)
    pts = bands[["grade", "J", "HRC_min", "HRC_max"]].copy()
    nb = comps["nom"].index[~boron.to_numpy()]
    h_q = 2.0e4
    if s08.J_RATES.exists():
        h_q, _ = s08.calibrate_quench(pd.read_csv(s08.J_RATES))
    return kcal.JominyData(pts[pts.grade.isin(nb)], comps["nom"], T_aust, G=s08.JOMINY_G, h_q=h_q)


def theta_map():
    return np.array([np.log(config.FERRITE_RATE_FACTOR), np.log(config.PEARLITE_RATE_FACTOR),
                     np.log(config.BAINITE_RATE_FACTOR), config.BAINITE_START_SHIFT_C, 1.49])


def adaptive_metropolis(logp, x0, n_iter, n_adapt, scale0, seed):
    """Haario-style adaptive Metropolis: the proposal covariance is the scaled
    empirical covariance of the chain, adapted during the burn-in only."""
    rng = np.random.default_rng(seed)
    d = len(x0)
    cov = np.diag(np.asarray(scale0, float) ** 2)
    x, lp = np.asarray(x0, float), logp(x0)
    chain, lps, acc = np.empty((n_iter, d)), np.empty(n_iter), 0
    for i in range(n_iter):
        if i >= 200 and i < n_adapt and i % 100 == 0:
            emp = np.cov(chain[max(0, i - 1000):i].T)
            cov = (2.38 ** 2 / d) * emp + 1e-8 * np.eye(d)
        prop = rng.multivariate_normal(x, cov)
        lq = logp(prop)
        if np.log(rng.uniform()) < lq - lp:
            x, lp = prop, lq
            if i >= n_adapt:
                acc += 1
        chain[i], lps[i] = x, lp
    return chain[n_adapt:], lps[n_adapt:], acc / max(n_iter - n_adapt, 1)


def split_rhat(chains):
    """Split R-hat for one parameter; chains: (m, n)."""
    m, n = chains.shape
    h = n // 2
    s = np.vstack([chains[:, :h], chains[:, h:2 * h]])
    W = s.var(axis=1, ddof=1).mean()
    B = h * s.mean(axis=1).var(ddof=1)
    var = (h - 1) / h * W + B / h
    return float(np.sqrt(var / max(W, 1e-300)))


def ess(chains):
    """Effective sample size from the chain autocorrelation (Geyer's initial
    positive sequence, pooled over chains)."""
    m, n = chains.shape
    x = chains - chains.mean(axis=1, keepdims=True)
    var = x.var()
    if var <= 0:
        return float(m * n)
    acf = np.array([np.mean(x[:, :n - k] * x[:, k:]) / var for k in range(min(n // 2, 500))])
    s = 0.0
    for k in range(1, len(acf) - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair < 0:
            break
        s += pair
    return float(m * n / (1.0 + 2.0 * s))


def cct_atlas_frame():
    from steelml.physics import microstructure as ms
    d = pd.read_csv(config.RAW_DIR / "cct_rose_hougardy_rijeka.csv")
    comps = pd.DataFrame({el: d[el] if el in d else config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS})
    paths = []
    for T0, t5 in zip(d.T_aust_C, d.t500_s):
        tau = t5 / np.log((T0 - 20.0) / (500.0 - 20.0))
        paths.append((cooling._time_on_grid(lambda T, tau=tau: (np.asarray(T, float) - 20.0) / tau, T0),
                      cooling.T_GRID.copy()))
    G = ms.austenite_astm(d.T_aust_C.to_numpy())
    rate = np.array([cooling.rate_at(t, T) for t, T in paths])
    return d, comps, G, paths, rate


def representative_states():
    """Representative cooling states whose prediction we want with a band."""
    from steelml.features import featurize_quench
    cat = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv").set_index("grade")
    rows = []
    for g, name in (("1045", "AISI 1045"), ("4140", "AISI 4140H"), ("4340", "AISI 4340H")):
        comp = {el: float(cat.loc[name, f"{el}_nom"]) for el in config.ELEMENTS if pd.notna(cat.loc[name, f"{el}_nom"])}
        rows.append((g, "furnace cooled", comp, cooling.furnace_cooling_path(870.0, config.FC_RATE_700C)))
        rows.append((g, "air cooled 25 mm", comp, cooling.air_cooling_path(870.0, 25.0)))
        q = featurize_quench(pd.DataFrame([{**comp, "quench_medium": "oil", "section_mm": 50.0, "pos_rR": 0.5}]))
        from steelml.physics import quench
        p = quench.quench_paths(float(q.treat_T_C.iloc[0]), 50.0, medium="oil", positions=(0.5,))[0]
        rows.append((g, "oil quenched 50 mm, mid-radius", comp, p))
    return rows


def fig_propagation(band, cov, rep):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.55})
    ax = axes[0]
    cv = cov.set_index("quantity")
    for key, col in (("f_FP", viz.BLUE), ("f_B", viz.AQUA), ("f_M", viz.YELLOW)):
        ax.errorbar(band[f"{key}_meas"], band[f"{key}_q50"],
                    yerr=[band[f"{key}_q50"] - band[f"{key}_q05"], band[f"{key}_q95"] - band[f"{key}_q50"]],
                    fmt="o", ms=2.5, color=col, ecolor=col, alpha=0.45, lw=0.6,
                    label=f"{key[2:]}: {cv.loc[key, 'coverage_parametric_90']:.0%} inside band")
    ax.plot([0, 1], [0, 1], color=viz.GRID, lw=1)
    ax.set_xlabel("measured fraction (CCT atlas, 19 steels)")
    ax.set_ylabel("predicted: posterior median and 90 % band")
    ax.set_title("Parameter uncertainty vs independent data", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axes[1]
    s = rep[rep.quantity.isin(["f_bainite", "f_martensite"])].reset_index(drop=True)
    short = {"furnace cooled": "FC", "air cooled 25 mm": "AC 25 mm", "oil quenched 50 mm, mid-radius": "oil 50 mm"}
    labs = [f"{g} {short.get(st, st)}: {'B' if q == 'f_bainite' else 'M'}" for g, st, q in zip(s.grade, s.state, s.quantity)]
    y = np.arange(len(s))
    ax.errorbar(s.q50, y, xerr=[s.q50 - s.q05, s.q95 - s.q50], fmt="o", color=viz.BLUE, ms=4)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("fraction: posterior median and 90 % credible interval")
    ax.set_title("Propagated to representative treatments", loc="left")
    return viz.savefig(fig, "fig41_kinetic_uncertainty_propagation")


def main():
    viz.apply_style()
    t0 = time.time()
    D = jominy_data()
    logp = lambda th: kcal.log_likelihood(th, D) + kcal.log_prior(th)  # noqa: E731
    th0 = theta_map()
    starts = [th0, th0 + np.array([0.8, -0.8, -1.0, 15.0, 0.3]), th0 + np.array([-0.8, 0.8, 1.0, -15.0, -0.3]),
              th0 + np.array([0.5, 0.5, -0.5, 10.0, 0.0])]
    scale0 = np.array([0.3, 0.3, 0.3, 4.0, 0.08])
    n_iter, n_adapt = 5000, 1500
    res = [adaptive_metropolis(logp, s, n_iter, n_adapt, scale0, seed=i) for i, s in enumerate(starts)]
    chains = np.stack([r[0] for r in res])          # (4, n, 5)
    acc = [r[2] for r in res]
    print(f"  MCMC: {time.time() - t0:.0f} s, acceptance {np.round(acc, 2)}")

    diag, summ = [], []
    for j, n in enumerate(NAMES):
        c = chains[:, :, j]
        flat = c.ravel()
        diag.append({"parameter": n, "split_Rhat": split_rhat(c), "ESS": ess(c)})
        summ.append({"parameter": n, "prior_mean": kcal.PRIOR_MEAN[j], "prior_sd": kcal.PRIOR_SD[j],
                     "MAP_stage8": th0[j], "posterior_mean": flat.mean(), "posterior_sd": flat.std(),
                     "q05": np.quantile(flat, 0.05), "q95": np.quantile(flat, 0.95),
                     "sd_ratio_to_prior": flat.std() / kcal.PRIOR_SD[j]})
    diag = pd.DataFrame(diag).assign(acceptance_mean=float(np.mean(acc)))
    summ = pd.DataFrame(summ)
    diag.to_csv(RES / "s11_mcmc_diagnostics.csv", index=False)
    summ.to_csv(RES / "s11_posterior_summary.csv", index=False)
    flat = chains.reshape(-1, len(NAMES))
    corr = pd.DataFrame(np.corrcoef(flat.T), index=NAMES, columns=NAMES)
    corr.to_csv(RES / "s11_posterior_correlation.csv")
    rng = np.random.default_rng(1)
    draws = flat[rng.choice(len(flat), 200, replace=False)]
    pd.DataFrame(draws, columns=NAMES).to_csv(RES / "s11_posterior_draws.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print(diag.round(3).to_string(index=False))

    # ---- propagation: CCT atlas
    d, comps, G, paths, rate = cct_atlas_frame()
    preds = []
    for th in draws:
        with kcal.kinetic_params(th):
            k = kinetics.simulate_batch(comps, G, paths)
        fp = (k.f_ferrite + k.f_pearlite + k.f_cementite_pro).to_numpy()
        hv = np.asarray(pp.hv_mixture(comps, rate * 3600.0, fp, k.f_bainite, k.f_martensite))
        preds.append(np.column_stack([fp, k.f_bainite.to_numpy(), k.f_martensite.to_numpy(), hv]))
    P = np.stack(preds)    # (draws, points, 4)
    cov_rows = []
    meas = {"f_FP": d.f_FP.to_numpy(), "f_B": d.f_B.to_numpy(), "f_M": d.f_M.to_numpy(), "HV": d.HV.to_numpy()}
    band = pd.DataFrame({"steel": d.steel, "t500_s": d.t500_s})
    for j, key in enumerate(("f_FP", "f_B", "f_M", "HV")):
        lo, med, hi = (np.quantile(P[:, :, j], q, axis=0) for q in (0.05, 0.5, 0.95))
        y = meas[key]
        ok = np.isfinite(y)
        inside = (y >= lo) & (y <= hi)
        resid = y[ok] - med[ok]
        s_struct = float(np.sqrt(max(np.mean(resid ** 2) - np.mean(((hi - lo)[ok] / 3.29) ** 2), 0.0)))
        lo2 = lo - 1.645 * s_struct
        hi2 = hi + 1.645 * s_struct
        inside2 = (y >= lo2) & (y <= hi2)
        cov_rows.append({"quantity": key, "points": int(ok.sum()), "median_band_width": float(np.median((hi - lo)[ok])),
                         "coverage_parametric_90": float(np.mean(inside[ok])),
                         "structural_sd": s_struct, "coverage_with_structural_90": float(np.mean(inside2[ok])),
                         "MAE_median_prediction": float(np.mean(np.abs(resid)))})
        band[f"{key}_meas"], band[f"{key}_q05"], band[f"{key}_q50"], band[f"{key}_q95"] = y, lo, med, hi
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(RES / "s11_cct_atlas_coverage.csv", index=False)
    band.to_csv(RES / "s11_cct_atlas_bands.csv", index=False)
    print(cov.round(3).to_string(index=False))

    # ---- propagation: dilatometry start temperatures
    dil = kcal.DilatometryData()
    rows = []
    for (cd, Gd, t, T, m) in dil.cases:
        sims = []
        for th in draws[:100]:
            with kcal.kinetic_params(th):
                sims.append(kinetics.simulate(cd, Gd, t, T))
        for key, mk in dil.KEYS:
            if pd.notna(m[mk]):
                v = np.array([s[key] for s in sims], float)
                v = v[np.isfinite(v)]
                rows.append({"steel": m.steel, "rate": m.rate, "start": key, "measured": m[mk],
                             "q05": np.quantile(v, 0.05) if len(v) else np.nan,
                             "q50": np.quantile(v, 0.5) if len(v) else np.nan,
                             "q95": np.quantile(v, 0.95) if len(v) else np.nan,
                             "share_predicting_start": len(v) / 100.0})
    dl = pd.DataFrame(rows)
    dl["inside"] = (dl.measured >= dl.q05) & (dl.measured <= dl.q95)
    dl.to_csv(RES / "s11_dilatometry_bands.csv", index=False)

    # ---- propagation: representative states
    rep = []
    for g, state, comp, path in representative_states():
        out = []
        for th in draws[:100]:
            with kcal.kinetic_params(th):
                k = kinetics.simulate(comp, 7.5, *path)
            out.append([k["f_bainite"], k["f_martensite"], k["f_ferrite"] + k["f_pearlite"] + k["f_cementite_pro"]])
        out = np.array(out)
        for j, key in enumerate(("f_bainite", "f_martensite", "f_ferrite_pearlite")):
            rep.append({"grade": g, "state": state, "quantity": key, "q05": np.quantile(out[:, j], 0.05),
                        "q50": np.quantile(out[:, j], 0.5), "q95": np.quantile(out[:, j], 0.95)})
    rep = pd.DataFrame(rep)
    rep.to_csv(RES / "s11_representative_bands.csv", index=False)

    # ---- figures
    fig, axes = plt.subplots(2, len(NAMES), figsize=(14.5, 5.2))
    labels = {"ln_f_ferrite": "ln ferrite factor", "ln_f_pearlite": "ln pearlite factor",
              "ln_f_bainite": "ln bainite factor", "dBs_C": "bainite-start shift (C)", "ln_sigma_model": "ln model sd (HRC)"}
    for j, n in enumerate(NAMES):
        for c, col in zip(chains[:, :, j], viz.SLOTS):
            axes[0, j].plot(c, lw=0.5, color=col, alpha=0.8)
        axes[0, j].set_title(labels[n], loc="left", fontsize=9)
        axes[1, j].hist(chains[:, :, j].ravel(), bins=40, color=viz.BLUE, alpha=0.8, density=True)
        xs = np.linspace(*axes[1, j].get_xlim(), 200)
        axes[1, j].plot(xs, np.exp(-0.5 * ((xs - kcal.PRIOR_MEAN[j]) / kcal.PRIOR_SD[j]) ** 2)
                        / (kcal.PRIOR_SD[j] * np.sqrt(2 * np.pi)), color=viz.INK2, lw=1, ls="--")
        axes[1, j].set_yticks([])
        r = diag.iloc[j]
        axes[1, j].set_xlabel(f"R-hat {r.split_Rhat:.2f}, ESS {r.ESS:.0f}", fontsize=8)
    axes[0, 0].set_ylabel("chains (after burn-in)"); axes[1, 0].set_ylabel("posterior (dashed: prior)")
    viz.savefig(fig, "fig40_kinetic_posterior")

    fig_propagation(band, cov, rep)
    print(f"stage 11 done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
