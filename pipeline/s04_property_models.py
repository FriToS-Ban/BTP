"""Stage 4 - property models: from textbook physics to physics-anchored ML.

Training data: ASM/Bethlehem typical properties of 33 AISI grades (annealed =
furnace cooled, normalized = air cooled, hot rolled = as-rolled), featurised
by the transformation model at the calibrated cooling conditions.

Evaluation
  * leave-one-grade-out (LOGO) cross-validation for every model and target
  * applicability-domain flag per held-out grade; headline metrics on the
    in-domain annealed + normalized rows
  * accuracy of the predicted annealed -> normalized *change* (the what-if)
  * paired comparison against the production model (Wilcoxon signed-rank on
    per-grade errors, bootstrap CI of the RMSE difference)
  * physical-consistency tests (cooling-rate and carbon monotonicity)
  * cooling-rate extrapolation: hardness of three dilatometry steels at nine
    rates (0.1-50 K/s) that the handbook data never contain
  * normalised split-conformal 90 % intervals and their LOGO coverage

Outputs: results/s04_*.csv, results/models/steel_predictor.joblib, figures fig11-fig14.
"""

import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import config, viz  # noqa: E402
from steelml.features import featurize  # noqa: E402
from steelml.models.consistency import consistency_report, featurized_sweeps, silicon_span  # noqa: E402
from steelml.models.domain import logo_domain  # noqa: E402
from steelml.models.evaluation import logo_oof, metrics, paired_route_difference  # noqa: E402
from steelml.models.property_models import (GREYBOX_BOUNDS, MODEL_ORDER, PRODUCTION_MODEL, TARGET_LABELS, TARGETS,  # noqa: E402
                                            GatedEnsemble, GreyBox, make_model)
from steelml.physics.properties import hb_from_hv  # noqa: E402
from steelml.predictor import SteelPredictor  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt
MODEL_LABELS = {
    "physics_lit": "Literature physics only", "ridge_comp": "Ridge (composition)",
    "gbm_comp": "Gradient boosting (composition)", "gp_comp": "GP (composition)",
    "physlit_gp": "Literature physics + GP residual", "gp_phys": "GP (physics descriptors)",
    "greybox": "Grey box (learned Gladman law)", "greybox_gp": "Grey box + GP residual",
    "gp_hybrid": "GP (composition + physics)", "anchored_gp": "Physics-anchored GP",
    "anchored_chain": "Physics-anchored chain", "gated_ensemble": "Gated ensemble (production)",
}


def collins_frame():
    comp = pd.read_csv(config.RAW_DIR / "collins2023_compositions.csv")
    hv = pd.read_csv(config.RAW_DIR / "collins2023_hardness.csv")
    rows = []
    for _, c in comp.iterrows():
        for r in sorted(hv[hv.steel == c.steel].rate):
            d = {el: float(c[el]) if el in c and pd.notna(c[el]) else 0.0 for el in config.ELEMENTS}
            d.update(steel=c.steel, route="CR", treat_T_C=c.T_aust_C, cooling_rate=r, section_mm=5.0)
            rows.append(d)
    F = featurize(pd.DataFrame(rows)).reset_index(drop=True)
    F = F.merge(hv, left_on=["steel", "cooling_rate"], right_on=["steel", "rate"])
    F["HB_meas"] = hb_from_hv(F.HV_mean)
    F["HB_meas_sd"] = hb_from_hv(F.HV_std)
    return F


def paired_tests(oof, F, ind, target, prod=PRODUCTION_MODEL, n_boot=4000, seed=0):
    """Per-grade squared errors of every model vs the production model."""
    rng = np.random.default_rng(seed)
    keep = ind & (F.route != "AR").to_numpy()
    g = F.grade.to_numpy()[keep]
    y = F[target].to_numpy()[keep]
    e_prod = (oof[(prod, target)][0][keep] - y) ** 2
    grades = np.unique(g)
    se_prod = np.array([e_prod[g == k].mean() for k in grades])
    rows = []
    for m in MODEL_ORDER:
        if m == prod:
            continue
        e = (oof[(m, target)][0][keep] - y) ** 2
        se = np.array([e[g == k].mean() for k in grades])
        try:
            p = wilcoxon(se, se_prod).pvalue
        except ValueError:
            p = np.nan
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(grades), len(grades))
            boots.append(np.sqrt(se[idx].mean()) - np.sqrt(se_prod[idx].mean()))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"target": target, "model": m, "RMSE_minus_production": np.sqrt(se.mean()) - np.sqrt(se_prod.mean()),
                     "CI95_lo": lo, "CI95_hi": hi, "wilcoxon_p": p, "n_grades": len(grades)})
    return rows


def holm(pvals):
    """Holm step-down adjustment of a family of p-values."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def normalised_conformal(y, mu, sd, groups, alpha=0.1):
    """q such that |y-mu| <= q*sd with LOGO-checked coverage."""
    s = np.abs(y - mu) / np.maximum(sd, 1e-6)
    n = len(s)
    q_all = float(np.quantile(s, min(1.0, np.ceil((1 - alpha) * (n + 1)) / n), method="higher"))
    cov, width = [], []
    for g in np.unique(groups):
        te = groups == g
        other = s[~te]
        m = len(other)
        q = np.quantile(other, min(1.0, np.ceil((1 - alpha) * (m + 1)) / m), method="higher")
        cov.extend(list(s[te] <= q))
        width.extend(list(2 * q * sd[te]))
    return q_all, float(np.mean(cov)), float(np.mean(width))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

# Gating rules of the production ensemble that were tried: (composition gate,
# cooling-rate gate, martensite gate); every rule also has the hard-phase gate.
GATE_RULES = {
    "final: hard phase + martensite": (False, False, True),
    "final + composition gate": (True, False, True),
    "final + cooling-rate gate": (False, True, True),
    "first version: composition + rate + hard phase": (True, True, False),
}


def _gate_weight(F, rule, max_hard, max_mart, max_rate, in_domain):
    comp, rate, mart = rule
    ex = [np.clip((F["f_hard"].to_numpy(float) - max_hard) / 0.2, 0.0, 1.0)]
    if mart:
        ex.append(np.clip((F["k_f_martensite"].to_numpy(float) - max_mart) / 0.2, 0.0, 1.0))
    if rate:
        ex.append(np.clip((F["log10_rate"].to_numpy(float) - max_rate) / 0.5, 0.0, 1.0))
    w = GatedEnsemble.DATA_WEIGHT * (1.0 - np.maximum.reduce(ex))
    return np.where(in_domain, w, 0.0) if comp else w


class _GateVariant:
    """The fitted production ensemble with another gating rule."""

    def __init__(self, g, rule, max_rate):
        self.g, self.rule, self.max_rate = g, rule, max_rate

    def predict(self, F):
        ind = self.g.dom.check(F)["in_domain"].to_numpy()
        w = _gate_weight(F, self.rule, self.g.max_hard, self.g.max_mart, self.max_rate, ind)
        return self.g.blend(F, w)


def gate_ablation(F, oof, ind, fitted, Fx, Fr, Fc, rng, Fs=None):
    """Re-score the gating rules on the same evidence, without refitting:
    LOGO errors from the members' out-of-fold predictions (fold-specific
    training maxima and domain flags), dilatometry hardness at <= 5 K/s, and
    monotonicity in the cooling-rate sweep."""
    fcac = (F.route != "AR").to_numpy()
    grades = F.grade.to_numpy()
    max_rate = float(F.log10_rate.max())
    rows = []
    for name, rule in GATE_RULES.items():
        rec = {"rule": name}
        for t in TARGETS:
            ma, _ = oof[("anchored_chain", t)]
            mp, _ = oof[("physlit_gp", t)]
            mh, _ = oof[("gp_hybrid", t)]
            w = np.zeros(len(F))
            for g in np.unique(grades):
                te = grades == g
                w[te] = _gate_weight(F[te], rule, F.f_hard[~te].max(), F.k_f_martensite[~te].max(),
                                     F.log10_rate[~te].max(), ind[te])
            mu = (1 - w) / 2 * ma + (1 - w) / 2 * mp + w * mh
            y = F[t].to_numpy(float)
            rec[f"{t}_RMSE_outAD"] = metrics(y[fcac & ~ind], mu[fcac & ~ind])["RMSE"]
            rec[f"{t}_RMSE_inAD"] = metrics(y[fcac & ind], mu[fcac & ind])["RMSE"]
            if rule == GATE_RULES["final: hard phase + martensite"]:
                rec[f"{t}_check_vs_production"] = float(np.max(np.abs(mu - oof[(PRODUCTION_MODEL, t)][0])))
        mu, _ = _GateVariant(fitted[(PRODUCTION_MODEL, "HB")], rule, max_rate).predict(Fx)
        sel = (Fx.cooling_rate <= 5).to_numpy()
        rec["HB_RMSE_dilatometry_le_5Ks"] = float(np.sqrt(np.mean((mu[sel] - Fx.HB_meas[sel]) ** 2)))
        rec["HB_RMSE_dilatometry_all"] = float(np.sqrt(np.mean((mu - Fx.HB_meas) ** 2)))
        variants = {(name, t): _GateVariant(fitted[(PRODUCTION_MODEL, t)], rule, max_rate) for t in TARGETS}
        for _, c in consistency_report(variants, Fr, Fc, rng, F_si=Fs).iterrows():
            rec[f"{c.target}_rate_violation_frac"] = c.rate_violation_frac
        rows.append(rec)
    return pd.DataFrame(rows)


def anchor_ablation(F, ind, Fx, Fr, Fc, Fs):
    """Production model with the first anchor (Maynier hardness for every
    constituent) and with the microstructure-based anchor (Gladman law for
    ferrite + pearlite, Maynier for bainite and martensite)."""
    from steelml.models import property_models as pm
    saved = pm.ANCHOR_COLUMN
    fcac = (F.route != "AR").to_numpy()
    rng = {t: float(F[t].max() - F[t].min()) for t in TARGETS}
    rows = []
    try:
        for anchor, label in (("HB_phys", "Maynier hardness, all constituents (first version)"),
                              ("UTS_phys", "Gladman law for F+P, Maynier for B+M (production)")):
            pm.ANCHOR_COLUMN = anchor
            rec = {"anchor": label}
            fitted = {}
            for t in TARGETS:
                y = F[t].to_numpy(float)
                mu, _ = logo_oof(F, y, F.grade.to_numpy(), PRODUCTION_MODEL, t)
                rec[f"{t}_RMSE_inAD"] = metrics(y[fcac & ind], mu[fcac & ind])["RMSE"]
                rec[f"{t}_RMSE_outAD"] = metrics(y[fcac & ~ind], mu[fcac & ~ind])["RMSE"]
                _, pmr = paired_route_difference(F[ind].reset_index(drop=True), y[ind], mu[ind])
                rec[f"{t}_change_RMSE"] = pmr["RMSE"]
                fitted[(PRODUCTION_MODEL, t)] = make_model(PRODUCTION_MODEL, t).fit(F, y)
            cons = consistency_report(fitted, Fr, Fc, rng, F_si=Fs).set_index("target")
            for t in TARGETS:
                rec[f"{t}_rate_violation_frac"] = cons.loc[t, "rate_violation_frac"]
                rec[f"{t}_carbon_violation_frac"] = cons.loc[t, "carbon_violation_frac"]
            mu, _ = fitted[(PRODUCTION_MODEL, "HB")].predict(Fx)
            sel = (Fx.cooling_rate <= 5).to_numpy()
            rec["HB_RMSE_dilatometry_le_5Ks"] = float(np.sqrt(np.mean((mu[sel] - Fx.HB_meas[sel]) ** 2)))
            rec["HB_RMSE_dilatometry_all"] = float(np.sqrt(np.mean((mu - Fx.HB_meas) ** 2)))
            rows.append(rec)
    finally:
        pm.ANCHOR_COLUMN = saved
    return pd.DataFrame(rows)


def silicon_ablation(F, ind, Fs):
    """The lesson behind property_models.NO_DATA_ELEMENTS: the same models with
    silicon added back to their data-driven inputs. Leave-one-grade-out looks
    better with it (the 0.225 vs 0.25 % nominal Si labels carbon vs alloy
    H-band grades), while the response to a real 0.10-0.40 % Si is unphysical."""
    import steelml.models.property_models as pm
    lists = {pm.GPComp: "features", pm.GPHybrid: "features", pm.AnchoredGP: "resid_features",
             pm._RatioGP: "features", pm.PhysLitGP: "resid_features"}
    saved = {c: list(getattr(c, a)) for c, a in lists.items()}
    fcac = (F.route != "AR").to_numpy()
    y = F.UTS_MPa.to_numpy(float)
    rows = []
    try:
        for with_si in (False, True):
            for c, a in lists.items():
                setattr(c, a, saved[c] + (["Si"] if with_si else []))
            for m in ("gp_comp", PRODUCTION_MODEL):
                mu, _ = logo_oof(F, y, F.grade.to_numpy(), m, "UTS_MPa")
                fit = make_model(m, "UTS_MPa").fit(F, y)
                rows.append({"silicon_input": with_si, "model": m,
                             "UTS_RMSE_inAD": metrics(y[fcac & ind], mu[fcac & ind])["RMSE"],
                             "UTS_RMSE_outAD": metrics(y[fcac & ~ind], mu[fcac & ~ind])["RMSE"],
                             "UTS_span_Si_0.10_to_0.40": silicon_span(fit, Fs)})
    finally:
        for c, a in lists.items():
            setattr(c, a, saved[c])
    return pd.DataFrame(rows)


def fig_ladder(res):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.4), sharey=True)
    order = MODEL_ORDER
    y = np.arange(len(order))
    for ax, t in zip(axes, TARGETS):
        d = res[res.target == t].set_index("model").loc[order]
        colors = [viz.BLUE if m == PRODUCTION_MODEL else viz.NEUTRAL for m in order]
        ax.barh(y, d.RMSE_inAD, color=colors, height=0.62)
        for yi, v in zip(y, d.RMSE_inAD):
            ax.text(v, yi, f" {v:.1f}" if t == "El_pct" else f" {v:.0f}", va="center", fontsize=8, color=viz.INK2)
        ax.set_title(TARGET_LABELS[t].replace(" (", "\nRMSE ("))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [MODEL_LABELS[m] for m in order], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.suptitle("Leave-one-grade-out error on unseen, in-domain grades (annealed + normalized rows)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig12_model_ladder")


def fig_parity(F, oof, ind, q90):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, t in zip(axes, TARGETS):
        mu, sd = oof[(PRODUCTION_MODEL, t)]
        y = F[t].to_numpy()
        lo, hi = min(y.min(), mu.min()), max(y.max(), mu.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=viz.AXIS, lw=1)
        for route in ("FC", "AC", "AR"):
            s = (F.route == route).to_numpy()
            ax.errorbar(y[s], mu[s], yerr=q90[t] * sd[s], fmt="o", ms=5, color=viz.ROUTE_COLORS[route],
                        ecolor=viz.ROUTE_COLORS[route], elinewidth=0.8, alpha=0.9, mec=viz.SURFACE, mew=0.8,
                        label=viz.ROUTE_LABELS[route])
        out = ~ind
        ax.plot(y[out], mu[out], "s", mfc="none", mec=viz.INK, ms=9, mew=1.0, label="outside domain (flagged)")
        m = metrics(y[ind & (F.route != "AR").to_numpy()], mu[ind & (F.route != "AR").to_numpy()])
        ax.set_title(f"{TARGET_LABELS[t]}\nin-domain RMSE {m['RMSE']:.3g}, R2 {m['R2']:.2f}")
        ax.set_xlabel("Measured")
    axes[0].set_ylabel("Predicted (grade held out)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Production model, leave-one-grade-out predictions with 90 % conformal intervals",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig13_parity_production")


def fig_extrapolation(Fx, preds):
    steels = [("EN3B", "EN3B (0.18C)"), ("EN8", "EN8 (0.44C)"), ("SA540", "SA-540 (Ni-Cr-Mo)")]
    show = [("gp_comp", viz.YELLOW, "GP on composition (no physics)"),
            ("physics_lit", viz.VIOLET, "Literature physics only"),
            (PRODUCTION_MODEL, viz.BLUE, "Production (physics-anchored, gated)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (s, title) in zip(axes, steels):
        d = Fx[Fx.steel == s].sort_values("cooling_rate")
        ax.errorbar(d.cooling_rate, d.HB_meas, yerr=d.HB_meas_sd, fmt="o", color=viz.INK, ms=6,
                    elinewidth=1, label="Measured (mean ± s.d. of 16 indents)", zorder=5)
        for m, col, lab in show:
            ax.plot(d.cooling_rate, preds[m][d.index], "-", color=col, lw=2, label=lab)
        # the training data have only two cooling conditions per grade:
        # furnace cooling (~0.005 K/s) and air cooling (~0.5-1 K/s at 700 C)
        for lo, hi in ((0.003, 0.008), (0.4, 1.1)):
            ax.axvspan(lo, hi, color=viz.GRID, alpha=0.8, lw=0)
        ax.text(0.02, 0.97, "shaded: the two cooling\nconditions in the training data", transform=ax.transAxes,
                fontsize=8, color=viz.MUTED, va="top")
        ax.set_xscale("log")
        ax.set_xlim(0.002, 80)
        ax.set_xlabel("Cooling rate at 700 °C (K/s)")
        ax.set_title(title)
    axes[0].set_ylabel("Hardness, HB equivalent (0.95 x measured HV)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Cooling-rate extrapolation: models trained on annealed/normalized data, tested on dilatometry",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig14_cooling_rate_extrapolation")


def fig_greybox(F):
    rows = []
    for t in ("UTS_MPa", "YS_MPa", "HB"):
        c = GreyBox(t).fit(F, F[t].to_numpy(float)).coefficients()
        for k, v in c.items():
            rows.append({"target": t, "term": k, "learned": v})
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "s04_greybox_coefficients.csv", index=False)
    u = df[df.target == "UTS_MPa"].set_index("term").learned
    lit = {"ferrite_matrix": 246.0, "pearlite_base": 719.0, "ferrite_hall_petch": 18.1, "pearlite_spacing": 3.5,
           "solid_solution": 1.0, "hard_phase_hv": 330.0}
    labels = {"ferrite_matrix": "ferrite base strength", "pearlite_base": "pearlite base strength",
              "ferrite_hall_petch": "ferrite Hall-Petch coefficient", "pearlite_spacing": "pearlite spacing coefficient",
              "solid_solution": "solid-solution strengthening", "hard_phase_hv": "bainite/martensite per HV"}
    lo_bounds = {k: v[0] for k, v in GREYBOX_BOUNDS["strength"].items()}
    hi_bounds = {k: v[1] for k, v in GREYBOX_BOUNDS["strength"].items()}
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    terms = list(lit)
    y = np.arange(len(terms))
    ratio = np.array([u[t] / lit[t] for t in terms])
    at_bound = np.array([np.isclose(u[t], lo_bounds[t] if np.isfinite(lo_bounds[t]) else np.nan, atol=1e-6)
                         or np.isclose(u[t], hi_bounds[t] if np.isfinite(hi_bounds[t]) else np.nan, atol=1e-6)
                         for t in terms])
    ax.axvline(1.0, color=viz.INK2, lw=1.0)
    ax.plot(ratio[~at_bound], y[~at_bound], "o", ms=9, color=viz.BLUE, mec=viz.SURFACE, label="learned / literature")
    ax.plot(ratio[at_bound], y[at_bound], "o", ms=9, mfc=viz.SURFACE, mec=viz.MUTED, mew=1.5,
            label="pinned at its bound: not identifiable from these data")
    for yy, r, t in zip(y, ratio, terms):
        ax.annotate(f"{u[t]:.3g} vs {lit[t]:.3g}", (r, yy), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=viz.INK2)
    ax.set_yticks(y, [labels[t] for t in terms], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-0.1, max(2.6, float(np.nanmax(ratio)) + 0.6))
    ax.set_xlabel("Learned coefficient / literature value (Gladman et al. UTS law; 3.3 MPa per HV)")
    sph = float(u["spheroidisation_risk"])
    ax.set_title(f"Grey-box UTS law fitted to the handbook data (slow-cooled pearlite penalty {sph:.0f} MPa)")
    ax.legend(loc="lower right", fontsize=8)
    return viz.savefig(fig, "fig15_greybox_coefficients")


def main():
    config.ensure_dirs()
    p = pd.read_csv(config.PROCESSED_DIR / "properties_handbook.csv")
    F = featurize(p).reset_index(drop=True)
    ad = logo_domain(F, F.grade.values)
    ind = ad.in_domain.to_numpy()
    fcac = (F.route != "AR").to_numpy()
    pd.concat([F[["grade", "condition", "route"]], ad], axis=1).to_csv(
        config.RESULTS_DIR / "s04_domain_flags.csv", index=False)
    print("out-of-domain grades:", sorted(set(F.grade[~ind])))

    oof, rows, paired = {}, [], []
    t0 = time.time()
    for t in TARGETS:
        for m in MODEL_ORDER:
            mu, sd = logo_oof(F, F[t].to_numpy(float), F.grade.to_numpy(), m, t)
            oof[(m, t)] = (mu, sd)
            y = F[t].to_numpy(float)
            a = metrics(y[fcac], mu[fcac])
            b = metrics(y[fcac & ind], mu[fcac & ind])
            c = metrics(y[fcac & ~ind], mu[fcac & ~ind])
            byr = {r: metrics(y[(F.route == r).to_numpy() & ind], mu[(F.route == r).to_numpy() & ind])["RMSE"]
                   for r in ("FC", "AC", "AR")}
            _, pm = paired_route_difference(F[ind].reset_index(drop=True), y[ind], mu[ind])
            rows.append({"target": t, "model": m, "RMSE_all": a["RMSE"], "R2_all": a["R2"],
                         "RMSE_inAD": b["RMSE"], "MAE_inAD": b["MAE"], "R2_inAD": b["R2"], "bias_inAD": b["bias"],
                         "MAPE_inAD": b["MAPE_pct"], "RMSE_outAD": c["RMSE"],
                         "RMSE_FC": byr["FC"], "RMSE_AC": byr["AC"], "RMSE_AR": byr["AR"],
                         "delta_RMSE": pm["RMSE"], "delta_R2": pm["R2"], "delta_sign_agreement": pm["sign_agreement"]})
        paired.extend(paired_tests(oof, F, ind, t))
        print(f"{t} done ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "s04_model_benchmark.csv", index=False)
    paired = pd.DataFrame(paired)
    # about eleven comparisons per property: Holm-adjust within each property
    paired["wilcoxon_p_holm"] = paired.groupby("target").wilcoxon_p.transform(lambda x: holm(x.to_numpy()))
    paired.to_csv(config.RESULTS_DIR / "s04_paired_tests.csv", index=False)
    oof_df = F[["grade", "condition", "route"] + TARGETS].copy()
    for (m, t), (mu, sd) in oof.items():
        oof_df[f"{m}|{t}|mu"] = mu
        oof_df[f"{m}|{t}|sd"] = sd
    oof_df.to_csv(config.RESULTS_DIR / "s04_oof_predictions.csv", index=False)
    print(res.pivot(index="model", columns="target", values="RMSE_inAD").round(1).to_string())

    # conformal calibration of the production model
    q90, cal_rows = {}, []
    for t in TARGETS:
        mu, sd = oof[(PRODUCTION_MODEL, t)]
        y = F[t].to_numpy(float)
        q, cov, width = normalised_conformal(y[ind], mu[ind], sd[ind], F.grade.to_numpy()[ind])
        q90[t] = q
        cal_rows.append({"target": t, "q90_normalised": q, "LOGO_coverage_90": cov, "mean_interval_width": width})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(config.RESULTS_DIR / "s04_conformal.csv", index=False)
    print(cal.round(3).to_string(index=False))

    # physical consistency
    Fr, Fc, Fs = featurized_sweeps(p)
    fitted = {(m, t): make_model(m, t).fit(F, F[t].to_numpy(float)) for m in MODEL_ORDER for t in TARGETS}
    rng = {t: float(F[t].max() - F[t].min()) for t in TARGETS}
    cons = consistency_report(fitted, Fr, Fc, rng, F_si=Fs)
    cons.to_csv(config.RESULTS_DIR / "s04_consistency.csv", index=False)

    # cooling-rate extrapolation on dilatometry hardness
    Fx = collins_frame()
    preds, ext = {}, []
    for m in MODEL_ORDER:
        mu, _ = fitted[(m, "HB")].predict(Fx)
        preds[m] = pd.Series(mu, index=Fx.index)
        rec = {"model": m}
        for s in ("EN3B", "EN8", "SA540"):
            sel = (Fx.steel == s).to_numpy()
            rec[f"HB_RMSE_{s}"] = float(np.sqrt(np.mean((mu[sel] - Fx.HB_meas[sel]) ** 2)))
        sel = (Fx.cooling_rate <= 5).to_numpy()
        rec["HB_RMSE_rates_le_5Ks"] = float(np.sqrt(np.mean((mu[sel] - Fx.HB_meas[sel]) ** 2)))
        rec["HB_RMSE_all"] = float(np.sqrt(np.mean((mu - Fx.HB_meas) ** 2)))
        ext.append(rec)
    ext = pd.DataFrame(ext)
    ext.to_csv(config.RESULTS_DIR / "s04_cooling_rate_extrapolation.csv", index=False)
    print(ext.round(1).to_string(index=False))

    # which evidence decided the gating rule of the production ensemble
    abl = gate_ablation(F, oof, ind, fitted, Fx, Fr, Fc, rng, Fs)
    abl.to_csv(config.RESULTS_DIR / "s04_gate_ablation.csv", index=False)
    print(abl.round(3).T.to_string())

    # why silicon (and vanadium) are not data-driven inputs
    sil = silicon_ablation(F, ind, Fs)
    sil.to_csv(config.RESULTS_DIR / "s04_silicon_ablation.csv", index=False)
    print(sil.round(1).to_string(index=False))

    # why the physics anchor is the microstructure-based strength
    anc = anchor_ablation(F, ind, Fx, Fr, Fc, Fs)
    anc.to_csv(config.RESULTS_DIR / "s04_anchor_ablation.csv", index=False)
    print(anc.round(3).T.to_string())

    fig_ladder(res)
    fig_parity(F, oof, ind, q90)
    fig_extrapolation(Fx, preds)
    fig_greybox(F)

    # production bundle
    cat_model = None
    cpath = config.MODELS_DIR / "category_model.joblib"
    if cpath.exists():
        cat_model = joblib.load(cpath)
    sp = SteelPredictor.train(F, q90, cat_model, meta={
        "production_model": PRODUCTION_MODEL, "trained_on": "ASM/Bethlehem handbook, 33 grades, 80 rows",
        "handbook_ac_section_mm": config.HANDBOOK_AC_SECTION_MM, "handbook_fc_rate": config.HANDBOOK_FC_RATE})
    sp.save()
    print("saved", config.MODELS_DIR / "steel_predictor.joblib")


if __name__ == "__main__":
    main()
