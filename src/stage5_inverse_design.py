"""Stage 5. Local inverse design inside the training composition box.

Declared before the search runs:
- Screening model: default random forest, 13 non-iron wt% features, fit on all
  312 alloys. This is an in-sample scoring surface. It is not a new cross-validated
  accuracy number. Tuning did not beat this forest in Stage 3.
- Lower confidence bound: predicted yield minus 1.0 times the standard deviation
  across the forest's trees. That spread is tree disagreement, not a calibrated
  standard error.
- A Gaussian process with the Stage 3 ARD Matérn kernel supplies a second
  uncertainty and the length scales that Stage 3 never saved.
- Search: Optuna TPE, 400 trials, seeds 42 and 43. Each trial starts at a
  training alloy and moves each solute by at most half of its training standard
  deviation (floor 0.02 wt% for C, N, Nb, V and 0.05 wt% otherwise), clipped to
  the training min and max. Iron is the closure.
- A trial is feasible only if iron stays inside its training range, the
  applicability domain says inside, and the predicted yield is at least 1500 MPa.
- Criticality is the sum of wt% times the EU 2023 supply-risk weights in
  btp_common. It is not a price.
"""

import os
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from applicability_domain import ApplicabilityDomain
from btp_common import (
    ELEMENTS,
    EU2023_SUPPLY_RISK,
    FIG_DIR,
    NON_FE_ELEMENTS,
    RANDOM_SEED,
    RES_DIR,
    TARGET_YIELD_MPA,
    UNCERTAINTY_K,
    assign_family,
    carbon_equivalents,
    cet_violation_frame,
    criticality_index,
    load_data,
    make_repeated_group_splits,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 400
SEEDS = (42, 43)
SPARSE_FLOOR = {"C": 0.02, "N": 0.02, "Nb": 0.02, "V": 0.02}


def gp_kernel():
    return ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
        length_scale=[1.0] * len(NON_FE_ELEMENTS),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1.0))


def make_forest():
    return RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=4,
        random_state=RANDOM_SEED, n_jobs=-1,
    )


def tree_std(model, x_values):
    stacked = np.stack([tree.predict(x_values) for tree in model.estimators_], axis=0)
    return stacked.std(axis=0, ddof=1)


def step_halfwidths(df_wt):
    widths = {}
    for el in NON_FE_ELEMENTS:
        floor = SPARSE_FLOOR.get(el, 0.05)
        widths[el] = float(max(0.5 * df_wt[el].std(ddof=1), floor))
    return widths


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


class ScreeningSurface:
    def __init__(self, df_wt, y):
        self.df_wt = df_wt
        self.y = y
        self.x13 = df_wt[NON_FE_ELEMENTS].to_numpy(dtype=float)
        self.forest = make_forest()
        self.forest.fit(self.x13, y)
        self.scaler = StandardScaler().fit(self.x13)
        self.gp = GaussianProcessRegressor(
            kernel=gp_kernel(), normalize_y=True, n_restarts_optimizer=5, random_state=RANDOM_SEED,
        )
        self.gp.fit(self.scaler.transform(self.x13), y)
        self.domain = ApplicabilityDomain(
            metric="euclidean", distance_percentile=95.0, box_tolerance=0.0,
            exclude_element="Fe", standardize=True,
        )
        self.domain.fit(df_wt[ELEMENTS], feature_names=ELEMENTS)
        self.retrieval_cutoff = float(np.median(self.domain.train_nn_distances_))
        self.bounds = {
            el: (float(df_wt[el].min()), float(df_wt[el].max())) for el in ELEMENTS
        }
        self.halfwidths = step_halfwidths(df_wt)

    def score_frame(self, frame):
        x13 = frame[NON_FE_ELEMENTS].to_numpy(dtype=float)
        pred = self.forest.predict(x13)
        spread = tree_std(self.forest, x13)
        gp_mean, gp_std = self.gp.predict(self.scaler.transform(x13), return_std=True)
        ad = self.domain.query(frame[ELEMENTS])
        out = frame.copy()
        out["pred_mpa"] = pred
        out["tree_std_mpa"] = spread
        out["lcb_mpa"] = pred - UNCERTAINTY_K * spread
        out["gp_mean_mpa"] = gp_mean
        out["gp_std_mpa"] = gp_std
        out["inside_domain"] = ad["inside_domain"].to_numpy()
        out["nearest_dist_std"] = ad["nearest_dist_std"].to_numpy()
        out["nearest_train_idx"] = ad["nearest_train_idx"].to_numpy()
        out["criticality"] = criticality_index(frame)
        ce, pcm = carbon_equivalents(frame)
        out["ce_iiw"] = ce
        out["pcm"] = pcm
        out["cet_valid"] = cet_violation_frame(frame)["cet_valid"].to_numpy()
        out["family"] = assign_family(frame)
        labels = []
        for inside, dist in zip(out["inside_domain"], out["nearest_dist_std"]):
            if not inside:
                labels.append("outside_domain")
            elif dist <= self.retrieval_cutoff:
                labels.append("retrieval")
            else:
                labels.append("in_domain_variant")
        out["domain_label"] = labels
        return out


def propose(trial, surface):
    base = trial.suggest_int("base_idx", 0, len(surface.df_wt) - 1)
    base_row = surface.df_wt.iloc[base]
    values = {el: float(base_row[el]) for el in ELEMENTS}
    for el in NON_FE_ELEMENTS:
        low, high = surface.bounds[el]
        delta = trial.suggest_float(f"d_{el}", -surface.halfwidths[el], surface.halfwidths[el])
        values[el] = float(np.clip(base_row[el] + delta, low, high))
    values["Fe"] = 100.0 - sum(values[el] for el in NON_FE_ELEMENTS)
    return base, values


def fe_violation(values, surface):
    low, high = surface.bounds["Fe"]
    if values["Fe"] < low:
        return low - values["Fe"]
    if values["Fe"] > high:
        return values["Fe"] - high
    return 0.0


def load_surface(path, df_wt, y):
    """Rebuild the screening object from the saved bundle without refitting."""
    bundle = joblib.load(path)
    surface = ScreeningSurface.__new__(ScreeningSurface)
    surface.df_wt = df_wt
    surface.y = y
    surface.x13 = df_wt[NON_FE_ELEMENTS].to_numpy(dtype=float)
    surface.forest = bundle["forest"]
    surface.scaler = bundle["scaler"]
    surface.gp = bundle["gp"]
    surface.domain = bundle["domain"]
    surface.retrieval_cutoff = bundle["retrieval_cutoff"]
    surface.bounds = bundle["bounds"]
    surface.halfwidths = bundle["halfwidths"]
    return surface


def run_study(surface, seed):
    records = []

    def objective(trial):
        base, values = propose(trial, surface)
        frame = pd.DataFrame([values])[ELEMENTS]
        scored = surface.score_frame(frame).iloc[0]
        violation = fe_violation(values, surface)
        trial.set_user_attr("base_idx", int(base))
        trial.set_user_attr("pred_mpa", float(scored["pred_mpa"]))
        trial.set_user_attr("tree_std_mpa", float(scored["tree_std_mpa"]))
        trial.set_user_attr("lcb_mpa", float(scored["lcb_mpa"]))
        trial.set_user_attr("gp_std_mpa", float(scored["gp_std_mpa"]))
        trial.set_user_attr("inside_domain", bool(scored["inside_domain"]))
        trial.set_user_attr("nearest_dist_std", float(scored["nearest_dist_std"]))
        trial.set_user_attr("nearest_train_idx", int(scored["nearest_train_idx"]))
        trial.set_user_attr("criticality", float(scored["criticality"]))
        trial.set_user_attr("domain_label", scored["domain_label"])
        trial.set_user_attr("fe_violation", float(violation))
        trial.set_user_attr("composition", values)
        # Constraints are <= 0 when feasible. Returned here only as attributes.
        return float(scored["lcb_mpa"])

    def constraints(trial):
        return (
            TARGET_YIELD_MPA - trial.user_attrs["pred_mpa"],
            0.0 if trial.user_attrs["inside_domain"] else 1.0,
            trial.user_attrs["fe_violation"],
        )

    sampler = optuna.samplers.TPESampler(seed=seed, constraints_func=constraints)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "seed": seed,
            "trial": trial.number,
            "lcb_mpa": trial.user_attrs["lcb_mpa"],
            "pred_mpa": trial.user_attrs["pred_mpa"],
            "tree_std_mpa": trial.user_attrs["tree_std_mpa"],
            "gp_std_mpa": trial.user_attrs["gp_std_mpa"],
            "criticality": trial.user_attrs["criticality"],
            "inside_domain": trial.user_attrs["inside_domain"],
            "fe_violation": trial.user_attrs["fe_violation"],
            "nearest_dist_std": trial.user_attrs["nearest_dist_std"],
            "nearest_train_idx": trial.user_attrs["nearest_train_idx"],
            "base_idx": trial.user_attrs["base_idx"],
            "domain_label": trial.user_attrs["domain_label"],
            "feasible": bool(
                trial.user_attrs["pred_mpa"] >= TARGET_YIELD_MPA
                and trial.user_attrs["inside_domain"]
                and trial.user_attrs["fe_violation"] <= 0
            ),
        }
        row.update(trial.user_attrs["composition"])
        records.append(row)
    return pd.DataFrame(records)


def length_scales_by_fold(df_wt, y, groups):
    x13 = df_wt[NON_FE_ELEMENTS].to_numpy(dtype=float)
    rows = []
    for split_idx, (tr_idx, _val_idx) in enumerate(make_repeated_group_splits(groups), start=1):
        scaler = StandardScaler().fit(x13[tr_idx])
        gp = GaussianProcessRegressor(
            kernel=gp_kernel(), normalize_y=True, n_restarts_optimizer=2, random_state=RANDOM_SEED,
        )
        gp.fit(scaler.transform(x13[tr_idx]), y[tr_idx])
        # kernel_ is Constant * Matern + White. The Matern is k1.k2.
        matern = gp.kernel_.k1.k2
        scales = np.atleast_1d(np.array(matern.length_scale, dtype=float))
        row = {"split": split_idx}
        for el, scale in zip(NON_FE_ELEMENTS, scales):
            row[el] = float(scale)
            row[f"{el}_at_upper_bound"] = bool(scale >= 99.0)
        rows.append(row)
        print(f"  GP length scales, split {split_idx}/25")
    return pd.DataFrame(rows)


def top_feasible(frame, n=10):
    feasible = frame[frame["feasible"]].sort_values("lcb_mpa", ascending=False)
    if feasible.empty:
        return feasible
    rounded = feasible[NON_FE_ELEMENTS].round(2).astype(str).agg("|".join, axis=1)
    feasible = feasible.loc[~rounded.duplicated()]
    return feasible.head(n)


def mean_distance(a, b):
    if a.empty or b.empty:
        return np.nan
    return float(np.linalg.norm(a[NON_FE_ELEMENTS].mean().to_numpy() - b[NON_FE_ELEMENTS].mean().to_numpy()))


def run_stage5():
    apply_style()
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    started = time.time()
    df_wt, y, groups, _formulas = load_data()
    surface = ScreeningSurface(df_wt, y)

    bounds_rows = []
    for el in ELEMENTS:
        low, high = surface.bounds[el]
        bounds_rows.append({
            "element": el,
            "min_wt": low,
            "max_wt": high,
            "step_halfwidth_wt": surface.halfwidths.get(el, 0.0),
            "eu2023_supply_risk": EU2023_SUPPLY_RISK[el],
        })
    pd.DataFrame(bounds_rows).to_csv(os.path.join(RES_DIR, "stage5_search_bounds.csv"), index=False)
    pd.DataFrame([
        {"element": el, "eu2023_supply_risk": weight, "source": "European Commission, Study on the Critical Raw Materials for the EU 2023"}
        for el, weight in EU2023_SUPPLY_RISK.items()
    ]).to_csv(os.path.join(RES_DIR, "stage5_element_cost_proxy.csv"), index=False)

    print("Fitting fold-wise GP length scales.")
    scales = length_scales_by_fold(df_wt, y, groups)
    scales.to_csv(os.path.join(RES_DIR, "stage5_gp_length_scales_by_fold.csv"), index=False)
    summary = {"element": [], "median_length_scale": [], "min_length_scale": [], "max_length_scale": [], "frac_at_upper_bound": []}
    for el in NON_FE_ELEMENTS:
        summary["element"].append(el)
        summary["median_length_scale"].append(float(scales[el].median()))
        summary["min_length_scale"].append(float(scales[el].min()))
        summary["max_length_scale"].append(float(scales[el].max()))
        summary["frac_at_upper_bound"].append(float(scales[f"{el}_at_upper_bound"].mean()))
    pd.DataFrame(summary).sort_values("median_length_scale").to_csv(
        os.path.join(RES_DIR, "stage5_gp_length_scale_summary.csv"), index=False
    )
    print(pd.DataFrame(summary).sort_values("median_length_scale").to_string(index=False))

    full_scales = np.atleast_1d(np.array(surface.gp.kernel_.k1.k2.length_scale, dtype=float))
    pd.DataFrame({
        "element": NON_FE_ELEMENTS,
        "full_data_length_scale": full_scales,
        "at_upper_bound": full_scales >= 99.0,
        "note": "Full-data explanatory fit. Not a cross-validated importance.",
    }).to_csv(os.path.join(RES_DIR, "stage5_gp_length_scales_full_data.csv"), index=False)

    print("Scoring the 312 training alloys on the in-sample surface.")
    training = surface.score_frame(df_wt)
    training.insert(0, "alloy_idx", np.arange(len(df_wt)))
    training["y_true"] = y
    training["set"] = "training_in_sample"
    training.to_csv(os.path.join(RES_DIR, "stage5_training_surface.csv"), index=False)

    trial_frames = []
    for seed in SEEDS:
        print(f"Optuna seed {seed}, {N_TRIALS} trials.")
        trial_frames.append(run_study(surface, seed))
    trials = pd.concat(trial_frames, ignore_index=True)
    trials.to_csv(os.path.join(RES_DIR, "stage5_all_trials.csv"), index=False)

    tops = []
    for seed, part in trials.groupby("seed"):
        chosen = top_feasible(part, n=10)
        chosen = chosen.copy()
        chosen["rank_within_seed"] = np.arange(1, len(chosen) + 1)
        tops.append(chosen)
        n_feas = int(part["feasible"].sum())
        print(f"Seed {seed}: {n_feas}/{len(part)} feasible trials, {len(chosen)} unique top candidates.")
    candidates = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()
    candidates.to_csv(os.path.join(RES_DIR, "stage5_candidates.csv"), index=False)

    top42 = top_feasible(trials[trials["seed"] == 42], n=10)
    top43 = top_feasible(trials[trials["seed"] == 43], n=10)
    train_lcb = training["lcb_mpa"]
    overlap = 0.0
    if not top42.empty and not top43.empty:
        a = set(top42["nearest_train_idx"].tolist())
        b = set(top43["nearest_train_idx"].tolist())
        overlap = len(a & b) / len(a | b)
    audit = pd.DataFrame([{
        "n_trials_per_seed": N_TRIALS,
        "target_yield_mpa": TARGET_YIELD_MPA,
        "uncertainty_k": UNCERTAINTY_K,
        "retrieval_distance_cutoff": surface.retrieval_cutoff,
        "ad_distance_cutoff": float(surface.domain.distance_cutoff_),
        "feasible_seed_42": int(((trials["seed"] == 42) & trials["feasible"]).sum()),
        "feasible_seed_43": int(((trials["seed"] == 43) & trials["feasible"]).sum()),
        "mean_composition_distance_top10": mean_distance(top42, top43),
        "jaccard_nearest_training_idx_top10": overlap,
        "training_lcb_p95": float(np.percentile(train_lcb, 95)),
        "training_lcb_max": float(train_lcb.max()),
        "best_candidate_lcb": float(candidates["lcb_mpa"].max()) if len(candidates) else np.nan,
        "best_candidate_beats_training_max": bool(len(candidates) and candidates["lcb_mpa"].max() > train_lcb.max()),
        "n_candidates_labeled_retrieval": int((candidates["domain_label"] == "retrieval").sum()) if len(candidates) else 0,
        "n_candidates_labeled_variant": int((candidates["domain_label"] == "in_domain_variant").sum()) if len(candidates) else 0,
    }])
    audit.to_csv(os.path.join(RES_DIR, "stage5_search_audit.csv"), index=False)
    print(audit.T.to_string(header=False))

    joblib.dump(
        {
            "forest": surface.forest,
            "gp": surface.gp,
            "scaler": surface.scaler,
            "domain": surface.domain,
            "bounds": surface.bounds,
            "halfwidths": surface.halfwidths,
            "retrieval_cutoff": surface.retrieval_cutoff,
            "elements": ELEMENTS,
            "non_fe": NON_FE_ELEMENTS,
            "supply_risk": EU2023_SUPPLY_RISK,
            "target_yield_mpa": TARGET_YIELD_MPA,
            "uncertainty_k": UNCERTAINTY_K,
        },
        os.path.join(RES_DIR, "stage5_screening_bundle.joblib"),
    )
    _plot_search(trials, training, candidates)
    _plot_ti_ni(df_wt, y, candidates)
    print(f"Stage 5 finished in {time.time() - started:.1f} seconds.")
    return audit, candidates


def _plot_search(trials, training, candidates):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(training["criticality"], training["lcb_mpa"], s=12, c="#d5d8dc", label="Training alloys, in-sample")
    for seed, color in ((42, "#1a5276"), (43, "#b9770e")):
        part = trials[trials["seed"] == seed]
        ax.scatter(part["criticality"], part["lcb_mpa"], s=10, alpha=0.25, c=color, label=f"Trials, seed {seed}")
    if len(candidates):
        ax.scatter(candidates["criticality"], candidates["lcb_mpa"], s=36, c="#922b21", label="Top feasible", zorder=3)
    ax.axhline(TARGET_YIELD_MPA, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Criticality index (wt% × EU 2023 supply risk)")
    ax.set_ylabel("Lower confidence bound (MPa)")
    ax.set_title("In-sample screening surface")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "fig22_inverse_search.png"))
    plt.close(fig)


def _plot_ti_ni(df_wt, y, candidates):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(df_wt["Ni"], df_wt["Ti"], c=y, cmap="viridis", s=16)
    fig.colorbar(sc, ax=ax, label="Measured yield strength (MPa)")
    if len(candidates):
        ax.scatter(candidates["Ni"], candidates["Ti"], s=40, facecolors="none", edgecolors="#922b21", linewidths=1.2, label="Top feasible")
        ax.legend(frameon=False)
    ax.set_xlabel("Ni wt%")
    ax.set_ylabel("Ti wt%")
    ax.set_title("Candidates against the training alloys")
    fig.savefig(os.path.join(FIG_DIR, "fig23_candidates_ti_ni.png"))
    plt.close(fig)


if __name__ == "__main__":
    run_stage5()
