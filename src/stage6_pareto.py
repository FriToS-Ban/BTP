"""Stage 6. Two-objective front, plus a weldability indicator that is audited and not optimized.

Objectives, declared before the search:
- Maximize the same lower confidence bound as Stage 5.
- Minimize the same EU 2023 criticality index.

The IIW carbon equivalent and Pcm are computed and then set aside as objectives.
Their published validity window (CET bounds in EN 1011-2 method B) does not
cover the high nickel, molybdenum, and chromium levels in this dataset. Using
them as a third objective would treat an out-of-range indicator as if it were
measured weldability. The script counts how often the training alloys fall
outside that window and stops there.

Search: NSGA-II, 600 trials, seed 42, same local moves as Stage 5.
Feasible means predicted yield at least 1500 MPa, inside the applicability
domain, and iron inside its training range.
"""

import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from btp_common import (
    ELEMENTS,
    FIG_DIR,
    NON_FE_ELEMENTS,
    RES_DIR,
    TARGET_YIELD_MPA,
    cet_violation_frame,
    load_data,
)
from stage5_inverse_design import fe_violation, load_surface, propose

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 600


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def run_stage6():
    apply_style()
    started = time.time()
    df_wt, y, _groups, _formulas = load_data()
    surface = load_surface(os.path.join(RES_DIR, "stage5_screening_bundle.joblib"), df_wt, y)

    flags = cet_violation_frame(df_wt)
    validity_rows = []
    for column in flags.columns:
        if column == "cet_valid":
            continue
        validity_rows.append({
            "bound": column.replace("cet_outside_", ""),
            "n_outside": int(flags[column].sum()),
            "fraction_outside": float(flags[column].mean()),
        })
    validity = pd.DataFrame(validity_rows).sort_values("n_outside", ascending=False)
    validity.to_csv(os.path.join(RES_DIR, "stage6_cet_validity_counts.csv"), index=False)
    n_valid = int(flags["cet_valid"].sum())
    print(f"Training alloys inside the published CET window: {n_valid} of {len(df_wt)}.")
    print(validity.to_string(index=False))

    def objective(trial):
        _base, values = propose(trial, surface)
        frame = pd.DataFrame([values])[ELEMENTS]
        scored = surface.score_frame(frame).iloc[0]
        violation = fe_violation(values, surface)
        trial.set_user_attr("pred_mpa", float(scored["pred_mpa"]))
        trial.set_user_attr("lcb_mpa", float(scored["lcb_mpa"]))
        trial.set_user_attr("tree_std_mpa", float(scored["tree_std_mpa"]))
        trial.set_user_attr("gp_std_mpa", float(scored["gp_std_mpa"]))
        trial.set_user_attr("criticality", float(scored["criticality"]))
        trial.set_user_attr("inside_domain", bool(scored["inside_domain"]))
        trial.set_user_attr("fe_violation", float(violation))
        trial.set_user_attr("domain_label", scored["domain_label"])
        trial.set_user_attr("nearest_dist_std", float(scored["nearest_dist_std"]))
        trial.set_user_attr("nearest_train_idx", int(scored["nearest_train_idx"]))
        trial.set_user_attr("ce_iiw", float(scored["ce_iiw"]))
        trial.set_user_attr("pcm", float(scored["pcm"]))
        trial.set_user_attr("cet_valid", bool(scored["cet_valid"]))
        trial.set_user_attr("family", scored["family"])
        trial.set_user_attr("composition", values)
        return float(scored["lcb_mpa"]), float(scored["criticality"])

    def constraints(trial):
        return (
            TARGET_YIELD_MPA - trial.user_attrs["pred_mpa"],
            0.0 if trial.user_attrs["inside_domain"] else 1.0,
            trial.user_attrs["fe_violation"],
        )

    sampler = optuna.samplers.NSGAIISampler(seed=42, population_size=50, constraints_func=constraints)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        feasible = bool(
            trial.user_attrs["pred_mpa"] >= TARGET_YIELD_MPA
            and trial.user_attrs["inside_domain"]
            and trial.user_attrs["fe_violation"] <= 0
        )
        row = {
            "trial": trial.number,
            "lcb_mpa": trial.user_attrs["lcb_mpa"],
            "pred_mpa": trial.user_attrs["pred_mpa"],
            "tree_std_mpa": trial.user_attrs["tree_std_mpa"],
            "gp_std_mpa": trial.user_attrs["gp_std_mpa"],
            "criticality": trial.user_attrs["criticality"],
            "ce_iiw": trial.user_attrs["ce_iiw"],
            "pcm": trial.user_attrs["pcm"],
            "cet_valid": trial.user_attrs["cet_valid"],
            "inside_domain": trial.user_attrs["inside_domain"],
            "domain_label": trial.user_attrs["domain_label"],
            "nearest_dist_std": trial.user_attrs["nearest_dist_std"],
            "nearest_train_idx": trial.user_attrs["nearest_train_idx"],
            "family": trial.user_attrs["family"],
            "feasible": feasible,
        }
        row.update(trial.user_attrs["composition"])
        rows.append(row)
    all_trials = pd.DataFrame(rows)
    all_trials.to_csv(os.path.join(RES_DIR, "stage6_all_trials.csv"), index=False)

    pareto_numbers = {trial.number for trial in study.best_trials}
    feasible = all_trials[all_trials["feasible"]].copy()
    feasible["on_optuna_front"] = feasible["trial"].isin(pareto_numbers)
    # Recompute the non-dominated set on feasible trials only. Optuna's front
    # can include infeasible trials, and those are not candidates.
    keep = _nondominated(feasible["lcb_mpa"].to_numpy(), feasible["criticality"].to_numpy())
    front = feasible.loc[keep].sort_values("criticality")
    front.to_csv(os.path.join(RES_DIR, "stage6_pareto_front.csv"), index=False)
    print(f"Feasible trials: {len(feasible)} of {len(all_trials)}. Front size: {len(front)}.")
    if len(front):
        print(front[["lcb_mpa", "pred_mpa", "criticality", "domain_label", "family", "cet_valid"]].to_string(index=False))

    audit = pd.DataFrame([{
        "n_trials": N_TRIALS,
        "n_feasible": int(len(feasible)),
        "n_front": int(len(front)),
        "training_cet_valid": n_valid,
        "training_n": int(len(df_wt)),
        "front_cet_valid": int(front["cet_valid"].sum()) if len(front) else 0,
        "front_retrieval": int((front["domain_label"] == "retrieval").sum()) if len(front) else 0,
        "front_variant": int((front["domain_label"] == "in_domain_variant").sum()) if len(front) else 0,
        "ce_was_optimized": False,
    }])
    audit.to_csv(os.path.join(RES_DIR, "stage6_audit.csv"), index=False)

    training = pd.read_csv(os.path.join(RES_DIR, "stage5_training_surface.csv"))
    _plot_front(all_trials, front, training)
    _plot_validity(validity, n_valid, len(df_wt))
    print(f"Stage 6 finished in {time.time() - started:.1f} seconds.")
    return front, audit


def _nondominated(lcb, criticality):
    """Maximize lcb, minimize criticality."""
    n = len(lcb)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        better_or_equal = (lcb >= lcb[i]) & (criticality <= criticality[i])
        strictly = (lcb > lcb[i]) | (criticality < criticality[i])
        if np.any(better_or_equal & strictly):
            keep[i] = False
    return keep


def _plot_front(trials, front, training):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(training["criticality"], training["lcb_mpa"], s=12, c="#d5d8dc", label="Training, in-sample")
    rejected = trials[~trials["feasible"]]
    ax.scatter(rejected["criticality"], rejected["lcb_mpa"], s=10, alpha=0.2, c="#85929e", label="Rejected trials")
    accepted = trials[trials["feasible"]]
    ax.scatter(accepted["criticality"], accepted["lcb_mpa"], s=14, alpha=0.45, c="#1a5276", label="Feasible trials")
    if len(front):
        ordered = front.sort_values("criticality")
        ax.plot(ordered["criticality"], ordered["lcb_mpa"], c="#922b21", lw=1.2, marker="o", label="Feasible front")
    ax.set_xlabel("Criticality index (wt% × EU 2023 supply risk)")
    ax.set_ylabel("Lower confidence bound (MPa)")
    ax.set_title("Feasible front. Carbon equivalent is not an axis.")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "fig24_pareto_front.png"))
    plt.close(fig)


def _plot_validity(validity, n_valid, n_total):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(validity["bound"], validity["n_outside"], color="#1a5276")
    ax.set_xlabel("Training alloys outside the published bound")
    ax.set_title(f"CET window holds for {n_valid} of {n_total} alloys")
    fig.savefig(os.path.join(FIG_DIR, "fig25_carbon_equivalent_validity.png"))
    plt.close(fig)


if __name__ == "__main__":
    run_stage6()
