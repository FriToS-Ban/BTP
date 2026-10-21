"""Stage 4. SHAP interpretation of the composition-only yield-strength models.

Headline explainer, declared before any SHAP value is computed:
default random forest on the 13 non-iron weight-percent features,
tree-path-dependent TreeSHAP, evaluated on the outer test rows of the
same 25 grouped splits used in Stage 3.

Tuned random forest and tuned XGBoost are robustness checks. Their
hyperparameters are read from results/stage3_nested_hyperparameters_per_split.csv
and are not retuned. On the 13-feature models those hyperparameters were
chosen on 14 features, and the script says so.

A metallurgical label can be "consistent" or "inconsistent" only when
default RF, tuned RF, and tuned XGB agree. Otherwise the label is "unclear".
SHAP is an attribution of the model, not a proof of a mechanism.
"""

import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from btp_common import (
    ELEMENTS,
    FAMILY_COLORS,
    FIG_DIR,
    NON_FE_ELEMENTS,
    RANDOM_SEED,
    RES_DIR,
    RHO_MIN,
    PRESENCE_FLOOR_WT,
    assign_family,
    load_data,
    make_repeated_group_splits,
    nadeau_bengio,
)

DEPENDENCE_ELEMENTS = ["Ti", "Ni", "Co", "Cr", "Mo", "C", "Al", "Si"]
INTERACTION_PAIRS = [("Ni", "Ti"), ("Co", "Mo"), ("Cr", "Ni"), ("C", "Cr")]
HEADLINE = "default_rf_13"
CONSENSUS_MODELS = ["default_rf_13", "tuned_rf_13", "tuned_xgb_13"]

# Directions declared before the run. Only these can receive a
# consistent or inconsistent label. Every other element stays unclear.
EXPECTATIONS = {
    "Ti": ("positive", "Early Ni3Ti-type precipitation in Fe-Ni martensite is the usual expectation. Sha, Cerezo and Smith, Metall. Trans. A 1993."),
    "Ni": ("positive", "Nickel stabilizes the martensite and supplies Ni3Ti and Ni3Mo-type precipitates. Floreen, Metall. Rev. 1968."),
    "Co": ("weak_positive", "Cobalt is reported to lower molybdenum solubility and to amplify molybdenum hardening, while precipitating little itself. Floreen and Speich 1964."),
    "Mo": ("positive", "Ni3Mo and later Fe7Mo6 or Fe2Mo are the usual molybdenum-bearing precipitates. Sha et al. 1993."),
    "Cr": ("confounded", "A global negative association is expected to be the stainless-versus-maraging mix, not a within-steel weakening by chromium."),
    "Al": ("conditional_positive", "An Ni-Al contribution is expected only where nickel is already high (Ni >= 12 wt%). Floreen 1964."),
    "C": ("no_sign", "No global sign was declared. Carbon is an interstitial and a carbide former, and it also separates the small tool-steel group."),
    "Si": ("no_sign", "No global sign was declared. A G-phase contribution is reported in some maraging alloys, but it is not assumed here."),
    "Mn": ("no_sign", "No global sign was declared."),
    "Nb": ("no_sign", "No global sign was declared. Niobium is present in nearly every row."),
    "V": ("no_sign", "No global sign was declared. Vanadium is present in nearly every row."),
    "N": ("sparse", "Nitrogen is sparse. No mechanism test is attempted."),
    "W": ("sparse", "Tungsten is sparse. No mechanism test is attempted."),
    "Fe": ("closure", "Iron closes the composition. Its attribution is not a separate mechanism."),
}


def _rho(x, s):
    if len(x) < 30 or np.std(x) < 1e-8 or np.std(s) < 1e-8:
        return None
    rho, _ = spearmanr(x, s)
    if np.isnan(rho):
        return None
    return float(rho)


def verdict_for_element(direction, rho, rho_ss, rho_mg, rho_high_ni, important, presence_frac):
    """Predeclared rules. Synthetic checks below lock the thresholds."""
    if direction == "closure":
        return "unclear", "Iron closes the composition, so this attribution is not read as its own mechanism."
    if direction == "sparse" or presence_frac < 0.20:
        return "unclear", "Present in under 20% of alloys. The attribution is too sparse to test a mechanism."
    if direction == "no_sign":
        return "unclear", "No sign was declared before the run, so the measured association is reported and not scored."
    if not important:
        return "unclear", "Mean absolute SHAP is below the median element, so the association is too small for a mechanism claim."
    if direction in ("positive", "weak_positive"):
        families_do_not_reverse = (rho_ss is None or rho_ss > -RHO_MIN) and (rho_mg is None or rho_mg > -RHO_MIN)
        one_family_repeats = (rho_ss is not None and rho_ss >= RHO_MIN) or (rho_mg is not None and rho_mg >= RHO_MIN)
        if rho is not None and rho >= RHO_MIN and families_do_not_reverse and one_family_repeats:
            return "consistent", "Higher concentration receives positive SHAP globally and inside at least one large family, and the other large family does not reverse it."
        if rho is not None and rho <= -RHO_MIN:
            return "inconsistent", "Higher concentration receives negative SHAP, opposite the predeclared positive expectation."
        return "unclear", "The association does not clear the predeclared Spearman threshold of 0.20 inside a large family."
    if direction == "confounded":
        within_quiet = (rho_ss is None or abs(rho_ss) < RHO_MIN) and (rho_mg is None or abs(rho_mg) < RHO_MIN)
        if rho is not None and rho <= -RHO_MIN and within_quiet:
            return "consistent", "The global negative association is not repeated inside the large families, which fits a between-family mix. It does not mean chromium lowers the strength of a given steel."
        return "unclear", "The chromium pattern does not match the predeclared between-family pattern cleanly."
    if direction == "conditional_positive":
        if rho_high_ni is not None and rho_high_ni >= RHO_MIN and (rho is None or rho > -RHO_MIN):
            return "consistent", "Where nickel is at least 12 wt%, higher aluminium receives positive SHAP."
        if rho_high_ni is not None and rho_high_ni <= -RHO_MIN:
            return "inconsistent", "Where nickel is at least 12 wt%, higher aluminium receives negative SHAP, opposite the Ni-Al expectation."
        return "unclear", "The high-nickel aluminium association does not clear the predeclared threshold."
    return "unclear", "No rule fired."


def _self_check_rules():
    v, _ = verdict_for_element("positive", 0.55, 0.40, 0.30, None, True, 0.9)
    assert v == "consistent"
    v, _ = verdict_for_element("positive", -0.40, -0.30, -0.10, None, True, 0.9)
    assert v == "inconsistent"
    v, _ = verdict_for_element("positive", 0.55, 0.05, 0.05, None, True, 0.9)
    assert v == "unclear"
    v, _ = verdict_for_element("positive", 0.55, 0.40, 0.30, None, False, 0.9)
    assert v == "unclear"
    v, _ = verdict_for_element("sparse", 0.80, 0.80, None, None, True, 0.05)
    assert v == "unclear"
    v, _ = verdict_for_element("confounded", -0.45, 0.05, -0.05, None, True, 0.9)
    assert v == "consistent"
    v, _ = verdict_for_element("confounded", -0.45, -0.35, 0.0, None, True, 0.9)
    assert v == "unclear"
    v, _ = verdict_for_element("conditional_positive", 0.10, None, None, 0.40, True, 0.8)
    assert v == "consistent"
    v, _ = verdict_for_element("no_sign", 0.90, 0.90, 0.90, None, True, 0.9)
    assert v == "unclear"
    v, _ = verdict_for_element("closure", 0.20, None, None, None, True, 1.0)
    assert v == "unclear"


def parse_max_features(value):
    text = str(value)
    if text == "sqrt":
        return "sqrt"
    return float(text)


def make_default_rf():
    return RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=4,
        random_state=RANDOM_SEED, n_jobs=-1,
    )


def make_default_xgb():
    return XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    )


def make_tuned_rf(row):
    return RandomForestRegressor(
        n_estimators=int(row.RF_n_estimators),
        max_depth=int(row.RF_max_depth),
        min_samples_split=int(row.RF_min_samples_split),
        max_features=parse_max_features(row.RF_max_features),
        random_state=RANDOM_SEED, n_jobs=-1,
    )


def make_tuned_xgb(row):
    return XGBRegressor(
        n_estimators=int(row.XGB_n_estimators),
        max_depth=int(row.XGB_max_depth),
        learning_rate=float(row.XGB_learning_rate),
        subsample=float(row.XGB_subsample),
        colsample_bytree=float(row.XGB_colsample_bytree),
        reg_alpha=float(row.XGB_reg_alpha),
        reg_lambda=float(row.XGB_reg_lambda),
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    )


def rmse_scorer(estimator, x_values, y_values):
    return -root_mean_squared_error(y_values, estimator.predict(x_values))


def shap_matrix(model, x_values, kind):
    if kind == "rf":
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        values = np.asarray(explainer.shap_values(x_values))
        base = explainer.expected_value
        base = float(np.ravel(base)[0])
        return values, np.full(len(x_values), base)
    contrib = model.get_booster().predict(xgb.DMatrix(x_values), pred_contribs=True)
    return contrib[:, :-1], contrib[:, -1]


def interventional_matrix(model, background, x_values):
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    values = np.asarray(explainer.shap_values(x_values))
    return values


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def run_stage4():
    _self_check_rules()
    apply_style()
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    started = time.time()
    print("Stage 4 SHAP. Rules self-check passed.")

    df_wt, y, groups, _formulas = load_data()
    families = assign_family(df_wt)
    splits = make_repeated_group_splits(groups)
    hyper = pd.read_csv(os.path.join(RES_DIR, "stage3_nested_hyperparameters_per_split.csv"))
    if len(hyper) != len(splits):
        raise RuntimeError("Hyperparameter table does not have 25 rows for 25 splits.")

    x14 = df_wt[ELEMENTS].to_numpy(dtype=float)
    x13 = df_wt[NON_FE_ELEMENTS].to_numpy(dtype=float)
    names14 = list(ELEMENTS)
    names13 = list(NON_FE_ELEMENTS)

    model_names = [
        "default_rf_14", "default_rf_13",
        "tuned_rf_14", "tuned_rf_13",
        "tuned_xgb_14", "tuned_xgb_13",
    ]
    # Accumulated SHAP, summed over the repeats in which an alloy is tested.
    shap_sum = {name: np.zeros((len(y), 14 if name.endswith("14") else 13)) for name in model_names}
    shap_count = {name: np.zeros(len(y)) for name in model_names}
    fold_importance = []
    metric_rows = []
    perm_rows = []
    interaction_sum = np.zeros((13, 13))
    interaction_count = 0
    inter_rank_rows = []
    additivity_errors = []

    for split_idx, ((tr_idx, val_idx), row) in enumerate(zip(splits, hyper.itertuples(index=False)), start=1):
        if len(tr_idx) != int(row.N_Train_Alloys) or int(row.Repeat) != ((split_idx - 1) // 5) + 1:
            raise RuntimeError(f"Split {split_idx} does not match the Stage 3 hyperparameter row.")
        builders = {
            "default_rf_14": (make_default_rf, x14, names14, "rf"),
            "default_rf_13": (make_default_rf, x13, names13, "rf"),
            "tuned_rf_14": (lambda r=row: make_tuned_rf(r), x14, names14, "rf"),
            "tuned_rf_13": (lambda r=row: make_tuned_rf(r), x13, names13, "rf"),
            "tuned_xgb_14": (lambda r=row: make_tuned_xgb(r), x14, names14, "xgb"),
            "tuned_xgb_13": (lambda r=row: make_tuned_xgb(r), x13, names13, "xgb"),
        }
        print(f"  split {split_idx}/25")
        for name, (builder, x_all, feature_names, kind) in builders.items():
            model = builder()
            model.fit(x_all[tr_idx], y[tr_idx])
            pred = model.predict(x_all[val_idx])
            metric_rows.append({
                "split": split_idx,
                "model": name,
                "rmse": root_mean_squared_error(y[val_idx], pred),
                "mae": mean_absolute_error(y[val_idx], pred),
                "r2": r2_score(y[val_idx], pred),
                "n_train": len(tr_idx),
                "n_test": len(val_idx),
            })
            values, base = shap_matrix(model, x_all[val_idx], kind)
            reconstructed = values.sum(axis=1) + base
            additivity_errors.append(float(np.max(np.abs(reconstructed - pred))))
            shap_sum[name][val_idx, : values.shape[1]] += values
            shap_count[name][val_idx] += 1
            mean_abs = np.mean(np.abs(values), axis=0)
            for j, el in enumerate(feature_names):
                fold_importance.append({
                    "split": split_idx, "model": name, "element": el, "mean_abs_shap": float(mean_abs[j]),
                })
            perm = permutation_importance(
                model, x_all[val_idx], y[val_idx], scoring=rmse_scorer,
                n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1,
            )
            for j, el in enumerate(feature_names):
                perm_rows.append({
                    "split": split_idx, "model": name, "element": el,
                    "rmse_increase_mpa": float(perm.importances_mean[j]),
                })
            if name == HEADLINE:
                inter = shap.TreeExplainer(
                    model, feature_perturbation="tree_path_dependent"
                ).shap_interaction_values(x_all[val_idx])
                interaction_sum += np.mean(np.asarray(inter), axis=0)
                interaction_count += 1
                rng_bg = np.random.RandomState(RANDOM_SEED + split_idx)
                take = rng_bg.choice(len(tr_idx), size=80, replace=False)
                iv = interventional_matrix(model, x_all[tr_idx][take], x_all[val_idx])
                rho_chk, p_chk = spearmanr(mean_abs, np.mean(np.abs(iv), axis=0))
                inter_rank_rows.append({
                    "split": split_idx,
                    "spearman_path_vs_interventional": float(rho_chk),
                    "p_value": float(p_chk),
                })

    if max(additivity_errors) > 1.0:
        raise RuntimeError(f"SHAP additivity failed. Worst reconstruction error {max(additivity_errors):.3f} MPa.")
    print(f"Additivity worst error {max(additivity_errors):.4e} MPa.")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(RES_DIR, "stage4_cv_metrics_by_split.csv"), index=False)
    summary_rows = []
    for name, part in metrics.groupby("model"):
        summary_rows.append({
            "model": name,
            "rmse_mean": part["rmse"].mean(),
            "rmse_std": part["rmse"].std(ddof=1),
            "mae_mean": part["mae"].mean(),
            "mae_std": part["mae"].std(ddof=1),
            "r2_mean": part["r2"].mean(),
            "r2_std": part["r2"].std(ddof=1),
        })
    summary = pd.DataFrame(summary_rows).sort_values("model")
    summary.to_csv(os.path.join(RES_DIR, "stage4_nofe_cv_metrics.csv"), index=False)
    print(summary.to_string(index=False))

    # Archived Stage 3 default-RF RMSE. A gap of a few MPa is a fresh execution
    # of the same protocol (sklearn / BLAS), not a new result. A gap above
    # 10 MPa means the splits or the model are not the ones Stage 3 used.
    archived = 158.30457041324203
    ref = float(metrics.loc[metrics["model"] == "default_rf_14", "rmse"].mean())
    gap = ref - archived
    if abs(gap) > 10.0:
        raise RuntimeError(
            f"Default RF 14-feature RMSE {ref:.3f} is {gap:+.3f} MPa from the archived Stage 3 value {archived:.3f}."
        )
    print(f"Default RF 14-feature RMSE {ref:.3f} versus archived Stage 3 {archived:.3f} (gap {gap:+.3f} MPa).")
    pd.DataFrame([{
        "model": "default_rf_14",
        "this_run_rmse": ref,
        "archived_stage3_rmse": archived,
        "gap_mpa": gap,
        "note": "Same protocol. A gap of a few MPa is execution difference, judged against a 10 MPa stop.",
    }]).to_csv(os.path.join(RES_DIR, "stage4_reproduction_gap.csv"), index=False)

    n_train = float(metrics["n_train"].mean())
    n_test = float(metrics["n_test"].mean())
    paired_rows = []
    pairs = [
        ("default_rf_14", "default_rf_13"),
        ("tuned_rf_14", "tuned_rf_13"),
        ("tuned_xgb_14", "tuned_xgb_13"),
    ]
    wide = metrics.pivot(index="split", columns="model", values="rmse")
    for left, right in pairs:
        delta = wide[right] - wide[left]
        mean, low, high, p_value = nadeau_bengio(delta.to_numpy(), n_train, n_test)
        paired_rows.append({
            "comparison": f"{right} minus {left}",
            "mean_delta_rmse_mpa": mean,
            "ci95_low": low,
            "ci95_high": high,
            "p_corrected": p_value,
            "n_train_mean": n_train,
            "n_test_mean": n_test,
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(os.path.join(RES_DIR, "stage4_nofe_paired_tests.csv"), index=False)
    print(paired.to_string(index=False))

    fold_df = pd.DataFrame(fold_importance)
    fold_df.to_csv(os.path.join(RES_DIR, "stage4_shap_stability.csv"), index=False)
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(os.path.join(RES_DIR, "stage4_permutation_importance.csv"), index=False)
    pd.DataFrame(inter_rank_rows).to_csv(
        os.path.join(RES_DIR, "stage4_interventional_rank_check.csv"), index=False
    )

    # Alloy-averaged SHAP. Each alloy is a test row once per repeat.
    averaged = {}
    long_parts = []
    for name in model_names:
        counts = np.maximum(shap_count[name], 1.0)[:, None]
        avg = shap_sum[name] / counts
        width = avg.shape[1]
        feature_names = names14 if width == 14 else names13
        averaged[name] = pd.DataFrame(avg[:, :width], columns=feature_names)
        part = averaged[name].copy()
        part.insert(0, "alloy_idx", np.arange(len(y)))
        part.insert(1, "model", name)
        part.insert(2, "family", families)
        part.insert(3, "y_true", y)
        long_parts.append(part)
    long_df = pd.concat(long_parts, ignore_index=True)
    long_df.to_csv(os.path.join(RES_DIR, "stage4_shap_oof_alloy_mean.csv"), index=False)

    # Stability of the importance vector across folds.
    stability_rows = []
    for name, part in fold_df.groupby("model"):
        vectors = []
        for _split, sub in part.groupby("split"):
            sub = sub.set_index("element")
            order = names13 if name.endswith("13") else names14
            vectors.append(sub.loc[order, "mean_abs_shap"].to_numpy())
        mat = np.vstack(vectors)
        rhos = []
        for i in range(len(mat)):
            for j in range(i + 1, len(mat)):
                rho, _ = spearmanr(mat[i], mat[j])
                rhos.append(rho)
        order = names13 if name.endswith("13") else names14
        for j, el in enumerate(order):
            stability_rows.append({
                "model": name,
                "element": el,
                "mean_abs_shap_across_folds": float(mat[:, j].mean()),
                "std_abs_shap_across_folds": float(mat[:, j].std(ddof=1)),
                "median_pairwise_rank_spearman": float(np.median(rhos)),
                "min_pairwise_rank_spearman": float(np.min(rhos)),
            })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(os.path.join(RES_DIR, "stage4_shap_rank_stability.csv"), index=False)

    headline = averaged[HEADLINE]
    presence = (df_wt[names13] > PRESENCE_FLOOR_WT).mean()
    mean_abs_headline = headline.abs().mean()
    importance_cut = float(mean_abs_headline.median())

    def model_stats(name):
        frame = averaged[name]
        stats = {}
        for el in frame.columns:
            if el == "Fe":
                stats[el] = {"rho": None, "rho_ss": None, "rho_mg": None, "rho_high_ni": None}
                continue
            wt = df_wt[el].to_numpy()
            sv = frame[el].to_numpy()
            stats[el] = {
                "rho": _rho(wt, sv),
                "rho_ss": _rho(wt[families == "Stainless"], sv[families == "Stainless"]),
                "rho_mg": _rho(wt[families == "Maraging"], sv[families == "Maraging"]),
                "rho_high_ni": _rho(wt[df_wt["Ni"].to_numpy() >= 12.0], sv[df_wt["Ni"].to_numpy() >= 12.0]),
            }
        return stats

    per_model_stats = {name: model_stats(name) for name in CONSENSUS_MODELS}
    sanity_rows = []
    for el in ELEMENTS:
        direction, expectation = EXPECTATIONS[el]
        votes = []
        vote_detail = {}
        for name in CONSENSUS_MODELS:
            if el not in averaged[name].columns and el != "Fe":
                continue
            if el == "Fe":
                vote, reason = verdict_for_element("closure", None, None, None, None, False, 1.0)
                votes.append(vote)
                vote_detail[name] = vote
                continue
            st = per_model_stats[name][el]
            important = bool(averaged[name][el].abs().mean() >= averaged[name].abs().mean().median())
            vote, reason = verdict_for_element(
                direction, st["rho"], st["rho_ss"], st["rho_mg"], st["rho_high_ni"],
                important, float(presence.get(el, 0.0)),
            )
            votes.append(vote)
            vote_detail[name] = vote
        if el == "Fe":
            consensus, consensus_reason = votes[0], reason
        elif len(set(votes)) == 1:
            consensus = votes[0]
            consensus_reason = reason + " Default RF, tuned RF, and tuned XGB agree."
        else:
            consensus = "unclear"
            consensus_reason = (
                "The three 13-feature models do not agree: "
                + ", ".join(f"{k}={v}" for k, v in vote_detail.items())
                + "."
            )
        st = per_model_stats[HEADLINE].get(el, {})
        sanity_rows.append({
            "element": el,
            "expectation": expectation,
            "predeclared_direction": direction,
            "presence_frac": float(presence[el]) if el in presence.index else 1.0,
            "headline_mean_abs_shap": float(mean_abs_headline[el]) if el in mean_abs_headline.index else np.nan,
            "headline_rho": st.get("rho"),
            "headline_rho_stainless": st.get("rho_ss"),
            "headline_rho_maraging": st.get("rho_mg"),
            "headline_rho_ni_ge_12": st.get("rho_high_ni"),
            "vote_default_rf_13": vote_detail.get("default_rf_13"),
            "vote_tuned_rf_13": vote_detail.get("tuned_rf_13"),
            "vote_tuned_xgb_13": vote_detail.get("tuned_xgb_13"),
            "consensus": consensus,
            "reason": consensus_reason,
            "importance_median_cutoff_mpa": importance_cut,
            "rho_threshold": RHO_MIN,
        })
    sanity = pd.DataFrame(sanity_rows)
    sanity.to_csv(os.path.join(RES_DIR, "stage4_metallurgical_sanity.csv"), index=False)
    print(sanity[["element", "consensus", "headline_rho", "vote_default_rf_13", "vote_tuned_rf_13", "vote_tuned_xgb_13"]].to_string(index=False))

    # Family mean |SHAP| on the headline model. Tool steels stay in the table and are flagged.
    family_rows = []
    for fam in ["Stainless", "Maraging", "Structural", "Tool"]:
        mask = families == fam
        block = headline.loc[mask].abs().mean()
        for el, value in block.items():
            family_rows.append({
                "family": fam,
                "n": int(mask.sum()),
                "underpowered": bool(mask.sum() < 30),
                "element": el,
                "mean_abs_shap": float(value),
            })
    family_df = pd.DataFrame(family_rows)
    family_df.to_csv(os.path.join(RES_DIR, "stage4_family_shap.csv"), index=False)

    interaction = interaction_sum / interaction_count
    inter_rows = []
    for i, left in enumerate(names13):
        for j, right in enumerate(names13):
            if j <= i:
                continue
            inter_rows.append({
                "element_a": left,
                "element_b": right,
                "mean_interaction_mpa": float(interaction[i, j]),
                "predeclared_pair": (left, right) in INTERACTION_PAIRS or (right, left) in INTERACTION_PAIRS,
            })
    interaction_df = pd.DataFrame(inter_rows).sort_values("mean_interaction_mpa", key=np.abs, ascending=False)
    interaction_df.to_csv(os.path.join(RES_DIR, "stage4_interactions.csv"), index=False)

    # Explanatory full-data PDP. Not a cross-validated accuracy claim.
    full_model = make_default_rf()
    full_model.fit(x13, y)
    pdp_rows = []
    for left, right in INTERACTION_PAIRS:
        i, j = names13.index(left), names13.index(right)
        bunch = partial_dependence(full_model, x13, features=[(i, j)], kind="average", grid_resolution=12)
        grid = bunch["average"][0]
        # Corner contrast on the PDP grid: high/high - high/low - low/high + low/low.
        contrast = float(grid[-1, -1] - grid[-1, 0] - grid[0, -1] + grid[0, 0])
        shap_value = float(interaction[i, j])
        pdp_rows.append({
            "element_a": left,
            "element_b": right,
            "pdp_corner_contrast_mpa": contrast,
            "mean_shap_interaction_mpa": shap_value,
            "signs_agree": bool(np.sign(contrast) == np.sign(shap_value) and contrast != 0 and shap_value != 0),
            "model": "full-data default RF, 13 features, explanatory only",
        })
    pdp_df = pd.DataFrame(pdp_rows)
    pdp_df.to_csv(os.path.join(RES_DIR, "stage4_pdp_contrasts.csv"), index=False)
    print(pdp_df.to_string(index=False))

    _plot_bars(stability)
    _plot_beeswarm(headline, df_wt, names13)
    _plot_dependence(headline, df_wt, families)
    _plot_stability(fold_df)
    _plot_family(family_df)
    _plot_interactions(interaction, names13)
    _plot_pdp(full_model, x13, names13)
    _plot_perm_vs_shap(perm_df, stability)
    print(f"Stage 4 finished in {time.time() - started:.1f} seconds.")
    return summary, sanity, paired


def _plot_bars(stability):
    focus = stability[stability["model"].isin(CONSENSUS_MODELS)].copy()
    elements = names_order(focus)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    xpos = np.arange(len(elements))
    for offset, name, color in zip(
        (-width, 0, width), CONSENSUS_MODELS, ("#1a5276", "#b9770e", "#196f3d")
    ):
        sub = focus[focus["model"] == name].set_index("element").loc[elements]
        ax.bar(xpos + offset, sub["mean_abs_shap_across_folds"], width=width, color=color, label=name)
    ax.set_xticks(xpos)
    ax.set_xticklabels(elements)
    ax.set_ylabel("Mean |SHAP| across 25 folds (MPa)")
    ax.set_title("13-feature models, outer-test TreeSHAP")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "fig14_shap_importance_bars.png"))
    plt.close(fig)


def names_order(focus):
    headline = focus[focus["model"] == HEADLINE].sort_values("mean_abs_shap_across_folds", ascending=False)
    return headline["element"].tolist()


def _plot_beeswarm(headline, df_wt, feature_names):
    order = headline.abs().mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.RandomState(RANDOM_SEED)
    for pos, el in enumerate(order):
        values = headline[el].to_numpy()
        color = df_wt[el].to_numpy()
        color = (color - color.min()) / (color.max() - color.min() + 1e-12)
        jitter = rng.uniform(-0.28, 0.28, size=len(values))
        ax.scatter(values, np.full(len(values), pos) + jitter, c=color, cmap="coolwarm", s=8, alpha=0.75)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Alloy-averaged SHAP on outer-test rows (MPa)")
    ax.set_title("Default random forest, 13 non-Fe features")
    fig.savefig(os.path.join(FIG_DIR, "fig15_shap_beeswarm_default_rf.png"))
    plt.close(fig)


def _plot_dependence(headline, df_wt, families):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, el in zip(axes.ravel(), DEPENDENCE_ELEMENTS):
        for fam, color in FAMILY_COLORS.items():
            mask = families == fam
            ax.scatter(df_wt.loc[mask, el], headline.loc[mask, el], s=10, alpha=0.75, c=color, label=fam)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(el)
        ax.set_xlabel("wt%")
        ax.set_ylabel("SHAP (MPa)")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, label=f, markersize=6) for f, c in FAMILY_COLORS.items()]
    fig.legend(handles=handles, frameon=False, loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig16_shap_dependence.png"))
    plt.close(fig)


def _plot_stability(fold_df):
    sub = fold_df[fold_df["model"] == HEADLINE]
    order = sub.groupby("element")["mean_abs_shap"].mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = [sub.loc[sub["element"] == el, "mean_abs_shap"].to_numpy() for el in order]
    try:
        ax.boxplot(data, tick_labels=order, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("Fold mean |SHAP| (MPa)")
    ax.set_title("Stability across 25 grouped folds, default RF, 13 features")
    fig.savefig(os.path.join(FIG_DIR, "fig17_shap_stability_folds.png"))
    plt.close(fig)


def _plot_family(family_df):
    pivot = family_df.pivot(index="family", columns="element", values="mean_abs_shap")
    pivot = pivot.loc[["Stainless", "Maraging", "Structural", "Tool"]]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"{idx} (n={int(family_df.loc[family_df.family == idx, 'n'].iloc[0])})" for idx in pivot.index])
    fig.colorbar(image, ax=ax, label="Mean |SHAP| (MPa)")
    ax.set_title("Within-family |SHAP|. Tool steels, n=17, are underpowered.")
    fig.savefig(os.path.join(FIG_DIR, "fig18_shap_family_heatmap.png"))
    plt.close(fig)


def _plot_interactions(interaction, feature_names):
    fig, ax = plt.subplots(figsize=(7, 6))
    off = interaction.copy()
    np.fill_diagonal(off, 0.0)
    limit = np.max(np.abs(off))
    image = ax.imshow(off, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    fig.colorbar(image, ax=ax, label="Mean SHAP interaction (MPa)")
    ax.set_title("Default RF, 13 features, averaged over 25 test folds")
    fig.savefig(os.path.join(FIG_DIR, "fig19_shap_interactions.png"))
    plt.close(fig)


def _plot_pdp(model, x_values, feature_names):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, (left, right) in zip(axes.ravel(), INTERACTION_PAIRS):
        i, j = feature_names.index(left), feature_names.index(right)
        bunch = partial_dependence(model, x_values, features=[(i, j)], kind="average", grid_resolution=20)
        grid = bunch["average"][0]
        xs, ys = bunch["grid_values"]
        image = ax.contourf(xs, ys, grid.T, levels=12, cmap="viridis")
        ax.set_xlabel(left + " wt%")
        ax.set_ylabel(right + " wt%")
        fig.colorbar(image, ax=ax, label="Predicted MPa")
    fig.suptitle("Full-data PDP, explanatory only, not a CV score")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig20_partial_dependence_pairs.png"))
    plt.close(fig)


def _plot_perm_vs_shap(perm_df, stability):
    perm = perm_df[perm_df["model"] == HEADLINE].groupby("element")["rmse_increase_mpa"].mean()
    shap_imp = stability[stability["model"] == HEADLINE].set_index("element")["mean_abs_shap_across_folds"]
    shared = [el for el in perm.index if el in shap_imp.index]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(perm.loc[shared], shap_imp.loc[shared], c="#1a5276")
    for el in shared:
        ax.annotate(el, (perm.loc[el], shap_imp.loc[el]), fontsize=8)
    rho, p_value = spearmanr(perm.loc[shared], shap_imp.loc[shared])
    ax.set_xlabel("Permutation importance, RMSE increase (MPa)")
    ax.set_ylabel("Mean |SHAP| (MPa)")
    ax.set_title(f"Default RF, 13 features\nSpearman {rho:.2f}, p={p_value:.3f}")
    fig.savefig(os.path.join(FIG_DIR, "fig21_shap_vs_permutation.png"))
    plt.close(fig)


if __name__ == "__main__":
    run_stage4()
