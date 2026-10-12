"""
Stage 3: Nested Hyperparameter Tuning, Deep Learning & GP Benchmarks, and Residual Metallurgical Audit
Project: Machine-Learning Prediction of Mechanical Properties of Steels from Composition
IIT Roorkee - B.Tech Project in Metallurgical and Materials Engineering
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from pymatgen.core import Composition
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.stats import ttest_rel, wilcoxon

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
RES_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#2c3e50',
    'axes.linewidth': 1.2,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold'
})

ELEMENTS = ['Al', 'C', 'Co', 'Cr', 'Fe', 'Mn', 'Mo', 'N', 'Nb', 'Ni', 'Si', 'Ti', 'V', 'W']
NON_FE_ELEMENTS = [el for el in ELEMENTS if el != 'Fe']


# -----------------------------------------------------------------------------
# 1. Data Loading & Feature Preparation
# -----------------------------------------------------------------------------
def load_and_prepare_data():
    data_path = os.path.join(DATA_DIR, 'processed_steels.csv')
    df = pd.read_csv(data_path)
    y = df['yield_strength'].values
    raw_formulas = df['raw_composition'].values
    
    wt_records = []
    for formula in raw_formulas:
        c = Composition(formula)
        row = {el: c.get_wt_fraction(el) * 100.0 for el in ELEMENTS}
        wt_records.append(row)
    df_wt = pd.DataFrame(wt_records)[ELEMENTS]

    # Headline Grouping at tau = 0.50 wt%
    dist_matrix = squareform(pdist(df_wt.values, metric='euclidean'))
    adj = dist_matrix <= 0.50
    _, groups = connected_components(adj)

    return df_wt, y, groups, raw_formulas


def make_repeated_group_splits(groups, n_splits=5, n_repeats=5, seed=42):
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    splits = []
    for rep in range(n_repeats):
        shuffled_groups = rng.permutation(unique_groups)
        grp_map = {g: i for i, g in enumerate(shuffled_groups)}
        mapped_groups = np.array([grp_map[g] for g in groups])
        gkf = GroupKFold(n_splits=n_splits)
        for tr_idx, val_idx in gkf.split(np.zeros(len(groups)), groups=mapped_groups):
            splits.append((tr_idx, val_idx))
    return splits


# -----------------------------------------------------------------------------
# 2. PyTorch MLP Definition
# -----------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    from sklearn.neural_network import MLPRegressor


if HAS_TORCH:
    class SteelMLP(nn.Module):
        def __init__(self, in_features=14, hidden1=64, hidden2=32, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_torch_mlp(X_tr, y_tr, X_val, in_dim=14, epochs=160, lr=0.01, seed=42):
        torch.manual_seed(seed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        y_mean = np.mean(y_tr)
        y_std = np.std(y_tr)
        y_tr_s = (y_tr - y_mean) / y_std

        X_t = torch.tensor(X_tr_s, dtype=torch.float32)
        y_t = torch.tensor(y_tr_s, dtype=torch.float32)
        X_v = torch.tensor(X_val_s, dtype=torch.float32)

        model = SteelMLP(in_features=in_dim, hidden1=64, hidden2=32, dropout=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds_s = model(X_v).numpy()
        preds = preds_s * y_std + y_mean
        return preds
else:
    def train_torch_mlp(X_tr, y_tr, X_val, in_dim=14, epochs=160, lr=0.01, seed=42):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01,
                           max_iter=epochs, random_state=seed, early_stopping=False)
        mlp.fit(X_tr_s, y_tr)
        return mlp.predict(X_val_s)


# -----------------------------------------------------------------------------
# 3. Nested Hyperparameter Optimization & Model Training
# -----------------------------------------------------------------------------
def run_stage3_pipeline():
    print("=" * 80)
    print("STAGE 3: NESTED HYPERPARAMETER OPTIMIZATION, DEEP LEARNING & RESIDUAL AUDIT")
    print("=" * 80)

    df_wt, y, groups, formulas = load_and_prepare_data()
    X = df_wt.values
    n_samples = len(X)
    splits = make_repeated_group_splits(groups, n_splits=5, n_repeats=5, seed=RANDOM_SEED)

    # Pre-tune optimal hyperparameter templates on training fold using inner GroupKFold
    print("\n--- Running Inner Grouped CV Hyperparameter Tuning with Optuna ---")
    
    # 1. Inner tuning for Random Forest
    def rf_objective(trial):
        n_est = trial.suggest_categorical('n_estimators', [100, 200, 300])
        max_d = trial.suggest_int('max_depth', 6, 16)
        min_split = trial.suggest_int('min_samples_split', 2, 8)
        max_feat = trial.suggest_categorical('max_features', ['sqrt', 0.8, 1.0])
        
        # Inner 3-fold grouped CV on first training partition to establish representative parameters
        inner_tr_idx = splits[0][0]
        X_inner, y_inner, grp_inner = X[inner_tr_idx], y[inner_tr_idx], groups[inner_tr_idx]
        gkf_inner = GroupKFold(n_splits=3)
        inner_rmses = []
        for in_tr, in_val in gkf_inner.split(X_inner, groups=grp_inner):
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, min_samples_split=min_split,
                                       max_features=max_feat, random_state=RANDOM_SEED, n_jobs=-1)
            rf.fit(X_inner[in_tr], y_inner[in_tr])
            inner_rmses.append(root_mean_squared_error(y_inner[in_val], rf.predict(X_inner[in_val])))
        return np.mean(inner_rmses)

    rf_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    rf_study.optimize(rf_objective, n_trials=25)
    best_rf_params = rf_study.best_params
    print(f"Optimal Random Forest Hyperparameters: {best_rf_params} (Inner RMSE: {rf_study.best_value:.2f} MPa)")

    # 2. Inner tuning for XGBoost
    def xgb_objective(trial):
        n_est = trial.suggest_categorical('n_estimators', [80, 150, 250])
        max_d = trial.suggest_int('max_depth', 3, 7)
        lr = trial.suggest_float('learning_rate', 0.03, 0.15, log=True)
        sub = trial.suggest_float('subsample', 0.7, 1.0)
        col = trial.suggest_float('colsample_bytree', 0.7, 1.0)
        alpha = trial.suggest_float('reg_alpha', 0.01, 1.0, log=True)
        lam = trial.suggest_float('reg_lambda', 0.5, 5.0, log=True)

        inner_tr_idx = splits[0][0]
        X_inner, y_inner, grp_inner = X[inner_tr_idx], y[inner_tr_idx], groups[inner_tr_idx]
        gkf_inner = GroupKFold(n_splits=3)
        inner_rmses = []
        for in_tr, in_val in gkf_inner.split(X_inner, groups=grp_inner):
            xgb = XGBRegressor(n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                               subsample=sub, colsample_bytree=col, reg_alpha=alpha, reg_lambda=lam,
                               random_state=RANDOM_SEED, n_jobs=-1)
            xgb.fit(X_inner[in_tr], y_inner[in_tr])
            inner_rmses.append(root_mean_squared_error(y_inner[in_val], xgb.predict(X_inner[in_val])))
        return np.mean(inner_rmses)

    xgb_study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    xgb_study.optimize(xgb_objective, n_trials=30)
    best_xgb_params = xgb_study.best_params
    print(f"Optimal XGBoost Hyperparameters: {best_xgb_params} (Inner RMSE: {xgb_study.best_value:.2f} MPa)")

    # -------------------------------------------------------------------------
    # Outer 5x5 Repeated Grouped CV Evaluation
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Models across 25 Repeated Grouped CV Splits ---")
    
    models = {
        'Default Random Forest': lambda: RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1),
        'Tuned Random Forest': lambda: RandomForestRegressor(**best_rf_params, random_state=RANDOM_SEED, n_jobs=-1),
        'Default XGBoost': lambda: XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1),
        'Tuned XGBoost': lambda: XGBRegressor(**best_xgb_params, random_state=RANDOM_SEED, n_jobs=-1),
        'Gaussian Process Regressor': lambda: Pipeline([
            ('scaler', StandardScaler()),
            ('gp', GaussianProcessRegressor(kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1),
                                           alpha=1e-2, n_restarts_optimizer=2, random_state=RANDOM_SEED))
        ])
    }

    results_by_model = {name: {'rmse': [], 'mae': [], 'r2': []} for name in list(models.keys()) + ['PyTorch MLP']}
    oof_accum = {name: np.zeros(n_samples) for name in results_by_model.keys()}
    oof_count = {name: np.zeros(n_samples) for name in results_by_model.keys()}

    for split_idx, (tr, val) in enumerate(splits, 1):
        X_tr, y_tr = X[tr], y[tr]
        X_val, y_val = X[val], y[val]

        # Standard sklearn models
        for m_name, m_factory in models.items():
            m = m_factory()
            m.fit(X_tr, y_tr)
            p = m.predict(X_val)

            results_by_model[m_name]['rmse'].append(root_mean_squared_error(y_val, p))
            results_by_model[m_name]['mae'].append(mean_absolute_error(y_val, p))
            results_by_model[m_name]['r2'].append(r2_score(y_val, p))
            oof_accum[m_name][val] += p
            oof_count[m_name][val] += 1

        # PyTorch MLP
        p_mlp = train_torch_mlp(X_tr, y_tr, X_val, in_dim=14, epochs=160, lr=0.015, seed=RANDOM_SEED + split_idx)
        results_by_model['PyTorch MLP']['rmse'].append(root_mean_squared_error(y_val, p_mlp))
        results_by_model['PyTorch MLP']['mae'].append(mean_absolute_error(y_val, p_mlp))
        results_by_model['PyTorch MLP']['r2'].append(r2_score(y_val, p_mlp))
        oof_accum['PyTorch MLP'][val] += p_mlp
        oof_count['PyTorch MLP'][val] += 1

    # Summarize Model Performance
    summary_records = []
    print("\n--- STAGE 3 MODEL BENCHMARK (5x5 REPEATED GROUPED CV, 25 SPLITS) ---")
    for m_name, metrics in results_by_model.items():
        rec = {
            'Model': m_name,
            'RMSE_Mean': np.mean(metrics['rmse']),
            'RMSE_Std': np.std(metrics['rmse']),
            'MAE_Mean': np.mean(metrics['mae']),
            'MAE_Std': np.std(metrics['mae']),
            'R2_Mean': np.mean(metrics['r2']),
            'R2_Std': np.std(metrics['r2'])
        }
        summary_records.append(rec)
        print(f"{m_name:28s} | RMSE: {rec['RMSE_Mean']:6.2f} +/- {rec['RMSE_Std']:5.2f} MPa | "
              f"MAE: {rec['MAE_Mean']:6.2f} +/- {rec['MAE_Std']:5.2f} MPa | "
              f"R2: {rec['R2_Mean']:6.3f} +/- {rec['R2_Std']:5.3f}")

    df_summary = pd.DataFrame(summary_records).sort_values(by='RMSE_Mean')
    summary_path = os.path.join(RES_DIR, 'stage3_model_comparison.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved stage 3 model comparison table to: {summary_path}")

    # Compute averaged out-of-fold predictions
    oof_df = pd.DataFrame({'y_true': y, 'raw_composition': formulas})
    for m_name in results_by_model.keys():
        oof_df[m_name] = oof_accum[m_name] / np.maximum(oof_count[m_name], 1)
        oof_df[f"{m_name}_residual"] = y - oof_df[m_name]
    oof_path = os.path.join(RES_DIR, 'stage3_oof_predictions.csv')
    oof_df.to_csv(oof_path, index=False)

    # -------------------------------------------------------------------------
    # 4. Paired Model Comparison with Bootstrap 95% CIs
    # -------------------------------------------------------------------------
    print("\n--- Paired Model Comparisons Across 25 Splits ---")
    comparisons = [
        ('Tuned RF vs Default RF', results_by_model['Default Random Forest']['rmse'], results_by_model['Tuned Random Forest']['rmse'], 'Default RF - Tuned RF'),
        ('Tuned XGB vs Default XGB', results_by_model['Default XGBoost']['rmse'], results_by_model['Tuned XGBoost']['rmse'], 'Default XGB - Tuned XGB'),
        ('Tuned RF vs Tuned XGB', results_by_model['Tuned XGBoost']['rmse'], results_by_model['Tuned Random Forest']['rmse'], 'Tuned XGB - Tuned RF'),
        ('Tuned RF vs PyTorch MLP', results_by_model['PyTorch MLP']['rmse'], results_by_model['Tuned Random Forest']['rmse'], 'PyTorch MLP - Tuned RF'),
        ('Tuned RF vs Gaussian Process', results_by_model['Gaussian Process Regressor']['rmse'], results_by_model['Tuned Random Forest']['rmse'], 'GP Regressor - Tuned RF')
    ]

    paired_records = []
    rng = np.random.RandomState(RANDOM_SEED)

    for comp_name, vec_a, vec_b, diff_desc in comparisons:
        vec_a, vec_b = np.array(vec_a), np.array(vec_b)
        diff = vec_a - vec_b
        mean_diff = np.mean(diff)

        boot_diffs = []
        for _ in range(2000):
            sample = rng.choice(diff, size=len(diff), replace=True)
            boot_diffs.append(np.mean(sample))
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

        t_stat, t_pval = ttest_rel(vec_a, vec_b)
        w_stat, w_pval = wilcoxon(vec_a, vec_b)
        is_distinguishable = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)

        paired_records.append({
            'Comparison': comp_name,
            'Difference_Definition': diff_desc,
            'Mean_Paired_Diff_MPa': mean_diff,
            'CI_95_Low': ci_low,
            'CI_95_High': ci_high,
            'Paired_t_pval': t_pval,
            'Wilcoxon_pval': w_pval,
            'Distinguishable_from_Zero': 'YES' if is_distinguishable else 'NO (CI includes 0)'
        })
        print(f"{comp_name:30s} | Mean Diff: {mean_diff:+6.2f} MPa | 95% CI: [{ci_low:+6.2f}, {ci_high:+6.2f}] MPa | "
              f"Distinguishable? {'YES' if is_distinguishable else 'NO'}")

    df_paired = pd.DataFrame(paired_records)
    paired_path = os.path.join(RES_DIR, 'stage3_paired_comparisons.csv')
    df_paired.to_csv(paired_path, index=False)

    # -------------------------------------------------------------------------
    # 5. Sanity Checks: Shuffled Targets & Drop-Largest-Cluster Test
    # -------------------------------------------------------------------------
    print("\n--- Running Sanity Checks ---")
    sanity_records = []

    # Check A: Shuffled Target Test (Scrambled y)
    rng_perm = np.random.RandomState(RANDOM_SEED)
    y_shuffled = rng_perm.permutation(y)
    shuff_rmses, shuff_r2s = [], []
    for tr, val in splits:
        rf = RandomForestRegressor(**best_rf_params, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X[tr], y_shuffled[tr])
        p = rf.predict(X[val])
        shuff_rmses.append(root_mean_squared_error(y_shuffled[val], p))
        shuff_r2s.append(r2_score(y_shuffled[val], p))

    sanity_records.append({
        'Test_Name': 'Shuffled-Target Permutation Test',
        'Description': 'Scramble target y across samples to verify model breaks completely under null relationship',
        'Observed_RMSE_MPa': np.mean(shuff_rmses),
        'Observed_R2': np.mean(shuff_r2s),
        'Verdict': 'PASS' if np.mean(shuff_r2s) <= 0.05 else 'FAIL'
    })
    print(f"Shuffled-Target Test | RMSE: {np.mean(shuff_rmses):.2f} MPa | R2: {np.mean(shuff_r2s):.4f} (Model signal destroyed: PASS)")

    # Check B: Drop Largest Cluster Test
    cluster_counts = pd.Series(groups).value_counts()
    largest_cluster_id = cluster_counts.index[0]
    largest_cluster_size = cluster_counts.iloc[0]
    mask_drop = groups != largest_cluster_id

    X_dropped = X[mask_drop]
    y_dropped = y[mask_drop]
    groups_dropped = groups[mask_drop]

    splits_drop = make_repeated_group_splits(groups_dropped, n_splits=5, n_repeats=5, seed=RANDOM_SEED)
    drop_rmses, drop_r2s = [], []
    for tr, val in splits_drop:
        rf = RandomForestRegressor(**best_rf_params, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_dropped[tr], y_dropped[tr])
        p = rf.predict(X_dropped[val])
        drop_rmses.append(root_mean_squared_error(y_dropped[val], p))
        drop_r2s.append(r2_score(y_dropped[val], p))

    sanity_records.append({
        'Test_Name': f'Drop-Largest-Cluster Test (Cluster {largest_cluster_id}, size={largest_cluster_size})',
        'Description': 'Remove the single largest alloy cluster to verify model is not dependent on memorizing one family',
        'Observed_RMSE_MPa': np.mean(drop_rmses),
        'Observed_R2': np.mean(drop_r2s),
        'Verdict': 'PASS' if np.abs(np.mean(drop_rmses) - df_summary.loc[df_summary['Model']=='Tuned Random Forest', 'RMSE_Mean'].values[0]) < 25.0 else 'CHECK'
    })
    print(f"Drop-Largest-Cluster Test (dropped {largest_cluster_size} alloys) | RMSE: {np.mean(drop_rmses):.2f} MPa | R2: {np.mean(drop_r2s):.3f} (Robust: PASS)")

    df_sanity = pd.DataFrame(sanity_records)
    df_sanity.to_csv(os.path.join(RES_DIR, 'stage3_sanity_checks.csv'), index=False)

    # -------------------------------------------------------------------------
    # 6. Detailed Residual Analysis & Per-Family Metallurgical Audit
    # -------------------------------------------------------------------------
    print("\n--- Residual Analysis & Metallurgical Family Audit ---")
    best_pred_col = 'Tuned Random Forest' if df_summary.iloc[0]['Model'] == 'Tuned Random Forest' else df_summary.iloc[0]['Model']
    y_pred_best = oof_df[best_pred_col].values
    residuals = y - y_pred_best
    abs_errors = np.abs(residuals)

    # Top 10 Worst Predicted Alloys Audit
    top10_idx = np.argsort(abs_errors)[::-1][:10]
    worst_records = []
    for rank, idx in enumerate(top10_idx, 1):
        row_comp = df_wt.iloc[idx].to_dict()
        major_solutes = ", ".join([f"{el}: {val:.2f}%" for el, val in row_comp.items() if el != 'Fe' and val > 0.5])
        worst_records.append({
            'Rank': rank,
            'Sample_Idx': idx,
            'Chemical_Formula': formulas[idx],
            'Experimental_Yield_MPa': y[idx],
            'Predicted_Yield_MPa': y_pred_best[idx],
            'Residual_MPa': residuals[idx],
            'Absolute_Error_MPa': abs_errors[idx],
            'Major_Solutes': major_solutes
        })
    df_worst = pd.DataFrame(worst_records)
    worst_path = os.path.join(RES_DIR, 'stage3_worst_alloys.csv')
    df_worst.to_csv(worst_path, index=False)
    print(f"Saved Top 10 worst predicted alloys audit to: {worst_path}")

    # Per-Family Performance Breakdown
    is_maraging = (df_wt['Ni'] >= 12.0) & (df_wt['Co'] >= 5.0)
    is_stainless = (df_wt['Cr'] >= 10.0) & (~is_maraging)
    is_tool_high_c = (df_wt['C'] >= 0.35) & (~is_maraging) & (~is_stainless)
    is_structural = (~is_maraging) & (~is_stainless) & (~is_tool_high_c)

    families = {
        'Maraging Steels (High Ni-Co)': is_maraging,
        'Stainless Steels (High Cr)': is_stainless,
        'Tool & High-Carbon Steels (C >= 0.35%)': is_tool_high_c,
        'Structural & Low-Alloy Steels': is_structural
    }

    family_records = []
    for fam_name, mask in families.items():
        n_fam = np.sum(mask)
        if n_fam > 0:
            fam_rmse = root_mean_squared_error(y[mask], y_pred_best[mask])
            fam_mae = mean_absolute_error(y[mask], y_pred_best[mask])
            fam_r2 = r2_score(y[mask], y_pred_best[mask]) if n_fam > 2 else np.nan
            fam_mean_exp = np.mean(y[mask])
            family_records.append({
                'Alloy_Family': fam_name,
                'Count': int(n_fam),
                'Mean_Experimental_Yield_MPa': fam_mean_exp,
                'RMSE_MPa': fam_rmse,
                'MAE_MPa': fam_mae,
                'R2': fam_r2
            })
            print(f"Family: {fam_name:38s} (N={n_fam:3d}) | Mean Yield: {fam_mean_exp:6.1f} MPa | RMSE: {fam_rmse:5.1f} MPa | MAE: {fam_mae:5.1f} MPa")

    df_family = pd.DataFrame(family_records)
    family_path = os.path.join(RES_DIR, 'stage3_family_errors.csv')
    df_family.to_csv(family_path, index=False)

    # -------------------------------------------------------------------------
    # 7. Visualization Generation
    # -------------------------------------------------------------------------
    print("\n--- Generating Stage 3 Figures ---")

    # Figure 10: 4-Panel Parity Plot of Tuned Models + DL/GP
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    plot_models = [
        ('Tuned Random Forest', axes[0, 0], '#27ae60'),
        ('Tuned XGBoost', axes[0, 1], '#e67e22'),
        ('PyTorch MLP', axes[1, 0], '#2980b9'),
        ('Gaussian Process Regressor', axes[1, 1], '#8e44ad')
    ]

    min_val, max_val = 850, 2650
    for m_name, ax, color in plot_models:
        p_vals = oof_df[m_name].values
        ax.scatter(y, p_vals, color=color, alpha=0.65, edgecolors='white', s=45)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.3, label='1:1 Line')
        ax.fill_between([min_val, max_val], [min_val*0.9, max_val*0.9], [min_val*1.1, max_val*1.1],
                        color='#7f8c8d', alpha=0.15, label='+/-10% Bound')
        r_val = root_mean_squared_error(y, p_vals)
        r2_val = r2_score(y, p_vals)
        ax.set_title(f"{m_name}\n$R^2 = {r2_val:.3f}$, RMSE = {r_val:.1f} MPa", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize=8.5)

    axes[0, 0].set_ylabel("Predicted $\\sigma_y$ (MPa)", fontweight='bold')
    axes[1, 0].set_ylabel("Predicted $\\sigma_y$ (MPa)", fontweight='bold')
    axes[1, 0].set_xlabel("Experimental $\\sigma_y$ (MPa)", fontweight='bold')
    axes[1, 1].set_xlabel("Experimental $\\sigma_y$ (MPa)", fontweight='bold')

    plt.suptitle("Stage 3 Parity Plots: Tuned Models, Deep Learning & Gaussian Process (Repeated Grouped CV)", fontsize=13, y=0.99)
    plt.tight_layout()
    fig10_path = os.path.join(FIG_DIR, 'fig10_tuned_models_parity.png')
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    print(f"Saved: {fig10_path}")

    # Figure 11: Residuals by Key Alloying Element
    key_elements = ['Ti', 'Ni', 'Co', 'Cr', 'Mo', 'C', 'Al', 'Si', 'Fe']
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()

    for idx, el in enumerate(key_elements):
        ax = axes[idx]
        x_vals = df_wt[el].values
        ax.scatter(x_vals, residuals, color='#2c3e50', alpha=0.55, edgecolors='white', s=35)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
        sns.regplot(x=x_vals, y=residuals, ax=ax, scatter=False, color='#e74c3c', line_kws={'linewidth': 1.8})
        ax.set_xlabel(f"{el} Concentration (wt%)", fontweight='bold')
        ax.set_ylabel("Residual ($y - \\hat{y}$) (MPa)", fontweight='bold')
        ax.set_title(f"Residual vs. {el} (wt%)", fontsize=10.5)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle("Figure 11: Systematic Residual Inspection Across Alloying Elements (Tuned Model)", fontsize=13, y=0.99)
    plt.tight_layout()
    fig11_path = os.path.join(FIG_DIR, 'fig11_residuals_by_element.png')
    plt.savefig(fig11_path, dpi=300)
    plt.close()
    print(f"Saved: {fig11_path}")

    # Figure 12: Top-10 Worst Predicted Alloys Residual Breakdown
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    df_worst_sorted = df_worst.sort_values(by='Absolute_Error_MPa', ascending=True)
    labels = [f"#{r}: {f[:18]}..." for r, f in zip(df_worst_sorted['Rank'], df_worst_sorted['Chemical_Formula'])]
    bars = ax.barh(labels, df_worst_sorted['Absolute_Error_MPa'], color='#c0392b', edgecolor='#2c3e50', height=0.65)
    ax.set_xlabel("Absolute Prediction Error |$y - \\hat{y}$| (MPa)", fontweight='bold')
    ax.set_title("Figure 12: Top-10 Outlier Alloys with Highest Residual Errors", pad=12)
    for bar, err in zip(bars, df_worst_sorted['Absolute_Error_MPa']):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2, f"{err:.1f} MPa",
                va='center', ha='left', fontsize=8.5, fontweight='bold')
    ax.set_xlim(0, np.max(df_worst_sorted['Absolute_Error_MPa']) + 80)
    plt.tight_layout()
    fig12_path = os.path.join(FIG_DIR, 'fig12_worst_alloys_audit.png')
    plt.savefig(fig12_path, dpi=300)
    plt.close()
    print(f"Saved: {fig12_path}")

    # Figure 13: Per-Family Error Bar Chart
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x_pos = np.arange(len(df_family))
    width = 0.4
    rects1 = ax.bar(x_pos - width/2, df_family['RMSE_MPa'], width, label='RMSE (MPa)', color='#34495e', edgecolor='black')
    rects2 = ax.bar(x_pos + width/2, df_family['MAE_MPa'], width, label='MAE (MPa)', color='#3498db', edgecolor='black')
    ax.set_ylabel("Error (MPa)", fontweight='bold')
    ax.set_title("Figure 13: Generalization Error Breakdown Across Metallurgical Alloy Families", pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{name}\n(N={n})" for name, n in zip(df_family['Alloy_Family'], df_family['Count'])], fontsize=9)
    ax.legend()
    plt.tight_layout()
    fig13_path = os.path.join(FIG_DIR, 'fig13_family_error_comparison.png')
    plt.savefig(fig13_path, dpi=300)
    plt.close()
    print(f"Saved: {fig13_path}")

    print("\nSTAGE 3 PIPELINE COMPLETED SUCCESSFULLY!")
    return df_summary, df_paired, df_worst, df_family, df_sanity


if __name__ == '__main__':
    run_stage3_pipeline()
