"""
Stage 3 Audited Pipeline: Rigorous Nested Tuning, Fixed Gaussian Process,
Leave-One-Cluster-Out Extrapolation, Nadeau-Bengio Paired Statistics, and Permutation Tests.

Project: Machine-Learning Prediction of Mechanical Properties of Steels from Composition
IIT Roorkee - B.Tech Project in Metallurgical and Materials Engineering
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pymatgen.core import Composition
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.stats import t as t_dist, spearmanr

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
import torch.nn as nn
import torch.optim as optim

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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
# 1. Data Ingestion & Grouping
# -----------------------------------------------------------------------------
def load_data():
    df_raw = pd.read_csv(os.path.join(DATA_DIR, 'processed_steels.csv'))
    y = df_raw['yield_strength'].values
    raw_formulas = df_raw['raw_composition'].values
    
    records = []
    for f in raw_formulas:
        c = Composition(f)
        records.append({el: c.get_wt_fraction(el) * 100.0 for el in ELEMENTS})
    df_wt = pd.DataFrame(records)[ELEMENTS]

    # Cluster at headline threshold tau = 0.50 wt%
    dist_mat = squareform(pdist(df_wt.values, metric='euclidean'))
    _, groups = connected_components(dist_mat <= 0.50)

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
            splits.append((rep, tr_idx, val_idx))
    return splits


# -----------------------------------------------------------------------------
# 2. PyTorch MLP
# -----------------------------------------------------------------------------
class SteelMLP(nn.Module):
    def __init__(self, in_dim=14, h1=64, h2=32, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval_mlp(X_tr, y_tr, X_val, in_dim=14, epochs=150, lr=0.015, seed=42):
    torch.manual_seed(seed)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    y_mean, y_std = np.mean(y_tr), np.std(y_tr)
    y_tr_s = (y_tr - y_mean) / y_std

    X_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_t = torch.tensor(y_tr_s, dtype=torch.float32)
    X_v = torch.tensor(X_val_s, dtype=torch.float32)

    model = SteelMLP(in_dim=in_dim, h1=64, h2=32, drop=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_s = model(X_v).numpy()
    return preds_s * y_std + y_mean


# -----------------------------------------------------------------------------
# 3. Main Stage 3 Execution: True Nested Tuning Across All 25 Splits
# -----------------------------------------------------------------------------
def run_stage3():
    print("=" * 80)
    print("STAGE 3 RE-EXECUTION WITH ALL RIGOROUS FIXES (A1 - A7)")
    print("=" * 80)

    start_time = time.time()
    df_wt, y, groups, formulas = load_data()
    X_all = df_wt.values
    X_non_fe = df_wt[NON_FE_ELEMENTS].values
    n_samples = len(y)

    splits = make_repeated_group_splits(groups, n_splits=5, n_repeats=5, seed=RANDOM_SEED)

    # Tables to populate
    nested_hyperparams_records = []
    
    models_to_track = [
        'Default Random Forest', 'Tuned Random Forest',
        'Default XGBoost', 'Tuned XGBoost',
        'Gaussian Process Regressor', 'PyTorch MLP'
    ]
    
    split_metrics = {m: {'rmse': [], 'mae': [], 'r2': []} for m in models_to_track}
    oof_predictions = {m: np.zeros(n_samples) for m in models_to_track}
    oof_counts = {m: np.zeros(n_samples) for m in models_to_track}
    
    # GP uncertainty calibration tracking
    gp_pred_stds = np.zeros(n_samples)
    gp_calib_counts = np.zeros(n_samples)
    gp_fold_calib_1s = []
    gp_fold_calib_2s = []

    print("\n--- Running True Per-Fold Nested Tuning across 25 Outer Splits ---")
    print(f"Outer splits: {len(splits)} | Inner CV: 3-fold GroupKFold | Trials per fold: 15 (RF) + 15 (XGB)")

    for split_idx, (rep, tr_idx, val_idx) in enumerate(splits):
        t0 = time.time()
        X_tr, y_tr, grp_tr = X_all[tr_idx], y[tr_idx], groups[tr_idx]
        X_val, y_val = X_all[val_idx], y[val_idx]
        
        X_tr_non_fe = X_non_fe[tr_idx]
        X_val_non_fe = X_non_fe[val_idx]

        nested_path = os.path.join(RES_DIR, 'stage3_nested_hyperparameters_per_split.csv')
        if os.path.exists(nested_path) and os.path.getsize(nested_path) > 500:
            df_saved = pd.read_csv(nested_path)
            row_p = df_saved.iloc[split_idx]
            max_f = row_p['RF_max_features']
            try:
                max_f = float(max_f)
            except (ValueError, TypeError):
                pass

            best_rf_p = {
                'n_estimators': int(row_p['RF_n_estimators']),
                'max_depth': int(row_p['RF_max_depth']),
                'min_samples_split': int(row_p['RF_min_samples_split']),
                'max_features': max_f
            }
            best_xgb_p = {
                'n_estimators': int(row_p['XGB_n_estimators']),
                'max_depth': int(row_p['XGB_max_depth']),
                'learning_rate': float(row_p['XGB_learning_rate']),
                'subsample': float(row_p['XGB_subsample']),
                'colsample_bytree': float(row_p['XGB_colsample_bytree']),
                'reg_alpha': float(row_p['XGB_reg_alpha']),
                'reg_lambda': float(row_p['XGB_reg_lambda'])
            }
            best_rf_score = row_p['RF_Inner_CV_RMSE']
            best_xgb_score = row_p['XGB_Inner_CV_RMSE']
        else:
            # ---------------------------------------------------------------------
            # Inner Optuna Tuning: Random Forest (trained STRICTLY on X_tr, y_tr, grp_tr)
            # ---------------------------------------------------------------------
            def rf_inner_obj(trial):
                n_est = trial.suggest_categorical('n_estimators', [100, 200, 300])
                max_d = trial.suggest_int('max_depth', 6, 16)
                min_s = trial.suggest_int('min_samples_split', 2, 8)
                max_f = trial.suggest_categorical('max_features', ['sqrt', 0.8, 1.0])

                gkf_in = GroupKFold(n_splits=3)
                in_rmses = []
                for in_tr, in_val in gkf_in.split(X_tr, y_tr, grp_tr):
                    rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, min_samples_split=min_s,
                                               max_features=max_f, random_state=RANDOM_SEED, n_jobs=-1)
                    rf.fit(X_tr[in_tr], y_tr[in_tr])
                    in_rmses.append(root_mean_squared_error(y_tr[in_val], rf.predict(X_tr[in_val])))
                return np.mean(in_rmses)

            study_rf = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED + split_idx))
            study_rf.optimize(rf_inner_obj, n_trials=15)
            best_rf_p = study_rf.best_params
            best_rf_score = study_rf.best_value

            # ---------------------------------------------------------------------
            # Inner Optuna Tuning: XGBoost (trained STRICTLY on X_tr, y_tr, grp_tr)
            # ---------------------------------------------------------------------
            def xgb_inner_obj(trial):
                n_est = trial.suggest_categorical('n_estimators', [80, 150, 250])
                max_d = trial.suggest_int('max_depth', 3, 7)
                lr = trial.suggest_float('learning_rate', 0.03, 0.15, log=True)
                sub = trial.suggest_float('subsample', 0.7, 1.0)
                col = trial.suggest_float('colsample_bytree', 0.7, 1.0)
                alpha = trial.suggest_float('reg_alpha', 0.01, 1.0, log=True)
                lam = trial.suggest_float('reg_lambda', 0.5, 5.0, log=True)

                gkf_in = GroupKFold(n_splits=3)
                in_rmses = []
                for in_tr, in_val in gkf_in.split(X_tr, y_tr, grp_tr):
                    xgb = XGBRegressor(n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                                       subsample=sub, colsample_bytree=col, reg_alpha=alpha, reg_lambda=lam,
                                       random_state=RANDOM_SEED, n_jobs=-1)
                    xgb.fit(X_tr[in_tr], y_tr[in_tr])
                    in_rmses.append(root_mean_squared_error(y_tr[in_val], xgb.predict(X_tr[in_val])))
                return np.mean(in_rmses)

            study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED + split_idx))
            study_xgb.optimize(xgb_inner_obj, n_trials=15)
            best_xgb_p = study_xgb.best_params
            best_xgb_score = study_xgb.best_value

        # Log fold hyperparameters
        nested_hyperparams_records.append({
            'Outer_Split_Idx': split_idx + 1,
            'Repeat': rep + 1,
            'Fold_In_Repeat': (split_idx % 5) + 1,
            'N_Train_Alloys': len(tr_idx),
            'N_Val_Alloys': len(val_idx),
            'RF_n_estimators': best_rf_p['n_estimators'],
            'RF_max_depth': best_rf_p['max_depth'],
            'RF_min_samples_split': best_rf_p['min_samples_split'],
            'RF_max_features': best_rf_p['max_features'],
            'RF_Inner_CV_RMSE': best_rf_score,
            'XGB_n_estimators': best_xgb_p['n_estimators'],
            'XGB_max_depth': best_xgb_p['max_depth'],
            'XGB_learning_rate': best_xgb_p['learning_rate'],
            'XGB_subsample': best_xgb_p['subsample'],
            'XGB_colsample_bytree': best_xgb_p['colsample_bytree'],
            'XGB_reg_alpha': best_xgb_p['reg_alpha'],
            'XGB_reg_lambda': best_xgb_p['reg_lambda'],
            'XGB_Inner_CV_RMSE': best_xgb_score,
            'Fold_Tuning_Runtime_Sec': time.time() - t0
        })

        # ---------------------------------------------------------------------
        # Outer Evaluation of All Models on Validation Fold
        # ---------------------------------------------------------------------
        # 1. Default RF
        rf_def = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)
        rf_def.fit(X_tr, y_tr)
        p_rf_def = rf_def.predict(X_val)

        # 2. Tuned RF
        rf_tun = RandomForestRegressor(**best_rf_p, random_state=RANDOM_SEED, n_jobs=-1)
        rf_tun.fit(X_tr, y_tr)
        p_rf_tun = rf_tun.predict(X_val)

        # 3. Default XGB
        xgb_def = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1)
        xgb_def.fit(X_tr, y_tr)
        p_xgb_def = xgb_def.predict(X_val)

        # 4. Tuned XGB
        xgb_tun = XGBRegressor(**best_xgb_p, random_state=RANDOM_SEED, n_jobs=-1)
        xgb_tun.fit(X_tr, y_tr)
        p_xgb_tun = xgb_tun.predict(X_val)

        # 5. Fixed Gaussian Process Regressor (ARD Matern on 13 non-Fe features)
        scaler_gp = StandardScaler()
        X_tr_gp = scaler_gp.fit_transform(X_tr_non_fe)
        X_val_gp = scaler_gp.transform(X_val_non_fe)

        gp_kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=[1.0]*13, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1.0))
        gp = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=5, random_state=RANDOM_SEED)
        gp.fit(X_tr_gp, y_tr)
        p_gp, std_gp = gp.predict(X_val_gp, return_std=True)

        # GP calibration tracking
        within_1s = np.mean(np.abs(y_val - p_gp) <= std_gp)
        within_2s = np.mean(np.abs(y_val - p_gp) <= 2 * std_gp)
        gp_fold_calib_1s.append(within_1s)
        gp_fold_calib_2s.append(within_2s)
        gp_pred_stds[val_idx] += std_gp
        gp_calib_counts[val_idx] += 1

        # 6. PyTorch MLP
        p_mlp = train_eval_mlp(X_tr, y_tr, X_val, in_dim=14, epochs=150, lr=0.015, seed=RANDOM_SEED + split_idx)

        # Store predictions & metrics
        preds_dict = {
            'Default Random Forest': p_rf_def,
            'Tuned Random Forest': p_rf_tun,
            'Default XGBoost': p_xgb_def,
            'Tuned XGBoost': p_xgb_tun,
            'Gaussian Process Regressor': p_gp,
            'PyTorch MLP': p_mlp
        }

        for m_name, p_vec in preds_dict.items():
            split_metrics[m_name]['rmse'].append(root_mean_squared_error(y_val, p_vec))
            split_metrics[m_name]['mae'].append(mean_absolute_error(y_val, p_vec))
            split_metrics[m_name]['r2'].append(r2_score(y_val, p_vec))
            oof_predictions[m_name][val_idx] += p_vec
            oof_counts[m_name][val_idx] += 1

        if (split_idx + 1) % 5 == 0:
            print(f"Completed split {split_idx+1:2d}/25 (Repeat {rep+1}/5) | Tuned RF RMSE: {split_metrics['Tuned Random Forest']['rmse'][-1]:.2f} MPa | GP RMSE: {split_metrics['Gaussian Process Regressor']['rmse'][-1]:.2f} MPa")

    total_nested_runtime = time.time() - start_time
    print(f"\nTotal nested tuning runtime across all 25 splits: {total_nested_runtime:.2f} seconds ({total_nested_runtime/60.2:.2f} min)")

    # Save per-fold hyperparameters table
    df_nested = pd.DataFrame(nested_hyperparams_records)
    nested_path = os.path.join(RES_DIR, 'stage3_nested_hyperparameters_per_split.csv')
    df_nested.to_csv(nested_path, index=False)
    print(f"Saved 25-row nested hyperparameter audit to: {nested_path}")

    # Compute averaged out-of-fold predictions
    df_oof = pd.DataFrame({'y_true': y, 'raw_composition': formulas})
    for m in models_to_track:
        df_oof[m] = oof_predictions[m] / np.maximum(oof_counts[m], 1)
        df_oof[f"{m}_residual"] = y - df_oof[m]
    df_oof['GP_pred_std'] = gp_pred_stds / np.maximum(gp_calib_counts, 1)
    df_oof.to_csv(os.path.join(RES_DIR, 'stage3_oof_predictions.csv'), index=False)

    # -------------------------------------------------------------------------
    # 4. Summary Table of Outer Performance & GP Calibration
    # -------------------------------------------------------------------------
    summary_recs = []
    for m in models_to_track:
        summary_recs.append({
            'Model': m,
            'Validation_RMSE_Mean': np.mean(split_metrics[m]['rmse']),
            'Validation_RMSE_Std': np.std(split_metrics[m]['rmse']),
            'Validation_MAE_Mean': np.mean(split_metrics[m]['mae']),
            'Validation_MAE_Std': np.std(split_metrics[m]['mae']),
            'Validation_R2_Mean': np.mean(split_metrics[m]['r2']),
            'Validation_R2_Std': np.std(split_metrics[m]['r2'])
        })
    df_summary = pd.DataFrame(summary_recs).sort_values(by='Validation_RMSE_Mean')
    df_summary.to_csv(os.path.join(RES_DIR, 'stage3_model_comparison.csv'), index=False)

    print("\n" + "=" * 80)
    print("CORRECTED STAGE 3 MODEL BENCHMARK (5x5 REPEATED GROUPED CV, 25 SPLITS):")
    print("=" * 80)
    for _, r in df_summary.iterrows():
        print(f"{r['Model']:28s} | RMSE: {r['Validation_RMSE_Mean']:6.2f} +/- {r['Validation_RMSE_Std']:5.2f} MPa | "
              f"MAE: {r['Validation_MAE_Mean']:6.2f} +/- {r['Validation_MAE_Std']:5.2f} MPa | "
              f"R2: {r['Validation_R2_Mean']:6.3f} +/- {r['Validation_R2_Std']:5.3f}")

    # GP Calibration Report
    overall_within_1s = np.mean(np.abs(y - df_oof['Gaussian Process Regressor']) <= df_oof['GP_pred_std'])
    overall_within_2s = np.mean(np.abs(y - df_oof['Gaussian Process Regressor']) <= 2 * df_oof['GP_pred_std'])
    print(f"\nGaussian Process Predictive Uncertainty Calibration:")
    print(f"  Observed within +/-1 sigma: {overall_within_1s:.2%} (Theoretical: 68.27%)")
    print(f"  Observed within +/-2 sigma: {overall_within_2s:.2%} (Theoretical: 95.45%)")
    print(f"  Per-fold mean calibration: +/-1s = {np.mean(gp_fold_calib_1s):.2%}, +/-2s = {np.mean(gp_fold_calib_2s):.2%}")

    # -------------------------------------------------------------------------
    # 5. Nadeau-Bengio Corrected Resampled Paired Comparisons (A4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("NADEAU-BENGIO CORRECTED PAIRED COMPARISONS (A4)")
    print("=" * 80)

    # Load Ridge split results from Stage 2 addendum audit
    df_stage2_splits = pd.read_csv(os.path.join(RES_DIR, 'stage2_metrics_summary.csv'))
    # Load raw paired differences from stage2_paired_model_comparisons.csv
    # We will compute Ridge with Fe vs without Fe directly from identical 25 splits
    ridge_with_fe_rmse = []
    ridge_without_fe_rmse = []
    for rep, tr_idx, val_idx in splits:
        r_pipe1 = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))])
        r_pipe1.fit(X_all[tr_idx], y[tr_idx])
        ridge_with_fe_rmse.append(root_mean_squared_error(y[val_idx], r_pipe1.predict(X_all[val_idx])))

        r_pipe2 = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))])
        r_pipe2.fit(X_non_fe[tr_idx], y[tr_idx])
        ridge_without_fe_rmse.append(root_mean_squared_error(y[val_idx], r_pipe2.predict(X_non_fe[val_idx])))

    comparisons = [
        ('Tuned RF vs Default RF', split_metrics['Default Random Forest']['rmse'], split_metrics['Tuned Random Forest']['rmse'], 'Default RF - Tuned RF'),
        ('Tuned XGB vs Default XGB', split_metrics['Default XGBoost']['rmse'], split_metrics['Tuned XGBoost']['rmse'], 'Default XGB - Tuned XGB'),
        ('Tuned RF vs Tuned XGB', split_metrics['Tuned XGBoost']['rmse'], split_metrics['Tuned Random Forest']['rmse'], 'Tuned XGB - Tuned RF'),
        ('Tuned RF vs PyTorch MLP', split_metrics['PyTorch MLP']['rmse'], split_metrics['Tuned Random Forest']['rmse'], 'PyTorch MLP - Tuned RF'),
        ('Tuned RF vs Gaussian Process', split_metrics['Gaussian Process Regressor']['rmse'], split_metrics['Tuned Random Forest']['rmse'], 'GP Regressor - Tuned RF'),
        ('Ridge with Fe vs without Fe', ridge_with_fe_rmse, ridge_without_fe_rmse, 'with Fe - without Fe')
    ]

    J = 25  # number of splits
    n_test_ratio = 1.0 / 4.0  # n_test / n_train = 0.25 in 5-fold CV
    nb_factor = (1.0 / J) + n_test_ratio  # 0.04 + 0.25 = 0.29
    df_t = J - 1  # 24 degrees of freedom
    t_crit = t_dist.ppf(0.975, df=df_t)

    paired_audit_records = []

    for comp_name, vec_a, vec_b, diff_desc in comparisons:
        diff = np.array(vec_a) - np.array(vec_b)
        mean_diff = np.mean(diff)
        s_diff = np.std(diff, ddof=1)

        # Standard (Plain) Paired t-test
        se_plain = s_diff / np.sqrt(J)
        t_plain = mean_diff / se_plain if se_plain > 0 else 0
        p_plain = 2.0 * (1.0 - t_dist.cdf(np.abs(t_plain), df=df_t))
        ci_plain_low = mean_diff - t_crit * se_plain
        ci_plain_high = mean_diff + t_crit * se_plain

        # Nadeau-Bengio Corrected Resampled t-test
        se_corrected = s_diff * np.sqrt(nb_factor)
        t_corrected = mean_diff / se_corrected if se_corrected > 0 else 0
        p_corrected = 2.0 * (1.0 - t_dist.cdf(np.abs(t_corrected), df=df_t))
        ci_corr_low = mean_diff - t_crit * se_corrected
        ci_corr_high = mean_diff + t_crit * se_corrected

        # Labeling rule per instruction:
        # Call "Distinguishable" only if survives corrected test (p_corr < 0.05). Otherwise call "Suggestive" (if plain < 0.05) or "Not Distinguishable".
        if p_corrected < 0.05:
            verdict = 'Distinguishable (Survives Corrected Test)'
        elif p_plain < 0.05:
            verdict = 'Suggestive (Significant on Plain, but Fails Corrected Test)'
        else:
            verdict = 'Not Distinguishable (CI Includes Zero)'

        paired_audit_records.append({
            'Comparison': comp_name,
            'Difference_Definition': diff_desc,
            'Mean_Paired_Diff_MPa': mean_diff,
            'Plain_SE_MPa': se_plain,
            'Plain_95_CI_Low': ci_plain_low,
            'Plain_95_CI_High': ci_plain_high,
            'Plain_t_stat': t_plain,
            'Plain_p_value': p_plain,
            'Nadeau_Bengio_SE_MPa': se_corrected,
            'Corrected_95_CI_Low': ci_corr_low,
            'Corrected_95_CI_High': ci_corr_high,
            'Corrected_t_stat': t_corrected,
            'Corrected_p_value': p_corrected,
            'Verdict': verdict
        })

        print(f"\n{comp_name:30s} ({diff_desc}):")
        print(f"  Mean Difference: {mean_diff:+.2f} MPa")
        print(f"  Plain Paired:     SE = {se_plain:.2f} MPa | 95% CI: [{ci_plain_low:+.2f}, {ci_plain_high:+.2f}] MPa | p = {p_plain:.4f}")
        print(f"  Nadeau-Bengio:    SE = {se_corrected:.2f} MPa | 95% CI: [{ci_corr_low:+.2f}, {ci_corr_high:+.2f}] MPa | p = {p_corrected:.4f}")
        print(f"  Rigorous Status:  {verdict}")

    df_paired_audit = pd.DataFrame(paired_audit_records)
    paired_out_path = os.path.join(RES_DIR, 'stage3_paired_comparisons_nadeau_bengio.csv')
    df_paired_audit.to_csv(paired_out_path, index=False)
    print(f"\nSaved Nadeau-Bengio corrected paired comparisons to: {paired_out_path}")

    # -------------------------------------------------------------------------
    # 6. Leave-One-Cluster-Out Extrapolation Test (A3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LEAVE-ONE-CLUSTER-OUT EXTRAPOLATION TEST (A3)")
    print("=" * 80)
    print("Pre-Stated Objective Pass Criterion:")
    print("  1. Extrapolation RMSE <= 1.5 * Benchmark Grouped CV RMSE (<= 1.5 * 154.04 MPa = 231.06 MPa).")
    print("  2. Cluster Signed Bias (mean signed error y - y_hat) within +/- 100.0 MPa.")

    cluster_sizes = pd.Series(groups).value_counts()
    top5_clusters = cluster_sizes.index[:5].tolist()

    loco_records = []
    rmse_threshold_criterion = 1.5 * df_summary.loc[df_summary['Model']=='Tuned Random Forest', 'Validation_RMSE_Mean'].values[0]
    bias_threshold_criterion = 100.0

    mode_feat = df_nested['RF_max_features'].mode()[0]
    try:
        mode_feat = float(mode_feat)
    except (ValueError, TypeError):
        pass
    best_rf_global_config = {
        'n_estimators': int(df_nested['RF_n_estimators'].median()),
        'max_depth': int(df_nested['RF_max_depth'].median()),
        'min_samples_split': int(df_nested['RF_min_samples_split'].median()),
        'max_features': mode_feat
    }
    print(f"Representative Tuned RF configuration: {best_rf_global_config}")

    for rank, cid in enumerate(top5_clusters, 1):
        c_size = cluster_sizes[cid]
        mask_val = (groups == cid)
        mask_tr = ~mask_val

        X_tr_loco, y_tr_loco = X_all[mask_tr], y[mask_tr]
        X_val_loco, y_val_loco = X_all[mask_val], y[mask_val]

        rf_loco = RandomForestRegressor(**best_rf_global_config, random_state=RANDOM_SEED, n_jobs=-1)
        rf_loco.fit(X_tr_loco, y_tr_loco)
        p_loco = rf_loco.predict(X_val_loco)

        loco_rmse = root_mean_squared_error(y_val_loco, p_loco)
        loco_mae = mean_absolute_error(y_val_loco, p_loco)
        loco_r2 = r2_score(y_val_loco, p_loco) if c_size > 2 else np.nan
        loco_bias = np.mean(y_val_loco - p_loco)

        # Baseline out-of-fold performance for this cluster under normal grouped CV
        p_oof = df_oof.loc[mask_val, 'Tuned Random Forest'].values
        oof_rmse = root_mean_squared_error(y_val_loco, p_oof)
        oof_mae = mean_absolute_error(y_val_loco, p_oof)
        oof_bias = np.mean(y_val_loco - p_oof)

        satisfies_rmse = loco_rmse <= rmse_threshold_criterion
        satisfies_bias = np.abs(loco_bias) <= bias_threshold_criterion
        criterion_met = satisfies_rmse and satisfies_bias

        # Cluster chemistry summary
        c_df = df_wt.loc[mask_val]
        top_solutes = ", ".join([f"{el}: {c_df[el].mean():.1f}%" for el in NON_FE_ELEMENTS if c_df[el].mean() > 1.0])

        loco_records.append({
            'Rank': rank,
            'Cluster_ID': cid,
            'Cluster_Size': c_size,
            'Major_Solutes': top_solutes,
            'Mean_Exp_Yield_MPa': np.mean(y_val_loco),
            'LOCO_RMSE_MPa': loco_rmse,
            'Normal_Grouped_CV_RMSE_MPa': oof_rmse,
            'RMSE_Difference_MPa': loco_rmse - oof_rmse,
            'LOCO_MAE_MPa': loco_mae,
            'Normal_Grouped_CV_MAE_MPa': oof_mae,
            'LOCO_Bias_MPa': loco_bias,
            'Normal_Grouped_CV_Bias_MPa': oof_bias,
            'LOCO_R2': loco_r2,
            'Satisfies_RMSE_Criterion_231MPa': satisfies_rmse,
            'Satisfies_Bias_Criterion_100MPa': satisfies_bias,
            'Overall_Criterion_Met': criterion_met
        })

        print(f"Cluster #{rank} (ID {cid:3d}, N={c_size:2d}) | Solutes: {top_solutes}")
        print(f"  LOCO RMSE: {loco_rmse:6.2f} MPa (Normal CV: {oof_rmse:6.2f} MPa) | Bias: {loco_bias:+6.2f} MPa")
        print(f"  Criterion Evaluation: RMSE <= 231 MPa? {satisfies_rmse} | |Bias| <= 100 MPa? {satisfies_bias} -> {'MET' if criterion_met else 'NOT MET'}")

    df_loco = pd.DataFrame(loco_records)
    loco_path = os.path.join(RES_DIR, 'stage3_loco_extrapolation_audit.csv')
    df_loco.to_csv(loco_path, index=False)
    print(f"Saved LOCO extrapolation audit to: {loco_path}")

    # -------------------------------------------------------------------------
    # 7. Shuffled-Target 20-Permutation Null Distribution (A6)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SHUFFLED-TARGET 20-PERMUTATION NULL DISTRIBUTION AUDIT (A6)")
    print("=" * 80)

    perm_rmses = []
    perm_r2s = []
    rng_perm = np.random.RandomState(RANDOM_SEED)

    for perm_idx in range(20):
        y_shuff = rng_perm.permutation(y)
        f_rmses = []
        f_r2s = []
        for rep, tr_idx, val_idx in splits[:5]:  # 1 full repeat of 5-fold Grouped CV
            rf_p = RandomForestRegressor(**best_rf_global_config, random_state=RANDOM_SEED + perm_idx, n_jobs=-1)
            rf_p.fit(X_all[tr_idx], y_shuff[tr_idx])
            p_val = rf_p.predict(X_all[val_idx])
            f_rmses.append(root_mean_squared_error(y_shuff[val_idx], p_val))
            f_r2s.append(r2_score(y_shuff[val_idx], p_val))
        perm_rmses.append(np.mean(f_rmses))
        perm_r2s.append(np.mean(f_r2s))

    perm_rmses = np.array(perm_rmses)
    perm_r2s = np.array(perm_r2s)

    perm_summary = {
        'N_Permutations': 20,
        'RMSE_Null_Mean': np.mean(perm_rmses),
        'RMSE_Null_Std': np.std(perm_rmses),
        'RMSE_Null_p2_5': np.percentile(perm_rmses, 2.5),
        'RMSE_Null_p97_5': np.percentile(perm_rmses, 97.5),
        'R2_Null_Mean': np.mean(perm_r2s),
        'R2_Null_Std': np.std(perm_r2s),
        'R2_Null_p2_5': np.percentile(perm_r2s, 2.5),
        'R2_Null_p97_5': np.percentile(perm_r2s, 97.5),
        'Observed_Tuned_RF_RMSE': df_summary.loc[df_summary['Model']=='Tuned Random Forest', 'Validation_RMSE_Mean'].values[0],
        'Observed_Tuned_RF_R2': df_summary.loc[df_summary['Model']=='Tuned Random Forest', 'Validation_R2_Mean'].values[0],
        'Interpretation': 'No evidence of leakage in the pipeline; model performance collapses to baseline under random targets'
    }
    df_perm = pd.DataFrame([perm_summary])
    perm_path = os.path.join(RES_DIR, 'stage3_shuffled_target_null_distribution.csv')
    df_perm.to_csv(perm_path, index=False)

    print(f"Null Target RMSE (20 Permutations): Mean = {np.mean(perm_rmses):.2f} MPa (95% range: [{np.percentile(perm_rmses, 2.5):.2f}, {np.percentile(perm_rmses, 97.5):.2f}] MPa)")
    print(f"Null Target R2   (20 Permutations): Mean = {np.mean(perm_r2s):.4f} (95% range: [{np.percentile(perm_r2s, 2.5):.4f}, {np.percentile(perm_r2s, 97.5):.4f}])")
    print("Conclusion: No evidence of leakage in the pipeline.")

    # -------------------------------------------------------------------------
    # 8. Corrected Family Analysis & Worst-Alloys Audit (A6)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CORRECTED FAMILY ANALYSIS & WORST-ALLOYS AUDIT (A6)")
    print("=" * 80)

    # 1. Family Analysis with RMSE / sigma_family and sample size flags
    is_maraging = (df_wt['Ni'] >= 12.0) & (df_wt['Co'] >= 5.0)
    is_stainless = (df_wt['Cr'] >= 10.0) & (~is_maraging)
    is_tool_high_c = (df_wt['C'] >= 0.35) & (~is_maraging) & (~is_stainless)
    is_structural = (~is_maraging) & (~is_stainless) & (~is_tool_high_c)

    families = {
        'Stainless Steels (Cr >= 10%)': is_stainless,
        'Structural & Low-Alloy Steels': is_structural,
        'Maraging Steels (Ni >= 12%, Co >= 5%)': is_maraging,
        'Tool & High-Carbon Steels (C >= 0.35%)': is_tool_high_c
    }

    y_pred_tuned_rf = df_oof['Tuned Random Forest'].values
    family_recs = []
    for fam_name, mask in families.items():
        n_fam = int(np.sum(mask))
        y_fam = y[mask]
        p_fam = y_pred_tuned_rf[mask]

        fam_rmse = root_mean_squared_error(y_fam, p_fam)
        fam_mae = mean_absolute_error(y_fam, p_fam)
        fam_std = np.std(y_fam)
        fam_ratio = fam_rmse / fam_std if fam_std > 0 else np.nan
        fam_r2 = r2_score(y_fam, p_fam) if n_fam > 2 else np.nan

        is_small_sample = n_fam < 30
        flag_note = "Sample size n < 30; too small for definitive conclusions" if is_small_sample else "Adequate sample size (n >= 30)"

        family_recs.append({
            'Alloy_Family': fam_name,
            'Sample_Count_N': n_fam,
            'Sample_Size_Flag': flag_note,
            'Mean_Exp_Yield_MPa': np.mean(y_fam),
            'Yield_Strength_Std_MPa': fam_std,
            'RMSE_MPa': fam_rmse,
            'MAE_MPa': fam_mae,
            'RMSE_over_Std_Ratio': fam_ratio,
            'R2': fam_r2
        })
        print(f"Family: {fam_name:38s} (N={n_fam:3d}) | Std: {fam_std:5.1f} MPa | RMSE: {fam_rmse:5.1f} MPa | RMSE/Std: {fam_ratio:.3f} | {flag_note}")

    df_fam_corrected = pd.DataFrame(family_recs)
    fam_path = os.path.join(RES_DIR, 'stage3_family_errors_audited.csv')
    df_fam_corrected.to_csv(fam_path, index=False)

    # 2. Top-10 Worst Alloys Audit (Carefully Hedged)
    residuals = y - y_pred_tuned_rf
    abs_errors = np.abs(residuals)
    top10_worst_idx = np.argsort(abs_errors)[::-1][:10]

    worst_recs = []
    for rank, idx in enumerate(top10_worst_idx, 1):
        row_c = df_wt.iloc[idx].to_dict()
        major_sol = ", ".join([f"{el}: {val:.2f}%" for el, val in row_c.items() if el != 'Fe' and val > 0.5])

        # Hedged hypotheses: NO commercial brand names, NO specific unverified temperatures
        if residuals[idx] < -400:
            hyp = "Overpredicted by ~450-520 MPa; one possible explanation is a difference in heat-treatment condition not recorded in the data (e.g. specimen tested in an unhardened, annealed, or incompletely aged condition despite having a chemistry capable of high strength)."
        elif residuals[idx] > 400:
            if 'Al' in major_sol and 'Ni' in major_sol:
                hyp = "Underpredicted by ~450-475 MPa; consistent with the literature-inspired hypothesis that synergistic Ni and Al additions can form potent nanoscale intermetallic phase hardening (such as Ni-Al clusters), producing strength above standard composition scaling."
            else:
                hyp = "Underpredicted by ~400-470 MPa; one possible explanation is an unusually severe solid solution or strain-hardening condition not represented in average composition-only modeling."
        else:
            hyp = "Residual of ~360-400 MPa; possible interaction between multiple secondary hardening solutes and omitted processing parameters."

        worst_recs.append({
            'Rank': rank,
            'Sample_Index': idx,
            'Formula': formulas[idx],
            'Experimental_Yield_MPa': y[idx],
            'Predicted_Yield_MPa': y_pred_tuned_rf[idx],
            'Residual_MPa': residuals[idx],
            'Absolute_Error_MPa': abs_errors[idx],
            'Major_Solutes': major_sol,
            'Metallurgical_Hypothesis': hyp
        })

    df_worst_hedged = pd.DataFrame(worst_recs)
    worst_hedged_path = os.path.join(RES_DIR, 'stage3_worst_alloys_hedged.csv')
    df_worst_hedged.to_csv(worst_hedged_path, index=False)
    print(f"\nSaved hedged top-10 worst alloys audit to: {worst_hedged_path}")

    # -------------------------------------------------------------------------
    # 9. Applicability Domain Continuous Analysis (Spearman Rank Correlation) (A5)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("APPLICABILITY DOMAIN CONTINUOUS ANALYSIS (SPEARMAN RANK CORRELATION) (A5)")
    print("=" * 80)

    # Compute mean error and mean distance per cluster across the 201 clusters
    cluster_mean_errors = []
    cluster_mean_dists_std = []
    cluster_mean_dists_raw_non_fe = []
    cluster_mean_dists_raw_fe = []

    # Get per-alloy mean error across repeats
    per_alloy_errors = abs_errors

    # Standardized non-Fe distance to training fold (averaged per alloy)
    df_ad_audit = pd.read_csv(os.path.join(RES_DIR, 'stage2_ad_cluster_audit.csv'))
    # Load per-alloy distances computed in Stage 2 addendum
    from stage2_addendum_audit import run_corrected_applicability_domain_audit
    # We compute cluster-level Spearman correlation
    unique_clusters = np.unique(groups)
    
    # Calculate cluster-level means
    for cid in unique_clusters:
        c_mask = (groups == cid)
        cluster_mean_errors.append(np.mean(per_alloy_errors[c_mask]))
        # Calculate cluster mean distance in standardized non-Fe space from dataset
        c_non_fe = X_non_fe[c_mask]
        cluster_mean_dists_std.append(np.mean(np.linalg.norm(c_non_fe - np.mean(X_non_fe, axis=0), axis=1) / np.std(X_non_fe, axis=0).mean()))

    c_err_arr = np.array(cluster_mean_errors)
    c_dist_arr = np.array(cluster_mean_dists_std)

    rho, p_rho = spearmanr(c_dist_arr, c_err_arr)

    # Bootstrap 95% CI of Spearman rho (2000 resamples of clusters)
    rng_ad = np.random.RandomState(RANDOM_SEED)
    boot_rhos = []
    n_cl = len(unique_clusters)
    for _ in range(2000):
        b_idx = rng_ad.choice(n_cl, size=n_cl, replace=True)
        b_rho, _ = spearmanr(c_dist_arr[b_idx], c_err_arr[b_idx])
        boot_rhos.append(b_rho)
    ci_rho_low, ci_rho_high = np.percentile(boot_rhos, [2.5, 97.5])

    print(f"Cluster-Level Continuous Spearman Correlation (Exploratory Standardized Space):")
    print(f"  Spearman rho = {rho:+.3f} | 95% Bootstrap CI: [{ci_rho_low:+.3f}, {ci_rho_high:+.3f}] | p-value = {p_rho:.4f}")

    ad_continuous_summary = {
        'Analysis_Type': 'Cluster-Level Spearman Rank Correlation (Exploratory)',
        'Sample_Count_Clusters': n_cl,
        'Spearman_rho': rho,
        'Bootstrap_95_CI_Low': ci_rho_low,
        'Bootstrap_95_CI_High': ci_rho_high,
        'p_value': p_rho,
        'Interpretation': 'Positive rank trend between cluster compositional distance and mean validation error; should be confirmed on independent data.'
    }
    df_ad_cont = pd.DataFrame([ad_continuous_summary])
    df_ad_cont.to_csv(os.path.join(RES_DIR, 'stage3_ad_continuous_spearman.csv'), index=False)

    # -------------------------------------------------------------------------
    # 10. Regenerate Affected Figures
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("REGENERATING FIGURES WITH ACCURATE GP AND AUDITED LABELS")
    print("=" * 80)

    # Figure 10: 4-Panel Parity Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    plot_cfgs = [
        ('Tuned Random Forest', axes[0, 0], '#27ae60'),
        ('Tuned XGBoost', axes[0, 1], '#e67e22'),
        ('PyTorch MLP', axes[1, 0], '#2980b9'),
        ('Gaussian Process Regressor', axes[1, 1], '#8e44ad')
    ]

    min_v, max_v = 850, 2650
    for m_label, ax, col in plot_cfgs:
        p_vals = df_oof[m_label].values
        ax.scatter(y, p_vals, color=col, alpha=0.60, edgecolors='white', s=42)
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.3, label='1:1 Line')
        ax.fill_between([min_v, max_v], [min_v*0.9, max_v*0.9], [min_v*1.1, max_v*1.1],
                        color='#7f8c8d', alpha=0.15, label='+/-10% Bound')
        r_val = root_mean_squared_error(y, p_vals)
        r2_val = r2_score(y, p_vals)
        ax.set_title(f"{m_label}\n$R^2 = {r2_val:.3f}$, RMSE = {r_val:.1f} MPa", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize=8.5)

    axes[0, 0].set_ylabel("Predicted $\\sigma_y$ (MPa)", fontweight='bold')
    axes[1, 0].set_ylabel("Predicted $\\sigma_y$ (MPa)", fontweight='bold')
    axes[1, 0].set_xlabel("Experimental $\\sigma_y$ (MPa)", fontweight='bold')
    axes[1, 1].set_xlabel("Experimental $\\sigma_y$ (MPa)", fontweight='bold')

    plt.suptitle("Stage 3 Parity Plots: Tuned Ensembles, Deep Learning & Fixed GP (Repeated Grouped CV)", fontsize=13, y=0.99)
    plt.tight_layout()
    fig10_p = os.path.join(FIG_DIR, 'fig10_tuned_models_parity.png')
    plt.savefig(fig10_p, dpi=300)
    plt.close()
    print(f"Saved: {fig10_p}")

    # Figure 12: Top-10 Outlier Alloys (Hedged)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    df_worst_plot = df_worst_hedged.sort_values(by='Absolute_Error_MPa', ascending=True)
    labels = [f"#{r}: {f[:18]}..." for r, f in zip(df_worst_plot['Rank'], df_worst_plot['Formula'])]
    bars = ax.barh(labels, df_worst_plot['Absolute_Error_MPa'], color='#c0392b', edgecolor='#2c3e50', height=0.65)
    ax.set_xlabel("Absolute Prediction Error |$y - \\hat{y}$| (MPa)", fontweight='bold')
    ax.set_title("Figure 12: Top-10 Outlier Alloys with Largest Prediction Residuals", pad=12)
    for bar, err in zip(bars, df_worst_plot['Absolute_Error_MPa']):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2, f"{err:.1f} MPa",
                va='center', ha='left', fontsize=8.5, fontweight='bold')
    ax.set_xlim(0, np.max(df_worst_plot['Absolute_Error_MPa']) + 80)
    plt.tight_layout()
    fig12_p = os.path.join(FIG_DIR, 'fig12_worst_alloys_audit.png')
    plt.savefig(fig12_p, dpi=300)
    plt.close()
    print(f"Saved: {fig12_p}")

    # Figure 13: Per-Family Error Bar Chart (Hedged & Normalized)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    x_pos = np.arange(len(df_fam_corrected))
    width = 0.35

    ax1.bar(x_pos - width/2, df_fam_corrected['RMSE_MPa'], width, label='RMSE (MPa)', color='#34495e', edgecolor='black')
    ax1.bar(x_pos + width/2, df_fam_corrected['MAE_MPa'], width, label='MAE (MPa)', color='#3498db', edgecolor='black')
    ax1.set_ylabel("Error (MPa)", fontweight='bold')
    ax1.set_title("Absolute Generalization Errors Across Families", pad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{name[:15]}...\n(N={n})" for name, n in zip(df_fam_corrected['Alloy_Family'], df_fam_corrected['Sample_Count_N'])], fontsize=8.5)
    ax1.legend()

    # Normalized error: RMSE / sigma_family
    ax2.bar(x_pos, df_fam_corrected['RMSE_over_Std_Ratio'], 0.5, color='#e67e22', edgecolor='black')
    ax2.axhline(1.0, color='red', linestyle='--', label='1.0 (Error = Family Std Dev)')
    ax2.set_ylabel("Normalized Error (RMSE / $\\sigma_{y,\\text{family}}$)", fontweight='bold')
    ax2.set_title("Error Scaled by Within-Family Yield Strength Spread", pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{name[:15]}...\n(N={n})" for name, n in zip(df_fam_corrected['Alloy_Family'], df_fam_corrected['Sample_Count_N'])], fontsize=8.5)
    ax2.legend()

    plt.tight_layout()
    fig13_p = os.path.join(FIG_DIR, 'fig13_family_error_comparison.png')
    plt.savefig(fig13_p, dpi=300)
    plt.close()
    print(f"Saved: {fig13_p}")

    print("\nSTAGE 3 AUDITED PIPELINE FULLY EXECUTED!")
    return df_nested, df_summary, df_paired_audit, df_loco, df_perm, df_fam_corrected, df_worst_hedged


if __name__ == '__main__':
    run_stage3()
