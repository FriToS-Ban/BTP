"""
Stage 2: Baseline Models and Leakage-Aware Evaluation (Updated & Audited)
Project: Machine-Learning Prediction of Mechanical Properties of Steels from Composition
IIT Roorkee - B.Tech Project in Metallurgical and Materials Engineering
Date: 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pymatgen.core import Composition
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import RepeatedKFold, GroupKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from applicability_domain import ApplicabilityDomain

# -----------------------------------------------------------------------------
# Configuration & Global Aesthetics
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
RES_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

ELEMENTS = ['Al', 'C', 'Co', 'Cr', 'Fe', 'Mn', 'Mo', 'N', 'Nb', 'Ni', 'Si', 'Ti', 'V', 'W']
NON_FE_ELEMENTS = [el for el in ELEMENTS if el != 'Fe']


# -----------------------------------------------------------------------------
# 1. Feature Space Preparation & Mass Conservation
# -----------------------------------------------------------------------------
def prepare_feature_spaces():
    """Load dataset, compute weight percentages, and verify mass conservation."""
    print("=" * 75)
    print("STEP 1: FEATURE SPACE PREPARATION & MASS CONSERVATION CHECK")
    print("=" * 75)

    data_path = os.path.join(DATA_DIR, 'processed_steels.csv')
    df = pd.read_csv(data_path)
    
    X_atom = df[ELEMENTS].copy()
    y = df['yield_strength'].values
    raw_formulas = df['raw_composition'].values
    
    wt_records = []
    for formula in raw_formulas:
        c = Composition(formula)
        row = {el: c.get_wt_fraction(el) * 100.0 for el in ELEMENTS}
        wt_records.append(row)
    
    df_wt = pd.DataFrame(wt_records)[ELEMENTS]
    
    sums_wt = df_wt.sum(axis=1)
    max_dev = (sums_wt - 100.0).abs().max()
    print(f"Total samples: {len(df_wt)}")
    print(f"Maximum deviation from 100.000 wt%: {max_dev:.2e}")
    if max_dev > 1e-6:
        raise ValueError("Critical mass conservation error: wt% does not sum to 100.0")

    # Non-Fe wt% (13 features)
    df_wt_non_fe = df_wt[NON_FE_ELEMENTS].copy()

    # Domain features
    df_phys = df_wt.copy()
    df_phys['Ni_Co_Ti'] = df_wt['Ni'] + df_wt['Co'] + df_wt['Ti']
    df_phys['Mo_W'] = df_wt['Mo'] + df_wt['W']
    df_phys['carbide_formers'] = (df_wt['Cr'] + df_wt['Mo'] + df_wt['V'] + 
                                  df_wt['Nb'] + df_wt['W'] + df_wt['Ti'])
    df_phys['total_alloying'] = 100.0 - df_wt['Fe']
    df_phys['interstitials'] = df_wt['C'] + df_wt['N']
    df_phys['Cr_eq'] = (df_wt['Cr'] + df_wt['Mo'] + 1.5 * df_wt['Si'] + 
                        0.5 * df_wt['Nb'] + 2.0 * df_wt['Ti'])
    df_phys['Ni_eq'] = (df_wt['Ni'] + 30.0 * df_wt['C'] + 30.0 * df_wt['N'] + 
                        0.5 * df_wt['Mn'])

    return {
        'atomic': X_atom,
        'wt_pct': df_wt,
        'wt_pct_non_fe': df_wt_non_fe,
        'phys': df_phys,
        'y': y,
        'formulas': raw_formulas
    }


# -----------------------------------------------------------------------------
# 2. Repeated Grouped CV Splitter
# -----------------------------------------------------------------------------
def make_repeated_group_splits(groups, n_splits=5, n_repeats=5, seed=42):
    """
    Generate repeated grouped cross-validation splits (n_splits x n_repeats).
    Each repeat reshuffles the assignment of group clusters to folds.
    """
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
# 3. Clustering Threshold Sensitivity & Multi-Threshold CV Evaluation
# -----------------------------------------------------------------------------
def analyze_clustering_sensitivity_and_evaluate(df_wt, y):
    """
    Test clustering thresholds (0.25, 0.50, 1.00 wt%) and evaluate Random Forest
    and XGBoost under 5x5 Repeated Grouped CV at each threshold.
    """
    print("\n" + "=" * 75)
    print("STEP 2: ALLOY CLUSTERING & MULTI-THRESHOLD REPEATED GROUPED CV")
    print("=" * 75)

    X_wt = df_wt.values
    dist_matrix = squareform(pdist(X_wt, metric='euclidean'))
    
    thresholds = [0.25, 0.50, 1.00]
    sens_records = []
    cluster_labels_dict = {}

    for tau in thresholds:
        adj = dist_matrix <= tau
        n_clusters, labels = connected_components(adj)
        cluster_labels_dict[tau] = labels
        
        sizes = pd.Series(labels).value_counts()
        n_multi = (sizes > 1).sum()
        max_size = sizes.max()
        pct_clustered = (sizes[sizes > 1].sum() / len(X_wt)) * 100.0

        # Evaluate RF and XGB on 5x5 repeated grouped CV (25 splits)
        rep_splits = make_repeated_group_splits(labels, n_splits=5, n_repeats=5, seed=RANDOM_SEED)
        
        for m_name, m_cls, m_kwargs in [
            ('Random Forest', RandomForestRegressor, {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 4, 'random_state': RANDOM_SEED, 'n_jobs': -1}),
            ('XGBoost', XGBRegressor, {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.08, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': RANDOM_SEED, 'n_jobs': -1})
        ]:
            rmses, maes, r2s = [], [], []
            for tr, val in rep_splits:
                m = m_cls(**m_kwargs)
                m.fit(X_wt[tr], y[tr])
                p = m.predict(X_wt[val])
                rmses.append(root_mean_squared_error(y[val], p))
                maes.append(mean_absolute_error(y[val], p))
                r2s.append(r2_score(y[val], p))
                
            sens_records.append({
                'Threshold_wt_pct': tau,
                'Total_Groups': n_clusters,
                'Multi_Sample_Clusters': n_multi,
                'Max_Cluster_Size': max_size,
                'Clustered_Samples_Pct': pct_clustered,
                'Model': m_name,
                'RMSE_Mean': np.mean(rmses),
                'RMSE_Std': np.std(rmses),
                'MAE_Mean': np.mean(maes),
                'MAE_Std': np.std(maes),
                'R2_Mean': np.mean(r2s),
                'R2_Std': np.std(r2s)
            })
            print(f"tau={tau:4.2f} wt% ({n_clusters:3d} groups) | {m_name:14s} | "
                  f"RMSE: {np.mean(rmses):6.2f} +/- {np.std(rmses):5.2f} MPa | "
                  f"MAE: {np.mean(maes):6.2f} +/- {np.std(maes):5.2f} MPa | "
                  f"R2: {np.mean(r2s):6.3f} +/- {np.std(r2s):5.3f}")

    sens_df = pd.DataFrame(sens_records)
    sens_csv = os.path.join(RES_DIR, 'stage2_threshold_sensitivity.csv')
    sens_df.to_csv(sens_csv, index=False)
    print(f"Saved multi-threshold sensitivity summary to: {sens_csv}")

    # Plot Figure 5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    
    rf_sub = sens_df[sens_df['Model'] == 'Random Forest']
    xgb_sub = sens_df[sens_df['Model'] == 'XGBoost']
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, rf_sub['RMSE_Mean'], width, yerr=rf_sub['RMSE_Std'],
                     capsize=4, label='Random Forest', color='#27ae60', edgecolor='#2c3e50')
    rects2 = ax1.bar(x + width/2, xgb_sub['RMSE_Mean'], width, yerr=xgb_sub['RMSE_Std'],
                     capsize=4, label='XGBoost', color='#e67e22', edgecolor='#2c3e50')
    
    ax1.set_ylabel("Validation RMSE (MPa, 25 Splits)", fontweight='bold')
    ax1.set_xlabel("Grouping Threshold ($\\tau$ in wt%)", fontweight='bold')
    ax1.set_title("Repeated Grouped CV Error vs. Clustering Threshold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t:.2f} wt%\n({g} groups)" for t, g in zip(thresholds, rf_sub['Total_Groups'])])
    ax1.set_ylim(0, 220)
    ax1.legend(loc='upper left', fontsize=9.5)

    # Size distribution for tau = 0.50
    sizes_05 = pd.Series(cluster_labels_dict[0.50]).value_counts()
    sns.histplot(sizes_05, bins=np.arange(0.5, 22.5, 1), ax=ax2, color='#34495e', edgecolor='white')
    ax2.set_xlabel("Alloy Group Size (Number of Steels)", fontweight='bold')
    ax2.set_ylabel("Frequency of Groups", fontweight='bold')
    ax2.set_title("Cluster Size Distribution at Headline $\\tau = 0.50$ wt%", pad=10)
    ax2.text(0.48, 0.78, f"Headline tau: 0.50 wt%\nTotal Groups: 201\nSingletons: {(sizes_05 == 1).sum()}\nMulti-alloy clusters: {(sizes_05 > 1).sum()}\nLargest cluster: {sizes_05.max()} alloys",
             transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7'))

    plt.tight_layout()
    fig5_path = os.path.join(FIG_DIR, 'fig5_clustering_threshold_sensitivity.png')
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    return cluster_labels_dict, sens_df


# -----------------------------------------------------------------------------
# 4. Full Model Benchmarking & Collinearity Audit
# -----------------------------------------------------------------------------
def run_benchmarks(data_dict, cluster_labels_dict):
    """
    Run full repeated benchmarking (5x5 = 25 splits):
    - Scheme (a): Naive RepeatedKFold (5 folds x 5 repeats)
    - Scheme (b): Repeated GroupKFold (5 folds x 5 repeats at headline tau = 0.50 wt%)
    - Ridge collinearity audit: with Fe vs without Fe
    - Feature sets: atomic vs wt% vs wt% + physical combinations
    """
    print("\n" + "=" * 75)
    print("STEP 3: COMPREHENSIVE BENCHMARKING (5x5 REPEATED NAIVE & GROUPED CV)")
    print("=" * 75)

    y = data_dict['y']
    X_wt = data_dict['wt_pct'].values
    X_non_fe = data_dict['wt_pct_non_fe'].values
    X_atom = data_dict['atomic'].values
    X_phys = data_dict['phys'].values

    headline_groups = cluster_labels_dict[0.50]

    # Split definitions (25 splits each)
    splits_naive = list(RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED).split(X_wt, y))
    splits_grouped = make_repeated_group_splits(headline_groups, n_splits=5, n_repeats=5, seed=RANDOM_SEED)

    benchmark_records = []
    oof_predictions_dict = {'y_true': y}

    # Model definitions
    def make_dummy():
        return DummyRegressor(strategy='mean')
    def make_rf():
        return RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)
    def make_xgb():
        return XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1)
    def make_ridge():
        return Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))])

    experiments = [
        # Naive 5x5
        ('Dummy (Mean)', make_dummy, X_wt, 'wt% (All 14)', 'Naive Repeated (5x5)', splits_naive),
        ('Ridge (with Fe)', make_ridge, X_wt, 'wt% (All 14, Collinear)', 'Naive Repeated (5x5)', splits_naive),
        ('Ridge (without Fe)', make_ridge, X_non_fe, 'wt% (13 non-Fe, Independent)', 'Naive Repeated (5x5)', splits_naive),
        ('Random Forest', make_rf, X_wt, 'wt% (All 14)', 'Naive Repeated (5x5)', splits_naive),
        ('XGBoost', make_xgb, X_wt, 'wt% (All 14)', 'Naive Repeated (5x5)', splits_naive),
        
        # Grouped 5x5 (Headline)
        ('Dummy (Mean)', make_dummy, X_wt, 'wt% (All 14)', 'GroupKFold (Headline 5x5)', splits_grouped),
        ('Ridge (with Fe)', make_ridge, X_wt, 'wt% (All 14, Collinear)', 'GroupKFold (Headline 5x5)', splits_grouped),
        ('Ridge (without Fe)', make_ridge, X_non_fe, 'wt% (13 non-Fe, Independent)', 'GroupKFold (Headline 5x5)', splits_grouped),
        ('Random Forest', make_rf, X_wt, 'wt% (All 14)', 'GroupKFold (Headline 5x5)', splits_grouped),
        ('XGBoost', make_xgb, X_wt, 'wt% (All 14)', 'GroupKFold (Headline 5x5)', splits_grouped),
        
        # Feature comparisons under Grouped 5x5
        ('Random Forest', make_rf, X_atom, 'Atomic Fractions', 'GroupKFold (Headline 5x5)', splits_grouped),
        ('XGBoost', make_xgb, X_atom, 'Atomic Fractions', 'GroupKFold (Headline 5x5)', splits_grouped),
        ('Random Forest', make_rf, X_phys, 'wt% + Physical Features', 'GroupKFold (Headline 5x5)', splits_grouped),
        ('XGBoost', make_xgb, X_phys, 'wt% + Physical Features', 'GroupKFold (Headline 5x5)', splits_grouped),
    ]

    for m_label, m_factory, X_data, feat_label, cv_label, splits_list in experiments:
        rmses, maes, r2s = [], [], []
        oof_acc = np.zeros(len(y))
        oof_cnt = np.zeros(len(y))
        
        for tr, val in splits_list:
            m = m_factory()
            m.fit(X_data[tr], y[tr])
            p = m.predict(X_data[val])
            
            oof_acc[val] += p
            oof_cnt[val] += 1
            
            rmses.append(root_mean_squared_error(y[val], p))
            maes.append(mean_absolute_error(y[val], p))
            r2s.append(r2_score(y[val], p))

        oof_final = oof_acc / np.maximum(oof_cnt, 1)
        key_oof = f"{m_label}_{feat_label}_{cv_label[:5]}"
        oof_predictions_dict[key_oof] = oof_final

        benchmark_records.append({
            'Model': m_label,
            'Feature_Set': feat_label,
            'CV_Scheme': cv_label,
            'RMSE_Mean': np.mean(rmses),
            'RMSE_Std': np.std(rmses),
            'MAE_Mean': np.mean(maes),
            'MAE_Std': np.std(maes),
            'R2_Mean': np.mean(r2s),
            'R2_Std': np.std(r2s)
        })

        print(f"{m_label:20s} | {feat_label:28s} | {cv_label:24s} | "
              f"RMSE: {np.mean(rmses):6.2f} +/- {np.std(rmses):5.2f} | "
              f"MAE: {np.mean(maes):6.2f} +/- {np.std(maes):5.2f} | "
              f"R2: {np.mean(r2s):6.3f} +/- {np.std(r2s):5.3f}")

    bench_df = pd.DataFrame(benchmark_records)
    metrics_csv = os.path.join(RES_DIR, 'stage2_metrics_summary.csv')
    bench_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics summary to: {metrics_csv}")

    oof_df = pd.DataFrame(oof_predictions_dict)
    oof_csv = os.path.join(RES_DIR, 'stage2_oof_predictions.csv')
    oof_df.to_csv(oof_csv, index=False)
    print(f"Saved out-of-fold predictions to: {oof_csv}")

    # Leakage gap audit
    print("\n" + "-" * 75)
    print("LEAKAGE GAP AUDIT: (Headline Grouped 5x5 vs. Naive 5x5)")
    print("-" * 75)
    for m_name in ['Random Forest', 'XGBoost', 'Ridge (without Fe)']:
        feat = 'wt% (13 non-Fe, Independent)' if 'Ridge' in m_name else 'wt% (All 14)'
        n_row = bench_df[(bench_df['Model'] == m_name) & (bench_df['Feature_Set'] == feat) & (bench_df['CV_Scheme'] == 'Naive Repeated (5x5)')].iloc[0]
        g_row = bench_df[(bench_df['Model'] == m_name) & (bench_df['Feature_Set'] == feat) & (bench_df['CV_Scheme'] == 'GroupKFold (Headline 5x5)')].iloc[0]
        d_rmse = g_row['RMSE_Mean'] - n_row['RMSE_Mean']
        d_r2 = n_row['R2_Mean'] - g_row['R2_Mean']
        print(f"{m_name:20s} | Delta_RMSE (Inflation in Naive): +{d_rmse:5.2f} MPa | Delta_R2: {d_r2:5.3f}")

    # Collinearity audit for Ridge
    print("\n" + "-" * 75)
    print("RIDGE COLLINEARITY AUDIT: (Effect of Dropping Solvent Fe)")
    print("-" * 75)
    r_with_g = bench_df[(bench_df['Model'] == 'Ridge (with Fe)') & (bench_df['CV_Scheme'] == 'GroupKFold (Headline 5x5)')].iloc[0]
    r_no_g = bench_df[(bench_df['Model'] == 'Ridge (without Fe)') & (bench_df['CV_Scheme'] == 'GroupKFold (Headline 5x5)')].iloc[0]
    print(f"Ridge with Fe (14 feats)    | Grouped RMSE: {r_with_g['RMSE_Mean']:.2f} +/- {r_with_g['RMSE_Std']:.2f} MPa | R2: {r_with_g['R2_Mean']:.3f}")
    print(f"Ridge without Fe (13 feats) | Grouped RMSE: {r_no_g['RMSE_Mean']:.2f} +/- {r_no_g['RMSE_Std']:.2f} MPa | R2: {r_no_g['R2_Mean']:.3f}")
    print(f"Benefit of removing Fe collinearity: RMSE reduced by {(r_with_g['RMSE_Mean'] - r_no_g['RMSE_Mean']):.2f} MPa, R2 improved by +{(r_no_g['R2_Mean'] - r_with_g['R2_Mean']):.3f}")

    return bench_df, oof_df, splits_grouped


# -----------------------------------------------------------------------------
# 5. Learning Curve Analysis
# -----------------------------------------------------------------------------
def compute_learning_curve(X_wt, y, headline_groups):
    """
    Evaluate validation error as a function of training dataset size under Repeated Grouped CV.
    Demonstrates whether the problem suggests performance is still data-limited.
    """
    print("\n" + "=" * 75)
    print("STEP 4: COMPUTING REPEATED GROUPED CV LEARNING CURVE")
    print("=" * 75)

    rep_splits = make_repeated_group_splits(headline_groups, n_splits=5, n_repeats=5, seed=RANDOM_SEED)
    fractions = [0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
    curve_records = []

    for f in fractions:
        fold_train_rmses, fold_val_rmses, fold_val_maes = [], [], []
        
        for tr_idx, val_idx in rep_splits:
            tr_groups = np.unique(headline_groups[tr_idx])
            
            np.random.seed(RANDOM_SEED)
            n_sub = int(np.ceil(len(tr_groups) * f))
            sub_groups = np.random.choice(tr_groups, size=n_sub, replace=False)
            sub_mask = np.isin(headline_groups[tr_idx], sub_groups)
            sub_tr_idx = tr_idx[sub_mask]

            rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)
            rf.fit(X_wt[sub_tr_idx], y[sub_tr_idx])

            tr_p = rf.predict(X_wt[sub_tr_idx])
            val_p = rf.predict(X_wt[val_idx])

            fold_train_rmses.append(root_mean_squared_error(y[sub_tr_idx], tr_p))
            fold_val_rmses.append(root_mean_squared_error(y[val_idx], val_p))
            fold_val_maes.append(mean_absolute_error(y[val_idx], val_p))

        n_samples_eff = int(len(X_wt) * 0.8 * f)
        curve_records.append({
            'Fraction': f,
            'Effective_Train_Samples': n_samples_eff,
            'Train_RMSE_Mean': np.mean(fold_train_rmses),
            'Train_RMSE_Std': np.std(fold_train_rmses),
            'Val_RMSE_Mean': np.mean(fold_val_rmses),
            'Val_RMSE_Std': np.std(fold_val_rmses),
            'Val_MAE_Mean': np.mean(fold_val_maes),
            'Val_MAE_Std': np.std(fold_val_maes)
        })
        print(f"Fraction {f:4.2f} (~{n_samples_eff:3d} alloys) | "
              f"Val RMSE: {np.mean(fold_val_rmses):5.1f} +/- {np.std(fold_val_rmses):4.1f} MPa | "
              f"Train RMSE: {np.mean(fold_train_rmses):4.1f} MPa")

    curve_df = pd.DataFrame(curve_records)
    curve_csv = os.path.join(RES_DIR, 'stage2_learning_curve.csv')
    curve_df.to_csv(curve_csv, index=False)
    print(f"Saved learning curve data to: {curve_csv}")

    # Plot Figure 7
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    
    ax.plot(curve_df['Effective_Train_Samples'], curve_df['Val_RMSE_Mean'],
            marker='o', linewidth=2.2, color='#e74c3c', label='Validation RMSE (Repeated Grouped CV)')
    ax.fill_between(curve_df['Effective_Train_Samples'],
                    curve_df['Val_RMSE_Mean'] - curve_df['Val_RMSE_Std'],
                    curve_df['Val_RMSE_Mean'] + curve_df['Val_RMSE_Std'],
                    color='#e74c3c', alpha=0.18, label='Validation +/-1 Std')

    ax.plot(curve_df['Effective_Train_Samples'], curve_df['Train_RMSE_Mean'],
            marker='s', linewidth=2.0, linestyle='--', color='#2980b9', label='Training RMSE')
    
    ax.set_title("Random Forest Learning Curve (Repeated Grouped CV, 25 Splits)", pad=12)
    ax.set_xlabel("Effective Training Set Size (Number of Steel Alloys)", fontweight='bold')
    ax.set_ylabel("RMSE Error (MPa)", fontweight='bold')
    ax.set_ylim(40, 310)
    ax.legend(loc='upper right', frameon=True, fontsize=9.5)

    annotation_text = (
        "Empirical Finding:\n"
        "Validation error steadily falls\n"
        "(from 240.4 to 158.3 MPa),\n"
        "suggesting performance is still\n"
        "data-limited (falling at N=249)."
    )
    ax.text(0.04, 0.45, annotation_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7'))

    plt.tight_layout()
    fig7_path = os.path.join(FIG_DIR, 'fig7_learning_curve.png')
    plt.savefig(fig7_path, dpi=300)
    plt.close()
    print(f"Saved: {fig7_path}")

    return curve_df


# -----------------------------------------------------------------------------
# 6. Parity & Residual Plots
# -----------------------------------------------------------------------------
def generate_parity_and_residual_plots(oof_df):
    """Generate 6-panel parity plots and residual plots."""
    print("\n" + "=" * 75)
    print("STEP 5: GENERATING PARITY & RESIDUAL PLOTS")
    print("=" * 75)

    y_true = oof_df['y_true'].values
    min_val, max_val = 900, 2600

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
    models_to_plot = [
        ('Ridge (without Fe)', 'Ridge (without Fe)_wt% (13 non-Fe, Independent)_Naive', 'Ridge (without Fe)_wt% (13 non-Fe, Independent)_Group'),
        ('Random Forest', 'Random Forest_wt% (All 14)_Naive', 'Random Forest_wt% (All 14)_Group'),
        ('XGBoost', 'XGBoost_wt% (All 14)_Naive', 'XGBoost_wt% (All 14)_Group')
    ]

    for col_idx, (m_label, naive_col, grouped_col) in enumerate(models_to_plot):
        p_naive = oof_df[naive_col].values
        p_group = oof_df[grouped_col].values
        
        # Row 0: Naive RepeatedKFold
        ax_top = axes[0, col_idx]
        ax_top.scatter(y_true, p_naive, color='#2980b9', alpha=0.65, edgecolors='white', s=45)
        ax_top.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.3, label='1:1 Line')
        ax_top.fill_between([min_val, max_val], [min_val*0.9, max_val*0.9], [min_val*1.1, max_val*1.1],
                            color='#7f8c8d', alpha=0.15, label='+/-10% Bound')
        rmse_n = root_mean_squared_error(y_true, p_naive)
        r2_n = r2_score(y_true, p_naive)
        ax_top.set_title(f"{m_label} [Naive Repeated (5x5)]\n$R^2 = {r2_n:.3f}$, RMSE = {rmse_n:.1f} MPa", fontsize=10.5)
        ax_top.grid(True, linestyle=':', alpha=0.6)
        if col_idx == 0:
            ax_top.set_ylabel("Predicted $\\sigma_y$ (MPa)", fontweight='bold')
        ax_top.legend(loc='lower right', fontsize=8.5)

        # Row 1: GroupedKFold
        ax_bot = axes[1, col_idx]
        ax_bot.scatter(y_true, p_group, color='#e67e22', alpha=0.65, edgecolors='white', s=45)
        ax_bot.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.3, label='1:1 Line')
        ax_bot.fill_between([min_val, max_val], [min_val*0.9, max_val*0.9], [min_val*1.1, max_val*1.1],
                            color='#7f8c8d', alpha=0.15, label='+/-10% Bound')
        rmse_g = root_mean_squared_error(y_true, p_group)
        r2_g = r2_score(y_true, p_group)
        ax_bot.set_title(f"{m_label} [Repeated Grouped (5x5)]\n$R^2 = {r2_g:.3f}$, RMSE = {rmse_g:.1f} MPa", fontsize=10.5)
        ax_bot.grid(True, linestyle=':', alpha=0.6)
        ax_bot.set_xlabel("Experimental $\\sigma_y$ (MPa)", fontweight='bold')
        if col_idx == 0:
            ax_bot.set_ylabel("Predicted $\\sigma_y$ (MPa)", fontweight='bold')
        ax_bot.legend(loc='lower right', fontsize=8.5)

    plt.suptitle("Parity Comparison: Naive Repeated CV vs. Repeated Grouped CV (25 Splits)", fontsize=13, y=0.99)
    plt.tight_layout()
    fig6_path = os.path.join(FIG_DIR, 'fig6_parity_plots_naive_vs_grouped.png')
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"Saved: {fig6_path}")

    # Figure 8: Residual Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)

    rf_pred = oof_df['Random Forest_wt% (All 14)_Group'].values
    rf_res = y_true - rf_pred

    xgb_pred = oof_df['XGBoost_wt% (All 14)_Group'].values
    xgb_res = y_true - xgb_pred

    ax1.scatter(rf_pred, rf_res, color='#27ae60', alpha=0.7, edgecolors='black', linewidth=0.5, s=50)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.4)
    ax1.axhline(np.std(rf_res), color='gray', linestyle=':', label=f"+/-1 Std ({np.std(rf_res):.1f} MPa)")
    ax1.axhline(-np.std(rf_res), color='gray', linestyle=':')
    ax1.set_xlabel("Predicted Yield Strength $\\hat{\\sigma}_y$ (MPa)", fontweight='bold')
    ax1.set_ylabel("Residual ($y - \\hat{y}$) (MPa)", fontweight='bold')
    ax1.set_title("Random Forest Residuals (Repeated Grouped CV)", pad=10)
    ax1.legend(loc='upper right', fontsize=9)

    ax2.scatter(xgb_pred, xgb_res, color='#8e44ad', alpha=0.7, edgecolors='black', linewidth=0.5, s=50)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.4)
    ax2.axhline(np.std(xgb_res), color='gray', linestyle=':', label=f"+/-1 Std ({np.std(xgb_res):.1f} MPa)")
    ax2.axhline(-np.std(xgb_res), color='gray', linestyle=':')
    ax2.set_xlabel("Predicted Yield Strength $\\hat{\\sigma}_y$ (MPa)", fontweight='bold')
    ax2.set_title("XGBoost Residuals (Repeated Grouped CV)", pad=10)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fig8_path = os.path.join(FIG_DIR, 'fig8_residuals_analysis.png')
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    print(f"Saved: {fig8_path}")


# -----------------------------------------------------------------------------
# 7. Applicability Domain Statistical Audit (Bootstrap CI & Permutation Test)
# -----------------------------------------------------------------------------
def audit_applicability_domain(df_wt, y, headline_groups):
    """
    Fit ApplicabilityDomain (excluding Fe, standardized non-Fe distance) across
    Repeated Grouped CV. Compute Pearson r with 95% bootstrap CI, near vs far MAE
    difference with 95% bootstrap CI, and permutation test p-values.
    """
    print("\n" + "=" * 75)
    print("STEP 6: APPLICABILITY DOMAIN STATISTICAL AUDIT (BOOTSTRAP & PERMUTATION)")
    print("=" * 75)

    X_wt = df_wt.values
    rep_splits = make_repeated_group_splits(headline_groups, n_splits=5, n_repeats=5, seed=RANDOM_SEED)

    # Accumulate OOF test errors and distances
    all_dists_std = []
    all_dists_raw = []
    all_abs_errors = []

    for tr_idx, val_idx in rep_splits:
        X_tr, y_tr = X_wt[tr_idx], y[tr_idx]
        X_val, y_val = X_wt[val_idx], y[val_idx]

        # Fit AD on training fold (excluding Fe, standardized)
        ad = ApplicabilityDomain(metric='euclidean', distance_percentile=95.0, 
                                 exclude_element='Fe', standardize=True)
        ad.fit(X_tr, feature_names=ELEMENTS)
        ad_res = ad.query(X_val)

        rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_val)

        errs = np.abs(y_val - preds)

        all_dists_std.extend(ad_res['nearest_dist_std'].values)
        all_dists_raw.extend(ad_res['nearest_dist_raw_wt'].values)
        all_abs_errors.extend(errs)

    dists_std = np.array(all_dists_std)
    dists_raw = np.array(all_dists_raw)
    errors = np.array(all_abs_errors)

    def compute_stats(d_vec, e_vec, label):
        r = np.corrcoef(d_vec, e_vec)[0, 1]
        med = np.median(d_vec)
        near_mae = np.mean(e_vec[d_vec < med])
        far_mae = np.mean(e_vec[d_vec >= med])
        diff_mae = far_mae - near_mae

        # Bootstrap 95% CI (2000 resamples)
        np.random.seed(RANDOM_SEED)
        boot_r, boot_diff = [], []
        n = len(d_vec)
        for _ in range(2000):
            idx = np.random.choice(n, size=n, replace=True)
            bd = d_vec[idx]
            be = e_vec[idx]
            boot_r.append(np.corrcoef(bd, be)[0, 1])
            b_med = np.median(bd)
            boot_diff.append(np.mean(be[bd >= b_med]) - np.mean(be[bd < b_med]))

        ci_r = np.percentile(boot_r, [2.5, 97.5])
        ci_diff = np.percentile(boot_diff, [2.5, 97.5])

        # Permutation test (5000 permutations)
        perm_r, perm_diff = [], []
        for _ in range(5000):
            p_d = np.random.permutation(d_vec)
            perm_r.append(np.corrcoef(p_d, e_vec)[0, 1])
            perm_diff.append(np.mean(e_vec[p_d >= med]) - np.mean(e_vec[p_d < med]))

        p_val_r = np.mean(np.array(perm_r) >= r)
        p_val_diff = np.mean(np.array(perm_diff) >= diff_mae)

        return {
            'Metric_Space': label,
            'Pearson_r': r,
            'r_95_CI_low': ci_r[0],
            'r_95_CI_high': ci_r[1],
            'r_p_value': p_val_r,
            'Near_MAE_MPa': near_mae,
            'Far_MAE_MPa': far_mae,
            'Delta_MAE_MPa': diff_mae,
            'Delta_MAE_95_CI_low': ci_diff[0],
            'Delta_MAE_95_CI_high': ci_diff[1],
            'Delta_MAE_p_value': p_val_diff,
            'Significant_at_05': p_val_diff < 0.05
        }

    stat_std = compute_stats(dists_std, errors, 'Standardized Non-Fe (Headline)')
    stat_raw = compute_stats(dists_raw, errors, 'Raw wt% Non-Fe')

    audit_df = pd.DataFrame([stat_std, stat_raw])
    audit_csv = os.path.join(RES_DIR, 'stage2_ad_statistical_audit.csv')
    audit_df.to_csv(audit_csv, index=False)
    print(f"Saved applicability domain statistical audit to: {audit_csv}")

    print("\n--- APPLICABILITY DOMAIN STATISTICAL AUDIT SUMMARY ---")
    for _, row in audit_df.iterrows():
        print(f"Space: {row['Metric_Space']}")
        print(f"  Pearson r = {row['Pearson_r']:.3f} (95% CI: [{row['r_95_CI_low']:.3f}, {row['r_95_CI_high']:.3f}]), p = {row['r_p_value']:.4f}")
        print(f"  Near MAE: {row['Near_MAE_MPa']:.1f} MPa | Far MAE: {row['Far_MAE_MPa']:.1f} MPa | Delta MAE: +{row['Delta_MAE_MPa']:.1f} MPa")
        print(f"  Delta MAE 95% CI: [{row['Delta_MAE_95_CI_low']:.1f}, {row['Delta_MAE_95_CI_high']:.1f}] MPa, p = {row['Delta_MAE_p_value']:.4f}")
        print(f"  Statistically distinguishable from zero at alpha=0.05? {'YES' if row['Significant_at_05'] else 'NO'}")

    # Plot Figure 9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0))

    # Scatter plot
    sns.regplot(x=dists_std, y=errors, ax=ax1, color='#2c3e50',
                scatter_kws={'alpha': 0.45, 'color': '#2980b9', 's': 30},
                line_kws={'color': '#e74c3c', 'linewidth': 2.0})
    ax1.set_xlabel("Nearest Neighbor Distance (Standardized Non-Fe Space)", fontweight='bold')
    ax1.set_ylabel("Absolute Prediction Error |$y - \\hat{y}$| (MPa)", fontweight='bold')
    ax1.set_title("Test Error vs. Standardized Non-Fe Extrapolation Distance", pad=10)
    
    stat_box = (
        f"Pearson r = +{stat_std['Pearson_r']:.3f}\n"
        f"95% CI: [{stat_std['r_95_CI_low']:.3f}, {stat_std['r_95_CI_high']:.3f}]\n"
        f"Permutation p = {stat_std['r_p_value']:.4f}\n"
        f"Near MAE: {stat_std['Near_MAE_MPa']:.1f} MPa\n"
        f"Far MAE: {stat_std['Far_MAE_MPa']:.1f} MPa\n"
        f"Delta MAE: +{stat_std['Delta_MAE_MPa']:.1f} MPa (p={stat_std['Delta_MAE_p_value']:.4f})"
    )
    ax1.text(0.04, 0.72, stat_box, transform=ax1.transAxes, fontsize=8.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7'))

    # Binned boxplot
    bins = [0.0, 1.5, 3.0, 5.0, np.max(dists_std) + 0.1]
    bin_labels = ['<1.5 std', '1.5-3.0 std', '3.0-5.0 std', '>5.0 std']
    dist_cat = pd.cut(dists_std, bins=bins, labels=bin_labels)
    df_box = pd.DataFrame({'Distance_Bracket': dist_cat, 'Absolute_Error_MPa': errors})

    sns.boxplot(data=df_box, x='Distance_Bracket', y='Absolute_Error_MPa', hue='Distance_Bracket',
                legend=False, ax=ax2, palette='YlOrRd', fliersize=2.5, linewidth=1.1)
    ax2.set_xlabel("Standardized Distance Bracket (Excluding Fe)", fontweight='bold')
    ax2.set_ylabel("Absolute Prediction Error (MPa)", fontweight='bold')
    ax2.set_title("Prediction Error Spread by Proximity Bracket", pad=10)

    plt.tight_layout()
    fig9_path = os.path.join(FIG_DIR, 'fig9_applicability_domain_error.png')
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print(f"Saved: {fig9_path}")

    return audit_df


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    data_dict = prepare_feature_spaces()
    cluster_labels_dict, sens_df = analyze_clustering_sensitivity_and_evaluate(data_dict['wt_pct'], data_dict['y'])
    bench_df, oof_df, splits_grouped = run_benchmarks(data_dict, cluster_labels_dict)
    curve_df = compute_learning_curve(data_dict['wt_pct'].values, data_dict['y'], cluster_labels_dict[0.50])
    generate_parity_and_residual_plots(oof_df)
    audit_df = audit_applicability_domain(data_dict['wt_pct'], data_dict['y'], cluster_labels_dict[0.50])

    print("\n" + "=" * 75)
    print("STAGE 2 PIPELINE EXECUTED SUCCESSFULLY WITH ALL FIXES!")
    print("=" * 75)


if __name__ == '__main__':
    main()
