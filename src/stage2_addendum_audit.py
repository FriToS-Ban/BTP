"""
Stage 2 Addendum: Cluster-Level Applicability Domain Audit & Paired Comparisons
Project: Machine-Learning Prediction of Mechanical Properties of Steels from Composition
IIT Roorkee - B.Tech Project in Metallurgical and Materials Engineering
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.stats import ttest_rel, wilcoxon
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from pymatgen.core import Composition

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RES_DIR = os.path.join(BASE_DIR, 'results')

ELEMENTS = ['Al', 'C', 'Co', 'Cr', 'Fe', 'Mn', 'Mo', 'N', 'Nb', 'Ni', 'Si', 'Ti', 'V', 'W']
NON_FE_ELEMENTS = [el for el in ELEMENTS if el != 'Fe']


def load_data():
    df_raw = pd.read_csv(os.path.join(DATA_DIR, 'processed_steels.csv'))
    y = df_raw['yield_strength'].values
    raw_formulas = df_raw['raw_composition'].values
    
    wt_records = []
    for formula in raw_formulas:
        c = Composition(formula)
        row = {el: c.get_wt_fraction(el) * 100.0 for el in ELEMENTS}
        wt_records.append(row)
    df_wt = pd.DataFrame(wt_records)[ELEMENTS]
    
    # Non-Fe wt%
    df_wt_non_fe = df_wt[NON_FE_ELEMENTS].copy()
    
    # Atomic
    df_atom = df_raw[ELEMENTS].copy()
    
    # Phys
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

    # Clustering at tau = 0.50
    dist_matrix = squareform(pdist(df_wt.values, metric='euclidean'))
    adj = dist_matrix <= 0.50
    _, groups = connected_components(adj)

    return df_wt, df_wt_non_fe, df_atom, df_phys, y, groups


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


def run_paired_model_comparisons(df_wt, df_wt_non_fe, df_atom, df_phys, y, groups):
    print("=" * 80)
    print("PART 1: PAIRED MODEL COMPARISONS ACROSS 25 GROUPED CV SPLITS")
    print("=" * 80)
    
    splits = make_repeated_group_splits(groups, n_splits=5, n_repeats=5, seed=RANDOM_SEED)
    
    # Store per-split RMSE
    ridge_with_fe_rmse = []
    ridge_without_fe_rmse = []
    
    rf_wt_rmse, xgb_wt_rmse = [], []
    rf_atom_rmse, xgb_atom_rmse = [], []
    rf_phys_rmse, xgb_phys_rmse = [], []

    for tr, val in splits:
        # 1. Ridge with vs without Fe
        r_pipe1 = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))])
        r_pipe1.fit(df_wt.values[tr], y[tr])
        p1 = r_pipe1.predict(df_wt.values[val])
        ridge_with_fe_rmse.append(root_mean_squared_error(y[val], p1))
        
        r_pipe2 = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))])
        r_pipe2.fit(df_wt_non_fe.values[tr], y[tr])
        p2 = r_pipe2.predict(df_wt_non_fe.values[val])
        ridge_without_fe_rmse.append(root_mean_squared_error(y[val], p2))

        # 2. RF vs XGB on wt%
        rf_wt = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)
        rf_wt.fit(df_wt.values[tr], y[tr])
        rf_wt_rmse.append(root_mean_squared_error(y[val], rf_wt.predict(df_wt.values[val])))
        
        xgb_wt = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1)
        xgb_wt.fit(df_wt.values[tr], y[tr])
        xgb_wt_rmse.append(root_mean_squared_error(y[val], xgb_wt.predict(df_wt.values[val])))

        # 3. RF vs XGB on atomic
        rf_atom = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)
        rf_atom.fit(df_atom.values[tr], y[tr])
        rf_atom_rmse.append(root_mean_squared_error(y[val], rf_atom.predict(df_atom.values[val])))
        
        xgb_atom = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1)
        xgb_atom.fit(df_atom.values[tr], y[tr])
        xgb_atom_rmse.append(root_mean_squared_error(y[val], xgb_atom.predict(df_atom.values[val])))

        # 4. RF vs XGB on phys
        rf_phys = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)
        rf_phys.fit(df_phys.values[tr], y[tr])
        rf_phys_rmse.append(root_mean_squared_error(y[val], rf_phys.predict(df_phys.values[val])))
        
        xgb_phys = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1)
        xgb_phys.fit(df_phys.values[tr], y[tr])
        xgb_phys_rmse.append(root_mean_squared_error(y[val], xgb_phys.predict(df_phys.values[val])))

    comparisons = [
        ('Ridge: with Fe vs without Fe', np.array(ridge_with_fe_rmse), np.array(ridge_without_fe_rmse), 'with Fe - without Fe'),
        ('RF vs XGBoost (wt% All 14)', np.array(xgb_wt_rmse), np.array(rf_wt_rmse), 'XGBoost - RF'),
        ('RF vs XGBoost (Atomic Fractions)', np.array(xgb_atom_rmse), np.array(rf_atom_rmse), 'XGBoost - RF'),
        ('RF vs XGBoost (wt% + Physical)', np.array(xgb_phys_rmse), np.array(rf_phys_rmse), 'XGBoost - RF'),
    ]

    paired_results = []
    rng = np.random.RandomState(RANDOM_SEED)
    
    for comp_name, vec_a, vec_b, diff_definition in comparisons:
        diff = vec_a - vec_b
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        # Bootstrap 95% CI of the mean difference across splits (2000 resamples)
        boot_means = []
        n_splits = len(diff)
        for _ in range(2000):
            sample = rng.choice(diff, size=n_splits, replace=True)
            boot_means.append(np.mean(sample))
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        
        # Paired tests
        t_stat, t_pval = ttest_rel(vec_a, vec_b)
        w_stat, w_pval = wilcoxon(vec_a, vec_b)
        
        is_significant = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
        
        paired_results.append({
            'Comparison': comp_name,
            'Model_A_Mean_RMSE': np.mean(vec_a),
            'Model_B_Mean_RMSE': np.mean(vec_b),
            'Mean_Paired_Diff': mean_diff,
            'Diff_Definition': diff_definition,
            'CI_95_Low': ci_low,
            'CI_95_High': ci_high,
            'Paired_t_pvalue': t_pval,
            'Wilcoxon_pvalue': w_pval,
            'Distinguishable_from_Zero': 'YES' if is_significant else 'NO (CI includes 0)'
        })
        
        print(f"\n{comp_name}:")
        print(f"  Mean Paired Difference ({diff_definition}): {mean_diff:+.2f} MPa")
        print(f"  95% Bootstrap CI: [{ci_low:+.2f}, {ci_high:+.2f}] MPa")
        print(f"  Paired t-test p-value: {t_pval:.4f} | Wilcoxon p-value: {w_pval:.4f}")
        print(f"  Distinguishable from zero at alpha=0.05? {'YES' if is_significant else 'NO (CI crosses zero)'}")

    df_paired = pd.DataFrame(paired_results)
    df_paired.to_csv(os.path.join(RES_DIR, 'stage2_paired_model_comparisons.csv'), index=False)
    print(f"\nSaved paired comparisons to: {os.path.join(RES_DIR, 'stage2_paired_model_comparisons.csv')}")
    return df_paired


def run_corrected_applicability_domain_audit(df_wt, y, groups):
    print("\n" + "=" * 80)
    print("PART 2: CORRECTED APPLICABILITY DOMAIN STATISTICAL AUDIT")
    print("=" * 80)
    
    n_samples = len(df_wt)
    splits = make_repeated_group_splits(groups, n_splits=5, n_repeats=5, seed=RANDOM_SEED)
    
    # Prepare matrices
    X_all_14 = df_wt.values # with Fe
    X_non_fe = df_wt[NON_FE_ELEMENTS].values # 13 elements
    
    # Track per-alloy per-repeat error and distance
    # Each alloy appears once per repeat (5 repeats total)
    alloy_errors = [[] for _ in range(n_samples)]
    alloy_dists_std_non_fe = [[] for _ in range(n_samples)]
    alloy_dists_raw_non_fe = [[] for _ in range(n_samples)]
    alloy_dists_raw_with_fe = [[] for _ in range(n_samples)]
    
    # Also track pooled arrays for exact comparison
    pooled_errors = []
    pooled_dists_std_non_fe = []
    pooled_dists_raw_non_fe = []
    pooled_dists_raw_with_fe = []

    for tr_idx, val_idx in splits:
        # Check confirmation: nearest neighbors fitted STRICTLY on training set
        X_tr_14, y_tr = X_all_14[tr_idx], y[tr_idx]
        X_val_14, y_val = X_all_14[val_idx], y[val_idx]
        
        X_tr_non_fe = X_non_fe[tr_idx]
        X_val_non_fe = X_non_fe[val_idx]
        
        # 1. Raw wt% with Fe (14 elements)
        nn_with_fe = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_tr_14)
        d_raw_with_fe, _ = nn_with_fe.kneighbors(X_val_14)
        d_raw_with_fe = d_raw_with_fe.flatten()
        
        # 2. Raw wt% non-Fe (13 elements)
        nn_non_fe = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_tr_non_fe)
        d_raw_non_fe, _ = nn_non_fe.kneighbors(X_val_non_fe)
        d_raw_non_fe = d_raw_non_fe.flatten()
        
        # 3. Standardized non-Fe (13 elements standardized strictly on training set)
        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr_non_fe)
        X_val_std = scaler.transform(X_val_non_fe)
        nn_std = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_tr_std)
        d_std_non_fe, _ = nn_std.kneighbors(X_val_std)
        d_std_non_fe = d_std_non_fe.flatten()
        
        # Train RF model
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_tr_14, y_tr)
        preds = rf.predict(X_val_14)
        errs = np.abs(y_val - preds)
        
        # Record into per-alloy lists
        for i, idx in enumerate(val_idx):
            alloy_errors[idx].append(errs[i])
            alloy_dists_std_non_fe[idx].append(d_std_non_fe[i])
            alloy_dists_raw_non_fe[idx].append(d_raw_non_fe[i])
            alloy_dists_raw_with_fe[idx].append(d_raw_with_fe[i])
            
        pooled_errors.extend(errs)
        pooled_dists_std_non_fe.extend(d_std_non_fe)
        pooled_dists_raw_non_fe.extend(d_raw_non_fe)
        pooled_dists_raw_with_fe.extend(d_raw_with_fe)

    # Per-alloy averaged values (n = 312)
    mean_errors = np.array([np.mean(errs) for errs in alloy_errors])
    mean_dists_std_non_fe = np.array([np.mean(d) for d in alloy_dists_std_non_fe])
    mean_dists_raw_non_fe = np.array([np.mean(d) for d in alloy_dists_raw_non_fe])
    mean_dists_raw_with_fe = np.array([np.mean(d) for d in alloy_dists_raw_with_fe])
    
    unique_clusters = np.unique(groups)
    n_clusters = len(unique_clusters)
    print(f"Total alloys: {n_samples}, Total clusters at headline tau=0.50 wt%: {n_clusters}")

    # Helper function for cluster-level bootstrap and permutation testing
    def compute_cluster_level_stats(dist_vec, err_vec, cluster_labels, label_name):
        # Point estimates on n = 312
        r_point = np.corrcoef(dist_vec, err_vec)[0, 1]
        med_dist = np.median(dist_vec)
        near_mask = dist_vec < med_dist
        far_mask = dist_vec >= med_dist
        near_mae = np.mean(err_vec[near_mask])
        far_mae = np.mean(err_vec[far_mask])
        diff_mae = far_mae - near_mae

        # Cluster-level Bootstrap (2000 resamples of clusters)
        rng = np.random.RandomState(RANDOM_SEED)
        boot_r, boot_diff = [], []
        
        # Pre-group alloy indices by cluster
        cluster_to_indices = {c: np.where(cluster_labels == c)[0] for c in unique_clusters}
        
        for _ in range(2000):
            sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
            boot_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
            
            b_d = dist_vec[boot_indices]
            b_e = err_vec[boot_indices]
            
            boot_r.append(np.corrcoef(b_d, b_e)[0, 1])
            b_med = np.median(b_d)
            boot_diff.append(np.mean(b_e[b_d >= b_med]) - np.mean(b_e[b_d < b_med]))
            
        ci_r = np.percentile(boot_r, [2.5, 97.5])
        ci_diff = np.percentile(boot_diff, [2.5, 97.5])

        # Cluster-level Permutation Test (5000 permutations)
        # Under H0, error distributions across clusters are independent of cluster distance.
        # Permute cluster identities assigned to distance profiles:
        perm_r, perm_diff = [], []
        for _ in range(5000):
            permuted_clusters = rng.permutation(unique_clusters)
            # Map each cluster to a permuted cluster's alloys for the distance vector
            perm_dist_vec = np.zeros_like(dist_vec)
            for orig_c, perm_c in zip(unique_clusters, permuted_clusters):
                orig_idx = cluster_to_indices[orig_c]
                perm_idx = cluster_to_indices[perm_c]
                # If cluster sizes match, copy directly; otherwise sample with replacement from perm_c
                if len(orig_idx) == len(perm_idx):
                    perm_dist_vec[orig_idx] = dist_vec[perm_idx]
                else:
                    sampled_perm = rng.choice(dist_vec[perm_idx], size=len(orig_idx), replace=True)
                    perm_dist_vec[orig_idx] = sampled_perm
                    
            perm_r.append(np.corrcoef(perm_dist_vec, err_vec)[0, 1])
            p_med = np.median(perm_dist_vec)
            perm_diff.append(np.mean(err_vec[perm_dist_vec >= p_med]) - np.mean(err_vec[perm_dist_vec < p_med]))
            
        p_val_r = np.mean(np.array(perm_r) >= r_point)
        p_val_diff = np.mean(np.array(perm_diff) >= diff_mae)

        return {
            'Metric_Definition': label_name,
            'Evaluation_Level': 'Corrected Cluster-Level (n=312 alloys, 201 clusters)',
            'Validity': 'VALID (Independent alloy averages & cluster-level resampling)',
            'Pearson_r': r_point,
            'r_95_CI_low': ci_r[0],
            'r_95_CI_high': ci_r[1],
            'r_p_value': p_val_r,
            'Near_MAE_MPa': near_mae,
            'Far_MAE_MPa': far_mae,
            'Delta_MAE_MPa': diff_mae,
            'Delta_MAE_95_CI_low': ci_diff[0],
            'Delta_MAE_95_CI_high': ci_diff[1],
            'Delta_MAE_p_value': p_val_diff,
            'Significant_05': 'YES' if (ci_diff[0] > 0 and p_val_diff < 0.05) else 'NO'
        }

    # Helper for pooled calculations (to reproduce previous numbers side by side)
    def compute_pooled_stats(p_dist, p_err, label_name):
        p_dist = np.array(p_dist)
        p_err = np.array(p_err)
        r_point = np.corrcoef(p_dist, p_err)[0, 1]
        med_dist = np.median(p_dist)
        near_mae = np.mean(p_err[p_dist < med_dist])
        far_mae = np.mean(p_err[p_dist >= med_dist])
        diff_mae = far_mae - near_mae
        
        rng = np.random.RandomState(RANDOM_SEED)
        boot_r, boot_diff = [], []
        n = len(p_dist)
        for _ in range(2000):
            idx = rng.choice(n, size=n, replace=True)
            bd = p_dist[idx]
            be = p_err[idx]
            boot_r.append(np.corrcoef(bd, be)[0, 1])
            b_med = np.median(bd)
            boot_diff.append(np.mean(be[bd >= b_med]) - np.mean(be[bd < b_med]))
        ci_r = np.percentile(boot_r, [2.5, 97.5])
        ci_diff = np.percentile(boot_diff, [2.5, 97.5])
        
        perm_r, perm_diff = [], []
        for _ in range(5000):
            pd_samp = rng.permutation(p_dist)
            perm_r.append(np.corrcoef(pd_samp, p_err)[0, 1])
            perm_diff.append(np.mean(p_err[pd_samp >= med_dist]) - np.mean(p_err[pd_samp < med_dist]))
        p_val_r = np.mean(np.array(perm_r) >= r_point)
        p_val_diff = np.mean(np.array(perm_diff) >= diff_mae)
        
        return {
            'Metric_Definition': label_name,
            'Evaluation_Level': 'Previous Pooled (n=7800 predictions across 25 splits)',
            'Validity': 'INVALID (Inflated sample size, pools repeats & correlated near-twins)',
            'Pearson_r': r_point,
            'r_95_CI_low': ci_r[0],
            'r_95_CI_high': ci_r[1],
            'r_p_value': p_val_r,
            'Near_MAE_MPa': near_mae,
            'Far_MAE_MPa': far_mae,
            'Delta_MAE_MPa': diff_mae,
            'Delta_MAE_95_CI_low': ci_diff[0],
            'Delta_MAE_95_CI_high': ci_diff[1],
            'Delta_MAE_p_value': p_val_diff,
            'Significant_05': 'YES' if (ci_diff[0] > 0 and p_val_diff < 0.05) else 'NO'
        }

    # Run for all three definitions:
    # 1. Raw wt% with Fe (14 elements)
    c_raw_fe = compute_cluster_level_stats(mean_dists_raw_with_fe, mean_errors, groups, '1. Raw wt% with Fe (14 elements)')
    p_raw_fe = compute_pooled_stats(pooled_dists_raw_with_fe, pooled_errors, '1. Raw wt% with Fe (14 elements)')
    
    # 2. Raw wt% non-Fe (13 elements)
    c_raw_non_fe = compute_cluster_level_stats(mean_dists_raw_non_fe, mean_errors, groups, '2. Raw wt% non-Fe (13 elements)')
    p_raw_non_fe = compute_pooled_stats(pooled_dists_raw_non_fe, pooled_errors, '2. Raw wt% non-Fe (13 elements)')
    
    # 3. Standardized non-Fe (13 elements)
    c_std_non_fe = compute_cluster_level_stats(mean_dists_std_non_fe, mean_errors, groups, '3. Standardized non-Fe (13 elements)')
    p_std_non_fe = compute_pooled_stats(pooled_dists_std_non_fe, pooled_errors, '3. Standardized non-Fe (13 elements)')

    all_ad_rows = [p_raw_fe, c_raw_fe, p_raw_non_fe, c_raw_non_fe, p_std_non_fe, c_std_non_fe]
    df_ad_all = pd.DataFrame(all_ad_rows)
    df_ad_all.to_csv(os.path.join(RES_DIR, 'stage2_ad_cluster_audit.csv'), index=False)
    print(f"\nSaved full side-by-side AD statistical audit to: {os.path.join(RES_DIR, 'stage2_ad_cluster_audit.csv')}")

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE APPLICABILITY DOMAIN COMPARISON (POOLED VS. CORRECTED CLUSTER-LEVEL):")
    print("=" * 80)
    for row in all_ad_rows:
        print(f"\n[{row['Evaluation_Level']}]")
        print(f"Metric: {row['Metric_Definition']} | Validity: {row['Validity']}")
        print(f"  Pearson r: {row['Pearson_r']:.3f} | 95% CI: [{row['r_95_CI_low']:.3f}, {row['r_95_CI_high']:.3f}] | p-value: {row['r_p_value']:.4f}")
        print(f"  Near MAE: {row['Near_MAE_MPa']:.1f} MPa | Far MAE: {row['Far_MAE_MPa']:.1f} MPa | Delta MAE: +{row['Delta_MAE_MPa']:.1f} MPa")
        print(f"  Delta MAE 95% CI: [{row['Delta_MAE_95_CI_low']:.1f}, {row['Delta_MAE_95_CI_high']:.1f}] MPa | p-value: {row['Delta_MAE_p_value']:.4f}")
        print(f"  Statistically significant at alpha=0.05? {row['Significant_05']}")

    return df_ad_all


if __name__ == '__main__':
    df_wt, df_wt_non_fe, df_atom, df_phys, y, groups = load_data()
    df_paired = run_paired_model_comparisons(df_wt, df_wt_non_fe, df_atom, df_phys, y, groups)
    df_ad = run_corrected_applicability_domain_audit(df_wt, y, groups)
    print("\nSTAGE 2 ADDENDUM AUDIT COMPLETED.")
