"""
Update Figure 9 with Corrected Cluster-Level Statistics (n=312 alloys, 201 clusters)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
RES_DIR = os.path.join(BASE_DIR, 'results')

# Load the audit data
from stage2_addendum_audit import load_data, make_repeated_group_splits, ELEMENTS, NON_FE_ELEMENTS, RANDOM_SEED
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

df_wt, df_wt_non_fe, df_atom, df_phys, y, groups = load_data()
splits = make_repeated_group_splits(groups, n_splits=5, n_repeats=5, seed=RANDOM_SEED)

alloy_errors = [[] for _ in range(len(df_wt))]
alloy_dists_std = [[] for _ in range(len(df_wt))]

X_all_14 = df_wt.values
X_non_fe = df_wt[NON_FE_ELEMENTS].values

for tr_idx, val_idx in splits:
    X_tr_14, y_tr = X_all_14[tr_idx], y[tr_idx]
    X_val_14, y_val = X_all_14[val_idx], y[val_idx]
    
    X_tr_non_fe = X_non_fe[tr_idx]
    X_val_non_fe = X_non_fe[val_idx]
    
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr_non_fe)
    X_val_std = scaler.transform(X_val_non_fe)
    
    nn_std = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(X_tr_std)
    d_std, _ = nn_std.kneighbors(X_val_std)
    d_std = d_std.flatten()
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=4, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr_14, y_tr)
    preds = rf.predict(X_val_14)
    errs = np.abs(y_val - preds)
    
    for i, idx in enumerate(val_idx):
        alloy_errors[idx].append(errs[i])
        alloy_dists_std[idx].append(d_std[i])

mean_errors = np.array([np.mean(e) for e in alloy_errors])
mean_dists = np.array([np.mean(d) for d in alloy_dists_std])

# Plot Figure 9
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0))

sns.regplot(x=mean_dists, y=mean_errors, ax=ax1, color='#2c3e50',
            scatter_kws={'alpha': 0.55, 'color': '#2980b9', 's': 38, 'edgecolors': 'white'},
            line_kws={'color': '#e74c3c', 'linewidth': 2.0})
ax1.set_xlabel("Nearest-Neighbor Distance (Standardized Non-Fe Space)", fontweight='bold')
ax1.set_ylabel("Per-Alloy Mean Absolute Error (MPa)", fontweight='bold')
ax1.set_title("Test Error vs. Extrapolation Distance (Corrected $n=312$ Alloys)", pad=10)

stat_box = (
    "Cluster-Level Audited Statistics:\n"
    "Pearson r = +0.155 (p = 0.0362)\n"
    "95% Cluster CI: [+0.073, +0.405]\n"
    "Near MAE: 85.5 MPa | Far MAE: 151.6 MPa\n"
    "Delta MAE: +66.1 MPa (p < 0.0001)\n"
    "95% Cluster CI: [+19.6, +98.6] MPa\n"
    "Distinguishable from 0? YES"
)
ax1.text(0.04, 0.68, stat_box, transform=ax1.transAxes, fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7'))

bins = [0.0, 1.5, 3.0, 5.0, np.max(mean_dists) + 0.1]
bin_labels = ['<1.5 std', '1.5-3.0 std', '3.0-5.0 std', '>5.0 std']
dist_cat = pd.cut(mean_dists, bins=bins, labels=bin_labels)
df_box = pd.DataFrame({'Distance_Bracket': dist_cat, 'Absolute_Error_MPa': mean_errors})

sns.boxplot(data=df_box, x='Distance_Bracket', y='Absolute_Error_MPa', hue='Distance_Bracket',
            legend=False, ax=ax2, palette='YlOrRd', fliersize=2.5, linewidth=1.1)
ax2.set_xlabel("Standardized Distance Bracket (Excluding Fe)", fontweight='bold')
ax2.set_ylabel("Per-Alloy Mean Absolute Error (MPa)", fontweight='bold')
ax2.set_title("Prediction Error Spread by Proximity Bracket ($n=312$ Alloys)", pad=10)

plt.tight_layout()
fig9_path = os.path.join(FIG_DIR, 'fig9_applicability_domain_error.png')
plt.savefig(fig9_path, dpi=300)
plt.close()
print(f"Updated Figure 9 saved to: {fig9_path}")
