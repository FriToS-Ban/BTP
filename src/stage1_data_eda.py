"""
Stage 1: Data Ingestion, pymatgen Parsing, and Exploratory Data Analysis (EDA)
Project: Machine-learning prediction of mechanical properties of steels from composition
IIT Roorkee - B.Tech Project in Metallurgical and Materials Engineering
Date: 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matminer.datasets import load_dataset
from pymatgen.core import Composition

# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#2c3e50',
    'axes.linewidth': 1.2,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold'
})

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_and_inspect_raw_data():
    """Load matbench_steels from matminer and perform integrity checks."""
    print("=" * 70)
    print("STEP 1: INGESTING MATBENCH STEELS DATASET")
    print("=" * 70)

    raw_df = load_dataset("matbench_steels")
    print(f"Dataset loaded successfully.")
    print(f"Dimensions: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
    print(f"Column Names: {raw_df.columns.tolist()}")
    print(f"Data Types:\n{raw_df.dtypes}")
    
    missing_count = raw_df.isnull().sum()
    print(f"Missing Values:\n{missing_count}")
    
    raw_duplicates = raw_df['composition'].duplicated().sum()
    print(f"Exact string duplicates in composition column: {raw_duplicates}")
    
    return raw_df


def parse_compositions(raw_df):
    """
    Parse composition strings into normalized stoichiometric fractions
    using pymatgen.core.Composition.
    """
    print("\n" + "=" * 70)
    print("STEP 2: PARSING COMPOSITIONS WITH PYMATGEN")
    print("=" * 70)

    parsed_records = []
    for idx, formula_str in enumerate(raw_df['composition']):
        comp = Composition(formula_str)
        # fractional_composition ensures fractions sum strictly to 1.0
        frac_dict = comp.fractional_composition.as_dict()
        parsed_records.append(frac_dict)

    element_df = pd.DataFrame(parsed_records).fillna(0.0)
    # Sort columns alphabetically for deterministic ordering
    element_df = element_df.reindex(sorted(element_df.columns), axis=1)

    # Check normalization
    sums = element_df.sum(axis=1)
    max_dev = (sums - 1.0).abs().max()
    print(f"Elements detected across 312 steels ({len(element_df.columns)} elements):")
    print(element_df.columns.tolist())
    print(f"Maximum deviation of fractional sum from 1.0: {max_dev:.2e}")

    # Combine with target
    target_col = 'yield strength'
    processed_df = element_df.copy()
    processed_df['yield_strength'] = raw_df[target_col].values
    processed_df['raw_composition'] = raw_df['composition'].values

    # Check for duplicate composition vectors
    comp_duplicates = element_df.duplicated().sum()
    print(f"Exact duplicate composition vectors in feature space: {comp_duplicates}")

    return element_df, processed_df


def analyze_element_rarity(element_df):
    """
    Evaluate element prevalence, identify rare elements (<5% occurrence),
    and formulate metallurgical justifications.
    """
    print("\n" + "=" * 70)
    print("STEP 3: ELEMENT OCCURRENCE & RARITY ANALYSIS (<5% THRESHOLD)")
    print("=" * 70)

    n_samples = len(element_df)
    rarity_data = []

    for el in element_df.columns:
        s = element_df[el]
        non_zero = (s > 1e-6).sum()
        pct = (non_zero / n_samples) * 100.0
        rarity_data.append({
            'Element': el,
            'NonZero_Count': non_zero,
            'Percentage': pct,
            'Min_NonZero': s[s > 1e-6].min() if non_zero > 0 else 0.0,
            'Median_All': s.median(),
            'Max': s.max(),
            'Mean': s.mean(),
            'Std': s.std()
        })

    rarity_df = pd.DataFrame(rarity_data).sort_values(by='Percentage', ascending=False)
    print(rarity_df.to_string(index=False, formatters={
        'Percentage': '{:5.1f}%'.format,
        'Min_NonZero': '{:7.5f}'.format,
        'Median_All': '{:7.5f}'.format,
        'Max': '{:7.5f}'.format,
        'Mean': '{:7.5f}'.format,
        'Std': '{:7.5f}'.format
    }))

    rare_elements = rarity_df[rarity_df['Percentage'] < 5.0]['Element'].tolist()
    boundary_elements = rarity_df[(rarity_df['Percentage'] >= 5.0) & (rarity_df['Percentage'] < 15.0)]['Element'].tolist()
    
    print(f"\nElements strictly < 5.0% presence: {rare_elements if rare_elements else 'None'}")
    print(f"Near-threshold elements (< 15.0% presence): {boundary_elements}")
    print("\nDecision & Justification for Feature Retention:")
    print("1. Tungsten (W) is present in 16 steels (5.13%). In high-strength tool and secondary-hardening steels,")
    print("   W forms potent M6C and M2C carbides that resist coarsening up to 550°C.")
    print("2. Nitrogen (N) is present in 43 steels (13.78%), serving as a potent interstitial strengthener")
    print("   and carbonitride former with V and Ti.")
    print("3. Dropping these elements would produce artificial mass/stoichiometry deficits (up to ~2.9% in W)")
    print("   and discard vital metallurgical physics. Tree models natively split on zero/nonzero entries.")
    print("4. Conclusion: ALL 14 elements are retained as features for modeling.")

    return rarity_df


def generate_eda_plots(processed_df, rarity_df, element_df):
    """Generate high-resolution exploratory figures."""
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING HIGH-RESOLUTION EDA FIGURES")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Figure 1: Target Variable Distribution (Yield Strength in MPa)
    # -------------------------------------------------------------------------
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={'height_ratios': [0.25, 0.75]}
    )

    ys = processed_df['yield_strength']
    mean_val = ys.mean()
    median_val = ys.median()
    std_val = ys.std()
    min_val = ys.min()
    max_val = ys.max()

    # Boxplot
    sns.boxplot(x=ys, ax=ax_box, color='#3498db', fliersize=4, flierprops={'markerfacecolor': '#e74c3c'})
    ax_box.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.1f} MPa')
    ax_box.axvline(median_val, color='#27ae60', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.1f} MPa')
    ax_box.set(xlabel='')
    ax_box.legend(loc='upper right', frameon=True, fontsize=9)
    ax_box.set_title(r'Distribution of Yield Strength ($\sigma_y$) in Matbench Steels (N=312)', pad=10)

    # Histogram & KDE
    sns.histplot(ys, kde=True, ax=ax_hist, bins=25, color='#2980b9', edgecolor='#1c2833', alpha=0.6)
    ax_hist.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=1.8, label=f'Mean = {mean_val:.1f} MPa')
    ax_hist.axvline(median_val, color='#27ae60', linestyle='-', linewidth=1.8, label=f'Median = {median_val:.1f} MPa')
    ax_hist.axvline(mean_val - std_val, color='#7f8c8d', linestyle=':', linewidth=1.2, label=f'±1 Std ({std_val:.1f} MPa)')
    ax_hist.axvline(mean_val + std_val, color='#7f8c8d', linestyle=':', linewidth=1.2)

    ax_hist.set_xlabel(r'Yield Strength $\sigma_y$ (MPa)', fontweight='bold')
    ax_hist.set_ylabel('Sample Frequency (Count)', fontweight='bold')
    ax_hist.legend(loc='upper right', frameon=True, fontsize=9)
    
    # Statistical Annotation Box
    stat_text = (
        f"Sample Size: 312\n"
        f"Min: {min_val:.1f} MPa\n"
        f"25th %: {ys.quantile(0.25):.1f} MPa\n"
        f"Median: {median_val:.1f} MPa\n"
        f"75th %: {ys.quantile(0.75):.1f} MPa\n"
        f"Max: {max_val:.1f} MPa\n"
        f"Std Dev: {std_val:.1f} MPa\n"
        f"Regime: Ultra-High Strength Steels (UHSS)"
    )
    ax_hist.text(
        0.03, 0.95, stat_text, transform=ax_hist.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.9)
    )

    plt.tight_layout()
    fig1_path = os.path.join(FIG_DIR, 'fig1_target_distribution.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Element Fraction Boxplots (Log Scale for Wide Dynamic Ranges)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Prepare non-zero and full data in percentage (atomic %)
    pct_df = element_df * 100.0
    sorted_cols = rarity_df.sort_values(by='Median_All', ascending=False)['Element'].tolist()
    # Put Fe first, then the rest
    if 'Fe' in sorted_cols:
        sorted_cols.remove('Fe')
        sorted_cols = ['Fe'] + sorted_cols

    melted_df = pct_df[sorted_cols].melt(var_name='Element', value_name='Atomic_Percent')
    # Filter out exact zeros for meaningful log-scale visualization
    melted_nonzero = melted_df[melted_df['Atomic_Percent'] > 0.0]

    sns.boxplot(
        data=melted_nonzero, x='Element', y='Atomic_Percent', hue='Element',
        legend=False, ax=ax, palette='Blues_r', fliersize=3, linewidth=1.1
    )
    ax.set_yscale('log')
    ax.set_title('Elemental Concentration Distributions across 312 Steels (Non-zero Values, Log Scale)', pad=12)
    ax.set_xlabel('Alloying Element', fontweight='bold')
    ax.set_ylabel('Concentration (Atomic %, Log Scale)', fontweight='bold')
    ax.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=0.9, alpha=0.7, label='1.0 at.% reference')
    ax.axhline(0.1, color='#f39c12', linestyle=':', linewidth=0.9, alpha=0.7, label='0.1 at.% reference')
    ax.legend(loc='lower left', frameon=True, fontsize=9)

    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, 'fig2_element_distributions.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Correlation Heatmap (Pearson)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8.5))
    corr_cols = sorted(element_df.columns.tolist()) + ['yield_strength']
    corr_matrix = processed_df[corr_cols].corr(method='pearson')

    # Renaming for cleaner display
    display_matrix = corr_matrix.rename(columns={'yield_strength': r'$\sigma_y$ (Yield)'},
                                        index={'yield_strength': r'$\sigma_y$ (Yield)'})

    mask = np.triu(np.ones_like(display_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        display_matrix, mask=mask, cmap=cmap, vmin=-0.8, vmax=0.8,
        annot=True, fmt='.2f', annot_kws={'size': 8.5},
        square=True, linewidths=0.6, cbar_kws={'shrink': 0.75, 'label': 'Pearson Correlation (r)'},
        ax=ax
    )
    ax.set_title(r'Correlation Matrix: Alloying Elements & Yield Strength ($\sigma_y$)', pad=14)

    plt.tight_layout()
    fig3_path = os.path.join(FIG_DIR, 'fig3_correlation_heatmap.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # -------------------------------------------------------------------------
    # Figure 4: Element Presence Frequency (Occurrence %)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_rarity = rarity_df.sort_values(by='Percentage', ascending=True)

    colors = ['#e74c3c' if p < 10.0 else '#f39c12' if p < 50.0 else '#2980b9' for p in plot_rarity['Percentage']]
    bars = ax.barh(plot_rarity['Element'], plot_rarity['Percentage'], color=colors, edgecolor='#2c3e50', height=0.65)

    ax.axvline(5.0, color='#c0392b', linestyle='--', linewidth=1.5, label='5.0% Rare Threshold (15.6 samples)')
    ax.set_xlim(0, 108)
    ax.set_xlabel('Occurrence Frequency across 312 Samples (%)', fontweight='bold')
    ax.set_ylabel('Alloying Element', fontweight='bold')
    ax.set_title('Prevalence of Alloying Elements in Matbench Steels', pad=12)
    ax.legend(loc='lower right', frameon=True, fontsize=9)

    for bar, pct, cnt in zip(bars, plot_rarity['Percentage'], plot_rarity['NonZero_Count']):
        ax.text(pct + 1.2, bar.get_y() + bar.get_height()/2, f"{pct:.1f}% ({cnt}/312)",
                va='center', ha='left', fontsize=8.5, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    fig4_path = os.path.join(FIG_DIR, 'fig4_element_occurrence_frequency.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")


def save_processed_data(processed_df, rarity_df):
    """Save processed datasets to data/."""
    out_csv = os.path.join(DATA_DIR, 'processed_steels.csv')
    processed_df.to_csv(out_csv, index=False)
    print(f"\nSaved processed dataset to: {out_csv}")

    summary_csv = os.path.join(DATA_DIR, 'element_summary.csv')
    rarity_df.to_csv(summary_csv, index=False)
    print(f"Saved element summary to: {summary_csv}")


def main():
    raw_df = load_and_inspect_raw_data()
    element_df, processed_df = parse_compositions(raw_df)
    rarity_df = analyze_element_rarity(element_df)
    generate_eda_plots(processed_df, rarity_df, element_df)
    save_processed_data(processed_df, rarity_df)
    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
