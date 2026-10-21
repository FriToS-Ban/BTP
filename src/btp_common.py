"""Shared data, splits, and screening definitions for Stages 4–7.

The grouped splits, the element order, and the family rules match
src/stage3_complete_audited.py. Later stages import this module so those
choices are not retyped.
"""

import os

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.stats import t as student_t
from sklearn.model_selection import GroupKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RES_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")

RANDOM_SEED = 42
TAU_WT = 0.50
TARGET_YIELD_MPA = 1500.0
UNCERTAINTY_K = 1.0
RHO_MIN = 0.20
PRESENCE_FLOOR_WT = 0.01

ELEMENTS = ["Al", "C", "Co", "Cr", "Fe", "Mn", "Mo", "N", "Nb", "Ni", "Si", "Ti", "V", "W"]
NON_FE_ELEMENTS = [el for el in ELEMENTS if el != "Fe"]

# Supply risk from the European Commission 2023 critical-raw-materials assessment.
# Weights are the published supply-risk numbers, not prices. Iron, carbon, and
# nitrogen are not entries in that assessment and are left at zero so the
# balance and the interstitial pair are not inventively penalized.
# Aluminium/bauxite extraction SR = 1.2. Silicon is the silicon-metal row
# (SR = 1.3), which is the nearest published row to alloying silicon.
# Titanium is the titanium-metal row (SR = 1.6), because the alloy addition is
# metal rather than the mineral row (SR = 0.5).
EU2023_SUPPLY_RISK = {
    "Al": 1.2,
    "C": 0.0,
    "Co": 2.8,
    "Cr": 0.7,
    "Fe": 0.0,
    "Mn": 1.2,
    "Mo": 0.8,
    "N": 0.0,
    "Nb": 4.4,
    "Ni": 0.5,
    "Si": 1.3,
    "Ti": 1.6,
    "V": 2.3,
    "W": 1.2,
}

# Published CET validity window, EN 1011-2 method B as restated by Dillinger.
# Copper and boron are absent from this dataset and are treated as zero.
CET_BOUNDS = {
    "C": (0.05, 0.32),
    "Si": (None, 0.80),
    "Mn": (0.50, 1.90),
    "Cr": (None, 1.50),
    "Ni": (None, 2.50),
    "Mo": (None, 0.75),
    "V": (None, 0.18),
    "Nb": (None, 0.06),
    "Ti": (None, 0.12),
}

FAMILY_COLORS = {
    "Stainless": "#2980b9",
    "Maraging": "#c0392b",
    "Structural": "#1e8449",
    "Tool": "#8e44ad",
}


def load_data():
    """Rebuild wt% from the stored formulas. Same clustering as Stage 3."""
    df_raw = pd.read_csv(os.path.join(DATA_DIR, "processed_steels.csv"))
    y = df_raw["yield_strength"].to_numpy(dtype=float)
    formulas = df_raw["raw_composition"].to_numpy()
    records = []
    for formula in formulas:
        comp = Composition(formula)
        records.append({el: comp.get_wt_fraction(el) * 100.0 for el in ELEMENTS})
    df_wt = pd.DataFrame(records)[ELEMENTS]
    dist = squareform(pdist(df_wt.to_numpy(), metric="euclidean"))
    _, groups = connected_components(dist <= TAU_WT)
    return df_wt, y, groups, formulas


def make_repeated_group_splits(groups, n_splits=5, n_repeats=5, seed=RANDOM_SEED):
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    splits = []
    for _rep in range(n_repeats):
        shuffled = rng.permutation(unique_groups)
        grp_map = {g: i for i, g in enumerate(shuffled)}
        mapped = np.array([grp_map[g] for g in groups])
        gkf = GroupKFold(n_splits=n_splits)
        for tr_idx, val_idx in gkf.split(np.zeros(len(groups)), groups=mapped):
            splits.append((tr_idx, val_idx))
    return splits


def assign_family(df_wt):
    """Same mutually exclusive rules as the audited Stage 3 script."""
    ni = df_wt["Ni"].to_numpy()
    co = df_wt["Co"].to_numpy()
    cr = df_wt["Cr"].to_numpy()
    carbon = df_wt["C"].to_numpy()
    maraging = (ni >= 12.0) & (co >= 5.0)
    stainless = (cr >= 10.0) & (~maraging)
    tool = (carbon >= 0.35) & (~maraging) & (~stainless)
    labels = np.full(len(df_wt), "Structural", dtype=object)
    labels[stainless] = "Stainless"
    labels[maraging] = "Maraging"
    labels[tool] = "Tool"
    return labels


def nadeau_bengio(deltas, n_train, n_test):
    """Corrected paired test. Returns mean, low CI, high CI, two-sided p."""
    deltas = np.asarray(deltas, dtype=float)
    j = len(deltas)
    mean = float(np.mean(deltas))
    var_d = float(np.var(deltas, ddof=1))
    inflated = (1.0 / j + n_test / n_train) * var_d
    se = np.sqrt(inflated)
    df = j - 1
    t_stat = mean / se if se > 0 else np.inf
    p_value = float(2 * student_t.sf(np.abs(t_stat), df))
    critical = float(student_t.ppf(0.975, df))
    half = critical * se
    return mean, mean - half, mean + half, p_value


def criticality_index(df_wt):
    """Sum of wt% times EU 2023 supply risk. A screening index, not a cost in currency."""
    total = np.zeros(len(df_wt), dtype=float)
    for el, weight in EU2023_SUPPLY_RISK.items():
        total += df_wt[el].to_numpy(dtype=float) * weight
    return total


def carbon_equivalents(df_wt):
    """IIW carbon equivalent and Pcm. Indicators only. See cet_validity_mask."""
    c = df_wt["C"].to_numpy(dtype=float)
    mn = df_wt["Mn"].to_numpy(dtype=float)
    cr = df_wt["Cr"].to_numpy(dtype=float)
    mo = df_wt["Mo"].to_numpy(dtype=float)
    v = df_wt["V"].to_numpy(dtype=float)
    ni = df_wt["Ni"].to_numpy(dtype=float)
    si = df_wt["Si"].to_numpy(dtype=float)
    ce_iiw = c + mn / 6.0 + (cr + mo + v) / 5.0 + ni / 15.0
    pcm = c + si / 30.0 + mn / 20.0 + ni / 60.0 + cr / 20.0 + mo / 15.0 + v / 10.0
    return ce_iiw, pcm


def schaeffler_coordinates(df_wt):
    cr_eq = (
        df_wt["Cr"].to_numpy(dtype=float)
        + df_wt["Mo"].to_numpy(dtype=float)
        + 1.5 * df_wt["Si"].to_numpy(dtype=float)
        + 0.5 * df_wt["Nb"].to_numpy(dtype=float)
        + 2.0 * df_wt["Ti"].to_numpy(dtype=float)
    )
    ni_eq = (
        df_wt["Ni"].to_numpy(dtype=float)
        + 30.0 * df_wt["C"].to_numpy(dtype=float)
        + 30.0 * df_wt["N"].to_numpy(dtype=float)
        + 0.5 * df_wt["Mn"].to_numpy(dtype=float)
    )
    return cr_eq, ni_eq


def cet_violation_frame(df_wt):
    """Boolean columns for each published CET bound. True means the bound is broken."""
    flags = {}
    for el, (low, high) in CET_BOUNDS.items():
        values = df_wt[el].to_numpy(dtype=float)
        bad = np.zeros(len(df_wt), dtype=bool)
        if low is not None:
            bad |= values < low
        if high is not None:
            bad |= values > high
        flags[f"cet_outside_{el}"] = bad
    frame = pd.DataFrame(flags)
    frame["cet_valid"] = ~frame.any(axis=1)
    return frame


def composition_row_to_frame(rows):
    return pd.DataFrame(rows)[ELEMENTS]
