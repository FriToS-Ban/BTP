"""Grade catalogue: composition ranges, nominal compositions and categories.

Sources merged (priority order for duplicated grade names):
  1. Summit Steel tables (SAE J403 carbon, SAE J1268 alloy H-steels)
  2. Wikipedia SAE stainless and tool-steel tables
  3. Zenodo 17704410 (EN, GB, ASTM, UNS carbon and stainless grades)
  4. Two alloy grades of the ASM property table that none of the range
     tables lists (3140, 9255): ranges centred on the Wikipedia SAE family
     nominal values (31xx: Ni 1.25, Cr 0.65; 92xx: Si 2.00, Mn 0.82) with the
     carbon implied by the AISI number.

Nominal composition rules
  * range given          -> midpoint
  * maximum only         -> 0.6 x maximum for alloying elements and C,
                            typical residual (capped at the maximum) for P, S
  * element not specified -> typical residual (config.RESIDUAL_TYPICAL)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import categories as cat
from .. import config

EL = config.ELEMENTS
MAX_ONLY_FRACTION = 0.6
IMPURITIES = ("P", "S")
# incidental (unspecified) residual maxima, SAE J1268 note 3 / common practice
INCIDENTAL_MAX = {"Cu": 0.35, "Ni": 0.25, "Cr": 0.20, "Mo": 0.06, "V": 0.03, "Al": 0.06,
                  "Ti": 0.01, "Nb": 0.01, "N": 0.012, "W": 0.0, "Co": 0.0, "B": 0.0,
                  "P": 0.04, "S": 0.05, "Si": 0.35}


def _std_cols(df):
    for el in EL:
        for suf in ("_min", "_max"):
            if f"{el}{suf}" not in df.columns:
                df[f"{el}{suf}"] = np.nan
    return df


def nominal_composition(row):
    nom = {}
    for el in EL:
        lo, hi = row.get(f"{el}_min", np.nan), row.get(f"{el}_max", np.nan)
        lo = 0.0 if (pd.isna(lo) or lo <= 1e-3 and not pd.isna(hi)) else lo
        if pd.isna(hi):
            nom[el] = config.RESIDUAL_TYPICAL.get(el, 0.0) if pd.isna(row.get(f"{el}_min")) else lo
        elif lo > 0 and hi >= lo:
            nom[el] = 0.5 * (lo + hi)
        elif el in IMPURITIES:
            nom[el] = min(config.RESIDUAL_TYPICAL[el], hi)
        else:
            nom[el] = MAX_ONLY_FRACTION * hi
    return nom


def _from_summit_carbon(raw):
    df = pd.read_csv(raw / "summit_carbon_ranges.csv")
    df = _std_cols(df)
    # SAE J403 bars: Si not tabulated by Summit for non-H grades; killed practice 0.10-0.35
    nosi = df.Si_min.isna() & df.Si_max.isna()
    df.loc[nosi, ["Si_min", "Si_max"]] = [0.10, 0.35]
    df["source"] = "summit_carbon"
    df["standard"] = "AISI/SAE"
    return df


def _from_summit_alloy(raw):
    df = _std_cols(pd.read_csv(raw / "summit_alloy_h_ranges.csv"))
    df["source"] = "summit_alloy_h"
    df["standard"] = "AISI/SAE"
    return df


def _manual_alloy():
    rows = [
        {"grade": "AISI 3140", "C_min": 0.38, "C_max": 0.43, "Mn_min": 0.70, "Mn_max": 0.90,
         "Si_min": 0.15, "Si_max": 0.35, "Ni_min": 1.10, "Ni_max": 1.40, "Cr_min": 0.55, "Cr_max": 0.75},
        {"grade": "AISI 9255", "C_min": 0.51, "C_max": 0.59, "Mn_min": 0.70, "Mn_max": 0.95,
         "Si_min": 1.80, "Si_max": 2.20},
    ]
    df = _std_cols(pd.DataFrame(rows))
    df["source"] = "sae_family_nominal"
    df["standard"] = "AISI/SAE"
    df["series"] = "alloy"
    return df


def _from_wiki_stainless(raw):
    df = _std_cols(pd.read_csv(raw / "wiki_stainless_ranges.csv"))
    df["source"] = "wikipedia_stainless"
    df["standard"] = "AISI/SAE"
    df["series"] = "stainless"
    return df


def _from_wiki_tool(raw):
    df = _std_cols(pd.read_csv(raw / "wiki_tool_ranges.csv"))
    df["source"] = "wikipedia_tool"
    df["standard"] = "AISI/SAE"
    df["series"] = "tool"
    return df


def _from_zenodo(raw):
    z = pd.read_excel(raw / "zenodo17704410_Carbon_stainless_grades.xlsx")
    keep = z.Standard.isin(["EN", "GB", "ASTM", "UNS"])
    z = z[keep].copy()
    recs = []
    for g, grp in z.groupby("Grade"):
        r = grp.iloc[0]
        rec = {"grade": str(g).strip(), "standard": r["Standard"], "series": r["Type"],
               "zenodo_type": r["Type"], "source": "zenodo17704410"}
        for el in ["Mn", "C", "Si", "P", "S", "Cr", "Ni", "Mo", "Cu", "Ti", "Al", "V", "N", "Nb"]:
            lo, hi = r.get(f"{el}_Min", np.nan), r.get(f"{el}_Max", np.nan)
            if pd.isna(hi) or hi <= 0:
                lo, hi = np.nan, np.nan
            elif lo <= 0.0001:
                lo = 0.0
            rec[f"{el}_min"], rec[f"{el}_max"] = lo, hi
        recs.append(rec)
    df = _std_cols(pd.DataFrame(recs))
    # drop grades without a usable carbon specification
    df = df[df.C_max.notna() & (df.C_max > 0)]
    return df


def _assign_category(row, nom):
    src = row["source"]
    if src.startswith("summit") or src == "sae_family_nominal":
        return cat.aisi_series_category(row["grade"], nom["C"])
    if src == "wikipedia_stainless":
        return cat.stainless_category(row.get("stainless_group", ""), nom["Ni"], nom["C"])
    if src == "wikipedia_tool":
        return "tool"
    if src == "zenodo17704410":
        t = str(row.get("zenodo_type", "")).lower()
        if t == "austenitic":
            return "stainless_austenitic"
        if t == "ferritic":
            return "stainless_ferritic"
        if t == "martensitic":
            return "stainless_martensitic"
        # carbon grades from EN/GB/ASTM
        if nom["Cr"] >= 0.5 or nom["Mo"] >= 0.15 or nom["Ni"] >= 0.5:
            return None  # low-alloy grades without an AISI series are left unlabelled
        if nom["Mn"] >= 1.0:
            return "carbon_manganese"
        return str(cat.carbon_class(nom["C"]))
    return None


def _canonical(name):
    return re.sub(r"\s+", " ", str(name).upper().replace("SAE", "AISI")).strip()


def build_catalog(raw_dir=config.RAW_DIR):
    parts = [_from_summit_carbon(raw_dir), _from_summit_alloy(raw_dir), _manual_alloy(),
             _from_wiki_stainless(raw_dir), _from_wiki_tool(raw_dir), _from_zenodo(raw_dir)]
    df = pd.concat(parts, ignore_index=True)
    df["grade_key"] = df.grade.map(_canonical)
    df = df.drop_duplicates("grade_key", keep="first").reset_index(drop=True)
    noms, cats = [], []
    for _, row in df.iterrows():
        nom = nominal_composition(row)
        noms.append(nom)
        cats.append(_assign_category(row, nom))
    nom_df = pd.DataFrame(noms).add_suffix("_nom")
    out = pd.concat([df, nom_df], axis=1)
    out["category"] = cats
    out = out[out.category.notna()].reset_index(drop=True)
    out["family"] = out.category.map(cat.FAMILY_OF)
    out["in_fp_scope"] = out.category.isin(cat.FERRITE_PEARLITE_SCOPE)
    cols = (["grade", "standard", "source", "series", "category", "family", "in_fp_scope"]
            + [f"{el}_{s}" for el in EL for s in ("min", "max")] + [f"{el}_nom" for el in EL])
    return out[cols]


# ---------------------------------------------------------------------------
# Heat sampling within specification ranges
# ---------------------------------------------------------------------------

def sample_heats(catalog, n_per_grade, rng, spread=0.25):
    """Simulated heat analyses: truncated normal inside each specified range.

    For a range [lo, hi] the aim is the midpoint and sigma = spread*(hi-lo);
    for a maximum-only limit the value is uniform in [0.2, 1.0] x max (for P
    and S: [0.2, 1.0] x the typical residual); for an unspecified element a
    residual is drawn uniformly in [0, 2 x typical], capped at the incidental
    maximum. Returns a DataFrame with ELEMENTS columns plus grade/category.
    """
    rows = []
    for _, g in catalog.iterrows():
        n = n_per_grade
        sample = {}
        for el in EL:
            lo, hi = g[f"{el}_min"], g[f"{el}_max"]
            if pd.notna(hi) and pd.notna(lo) and lo > 1e-3 and hi > lo:
                mu, sd = 0.5 * (lo + hi), spread * (hi - lo)
                x = rng.normal(mu, sd, n)
                sample[el] = np.clip(x, lo, hi)
            elif pd.notna(hi) and pd.notna(lo) and lo > 1e-3 and hi == lo:
                sample[el] = rng.normal(lo, 0.03 * lo + 1e-4, n).clip(min=0)
            elif pd.notna(hi):
                top = min(hi, config.RESIDUAL_TYPICAL[el]) if el in IMPURITIES else hi
                sample[el] = rng.uniform(0.2 * top, top, n)
            else:
                typ = config.RESIDUAL_TYPICAL.get(el, 0.0)
                cap = INCIDENTAL_MAX.get(el, typ * 2)
                sample[el] = np.minimum(rng.uniform(0, 2 * typ, n), cap) if typ > 0 else np.zeros(n)
        block = pd.DataFrame(sample)
        block["grade"] = g["grade"]
        block["category"] = g["category"]
        block["family"] = g["family"]
        rows.append(block)
    return pd.concat(rows, ignore_index=True)
