"""Assemble the analysis-ready datasets from the parsed sources.

Outputs (data/processed/):
  grade_catalog.csv           composition ranges, nominal compositions, categories
  properties_handbook.csv     ASM/Bethlehem typical properties (annealed = furnace
                              cooled, normalized = air cooled, hot rolled = as-rolled)
                              joined to nominal compositions and cooling routes
  properties_secondary.csv    the same grades/conditions from a second compilation
                              (Zenodo 17704410, MatWeb-derived) for an inter-source check
  structural_external.csv     measured heat chemistry + tensile data of hot-rolled
                              structural members (Kingsley et al. 2026)
  heats_steelbench.csv        real heat analyses with family labels (classifier test)
  qt_bars.csv                 quenched-and-tempered bars (Birbilis et al. 2025): quench
                              medium, austenitising and tempering temperature, bar size
  normalized_bars.csv         normalised bars of 13-100 mm and annealed bars, same source
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import config
from . import catalog as catmod

EL = config.ELEMENTS

ROUTE_OF_CONDITION = {"annealed": "FC", "normalized": "AC", "hot_rolled": "AR"}


def _catalog_key(grade):
    """ASM grade name -> catalogue grade name (alloy grades use the H-band)."""
    num = grade.replace("AISI", "").strip()
    if re.fullmatch(r"10\d\d|11\d\d|12\d\d|15\d\d", num):
        return f"AISI {num}"
    if num in ("3140", "9255"):
        return f"AISI {num}"
    return f"AISI {num}H"


def property_table(catalog):
    asm = pd.read_csv(config.RAW_DIR / "asm_bethlehem_properties.csv")
    asm["catalog_grade"] = asm.grade.map(_catalog_key)
    cat_idx = catalog.set_index("grade")
    missing = sorted(set(asm.catalog_grade) - set(cat_idx.index))
    if missing:
        raise KeyError(f"grades missing from catalogue: {missing}")
    comp = cat_idx.loc[asm.catalog_grade, [f"{el}_nom" for el in EL] + [f"{el}_min" for el in EL]
                       + [f"{el}_max" for el in EL] + ["category", "family"]].reset_index(drop=True)
    comp.columns = [c.replace("_nom", "") for c in comp.columns]
    df = pd.concat([asm.reset_index(drop=True), comp], axis=1)
    df["route"] = df.condition.map(ROUTE_OF_CONDITION)
    # All grades are treated as fine-grained practice. (Resulfurised 11xx steels
    # are sometimes made coarse-grained, but their MnS inclusions also nucleate
    # ferrite and lower hardenability; with a coarse-grain assumption the
    # kinetic model wrongly predicts bainite for normalised 1137-1144, whose
    # measured hardness is that of ferrite-pearlite.)
    df["coarse_grained"] = False
    # calibrated effective cooling conditions of the handbook data (see config)
    df["section_mm"] = np.where(df.condition == "normalized", config.HANDBOOK_AC_SECTION_MM,
                                config.AC_DEFAULT_DIAMETER_MM)
    df["fc_rate"] = config.HANDBOOK_FC_RATE
    df["YS_UTS"] = df.YS_MPa / df.UTS_MPa
    df["sample_id"] = df.grade + "|" + df.condition
    return df


def secondary_property_table(catalog, primary):
    z = pd.read_excel(config.RAW_DIR / "zenodo17704410_Carbon_stainless_grades.xlsx")
    z = z[(z.Standard == "AISI") & (z.Type == "Carbon")]
    cond_map = {"Annealed": "annealed", "Normalised": "normalized", "Hot Rolled": "hot_rolled"}
    z = z[z.Treatment_Condition.isin(cond_map)].copy()
    z["condition"] = z.Treatment_Condition.map(cond_map)
    z = z.rename(columns={"Grade": "grade", "UTS": "UTS_MPa", "YS": "YS_MPa",
                          "Elongation": "El_pct", "Hardness_Brinell": "HB"})
    z = z[["grade", "condition", "UTS_MPa", "YS_MPa", "El_pct", "HB", "Name"]]
    z["catalog_grade"] = z.grade
    z = z[z.catalog_grade.isin(set(catalog.grade))]
    cat_idx = catalog.set_index("grade")
    comp = cat_idx.loc[z.catalog_grade, [f"{el}_nom" for el in EL] + ["category"]].reset_index(drop=True)
    comp.columns = [c.replace("_nom", "") for c in comp.columns]
    z = pd.concat([z.reset_index(drop=True), comp], axis=1)
    z["route"] = z.condition.map(ROUTE_OF_CONDITION)
    # flag rows that repeat the ASM value (same handbook origin) vs independent values
    prim = primary.set_index(["grade", "condition"])
    same = []
    for _, r in z.iterrows():
        key = (r.grade, r.condition)
        if key in prim.index:
            p = prim.loc[key]
            same.append(bool(abs(p.UTS_MPa - r.UTS_MPa) / p.UTS_MPa < 0.02
                             and abs(p.YS_MPa - r.YS_MPa) / p.YS_MPa < 0.02))
        else:
            same.append(False)
    z["repeats_asm"] = same
    z["in_asm"] = [(g, c) in prim.index for g, c in zip(z.grade, z.condition)]
    z["source"] = "zenodo17704410_matweb"
    return z


def structural_external():
    """Hot-rolled structural members (as-rolled, still-air cooled from rolling).

    Cold-formed hollow sections (EN 10219 or SHS/RHS/CHS from strip) are
    excluded because cold work raises the yield strength; values outside
    physical ranges (P > 0.1, Mn < 0.2, C missing) are dropped as entry errors.
    """
    df = pd.read_csv(config.RAW_DIR / "cleveland_members_parsed.csv")
    spec = df.spec.str.upper()
    hollow = df.member_type.str.upper().isin(["SHS", "RHS", "CHS"])
    cold_formed = spec.str.contains("10219") | (hollow & ~spec.str.contains("10210"))
    df["cold_formed"] = cold_formed
    ok = (df.C.notna() & (df.P < 0.1) & (df.Mn > 0.2) & df.YS_MPa.notna() & df.UTS_MPa.notna())
    df["usable"] = ok & ~cold_formed
    for el in EL:
        if el not in df.columns:
            df[el] = 0.0
        df[el] = df[el].fillna(config.RESIDUAL_TYPICAL.get(el, 0.0) if el not in ("C",) else np.nan)
    df["route"] = "AR"
    # an I/H-section or plate cools roughly like a bar of diameter ~2x thickness
    df["section_mm"] = 2.0 * df.thickness_mm.clip(lower=3.0)
    df["microalloyed"] = (df.Nb.fillna(0) >= 0.01) | (df.V.fillna(0) >= 0.02) | (df.Ti.fillna(0) >= 0.01)
    return df


def steelbench_heats():
    sb = pd.read_csv(config.RAW_DIR / "steelbench_core_open.csv")
    fam_map = {
        "carbon_steel": "plain_carbon", "carbon_low": "plain_carbon", "carbon_medium": "plain_carbon",
        "carbon_high": "plain_carbon", "low_alloy_normalized": "low_alloy", "low_alloy_other": "low_alloy",
        "CrMo_QT": "low_alloy", "low_alloy_QT": "low_alloy", "NiCrMo_QT": "low_alloy",
        "stainless_austenitic": "stainless", "stainless_ferritic": "stainless",
        "stainless_duplex": "stainless", "stainless_martensitic": "stainless",
        "stainless_PH": "stainless", "stainless_steel": "stainless",
    }
    sb["family_coarse"] = sb.steel_family.map(fam_map)
    sb = sb[sb.family_coarse.notna() & sb.C.notna() & sb.Mn.notna()].copy()
    for el in EL:
        if el not in sb.columns:
            sb[el] = np.nan
    return sb[["heat_id", "grade_id", "source", "data_tier", "steel_family", "family_coarse"] + EL]


def _catalog_key_from_base(base):
    if re.fullmatch(r"10\d\d|11\d\d|12\d\d|15\d\d|3140|9255", str(base)):
        return f"AISI {base}"
    return f"AISI {base}H"


def _birbilis_compositions(rows, catalog):
    """Nominal compositions for Birbilis rows: the grade catalogue's nominal
    composition (with its residual levels, as for the handbook data) when the
    grade is in it; otherwise the database's own composition with unlisted
    residual elements set to the typical levels of config.RESIDUAL_TYPICAL."""
    cat = catalog.set_index("grade")
    out = rows.copy()
    out["catalog_grade"] = out.base_grade.map(_catalog_key_from_base)
    in_cat = out.catalog_grade.isin(cat.index)
    for el in EL:
        own = out[el].astype(float) if el in out.columns else pd.Series(0.0, index=out.index)
        if el in config.RESIDUAL_TYPICAL:
            own = own.where(own > 0, config.RESIDUAL_TYPICAL[el])
        nom = out.catalog_grade.map(cat[f"{el}_nom"]) if f"{el}_nom" in cat.columns else own
        out[el] = np.where(in_cat & nom.notna(), nom, own)
    out["composition_source"] = np.where(in_cat, "grade catalogue", "database (+typical residuals)")
    return out


def heat_treatment_tables(catalog):
    """Quenched-and-tempered, normalised and annealed bars of Birbilis et al.
    (2025), deduplicated (the database repeats the H and non-H versions of a
    grade with identical values)."""
    from . import birbilis
    p = birbilis.parse(birbilis.load_raw())
    u = p[p.in_scope & p.dup_of.isna()].copy()
    u = u[u.UTS_MPa.notna() & u.YS_MPa.notna() & u.El_pct.notna() & (u.YS_MPa <= u.UTS_MPa * 1.001)]
    u["grade"] = u.base_grade
    u["sample_id"] = "BB" + u.entry.astype(int).astype(str)
    qt = u[(u.route == "QT") & u.quench_medium.isin(["oil", "water"]) & u.temper_T_C.notna()
           & u.section_mm.notna() & ~u.processing.str.contains("Reheated", case=False)
           & ~u.double_quench.astype(bool)].copy()
    qt["route"] = "QT"
    qt["treat_T_C_given"] = qt.austenitize_T_C.notna()
    qt["treat_T_C"] = qt.austenitize_T_C
    qt["temper_time_h"] = qt.temper_time_h
    qt = _birbilis_compositions(qt, catalog)
    no = u[u.route.isin(["NO", "AN"])].copy()
    no["route"] = np.where(no.route == "NO", "AC", "FC")
    no["treat_T_C_given"] = no.austenitize_T_C.notna()
    no["treat_T_C"] = no.austenitize_T_C
    no = _birbilis_compositions(no, catalog)
    # physically implausible rows are kept, flagged and left out of every metric
    # (as the literature compilation does): an "annealed" 0.5 % C steel at
    # 1255 MPa with a yield ratio of 0.86 is a quenched-and-tempered value under
    # the wrong label; annealed 4130's own yield ratio of 0.82 is genuine.
    yr = no.YS_MPa / no.UTS_MPa
    bad = ((no.route == "FC") & ((no.UTS_MPa > 1100) | (yr > 0.85))) | ((no.route == "AC") & (yr > 0.85))
    no["quality_flag"] = np.where(bad, "implausible for the stated treatment", "")
    qt["quality_flag"] = ""
    keep = ["sample_id", "entry", "grade", "family", "route", "processing", "quench_medium", "treat_T_C",
            "treat_T_C_given", "temper_T_C", "temper_time_h", "section_mm", "catalog_grade",
            "composition_source", "quality_flag"] + EL + ["UTS_MPa", "YS_MPa", "El_pct"]
    return qt[keep].reset_index(drop=True), no[[k for k in keep if k in no.columns]].reset_index(drop=True)


def build_all():
    config.ensure_dirs()
    catalog = catmod.build_catalog()
    catalog.to_csv(config.PROCESSED_DIR / "grade_catalog.csv", index=False)
    prim = property_table(catalog)
    prim.to_csv(config.PROCESSED_DIR / "properties_handbook.csv", index=False)
    sec = secondary_property_table(catalog, prim)
    sec.to_csv(config.PROCESSED_DIR / "properties_secondary.csv", index=False)
    ext = structural_external()
    ext.to_csv(config.PROCESSED_DIR / "structural_external.csv", index=False)
    sb = steelbench_heats()
    sb.to_csv(config.PROCESSED_DIR / "heats_steelbench.csv", index=False)
    qt, nb = heat_treatment_tables(catalog)
    qt.to_csv(config.PROCESSED_DIR / "qt_bars.csv", index=False)
    nb.to_csv(config.PROCESSED_DIR / "normalized_bars.csv", index=False)
    return {"catalog": catalog, "primary": prim, "secondary": sec, "external": ext, "steelbench": sb,
            "qt": qt, "normalized_bars": nb}
