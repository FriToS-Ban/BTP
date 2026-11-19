"""Physics featurizer: composition + route -> microstructure descriptors.

For each sample (a composition, a cooling route, an austenitising temperature
and a section size) this module

1. builds the cooling path (furnace, still-air or constant rate),
2. runs the transformation model to get phase fractions and the temperatures
   at which ferrite and pearlite formed,
3. converts those temperatures into length scales (pearlite spacing,
   ferrite grain size) and the prior-austenite grain size,
4. evaluates the literature structure-property relations (physics baseline),
5. returns everything as columns, together with derived "physics features"
   (f_alpha^1/3, S0^-1/2, d^-1/2, log cooling rate, ...) used by the ML models.

Routes
  FC  furnace cooling from T_aust (rate at 700 C = config.FC_RATE_700C unless given)
  AC  still-air cooling of a bar of diameter section_mm from T_aust
  AR  as-rolled: still-air cooling from 950 C with a fine recrystallised
      austenite (ASTM 8); used only for the auxiliary hot-rolled data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .physics import cooling, kinetics, microstructure as ms, properties as pp, thermo

EL = config.ELEMENTS


def _paths_for(df, fc_rate):
    paths, rates, t85s, Gs, dg = [], [], [], [], []
    for _, r in df.iterrows():
        route = r["route"]
        coarse = bool(r.get("coarse_grained", False))
        if route == "AR":
            T0, G = 950.0, 8.0 - (2.0 if coarse else 0.0)
            t, T = cooling.air_cooling_path(T0, float(r.get("section_mm", 25.0)))
        else:
            T0 = float(r["treat_T_C"])
            if not np.isfinite(T0) or T0 < 650.0:
                raise ValueError(f"austenitising temperature missing or too low ({T0}) for route {route}")
            G = float(ms.austenite_astm(T0, coarse))
            if route == "FC":
                t, T = cooling.furnace_cooling_path(T0, float(r.get("fc_rate", fc_rate)))
            elif route == "AC":
                t, T = cooling.air_cooling_path(T0, float(r.get("section_mm", 25.0)))
            elif route == "CR":
                t, T = cooling.constant_rate_path(T0, float(r["cooling_rate"]))
            else:
                raise ValueError(route)
        paths.append((t, T))
        rates.append(cooling.rate_at(t, T, 700.0))
        t85s.append(cooling.t85(t, T))
        Gs.append(G)
        dg.append(float(ms.astm_to_um(G)))
    return paths, np.array(rates), np.array(t85s), np.array(Gs), np.array(dg)


def featurize(df: pd.DataFrame, fc_rate=config.FC_RATE_700C) -> pd.DataFrame:
    """Return a copy of ``df`` with physics descriptors appended.

    Required columns: element columns (missing -> 0), 'route', and
    'treat_T_C' for FC/AC/CR rows; optional 'section_mm', 'coarse_grained',
    'fc_rate', 'cooling_rate' (CR rows).
    """
    df = df.copy()
    for el in EL:
        if el not in df.columns:
            df[el] = 0.0
        df[el] = df[el].fillna(0.0)
    paths, rate700, t85, G, d_gamma = _paths_for(df, fc_rate)
    kin = kinetics.simulate_batch(df[EL], G, paths)
    kin.index = df.index
    out = pd.concat([df, kin.add_prefix("k_")], axis=1)
    out["cool_rate_700"] = rate700
    out["log10_rate"] = np.log10(rate700)
    out["t85_s"] = t85
    out["astm_G"] = G
    out["d_gamma_um"] = d_gamma

    lr = thermo.lever_rule(df)
    out["f_ferrite_eq"] = lr["f_ferrite_eq"]
    out["f_pearlite_eq"] = lr["f_pearlite_eq"]
    out["f_cementite_eq"] = lr["f_cementite_eq"]
    out["C_eutectoid"] = lr["C_eutectoid"]
    out["T_eutectoid"] = lr["T_eutectoid"]
    out["eutectoid_class"] = thermo.eutectoid_class(df)
    out["CE_IIW"] = thermo.carbon_equivalent_iiw(df)
    out["in_validity_window"] = thermo.in_validity_window(df)

    fF, fP = out.k_f_ferrite.to_numpy(), out.k_f_pearlite.to_numpy()
    fB, fM = out.k_f_bainite.to_numpy(), out.k_f_martensite.to_numpy()
    fC = out.k_f_cementite_pro.to_numpy()
    # pearlite spacing from the mean pearlite formation temperature
    Tp = np.where(np.isnan(out.k_T_pearlite_mean), out.k_Ae1 - 20.0, out.k_T_pearlite_mean)
    out["pearlite_undercooling"] = np.maximum(out.k_Ae1 - Tp, 5.0)
    out["S0_um"] = ms.pearlite_spacing_um(Tp, out.k_Ae1)
    out["d_alpha_um"] = ms.ferrite_grain_size_um(df["C"], df["Mn"], rate700, d_gamma)
    out["ferrite_undercooling"] = np.where(np.isnan(out.k_T_ferrite_mean), 0.0,
                                           out.k_Ae3 - out.k_T_ferrite_mean)

    vr = rate700 * 3600.0  # C/h for Maynier
    base = pp.physics_baseline(fF, fP + fC, fB, fM, df, out.d_alpha_um, out.S0_um, vr)
    for k, v in base.items():
        out[k] = v
    out["ITT_phys"] = pp.itt_pickering(df["Si"], fP, out.d_alpha_um)

    # derived physics features (dimensionless or SI-like)
    f_fp = np.maximum(fF + fP + fC, 1e-6)
    fa = np.clip(fF / f_fp, 0, 1)
    out["fa_third"] = fa ** (1 / 3)
    out["inv_sqrt_d_mm"] = (out.d_alpha_um / 1000.0) ** -0.5
    out["inv_sqrt_S0_mm"] = (out.S0_um / 1000.0) ** -0.5
    out["f_hard"] = fB + fM            # bainite + martensite
    out["f_fp"] = fF + fP + fC
    # Descriptors for the data-driven models: the three microstructural terms of
    # the Gladman mixture law (ferrite share, ferrite Hall-Petch, pearlite
    # fineness), each weighted by f_FP as in the law itself. They only matter in
    # proportion to how much ferrite + pearlite is present and go smoothly to
    # zero for a bainitic or martensitic structure, whereas the raw ratio
    # fa_third flips between 0 and 1 on a trace of ferrite and the spacing of
    # absent pearlite is only a placeholder (Ae1 - 20 K).
    out["ferrite_term"] = out["f_fp"] * out["fa_third"]
    out["ferrite_hp_term"] = out["ferrite_term"] * out["inv_sqrt_d_mm"]
    out["pearlite_term"] = out["f_fp"] * (1.0 - out["fa_third"]) * out["inv_sqrt_S0_mm"]
    # fraction-weighted mean transformation temperature (martensite taken at
    # Ms - 50 C): a single scalar for "how cold did the austenite transform"
    Tf = np.where(np.isnan(out.k_T_ferrite_mean), 0.0, out.k_T_ferrite_mean)
    Tpm = np.where(np.isnan(out.k_T_pearlite_mean), 0.0, out.k_T_pearlite_mean)
    Tbm = np.where(np.isnan(out.k_T_bainite_mean), 0.0, out.k_T_bainite_mean)
    Tm = np.where(np.isnan(out.k_T_martensite_start), 0.0, out.k_T_martensite_start - 50.0)
    wsum = np.maximum(fF + fP + fC + fB + fM, 1e-6)
    out["T_trans_mean"] = (fF * Tf + (fP + fC) * Tpm + fB * Tbm + fM * Tm) / wsum

    # austenitising superheat: how far above A1 / A3 (or Acm) the steel was
    # soaked. Small superheat leaves undissolved carbide, which favours a
    # divorced (spheroidised) eutectoid product on slow cooling.
    T_aust = np.where(df["route"].to_numpy() == "AR", 950.0, df.get("treat_T_C", pd.Series(np.nan, index=df.index)).to_numpy(float))
    A_upper = np.where(df["C"] > out["C_eutectoid"], thermo.acm(df), out.k_Ae3)
    out["superheat_A1"] = T_aust - out.k_Ae1
    out["superheat_A3cm"] = T_aust - A_upper

    # ferrite solid-solution strengthening (MPa per wt%, Pickering 1978):
    # Mn 32, Si 83, P 678, Cu 39, Ni 0, Cr -31, Mo 11
    out["ss_index"] = (32.0 * df["Mn"] + 83.0 * df["Si"] + 678.0 * df["P"] + 39.0 * df["Cu"]
                       - 31.0 * df["Cr"] + 11.0 * df["Mo"])
    return out


PHYSICS_FEATURES = [
    "k_f_ferrite", "k_f_pearlite", "k_f_bainite", "k_f_martensite", "f_hard",
    "fa_third", "inv_sqrt_d_mm", "inv_sqrt_S0_mm", "ferrite_term", "ferrite_hp_term", "pearlite_term",
    "pearlite_undercooling",
    "ferrite_undercooling", "T_trans_mean", "superheat_A1", "superheat_A3cm",
    "log10_rate", "d_gamma_um", "CE_IIW", "ss_index",
    "UTS_phys", "YS_phys", "HV_phys",
]


# ---------------------------------------------------------------------------
# Hardened (quenched) and quenched-and-tempered states
# ---------------------------------------------------------------------------
# Tensile specimens of bars up to 1.5 in. (38 mm) are machined from the centre,
# those of larger bars at mid-radius (ASTM A370, steel bar products). The
# quench path is computed at that position; stage 10 tests the choice.
SPECIMEN_CENTRE_MAX_MM = 38.0
DEFAULT_TEMPER_TIME_H = 1.0   # when a source gives no tempering time (stage 10 tests 2 h)
# As-quenched martensite hardness levels off above about 0.6 % C at about
# 66 HRC (865 HV): retained austenite offsets the harder plate martensite
# (Krauss 2015, Steels: Processing, Structure, and Performance, ch. 17).
# Maynier's linear carbon term keeps rising (1080 would be 68 HRC), so it is capped.
HV_MARTENSITE_MAX = 865.0
QT_PHYSICS_FEATURES = ["log10_rate", "f_M", "f_B", "f_FP", "HV_M0", "HV_asQ", "P_k", "S_soft",
                       "HV_T_phys", "UTS_qt_phys"]


def specimen_position(D_mm):
    """r/R of the tensile specimen axis: 0 (centre) up to 38 mm, 0.5 above."""
    return np.where(np.asarray(D_mm, float) <= SPECIMEN_CENTRE_MAX_MM, 0.0, 0.5)


def quench_H(medium, h_mult=None):
    """Grossmann H (1/in) for each row's quench medium ('water', 'oil', 'brine',
    'ideal'), optionally scaled by a calibrated multiplier per medium."""
    from .physics import quench
    out = []
    for m in medium:
        if m == "ideal":
            out.append(np.inf)
            continue
        H = quench.severity(m, "mild")["H_nom"]
        if h_mult is not None:
            H *= float(h_mult.get(m, 1.0))
        out.append(H)
    return np.array(out, float)


def featurize_quench(df: pd.DataFrame, theta=None, h_mult=None) -> pd.DataFrame:
    """Physics descriptors of a quenched (route 'Q') or quenched-and-tempered
    (route 'QT') bar.

    Required columns: elements, 'quench_medium' ('water', 'oil', 'brine' or
    'ideal'), 'section_mm' (bar diameter). Optional: 'treat_T_C' (austenitising
    temperature; default thermo.default_austenitising_T(..., 'Q')),
    'temper_T_C' (NaN = untempered), 'temper_time_h' (default 1 h), 'pos_rR'
    (specimen position, default ASTM A370), 'H' (Grossmann severity, default
    the nominal value of the medium, times ``h_mult[medium]``).

    Steps: radial quench path at the specimen position (physics.quench) ->
    transformation kinetics (martensite, bainite, ferrite + pearlite) ->
    as-quenched hardness of each constituent (Maynier et al. 1978) at the
    local 700 C cooling rate -> Hollomon-Jaffe parameter -> tempered hardness
    (physics.tempering, constants ``theta``; prior means if None) -> UTS = 3.3 HV.
    """
    from .physics import quench, tempering

    df = df.copy()
    for el in EL:
        if el not in df.columns:
            df[el] = 0.0
        df[el] = df[el].fillna(0.0)
    if "treat_T_C" not in df.columns:
        df["treat_T_C"] = np.nan
    df["treat_T_C"] = df["treat_T_C"].fillna(pd.Series(
        np.asarray(thermo.default_austenitising_T(df, np.full(len(df), "Q")), float), index=df.index))
    if "temper_T_C" not in df.columns:
        df["temper_T_C"] = np.nan
    if "temper_time_h" not in df.columns:
        df["temper_time_h"] = DEFAULT_TEMPER_TIME_H
    df["temper_time_h"] = df["temper_time_h"].fillna(DEFAULT_TEMPER_TIME_H)
    if "pos_rR" not in df.columns:
        df["pos_rR"] = specimen_position(df["section_mm"])
    H = df["H"].to_numpy(float) if "H" in df.columns else quench_H(df["quench_medium"], h_mult)
    df["H_used"] = H

    cache, paths = {}, []
    for T0, D, Hh, med, pos in zip(df.treat_T_C, df.section_mm, H, df.quench_medium, df.pos_rR):
        key = (round(float(T0)), round(float(D), 1), float(Hh), str(med))
        if key not in cache:
            ps = quench.quench_paths(key[0], key[1], H=Hh, medium=None if np.isfinite(Hh) else None,
                                     T_q=quench.T_QUENCHANT.get(str(med), 25.0),
                                     positions=(0.0, 0.25, 0.5, 0.75, 1.0))
            cache[key] = ps
        ps = cache[key]
        # linear interpolation in r/R between the stored positions
        x = float(np.clip(pos, 0.0, 1.0)) * 4.0
        i = min(int(np.floor(x)), 3)
        w = x - i
        paths.append(((1.0 - w) * ps[i][0] + w * ps[i + 1][0], ps[i][1]))

    G = ms.austenite_astm(df.treat_T_C.to_numpy(float), df.get("coarse_grained", False))
    kin = kinetics.simulate_batch(df[EL], G, paths)
    kin.index = df.index
    out = pd.concat([df, kin.add_prefix("k_")], axis=1)
    rate = np.array([cooling.rate_at(t, T, 700.0) for t, T in paths])
    out["cool_rate_700"] = rate
    out["log10_rate"] = np.log10(np.maximum(rate, 1e-4))
    out["t85_s"] = [cooling.t85(t, T) for t, T in paths]
    out["astm_G"] = G
    fM = out.k_f_martensite.to_numpy() + out.k_f_austenite_ret.to_numpy()   # retained austenite counted with martensite
    fB = out.k_f_bainite.to_numpy()
    fFP = out.k_f_ferrite.to_numpy() + out.k_f_pearlite.to_numpy() + out.k_f_cementite_pro.to_numpy()
    tot = np.maximum(fM + fB + fFP, 1e-9)
    out["f_M"], out["f_B"], out["f_FP"] = fM / tot, fB / tot, fFP / tot
    hv = pp.hv_maynier(df, rate * 3600.0)
    out["HV_M0"] = np.minimum(hv["HV_M"], HV_MARTENSITE_MAX)
    out["HV_B0"] = np.minimum(hv["HV_B"], out["HV_M0"])
    out["HV_FP0"] = hv["HV_FP"]
    out["HV_asQ"] = out.f_M * out.HV_M0 + out.f_B * out.HV_B0 + out.f_FP * out.HV_FP0
    P = np.where(np.isfinite(df.temper_T_C.to_numpy(float)),
                 tempering.hollomon_jaffe(df.temper_T_C.to_numpy(float), df.temper_time_h.to_numpy(float)), np.nan)
    out["P_HJ"] = P
    out["tempered"] = np.isfinite(P).astype(float)
    out["P_k"] = np.where(np.isfinite(P), P / 1000.0, 7.0)   # untempered ~ room-temperature ageing
    out["S_soft"] = np.where(np.isfinite(P), tempering.softening(np.nan_to_num(P), df, theta), 1.0)
    add_tempered_physics(out, theta)
    out["DI_mm"] = __import__("steelml.physics.hardenability", fromlist=["x"]).ideal_critical_diameter(df)[1]
    out["CE_IIW"] = thermo.carbon_equivalent_iiw(df)
    return out


def add_tempered_physics(out, theta=None):
    """(Re)compute the tempered hardness and its UTS for given constants theta."""
    from .physics import tempering
    hvT = tempering.tempered_hv(out.HV_M0, out.HV_B0, out.HV_FP0, out.f_M, out.f_B, out.f_FP,
                                out.P_HJ, out, theta)
    out["HV_T_phys"] = hvT
    out["UTS_qt_phys"] = tempering.UTS_PER_HV * hvT
    return out
