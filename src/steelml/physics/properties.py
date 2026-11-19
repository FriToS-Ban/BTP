"""Literature structure-property relations used as the *physics baseline*.

These equations are the textbook starting point for ferrite-pearlite steels.
The machine-learning models of :mod:`steelml.models.property_models` either
re-learn their coefficients (grey box) or correct their residuals, and the
benchmark in ``pipeline/s05_property_models.py`` measures how much each step
adds over the literature values below.

Units: stresses in MPa, ferrite grain size d and pearlite spacing S0 in um
(converted to mm inside, as in the original papers), compositions in wt%.

Relations
---------
Pickering & Gladman (1963), mild steels with little pearlite:
    YS  = 53.9 + 32.3 Mn + 83.2 Si + 354 sqrt(N_f) + 17.4 d^-1/2
    UTS = 294 + 27.7 Mn + 83.2 Si + 3.85 (%pearlite) + 7.7 d^-1/2

Gladman, McIvor & Pickering (1972), ferrite-pearlite of any pearlite content,
non-linear mixture in f_alpha^(1/3):
    YS  = fa^(1/3) (35.4 + 58.5 Mn + 17.4 d^-1/2)
          + (1 - fa^(1/3)) (178.9 + 3.9 S0^-1/2) + 63 Si + 425 sqrt(N_f)
    UTS = fa^(1/3) (246 + 1143 sqrt(N_f) + 18.1 d^-1/2)
          + (1 - fa^(1/3)) (719 + 3.5 S0^-1/2) + 95 Si

Maynier, Jungmann & Dollet (1978), Vickers hardness of each constituent as a
function of the cooling rate Vr (C/h at 700 C):
    HV_M  = 127 + 949C + 27Si + 11Mn + 8Ni + 16Cr + 21 log Vr
    HV_B  = -323 + 185C + 330Si + 153Mn + 65Ni + 144Cr + 191Mo
            + (89 + 53C - 55Si - 22Mn - 10Ni - 20Cr - 33Mo) log Vr
    HV_FP = 42 + 223C + 53Si + 30Mn + 12.6Ni + 7Cr + 19Mo
            + (10 - 19Si + 4Ni + 8Cr + 130V) log Vr

Pickering (1978), impact transition temperature of ferrite-pearlite steels:
    ITT (C) = -19 + 44 Si + 700 sqrt(N_f) + 2.2 (%pearlite) - 11.5 d^-1/2

References
----------
Pickering, F.B., Gladman, T. (1963) ISI Special Report 81, 10-20.
Gladman, T., McIvor, I.D., Pickering, F.B. (1972) J. Iron Steel Inst. 210, 916.
Pickering, F.B. (1978) Physical Metallurgy and the Design of Steels, Applied
    Science, London.
Maynier, P., Jungmann, B., Dollet, J. (1978) in Hardenability Concepts with
    Applications to Steel, AIME, 518-545.
"""

from __future__ import annotations

import numpy as np

N_FREE_DEFAULT = 0.002   # wt% free nitrogen assumed for Al-killed steels


def _inv_sqrt_mm(length_um):
    return (np.maximum(np.asarray(length_um, float), 1e-3) / 1000.0) ** -0.5


def ys_pickering_mild(Mn, Si, d_um, N_free=N_FREE_DEFAULT):
    return (53.9 + 32.3 * np.asarray(Mn) + 83.2 * np.asarray(Si)
            + 354.0 * np.sqrt(N_free) + 17.4 * _inv_sqrt_mm(d_um))


def uts_pickering_mild(Mn, Si, f_pearlite, d_um):
    return (294.0 + 27.7 * np.asarray(Mn) + 83.2 * np.asarray(Si)
            + 3.85 * 100.0 * np.asarray(f_pearlite) + 7.7 * _inv_sqrt_mm(d_um))


def ys_gmp(f_alpha, Mn, Si, d_um, S0_um, N_free=N_FREE_DEFAULT):
    fa3 = np.clip(np.asarray(f_alpha, float), 0.0, 1.0) ** (1.0 / 3.0)
    return (fa3 * (35.4 + 58.5 * np.asarray(Mn) + 17.4 * _inv_sqrt_mm(d_um))
            + (1.0 - fa3) * (178.9 + 3.9 * _inv_sqrt_mm(S0_um))
            + 63.0 * np.asarray(Si) + 425.0 * np.sqrt(N_free))


def uts_gmp(f_alpha, Si, d_um, S0_um, N_free=N_FREE_DEFAULT):
    fa3 = np.clip(np.asarray(f_alpha, float), 0.0, 1.0) ** (1.0 / 3.0)
    return (fa3 * (246.0 + 1143.0 * np.sqrt(N_free) + 18.1 * _inv_sqrt_mm(d_um))
            + 95.0 * np.asarray(Si)
            + (1.0 - fa3) * (719.0 + 3.5 * _inv_sqrt_mm(S0_um)))


def hv_maynier(comp, vr_c_per_h):
    """Return dict of HV for martensite, bainite and ferrite-pearlite."""
    g = lambda k: np.asarray(comp.get(k, 0.0) if hasattr(comp, "get") else comp[k], float)  # noqa: E731
    C, Si, Mn, Ni, Cr, Mo, V = (g(k) for k in ("C", "Si", "Mn", "Ni", "Cr", "Mo", "V"))
    lv = np.log10(np.maximum(np.asarray(vr_c_per_h, float), 1e-3))
    hv_m = 127 + 949 * C + 27 * Si + 11 * Mn + 8 * Ni + 16 * Cr + 21 * lv
    hv_b = (-323 + 185 * C + 330 * Si + 153 * Mn + 65 * Ni + 144 * Cr + 191 * Mo
            + (89 + 53 * C - 55 * Si - 22 * Mn - 10 * Ni - 20 * Cr - 33 * Mo) * lv)
    hv_fp = (42 + 223 * C + 53 * Si + 30 * Mn + 12.6 * Ni + 7 * Cr + 19 * Mo
             + (10 - 19 * Si + 4 * Ni + 8 * Cr + 130 * V) * lv)
    return {"HV_M": hv_m, "HV_B": hv_b, "HV_FP": hv_fp}


def hv_mixture(comp, vr_c_per_h, f_fp, f_b, f_m):
    h = hv_maynier(comp, vr_c_per_h)
    tot = np.maximum(np.asarray(f_fp) + np.asarray(f_b) + np.asarray(f_m), 1e-9)
    return (np.asarray(f_fp) * h["HV_FP"] + np.asarray(f_b) * h["HV_B"]
            + np.asarray(f_m) * h["HV_M"]) / tot


def itt_pickering(Si, f_pearlite, d_um, N_free=N_FREE_DEFAULT):
    return (-19.0 + 44.0 * np.asarray(Si) + 700.0 * np.sqrt(N_free)
            + 2.2 * 100.0 * np.asarray(f_pearlite) - 11.5 * _inv_sqrt_mm(d_um))


def hb_from_hv(hv):
    """Approximate Brinell (10 mm ball, 3000 kgf) from Vickers for steels.

    HB ~= 0.95 HV over 100-450 HV (ASTM E140 conversion tables for non-
    austenitic steels are close to this ratio in that range).
    """
    return 0.95 * np.asarray(hv, float)


def hv_from_hb(hb):
    return np.asarray(hb, float) / 0.95


def uts_from_hv(hv):
    """Rule-of-thumb UTS (MPa) ~= 3.3 HV for carbon and low-alloy steels."""
    return 3.3 * np.asarray(hv, float)


def physics_baseline(f_ferrite, f_pearlite, f_bainite, f_martensite, comp,
                     d_um, S0_um, vr_c_per_h, N_free=N_FREE_DEFAULT):
    """Literature-only predictions of UTS, YS and hardness for one route.

    Ferrite-pearlite part: Gladman-McIvor-Pickering (UTS, YS) and Maynier
    (hardness). Bainite/martensite part: Maynier hardness converted to UTS
    with 3.3 HV and a yield ratio of 0.75 (an assumption for as-transformed,
    untempered constituents). The two parts are mixed linearly by volume.
    """
    g = lambda k: np.asarray(comp.get(k, 0.0) if hasattr(comp, "get") else comp[k], float)  # noqa: E731
    f_fp = np.asarray(f_ferrite) + np.asarray(f_pearlite)
    f_bm = np.asarray(f_bainite) + np.asarray(f_martensite)
    fa_in_fp = np.where(f_fp > 1e-6, np.asarray(f_ferrite) / np.maximum(f_fp, 1e-6), 0.0)
    uts_fp = uts_gmp(fa_in_fp, g("Si"), d_um, S0_um, N_free)
    ys_fp = ys_gmp(fa_in_fp, g("Mn"), g("Si"), d_um, S0_um, N_free)
    hv = hv_maynier(comp, vr_c_per_h)
    hv_bm = np.where(f_bm > 1e-6,
                     (np.asarray(f_bainite) * hv["HV_B"] + np.asarray(f_martensite) * hv["HV_M"])
                     / np.maximum(f_bm, 1e-6), 0.0)
    uts = f_fp * uts_fp + f_bm * uts_from_hv(hv_bm)
    ys = f_fp * ys_fp + f_bm * 0.75 * uts_from_hv(hv_bm)
    hv_tot = f_fp * hv["HV_FP"] + f_bm * hv_bm
    return {"UTS_phys": uts, "YS_phys": ys, "HV_phys": hv_tot, "HB_phys": hb_from_hv(hv_tot)}


# uniform / total elongation, with every total elongation first converted to
# the handbook gauge (4.5 sqrt(S0), Oliver / ISO 2566-1): mean and SD of 10
# literature tests from 4 sources (stage 6). As tested, on their own gauges,
# the ratio is 0.555 +/- 0.075; on one gauge basis the scatter falls by a third.
UNIFORM_TOTAL_RATIO = 0.508
UNIFORM_TOTAL_RATIO_SD = 0.051


def uniform_elongation_estimate(el_total, el_lo=None, el_hi=None, ratio=UNIFORM_TOTAL_RATIO,
                                ratio_sd=UNIFORM_TOTAL_RATIO_SD, z=1.645):
    """Uniform elongation (engineering, %) from total elongation, and n.

    In every open-access ferrite-pearlite tensile test we found that reports
    both quantities (10 tests, 4 independent sources, annealed and air-cooled
    states, 0.1-0.8 % C), uniform elongation is a stable fraction of total
    elongation once the total elongation is put on the handbook gauge
    (4.5 sqrt(S0), Oliver's relation): e_u / e_t = 0.508 +/- 0.051. Total
    elongation depends on the gauge geometry and uniform elongation does
    not, so the ratio only applies to handbook-basis total elongations, which
    is what the models predict. The value n = ln(1 + e_u) is Considere's
    true uniform strain; for steels with a yield-point plateau it includes the
    Lueders strain, so it is a true uniform strain, not a fitted Hollomon
    exponent.

    A Hollomon curve through the yield point and the UTS (see
    hollomon_n_from_ratio) was also tested. Started at the end of a Lueders
    plateau it over-predicted e_u by ~12 points on these tests; started at
    0.2 % strain it under-predicted by ~3.5 points (4 annealed tests) and, for
    fine pearlite, gives uniform elongations larger than the total
    elongation. Lueders-yielding ferrite-pearlite steels do not follow a
    single power law from the yield point, so the Hollomon route is kept only
    as a cross-check.

    Returns n, e_u and, if total-elongation bounds are given, a 90 % band
    combining the ratio's spread with the total-elongation interval.
    """
    et = np.asarray(el_total, float)
    eu = ratio * et
    n = np.log1p(eu / 100.0)
    if el_lo is None or el_hi is None:
        return n, eu
    lo = np.maximum(ratio - z * ratio_sd, 0.05) * np.asarray(el_lo, float)
    hi = (ratio + z * ratio_sd) * np.asarray(el_hi, float)
    return n, eu, lo, hi


def hollomon_n_from_ratio(ys, uts, luders_strain):
    """Strain-hardening exponent n of a Hollomon curve sigma = K eps^n that
    passes through the yield point at ``luders_strain`` and whose Considere
    maximum gives the engineering UTS:  UTS/YS = (n/eps_L)^n exp(-n).

    Solved by bisection on n in (0.01, 0.6). Returns n and the engineering
    uniform elongation exp(n) - 1 (%). A *model-derived* estimate; see the
    report for its assumptions and uncertainty.
    """
    ys = np.atleast_1d(np.asarray(ys, float))
    uts = np.atleast_1d(np.asarray(uts, float))
    eL = np.broadcast_to(np.asarray(luders_strain, float), ys.shape)
    target = np.log(np.maximum(uts / ys, 1.0001))
    lo = np.full(ys.shape, 1e-3)
    hi = np.full(ys.shape, 0.6)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = mid * np.log(mid / eL) - mid - target
        # f is increasing in n for n > eL
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f >= 0, mid, hi)
    n = 0.5 * (lo + hi)
    return n, 100.0 * (np.exp(n) - 1.0)


# ---------------------------------------------------------------------------
# Elongation on a common gauge basis
# ---------------------------------------------------------------------------
# Total elongation depends on the gauge length L0 relative to the specimen
# size (sqrt(S0)), because the necking strain is spread over L0 (Barba's
# law). The handbook values are ASTM elongations in 50 mm on a 12.5-12.8 mm
# round, L0 = 4.5 sqrt(S0); European tests use 5.65 sqrt(S0). Oliver's
# relation (ISO 2566-1, carbon and low-alloy steels) converts between gauge
# geometries: A2 = A1 (k1 / k2)^0.4 with k = L0 / sqrt(S0).
HANDBOOK_GAUGE_K = 4.5
PROPORTIONAL_GAUGE_K = 5.65


def elongation_to_gauge(A, k_from, k_to=HANDBOOK_GAUGE_K):
    """Convert a total elongation measured with gauge ratio k_from to k_to."""
    return np.asarray(A, float) * (np.asarray(k_from, float) / k_to) ** 0.4


def gauge_ratio_from_text(text):
    """k = L0 / sqrt(S0) from a free-text specimen description such as
    'round, GL 40 mm, dia 8 mm' or 'flat, GL 10 mm x W 5 mm x t 1 mm'.
    Returns NaN when the geometry cannot be read."""
    import re
    if not isinstance(text, str):
        return np.nan
    t = text.lower().replace(",", " ")
    num = r"(\d+(?:\.\d+)?)"
    gl = re.search(r"(?:gl|gauge length|gage length)\s*=?\s*" + num + r"\s*mm", t) or \
        re.search(r"parallel length\s*" + num + r"\s*mm", t)
    dia = re.search(r"(?:dia(?:meter)?|ø|d\s*=)\s*" + num + r"\s*mm", t)
    w = re.search(r"\bw\s*" + num + r"\s*mm", t)
    th = re.search(r"\bt\s*" + num + r"\s*mm", t)
    if gl is None:
        return np.nan
    L0 = float(gl.group(1))
    if dia is not None:
        S0 = np.pi * float(dia.group(1)) ** 2 / 4.0
    elif w is not None and th is not None:
        S0 = float(w.group(1)) * float(th.group(1))
    else:
        return np.nan
    return L0 / np.sqrt(S0)
