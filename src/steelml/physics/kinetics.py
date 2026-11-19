"""Austenite decomposition under an arbitrary cooling path.

Model
-----
Semi-empirical Kirkaldy-Venugopalan kinetics as modified by Li et al. (1998),
following the open-source implementation of Collins et al. (2023), and
integrated with Scheil's additivity rule. For a product j the isothermal time
to reach a normalised fraction X is

    tau_j(X, T) = S(X) / k_j(T),
    S(X) = integral_0^X dx / [x^(0.4(1-x)) (1-x)^(0.4x)],

    k_F = 2^(0.41 G) (Ae3 - T)^3 exp(-Q/RT) / FC
    k_P = 2^(0.32 G) (Ae1 - T)^3 exp(-Q/RT) / PC
    k_B = 2^(0.29 G) (Bs  - T)^2 exp(-Q/RT) / BC

    FC = exp(1.00 + 6.31C + 1.78Mn + 0.31Si + 1.12Ni + 2.70Cr + 4.06Mo)
    PC = exp(-4.25 + 4.12C + 4.36Mn + 0.44Si + 1.71Ni + 3.33Cr + 5.19 sqrt(Mo))
    BC = exp(-10.23 + 10.18C + 0.85Mn + 0.55Ni + 0.90Cr + 0.36Mo)

with Q = 27500 cal/mol, R = 1.987 cal/(mol K) and G the ASTM prior-austenite
grain-size number. Under additivity the transformation reaches X when the
integral of k_j(T(t)) dt equals S(X); we therefore integrate in "S-space",
which is exact for an isokinetic reaction and unconditionally stable.

Coupling rules (as in Li et al. / Collins et al.):
* Ferrite grows below Ae3 until pearlite or bainite starts; its amount is
  bounded by the para-equilibrium lever rule on the extrapolated Ae3 line.
  Ferrite enriches the remaining austenite in carbon (mass balance).
* Pearlite may start below Ae1 once the (enriched) austenite lies below both
  its own Ae3 and Acm (Hultgren extrapolation).
* Bainite starts below Bs (evaluated with the enriched austenite carbon) and
  stops pearlite/ferrite growth.
* Martensite forms below Ms of the enriched austenite following
  Koistinen-Marburger, f = 1 - exp(-0.011 (Ms - T)).

Extension made here (flagged as an assumption): in hypereutectoid steels a
grain-boundary proeutectoid cementite reaction is allowed between Acm and the
pearlite start, with the pearlite rate constant and an (Acm - T)^3 driving
force, bounded by the lever rule on the Acm line. Li et al. do not model this
constituent; it is only used to show the qualitative furnace-versus-air
difference in hypereutectoid steels.

References: Kirkaldy & Venugopalan (1984) in Phase Transformations in Ferrous
Alloys, TMS; Li et al. (1998) Metall. Mater. Trans. B 29, 661; Collins et al.
(2023) Metals 13, 1168; Koistinen & Marburger (1959) Acta Metall. 7, 59.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from . import thermo

Q_CAL = 27500.0
R_CAL = 1.987
C_ALPHA = thermo.C_FERRITE
C_THETA = thermo.C_CEMENTITE
X_START = 0.01   # normalised fraction that defines a transformation "start"
from .. import config as _config  # noqa: E402


def _kinetic_params():
    """Adjustable constants of the kinetic model, read at call time (not
    frozen into the compiled kernel):

    [0] multiplier on the ferrite rate constant      (config.FERRITE_RATE_FACTOR)
    [1] multiplier on the pearlite (and proeutectoid cementite) rate
                                                      (config.PEARLITE_RATE_FACTOR)
    [2] multiplier on the bainite rate constant      (config.BAINITE_RATE_FACTOR)
    [3] shift of the bainite-start temperature, C    (config.BAINITE_START_SHIFT_C)

    The published Li et al. model is (1, 1, 1, 0). The calibrated values, and
    the data they come from, are set in config (see pipeline/s08_hardenability.py)."""
    return np.array([float(getattr(_config, "FERRITE_RATE_FACTOR", 1.0)),
                     float(getattr(_config, "PEARLITE_RATE_FACTOR", 1.0)),
                     float(getattr(_config, "BAINITE_RATE_FACTOR", 1.0)),
                     float(getattr(_config, "BAINITE_START_SHIFT_C", 0.0))])


def _ferrite_factor():
    return float(_kinetic_params()[0])


def _pearlite_mode():
    """0: pearlite incubation accumulates from Ae1 (Li et al. / Collins et al.);
    1: it accumulates only once the austenite is inside the Hultgren region.
    See config.PEARLITE_INCUBATION and pipeline/s03_transformation_model.py."""
    return 1 if str(getattr(_config, "PEARLITE_INCUBATION", "ae1")).lower() == "hultgren" else 0

# Order of elements in the composition vector passed to the numba kernels
KIN_ELEMENTS = ("C", "Mn", "Si", "Ni", "Cr", "Mo", "Co")

# Output columns of the kernel
OUT_COLUMNS = (
    "f_ferrite", "f_pearlite", "f_bainite", "f_martensite", "f_cementite_pro",
    "f_austenite_ret", "T_ferrite_start", "T_pearlite_start", "T_bainite_start",
    "T_martensite_start", "T_ferrite_mean", "T_pearlite_mean", "T_bainite_mean",
    "C_austenite_at_pearlite", "Ae3", "Ae1", "Bs", "Ms_bulk", "T_finish",
)


# ---------------------------------------------------------------------------
# S(X) table
# ---------------------------------------------------------------------------

def _build_s_table(n=4001):
    x = np.linspace(0.0, 1.0, n)
    xm = 0.5 * (x[1:] + x[:-1])
    integrand = 1.0 / (xm ** (0.4 * (1.0 - xm)) * (1.0 - xm) ** (0.4 * xm))
    s = np.concatenate([[0.0], np.cumsum(integrand * np.diff(x))])
    return x, s


S_X, S_VAL = _build_s_table()
S_START = float(np.interp(X_START, S_X, S_VAL))
S_MAX = float(S_VAL[-1])


@njit(cache=True, error_model="numpy")
def _s_inverse(s, s_x, s_val):
    if s <= 0.0:
        return 0.0
    if s >= s_val[-1]:
        return 1.0
    # binary search on the monotone table
    lo, hi = 0, s_val.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if s_val[mid] < s:
            lo = mid
        else:
            hi = mid
    w = (s - s_val[lo]) / (s_val[hi] - s_val[lo])
    return s_x[lo] + w * (s_x[hi] - s_x[lo])


# ---------------------------------------------------------------------------
# Thermodynamic helpers inside numba (Celsius)
# ---------------------------------------------------------------------------

@njit(cache=True, error_model="numpy")
def _ae3(c, mn, si, ni, cr):
    return ((1570.0 - 323.0 * c - 25.0 * mn + 80.0 * si - 3.0 * cr - 32.0 * ni) - 32.0) * 5.0 / 9.0


@njit(cache=True, error_model="numpy")
def _ae1(mn, si, ni, cr):
    return ((1333.0 - 25.0 * mn + 40.0 * si + 42.0 * cr - 26.0 * ni) - 32.0) * 5.0 / 9.0


@njit(cache=True, error_model="numpy")
def _acm(c, mn, si, ni, cr, mo):
    return (224.4 + 992.4 * c - 465.1 * c * c + 46.7 * cr + 19.0 * c * cr - 6.1 * cr * cr
            + 7.6 * mn + 10.0 * mo - 6.8 * cr * mo - 6.9 * ni + 3.7 * c * ni - 2.7 * cr * ni
            + 0.8 * ni * ni + 16.7 * si)


@njit(cache=True, error_model="numpy")
def _ae3_carbon(T, mn, si, ni, cr):
    tf = T * 9.0 / 5.0 + 32.0
    return (tf - 1570.0 + 25.0 * mn - 80.0 * si + 3.0 * cr + 32.0 * ni) / (-323.0)


@njit(cache=True, error_model="numpy")
def _acm_carbon(T, mn, si, ni, cr, mo):
    """Carbon on the Acm curve at temperature T (root of the quadratic)."""
    b2 = -465.1
    b1 = 992.4 + 19.0 * cr + 3.7 * ni
    b0 = (224.4 + 46.7 * cr - 6.1 * cr * cr + 7.6 * mn + 10.0 * mo - 6.8 * cr * mo
          - 6.9 * ni - 2.7 * cr * ni + 0.8 * ni * ni + 16.7 * si) - T
    disc = b1 * b1 - 4.0 * b2 * b0
    if disc < 0.0:
        return 10.0
    r = (-b1 + math.sqrt(disc)) / (2.0 * b2)
    if r < 0.0:
        r = (-b1 - math.sqrt(disc)) / (2.0 * b2)
    return r


@njit(cache=True, error_model="numpy")
def _bs(c, mn, ni, cr, mo):
    return 637.0 - 58.0 * c - 35.0 * mn - 15.0 * ni - 34.0 * cr - 41.0 * mo


@njit(cache=True, error_model="numpy")
def _ms(c, mn, si, ni, cr, mo, co):
    return 561.0 - 474.0 * c - 33.0 * mn - 17.0 * ni - 17.0 * cr - 21.0 * mo - 7.5 * si + 10.0 * co


@njit(cache=True, error_model="numpy")
def _arr(T):
    return math.exp(-Q_CAL / (R_CAL * (T + 273.15)))


@njit(cache=True, error_model="numpy")
def _austenite_carbon(c0, xf, xc, c_prev):
    """Carbon of the remaining austenite by mass balance; unchanged once
    (almost) no austenite is left, e.g. in extra-low-carbon steels whose
    carbon is below the ferrite solubility limit."""
    rem = 1.0 - xf - xc
    if rem < 1e-6:
        return c_prev
    c = (c0 - xf * C_ALPHA - xc * C_THETA) / rem
    return c if c > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Core kernel: one composition, one cooling path
# ---------------------------------------------------------------------------

@njit(cache=True, error_model="numpy")
def _simulate_one(comp, G, t_arr, T_arr, s_x, s_val, s_start, kp, p_mode=0):
    c0, mn, si, ni, cr, mo, co = comp[0], comp[1], comp[2], comp[3], comp[4], comp[5], comp[6]
    out = np.full(len(OUT_COLUMNS), np.nan)

    ae3b = _ae3(c0, mn, si, ni, cr)
    ae1b = _ae1(mn, si, ni, cr)
    bs_b = _bs(c0, mn, ni, cr, mo) + kp[3]
    ms_b = _ms(c0, mn, si, ni, cr, mo, co)
    acm_b = _acm(c0, mn, si, ni, cr, mo)
    ce_hyper = c0 > _ae3_carbon(ae1b, mn, si, ni, cr)  # beyond eutectoid

    FC = math.exp(1.00 + 6.31 * c0 + 1.78 * mn + 0.31 * si + 1.12 * ni + 2.70 * cr + 4.06 * mo)
    PC = math.exp(-4.25 + 4.12 * c0 + 4.36 * mn + 0.44 * si + 1.71 * ni + 3.33 * cr + 5.19 * math.sqrt(mo))
    gF = 2.0 ** (0.41 * G)
    gP = 2.0 ** (0.32 * G)
    gB = 2.0 ** (0.29 * G)

    xf = 0.0; xp = 0.0; xb = 0.0; xm = 0.0; xc = 0.0
    sF = 0.0; sP = 0.0; sB = 0.0; sC = 0.0
    p_started = False; b_started = False; m_started = False; f_started = False
    p_possible = False
    xg_p0 = 0.0; xg_b0 = 0.0; xg_m0 = 0.0
    cg = c0          # carbon of the remaining austenite
    BC = 0.0; ms_eff = ms_b
    TFs = np.nan; TPs = np.nan; TBs = np.nan; TMs = np.nan
    wf = 0.0; wp = 0.0; wb = 0.0; tfsum = 0.0; tpsum = 0.0; tbsum = 0.0
    c_at_p = np.nan
    T_fin = np.nan   # diffusional transformation finished (< 1 % austenite left before Ms)

    n = T_arr.shape[0]
    for i in range(1, n):
        T = 0.5 * (T_arr[i] + T_arr[i - 1])
        dt = t_arr[i] - t_arr[i - 1]
        if dt <= 0.0:
            continue
        xa = 1.0 - xf - xp - xb - xm - xc
        if xa <= 0.01 and (not m_started) and math.isnan(T_fin):
            T_fin = T_arr[i - 1]
        if xa <= 1e-6:
            break
        arr = _arr(T)

        # ---------------- martensite ----------------
        if not m_started:
            ms_eff = _ms(cg, mn, si, ni, cr, mo, co)
        if T < ms_eff:
            if not m_started:
                m_started = True
                xg_m0 = xa
                TMs = ms_eff
            xm_new = xg_m0 * (1.0 - math.exp(-0.011 * (ms_eff - T_arr[i])))
            if xm_new > xm:
                xm = xm_new
            continue  # diffusional reactions are frozen below Ms

        # ---------------- bainite ----------------
        bs_eff = _bs(cg, mn, ni, cr, mo) + kp[3]
        if T < bs_eff:
            if BC == 0.0:
                BC = math.exp(-10.23 + 10.18 * cg + 0.85 * mn + 0.55 * ni + 0.90 * cr + 0.36 * mo)
            sB += kp[2] * gB * (bs_eff - T) ** 2 * arr / BC * dt
            if (not b_started) and sB >= s_start:
                b_started = True
                xg_b0 = xa
                TBs = T
            if b_started:
                yb = _s_inverse(sB, s_x, s_val)
                xb_new = yb * xg_b0
                if xb_new > xb:
                    dxb = min(xb_new - xb, xa)
                    xb += dxb
                    wb += dxb; tbsum += dxb * T
            if b_started:
                continue

        # ---------------- ferrite ----------------
        if (not p_started) and (not ce_hyper) and T < ae3b:
            sF += kp[0] * gF * (ae3b - T) ** 3 * arr / FC * dt
            if sF >= s_start:
                if not f_started:
                    f_started = True
                    TFs = T
                yf = _s_inverse(sF, s_x, s_val)
                c_lim = _ae3_carbon(T, mn, si, ni, cr)
                xf_eq = 0.0
                if c_lim > c0:
                    xf_eq = (c_lim - c0) / max(c_lim - C_ALPHA, 1e-9)
                    if xf_eq > 1.0:
                        xf_eq = 1.0
                xf_new = yf * xf_eq
                if xf_new > xf:
                    dxf = min(xf_new - xf, xa)
                    xf += dxf
                    wf += dxf; tfsum += dxf * T
                    cg = _austenite_carbon(c0, xf, xc, cg)

        # ---------------- proeutectoid cementite (hypereutectoid) ----------------
        if (not p_started) and ce_hyper and T < acm_b:
            sC += kp[1] * gP * (acm_b - T) ** 3 * arr / PC * dt
            if sC >= s_start:
                yc = _s_inverse(sC, s_x, s_val)
                # bounded by the equilibrium amount at Ae1 (no Hultgren
                # extrapolation of the Acm line below the eutectoid)
                c_lim = _acm_carbon(max(T, ae1b), mn, si, ni, cr, mo)
                xc_eq = 0.0
                if c_lim < c0:
                    xc_eq = (c0 - c_lim) / max(C_THETA - c_lim, 1e-9)
                xc_new = yc * xc_eq
                if xc_new > xc:
                    xc += min(xc_new - xc, xa)
                    cg = _austenite_carbon(c0, xf, xc, cg)

        # ---------------- pearlite ----------------
        # Incubation (the additivity sum) accumulates as soon as T < Ae1, as in
        # Li et al. / Collins et al.; the Hultgren condition (austenite below
        # its own Ae3 and Acm) gates only the start of growth.
        if not p_possible:
            if T < ae1b and T <= _ae3(cg, mn, si, ni, cr) and T <= _acm(cg, mn, si, ni, cr, mo):
                p_possible = True
        if ae1b > T and (p_mode == 0 or p_possible):
            sP += kp[1] * gP * (ae1b - T) ** 3 * arr / PC * dt
            if p_possible and (not p_started) and sP >= s_start:
                p_started = True
                xa = 1.0 - xf - xp - xb - xm - xc
                xg_p0 = xa
                TPs = T
                c_at_p = cg
            if p_started:
                yp = _s_inverse(sP, s_x, s_val)
                xp_new = yp * xg_p0
                if xp_new > xp:
                    dxp = min(xp_new - xp, 1.0 - xf - xp - xb - xm - xc)
                    xp += dxp
                    wp += dxp; tpsum += dxp * T

    xa = 1.0 - xf - xp - xb - xm - xc
    if xa < 0.0:
        xa = 0.0
    out[0] = xf; out[1] = xp; out[2] = xb; out[3] = xm; out[4] = xc; out[5] = xa
    out[6] = TFs; out[7] = TPs; out[8] = TBs; out[9] = TMs
    out[10] = tfsum / wf if wf > 0 else np.nan
    out[11] = tpsum / wp if wp > 0 else np.nan
    out[12] = tbsum / wb if wb > 0 else np.nan
    out[13] = c_at_p
    out[14] = ae3b; out[15] = ae1b; out[16] = bs_b; out[17] = ms_b
    out[18] = T_fin
    return out


@njit(parallel=True, cache=True, error_model="numpy")
def _simulate_many(comps, Gs, t_paths, T_paths, s_x, s_val, s_start, kp, p_mode=0):
    n = comps.shape[0]
    res = np.empty((n, len(OUT_COLUMNS)))
    for k in prange(n):
        res[k] = _simulate_one(comps[k], Gs[k], t_paths[k], T_paths[k], s_x, s_val, s_start, kp, p_mode)
    return res


# ---------------------------------------------------------------------------
# Python-facing API
# ---------------------------------------------------------------------------

def comp_vector(comp):
    """Composition dict/Series -> numpy vector in KIN_ELEMENTS order."""
    return np.array([float(comp.get(k, 0.0) or 0.0) if hasattr(comp, "get") else 0.0
                     for k in KIN_ELEMENTS])


def simulate(comp, G, t, T):
    """Simulate one composition along one cooling path (arrays t [s], T [C])."""
    res = _simulate_one(comp_vector(comp), float(G), np.asarray(t, float),
                        np.asarray(T, float), S_X, S_VAL, S_START, _kinetic_params(), _pearlite_mode())
    return dict(zip(OUT_COLUMNS, res))


def simulate_batch(comps, Gs, paths):
    """Simulate many (composition, grain size, path) triples in parallel.

    comps : DataFrame with element columns (missing columns -> 0)
    Gs    : array of ASTM grain-size numbers
    paths : list of (t, T) tuples; all paths must share the same length
    """
    import pandas as pd

    mat = np.column_stack([comps[k].to_numpy(float) if k in comps else np.zeros(len(comps))
                           for k in KIN_ELEMENTS])
    t_paths = np.stack([np.asarray(p[0], float) for p in paths])
    T_paths = np.stack([np.asarray(p[1], float) for p in paths])
    res = _simulate_many(mat, np.asarray(Gs, float), t_paths, T_paths, S_X, S_VAL, S_START, _kinetic_params(),
                         _pearlite_mode())
    out = pd.DataFrame(res, columns=OUT_COLUMNS, index=comps.index)
    _sanity_check(out)
    return out


def _sanity_check(out):
    """Fail loudly instead of passing non-physical numbers downstream."""
    fr = out[["f_ferrite", "f_pearlite", "f_bainite", "f_martensite", "f_cementite_pro", "f_austenite_ret"]]
    bad = (~np.isfinite(fr).all(axis=1)) | (fr.min(axis=1) < -1e-6) | ((fr.sum(axis=1) - 1.0).abs() > 1e-3)
    bad |= ~np.isfinite(out["Ae3"]) | (out["Ae3"].abs() > 2000)
    if bad.any():
        raise RuntimeError(f"transformation model returned non-physical output for {int(bad.sum())} rows")


def linear_path(rate, T0=900.0, T_end=20.0, dT=0.5):
    """Constant cooling-rate path (rate in K/s)."""
    T = np.arange(T0, T_end - dT, -dT)
    t = (T0 - T) / rate
    return t, T


def cct_diagram(comp, G, rates, T0=None, dT=0.5):
    """Constant-rate CCT data for plotting (start temperatures and fractions)."""
    import pandas as pd

    if T0 is None:
        T0 = float(thermo.ae3(comp)) + 50.0
    rows = []
    for r in rates:
        t, T = linear_path(r, T0=T0, dT=dT)
        d = simulate(comp, G, t, T)
        d["rate"] = r
        rows.append(d)
    return pd.DataFrame(rows)
