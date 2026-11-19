"""Jominy end-quench test: cooling paths along the bar and the hardness profile.

The ASTM A255 / SAE J406 specimen is a round bar 25.4 mm (1 in.) in diameter
and 101.6 mm (4 in.) long. It is austenitised and then quenched at one end by
a water jet while the rest of the bar cools in still air. Distances from the
quenched end are counted in sixteenths of an inch, J1 = 1.59 mm ... J32 =
50.8 mm. The test is the classic measure of hardenability, and its far end
cools at a few K/s at 700 C, the same rates as still-air cooling of bars of
roughly 10 to 25 mm, so it tests the transformation model exactly where the
furnace-versus-air question becomes interesting for alloy steels.

Heat flow
---------
One-dimensional transient conduction along the bar is solved by explicit
finite differences, with the temperature-dependent conductivity and specific
heat of carbon steel from EN 1993-1-2 (the specific-heat peak near 735 C
stands in for the transformation and magnetic heat effects, as in
:mod:`steelml.physics.cooling`):

    rho c(T) dT/dt = d/dx (k(T) dT/dx) - (4/D) q_air(T)

* quenched end: q = h_q (T - T_w), water at 24 C. ``h_q`` is the one free
  constant; it is set so that the computed cooling rates at 700 C follow the
  standard Jominy table (see ``pipeline/s08_hardenability.py``).
* sides and top end: still-air convection and radiation, as for the
  normalised bars (cooling.h_convection, emissivity 0.8). The radial
  temperature gradient is neglected (the usual one-dimensional Jominy model).

Each position's T(t) is mapped onto the shared temperature grid of the
cooling module, so the same kinetic kernel, phase-hardness relations and
hardness conversion are used as for the furnace and air routes.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from . import cooling

BAR_D_M = 0.0254
BAR_L_M = 0.1016
T_WATER = 24.0
SIXTEENTH_MM = 25.4 / 16.0
STANDARD_J = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12, 13, 14, 15, 16,
              18, 20, 22, 24, 26, 28, 30, 32)


def conductivity_eurocode(T):
    """Thermal conductivity of carbon steel, EN 1993-1-2 (W m^-1 K^-1), T in C."""
    T = np.asarray(T, float)
    return np.where(T < 800.0, 54.0 - 3.33e-2 * T, 27.3)


@njit(cache=True)
def _k(T):
    return 54.0 - 3.33e-2 * T if T < 800.0 else 27.3


@njit(cache=True)
def _cp(T):
    if T < 600.0:
        return 425.0 + 0.773 * T - 1.69e-3 * T * T + 2.22e-6 * T * T * T
    if T < 735.0:
        return 666.0 + 13002.0 / (738.0 - min(T, 737.0))
    if T < 900.0:
        return 545.0 + 17820.0 / (max(T, 732.0) - 731.0)
    return 650.0


@njit(cache=True)
def _q_air(T, D, T_amb, eps):
    dT = max(T - T_amb, 1e-6)
    hc = 1.32 * (dT / D) ** 0.25
    Tk = T + 273.15
    Ta = T_amb + 273.15
    return hc * (T - T_amb) + eps * 5.670374e-8 * (Tk ** 4 - Ta ** 4)


@njit(cache=True)
def _solve(T0, h_q, T_w, T_amb, eps, D, L, dx, dt, t_end, out_idx, grid, T_stop):
    n = int(round(L / dx)) + 1
    T = np.full(n, T0)
    Tn = np.empty(n)
    rho = 7850.0
    m = out_idx.shape[0]
    g = grid.shape[0]
    times = np.zeros((m, g))
    ptr = np.zeros(m, np.int64)
    for j in range(m):
        p = 0
        while p < g and grid[p] > T0:
            p += 1
        ptr[j] = p
    t = 0.0
    sink_coef = 4.0 / D
    while t < t_end:
        for i in range(n):
            ki = _k(T[i])
            ci = rho * _cp(T[i])
            sink = sink_coef * _q_air(T[i], D, T_amb, eps)
            if i == 0:
                kr = 0.5 * (ki + _k(T[1]))
                flux = kr * (T[1] - T[0]) / dx - h_q * (T[0] - T_w)
                Tn[0] = T[0] + dt * (flux / (0.5 * dx) - sink) / ci
            elif i == n - 1:
                kl = 0.5 * (ki + _k(T[i - 1]))
                flux = -kl * (T[i] - T[i - 1]) / dx - _q_air(T[i], D, T_amb, eps)
                Tn[i] = T[i] + dt * (flux / (0.5 * dx) - sink) / ci
            else:
                kl = 0.5 * (ki + _k(T[i - 1]))
                kr = 0.5 * (ki + _k(T[i + 1]))
                lap = (kr * (T[i + 1] - T[i]) - kl * (T[i] - T[i - 1])) / (dx * dx)
                Tn[i] = T[i] + dt * (lap - sink) / ci
        t_new = t + dt
        done = True
        for j in range(m):
            i = out_idx[j]
            while ptr[j] < g and Tn[i] <= grid[ptr[j]]:
                frac = (T[i] - grid[ptr[j]]) / max(T[i] - Tn[i], 1e-12)
                times[j, ptr[j]] = t + frac * dt
                ptr[j] += 1
            if Tn[i] > T_stop:
                done = False
        for i in range(n):
            T[i] = Tn[i]
        t = t_new
        if done:
            break
    # below the last recorded grid point continue at the last cooling rate
    for j in range(m):
        p = ptr[j]
        if p >= g:
            continue
        if p >= 2:
            rate = (grid[p - 2] - grid[p - 1]) / max(times[j, p - 1] - times[j, p - 2], 1e-9)
        else:
            rate = 1.0
        rate = max(rate, 1e-3)
        for q in range(p, g):
            times[j, q] = times[j, p - 1] + (grid[p - 1] - grid[q]) / rate if p >= 1 else t
    return times


def jominy_paths(T0, J=STANDARD_J, h_q=2.0e4, dx=0.5e-3, dt=2.0e-3, t_end=3000.0, T_stop=150.0):
    """Cooling paths (t on the shared temperature grid) at Jominy distances J (sixteenths of an inch)."""
    J = np.asarray(J, float)
    idx = np.round(J * SIXTEENTH_MM * 1e-3 / dx).astype(np.int64)
    grid = cooling.T_GRID.astype(float)
    times = _solve(float(T0), float(h_q), T_WATER, cooling.T_AMB, cooling.EPS, BAR_D_M, BAR_L_M,
                   dx, dt, t_end, idx, grid, T_stop)
    return [(times[j].copy(), grid.copy()) for j in range(len(J))]


def rate_at_700(paths):
    """Cooling rate (K/s) at 700 C of each Jominy path."""
    return np.array([cooling.rate_at(t, T, 700.0) for t, T in paths])


# ASTM E140 (non-austenitic steels): Vickers HV for Rockwell C 20 ... 65.
HRC_POINTS = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0])
HV_AT_HRC = np.array([238.0, 266.0, 302.0, 345.0, 392.0, 446.0, 513.0, 595.0, 697.0, 832.0])


def hrc_from_hv(hv):
    """Rockwell C from Vickers by interpolation in the ASTM E140 table. Values
    below HRC 20 (HV 238) are returned as NaN, because the C scale is not used
    there and the H-bands stop at 20 HRC."""
    hv = np.asarray(hv, float)
    out = np.interp(hv, HV_AT_HRC, HRC_POINTS, left=np.nan, right=np.nan)
    out = np.where(hv > HV_AT_HRC[-1], HRC_POINTS[-1] + (hv - HV_AT_HRC[-1]) / 27.0, out)
    return out


def simulate_profiles(comps, T_aust, G=7.0, J=STANDARD_J, h_q=2.0e4):
    """Predicted Jominy profiles.

    comps : DataFrame of compositions (one row per steel; element columns)
    T_aust: austenitising temperature(s) (C), scalar or one per row
    Returns a long DataFrame: row index of comps, J, rate at 700 C, the
    constituent fractions, Maynier HV and HRC.
    """
    import pandas as pd

    from . import kinetics
    from . import properties as pp

    T_aust = np.broadcast_to(np.asarray(T_aust, float), (len(comps),))
    cache = {}
    rows = []
    for i, (idx, comp) in enumerate(comps.iterrows()):
        key = round(float(T_aust[i]), 0)
        if key not in cache:
            cache[key] = jominy_paths(key, J, h_q=h_q)
        paths = cache[key]
        rep = pd.DataFrame([comp] * len(paths)).reset_index(drop=True)
        kin = kinetics.simulate_batch(rep, np.full(len(paths), G), paths)
        rate = rate_at_700(paths)
        f_fp = kin.f_ferrite + kin.f_pearlite + kin.f_cementite_pro
        hv = pp.hv_mixture(rep, rate * 3600.0, f_fp, kin.f_bainite, kin.f_martensite)
        rows.append(pd.DataFrame({"row": idx, "J": np.asarray(J, float), "rate700": rate,
                                  "f_fp": f_fp.to_numpy(), "f_bainite": kin.f_bainite.to_numpy(),
                                  "f_martensite": kin.f_martensite.to_numpy(), "HV": np.asarray(hv),
                                  "HRC": hrc_from_hv(hv)}))
    return pd.concat(rows, ignore_index=True)
