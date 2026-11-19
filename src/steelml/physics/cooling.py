"""Cooling paths for the two heat-treatment routes studied.

Air cooling (normalising)
    A long round bar of diameter D cools in still air. Biot numbers are
    below 0.1 for D <= 100 mm, so a lumped-capacitance balance is used:

        rho c(T) dT/dt = -(4/D) [h_c (T - T_inf) + eps sigma (T^4 - T_inf^4)]

    c(T) is the EN 1993-1-2 (Eurocode 3) specific heat of carbon steel, whose
    peak near 735 C stands in for the transformation and magnetic heat
    effects; h_c = 1.32 (dT/D)^0.25 W m^-2 K^-1 is the simplified laminar
    natural-convection correlation for a horizontal cylinder in air; the
    emissivity of an oxidised steel surface is taken as 0.8.

Furnace cooling (full annealing)
    Newtonian cooling with the furnace: T(t) = T_inf + (T0 - T_inf) exp(-t/tau),
    with tau chosen so that the rate at 700 C equals the requested value.

Constant-rate paths are also provided for CCT diagrams.

All paths are returned on a shared descending temperature grid so that
many simulations can be stacked and run in parallel; above the austenitising
temperature the time stays at zero, which the kinetic kernel skips.
"""

from __future__ import annotations

import numpy as np

from .. import config

RHO = 7850.0           # kg/m3
SIGMA = 5.670374e-8    # W m^-2 K^-4
EPS = 0.8
T_AMB = 25.0
# multiplier on the natural-convection coefficient (1.0 = the correlation as
# published); stage 10 varies it, with EPS, in the sensitivity analysis
H_CONV_FACTOR = 1.0

T_GRID_TOP = 1000.0
T_GRID_BOTTOM = 20.0
T_GRID_STEP = 0.5
T_GRID = np.arange(T_GRID_TOP, T_GRID_BOTTOM - 1e-9, -T_GRID_STEP)


def cp_eurocode(T):
    """Specific heat of carbon steel, EN 1993-1-2 (J kg^-1 K^-1), T in C."""
    T = np.asarray(T, dtype=float)
    c = np.where(T < 600.0, 425.0 + 0.773 * T - 1.69e-3 * T ** 2 + 2.22e-6 * T ** 3,
        np.where(T < 735.0, 666.0 + 13002.0 / (738.0 - np.minimum(T, 737.0)),
        np.where(T < 900.0, 545.0 + 17820.0 / (np.maximum(T, 732.0) - 731.0), 650.0)))
    return c


def h_convection(T, D_m):
    dT = np.maximum(np.asarray(T, float) - T_AMB, 1e-6)
    return H_CONV_FACTOR * 1.32 * (dT / D_m) ** 0.25


def air_cooling_rate(T, D_mm):
    """Instantaneous still-air cooling rate (K/s) of a bar of diameter D at T."""
    D = D_mm / 1000.0
    Tk, Tak = np.asarray(T, float) + 273.15, T_AMB + 273.15
    q = h_convection(T, D) * (np.asarray(T, float) - T_AMB) + EPS * SIGMA * (Tk ** 4 - Tak ** 4)
    return (4.0 / D) * q / (RHO * cp_eurocode(T))


def _time_on_grid(rate_fn, T0, grid=T_GRID):
    """Integrate dt = dT / rate(T) down the grid starting at T0 (zero above T0)."""
    T = grid
    rate = np.maximum(rate_fn(T), 1e-9)
    t = np.zeros_like(T)
    active = T <= T0
    if not active.any():
        return t
    i0 = np.argmax(active)
    # trapezoidal integration of 1/rate in temperature
    inv = 1.0 / rate
    seg = 0.5 * (inv[i0 + 1:] + inv[i0:-1]) * (T[i0:-1] - T[i0 + 1:])
    t[i0 + 1:] = np.cumsum(seg)
    # small offset for the partial step between T0 and the first grid point
    t[i0:] += (T0 - T[i0]) * inv[i0]
    t[:i0] = 0.0
    return t


def air_cooling_path(T0, D_mm=config.AC_DEFAULT_DIAMETER_MM):
    t = _time_on_grid(lambda T: air_cooling_rate(T, D_mm), T0)
    return t, T_GRID.copy()


def furnace_cooling_path(T0, rate_700=config.FC_RATE_700C):
    tau = (700.0 - T_AMB) / rate_700

    def rate(T):
        return (np.asarray(T, float) - T_AMB) / tau

    t = _time_on_grid(rate, T0)
    return t, T_GRID.copy()


def constant_rate_path(T0, rate):
    t = _time_on_grid(lambda T: np.full_like(np.asarray(T, float), rate), T0)
    return t, T_GRID.copy()


def route_path(route, T0, D_mm=config.AC_DEFAULT_DIAMETER_MM,
               fc_rate=config.FC_RATE_700C):
    """Cooling path for a named route: 'FC' (furnace) or 'AC' (still air)."""
    if route == "FC":
        return furnace_cooling_path(T0, fc_rate)
    if route == "AC":
        return air_cooling_path(T0, D_mm)
    raise ValueError(f"unknown route {route!r}")


def rate_at(t, T, T_query=700.0):
    """Cooling rate (K/s) of a path at temperature T_query."""
    order = np.argsort(T)
    Ts, ts = np.asarray(T)[order], np.asarray(t)[order]
    dTdt = -np.gradient(ts, Ts)  # dt/dT (negative because t falls with T)
    dtdT = np.interp(T_query, Ts, np.abs(dTdt))
    return 1.0 / dtdT if dtdT > 0 else np.inf


def t85(t, T):
    """Cooling time from 800 to 500 C (s), a common weld/heat-treatment descriptor."""
    order = np.argsort(T)
    Ts, ts = np.asarray(T)[order], np.asarray(t)[order]
    return float(np.interp(500.0, Ts, ts) - np.interp(800.0, Ts, ts))
