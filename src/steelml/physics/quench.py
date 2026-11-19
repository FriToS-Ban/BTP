"""Quenching of round bars: radial heat flow and Grossmann severity of quench.

Equations
---------
Transient conduction in an infinitely long round bar of diameter D:

    rho c(T) dT/dt = (1/r) d/dr ( k(T) r dT/dr ),         rho = 7850 kg m^-3

with dT/dr = 0 at the centre and, at the surface r = R = D/2,

    -k dT/dr = h (T_s - T_q)                  (constant h, Grossmann's convention)
     T_s = T_q                                (ideal quench, H = infinity)
    -k dT/dr = h_c (T_s - T_amb) + eps sigma (T_s^4 - T_amb^4)   (still air)

The equation is discretised by an explicit finite-volume scheme on N radial
nodes (node 0 at the centre, node N-1 at the surface) with the exact
cylindrical control volumes (per radian and unit length): centre
(dr/2)^2/2, interior r_i dr, surface half-cell (R^2 - (R - dr/2)^2)/2. Face
conductivities are arithmetic means of the node values, as in
:mod:`steelml.physics.jominy`, whose EN 1993-1-2 conductivity and specific heat
(``jominy._k``, ``jominy._cp``) are reused. The time step is recomputed every
step from the stability limit of every node (the surface heat-transfer term
included) with a safety factor of 0.4.

Grossmann severity of quench
----------------------------
H = h / (2 k) in 1/inch. Tabulated ranges (Grossmann, Asimow & Urban 1939;
ASM Handbook Vol. 4) are held in ``H_TABLE``; the nominal H is the midpoint.
Converting H to a heat-transfer coefficient needs a conductivity, taken as the
constant ``K_REF`` = 30 W m^-1 K^-1 (assumption: conductivity of hot steel).

Output
------
Each position's T(t) is mapped onto the shared temperature grid of
:mod:`steelml.physics.cooling` (time at which the point first reaches each grid
temperature, 0 above the start temperature), continuing below the last
recorded point at the last rate, so the paths can go straight into
``kinetics.simulate_batch``. Positions between nodes are obtained by linear
interpolation of the crossing times between the two nearest nodes.

Assumptions
-----------
No latent-heat source except the Eurocode specific-heat peak; constant h
(Grossmann), i.e. no vapour-blanket / boiling-curve effects; long bar without
end effects; constant quenchant temperature. For an ideal quench the surface
node is held at T_q from the first step, so its own cooling rate is
meaningless (the surface path is essentially a step); use an inner position.

References
----------
M. A. Grossmann, M. Asimow, S. F. Urban (1939), "Hardenability, its relation
to quenching and some quantitative data", in Hardenability of Alloy Steels, ASM.
EN 1993-1-2 (Eurocode 3), steel thermal properties.
F. P. Incropera, D. P. DeWitt, Fundamentals of Heat and Mass Transfer
(one-term/multi-term series solution for the long cylinder).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from . import cooling
from .jominy import _cp, _k, _q_air

RHO = cooling.RHO
K_REF = 30.0   # W m^-1 K^-1, assumed conductivity of hot steel for H <-> h
INCH = 0.0254

# (medium, agitation) -> (H_low, H_high) in 1/inch
H_TABLE = {
    ("oil", "none"): (0.25, 0.30), ("oil", "mild"): (0.30, 0.35),
    ("oil", "moderate"): (0.35, 0.40), ("oil", "good"): (0.40, 0.50),
    ("oil", "strong"): (0.50, 0.80), ("oil", "violent"): (0.80, 1.10),
    ("water", "none"): (0.90, 1.00), ("water", "mild"): (1.00, 1.10),
    ("water", "moderate"): (1.20, 1.30), ("water", "good"): (1.40, 1.50),
    ("water", "strong"): (1.60, 2.00), ("water", "violent"): (4.0, 4.0),
    ("brine", "none"): (2.0, 2.0), ("brine", "mild"): (2.0, 2.2),
    ("brine", "violent"): (5.0, 5.0),
}
T_QUENCHANT = {"water": 25.0, "brine": 25.0, "oil": 60.0, "air": cooling.T_AMB}

_MODE_H, _MODE_IDEAL, _MODE_AIR = 0, 1, 2


def h_from_H(H):
    """Heat-transfer coefficient (W m^-2 K^-1) from Grossmann H (1/in)."""
    return 2.0 * K_REF * np.asarray(H, float) / INCH


def H_from_h(h):
    """Grossmann H (1/in) from heat-transfer coefficient (W m^-2 K^-1)."""
    return np.asarray(h, float) * INCH / (2.0 * K_REF)


def severity(medium, agitation="mild"):
    """Grossmann severity of quench for a medium and agitation level."""
    try:
        lo, hi = H_TABLE[(medium, agitation)]
    except KeyError:
        raise KeyError(f"no Grossmann H for ({medium!r}, {agitation!r}); "
                       f"available: {sorted(H_TABLE)}") from None
    nom = 0.5 * (lo + hi)
    return dict(H_low=lo, H_high=hi, H_nom=nom, h_nom=float(h_from_H(nom)))


@njit(cache=True)
def _kk(T, const_k):
    return const_k if const_k > 0.0 else _k(T)


SMOOTH_CP_LIQUID = True    # liquid quenches use _cp_smooth (stage 10 tests the Eurocode peak instead)
CP_AUSTENITE_CAP = 650.0   # J kg^-1 K^-1, specific heat of austenite (no transformation peak)


@njit(cache=True)
def _cp_smooth(T):
    """Specific heat without the EN 1993-1-2 peak at 735 C: the Eurocode
    polynomial, capped at the austenite value (650 J/kg/K). The Eurocode peak
    stands for the heat of the ferrite + pearlite reaction and the magnetic
    change during *slow* cooling; in a quench the austenite is undercooled far
    below 735 C without transforming, so the peak would wrongly hold the
    surface near 700-750 C while the core cools through it (it made the
    surface of an oil-quenched 50 mm bar less martensitic than its centre)."""
    if T >= 600.0:
        return CP_AUSTENITE_CAP
    c = 425.0 + 0.773 * T - 1.69e-3 * T * T + 2.22e-6 * T * T * T
    return c if c < CP_AUSTENITE_CAP else CP_AUSTENITE_CAP


@njit(cache=True)
def _cc(T, const_cp):
    """const_cp > 0: constant; const_cp < 0: smooth (quench); 0: Eurocode with peak."""
    if const_cp > 0.0:
        return const_cp
    if const_cp < 0.0:
        return _cp_smooth(T)
    return _cp(T)


@njit(cache=True)
def _solve(T0, h, T_q, mode, D, N, grid, T_stop, t_end, const_k, const_cp, T_amb, eps):
    """Radial explicit finite-volume solver; returns crossing times (N, len(grid)).

    const_k / const_cp > 0 replace the temperature-dependent properties (used
    to verify the scheme against the analytic series solution).
    """
    R = 0.5 * D
    dr = R / (N - 1)
    T = np.full(N, T0)
    Tn = np.empty(N)
    kn = np.empty(N)
    cn = np.empty(N)
    G = np.empty(N - 1)
    V = np.empty(N)
    V[0] = 0.5 * (0.5 * dr) ** 2
    for i in range(1, N - 1):
        V[i] = i * dr * dr
    V[N - 1] = 0.5 * (R * R - (R - 0.5 * dr) ** 2)
    g = grid.shape[0]
    times = np.zeros((N, g))
    p0 = 0
    while p0 < g and grid[p0] > T0:
        p0 += 1
    ptr = np.full(N, p0, np.int64)
    ideal = mode == 1
    t = 0.0
    while t < t_end:
        for i in range(N):
            kn[i] = _kk(T[i], const_k)
            cn[i] = RHO * _cc(T[i], const_cp)
        dt = 1e30
        for i in range(N - 1):
            G[i] = 0.5 * (kn[i] + kn[i + 1]) * (i + 0.5) * dr / dr
        for i in range(N):
            s = 0.0
            if i > 0:
                s += G[i - 1]
            if i < N - 1:
                s += G[i]
            if i == N - 1:
                if ideal:
                    continue
                if mode == 0:
                    hs = h
                else:
                    hs = _q_air(T[i], D, T_amb, eps) / max(T[i] - T_amb, 1e-6)
                s += hs * R
            lim = 0.4 * cn[i] * V[i] / s
            if lim < dt:
                dt = lim
        if t + dt > t_end:
            dt = t_end - t
        for i in range(N):
            if i == N - 1 and ideal:
                Tn[i] = T_q
                continue
            q = 0.0
            if i > 0:
                q -= G[i - 1] * (T[i] - T[i - 1])
            if i < N - 1:
                q += G[i] * (T[i + 1] - T[i])
            if i == N - 1:
                if mode == 0:
                    q -= R * h * (T[i] - T_q)
                else:
                    q -= R * _q_air(T[i], D, T_amb, eps)
            Tn[i] = T[i] + dt * q / (cn[i] * V[i])
        for i in range(N):
            while ptr[i] < g and Tn[i] <= grid[ptr[i]]:
                frac = (T[i] - grid[ptr[i]]) / max(T[i] - Tn[i], 1e-12)
                times[i, ptr[i]] = t + frac * dt
                ptr[i] += 1
        for i in range(N):
            T[i] = Tn[i]
        t += dt
        if Tn[0] <= T_stop:
            break
    # below the last recorded grid point continue at the last cooling rate
    for j in range(N):
        p = ptr[j]
        if p >= g:
            continue
        if p >= p0 + 2:
            rate = (grid[p - 2] - grid[p - 1]) / max(times[j, p - 1] - times[j, p - 2], 1e-9)
        else:
            rate = 1.0
        rate = max(rate, 1e-3)
        base = times[j, p - 1] if p >= 1 else t
        gbase = grid[p - 1] if p >= 1 else T0
        for q in range(p, g):
            times[j, q] = base + (gbase - grid[q]) / rate
    return times


def quench_paths(T0, D_mm, H=None, h=None, medium=None, agitation="mild",
                 positions=(0.0, 0.5, 1.0), T_q=None, n_nodes=41, T_stop=100.0, t_end=None,
                 smooth_cp=None):
    """Cooling paths (t on the shared grid, T_GRID) at r/R = positions.

    Boundary: ``h`` (W m^-2 K^-1) if given, else ``H`` (1/in; np.inf = ideal
    quench), else the nominal Grossmann H of ``medium``/``agitation``. For
    ``medium='air'`` (and no H/h) the still-air boundary is used.

    ``smooth_cp`` (default: True for liquid quenches, False for still air)
    removes the Eurocode transformation peak from the specific heat, see
    :func:`_cp_smooth`; still air keeps it, so that the air-cooling paths agree
    with :mod:`steelml.physics.cooling`.
    """
    mode, hval = _MODE_H, 0.0
    if h is not None:
        hval = float(h)
    elif H is not None:
        if np.isinf(H):
            mode = _MODE_IDEAL
        else:
            hval = float(h_from_H(H))
    elif medium == "air":
        mode = _MODE_AIR
    elif medium is not None:
        hval = severity(medium, agitation)["h_nom"]
    else:
        raise ValueError("give H, h or medium")
    if T_q is None:
        T_q = T_QUENCHANT.get(medium, 25.0) if medium != "air" else cooling.T_AMB
    if mode == _MODE_AIR:
        T_q = cooling.T_AMB
    D = D_mm * 1e-3
    if t_end is None:
        t_end = 3.0e5
    grid = cooling.T_GRID.astype(float)
    if smooth_cp is None:
        smooth_cp = (mode != _MODE_AIR) and SMOOTH_CP_LIQUID
    times = _solve(float(T0), hval, float(T_q), mode, D, int(n_nodes), grid, float(T_stop),
                   float(t_end), 0.0, -1.0 if smooth_cp else 0.0, cooling.T_AMB, cooling.EPS)
    out = []
    for p in positions:
        x = float(np.clip(p, 0.0, 1.0)) * (n_nodes - 1)
        i = min(int(np.floor(x)), n_nodes - 2)
        w = x - i
        out.append(((1.0 - w) * times[i] + w * times[i + 1], grid.copy()))
    return out


def rate_at_700(paths):
    """Cooling rate (K/s) at 700 C of each path."""
    return np.array([cooling.rate_at(t, T, 700.0) for t, T in paths])
