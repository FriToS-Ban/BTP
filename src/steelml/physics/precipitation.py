"""Precipitation strengthening by microalloy carbonitrides (V, Nb).

A deliberately simple, additive estimate of the strengthening that the
microalloying elements add on top of the ferrite-pearlite relations, which do
not contain it. It is reported separately, labelled as an estimate, and
enters a prediction only for steels with more than residual V or Nb.

1. How much dissolves at the austenitising temperature: the solubility
   products in austenite (wt %, T in K), as compiled by Gladman (1997):
       VC      log10([V][C])           = 6.72 - 9500 / T    (Narita 1975)
       VN      log10([V][N])           = 3.63 - 8700 / T    (Irvine et al. 1967)
       Nb(C,N) log10([Nb][C + 12/14 N]) = 2.26 - 6770 / T    (Irvine et al. 1967)
   Each is solved with the stoichiometric mass balance. Vanadium takes the
   more stable of VN and VC. Titanium is taken as TiN that never dissolves at
   heat-treatment temperatures, so it adds no precipitation strengthening.

2. How much precipitates, and how fine: the dissolved amount re-precipitates
   in the ferrite during or after the transformation. The particle size
   depends on the route. Interphase and general precipitation during air
   cooling gives particles of about 5-10 nm. During furnace cooling the
   particles have time to coarsen to about 20-40 nm. During tempering at
   550-650 C, fine alloy carbides form; that secondary hardening belongs to
   the tempering model (physics.tempering) and is not counted twice.

3. Strengthening: Ashby-Orowan, in Gladman's form (Gladman 1999),
       d_sigma (MPa) = (5.9 sqrt(f) / x) ln(x / 2.5e-4),   x in um,
   with f the volume fraction of precipitate. f is about 1.24 x wt % V
   (as VC) x 7.87 / 5.77, the density ratio of ferrite to VC.

The particle sizes and the precipitated fraction are assumptions (ROUTE_X_NM,
PRECIPITATED). Stage 10 varies them, and the report gives the term with a
+/- 50 % band. A check against the literature steels with V or Nb is made in
stage 6.

References: Gladman, T. (1997) The Physical Metallurgy of Microalloyed
Steels, Institute of Materials; Gladman, T. (1999) Mater. Sci. Technol. 15,
30-36; Irvine, K.J., Pickering, F.B., Gladman, T. (1967) JISI 205, 161;
Narita, K. (1975) Trans. ISIJ 15, 145.
"""

from __future__ import annotations

import numpy as np

# route -> assumed mean precipitate size (nm) and fraction of the dissolved
# microalloy that precipitates in the ferrite
ROUTE_X_NM = {"AC": 8.0, "AR": 8.0, "FC": 30.0, "CR": 10.0}
PRECIPITATED = {"AC": 0.7, "AR": 0.7, "FC": 0.9, "CR": 0.7}
MICROALLOY_V, MICROALLOY_NB = 0.03, 0.01   # wt % above which a steel counts as microalloyed

A_V, A_NB, A_C, A_N = 50.94, 92.91, 12.01, 14.01


def _dissolve(M0, X0, logK, ratio):
    """Dissolved M (wt %) at equilibrium for M + X -> MX with [M][X] = K.
    ratio = A_X / A_M (wt % of X taken per wt % of M)."""
    K = 10.0 ** logK
    M0, X0 = np.asarray(M0, float), np.asarray(X0, float)
    # (M0 - p)(X0 - ratio p) = K, p = precipitated M; smallest root in [0, min(M0, X0/ratio)]
    a, b, c = ratio, -(X0 + ratio * M0), M0 * X0 - K
    disc = np.maximum(b * b - 4 * a * c, 0.0)
    p = (-b - np.sqrt(disc)) / (2 * a)
    p = np.where(M0 * X0 <= K, 0.0, np.clip(p, 0.0, np.minimum(M0, X0 / ratio)))
    return M0 - p


def dissolved_at(comp, T_aust_C):
    """Dissolved V and Nb (wt %) at the austenitising temperature."""
    g = lambda k: np.nan_to_num(np.asarray(comp.get(k, 0.0) if hasattr(comp, "get") else comp[k], float))  # noqa: E731
    T = np.asarray(T_aust_C, float) + 273.15
    V, Nb, C, N = g("V"), g("Nb"), g("C"), g("N")
    v_vn = _dissolve(V, N, 3.63 - 8700.0 / T, A_N / A_V)
    v_vc = _dissolve(V, C, 6.72 - 9500.0 / T, A_C / A_V)
    v_dis = np.minimum(v_vn, v_vc)
    nb_dis = _dissolve(Nb, C + 12.0 / 14.0 * N, 2.26 - 6770.0 / T, A_C / A_NB)
    return v_dis, nb_dis


def orowan_mpa(f, x_nm):
    x = np.maximum(np.asarray(x_nm, float), 1.0) / 1000.0
    return 5.9 * np.sqrt(np.maximum(np.asarray(f, float), 0.0)) / x * np.log(x / 2.5e-4)


def strengthening(comp, T_aust_C, route):
    """Precipitation strengthening (MPa) of V and Nb for a ferrite-pearlite
    route ('FC', 'AC', 'AR', 'CR'); zero for steels below microalloy levels."""
    v_dis, nb_dis = dissolved_at(comp, T_aust_C)
    route = np.broadcast_to(np.asarray(route).astype(str), np.shape(v_dis))
    x = np.array([ROUTE_X_NM.get(r, 10.0) for r in route.ravel()]).reshape(route.shape)
    frac = np.array([PRECIPITATED.get(r, 0.7) for r in route.ravel()]).reshape(route.shape)
    # volume fraction: VC (62.95/50.94 wt ratio, density 5.77) and NbC (104.92/92.91, density 7.8)
    f = frac * (v_dis * (62.95 / A_V) * 7.87 / 5.77 + nb_dis * (104.92 / A_NB) * 7.87 / 7.8) / 100.0
    g = lambda k: np.nan_to_num(np.asarray(comp.get(k, 0.0) if hasattr(comp, "get") else comp[k], float))  # noqa: E731
    micro = (g("V") >= MICROALLOY_V) | (g("Nb") >= MICROALLOY_NB)
    return np.where(micro, orowan_mpa(f, x), 0.0)
