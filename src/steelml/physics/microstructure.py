"""Microstructural length scales derived from the transformation history.

* Prior-austenite grain size from the austenitising temperature.
* Pearlite interlamellar spacing from the pearlite undercooling (power law
  fitted on measured continuous-cooling spacings).
* Ferrite grain size from carbon equivalent, cooling rate and austenite grain
  size (Hodgson & Gibbs 1992), clipped to its calibrated range.

These are deliberately simple, cited relations. They feed the physics
baseline and serve as physically meaningful *features* for the machine-
learning property models, which learn their own coefficients; errors in the
absolute scale of a length therefore become a rescaling that the learned
model absorbs, while the monotone trends (finer structure at lower
transformation temperature) are what matter.

References
----------
ASTM E112-13, Standard Test Methods for Determining Average Grain Size.
Marder, A.R., Bramfitt, B.L. (1976) Metall. Trans. A 7, 365 (spacing versus
    undercooling in eutectoid steel), as summarised in Krauss, G. (2015)
    Steels: Processing, Structure, and Performance, 2nd ed., ASM.
Hodgson, P.D., Gibbs, R.K. (1992) ISIJ Int. 32, 1329-1338.
"""

from __future__ import annotations

import numpy as np


def astm_to_um(G):
    """ASTM E112 grain-size number -> mean lineal intercept (um)."""
    G = np.asarray(G, dtype=float)
    return 1000.0 * 10.0 ** (-(G + 3.2877) / 6.6439)


def um_to_astm(d_um):
    d_mm = np.asarray(d_um, dtype=float) / 1000.0
    return -3.2877 - 6.6439 * np.log10(d_mm)


def austenite_astm(T_aust, coarse_grained=False):
    """Prior-austenite ASTM number after a ~1 h soak at T_aust (C).

    Fine-grained (Al-killed) steels: AlN pinning keeps the grain near ASTM 8
    at 800-850 C, coarsening by about one number per 100 C above 850 C.
    Coarse-grained (non-Al-killed, e.g. the resulfurised 11xx series in the
    ASM data) steels are taken two numbers coarser. Clipped to 4-9.
    """
    T = np.asarray(T_aust, dtype=float)
    G = 8.0 - np.maximum(T - 850.0, 0.0) / 100.0
    G = np.where(coarse_grained, G - 2.0, G)
    return np.clip(G, 4.0, 9.0)


def pearlite_spacing_um(T_pearlite, Ae1, min_undercooling=5.0, A=None, m=None):
    """Interlamellar spacing S0 = A * dT^(-m) (um), dT = Ae1 - T (K).

    A and m default to config.SPACING_A_UM / SPACING_EXPONENT, fitted on
    measured continuous-cooling spacings (pipeline/s03). With m = 1 and
    A = 8.02 this is the isothermal Marder-Bramfitt relation.
    """
    from .. import config

    A = config.SPACING_A_UM if A is None else A
    m = config.SPACING_EXPONENT if m is None else m
    dT = np.maximum(np.asarray(Ae1, float) - np.asarray(T_pearlite, float), min_undercooling)
    return A * dT ** (-m)


def ferrite_grain_size_um(C, Mn, cooling_rate, d_gamma_um, rate_floor=0.1):
    """Ferrite grain size (um) from the Hodgson & Gibbs (1992) relation.

    Hodgson & Gibbs give two branches, fitted on hot-rolled C-Mn strip:
        Ceq < 0.35: d = (-0.4 + 6.37 Ceq) + (24.2 - 59.0 Ceq) Tdot^-1/2 + 22 (1 - exp(-0.015 d_gamma))
        Ceq > 0.35: d = (22.6 - 57.0 Ceq) + 3.0 Tdot^-1/2 + 22 (1 - exp(-0.015 d_gamma))
    with Ceq = C + Mn/6, Tdot the cooling rate (K/s) and d_gamma in um.

    The second branch collapses to a few micrometres (or below zero) for
    medium- and high-carbon steels at furnace and air-cooling rates; with it
    a normalised 1045 came out at 3 um and an annealed one at 4 um, against
    11 to 25 um measured in the literature compilation. We therefore use the
    low-Ceq branch for every steel and hold Ceq at its 0.35 limit for the
    richer steels (their ferrite is then treated like that of a Ceq = 0.35
    steel). Against the 20 measured grain sizes this lowers the typical error
    from a factor of 2.7 to 2.1 and leaves no steel at the 3 um floor
    (pipeline/s03, results/s03_grain_size_check.csv), with nothing fitted.
    The relation was calibrated for roughly 0.1-100 K/s, so slower rates are
    clipped to ``rate_floor`` (furnace cooling then gives the coarsest grain
    the relation supports), and the result is clipped to 3-60 um.
    """
    ceq = np.minimum(np.asarray(C, float) + np.asarray(Mn, float) / 6.0, 0.35)
    r = np.maximum(np.asarray(cooling_rate, float), rate_floor) ** -0.5
    dg = 22.0 * (1.0 - np.exp(-0.015 * np.asarray(d_gamma_um, float)))
    d = (-0.4 + 6.37 * ceq) + (24.2 - 59.0 * ceq) * r + dg
    return np.clip(d, 3.0, 60.0)


def ferrite_grain_size_um_hg_original(C, Mn, cooling_rate, d_gamma_um, rate_floor=0.1):
    """The original two-branch Hodgson & Gibbs relation (kept for the comparison in stage 3)."""
    ceq = np.asarray(C, float) + np.asarray(Mn, float) / 6.0
    r = np.maximum(np.asarray(cooling_rate, float), rate_floor) ** -0.5
    dg = 22.0 * (1.0 - np.exp(-0.015 * np.asarray(d_gamma_um, float)))
    low = (-0.4 + 6.37 * ceq) + (24.2 - 59.0 * ceq) * r + dg
    high = (22.6 - 57.0 * ceq) + 3.0 * r + dg
    return np.clip(np.where(ceq < 0.35, low, high), 3.0, 60.0)
