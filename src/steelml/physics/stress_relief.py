"""Stress relieving: what a subcritical soak does, and what it does not.

A stress relief is a soak below Ac1 (typically 550-650 C, about 1 h per
25 mm of section) followed by slow cooling. It is meant to remove residual
stresses without changing the microstructure. Three questions decide what
the model reports.

1. Is the temperature subcritical? Above Ac1 the steel partly re-austenitises
   and the treatment is no longer a stress relief. The check uses the Andrews
   (1965) Ac1 with a 25 C margin.

2. Does the hardness change? A ferrite-pearlite (annealed, normalised or
   as-rolled) structure keeps its hardness: only very long soaks spheroidise
   the pearlite, and that is not modelled. A tempered structure softens only
   if the soak tempers it further. Its tempering parameter after the soak
   follows Hollomon-Jaffe additivity (physics.tempering.combine). A soak
   below the prior tempering parameter leaves the hardness unchanged, which
   is why stress relieving is done at least 25-30 C below the tempering
   temperature.

3. How much residual stress can remain? At the soak temperature a residual
   stress cannot exceed the yield strength there, so plastic flow brings it
   down to that level at once. Creep then relaxes it further over the soak,
   which is not quantified here. The upper bound is

       sigma_res <= k_y(T) x YS(20 C)

   with k_y the effective-yield reduction factor of carbon steel at
   temperature from EN 1993-1-2 (Eurocode 3, Table 3.1). So it is an upper
   bound, and the real level after a full soak is lower.

References: EN 1993-1-2:2005 Table 3.1; Andrews, K.W. (1965) JISI 203, 721;
Hollomon & Jaffe (1945); Totten, G.E. (ed.) (2006) Steel Heat Treatment:
Metallurgy and Technologies, CRC, ch. on stress relieving.
"""

from __future__ import annotations

import numpy as np

from . import tempering, thermo

# EN 1993-1-2 Table 3.1: reduction factor k_y,theta of the effective yield strength
EC3_T = np.array([20.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0])
EC3_KY = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.78, 0.47, 0.23, 0.11, 0.06, 0.04, 0.02, 0.0])
AC1_MARGIN = 25.0


def ky(T_C):
    """Eurocode 3 reduction factor of the yield strength at T (C)."""
    return np.interp(np.asarray(T_C, float), EC3_T, EC3_KY)


def residual_stress_bound(ys_mpa, T_C):
    """Upper bound (MPa) of the residual stress left after a soak at T_C."""
    return ky(T_C) * np.asarray(ys_mpa, float)


def is_subcritical(comp, T_C, margin=AC1_MARGIN):
    return np.asarray(T_C, float) <= thermo.andrews_ac1(comp) - margin


def tempering_after_relief(P_prior, T_sr_C, t_sr_h):
    """Tempering parameter of a tempered steel after a stress relief.
    P_prior NaN (an untempered or ferrite-pearlite state) gives the parameter
    of the soak alone."""
    P_prior = np.asarray(P_prior, float)
    alone = tempering.hollomon_jaffe(T_sr_C, t_sr_h)
    comb = tempering.combine(np.nan_to_num(P_prior, nan=0.0), T_sr_C, t_sr_h)
    return np.where(np.isfinite(P_prior), np.maximum(comb, P_prior), alone)
