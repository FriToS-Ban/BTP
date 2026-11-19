"""Hardenability from composition: Grossmann's ideal critical diameter D_I.

D_I is the diameter of a bar that hardens to 50 % martensite at its centre
under an ideal quench (H = infinity). Grossmann (1942) showed that it is the
product of a base value set by carbon and the austenite grain size and one
multiplying factor per alloying element:

    D_I = D_I,base(C, G) x f_Mn x f_Si x f_Ni x f_Cr x f_Mo x f_Cu x f_V

The factors below are those of ASTM A255-10 (the tabulated factors written as
the usual piecewise formulas, including the cubic Ni factor above 1.5 % Ni),
for ASTM grain size 7, checked against a published worked example (an 8640
heat: Mn, Si, Ni, Cr and Mo factors agree to three significant figures).
D_I is in inches; it is also returned in mm. Residual Cr, Ni, Mo and Cu of
about 0.1 % each raise D_I by 20-40 %, which is why our values for plain
carbon steels sit above handbook values computed without residuals.

Why it matters here: whether air cooling a bar gives ferrite + pearlite or
bainite and martensite is a hardenability question. Plain carbon steels have
D_I of roughly 0.3 to 1.2 in. (8 to 30 mm) and form ferrite + pearlite on
air cooling in any normal section, while 4140 or 4340 reach 3 to 7 in., which
is why a 25 mm bar of them comes out bainitic or martensitic in still air.
D_I is independent of the kinetic model, so it is also a check on it
(pipeline/s08_hardenability.py).

References: Grossmann, M.A. (1942) Trans. AIME 150, 227-259; ASTM A255,
Standard Test Methods for Determining Hardenability of Steel.
"""

from __future__ import annotations

import numpy as np


def _g(comp, key):
    v = comp.get(key, 0.0) if hasattr(comp, "get") else comp[key]
    return np.nan_to_num(np.asarray(v, float))


def di_base_inch(C):
    """Base ideal diameter (in.) for ASTM grain size 7 (ASTM A255)."""
    C = np.asarray(C, float)
    return np.select(
        [C <= 0.39, C <= 0.55, C <= 0.65, C <= 0.75],
        [0.54 * C,
         0.171 + 0.001 * C + 0.265 * C ** 2,
         0.115 + 0.268 * C - 0.038 * C ** 2,
         0.143 + 0.2 * C],
        default=0.062 + 0.409 * C - 0.135 * C ** 2)


def multiplying_factors(comp):
    Mn, Si, Ni, Cr, Mo, Cu, V = (_g(comp, k) for k in ("Mn", "Si", "Ni", "Cr", "Mo", "Cu", "V"))
    return {
        "Mn": np.where(Mn <= 1.20, 1.0 + 3.3333 * Mn, 5.10 * Mn - 1.12),
        "Si": 1.0 + 0.7 * Si,
        "Ni": np.where(Ni <= 1.50, 1.0 + 0.363 * Ni, 0.3211 + 1.4501 * Ni - 0.6119 * Ni ** 2 + 0.1253 * Ni ** 3),
        "Cr": 1.0 + 2.16 * Cr,
        "Mo": 1.0 + 3.0 * Mo,
        "Cu": 1.0 + 0.365 * Cu,
        "V": 1.0 + 1.73 * V,
    }


def ideal_critical_diameter(comp):
    """D_I in inches and in mm (ASTM grain size 7)."""
    d = di_base_inch(_g(comp, "C"))
    for f in multiplying_factors(comp).values():
        d = d * f
    return d, 25.4 * d
