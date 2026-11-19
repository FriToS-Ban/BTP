"""Critical temperatures, eutectoid point and equilibrium (lever-rule) phase
fractions for carbon and low-alloy steels.

All functions are vectorised: element arguments may be floats or numpy
arrays (wt%). Temperatures are in degrees Celsius.

Equation set
------------
The kinetic model in :mod:`steelml.physics.kinetics` follows the open-source
implementation of Collins et al. (2023) of the Li et al. (1998) model. To keep
thermodynamic boundaries and kinetics consistent, the same empirical boundary
equations are used here:

* Ae3 and Ae1: Grange (1961), originally in Fahrenheit
    Ae3[F] = 1570 - 323 C - 25 Mn + 80 Si - 3 Cr - 32 Ni
    Ae1[F] = 1333 - 25 Mn + 40 Si + 42 Cr - 26 Ni
* Acm: the quadratic composition fit used by Collins et al. (2023).
* Bs: Li et al. (1998)            Bs = 637 - 58 C - 35 Mn - 15 Ni - 34 Cr - 41 Mo
* Ms: Steven & Haynes (1956) with the Si and Co terms used by Collins et al.
    Ms = 561 - 474 C - 33 Mn - 17 Ni - 17 Cr - 21 Mo - 7.5 Si + 10 Co

Andrews (1965) Ac1/Ac3 are provided for comparison only.

Limitations (documented, not hidden): these are regression formulas for
low-alloy steels (roughly C < 0.8, Mn < 2, Si < 2, Ni < 4, Cr < 2, Mo < 1).
Outside that window results are extrapolations and are flagged by
:func:`in_validity_window`.

References
----------
Grange, R.A. (1961) Metal Progress 79, 73.
Li, M.V., Niebuhr, D.V., Meekisho, L.L., Atteridge, D.G. (1998) Metall. Mater.
    Trans. B 29, 661-672.
Collins, J., Piemonte, M., Taylor, M., Fellowes, J., Pickering, E. (2023)
    Metals 13, 1168 (open-source CCT predictor, MIT licence).
Steven, W., Haynes, A.G. (1956) J. Iron Steel Inst. 183, 349-359.
Andrews, K.W. (1965) J. Iron Steel Inst. 203, 721-727.
"""

from __future__ import annotations

import numpy as np

C_FERRITE = 0.02      # wt% C in ferrite at the eutectoid temperature (~0.022)
C_CEMENTITE = 6.67    # wt% C in Fe3C

# Composition window in which the boundary equations behave. Silicon is
# limited to 1 %: Grange's +80 Si (F) on Ae3 is larger than the +16.7 Si of
# the Acm fit, so their intersection moves the eutectoid carbon *up* with
# silicon (0.74 -> 0.87 % C at 2 % Si), whereas silicon, like every common
# alloying element, lowers it (Bain; about 0.6 % C at 2 % Si). For the 2 % Si
# spring steels the eutectoid point and the lever split are therefore wrong.
# Carbon is limited to 1.05 %, below the turning point of the Acm fit.
VALIDITY_WINDOW = {"C": (0.0, 1.05), "Mn": (0.0, 2.0), "Si": (0.0, 1.0),
                   "Ni": (0.0, 4.0), "Cr": (0.0, 2.0), "Mo": (0.0, 1.0)}


def _g(comp, key):
    """Fetch an element from a dict/DataFrame/Series as float array (missing -> 0)."""
    try:
        val = comp[key]
    except (KeyError, IndexError):
        return np.asarray(0.0)
    return np.asarray(val, dtype=float)


def f_to_c(t_f):
    return (np.asarray(t_f, dtype=float) - 32.0) * 5.0 / 9.0


def c_to_f(t_c):
    return np.asarray(t_c, dtype=float) * 9.0 / 5.0 + 32.0


# ---------------------------------------------------------------------------
# Critical temperatures
# ---------------------------------------------------------------------------

def ae3(comp, C=None):
    """Ae3 (C) from Grange (1961). ``C`` overrides the bulk carbon content."""
    c = _g(comp, "C") if C is None else np.asarray(C, dtype=float)
    t_f = (1570.0 - 323.0 * c - 25.0 * _g(comp, "Mn") + 80.0 * _g(comp, "Si")
           - 3.0 * _g(comp, "Cr") - 32.0 * _g(comp, "Ni"))
    return f_to_c(t_f)


def ae1(comp):
    """Ae1 / eutectoid temperature (C) from Grange (1961)."""
    t_f = (1333.0 - 25.0 * _g(comp, "Mn") + 40.0 * _g(comp, "Si")
           + 42.0 * _g(comp, "Cr") - 26.0 * _g(comp, "Ni"))
    return f_to_c(t_f)


def acm(comp, C=None):
    """Acm (C): cementite-solvus fit used by Collins et al. (2023)."""
    c = _g(comp, "C") if C is None else np.asarray(C, dtype=float)
    cr, mn, mo, ni, si = (_g(comp, k) for k in ("Cr", "Mn", "Mo", "Ni", "Si"))
    return (224.4 + 992.4 * c - 465.1 * c ** 2 + 46.7 * cr + 19.0 * c * cr
            - 6.1 * cr ** 2 + 7.6 * mn + 10.0 * mo - 6.8 * cr * mo - 6.9 * ni
            + 3.7 * c * ni - 2.7 * cr * ni + 0.8 * ni ** 2 + 16.7 * si)


def ae3_carbon(comp, T):
    """Carbon content (wt%) of austenite on the (extrapolated) Ae3 line at T.

    Inverse of :func:`ae3` in C. Below Ae1 this is the Hultgren extrapolation
    used by Li et al. to bound ferrite formation.
    """
    t_f = c_to_f(T)
    num = (t_f - 1570.0 + 25.0 * _g(comp, "Mn") - 80.0 * _g(comp, "Si")
           + 3.0 * _g(comp, "Cr") + 32.0 * _g(comp, "Ni"))
    return num / (-323.0)


def eutectoid_point(comp):
    """Alloy-adjusted eutectoid (carbon wt%, temperature C).

    The eutectoid carbon content is the intersection of the Ae3(C) line and
    the Acm(C) curve for the given substitutional alloy content, the same
    pair of boundaries the kinetic model uses. Returns arrays (Ce, Te).
    """
    mn, si, cr, ni, mo = (_g(comp, k) for k in ("Mn", "Si", "Cr", "Ni", "Mo"))
    # Ae3(C) = a0 + a1*C   (Celsius)
    a1 = -323.0 * 5.0 / 9.0
    a0 = f_to_c(1570.0 - 25.0 * mn + 80.0 * si - 3.0 * cr - 32.0 * ni)
    # Acm(C) = b0 + b1*C + b2*C^2
    b2 = -465.1
    b1 = 992.4 + 19.0 * cr + 3.7 * ni
    b0 = (224.4 + 46.7 * cr - 6.1 * cr ** 2 + 7.6 * mn + 10.0 * mo
          - 6.8 * cr * mo - 6.9 * ni - 2.7 * cr * ni + 0.8 * ni ** 2 + 16.7 * si)
    # b2 C^2 + (b1 - a1) C + (b0 - a0) = 0 ; take the root in (0, 1.5)
    qa, qb, qc = b2, b1 - a1, b0 - a0
    disc = np.sqrt(np.maximum(qb ** 2 - 4.0 * qa * qc, 0.0))
    r1 = (-qb + disc) / (2.0 * qa)
    r2 = (-qb - disc) / (2.0 * qa)
    ce = np.where((r1 > 0) & (r1 < 1.5), r1, r2)
    te = a0 + a1 * ce
    return ce, te


# Slope of the cementite solvus (Acm) of the binary Fe-Fe3C diagram, from
# (0.76 % C, 727 C) to (2.14 % C, 1147 C) (ASM Handbook Vol. 3), about 304 C
# per % C. The Acm fit used inside the kinetic model (acm(), after Collins et
# al.) serves the Hultgren extrapolation below the eutectoid; above about
# 0.9 % C it falls below the real solvus (752 C instead of about 800 C at
# 1.0 % C) and it bends down beyond 1.07 % C, so it is not used for the
# upper critical temperature of hypereutectoid steels.
ACM_SLOPE_BINARY = (1147.0 - 727.0) / (2.14 - 0.76)


def upper_critical(comp):
    """Upper critical temperature (C) on equilibrium: Ae3 for hypoeutectoid
    steels, and above the eutectoid the cementite solvus drawn from the
    alloy-adjusted eutectoid point with the binary Fe-Fe3C slope."""
    c = np.asarray(_g(comp, "C"), float)
    ce, te = eutectoid_point(comp)
    return np.where(c <= ce, ae3(comp), te + ACM_SLOPE_BINARY * (c - ce))


# Austenitising temperature by route, following the practice of the handbook
# data the property models learn from (fitted to its annealed and normalized
# rows, MAE 14 and 15 C):
#   full annealing: max(Ae3 + 50, Ae1 + 65) C, i.e. about 50 C above Ae3 for
#       hypoeutectoid steels and 65 C above Ae1 (between A1 and Acm, as usual
#       practice for hypereutectoid steels) near and above the eutectoid;
#   normalising:   upper critical temperature + 110 C (Ae3, or Acm above the
#       eutectoid), but not below 870 C (1600 F), the lowest normalising
#       temperature of the handbook practice; always fully austenitic and
#       hotter than annealing (MAE 15 C against the handbook temperatures).
# Both are continuous in composition; an earlier piecewise version jumped by
# 66 C at the eutectoid and put a step into the carbon sweep, and without the
# floor the rule dipped to about 830 C at the eutectoid, where the handbook
# normalises at 899 C.
ANNEAL_ABOVE_AE3, ANNEAL_ABOVE_AE1 = 50.0, 65.0
NORMALISE_ABOVE_UPPER, NORMALISE_MIN_C = 110.0, 870.0
NORMALISING_ROUTES = ("AC", "AR")


def default_austenitising_T(comp, route=None):
    """Usual austenitising temperature (C) for the cooling route: furnace
    cooling (full annealing, also used for dilatometry-type routes) or air
    cooling (normalising). ``route`` may be a scalar or an array; without it
    the annealing temperature is returned."""
    anneal = np.maximum(ae3(comp) + ANNEAL_ABOVE_AE3, ae1(comp) + ANNEAL_ABOVE_AE1)
    if route is None:
        return anneal
    normalise = np.maximum(upper_critical(comp) + NORMALISE_ABOVE_UPPER, NORMALISE_MIN_C)
    return np.where(np.isin(np.asarray(route).astype(str), NORMALISING_ROUTES), normalise, anneal)


def bainite_start(comp, C=None):
    """Bs (C), Li et al. (1998)."""
    c = _g(comp, "C") if C is None else np.asarray(C, dtype=float)
    return (637.0 - 58.0 * c - 35.0 * _g(comp, "Mn") - 15.0 * _g(comp, "Ni")
            - 34.0 * _g(comp, "Cr") - 41.0 * _g(comp, "Mo"))


def martensite_start(comp, C=None):
    """Ms (C), Steven & Haynes (1956) with Si and Co terms (Collins et al. 2023)."""
    c = _g(comp, "C") if C is None else np.asarray(C, dtype=float)
    return (561.0 - 474.0 * c - 33.0 * _g(comp, "Mn") - 17.0 * _g(comp, "Ni")
            - 17.0 * _g(comp, "Cr") - 21.0 * _g(comp, "Mo") - 7.5 * _g(comp, "Si")
            + 10.0 * _g(comp, "Co"))


def andrews_ac1(comp):
    """Andrews (1965) Ac1 (heating), for comparison."""
    return (723.0 - 10.7 * _g(comp, "Mn") - 16.9 * _g(comp, "Ni") + 29.1 * _g(comp, "Si")
            + 16.9 * _g(comp, "Cr") + 6.38 * _g(comp, "W"))


def andrews_ac3(comp):
    """Andrews (1965) Ac3 (heating, hypoeutectoid), for comparison."""
    return (910.0 - 203.0 * np.sqrt(np.maximum(_g(comp, "C"), 0.0)) - 15.2 * _g(comp, "Ni")
            + 44.7 * _g(comp, "Si") + 104.0 * _g(comp, "V") + 31.5 * _g(comp, "Mo")
            + 13.1 * _g(comp, "W"))


# ---------------------------------------------------------------------------
# Equilibrium constitution
# ---------------------------------------------------------------------------

def lever_rule(comp):
    """Equilibrium (very slow cooling) constitution just below the eutectoid.

    Returns a dict of arrays with mass fractions of proeutectoid ferrite,
    pearlite and proeutectoid cementite, plus the eutectoid point used. Mass
    and volume fractions differ by < 1 % for ferrite/pearlite and are
    treated as equal.
    """
    c = _g(comp, "C")
    ce, te = eutectoid_point(comp)
    hypo = c <= ce
    f_pearlite_hypo = np.clip((c - C_FERRITE) / (ce - C_FERRITE), 0.0, 1.0)
    f_cem = np.clip((c - ce) / (C_CEMENTITE - ce), 0.0, 1.0)
    f_pearlite = np.where(hypo, f_pearlite_hypo, 1.0 - f_cem)
    f_ferrite = np.where(hypo, 1.0 - f_pearlite_hypo, 0.0)
    f_cementite = np.where(hypo, 0.0, f_cem)
    return {"f_ferrite_eq": f_ferrite, "f_pearlite_eq": f_pearlite,
            "f_cementite_eq": f_cementite, "C_eutectoid": ce, "T_eutectoid": te}


def eutectoid_class(comp, band=0.05):
    """'hypoeutectoid' / 'eutectoid' / 'hypereutectoid' relative to the
    alloy-adjusted eutectoid carbon, with a +/- ``band`` wt% tolerance."""
    c = _g(comp, "C")
    ce, _ = eutectoid_point(comp)
    out = np.where(c < ce - band, "hypoeutectoid",
                   np.where(c > ce + band, "hypereutectoid", "eutectoid"))
    return out


def carbon_equivalent_iiw(comp):
    """IIW carbon equivalent CE = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15."""
    return (_g(comp, "C") + _g(comp, "Mn") / 6.0
            + (_g(comp, "Cr") + _g(comp, "Mo") + _g(comp, "V")) / 5.0
            + (_g(comp, "Ni") + _g(comp, "Cu")) / 15.0)


def in_validity_window(comp):
    """Boolean array: composition inside the window of the empirical equations."""
    ok = np.asarray(True)
    for el, (lo, hi) in VALIDITY_WINDOW.items():
        v = _g(comp, el)
        ok = ok & (v >= lo) & (v <= hi)
    return ok
