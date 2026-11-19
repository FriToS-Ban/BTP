"""Tempering of quenched steel: the Hollomon-Jaffe parameter and a
semi-empirical tempered-hardness model with a secondary-hardening term.

Tempering parameter (Hollomon & Jaffe 1945)
-------------------------------------------
    P = T_K (C_HJ + log10 t),     T_K in kelvin, t in hours, C_HJ = 20.

For one steel, equal P gives equal tempered hardness, so a time at one
temperature converts into an equivalent time at another. Stress relieving
after tempering uses this (:mod:`steelml.physics.stress_relief`). One hour at
205, 425, 540 and 650 C gives P = 9570, 13960, 16260 and 18460.

Tempered hardness (fitted in stage 9 on measured quenched-and-tempered bars)
-----------------------------------------------------------------------------
Martensite softens along a master curve in P::

    HV_M(P) = HV_soft + (HV_M0 - HV_soft) S(P) + dHV_sec(P)
    S(P)    = 1 / (1 + exp((P - P50) / W))
    P50     = p0 + p_Mn Mn + p_Si Si + p_Cr Cr + p_Mo Mo + p_V V
    HV_soft = s0 + s_C C
    dHV_sec = (a_Mo Mo + a_V V + a_Cr Cr) exp(-((P - P_SEC) / W_SEC)^2)

* HV_M0 is the as-quenched hardness of martensite at the cooling rate of the
  specimen location (Maynier et al. 1978, as in :mod:`.properties`). It
  depends mostly on carbon. As tempering proceeds the hardness falls towards
  HV_soft, the hardness of ferrite with coarse spheroidised carbide, which
  depends far less on carbon. So the carbon effect shrinks with tempering,
  as Grange, Hribal & Porter (1977) found.
* Alloying elements shift the softening to higher P (P50). Silicon delays
  the change from epsilon carbide to cementite. Cr, Mo and V form their own
  carbides, which delays the coarsening of cementite.
* Mo, V and Cr also give a bump around P_SEC: secondary hardening by fine
  alloy carbides at 500 to 650 C. Low-alloy steels with 0.2 % Mo show it only
  as a delay in softening, so its amplitudes are fitted with a prior centred
  on small values.
* Bainite starts softer than martensite. It softens only once tempered
  martensite of the same steel would be softer than it:
  HV_B(P) = min(HV_B0, HV_M(P)).
  Ferrite and pearlite do not change below Ac1.

The constants have physically sized priors (PRIOR) and are fitted by
penalised least squares on tensile strengths, through UTS = 3.3 HV, the
conversion used everywhere in this project. The fit is done inside every
cross-validation fold of stage 9. A literature-only version (the prior means)
is kept as the reference rung.

References
----------
Hollomon, J.H., Jaffe, L.D. (1945) Trans. AIME 162, 223-249.
Grange, R.A., Hribal, C.R., Porter, L.F. (1977) Metall. Trans. A 8, 1775-1785.
Maynier, P., Jungmann, B., Dollet, J. (1978) in Hardenability Concepts with
    Applications to Steel, AIME, 518-545.
Krauss, G. (2015) Steels: Processing, Structure, and Performance, 2nd ed., ASM,
    ch. 17 (tempering stages, secondary hardening).
"""

from __future__ import annotations

import numpy as np

C_HJ = 20.0
UTS_PER_HV = 3.3          # same conversion as properties.uts_from_hv
P_SEC = 17500.0           # secondary-hardening peak (about 1 h at 600 C)
W_SEC = 1500.0

# parameter: (prior mean, prior sd, lower bound, upper bound)
PRIOR = {
    "p0":    (14500.0, 2000.0, 9000.0, 20000.0),
    "p_Mn":  (300.0, 400.0, -1000.0, 2000.0),
    "p_Si":  (1000.0, 700.0, -500.0, 4000.0),
    "p_Cr":  (500.0, 500.0, -1000.0, 3000.0),
    "p_Mo":  (1500.0, 1500.0, -500.0, 10000.0),
    "p_V":   (3000.0, 2000.0, -1000.0, 10000.0),
    "logW":  (np.log(2500.0), 0.5, np.log(500.0), np.log(8000.0)),
    "s0":    (120.0, 40.0, 50.0, 250.0),
    "s_C":   (150.0, 100.0, 0.0, 600.0),
    "a_Mo":  (60.0, 80.0, 0.0, 400.0),
    "a_V":   (150.0, 200.0, 0.0, 800.0),
    "a_Cr":  (10.0, 20.0, 0.0, 100.0),
}
PARAMS = tuple(PRIOR)
PRIOR_MEAN = {k: v[0] for k, v in PRIOR.items()}


def hollomon_jaffe(T_C, t_h=1.0, C=None):
    """Tempering parameter P = T_K (C + log10 t_h); C defaults to C_HJ (read at
    call time, so the sensitivity analysis can vary it)."""
    C = C_HJ if C is None else C
    T_K = np.asarray(T_C, float) + 273.15
    t = np.maximum(np.asarray(t_h, float), 1e-6)
    return T_K * (C + np.log10(t))


def equivalent_time_h(P, T_C, C=None):
    """Time (h) at T_C that gives tempering parameter P."""
    C = C_HJ if C is None else C
    return 10.0 ** (np.asarray(P, float) / (np.asarray(T_C, float) + 273.15) - C)


def combine(P_first, T2_C, t2_h, C=None):
    """Tempering parameter after a second treatment (T2, t2) that follows one of
    parameter P_first (Hollomon-Jaffe additivity: the first treatment counts as
    its equivalent time at T2). A second treatment at a lower parameter adds
    almost nothing."""
    t_eq = equivalent_time_h(P_first, T2_C, C)
    return hollomon_jaffe(T2_C, t_eq + np.asarray(t2_h, float), C)


def _g(comp, k):
    v = comp.get(k, 0.0) if hasattr(comp, "get") else comp[k]
    return np.nan_to_num(np.asarray(v, float))


def _theta(theta):
    th = dict(PRIOR_MEAN)
    if theta is not None:
        th.update(theta)
    return th


def p50(comp, theta=None):
    th = _theta(theta)
    return (th["p0"] + th["p_Mn"] * _g(comp, "Mn") + th["p_Si"] * _g(comp, "Si")
            + th["p_Cr"] * _g(comp, "Cr") + th["p_Mo"] * _g(comp, "Mo") + th["p_V"] * _g(comp, "V"))


def softening(P, comp, theta=None):
    """S(P): 1 for untempered martensite, falling to 0 when fully softened."""
    th = _theta(theta)
    z = (np.asarray(P, float) - p50(comp, th)) / np.exp(th["logW"])
    return 1.0 / (1.0 + np.exp(np.clip(z, -50, 50)))


def soft_hardness(comp, theta=None):
    th = _theta(theta)
    return th["s0"] + th["s_C"] * _g(comp, "C")


def secondary_hardening(P, comp, theta=None):
    th = _theta(theta)
    amp = th["a_Mo"] * _g(comp, "Mo") + th["a_V"] * _g(comp, "V") + th["a_Cr"] * _g(comp, "Cr")
    return amp * np.exp(-((np.asarray(P, float) - P_SEC) / W_SEC) ** 2)


def tempered_martensite_hv(HV_M0, P, comp, theta=None):
    """HV of martensite (as-quenched hardness HV_M0) tempered to parameter P."""
    soft = soft_hardness(comp, theta)
    hv0 = np.maximum(np.asarray(HV_M0, float), soft)
    return soft + (hv0 - soft) * softening(P, comp, theta) + secondary_hardening(P, comp, theta)


def tempered_hv(HV_M0, HV_B0, HV_FP0, f_M, f_B, f_FP, P, comp, theta=None):
    """Volume-weighted HV of a quenched structure after tempering to P.

    P may be NaN for an untempered (as-quenched) state, in which case the
    as-quenched hardnesses are mixed unchanged."""
    P = np.asarray(P, float)
    hvm = np.where(np.isfinite(P), tempered_martensite_hv(HV_M0, np.nan_to_num(P, nan=0.0), comp, theta),
                   np.asarray(HV_M0, float))
    hvb = np.minimum(np.asarray(HV_B0, float), np.where(np.isfinite(P), hvm, np.inf))
    tot = np.maximum(np.asarray(f_M) + np.asarray(f_B) + np.asarray(f_FP), 1e-9)
    return (np.asarray(f_M) * hvm + np.asarray(f_B) * hvb + np.asarray(f_FP) * np.asarray(HV_FP0, float)) / tot


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def _pack(theta):
    return np.array([theta[k] for k in PARAMS], float)


def _unpack(x):
    return dict(zip(PARAMS, (float(v) for v in x)))


def fit(D, uts, sigma_rel=0.06, prior_weight=1.0, theta0=None):
    """Fit the tempered-hardness constants to measured tensile strengths.

    D : DataFrame with HV_M0, HV_B0, HV_FP0, f_M, f_B, f_FP, P_HJ (the
        Hollomon-Jaffe parameter; NaN = untempered; not "P", which is the
        phosphorus column) and the element columns; uts: measured UTS (MPa).
    Loss: sum of squared relative residuals (scale sigma_rel, about the
    scatter of typical handbook values) plus the Gaussian prior.
    Returns the fitted constants (dict).
    """
    from scipy.optimize import minimize

    uts = np.asarray(uts, float)
    cols = {k: D[k].to_numpy(float) for k in ("HV_M0", "HV_B0", "HV_FP0", "f_M", "f_B", "f_FP")}
    cols["P"] = D["P_HJ"].to_numpy(float)
    mean = np.array([PRIOR[k][0] for k in PARAMS])
    sd = np.array([PRIOR[k][1] for k in PARAMS])
    # optimise in prior-standardised coordinates z = (theta - mean) / sd: the
    # constants span four orders of magnitude, and an unscaled L-BFGS-B run
    # stalls on the bounds
    zb = [((PRIOR[k][2] - PRIOR[k][0]) / PRIOR[k][1], (PRIOR[k][3] - PRIOR[k][0]) / PRIOR[k][1]) for k in PARAMS]
    comp = {k: np.asarray(D[k], float) for k in ("Mn", "Si", "Cr", "Mo", "V", "C") if k in D}

    def loss(z):
        th = _unpack(mean + sd * z)
        hv = tempered_hv(cols["HV_M0"], cols["HV_B0"], cols["HV_FP0"], cols["f_M"], cols["f_B"], cols["f_FP"],
                         cols["P"], comp, th)
        r = (UTS_PER_HV * hv - uts) / (sigma_rel * uts)
        return 0.5 * np.sum(r ** 2) + prior_weight * 0.5 * np.sum(z ** 2)

    starts = [np.zeros(len(PARAMS))]
    if theta0 is not None:
        starts.append((_pack(_theta(theta0)) - mean) / sd)
    rng = np.random.default_rng(0)
    starts += [np.clip(rng.normal(0, 0.7, len(PARAMS)), [b[0] for b in zb], [b[1] for b in zb]) for _ in range(3)]
    best = None
    for z0 in starts:
        res = minimize(loss, z0, method="L-BFGS-B", bounds=zb, options={"maxiter": 5000})
        if best is None or res.fun < best.fun:
            best = res
    return _unpack(mean + sd * best.x)
