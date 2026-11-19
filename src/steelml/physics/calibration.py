"""Bayesian calibration of the transformation kinetics on hardenability data.

Li et al. (1998) fitted their rate constants on a limited set of steels. We
keep their equations and their composition dependence, and let a large body
of standard data rescale the four quantities that decide what forms on
continuous cooling:

    theta = (ln f_F, ln f_P, ln f_B, dBs)

f_F, f_P and f_B multiply the ferrite, pearlite and bainite rate constants
and dBs shifts the bainite-start temperature (kinetics._kinetic_params).

Data: the SAE/ASTM Jominy H-bands of the H-steels. At every distance the
band spans the hardness of the heats that meet the grade's composition
range, so the middle of the band is compared with the prediction for the
nominal composition, with a standard deviation of a quarter of the band
width (the band taken as +/- 2 sigma) plus a model error sigma_m that is
inferred with the other parameters. Boron grades are excluded because
neither the Li model nor the hardness relations contain boron's large
effect on hardenability.

A second, joint calibration adds the measured dilatometry start
temperatures of Collins et al. (DilatometryData). What is left out of each
calibration (the dilatometry hardness, the measured phase fractions of the
literature and of the CCT atlas, the handbook hardness) is used to test
whether the calibrated kinetics transfer (pipeline/s08_hardenability.py).

Prior: the published model, ln f ~ N(0, 1) (a factor of e either way at one
standard deviation), dBs ~ N(0, 50 C), ln sigma_m ~ N(ln 3 HRC, 1).
Posterior: the maximum a posteriori point (Nelder-Mead from several starts)
with a Laplace (Gaussian) approximation around it. A random-walk Metropolis
sampler is kept for completeness; on these data it mixes poorly (the
hardness data alone leave the factors strongly correlated), which is itself
a finding: hardness cannot tell which reaction is too slow.

Result (stage 8): the calibration fits the H-bands of grades it never saw
much better than the published constants, but it does not transfer to the
independent phase-fraction and hardness data, so the production chain keeps
the published Li et al. constants.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

from .. import config
from . import jominy, kinetics
from . import properties as pp

PARAM_NAMES = ("ln_f_ferrite", "ln_f_pearlite", "ln_f_bainite", "dBs_C", "ln_sigma_model")
PRIOR_MEAN = np.array([0.0, 0.0, 0.0, 0.0, np.log(3.0)])
PRIOR_SD = np.array([1.0, 1.0, 1.0, 50.0, 1.0])


@contextmanager
def kinetic_params(theta):
    """Temporarily run the kinetic model with the factors of theta."""
    keys = ("FERRITE_RATE_FACTOR", "PEARLITE_RATE_FACTOR", "BAINITE_RATE_FACTOR", "BAINITE_START_SHIFT_C")
    saved = {k: getattr(config, k, None) for k in keys}
    vals = (np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2]), float(theta[3]))
    try:
        for k, v in zip(keys, vals):
            setattr(config, k, float(v))
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                if hasattr(config, k):
                    delattr(config, k)
            else:
                setattr(config, k, v)


def hrc_continuous(hv):
    """HRC from HV, continued linearly below HRC 20 so the likelihood stays smooth."""
    hv = np.asarray(hv, float)
    h = jominy.hrc_from_hv(hv)
    below = hv < jominy.HV_AT_HRC[0]
    slope = (jominy.HV_AT_HRC[1] - jominy.HV_AT_HRC[0]) / (jominy.HRC_POINTS[1] - jominy.HRC_POINTS[0])
    return np.where(below, jominy.HRC_POINTS[0] + (hv - jominy.HV_AT_HRC[0]) / slope, h)


class JominyData:
    """Stacked Jominy points: one row per (grade, distance), with the cooling
    path of that distance precomputed for the grade's austenitising temperature."""

    def __init__(self, points: pd.DataFrame, comps: pd.DataFrame, T_aust: pd.Series, G=7.0, h_q=2.0e4):
        """points: grade, J, HRC_min, HRC_max; comps: element columns indexed by grade;
        T_aust: austenitising temperature indexed by grade."""
        self.points = points.reset_index(drop=True)
        rows, paths, rates = [], [], []
        cache = {}
        for g, grp in self.points.groupby("grade", sort=False):
            T0 = round(float(T_aust[g]), 0)
            Js = tuple(sorted(set(self.points.J)))
            if (T0, Js) not in cache:
                ps = jominy.jominy_paths(T0, Js, h_q=h_q)
                cache[(T0, Js)] = dict(zip(Js, ps)), dict(zip(Js, jominy.rate_at_700(ps)))
            pmap, rmap = cache[(T0, Js)]
            for i, r in grp.iterrows():
                rows.append(i)
                paths.append(pmap[r.J])
                rates.append(rmap[r.J])
        order = np.array(rows)
        self.points = self.points.loc[order].reset_index(drop=True)
        self.paths = paths
        self.rate700 = np.array(rates)
        self.comps = comps.loc[self.points.grade].reset_index(drop=True)
        self.G = np.full(len(self.points), G)
        self.mid = 0.5 * (self.points.HRC_min + self.points.HRC_max).to_numpy(float)
        self.sd_band = ((self.points.HRC_max - self.points.HRC_min) / 4.0).to_numpy(float)

    def predict(self, theta):
        with kinetic_params(theta):
            kin = kinetics.simulate_batch(self.comps, self.G, self.paths)
        f_fp = kin.f_ferrite + kin.f_pearlite + kin.f_cementite_pro
        hv = pp.hv_mixture(self.comps, self.rate700 * 3600.0, f_fp, kin.f_bainite, kin.f_martensite)
        return hrc_continuous(hv), kin


def log_prior(theta):
    return float(-0.5 * np.sum(((np.asarray(theta) - PRIOR_MEAN) / PRIOR_SD) ** 2))


def log_likelihood(theta, data: JominyData, idx=None):
    pred, _ = data.predict(theta)
    sm = np.exp(theta[4])
    var = data.sd_band ** 2 + sm ** 2
    r = pred - data.mid
    if idx is not None:
        r, var = r[idx], var[idx]
    return float(-0.5 * np.sum(r ** 2 / var + np.log(2.0 * np.pi * var)))


def fit_map(data: JominyData, idx=None, x0=None, maxiter=400):
    from scipy.optimize import minimize

    x0 = PRIOR_MEAN.copy() if x0 is None else np.asarray(x0, float)
    f = lambda th: -(log_likelihood(th, data, idx) + log_prior(th))  # noqa: E731
    res = minimize(f, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-2, "adaptive": True})
    return res.x, -res.fun


def laplace_cov(data: JominyData, theta, steps=None):
    """Inverse of the numerical Hessian of the negative log posterior."""
    steps = np.array([0.05, 0.05, 0.05, 2.0, 0.05]) if steps is None else steps
    n = len(theta)
    f = lambda th: -(log_likelihood(th, data) + log_prior(th))  # noqa: E731
    f0 = f(theta)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = steps[i], steps[j]
            if i == j:
                H[i, i] = (f(theta + ei) - 2 * f0 + f(theta - ei)) / steps[i] ** 2
            else:
                H[i, j] = H[j, i] = (f(theta + ei + ej) - f(theta + ei - ej) - f(theta - ei + ej)
                                     + f(theta - ei - ej)) / (4 * steps[i] * steps[j])
    w, V = np.linalg.eigh(H)
    w = np.maximum(w, 1e-6)
    return V @ np.diag(1.0 / w) @ V.T


def metropolis(data: JominyData, theta0, cov, n=1500, seed=config.RANDOM_SEED, scale=2.38):
    rng = np.random.default_rng(seed)
    d = len(theta0)
    L = np.linalg.cholesky(cov * scale ** 2 / d + 1e-10 * np.eye(d))
    lp = lambda th: log_likelihood(th, data) + log_prior(th)  # noqa: E731
    cur, cur_lp = np.asarray(theta0, float), lp(theta0)
    chain, acc = [], 0
    for _ in range(n):
        prop = cur + L @ rng.standard_normal(d)
        p_lp = lp(prop)
        if np.log(rng.uniform()) < p_lp - cur_lp:
            cur, cur_lp = prop, p_lp
            acc += 1
        chain.append(cur.copy())
    return np.array(chain), acc / n


class DilatometryData:
    """Measured transformation-start temperatures (Collins et al. 2023) as a
    likelihood term: each measured start (ferrite, pearlite, bainite,
    martensite) against the model's, with a standard deviation SIG_T. A start
    that was measured but is not predicted counts as a 100 C miss."""

    SIG_T = 20.0
    KEYS = (("T_ferrite_start", "Tf"), ("T_pearlite_start", "Tp"), ("T_bainite_start", "Tb"),
            ("T_martensite_start", "Tm"))

    def __init__(self):
        from . import cooling
        comp = pd.read_csv(config.RAW_DIR / "collins2023_compositions.csv").set_index("steel")
        ts = pd.read_csv(config.RAW_DIR / "collins2023_Ts.csv")
        self.cases = []
        for s, c in comp.iterrows():
            for r in sorted(ts[ts.steel == s].rate):
                t, T = cooling.constant_rate_path(c.T_aust_C, r)
                m = ts[(ts.steel == s) & (ts.rate == r)].iloc[0]
                self.cases.append((c.to_dict(), float(c.astm_G), t, T, m))

    def log_likelihood(self, theta):
        ll = 0.0
        with kinetic_params(theta):
            for cd, G, t, T, m in self.cases:
                d = kinetics.simulate(cd, G, t, T)
                for key, mk in self.KEYS:
                    if pd.notna(m[mk]):
                        e = (d[key] - m[mk]) if np.isfinite(d[key]) else 100.0
                        ll += -0.5 * (e / self.SIG_T) ** 2
        return ll


def fit_map_multistart(neg_log_post, starts, maxiter=600):
    """Nelder-Mead from several starting points; the best optimum is returned."""
    from scipy.optimize import minimize
    best = None
    for x0 in starts:
        res = minimize(neg_log_post, np.asarray(x0, float), method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-2, "adaptive": True})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, -best.fun


STARTS = ([0, 0, 0, 0, np.log(3.0)], [0, 0, 1.0, -60, np.log(3.0)], [-0.8, 0, 1.4, 0, np.log(3.0)],
          [1.3, -2, 2, -40, np.log(3.0)])
