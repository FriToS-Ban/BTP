"""Property models for hardened (quenched) and quenched-and-tempered steel.

Same interface as :mod:`steelml.models.property_models`::

    m = make_qt_model(name, target).fit(F, y)    # F from features.featurize_quench
    mu, sd = m.predict(F)

Model ladder (stage 9 scores every rung with leave-one-grade-out)
-----------------------------------------------------------------
qt_physics     Literature-only chain: Maynier as-quenched hardness at the
               simulated cooling rate, the tempering master curve at its
               *prior* constants, UTS = 3.3 HV; yield ratio 0.85 when
               tempered, 0.75 untempered; elongation = training mean.
qt_gbm         Gradient boosting on composition + process variables (bar
               size, quench medium, austenitising and tempering temperature):
               the usual data-only approach, no physics.
qt_gp          Gaussian process on the same inputs.
qt_greybox     The physics chain with the tempering constants *fitted* on
               the training grades (physics.tempering.fit), a yield ratio
               that rises with tempering, and elongation from a
               strength-ductility relation with a tempering term.
qt_anchored    Grey box x exp(GP correction): a GP on composition and the
               physics descriptors learns the multiplicative error of the
               grey box. Far from the data the correction fades and the
               prediction falls back to the physics. Production candidate.
qt_ensemble    Mixture of qt_anchored and qt_gp (equal weights), interval
               from the mixture spread. The production model: the physics
               member carries the extrapolation (new bar sizes, tempering
               temperatures), the data member the grade-to-grade detail.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ..physics import tempering

QT_TARGETS = ["UTS_MPa", "YS_MPa", "El_pct"]
PROCESS = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "section_mm", "is_water", "treat_T_C", "temper_T_eff"]
RESID = ["C", "Mn", "Cr", "Ni", "Mo", "log10_rate", "f_M", "P_k", "S_soft"]


def _process_matrix(F):
    X = F.copy()
    X["is_water"] = (X["quench_medium"].astype(str).isin(["water", "brine"])).astype(float)
    # the tempering enters as its equivalent 1-h temperature, P / C_HJ - 273.15:
    # identical to the stated temperature for the 1-h training bars, and it makes
    # the data member respect Hollomon-Jaffe equivalence for any other time (a
    # long soak at a lower temperature gives what its parameter says). Untempered
    # bars count as tempered at 100 C (no softening).
    P = X["P_HJ"].to_numpy(float)
    X["temper_T_eff"] = np.where(np.isfinite(P), P / tempering.C_HJ - 273.15, 100.0)
    return X[PROCESS].to_numpy(float)


class _GPCore:
    def __init__(self, ls0=2.0, noise=0.05):
        self.ls0, self.noise = ls0, noise

    def fit(self, X, y):
        self.xs = StandardScaler().fit(X)
        self.ym, self.ys = float(np.mean(y)), float(np.std(y) + 1e-9)
        d = X.shape[1]
        k = (ConstantKernel(1.0, (1e-2, 1e2)) * Matern(np.ones(d) * self.ls0, (0.1, 100.0), nu=2.5)
             + WhiteKernel(self.noise, (1e-4, 1.0)))
        self.m = GaussianProcessRegressor(k, n_restarts_optimizer=2, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(self.xs.transform(X), (np.asarray(y) - self.ym) / self.ys)
        return self

    def predict(self, X):
        mu, sd = self.m.predict(self.xs.transform(X), return_std=True)
        return mu * self.ys + self.ym, sd * self.ys


class _Base:
    name = "base"

    def __init__(self, target, theta=None):
        self.target = target
        self.theta = theta

    def fit(self, F, y):
        raise NotImplementedError

    def predict(self, F):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# physics chain (prior or fitted constants)
# ---------------------------------------------------------------------------

def _uts_chain(F, theta):
    hv = tempering.tempered_hv(F.HV_M0, F.HV_B0, F.HV_FP0, F.f_M, F.f_B, F.f_FP, F.P_HJ, F, theta)
    return tempering.UTS_PER_HV * np.asarray(hv, float)


def _ratio_design(F):
    """Yield-ratio regressors: tempering progress of the martensite (1 - S),
    whether the bar was tempered at all, and the share of ferrite + pearlite
    (which yields early)."""
    temp = F["tempered"].to_numpy(float)
    return np.column_stack([np.ones(len(F)), temp * (1.0 - F["S_soft"].to_numpy(float)), temp,
                            F["f_FP"].to_numpy(float)])


def _el_design(F, uts):
    return np.column_stack([np.ones(len(F)), np.log(np.maximum(uts, 100.0)), F["P_k"].to_numpy(float) / 10.0,
                            F["f_FP"].to_numpy(float)])


class QTPhysics(_Base):
    name = "qt_physics"

    def fit(self, F, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, F):
        uts = _uts_chain(F, None)
        if self.target == "UTS_MPa":
            mu = uts
        elif self.target == "YS_MPa":
            mu = uts * np.where(F["tempered"].to_numpy(float) > 0, 0.85, 0.75)
        else:
            mu = np.full(len(F), self.mean_)
        return mu, np.full(len(F), np.nan)


class QTGreyBox(_Base):
    """Physics chain with constants fitted on the training rows.

    Every target model refits the tempering constants on the measured UTS of
    its training rows (they are shared by the three targets, so the fit is
    cached on the data it saw)."""
    name = "qt_greybox"
    _cache = {}

    def _theta_for(self, F, uts):
        key = (len(F), float(np.round(np.sum(uts), 3)), float(np.round(F["P_k"].sum(), 3)))
        if key not in QTGreyBox._cache:
            QTGreyBox._cache[key] = tempering.fit(F, uts)
        return QTGreyBox._cache[key]

    def fit(self, F, y):
        uts = F["UTS_MPa"].to_numpy(float)
        self.th = self.theta or self._theta_for(F, uts)
        u_hat = _uts_chain(F, self.th)
        y = np.asarray(y, float)
        if self.target == "YS_MPa":
            A = _ratio_design(F)
            self.beta, *_ = np.linalg.lstsq(A, y / uts, rcond=None)
        elif self.target == "El_pct":
            A = _el_design(F, u_hat)
            self.beta, *_ = np.linalg.lstsq(A, np.log(np.maximum(y, 0.5)), rcond=None)
        resid = np.log(y) - np.log(np.maximum(self._mu(F), 1e-6))
        self.sd_log = float(np.std(resid))
        return self

    def _mu(self, F):
        u = _uts_chain(F, self.th)
        if self.target == "UTS_MPa":
            return u
        if self.target == "YS_MPa":
            return u * np.clip(_ratio_design(F) @ self.beta, 0.4, 1.0)
        return np.exp(np.clip(_el_design(F, u) @ self.beta, -1.0, 4.5))

    def predict(self, F):
        mu = self._mu(F)
        return mu, mu * self.sd_log


class QTData(_Base):
    """Data-only models on composition + process variables."""

    def __init__(self, target, theta=None, kind="gp"):
        super().__init__(target, theta)
        self.kind = kind
        self.name = "qt_gp" if kind == "gp" else "qt_gbm"

    def fit(self, F, y):
        X = _process_matrix(F)
        y = np.asarray(y, float)
        if self.kind == "gp":
            self.m = _GPCore().fit(X, y)
        else:
            self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                   min_samples_leaf=8, l2_regularization=1.0, random_state=0)
            self.m.fit(X, y)
            self.sd_ = float(np.std(y - self.m.predict(X)))
        return self

    def predict(self, F):
        X = _process_matrix(F)
        if self.kind == "gp":
            return self.m.predict(X)
        mu = self.m.predict(X)
        return mu, np.full(len(F), max(self.sd_, 1e-6))


class QTAnchored(_Base):
    """Grey box times a GP-learned multiplicative correction (log space)."""
    name = "qt_anchored"

    def fit(self, F, y):
        self.gb = QTGreyBox(self.target, self.theta).fit(F, y)
        a = np.maximum(self.gb._mu(F), 1e-6)
        r = np.log(np.asarray(y, float)) - np.log(a)
        self.gp = _GPCore(ls0=3.0, noise=0.2).fit(F[RESID].to_numpy(float), r)
        return self

    def predict(self, F):
        a = np.maximum(self.gb._mu(F), 1e-6)
        dm, sd = self.gp.predict(F[RESID].to_numpy(float))
        mu = a * np.exp(dm)
        return mu, mu * sd


class QTEnsemble(_Base):
    name = "qt_ensemble"

    def fit(self, F, y):
        self.a = QTAnchored(self.target, self.theta).fit(F, y)
        self.g = QTData(self.target, kind="gp").fit(F, y)
        return self

    def predict(self, F):
        (m1, s1), (m2, s2) = self.a.predict(F), self.g.predict(F)
        mu = 0.5 * (m1 + m2)
        var = 0.5 * (s1 ** 2 + (m1 - mu) ** 2) + 0.5 * (s2 ** 2 + (m2 - mu) ** 2)
        return mu, np.sqrt(var)


QT_MODEL_ORDER = ["qt_physics", "qt_gbm", "qt_gp", "qt_greybox", "qt_anchored", "qt_ensemble"]
QT_PRODUCTION = "qt_ensemble"   # best rung for UTS, YS and El (stage 9)


def make_qt_model(name, target, theta=None):
    if name == "qt_physics":
        return QTPhysics(target, theta)
    if name == "qt_gbm":
        return QTData(target, theta, kind="gbm")
    if name == "qt_gp":
        return QTData(target, theta, kind="gp")
    if name == "qt_greybox":
        return QTGreyBox(target, theta)
    if name == "qt_anchored":
        return QTAnchored(target, theta)
    if name == "qt_ensemble":
        return QTEnsemble(target, theta)
    raise KeyError(name)


class QTBundle:
    """Production models for hardened and quenched-and-tempered bars (saved by
    stage 9 to results/models/qt_models.joblib).

    models : {target: fitted model}; q90 : {target: normalised conformal
    quantile}; theta : tempering constants fitted on all bars."""

    def __init__(self, models, q90, theta, meta=None):
        self.models, self.q90, self.theta, self.meta = models, q90, theta, meta or {}

    def featurize(self, df):
        from ..features import featurize_quench
        return featurize_quench(df, self.theta)

    def predict(self, df):
        """df: elements + quench_medium + section_mm (+ treat_T_C, temper_T_C,
        temper_time_h, pos_rR). Untempered rows get physics-only values."""
        F = self.featurize(df)
        out = F.copy()
        tempered = F["tempered"].to_numpy(float) > 0
        for t, m in self.models.items():
            mu, sd = m.predict(F)
            half = self.q90[t] * np.maximum(sd, 1e-6)
            out[f"{t}_pred"] = np.maximum(mu, 0.0)
            out[f"{t}_lo90"] = np.maximum(mu - half, 0.0)
            out[f"{t}_hi90"] = mu + half
        # as-quenched (untempered) martensite: the tensile test is not meaningful
        # (brittle, premature fracture) and no training bar is untempered, so only
        # the physics hardness is reported, with UTS = 3.3 HV as an indication
        hv_q = F["HV_asQ"].to_numpy(float)
        for t in ("UTS_MPa", "YS_MPa", "El_pct"):
            for s in ("pred", "lo90", "hi90"):
                col = f"{t}_{s}"
                if t == "UTS_MPa":
                    val = tempering.UTS_PER_HV * hv_q
                elif t == "YS_MPa":
                    val = 0.75 * tempering.UTS_PER_HV * hv_q
                else:
                    val = np.full(len(F), np.nan)
                out[col] = np.where(tempered, out[col], val if s == "pred" else np.nan)
        hv = np.where(tempered, out["UTS_MPa_pred"] / tempering.UTS_PER_HV, hv_q)
        out["HV_est"] = hv
        out["HB_est"] = 0.95 * hv
        from ..physics.jominy import hrc_from_hv
        out["HRC_est"] = hrc_from_hv(hv)
        from ..physics.toughness import embrittlement_flags
        fl = embrittlement_flags(F, F["temper_T_C"].to_numpy(float), F["f_M"].to_numpy(float))
        for k, v in fl.items():
            out[k] = v
        return out
