"""Property models: from physics baseline to physics-informed ML.

All models share one interface::

    m = make_model(name, target)
    m.fit(F, y)              # F = featurised DataFrame (steelml.features.featurize)
    mu, sd = m.predict(F)    # sd is NaN for models without a native uncertainty

Model ladder (what each rung adds)
----------------------------------
physics_lit    Literature equations only (Gladman-McIvor-Pickering, Maynier).
ridge_comp     Linear ML on composition + cooling descriptors.
gbm_comp       Gradient boosting on composition + cooling descriptors with
               metallurgically signed monotone constraints.
gp_comp        Gaussian process (ARD Matern) on composition + cooling.
gp_phys        Gaussian process on physics descriptors from the
               transformation model (phase fractions, spacing, grain size,
               transformation temperature, superheat).
greybox        Physics-structured regression: the Gladman mixture form
               (f_alpha^1/3 weighting, Hall-Petch and spacing terms, hard-
               phase and solid-solution terms) with coefficients *learned*
               under sign constraints.
greybox_gp     greybox + GP on its residuals (Delta-learning): the physics
               form carries the extrapolation, the GP learns what it misses.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ["UTS_MPa", "YS_MPa", "El_pct", "HB"]
TARGET_LABELS = {"UTS_MPa": "UTS (MPa)", "YS_MPa": "Yield strength (MPa)",
                 "El_pct": "Total elongation (%)", "HB": "Brinell hardness (HB)"}

# No data-driven input contains silicon or vanadium. Every training grade has
# the same nominal Si apart from the 2 % Si spring steel 9255 - 0.225 % for the
# carbon grades and 0.25 % for the alloy H-band grades, an artefact of the two
# specification sources - and only 6150 carries V. With Si as an input the
# models learned that 0.025 % difference as an "alloy grade" label: they looked
# better in leave-one-grade-out, but a real heat with 0.30 % Si was predicted
# 100-150 MPa stronger than one with 0.225 % (results/s04_silicon_ablation.csv;
# the silicon sweep of models.consistency measures this). The real effects of Si
# and V enter through the physics: the Maynier hardness anchor, the
# solid-solution index and the transformation kinetics.
NO_DATA_ELEMENTS = ("Si", "V")
COMP = ["C", "Mn", "Cr", "Ni", "Mo", "S"]
ROUTE = ["log10_rate", "is_AR", "treat_T_eff"]
PHYS = ["ferrite_term", "f_hard", "pearlite_term", "inv_sqrt_d_mm", "T_trans_mean",
        "superheat_A1", "ss_index", "C", "log10_rate"]

# expected sign of the effect of each raw input on strength (+1) / ductility (-1)
STRENGTH_SIGNS = {"C": 1, "Mn": 1, "Si": 1, "Cr": 1, "Ni": 0, "Mo": 1, "V": 1, "S": 0,
                  "log10_rate": 1, "is_AR": 0, "treat_T_eff": 0}


def add_route_columns(F):
    F = F.copy()
    F["is_AR"] = (F["route"] == "AR").astype(float)
    F["treat_T_eff"] = np.where(F["route"] == "AR", 950.0, F.get("treat_T_C", np.nan))
    return F


# ---------------------------------------------------------------------------
# base classes
# ---------------------------------------------------------------------------

class _Base:
    name = "base"

    def __init__(self, target):
        self.target = target

    def fit(self, F, y):
        raise NotImplementedError

    def predict(self, F):
        raise NotImplementedError


class PhysicsLit(_Base):
    name = "physics_lit"
    col = {"UTS_MPa": "UTS_phys", "YS_MPa": "YS_phys", "HB": "HB_phys"}

    def fit(self, F, y):
        self.mean_ = float(np.mean(y))  # elongation has no literature equation
        return self

    def predict(self, F):
        if self.target in self.col:
            mu = F[self.col[self.target]].to_numpy(float)
        else:
            mu = np.full(len(F), self.mean_)
        return mu, np.full(len(F), np.nan)


class _SK(_Base):
    features: list = []

    def _X(self, F):
        return add_route_columns(F)[self.features].to_numpy(float)


class RidgeComp(_SK):
    name = "ridge_comp"
    features = COMP + ROUTE

    def fit(self, F, y):
        X = self._X(F)
        # route-carbon interaction lets the pearlite effect differ by route
        X = np.column_stack([X, X[:, 0] * X[:, len(COMP)]])
        self.m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
        self.m.fit(X, y)
        return self

    def predict(self, F):
        X = self._X(F)
        X = np.column_stack([X, X[:, 0] * X[:, len(COMP)]])
        return self.m.predict(X), np.full(len(F), np.nan)


class GBMComp(_SK):
    name = "gbm_comp"
    features = COMP + ROUTE

    def fit(self, F, y):
        from xgboost import XGBRegressor
        sign = 1 if self.target != "El_pct" else -1
        mono = tuple(int(sign * STRENGTH_SIGNS.get(f, 0)) for f in self.features)
        self.m = XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
                              monotone_constraints=mono, random_state=0, n_jobs=1)
        self.m.fit(self._X(F), y)
        return self

    def predict(self, F):
        return self.m.predict(self._X(F)), np.full(len(F), np.nan)


class _GP(_SK):
    def fit(self, F, y):
        X = self._X(F)
        self.xs = StandardScaler().fit(X)
        self.ym, self.ys = float(np.mean(y)), float(np.std(y) + 1e-9)
        d = X.shape[1]
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * Matern(length_scale=np.ones(d) * 2.0, length_scale_bounds=(0.1, 100.0), nu=2.5)
                  + WhiteKernel(0.05, (1e-4, 1.0)))
        self.m = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=3,
                                          random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(self.xs.transform(X), (np.asarray(y) - self.ym) / self.ys)
        return self

    def predict(self, F):
        mu, sd = self.m.predict(self.xs.transform(self._X(F)), return_std=True)
        return mu * self.ys + self.ym, sd * self.ys


class GPComp(_GP):
    name = "gp_comp"
    features = COMP + ROUTE


class GPPhys(_GP):
    name = "gp_phys"
    features = PHYS + ["Mn", "Cr", "Mo", "Ni"]


class GPHybrid(_GP):
    """GP on composition plus the few physics descriptors that carry the
    cooling-rate dependence: the hard-phase share, the transformation
    temperature, and the three Gladman terms (ferrite share, ferrite grain
    size, pearlite fineness, each weighted by the ferrite + pearlite share so
    that they vanish smoothly in bainitic or martensitic structures)."""
    name = "gp_hybrid"
    features = ["C", "Mn", "Cr", "Ni", "Mo", "log10_rate", "is_AR",
                "f_hard", "ferrite_term", "ferrite_hp_term", "T_trans_mean", "pearlite_term", "superheat_A1"]


class PhysLitGP(_Base):
    """Delta-learning on the *literature* physics: GP on the residual of the
    textbook equations (the classical way of 'correcting' physics with ML)."""
    name = "physlit_gp"
    resid_features = ["C", "Mn", "Cr", "Mo", "Ni", "log10_rate", "f_hard", "superheat_A1", "T_trans_mean"]

    def fit(self, F, y):
        self.base = PhysicsLit(self.target).fit(F, y)
        r = np.asarray(y, float) - self.base.predict(F)[0]
        self.gp = _ResidGP(self.target, self.resid_features).fit(F, r)
        return self

    def predict(self, F):
        mu0, _ = self.base.predict(F)
        dm, sd = self.gp.predict(F)
        return mu0 + dm, sd


# ---------------------------------------------------------------------------
# grey box
# ---------------------------------------------------------------------------

INF = np.inf
# Coefficient bounds (lower, upper) in the physical units of each regressor.
# Strength-like targets follow the Gladman mixture law. Where the training
# data cannot identify a parameter (only one grade, 9255, carries more than
# 0.25 % Si) the bound carries the literature value: the solid-solution
# regressor is Pickering's coefficient sum, so its multiplier is held to
# 0.5-1.5 x the literature strengthening. HB bounds are the UTS bounds / 3.4.
_SS_UTS = (0.5, 1.5)
GREYBOX_BOUNDS = {
    "strength": {"ferrite_matrix": (0, INF), "ferrite_hall_petch": (0, INF),
                 "pearlite_base": (0, INF), "pearlite_spacing": (0, INF),
                 "hard_phase_hv": (0, INF), "solid_solution": _SS_UTS,
                 "spheroidisation_risk": (-INF, 0)},
    "ductility": {"ferrite_matrix": (0, INF), "ferrite_hall_petch": (-INF, INF),
                  "pearlite_base": (0, INF), "pearlite_spacing": (-INF, INF),
                  "hard_phase_hv": (-INF, 0), "hard_phase_share": (0, INF),
                  "spheroidisation_risk": (0, INF), "sulphur": (-INF, 0)},
}


def greybox_terms(F, kind="strength"):
    """Physics-structured regressors of a Gladman-type mixture law.

    strength ~ f_FP [ fa3 (s_f + k_d d^-1/2) + (1 - fa3)(s_p + k_s S0^-1/2) ]
               + k_h (f_B HV_B + f_M HV_M)/100 + k_ss SS - k_sph SPH

    fa3 = f_alpha^(1/3) (Gladman et al. 1972). The phase-share terms sum to
    one, so each phase's base strength is learned directly and there is no
    free intercept. HV_B and HV_M are Maynier's hardnesses of bainite and
    martensite at the actual cooling rate. SS is Pickering's ferrite solid-
    solution sum (MPa). SPH is the pearlite share of slowly cooled steels
    weighted by exp(-superheat_A1 / 60 K): large when the steel was annealed
    only just above A1, where a divorced (spheroidised) eutectoid forms.
    Ductility uses the same phase shares with ductility-appropriate bounds.
    """
    from ..physics.properties import hv_maynier

    F = add_route_columns(F)
    f_fp = F["f_fp"].to_numpy(float)
    fa3 = F["fa_third"].to_numpy(float)
    d = F["inv_sqrt_d_mm"].to_numpy(float)
    s = F["inv_sqrt_S0_mm"].to_numpy(float)
    fB = F["k_f_bainite"].to_numpy(float)
    fM = F["k_f_martensite"].to_numpy(float)
    hv = hv_maynier(F, F["cool_rate_700"].to_numpy(float) * 3600.0)
    hard_hv = (fB * hv["HV_B"] + fM * hv["HV_M"]) / 100.0
    slow = (F["log10_rate"].to_numpy(float) < -1).astype(float)
    sph = f_fp * (1 - fa3) * np.exp(-np.maximum(F["superheat_A1"].to_numpy(float), 0.0) / 60.0) * slow
    terms = {
        "ferrite_matrix": f_fp * fa3,
        "ferrite_hall_petch": f_fp * fa3 * d,
        "pearlite_base": f_fp * (1 - fa3),
        "pearlite_spacing": f_fp * (1 - fa3) * s,
        "hard_phase_hv": hard_hv,
        "solid_solution": F["ss_index"].to_numpy(float),
        "spheroidisation_risk": sph,
    }
    if kind == "ductility":
        terms.pop("solid_solution")
        terms["hard_phase_share"] = fB + fM
        terms["sulphur"] = F["S"].to_numpy(float)
    names = list(terms)
    return np.column_stack([terms[n] for n in names]), names


class GreyBox(_Base):
    name = "greybox"

    def _kind(self):
        return "ductility" if self.target == "El_pct" else "strength"

    def fit(self, F, y):
        kind = self._kind()
        X, self.names = greybox_terms(F, kind)
        scale = 1.0 / 3.4 if self.target == "HB" else 1.0
        bnd = GREYBOX_BOUNDS[kind]
        lb = np.array([bnd[n][0] * (scale if n == "solid_solution" else 1.0) for n in self.names])
        ub = np.array([bnd[n][1] * (scale if n == "solid_solution" else 1.0) for n in self.names])
        self.sc = np.maximum(np.abs(X).max(axis=0), 1e-9)
        Xs = X / self.sc
        res = lsq_linear(Xs, np.asarray(y, float), bounds=(lb * self.sc, ub * self.sc), lsmr_tol="auto")
        self.coef_ = res.x
        return self

    def coefficients(self):
        """Coefficients in the physical units of each regressor."""
        return pd.Series(self.coef_ / self.sc, index=self.names)

    def predict(self, F):
        X, _ = greybox_terms(F, self._kind())
        return (X / self.sc) @ self.coef_, np.full(len(F), np.nan)


class GreyBoxGP(_Base):
    name = "greybox_gp"
    resid_features = ["C", "Mn", "Cr", "Mo", "Ni", "log10_rate", "f_hard", "superheat_A1", "T_trans_mean"]

    def fit(self, F, y):
        self.gb = GreyBox(self.target).fit(F, y)
        r = np.asarray(y, float) - self.gb.predict(F)[0]
        self.gp = _ResidGP(self.target, self.resid_features).fit(F, r)
        return self

    def predict(self, F):
        mu0, _ = self.gb.predict(F)
        dm, sd = self.gp.predict(F)
        return mu0 + dm, sd


class _ResidGP(_GP):
    def __init__(self, target, features):
        super().__init__(target)
        self.features = features

    def fit(self, F, y):
        X = self._X(F)
        self.xs = StandardScaler().fit(X)
        self.ym, self.ys = 0.0, float(np.std(y) + 1e-9)
        d = X.shape[1]
        kernel = (ConstantKernel(0.5, (1e-3, 1e1))
                  * Matern(length_scale=np.ones(d) * 3.0, length_scale_bounds=(0.3, 100.0), nu=2.5)
                  + WhiteKernel(0.2, (1e-3, 1.0)))
        self.m = GaussianProcessRegressor(kernel, n_restarts_optimizer=3, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(self.xs.transform(X), np.asarray(y) / self.ys)
        return self


# Physics quantity the anchored models scale. "UTS_phys" is microstructure-
# based: the Gladman-McIvor-Pickering mixture law on the predicted ferrite
# fraction, pearlite spacing and ferrite grain size for the ferrite + pearlite
# part, and the Maynier hardness (x 3.3) for bainite and martensite.
# "HB_phys" (the first version) is the Maynier hardness for every
# constituent, whose ferrite-pearlite term depends on composition and cooling
# rate only and so cannot see pearlite refinement. Compared in stage 4
# (results/s04_anchor_ablation.csv).
ANCHOR_COLUMN = "UTS_phys"


class AnchoredGP(_Base):
    """Physics-anchored GP (the project's main model).

    Every target is anchored on the strength UTS_phys that the literature
    relations assign to the microstructure predicted by the transformation
    model: the Gladman-McIvor-Pickering mixture law (ferrite fraction, ferrite
    grain size, pearlite spacing) for the ferrite + pearlite part and the
    Maynier hardness (UTS ~ 3.3 HV) for bainite and martensite, mixed by
    volume fraction (properties.physics_baseline):

        UTS ~ c * UTS_phys         HB ~ c * UTS_phys       (UTS/HB ~ 3.4)
        YS  ~ c * UTS_phys         (yield ratio ~ 0.63)
        El  ~ c / UTS_phys         (El x UTS ~ constant: strength-ductility trade-off)

    (ANCHOR_COLUMN; the first version used the Maynier hardness for every
    constituent, whose ferrite-pearlite term cannot see pearlite refinement;
    see results/s04_anchor_ablation.csv.)

    The single constant c is refitted on every training fold (least squares
    in log space), and a GP learns a *multiplicative* correction
    log(y / anchor) from composition and physics descriptors. Far from the
    training data the GP correction decays to zero, so the prediction falls
    back to physics instead of to the training mean - the property that
    makes the model safe to query at cooling rates absent from the data.
    Just *outside* the data (one element slightly out of range) the
    correction is still active and can carry a neighbouring grade's
    correction over, which is why GatedEnsemble keeps its data-driven member
    for new compositions.
    """
    name = "anchored_gp"
    resid_features = ["C", "Mn", "Cr", "Ni", "Mo", "log10_rate", "f_hard", "ferrite_term", "ferrite_hp_term",
                      "pearlite_term", "T_trans_mean", "superheat_A1"]

    def _anchor(self, F):
        a = np.maximum(F[ANCHOR_COLUMN].to_numpy(float), 50.0)
        return 1.0 / a if self.target == "El_pct" else a

    def fit(self, F, y):
        a = self._anchor(F)
        ly = np.log(np.asarray(y, float))
        self.logc = float(np.mean(ly - np.log(a)))
        r = ly - np.log(a) - self.logc
        self.gp = _ResidGP(self.target, self.resid_features).fit(F, r)
        return self

    def predict(self, F):
        a = self._anchor(F)
        dm, sd = self.gp.predict(F)
        mu = np.exp(self.logc + np.log(a) + dm)
        return mu, mu * sd


class _RatioGP(_GP):
    """GP for the dimensionless yield ratio YS/UTS on microstructure descriptors."""
    features = ["C", "Mn", "Cr", "Ni", "Mo", "log10_rate", "is_AR", "f_hard",
                "ferrite_term", "ferrite_hp_term", "T_trans_mean", "pearlite_term", "superheat_A1"]


class AnchoredChain(_Base):
    """Physics-anchored property chain.

    UTS, HB and El come from :class:`AnchoredGP`. Yield strength is obtained
    as YS = UTS x (YS/UTS), with the yield ratio learned by a GP on
    microstructure descriptors: the ratio is bounded (0.3-1.0) and depends on
    microstructure type (spheroidised < ferrite-pearlite < bainite), so it
    is a safer quantity to learn and extrapolate than YS itself.
    """
    name = "anchored_chain"

    def fit(self, F, y):
        if self.target != "YS_MPa":
            self.m = AnchoredGP(self.target).fit(F, y)
            return self
        uts = F["UTS_MPa"].to_numpy(float)
        self.uts_m = AnchoredGP("UTS_MPa").fit(F, uts)
        self.ratio_m = _RatioGP("ratio").fit(F, np.asarray(y, float) / uts)
        return self

    def predict(self, F):
        if self.target != "YS_MPa":
            return self.m.predict(F)
        u, su = self.uts_m.predict(F)
        r, sr = self.ratio_m.predict(F)
        r = np.clip(r, 0.3, 1.0)
        mu = u * r
        sd = mu * np.sqrt((su / np.maximum(u, 1e-9)) ** 2 + (sr / np.maximum(r, 1e-9)) ** 2)
        return mu, sd


class GatedEnsemble(_Base):
    """Gated ensemble: the project's production predictor.

    The prediction is the average of three partly independent routes from
    microstructure to properties: two physics-anchored ones - the
    hardness-anchored chain (AnchoredChain) and the literature equations with a
    learned GP residual (PhysLitGP) - and the composition+physics GP
    (gp_hybrid). When the cooling produces a microstructure the data do not
    contain - more martensite, or more bainite+martensite, than any training
    steel - the data-driven member's weight moves smoothly to the two
    physics-anchored members, because a GP cannot predict values beyond its
    training range and the physics anchors can. The interval is the spread of
    the mixture, so it widens where the members disagree.

    Two broader gates were tried first and dropped on evidence; stage 4
    re-scores them in results/s04_gate_ablation.csv:

    * handing over for any composition outside the applicability domain: on
      the six leave-one-grade-out grades outside it, physics-anchoring alone
      was worse than the blend for UTS and hardness, because a learned
      correction does not fade when only one element is out of range. Such
      steels are still flagged (``in_domain``);
    * handing over above the fastest training cooling rate: switching
      members made strength *fall* with faster cooling of ferrite-pearlite
      bars (more monotonicity violations in the rate sweep), while the
      dilatometry hardness test did not need it. For a ferrite-pearlite steel
      a faster cool only refines the structure, which the GP's physics
      descriptors (Gladman terms, transformation temperature) already carry.
    """
    name = "gated_ensemble"
    MEMBERS = ("anchored_chain", "physlit_gp", "gp_hybrid")   # the last one is data-driven
    DATA_WEIGHT = 1.0 / 3.0   # weight of the data-driven member inside the data

    def fit(self, F, y):
        from .domain import Domain
        self.a = AnchoredChain(self.target).fit(F, y)
        self.p = PhysLitGP(self.target).fit(F, y)
        self.h = GPHybrid(self.target).fit(F, y)
        self.dom = Domain().fit(F)
        self.max_hard = float(F["f_hard"].max())
        self.max_mart = float(F["k_f_martensite"].max())
        return self

    def weight_data_model(self, F):
        """Weight of the data-driven member (0 = physics-anchored members only)."""
        excess_hard = np.clip((F["f_hard"].to_numpy(float) - self.max_hard) / 0.2, 0.0, 1.0)
        excess_mart = np.clip((F["k_f_martensite"].to_numpy(float) - self.max_mart) / 0.2, 0.0, 1.0)
        return self.DATA_WEIGHT * (1.0 - np.maximum(excess_hard, excess_mart))

    def blend(self, F, w_data):
        """Mixture of the three members for a given data-member weight."""
        members = [self.a.predict(F), self.p.predict(F), self.h.predict(F)]
        w_data = np.asarray(w_data, float)
        weights = [(1.0 - w_data) / 2.0, (1.0 - w_data) / 2.0, w_data]
        mu = sum(w * m for w, (m, _) in zip(weights, members))
        var = sum(w * (s ** 2 + (m - mu) ** 2) for w, (m, s) in zip(weights, members))
        return mu, np.sqrt(var)

    def predict(self, F):
        return self.blend(F, self.weight_data_model(F))


MODEL_CLASSES = {c.name: c for c in (PhysicsLit, RidgeComp, GBMComp, GPComp, GPPhys, GPHybrid,
                                      PhysLitGP, GreyBox, GreyBoxGP, AnchoredGP, AnchoredChain,
                                      GatedEnsemble)}
MODEL_ORDER = ["physics_lit", "ridge_comp", "gbm_comp", "gp_comp", "physlit_gp", "gp_phys",
               "greybox", "greybox_gp", "gp_hybrid", "anchored_gp", "anchored_chain", "gated_ensemble"]
PRODUCTION_MODEL = "gated_ensemble"


def make_model(name, target):
    return MODEL_CLASSES[name](target)
