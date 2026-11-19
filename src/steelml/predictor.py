"""End-to-end predictor: composition + cooling route -> category,
microstructure and mechanical properties with calibrated intervals.

    from steelml.predictor import SteelPredictor
    sp = SteelPredictor.load()
    out = sp.what_if({"C": 0.45, "Mn": 0.75, "Si": 0.25})   # furnace vs air cooling

The property models are the gated ensembles of
:mod:`steelml.models.property_models`; their 90 % intervals are
*normalised split-conformal* intervals: the half-width is q * sigma, where
sigma is the model's own predictive standard deviation and q is the 90 %
quantile of |residual| / sigma over the leave-one-grade-out predictions.
Intervals therefore widen automatically where the model is uncertain or
out of domain, while keeping the empirical 90 % coverage.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import featurize
from .models.property_models import TARGETS, make_model, PRODUCTION_MODEL
from .physics import thermo
from .physics.properties import uniform_elongation_estimate

BUNDLE = config.MODELS_DIR / "steel_predictor.joblib"


def default_austenitising_T(comp, route=None):
    """Usual austenitising temperature for the route (see
    :func:`steelml.physics.thermo.default_austenitising_T`): hotter for
    normalising than for annealing, as in the handbook data."""
    return thermo.default_austenitising_T(comp, route)


def best_estimate_fractions(F):
    """Reported constituent fractions, one rule at every cooling rate.

    * Bainite, martensite and proeutectoid cementite come from the kinetic
      model, whose transformation temperatures are the validated part
      (dilatometry, stage 3; Jominy H-bands, stage 8).
    * The remaining ferrite + pearlite is split in lever-rule proportions
      (F_best, P_best). The measured fractions of annealed and of normalised
      steels in the literature compilation lie at or below the lever value
      (stage 3), so this is the central value we report.
    * The kinetic split of the same ferrite + pearlite (F_lower, P_upper) is
      reported as the other end of a band. It contains the quasi-eutectoid
      excess of pearlite that textbooks describe for faster cooling (the
      austenite enters the Hultgren region before all the equilibrium ferrite
      has formed), which the Li model overstates: about 82-86 % pearlite for
      normalised 1045 / S45C, where 50-56 % was measured.
    An earlier version switched from the lever to the kinetic split above
    3 K/s, which put a step into the fractions; the band replaces it.
    The raw kinetic fractions remain available as k_f_* columns.
    """
    fB, fM = F["k_f_bainite"].to_numpy(float), F["k_f_martensite"].to_numpy(float)
    fC = F["k_f_cementite_pro"].to_numpy(float)
    fFP = np.clip(1.0 - fB - fM - fC - F["k_f_austenite_ret"].to_numpy(float), 0.0, 1.0)
    p_lever = F["f_pearlite_eq"].to_numpy(float)
    # lever split of the ferrite + pearlite part (hypoeutectoid; hyper: all pearlite)
    p_share = np.where(F["f_ferrite_eq"].to_numpy(float) > 0, p_lever, 1.0)
    kF, kP = F["k_f_ferrite"].to_numpy(float), F["k_f_pearlite"].to_numpy(float)
    k_share = np.where(kF + kP > 1e-6, kP / np.maximum(kF + kP, 1e-12), p_share)
    upper = np.maximum(k_share, p_share)
    return {
        "F_best": fFP * (1.0 - p_share), "P_best": fFP * p_share,
        "B_best": fB, "M_best": fM, "Fe3C_best": fC,
        "F_lower": fFP * (1.0 - upper), "P_upper": fFP * upper,
    }


def feature_signature():
    """Structure and input features of the production ensemble. Pickled
    models read their feature lists from the current class definitions, so a
    bundle trained with other features or members would silently get the
    wrong inputs; load() checks this."""
    from .models import property_models as pm
    return {"ensemble": list(pm.GatedEnsemble.MEMBERS), "data_weight": pm.GatedEnsemble.DATA_WEIGHT,
            "anchor": pm.ANCHOR_COLUMN,
            "kinetics": [float(getattr(config, k, d)) for k, d in (("FERRITE_RATE_FACTOR", 1.0),
                         ("PEARLITE_RATE_FACTOR", 1.0), ("BAINITE_RATE_FACTOR", 1.0), ("BAINITE_START_SHIFT_C", 0.0))],
            "gp_hybrid": list(pm.GPHybrid.features), "anchored_gp": list(pm.AnchoredGP.resid_features),
            "physlit_gp": list(pm.PhysLitGP.resid_features), "ratio_gp": list(pm._RatioGP.features)}


class SteelPredictor:
    def __init__(self, models, q90, category_model=None, meta=None):
        self.models = models
        self.q90 = q90
        self.category_model = category_model
        self.meta = meta or {}

    # ------------------------------------------------------------------ io
    def save(self, path=BUNDLE):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=BUNDLE):
        sp = joblib.load(path)
        sig = sp.meta.get("feature_signature")
        if sig is not None and sig != feature_signature():
            raise RuntimeError(f"{path} was trained with other model features than the current code; "
                               "rerun the pipeline from stage 4 (python pipeline/run_all.py --from 4).")
        return sp

    @classmethod
    def train(cls, F_train, q90, category_model=None, meta=None):
        models = {t: make_model(PRODUCTION_MODEL, t).fit(F_train, F_train[t].to_numpy(float)) for t in TARGETS}
        meta = dict(meta or {}, feature_signature=feature_signature())
        return cls(models, q90, category_model, meta)

    # ------------------------------------------------------------- predict
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """df: element columns + route (FC/AC/CR) + optional treat_T_C, section_mm, fc_rate, cooling_rate."""
        df = df.copy()
        for el in config.ELEMENTS:
            # unspecified elements (missing column or blank cell) are taken at
            # typical residual levels; C, Mn and Si must be given
            default = 0.0 if el in ("C", "Mn", "Si") else config.RESIDUAL_TYPICAL.get(el, 0.0)
            if el not in df.columns:
                df[el] = default
            df[el] = df[el].astype(float).fillna(default)
        if "treat_T_C" not in df.columns or df["treat_T_C"].isna().any():
            auto = default_austenitising_T(df, df["route"] if "route" in df.columns else None)
            df["treat_T_C"] = df.get("treat_T_C", pd.Series(np.nan, index=df.index)).fillna(
                pd.Series(np.asarray(auto, float), index=df.index))
        df["section_mm"] = df.get("section_mm", pd.Series(config.AC_DEFAULT_DIAMETER_MM, index=df.index))
        df["fc_rate"] = df.get("fc_rate", pd.Series(config.FC_RATE_700C, index=df.index))
        F = featurize(df)
        out = F.copy()
        for t, m in self.models.items():
            mu, sd = m.predict(F)
            half = self.q90[t] * np.maximum(sd, 1e-6)
            out[f"{t}_pred"] = np.maximum(mu, 0.0)
            out[f"{t}_lo90"] = np.maximum(mu - half, 0.0)   # no property is negative
            out[f"{t}_hi90"] = mu + half
        # model-derived quantities (not trained on measured data; see report)
        n, eu, eu_lo, eu_hi = uniform_elongation_estimate(out["El_pct_pred"], out["El_pct_lo90"],
                                                          out["El_pct_hi90"])
        out["n_est"] = n
        out["uniform_El_est_pct"] = eu
        out["uniform_El_lo90"] = eu_lo
        out["uniform_El_hi90"] = eu_hi
        best = best_estimate_fractions(out)
        for k, v in best.items():
            out[k] = v
        # toughness indicator: Pickering's impact transition temperature of
        # ferrite-pearlite steels on the reported microstructure (C); not
        # defined when bainite + martensite make up half the structure or more
        from .physics.properties import itt_pickering
        itt = itt_pickering(df["Si"].to_numpy(float), out["P_best"] + out["Fe3C_best"], out["d_alpha_um"])
        out["ITT_best_C"] = np.where((out["B_best"] + out["M_best"]) < 0.5, itt, np.nan)
        out["in_domain"] = self.models["UTS_MPa"].dom.check(F)["in_domain"].to_numpy()
        ens = self.models["UTS_MPa"]
        out["data_weight"] = ens.weight_data_model(F)
        # the microstructure is of a kind the training data contain (no more
        # martensite than any training steel), so the data-driven member keeps
        # its full weight; otherwise the prediction is a physics extrapolation
        out["microstructure_in_data"] = out["data_weight"].to_numpy() >= 0.9 * ens.DATA_WEIGHT
        if self.category_model is not None:
            cat = self.category_model.predict(df[config.ELEMENTS])
            out["category"] = cat.predicted.to_numpy()
            out["category_confidence"] = cat.confidence.to_numpy()
            out["category_unknown"] = cat.unknown.to_numpy()
            # independent cross-check: the AISI/SAE composition limits as a
            # decision list. On 1358 real heats the family was right for 97 %
            # of the heats where both agree and 74 % where they do not.
            from .models.rules import rule_category
            rules, _ = rule_category(df[config.ELEMENTS])
            out["category_rules"] = rules
            out["category_rules_agree"] = rules == out["category"].to_numpy()
        # ideal critical diameter (Grossmann / ASTM A255), a hardenability index
        from .physics.hardenability import ideal_critical_diameter
        out["DI_mm"] = ideal_critical_diameter(df)[1]
        return out

    def what_if(self, comp: dict, section_mm=config.AC_DEFAULT_DIAMETER_MM, fc_rate=config.FC_RATE_700C,
                T_aust=None):
        """Furnace cooling vs air cooling for one composition."""
        rows = []
        for route in ("FC", "AC"):
            r = dict(comp)
            r.update(route=route, section_mm=section_mm, fc_rate=fc_rate)
            if T_aust is not None:
                r["treat_T_C"] = T_aust
            rows.append(r)
        return self.predict(pd.DataFrame(rows))
