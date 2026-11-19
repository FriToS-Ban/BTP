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
QT_BUNDLE = config.MODELS_DIR / "qt_models.joblib"


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
        out["fraction_pm90"] = 1.645 * config.PHASE_FRACTION_STRUCTURAL_SD   # structural band, stage 11
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

    # ------------------------------------------------- heat treatments (v3)
    def _qt(self):
        b = getattr(self, "_qt_bundle", None)
        if b is None:
            if not QT_BUNDLE.exists():
                raise FileNotFoundError(f"{QT_BUNDLE} not found; run pipeline stage 9")
            b = joblib.load(QT_BUNDLE)
            self._qt_bundle = b
        return b

    @staticmethod
    def _comp_frame(comp):
        df = pd.DataFrame([dict(comp)])
        for el in config.ELEMENTS:
            default = 0.0 if el in ("C", "Mn", "Si") else config.RESIDUAL_TYPICAL.get(el, 0.0)
            if el not in df.columns:
                df[el] = default
            df[el] = df[el].astype(float).fillna(default)
        return df

    @staticmethod
    def microstructure_label(F, P, B, M, prefix="", fine=None):
        """Constituents above 2 %, largest first, e.g. 'fine pearlite 54 % + ferrite 46 %'."""
        parts = [(M, f"{prefix}martensite"), (B, f"{prefix}bainite"), (F, "ferrite"),
                 (P, f"{fine} pearlite" if fine else "pearlite")]
        parts = [(v, n) for v, n in parts if np.isfinite(v) and v >= 0.02]
        parts.sort(key=lambda x: -x[0])
        return " + ".join(f"{n} {v:.0%}" for v, n in parts) if parts else "n/a"

    @staticmethod
    def default_quench_medium(comp, section_mm):
        """Water for shallow-hardening steels (D_I below 1.6 x the bar size),
        oil otherwise - the usual shop choice that avoids quench cracking."""
        from .physics.hardenability import ideal_critical_diameter
        di_mm = float(np.asarray(ideal_critical_diameter(comp)[1]).ravel()[0])
        return "water" if di_mm < 1.6 * float(section_mm) else "oil"

    def heat_treat(self, comp: dict, treatment: str, section_mm=config.AC_DEFAULT_DIAMETER_MM, medium=None,
                   temper_T=600.0, temper_h=1.0, sr_T=600.0, sr_h=1.0, sr_base="normalise", T_aust=None,
                   fc_rate=config.FC_RATE_700C):
        """One heat treatment of one steel -> one row of HEAT_TREAT_COLUMNS.

        treatment: 'anneal' (furnace cooling), 'normalise' (still air, bar of
        section_mm), 'harden' (quench, untempered), 'quench_temper' (quench +
        temper at temper_T for temper_h), 'stress_relieve' (soak at sr_T for
        sr_h after the state sr_base: 'normalise', 'anneal' or 'quench_temper').
        Properties are at the tensile-specimen position of the bar (ASTM A370).
        """
        from .models.flags import composition_flags
        from .physics import precipitation, stress_relief, thermo as th
        df = self._comp_frame(comp)
        flags = composition_flags(df)
        row = {"treatment": treatment, "section_mm": float(section_mm)}
        if treatment in ("anneal", "normalise"):
            route = "FC" if treatment == "anneal" else "AC"
            d = df.assign(route=route, section_mm=section_mm, fc_rate=fc_rate)
            if T_aust is not None:
                d["treat_T_C"] = T_aust
            o = self.predict(d).iloc[0]
            dppt = float(precipitation.strengthening(d, o["treat_T_C"], route)[0])
            row.update(route=route, medium="furnace" if route == "FC" else "still air", T_aust_C=o["treat_T_C"],
                       temper_T_C=np.nan, F=o["F_best"], P=o["P_best"] + o["Fe3C_best"], B=o["B_best"],
                       M=o["M_best"], S0_um=o["S0_um"], d_alpha_um=o["d_alpha_um"],
                       cool_rate_700=o["cool_rate_700"], ITT_C=o["ITT_best_C"], dsigma_ppt_MPa=dppt,
                       in_domain=bool(o["in_domain"]))
            for t in ("UTS_MPa", "YS_MPa", "El_pct"):
                add = dppt if t == "YS_MPa" else (0.7 * dppt if t == "UTS_MPa" else 0.0)
                for s in ("pred", "lo90", "hi90"):
                    row[f"{t}_{s}"] = o[f"{t}_{s}"] + add
            row.update(HB=o["HB_pred"] + 0.7 * dppt / 3.45, HRC=np.nan)
            row["HV"] = row["HB"] / 0.95
            row["microstructure"] = self.microstructure_label(row["F"], row["P"], row["B"], row["M"],
                                                              fine="coarse" if route == "FC" else "fine")
        elif treatment in ("harden", "quench_temper"):
            medium = medium or self.default_quench_medium(df, section_mm)
            d = df.assign(quench_medium=medium, section_mm=section_mm,
                          temper_T_C=np.nan if treatment == "harden" else temper_T,
                          temper_time_h=temper_h)
            if T_aust is not None:
                d["treat_T_C"] = T_aust
            o = self._qt().predict(d).iloc[0]
            row.update(route="Q" if treatment == "harden" else "QT", medium=medium, T_aust_C=o["treat_T_C"],
                       temper_T_C=o["temper_T_C"], F=float(o["k_f_ferrite"]),
                       P=float(o["k_f_pearlite"] + o["k_f_cementite_pro"]), B=float(o["f_B"]), M=float(o["f_M"]),
                       S0_um=np.nan, d_alpha_um=np.nan, cool_rate_700=o["cool_rate_700"], ITT_C=np.nan,
                       dsigma_ppt_MPa=0.0, HB=o["HB_est"], HRC=o["HRC_est"], in_domain=True,
                       TME_risk=bool(o["TME_risk"]), TE_risk=bool(o["TE_risk"]))
            for t in ("UTS_MPa", "YS_MPa", "El_pct"):
                for s in ("pred", "lo90", "hi90"):
                    row[f"{t}_{s}"] = o[f"{t}_{s}"]
            row["HV"] = o["HV_est"]
            if treatment == "harden":
                # untempered martensite: brittle, a tensile test is not meaningful
                row["UTS_equiv_MPa"] = row["UTS_MPa_pred"]
                for t in ("UTS_MPa", "YS_MPa", "El_pct"):
                    for s in ("pred", "lo90", "hi90"):
                        row[f"{t}_{s}"] = np.nan
            pre = "tempered " if treatment == "quench_temper" else ""
            row["microstructure"] = self.microstructure_label(row["F"], row["P"], row["B"], row["M"], prefix=pre)
        elif treatment == "stress_relieve":
            base = self.heat_treat(comp, sr_base, section_mm, medium, temper_T, temper_h, T_aust=T_aust,
                                   fc_rate=fc_rate).iloc[0].to_dict()
            if sr_base == "quench_temper":
                # Hollomon-Jaffe additivity: the soak counts only if it tempers the
                # steel further (a parameter more than ~50 above the prior one, about
                # 2.5 C); the softer state is then predicted as its equivalent tempering
                # temperature for the original time, the form the models were trained on
                from .physics import tempering as tp
                P1 = float(tp.hollomon_jaffe(temper_T, temper_h))
                P_after = float(stress_relief.tempering_after_relief(P1, sr_T, sr_h))
                row["P_before"], row["P_after"] = P1, P_after
                if P_after > P1 + 50.0:
                    T_eq = P_after / (tp.C_HJ + np.log10(temper_h)) - 273.15
                    re = self.heat_treat(comp, "quench_temper", section_mm, base["medium"], T_eq, temper_h,
                                         T_aust=T_aust).iloc[0].to_dict()
                    base.update({k: re[k] for k in re if k.startswith(("UTS", "YS", "El", "HB", "HRC", "HV"))})
            row.update({k: v for k, v in base.items() if k not in ("treatment",)})
            row["microstructure"] = "unchanged: " + str(base["microstructure"])
            row["sr_T_C"], row["sr_h"] = sr_T, sr_h
            row["subcritical"] = bool(stress_relief.is_subcritical(df, sr_T)[0])
            row["residual_stress_bound_MPa"] = float(stress_relief.residual_stress_bound(base["YS_MPa_pred"], sr_T))
            row["Ac1_C"] = float(th.andrews_ac1(df)[0])
        else:
            raise ValueError(f"unknown treatment {treatment!r}")
        # structural uncertainty of every phase fraction (stage 11): +/- at 90 %
        row["fraction_pm90"] = 1.645 * config.PHASE_FRACTION_STRUCTURAL_SD
        row["microalloyed"] = bool(flags["microalloyed"][0])
        row["tmcp_like"] = bool(flags["tmcp_like"][0])
        row["boron"] = bool(flags["boron"][0])
        row["notes"] = str(flags["composition_notes"][0])
        return pd.DataFrame([row])

    def compare_treatments(self, comp: dict, section_mm=config.AC_DEFAULT_DIAMETER_MM, medium=None,
                           temper_T=600.0, temper_h=1.0, sr_T=600.0, sr_h=1.0):
        """The five classic heat treatments of one steel, side by side:
        annealing, normalising, hardening, hardening + tempering, and stress
        relieving (of the normalised bar)."""
        rows = [self.heat_treat(comp, "anneal", section_mm),
                self.heat_treat(comp, "normalise", section_mm),
                self.heat_treat(comp, "harden", section_mm, medium),
                self.heat_treat(comp, "quench_temper", section_mm, medium, temper_T, temper_h),
                self.heat_treat(comp, "stress_relieve", section_mm, medium, temper_T, temper_h, sr_T, sr_h,
                                sr_base="normalise")]
        return pd.concat(rows, ignore_index=True)

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
