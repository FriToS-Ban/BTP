"""Stage 6 - external validation on data the models never saw.

A. Hot-rolled structural members (Kingsley et al. 2026, Zenodo 19000367):
   measured heat chemistry and tensile properties of 165 S275/S355 sections.
   They are *as-rolled* (air cooled from rolling), so they are predicted with
   the as-rolled route at an equivalent bar diameter of 2 x thickness.
   Microalloyed members (Nb/V/Ti) carry precipitation strengthening that no
   training steel has, so they are reported separately.
B. Second compilation of the same AISI grades (Zenodo 17704410, MatWeb-
   derived): rows whose values differ from the ASM table are an
   independent "typical value" check; SAE J1397 hot-rolled *minimum*
   values are used as a sanity check of the lower prediction bound.
C. EN/ASTM normalised grades (Zenodo 17704410): specification-level values
   of plain C and C-Mn normalised steels - a plausibility check.

Outputs: results/s06_*.csv, figure fig19.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import config, viz  # noqa: E402
from steelml.data.catalog import nominal_composition  # noqa: E402
from steelml.models.evaluation import metrics  # noqa: E402
from steelml.physics.properties import (PROPORTIONAL_GAUGE_K, elongation_to_gauge,  # noqa: E402
                                        gauge_ratio_from_text, itt_pickering)
from steelml.predictor import SteelPredictor  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt
TARGETS = [("YS_MPa", "Yield strength (MPa)"), ("UTS_MPa", "Tensile strength (MPa)"), ("El_pct", "Elongation (%)")]


def structural(sp):
    ext = pd.read_csv(config.PROCESSED_DIR / "structural_external.csv")
    ext = ext[ext.usable].reset_index(drop=True)
    # elongation on the handbook basis (4.5 sqrt(S0)): the source gives each
    # value converted to the proportional gauge 5.65 sqrt(S0) (ISO 2566-1)
    ext["El_pct_as_tested"] = ext.El_pct
    ext["El_pct"] = elongation_to_gauge(ext.El_converted_pct.fillna(ext.El_pct), PROPORTIONAL_GAUGE_K)
    X = ext[config.ELEMENTS + ["section_mm"]].copy()
    X["route"] = "AR"
    out = sp.predict(X)
    for t, _ in TARGETS:
        ext[f"{t}_pred"] = out[f"{t}_pred"].to_numpy()
        ext[f"{t}_lo90"] = out[f"{t}_lo90"].to_numpy()
        ext[f"{t}_hi90"] = out[f"{t}_hi90"].to_numpy()
    ext["in_domain"] = out.in_domain.to_numpy()
    ext["category_pred"] = out.category.to_numpy()
    ext["f_pearlite_pred"] = out.k_f_pearlite.to_numpy()
    ext["d_alpha_pred_um"] = out.d_alpha_um.to_numpy()
    # Pickering's impact transition temperature on the predicted microstructure
    ext["ITT_pred_C"] = itt_pickering(ext.Si, out.P_best + out.Fe3C_best, out.d_alpha_um)
    rows = []
    for grp_name, sel in (("all", np.ones(len(ext), bool)), ("not microalloyed", ~ext.microalloyed.to_numpy()),
                          ("microalloyed (Nb/V/Ti)", ext.microalloyed.to_numpy())):
        for t, _ in TARGETS:
            y = ext[t].to_numpy()[sel]
            p = ext[f"{t}_pred"].to_numpy()[sel]
            m = metrics(y, p)
            cov = float(np.mean((y >= ext[f"{t}_lo90"].to_numpy()[sel]) & (y <= ext[f"{t}_hi90"].to_numpy()[sel])))
            ok = np.isfinite(y) & np.isfinite(p)
            rows.append({"subset": grp_name, "target": t, "n": m["n"], "RMSE": m["RMSE"], "MAE": m["MAE"],
                         "bias": m["bias"], "MAPE_pct": m["MAPE_pct"], "R2": m["R2"],
                         "spearman": float(spearmanr(y[ok], p[ok]).correlation), "coverage_90": cov})
    # toughness: does the predicted transition temperature rank the measured
    # Charpy energy at 0 C? (lower ITT -> tougher -> more energy expected)
    c0 = (ext.charpy_T_C == 0) & ext.charpy_avg_J.notna()
    rho = spearmanr(ext.ITT_pred_C[c0], ext.charpy_avg_J[c0]).correlation
    rows.append({"subset": "all", "target": "Charpy energy at 0 C vs predicted ITT (Pickering)",
                 "n": int(c0.sum()), "spearman": float(rho)})
    return ext, pd.DataFrame(rows)


def secondary(sp):
    sec = pd.read_csv(config.PROCESSED_DIR / "properties_secondary.csv")
    rows = []
    for _, r in sec.iterrows():
        d = {el: r[el] for el in config.ELEMENTS}
        d["route"] = r.route
        d["section_mm"] = config.HANDBOOK_AC_SECTION_MM if r.route == "AC" else 25.0
        d["grade"] = r.grade
        d["condition"] = r.condition
        d["repeats_asm"] = r.repeats_asm
        d["in_asm"] = r.in_asm
        for t in ("UTS_MPa", "YS_MPa", "El_pct", "HB"):
            d[f"{t}_obs"] = r[t]
        rows.append(d)
    X = pd.DataFrame(rows)
    out = sp.predict(X.drop(columns=["grade", "condition"]))
    for t in ("UTS_MPa", "YS_MPa", "El_pct", "HB"):
        X[f"{t}_pred"] = out[f"{t}_pred"].to_numpy()
        X[f"{t}_lo90"] = out[f"{t}_lo90"].to_numpy()
    X["in_domain"] = out.in_domain.to_numpy()
    res = []
    indep = X[(~X.repeats_asm) & (X.route != "AR")]
    for t in ("UTS_MPa", "YS_MPa", "El_pct", "HB"):
        m = metrics(indep[f"{t}_obs"], indep[f"{t}_pred"])
        res.append({"check": "independent typical values (annealed/normalized)", "target": t, **m})
    hr = X[X.route == "AR"]
    for t in ("UTS_MPa", "YS_MPa"):
        below = float(np.mean(hr[f"{t}_pred"] >= hr[f"{t}_obs"]))
        res.append({"check": "SAE J1397 hot-rolled minimum <= predicted typical", "target": t, "n": len(hr),
                    "fraction_satisfied": below})
    return X, pd.DataFrame(res)


def bar_sizes(sp):
    """Normalised bars of 13-100 mm and annealed bars (Birbilis et al. 2025).

    Rows that repeat the training data (the 25 mm normalised and the annealed
    rows of handbook grades) are kept apart from the independent ones: other
    bar sizes of handbook grades (new cooling rates) and grades the models
    never saw. The bar-size effect (13 mm minus 100 mm) is compared as a
    paired difference, which removes each grade's offset."""
    nb = pd.read_csv(config.PROCESSED_DIR / "normalized_bars.csv")
    hb = pd.read_csv(config.PROCESSED_DIR / "properties_handbook.csv")
    seen = set(hb.grade.str.replace("AISI ", "", regex=False))
    X = nb[((nb.route == "FC") | nb.section_mm.notna()) & (nb.quality_flag.fillna("") == "")].copy()
    X["section_mm"] = X.section_mm.where(X.route == "AC", 25.0)
    X["fc_rate"] = config.HANDBOOK_FC_RATE
    out = sp.predict(X[config.ELEMENTS + ["route", "treat_T_C", "section_mm", "fc_rate"]].reset_index(drop=True))
    X = X.reset_index(drop=True)
    for t in ("UTS_MPa", "YS_MPa", "El_pct"):
        X[f"{t}_pred"] = out[f"{t}_pred"].to_numpy()
        X[f"{t}_lo90"] = out[f"{t}_lo90"].to_numpy()
        X[f"{t}_hi90"] = out[f"{t}_hi90"].to_numpy()
    X["in_domain"] = out.in_domain.to_numpy()
    X["seen_grade"] = X.grade.isin(seen)
    X["repeats_training"] = X.seen_grade & ((X.route == "FC") | (X.section_mm.round() == 25))
    rows = []
    groups = {"repeats training rows": X.repeats_training,
              "normalised, handbook grades, other bar sizes": (X.route == "AC") & X.seen_grade & ~X.repeats_training,
              "normalised, grades never seen": (X.route == "AC") & ~X.seen_grade,
              "annealed, grades never seen": (X.route == "FC") & ~X.seen_grade}
    for lab, msk in groups.items():
        for t in ("UTS_MPa", "YS_MPa", "El_pct"):
            s_ = X[msk & X[f"{t}"].notna()]
            if len(s_) < 3:
                continue
            e = s_[f"{t}_pred"] - s_[t]
            cover = float(np.mean((s_[t] >= s_[f"{t}_lo90"]) & (s_[t] <= s_[f"{t}_hi90"])))
            rows.append({"check": lab, "target": t, "n": len(s_), "grades": s_.grade.nunique(),
                         "RMSE": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)), "coverage90": cover})
    for D in (13, 25, 50, 100):
        s_ = X[(X.route == "AC") & (X.section_mm.round() == D) & ~X.repeats_training]
        if len(s_) >= 3:
            e = s_.UTS_MPa_pred - s_.UTS_MPa
            rows.append({"check": f"normalised {D} mm bars (independent)", "target": "UTS_MPa", "n": len(s_),
                         "grades": s_.grade.nunique(), "RMSE": float(np.sqrt(np.mean(e ** 2))),
                         "bias": float(np.mean(e)), "coverage90": float(np.mean((s_.UTS_MPa >= s_.UTS_MPa_lo90)
                                                                                 & (s_.UTS_MPa <= s_.UTS_MPa_hi90)))})
    ac = X[X.route == "AC"].pivot_table(index="grade", columns=X.section_mm.round(), values=["UTS_MPa", "UTS_MPa_pred"])
    pairs = pd.DataFrame()
    if ("UTS_MPa", 13.0) in ac.columns and ("UTS_MPa", 100.0) in ac.columns:
        pairs = pd.DataFrame({"measured_13_minus_100": ac[("UTS_MPa", 13.0)] - ac[("UTS_MPa", 100.0)],
                              "predicted_13_minus_100": ac[("UTS_MPa_pred", 13.0)] - ac[("UTS_MPa_pred", 100.0)]}).dropna()
        e = pairs.predicted_13_minus_100 - pairs.measured_13_minus_100
        rows.append({"check": "bar-size effect, UTS(13 mm) - UTS(100 mm), paired", "target": "UTS_MPa",
                     "n": len(pairs), "grades": len(pairs), "RMSE": float(np.sqrt(np.mean(e ** 2))),
                     "bias": float(np.mean(e)),
                     "sign_agreement": float(np.mean(np.sign(pairs.predicted_13_minus_100)
                                                     == np.sign(pairs.measured_13_minus_100))),
                     "mean_measured": float(pairs.measured_13_minus_100.mean()),
                     "mean_predicted": float(pairs.predicted_13_minus_100.mean())})
    return X, pd.DataFrame(rows), pairs.reset_index()


def fig_bar_sizes(X):
    grades = [g for g in ("1020", "1045", "1095", "4140", "4340", "8620") if g in set(X.grade)]
    fig, axes = plt.subplots(1, len(grades), figsize=(2.6 * len(grades), 3.6), sharey=False)
    for ax, g in zip(np.atleast_1d(axes), grades):
        s = X[(X.grade == g) & (X.route == "AC")].sort_values("section_mm")
        ax.plot(s.section_mm, s.UTS_MPa_pred, "-", color=viz.ORANGE, lw=1.6, label="predicted")
        ax.fill_between(s.section_mm, s.UTS_MPa_lo90, s.UTS_MPa_hi90, color=viz.ORANGE, alpha=0.15, lw=0)
        ax.plot(s.section_mm, s.UTS_MPa, "o", color=viz.INK, ms=4, label="measured")
        a = X[(X.grade == g) & (X.route == "FC")]
        if len(a):
            ax.axhline(a.UTS_MPa.iloc[0], color=viz.BLUE, lw=1, ls="--", label="annealed (measured)")
        ax.set_xscale("log"); ax.set_xticks([13, 25, 50, 100]); ax.set_xticklabels(["13", "25", "50", "100"])
        ax.xaxis.set_minor_formatter(viz.matplotlib.ticker.NullFormatter())
        ax.set_title(f"AISI {g}", loc="left"); ax.set_xlabel("bar diameter (mm)")
    np.atleast_1d(axes)[0].set_ylabel("UTS of normalised bars (MPa)")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    return viz.savefig(fig, "fig42_normalising_bar_size")


def en_normalised(sp):
    z = pd.read_excel(config.RAW_DIR / "zenodo17704410_Carbon_stainless_grades.xlsx")
    z = z[(z.Type == "Carbon") & (z.Standard.isin(["EN", "ASTM", "GB"])) &
          (z.Treatment_Condition == "Normalised")].copy()
    rows = []
    for _, r in z.iterrows():
        rec = {}
        for el in ["Mn", "C", "Si", "P", "S", "Cr", "Ni", "Mo", "Cu", "Ti", "Al", "V", "N", "Nb"]:
            lo, hi = r.get(f"{el}_Min", np.nan), r.get(f"{el}_Max", np.nan)
            rec[f"{el}_min"] = 0.0 if (pd.notna(lo) and lo <= 1e-4) else lo
            rec[f"{el}_max"] = hi if (pd.notna(hi) and hi > 0) else np.nan
        nom = nominal_composition(pd.Series(rec))
        if not np.isfinite(nom["C"]) or nom["C"] <= 0:
            continue
        nom.update(route="AC", section_mm=25.0)
        nom["grade"] = r.Grade
        nom["UTS_obs"], nom["YS_obs"], nom["El_obs"] = r.UTS, r.YS, r.Elongation
        rows.append(nom)
    X = pd.DataFrame(rows)
    out = sp.predict(X.drop(columns=["grade"]))
    X["UTS_pred"] = out.UTS_MPa_pred.to_numpy()
    X["YS_pred"] = out.YS_MPa_pred.to_numpy()
    X["El_pred"] = out.El_pct_pred.to_numpy()
    X["in_domain"] = out.in_domain.to_numpy()
    res = []
    for t in ("UTS", "YS", "El"):
        m = metrics(X[f"{t}_obs"], X[f"{t}_pred"])
        res.append({"check": "EN/ASTM/GB normalised grades (specification-level values)", "target": t, **m})
    return X, pd.DataFrame(res)


def _rate(v):
    """Cooling rate cell -> float; a range 'a-b' becomes its geometric mean."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if "-" in s[1:]:
        a, b = (float(x) for x in s.split("-", 1))
        return float(np.sqrt(a * b))
    return float(s)


def _clean(df):
    """Drop rows flagged as suspect during transcription (implausible or
    internally inconsistent values); they stay in the compilation for the record."""
    flag = df.get("quality_flag", pd.Series("", index=df.index)).fillna("")
    return df[~flag.str.startswith("suspect")].copy()


def _hb_observed(lit):
    """Measured hardness as an HB equivalent: HB as given, HV x 0.95 (ASTM E140
    is close to this ratio below about 450 HV); other scales are left out."""
    sc = lit.hardness_scale.fillna("").str.upper()
    hv, hb = sc.str.startswith("HV"), sc.str.startswith("HB")
    return np.where(hv, 0.95 * lit.hardness_value, np.where(hb, lit.hardness_value, np.nan))


def paired_routes(sp):
    """The same steel in the same paper, furnace cooled and air cooled:
    measured against predicted change (normalised minus annealed)."""
    lit = _clean(pd.read_csv(config.RAW_DIR / "literature_compilation.csv"))
    lit = lit[lit.condition.isin(["annealed_furnace_cooled", "normalized_air_cooled"]) & lit.C.notna()].copy()
    lit["route"] = np.where(lit.condition == "annealed_furnace_cooled", "FC", "AC")
    lit["HB_obs"] = _hb_observed(lit)
    lit["key"] = lit.source_id + "|" + lit.C.round(3).astype(str) + "|" + lit.Mn.round(3).astype(str)
    both = lit.groupby("key").route.transform(lambda r: set(r) == {"FC", "AC"})
    lit = lit[both].reset_index(drop=True)
    if lit.empty:
        return pd.DataFrame()
    rows = []
    for _, r in lit.iterrows():
        d = {el: (float(r[el]) if pd.notna(r.get(el, np.nan)) else
                  (0.25 if el == "Si" else config.RESIDUAL_TYPICAL.get(el, 0.0))) for el in config.ELEMENTS}
        d["route"] = r.route
        size = r.section_size_mm
        d["section_mm"] = float(size) if (pd.notna(size) and size >= 5) else 25.0
        if pd.notna(r.austenitizing_T_C):
            d["treat_T_C"] = float(r.austenitizing_T_C)
        rows.append(d)
    out = sp.predict(pd.DataFrame(rows))
    for t in ("UTS_MPa", "YS_MPa", "HB"):
        lit[f"{t}_pred"] = out[f"{t}_pred"].to_numpy()
    lit["UTS_obs"], lit["YS_obs"] = lit.UTS_MPa, lit.YS_MPa
    rec = []
    for key, g in lit.groupby("key"):
        fc, ac = g[g.route == "FC"].iloc[0], g[g.route == "AC"].iloc[0]
        for t, obs in (("HB", "HB_obs"), ("UTS_MPa", "UTS_obs"), ("YS_MPa", "YS_obs")):
            if pd.notna(fc[obs]) and pd.notna(ac[obs]):
                rec.append({"source_id": fc.source_id, "steel": fc.steel_designation, "target": t,
                            "measured_FC": fc[obs], "measured_AC": ac[obs], "measured_change": ac[obs] - fc[obs],
                            "predicted_FC": fc[f"{t}_pred"], "predicted_AC": ac[f"{t}_pred"],
                            "predicted_change": ac[f"{t}_pred"] - fc[f"{t}_pred"]})
    return pd.DataFrame(rec)


def continuous_cooling_hardness(sp):
    """Hardness after controlled continuous cooling at stated rates (dilatometer
    tests in the compilation): a further cooling-rate extrapolation test."""
    lit = _clean(pd.read_csv(config.RAW_DIR / "literature_compilation.csv"))
    sel = lit.condition.str.startswith("other: controlled continuous cooling") & lit.hardness_value.notna()
    lit = lit[sel].reset_index(drop=True)
    if lit.empty:
        return pd.DataFrame()
    rows = []
    for _, r in lit.iterrows():
        d = {el: (float(r[el]) if pd.notna(r.get(el, np.nan)) else config.RESIDUAL_TYPICAL.get(el, 0.0))
             for el in config.ELEMENTS}
        d.update(route="CR", cooling_rate=float(r.cooling_rate_C_per_s), treat_T_C=float(r.austenitizing_T_C))
        rows.append(d)
    out = sp.predict(pd.DataFrame(rows))
    return pd.DataFrame({"source_id": lit.source_id, "steel": lit.steel_designation,
                         "rate_K_s": lit.cooling_rate_C_per_s, "T_aust_C": lit.austenitizing_T_C,
                         "HB_measured": _hb_observed(lit), "HB_pred": out.HB_pred.to_numpy(),
                         "HB_lo90": out.HB_lo90.to_numpy(), "HB_hi90": out.HB_hi90.to_numpy(),
                         "B_pred": out.B_best.to_numpy(), "M_pred": out.M_best.to_numpy(),
                         "microstructure_in_data": out.microstructure_in_data.to_numpy()})


# Steels of Carvajal et al. (2017), Materials Research 20(Suppl. 2) 347-352, Table 1 (CC BY 4.0)
MICROGRAPH_STEELS = {
    "SAE 1045": {"comp": dict(C=0.427, Si=0.154, Mn=0.76, P=0.029, S=0.042, Cr=0.128, Ni=0.126, Mo=0.013, Cu=0.435),
                 "bar_mm": 12.7, "T_FC": 900.0, "T_AC": 1000.0, "file": "1045"},
    "SAE 4140": {"comp": dict(C=0.396, Si=0.278, Mn=0.79, P=0.019, S=0.023, Cr=0.93, Ni=0.193, Mo=0.178, Cu=0.22),
                 "bar_mm": 19.05, "T_FC": 900.0, "T_AC": 900.0, "file": "4140"},
}


def micrograph_predictions(sp):
    rows = []
    for name, m in MICROGRAPH_STEELS.items():
        for route, T in (("FC", m["T_FC"]), ("AC", m["T_AC"])):
            rows.append({**m["comp"], "route": route, "treat_T_C": T, "section_mm": m["bar_mm"], "steel": name})
    X = pd.DataFrame(rows)
    out = sp.predict(X.drop(columns=["steel"]))
    out["steel"] = X.steel.to_numpy()
    return out


def fig_micrographs(sp):
    """Published micrographs of the same steel annealed and normalised, with the
    model's prediction for that composition, austenitising temperature and bar."""
    from matplotlib.image import imread
    out = micrograph_predictions(sp)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.4))
    for i, (name, m) in enumerate(MICROGRAPH_STEELS.items()):
        for j, (route, cond) in enumerate((("FC", "annealed"), ("AC", "normalised"))):
            ax = axes[i, j]
            ax.imshow(imread(config.FIGURES_DIR / "micrographs" / f"carvajal2017_{m['file']}_{cond}.jpg"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            r = out[(out.steel == name) & (out.route == route)].iloc[0]
            how = (f"furnace cooled from {r.treat_T_C:.0f} °C" if route == "FC"
                   else f"{m['bar_mm']:.0f} mm bar, still air from {r.treat_T_C:.0f} °C")
            ax.set_title(f"{name}, {cond} ({how})", fontsize=10)
            parts = [f"{k} {100 * r[c]:.0f} %" for k, c in (("F", "F_best"), ("P", "P_best"), ("B", "B_best"),
                                                            ("M", "M_best")) if r[c] >= 0.02]
            txt = "Model: " + " + ".join(parts)
            if r.P_best >= 0.05 and r.B_best + r.M_best < 0.5:
                txt += f"; spacing {1000 * r.S0_um:.0f} nm; ferrite grain {r.d_alpha_um:.0f} µm"
            txt += f"\n{r.HB_pred:.0f} HB (90 %: {r.HB_lo90:.0f}-{r.HB_hi90:.0f}), UTS {r.UTS_MPa_pred:.0f} MPa"
            ax.text(0.01, -0.03, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9, color=viz.INK,
                    bbox=dict(boxstyle="round,pad=0.35", fc=viz.SURFACE, ec=viz.AXIS, lw=0.8))
    fig.text(0.01, 0.005, "Micrographs (cropped) reproduced from Carvajal, Artigas, Monsalve and Arévalo (2017), "
             "Materials Research 20 (Suppl. 2) 347-352, Figs. 2 and 3, doi:10.1590/1980-5373-MR-2016-1083, CC BY 4.0. "
             "50 µm scale bars.", fontsize=8, color=viz.INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=4.2)
    viz.savefig(fig, "fig30_micrographs_vs_model")
    return out


def literature(sp):
    """D. Measured microstructure and properties from open-access papers."""
    lit = _clean(pd.read_csv(config.RAW_DIR / "literature_compilation.csv"))
    route_of = {"annealed_furnace_cooled": "FC", "normalized_air_cooled": "AC", "hot_rolled_air_cooled": "AR"}
    lit = lit[lit.condition.isin(route_of)].copy().reset_index(drop=True)
    rows = []
    for _, r in lit.iterrows():
        d = {}
        for el in config.ELEMENTS:
            v = r.get(el, np.nan)
            d[el] = float(v) if pd.notna(v) else (0.25 if el == "Si" else config.RESIDUAL_TYPICAL.get(el, 0.0))
        route = route_of[r.condition]
        rate = _rate(r.cooling_rate_C_per_s)
        if pd.notna(rate) and route != "FC":
            d["route"], d["cooling_rate"] = "CR", rate
        else:
            d["route"] = route
        size = r.section_size_mm
        d["section_mm"] = float(size) if (pd.notna(size) and size >= 5) else 25.0
        if route == "AR":
            d["section_mm"] = 2.0 * d["section_mm"]
        if pd.notna(r.austenitizing_T_C):
            d["treat_T_C"] = float(r.austenitizing_T_C)
        rows.append(d)
    X = pd.DataFrame(rows)
    out = sp.predict(X)
    lit["route"] = [route_of[c] for c in lit.condition]
    lit["pearlite_pred_pct"] = 100 * (out.P_best + out.Fe3C_best).to_numpy()
    lit["pearlite_kinetic_pct"] = 100 * (out.k_f_pearlite + out.k_f_cementite_pro).to_numpy()
    lit["pearlite_lever_pct"] = 100 * out.f_pearlite_eq.to_numpy()
    lit["ferrite_d_pred_um"] = out.d_alpha_um.to_numpy()
    lit["spacing_pred_nm"] = 1000 * out.S0_um.to_numpy()
    lit["in_domain"] = out.in_domain.to_numpy()
    for t in ("YS_MPa", "UTS_MPa", "El_pct", "HB"):
        lit[f"{t}_pred"] = out[f"{t}_pred"].to_numpy()
    # total elongation on the handbook gauge basis where the specimen geometry is stated
    lit["gauge_k"] = lit.gauge_length_or_specimen.map(gauge_ratio_from_text)
    lit["El_hb_basis"] = elongation_to_gauge(lit.total_elongation_pct, lit.gauge_k)
    lit["HB_obs"] = _hb_observed(lit)
    from steelml.physics.properties import (UNIFORM_TOTAL_RATIO, hollomon_n_from_ratio)
    # Uniform elongation. The ratio method and the Hollomon alternative only
    # need the measured tensile values, not a cooling route, so they are
    # scored on every literature test that reports uniform elongation
    # (any processing); the full chain needs a route and uses `lit`.
    ue = _clean(pd.read_csv(config.RAW_DIR / "literature_compilation.csv"))
    ue = ue[ue.uniform_elongation_pct.notna()].copy().reset_index(drop=True)
    # total elongation of each test converted to the handbook gauge (Oliver,
    # ISO 2566-1), because uniform elongation does not depend on the gauge but
    # total elongation does; the ratio is then the one to apply to handbook-
    # basis predictions
    ue["gauge_k"] = ue.gauge_length_or_specimen.map(gauge_ratio_from_text)
    ue["El_hb_basis"] = elongation_to_gauge(ue.total_elongation_pct, ue.gauge_k)
    # (i) adopted estimate e_u = r * e_t, leave-one-source-out (each source is
    #     predicted with r fitted on the other sources)
    both = ue.El_hb_basis.notna()
    ue["ratio_loso"] = np.nan
    for src in ue.loc[both, "source_id"].unique():
        others = both & (ue.source_id != src)
        r = float((ue.uniform_elongation_pct[others] / ue.El_hb_basis[others]).mean())
        sel = both & (ue.source_id == src)
        ue.loc[sel, "ratio_loso"] = r * ue.El_hb_basis[sel]
    # (ii) rejected alternative: Hollomon curve through the measured yield point and UTS
    ok_m = ue[["YS_MPa", "UTS_MPa"]].notna().all(axis=1).to_numpy()
    _, eu_h = hollomon_n_from_ratio(ue.YS_MPa.fillna(1), ue.UTS_MPa.fillna(2), 0.002)
    ue["hollomon"] = np.where(ok_m, eu_h, np.nan)
    same = (ue.ratio_loso.notna() & ue.hollomon.notna()).to_numpy()
    # (iii) full chain: composition -> predicted total elongation -> e_u
    lit["uniformEl_est_chain"] = out.uniform_El_est_pct.to_numpy()
    lit["uniform_total_ratio_used"] = UNIFORM_TOTAL_RATIO
    # Hollomon on the handbook: a uniform elongation above the total is impossible
    hbk = pd.read_csv(config.PROCESSED_DIR / "properties_handbook.csv")
    _, eu_hbk = hollomon_n_from_ratio(hbk.YS_MPa, hbk.UTS_MPa, 0.002)
    over = np.asarray(eu_hbk - hbk.El_pct, float)

    res = []

    def add(name, obs, pred, sel=None):
        obs, pred = np.asarray(obs, float), np.asarray(pred, float)
        ok = np.isfinite(obs) & np.isfinite(pred)
        if sel is not None:
            ok &= sel
        if ok.sum() == 0:
            return
        m = metrics(obs[ok], pred[ok])
        res.append({"quantity": name, "n": m["n"], "MAE": m["MAE"], "RMSE": m["RMSE"], "bias": m["bias"]})

    add("pearlite fraction (%) - best estimate", lit.pearlite_fraction_pct, lit.pearlite_pred_pct)
    add("pearlite fraction (%) - raw kinetic model", lit.pearlite_fraction_pct, lit.pearlite_kinetic_pct)
    add("pearlite fraction (%) - lever rule baseline", lit.pearlite_fraction_pct, lit.pearlite_lever_pct)
    add("ferrite grain size (µm)", lit.ferrite_grain_size_um, lit.ferrite_d_pred_um)
    add("pearlite spacing (nm)", lit.pearlite_spacing_nm, lit.spacing_pred_nm)
    add("yield strength (MPa)", lit.YS_MPa, lit.YS_MPa_pred)
    add("tensile strength (MPa)", lit.UTS_MPa, lit.UTS_MPa_pred)
    add("total elongation (%), as tested (gauge lengths differ)", lit.total_elongation_pct, lit.El_pct_pred)
    add("total elongation (%), converted to the handbook gauge 4.5 sqrt(S0)", lit.El_hb_basis, lit.El_pct_pred)
    add("hardness (HB, HV x 0.95)", lit.HB_obs, lit.HB_pred)
    add("uniform elongation (%) - ratio method, leave-one-source-out", ue.uniform_elongation_pct, ue.ratio_loso)
    add("uniform elongation (%) - Hollomon through yield point (rejected)", ue.uniform_elongation_pct, ue.hollomon)
    add("uniform elongation (%) - ratio method, tests with YS and UTS", ue.uniform_elongation_pct, ue.ratio_loso,
        same)
    add("uniform elongation (%) - Hollomon, same tests", ue.uniform_elongation_pct, ue.hollomon, same)
    add("uniform elongation (%) - full chain from composition", lit.uniform_elongation_pct, lit.uniformEl_est_chain)
    ratio = (ue.uniform_elongation_pct / ue.El_hb_basis)[both]
    res.append({"quantity": "uniform/total elongation ratio (total on the handbook gauge), all literature tests",
                "n": int(both.sum()),
                "MAE": np.nan, "RMSE": np.nan, "bias": np.nan, "observed_mean": float(ratio.mean()),
                "observed_sd": float(ratio.std()), "sources": int(ue.loc[both, "source_id"].nunique())})
    res.append({"quantity": "handbook rows where Hollomon e_u exceeds total elongation (impossible)",
                "n": len(hbk), "MAE": np.nan, "RMSE": np.nan, "bias": np.nan, "count": int((over > 0).sum()),
                "max_excess_points": float(over.max())})
    # furnace vs air hardness change on the same 4140 heat (S12)
    s12 = lit[lit.source_id == "S12"].copy()
    s12["key"] = s12.steel_designation
    fc = s12[s12.route == "FC"].set_index("key")
    ac = s12[s12.route == "AC"].set_index("key")
    k = fc.index.intersection(ac.index)
    d_obs = (ac.loc[k, "HB_obs"] - fc.loc[k, "HB_obs"]).to_numpy()
    d_pred = (ac.loc[k, "HB_pred"] - fc.loc[k, "HB_pred"]).to_numpy()
    res.append({"quantity": "AISI 4140 (S12): air minus furnace hardness, HB", "n": len(k),
                "MAE": float(np.mean(np.abs(d_pred - d_obs))), "RMSE": float(np.sqrt(np.mean((d_pred - d_obs) ** 2))),
                "bias": float(np.mean(d_pred - d_obs)), "observed_mean": float(np.mean(d_obs)),
                "predicted_mean": float(np.mean(d_pred))})
    return lit, pd.DataFrame(res)


def fig_literature(lit):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax = axes[0]
    s = lit.pearlite_fraction_pct.notna()
    for route in ("FC", "AC", "AR"):
        r = s & (lit.route == route)
        ax.plot(lit.pearlite_fraction_pct[r], lit.pearlite_pred_pct[r], "o", color=viz.ROUTE_COLORS[route], ms=7,
                mec=viz.SURFACE, label=viz.ROUTE_LABELS[route])
    ax.plot(lit.pearlite_fraction_pct[s], lit.pearlite_lever_pct[s], "x", color=viz.MUTED, ms=6,
            label="lever rule (same steels)")
    ax.plot([0, 100], [0, 100], color=viz.AXIS, lw=1)
    ax.set_xlabel("Measured pearlite (%)")
    ax.set_ylabel("Predicted pearlite (%)")
    ax.set_title("Pearlite fraction (literature steels)")
    ax.legend(fontsize=8)
    ax = axes[1]
    s = lit.ferrite_grain_size_um.notna()
    for route in ("FC", "AC", "AR"):
        r = s & (lit.route == route)
        ax.plot(lit.ferrite_grain_size_um[r], lit.ferrite_d_pred_um[r], "o", color=viz.ROUTE_COLORS[route], ms=7,
                mec=viz.SURFACE)
    ax.plot([2, 100], [2, 100], color=viz.AXIS, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured ferrite grain size (µm)")
    ax.set_ylabel("Predicted (µm)")
    ax.set_title("Ferrite grain size")
    ax = axes[2]
    s12 = lit[lit.source_id == "S12"]
    for route in ("FC", "AC"):
        r = s12[s12.route == route]
        ax.plot(r.HB_obs, r.HB_pred, "o", color=viz.ROUTE_COLORS[route], ms=7, mec=viz.SURFACE,
                label=viz.ROUTE_LABELS[route])
    ax.plot([150, 320], [150, 320], color=viz.AXIS, lw=1)
    ax.set_xlabel("Measured hardness (HB from HV)")
    ax.set_ylabel("Predicted (HB)")
    ax.set_title("AISI 4140, annealed vs normalised (S12)")
    ax.legend(fontsize=8)
    fig.suptitle("Literature validation: measured microstructure and hardness from open-access papers",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig21_literature_validation")


def fig_structural(ext):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (t, lab) in zip(axes, TARGETS):
        y = ext[t]
        for ma, col, name in ((False, viz.AQUA, "not microalloyed"), (True, viz.VIOLET, "microalloyed (Nb/V/Ti)")):
            s = ext.microalloyed == ma
            ax.errorbar(y[s], ext[f"{t}_pred"][s],
                        yerr=[ext[f"{t}_pred"][s] - ext[f"{t}_lo90"][s], ext[f"{t}_hi90"][s] - ext[f"{t}_pred"][s]],
                        fmt="o", ms=4.5, color=col, ecolor=col, elinewidth=0.6, alpha=0.75, mec=viz.SURFACE,
                        mew=0.6, label=name)
        lo = min(y.min(), ext[f"{t}_pred"].min())
        hi = max(y.max(), ext[f"{t}_pred"].max())
        ax.plot([lo, hi], [lo, hi], color=viz.AXIS, lw=1)
        m = metrics(y, ext[f"{t}_pred"])
        ax.set_title(f"{lab}\nRMSE {m['RMSE']:.3g}, bias {m['bias']:+.3g}")
        ax.set_xlabel("Measured")
    axes[0].set_ylabel("Predicted (never seen in training)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("External test: 165 hot-rolled S275/S355 members with measured chemistry (Kingsley et al. 2026)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig20_external_structural")


def main():
    config.ensure_dirs()
    sp = SteelPredictor.load()
    ext, s_res = structural(sp)
    ext.to_csv(config.RESULTS_DIR / "s06_structural_predictions.csv", index=False)
    s_res.to_csv(config.RESULTS_DIR / "s06_structural_metrics.csv", index=False)
    print(s_res.round(2).to_string(index=False))
    sec, sec_res = secondary(sp)
    sec.to_csv(config.RESULTS_DIR / "s06_secondary_predictions.csv", index=False)
    en, en_res = en_normalised(sp)
    en.to_csv(config.RESULTS_DIR / "s06_en_normalised_predictions.csv", index=False)
    bars, bar_res, bar_pairs = bar_sizes(sp)
    bars.to_csv(config.RESULTS_DIR / "s06_bar_sizes_predictions.csv", index=False)
    bar_res.to_csv(config.RESULTS_DIR / "s06_bar_sizes_metrics.csv", index=False)
    bar_pairs.to_csv(config.RESULTS_DIR / "s06_bar_size_effect_pairs.csv", index=False)
    print(bar_res.round(2).to_string(index=False))
    fig_bar_sizes(bars)
    other = pd.concat([sec_res, en_res], ignore_index=True)
    other.to_csv(config.RESULTS_DIR / "s06_other_checks.csv", index=False)
    print(other.round(3).to_string(index=False))
    fig_structural(ext)
    lit, lit_res = literature(sp)
    lit.to_csv(config.RESULTS_DIR / "s06_literature_predictions.csv", index=False)
    lit_res.to_csv(config.RESULTS_DIR / "s06_literature_metrics.csv", index=False)
    print(lit_res.round(2).to_string(index=False))
    fig_literature(lit)
    pr = paired_routes(sp)
    pr.to_csv(config.RESULTS_DIR / "s06_literature_paired_routes.csv", index=False)
    if not pr.empty:
        print(pr.round(1).to_string(index=False))
        e = pr.predicted_change - pr.measured_change
        print("paired change: n", len(pr), "MAE", round(float(np.mean(np.abs(e))), 1), "sign agreement",
              round(float(np.mean(np.sign(pr.predicted_change) == np.sign(pr.measured_change))), 2))
    cc = continuous_cooling_hardness(sp)
    cc.to_csv(config.RESULTS_DIR / "s06_continuous_cooling_hardness.csv", index=False)
    if not cc.empty:
        print(cc.round(1).to_string(index=False))
    mg = fig_micrographs(sp)
    keep = ["steel", "route", "treat_T_C", "section_mm", "F_best", "P_best", "B_best", "M_best", "S0_um", "d_alpha_um",
            "HB_pred", "HB_lo90", "HB_hi90", "UTS_MPa_pred"]
    mg[keep].to_csv(config.RESULTS_DIR / "s06_micrograph_steels_predictions.csv", index=False)
    print(mg[keep].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
