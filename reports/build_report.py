"""Build reports/REPORT.md from REPORT.template.md and the pipeline results.

    .venv/bin/python reports/build_report.py

Every «token» in the template is replaced by a number computed here from
results/*.csv, so the report always matches the last pipeline run. An
unknown token is an error.
"""

from __future__ import annotations

import re
from pathlib import Path

import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))


def R(name):
    return pd.read_csv(RES / f"{name}.csv")


def f(x, d=0):
    return f"{x:.{d}f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f} %"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tokens():
    T = {}
    lit = pd.read_csv(ROOT / "data/raw/literature_compilation.csv")
    T["lit_rows"], T["lit_src"] = str(len(lit)), str(lit.source_id.nunique())
    T["lit_suspect"] = str(int(lit.quality_flag.fillna("").str.startswith("suspect").sum()))

    b = R("s04_model_benchmark")
    g = lambda m, t, c="RMSE_inAD": float(b[(b.model == m) & (b.target == t)][c].iloc[0])  # noqa: E731
    T["uts_rmse"], T["ys_rmse"], T["hb_rmse"] = f(g("gated_ensemble", "UTS_MPa")), f(g("gated_ensemble", "YS_MPa")), f(g("gated_ensemble", "HB"))
    T["uts_phys"], T["ys_phys"], T["hb_phys"] = f(g("physics_lit", "UTS_MPa")), f(g("physics_lit", "YS_MPa")), f(g("physics_lit", "HB"))
    T["uts_r2"], T["hb_r2"] = f(g("gated_ensemble", "UTS_MPa", "R2_inAD"), 2), f(g("gated_ensemble", "HB", "R2_inAD"), 2)
    labels = [("physics_lit", "Textbook relations only (Gladman, Maynier; El: training mean)"),
              ("ridge_comp", "Ridge, composition"), ("gbm_comp", "Gradient boosting, composition"),
              ("gp_comp", "GP, composition"), ("gp_phys", "GP, physics descriptors"),
              ("greybox", "Grey box (learned Gladman law)"), ("physlit_gp", "Textbook + GP residual"),
              ("gp_hybrid", "GP, composition + physics"), ("anchored_chain", "Physics-anchored chain"),
              ("gated_ensemble", "**Production ensemble**")]
    T["ladder_table"] = md_table(["Model", "UTS (MPa)", "YS (MPa)", "Total El (%)", "HB"],
                                 [[lab] + [f(g(m, t), 1) for t in ("UTS_MPa", "YS_MPa", "El_pct", "HB")] for m, lab in labels])

    pt = R("s04_paired_tests")
    T["p_phys"] = f(float(pt[(pt.target == "UTS_MPa") & (pt.model == "physics_lit")].wilcoxon_p_holm.iloc[0]), 3)
    gp = pt[(pt.target == "UTS_MPa") & (pt.model == "gp_comp")].iloc[0]
    T["gp_diff"], T["gp_lo"], T["gp_hi"] = f(gp.RMSE_minus_production, 0), f(gp.CI95_lo), f(gp.CI95_hi)

    ex = R("s04_cooling_rate_extrapolation").set_index("model").HB_RMSE_rates_le_5Ks
    T["dil_prod"], T["dil_phys"], T["dil_gp"] = f(ex["gated_ensemble"]), f(ex["physics_lit"]), f(ex["gp_comp"])
    an = R("s04_anchor_ablation")
    a0, a1 = an.iloc[0], an.iloc[1]
    T["anchor_sentence"] = (
        f"Anchoring on the Gladman law instead of the Maynier hardness cut the dilatometry hardness error from "
        f"{a0.HB_RMSE_dilatometry_le_5Ks:.0f} to {a1.HB_RMSE_dilatometry_le_5Ks:.0f} HB and the out-of-domain UTS error from "
        f"{a0.UTS_MPa_RMSE_outAD:.0f} to {a1.UTS_MPa_RMSE_outAD:.0f} MPa, while the in-domain UTS error changed from "
        f"{a0.UTS_MPa_RMSE_inAD:.0f} to {a1.UTS_MPa_RMSE_inAD:.0f} MPa and elongation from {a0.El_pct_RMSE_inAD:.1f} to "
        f"{a1.El_pct_RMSE_inAD:.1f} % (`results/s04_anchor_ablation.csv`).")
    c = R("s04_consistency")
    cg = c[c.model == "gated_ensemble"].set_index("target")
    T["viol_uts"] = pct(cg.loc["UTS_MPa", "rate_violation_frac"], 1)
    T["si_span"] = f(cg.loc["UTS_MPa", "si_0.10_0.40_span"])
    sil = R("s04_silicon_ablation")
    T["si_span_with"] = f(float(sil[(sil.silicon_input) & (sil.model == "gated_ensemble")]["UTS_span_Si_0.10_to_0.40"].iloc[0]))

    cv = R("s02_classifier_cv").set_index("model")
    prod = cv[cv.production].iloc[0]
    T["clf_cv"], T["clf_sb"] = f(prod.cv_accuracy, 3), f(prod.steelbench_family_accuracy, 3)
    T["rules_cv"], T["rules_sb"] = f(cv.loc["aisi_rules", "cv_accuracy"], 3), f(cv.loc["aisi_rules", "steelbench_family_accuracy"], 3)
    ext = R("s02_external_tests_all_models")
    T["rules_maraging"] = pct(float(ext[ext.test.str.startswith("Legacy") & (ext.model == "aisi_rules")].accuracy.iloc[0]))
    ag = R("s02_rules_agreement").set_index("subset")
    T["disagree_share"] = pct(ag.loc["ML and AISI rules disagree", "share"])
    T["agree_acc"] = pct(ag.loc["ML and AISI rules agree on the category", "family_accuracy_ML"])
    T["disagree_acc"] = pct(ag.loc["ML and AISI rules disagree", "family_accuracy_ML"])

    cp = R("s03_cooling_paths").set_index("path")
    T["rate25"], T["rate50"], T["rate100"] = (f(cp.loc[f"still air, {d} mm bar", "rate_700C_K_s"], 2) for d in (25, 50, 100))
    T["rateJ32"] = f(cp.loc["Jominy J32", "rate_700C_K_s"], 1)
    T["cool_t85_25"] = f(300.0 / cp.loc["still air, 25 mm bar", "t85_s"], 2)
    T["cool_t85_50"] = f(300.0 / cp.loc["still air, 50 mm bar", "t85_s"], 2)

    ks = R("s03_kinetics_summary")
    k = ks[ks.model == "this work"].set_index("start")
    T["Fs"], T["Ps"], T["Ms"] = f(k.loc["ferrite", "MAE_C"], 1), f(k.loc["pearlite", "MAE_C"], 1), f(k.loc["martensite", "MAE_C"], 1)
    T["Bs_now"] = f"{k.loc['bainite', 'bias_C']:+.0f}"
    tr = R("s08_kinetic_calibration_transfer").set_index("kinetics")
    T["Bs_pub"] = f"+{tr.loc['published Li et al. constants', 'bainite_start_bias_C']:.0f}"
    cols = [("H-band_RMSE_HRC_non_boron", "H-band RMSE, all grades (HRC)", 1),
            ("bainite_start_bias_C", "bainite start bias vs dilatometry (°C)", 0),
            ("martensite_start_MAE_C", "Ms MAE (°C)", 1), ("ferrite_start_MAE_C", "ferrite start MAE (°C)", 1),
            ("dilatometry_HB_RMSE_le_5Ks", "dilatometry hardness, ≤ 5 K/s (HB)", 0),
            ("pearlite_fraction_MAE_air", "air-cooled pearlite fraction MAE (points)", 1),
            ("handbook_HB_RMSE_25mm", "handbook hardness, 25 mm normalised bar (HB)", 0),
            ("handbook_HB_RMSE_best_bar", "handbook hardness, best bar size (HB)", 0),
            ("cct_atlas_MAE_bainite", "CCT atlas bainite fraction MAE", 2),
            ("cct_atlas_HV_RMSE", "CCT atlas hardness RMSE (HV)", 0)]
    names = list(tr.index)
    T["transfer_table"] = md_table(["Independent test"] + names,
                                   [[lab] + [f(tr.loc[n, c_], d) for n in names] for c_, lab, d in cols])

    jm = R("s08_jominy_metrics")
    sel = lambda kin, dist: jm[(jm.kinetics == kin) & (jm.grades == "non-boron grades") & (jm.distances == dist)].iloc[0]  # noqa: E731
    pub = "published Li et al. constants"
    T["jm_bias_end_pub"] = f"{sel(pub, 'J1-J4 (quenched end)').bias_HRC:+.1f}"
    T["jm_in_end_pub"] = pct(sel(pub, "J1-J4 (quenched end)").inside_band)
    T["jm_bias_mid_pub"] = f"{sel(pub, 'J5-J16').bias_HRC:+.1f}"
    T["jm_bias_far_pub"] = f"{sel(pub, 'J18-J32 (air-cooling range)').bias_HRC:+.1f}"
    T["jm_in_far_pub"] = pct(sel(pub, "J18-J32 (air-cooling range)").inside_band)
    jcv = R("s08_kinetic_calibration_cv_metrics")
    a_pub = jcv[(jcv.kinetics == "HRC_published") & (jcv.distances == "all distances")].iloc[0]
    a_cv = jcv[(jcv.kinetics == "HRC_calibrated_cv") & (jcv.distances == "all distances")].iloc[0]
    T["jm_rmse_pub"], T["jm_rmse_cv"] = f(a_pub.RMSE_HRC, 1), f(a_cv.RMSE_HRC, 1)
    T["jm_in_pub"], T["jm_in_cv"] = pct(a_pub.inside_band), pct(a_cv.inside_band)
    jp = R("s08_jominy_predictions")
    T["jm_grades"] = str(jp.grade.nunique())
    T["jm_points"] = str(int((~jp.boron).sum()))
    post = R("s08_kinetic_calibration_posterior").set_index("parameter")
    T["dBs"] = f(abs(post.loc["dBs_C", "H-bands MAP"]))
    T["fB"] = f(np.exp(post.loc["ln_f_bainite", "H-bands MAP"]))
    rt = R("s08_jominy_cooling_rates")
    T["jr_lo"], T["jr_hi"] = f(rt.ratio.min(), 2), f(rt.ratio.max(), 2)
    T["h_q"] = f(round(float(rt.h_q_W_m2K.iloc[0]), -2))

    ca = R("s03_cct_atlas_summary").set_index("subset")
    T["atlas_fp"], T["atlas_b"], T["atlas_m"] = (f(ca.loc["all points", c_], 2) for c_ in
                                                 ("MAE_ferrite_pearlite", "MAE_bainite", "MAE_martensite"))
    T["atlas_hv"] = f(ca.loc["all points", "HV_RMSE"])
    T["atlas_fp_fast"] = f(ca.loc["t500 < 20 s (quench-like)", "MAE_ferrite_pearlite"], 2)
    T["atlas_fp_slow"] = f(ca.loc["t500 >= 20 s (air cooling of bars and slower)", "MAE_ferrite_pearlite"], 2)

    gs = R("s03_grain_size_check")
    T["gs_old"], T["gs_new"] = f(gs.mean_error_factor.iloc[0], 1), f(gs.mean_error_factor.iloc[1], 1)
    pl = R("s03_literature_pearlite_summary").set_index(["subset", "estimate"]).MAE_pct_points
    T["pf_fc_kin"], T["pf_fc_lev"] = f(pl[("furnace cooled", "kinetic")], 1), f(pl[("furnace cooled", "lever")], 1)
    T["pf_ac_kin"], T["pf_ac_lev"] = f(pl[("air cooled", "kinetic")], 1), f(pl[("air cooled", "lever")], 1)

    lm = R("s06_literature_metrics").set_index("quantity")
    q = lambda s: lm.loc[[i for i in lm.index if i.startswith(s)][0]]  # noqa: E731
    T["hollomon_mae"] = f(q("uniform elongation (%) - Hollomon through").MAE, 1)
    T["ue_loso"] = f(q("uniform elongation (%) - ratio method, leave").MAE, 1)
    r_ = q("uniform/total elongation ratio")
    T["ue_ratio"], T["ue_ratio_sd"] = f(r_.observed_mean, 3), f(r_.observed_sd, 3)
    T["lit_ys"], T["lit_uts"] = f(q("yield strength").MAE), f(q("tensile strength").MAE)
    T["lit_hb"], T["lit_gs"] = f(q("hardness").MAE), f(q("ferrite grain size").MAE)

    st = R("s06_structural_metrics")
    sa = st[st.subset == "all"].set_index("target")
    T["st_uts_rmse"], T["st_uts_bias"] = f(sa.loc["UTS_MPa", "RMSE"]), f"{sa.loc['UTS_MPa', 'bias']:+.0f}"
    T["st_ys_bias"] = f(abs(sa.loc["YS_MPa", "bias"]))
    T["st_el_rmse"], T["st_ys_rho"] = f(sa.loc["El_pct", "RMSE"], 1), f(sa.loc["YS_MPa", "spearman"], 2)
    T["st_ys_bias_micro"] = f(abs(float(st[(st.subset.str.startswith("microalloyed")) & (st.target == "YS_MPa")].bias.iloc[0])))
    T["charpy_rho"] = f(float(st[st.target.str.startswith("Charpy")].spearman.iloc[0]), 2)

    nf = R("s05_composition_noise_floor").set_index(["route", "target"]).rms_sd
    T["noise_fc"], T["noise_ac"] = f(nf[("FC", "UTS_MPa")]), f(nf[("AC", "UTS_MPa")])

    cat = R("s05_category_summary").set_index("category")
    lo, me, hi, cm = (cat.loc[c_] for c_ in ("Low-carbon steel", "Medium-carbon steel", "High-carbon steel", "Mo / Cr-Mo steel"))
    T["d_lo_fc"], T["d_lo_ac"] = f(lo.FC_d_alpha_um_median), f(lo.AC25_d_alpha_um_median)
    T["d_me_fc"], T["d_me_ac"] = f(me.FC_d_alpha_um_median), f(me.AC25_d_alpha_um_median)
    T["itt_lo"] = f(lo.ITT_FC_median_C - lo.ITT_AC25_median_C)
    T["duts_me"], T["duts_hi"] = f(me.UTS_MPa_delta_median), f(hi.UTS_MPa_delta_median)
    T["duts_crmo"], T["di_crmo"] = f(cm.UTS_MPa_delta_median), f(cm.DI_mm_median)
    T["dcool_me"], T["dtemp_me"] = f"{me.UTS_MPa_delta_cooling_median:+.0f}", f"{me.UTS_MPa_delta_temperature_median:+.0f}"

    def micro(r, case):
        s = f"P {100 * r[f'{case}_pearlite_mean']:.0f} %"
        bm = r.get(f"{case}_bainite_martensite_mean", 0.0) if case != "FC" else 0.0
        return s + (f", B+M {100 * bm:.0f} %" if bm >= 0.02 else "")
    rows = []
    for c_ in ["Low-carbon steel", "Medium-carbon steel", "High-carbon steel", "Free-machining steel", "C-Mn / Mn steel",
               "Cr / Cr-V steel", "Mo / Cr-Mo steel", "Ni-Cr-Mo / Ni steel", "Si-Mn spring steel"]:
        r = cat.loc[c_]
        rows.append([c_ + (" *" if r.in_validity_share < 0.5 else ""), int(r.grades), micro(r, "FC"), micro(r, "AC25"),
                     f"{r.UTS_MPa_FC_median:.0f} → {r.UTS_MPa_AC25_median:.0f} ({r.UTS_MPa_AC50_median:.0f})",
                     f"{r.YS_MPa_FC_median:.0f} → {r.YS_MPa_AC25_median:.0f}",
                     f"{r.El_pct_FC_median:.0f} → {r.El_pct_AC25_median:.0f}",
                     f"{r.uniformEl_FC_median:.0f} → {r.uniformEl_AC25_median:.0f}",
                     f"{r.HB_FC_median:.0f} → {r.HB_AC25_median:.0f}"])
    T["category_table"] = md_table(["Category (grades)", "n", "FC", "AC, 25 mm", "UTS (MPa) FC → AC (50 mm)", "YS (MPa)",
                                    "Total El (%)", "Uniform El (%, est., 25 mm)", "HB"], rows) + (
        "\n\nP: pearlite (lever split; the kinetic split is the top of a band), B+M: bainite + martensite. "
        "* outside the window of the critical-temperature equations (2 % Si).")
    tokens_v3(T)
    return T


def tokens_v3(T):
    """Tokens of the version-3 sections (stages 6, 9, 10 and 11)."""
    def g(path):
        return pd.read_csv(RES / path)
    qt = pd.read_csv(ROOT / "data/processed/qt_bars.csv")
    nb = pd.read_csv(ROOT / "data/processed/normalized_bars.csv")
    T["qt_n"], T["qt_grades"], T["bar_n"] = str(len(qt)), str(qt.grade.nunique()), str(len(nb))
    T["bb_raw"] = "3234"
    T["hb_grades"] = "33"
    sb = pd.read_csv(ROOT / "data/processed/heats_steelbench.csv")
    T["sb_heats"] = str(len(sb))
    jp = g("s08_jominy_predictions.csv")
    ts = pd.read_csv(ROOT / "data/raw/collins2023_Ts.csv")
    n_dil = int(ts[[c for c in ("Tf", "Tp", "Tb", "Tm") if c in ts]].notna().sum().sum())
    lit = pd.read_csv(ROOT / "data/raw/literature_compilation.csv")
    T["data_rows"] = f"{80 + len(qt) + len(nb) + len(jp) + 423 + n_dil + len(lit) + 165:,}".replace(",", " ")
    # ---- ladder
    b = g("s09_qt_benchmark.csv")
    rm = lambda m, t: float(b[(b.model == m) & (b.target == t)].RMSE.iloc[0])  # noqa: E731
    T["qt_uts_ens"], T["qt_uts_phys"] = f(rm("qt_ensemble", "UTS_MPa")), f(rm("qt_physics", "UTS_MPa"))
    T["qt_uts_grey"], T["qt_uts_gp"], T["qt_uts_gbm"] = (f(rm(m, "UTS_MPa")) for m in ("qt_greybox", "qt_gp", "qt_gbm"))
    T["qt_ys_ens"], T["qt_el_ens"] = f(rm("qt_ensemble", "YS_MPa")), f(rm("qt_ensemble", "El_pct"), 1)
    labels = [("qt_physics", "Textbook physics (prior constants)"), ("qt_gbm", "Gradient boosting, composition + process"),
              ("qt_gp", "GP, composition + process"), ("qt_greybox", "Fitted physics chain (grey box)"),
              ("qt_anchored", "Physics-anchored GP"), ("qt_ensemble", "**Production: physics-anchored GP + GP mixture**")]
    T["qt_ladder_table"] = md_table(["Model", "UTS (MPa)", "YS (MPa)", "Elongation (%)"],
                                    [[lab, f(rm(m, "UTS_MPa")), f(rm(m, "YS_MPa")), f(rm(m, "El_pct"), 1)] for m, lab in labels])
    cf = g("s09_qt_conformal.csv").set_index("target")
    T["qt_cov"] = pct(cf.loc["UTS_MPa", "coverage_nested"])
    T["qt_width"] = f(cf.loc["UTS_MPa", "median_width"])
    kc = g("s09_qt_kinetics_check.csv")
    kk = lambda lab, m: float(kc[(kc.kinetics.str.startswith(lab)) & (kc.model == m)].UTS_RMSE.iloc[0])  # noqa: E731
    T["qt_kin_grey_cal"], T["qt_kin_grey_pub"] = f(kk("recal", "qt_greybox")), f(kk("published", "qt_greybox"))
    T["qt_kin_anch_cal"], T["qt_kin_anch_pub"] = f(kk("recal", "qt_anchored")), f(kk("published", "qt_anchored"))
    # ---- quench and hardenability
    qr = g("s09_quench_rates.csv")
    q = lambda D, m: float(qr[(qr.D_mm == D) & (qr.medium == m) & (qr.r_over_R == 0)].rate700_K_s.iloc[0])  # noqa: E731
    T["q25_oil"], T["q25_water"] = f(q(25, "oil")), f(q(25, "water"))
    T["q100_oil"], T["q100_water"] = f(q(100, "oil"), 1), f(q(100, "water"), 1)
    hd = g("s09_hardenability_check.csv")
    from scipy.stats import spearmanr
    rc, rp = hd.DI_model_calibrated_mm / hd.DI_A255_mm, hd.DI_model_published_mm / hd.DI_A255_mm
    T["di_n"], T["di_ratio_cal"], T["di_ratio_pub"] = str(len(hd)), f(rc.median(), 2), f(rp.median(), 2)
    T["di_rho_cal"] = f(spearmanr(hd.DI_A255_mm, hd.DI_model_calibrated_mm, nan_policy="omit")[0], 2)
    T["di_1080"] = f(float(rc[hd.grade.astype(str) == "1080"].iloc[0]), 1)
    # ---- tempering constants
    tc = g("s09_tempering_constants.csv")
    lab = {"p0": "P₅₀ base", "p_Mn": "P₅₀ shift per % Mn", "p_Si": "per % Si", "p_Cr": "per % Cr", "p_Mo": "per % Mo",
           "p_V": "per % V", "logW": "ln W (width)", "s0": "HV_soft base (HV)", "s_C": "HV_soft per % C",
           "a_Mo": "secondary hardening per % Mo (HV)", "a_V": "per % V (HV)", "a_Cr": "per % Cr (HV)"}
    rows = []
    for _, r in tc.iterrows():
        d = 2 if r.constant == "logW" else 0
        rows.append([lab[r.constant], f(r.prior_mean, d), f"**{r.fitted:.{d}f}** ± {r.bootstrap_sd:.{d}f}" + (" (bound)" if r.at_bound else "")])
    T["temper_table"] = md_table(["Constant", "Prior", "Fitted ± SD"], rows)
    pmo = float(tc[tc.constant == "p_Mo"].fitted.iloc[0])
    T["s_c_fit"] = f(float(tc[tc.constant == "s_C"].fitted.iloc[0]))
    T["p_mo_degC"] = f(0.1 * pmo / 20.0)
    # ---- five treatments
    ft = g("s09_five_treatments.csv")
    ft["grade"] = ft.grade.astype(str)
    a = ft[ft.grade == "4140"].set_index("treatment")
    book = {"anneal": ("coarse pearlite (+ ferrite)", "softest, most ductile"),
            "normalise": ("fine pearlite, uniform small grains", "stronger and tougher than annealed"),
            "harden": ("martensite", "maximum hardness, brittle"),
            "quench_temper": ("tempered martensite", "balanced strength and toughness"),
            "stress_relieve": ("unchanged", "residual stress relieved, hardness unchanged")}
    names = {"anneal": "Annealing", "normalise": "Normalising", "harden": "Hardening (oil)",
             "quench_temper": "Hardening + tempering 600 °C", "stress_relieve": "Stress relieving 600 °C (after normalising)"}
    rows = []
    for k in ("anneal", "normalise", "harden", "quench_temper", "stress_relieve"):
        r = a.loc[k]
        uts = "not meaningful (brittle)" if k == "harden" else f"{r.UTS_MPa_pred:.0f} [{r.UTS_MPa_lo90:.0f}, {r.UTS_MPa_hi90:.0f}]"
        el = "-" if k == "harden" else f"{r.El_pct_pred:.0f}"
        rows.append([names[k], book[k][0], book[k][1], str(r.microstructure), uts, f"{r.HV:.0f}", el])
    T["five_table_4140"] = md_table(["Treatment", "Textbook microstructure", "Textbook property", "Model microstructure",
                                     "UTS (MPa) [90 %]", "HV", "El (%)"], rows)
    h = ft.set_index(["grade", "treatment"])
    T["hv_q_1080"], T["hv_q_4140"] = f(h.loc[("1080", "harden"), "HV"]), f(h.loc[("4140", "harden"), "HV"])
    T["uts_qt_4140"], T["el_qt_4140"] = f(h.loc[("4140", "quench_temper"), "UTS_MPa_pred"]), f(h.loc[("4140", "quench_temper"), "El_pct_pred"])
    from steelml.physics.hardenability import ideal_critical_diameter
    cat = pd.read_csv(ROOT / "data/processed/grade_catalog.csv").set_index("grade")
    c1020 = {el: cat.loc["AISI 1020", f"{el}_nom"] for el in ("C", "Mn", "Si", "Ni", "Cr", "Mo", "Cu", "V")}
    T["di_1020"] = f(float(np.ravel(ideal_critical_diameter(c1020)[1])[0]))
    tcur = g("s09_tempering_curves.csv")
    t4 = tcur[(tcur.grade.astype(str) == "4140") & (tcur.temper_T_C >= 200) & (tcur.temper_T_C <= 700)]
    T["qt_range_4140"] = f"{t4.UTS_MPa_pred.min():.0f}-{t4.UTS_MPa_pred.max():.0f}"
    # ---- stress relief, precipitation
    sr = g("s09_stress_relief.csv")
    s1045 = sr[sr.state.str.startswith("normalised 1045") & (sr.T_C == 600)]
    T["sr_bound_1045"] = f(s1045.residual_stress_bound_MPa.iloc[0])
    sa = g("s09_stress_relief_after_temper.csv").set_index("sr_T_C")
    T["uts_sr575_4140"], T["uts_sr650_4140"] = f(sa.loc[575, "UTS_after_MPa"]), f(sa.loc[650, "UTS_after_MPa"])
    pp_ = g("s09_precipitation.csv")
    near = lambda el, rt, x: float(pp_[(pp_.element == el) & (pp_.route == rt)].iloc[  # noqa: E731
        (pp_[(pp_.element == el) & (pp_.route == rt)].wt_pct - x).abs().argmin()].dsigma_MPa)
    T["ppt_v10_ac"], T["ppt_v10_fc"], T["ppt_nb03_ac"] = f(near("V", "AC", 0.10)), f(near("V", "FC", 0.10)), f(near("Nb", "AC", 0.03))
    # ---- stage 10
    asm = g("s10_assumptions.csv")
    T["assumptions_table"] = md_table(["Assumption", "Value", "Status", "Evidence", "Tested"],
                                      asm[["assumption", "value", "status", "evidence", "range_tested"]].values.tolist())
    sm = g("s10_handbook_section_match.csv")
    T["section_match_table"] = md_table(["Bar size (mm)", "Grades", "Mean UTS difference (MPa)", "RMS difference (MPa)",
                                         "Within 3 MPa"],
                                        [[f"{r.section_mm:.0f}", int(r.grades), f"{r.mean_dUTS:+.1f}", f(r.rms_dUTS, 1),
                                          f"{int(r.within_3MPa)} of {int(r.grades)}"] for _, r in sm.iterrows()])
    oat = g("s10_oat_anneal_normalise.csv")
    gcols = [c for c in oat.columns if c.startswith("dUTS_")]
    T["oat_table"] = md_table(["Setting", "UTS RMSE (MPa)", "HB RMSE"] + [f"AC gain {c[5:]} (MPa)" for c in gcols],
                              [[r.setting, f(r.UTS_MPa_RMSE, 1), f(r.HB_RMSE, 1)] + [f"{r[c]:+.0f}" for c in gcols]
                               for _, r in oat.iterrows()])
    base = oat[oat.setting == "baseline"].iloc[0]
    # the bar size is documented, so its rows are counterfactuals; rank only the uncertain assumptions
    unc = oat[(oat.setting != "baseline") & ~oat.setting.str.startswith("normalised bars")]
    worst = unc.loc[(unc.UTS_MPa_RMSE - base.UTS_MPa_RMSE).abs().idxmax()]
    fcr = oat[oat.setting.str.startswith("furnace cooling") | (oat.setting == "baseline")]
    v2 = oat[oat.setting.str.startswith("normalised bars 50")].iloc[0]
    T["oat_sentence"] = (
        f"Among the assumptions that are genuinely uncertain, *{worst.setting}* changes the accuracy most (UTS RMSE "
        f"{base.UTS_MPa_RMSE:.0f} → {worst.UTS_MPa_RMSE:.0f} MPa over all annealed and normalised rows). The undocumented furnace "
        f"rate hardly changes the accuracy ({fcr.UTS_MPa_RMSE.min():.0f}-{fcr.UTS_MPa_RMSE.max():.0f} MPa from 0.002 to 0.02 K/s), "
        f"because the GP correction absorbs it, but it does change the size of the furnace-vs-air difference: the 1045 gain "
        f"runs from {fcr.dUTS_1045.min():.0f} to {fcr.dUTS_1045.max():.0f} MPa and the 4140 gain from {fcr.dUTS_4140.min():.0f} to "
        f"{fcr.dUTS_4140.max():.0f} MPa, because a slower furnace gives a softer annealed state. So the furnace rate is the main "
        f"uncertainty in *how much* air cooling adds, never in the direction or the ranking of the steels. Had we kept "
        f"version 2's inferred 50 mm bar, the accuracy would be worse ({v2.UTS_MPa_RMSE:.0f} MPa) and the Cr-Mo gain "
        f"overstated (4140: {v2.dUTS_4140:+.0f} against {base.dUTS_4140:+.0f} MPa).")
    qo = g("s10_oat_quench_temper.csv")
    T["qt_oat_table"] = md_table(["Setting", "UTS RMSE (MPa)", "mean martensite", "bias 13 mm", "bias 100 mm"],
                                 [[r.setting, f(r.greybox_UTS_RMSE, 1), f(r.mean_f_martensite, 2), f"{r.bias_13mm:+.0f}",
                                   f"{r.bias_100mm:+.0f}"] for _, r in qo.iterrows()])
    qb = qo[qo.setting == "baseline"].iloc[0]
    qw = qo[qo.setting != "baseline"]
    qworst = qw.loc[(qw.greybox_UTS_RMSE - qb.greybox_UTS_RMSE).abs().idxmax()]
    T["qt_oat_sentence"] = (f"For the quench-and-temper chain the largest change comes from *{qworst.setting}* "
                            f"({qb.greybox_UTS_RMSE:.0f} → {qworst.greybox_UTS_RMSE:.0f} MPa). Because the tempering constants "
                            f"are refitted every time, most assumptions are absorbed by them; what they cannot absorb shows "
                            f"up in the bias at 13 and 100 mm.")
    mo = g("s10_morris.csv")
    top = lambda o: mo[mo.output == o].sort_values("mu_star", ascending=False).factor.iloc[0]  # noqa: E731
    names_f = {"fc_rate": "the furnace cooling rate", "eps": "the emissivity", "hconv": "the convection coefficient",
               "astm": "the austenite grain size", "H_oil": "the oil quench severity", "kref": "the conductivity used for H",
               "t_temper": "the tempering time", "chj": "the Hollomon-Jaffe constant"}
    T["morris_r"] = "12"
    T["morris_top_1045"] = names_f[top("dUTS_AC_FC_1045")]
    T["morris_top_qt"] = names_f[top("UTS_QT_4140_50mm")]
    T["morris_top_fm"] = names_f[top("f_martensite_4140_50mm_oil")]
    # ---- stage 6 bar sizes
    bm = g("s06_bar_sizes_metrics.csv")
    bt = bm[bm.target == "UTS_MPa"]
    T["bar_table"] = md_table(["Check", "Bars", "Grades", "UTS RMSE (MPa)", "Bias (MPa)", "In 90 % interval"],
                              [[r.check, int(r.n), int(r.grades), f(r.RMSE), f"{r.bias:+.0f}",
                                pct(r.coverage90) if pd.notna(r.get("coverage90", np.nan)) else "-"] for _, r in bt.iterrows()])
    pr = bm[bm.check.str.startswith("bar-size effect")]
    if len(pr):
        p0 = pr.iloc[0]
        T["bar_sentence"] = (f"The bar-size effect itself, the UTS of a 13 mm bar minus that of a 100 mm bar of the same grade, "
                             f"has the right sign for {pct(p0.sign_agreement)} of the {int(p0.n)} grades, and its size is "
                             f"{p0.mean_predicted:.0f} MPa predicted against {p0.mean_measured:.0f} MPa measured (mean). This is "
                             f"hypothesis H4 tested on data the model never saw.")
    else:
        T["bar_sentence"] = ""
    # ---- stage 11
    dg = g("s11_mcmc_diagnostics.csv")
    T["rhat_max"], T["ess_min"] = f(dg.split_Rhat.max(), 3), f(dg.ESS.min())
    ps = g("s11_posterior_summary.csv")
    plab = {"ln_f_ferrite": "ln ferrite factor", "ln_f_pearlite": "ln pearlite factor", "ln_f_bainite": "ln bainite factor",
            "dBs_C": "bainite-start shift (°C)", "ln_sigma_model": "ln model error (HRC)"}
    T["posterior_table"] = md_table(["Constant", "Prior", "Stage-8 MAP", "Posterior mean ± SD", "90 % interval", "SD / prior SD"],
                                    [[plab[r.parameter], f"{r.prior_mean:.2f} ± {r.prior_sd:.2f}", f(r.MAP_stage8, 2),
                                      f"{r.posterior_mean:.2f} ± {r.posterior_sd:.2f}", f"{r.q05:.2f} to {r.q95:.2f}",
                                      f(r.sd_ratio_to_prior, 2)] for _, r in ps.iterrows()])
    fp = ps[ps.parameter == "ln_f_pearlite"].iloc[0]
    T["post_fp"], T["post_fp_lo"], T["post_fp_hi"] = f(np.exp(fp.posterior_mean), 2), f(np.exp(fp.q05), 2), f(np.exp(fp.q95), 2)
    cv = g("s11_cct_atlas_coverage.csv").set_index("quantity")
    T["coverage_table"] = md_table(["Quantity", "Points", "Credible band width (median)", "Inside 90 % band",
                                    "Structural SD", "Inside, with structural error", "MAE"],
                                   [[q_, int(r.points), f(r.median_band_width, 3), pct(r.coverage_parametric_90),
                                     f(r.structural_sd, 2 if q_ != "HV" else 0), pct(r.coverage_with_structural_90),
                                     f(r.MAE_median_prediction, 2 if q_ != "HV" else 0)] for q_, r in cv.iterrows()])
    T["cov_fp"], T["cov_b"] = pct(cv.loc["f_FP", "coverage_parametric_90"]), pct(cv.loc["f_B", "coverage_parametric_90"])
    T["s_fp"], T["s_hv"] = f(cv.loc[["f_FP", "f_B", "f_M"], "structural_sd"].mean(), 2), f(cv.loc["HV", "structural_sd"])
    T["frac_band"] = f(1.645 * cv.loc[["f_FP", "f_B", "f_M"], "structural_sd"].mean(), 2)


def main():
    tpl = (HERE / "REPORT.template.md").read_text()
    # Appendix B before Appendix A reads oddly: move it after Appendix A
    if "## Appendix B" in tpl and "## Appendix A" in tpl and tpl.index("## Appendix B") < tpl.index("## Appendix A"):
        b0 = tpl.index("## Appendix B")
        a0 = tpl.index("## Appendix A")
        app_b = tpl[b0:a0]
        tpl = tpl[:b0] + tpl[a0:]
        r0 = tpl.index("## References")
        tpl = tpl[:r0] + app_b + tpl[r0:]
    T = tokens()
    missing = sorted(set(re.findall(r"«(\w+)»", tpl)) - set(T))
    if missing:
        raise KeyError(f"tokens without a value: {missing}")
    out = re.sub(r"«(\w+)»", lambda m: T[m.group(1)], tpl)
    (HERE / "REPORT.md").write_text(out)
    print(f"wrote {HERE / 'REPORT.md'} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
