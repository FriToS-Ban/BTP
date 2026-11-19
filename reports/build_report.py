"""Build reports/REPORT.md from REPORT.template.md and the pipeline results.

    .venv/bin/python reports/build_report.py

Every «token» in the template is replaced by a number computed here from
results/*.csv, so the report always matches the last pipeline run. An
unknown token is an error.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"


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
    T["duts_me"], T["duts_hi"] = f(me.UTS_MPa_delta50_median), f(hi.UTS_MPa_delta50_median)
    T["duts_crmo"], T["di_crmo"] = f(cm.UTS_MPa_delta50_median), f(cm.DI_mm_median)
    T["dcool_me"], T["dtemp_me"] = f"{me.UTS_MPa_delta_cooling_median:+.0f}", f"{me.UTS_MPa_delta_temperature_median:+.0f}"

    def micro(r, case):
        s = f"P {100 * r[f'{case}_pearlite_mean']:.0f} %"
        bm = r.get(f"{case}_bainite_martensite_mean", 0.0) if case != "FC" else 0.0
        return s + (f", B+M {100 * bm:.0f} %" if bm >= 0.02 else "")
    rows = []
    for c_ in ["Low-carbon steel", "Medium-carbon steel", "High-carbon steel", "Free-machining steel", "C-Mn / Mn steel",
               "Cr / Cr-V steel", "Mo / Cr-Mo steel", "Ni-Cr-Mo / Ni steel", "Si-Mn spring steel"]:
        r = cat.loc[c_]
        rows.append([c_ + (" *" if r.in_validity_share < 0.5 else ""), int(r.grades), micro(r, "FC"), micro(r, "AC50"),
                     f"{r.UTS_MPa_FC_median:.0f} → {r.UTS_MPa_AC50_median:.0f} ({r.UTS_MPa_AC25_median:.0f})",
                     f"{r.YS_MPa_FC_median:.0f} → {r.YS_MPa_AC50_median:.0f}",
                     f"{r.El_pct_FC_median:.0f} → {r.El_pct_AC50_median:.0f}",
                     f"{r.uniformEl_FC_median:.0f} → {r.uniformEl_AC25_median:.0f}",
                     f"{r.HB_FC_median:.0f} → {r.HB_AC50_median:.0f}"])
    T["category_table"] = md_table(["Category (grades)", "n", "FC", "AC, 50 mm", "UTS (MPa) FC → AC (25 mm)", "YS (MPa)",
                                    "Total El (%)", "Uniform El (%, est., 25 mm)", "HB"], rows) + (
        "\n\nP: pearlite (lever split; the kinetic split is the top of a band), B+M: bainite + martensite. "
        "* outside the window of the critical-temperature equations (2 % Si).")
    return T


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
