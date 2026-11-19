"""Result, use and closing slides of the deck (numbers from results/)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from deck_lib import img, stat, table

CAT_ORDER = ["Low-carbon steel", "Medium-carbon steel", "High-carbon steel", "Free-machining steel", "C-Mn / Mn steel",
             "Cr / Cr-V steel", "Mo / Cr-Mo steel", "Ni-Cr-Mo / Ni steel", "Si-Mn spring steel"]


def bench(N, model, target, col="RMSE_inAD"):
    b = N["bench"]
    return float(b[(b.model == model) & (b.target == target)][col].iloc[0])


def micro(row, case):
    p = row[f"{case}_pearlite_mean"] if f"{case}_pearlite_mean" in row else np.nan
    bm = row.get(f"{case}_bainite_martensite_mean", 0.0) if case != "FC" else 0.0
    s = f"P {100 * p:.0f} %"
    if bm >= 0.02:
        s += f", B+M {100 * bm:.0f} %"
    return s


def slides_results(d, N):
    cat = N["cat"]

    # spacing and grain size
    gs = N["gs"]
    hg, used = gs.iloc[0], gs.iloc[1]
    lo = cat.loc["Low-carbon steel"]
    me = cat.loc["Medium-carbon steel"]
    body = (f'<div class="row"><div class="col-3">{img("fig21_literature_validation.png", height=500)}</div>'
            '<div class="col-2 small">'
            "<h3>Pearlite spacing</h3><p>S<sub>0</sub> = 3.33 ΔT<sup>-0.607</sup> µm, fitted on 14 measured continuous-cooling "
            "spacings from 4 papers; a factor of 1.7 when each paper is predicted from the others. The isothermal "
            "Marder-Bramfitt 8.02/ΔT is about 3 times too fine for air cooling, because the model forms that pearlite too "
            "cold; the exponent is a correction, not a new law.</p>"
            "<h3>Ferrite grain size</h3><p>Hodgson-Gibbs (hot strip) put every medium-carbon steel on a 3 µm floor. Using its "
            f"low-carbon branch with Ceq held at 0.35 brings the typical error from a factor of {hg.mean_error_factor:.1f} to "
            f"{used.mean_error_factor:.1f} on {int(used.n)} measured grains, with nothing fitted.</p>"
            f"<p>Normalising refines the ferrite: {lo.FC_d_alpha_um_median:.0f} → {lo.AC25_d_alpha_um_median:.0f} µm for "
            f"low-carbon and {me.FC_d_alpha_um_median:.0f} → {me.AC25_d_alpha_um_median:.0f} µm for medium-carbon steels "
            "(25 mm bar).</p></div></div>")
    d.add(body, title="Pearlite spacing and ferrite grain size come out within a factor of about 2",
          kicker="Microstructure: spacing and grain size")

    # hardenability
    di = N["di"]
    rows = [[f"{r.DI_mm_from:.0f}-{r.DI_mm_to:.0f} mm" if r.DI_mm_to < 999 else f"&gt; {r.DI_mm_from:.0f} mm",
             f"{int(r.grades)}", f"{100 * r.share_over_50pct_B_M_AC25:.0f} %", f"+{r.median_dUTS_AC25_MPa:.0f}"]
            for r in di.itertuples()]
    body = (f'<div class="row"><div class="col-3">{img("fig25_hardenability_vs_air_cooling.png", height=430)}</div>'
            '<div class="col-2 small">'
            + table(["Ideal critical diameter D<sub>I</sub>", "grades", "25 mm bars with &gt;50 % B+M", "median ΔUTS (MPa)"],
                    rows, num_cols=(1, 2, 3))
            + "<p style='margin-top:10px'>D<sub>I</sub> (Grossmann, ASTM A255 factors) is computed from the composition "
              "alone, independently of the kinetic model, and orders the steels the same way the model does. It explains "
              "the category results: the strength gain on air cooling is a hardenability question.</p></div></div>")
    d.add(body, title="Hardenability decides what air cooling forms, and how much strength it adds (H3)",
          kicker="Hardenability")

    # CCT atlas
    ca = pd.read_csv(N["res"] / "s03_cct_atlas_summary.csv").set_index("subset")
    a, sl = ca.loc["all points"], ca.loc["t500 >= 20 s (air cooling of bars and slower)"]
    d.add(img("fig28_cct_atlas_validation.png", height=380)
          + f'<p class="small" style="margin-top:10px">423 measured continuous-cooling points of 19 low-alloy steels (Rose &amp; '
            f"Hougardy atlas, open CC BY tables). Ferrite + pearlite within {a.MAE_ferrite_pearlite:.2f}, bainite "
            f"{a.MAE_bainite:.2f}, martensite {a.MAE_martensite:.2f} (MAE, fractions); hardness {a.HV_RMSE:.0f} HV (RMSE). Good "
            f"for quench-like cooling, weaker at air-cooling times (ferrite + pearlite {sl.MAE_ferrite_pearlite:.2f}), where "
            "bainite competes.</p>",
          title="The kinetics against an independent CCT atlas: 19 steels, 423 measured points",
          kicker="Transformation kinetics")

    # Jominy and calibration transfer
    tr = pd.read_csv(N["res"] / "s08_kinetic_calibration_transfer.csv").set_index("kinetics")
    cvm = pd.read_csv(N["res"] / "s08_kinetic_calibration_cv_metrics.csv")
    ap = cvm[(cvm.kinetics == "HRC_published") & (cvm.distances == "all distances")].iloc[0]
    ac = cvm[(cvm.kinetics == "HRC_calibrated_cv") & (cvm.distances == "all distances")].iloc[0]
    P, C = "published Li et al. constants", "calibrated on the H-bands"
    rows = [[lab, f"{tr.loc[P, c]:{fm}}", f"{tr.loc[C, c]:{fm}}"] for c, lab, fm in (
        ("bainite_start_bias_C", "bainite start vs dilatometry (°C)", "+.0f"),
        ("martensite_start_MAE_C", "Ms MAE (°C)", ".1f"),
        ("handbook_HB_RMSE_25mm", "handbook HB, 25 mm normalised bar", ".0f"),
        ("cct_atlas_HV_RMSE", "CCT atlas hardness (HV)", ".0f"),
        ("pearlite_fraction_MAE_air", "air-cooled pearlite fraction (points)", ".1f"),
        ("dilatometry_HB_RMSE_le_5Ks", "dilatometry hardness (HB)", ".0f"))]
    body = (f'<div class="row"><div class="col-3">{img("fig29_jominy_hbands.png", height=520)}</div>'
            '<div class="col-2 small">'
            f"<p>Against the SAE H-bands of 80 grades the published kinetics are right at the quenched end but too "
            f"hardenable further along. A Bayesian calibration (Bs lower, faster bainite) cut the band error from "
            f"{ap.RMSE_HRC:.1f} to {ac.RMSE_HRC:.1f} HRC on grades it never saw ({100 * ap.inside_band:.0f} → "
            f"{100 * ac.inside_band:.0f} % inside the band).</p><p><b>Does it transfer to data it never saw?</b></p>"
            + table(["Independent test", "published", "calibrated"], rows, num_cols=(1, 2))
            + "<p class='tiny' style='margin-top:6px'>Yes, except the very rich SA-540 (3.5Ni-1.7Cr), which it makes too "
              "bainitic; adopted with that limit stated.</p></div></div>")
    d.add(body, title="80 Jominy bands: the published kinetics were too hardenable, and the fix transfers",
          kicker="Hardenability: calibration")

    # micrographs
    d.add(f'<div class="row"><div class="col-3">{img("fig30_micrographs_vs_model.png", height=560)}</div>'
          '<div class="col-2 small"><p>Published micrographs of the same steels after both routes, with the model\'s '
          "prediction for that composition, temperature and bar size underneath.</p><p>Annealed: ferrite + pearlite, as "
          "predicted. Normalised 1045: finer ferrite + pearlite. Normalised 4140: a fine acicular structure, consistent with "
          "the bainite the model predicts (the authors call it Widmanstätten ferrite and fine pearlite; reading it as "
          "bainite is ours).</p><p class='tiny'>Micrographs: Carvajal et al. (2017), Materials Research 20 (Suppl. 2) "
          "347-352, CC BY 4.0.</p></div></div>",
          title="What the micrographs show: same phases for 1045, a different steel for 4140", kicker="Microstructure")

    # property ladder
    rows = []
    labels = [("physics_lit", "Textbook relations only (Gladman, Maynier)"), ("ridge_comp", "Ridge regression, composition"),
              ("gp_comp", "Gaussian process, composition"), ("physlit_gp", "Textbook + GP residual"),
              ("anchored_chain", "Physics-anchored chain"), ("gated_ensemble", "Production ensemble")]
    for m, lab in labels:
        rows.append([lab] + [f"{bench(N, m, t):.1f}" for t in ("UTS_MPa", "YS_MPa", "El_pct", "HB")])
    pt = N["paired"]
    p_phys = float(pt[(pt.target == "UTS_MPa") & (pt.model == "physics_lit")].wilcoxon_p_holm.iloc[0])
    gp = pt[(pt.target == "UTS_MPa") & (pt.model == "gp_comp")].iloc[0]
    body = (f'<div class="row"><div class="col-2">{img("fig12_model_ladder.png", height=470)}</div><div class="col-2 small">'
            + table(["Model (leave-one-grade-out RMSE)", "UTS", "YS", "El %", "HB"], rows, num_cols=(1, 2, 3, 4),
                    hl_rows=(len(rows) - 1,))
            + f"<p style='margin-top:10px'>Unseen grades inside the training domain (27 grades). The ensemble is clearly "
              f"better than the textbook relations (UTS, Holm-adjusted p = {p_phys:.3f}), but inside the data it ties with a "
              f"plain composition GP (difference {gp.RMSE_minus_production:+.0f} MPa, 95 % CI {gp.CI95_lo:.0f} to "
              f"{gp.CI95_hi:.0f}). The physics is not there for interpolation; the next slide shows what it is for.</p>"
              "</div></div>")
    d.add(body, title=f"Q3: physics + ML predicts unseen grades within {bench(N, 'gated_ensemble', 'UTS_MPa'):.0f} MPa (UTS) and {bench(N, 'gated_ensemble', 'HB'):.0f} HB", kicker="Properties")

    # extrapolation and consistency
    ex = N["extrap"]
    an = N["anchor"]
    cons = N["cons"]
    cg = cons[cons.model == "gated_ensemble"].set_index("target")
    stats = (stat(f"{ex.loc['gated_ensemble', 'HB_RMSE_rates_le_5Ks']:.0f} HB", "production, new cooling rates ≤ 5 K/s")
             + stat(f"{ex.loc['gp_comp', 'HB_RMSE_rates_le_5Ks']:.0f} HB", "composition-only GP, same test")
             + stat(f"{100 * cg.loc['UTS_MPa', 'rate_violation_frac']:.1f} %", "wrong-way steps in UTS over cooling-rate sweeps")
             + stat(f"{cg.loc['UTS_MPa', 'si_0.10_0.40_span']:.0f} MPa", "UTS change over 0.10-0.40 % Si (physics expects tens)"))
    a0, a1 = an.iloc[0], an.iloc[1]
    body = (f'<div class="stats">{stats}</div><div class="row" style="margin-top:14px">'
            f'<div class="col-3">{img("fig14_cooling_rate_extrapolation.png", height=330)}</div>'
            '<div class="col-2 small">'
            "<p>Dilatometry steels cooled at 0.1-50 K/s: rates and microstructures the handbook data do not contain. Purely "
            "data-driven models cannot invent martensite hardness; the physics-anchored ones can.</p>"
            "<p><b>The anchor matters.</b> Anchoring on the Gladman law (ferrite fraction, grain size, spacing) instead of "
            f"Maynier's composition-only ferrite-pearlite hardness cut this error from {a0.HB_RMSE_dilatometry_le_5Ks:.0f} to "
            f"{a1.HB_RMSE_dilatometry_le_5Ks:.0f} HB and the out-of-domain UTS error from {a0.UTS_MPa_RMSE_outAD:.0f} to "
            f"{a1.UTS_MPa_RMSE_outAD:.0f} MPa, at a cost of {a1.UTS_MPa_RMSE_inAD - a0.UTS_MPa_RMSE_inAD:.0f} MPa inside the "
            "data.</p><p>Silicon is kept out of the ML inputs: with it, cross-validation looked better but a plain steel's UTS "
            "swung by 176 MPa over 0.10-0.40 % Si.</p></div></div>")
    d.add(body, title="What the physics buys: sensible answers at new cooling rates and physically consistent trends",
          kicker="Extrapolation and consistency")

    # answer table
    rows = []
    for c in CAT_ORDER:
        if c not in cat.index:
            continue
        r = cat.loc[c]
        flag = " *" if r.get("in_validity_share", 1.0) < 0.5 else ""
        rows.append([c.replace(" steel", "") + flag, f"{int(r.grades)}", micro(r, "FC"), micro(r, "AC25"),
                     f"{r.UTS_MPa_FC_median:.0f} → {r.UTS_MPa_AC25_median:.0f} ({r.UTS_MPa_AC50_median:.0f})",
                     f"{r.YS_MPa_FC_median:.0f} → {r.YS_MPa_AC25_median:.0f}",
                     f"{r.El_pct_FC_median:.0f} → {r.El_pct_AC25_median:.0f}",
                     f"{r.HB_FC_median:.0f} → {r.HB_AC25_median:.0f}"])
    body = (table(["Category", "grades", "furnace cooled", "air cooled, 25 mm", "UTS (50 mm)", "YS", "El %", "HB"], rows,
                  num_cols=(1,), col_widths=["18%", "6%", "13%", "17%", "15%", "11%", "9%", "11%"])
            + '<p class="tiny" style="margin-top:8px">Medians over the catalogue grades of each category; furnace cooling '
              "0.005 K/s, air cooling of a 25 mm bar (the handbook's documented normalising condition), and of a 50 mm bar in brackets. "
              "P: pearlite (lever split; the kinetic split is the top of a band), B+M: bainite + martensite. "
              "* outside the window of the critical-temperature equations (2 % Si): treat with caution.</p>")
    d.add(body, title="Q4, the answer: air cooling refines plain-carbon steels and changes alloy steels into a different steel",
          kicker="Furnace cooling vs air cooling, by category")

    # temperature vs cooling split
    rows = []
    for c in CAT_ORDER:
        if c not in cat.index:
            continue
        r = cat.loc[c]
        rows.append([c.replace(" steel", ""), f"{r.UTS_MPa_delta_median:+.0f}", f"{r.UTS_MPa_delta_cooling_median:+.0f}",
                     f"{r.UTS_MPa_delta_temperature_median:+.0f}", f"{r.ITT_FC_median_C:.0f}" if pd.notna(r.ITT_FC_median_C) else "-",
                     f"{r.ITT_AC25_median_C:.0f}" if pd.notna(r.ITT_AC25_median_C) else "-"])
    body = ('<div class="row"><div class="col">'
            + table(["Category", "ΔUTS, 25 mm (MPa)", "from faster cooling", "from hotter austenitisation",
                     "ITT FC (°C)", "ITT AC (°C)"], rows, num_cols=(1, 2, 3, 4, 5))
            + '</div><div class="col small" style="max-width:360px"><p>Normalising differs from annealing in two ways. Air '
              "cooling a 25 mm bar from the <i>annealing</i> temperature isolates the effect of the faster cooling; the rest "
              "is the hotter austenitisation (coarser austenite, more hardenability).</p>"
              "<p>ITT is Pickering's impact transition temperature on the predicted microstructure, a toughness indicator "
              "(lower is tougher), defined for ferrite-pearlite steels only.</p>"
              "<p class='tiny'>The temperature part is learned from data in which temperature and route go together, so it "
              "is less certain than the cooling part.</p></div></div>")
    d.add(body, title="Most of the gain comes from the faster cooling; normalising also lowers the transition temperature",
          kicker="Why air cooling is stronger")

    # carbon sweep
    d.add(img("fig16_carbon_sweep_fc_vs_ac.png", height=560),
          title="Coarse vs fine pearlite: the classic chart, now with intervals (H1, H2)", kicker="Plain-carbon steels")

    # section size
    d.add(img("fig17_section_size_effect.png", height=430)
          + '<p class="small" style="margin-top:10px">For 1020 and 1045 the bar size changes the UTS by only tens of MPa, '
            "because the structure stays ferrite + pearlite and only coarsens. For 4140 it decides between martensite, "
            "bainite and ferrite + pearlite. Normalised 220 mm C-Mn axles from the literature are a direct check of a thick "
            "section (next slides).</p>",
          title="Section size: a thick air-cooled bar drifts towards the annealed state (H4)", kicker="Mass effect")

    # external checks
    st = N["struct"]
    sa = st[(st.subset == "all")].set_index("target")
    lm = N["litm"]
    pr = N.get("pairs")
    pair_txt = ""
    if pr is not None and len(pr):
        e = pr.predicted_change - pr.measured_change
        sign = np.mean(np.sign(pr.predicted_change) == np.sign(pr.measured_change))
        pair_txt = (f"<p><b>Same steel, both routes</b> ({pr.source_id.nunique()} papers, {len(pr)} comparisons): the "
                    f"predicted change has the right sign in {100 * sign:.0f} % of cases, MAE {np.mean(np.abs(e)):.0f} "
                    "(MPa or HB).</p>")
    charpy = st[st.target.str.startswith("Charpy")]
    ch_txt = (f"<p><b>Toughness:</b> the predicted transition temperature ranks the measured Charpy energy of the members "
              f"with Spearman {charpy.spearman.iloc[0]:+.2f} (n = {int(charpy.n.iloc[0])}).</p>") if len(charpy) else ""
    body = (f'<div class="row"><div class="col-3">{img("fig20_external_structural.png", height=420)}</div>'
            '<div class="col-2 small">'
            f"<p><b>165 hot-rolled S275/S355 members</b> (never used for training): UTS RMSE {sa.loc['UTS_MPa', 'RMSE']:.0f} MPa "
            f"(bias {sa.loc['UTS_MPa', 'bias']:+.0f}), YS bias {sa.loc['YS_MPa', 'bias']:+.0f} MPa, elongation on the handbook "
            f"gauge within {sa.loc['El_pct', 'RMSE']:.1f} %. The yield bias is larger for the microalloyed members; the "
            "predictions hardly rank the members (Spearman "
            f"{sa.loc['YS_MPa', 'spearman']:.2f} for YS), so for a new product a few local tests are needed.</p>"
            + pair_txt + ch_txt + "</div></div>")
    d.add(body, title="Tests on data the models never saw: good for typical values, weaker for modern rolled steel",
          kicker="External validation")

    # uniform elongation and toughness
    ue = lm.loc[[q for q in lm.index if q.startswith("uniform elongation (%) - ratio method, leave")][0]]
    hol = lm.loc[[q for q in lm.index if q.startswith("uniform elongation (%) - Hollomon through")][0]]
    ratio = lm.loc[[q for q in lm.index if q.startswith("uniform/total elongation ratio")][0]]
    body = ('<div class="row"><div class="col small">'
            "<h3>Uniform elongation</h3>"
            "<p>None of the open handbook data report it. A Hollomon curve through the yield point and the UTS fails for "
            "ferrite-pearlite steels with a Lüders plateau (MAE "
            f"{hol.MAE:.1f} points, and above the total elongation for some handbook rows).</p>"
            "<p>What works: after putting every test's total elongation on the handbook gauge (Oliver, ISO 2566-1), uniform "
            f"elongation is a stable fraction of it, <b>{ratio.observed_mean:.3f} ± {ratio.observed_sd:.3f}</b> "
            f"({int(ratio.n)} tests, {int(ratio.sources)} papers); predicting each paper from the others is "
            f"{ue.MAE:.1f} points off. The gauge conversion alone cut the scatter of the ratio by a third.</p></div>"
            '<div class="col small"><h3>Toughness</h3>'
            "<p>The handbook copy we could use has no impact data, so toughness is reported as Pickering's impact "
            "transition temperature on the predicted microstructure (ferrite grain size, pearlite share, Si, free N).</p>"
            "<p>It is an indicator, not a measurement; the Charpy energies of the structural members are used only to "
            "check that it ranks steels the right way.</p></div></div>")
    d.add(body, title="Uniform elongation and toughness: estimated with care, and labelled as estimates",
          kicker="Ductility and toughness")

    # use
    recs = N["recs"]
    rows = []
    for spec, g in recs.groupby("spec", sort=False):
        top = g.head(3)
        rows.append([spec, ", ".join(f"{a.replace('AISI ', '')} ({b})" for a, b in zip(top.grade, top.option))])
    body = (f'<div class="row"><div class="col-2">{img("fig22_grade_front_fc_vs_ac.png", height=470)}</div>'
            '<div class="col-2 small">' + table(["Requirement (on the lower 90 % bound)", "Best real grade + route"], rows)
            + "<p style='margin-top:10px'>Only standard grades and routes are searched, inside the training domain and with "
              "a microstructure the data contain. No composition is invented.</p></div></div>")
    d.add(body, title="Using the model: which standard grade and cooling route meets a specification", kicker="Proactive use")


def slides_closing(d, N):
    plan = N["plan"]
    rows = []
    for r in plan.sort_values("priority_rank").head(9).itertuples():
        micro = " + ".join(f"{k} {100 * v:.0f}%" for k, v in (("F", r.ferrite_frac), ("P", r.pearlite_frac),
                                                               ("B", r.bainite_frac), ("M", r.martensite_frac)) if v >= 0.02)
        rows.append([r.steel, r.option, micro, f"{r.UTS_MPa_pred:.0f} {r.UTS_MPa_90PI}", f"{r.HB_pred:.0f}"])
    body = ('<div class="row"><div class="col-3">'
            + table(["Steel", "Route", "Predicted microstructure", "UTS (MPa), 90 % interval", "HB"], rows,
                    col_widths=["16%", "24%", "26%", "22%", "12%"])
            + '</div><div class="col-2 small"><h3>Protocol</h3><ul>'
              "<li>Spark-OES chemistry of the bars actually bought; predictions re-issued with it before testing.</li>"
              "<li>Thermocouple in each bar: log the real furnace and air-cooling curves.</li>"
              "<li>Tensile tests with an extensometer (uniform elongation measured), gauge geometry recorded.</li>"
              "<li>Brinell/Vickers after removing the decarburised layer; point counting (ASTM E562), grain size (E112), "
              "SEM pearlite spacing on a few samples.</li>"
              "<li>As-cooled martensitic bars: hardness and microstructure only.</li></ul>"
              "<p>The predictions are written down before the tests, so the results will be a genuine test.</p></div></div>")
    d.add(body, title="Part of the study: a pre-registered validation protocol with steels available in India", kicker="Validation protocol")

    d.add('<div class="row"><div class="col small"><ul>'
          "<li><b>Training data are small and old:</b> 33 handbook grades, one typical value per grade and condition, at "
          "nominal compositions; the spread of compositions inside a grade's own range is already a large part of the "
          "error.</li>"
          "<li><b>One inferred condition left:</b> the furnace rate of the handbook's annealed bars (the bar size is now "
          "documented); its effect is measured by retraining (stage 10).</li>"
          "<li><b>Semi-empirical kinetics:</b> errors on independent data are structural; still too hardenable against "
          "Grossmann's D<sub>I</sub>; the Acm fit and Grange's silicon terms are poor above 0.9 % C and 1 % Si.</li>"
          "<li><b>Estimated quantities:</b> uniform elongation (a ratio), ferrite grain size (about ×2), toughness (an "
          "indicator).</li></ul></div><div class='col small'><ul>"
          "<li><b>Domain shift:</b> modern thermomechanically rolled and microalloyed products are stronger than "
          "predicted until a few local tests are added.</li>"
          "<li><b>Design choices made on evaluation data</b> (descriptors, gates, silicon) are documented as ablations, "
          "so the cross-validation is not fully blind.</li>"
          "<li><b>Estimated or out of scope:</b> precipitation (±50 %), Q+T Charpy, as-delivered TMCP, boron hardenability, "
          "carburising, decarburisation, banding; a constant-h (Grossmann) quench.</li>"
          "<li><b>Nothing here replaces a tensile test;</b> the intervals say how much to trust a number.</li></ul></div></div>",
          title="Scope and limitations", kicker="Limitations")

    cat = N["cat"]
    lo, hi, crmo = cat.loc["Low-carbon steel"], cat.loc["High-carbon steel"], cat.loc["Mo / Cr-Mo steel"]
    d.add('<div class="small"><ol style="padding-left:22px">'
          f"<li><b>Low-carbon steels:</b> furnace and air cooling give almost the same steel; air cooling only refines it "
          f"(UTS {lo.UTS_MPa_FC_median:.0f} → {lo.UTS_MPa_AC25_median:.0f} MPa), which is why they are normalised for grain "
          "refinement and uniformity, not for strength.</li>"
          f"<li><b>Medium- and high-carbon steels:</b> the same phases, much finer pearlite and finer ferrite; the gain grows "
          f"with the pearlite share (high carbon: {hi.UTS_MPa_FC_median:.0f} → {hi.UTS_MPa_AC25_median:.0f} MPa).</li>"
          f"<li><b>Hardenable alloy steels:</b> air cooling gives bainite and, in thin bars, martensite: a different steel "
          f"(Cr-Mo median {crmo.UTS_MPa_FC_median:.0f} → {crmo.UTS_MPa_AC25_median:.0f} MPa), so bar size and route must be "
          "chosen together, and such steels are tempered afterwards.</li>"
          "<li><b>The rest of the route:</b> hardening and tempering give each grade a range of strengths (Q+T UTS within "
          "about 75 MPa on unseen grades); a stress relief below the tempering parameter changes nothing but the residual stress.</li>"
          "<li><b>Method:</b> physics carries the extrapolation (new cooling rates, martensite, consistent trends); machine "
          "learning corrects the textbook relations and gives honest intervals; every link is tested on data the models "
          "never saw.</li></ol>"
          '<div class="callout" style="margin-top:14px"><p>Quantitatively unsettled in the literature and in our data: how much extra '
          "pearlite normalising gives a medium-carbon steel (H5). We report it as a band.</p></div></div>",
          title="Conclusions", kicker="Summary")

    refs = [
        "Li, M.V. et al. (1998) Metall. Mater. Trans. B 29, 661-672 (transformation kinetics).",
        "Kirkaldy, J.S., Venugopalan, D. (1984) Phase Transformations in Ferrous Alloys, TMS, 125-148.",
        "Collins, J. et al. (2023) Metals 13, 1168; data Zenodo 7770264 (open CCT predictor, dilatometry).",
        "Gladman, T., McIvor, I.D., Pickering, F.B. (1972) J. Iron Steel Inst. 210, 916-930.",
        "Pickering, F.B. (1978) Physical Metallurgy and the Design of Steels, Applied Science.",
        "Maynier, P., Jungmann, B., Dollet, J. (1978) Hardenability Concepts with Applications to Steel, AIME, 518-545.",
        "Grange, R.A. (1961) Metal Progress 79, 73; Grossmann, M.A. (1942) Trans. AIME 150, 227-259; ASTM A255.",
        "Hodgson, P.D., Gibbs, R.K. (1992) ISIJ Int. 32, 1329-1338; Marder, A.R., Bramfitt, B.L. (1976) Metall. Trans. A 7, 365.",
        "Koistinen, D.P., Marburger, R.E. (1959) Acta Metall. 7, 59-60; Steven, W., Haynes, A.G. (1956) JISI 183, 349.",
        "Lei, J. et al. (2018) J. Am. Stat. Assoc. 113, 1094-1111 (conformal prediction).",
        "Kennedy, M.C., O'Hagan, A. (2001) J. R. Stat. Soc. B 63, 425-464 (physics + discrepancy modelling).",
        "ASM Handbook Vol. 1 (1990) and Vol. 4 (1991); ISO 2566-1 (elongation conversion); ASTM E112, E562, E140.",
    ]
    d.add('<ol class="small" style="padding-left:22px">' + "".join(f"<li>{r}</li>" for r in refs) + "</ol>"
          '<p class="tiny">The full list, with DOIs and the 52 papers of the literature compilation, is in the project report.</p>',
          title="Key references", kicker="References")

    d.add('<div class="kicker">Furnace cooling vs air cooling of carbon and low-alloy steels</div>'
          '<h1>Thank you</h1>'
          '<p style="font-size:22px;line-height:1.4;color:#52514e;max-width:980px">Composition, cooling route and bar size '
          'in; microstructure and properties out, each with a range, and every step checked against measured data.</p>'
          '<div class="meta">Aviral Singh &nbsp;·&nbsp; Shibasish Banerjee<br>Supervisor: Dr. Anish Karmakar<br>'
          'Department of Metallurgical and Materials Engineering, IIT Roorkee</div>',
          cls="title-slide", footer=False)
