"""Version-3 slides: data, the documented bar size, hardening, tempering, the five
treatments, assumptions and sensitivity, kinetic uncertainty (numbers from results/)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from deck_lib import img, stat, table


def _r(N, name):
    return pd.read_csv(N["res"] / f"{name}.csv")


def slides_v3(d, N):
    res = N["res"]
    qt = pd.read_csv(res.parent / "data/processed/qt_bars.csv")
    nb = pd.read_csv(res.parent / "data/processed/normalized_bars.csv")
    b = _r(N, "s09_qt_benchmark")
    rm = lambda m, t: float(b[(b.model == m) & (b.target == t)].RMSE.iloc[0])  # noqa: E731

    # ---- what version 3 adds
    rows = [["1", "Inferred cooling conditions", "Assumptions register; handbook bar size documented (25 mm, not 50); "
             "retraining over every range; Morris screen", "done"],
            ["2", "Kinetics uncertainty", "MCMC posterior (4 chains, R-hat, ESS); propagated to predictions; "
             "independent coverage on 423 CCT points", "done"],
            ["3", "Hardening (quench)", "radial heat flow, Grossmann H, oil / water / brine, any bar size", "done"],
            ["4", "Tempering", "Hollomon-Jaffe master curve + secondary hardening, fitted on Q+T bars", "done"],
            ["5", "Precipitation strengthening", "V, Nb solubility + Ashby-Orowan, route-dependent", "estimate (±50 %)"],
            ["6", "Toughness", "ITT for ferrite-pearlite; Q+T Charpy out of scope, embrittlement flags", "declared"],
            ["7", "TMCP / microalloyed", "detected and refused for as-delivered properties; boron flagged", "done"]]
    body = ('<div class="row"><div class="col-3">' + table(["#", "Reviewer's item", "What we did", "Status"], rows,
                                                           col_widths=["5%", "25%", "52%", "18%"])
            + '</div><div class="col-2"><div class="stats">'
            + stat(f"{len(qt) + len(nb)}", "new measured bars with the heat treatment written out")
            + stat("5", "classic treatments now modelled")
            + stat("25 mm", "documented handbook bar size (was inferred as 50 mm)")
            + '</div><p class="small">A fellow student reviewed version 2 and ranked what was missing. We followed the list '
              "in order, and asked whether our data were simply too small.</p></div></div>")
    d.add(body, title="Version 3: the reviewer's list, item by item", kicker="What changed")

    # ---- data and the 25 mm finding
    sm = _r(N, "s10_handbook_section_match")
    rows = [[f"{r.section_mm:.0f} mm", f"{r.mean_dUTS:+.0f}", f"{int(r.within_3MPa)} of {int(r.grades)}"] for _, r in sm.iterrows()]
    body = ('<div class="row"><div class="col small">'
            "<h3>A bigger, relevant dataset</h3>"
            f"<p>Birbilis et al. (2025, CC BY 4.0): 3234 steels with the heat treatment written out. After removing "
            f"stainless, tool, cast, carburised and cold-drawn products and duplicate rows: <b>{len(qt)} quenched-and-tempered "
            f"bars of {qt.grade.nunique()} grades</b> (oil and water, tempered 150-705 °C, 13-100 mm) and <b>{len(nb)} "
            "normalised or annealed bars</b>.</p>"
            "<h3>Is our dataset too small?</h3><p>A 1 GB steel dataset is almost always images (micrographs) or a big table "
            "without the processing; neither tells a model what a heat treatment does. What matters is how many steels "
            "and treatments have a measured property <i>and</i> a known treatment. Every row here has one.</p></div>"
            '<div class="col small"><h3>One inferred assumption was wrong</h3>'
            "<p>The handbook's normalised values appear in the new database with the bar size written out:</p>"
            + table(["Bar", "UTS bar − handbook (MPa)", "Within 3 MPa"], rows)
            + "<p>They are the <b>25 mm (1 in.)</b> bars. Version 2 had inferred 50 mm from the physics alone, which "
              "measured the physics' bias, not the bar. All results now use the documented size.</p></div></div>")
    d.add(body, title="More data, and the bar size of the handbook data documented", kicker="Data")

    # ---- hardening
    hd = _r(N, "s09_hardenability_check")
    rc = (hd.DI_model_calibrated_mm / hd.DI_A255_mm).median()
    rp = (hd.DI_model_published_mm / hd.DI_A255_mm).median()
    qr = _r(N, "s09_quench_rates")
    q = lambda D, m: float(qr[(qr.D_mm == D) & (qr.medium == m) & (qr.r_over_R == 0)].rate700_K_s.iloc[0])  # noqa: E731
    body = (f'<div class="row"><div class="col-3">{img("fig31_quench_and_hardenability.png", height=470)}</div>'
            '<div class="col-2 small"><h3>Quench heat flow</h3><p>Radial conduction in a round bar; Grossmann H for oil, '
            "water, brine and the agitation, h = 2kH; checked against the exact Bessel solution (within 2 %). The 735 °C "
            "heat-capacity peak of slow cooling is removed for a quench (it made the surface softer than the centre).</p>"
            f"<p>Centre of a 25 mm bar at 700 °C: <b>{q(25, 'oil'):.0f} K/s</b> in oil, <b>{q(25, 'water'):.0f} K/s</b> in water.</p>"
            "<h3>Hardenability check</h3><p>D<sub>I</sub> (50 % martensite at the centre, ideal quench) from heat flow + "
            f"kinetics against the ASTM A255 factors: <b>{rc:.2f}×</b> with the recalibrated kinetics, {rp:.2f}× with the "
            "published ones. Still too hardenable, most for high-carbon steels, but half the bias.</p></div></div>")
    d.add(body, title="Hardening: quench heat flow, and a hardenability check that uses none of our data",
          kicker="Hardening")

    # ---- tempering
    tc = _r(N, "s09_tempering_constants").set_index("constant")
    body = (f'<div class="row"><div class="col-3">{img("fig35_tempering_curves.png", height=440)}</div>'
            '<div class="col-2 small"><h3>Hollomon-Jaffe master curve</h3>'
            "<p>P = T(20 + log t). Martensite softens along a logistic curve in P towards a fully tempered hardness; "
            "alloying shifts it to higher P; Mo, V, Cr add secondary hardening near 600 °C. As-quenched hardness is "
            "capped at 66 HRC (retained austenite).</p>"
            f"<p>Fitted on {len(qt)} bars (grade bootstrap):<br>Mo delays softening by <b>{0.1 * tc.loc['p_Mo', 'fitted'] / 20:.0f} °C per "
            f"0.1 % Mo</b>; Si {0.1 * tc.loc['p_Si', 'fitted'] / 20:.0f} °C and Cr {0.1 * tc.loc['p_Cr', 'fitted'] / 20:.0f} °C per 0.1 %.</p>"
            f"<p>The fully tempered hardness rises by only {tc.loc['s_C', 'fitted']:.0f} HV per % C (untempered martensite: about 950): "
            "carbon's effect fades with tempering, as Grange et al. (1977) found. Shaded: the temper-embrittlement windows "
            "the tool flags.</p></div></div>")
    d.add(body, title="Tempering: one master curve, and alloying delays the softening (H6)", kicker="Tempering")

    # ---- Q+T models
    cf = _r(N, "s09_qt_conformal").set_index("target")
    kc = _r(N, "s09_qt_kinetics_check")
    kg = lambda lab: float(kc[kc.kinetics.str.startswith(lab) & (kc.model == "qt_greybox")].UTS_RMSE.iloc[0])  # noqa: E731
    body = (f'<div class="row"><div class="col">{img("fig32_qt_model_ladder.png", width=640)}'
            f'{img("fig34_qt_mass_effect.png", width=640)}</div>'
            '<div class="col small"><div class="stats">'
            + stat(f"{rm('qt_ensemble', 'UTS_MPa'):.0f} MPa", "UTS error, grade held out")
            + stat(f"{rm('qt_physics', 'UTS_MPa'):.0f} MPa", "textbook physics alone")
            + stat(f"{100 * cf.loc['UTS_MPa', 'coverage_nested']:.0f} %", "inside the 90 % interval")
            + "</div><p>Production: physics-anchored GP + data GP. Best rung for UTS, YS "
              f"({rm('qt_ensemble', 'YS_MPa'):.0f} MPa) and elongation ({rm('qt_ensemble', 'El_pct'):.1f} %). "
              "No bias with bar size: the mass effect (H7) comes from the heat flow and the kinetics.</p>"
              f"<p><b>Independent test of the recalibrated kinetics:</b> grey box {kg('recal'):.0f} MPa with them, {kg('published'):.0f} MPa "
              "with the published constants; the physics-anchored GP the other way round. After tempering, strength hardly "
              "depends on the bainite share, so these bars cannot settle the recalibration, and we say so.</p></div></div>")
    d.add(body, title="Quenched and tempered: physics + ML within 75 MPa across 13-100 mm bars", kicker="Q+T properties")

    # ---- five treatments
    ft = _r(N, "s09_five_treatments")
    ft["grade"] = ft.grade.astype(str)
    a = ft[ft.grade == "4140"].set_index("treatment")
    rows = []
    lab = {"anneal": "Annealed", "normalise": "Normalised", "harden": "Hardened (oil)", "quench_temper": "+ tempered 600 °C",
           "stress_relieve": "Stress relieved 600 °C"}
    for k in lab:
        r = a.loc[k]
        uts = "brittle" if k == "harden" else f"{r.UTS_MPa_pred:.0f}"
        rows.append([lab[k], str(r.microstructure).replace("unchanged: ", "unchanged "), uts, f"{r.HV:.0f}"])
    body = (f'<div class="row"><div class="col-3">{img("fig36_five_heat_treatments.png", height=500)}</div>'
            '<div class="col-2 small"><h3>AISI 4140, 25 mm bar</h3>'
            + table(["Treatment", "Microstructure", "UTS", "HV"], rows, col_widths=["26%", "44%", "15%", "15%"])
            + "<p>The textbook table, computed for any steel: <code>compare_treatments(comp)</code>.</p></div></div>")
    d.add(body, title="The five classic heat treatments, side by side", kicker="Full route")

    # ---- stress relief, precipitation, toughness, TMCP
    sa = _r(N, "s09_stress_relief_after_temper").set_index("sr_T_C")
    body = (f'<div class="row"><div class="col-3">{img("fig37_stress_relief_and_precipitation.png", height=420)}</div>'
            '<div class="col-2 small"><ul>'
            "<li><b>Stress relief:</b> residual stress ≤ k<sub>y</sub>(T)·YS (EN 1993-1-2). A Q+T 4140 (600 °C) keeps "
            f"{sa.loc[575, 'UTS_after_MPa']:.0f} MPa after a 575 °C relief but falls to {sa.loc[650, 'UTS_after_MPa']:.0f} MPa after 650 °C: "
            "relieve below the tempering temperature (H8).</li>"
            "<li><b>Precipitation:</b> dissolved V and Nb (solubility products) + Ashby-Orowan; finer after air cooling "
            "than after furnace cooling. An estimate (±50 %), only for microalloyed steels.</li>"
            "<li><b>Toughness:</b> ITT for ferrite-pearlite; for Q+T, Charpy is out of scope and the TME and "
            "temper-embrittlement windows are flagged.</li>"
            "<li><b>TMCP / HSLA, boron:</b> recognised from the chemistry; the tool predicts them only after "
            "re-austenitising.</li></ul></div></div>")
    d.add(body, title="Stress relieving, precipitation, toughness and rolled steels: modelled, estimated or declared",
          kicker="Scope")

    # ---- assumptions
    oat = _r(N, "s10_oat_anneal_normalise")
    base = oat[oat.setting == "baseline"].iloc[0]
    other = oat[(oat.setting != "baseline") & ~oat.setting.str.startswith("normalised bars")]
    body = (f'<div class="row"><div class="col">{img("fig38_assumption_sensitivity.png", width=1180)}</div></div>'
            '<p class="small">Every assumption is in a register with its status (documented, fitted, literature, assumed, '
            "inferred) and range. Retraining with each uncertain one at the ends of its range moves the 1045 air-cooling gain "
            f"between {other.dUTS_1045.min():.0f} and {other.dUTS_1045.max():.0f} MPa (baseline {base.dUTS_1045:.0f}) and the 4140 gain between "
            f"{other.dUTS_4140.min():.0f} and {other.dUTS_4140.max():.0f} MPa (baseline {base.dUTS_4140:.0f}). The undocumented furnace rate sets "
            "<i>how much</i> air cooling adds, never the direction or the ranking of the steels.</p>")
    d.add(body, title="Every assumption made explicit, and how much each one matters", kicker="Sensitivity")

    # ---- kinetic uncertainty
    cv = _r(N, "s11_cct_atlas_coverage").set_index("quantity")
    dg = _r(N, "s11_mcmc_diagnostics")
    body = (f'<div class="row"><div class="col">{img("fig40_kinetic_posterior.png", width=620)}</div>'
            f'<div class="col">{img("fig41_kinetic_uncertainty_propagation.png", width=620)}</div></div>'
            '<p class="small">Four adaptive Metropolis chains, split R-hat ≤ '
            f"{dg.split_Rhat.max():.2f}, ESS ≥ {dg.ESS.min():.0f}. The bainite constants are pinned; the ferrite factor is not "
            "identified by hardness. Pushed to 423 independent CCT-atlas points, the parameter bands contain only "
            f"{100 * cv.loc['f_FP', 'coverage_parametric_90']:.0f} % (ferrite + pearlite) and {100 * cv.loc['f_B', 'coverage_parametric_90']:.0f} % "
            "(bainite) of the measurements: the error is in the form of the equations, not their constants. A structural "
            f"term of about {cv.loc[['f_FP', 'f_B', 'f_M'], 'structural_sd'].mean():.2f} in phase fraction brings the coverage to 90 %.</p>")
    d.add(body, title="How uncertain is the transformation model? Structurally, not parametrically",
          kicker="Kinetic uncertainty")
