"""Method, validation and result slides of the deck (numbers from results/)."""

from __future__ import annotations

from deck_lib import img, stat, table


def pct(x, d=0):
    return f"{100 * x:.{d}f} %"


def bench(N, model, target, col="RMSE_inAD"):
    b = N["bench"]
    return float(b[(b.model == model) & (b.target == target)][col].iloc[0])


def slides_methods(d, N):
    # data
    rows = [
        ["ASM Handbook / Bethlehem Steel table", "typical YS, UTS, elongation, HB of 33 AISI grades annealed, normalised, hot rolled",
         "80 rows", "training"],
        ["SAE J403 / J1268 ranges, EN/GB/ASTM catalogue", "composition ranges and families", "547 grades", "classifier"],
        ["SteelBench (Zenodo)", "real heat analyses with family labels", "1358 heats", "classifier test"],
        ["Collins et al. 2023 (Zenodo)", "dilatometry start temperatures and hardness at 9 cooling rates", "3 steels",
         "kinetics, extrapolation"],
        ["Literature compilation (this work)", "measured fractions, spacing, grain size, tensile, hardness, impact",
         f"{N['lit_rows']} rows, {N['lit_src']} papers", "microstructure and property checks"],
        ["Kingsley et al. 2026 (Zenodo)", "chemistry, tensile and Charpy of hot-rolled S275/S355 members", "165 members",
         "external test, recalibration"],
    ]
    if N.get("jm") is not None:
        rows.append(["SAE J1268 / ASTM A304 H-bands", "Jominy hardness bands of the H-steels", N.get("jominy_size", "H-grades"),
                     "kinetics calibration and test"])
    body = table(["Source", "What it gives", "Size", "Role"], rows, col_widths=["27%", "45%", "13%", "15%"])
    body += (
        '<div class="row" style="margin-top:14px"><div class="col"><p class="small">All open data or factual handbook values, '
        "each file with its source, licence and checksum. Handbook <i>typical</i> values and specification <i>minimum</i> "
        "values are never mixed. Errors in the sources (a row shift in the H-steel table, swapped Mn ranges of 1117/1118, "
        "&ldquo;15821H&rdquo; for 15B21H) are corrected and logged.</p></div>"
        '<div class="col"><p class="small">Every number of the literature compilation is traceable to a table or sentence of '
        "an open paper and was checked by script against the saved full text. "
        f"{N['lit_suspect']} rows whose values are implausible are kept but flagged and left out of every metric.</p></div></div>")
    d.add(body, title="Real, open data, each with a clear role", kicker="Data")

    # chain
    def box(head, text):
        return f'<div class="box"><b>{head}</b>{text}</div>'
    arrow = '<div class="arrow">&rarr;</div>'
    row1 = arrow.join([box("Composition", "wt% C, Mn, Si, Cr, Ni, Mo, ..."),
                       box("Category", "XGBoost classifier, cross-checked by the AISI/SAE rules; &ldquo;unknown&rdquo; if novel"),
                       box("Thermodynamics", "Ae1, Ae3, Acm (Grange), eutectoid point, lever rule"),
                       box("Cooling path", "furnace (Newtonian), round bar in still air (heat balance), Jominy bar (1-D conduction)")])
    row2 = arrow.join([box("Kinetics", "Kirkaldy-Venugopalan / Li et al. (1998), additivity, Hultgren start, Koistinen-Marburger"),
                       box("Microstructure", "ferrite, pearlite, bainite, martensite; spacing; ferrite grain size"),
                       box("Properties", "physics-anchored GP ensemble on Gladman / Maynier laws; 90 % conformal intervals"),
                       box("Use", "what-if, grade + route recommender, active learning, pre-registered lab plan")])
    phys = bench(N, "physics_lit", "UTS_MPa")
    body = (f'<div class="chain">{row1}</div><div class="chain" style="margin-top:12px">{row2}</div>'
            '<div class="row" style="margin-top:18px">'
            '<div class="col"><div class="callout small"><p><b>Physics</b> carries what the data cannot: new cooling rates, new '
            "microstructures (martensite), and the direction of every effect.</p></div></div>"
            f'<div class="col"><div class="callout good small"><p><b>Machine learning</b> corrects what the textbook relations '
            f"get wrong (physics alone is {phys:.0f} MPa off for UTS) and says how sure it is.</p></div></div>"
            '<div class="col"><div class="callout warn small"><p><b>Checks</b> on every link: dilatometry, measured fractions and '
            "spacings, Jominy bands, physical-consistency sweeps, three external datasets.</p></div></div></div>")
    d.add(body, title="One chain that follows the metallurgy: composition &rarr; microstructure &rarr; properties",
          kicker="Method")

    # category
    stats = (stat(f"{N['clf_sb']:.3f}", "family accuracy on 1358 real heats (ML)")
             + stat(f"{N['rules_sb']:.3f}", "the same for the AISI/SAE composition rules")
             + stat(pct(N["mar_ml"]), f"maraging steels refused (rules: {pct(N['mar_rules'])})")
             + stat(pct(N["agree_acc"]), f"family right when ML and rules agree ({pct(N['agree_share'])} of heats)"))
    body = (f'<div class="stats">{stats}</div><div class="row" style="margin-top:16px">'
            f'<div class="col-3">{img("fig04_category_shap.png", height=370)}</div>'
            '<div class="col-2 small">'
            "<p>A fair baseline first: the AISI/SAE composition limits written as a decision list, with nothing fitted. "
            f"ML beats it on the 13 categories of unseen grades ({N['clf_cv']:.2f} vs {N['rules_cv']:.2f}) and on real "
            "heats, and only the ML model recognises compositions that belong to no category.</p>"
            f"<p>The two disagree on {pct(1 - N['agree_share'])} of real heats, almost all near a limit (Mn about 1.0 %, "
            f"C about 0.25 %); there the family is right only {pct(N['disagree_acc'])} of the time, so the app shows "
            "both.</p><p>SHAP points to the elements a metallurgist would look at: S for free-machining, Si for spring "
            "steels, Mo for Cr-Mo, Ni for Ni-Cr-Mo, C for the plain-carbon classes. Cr matters for almost every class, "
            "because its absence (a residual 0.1 %) is what separates a plain-carbon from a Cr steel.</p></div></div>")
    d.add(body, title="Q1: the classifier recognises the category, and beats the AISI rules where it matters",
          kicker="Steel category")

    # cooling paths
    r = N["rates"]
    body = (f'<div class="row"><div class="col-3">{img("fig27_cooling_paths.png", height=520)}</div>'
            '<div class="col-2 small"><h3>How the paths are computed</h3>'
            f"<p><b>Furnace:</b> Newtonian cooling with the furnace, {r['furnace cooling']:.3f} K/s at 700 °C.</p>"
            "<p><b>Still air:</b> heat balance of a long round bar (natural convection and radiation, emissivity 0.8, "
            "Eurocode heat capacity). The Biot number stays below 0.1 up to 100 mm, so the bar is treated as lumped.</p>"
            "<p><b>Jominy bar:</b> 1-D conduction along the bar, with the water-jet coefficient set from the standard "
            "cooling-rate table.</p>"
            f"<p>J32 of the Jominy bar cools like a 10 mm bar in still air ({r['Jominy J32']:.1f} vs "
            f"{r['still air, 10 mm bar']:.1f} K/s), so the H-bands test the kinetics from thin air-cooled bars up to quench "
            "rates.</p><p class='tiny'>Simplification: the Eurocode heat capacity puts the transformation heat near 735 °C for "
            "every steel; for alloy steels that transform lower this shifts the rate at 700 °C (a stated limitation).</p>"
            "</div></div>")
    d.add(body, title=f"A 25 mm bar in still air cools at {r['still air, 25 mm bar']:.1f} K/s at 700 °C, "
                      f"a 100 mm bar at {r['still air, 100 mm bar']:.2f} K/s", kicker="Cooling paths")

    # kinetics
    tab = table(["Start temperature", "MAE", "n"],
                [["Ferrite (Ar3)", f"{N['Fs']:.1f} °C", str(N["n_Fs"])], ["Pearlite", f"{N['Ps']:.1f} °C", "10"],
                 ["Martensite (Ms)", f"{N['Ms']:.1f} °C", str(N["n_Ms"])],
                 ["Bainite", f"+{N['Bs_bias']:.0f} °C (bias)", "5"]], num_cols=(1, 2))
    body = (f'<div class="row"><div class="col-3">{img("fig07_kinetics_validation.png", height=470)}</div>'
            f'<div class="col-2 small">{tab}'
            '<p style="margin-top:10px">Measured dilatometry of EN3B, EN8 and SA-540 at 0.1-50 K/s (Collins et al. 2023). '
            "Our implementation agrees with the authors' own code to 1-2 °C for ferrite and bainite start, so the model is "
            "written down correctly.</p>"
            f"<p>The bainite start is about {N['Bs_bias']:.0f} °C too high, a known weakness of the Li model.</p>"
            "</div></div>")
    d.add(body, title="Q2: the transformation model predicts the measured start temperatures to within 10-23 °C",
          kicker="Transformation kinetics")

    # CCT
    d.add(img("fig09b_cct_1045_vs_4140.png", height=560),
          title="The same air-cooling path gives ferrite + pearlite in 1045 but bainite in 4140", kicker="CCT diagrams")

    # pearlite fraction
    pl = N["pearl"].set_index(["subset", "estimate"])
    fk, fl = pl.loc[("furnace cooled", "kinetic"), "MAE_pct_points"], pl.loc[("furnace cooled", "lever"), "MAE_pct_points"]
    ak, al = pl.loc[("air cooled", "kinetic"), "MAE_pct_points"], pl.loc[("air cooled", "lever"), "MAE_pct_points"]
    body = (f'<div class="row"><div class="col-3">{img("fig11_microstructure_literature_validation.png", height=470)}</div>'
            '<div class="col-2 small">'
            f"<p><b>Furnace cooled:</b> kinetics and lever rule both agree with the measured fractions (MAE {fk:.1f} and "
            f"{fl:.1f} points).</p>"
            "<p><b>Air cooled:</b> the textbooks say normalised steel holds more pearlite (quasi-eutectoid: the austenite "
            "enters the Hultgren region before the equilibrium ferrite has formed). The Li model shows this, but far too "
            f"strongly: 82-86 % pearlite for normalised 1045 / S45C, where 50-56 % was measured (MAE {ak:.1f} vs {al:.1f} "
            "points for the lever rule).</p>"
            "<p>We report the lever split, with the kinetic split as the top of a band. Starting the pearlite incubation "
            "only inside the Hultgren region did not help (an ablation).</p>"
            '<div class="callout warn"><p>So the quasi-eutectoid excess is reported as a band, and point counting of the same '
            "heat after both routes is part of the validation protocol.</p></div></div></div>")
    d.add(body, title="H5 is the least settled: measured pearlite of normalised steels sits at or below the lever rule",
          kicker="Microstructure: phase fractions")
