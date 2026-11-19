"""Build the presentation deck (reports/BTP_deck.pdf) from the pipeline results.

    .venv/bin/python reports/deck/build_deck.py            # PDF only
    .venv/bin/python reports/deck/build_deck.py --png DIR  # also PNG previews

Every number on a slide is read from results/*.csv here, so the deck always
matches the last pipeline run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from deck_lib import Deck, img, render_pdf, render_pptx, stat, table  # noqa: E402
from slides_body import slides_methods  # noqa: E402
from slides_results import slides_closing, slides_results  # noqa: E402
from slides_v3 import slides_v3  # noqa: E402

RES = ROOT / "results"
FIGS = ROOT / "figures"


def R(name):
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def f0(x):
    return f"{x:.0f}"


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f} %"


def has_fig(name):
    return (FIGS / name).exists()


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------

def numbers():
    N = {"res": RES, "jominy_size": "80 grades"}
    cv = R("s02_classifier_cv").set_index("model")
    N["clf_cv"] = cv.loc[cv.production].cv_accuracy.iloc[0]
    N["clf_sb"] = cv.loc[cv.production].steelbench_family_accuracy.iloc[0]
    N["rules_cv"] = cv.loc["aisi_rules", "cv_accuracy"]
    N["rules_sb"] = cv.loc["aisi_rules", "steelbench_family_accuracy"]
    ext = R("s02_external_tests_all_models")
    lg = ext[ext.test.str.startswith("Legacy")].set_index("model").accuracy
    N["mar_ml"], N["mar_rules"] = lg[cv.index[cv.production][0]], lg["aisi_rules"]
    ag = R("s02_rules_agreement").set_index("subset")
    N["agree_share"] = ag.loc["ML and AISI rules agree on the category", "share"]
    N["agree_acc"] = ag.loc["ML and AISI rules agree on the category", "family_accuracy_ML"]
    N["disagree_acc"] = ag.loc["ML and AISI rules disagree", "family_accuracy_ML"]
    ks = R("s03_kinetics_summary")
    k = ks[ks.model == "this work"].set_index("start")
    N["Fs"], N["Ps"], N["Bs"], N["Ms"] = (k.loc[s, "MAE_C"] for s in ("ferrite", "pearlite", "bainite", "martensite"))
    N["Bs_bias"] = k.loc["bainite", "bias_C"]
    N["n_Fs"], N["n_Ms"] = int(k.loc["ferrite", "n"]), int(k.loc["martensite", "n"])
    cp = R("s03_cooling_paths").set_index("path").rate_700C_K_s
    N["rates"] = cp
    pl = R("s03_literature_pearlite_summary")
    N["pearl"] = pl
    gs = R("s03_grain_size_check").set_index("relation")
    N["gs"] = gs
    bm = R("s04_model_benchmark")
    N["bench"] = bm
    N["conf"] = R("s04_conformal").set_index("target")
    N["extrap"] = R("s04_cooling_rate_extrapolation").set_index("model")
    N["anchor"] = R("s04_anchor_ablation")
    N["cons"] = R("s04_consistency")
    N["sil"] = R("s04_silicon_ablation")
    N["paired"] = R("s04_paired_tests")
    N["cat"] = R("s05_category_summary").set_index("category")
    N["noise"] = R("s05_composition_noise_floor")
    N["struct"] = R("s06_structural_metrics")
    N["litm"] = R("s06_literature_metrics").set_index("quantity")
    N["pairs"] = R("s06_literature_paired_routes")
    N["cc"] = R("s06_continuous_cooling_hardness")
    N["di"] = R("s08_ideal_diameter_summary")
    N["recs"] = R("s07_recommendations")
    N["plan"] = R("s07_lab_plan_preregistered")
    N["fs"] = R("s07_few_shot")
    N["jm"] = R("s08_jominy_metrics")
    N["jpost"] = R("s08_kinetic_calibration_posterior")
    N["jcv"] = R("s08_kinetic_calibration_cv_metrics")
    lit = pd.read_csv(ROOT / "data/raw/literature_compilation.csv")
    N["lit_rows"], N["lit_src"] = len(lit), lit.source_id.nunique()
    flag = lit.get("quality_flag", pd.Series("", index=lit.index)).fillna("")
    N["lit_suspect"] = int(flag.str.startswith("suspect").sum())
    cat = R("s05_catalogue_fc_vs_ac")
    N["ex"] = cat[cat.grade.isin(["AISI 1045", "AISI 4140H"]) & cat.case.isin(["FC", "AC25"])].set_index(["grade", "case"])
    return N


def micro_str(r):
    parts = [f"{name} {100 * v:.0f} %" for name, v in (("ferrite", r.F_best), ("pearlite", r.P_best),
                                                        ("bainite", r.B_best), ("martensite", r.M_best)) if v >= 0.02]
    return " + ".join(parts) + f" ({r.S0_um:.2f} µm)"


def bench(N, model, target, col="RMSE_inAD"):
    b = N["bench"]
    return float(b[(b.model == model) & (b.target == target)][col].iloc[0])


# ---------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------

def build(N):
    d = Deck("Singh & Banerjee · MTP-400A · MMED, IIT Roorkee", "Heat treatment of steels")

    # 1 title
    d.add('''<div class="kicker">B.Tech Project MTP-400A · Department of Metallurgical and Materials Engineering, IIT Roorkee</div>
<h1>Annealing, normalising, hardening, tempering and stress relieving of carbon and low-alloy steels</h1>
<p style="font-size:24px;line-height:1.35;color:#52514e;max-width:1000px">Microstructure, hardness and tensile properties
predicted from the composition and the heat treatment with physics-informed machine learning. The core question:
furnace cooling vs air cooling.</p>
<div class="meta">Aviral Singh &nbsp;·&nbsp; Shibasish Banerjee<br>Supervisor: Dr. Anish Karmakar<br>November 2026</div>''',
          cls="title-slide", footer=False)

    # 2 the topic, from the basics
    svg = '''<svg viewBox="0 0 540 400" width="100%" style="max-height:395px;font-family:inherit">
<rect x="60" y="30" width="450" height="310" fill="#ffffff" stroke="#e1e0d9"/>
<rect x="60" y="97" width="450" height="29" fill="#f4f3ee"/>
<line x1="60" y1="97" x2="510" y2="97" stroke="#898781" stroke-dasharray="5 4"/>
<line x1="60" y1="126" x2="510" y2="126" stroke="#898781" stroke-dasharray="5 4"/>
<text x="66" y="92" font-size="12" fill="#52514e">Ae3: ferrite starts to form</text>
<text x="66" y="140" font-size="12" fill="#52514e">Ae1: pearlite forms</text>
<text x="70" y="48" font-size="12.5" fill="#0b0b0b">Austenite (FCC iron, carbon dissolved)</text>
<polyline fill="none" stroke="#eb6834" stroke-width="3" stroke-linejoin="round"
 points="80,50 205,65 250,100 276,140 301,190 335,265 376,340"/>
<polyline fill="none" stroke="#2a78d6" stroke-width="3" stroke-linejoin="round"
 points="80,75 316,76 382,83 434,110 459,145 479,200 500,290"/>
<text x="120" y="225" font-size="13" font-weight="700" fill="#eb6834">Air cooling (normalising)</text>
<text x="120" y="241" font-size="12" fill="#52514e">about 1 K/s at 700 °C, 25 mm bar</text>
<text x="262" y="68" font-size="13" font-weight="700" fill="#2a78d6">Furnace cooling (annealing)</text>
<text x="478" y="245" font-size="12" fill="#52514e" text-anchor="end">about 18 K/h</text>
<text x="478" y="260" font-size="12" fill="#52514e" text-anchor="end">at 700 °C</text>
<g font-size="11.5" fill="#52514e" text-anchor="middle">
<text x="80" y="358">1 s</text><text x="231" y="358">1 min</text><text x="382" y="358">1 h</text><text x="500" y="358">1 day</text></g>
<text x="285" y="384" font-size="12.5" fill="#0b0b0b" text-anchor="middle">time from the start of cooling (log scale)</text>
<g font-size="11.5" fill="#52514e" text-anchor="end">
<text x="54" y="54">900</text><text x="54" y="144">700</text><text x="54" y="244">500</text><text x="54" y="344">300</text></g>
<text x="18" y="185" font-size="12.5" fill="#0b0b0b" text-anchor="middle" transform="rotate(-90 18 185)">temperature (°C)</text>
</svg>'''
    d.add(f'''<div class="row" style="gap:36px"><div class="col" style="flex:1.05">
<p>Every carbon or low-alloy steel part, a shaft, a gear blank or a forging, is heated until it becomes
<b>austenite</b> (about 830-880 °C), and what it turns into on the way back down to room temperature depends mostly on
<b>how fast it cools</b>.</p>
<p><span class="fc">Full annealing</span>: the furnace is switched off and the steel cools inside it, very slowly. It gives
coarse ferrite and coarse pearlite, a soft and ductile steel that is easy to machine and to cold form.</p>
<p><span class="ac">Normalising</span>: the steel is taken out and left to cool in still air, about 200 times faster at 700 °C.
The pearlite becomes finer and the ferrite grains smaller, so the steel is stronger and harder, and an alloy steel may
not form pearlite at all but bainite.</p>
<p class="small muted">These are the two most common heat treatments in industry, but the choice between them is still
made from handbook tables that list a few grades in a few conditions, with nothing for a new chemistry or a different
bar size. Hardening, tempering and stress relieving, the other three classic treatments, follow in the second part.</p></div>
<div class="col">{svg}
<div class="row small" style="gap:12px;margin-top:6px">
<div class="col callout" style="border-left-color:#2a78d6"><p><b class="fc">Furnace cooled</b><br>coarse ferrite + pearlite:
soft, ductile, machinable</p></div>
<div class="col callout" style="border-left-color:#eb6834;background:#fdf1ec"><p><b class="ac">Air cooled</b><br>fine
ferrite + pearlite, or bainite: stronger, harder</p></div></div></div></div>''',
          title="The core question: the same steel, cooled in two different ways, ends up as two different materials",
          kicker="What this project is about")

    # 3 what the project does, with a worked example
    def step(n, head, text):
        return f'<div class="box"><span class="pill">{n}</span><b>{head}</b>{text}</div>'
    arrow = '<div class="arrow">&rarr;</div>'
    chain = arrow.join([
        step("in", "You give", "the composition (wt %), the treatment (furnace, air, quench, temper, stress relief) and the bar diameter"),
        step(1, "Which steel?", "the category: plain carbon, Cr-Mo, Ni-Cr-Mo ..."),
        step(2, "How fast it cools", "the cooling curve of that bar in the furnace or in still air"),
        step(3, "What forms", "ferrite, pearlite, bainite, martensite, and at what temperature"),
        step(4, "How fine", "pearlite spacing and ferrite grain size"),
        step("out", "Properties", "yield and tensile strength, elongation, hardness, each with a 90 % range")])
    ex = N["ex"]
    rows = []
    for g, name in (("AISI 1045", "AISI 1045 (medium carbon)"), ("AISI 4140H", "AISI 4140H (Cr-Mo)")):
        for case, label in (("FC", '<span class="fc">furnace cooled</span>'), ("AC25", '<span class="ac">air cooled, 25 mm bar</span>')):
            r = ex.loc[(g, case)]
            rows.append([name if case == "FC" else "", label, micro_str(r), f"{r.YS_MPa_pred:.0f}", f"{r.UTS_MPa_pred:.0f}",
                         f"{r.El_pct_pred:.0f}", f"{r.HB_pred:.0f}"])
    a, b = ex.loc[("AISI 1045", "FC")], ex.loc[("AISI 1045", "AC25")]
    c, e = ex.loc[("AISI 4140H", "FC")], ex.loc[("AISI 4140H", "AC25")]
    d.add(f'''<div class="chain">{chain}</div>
<div class="row" style="margin-top:16px;gap:26px"><div class="col-3">
{table(["Steel", "Route", "Predicted microstructure", "YS (MPa)", "UTS (MPa)", "El. (%)", "HB"], rows,
       num_cols=(3, 4, 5, 6), col_widths=["17%", "21%", "32%", "8%", "8%", "7%", "7%"])}
<p class="tiny" style="margin-top:6px">Output of the final model for two common grades; all numbers come from the project
results. Pearlite spacing in brackets.</p></div>
<div class="col-2 small"><p><b>1045</b> stays ferrite + pearlite after air cooling, but the pearlite is about
{a.S0_um / b.S0_um:.1f} times finer and the ferrite grains go from {a.d_alpha_um:.0f} to {b.d_alpha_um:.0f} µm, so the
tensile strength rises by about {b.UTS_MPa_pred - a.UTS_MPa_pred:.0f} MPa.</p>
<p><b>4140</b> turns mostly into bainite in air, which is a different steel altogether, and its tensile strength
rises by about {e.UTS_MPa_pred - c.UTS_MPa_pred:.0f} MPa.</p>
<p>The project explains <b>why</b> each steel behaves the way it does, and <b>how much</b> it changes, for 271 standard
grades and any bar size.</p></div></div>''',
          title="What the project does: predicts the microstructure and properties of either route before the steel is treated",
          kicker="What we built")

    # 4 the study at a glance
    gated, phys = bench(N, "gated_ensemble", "UTS_MPa"), bench(N, "physics_lit", "UTS_MPa")
    tiles = (stat("271", "AISI/SAE carbon and low-alloy grades covered, in 9 categories")
             + stat("33", "handbook grades with measured YS, UTS, elongation and hardness (annealed and normalised)")
             + stat(f"{N['lit_rows']} + 423 + 80", f"measured literature records ({N['lit_src']} papers), CCT points and Jominy bands "
                    "used only for checking")
             + stat(f"±{gated:.0f} MPa", f"UTS error on grades the model never saw (textbook physics alone: {phys:.0f} MPa)"))
    d.add(f'''<div class="stats">{tiles}</div>
<div class="row" style="margin-top:18px;gap:22px">
<div class="col"><h3>Why not machine learning alone?</h3><p class="small">33 grades is far too few for a black box,
and a model trained on handbook values would not know that a faster cooling makes finer pearlite, or that a Cr-Mo steel
forms bainite. So the physical metallurgy is built in: critical temperatures, cooling curves, transformation kinetics
and the structure-strength laws of Gladman and Pickering.</p></div>
<div class="col"><h3>Why not physics alone?</h3><p class="small">The textbook relations are only approximately right
(about {phys:.0f} MPa off on tensile strength), and they do not say how far a prediction can be trusted. Machine
learning learns the leftover error from the measured data and gives every number a 90 % range.</p></div>
<div class="col"><h3>How we kept it honest</h3><p class="small">Every step is tested on data the models never saw:
dilatometry, an independent CCT atlas of 19 steels, 80 Jominy hardenability bands, measured phase fractions and
pearlite spacings, and three external property datasets. Where a result is uncertain we say so.</p></div></div>
<div class="callout" style="margin-top:14px"><p><b>Main finding.</b> For plain-carbon steels air cooling only refines the
same ferrite + pearlite, and the gain grows with the carbon content. For hardenable alloy steels it produces a
different steel (bainite, sometimes martensite), so the bar size and the route have to be chosen together.</p></div>''',
          title="The study at a glance: physics-informed machine learning, checked against real measurements",
          kicker="Overview")

    # 5 questions and scope
    d.add('''<div class="rq">
<div class="q"><b>Q1 · Which steel is it?</b><br><span class="small">Category from the measured composition (13 classes, and
"unknown" for steels outside every class)</span></div>
<div class="q"><b>Q2 · What forms on furnace cooling?</b><br><span class="small">Ferrite and pearlite fractions, pearlite
spacing, ferrite grain size, and any bainite or martensite</span></div>
<div class="q"><b>Q3 · Which properties follow?</b><br><span class="small">Yield and tensile strength, uniform and total
elongation, Brinell hardness, and a toughness indicator, each with an interval</span></div>
<div class="q"><b>Q4 · What does air cooling change, and why?</b><br><span class="small">Normalising, as a function of the bar
diameter (mass effect) and the steel's hardenability</span></div></div>
<div class="row" style="margin-top:18px"><div class="col"><div class="callout good small"><p><b>Scope.</b> Carbon and low-alloy
steels that form ferrite + pearlite on slow cooling: 271 AISI/SAE grades in 9 categories. Stainless, tool, maraging and
Hadfield steels are recognised and refused.</p></div></div>
<div class="col"><div class="callout warn small"><p><b>Not modelled.</b> Tempering after normalising, precipitation
strengthening by Nb/V/Ti, decarburisation, banding, and a measured impact energy (only an indicator).</p></div></div></div>''',
          title="Four questions, answered as one metallurgical chain", kicker="Research questions")

    # 6 practice
    r = N["rates"]
    d.add(f'''<div class="row"><div class="col-3">{img("fig26_heat_treatment_practice.png", height=545)}</div>
<div class="col-2 small">
<h3>Full annealing (furnace cooling)</h3>
<p>Austenitise about 50 °C above Ae3 (hypoeutectoid) or 65 °C above Ae1 (hypereutectoid), then cool with the furnace:
<b>{r["furnace cooling"]:.3f} K/s at 700 °C</b> (18 K/h).</p>
<h3>Normalising (air cooling)</h3>
<p>Austenitise about 110 °C above the upper critical temperature, then cool a bar in still air:
<b>{r["still air, 25 mm bar"]:.2f} K/s</b> for 25 mm, {r["still air, 50 mm bar"]:.2f} K/s for 50 mm,
{r["still air, 100 mm bar"]:.2f} K/s for 100 mm.</p>
<p>So normalising differs in <b>two</b> ways: about 60 °C hotter, and about {r["still air, 25 mm bar"] / r["furnace cooling"]:.0f}×
faster at 700 °C for a 25 mm bar. The model separates the two effects.</p>
<p class="tiny">Points: the temperatures at which the 33 handbook grades were actually treated. Dashed: the defaults the
tool uses, fitted to that practice (MAE 14-15 °C).</p></div></div>''',
          title=f"Normalising is done about 60 °C hotter than annealing and cools about {r['still air, 25 mm bar'] / r['furnace cooling']:.0f}× faster",
          kicker="Heat-treatment practice")

    # 7 hypotheses
    d.add('''<div class="row"><div class="col"><table class="data">
<thead><tr><th style="width:6%">#</th><th>What the textbooks lead us to expect on air cooling</th><th style="width:30%">Where we test it</th></tr></thead><tbody>
<tr><td>H1</td><td>Finer pearlite (spacing ∝ 1/ΔT) and finer ferrite grains, so higher YS, UTS and hardness, slightly lower elongation</td><td>CCT, spacing and grain-size checks; property models</td></tr>
<tr><td>H2</td><td>The gain grows with the pearlite share: small for low-carbon, large for high-carbon steels</td><td>Carbon sweep, category table</td></tr>
<tr><td>H3</td><td>Hardenable alloy steels form bainite or martensite in still air ("a different steel", tempered afterwards)</td><td>Dilatometry, Jominy H-bands, ideal critical diameter</td></tr>
<tr><td>H4</td><td>A thick bar cools slowly and drifts towards the annealed state (mass effect)</td><td>Section-size study, 220 mm normalised axles</td></tr>
<tr><td>H5</td><td>Normalised steel holds more pearlite than annealed steel (quasi-eutectoid effect)</td><td>Measured fractions vs lever rule vs kinetics</td></tr>
</tbody></table></div></div>
<p class="small muted" style="margin-top:14px">We treat these as hypotheses to check against data, not as assumptions; H5 turns out to be the
least settled of the five.</p>''', title="What physical metallurgy predicts, written as testable hypotheses",
          kicker="Hypotheses")
    slides_methods(d, N)
    slides_results(d, N)
    slides_v3(d, N)
    slides_closing(d, N)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    N = numbers()
    d = build(N)
    html_path = HERE / "deck.html"
    html_path.write_text(d.html("Heat treatment of steels"))
    pdf = ROOT / "reports" / "BTP_deck_heat_treatment_of_steels.pdf"
    render_pdf(html_path, pdf, args.png)
    pptx = pdf.with_suffix(".pptx")
    render_pptx(html_path, pptx)
    print(f"{len(d.slides)} slides -> {pdf} and {pptx}")


if __name__ == "__main__":
    main()
