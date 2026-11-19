"""Interactive what-if tool: furnace cooling vs air cooling of a steel, and the
five classic heat treatments side by side (second tab).

    python app/app.py            # then open http://127.0.0.1:7860

Pick a standard grade or type a composition; the app classifies the steel,
predicts the microstructure after furnace cooling and after still-air
cooling, the tensile properties with 90 % intervals, draws the CCT diagram
with both cooling paths, and explains in words what air cooling changes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steelml import categories as cat  # noqa: E402
from steelml import config, viz  # noqa: E402
from steelml.cct import draw_cct  # noqa: E402
from steelml.physics import thermo  # noqa: E402
from steelml.predictor import SteelPredictor  # noqa: E402

plt = viz.plt
SP = SteelPredictor.load()
CATALOG = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv")
SCOPE = CATALOG[CATALOG.in_fp_scope & CATALOG.standard.eq("AISI/SAE")].sort_values("grade")
GRADES = ["(custom)"] + SCOPE.grade.tolist()
INPUT_ELEMENTS = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Cu", "S", "P"]


def grade_composition(grade):
    if grade == "(custom)":
        return [None] * len(INPUT_ELEMENTS)
    r = SCOPE[SCOPE.grade == grade].iloc[0]
    return [round(float(r[f"{el}_nom"]), 3) for el in INPUT_ELEMENTS]


def _comp(values):
    comp = {el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS}
    for el, v in zip(INPUT_ELEMENTS, values):
        if v is not None and np.isfinite(v):
            comp[el] = float(v)
    return comp


def explain(fc, ac):
    d = lambda t: ac[f"{t}_pred"] - fc[f"{t}_pred"]  # noqa: E731
    lines = []
    hard_fc = fc.B_best + fc.M_best
    hard_ac = ac.B_best + ac.M_best
    if hard_ac - hard_fc > 0.1:
        lines.append(f"Air cooling is fast enough for this steel's hardenability to suppress part of the "
                     f"ferrite-pearlite reaction: about {100 * hard_ac:.0f} % bainite + martensite forms instead "
                     f"(furnace cooling: {100 * hard_fc:.0f} %). This is why the strength jump is large; such a "
                     f"steel is usually tempered after normalising.")
    else:
        lines.append(f"Both routes give ferrite + pearlite. Air cooling transforms the austenite at a lower "
                     f"temperature, so the pearlite is finer (spacing about {1000 * fc.S0_um:.0f} -> "
                     f"{1000 * ac.S0_um:.0f} nm) and the ferrite grains are smaller (about {fc.d_alpha_um:.0f} -> "
                     f"{ac.d_alpha_um:.0f} µm). The pearlite fraction is {100 * fc.P_best:.0f} % -> "
                     f"{100 * ac.P_best:.0f} % on the lever-rule split that measured normalised steels follow; "
                     f"the textbook quasi-eutectoid effect (extra pearlite on faster cooling) could raise the "
                     f"air-cooled value towards {100 * ac.P_upper:.0f} %, the kinetic model's upper bound.")
    lines.append(f"Predicted change on air cooling: UTS {d('UTS_MPa'):+.0f} MPa, yield {d('YS_MPa'):+.0f} MPa, "
                 f"total elongation {d('El_pct'):+.1f} %, hardness {d('HB'):+.0f} HB.")
    if not bool(fc.in_validity_window):
        lines.append("Caution: the composition is outside the window of the empirical critical-temperature "
                     "equations (for example more than 1 % Si, where they move the eutectoid carbon the wrong way), "
                     "so the eutectoid point, the lever split and the phase fractions are unreliable.")
    if not bool(ac.microstructure_in_data):
        lines.append("Caution: air cooling this bar gives martensite, which none of the training steels had, so "
                     "the properties come from the physics-anchored members alone; treat them as estimates.")
    if not bool(fc.in_domain):
        lines.append("Caution: this composition is outside the range of the training grades, so the prediction "
                     "is an extrapolation; its interval widens where the two model members disagree, and it "
                     "should be checked by a test.")
    return " ".join(lines)


def run(grade, C, Mn, Si, Cr, Ni, Mo, V, Cu, S, P, section_mm, fc_rate_kh, T_aust):
    comp = _comp([C, Mn, Si, Cr, Ni, Mo, V, Cu, S, P])
    T = None if (T_aust is None or T_aust <= 0) else float(T_aust)
    out = SP.what_if(comp, section_mm=float(section_mm), fc_rate=float(fc_rate_kh) / 3600.0, T_aust=T)
    fc, ac = out.iloc[0], out.iloc[1]
    # category
    c_lab = cat.CATEGORY_LABELS.get(fc.category, fc.category)
    cat_txt = f"**{c_lab}** (confidence {100 * fc.category_confidence:.0f} %)"
    rules_lab = cat.CATEGORY_LABELS.get(fc.category_rules, fc.category_rules)
    cat_txt += ("  \nAISI/SAE composition rules: " + rules_lab +
                (" (agrees)" if fc.category_rules_agree else " (disagrees: a borderline composition)"))
    if fc.category_unknown:
        cat_txt += "  \n:warning: composition unlike any known category - treat everything below with caution."
    if fc.category not in cat.FERRITE_PEARLITE_SCOPE:
        # stainless and tool steels: the critical-temperature formulas, the
        # kinetics and the property models are all outside their validity, so
        # nothing is predicted rather than showing misleading numbers
        cat_txt += "  \n:warning: this category does not decompose to ferrite + pearlite on slow cooling."
        msg = ("This steel is outside the scope of the cooling-route models, which are built for carbon and "
               "low-alloy steels that form ferrite + pearlite when furnace cooled. No microstructure or property "
               "prediction is shown for it.")
        return cat_txt, "Critical temperatures not shown (the formulas are valid for low-alloy steels only).", \
            pd.DataFrame(columns=["", "Furnace cooled", "Air cooled"]), None, None, msg
    ce, te = thermo.eutectoid_point(comp)
    crit = (f"Ae1 {float(thermo.ae1(comp)):.0f} °C | Ae3 {float(thermo.ae3(comp)):.0f} °C | "
            f"eutectoid {float(ce):.2f} wt% C at {float(te):.0f} °C | {str(thermo.eutectoid_class(comp))} | "
            f"austenitised at {fc.treat_T_C:.0f} °C (furnace) / {ac.treat_T_C:.0f} °C (air) | CE(IIW) {fc.CE_IIW:.2f} | "
            f"ideal critical diameter D_I {fc.DI_mm:.0f} mm")
    rows = []
    band = lambda r, a, b: (f"{100 * r[a]:.0f}" if abs(r[a] - r[b]) < 0.015  # noqa: E731
                            else f"{100 * r[a]:.0f}  ({100 * min(r[a], r[b]):.0f}-{100 * max(r[a], r[b]):.0f})")
    rows.append(["Ferrite (%), lever split (band to the kinetic split)", band(fc, "F_best", "F_lower"),
                 band(ac, "F_best", "F_lower")])
    rows.append(["Pearlite (%), lever split (band to the kinetic split)", band(fc, "P_best", "P_upper"),
                 band(ac, "P_best", "P_upper")])
    for lab, key in [("Proeutectoid cementite (%)", "Fe3C_best"), ("Bainite (%)", "B_best"),
                     ("Martensite (%)", "M_best")]:
        rows.append([lab, f"{100 * fc[key]:.0f}", f"{100 * ac[key]:.0f}"])
    rows.append(["Pearlite spacing (nm)", f"{1000 * fc.S0_um:.0f}" if fc.P_best > 0.05 else "-",
                 f"{1000 * ac.S0_um:.0f}" if ac.P_best > 0.05 else "-"])
    rows.append(["Ferrite grain size (µm, about ±2x)", f"{fc.d_alpha_um:.0f}" if fc.F_best > 0.05 else "-",
                 f"{ac.d_alpha_um:.0f}" if ac.F_best > 0.05 else "-"])
    rows.append(["Cooling rate at 700 °C (K/s)", f"{fc.cool_rate_700:.3g}", f"{ac.cool_rate_700:.3g}"])
    for lab, t in [("Yield strength (MPa)", "YS_MPa"), ("Tensile strength (MPa)", "UTS_MPa"),
                   ("Total elongation (%)", "El_pct"), ("Brinell hardness (HB)", "HB")]:
        rows.append([lab, f"{fc[t + '_pred']:.0f}  [{fc[t + '_lo90']:.0f}, {fc[t + '_hi90']:.0f}]",
                     f"{ac[t + '_pred']:.0f}  [{ac[t + '_lo90']:.0f}, {ac[t + '_hi90']:.0f}]"])
    rows.append(["Uniform elongation, estimate (%)",
                 f"{fc.uniform_El_est_pct:.1f}  [{fc.uniform_El_lo90:.0f}, {fc.uniform_El_hi90:.0f}]",
                 f"{ac.uniform_El_est_pct:.1f}  [{ac.uniform_El_lo90:.0f}, {ac.uniform_El_hi90:.0f}]"])
    rows.append(["Strain-hardening exponent n (Considère)", f"{fc.n_est:.2f}", f"{ac.n_est:.2f}"])
    rows.append(["Inside training domain", "yes" if fc.in_domain else "no", "yes" if ac.in_domain else "no"])
    rows.append(["Microstructure seen in training data", "yes" if fc.microstructure_in_data else "no",
                 "yes" if ac.microstructure_in_data else "no"])
    table = pd.DataFrame(rows, columns=["", "Furnace cooled", f"Air cooled ({section_mm:.0f} mm bar)"])

    # CCT figure (same drawing as figure 9 of the report)
    fig1, ax = plt.subplots(figsize=(6.6, 4.8))
    draw_cct(ax, "CCT diagram with the two cooling paths", comp, predictor=SP, fontsize=8.5,
             section_mm=float(section_mm), fc_rate=float(fc_rate_kh) / 3600.0, T_aust=T)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=7.5, loc="lower left")
    fig1.tight_layout()

    # section-size figure
    Ds = [10, 15, 25, 35, 50, 75, 100, 150]
    X = pd.DataFrame([{**comp, "route": "AC", "section_mm": float(D), **({"treat_T_C": T} if T else {})}
                      for D in Ds])
    so = SP.predict(X)
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.0))
    ax2.fill_between(Ds, so.UTS_MPa_lo90, so.UTS_MPa_hi90, color=viz.ORANGE, alpha=0.15, lw=0)
    ax2.plot(Ds, so.UTS_MPa_pred, "-o", color=viz.ORANGE, ms=4, label="UTS, air cooled")
    ax2.fill_between(Ds, so.YS_MPa_lo90, so.YS_MPa_hi90, color=viz.AQUA, alpha=0.15, lw=0)
    ax2.plot(Ds, so.YS_MPa_pred, "-o", color=viz.AQUA, ms=4, label="Yield, air cooled")
    ax2.axhline(fc.UTS_MPa_pred, color=viz.BLUE, lw=1.5, label="UTS, furnace cooled")
    ax2.axhline(fc.YS_MPa_pred, color=viz.BLUE, lw=1.0, alpha=0.6, label="Yield, furnace cooled")
    ax2.set_xscale("log")
    ax2.set_xlabel("Bar diameter (mm)")
    ax2.set_ylabel("MPa")
    ax2.legend(fontsize=8)
    ax2.set_title("Effect of section size on air cooling")
    fig2.tight_layout()
    expl = explain(fc, ac)
    if T is not None:
        usual = {r: float(thermo.default_austenitising_T(comp, r)) for r in ("FC", "AC")}
        if max(abs(T - usual["FC"]), abs(T - usual["AC"])) > 40.0:
            expl += (f" Note: the models learned from bars annealed at about {usual['FC']:.0f} °C and normalised at "
                     f"about {usual['AC']:.0f} °C for this steel; at {T:.0f} °C the predictions are less certain.")
    return cat_txt, crit, table, fig1, fig2, expl


TREAT_LABEL = {"anneal": "Annealed (furnace cooled)", "normalise": "Normalised (air cooled)",
               "harden": "Hardened (quenched)", "quench_temper": "Quenched + tempered",
               "stress_relieve": "Stress relieved (after normalising)"}


def run_five(grade, *vals):
    *elv, section_mm, medium, temper_T, sr_T = vals
    comp = _comp(elv)
    med = None if medium == "automatic" else medium
    out = SP.compare_treatments(comp, section_mm=float(section_mm), medium=med, temper_T=float(temper_T),
                                sr_T=float(sr_T))
    rows = []
    for _, r in out.iterrows():
        uts = ("brittle (hardness only)" if r.treatment == "harden"
               else f"{r.UTS_MPa_pred:.0f} [{r.UTS_MPa_lo90:.0f}-{r.UTS_MPa_hi90:.0f}]")
        ys = "" if r.treatment == "harden" else f"{r.YS_MPa_pred:.0f} [{r.YS_MPa_lo90:.0f}-{r.YS_MPa_hi90:.0f}]"
        el = "" if r.treatment == "harden" else f"{r.El_pct_pred:.0f}"
        rows.append({"treatment": TREAT_LABEL[r.treatment], "quench / cooling": r.medium,
                     "microstructure": r.microstructure, "UTS (MPa) [90 %]": uts, "YS (MPa) [90 %]": ys,
                     "El (%)": el, "HV": f"{r.HV:.0f}"})
    table = pd.DataFrame(rows)
    notes = []
    q = out[out.treatment == "quench_temper"].iloc[0]
    if bool(q.get("TME_risk", False)):
        notes.append("tempering at 250-400 C: tempered-martensite embrittlement window")
    if bool(q.get("TE_risk", False)):
        notes.append("tempering at 375-575 C: temper-embrittlement risk (slow cooling after tempering)")
    sr = out[out.treatment == "stress_relieve"].iloc[0]
    if not bool(sr.get("subcritical", True)):
        notes.append(f"stress relief at {sr_T:.0f} C is above Ac1 - 25 C ({sr.Ac1_C:.0f} C): not a stress relief")
    else:
        notes.append(f"stress relief at {sr_T:.0f} C: residual stress at most {sr.residual_stress_bound_MPa:.0f} MPa "
                     "(yield strength at temperature)")
    if str(out.notes.iloc[0]):
        notes.append(str(out.notes.iloc[0]))
    md = "**Notes:** " + "; ".join(notes) if notes else ""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(out))
    ax.bar(x, out.HV, color=[viz.BLUE, viz.ORANGE, viz.YELLOW, viz.AQUA, viz.NEUTRAL])
    ax.set_xticks(x)
    ax.set_xticklabels(["annealed", "normalised", "hardened", "Q + T", "stress rel."], fontsize=9)
    ax.set_ylabel("hardness (HV)")
    fig.tight_layout()
    return table, md, fig


def build():
    import gradio as gr

    with gr.Blocks(title="Heat treatment of steels") as demo:
        gr.Markdown("## Furnace cooling vs air cooling, and the rest of the heat-treatment route\n"
                    "Physics-informed ML (B.Tech project, IIT Roorkee). Pick a grade or type a composition (wt%).")
        with gr.Tab("Furnace vs air cooling"):
            with gr.Row():
                grade = gr.Dropdown(GRADES, value="AISI 1045", label="Standard grade")
                section = gr.Slider(10, 150, value=25, step=1, label="Air-cooled bar diameter (mm)")
                fc_rate = gr.Slider(5, 150, value=18, step=1, label="Furnace cooling rate at 700 °C (K/h)")
                T_aust = gr.Number(value=0, label="Austenitising temperature (°C), 0 = automatic")
            with gr.Row():
                els = [gr.Number(value=v, label=f"{el} (wt%)") for el, v in
                       zip(INPUT_ELEMENTS, grade_composition("AISI 1045"))]
            btn = gr.Button("Predict", variant="primary")
            cat_md = gr.Markdown()
            crit_md = gr.Markdown()
            table = gr.Dataframe(interactive=False)
            expl = gr.Markdown()
            with gr.Row():
                p1 = gr.Plot()
                p2 = gr.Plot()
            grade.change(grade_composition, grade, els)
            btn.click(run, [grade] + els + [section, fc_rate, T_aust], [cat_md, crit_md, table, p1, p2, expl])
        with gr.Tab("All five heat treatments"):
            gr.Markdown("Annealing, normalising, hardening, hardening + tempering and stress relieving of one bar.")
            with gr.Row():
                grade2 = gr.Dropdown(GRADES, value="AISI 4140H", label="Standard grade")
                section2 = gr.Slider(10, 150, value=25, step=1, label="Bar diameter (mm)")
                medium2 = gr.Dropdown(["automatic", "oil", "water", "brine"], value="automatic", label="Quench")
                temper2 = gr.Slider(150, 700, value=600, step=5, label="Tempering temperature (°C), 1 h")
                sr2 = gr.Slider(450, 700, value=600, step=5, label="Stress-relief temperature (°C), 1 h")
            with gr.Row():
                els2 = [gr.Number(value=v, label=f"{el} (wt%)") for el, v in
                        zip(INPUT_ELEMENTS, grade_composition("AISI 4140H"))]
            btn2 = gr.Button("Predict all five", variant="primary")
            table2 = gr.Dataframe(interactive=False)
            notes2 = gr.Markdown()
            plot2 = gr.Plot()
            grade2.change(grade_composition, grade2, els2)
            btn2.click(run_five, [grade2] + els2 + [section2, medium2, temper2, sr2], [table2, notes2, plot2])
    return demo


if __name__ == "__main__":
    build().launch()
