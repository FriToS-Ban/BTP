"""Build the A1 poster (594 x 841 mm), version 2, on the department template.

    python reports/poster_v2/make_figures.py     # figures at print size (SVG + PNG)
    python reports/poster_v2/build_poster.py     # poster.html -> BTP_poster_A1_v2.pdf, preview, layout.json
    python reports/poster_v2/export_pptx.py      # the same layout as an editable PPTX on Template.pptx

The frame is the department template (Template.pptx): its gradient header band, IIT Roorkee logo, page border
and column divider, with the footer band of the reference poster. The page is laid out by Chrome, which also
prints the PDF, so what is checked on screen is what is printed. Every number in the text is read from
results/ by numbers() below, never typed in.
"""

import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
CACHE = HERE / ".cache"
sys.path.insert(0, str(HERE))
from make_figures import HALF, ROUTE_PICK, SIZES  # noqa: E402  (slot sizes = figure sizes)
from poster_fonts import sans_fonts  # noqa: E402

OUT_NAME = "BTP_poster_A1_v2"

TITLE = "Classify, Heat-Treat, Predict: Machine Learning for Steels"
SUBTITLE = ("One model names a carbon or low-alloy steel, follows it through five heat treatments and predicts its "
            "microstructure and properties")
COURSE = ("B.Tech Project (MTP-400A), Autumn 2026 &nbsp; | &nbsp; Department of Metallurgical and Materials "
          "Engineering, Indian Institute of Technology Roorkee")
AUTHORS = ("Shibasish Banerjee (23118071), Aviral Singh (23118017) &nbsp; | &nbsp; "
           "Supervisor: Dr. Anish Karmakar")
REPO = "github.com/FriToS-Ban/BTP"

# ------------------------------------------------------------------ page geometry (mm) and type scale (pt)
PAGE_W, PAGE_H = 594.12, 840.98
MARGIN, GUTTER = 15.0, 15.0
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
HEADER_H, FOOTER_H = 60.0, 16.0
TOP = HEADER_H + 5.0
BOTTOM = PAGE_H - FOOTER_H - 4.0
GAP = COL_W - 2 * HALF          # gap between two half-width figures
# reference-poster scale: headings 28, body 20, takeaways 17, captions 16, small print 15; the title is set
# at 50 pt and still clears the logo (the template's own title line is 54 pt, the reference poster's 40 pt)
S_TITLE, S_SUB, S_HDR, S_AUTH, S_HEAD, S_BODY, S_TAKE, S_CAP, S_SMALL = 50, 24, 22, 28, 28, 20, 17, 16, 15
BLUE, NAVY, INK, PALE, FOOT = "#0070C0", "#1F3973", "#111111", "#E8F1FA", "#F1AA84"


# ------------------------------------------------------------------ numbers from results/
def numbers():
    n = {}
    rp = pd.read_csv(RES / "s01_route_pairs.csv").set_index("grade")
    n["fc4140"], n["ac4140"] = rp.loc["AISI 4140", "UTS_MPa_FC"], rp.loc["AISI 4140", "UTS_MPa_AC"]
    n["routes"] = (rp.UTS_MPa_AC - rp.UTS_MPa_FC).to_dict()
    b = pd.read_csv(RES / "s04_model_benchmark.csv").set_index(["target", "model"])
    n["an_uts"] = b.loc[("UTS_MPa", "gated_ensemble"), "RMSE_inAD"]
    n["an_uts_txt"] = b.loc[("UTS_MPa", "physics_lit"), "RMSE_inAD"]
    n["an_cov"] = pd.read_csv(RES / "s04_conformal.csv").set_index("target").loc["UTS_MPa", "LOGO_coverage_90"]
    q = pd.read_csv(RES / "s09_qt_benchmark.csv").set_index(["model", "target"])
    n["qt_uts"], n["qt_uts_txt"] = q.loc[("qt_ensemble", "UTS_MPa"), "RMSE"], q.loc[("qt_physics", "UTS_MPa"), "RMSE"]
    n["qt_cov"] = q.loc[("qt_ensemble", "UTS_MPa"), "coverage_nested"]
    qo = pd.read_csv(RES / "s09_qt_oof.csv")
    n["qt_bars"], n["qt_grades"] = len(qo), qo.grade.nunique()
    q41 = qo[qo.grade.astype(str) == "4140"]
    n["qt4140_lo"], n["qt4140_hi"] = q41.UTS_MPa.min(), q41.UTS_MPa.max()
    c = pd.read_csv(RES / "s05_category_summary.csv").set_index("category")
    g = c.UTS_MPa_delta_median
    n["gain_low"], n["gain_med"], n["gain_high"] = g["Low-carbon steel"], g["Medium-carbon steel"], g["High-carbon steel"]
    n["gain_crmo"] = g["Mo / Cr-Mo steel"]
    n["el_high"] = c.El_pct_delta_median["High-carbon steel"]
    n["bm_crmo"], n["bm_nicrmo"] = (c.AC25_bainite_martensite_mean["Mo / Cr-Mo steel"],
                                    c.AC25_bainite_martensite_mean["Ni-Cr-Mo / Ni steel"])
    n["fam_grades"] = int(c.grades.sum() - c.grades["Si-Mn spring steel"])
    j = pd.read_csv(RES / "s08_kinetic_calibration_cv_metrics.csv")
    j = j[j.distances == "all distances"].set_index("kinetics")
    n["jom_pub"], n["jom_cal"] = j.loc["HRC_published", "RMSE_HRC"], j.loc["HRC_calibrated_cv", "RMSE_HRC"]
    n["jom_pts"] = int(j.loc["HRC_calibrated_cv", "n"])
    n["jom_grades"] = pd.read_csv(RES / "s08_kinetic_calibration_cv.csv").grade.nunique()
    bs = pd.read_csv(RES / "s06_bar_sizes_metrics.csv")
    r = bs[bs.check.str.startswith("bar-size effect")].iloc[0]
    n["bs_n"], n["bs_sign"], n["bs_pred"], n["bs_meas"] = int(r.n), r.sign_agreement, r.mean_predicted, r.mean_measured
    u = bs[(bs.target == "UTS_MPa") & bs.check.isin(["normalised, handbook grades, other bar sizes",
                                                      "normalised, grades never seen"])]
    n["nb_bars"] = int(u.n.sum())
    n["nb_uts"] = float(np.sqrt((u.n * u.RMSE ** 2).sum() / u.n.sum()))
    hc = pd.read_csv(RES / "s09_hardenability_check.csv")
    n["di_ratio"] = float(np.median(hc.DI_model_calibrated_mm / hc.DI_A255_mm))
    ft = pd.read_csv(RES / "s09_five_treatments.csv")
    f = ft[ft.grade.astype(str) == "4140"].set_index("treatment")
    n["ft_min"], n["ft_max"] = f.UTS_MPa_pred.min(), f.UTS_MPa_pred.max()
    ds = pd.read_csv(RES / "s01_dataset_summary.csv").set_index("dataset")
    n["hb_grades"] = int(ds.loc["properties_handbook", "steels_or_grades"])
    lit = pd.read_csv(ROOT / "data/raw/literature_compilation.csv")
    n["lit_rows"], n["lit_papers"] = len(lit), lit.source_id.nunique()
    atlas = pd.read_csv(RES / "s03_cct_atlas_summary.csv").set_index("subset").loc["all points"]
    n["cct_pts"], n["cct_steels"] = int(atlas.n), int(atlas.steels)
    n["cct_hv"], n["cct_fp"], n["cct_b"] = atlas.HV_RMSE, atlas.MAE_ferrite_pearlite, atlas.MAE_bainite
    ks = pd.read_csv(RES / "s03_kinetics_summary.csv")
    ks = ks[ks.model == "this work"].set_index("start")
    n["fs_mae"], n["ms_mae"], n["ps_mae"] = ks.MAE_C["ferrite"], ks.MAE_C["martensite"], ks.MAE_C["pearlite"]
    n["start_pts"] = int(ks.n.sum())
    n["families"] = len(pd.read_csv(RES / "s02_per_category_recall.csv"))
    cl = pd.read_csv(RES / "s02_classifier_cv.csv")
    n["cls_family"] = float(cl[cl.production].steelbench_family_accuracy.iloc[0])
    ex = pd.read_csv(RES / "s02_external_tests.csv").set_index("test")
    n["cls_heats"] = int(ex.loc["SteelBench all", "n"])
    cool = pd.read_csv(RES / "s03_cooling_paths.csv").set_index("path").rate_700C_K_s
    n["air_vs_furnace"] = cool["still air, 25 mm bar"] / cool["furnace cooling"]
    tc = pd.read_csv(RES / "s09_tempering_curves.csv")
    t = tc[tc.grade.astype(str) == "4140"]
    m = qo[(qo.grade.astype(str) == "4140") & (qo.section_mm == 25) & (qo.quench_medium == "oil")]
    n["temp_meas_lo"], n["temp_meas_hi"] = m.temper_T_C.min(), m.temper_T_C.max()
    t = t[(t.temper_T_C >= 200) & (t.temper_T_C <= 700)]
    n["temp_lo"], n["temp_hi"] = t.UTS_MPa_pred.min(), t.UTS_MPa_pred.max()
    me = pd.read_csv(RES / "s09_mass_effect.csv")
    me = me[(me.grade.astype(str) == "4140") & (me.quench_medium == "oil")]
    full = [tt for tt, gg in me.groupby("temper_T_C") if gg.section_mm.nunique() >= 4]
    n["qm_temps"] = (min(full), max(full))
    c41 = json.loads((HERE / "data/cct_4140.json").read_text())["routes"]
    c45 = json.loads((HERE / "data/cct_1045.json").read_text())["routes"]
    n["cct45_same"] = all(abs(c45["FC"][k] - c45["AC"][k]) < 0.005 for k in "FPBM") and c45["AC"]["B"] < 0.01
    n["cct_ac_B"] = c41["AC"]["B"]
    hm = pd.read_csv(RES / "s10_handbook_section_match.csv").set_index("section_mm")
    n["hm_within"], n["hm_grades"] = int(hm.loc[25.0, "within_3MPa"]), int(hm.loc[25.0, "grades"])
    n["assumptions"] = len(pd.read_csv(RES / "s10_assumptions.csv"))
    mc = pd.read_csv(RES / "s11_mcmc_diagnostics.csv")
    n["rhat"], n["ess"] = mc.split_Rhat.max(), mc.ESS.min()
    tk = pd.read_csv(RES / "s09_tempering_constants.csv").set_index("constant").fitted
    n["temper_sC"] = tk["s_C"]                       # carbon's effect on fully tempered hardness, HV per % C
    shifts = tk[["p_Mn", "p_Si", "p_Cr", "p_Mo", "p_V"]]
    n["temper_slowest"] = {"p_Mn": "manganese", "p_Si": "silicon", "p_Cr": "chromium", "p_Mo": "molybdenum",
                           "p_V": "vanadium"}[shifts.idxmax()]
    di = pd.read_csv(RES / "s08_ideal_diameter_catalogue.csv")
    di = di[di.category_true != "si_mn"]
    lo, hi = di[di.DI_mm < 40], di[di.DI_mm >= 100]
    n["di_grades"] = len(di)
    n["di_lo"] = (int((lo.hard_AC25 > 0.5).sum()), len(lo))
    n["di_hi"] = (int((hi.hard_AC25 > 0.5).sum()), len(hi))
    n["di_rho"] = float(spearmanr(di.DI_mm, di.dUTS_AC25).correlation)
    return n


# ------------------------------------------------------------------ small HTML helpers
UNIT_RE = re.compile(r"(\d) (%|MPa|HRC|HV|HB|mm|°C|K/s|µm|nm|h(?![A-Za-z]))")


def nbsp(html):
    """Keep a number and its unit on one line."""
    return UNIT_RE.sub(lambda m: m.group(1) + " " + m.group(2), html)


def fig_img(key, name):
    w, h = SIZES[key]
    return f'<img class="pic" data-fig="{name}" src="figures/{name}.svg" style="width:{w}mm;height:{h}mm" alt="">'


def cap(no, text):
    return f'<div class="cap" data-tb="caption {no}"><b>Fig {no}:</b> {text}</div>'


def take(no, text):
    return f'<div class="take" data-tb="takeaway {no}">{text}</div>'


def fig_full(key, name, no, caption, takeaway, extra=""):
    return f'<div class="fig">{fig_img(key, name)}{extra}{cap(no, caption)}{take(no, takeaway)}</div>'


def fig_pair(a, b):
    """Two half-width figures whose captions and takeaways line up. a, b = (key, name, no, caption, takeaway)."""
    return ('<div class="pair">' + fig_img(a[0], a[1]) + fig_img(b[0], b[1]) + cap(a[2], a[3]) + cap(b[2], b[3])
            + take(a[2], a[4]) + take(b[2], b[4]) + "</div>")


def bullets(items, name, cls=""):
    lis = "".join(f"<li>{it}</li>" for it in items)
    return f'<ul class="bl {cls}" data-tb="{name}">{lis}</ul>'


ARROW = {
    "r": '<svg viewBox="0 0 20 14" class="arr"><path d="M0 4.2h11V0l9 7-9 7V9.8H0z"/></svg>',
    "l": '<svg viewBox="0 0 20 14" class="arr"><path d="M20 4.2H9V0L0 7l9 7V9.8h11z"/></svg>',
    "d": '<svg viewBox="0 0 14 14" class="arr dn"><path d="M4.2 0v6H0l7 8 7-8H9.8V0z"/></svg>',
}


def graphical_abstract(N):
    steps = [("Inputs", "composition, heat treatment, bar size"),
             ("Classify", f"steel family, one of {N['families']}, or “unknown”"),
             ("Heat-treat", "cooling path in the furnace, in air or in a quench"),
             ("Transform", "kinetics give ferrite, pearlite, bainite, martensite"),
             ("Predict", "textbook strength laws; a Gaussian process learns the rest"),
             ("Properties", "YS, UTS, elongation, hardness, with 90 % intervals")]
    cells = {0: (1, 1), 1: (1, 3), 2: (1, 5), 3: (3, 5), 4: (3, 3), 5: (3, 1)}
    out = ['<div class="ga">']
    for k, (t, s) in enumerate(steps):
        r, c = cells[k]
        out.append(f'<div class="gab{" out" if k == 5 else ""}" data-shape="GA box {k + 1}" '
                   f'style="grid-area:{r}/{c}"><span class="num" data-shape="GA number {k + 1}" '
                   f'data-tb="GA number {k + 1}">{k + 1}</span><div data-tb="GA text {k + 1}">'
                   f'<div class="gt par">{t}</div><div class="gs par">{s}</div></div></div>')
    for k, (r, c, d) in enumerate([(1, 2, "r"), (1, 4, "r"), (2, 5, "d"), (3, 4, "l"), (3, 2, "l")]):
        out.append(f'<div class="gar" data-arrow="{d}" style="grid-area:{r}/{c}">{ARROW[d]}</div>')
    out.append("</div>")
    return "".join(out)


def tiles(items, name):
    out = ['<div class="tiles">']
    for k, (num, lab) in enumerate(items):
        out.append(f'<div class="tile" data-shape="{name} tile {k + 1}"><div data-tb="{name} tile text {k + 1}">'
                   f'<div class="tn par">{num}</div><div class="tl par">{lab}</div></div></div>')
    out.append("</div>")
    return "".join(out)


def equations():
    v = lambda s: f"<i>{s}</i>"  # noqa: E731
    eqs = [("Bainite rate (Li et al.)",
            f"{v('k')}<sub>B</sub> = {v('f')}<sub>B</sub> 2<sup>0.29{v('G')}</sup> ({v('B')}<sub>s</sub> − {v('T')})"
            f"<sup>2</sup> e<sup>−{v('Q')}/{v('RT')}</sup> / BC"),
           ("Physics strength (the anchor)",
            f"σ<sub>phys</sub> = {v('f')}<sub>FP</sub> σ<sub>Gladman</sub> + {v('f')}<sub>BM</sub> · 3.3 HV"
            "<sub>Maynier</sub>"),
           ("Pearlite spacing", f"{v('S')}<sub>0</sub> = 3.33 Δ{v('T')}<sup> −0.607</sup> µm"),
           ("Anchored Gaussian process", f"UTS = {v('c')} σ<sub>phys</sub> e<sup>δ({v('x')})</sup>, &nbsp;δ ∼ GP"),
           ("Hollomon–Jaffe parameter", f"{v('P')} = {v('T')} (20 + log<sub>10</sub> {v('t')})"),
           ("90 % interval (conformal)", f"ŷ ± {v('q')} σ({v('x')})")]
    out = ['<div class="eqbox" data-shape="equations box"><div class="eqgrid">']
    for k, (lab, src) in enumerate(eqs):
        out.append(f'<div class="eqc"><div class="eql" data-tb="eq label {k + 1}">{k + 1}. {lab}</div>'
                   f'<div class="eq" data-tb="equation {k + 1}">{src}</div></div>')
    out.append("</div></div>")
    return "".join(out)


def novelty(N):
    """What exists today against what this work adds; the left column follows the report's research gap."""
    rows = [("Commercial tools are closed", "<b>Open:</b> code, data and every number can be rebuilt"),
            ("Open kinetic models stop at the microstructure",
             "<b>Goes on to properties,</b> each with a 90 % interval"),
            ("ML models usually skip the heat treatment",
             "<b>Five heat treatments</b> and any bar size in one chain"),
            ("Textbook laws alone, or composition-only ML",
             f"<b>Physics extrapolates, ML corrects:</b> UTS error {N['an_uts_txt']:.0f} → {N['an_uts']:.0f} MPa"),
            ("Li kinetics with the published constants",
             f"<b>Bayesian recalibration</b> on {N['jom_pts']} Jominy points"),
            ("Assumptions often left implicit",
             f"<b>{N['assumptions']} assumptions</b> registered and varied; one corrected")]
    out = ['<div class="nov">',
           '<div class="nh nl" data-shape="novelty head 1"><div data-tb="novelty head text 1">What exists today</div>'
           '</div>',
           '<div class="nh nr" data-shape="novelty head 2"><div data-tb="novelty head text 2">What is new in this '
           'work</div></div>']
    for i, (old, new) in enumerate(rows, 1):
        out.append(f'<div class="nc nl" data-shape="novelty before {i}"><div data-tb="novelty before text {i}">{old}'
                   f'</div></div><div class="nc nr" data-shape="novelty new {i}"><div data-tb="novelty new text {i}">'
                   f'{new}</div></div>')
    out.append("</div>")
    return "".join(out)


# ------------------------------------------------------------------ the page
def body(N):
    gains = [N["routes"][g] for g, _ in ROUTE_PICK]
    k = lambda s: f'<b class="k">{s}</b>'  # noqa: E731
    left = f"""
<section>
  <h2 data-tb="heading intro">Introduction and Motivation</h2>
  {bullets([
      f"<b>Heat treatment decides the steel:</b> AISI 4140 is {N['fc4140']:.0f} MPa annealed, {N['ac4140']:.0f} MPa "
      f"normalised and {N['qt4140_lo']:.0f}–{N['qt4140_hi']:.0f} MPa quenched and tempered",
      f"Handbooks list only a few conditions per grade; textbook laws alone err by {N['an_uts_txt']:.0f} MPa",
      "<b>Aim:</b> predict microstructure and properties with physics-informed machine learning",
      "<b>Our supervisor's question:</b> furnace or air cooling – what changes, how much, and why?",
  ], "intro")}
  {fig_full("routes", "f01_handbook_routes", 1,
            "Handbook tensile strength of the same grades, annealed vs normalised",
            f"The cooling route alone adds +{min(gains):.0f} to +{max(gains):.0f} MPa")}
</section>
<section>
  <h2 data-tb="heading novelty">Novelty of This Work</h2>
  {novelty(N)}
</section>
<section>
  <h2 data-tb="heading abstract">Graphical Abstract</h2>
  {graphical_abstract(N)}
  <h3 data-tb="subheading evidence">Every link tested on data the models never saw</h3>
  {tiles([
      (f"{N['cls_family']:.2f}", f"steel family right, {N['cls_heats']} real heats"),
      (f"{N['fs_mae']:.0f} °C", "ferrite-start error vs dilatometry"),
      (f"{N['jom_cal']:.1f} HRC", f"Jominy bands, {N['jom_grades']} unseen H-steels"),
      (f"{N['an_uts']:.0f} MPa", "UTS, annealed and normalised grades"),
      (f"{N['qt_uts']:.0f} MPa", "UTS, quenched + tempered bars"),
      (f"{100 * N['an_cov']:.1f} %", "of values inside the 90 % interval"),
  ], "evidence")}
</section>
<section>
  <h2 data-tb="heading methodology">Methodology</h2>
  <p class="lead" data-tb="data">{k('Open data only,')} each row with its heat treatment: {N['hb_grades']} handbook
  grades, {N['qt_bars']} quenched and tempered bars, {N['jom_pts']} Jominy points, {N['cct_pts']} CCT points,
  {N['lit_rows']} values from {N['lit_papers']} papers</p>
  <div class="split">
    <div class="chain">
      <h3 data-tb="subheading chain">Model chain</h3>
      {bullets([
          f"<b>Steel family:</b> XGBoost, {N['families']} families",
          "<b>Thermodynamics:</b> Ae1, Ae3, Acm, lever rule",
          "<b>Heat flow:</b> furnace, air-cooled bar, quench",
          "<b>Kinetics:</b> Li et al.; bainite recalibrated on Jominy bands (Bayesian)",
          "<b>Tempering:</b> Hollomon–Jaffe master curve",
          "<b>Properties:</b> Gaussian processes anchored on physics strength",
          "<b>Tests:</b> grade held out; 90 % intervals",
      ], "methods", "tight")}
    </div>
    <div class="fig">{fig_img("cooling", "f02_cooling_rates")}
      {cap(2, "Model cooling rate at 700 °C (quenches: bar centre)")}
      {take(2, f"A 25 mm bar in air cools {N['air_vs_furnace']:.0f}× faster than the furnace")}</div>
  </div>
  {equations()}
</section>
<section>
  <h2 data-tb="heading validation">Validation Against Experiments</h2>
  {fig_pair(
      ("starts", "f03_transformation_starts", 3,
       "Transformation starts vs dilatometry of 3 steels at 0.1–50 K/s; bainite start is the weak link",
       f"Ferrite start within {N['fs_mae']:.0f} °C, Ms within {N['ms_mae']:.0f} °C"),
      ("atlas", "f04_cct_atlas_hardness", 4,
       f"{N['cct_pts']} measured continuous-cooling points of {N['cct_steels']} low-alloy steels (Rose & Hougardy "
       f"atlas)",
       f"Hardness within {N['cct_hv']:.0f} HV; F + P fraction within {N['cct_fp']:.2f}"))}
  {fig_pair(
      ("parity", "f05_parity_unseen_grades", 5,
       "Tensile strength of grades held out of training",
       f"UTS within {N['an_uts']:.0f} MPa; {N['qt_uts']:.0f} MPa after quench + temper"),
      ("ladder", "f06_model_ladder", 6,
       "The same held-out test for each kind of model",
       "Physics + GP halves the textbook-law error"))}
</section>
<section class="fine">
  <h2 data-tb="heading acknowledgements">Acknowledgements</h2>
  <p class="ack" data-tb="acknowledgements"><span class="par">We thank Dr. Anish Karmakar for the question behind
  this work and for always asking how we knew a result was true, and the authors who shared their data openly.
  Code and data: <b>{REPO}</b></span>
  <span class="par"><b>AI use:</b> AI tools assisted with language, structure and code; we reviewed and validated all
  output. The content and conclusions are our own; no data were generated or altered by AI.</span></p>
</section>
"""
    cct_take = (("1045 stays ferrite + pearlite; " if N["cct45_same"] else "")
                + f"4140 forms {100 * N['cct_ac_B']:.0f} % bainite in air: a different steel")
    right = f"""
<section>
  <h2 data-tb="heading results">Results and Interpretation</h2>
  <h3 data-tb="subheading A">A. Furnace or air cooling: what changes, and why</h3>
  {fig_full("carbon", "f07_furnace_vs_air_carbon", 7,
            "Plain-carbon steels (0.75 % Mn), furnace vs 25 mm bar in still air; lines: model with 90 % band",
            f"Air cooling adds +{N['gain_low']:.0f} MPa (low C) to +{N['gain_high']:.0f} MPa (high C): the same "
            f"phases, finer pearlite")}
</section>
<section>
  {fig_full("cct", "f08_cct_1045_4140", 8,
            "Model CCT diagrams of two steels with the same furnace path and 25 mm still-air path",
            cct_take)}
</section>
<section>
  {fig_full("harden", "f09_hardenability_catalogue", 9,
            f"All {N['di_grades']} catalogue grades as 25 mm bars; ideal critical diameter from the ASTM A255 "
            f"factors; (b) Spearman {N['di_rho']:.2f}",
            f"Below 40 mm, {N['di_lo'][0]} of {N['di_lo'][1]} bars are mostly bainite + martensite; above 100 mm, "
            f"{N['di_hi'][0]} of {N['di_hi'][1]}")}
</section>
<section>
  {fig_pair(
      ("family", "f10_steel_families", 10,
       f"Median of each steel family ({N['fam_grades']} catalogue grades); gain in MPa; right: 25 mm air-cooled bar",
       f"Cr-Mo, Ni-Cr-Mo: {100 * N['bm_crmo']:.0f}–{100 * N['bm_nicrmo']:.0f} % bainite + martensite"),
      ("barsize", "f11_bar_size", 11,
       f"Normalised bars, sizes never seen in training (Birbilis et al. 2025); effect overstated: "
       f"{N['bs_pred']:.0f} vs {N['bs_meas']:.0f} MPa",
       f"Mass effect: right sign for {N['bs_sign'] * N['bs_n']:.0f} of {N['bs_n']} grades"))}
</section>
<section>
  <h3 data-tb="subheading B">B. Hardening, tempering and stress relieving</h3>
  {fig_pair(
      ("jominy", "f12_jominy_4140H", 12,
       "Jominy end-quench hardness of 4140H against its SAE hardenability band",
       f"{N['jom_grades']} unseen H-steels: {N['jom_pub']:.1f} → {N['jom_cal']:.1f} HRC band error"),
      ("five", "f13_five_treatments_4140", 13,
       "Five treatments of a 25 mm 4140 bar; whiskers: 90 % interval",
       f"One steel, {N['ft_min']:.0f} to {N['ft_max']:.0f} MPa by treatment alone"))}
</section>
<section>
  {fig_pair(
      ("tempering", "f14_tempering_4140", 14,
       "Tempering a 25 mm 4140 bar (oil, 1 h); circles: measured; TME, TE: embrittlement ranges",
       f"Tracks the measured bars from {N['temp_meas_lo']:.0f} to {N['temp_meas_hi']:.0f} °C"),
      ("qtmass", "f15_qt_mass_effect", 15,
       "Oil-quenched and tempered 4140 bars, grade held out; circles: measured",
       "The mass effect follows from the quench heat flow"))}
</section>
<section>
  <h2 data-tb="heading conclusion">Conclusions</h2>
  {bullets([
      f"{k('Furnace vs air:')} low-C steels barely change (+{N['gain_low']:.0f} MPa); medium and high C gain "
      f"+{N['gain_med']:.0f} and +{N['gain_high']:.0f} MPa from finer pearlite",
      f"{k('Hardenable steels:')} Cr-Mo and Ni-Cr-Mo turn bainitic in air: bar size matters as much as route",
      f"{k('One chain, five treatments:')} unseen-grade UTS within {N['an_uts']:.0f} MPa (annealed, normalised) and "
      f"{N['qt_uts']:.0f} MPa (quenched + tempered); 90 % intervals hold {100 * N['an_cov']:.1f} % and "
      f"{100 * N['qt_cov']:.0f} %",
      f"{k('Recovered from data:')} tempering fades carbon's effect; {N['temper_slowest']} delays softening most",
      f"{k('Also built:')} precipitation strengthening, embrittlement and TMCP flags, route advice",
      f"{k('Limits:')} {N['di_ratio']:.1f}× too hardenable vs Grossmann's ideal diameter; kinetic error is structural "
      f"(MCMC); stress relief not tested (no open data)",
  ], "conclusion")}
</section>
<section class="fine">
  <h2 data-tb="heading references">References</h2>
  <div class="refs">
    {bullets(["M.V. Li et al., Metall. Mater. Trans. B 29, 661 (1998)",
              "M.A. Grossmann et al., Hardenability of Alloy Steels (1939)",
              "T. Gladman et al., J. Iron Steel Inst. 210, 916 (1972)",
              "Ph. Maynier et al., in Hardenability Concepts, AIME (1978)",
              "J.H. Hollomon, L.D. Jaffe, Trans. AIME 162, 223 (1945)",
              "D.M. Collins et al., Metals 13, 1168 (2023)"], "references 1", "ref")}
    {bullets(["N. Birbilis et al., Mendeley Data, doi:10.17632/jmwb9ddd43.1 (2025)",
              "J. Lei et al., J. Am. Stat. Assoc. 113, 1094 (2018)",
              "ASTM A255 (hardenability); ASM Handbook, Vol. 1 (1990)",
              "Metallus, Practical Data for Metallurgists (2024)"], "references 2", "ref")}
  </div>
</section>
"""
    return nbsp(left), nbsp(right)


CSS = """
@page { size: %(PW)smm %(PH)smm; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { width: %(PW)smm; height: %(PH)smm; position: relative; overflow: hidden; background: #fff; color: %(INK)s;
  font-family: "Calibri", "Carlito", sans-serif; font-size: %(S_BODY)spt; line-height: 1.16;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; font-kerning: normal; }
.hdr { position: absolute; left: 0; top: 0; width: 100%%; height: %(HH)smm;
  background: linear-gradient(to bottom, rgb(240,248,252) 0%%, rgb(116,196,233) 74%%, rgb(116,196,233) 83%%,
  rgb(162,216,240) 100%%); }
.logo { position: absolute; left: 14mm; top: 6mm; width: 48mm; height: 48mm; }
.hdr-text { position: absolute; left: %(M)smm; right: %(M)smm; top: 3.6mm; text-align: center;
  font-family: "Times New Roman", Times, serif; }
.hdr-text .t1 { font-size: %(S_TITLE)spt; font-weight: bold; line-height: 1.08; }
.hdr-text .t2 { font-size: %(S_SUB)spt; font-style: italic; color: %(NAVY)s; line-height: 1.18; margin-top: 0.6mm; }
.hdr-text .t3 { font-size: %(S_HDR)spt; line-height: 1.18; margin-top: 0.5mm; }
.hdr-text .t4 { font-size: %(S_AUTH)spt; line-height: 1.16; margin-top: 0.9mm; }
.border { position: absolute; left: 0.6mm; top: 0.6mm; right: 0.6mm; bottom: 0.6mm; border: 2.5pt solid #0E2841; }
.divider { position: absolute; left: calc(50%% - 0.75pt); top: %(TOP)smm; height: %(COLH)smm; width: 0;
  border-left: 1.5pt solid #156082; }
.ftr { position: absolute; left: 0; bottom: 0; width: 100%%; height: %(FH)smm; background: %(FOOT)s; display: flex;
  align-items: center; }
.ftr div { width: 50%%; text-align: center; font-size: %(S_BODY)spt; line-height: 1.2; }
.ftr .hi { font-family: "Nirmala UI", "Kohinoor Devanagari", "Devanagari Sangam MN", sans-serif; }
.ftr .en { font-weight: bold; }
.col { position: absolute; top: %(TOP)smm; height: %(COLH)smm; width: %(CW)smm; display: flex;
  flex-direction: column; justify-content: space-between; }
#colL { left: %(M)smm; }
#colR { left: %(XR)smm; }
section { margin: 0; }
h2 { font-family: "Trebuchet MS", sans-serif; font-weight: bold; font-size: %(S_HEAD)spt; color: %(BLUE)s;
  text-align: center; line-height: 1.02; margin: 0 0 2.2mm 0; }
h3 { font-weight: bold; font-size: %(S_BODY)spt; color: %(BLUE)s; line-height: 1.1; margin: 0 0 1.3mm 0; }
b.k { color: %(BLUE)s; }
ul.bl { list-style: none; margin: 0; padding: 0; }
ul.bl li { position: relative; padding-left: 5mm; margin: 0 0 1.1mm 0; }
ul.bl li:last-child { margin-bottom: 0; }
ul.bl li::before { content: "\\2022"; color: %(BLUE)s; position: absolute; left: 0.6mm; top: 0; }
ul.bl.tight li { margin-bottom: 1.4mm; }
p { margin: 0; }
.lead { margin-bottom: 2.4mm; }
.fig { margin: 0; }
.pic { display: block; }
.cap { font-size: %(S_CAP)spt; line-height: 1.14; text-align: center; margin-top: 1.0mm; }
.take { font-size: %(S_TAKE)spt; line-height: 1.14; text-align: center; font-weight: bold; color: %(BLUE)s;
  margin-top: 0.5mm; }
#colL section > .fig { margin-top: 2.2mm; }
.pair { display: grid; grid-template-columns: %(HALF)smm %(HALF)smm; column-gap: %(GAP)smm; }
.pair + .pair { margin-top: 4.2mm; }
.split { display: grid; grid-template-columns: 1fr %(HALF)smm; column-gap: 6mm; align-items: start; }
/* graphical abstract */
.ga { display: grid; grid-template-columns: 1fr 9mm 1fr 9mm 1fr; grid-template-rows: auto 6.4mm auto;
  padding: 2.5mm 0 0 2.5mm; margin-bottom: 3.2mm; }
.gab { position: relative; background: %(PALE)s; border: 1.5pt solid %(BLUE)s; border-radius: 3.4mm;
  padding: 3.6mm 2.6mm 2.6mm; text-align: center; }
.gab.out { background: #FDE9D9; }
.gab .num { position: absolute; left: -2.5mm; top: -2.5mm; width: 7.6mm; height: 7.6mm; border-radius: 50%%;
  background: %(BLUE)s; color: #fff; font-size: 15pt; font-weight: bold; line-height: 7.6mm; text-align: center; }
.gt { font-weight: bold; color: %(NAVY)s; font-size: 20pt; line-height: 1.1; }
.gs { font-size: 17pt; line-height: 1.12; margin-top: 0.4mm; }
.gar { display: flex; align-items: center; justify-content: center; }
.arr { width: 6.6mm; height: 6.6mm; fill: %(NAVY)s; }
.arr.dn { width: 5.6mm; height: 5.6mm; }
/* tiles */
.tiles { display: flex; gap: 3mm; }
.tile { flex: 1 1 0; background: %(PALE)s; border-radius: 2.2mm; padding: 1.7mm 1.2mm 1.8mm; text-align: center; }
.tn { font-size: 26pt; font-weight: bold; color: %(BLUE)s; line-height: 1.0; }
.tl { font-size: %(S_SMALL)spt; line-height: 1.1; margin-top: 0.7mm; }
/* equations */
.eqbox { border: 2pt solid %(BLUE)s; border-radius: 3.4mm; padding: 2.4mm 4mm 2.6mm; margin-top: 3.4mm; }
.eqgrid { display: grid; grid-auto-flow: column; grid-template-rows: auto auto; grid-auto-columns: max-content;
  column-gap: 6mm; row-gap: 1.6mm; justify-content: space-between; }
.eql { font-size: %(S_SMALL)spt; font-weight: bold; color: %(NAVY)s; line-height: 1.1; }
.eq { font-family: Cambria, "STIX Two Text", "Times New Roman", serif; font-size: 19pt; line-height: 1.34;
  white-space: nowrap; }
.eq sub, .eq sup { font-size: 68%%; line-height: 0; }
/* novelty table */
.nov { display: grid; grid-template-columns: 0.82fr 1fr; column-gap: 2mm; row-gap: 1.1mm; }
.nov > div { border-radius: 1.6mm; padding: 1.2mm 3mm 1.3mm; font-size: %(S_TAKE)spt; line-height: 1.14; }
.nh { font-weight: bold; }
.nh.nl { background: #D9D9D5; color: #333333; }
.nh.nr { background: %(BLUE)s; color: #FFFFFF; }
.nc.nl { background: #F1F1EE; color: #52514E; }
.nc.nr { background: %(PALE)s; }
/* small print */
.fine h2 { margin-bottom: 1.4mm; }
.refs { display: grid; grid-template-columns: 1fr 1fr; column-gap: 6mm; }
ul.bl.ref { font-size: %(S_SMALL)spt; line-height: 1.12; }
ul.bl.ref li { padding-left: 3.6mm; margin-bottom: 0.5mm; }
.ack { font-size: %(S_SMALL)spt; line-height: 1.14; }
.ack .par { display: block; }
.ack .par + .par { margin-top: 0.8mm; }
"""


def html(N):
    fonts = sans_fonts(CACHE)
    faces = ""
    if fonts["family"] != "Calibri":
        w = {"regular": ("normal", "normal"), "bold": ("bold", "normal"), "italic": ("normal", "italic"),
             "bolditalic": ("bold", "italic")}
        for face, path in fonts["files"].items():
            faces += (f'@font-face {{ font-family: "Carlito"; font-weight: {w[face][0]}; font-style: {w[face][1]}; '
                      f'src: url("{Path(path).resolve().as_uri()}"); }}\n')
    CACHE.mkdir(exist_ok=True)
    with zipfile.ZipFile(HERE / "Template.pptx") as z:      # the template's own vector logo
        (CACHE / "logo.svg").write_bytes(z.read("ppt/media/image1.svg"))
    css = CSS % dict(PW=PAGE_W, PH=PAGE_H, INK=INK, NAVY=NAVY, BLUE=BLUE, PALE=PALE, FOOT=FOOT, HH=HEADER_H,
                     FH=FOOTER_H, M=MARGIN, TOP=TOP, COLH=round(BOTTOM - TOP, 2), CW=round(COL_W, 2),
                     XR=round(MARGIN + COL_W + GUTTER, 2), HALF=HALF, GAP=round(GAP, 2),
                     S_TITLE=S_TITLE, S_SUB=S_SUB, S_HDR=S_HDR, S_AUTH=S_AUTH, S_HEAD=S_HEAD, S_BODY=S_BODY,
                     S_TAKE=S_TAKE, S_CAP=S_CAP, S_SMALL=S_SMALL)
    left, right = body(N)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>
<style>
{faces}{css}</style></head>
<body>
<div class="hdr" data-shape="header band"></div>
<img class="logo" data-logo="1" src=".cache/logo.svg" alt="IIT Roorkee">
<div class="hdr-text" data-tb="title block">
  <div class="t1 par">{TITLE}</div>
  <div class="t2 par">{SUBTITLE}</div>
  <div class="t3 par">{COURSE}</div>
  <div class="t4 par">{AUTHORS}</div>
</div>
<div class="col" id="colL">{left}</div>
<div class="col" id="colR">{right}</div>
<div class="divider" data-shape="column divider"></div>
<div class="ftr" data-shape="footer band">
  <div class="hi" data-tb="footer 0">श्रमं विना न किमपि साध्यम्!</div>
  <div class="en" data-tb="footer 1">Nothing can be achieved without hard work!</div>
</div>
<div class="border" data-shape="page border"></div>
</body></html>
"""


# ------------------------------------------------------------------ layout checks and export, in Chrome
CHECK_JS = r"""
() => {
  const mm = v => v * 25.4 / 96;
  const out = {cols: {}, problems: [], fonts: []};
  for (const id of ['colL', 'colR']) {
    const c = document.getElementById(id);
    const kids = [...c.children];
    const used = kids.reduce((s, k) => s + k.getBoundingClientRect().height, 0);
    out.cols[id] = {height: mm(c.clientHeight), used: mm(used), sections: kids.length,
                    gap: kids.length > 1 ? mm((c.clientHeight - used) / (kids.length - 1)) : 0};
    if (used > c.clientHeight + 0.5) out.problems.push(id + ' overflows by ' + mm(used - c.clientHeight).toFixed(1) + ' mm');
  }
  const page = document.body.getBoundingClientRect();
  for (const el of document.querySelectorAll('[data-tb], .pic, .tile, .gab, .eqbox')) {
    const r = el.getBoundingClientRect();
    const name = el.dataset.tb || el.dataset.fig || el.dataset.shape || el.className;
    if (r.right > page.right + 0.5 || r.bottom > page.bottom + 0.5) out.problems.push('off page: ' + name);
    if (el.scrollWidth > el.clientWidth + 1 && el.clientWidth > 0) out.problems.push('too wide: ' + name);
  }
  const hdr = document.querySelector('.hdr').getBoundingClientRect();
  const ht = document.querySelector('.hdr-text').getBoundingClientRect();
  out.header = {text_bottom: mm(ht.bottom), band: mm(hdr.bottom)};
  if (ht.bottom > hdr.bottom - 2) out.problems.push('header text runs out of the band');
  const logo = document.querySelector('.logo').getBoundingClientRect();
  for (const p of document.querySelectorAll('.hdr-text .par')) {
    const rg = document.createRange(); rg.selectNodeContents(p);
    const r = rg.getBoundingClientRect();
    if (r.left < logo.right + 4) out.problems.push('header line touches the logo: ' + p.className);
    if (rg.getClientRects().length > 1 && p.classList.contains('t1') === false && r.height > 1.6 * parseFloat(getComputedStyle(p).lineHeight))
      out.problems.push('header line wraps: ' + p.className);
  }
  out.fonts = [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family + ' ' + f.weight + ' ' + f.style);
  return out;
}
"""

DUMP_JS = r"""
() => {
  const mm = v => v * 25.4 / 96, pt = v => v * 72 / 96;
  const rect = el => { const r = el.getBoundingClientRect(); return {x: mm(r.left), y: mm(r.top), w: mm(r.width), h: mm(r.height)}; };
  const col = c => { const m = c.match(/[\d.]+/g).map(Number); return m.slice(0, 3).map(v => Math.round(v).toString(16).padStart(2, '0')).join('').toUpperCase(); };
  const runs = par => {
    const out = [];
    const w = document.createTreeWalker(par, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = w.nextNode())) {
      const txt = n.nodeValue.replace(/\s+/g, ' ');
      if (!txt) continue;
      const pe = n.parentElement, cs = getComputedStyle(pe);
      let base = 0;
      if (pe.closest('sub')) base = -25000; else if (pe.closest('sup')) base = 30000;
      let size = pt(parseFloat(cs.fontSize));
      if (base) size = pt(parseFloat(getComputedStyle(pe.closest('sub,sup').parentElement).fontSize));
      out.push({t: txt, size: Math.round(size * 100) / 100, bold: parseInt(cs.fontWeight) >= 600, italic: cs.fontStyle === 'italic',
                color: col(cs.color), font: cs.fontFamily.split(',')[0].replace(/"/g, '').trim(), base});
    }
    if (out.length) { out[0].t = out[0].t.replace(/^ /, ''); out[out.length - 1].t = out[out.length - 1].t.replace(/ $/, ''); }
    return out.filter(r => r.t.length);
  };
  const boxes = [];
  for (const el of document.querySelectorAll('[data-tb]')) {
    let pars = [...el.querySelectorAll(':scope > li, :scope .par')];
    if (!pars.length) pars = [el];
    const cs = getComputedStyle(el);
    boxes.push({name: el.dataset.tb, ...rect(el), valign: (el.classList.contains('num') || el.closest('.ftr')) ? 'middle' : 'top',
      pars: pars.map(p => { const ps = getComputedStyle(p); const r = p.getBoundingClientRect();
        return {align: ps.textAlign, line: pt(parseFloat(ps.lineHeight)), after: pt(parseFloat(ps.marginBottom) || 0),
                before: pt(parseFloat(ps.marginTop) || 0), bullet: p.tagName === 'LI', indent: mm(parseFloat(ps.paddingLeft) || 0),
                y: mm(r.top), h: mm(r.height), runs: runs(p)}; })});
  }
  const shapes = [];
  for (const el of document.querySelectorAll('[data-shape]')) {
    const cs = getComputedStyle(el);
    shapes.push({name: el.dataset.shape, ...rect(el), fill: cs.backgroundColor.startsWith('rgba(0, 0, 0, 0') ? null : col(cs.backgroundColor),
      line: parseFloat(cs.borderTopWidth) > 0 ? col(cs.borderTopColor) : null, line_pt: pt(parseFloat(cs.borderTopWidth) || 0),
      radius: mm(parseFloat(cs.borderTopLeftRadius) || 0), round: cs.borderTopLeftRadius.includes('%')});
  }
  const pics = [...document.querySelectorAll('.pic')].map(el => ({name: el.dataset.fig, ...rect(el)}));
  const arrows = [...document.querySelectorAll('.gar svg')].map(el => ({dir: el.parentElement.dataset.arrow, ...rect(el)}));
  const logo = rect(document.querySelector('.logo'));
  return {boxes, shapes, pics, arrows, logo};
}
"""


def render(html_path, pdf_path, png_path, layout_path, preview_width=2400):
    from playwright.sync_api import sync_playwright
    px = lambda v: int(round(v * 96 / 25.4))  # noqa: E731
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception:
            browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": px(PAGE_W), "height": px(PAGE_H)},
                                device_scale_factor=preview_width / px(PAGE_W))
        page.emulate_media(media="print")
        page.goto(html_path.resolve().as_uri())
        page.evaluate("document.fonts.ready")
        page.wait_for_load_state("networkidle")
        check = page.evaluate(CHECK_JS)
        layout_path.write_text(json.dumps(page.evaluate(DUMP_JS), indent=1, ensure_ascii=False), encoding="utf-8")
        page.screenshot(path=str(png_path), full_page=False)
        page.pdf(path=str(pdf_path), width=f"{PAGE_W}mm", height=f"{PAGE_H}mm", print_background=True,
                 prefer_css_page_size=True)
        browser.close()
    return check


def main():
    N = numbers()
    html_path = HERE / "poster.html"
    html_path.write_text(html(N), encoding="utf-8")
    (HERE / "preview").mkdir(exist_ok=True)
    check = render(html_path, HERE / f"{OUT_NAME}.pdf", HERE / "preview" / "poster_preview.png",
                   CACHE / "layout.json")
    for cid, c in check["cols"].items():
        print(f"{cid}: content {c['used']:.1f} of {c['height']:.1f} mm, spare {c['height'] - c['used']:.1f} mm "
              f"({c['gap']:.1f} mm between {c['sections']} sections)")
    print(f"header text ends at {check['header']['text_bottom']:.1f} mm, band at {check['header']['band']:.1f} mm")
    print("fonts loaded:", ", ".join(sorted(set(check["fonts"]))) or "(system fonts only)")
    lay = json.loads((CACHE / "layout.json").read_text(encoding="utf-8"))
    for bx in lay["boxes"]:
        if bx["name"] in ("intro", "methods", "conclusion") or bx["name"].startswith(("caption", "takeaway")):
            lines = [round(pp["h"] / (pp["line"] * 25.4 / 72)) for pp in bx["pars"]]
            print(f"  lines {bx['name']:14s} {lines}")
    print("layout problems:", check["problems"] or "none")
    return check


if __name__ == "__main__":
    main()
