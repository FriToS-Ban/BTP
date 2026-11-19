"""Build and execute the presentation notebooks from the pipeline results.

    python notebooks/make_notebooks.py

The notebooks only *read* results/ and figures/; every model is fitted by
the pipeline scripts, so the notebooks run in seconds and always show the
numbers of the latest pipeline run.
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

SETUP = """import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd, numpy as np
from IPython.display import Image, display, Markdown
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.3g}")
R = ROOT / "results"; FIG = ROOT / "figures"
def fig(name, width=1000):
    display(Image(filename=str(FIG / f"{name}.png"), width=width))"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


NOTEBOOKS = {
    "01_data_overview": [
        md("""# 1. Data: what is known about furnace-cooled and air-cooled steels

The project predicts what happens to a steel when it is **furnace cooled** (full annealing) or **air cooled**
(normalising): which microstructure forms and which tensile properties result. This notebook shows the open
data the study rests on. The training data are typical properties of 33 AISI grades measured in the
annealed, normalized and hot-rolled condition (ASM Handbook, data of Bethlehem Steel); measured data from
dilatometry, Jominy hardenability bands, an open CCT atlas, structural members and 52 open-access papers are
kept aside for validation."""),
        code(SETUP),
        code("""pd.read_csv(R / "s01_dataset_summary.csv")"""),
        md("""## Properties against carbon content, furnace vs air cooling
Plain-carbon steels (filled markers) show the textbook behaviour: strength and hardness rise with carbon,
air cooling adds strength and removes ductility, and the gap widens towards the eutectoid composition.
Alloy steels (hollow markers) respond much more strongly to air cooling because their hardenability lets
bainite form."""),
        code("""fig("fig01_handbook_properties_vs_carbon")"""),
        code("""pairs = pd.read_csv(R / "s01_route_pairs.csv")
pairs[[c for c in pairs.columns if c.endswith("_delta")]].describe().round(1)"""),
        code("""fig("fig02_route_effect_by_grade", 950)"""),
        md("""## Practice: at which temperatures were these steels treated?
The steel corner of the Fe-C diagram with the temperatures at which the 33 handbook grades were actually
annealed and normalised, and the defaults the tool uses (fitted to that practice). Normalising is done about
60 °C hotter than annealing, so the two routes differ in temperature as well as in cooling rate."""),
        code("""fig("fig26_heat_treatment_practice", 850)"""),
        md("""**Reading the data.** Air cooling never lowered the tensile strength of a handbook grade (all 33 deltas
are positive); elongation fell for most grades. The largest effects are in eutectoid/high-carbon steels
(finer pearlite) and in Cr-Mo / Ni-Cr-Mo steels (bainite formation)."""),
    ],
    "02_steel_categories": [
        md("""# 2. Differentiating steels by category

Before asking what furnace or air cooling does, the steel must be placed in the right category: a plain
low-carbon steel, a eutectoid steel and a Ni-Cr-Mo steel respond to the same cooling in completely different
ways, and stainless or tool steels are outside the ferrite-pearlite scope altogether.

A classifier is trained on heats simulated inside the published composition ranges of 547 grades and
evaluated on **unseen grades** (grade-grouped cross-validation) and on **real measured heats**."""),
        code(SETUP),
        code("""pd.read_csv(R / "s02_classifier_cv.csv")"""),
        md("""The production classifier is chosen by a rule fixed in advance: the best family accuracy on the real
SteelBench heats (simulated-heat CV is reported alongside). The last row is a fair baseline: the AISI/SAE
composition limits written as a decision list, with nothing fitted. The ML model beats it on unseen grades
and on real heats, and only the ML model recognises compositions that belong to no category. Where the two
agree, the predicted family is almost always right; where they disagree (borderline compositions) it is
much less reliable, so the app shows both."""),
        code("""pd.read_csv(R / "s02_rules_agreement.csv")"""),
        code("""pd.read_csv(R / "s02_external_tests.csv")"""),
        code("""pd.read_csv(R / "s02_per_category_recall.csv")"""),
        code("""fig("fig03_category_confusion", 800)"""),
        md("""## Which elements decide a category?
SHAP attributions agree with metallurgy: carbon separates low/medium/high-carbon steels, sulphur marks
free-machining steels, chromium the Cr and stainless families, nickel the Ni-bearing and austenitic grades,
silicon the Si-Mn spring steels."""),
        code("""fig("fig04_category_shap", 900)"""),
        md("""## Unsupervised view and open-set check
An unsupervised clustering of the catalogue recovers the families only partly (ARI below), which is why a
supervised model is needed. The novelty score rejects compositions unlike any known grade: all maraging
steels of the project's earlier dataset are flagged as *unknown*, so the cooling-route models refuse them."""),
        code("""pd.read_csv(R / "s02_unsupervised_agreement.csv")"""),
        code("""fig("fig05_catalog_embedding", 750); fig("fig06_category_novelty", 800)"""),
    ],
    "03_microstructure": [
        md("""# 3. Microstructure after furnace cooling and air cooling

The transformation model is the semi-empirical Kirkaldy-Venugopalan / Li et al. (1998) kinetics, written
from the equations and integrated with Scheil's additivity rule, with the thermodynamic start conditions of
Collins et al. (2023). Cooling paths come from a heat balance: a furnace cool, or a bar of given diameter
cooling in still air (convection + radiation, temperature-dependent heat capacity)."""),
        code(SETUP),
        md("""## Validation against measured dilatometry
Start temperatures of each constituent at nine cooling rates for EN3B, EN8 and SA-540 (Collins et al.).
The second block compares this implementation with the authors' own published Li-model numbers."""),
        code("""pd.read_csv(R / "s03_kinetics_summary.csv")"""),
        code("""fig("fig07_kinetics_validation")"""),
        md("""## Calibrating the cooling conditions of the handbook data
The handbook does not state how fast its bars cooled. Using physics only (no ML), the hardness error of the
normalized rows is large for small bars and flat between about 50 and 80 mm bars in still air; 50 mm (about
0.5 K/s at 700 C) is taken at the start of that flat region. For the annealed rows the error hardly changes
below 0.01 K/s, and 0.005 K/s (18 K/h), the usual full-annealing furnace cool, is taken."""),
        code("""fig("fig08_cooling_calibration", 900)"""),
        md("""## CCT diagrams with the two cooling paths
The furnace path passes through the ferrite and pearlite fields of every steel shown. The air path does so
for plain-carbon steels but crosses the bainite (and martensite) fields of the hardenable Cr-Mo and
Ni-Cr-Mo steels - the central difference between the categories."""),
        code("""fig("fig09_cct_with_cooling_paths")"""),
        code("""fig("fig10_microstructure_vs_cooling_rate")"""),
        md("""## The two routes as cooling curves
Furnace cooling at 0.005 K/s, still-air cooling of bars of 10 to 150 mm, and three positions of a Jominy
bar. A still-air 25 mm bar has a mean 800-500 °C rate of about 0.9 K/s, the value measured for a 25 mm bar in
a patent study (0.89 K/s)."""),
        code("""pd.read_csv(R / "s03_cooling_paths.csv")"""),
        code("""fig("fig27_cooling_paths", 850)"""),
        md("""## 423 measured CCT points of 19 low-alloy steels
Phase fractions and hardness after continuous cooling, from the Rose and Hougardy atlas as tabulated in two
open (CC BY) papers. The model is good for quench-like cooling and weaker in the slow-to-intermediate range
where bainite competes with ferrite and pearlite."""),
        code("""pd.read_csv(R / "s03_cct_atlas_summary.csv")"""),
        code("""fig("fig28_cct_atlas_validation")"""),
        md("""## Measured microstructures from the literature
The textbooks say that normalised steel holds more pearlite than annealed steel (the quasi-eutectoid
effect). The Li model shows this, but far too strongly for normalised medium-carbon bars, whose measured
fractions sit at or below the lever-rule value. We therefore report the lever split of the ferrite +
pearlite as the central value and the kinetic split as the top of a band, at every cooling rate. Starting
the pearlite incubation only inside the Hultgren region does not help (ablation below), and a faster ferrite
would fix the fractions but spoil the measured ferrite-start temperatures."""),
        code("""pd.read_csv(R / "s03_pearlite_coupling_ablation.csv")"""),
        code("""pd.read_csv(R / "s03_grain_size_check.csv")"""),
        code("""pd.read_csv(R / "s03_literature_pearlite_summary.csv")"""),
        code("""pd.read_csv(R / "s03_ferrite_factor_scan.csv")"""),
        code("""pd.read_csv(R / "s03_spacing_calibration.csv")"""),
        code("""fig("fig11_microstructure_literature_validation", 900)"""),
    ],
    "04_property_models": [
        md("""# 4. From microstructure to properties: physics-anchored machine learning

Training data: 80 rows (33 grades x annealed / normalized / hot rolled). Every model is evaluated by
**leave-one-grade-out** cross-validation, so a grade is never seen in training when it is predicted. The
ladder runs from textbook equations, through composition-only ML, to the physics-anchored models."""),
        code(SETUP),
        code("""b = pd.read_csv(R / "s04_model_benchmark.csv")
b.pivot(index="model", columns="target", values="RMSE_inAD").round(1)"""),
        code("""b.pivot(index="model", columns="target", values="R2_inAD").round(3)"""),
        md("""Accuracy on the **change** from annealed to normalized - the what-if this project is about:"""),
        code("""b.pivot(index="model", columns="target", values="delta_RMSE").round(1)"""),
        code("""fig("fig12_model_ladder")"""),
        md("""## Are the differences real? Paired tests against the production model
Per-grade errors, Wilcoxon signed-rank test (with the Holm adjustment for the eleven comparisons per property)
and a bootstrap 95 % interval of the RMSE difference (positive = the other model is worse)."""),
        code("""pd.read_csv(R / "s04_paired_tests.csv").round(3)"""),
        md("""## Uncertainty: normalised conformal intervals
The ensemble's own error bars are only roughly calibrated (a Gaussian 90 % interval would use 1.645 standard
deviations; the factors below range from about 1.5 to 2.1). Normalised split-conformal calibration rescales them
so that the intervals cover 90 % of unseen grades, and that coverage is checked here."""),
        code("""pd.read_csv(R / "s04_conformal.csv")"""),
        code("""fig("fig13_parity_production")"""),
        md("""## The decisive test: cooling rates the training data never contain
Models trained only on annealed/normalised data predict the measured hardness of three steels cooled at
0.1-50 K/s. ML without physics cannot invent the hardness of bainite or martensite; the physics-anchored
production model inherits it from the transformation model."""),
        code("""pd.read_csv(R / "s04_cooling_rate_extrapolation.csv").round(1)"""),
        code("""fig("fig14_cooling_rate_extrapolation")"""),
        md("""## Physical consistency and interpretability
Share of cooling-rate steps where a property moves the wrong way, and the span of the predicted UTS / HB
over a normal 0.10-0.40 % Si (physically a few tens of MPa)."""),
        code("""c = pd.read_csv(R / "s04_consistency.csv")
c.pivot(index="model", columns="target", values="rate_violation_frac").round(3)"""),
        code("""c.pivot(index="model", columns="target", values="si_0.10_0.40_span").round(1)"""),
        md("""## Two lessons, kept as ablations
**Silicon.** Every training grade has the same nominal Si (0.225 % carbon grades, 0.25 % alloy H-grades)
except 9255. With Si as an input, cross-validation looks better because Si labels the grade type, but the
response to a real change of Si is unphysical. Si and V are therefore left to the physics."""),
        code("""pd.read_csv(R / "s04_silicon_ablation.csv").round(1)"""),
        md("""**The gate.** Handing over to the physics-anchored members for every unfamiliar composition, or above
the fastest training cooling rate, was tried first; this table re-scores those rules on the same evidence."""),
        code("""pd.read_csv(R / "s04_gate_ablation.csv").set_index("rule").T.round(3)"""),
        md("""**The physics anchor.** The anchored models first scaled every property on the Maynier hardness, whose
ferrite-pearlite term depends on composition and cooling rate only and cannot see pearlite refinement. They
now scale on the Gladman law (ferrite fraction, grain size, spacing) for ferrite + pearlite and on Maynier
for bainite and martensite. Inside the data this costs a few MPa; at new cooling rates and for new
compositions it is clearly better."""),
        code("""pd.read_csv(R / "s04_anchor_ablation.csv").set_index("anchor").T.round(3)"""),
        code("""fig("fig15_greybox_coefficients", 800)"""),
    ],
    "05_furnace_vs_air": [
        md("""# 5. Furnace cooling vs air cooling, category by category

Every catalogue grade that forms ferrite + pearlite on slow cooling is taken through both routes. The table
gives, per category, the microstructure that forms and the median change in properties on air cooling
(25 mm bar)."""),
        code(SETUP),
        code("""s = pd.read_csv(R / "s05_category_summary.csv")
s[["category", "grades", "FC_regime_mode", "AC25_regime_mode", "FC_pearlite_mean", "AC25_pearlite_mean",
   "AC25_bainite_martensite_mean", "UTS_MPa_FC_median", "UTS_MPa_AC25_median", "UTS_MPa_delta_median",
   "YS_MPa_delta_median", "El_pct_delta_median", "uniformEl_FC_median", "uniformEl_AC25_median"]]"""),
        code("""fig("fig19_category_microstructures"); fig("fig18_category_route_effects")"""),
        md("""## Why air cooling is stronger: faster cooling or hotter austenitisation?
Normalising is done about 60 °C hotter than annealing. Air cooling a bar from the annealing temperature
separates the two: almost all of the gain comes from the faster cooling. The impact transition temperature
(Pickering, a toughness indicator) falls on air cooling because the ferrite grains are finer."""),
        code("""s[["category", "UTS_MPa_delta_median", "UTS_MPa_delta_cooling_median", "UTS_MPa_delta_temperature_median",
   "FC_d_alpha_um_median", "AC25_d_alpha_um_median", "ITT_FC_median_C", "ITT_AC25_median_C", "DI_mm_median"]]"""),
        md("""## How much of the error is the unknown heat composition?
The training data give one value per grade at the nominal composition. Varying each grade's composition
inside its own specification range moves the prediction by about this much, a floor under the error of any
model of nominal compositions."""),
        code("""pd.read_csv(R / "s05_composition_noise_floor.csv")"""),
        code("""fig("fig09b_cct_1045_vs_4140", 900)"""),
        md("""## Plain-carbon steels across the carbon range
The classic fine-pearlite versus coarse-pearlite chart, regenerated with 90 % intervals. Near the eutectoid
the furnace-cooled curve flattens because the handbook's annealed 1080 (615 MPa, 24.7 % elongation) is softer
and more ductile than its annealed 1060; that row looks partly spheroidised, and we flag it as suspect rather
than explain it."""),
        code("""fig("fig16_carbon_sweep_fc_vs_ac")"""),
        md("""## Section size: air cooling a thick bar approaches furnace cooling"""),
        code("""fig("fig17_section_size_effect")"""),
        md("""## Try it: one steel, both routes"""),
        code("""from steelml.predictor import SteelPredictor
sp = SteelPredictor.load()
out = sp.what_if({"C": 0.45, "Mn": 0.75, "Si": 0.25})
cols = ["route", "F_best", "P_best", "B_best", "M_best", "S0_um", "YS_MPa_pred", "YS_MPa_lo90", "YS_MPa_hi90",
        "UTS_MPa_pred", "UTS_MPa_lo90", "UTS_MPa_hi90", "El_pct_pred", "uniform_El_est_pct", "HB_pred", "in_domain"]
out[cols]"""),
    ],
    "06_external_validation": [
        md("""# 6. External validation

Data the models never saw: measured hot-rolled structural members, a second compilation of the same
grades, specification-level values of normalised EN/ASTM grades, and measured values from 52 open-access
papers (rows flagged as suspect during transcription are left out)."""),
        code(SETUP),
        code("""pd.read_csv(R / "s06_structural_metrics.csv").round(2)"""),
        md("""Strength of modern as-rolled sections is under-predicted by ~50-70 MPa: they are finer-grained (and in
part microalloyed) compared with the handbook's bars. Stage 7 shows that a handful of measured members
removes this bias."""),
        code("""fig("fig20_external_structural")"""),
        code("""pd.read_csv(R / "s06_other_checks.csv").round(3)"""),
        code("""pd.read_csv(R / "s06_literature_metrics.csv").round(2)"""),
        code("""fig("fig21_literature_validation")"""),
        md("""## The same steel after both routes
Papers that tested one heat furnace cooled and air cooled. The direction of the change is right; for the
alloy steels the predicted hardening is larger than measured, the same over-hardenability that the Jominy
bands show (notebook 8)."""),
        code("""pd.read_csv(R / "s06_literature_paired_routes.csv").round(1)"""),
        md("""## A dilatometer steel at stated cooling rates (23MnNiCrMo5-3)"""),
        code("""pd.read_csv(R / "s06_continuous_cooling_hardness.csv").round(1)"""),
        md("""## Micrographs of the same steels, annealed and normalised
Published micrographs (CC BY 4.0) of SAE 1045 and SAE 4140 after furnace cooling and after still-air cooling,
with the model's prediction for that composition, temperature and bar diameter underneath. The normalised
4140 has the acicular structure of the bainite the model predicts (the authors describe Widmanstätten
ferrite and fine pearlite; calling it bainite is our reading)."""),
        code("""fig("fig30_micrographs_vs_model", 950)"""),
    ],
    "07_design_and_active_learning": [
        md("""# 7. Using the models proactively

* **Recommend** a standard grade and a cooling route for a property specification, using the *lower*
  90 % bound so the recommendation is robust.
* **Map the strength-ductility trade-off** that standard grades can deliver, furnace vs air cooling (only real
  grades are searched; no composition is invented).
* **Plan experiments**: show that adding the most uncertain grade first learns faster than random, that a
  few local measurements recalibrate the model to a new product, and produce a pre-registered prediction
  sheet for the laboratory."""),
        code(SETUP),
        code("""r = pd.read_csv(R / "s07_recommendations.csv")
r[["spec", "grade", "option", "UTS_MPa_lo90", "YS_MPa_lo90", "El_pct_lo90", "CE_IIW"]]"""),
        code("""fig("fig22_grade_front_fc_vs_ac", 800)"""),
        code("""al = pd.read_csv(R / "s07_active_learning.csv")
al.groupby(["n_grades", "strategy"]).RMSE.mean().unstack().round(1)"""),
        code("""fig("fig23_active_learning", 750)"""),
        code("""pd.read_csv(R / "s07_few_shot.csv").groupby(["target", "k_added"])[["RMSE", "bias"]].mean().round(1)"""),
        code("""fig("fig24_few_shot_recalibration", 900)"""),
        md("""## Pre-registered laboratory plan
Predictions are fixed *before* the experiments, so the lab results are a genuine test. Steels commonly
stocked in Indian laboratories are listed with their AISI equivalents, ranked by expected information
(width of the prediction intervals)."""),
        code("""pd.read_csv(R / "s07_lab_plan_preregistered.csv")"""),
    ],
    "08_hardenability": [
        md("""# 8. Hardenability: why some steels change so much on air cooling

Whether air cooling gives ferrite + pearlite or bainite and martensite is a hardenability question. This
notebook shows (1) Grossmann's ideal critical diameter D_I (ASTM A255 factors) against the model's air-cooled
microstructures, (2) the model against the SAE Jominy hardenability bands of the H-steels, and (3) the
Bayesian calibration of the bainite reaction on those bands, with the test of whether it transfers to data
it never saw."""),
        code(SETUP),
        code("""pd.read_csv(R / "s08_ideal_diameter_summary.csv")"""),
        code("""fig("fig25_hardenability_vs_air_cooling", 950)"""),
        md("""## Jominy end-quench against the SAE H-bands
The Jominy bar is solved by 1-D conduction; its water-jet coefficient is fitted to the cooling-rate scale of
the ASM combined hardenability charts. The published Li constants are right at the quenched end and too
hardenable from J5 on."""),
        code("""pd.read_csv(R / "s08_jominy_cooling_rates.csv")"""),
        code("""m = pd.read_csv(R / "s08_jominy_metrics.csv"); m[m.grades == "non-boron grades"]"""),
        code("""fig("fig29_jominy_hbands", 900)"""),
        md("""## Calibration, cross-validation and transfer
The calibration changes only the bainite reaction (it starts lower and proceeds faster below that); the
H-bands do not constrain the ferrite and pearlite rates. Grade-grouped cross-validation, then the transfer
table: the calibrated kinetics against every independent dataset."""),
        code("""pd.read_csv(R / "s08_kinetic_calibration_posterior.csv")"""),
        code("""pd.read_csv(R / "s08_kinetic_calibration_cv_metrics.csv")"""),
        code("""pd.read_csv(R / "s08_kinetic_calibration_transfer.csv").set_index("kinetics").T"""),
    ],
    "09_hardening_tempering": [
        md("""# 9. The rest of the route: hardening, tempering and stress relieving

Versions 1-2 modelled furnace cooling and air cooling. This notebook adds the other three classic treatments:
**hardening** (a quench in oil, water or brine), **tempering** and **stress relieving**. The data are 324
quenched-and-tempered bars of 28 AISI grades from the open Birbilis et al. (2025) database, each with the quench
medium, the austenitising and tempering temperatures and the bar size (13, 25, 50 or 100 mm) written out."""),
        code(SETUP),
        code("""pd.read_csv(R / "s09_qt_dataset_summary.csv").pivot_table(index="section_mm", columns="quench_medium", values="bars")"""),
        md("""## Hardening: the quench heat flow
Radial conduction in the bar with Grossmann's severity of quench; checked against the exact Bessel solution
(tests/test_quench.py). Right: the model's ideal critical diameter against Grossmann's D_I from the ASTM A255
factors, a check that uses none of our data."""),
        code("""pd.read_csv(R / "s09_quench_rates.csv").pivot_table(index="D_mm", columns=["medium", "r_over_R"], values="rate700_K_s").round(1)"""),
        code("""fig("fig31_quench_and_hardenability", 1000)"""),
        code("""h = pd.read_csv(R / "s09_hardenability_check.csv")
h.assign(ratio_recalibrated=h.DI_model_calibrated_mm / h.DI_A255_mm, ratio_published=h.DI_model_published_mm / h.DI_A255_mm).round(2)"""),
        md("""## Tempering: one master curve
P = T(20 + log t). The constants were fitted on the bars, with a grade bootstrap."""),
        code("""pd.read_csv(R / "s09_tempering_constants.csv")"""),
        code("""fig("fig35_tempering_curves", 1000)"""),
        md("""## The model ladder (leave one grade out) and the mass effect"""),
        code("""b = pd.read_csv(R / "s09_qt_benchmark.csv"); b.pivot_table(index="model", columns="target", values="RMSE").round(1)"""),
        code("""fig("fig32_qt_model_ladder", 1000); fig("fig34_qt_mass_effect", 1000)"""),
        code("""pd.read_csv(R / "s09_qt_kinetics_check.csv").round(1)"""),
        md("""## The five classic treatments, side by side
Annealing, normalising, hardening, hardening + tempering and stress relieving of six grades as 25 mm bars."""),
        code("""ft = pd.read_csv(R / "s09_five_treatments.csv")
ft[["grade", "treatment", "medium", "microstructure", "UTS_MPa_pred", "UTS_MPa_lo90", "UTS_MPa_hi90", "El_pct_pred", "HV"]].round(0)"""),
        code("""fig("fig36_five_heat_treatments", 1000)"""),
        md("""## A live example: an EN19-type (42CrMo4) bar of 30 mm
The same call works for any composition; the tool flags microalloyed, TMCP-type and boron steels."""),
        code("""from steelml.predictor import SteelPredictor
sp = SteelPredictor.load()
comp = {"C": 0.42, "Mn": 0.75, "Si": 0.25, "Cr": 1.05, "Mo": 0.22, "Ni": 0.10}
out = sp.compare_treatments(comp, section_mm=30, temper_T=600)
out[["treatment", "medium", "microstructure", "UTS_MPa_pred", "UTS_MPa_lo90", "UTS_MPa_hi90", "YS_MPa_pred", "El_pct_pred", "HV", "notes"]].round(0)"""),
        md("""## Stress relieving and precipitation"""),
        code("""pd.read_csv(R / "s09_stress_relief_after_temper.csv").round(0)"""),
        code("""fig("fig37_stress_relief_and_precipitation", 1000)"""),
    ],
    "10_assumptions_sensitivity": [
        md("""# 10. Every assumption, and how much it matters

A model is only as credible as its assumptions. This notebook lists every condition the models rely on, shows
how one inferred condition (the bar size of the handbook's normalised data) turned out to be documented in a
second source, and measures how much each remaining assumption matters by retraining the models with it moved
to the ends of its range."""),
        code(SETUP),
        code("""pd.read_csv(R / "s10_assumptions.csv")"""),
        md("""## The handbook's normalised data are 25 mm bars
Version 2 inferred 50 mm from the physics alone. The Birbilis database has the same values with the bar size."""),
        code("""pd.read_csv(R / "s10_handbook_section_match.csv").round(1)"""),
        md("""## One assumption at a time (retrained, leave one grade out)"""),
        code("""pd.read_csv(R / "s10_oat_anneal_normalise.csv").round(1)"""),
        code("""pd.read_csv(R / "s10_oat_quench_temper.csv").round(1)"""),
        code("""fig("fig38_assumption_sensitivity", 1100)"""),
        md("""## Global sensitivity (Morris elementary effects)"""),
        code("""pd.read_csv(R / "s10_morris.csv").pivot_table(index="factor", columns="output", values="mu_star").round(2)"""),
        code("""fig("fig39_morris_sensitivity", 1100)"""),
        md("""## A test across bar sizes
Normalised bars of 13-100 mm and grades never seen in training."""),
        code("""pd.read_csv(R / "s06_bar_sizes_metrics.csv").round(2)"""),
        code("""fig("fig42_normalising_bar_size", 1100)"""),
    ],
    "11_kinetic_uncertainty": [
        md("""# 11. How uncertain is the transformation model?

The kinetic constants were calibrated on the SAE hardenability bands. Here their posterior is sampled with
four adaptive Metropolis chains (with convergence checks), the draws are pushed through to predictions, and
we ask whether the parameter uncertainty explains the errors on 423 independent CCT-atlas measurements."""),
        code(SETUP),
        code("""pd.read_csv(R / "s11_mcmc_diagnostics.csv")"""),
        code("""pd.read_csv(R / "s11_posterior_summary.csv").round(3)"""),
        code("""pd.read_csv(R / "s11_posterior_correlation.csv", index_col=0).round(2)"""),
        code("""fig("fig40_kinetic_posterior", 1100)"""),
        md("""## Propagation and independent coverage
If the parameter uncertainty explained the errors, about 90 % of the measured points would fall inside the
90 % band. They do not: the error is structural (the form of the equations)."""),
        code("""pd.read_csv(R / "s11_cct_atlas_coverage.csv").round(3)"""),
        code("""pd.read_csv(R / "s11_representative_bands.csv").round(3)"""),
        code("""d = pd.read_csv(R / "s11_dilatometry_bands.csv"); print("inside the 90 % band:", round(d.inside.mean(), 2)); d.round(0)"""),
        code("""fig("fig41_kinetic_uncertainty_propagation", 1100)"""),
    ],
}


def main():
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
        client.execute()
        path = HERE / f"{name}.ipynb"
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
