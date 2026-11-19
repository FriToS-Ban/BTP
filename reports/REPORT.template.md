# Furnace cooling vs air cooling of carbon and low-alloy steels, and the rest of the heat-treatment route: microstructure and tensile properties predicted with physics-informed machine learning

**B.Tech Project (MTP-400A), Department of Metallurgical and Materials Engineering, IIT Roorkee**
**Authors:** Aviral Singh, Shibasish Banerjee  |  **Supervisor:** Dr. Anish Karmakar

> This report is generated from `reports/REPORT.template.md` by `reports/build_report.py`, which fills every
> number from the files in `results/`, so the text always matches the last pipeline run.

---

## Abstract

We built a model that takes the composition of a carbon or low-alloy steel and tells what kind of steel it is,
what microstructure forms when it is furnace cooled (full annealing) or cooled in still air as a bar of a given
size (normalising), and what yield strength, tensile strength, uniform and total elongation, hardness and
impact transition temperature to expect, each with a 90 % interval. Version 3 adds the rest of the classic
route: hardening (oil, water or brine quench of a round bar), tempering and stress relieving, with
precipitation strengthening, embrittlement warnings and a check that refuses thermomechanically rolled
chemistries. The model follows the metallurgy step by step: empirical thermodynamics, a heat balance for the
cooling path (lumped for air, radial conduction for a quench), the Kirkaldy-Venugopalan / Li et al.
transformation kinetics, structure-property laws, a Hollomon-Jaffe tempering model, and Gaussian-process models
that learn what the textbook laws miss. We tested every link against measured data that the models never saw:
dilatometry of 3 steels, 423 points of an open CCT atlas, the Jominy bands of «jm_grades» SAE H-steels, «qt_n»
quenched-and-tempered bars of «qt_grades» grades at four bar sizes, «bar_n» normalised and annealed bars, «lit_rows»
values from «lit_src» open papers, published micrographs and 165 structural members. On unseen handbook grades the
model predicts the UTS of annealed and normalised steel within «uts_rmse» MPa (textbook relations alone: «uts_phys»), and the
UTS of quenched-and-tempered steel within «qt_uts_ens» MPa (textbook physics alone: «qt_uts_phys»). Making every assumption
explicit paid off at once. The bar size of the handbook's normalised data, which we had inferred as 50 mm,
is documented in a second source as 25 mm, and we corrected it. A Markov-chain Monte Carlo posterior of
the kinetic constants then shows that the transformation model's errors on independent data are structural
rather than parametric. The answer to our question is clear: for low-carbon steels furnace and air cooling
give almost the same steel; for medium- and high-carbon steels air cooling gives the same phases with much
finer pearlite; and for hardenable Cr-Mo and Ni-Cr-Mo steels it gives a different, bainitic steel, so for them
the bar size matters as much as the route. Hardening and tempering then give each grade a whole range of
strengths: «qt_range_4140» MPa for a 25 mm 4140 bar tempered at 200-700 °C.

## 1. Introduction

### 1.1 Why we rebuilt the project

Our first version of this project predicted the yield strength of 312 maraging-type steels from their
composition alone. It had careful statistics, but our supervisor told us, quite rightly, that it did not make
physical sense. A steel's strength depends on what was done to it, and that dataset had no processing
information at all, so two steels with the same chemistry and a different heat treatment looked identical to
the model, and an "inverse design" of compositions on such a model has no metallurgical meaning.

So we started again from the way a metallurgist actually thinks about a steel, and put the heat treatment at the
centre. The old work is kept in git history under the tag `legacy-composition-only-v1`.

### 1.2 What version 3 adds, and why

A fellow student reviewed version 2 and ranked what was missing: the cooling conditions we had inferred, the
uncertainty of the recalibrated kinetics, hardening, tempering, precipitation strengthening, toughness and
modern thermomechanically rolled steels. We followed that list, and also asked whether our data were simply too
small. Three things came out of it:

* **More and better data.** An open CC BY database (Birbilis et al. 2025) gives quenched-and-tempered,
  normalised and annealed bars of the same AISI grades with the quench medium, temperatures and bar size written
  out (Section 3.1). It is the data that hardening and tempering need, and it tests the old models at new bar sizes.
* **One inferred assumption was wrong.** The same database shows that the handbook's normalised values were
  measured on 25 mm bars, not the 50 mm we had inferred from the physics (Section 8.2). All results in this version
  use the documented size.
* **The full route.** Hardening, tempering and stress relieving are modelled with the same kinetics
  (Section 7), and every assumption is listed with a sensitivity analysis (Section 8) and a proper posterior of
  the kinetic constants (Section 9).

### 1.3 The questions, and what metallurgy expects

We ask four questions, in the order a metallurgist would:

1. **What kind of steel is it?** (low, medium or high carbon, free-machining, C-Mn, Cr, Cr-Mo, Ni-Cr-Mo, Si-Mn,
   stainless, tool, or "none of these")
2. **What happens when we furnace cool it?** Which constituents form, in what amounts, how fine.
3. **What properties does that give?** Yield and tensile strength, uniform and total elongation, hardness and a
   toughness indicator.
4. **And if we air cool it instead?** What changes, by how much, why, and how the bar size matters.
5. **And if we quench it, temper it or stress relieve it?** The rest of the classic route (Section 7).

The textbooks lead us to five expectations, which we treat as hypotheses to test rather than assume:

* **H1.** Air cooling transforms the austenite colder, so the pearlite is finer (the spacing goes roughly as
  1/ΔT) and the ferrite grains are finer, which raises yield strength, UTS and hardness and lowers elongation a
  little.
* **H2.** The gain grows with the amount of pearlite: small in low-carbon steels, large near the eutectoid.
* **H3.** Hardenable alloy steels form bainite or martensite when air cooled, so they become a different steel,
  which is why such steels are tempered after normalising.
* **H4.** A thick bar cools slowly in air and drifts towards the annealed state (the mass effect).
* **H5.** Normalised steel holds more pearlite than annealed steel of the same composition (the quasi-eutectoid
  effect).
* **H6.** Tempering softens martensite along one master curve of the Hollomon-Jaffe parameter, and alloying
  elements (Mo, Cr, Si, V) delay the softening.
* **H7.** A quenched bar is weaker the larger it is, even after tempering, because its centre forms less martensite.
* **H8.** A stress relief below the prior tempering parameter removes residual stress without changing hardness.

### 1.4 What is new in this work

* An open, end-to-end chain from composition to properties for the two routes, with honest 90 % intervals and a
  refusal to answer for steels unlike anything it knows.
* A Bayesian calibration of the transformation kinetics on «jm_points» points of the SAE hardenability bands, with
  grade-grouped cross-validation and a test of whether the calibration transfers to independent data.
* Physics-anchored machine learning in which the physics carries the extrapolation (new cooling rates, new
  microstructures) and the machine learning corrects what the textbook relations get wrong; the alternatives we
  tried and rejected are kept as ablations.
* A compilation of «lit_rows» measured values from «lit_src» open papers, each number traceable to a table and checked by
  script against the saved paper.
* (v3) The five classic treatments in one chain, a quench heat-flow model checked against the exact solution, a
  tempering model fitted and tested by grade, an explicit register of every assumption with its sensitivity, and an
  MCMC posterior of the kinetic constants propagated to predictions.

## 2. Background

### 2.1 Annealing and normalising

In **full annealing** a hypoeutectoid steel is austenitised a little above Ac3 (a hypereutectoid steel above
Ac1, so that the cementite network is not dissolved and re-formed) and cooled with the furnace, typically at
10 to 30 °C per hour through the transformation range. In **normalising** the steel is austenitised about 50 to
100 °C above the upper critical temperature (Ac3, or Accm for hypereutectoid steels) and cooled in still air
(ASM Handbook Vol. 4; Krauss 2015). The two routes therefore differ in two ways at once, the austenitising
temperature and the cooling rate. Air cooling makes the austenite transform at a larger undercooling, so the
pearlite is finer (Zener 1946; Marder and Bramfitt 1976) and the ferrite grains are smaller; in hypoeutectoid
steels the austenite can enter the extrapolated (Hultgren) region before all the equilibrium ferrite has formed,
which gives more than the equilibrium amount of pearlite. In steels with enough hardenability the ferrite and
pearlite reactions are delayed so much that bainite or martensite forms in an air-cooled bar; hardenability is
measured by the Jominy test and summarised by Grossmann's ideal critical diameter D_I (Grossmann 1942;
ASTM A255). Because a thicker bar cools more slowly, the same steel normalised as a thick and as a thin bar can
end up with different structures (the mass effect).

### 2.2 Predicting what forms on cooling

The classical route is a semi-empirical kinetic model for the ferrite, pearlite and bainite reactions with an
additivity rule for continuous cooling: Kirkaldy and Venugopalan (1984), refined by Li et al. (1998), who
calibrated it on hardenability data. Collins et al. (2023) published an open implementation with dilatometry
of three steels, which we use as a check. Commercial tools such as JMatPro combine CALPHAD thermodynamics with
kinetic and structure-property models (Saunders et al. 2003). Machine learning has been used to predict CCT and
TTT diagrams directly from composition (Trzaska and Dobrzański 2007; Huang et al. 2020), and recent work adds
physical constraints or physics-based features to such models (Cao et al. 2023; Hedström et al. 2026). These
data-driven CCT models need thousands of diagrams, often from closed databases, and their authors note that
extrapolation to new steels is hard.

### 2.3 From microstructure to properties

For ferrite-pearlite steels the Gladman, McIvor and Pickering (1972) mixture law gives yield and tensile strength
from the ferrite fraction, the ferrite grain size (Hall-Petch) and the pearlite spacing; Pickering (1978) gives the
impact transition temperature from the same quantities. Maynier et al. (1978) give the hardness of ferrite +
pearlite, bainite and martensite as functions of composition and cooling rate. Uniform elongation follows from the
Considère criterion if the flow curve is known, but steels with a yield-point plateau do not follow a single
Hollomon law (Hollomon 1945).

### 2.4 Machine learning with small materials data

Neural networks for steel properties go back to Bhadeshia (1999); later work predicts properties from
composition and processing (Agrawal et al. 2014; Xiong et al. 2020) or guides the model with physical
metallurgy (Shen et al. 2019). With small data the main risks are extrapolation and over-confidence: random
cross-validation hides how a model does on genuinely new materials (Meredig et al. 2018), a model should say
where it is reliable (Sutton et al. 2020), and intervals should be calibrated, for example by conformal prediction
(Lei et al. 2018). Correcting a physics model with a statistical discrepancy term is the idea of Kennedy and
O'Hagan (2001), and choosing the next experiment by uncertainty is active learning (Lookman et al. 2019).

### 2.5 The gap

We did not find an open model that answers our question end to end for arbitrary carbon and low-alloy
compositions, from the category through the microstructure to the tensile properties of both routes, with
honest uncertainty, and with every link tested on measured data. That is what we tried to build, and where we
fell short, we say so.

## 3. Data

We used only open data, or factual handbook values that are openly reproduced, and kept every file with its
source and licence (`data/README.md`, `data/raw/source_manifest.csv`).

| Data | What it gives us | Size | Role |
|---|---|---|---|
| ASM Handbook / Bethlehem Steel table (via Engineers Edge) | typical UTS, YS, elongation and hardness of 33 AISI grades **annealed**, **normalised** and hot rolled, with the treatment temperatures | 80 rows | training the property models |
| SAE J403 / J1268 ranges (Summit Steel), Wikipedia, Zenodo 17704410 | composition ranges and families | 547 grades | category classifier |
| SteelBench (Zenodo 18530558) | real heat analyses with family labels | 1358 heats | test of the classifier |
| Collins et al. 2023 (Zenodo 7770264) | measured transformation-start temperatures and hardness at 9 cooling rates for EN3B, EN8, SA-540 | 3 steels | kinetics, cooling-rate extrapolation |
| Rose and Hougardy atlas, as tabulated (CC BY) by Smokvina Hanza et al. 2021 and Marohnić et al. 2025 | measured ferrite + pearlite, bainite, martensite fractions and hardness after continuous cooling | 423 points, 19 steels | kinetics validation |
| Metallus (2024) Practical Data for Metallurgists | SAE end-quench hardenability bands (Jominy), cooling-rate scale of the ASM hardenability charts | «jm_grades» H-grades | kinetics calibration and test |
| Literature compilation (this work, 52 papers) | measured fractions, spacing, grain size, tensile, hardness, impact energy | «lit_rows» rows | microstructure and property checks |
| Kingsley et al. 2026 (Zenodo 19000367) | chemistry, tensile and Charpy results of hot-rolled S275/S355 members | 165 members | external test, recalibration |
| Carvajal et al. 2017 (CC BY) | micrographs of SAE 1045 and 4140, annealed and normalised | 2 steels | visual check |
| **Birbilis et al. 2025 (Mendeley Data, CC BY 4.0)** (v3) | UTS, YS and elongation of AISI grades with the treatment written out: quench medium, austenitising and tempering temperature, bar size (13, 25, 50, 100 mm) | «bb_raw» entries; «qt_n» Q+T bars of «qt_grades» grades and «bar_n» normalised / annealed bars after cleaning | training and testing hardening + tempering; test at new bar sizes; documents the handbook bar size |

A few things about the data were important to get right. Handbook "typical" values and specification "minimum"
values are very different numbers for the same steel (the minima are 25 to 35 % lower), so we never mixed them.
The composition tables had real errors (a row shift in the alloy H-steel table, swapped Mn ranges of 1117 and
1118, missing decimal points, "15821H" printed for 15B21H), which we corrected and logged in
`data/raw/source_corrections.csv`. The second literature search found many papers that test the same steel after
both routes, but most come from weak journals and contain inconsistent values (an annealed 1045 at 56 HV, a 0.2 %
C steel with 50 % pearlite); every number we transcribed was checked by script against the saved paper, and the
«lit_suspect» rows we judged implausible are kept with a `quality_flag` but left out of every metric. The phase-fraction
file of Collins et al. turned out to be their model output, not metallography, so it is used only to check our
implementation, never as evidence.

The Birbilis database needed the same care. It lists many grades twice (4140 and 4140H, for example) with
identical numbers, so after removing stainless, tool and cast steels, carburised and cold-drawn products, and the
repeated rows, «bb_raw» entries become «qt_n» quenched-and-tempered bars and «bar_n» normalised or annealed bars
(`src/steelml/data/birbilis.py`, with tests). Its compositions are nominal, as in the handbook. The tempering
time is not stated, so we assume 1 h and test 0.5 and 2 h (Section 8). Two other open sources we looked at were
not usable for this purpose. The NIMS rows of SteelBench give tensile strength for 360 steels but no heat
treatment, which is the flaw of our first version. The SteelBench EMK rows are specification minima with
compositions that look like upper limits.

### 3.1 Is our dataset big enough?

A fair question we were asked is why our data are a few megabytes when other student projects use a 1 GB
dataset. Steel datasets of that size are almost always images, such as micrograph collections (the NIST
ultrahigh-carbon steel set alone is 961 SEM images, 270 MB), or large tables without the processing.
Neither can teach a model what furnace cooling, air cooling or tempering do to strength. What limits a property
model is the number of distinct steels and treatments with a measured property and a known treatment, and
in tabular form that is never large. Our data are about «data_rows» such records (handbook and Birbilis
bars, Jominy band points, CCT-atlas points, dilatometry, literature values and structural members), plus «sb_heats»
real heat analyses for the classifier. Every row has its heat treatment.

Size still matters in one way: the property data cover «qt_grades» to «hb_grades» grades, so new grades are where the
models can fail. That is why every error in this report is measured on grades the models never saw, and why
the physics carries the extrapolation. A micrograph set could be added for a different purpose, an image check
of the predicted constituents, and we list it as future work.

## 4. Methods

The whole chain is summarised below; Appendix A gives every equation as implemented, and Appendix B lists every
constant that we chose after looking at data.

```
composition ─► classifier (ML, cross-checked by AISI/SAE rules) ─► category, or "unknown"
     ├─► thermodynamics (Grange Ae1/Ae3, Acm, eutectoid point, lever rule, default temperatures)
route + bar size ─► cooling path (furnace; round bar in still air; Jominy bar)
     └─► kinetics (Li et al. 1998, bainite calibrated on the H-bands) ─► ferrite, pearlite, bainite,
                                                                          martensite, spacing, grain size
          physics strength (Gladman for F+P, Maynier for B+M) ─► physics-anchored GP ensemble
                                                                 + conformal 90 % intervals
          YS, UTS, total and uniform elongation, HB, impact transition temperature
```

### 4.1 The steel category

We simulated 40 heats per grade inside the published composition ranges of 547 grades (21,880 heats), trained
5 classifiers, and tested them on grades they had never seen (grade-grouped cross-validation) and on 1358 real
heats. Before looking at the results we fixed the rule that the production model would be the one with the best
accuracy on the real heats. As a fair baseline we wrote the AISI/SAE composition limits as a decision list, with
nothing fitted (S ≥ 0.07 % free-machining, Si ≥ 1 % Si-Mn, Ni ≥ 0.25 % Ni-bearing, Mo ≥ 0.08 % Mo/Cr-Mo, Cr ≥
0.30 % Cr, Mn ≥ 1.0 % C-Mn, carbon limits 0.25 and 0.60 % following Callister; stainless and tool steels by the
usual Cr, Ni, W, Mo, V limits). A novelty score (distance to the nearest known grade) lets the classifier say "I
don't know".

### 4.2 Cooling paths

Furnace cooling is Newtonian cooling with the furnace at 0.005 K/s at 700 °C (18 K/h). Air cooling is a heat
balance of a long round bar in still air, with natural convection, radiation (emissivity 0.8) and the
temperature-dependent heat capacity of steel from EN 1993-1-2; the Biot number stays below 0.1 up to 100 mm, so
the bar is treated as lumped. The model gives a mean 800-500 °C rate of «cool_t85_25» K/s for a 25 mm bar and
«cool_t85_50» K/s for a 50 mm bar, close to measured values of 0.89 K/s for a 25 mm bar and 0.36 to 0.46 K/s for
45 to 50 mm bars (a patent study, US 9,777,354; Toshioka et al. 1968). The Jominy bar is solved by 1-D
conduction along its length with a water-jet coefficient of «h_q» W m⁻² K⁻¹, set so that the cooling rates match
the scale of the ASM combined hardenability charts (within «jr_lo» to «jr_hi» times). One simplification is worth
stating: the Eurocode heat capacity puts the transformation heat near 735 °C for every steel, which for an alloy
steel that transforms near 450 °C slightly misplaces the heat release.

### 4.3 Critical temperatures and the default heat treatment

Ae1 and Ae3 come from Grange (1961), Acm from the fit used by Collins et al. (2023), the eutectoid point from
their intersection and the equilibrium fractions from the lever rule. Two of these have known limits that we
checked: Grange's silicon terms move the eutectoid carbon up with silicon (0.74 → 0.87 % C at 2 % Si), the wrong
way, so compositions above 1 % Si are flagged as outside the window of the equations; and the Acm fit falls
below the binary Fe-Fe₃C solvus above about 0.9 % C, so above the eutectoid the upper critical temperature is
drawn with the binary slope. When the user gives no temperature, the tool uses the practice of its own training
data (Figure 26): full annealing at max(Ae3 + 50, Ae1 + 65) °C and normalising at the upper critical
temperature + 110 °C, but not below 870 °C, which reproduce the handbook temperatures within about 15 °C and are
continuous in composition.

### 4.4 Transformation kinetics

We wrote the Kirkaldy-Venugopalan model as modified by Li et al. (1998) from its equations and integrated it
along any cooling path with the additivity rule; it reproduces the authors' own implementation (Collins et al.
2023) to 1-2 °C for the ferrite and bainite start. Ferrite forms below Ae3 up to the extrapolated-Ae3 limit and
enriches the austenite; pearlite starts once the austenite is below both its Ae3 and Acm (the Hultgren
condition); bainite forms below Bs; martensite follows Koistinen-Marburger. The prior-austenite grain size is
ASTM 8 up to 850 °C and one number coarser per 100 °C above.

**Calibrating the bainite reaction on the hardenability bands.** Section 5.3 shows that the published constants
are right at the quenched end of the Jominy bar but too hardenable further along it. We therefore calibrated four
factors (multipliers on the ferrite, pearlite and bainite rate constants and a shift of Bs) on the SAE H-bands of
the non-boron H-steels, with the published model as the prior (log-factors ~ N(0, 1), ΔBs ~ N(0, 50 °C)), the
middle of each band as the target, a quarter of the band width plus a fitted model error as the noise, and the
maximum a posteriori point with a Laplace approximation. Boron grades are left out because neither the kinetics
nor the hardness relations contain boron. The bands do not constrain the ferrite and pearlite rates (they stay at
the published values), and the calibration lowers the bainite start by «dBs» °C and speeds the reaction below it
(factor «fB»). Because hardness alone could, in principle, be fitted by the wrong mechanism, we only adopted the
calibration after checking that it transfers to data it never saw (Section 5.3).

### 4.5 Microstructure quantities

**Phase fractions.** The kinetics decide how much bainite, martensite and proeutectoid cementite form. The
remaining ferrite + pearlite is reported with its lever-rule split as the central value and the kinetic split as
the top of a band, because measured normalised fractions sit at or below the lever value while the kinetics put
far too much pearlite into air-cooled medium-carbon steels (Section 5.4). **Pearlite spacing:** S₀ = 3.33 ΔT^−0.607
µm, fitted on 14 measured continuous-cooling spacings from 4 papers; the exponent below 1 compensates for the
model forming air-cooled pearlite too cold, so it is a correction, not a new law. **Ferrite grain size:** the
low-carbon branch of Hodgson and Gibbs (1992) with C_eq held at 0.35 at most. Their high-C_eq branch put every
medium-carbon steel on a 3 µm floor, and against 20 measured grain sizes the change lowers the typical error from
a factor of «gs_old» to «gs_new» with nothing fitted.

### 4.6 Property models

The property models learn from 80 handbook rows, so the important question was how to use physics so that they
generalise. We compared a ladder of models with leave-one-grade-out cross-validation (a grade is never in training
when it is predicted): the textbook relations alone; ridge, gradient boosting and a Gaussian process (GP) on
composition; GPs on physics descriptors; the textbook relations with a GP residual; a grey box (the Gladman law
with learned, sign-constrained coefficients); and physics-anchored GPs, in which each property is scaled on the
physics strength UTS_phys (the Gladman law on the predicted ferrite fraction, grain size and spacing for the
ferrite + pearlite part, Maynier hardness for bainite and martensite) and a GP learns only a multiplicative
correction that fades away far from the data.

The **production model** averages three partly independent routes from microstructure to properties: the
physics-anchored chain, the textbook relations with a GP residual, and a GP on composition and physics
descriptors. When the cooling produces more martensite, or more bainite + martensite, than any training steel,
the weight of the purely data-driven member moves to the two physics-anchored members.

Three decisions about the inputs mattered as much as the choice of model, and we found them by checking how the
models respond to a change of input, not by cross-validation. First, the microstructure enters the ML in the form
of the Gladman mixture law (ferrite share, ferrite Hall-Petch term, pearlite fineness, each weighted by the ferrite
+ pearlite fraction), so that it goes smoothly to zero for a bainitic steel. Second, silicon and vanadium are not
ML inputs: every training grade has the same nominal silicon apart from one, and a model that used silicon looked
better in cross-validation but swung by «si_span_with» MPa across a normal 0.10-0.40 % Si. Third, the anchor: our first
anchor was the Maynier hardness for every constituent, whose ferrite-pearlite term depends on composition and
cooling rate only and so cannot see pearlite refinement; anchoring on the Gladman law instead (ablation in Section
5.5) made the model much better at new cooling rates and new compositions, at a small cost inside the data.

### 4.7 Uncertainty, domain and evaluation

The 90 % intervals are normalised split-conformal intervals (half-width = q × the model's own standard
deviation), with coverage checked by leave-one-grade-out. A composition is "in the domain" if it lies within the
training range and close to a training grade. Paired comparisons use per-grade errors, a Wilcoxon test with the
Holm adjustment over the comparisons of each property, and a bootstrap interval of the RMSE difference. Physical
consistency is checked on sweeps that the data do not contain: cooling rate (furnace 0.003-0.03 K/s and air-cooled
bars of 10-150 mm), carbon (0.05-0.85 %) and silicon (0.10-0.40 %). Several design choices (descriptor forms, the
gate, silicon, the anchor) were made after seeing evaluation results; they are documented as ablations, and the
cross-validation is therefore not fully blind.

### 4.8 Uniform elongation and toughness

None of the open handbook data report uniform elongation. A Hollomon curve through the yield point and the UTS
fails for steels with a Lüders plateau: on the literature tests it is «hollomon_mae» points off, and for 12 of the 80
handbook rows it gives a uniform elongation larger than the total elongation. What works is simpler: once every
test's total elongation is put on the handbook gauge (4.5√S₀) with Oliver's relation (ISO 2566-1), uniform
elongation is a stable fraction of it, «ue_ratio» ± «ue_ratio_sd» (10 tests, 4 papers), and predicting each paper from
the others is «ue_loso» points off. The gauge conversion alone cut the scatter of the ratio by a third. The tool
reports uniform elongation as this fraction of the predicted total elongation, with an interval that combines
both uncertainties, and n = ln(1 + e_u) as the Considère uniform true strain (which includes the Lüders strain).
Toughness is reported as Pickering's impact transition temperature on the predicted microstructure, an indicator
defined for ferrite-pearlite steels only.

## 5. Results

Every number below comes from a file in `results/`, and every figure from `figures/`.

### 5.1 Which steel is it?

All 5 classifiers learned the 13 categories well on grades they had never seen (accuracy 0.84 to 0.93,
`results/s02_classifier_cv.csv`). XGBoost, chosen by the rule fixed in advance, reaches «clf_cv» on unseen grades
and «clf_sb» at the family level on the 1358 real SteelBench heats. The AISI/SAE decision list, with nothing fitted,
reaches «rules_cv» and «rules_sb»: a strong baseline, as it should be, since the categories are defined by those
limits. The ML model is better where it matters: on real heats, near the limits, and in recognising steels that
belong to no category (all 312 maraging steels of our old dataset are flagged as unknown, against «rules_maraging» for
the rules). The two disagree on «disagree_share» of the real heats, almost all near a limit such as Mn ≈ 1.0 % or
C ≈ 0.25 %; where they agree the family is right «agree_acc» of the time and where they disagree only «disagree_acc», so the
app shows both. SHAP (`fig04_category_shap`) points to the elements a metallurgist would look at: sulphur for
free-machining, silicon for spring steels, molybdenum for Cr-Mo, nickel for Ni-Cr-Mo, carbon for the plain-carbon
classes, and chromium for almost every class, because its absence (a residual 0.1 %) is what separates a plain
carbon steel from a Cr steel.

### 5.2 Cooling paths

A 25 mm bar in still air cools at «rate25» K/s at 700 °C, a 50 mm bar at «rate50» K/s, a 100 mm bar at «rate100» K/s
(`fig27_cooling_paths`), and the mean 800-500 °C rates agree with measured bars (Section 4.2). The far end of a
Jominy bar (J32, 51 mm from the quenched end) cools at «rateJ32» K/s, like a 10 mm bar in still air, so the
hardenability bands test the kinetics from thin air-cooled bars up to quench rates.

### 5.3 The transformation model

**Start temperatures.** Against the measured dilatometry of EN3B, EN8 and SA-540 at 0.1 to 50 K/s
(`fig07_kinetics_validation`), the model predicts the ferrite start within «Fs» °C (MAE, 18 cases), the pearlite
start within «Ps» °C and Ms within «Ms» °C. With the published constants the bainite start was «Bs_pub» °C too high on
average, a known weakness of the Li model; after the calibration described below it is «Bs_now» °C.

**CCT atlas.** Against 423 measured continuous-cooling points of 19 low-alloy steels (`fig28_cct_atlas_validation`),
the model predicts the ferrite + pearlite fraction within «atlas_fp» (MAE), bainite within «atlas_b», martensite within
«atlas_m» and hardness within «atlas_hv» HV (RMSE). It is good for quench-like cooling (t₅₀₀ < 20 s: ferrite + pearlite
within «atlas_fp_fast») and weaker in the slow-to-intermediate range where bainite competes with ferrite and pearlite
(«atlas_fp_slow»).

**Jominy hardenability bands.** Against the SAE H-bands of «jm_grades» grades (`fig29_jominy_hbands`), the published
constants are right at the quenched end (J1-J4: bias «jm_bias_end_pub» HRC, «jm_in_end_pub» of the points inside the band) but
too hardenable further along the bar (J5-J16 bias «jm_bias_mid_pub» HRC; J18-J32 bias «jm_bias_far_pub» HRC, only
«jm_in_far_pub» inside the band). This is the same direction as the too-high bainite start of the dilatometry and the too-hard
air-cooled 4140 in the literature, three independent signs of one problem.

**Calibration and transfer.** The Bayesian calibration (Section 4.4) changes only the bainite reaction: Bs lower by
«dBs» °C and a faster reaction below it. On grades it never saw (5-fold, grouped by grade) the band error falls from
«jm_rmse_pub» to «jm_rmse_cv» HRC and the share of points inside the band rises from «jm_in_pub» to «jm_in_cv». The real question is
whether this is better kinetics or just a better fit to one kind of data, so Table 1 checks it against everything
the calibration did not see.

**Table 1.** Does the calibration transfer? (`results/s08_kinetic_calibration_transfer.csv`)

«transfer_table»

It transfers on most counts: the bainite start moves onto the dilatometry, Ms improves, the handbook hardness of a
25 mm normalised bar (the probable size of the handbook bars) improves a lot, and so do the CCT-atlas hardness and
the air-cooled pearlite fractions. It does not transfer to the dilatometry hardness, and the reason is one steel:
SA-540 (3.5Ni-1.7Cr-0.5Mo), more highly alloyed than any H-grade, which the calibrated model makes bainitic at
0.5-1 K/s where it was measured martensitic. A joint calibration that also used the dilatometry start temperatures
gave almost the same result. We adopted the bainite calibration for the whole chain, and we state its limit: it is
learned on SAE H-steel chemistries and is not valid for steels as rich as SA-540.

### 5.4 Microstructure

**Phase fractions (H5).** For furnace-cooled steels the kinetics and the lever rule both agree with the measured
pearlite fractions (MAE «pf_fc_kin» and «pf_fc_lev» points). For air-cooled steels the textbook expects extra pearlite, and
the Li model shows it, but far too strongly: 82-86 % pearlite for normalised 1045 and S45C, where 50-56 % was
measured (MAE «pf_ac_kin» points against «pf_ac_lev» for the lever rule). Starting the pearlite incubation only inside the
Hultgren region did not help (`results/s03_pearlite_coupling_ablation.csv`). The evidence is honestly mixed: a 1966
regression of hot-rolled and normalised low-carbon steels (Grozier and Bucher, as compiled by Gorni) gives about
twice the lever-rule pearlite at 1 K/s, while the image-analysis measurements in our compilation sit at or below it,
perhaps partly because of what is counted as pearlite. We therefore report it as a band and regard H5 as quantitatively unsettled.

**Spacing and grain size.** The spacing law is good to a factor of about 1.7 when each paper is predicted from the
others; the ferrite grain size to a factor of about 2 (`fig21_literature_validation`). Normalising refines the
ferrite from «d_lo_fc» to «d_lo_ac» µm in low-carbon steels and from «d_me_fc» to «d_me_ac» µm in medium-carbon steels
(25 mm bar).

**Micrographs.** Figure 30 sets published micrographs of SAE 1045 and 4140, annealed and normalised (Carvajal et al.
2017, CC BY 4.0), beside the model's prediction for that composition, temperature and bar. The annealed 1045 and 4140
are ferrite + pearlite as predicted, the normalised 1045 is finer ferrite + pearlite, and the normalised 4140 has the
fine acicular structure of the bainite the model predicts (the authors call it Widmanstätten ferrite and fine
pearlite; reading it as bainite is ours).

### 5.5 From microstructure to properties

**Table 2.** Error on unseen grades inside the domain (leave-one-grade-out RMSE, 27 grades, `results/s04_model_benchmark.csv`)

«ladder_table»

The textbook relations alone are «uts_phys» MPa off for UTS, and for elongation there is no textbook equation (that row
is the training mean). The production ensemble predicts UTS within «uts_rmse» MPa (R² «uts_r2») and hardness within «hb_rmse» HB
(R² «hb_r2»). Paired tests with the Holm adjustment (`results/s04_paired_tests.csv`) show it clearly better than the textbook
relations for UTS (adjusted p = «p_phys»), but inside the data it ties with a plain composition GP (difference «gp_diff» MPa,
95 % interval «gp_lo» to «gp_hi»). So inside the data the physics buys little accuracy; what it buys is what happens outside.

**New cooling rates.** On the dilatometry steels at 0.1-5 K/s, rates and microstructures the handbook does not
contain, the production model is «dil_prod» HB off, the textbook relations «dil_phys» and the composition GP «dil_gp»
(`fig14_cooling_rate_extrapolation`): data-only models cannot invent the hardness of martensite.

**The anchor (ablation).** «anchor_sentence»

**Physical consistency.** Over cooling-rate sweeps of all 33 grades the production UTS moves the wrong way in «viol_uts»
of the steps, and over a normal 0.10-0.40 % Si it changes by «si_span» MPa, close to the few tens of MPa that physics
expects (with silicon as an ML input it swung by «si_span_with» MPa).

**How much error is irreducible?** The handbook gives one value per grade at the nominal composition. Moving each grade's
composition inside its own specification range moves the predicted UTS by «noise_fc» MPa (RMS) after furnace cooling and
«noise_ac» MPa after air cooling, where hardenability varies within a grade. The model's «uts_rmse» MPa is of the same size, so
much of what is left is the unknown real composition of the tested heats, not the model.

**What the grey box learned.** Fitted under sign limits, the Gladman law gives the pearlite spacing about twice the weight
Gladman et al. found (`fig15_greybox_coefficients`), but it pins the ferrite Hall-Petch coefficient and the solid-solution
term at their bounds: with 33 grades and predicted grain sizes these are not identifiable, and we do not read them as
physical results.

### 5.6 Data the models never saw

**Structural members.** For the 165 hot-rolled S275/S355 members the UTS is «st_uts_rmse» MPa off (bias «st_uts_bias»), the yield
strength «st_ys_bias» MPa too low on average, and the elongation, on the handbook gauge, within «st_el_rmse» %. The predictions do
not rank the members (Spearman «st_ys_rho» for yield strength): modern sections owe their strength to rolling practice and
microalloying (the bias is «st_ys_bias_micro» MPa for the 69 members with Nb, V or Ti), which our handbook bars did not have.
The predicted impact transition temperature ranks their Charpy energy at 0 °C with the right sign (Spearman «charpy_rho»).

**Literature.** On the unflagged literature values the yield strength is «lit_ys» MPa off (MAE), UTS «lit_uts» MPa, hardness
«lit_hb» HB and the ferrite grain size «lit_gs» µm (`results/s06_literature_metrics.csv`). For the three steels tested after both
routes in the same paper (4140, 42CrMo4, 8620) the direction of the change is right every time, but for these hardenable
steels the predicted hardening is larger than measured, the over-hardenability seen in the Jominy bands.

## 6. What happens when we furnace cool or air cool each kind of steel

This is the answer to the question we started with, for the 271 catalogue grades that form ferrite + pearlite on slow
cooling (medians per category, `results/s05_category_summary.csv`, `fig18_category_route_effects`,
`fig19_category_microstructures`). Furnace cooling is 0.005 K/s; air cooling is a 25 mm bar (the handbook's
normalising condition, now documented, Section 8), with a 50 mm bar in brackets.

**Table 3.** Furnace cooling (FC) against air cooling (AC), per category

«category_table»

**Low-carbon steels.** Both routes give ferrite with about 15 % pearlite; air cooling only refines it (finer pearlite,
ferrite «d_lo_fc» → «d_lo_ac» µm), so UTS rises modestly and ductility stays. These steels are normalised to refine the grain
and make it uniform, not for strength, and the impact transition temperature falls by about «itt_lo» °C.

**Medium- and high-carbon steels.** The same phases, but the pearlite is much finer and there is more of it to refine,
so the gain grows with carbon: about «duts_me» MPa for medium carbon and «duts_hi» MPa for high carbon (25 mm bar). This is
the classic coarse-pearlite against fine-pearlite chart (`fig16_carbon_sweep_fc_vs_ac`), now with intervals. Near the
eutectoid the furnace-cooled curve flattens because the handbook's annealed 1080 is softer and more ductile than its
annealed 1060; that row looks partly spheroidised and we treat it as suspect rather than explain it.

**Free-machining and C-Mn steels.** Like plain carbon steels of the same carbon content after furnace cooling; their
manganese delays the ferrite and pearlite enough that a few grades form some bainite in thin bars.

**Cr, Cr-Mo and Ni-Cr-Mo steels.** Here air cooling gives a different steel. Their hardenability (median D_I «di_crmo» mm for
Cr-Mo) delays ferrite and pearlite so much that a 25 mm bar forms bainite, and the UTS rises by «duts_crmo» MPa for the median
Cr-Mo grade (25 mm bar). Hardenability orders all grades this way (`fig25_hardenability_vs_air_cooling`): below D_I ≈
40 mm almost no 25 mm bar forms more than half bainite + martensite, above 100 mm almost all do. Such steels are tempered
after normalising, which we do not model.

**Si-Mn spring steels** (2 % Si) are outside the window of the critical-temperature equations; their row is shown only for
completeness.

**Why air cooling is stronger.** Air cooling a bar from the annealing temperature separates the two differences between
the routes: for medium-carbon steels «dcool_me» MPa of the gain comes from the faster cooling and «dtemp_me» MPa from the
hotter austenitisation. The temperature part is learned from data in which temperature and route go together, so it is
the less certain part.

**Bar size.** For 1020 and 1045 the bar size changes the UTS by only tens of MPa, because the structure stays ferrite +
pearlite and only coarsens; for 4140 it decides between martensite, bainite and ferrite + pearlite
(`fig17_section_size_effect`). For hardenable steels the bar size matters as much as the choice between the furnace and
the air.

In one sentence: for low-carbon steels furnace and air cooling give almost the same steel, for medium- and high-carbon
steels air cooling gives the same phases with much finer pearlite and much more strength, and for Cr-Mo and Ni-Cr-Mo
steels it gives a different, bainitic steel, so the route and the bar size have to be chosen together.

## 7. The rest of the route: hardening, tempering and stress relieving

A heat-treatment shop does five classic things to a carbon or low-alloy steel, and versions 1-2 of this project
modelled two of them. The chain now covers all five with the same kinetics (`pipeline/s09_hardening_tempering.py`).
Table 4 sets the textbook summary of each treatment against what the model predicts for one grade.

**Table 4.** The five treatments of a 25 mm AISI 4140 bar: textbook expectation and model prediction (90 % intervals;
quench in oil; tempering and stress relief at 600 °C for 1 h)

«five_table_4140»

### 7.1 Hardening: the quench heat flow

A quenched bar is no longer a lumped body, because its surface and centre cool very differently, so we solve
transient radial conduction in a long round bar (finite volumes, the EN 1993-1-2 conductivity). The surface
loses heat to the bath with Grossmann's severity of quench H, converted to a heat-transfer coefficient with
h = 2kH (k = 30 W m⁻¹ K⁻¹); an ideal quench holds the surface at the bath temperature. With constant
properties the solver matches the exact Bessel-series solution within 2 % of the temperature difference
(tests/test_quench.py). The Eurocode heat capacity has a large peak near 735 °C, which stands for the heat of the
ferrite + pearlite reaction during slow cooling. In a quench the austenite is undercooled far below 735 °C
without transforming, and with the peak the surface of an oil-quenched 50 mm bar came out softer than its
centre, which is the wrong way round. The quench therefore uses the heat capacity without the peak (Section 8
tests the alternative). The centre of a 25 mm bar cools at «q25_oil» K/s at 700 °C in oil and «q25_water» K/s in water;
the centre of a 100 mm bar at «q100_oil» and «q100_water» K/s. Tensile properties are predicted at the specimen position of
ASTM A370: the centre of bars up to 38 mm and mid-radius above.

**A hardenability check that does not use our data.** Grossmann's ideal critical diameter D_I is the bar that
just reaches 50 % martensite at its centre in an ideal quench. The ASTM A255 factors give it from measured
hardenability, while our heat flow and kinetics give it from first principles. For the «di_n» quenched-and-tempered
grades our D_I is «di_ratio_cal» times the A255 value (median) with the recalibrated kinetics and «di_ratio_pub» times with the
published constants; the ranking agrees (Spearman «di_rho_cal»). So the model is still too hardenable, most of all for
high-carbon steels («di_1080»× for 1080). It is the same bias the Jominy bands showed (Section 5.3), and the
recalibration roughly halved it (`fig31_quench_and_hardenability`).

### 7.2 Tempering

Tempering is described by the Hollomon-Jaffe parameter P = T(20 + log t), with T in kelvin and t in hours.
Martensite softens from its as-quenched hardness towards a fully tempered hardness along a logistic curve in
P. Alloying elements move the curve to higher P, and Mo, V and Cr add a secondary-hardening bump near 600 °C
(Appendix A.8). As-quenched hardness levels off above about 0.6 % C at 66 HRC because of retained
austenite, so the linear carbon term of Maynier's relation is capped at 865 HV. The constants were fitted on
the tensile strengths of the «qt_n» bars, through UTS = 3.3 HV, with a grade bootstrap for their uncertainty:

**Table 5.** Tempering constants (prior → fitted ± grade-bootstrap standard deviation)

«temper_table»

Two of the results are metallurgy we did not put in. First, the fully tempered hardness depends on carbon by only
«s_c_fit» HV per % C, against about 950 HV per % C for untempered martensite (Maynier). Grange, Hribal and Porter (1977)
found the same: carbon's effect fades as tempering proceeds. Second, molybdenum delays the softening most strongly,
about «p_mo_degC» °C of tempering temperature per 0.1 % Mo at 600 °C (H6). That is a large effect. Part of it is probably
the difference in as-quenched structure between the Mo-bearing grades and the others, which are less hardenable,
so we read it as a direction and an order of magnitude. The tempering curves of `fig35_tempering_curves` show the result for four grades. They also
mark the windows of tempered martensite embrittlement (250-400 °C) and temper embrittlement (375-575 °C), which
the tool flags.

### 7.3 Property models for quenched-and-tempered steel

The same ladder as Section 5.5, scored by leaving out one grade at a time:

**Table 6.** Leave-one-grade-out RMSE on «qt_n» quenched-and-tempered bars

«qt_ladder_table»

The textbook chain alone is far off («qt_uts_phys» MPa), because the tempering constants of the literature form are not
fitted. Fitted to the data (grey box), the physics chain alone reaches «qt_uts_grey» MPa, close to a Gaussian process on
composition and process variables («qt_uts_gp» MPa) and better than gradient boosting («qt_uts_gbm» MPa). The production
model, an equal mixture of the physics-anchored GP and the data GP, is best for every property: UTS «qt_uts_ens» MPa,
yield strength «qt_ys_ens» MPa and elongation «qt_el_ens» %. Its normalised conformal 90 % intervals contain «qt_cov» of the bars of
grades left out of their calibration (median width «qt_width» MPa). The bias does not change with bar size
(`fig34_qt_mass_effect`): the model reproduces the mass effect (H7) because the quench heat flow and the
kinetics carry it, not because it was told.

**An independent test of the recalibrated kinetics.** The Jominy recalibration of Section 5.3 was fitted on
hardness bands. These bars were never part of it, so we ran the whole Q+T chain with both sets of constants. The
result is mixed. The physics chain alone (grey box) is slightly worse with the recalibrated constants («qt_kin_grey_cal»
against «qt_kin_grey_pub» MPa), and the physics-anchored GP slightly better («qt_kin_anch_cal» against «qt_kin_anch_pub» MPa). After
tempering at 480-650 °C the strength is not very sensitive to how much of the structure was bainite, so these bars
can neither confirm nor reject the recalibration. We keep the recalibrated constants, which are better where they
matter most (Jominy bands, dilatometry, normalised alloy steel).

### 7.4 The five treatments, side by side

`fig36_five_heat_treatments` and `results/s09_five_treatments.csv` give the five treatments of 1020, 1045, 1080,
4140, 4340 and 5160 as 25 mm bars. They read like the textbook table they started from. Annealing gives coarse
pearlite and the softest state. Normalising gives finer pearlite and more strength, or bainite for the hardenable
grades (H3). Hardening gives martensite, up to «hv_q_1080» HV for 1080 and «hv_q_4140» HV for 4140. That state is too
brittle for a tensile test, so only its hardness is reported. Tempering at 600 °C brings 4140 to «uts_qt_4140» MPa with
«el_qt_4140» % elongation. A stress relief changes neither the microstructure nor the strength. Plain 1020 in a 25 mm bar
hardly hardens at all: its D_I is only «di_1020» mm, which is why it is case-hardened rather than through-hardened.

### 7.5 Stress relieving

A stress relief below Ac1 leaves the microstructure as it is. During the soak a residual stress cannot exceed
the yield strength at temperature, so plastic flow brings it down to k_y(T) × YS, with k_y from EN 1993-1-2
(0.47 at 600 °C), and creep lowers it further. For a normalised 1045 bar the bound is «sr_bound_1045» MPa after a soak at
600 °C. A tempered steel softens only if the soak tempers it further. By Hollomon-Jaffe additivity, a 4140 bar
tempered at 600 °C for 1 h («uts_qt_4140» MPa) keeps «uts_sr575_4140» MPa after a 575 °C stress relief but falls to «uts_sr650_4140» MPa after
one at 650 °C. That is the reason for the shop rule of relieving at least 25-30 °C below the tempering temperature (H8).

### 7.6 Precipitation strengthening, toughness and thermomechanically rolled steels

**Precipitation.** V and Nb that dissolve at the austenitising temperature (solubility products of Irvine et
al. 1967 and Narita 1975) re-precipitate in the ferrite. Ashby-Orowan strengthening then gives, for 0.10 % V,
about «ppt_v10_ac» MPa after air cooling (fine interphase precipitates) and «ppt_v10_fc» MPa after furnace cooling (coarsened),
but only about «ppt_nb03_ac» MPa for 0.03 % Nb air cooled. Little niobium dissolves at normalising temperatures,
so its main benefit in a normalised steel, a finer grain from undissolved Nb(C,N) pinning the austenite, is a
grain-size effect that we do not model. The training grades contain no microalloyed steel, so the term is an
estimate added only to steels above microalloy levels, reported with ±50 % and flagged.
**Toughness.** Pickering's impact transition temperature is given for ferrite-pearlite states. For
quenched-and-tempered states no Charpy value is given: toughness there depends on prior-austenite grain size,
impurities (P, Sn, Sb, As) and the cooling after tempering, none of which our data contain. Instead the tool
flags the two documented embrittlement windows, using Watanabe's J factor for temper embrittlement.
**Thermomechanically rolled steels.** Low-carbon chemistries with Nb, Ti or V and Mn ≥ 1 % are recognised as
TMCP/HSLA types. Their as-delivered strength comes from the rolling schedule, so the tool says it can only predict
them after re-austenitising. Boron steels are flagged because nothing in the kinetics contains boron's effect on
hardenability.

## 8. Every assumption, and how much it matters

### 8.1 The register

Every condition the models rely on is listed in `results/s10_assumptions.csv` (Table 7) with its status:
documented, fitted, from the literature, assumed or inferred.

**Table 7.** Assumptions, their status, and the range over which we tested them

«assumptions_table»

### 8.2 The bar size of the handbook's normalised data

Versions 1-2 inferred that the handbook's normalised values belong to 50 mm bars, because the physics-only
hardness error was lowest from 50 to 80 mm. The Birbilis database contains the same handbook values with the bar
size written out, and the match is unambiguous (Table 8, `fig38_assumption_sensitivity`):

**Table 8.** Birbilis normalised bars against the handbook's normalised values, same grades

«section_match_table»

The handbook values are the 25 mm (1 in.) bars. The physics-only scan had pointed to 50 mm because the physics
chain is too hard at 1 K/s: it over-predicts the hardness of 25 mm normalised bars, mostly through too much bainite in
the alloy grades, and a slower (larger) bar hid that. The scan was measuring the bias of the physics, not the bar size. We now train on the
documented size and let the GP correction absorb the bias, which is what it is for.

### 8.3 Retraining with each assumption changed

For the annealing/normalising model every inferred or assumed condition was moved to the ends of its plausible
range, the production model retrained and re-scored on all annealed and normalised rows, and the furnace-vs-air answer
recomputed for five grades as 25 mm bars (Table 9). The bar-size rows are counterfactuals now that the size is
documented; they show what the old inference cost.

**Table 9.** One assumption at a time: leave-one-grade-out RMSE and the UTS gain on air cooling (25 mm bar)

«oat_table»

«oat_sentence»

**Table 10.** The quench-and-temper physics (grey box, leave-one-grade-out UTS RMSE) with one assumption changed

«qt_oat_table»

«qt_oat_sentence»

### 8.4 Global sensitivity

The Morris elementary-effects screen varies all the physics assumptions together («morris_r» trajectories, eight factors) with
the fitted constants held fixed, so it measures how sensitive a finished model is to each assumption (Tables 9 and 10
measured how much a refit absorbs). For the physics air-cooling gain of 1045 the largest effect is «morris_top_1045»; for the
strength of a tempered 50 mm 4140 bar, «morris_top_qt», which a refit of the tempering constants absorbs almost entirely
(Table 10); for that bar's martensite fraction, «morris_top_fm» (`fig39_morris_sensitivity`).

### 8.5 A test across bar sizes

The Birbilis normalised bars also test the furnace-vs-air model where it was never trained: other bar sizes of the
handbook grades, and grades it never saw (`fig42_normalising_bar_size`, `results/s06_bar_sizes_metrics.csv`).

«bar_table»

«bar_sentence»

## 9. How uncertain is the transformation model?

Stage 8 summarised the kinetic calibration with a Laplace approximation, which gave a posterior standard
deviation of 1000 for the ferrite factor whose prior standard deviation is 1. That is impossible for a proper
posterior. The cause is that the likelihood is not smooth (a reaction either starts within a time step or it
does not), so the numerical Hessian fails. We therefore sampled the posterior with four adaptive Metropolis chains
(Haario et al. 2001) from overdispersed starts (`pipeline/s11_kinetic_uncertainty.py`, `fig40_kinetic_posterior`). All
chains converged: split R-hat at most «rhat_max» and effective sample size at least «ess_min» for every constant.

**Table 11.** Posterior of the kinetic constants on the SAE H-bands

«posterior_table»

The bands pin the bainite constants tightly, leave the ferrite factor almost where the prior put it (they
cannot tell how fast ferrite forms), and prefer a pearlite factor of «post_fp» («post_fp_lo»-«post_fp_hi») to the published 1.0 at
which the stage-8 search had stopped. Production keeps 1.0 for now, because changing it would require
re-running every transfer check of stage 8. We report it as the next step.

**Does the parameter uncertainty explain the errors?** 200 posterior draws were pushed through the kinetics to
the 423 measured points of the CCT atlas, which played no part in the calibration (Table 12, `fig41_kinetic_uncertainty_propagation`).

**Table 12.** Coverage of the independent CCT-atlas measurements by the propagated 90 % bands

«coverage_table»

The answer is no. The credible bands are narrow and contain only «cov_fp» of the measured ferrite + pearlite fractions and
«cov_b» of the bainite fractions. The errors come from the form of the equations, not from the uncertainty of their
constants: a structural error of about «s_fp» in phase fraction (and «s_hv» HV) has to be added before the bands cover
90 % of the points. Two consequences follow. More calibration data of the same kind would not fix the transformation
model; a better model form would. And any uncertainty we report for a microstructure must include the structural
term, which the tool now does: every predicted phase fraction carries a band of ±«frac_band» at 90 %.

## 10. Using the model

**Recommending a real grade and route.** For a specification on the lower 90 % bound the tool ranks every in-scope
AISI/SAE grade and cooling option inside the domain by weldability (`results/s07_recommendations.csv`). For YS ≥ 300 MPa
with El ≥ 28 % it proposes air-cooled plain carbon steels of about 0.12-0.20 % C; for UTS ≥ 950 MPa with El ≥ 12 % without
quench and temper, air-cooled 4137H-4142H. Only standard grades are searched (`fig22_grade_front_fc_vs_ac`); an earlier
version optimised free compositions, which is the kind of inverse design this project set out to avoid.

**Any of the five treatments.** `SteelPredictor.heat_treat(comp, treatment, ...)` returns the microstructure and
properties with 90 % intervals for 'anneal', 'normalise', 'harden', 'quench_temper' or 'stress_relieve', with the bar
size, quench medium, tempering and stress-relief temperatures as arguments. `compare_treatments(comp)` gives the five
side by side, which is Table 4 for any steel. The interactive tool has a tab for it.

**Learning efficiently.** Adding the grade the model is least sure about learns faster than adding a random one
(`fig23_active_learning`), and a few measured members of a new product remove most of the bias of Section 5.6
(`fig24_few_shot_recalibration`).

## 11. Limitations

* **The training data are small and old:** 33 handbook grades (annealing, normalising) and «qt_grades» grades (quench and
  temper), with typical values at nominal compositions. The composition spread within a grade is already a large
  part of the error, and new grades are where the models can fail.
* **One inferred condition remains:** the furnace cooling rate of the handbook's annealed bars (0.005 K/s) is not
  documented anywhere; Section 8.3 shows how much it matters. The tempering time of the Q+T bars (1 h) is assumed.
* **Semi-empirical kinetics with a structural error:** the transformation model's errors on independent data are
  structural (Section 9); it over-predicts the hardenability measured by Grossmann's D_I, most for high-carbon
  steels (Section 7.1), and the bainite calibration holds for H-steel chemistries, not for steels as rich as SA-540.
* **The quench is Grossmann's:** one constant heat-transfer coefficient per bath, no boiling stages, no latent heat of
  transformation; long bars only (no plates, no end effects). The radial martensite profile of a moderately agitated
  oil quench therefore carries the bainite kinetics' uncertainty.
* **Estimated, not trained:** precipitation strengthening (±50 %), uniform elongation, ferrite grain size (factor of 2)
  and the impact transition temperature; as-quenched tensile properties are not given.
* **Out of scope, and said so:** Charpy values of quenched-and-tempered steel, as-delivered TMCP products, boron's
  effect on hardenability, carburising, decarburisation, banding, stainless, tool and maraging steels.
* **Not fully blind:** several design choices were made after seeing evaluation results (Appendix B).
* **Nothing here replaces a tensile test.**

## 12. Pre-registered validation protocol

The study includes a validation protocol written before any test (`results/s07_lab_plan_preregistered.csv`):
predictions with 90 % intervals for common Indian grades (EN3B, EN8, EN9, EN18, EN19, EN24, EN42 and an 8620-type),
furnace cooled and air cooled as 10 and 25 mm bars. The protocol fixes how they are to be tested: spark-OES chemistry
of the bars actually bought, with the predictions re-issued for it; a thermocouple in each bar to record the real
furnace and air-cooling curves; tensile tests with an extensometer so that uniform elongation is measured; Brinell or
Vickers hardness after removing the decarburised layer; point counting (ASTM E562), grain size (ASTM E112) and SEM
pearlite spacing on a few samples; hardness and microstructure only for as-cooled martensite. Because the predictions
are fixed beforehand, any laboratory can use them as a genuine test of the model.

## 13. How to reproduce

```bash
make setup            # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
make pipeline         # stages 1-11, about 35 minutes on a laptop
make notebooks        # executed presentation notebooks
make test             # physics, data and model tests
make app              # interactive what-if tool at http://127.0.0.1:7860
.venv/bin/python reports/build_report.py        # this report from results/
.venv/bin/python reports/deck/build_deck.py     # the slide deck (PDF and PPTX)
```

## Appendix B. Constants we chose after looking at data

| Choice | Value | Evidence or reason |
|---|---|---|
| Bar size of the handbook normalised data | 25 mm (1 in.) round in still air | documented: same values as the 25 mm rows of Birbilis et al. 2025 (v2 inferred 50 mm) |
| Effective furnace cooling of the handbook annealed bars | 0.005 K/s | error flat below 0.01 K/s; usual full-annealing practice |
| Bainite factor and Bs shift | «fB», «dBs» °C | Bayesian calibration on the H-bands (stage 8) |
| Jominy water-jet coefficient | «h_q» W m⁻² K⁻¹ | fitted to the ASM chart cooling rates |
| Pearlite spacing law | 3.33 ΔT^−0.607 µm | 14 measured spacings, 4 papers |
| Ferrite grain size | Hodgson-Gibbs low-C_eq branch, C_eq ≤ 0.35 | removes the 3 µm floor; checked on 20 grains |
| Default austenitising temperatures | anneal max(Ae3 + 50, Ae1 + 65); normalise max(upper + 110, 870) | fitted to the handbook practice |
| Phase-fraction report | lever split with the kinetic split as a band | measured normalised fractions |
| 11xx grades treated as fine-grained | against the handbook footnote | a coarse-grain assumption made normalised 1137-1144 bainitic, which their hardness contradicts; open |
| Yield ratio of untempered bainite/martensite in the textbook baseline | 0.75 | assumption |
| Uniform / total elongation | «ue_ratio» ± «ue_ratio_sd» | 10 literature tests on one gauge basis |
| Gate of the data-driven member | weight 1/3, fading over 0.2 excess hard phase | gate ablation (stage 4) |
| Physics anchor | Gladman (F+P) + Maynier (B+M) | anchor ablation (stage 4) |
| Silicon and vanadium not ML inputs | | silicon ablation (stage 4) |
| Composition window of the thermodynamics | C ≤ 1.05, Si ≤ 1.0, Mn ≤ 2, Ni ≤ 4, Cr ≤ 2, Mo ≤ 1 | known failures of the equations |
| Default hardening temperature (v3) | max(Ae3 + 70, Ae1 + 90) | fitted to 349 Birbilis quench temperatures, MAE 16 °C |
| Quench heat capacity (v3) | Eurocode polynomial capped at 650 J/kg/K | the 735 °C peak made a quenched surface softer than the centre |
| Conductivity for H → h (v3) | 30 W/m/K | hot austenite; 25-40 tested (Section 8.3) |
| Tensile specimen position (v3) | centre ≤ 38 mm, mid-radius above | ASTM A370 |
| Cap on as-quenched martensite hardness (v3) | 865 HV (66 HRC) | retained austenite above about 0.6 % C (Krauss 2015) |
| Tempering constants (v3) | Table 5 | fitted on «qt_n» Q+T bars, priors in `physics/tempering.py` |
| Q+T production model (v3) | mixture of physics-anchored GP and data GP | best rung for UTS, YS and El (Table 6) |
| Precipitation particle sizes (v3) | 8 nm air cooled, 30 nm furnace cooled | assumption (Gladman 1997); ±50 % reported |

## Appendix A. The equations as implemented

Compositions in wt%, temperatures in °C unless stated, lengths as given. Each relation is the published one;
where we changed or limited it, the change is stated.

**A.1 Critical temperatures** (Grange 1961, converted from °F; `physics/thermo.py`)

Ae3 (°F) = 1570 − 323 C − 25 Mn + 80 Si − 3 Cr − 32 Ni
Ae1 (°F) = 1333 − 25 Mn + 40 Si + 42 Cr − 26 Ni
Acm (°C) = 224.4 + 992.4 C − 465.1 C² + 46.7 Cr + 19.0 C·Cr − 6.1 Cr² + 7.6 Mn + 10.0 Mo − 6.8 Cr·Mo − 6.9 Ni
          + 3.7 C·Ni − 2.7 Cr·Ni + 0.8 Ni² + 16.7 Si   (the fit used by Collins et al. 2023 for the Hultgren condition)

The eutectoid point is the intersection of Ae3 and Acm. Above the eutectoid the upper critical temperature used
for the default normalising temperature is the cementite solvus drawn from that point with the slope of the binary
Fe-Fe₃C diagram, (1147 − 727)/(2.14 − 0.76) ≈ 304 °C per % C, because the Acm fit falls below the real solvus
above about 0.9 % C. The equations are flagged outside C ≤ 1.05, Mn ≤ 2, Si ≤ 1, Ni ≤ 4, Cr ≤ 2, Mo ≤ 1.

Lever rule just below the eutectoid: f_P = (C − 0.02)/(C_e − 0.02) for hypoeutectoid steels.

Default austenitising temperatures (fitted to the practice of the handbook data):
full annealing max(Ae3 + 50, Ae1 + 65); normalising max(upper critical + 110, 870).

**A.2 Martensite and bainite start** (`physics/thermo.py`, `physics/kinetics.py`)

Ms = 561 − 474 C − 33 Mn − 17 Ni − 17 Cr − 21 Mo − 7.5 Si + 10 Co   (Steven & Haynes 1956; Si, Co terms as in Collins et al.)
Bs = 637 − 58 C − 35 Mn − 15 Ni − 34 Cr − 41 Mo + ΔBs   (Li et al. 1998; ΔBs = −107 °C from the H-band calibration)
Martensite: f_M = 1 − exp[−0.011 (Ms − T)]   (Koistinen & Marburger 1959)

**A.3 Transformation kinetics** (Kirkaldy & Venugopalan 1984; Li et al. 1998; `physics/kinetics.py`)

τ_j(X, T) = S(X) / k_j(T),   S(X) = ∫₀ˣ dx / [x^{0.4(1−x)} (1 − x)^{0.4x}]
k_F = f_F 2^{0.41G} (Ae3 − T)³ exp(−Q/RT) / FC
k_P = f_P 2^{0.32G} (Ae1 − T)³ exp(−Q/RT) / PC
k_B = f_B 2^{0.29G} (Bs − T)² exp(−Q/RT) / BC
FC = exp(1.00 + 6.31C + 1.78Mn + 0.31Si + 1.12Ni + 2.70Cr + 4.06Mo)
PC = exp(−4.25 + 4.12C + 4.36Mn + 0.44Si + 1.71Ni + 3.33Cr + 5.19√Mo)
BC = exp(−10.23 + 10.18C + 0.85Mn + 0.55Ni + 0.90Cr + 0.36Mo)
Q = 27 500 cal/mol, R = 1.987 cal/(mol K), G = ASTM prior-austenite grain-size number.
f_F = f_P = 1 (published); f_B = 79.4 (calibrated on the SAE H-bands, stage 8).
Additivity (Scheil) in S-space; ferrite bounded by the extrapolated Ae3 (para-equilibrium) lever rule; pearlite may start
once the enriched austenite is below its own Ae3 and Acm (Hultgren condition).

Prior-austenite grain size after about 1 h at T: G = 8 − max(T − 850, 0)/100, clipped to 4-9 (fine-grained, Al-killed).

**A.4 Cooling paths** (`physics/cooling.py`, `physics/jominy.py`)

Furnace: T(t) = T∞ + (T₀ − T∞) exp(−t/τ), τ set by the rate at 700 °C (0.005 K/s).
Still air, round bar of diameter D (lumped, Bi < 0.1 for D ≤ 100 mm):
ρ c(T) dT/dt = −(4/D) [h_c (T − T∞) + ε σ (T⁴ − T∞⁴)],  h_c = 1.32 (ΔT/D)^{1/4},  ε = 0.8,
c(T) and the conductivity k(T) from EN 1993-1-2 (k = 54 − 0.0333 T below 800 °C, 27.3 above).
Jominy bar (25.4 × 101.6 mm): 1-D conduction ρ c ∂T/∂t = ∂/∂x(k ∂T/∂x) − (4/D) q_air, water jet q = h_q (T − 24),
h_q ≈ 8000 W m⁻² K⁻¹ fitted to the cooling-rate scale of the ASM combined hardenability charts.

**A.5 Microstructure** (`physics/microstructure.py`)

Pearlite spacing S₀ = 3.33 ΔT^{−0.607} µm, ΔT = Ae1 − mean pearlite formation temperature (fitted, 14 spacings, 4 papers).
Ferrite grain size (Hodgson & Gibbs 1992, low-C_eq branch, C_eq = C + Mn/6 held at 0.35 at most):
d_α = (−0.4 + 6.37 C_eq) + (24.2 − 59.0 C_eq) Ṫ^{−1/2} + 22 [1 − exp(−0.015 d_γ)]  µm, Ṫ ≥ 0.1 K/s, clipped to 3-60 µm.

**A.6 Structure-property relations** (`physics/properties.py`)

Gladman, McIvor & Pickering (1972), with f_α^{1/3} weighting (d and S₀ in mm, N_f = 0.002):
YS  = f_α^{1/3} (35.4 + 58.5 Mn + 17.4 d^{−1/2}) + (1 − f_α^{1/3}) (178.9 + 3.9 S₀^{−1/2}) + 63 Si + 425 √N_f
UTS = f_α^{1/3} (246 + 1143 √N_f + 18.1 d^{−1/2}) + (1 − f_α^{1/3}) (719 + 3.5 S₀^{−1/2}) + 95 Si
Maynier, Jungmann & Dollet (1978), V_r = cooling rate at 700 °C in °C/h:
HV_M  = 127 + 949C + 27Si + 11Mn + 8Ni + 16Cr + 21 log V_r
HV_B  = −323 + 185C + 330Si + 153Mn + 65Ni + 144Cr + 191Mo + (89 + 53C − 55Si − 22Mn − 10Ni − 20Cr − 33Mo) log V_r
HV_FP = 42 + 223C + 53Si + 30Mn + 12.6Ni + 7Cr + 19Mo + (10 − 19Si + 4Ni + 8Cr + 130V) log V_r
Physics strength used as the anchor: UTS_phys = f_FP · UTS(Gladman) + f_BM · 3.3 HV(Maynier, B and M).
HB ≈ 0.95 HV (ASTM E140 is close to this below about 450 HV); HRC from HV by the E140 table.
Impact transition temperature (Pickering 1978): ITT = −19 + 44 Si + 700 √N_f + 2.2 (% pearlite) − 11.5 d^{−1/2}.
Ideal critical diameter (Grossmann 1942; ASTM A255 factors, grain size 7): D_I = D_I,base(C) · Π f_i.
Uniform elongation e_u = 0.508 × total elongation on the 4.5√S₀ gauge; Oliver's conversion between gauges
A₂ = A₁ (k₁/k₂)^{0.4}, k = L₀/√S₀; n = ln(1 + e_u).

**A.7 Uncertainty**

Normalised split-conformal intervals: half-width = q · σ_model(x), q the ⌈(1 − α)(n + 1)⌉/n quantile of |y − μ|/σ over
the leave-one-grade-out residuals (α = 0.1).

**A.8 Quench, tempering, stress relief and precipitation (v3)** (`physics/quench.py`, `tempering.py`,
`stress_relief.py`, `precipitation.py`)

Quench: ρ c(T) ∂T/∂t = (1/r) ∂/∂r (k(T) r ∂T/∂r), surface −k ∂T/∂r = h (T − T_bath), h = 2 k_ref H, k_ref = 30 W m⁻¹ K⁻¹,
H from the Grossmann table (oil 0.25-1.1, water 0.9-4.0, brine 2.0-5.0 per inch by agitation); ideal quench T_s = T_bath.
Hollomon-Jaffe: P = T_K (20 + log₁₀ t_h). Additivity: a first treatment counts as t_eq = 10^(P₁/T₂,K − 20) h at T₂.
Tempered martensite: HV_M(P) = HV_soft + (HV_M0 − HV_soft) / (1 + exp((P − P₅₀)/W)) + (a_Mo Mo + a_V V + a_Cr Cr)
exp(−((P − 17500)/1500)²); P₅₀ = p₀ + p_Mn Mn + p_Si Si + p_Cr Cr + p_Mo Mo + p_V V; HV_soft = s₀ + s_C C;
HV_M0 = min(Maynier HV_M, 865). Bainite: HV_B(P) = min(HV_B0, HV_M(P)); ferrite + pearlite unchanged. UTS = 3.3 HV.
Stress relief: subcritical if T ≤ Ac1(Andrews) − 25 °C; residual stress ≤ k_y(T) YS, k_y (EN 1993-1-2 Table 3.1) =
1.0 to 400 °C, 0.78 at 500, 0.47 at 600, 0.23 at 700 °C.
Precipitation: log[V][C] = 6.72 − 9500/T, log[V][N] = 3.63 − 8700/T, log[Nb][C + 12/14 N] = 2.26 − 6770/T (wt %, K);
Δσ = (5.9 √f / x) ln(x / 2.5×10⁻⁴) MPa, x in µm (Gladman 1999).
Temper embrittlement index: J = (Mn + Si)(P + Sn) × 10⁴, Sn = 0.01 % assumed.

## References

1. Li, M.V., Niebuhr, D.V., Meekisho, L.L., Atteridge, D.G. (1998). A computational model for the prediction of steel hardenability. *Metall. Mater. Trans. B* 29, 661-672. doi:10.1007/s11663-998-0101-3
2. Kirkaldy, J.S., Venugopalan, D. (1984). Prediction of microstructure and hardenability in low alloy steels. In *Phase Transformations in Ferrous Alloys* (Marder, Goldstein, eds.), TMS-AIME, 125-148.
3. Collins, J., Piemonte, M., Taylor, M., Fellowes, J., Pickering, E. (2023). A rapid, open-source CCT predictor for low-alloy steels, and its application to compositionally heterogeneous material. *Metals* 13, 1168. doi:10.3390/met13071168; data doi:10.5281/zenodo.7770264
4. Gladman, T., McIvor, I.D., Pickering, F.B. (1972). Some aspects of the structure-property relationships in high-carbon ferrite-pearlite steels. *J. Iron Steel Inst.* 210, 916-930.
5. Pickering, F.B. (1978). *Physical Metallurgy and the Design of Steels.* Applied Science, London.
6. Maynier, P., Jungmann, B., Dollet, J. (1978). In *Hardenability Concepts with Applications to Steel* (Doane, Kirkaldy, eds.), TMS-AIME, 518-545 (title and author list as cited by later work; not checked against the volume).
7. Grange, R.A. (1961). Estimating critical ranges in heat treatment of steels. *Metal Progress* 79, 73.
8. Andrews, K.W. (1965). Empirical formulae for the calculation of some transformation temperatures. *J. Iron Steel Inst.* 203, 721-727.
9. Steven, W., Haynes, A.G. (1956). The temperature of formation of martensite and bainite in low-alloy steels. *J. Iron Steel Inst.* 183, 349-359.
10. Koistinen, D.P., Marburger, R.E. (1959). A general equation prescribing the extent of the austenite-martensite transformation. *Acta Metall.* 7, 59-60.
11. Hodgson, P.D., Gibbs, R.K. (1992). A mathematical model to predict the mechanical properties of hot rolled C-Mn and microalloyed steels. *ISIJ Int.* 32, 1329-1338. doi:10.2355/isijinternational.32.1329
12. Marder, A.R., Bramfitt, B.L. (1976). The effect of morphology on the strength of pearlite. *Metall. Trans. A* 7, 365-372.
13. Zener, C. (1946). Kinetics of the decomposition of austenite. *Trans. AIME* 167, 550-595.
14. Hultgren, A. (1947). *Trans. ASM* 39, 915 (isothermal transformation of austenite).
15. Grossmann, M.A. (1942). Hardenability calculated from chemical composition. *Trans. AIME* 150, 227-259.
16. ASTM A255. Standard Test Methods for Determining Hardenability of Steel. ASTM International.
17. Hollomon, J.H. (1945). Tensile deformation. *Trans. AIME* 162, 268-290.
18. Morrison, W.B. (1966). The effect of grain size on the stress-strain relationship in low-carbon steel. *Trans. ASM* 59, 824-846.
19. Vander Voort, G.F., Roósz, A. (1984). Measurement of the interlamellar spacing of pearlite. *Metallography* 17, 1-17. doi:10.1016/0026-0800(84)90002-8
20. Krauss, G. (2015). *Steels: Processing, Structure, and Performance*, 2nd ed. ASM International.
21. Bhadeshia, H.K.D.H., Honeycombe, R.W.K. (2017). *Steels: Microstructure and Properties*, 4th ed. Butterworth-Heinemann.
22. Saunders, N., Guo, Z., Li, X., Miodownik, A.P., Schillé, J.-Ph. (2003). Using JMatPro to model materials properties and behavior. *JOM* 55(12), 60-65. doi:10.1007/s11837-003-0013-2
23. Bhadeshia, H.K.D.H. (1999). Neural networks in materials science. *ISIJ Int.* 39, 966-979. doi:10.2355/isijinternational.39.966
24. Trzaska, J., Dobrzański, L.A. (2007). Modelling of CCT diagrams for engineering and constructional steels. *J. Mater. Process. Technol.* 192-193, 504-510. doi:10.1016/j.jmatprotec.2007.04.099
25. Huang, X. et al. (2020). A combined machine learning model for the prediction of time-temperature-transformation diagrams of high-alloy steels. *J. Alloys Compd.* 823, 153694. doi:10.1016/j.jallcom.2020.153694
26. Hedström, P. et al. (2026). Physics-informed machine learning for steel development: a computational framework and CCT diagram modeling. *Metall. Mater. Trans. A* 57, 2816-2832. doi:10.1007/s11661-026-08181-4
27. Cao, Y. et al. (2023). Modeling continuous cooling transformations for HSLA steels with physical metallurgy guided hereditary machine learning. *Metall. Mater. Trans. A* 54, 4891-4904. doi:10.1007/s11661-023-07210-w
28. Shen, C. et al. (2019). Physical metallurgy-guided machine learning and artificial intelligent design of ultrahigh-strength stainless steel. *Acta Mater.* 179, 201-214. doi:10.1016/j.actamat.2019.08.033
29. Agrawal, A. et al. (2014). Exploration of data science techniques to predict fatigue strength of steel from composition and processing parameters. *Integr. Mater. Manuf. Innov.* 3, 8. doi:10.1186/2193-9772-3-8
30. Xiong, J., Zhang, T.-Y., Shi, S.-Q. (2020). Machine learning of mechanical properties of steels. *Sci. China Technol. Sci.* 63, 1247-1255. doi:10.1007/s11431-020-1599-5
31. Kennedy, M.C., O'Hagan, A. (2001). Bayesian calibration of computer models. *J. R. Stat. Soc. B* 63, 425-464. doi:10.1111/1467-9868.00294
32. Meredig, B. et al. (2018). Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery. *Mol. Syst. Des. Eng.* 3, 819-825. doi:10.1039/C8ME00012C
33. Sutton, C. et al. (2020). Identifying domains of applicability of machine learning models for materials science. *Nat. Commun.* 11, 4428. doi:10.1038/s41467-020-17112-9
34. Lookman, T., Balachandran, P.V., Xue, D., Yuan, R. (2019). Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design. *npj Comput. Mater.* 5, 21. doi:10.1038/s41524-019-0153-8
35. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J., Wasserman, L. (2018). Distribution-free predictive inference for regression. *J. Am. Stat. Assoc.* 113, 1094-1111.
36. Rasmussen, C.E., Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
37. Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS* 30.
38. Smokvina Hanza, S., Marohnić, T., Iljkić, D., Basan, R. (2021). Artificial neural networks-based prediction of hardness of low-alloy steels using specific Jominy distance. *Metals* 11, 714. doi:10.3390/met11050714 (CC BY 4.0; data from Rose & Hougardy's atlas)
39. Marohnić, T., Smokvina Hanza, S., Iljkić, D., Basan, R. (2025). Possibilities of using specific Jominy distance in ANN models for predicting low-alloy steels' microstructure. *Materials* 18, 564. doi:10.3390/ma18030564 (CC BY 4.0)
40. Metallus (2024). *Practical Data for Metallurgists*, 19th ed., End-Quench Hardenability Bands (pp. 49-59) and combined hardenability charts (p. 68).
41. Carvajal, L., Artigas, A., Monsalve, A., Arévalo, E. (2017). Monitoring heat treatments in steels by a non destructive ultrasonic method. *Mater. Res.* 20 (Suppl. 2), 347-352. doi:10.1590/1980-5373-MR-2016-1083 (CC BY 4.0; micrographs)
42. Kingsley et al. (2026). Experimental test results on reused steel members. Zenodo, doi:10.5281/zenodo.19000367.
43. Volkov, A. et al. (2026). SteelBench v1.0. Zenodo, doi:10.5281/zenodo.18530558.
44. Carbon and Stainless Steel Grades (2025). Zenodo, doi:10.5281/zenodo.17704410.
45. ASM International (1990, 1991). *ASM Handbook* Vol. 1 (Properties and Selection: Irons, Steels and High-Performance Alloys) and Vol. 4 (Heat Treating).
46. EN 1993-1-2 (2005). Eurocode 3: Design of steel structures, Part 1-2 (thermal properties of carbon steel).
47. ISO 2566-1. Steel - Conversion of elongation values - Part 1: Carbon and low alloy steels.
48. ASTM E112 (grain size), E562 (point counting), E140 (hardness conversion), E8 (tension testing).
49. The 52 open-access papers of the literature compilation are listed with DOIs and licences in `data/raw/literature_sources.md`.
50. Birbilis, N., Ghorbani, M., Zhao, S. (2025). A database of mechanical properties of steels. Mendeley Data, V1, doi:10.17632/jmwb9ddd43.1 (CC BY 4.0).
51. Grossmann, M.A., Asimow, M., Urban, S.F. (1939). Hardenability, its relation to quenching, and some quantitative data. In *Hardenability of Alloy Steels*, ASM, Cleveland, 124-196.
52. Hollomon, J.H., Jaffe, L.D. (1945). Time-temperature relations in tempering steel. *Trans. AIME* 162, 223-249.
53. Grange, R.A., Hribal, C.R., Porter, L.F. (1977). Hardness of tempered martensite in carbon and low-alloy steels. *Metall. Trans. A* 8, 1775-1785. doi:10.1007/BF02646882
54. Gladman, T. (1997). *The Physical Metallurgy of Microalloyed Steels.* Institute of Materials, London.
55. Gladman, T. (1999). Precipitation hardening in metals. *Mater. Sci. Technol.* 15, 30-36.
56. Irvine, K.J., Pickering, F.B., Gladman, T. (1967). Grain-refined C-Mn steels. *J. Iron Steel Inst.* 205, 161-182.
57. Narita, K. (1975). Physical chemistry of the groups IVa (Ti, Zr), Va (V, Nb, Ta) and the rare earth elements in steel. *Trans. ISIJ* 15, 145-152.
58. Watanabe, J., Shindo, Y., Murakami, Y., Adachi, T., Ajiki, S., Miyano, K. (1974). Temper embrittlement of 2¼Cr-1Mo pressure vessel steel. ASME 29th Petroleum Mechanical Engineering Conference, Dallas.
59. ASTM A370. Standard Test Methods and Definitions for Mechanical Testing of Steel Products. ASTM International.
60. Haario, H., Saksman, E., Tamminen, J. (2001). An adaptive Metropolis algorithm. *Bernoulli* 7, 223-242.
61. Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., Rubin, D.B. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
62. Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics* 33, 161-174.
63. Hecht, M.D., DeCost, B.L., Francis, T., Holm, E.A., Picard, Y.N., Webler, B.A. Ultrahigh carbon steel micrographs. NIST Materials Data Repository (CC BY 3.0 US); DeCost, B.L. et al. (2017) *Integr. Mater. Manuf. Innov.* 6, 197-205.
