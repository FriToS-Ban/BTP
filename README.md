# Classify, Heat-Treat, Predict: Machine Learning for Steels

**B.Tech Project (MTP-400A), Department of Metallurgical and Materials Engineering, IIT Roorkee**
**Authors:** Aviral Singh, Shibasish Banerjee  |  **Supervisor:** Dr. Anish Karmakar

**Report:** [`reports/submission/report.pdf`](reports/submission/report.pdf)  |  **Live demo:** `make demo`

---

## What this project does

You give it the composition of a carbon or low-alloy steel, a heat treatment and the size of the bar, and it tells
you what kind of steel it is, what microstructure forms and what yield strength, tensile strength, elongation and
hardness to expect, each with a 90 % interval. It covers the 5 classic treatments, annealing, normalising,
hardening, tempering and stress relieving, so one model can answer the everyday question of a heat-treatment shop,
which is simply what this steel will be like after this treatment.

The work started from a question set by our supervisor, how furnace cooling and air cooling change a steel, and
answering it properly needed the whole chain from the cooling of the bar to the strength of the final
microstructure. Once that chain was in place we extended it to the rest of the route.

| Treatment | What the model predicts | How it is tested (on data the models never saw) |
|---|---|---|
| **Annealing** (furnace cooling) | phases, pearlite spacing, grain size; YS, UTS, elongation, hardness with 90 % intervals | handbook grades left out (UTS within 47 MPa); annealed bars of 7 new grades (20 MPa) |
| **Normalising** (still air, any bar size) | the same, including bainite in hardenable steels | handbook grades left out; 114 bars at other sizes or of new grades (68 MPa); 215 literature values |
| **Hardening** (oil, water or brine quench) | phases and as-quenched hardness through the bar | SAE Jominy bands of 80 H-steels (5.9 HRC) |
| **Tempering** (any temperature and time) | YS, UTS, elongation with 90 % intervals, hardness, embrittlement warnings | 324 quenched and tempered bars of 28 grades, left out by grade (UTS within 75 MPa) |
| **Stress relieving** | subcritical check, residual-stress limit, softening from extra tempering | built on EN 1993-1-2 and Hollomon–Jaffe additivity |

## How the model works

The model is a chain that follows the metallurgy step by step. The physics carries every prediction, and machine
learning is used only to correct what the textbook relations get wrong:

```
composition ─► 1. XGBoost classifier ─► steel family (13 families, or "unknown" for steels outside the scope)
     ├─► 2. thermodynamics ─► Ae1, Ae3, Acm, Ms, lever rule, austenitising temperature
treatment + bar size ─► 3. heat flow ─► cooling path (furnace, still air, or a radial quench with Grossmann H)
     └─► 4. transformation kinetics (Li et al. 1998, Bayesian-calibrated bainite)
                 ─► ferrite, pearlite, bainite, martensite, pearlite spacing, grain size
                 └─► tempering and stress relief (Hollomon–Jaffe, with secondary hardening)
          5. physics-anchored Gaussian-process models + conformal 90 % intervals
                 ─► YS, UTS, elongation, hardness
```

1. **Classify.** An XGBoost classifier, trained on heats sampled from 547 SAE composition ranges, names the steel family, and a novelty
   score refuses steels it was not built for, such as stainless, tool and maraging steels.
2. **Thermodynamics.** Empirical equations (Grange for Ae1 and Ae3, Steven–Haynes for Ms) give the critical temperatures.
3. **Cooling.** A heat balance gives the cooling curve of the bar in a furnace, in still air or in a quench, so a
   thick bar cools more slowly than a thin one.
4. **Transformation.** The Kirkaldy–Venugopalan kinetics of Li et al. turn the cooling curve into phase fractions.
   We calibrated the bainite reaction on 1040 points of the SAE hardenability bands with an MCMC posterior.
5. **Properties.** The Gladman–McIvor–Pickering and Maynier relations give a physics estimate of the strength from
   the microstructure, and a Gaussian process learns the correction to it. Far from the training data the
   correction fades and the prediction falls back to the physics. Split-conformal calibration gives every
   prediction a 90 % interval that holds on unseen grades.

## Key results

Every number is in `results/`, and every error is measured with leave-one-grade-out, so the grade being predicted
was never in training.

| Question | What we found |
|---|---|
| Which steel is it? | 13 families; 0.94 family accuracy on 1358 real heats, against 0.92 for the AISI/SAE composition rules |
| Which microstructure forms? | ferrite start within 16 °C and Ms within 7 °C of dilatometry; 423 CCT-atlas points: ferrite + pearlite fraction within 0.12 |
| Hardenability | Bayesian calibration of the bainite reaction cut the Jominy band error on unseen grades from 10.1 to 5.9 HRC |
| Annealed and normalised properties | UTS 47 MPa, YS 44 MPa, hardness 16 HB (RMSE), against 94, 100 and 23 for the textbook relations; 92.5 % of values inside the 90 % interval |
| Quenched and tempered properties | 324 bars of 28 grades at 13–100 mm: UTS 75 MPa, YS 85 MPa, elongation 1.7 % (textbook chain: 247 MPa); 89 % inside the 90 % interval |
| Furnace vs air cooling | low-carbon steels: almost the same steel (+37 MPa); medium and high carbon: same phases, finer pearlite (+79 and +251 MPa); Cr-Mo and Ni-Cr-Mo steels: partly bainitic in air, so the bar size matters as much as the route |
| Bar size of the handbook data | the handbook's normalised values match 25 mm bars of an independent database (mean difference 0.4 MPa) |

## Quick start

```bash
make setup        # virtual environment + requirements (Python 3.10+, tested on 3.13)
make pipeline     # stages 1-11, about 25 minutes; writes results/ and figures/
make test         # physics, data and model tests
make demo         # one-page live demo: http://127.0.0.1:8050
make app          # full interactive tool
make notebooks    # 11 executed notebooks
```

`data/raw/` is committed, so the pipeline runs offline. `make fetch` downloads every source again and records
checksums in `data/raw/source_manifest.csv`.

Using the model from Python:

```python
import sys; sys.path.insert(0, "src")
from steelml.predictor import SteelPredictor

sp = SteelPredictor.load()
comp = {"C": 0.42, "Mn": 0.75, "Si": 0.25, "Cr": 1.05, "Mo": 0.22}        # an AISI 4140-type steel

sp.what_if(comp, section_mm=25)                                          # furnace vs air cooling
sp.heat_treat(comp, "quench_temper", section_mm=30, medium="oil", temper_T=550)
sp.compare_treatments(comp, section_mm=30)                               # all five treatments side by side
```

## Repository structure

```
BTP/
├── src/steelml/             the model, as a Python package
│   ├── physics/             thermodynamics, cooling, quenching, transformation kinetics, Jominy, Bayesian
│   │                        calibration, tempering, stress relief, precipitation, structure-property laws
│   ├── models/              steel classifier, property models, quenched-and-tempered models, domain checks
│   ├── data/                source registry, parsers, grade catalogue, dataset assembly
│   ├── features.py          composition + treatment -> physics descriptors
│   └── predictor.py         end-to-end predictor: what_if, heat_treat, compare_treatments
├── pipeline/                s00 fetch ... s11 kinetic uncertainty; run_all.py runs everything
├── app/                     app.py (full tool) and demo/ (one-page live demo)
├── notebooks/               11 executed notebooks, one per stage
├── data/                    raw source tables, processed datasets, data card (README.md)
├── results/                 every table behind every number, and the trained models
├── figures/                 result figures
├── reports/                 submission report, posters, slides and paper, with the scripts that build them
└── tests/                   pytest suite (98 tests)
```

## Notebooks

| Notebook | What it shows |
|---|---|
| `01_data_overview` | the open data, and what furnace vs air cooling does to 33 measured grades |
| `02_steel_categories` | steel classifier, tests on real heats, SHAP, detection of steels outside the scope |
| `03_microstructure` | transformation model, checks against dilatometry and literature, CCT diagrams |
| `04_property_models` | the model ladder, paired tests, conformal intervals, cooling-rate extrapolation |
| `05_furnace_vs_air` | the answer for each family of steel, carbon sweep, effect of bar size |
| `06_external_validation` | structural members, second-source and literature checks, bar sizes |
| `07_design_and_active_learning` | grade and route recommendations, active learning, recalibration |
| `08_hardenability` | ideal critical diameter, Jominy H-bands, Bayesian kinetics calibration |
| `09_hardening_tempering` | quench heat flow, tempering, Q+T model ladder, mass effect, the five treatments |
| `10_assumptions_sensitivity` | assumptions register, retraining per assumption, Morris screening |
| `11_kinetic_uncertainty` | MCMC posterior of the kinetic constants and how it carries into predictions |

## Data and licences

See `data/README.md`. In short, ASM/Bethlehem handbook values (factual data), SAE composition ranges, and CC BY
datasets from Zenodo and Mendeley Data (Birbilis et al. 2025; Collins et al. 2023; SteelBench; Carbon and Stainless
Steel Grades), together with 215 measured values transcribed from 52 open-access papers
(`data/raw/literature_sources.md`).

## Main references

Li et al. (1998) *Metall. Mater. Trans. B* 29, 661; Kirkaldy and Venugopalan (1984); Collins et al. (2023)
*Metals* 13, 1168; Gladman, McIvor and Pickering (1972) *JISI* 210, 916; Maynier, Jungmann and Dollet (1978);
Grossmann, Asimow and Urban (1939); Hollomon and Jaffe (1945) *Trans. AIME* 162, 223; Grange, Hribal and Porter
(1977) *Metall. Trans. A* 8, 1775; Haario et al. (2001) *Bernoulli* 7, 223; Lei et al. (2018) *JASA* 113, 1094;
Birbilis, Ghorbani and Zhao (2025), Mendeley Data, doi:10.17632/jmwb9ddd43.1. The full list is in the report.
