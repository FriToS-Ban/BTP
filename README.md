# Furnace cooling vs air cooling of steels: microstructure and tensile properties with physics-informed ML

**B.Tech Project (MTP-400A), Department of Metallurgical and Materials Engineering, IIT Roorkee**
**Authors:** Aviral Singh, Shibasish Banerjee  |  **Supervisor:** Dr. Anish Karmakar

---

## What this project does

Give it the composition of a steel, and it tells you what kind of steel it is, what microstructure forms
when it is **furnace cooled** (full annealing) and when it is **air cooled** (normalising), and what yield
strength, tensile strength, uniform and total elongation and hardness to expect, each with a 90 % interval.
It also says why the two routes differ, and it refuses to guess for steels it has never seen anything like.

We built it as a chain that follows the metallurgy, with machine learning used where physics alone is not
good enough and physics kept where data alone cannot be trusted:

```
composition ─► ML classifier ─► category  (low/medium/high C, free-machining, C-Mn, Cr, Cr-Mo, Ni-Cr-Mo,
     │                                     Si-Mn, stainless, tool, or "unknown")
     ├─► thermodynamics ─► Ae1, Ae3, Acm, eutectoid carbon, lever rule
cooling route ─► heat balance ─► cooling path (furnace, or a bar of diameter D in still air)
     └─► transformation kinetics (Li et al. 1998) ─► ferrite, pearlite, bainite, martensite,
                                                     pearlite spacing, ferrite grain size
                                     │
          physics-anchored Gaussian-process models + conformal intervals
                                     │
          YS, UTS, total and uniform elongation, hardness  ─►  what-if, inverse design,
                                                               active learning, lab plan
```

## Why it was rebuilt

The first version of this project predicted the yield strength of 312 maraging-type steels from
composition alone. Our supervisor pointed out that this does not make physical sense, because strength
depends on the heat treatment and that data had none. The new project is built around the heat treatment
itself. The old work is kept in git history under the tag `legacy-composition-only-v1`.

## Key results

Every number is in `results/` and explained in `reports/REPORT.md` (generated from the results, so the two
cannot disagree). "Unseen" means leave-one-grade-out: the grade being predicted was never in training.

| Question | What we found |
|---|---|
| Which steel is it? | 13 categories; 0.94 family accuracy on 1358 real heats, against 0.92 for the AISI/SAE composition rules; all 312 maraging steels of the old project refused as "unknown" |
| Which microstructure forms? | ferrite start within 16 °C and Ms within 7 °C of measured dilatometry; 423 points of an open CCT atlas: ferrite + pearlite within 0.12, bainite 0.16 (fraction MAE) |
| Is the kinetics right for hardenable steels? | against the SAE Jominy bands of 80 grades the published Li model was too hardenable (+12 HRC far from the quenched end); a Bayesian recalibration of the bainite reaction cut the band error from 10.1 to 5.9 HRC on unseen grades and moved the bainite start onto independent dilatometry |
| Which properties? | unseen grades: UTS 53 MPa, YS 42 MPa, hardness 14 HB (RMSE), against 82, 82 and 19 for the textbook relations alone; 90 % intervals cover 92.5 % |
| Furnace vs air cooling | low-carbon: same steel, only refined (+30 MPa UTS); medium/high carbon: same phases, finer pearlite and ferrite (+80 to +270 MPa); Cr-Mo and Ni-Cr-Mo: partly bainitic in air-cooled bars, so bar size matters as much as the route |
| Why stronger on air cooling? | almost all of it is the faster cooling, not the hotter austenitisation; normalising also lowers the impact transition temperature by about 30 °C |

What we learned the hard way is kept as ablations in `results/`: silicon as an ML input (looked better in
cross-validation, behaved unphysically), the physics anchor (a composition-only hardness anchor could not see
pearlite refinement), and calibrations that fit one dataset but must be checked on the others.

## Quick start

```bash
make setup        # virtual environment + requirements (Python 3.10+, tested on 3.13)
make pipeline     # stages 1-8, about 15 minutes; writes results/ and figures/
make notebooks    # 8 executed presentation notebooks
make test         # physics, data and model tests
make app          # interactive tool: http://127.0.0.1:7860
make report       # report, slide deck and journal-style manuscript (reports/paper/manuscript.pdf)
```

`data/raw/` is committed, so the pipeline runs offline. `make fetch` re-downloads every source and
re-parses it, and records checksums in `data/raw/source_manifest.csv`.

Using the model from Python:

```python
import sys; sys.path.insert(0, "src")
from steelml.predictor import SteelPredictor

sp = SteelPredictor.load()
out = sp.what_if({"C": 0.45, "Mn": 0.75, "Si": 0.25}, section_mm=25)   # furnace vs air cooling
print(out[["route", "F_best", "P_best", "B_best", "M_best",
           "YS_MPa_pred", "UTS_MPa_pred", "El_pct_pred", "uniform_El_est_pct", "HB_pred"]])
```

## Repository structure

```
BTP/
├── src/steelml/             the package
│   ├── physics/             thermo (Ae1/Ae3/Acm, lever rule), cooling (heat balance),
│   │                        kinetics (Li et al., numba), microstructure, structure-property relations
│   ├── data/                source registry, parsers, catalogue, dataset assembly
│   ├── models/              classifier, property models, domain, evaluation, consistency tests
│   ├── features.py          composition + route -> physics descriptors
│   ├── predictor.py         end-to-end predictor with conformal intervals
│   └── categories.py        steel category taxonomy
├── pipeline/                s00 fetch ... s08 hardenability; run_all.py runs everything
├── app/app.py               Gradio what-if tool
├── notebooks/               01-08 executed notebooks (+ make_notebooks.py)
├── data/                    raw (committed source tables), processed, README.md (data card)
├── results/                 every table behind every number, and the trained models
├── figures/                 fig01-fig30, and CC BY micrographs
├── reports/                 REPORT.md (generated), the slide deck (PDF and PPTX) and its sources
└── tests/                   pytest suite
```

## Notebooks

| Notebook | What it shows |
|---|---|
| `01_data_overview` | the open data, and what furnace vs air cooling does to 33 measured grades |
| `02_steel_categories` | category classifier, tests on real heats, SHAP, unknown-steel detection |
| `03_microstructure` | transformation model, validation against dilatometry and literature, CCT diagrams |
| `04_property_models` | the model ladder, paired tests, conformal intervals, cooling-rate extrapolation |
| `05_furnace_vs_air` | category-by-category answer, carbon sweep, section-size effect, a live example |
| `06_external_validation` | structural members, second-source and literature checks |
| `07_design_and_active_learning` | grade + route recommendations, strength-ductility front of real grades, active learning, recalibration, lab plan |
| `08_hardenability` | ideal critical diameter, Jominy H-bands, Bayesian kinetics calibration and its transfer test |

## Limitations

The full list is in `reports/REPORT.md`, section 7. The main ones:

* The property models learn from 33 grades with one typical value per grade and condition (ASM/Bethlehem),
  at nominal compositions, so the 90 % intervals are wide (about +/-95 MPa for strength).
* The cooling conditions of the handbook bars are inferred (a 50 mm bar in still air for normalising,
  0.005 K/s for annealing), and the default austenitising temperatures follow the handbook practice.
* The kinetic model is semi-empirical: transformation temperatures are good, air-cooled pearlite fractions
  need the lever-rule correction, and spacings and grain sizes are good to a factor of about 2.
* Uniform elongation is estimated as 0.51 x total elongation on the handbook gauge (10 literature tests), not measured.
* Modern thermomechanically rolled or microalloyed products are stronger than predicted until a few local
  tests are added. Tempering, precipitation strengthening and toughness are not modelled.
* Only carbon and low-alloy steels that form ferrite + pearlite on slow cooling are in scope; stainless,
  tool and maraging steels are recognised and turned away.

## Data and licences

See `data/README.md`. In short: ASM/Bethlehem handbook values (factual data, reproduced by Engineers
Edge), SAE composition ranges (Summit Steel, Wikipedia), and CC BY datasets from Zenodo (Collins et al.
2023; Kingsley et al. 2026; SteelBench; Carbon and Stainless Steel Grades), plus 215 measured values
transcribed from 52 open-access papers (`data/raw/literature_sources.md`).

## Main references

Li et al. (1998) *Metall. Mater. Trans. B* 29, 661; Kirkaldy and Venugopalan (1984); Collins et al.
(2023) *Metals* 13, 1168; Gladman, McIvor and Pickering (1972) *JISI* 210, 916; Pickering (1978);
Maynier, Jungmann and Dollet (1978); Lei et al. (2018) *JASA* 113, 1094. The full list is in
`reports/REPORT.md`.
