# Heat treatment of carbon and low-alloy steels: microstructure and tensile properties with physics-informed ML

**B.Tech Project (MTP-400A), Department of Metallurgical and Materials Engineering, IIT Roorkee**
**Authors:** Aviral Singh, Shibasish Banerjee  |  **Supervisor:** Dr. Anish Karmakar

---

## What this project does

Give it the composition of a carbon or low-alloy steel and a heat treatment, and it tells you what kind of
steel it is, what microstructure forms, and what yield strength, tensile strength, uniform and total
elongation and hardness to expect, each with a 90 % interval. It covers the five classic treatments:

| Treatment | What the model computes |
|---|---|
| **Annealing** (furnace cooling) | coarse pearlite + ferrite, softest state |
| **Normalising** (still air, any bar size) | finer pearlite and ferrite, or bainite in hardenable steels |
| **Hardening** (oil, water or brine quench of a round bar) | martensite, bainite, ferrite + pearlite at the specimen position, hardness |
| **Tempering** (any temperature and time) | tempered martensite: strength, ductility, embrittlement warnings |
| **Stress relieving** | microstructure unchanged, residual-stress bound, softening only above the tempering parameter |

The core research question, set by our supervisor, is **furnace cooling vs air cooling**: what changes, by how
much, why, and how the bar size matters. Version 3 adds the rest of the route after a review by a fellow student.

The model is a chain that follows the metallurgy. Machine learning is used where the physics alone is not good
enough, and the physics is kept where the data alone cannot be trusted:

```
composition ─► ML classifier ─► category (13 families, or "unknown"); microalloyed / TMCP / boron flags
     ├─► thermodynamics ─► Ae1, Ae3, Acm, eutectoid, lever rule, default temperatures
treatment ─► heat flow ─► cooling path (furnace; bar in still air; radial quench with Grossmann H)
     └─► transformation kinetics (Li et al. 1998, Bayesian-calibrated bainite) ─► F, P, B, M, spacing, grain size
                 │                                  └─► tempering (Hollomon-Jaffe master curve + secondary hardening)
     physics-anchored Gaussian-process models + conformal 90 % intervals
                 │
     YS, UTS, elongation, hardness, ITT ─► what-if, five-treatment comparison, grade recommendation
```

## Key results

Every number is in `results/` and explained in `reports/REPORT.md`, which is generated from the results so the
two cannot disagree. "Unseen" means leave-one-grade-out: the grade being predicted was never in training.

| Question | What we found |
|---|---|
| Which steel is it? | 13 categories; 0.94 family accuracy on 1358 real heats (AISI/SAE composition rules: 0.92); all 312 maraging steels of the old project refused as "unknown" |
| Which microstructure forms? | ferrite start within 16 °C and Ms within 7 °C of dilatometry; 423 CCT-atlas points: ferrite + pearlite within 0.12, bainite 0.16 (fraction MAE) |
| Is the kinetics right for hardenable steels? | SAE Jominy bands of 80 grades: a Bayesian recalibration of the bainite reaction cut the band error from 10.1 to 5.9 HRC on unseen grades; an MCMC posterior shows the remaining error is structural, not parametric |
| Annealed / normalised properties | unseen grades: UTS 47 MPa, YS 44 MPa, hardness 16 HB (RMSE), against 94, 100 and 23 for the textbook relations; 90 % intervals hold 92.5 % |
| Furnace vs air cooling | low carbon: almost the same steel (+37 MPa); medium / high carbon: same phases, finer pearlite (+79 / +251 MPa); Cr-Mo and Ni-Cr-Mo: partly bainitic in air, so bar size matters as much as the route |
| Bar size (normalising) | the 13 mm vs 100 mm strength difference has the right sign for all 32 grades tested, but the model overstates its size (151 vs 99 MPa) |
| Quenched and tempered | 324 bars of 28 grades at 13-100 mm: UTS within 75 MPa, YS 85 MPa, elongation 1.7 % on unseen grades (textbook physics: 247 MPa); 89 % of bars inside the 90 % interval; no bias with bar size |
| Assumptions | every one registered; the handbook's normalised data are **25 mm bars** (documented in a second source, not the 50 mm we had inferred); the one undocumented condition, the furnace rate, changes how much air cooling adds, never the direction |

## Version 3: what changed and why

A fellow student ranked what version 2 was missing. We followed the list:

1. **Inferred cooling conditions:** an assumptions register, retraining over every range, and a Morris screen.
   The bar size of the handbook's normalised data is now documented (25 mm), and that correction changed the
   results.
2. **Kinetic uncertainty:** four-chain MCMC with R-hat and ESS, posterior draws propagated to predictions, and
   coverage tested on independent data.
3. **Hardening:** radial quench heat flow in oil, water or brine, checked against the exact Bessel solution and
   against Grossmann's D_I.
4. **Tempering:** a Hollomon-Jaffe master curve with secondary hardening, fitted and tested by grade.
5. **Precipitation strengthening:** V and Nb solubility plus Ashby-Orowan, as an estimate with ±50 %.
6. **Toughness:** ITT for ferrite-pearlite. For quenched-and-tempered steel, Charpy is declared out of scope
   and the embrittlement windows are flagged.
7. **TMCP / microalloyed / boron steels:** detected from the chemistry, and declined for as-delivered properties.

**On data size:** the new open database (Birbilis et al. 2025, CC BY 4.0) multiplied the heat-treated property
data by about 8. The raw data are still only a few MB, because tabular data with a measured property *and* a known
treatment are small everywhere. Steel datasets of a gigabyte are image collections or tables without the
processing, and neither can teach a model what a heat treatment does. See `reports/REPORT.md`, Section 3.1.

The first version of the project, yield strength from composition alone, is kept in git history under the tag
`legacy-composition-only-v1`.

## Quick start

```bash
make setup        # virtual environment + requirements (Python 3.10+, tested on 3.13)
make pipeline     # stages 1-11, about 25 minutes; writes results/ and figures/
make notebooks    # 11 executed presentation notebooks
make test         # physics, data and model tests
make app          # interactive tool: http://127.0.0.1:7860 (furnace vs air, and all five treatments)
make report       # report, slide deck and journal-style manuscript
```

`data/raw/` is committed, so the pipeline runs offline. `make fetch` re-downloads every source, re-parses it,
and records checksums in `data/raw/source_manifest.csv`.

Using the model from Python:

```python
import sys; sys.path.insert(0, "src")
from steelml.predictor import SteelPredictor

sp = SteelPredictor.load()
comp = {"C": 0.42, "Mn": 0.75, "Si": 0.25, "Cr": 1.05, "Mo": 0.22}        # an EN19 / 42CrMo4-type steel

sp.what_if(comp, section_mm=25)                                          # furnace vs air cooling
sp.heat_treat(comp, "quench_temper", section_mm=30, medium="oil", temper_T=550)
sp.compare_treatments(comp, section_mm=30)                               # all five, side by side
```

## Repository structure

```
BTP/
├── src/steelml/             the package
│   ├── physics/             thermo, cooling (air), quench (radial heat flow), kinetics (Li et al., numba),
│   │                        microstructure, properties, tempering, stress_relief, precipitation,
│   │                        toughness, hardenability, jominy, calibration (Bayesian)
│   ├── data/                source registry, parsers (incl. birbilis.py), catalogue, dataset assembly
│   ├── models/              classifier, property models, qt_models, flags, domain, evaluation
│   ├── features.py          composition + treatment -> physics descriptors
│   └── predictor.py         end-to-end predictor: what_if, heat_treat, compare_treatments
├── pipeline/                s00 fetch ... s11 kinetic uncertainty; run_all.py runs everything
├── app/app.py               Gradio tool (two tabs)
├── notebooks/               01-11 executed notebooks (+ make_notebooks.py)
├── data/                    raw (committed source tables), processed, README.md (data card)
├── results/                 every table behind every number, and the trained models
├── figures/                 fig01-fig42 and CC BY micrographs
├── reports/                 REPORT.md (generated), slide deck (PDF, PPTX), manuscript, and their sources
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
| `06_external_validation` | structural members, second-source and literature checks, bar sizes |
| `07_design_and_active_learning` | grade + route recommendations, active learning, recalibration, lab plan |
| `08_hardenability` | ideal critical diameter, Jominy H-bands, Bayesian kinetics calibration |
| `09_hardening_tempering` | quench heat flow, tempering, Q+T model ladder, mass effect, the five treatments, a live example |
| `10_assumptions_sensitivity` | assumptions register, the documented bar size, retraining per assumption, Morris screen |
| `11_kinetic_uncertainty` | MCMC posterior of the kinetic constants, propagation, independent coverage |

## Limitations

The full list is in `reports/REPORT.md`, Section 11. The main ones:

* The property models learn from 33 handbook grades (annealing, normalising) and 28 grades (quench and temper),
  with typical values at nominal compositions, so the intervals are wide.
* The furnace cooling rate of the handbook's annealed bars is not documented (0.005 K/s assumed), and the
  tempering time of the Q+T bars is assumed to be 1 h. Section 8 measures how much each matters.
* The kinetics are semi-empirical, with a structural error. They over-predict Grossmann's D_I, most for
  high-carbon steels, and overstate the bar-size effect of normalising.
* The quench uses Grossmann's constant heat-transfer coefficient (no boiling stages) and long round bars.
* Estimated rather than trained: precipitation strengthening, uniform elongation, ferrite grain size and ITT.
  As-quenched tensile properties are not given, because untempered martensite is too brittle for a meaningful
  tensile test.
* Out of scope, and the tool says so: Charpy values of tempered steel, as-delivered TMCP products, boron's
  hardenability effect, carburising, and stainless, tool and maraging steels.

## Data and licences

See `data/README.md`. In short: ASM/Bethlehem handbook values (factual data, reproduced by Engineers Edge), SAE
composition ranges (Summit Steel, Wikipedia), and CC BY datasets from Zenodo and Mendeley Data (Collins et al.
2023; Kingsley et al. 2026; SteelBench; Carbon and Stainless Steel Grades; Birbilis et al. 2025). Also 215 measured
values transcribed from 52 open-access papers (`data/raw/literature_sources.md`).

## Main references

Li et al. (1998) *Metall. Mater. Trans. B* 29, 661; Kirkaldy and Venugopalan (1984); Collins et al. (2023)
*Metals* 13, 1168; Gladman, McIvor and Pickering (1972) *JISI* 210, 916; Maynier, Jungmann and Dollet (1978);
Grossmann, Asimow and Urban (1939); Hollomon and Jaffe (1945) *Trans. AIME* 162, 223; Grange, Hribal and Porter
(1977) *Metall. Trans. A* 8, 1775; Gladman (1997); Haario et al. (2001) *Bernoulli* 7, 223; Lei et al. (2018)
*JASA* 113, 1094; Birbilis, Ghorbani and Zhao (2025), Mendeley Data, doi:10.17632/jmwb9ddd43.1. The full list is in
`reports/REPORT.md`.
