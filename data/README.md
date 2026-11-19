# Data card

All data used in the project are open (or are factual handbook values reproduced openly), are recorded
with their source and licence, and are re-created by `python pipeline/s00_fetch_data.py`.

`data/external/` holds the raw downloads (git-ignored). `data/raw/` holds the parsed source tables
(committed, small). `data/processed/` holds the analysis-ready tables built by stage 1.

## Sources

| Key | What it provides | Rows | Value type | Licence | Used for |
|---|---|---|---|---|---|
| `asm_carbon`, `asm_alloy` | ASM Handbook / Bethlehem Steel table of **hot-rolled, normalized (air-cooled) and annealed (furnace-cooled)** properties of 33 AISI grades: UTS, YS, elongation (50 mm), Brinell hardness, and the normalizing/annealing temperatures. Reproduced by Engineers Edge (psi, converted to MPa). | 80 | typical | factual data | **training** the property models |
| `summit_carbon`, `summit_alloy_h` | SAE J403 (carbon) and SAE J1268 (alloy H-steel) composition ranges, as tabulated by Summit Steel | 101 + 75 grades | composition range | factual data | grade catalogue, compositions of the training grades |
| `wiki_sae`, `wiki_tool` | SAE family definitions, stainless (58) and tool (12) steel compositions (Wikipedia) | - | composition range | CC BY-SA 4.0 | grade catalogue |
| `zenodo_grades` | Zenodo 17704410 *Carbon and Stainless Steel Grades*: composition ranges, properties and processing flags of AISI/EN/GB/ASTM/UNS carbon and stainless grades (MatWeb/MakeItFrom-derived) | 538 | mixed: typical and specification-level | CC BY 4.0 | catalogue (EN/GB/ASTM grades), inter-source check, spec-level plausibility check |
| `zenodo_cleveland` | Zenodo 19000367 (Kingsley et al. 2026): **measured** heat chemistry, tensile and Charpy results of 182 reclaimed structural members | 182 (165 usable) | measured | CC BY 4.0 | external validation, few-shot recalibration |
| `steelbench` | Zenodo 18530558 (SteelBench v1.0): heat-level chemistry with steel-family labels | 1358 | mixed | CC BY 4.0 | external test of the category classifier |
| `collins2023` | Zenodo 7770264 (Collins et al. 2023): **measured** transformation-start temperatures (dilatometry) and microhardness (16 indents per rate) at 9 cooling rates, chemistry and prior-austenite grain size of EN3B, EN8 and SA-540 B24; also the authors' own Li-model output (start temperatures and the constituent fractions in `collins2023_fractions.csv`, which are **modelled, not measured**, and are used only to check that our implementation reproduces theirs) | 3 steels x 9 rates | measured | CC BY 4.0 | validation of the transformation model and of cooling-rate extrapolation. Only 15 small members of the 231 MB archive are read, by HTTP range requests. |
| `literature` | `raw/literature_compilation.csv`: 215 rows from 52 open-access papers (pearlite fraction, ferrite grain size, pearlite spacing, YS, UTS, uniform and total elongation, hardness, impact energy of annealed / normalized / other ferrite-pearlite and low-alloy steels). 93 rows (S01-S26) from the first search, 122 (S27-S52) from a second search for same-steel furnace-vs-air comparisons; every number of the second search was checked by script against the saved full text, and its verbatim excerpt is in `raw/literature_evidence.csv`. 51 rows carry `quality_flag = suspect: <reason>` and are excluded from every metric. See `raw/literature_sources.md`. | 215 | measured | per paper (mostly CC BY) | microstructure validation and calibration, uniform-elongation ratio, external property checks, paired furnace-vs-air changes |
| `birbilis2025` (v3) | Mendeley Data jmwb9ddd43 (Birbilis, Ghorbani & Zhao 2025): composition, a written **processing condition** (quench medium, austenitising and tempering temperature, bar size 13/25/50/100 mm) and YS, UTS, ductility of 3234 steels. Parsed by `src/steelml/data/birbilis.py`; after removing out-of-scope, carburised, cold-drawn and repeated (H / non-H) rows: 324 quenched-and-tempered bars (28 grades) and 241 normalised / annealed bars (39 grades). Its normalised 25 mm rows equal the ASM handbook values, which documents the handbook bar size. | 3234 (565 used) | typical (handbook-type, stated bar size) | CC BY 4.0 | hardening + tempering models (stage 9), bar-size tests (stages 6, 10) |
| `uhcsdb` (optional, not used) | NIST *Ultrahigh Carbon Steel Micrographs* (961 SEM images with anneal/cooling metadata). Located and documented as a future-work dataset; its steels (1.3-1.5 % C) lie outside the ferrite-pearlite scope. | 961 | measured | CC BY 3.0 US | - |

`raw/source_manifest.csv` records the URL, SHA-256, licence and citation of every file.

**Sources checked and not used for training (v3).** SteelBench's NIMS rows (360) give tensile strength without
any heat treatment, the flaw of this project's first version; its EMK rows are specification minima whose
compositions look like upper limits. The NIMS fatigue dataset (Agrawal et al. 2014) and the MAP datasets have
no licence we could confirm for redistribution, and eFunda's quenched-and-tempered tables are copyrighted.

**Why the data are a few MB.** Tabular data with a measured property *and* a known heat treatment are small
everywhere; steel datasets of a gigabyte are image collections (e.g. the NIST micrographs above, 270 MB) or
tables without the processing. The project counts measured records with their treatment, not megabytes.

## Typical values versus minimum values

Handbook "typical" values (ASM/Bethlehem) are averages of tested material. Specification values
(SAE J1397 hot-rolled bar minima, EN/ASTM normalized-grade requirements) are **minimum** guarantees and sit
25-35 % lower for the same grade. The two are never mixed in training: only typical values train the
models; specification values are used only as lower-bound checks (stage 6).

## Curation and corrections

Every correction is listed in `raw/source_corrections.csv` with its reason. In short:

* Summit alloy H table: a row shift moved the Ni/Mo ranges of 81B45H onto 6150H and the vanadium notes one
  row up (51B60H/6118H/6150H); 4037H carbon `.39-.41` -> `0.34-0.41`; 1335H `Si .15- 35` -> `0.15-0.35`.
* Summit carbon table: the Mn ranges of 1117 and 1118 are swapped relative to SAE J403 (cross-checked with
  Zenodo 17704410); 1212/1213/12L14 carbon lacked a decimal point.
* Summit carbon table: "15821H" is 15B21H (the boron "B" printed as "8"; its C and Mn ranges are those of
  15B21H).
* 3140 and 9255 are absent from both range tables; their ranges are centred on the Wikipedia SAE family
  nominal values (31xx: Ni 1.25, Cr 0.65; 92xx: Si 2.00, Mn 0.82).
* Alloy grades of the property table use the SAE J1268 H-band ranges of the same grade (midpoints are
  within 0.01-0.03 wt% of the standard ranges).
* Composition of a handbook grade is the midpoint of its range; unspecified residuals (Cr, Ni, Mo, Cu, ...)
  are set to typical residual levels (`config.RESIDUAL_TYPICAL`).
* Kingsley et al. members: cold-formed hollow sections (EN 10219, or SHS/RHS/CHS without EN 10210) are
  excluded because cold work raises the yield strength; entries with P > 0.1 % or Mn < 0.2 % are treated as
  typing errors; `<x` detection limits are set to x/2.

## Calibrated constants (not fitted to the property targets)

| Constant | Value | Evidence |
|---|---|---|
| Effective air-cooling of the handbook *normalized* data | 50 mm bar in still air (~0.5 K/s at 700 C) | physics-only hardness error (no ML) is flat between ~50 and 80 mm (21-22 HB RMSE, vs 58 HB at 25 mm); 50 mm taken at the start of the flat region; stage 3, `results/s03_calibration.csv` |
| Effective furnace cooling of the handbook *annealed* data | 0.005 K/s (18 K/h) | same scan: the error hardly changes below 0.01 K/s (28.5-29.5 HB); 0.005 K/s is the usual full-annealing practice for alloy steels |
| Pearlite spacing `S0 = A dT^-m` | A = 3.33 um, m = 0.607 | 14 measured continuous-cooling spacings from 4 papers; leave-one-source-out factor 1.7. A constant k cannot fit both routes: the isothermal Marder-Bramfitt 8.02/dT is ~3x too fine for air cooling, and a k fitted to air cooling makes furnace-cooled pearlite 2-3 um |
| Uniform / total elongation ratio | 0.555 +/- 0.075 | 10 measured tensile tests from 4 papers; leave-one-source-out MAE 3.3 % |
| Ferrite rate multiplier (Li et al.) | 1.0 (published) | a scan shows a faster ferrite rate would fit air-cooled fractions but spoil measured ferrite-start temperatures; kept at the published value |
