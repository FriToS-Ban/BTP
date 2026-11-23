# A1 poster, version 2

**One Steel, Five Heat Treatments**
*Predicting what changes, and why: physics-informed machine learning of microstructure and properties of carbon
and low-alloy steels*

`BTP_poster_A1_v2.pdf` (print, 594 × 841 mm, no bleed) and `BTP_poster_A1_v2.pptx` (editable) are built on
`Template.pptx`, the department frame with the IIT Roorkee logo. Every number and every plotted value comes
from `results/`; nothing is typed in by hand. The first poster is kept unchanged in `reports/poster/`.

## What changed from the first poster

* **Title.** "Furnace or Air Cooling? Predicting What Changes in a Steel" named one question. The new title is a
  short line for the whole project, set large as on the template (50 pt), and the subtitle says what is predicted,
  how, and for which steels.
* **Figures: 15 instead of 8.** New: transformation starts against dilatometry (3), hardness against the CCT atlas
  (4), the model ladder (6), hardenability across the whole grade catalogue (9), furnace against air by steel
  family (10), the bar-size effect against measured bars (11) and the mass effect after quenching and tempering
  (15).
* **Sections.** "Validation Against Experiments" is new. "Software Tool" and "Project Team" are gone: the team is in
  the header, and the repository address is in the acknowledgements.
* **Build.** Chrome lays out the page and prints the PDF, so it builds on macOS, Windows or Linux without
  PowerPoint. The same layout is then written to the template as an editable PPTX.

## How to build

| Step | Command | Writes |
|---|---|---|
| 1. CCT curves of 1045 and 4140 (only if the kinetics change) | `python reports/poster_v2/cct_data.py` | `data/cct_*.csv`, `data/cct_*.json` |
| 2. Figures at print size | `python reports/poster_v2/make_figures.py` | `figures/*.svg` (vector), `figures/*.png` (300 dpi) |
| 3. Page, PDF and preview | `python reports/poster_v2/build_poster.py` | `poster.html`, `BTP_poster_A1_v2.pdf`, `preview/poster_preview.png` |
| 4. Editable PPTX on the template | `python reports/poster_v2/export_pptx.py` | `BTP_poster_A1_v2.pptx` |
| 5. QA | `python reports/poster_v2/check_poster.py` | report on screen |

Run them with the project environment (`.venv`). Step 3 needs Google Chrome (Playwright drives it, as for the
report and the slide deck) and prints the spare height of each column and any text that does not fit. The title
and subtitle are the two constants at the top of `build_poster.py`.

## Type and fonts

Type sizes follow the reference BTP poster: headings 28 pt, body 20 pt, takeaways 17 pt, captions 16 pt, small
print 15 pt, figure tick and axis labels 15–16 pt. The title is 50 pt (the template's own title line is 54 pt).

The body font is Calibri. Where Calibri is not installed (macOS without Office, Linux), the build uses Carlito,
the open font with the same metrics, so the line breaks are the same on every machine (`poster_fonts.py`).
Figure text is stored as outlines. Equations use Cambria, or STIX Two where Cambria is missing.

## The PPTX

The PDF is the print master. The PPTX holds the same blocks at the same positions as native text boxes and
shapes, with each figure as a vector SVG and a 300 dpi PNG behind it. It was written and checked with
python-pptx and previewed with macOS Quick Look. It has **not** been opened in PowerPoint on the machine that
built it, so open it once and look it over before printing from it.

## The steels in the figures

The "1045" and "4140" of figures 8, 13, 14 and 15 are the nominal AISI 1045 and AISI 4140H compositions that
stage 9 uses for its five-treatment comparison (`results/s09_five_treatments.csv`), so the phase fractions agree
across panels. Figures 9 and 10 use the 269 catalogue grades that form ferrite + pearlite on slow cooling; the two
Si-Mn spring steels (2 % Si) are left out because they are outside the window of the critical-temperature
equations.
