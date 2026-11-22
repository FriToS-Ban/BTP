# A1 poster

`BTP_poster_A1.pdf` (print, 594 × 841 mm, no bleed) and `BTP_poster_A1.pptx` (editable) are built from
`Template.pptx` (the department frame with the IIT Roorkee logo). Every number and every plotted value
comes from `results/`; nothing is typed in by hand.

| Step | Command | Writes |
|---|---|---|
| 1. CCT curves of 4140 (needs the project `.venv`, numba) | `.venv/Scripts/python reports/poster/cct_data.py` | `data/cct_4140.csv`, `data/cct_4140.json` |
| 2. Figures at print size, vector | `python reports/poster/make_figures.py` | `figures/f1..f6.svg` (+ PNG previews) |
| 3. Layout and text on the template | `python reports/poster/build_poster.py` | `poster_draft.pptx`, `placements.json` |
| 4. PowerPoint places the SVGs, checks text fit, exports | `powershell -ExecutionPolicy Bypass -File reports/poster/finish_poster.ps1` | `BTP_poster_A1.pptx`, `BTP_poster_A1.pdf`, `preview/poster_preview.png` |
| 5. QA: font sizes by role, word count | `python reports/poster/check_poster.py` | report on screen |

Step 4 needs Microsoft PowerPoint (Windows). Sizes at print scale: title 80 pt, headings 48 pt, body 28 pt,
captions, references and acknowledgements 22 pt, tiles and diagram text 24 pt, figure tick and axis labels
24–26 pt.

The "4140" of figures 2, 5 and 6 is the nominal AISI 4140H composition that stage 9 uses for its
five-treatment comparison (`results/s09_five_treatments.csv`), so the phase fractions agree across panels.
