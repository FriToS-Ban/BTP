# A1 poster

`BTP_poster_A1.pdf` (print, 594 × 841 mm, no bleed) and `BTP_poster_A1.pptx` (editable) are built from
`Template.pptx` (the department frame with the IIT Roorkee logo). Every number and every plotted value
comes from `results/`; nothing is typed in by hand.

| Step | Command | Writes |
|---|---|---|
| 1. CCT curves of 1045 and 4140 (needs the project `.venv`, numba) | `.venv/Scripts/python reports/poster/cct_data.py` | `data/cct_1045.*`, `data/cct_4140.*` |
| 2. Figures at print size, vector | `python reports/poster/make_figures.py` | `figures/*.svg` (+ PNG previews) |
| 3. Layout and text on the template | `python reports/poster/build_poster.py` | `poster_draft.pptx`, `placements.json` |
| 4. PowerPoint places the SVGs, checks text fit, exports | `powershell -ExecutionPolicy Bypass -File reports/poster/finish_poster.ps1` | `BTP_poster_A1.pptx`, `BTP_poster_A1.pdf`, `preview/poster_preview.png` |
| 5. QA: font sizes by role, word count | `python reports/poster/check_poster.py` | report on screen |

Step 4 needs Microsoft PowerPoint (Windows). Type sizes follow the reference BTP poster: title 40 pt, headings
28 pt, body 20 pt, takeaways 17 pt, captions 16 pt, references and acknowledgements 15 pt, figure tick and axis
labels 15–16 pt. `build_poster.py` reads the figure sizes from `make_figures.SIZES`, so slots and figures match,
and spreads any spare height evenly between sections.

The "1045" and "4140" of figures 3, 6 and 7 are the nominal AISI 1045 and AISI 4140H compositions that stage 9
uses for its five-treatment comparison (`results/s09_five_treatments.csv`), so the phase fractions agree across
panels.
