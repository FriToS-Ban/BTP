# Live demo

A clean, one-page version of the project for presenting. It runs offline on your laptop and follows the poster's
story: **1 Classify** the steel, **2 Heat-treat** it (furnace against air cooling, with its CCT diagram), **3 Predict**
its microstructure and properties with 90 % intervals, for all five treatments. Every number comes from the same
trained predictor as the report and the poster.

## Start and stop

```bash
make demo
```

(or `.venv/bin/python app/demo/server.py`). It loads the models in about two seconds and opens
<http://127.0.0.1:8050> in your browser. Stop it with **Ctrl+C** in the terminal. If port 8050 is busy it takes the
next free one and prints the address. `--no-browser` starts it without opening a tab.

## The controls

| Control | What it does |
|---|---|
| **Steel** list, or the quick buttons | any of the 177 AISI/SAE carbon and low-alloy grades; *304 stainless* shows the tool declining a steel outside its scope |
| **Composition** | edit C, Mn, Si, Cr, Ni, Mo (wt %); the list switches to *Custom composition* |
| **Bar diameter** | the bar that is air cooled (and quenched): 10–150 mm; acts on both views |
| **Quench** | Automatic (oil or water, by hardenability), Oil or Water; opens the five-treatment view |
| **Tempering temperature, 1 h** | the tempering of the quenched-and-tempered bar; opens the five-treatment view and highlights that row |
| **Furnace vs air / Five treatments** | switches the view |

Every change updates the page at once; there is no Predict button. Quench and tempering act only on the
five-treatment view, which is why changing them opens it. For a steel with little martensite when quenched (1020
in a 25 mm bar), the page says that tempering changes it little.

Run `.venv/bin/python app/demo/check_demo.py` to check the app: every grade, every control, and its numbers against
the project's result files.

## A three-minute demo

1. **1045, Furnace vs air.** Both routes give ferrite + pearlite; air cooling refines the pearlite, so the strength
   rises by about 100 MPa. Point at the CCT diagram: the air path crosses the ferrite and pearlite curves lower down.
2. **4140H.** The air-cooled bar is 92 % bainite and 360 MPa stronger: "a different steel". On the CCT diagram the
   air path slips under the ferrite and pearlite curves.
3. **Drag the bar diameter to 100 mm.** Only 17 % bainite now, and the strength falls: the mass effect.
4. **Five treatments.** One bar, five treatments, as on poster Fig 13. Drag the tempering temperature to 300 °C: the
   strength rises to about 1,620 MPa and the tool warns of tempered-martensite embrittlement.
5. **304 stainless.** The tool says the steel is outside what it knows instead of guessing.

Before presenting, run it once on the laptop you will use. On a small screen, zoom the browser out (Cmd −) so the
whole view fits.
