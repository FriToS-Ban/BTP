"""Self-check of the live demo: run it before presenting.

    .venv/bin/python app/demo/check_demo.py

1. Every one of the 177 grades in the list runs without an error and gives valid JSON.
2. The quick-pick steels across bar sizes, tempering temperatures and quenches: no errors, and the physics
   trends hold (thicker bar -> weaker normalised bar; hotter tempering -> weaker tempered bar, apart from
   secondary hardening; water quench -> at least as much martensite as oil).
3. The app's numbers against the project's own result files: the five treatments of stage 9, the bar-size
   sweep of stage 5, the CCT curves of the poster, and the measured handbook values.
4. Odd inputs (no carbon, very high carbon or silicon, stainless, tool-steel chemistry, bad values).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import server as s  # noqa: E402

from steelml import config  # noqa: E402
from steelml.predictor import SteelPredictor  # noqa: E402

ROOT = HERE.parents[1]
RES = ROOT / "results"
problems = []


def check(ok, msg):
    if not ok:
        problems.append(msg)
    return ok


def run(payload):
    r = s.run(payload)
    json.dumps(r, allow_nan=False)          # what the server sends; fails on NaN
    return r


def main():
    s.SP = SteelPredictor.load()
    cat = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv")
    s.SCOPE = cat[cat.in_fp_scope & cat.standard.eq("AISI/SAE")].sort_values("grade")
    G = {x["name"]: x["comp"] for x in s.grades()["grades"]}

    # 1. every grade
    out_scope, cautions = [], 0
    for name, comp in G.items():
        try:
            r = run({"comp": comp, "bar_mm": 25, "temper_C": 600})
        except Exception as e:
            check(False, f"{name}: {type(e).__name__}: {e}")
            continue
        if not r["in_scope"]:
            out_scope.append(f"{name} -> {r['steel']['family']}")
        else:
            cautions += bool(r["change"]["cautions"])
    print(f"1. all {len(G)} grades: {len(G) - len(out_scope)} predicted, {len(out_scope)} declined {out_scope}; "
          f"{cautions} show a caution")

    # 2. quick picks over the controls
    n = 0
    for name in ("AISI 1020", "AISI 1045", "AISI 1080", "AISI 4140H", "AISI 4340H"):
        norm, temp = [], {}
        for bar in (10, 25, 50, 100, 150):
            r = run({"comp": G[name], "bar_mm": bar, "temper_C": 600})
            norm.append(r["routes"]["AC"]["UTS_MPa"][0])
            for medium in ("auto", "oil", "water"):
                for T in (150, 300, 450, 600, 700):
                    q = run({"comp": G[name], "bar_mm": bar, "temper_C": T, "medium": medium})
                    rows = {x["key"]: x for x in q["five"]["rows"]}
                    temp[(bar, medium, T)] = (rows["quench_temper"]["UTS"][0], rows["harden"]["M"])
                    n += 1
        check(all(b <= a + 5 for a, b in zip(norm, norm[1:])), f"{name}: normalised UTS rises with bar size {norm}")
        for bar in (10, 25, 50, 100, 150):
            for medium in ("auto", "oil", "water"):
                u = [temp[(bar, medium, T)][0] for T in (150, 300, 450, 600, 700)]
                check(all(b <= a + 25 for a, b in zip(u, u[1:])),
                      f"{name}, {bar} mm, {medium}: tempered UTS rises with temperature {u}")
            check(temp[(bar, "water", 600)][1] >= temp[(bar, "oil", 600)][1] - 0.01,
                  f"{name}, {bar} mm: water quench gives less martensite than oil")
    print(f"2. quick picks: {n} runs over bar size, quench and tempering")

    # 3a. five treatments against stage 9 (same nominal compositions)
    ft = pd.read_csv(RES / "s09_five_treatments.csv")
    worst = 0.0
    for g, name in (("1020", "AISI 1020"), ("1045", "AISI 1045"), ("1080", "AISI 1080"), ("4140", "AISI 4140H"),
                    ("4340", "AISI 4340H"), ("5160", "AISI 5160H")):
        if name not in G:
            continue
        ref = ft[ft.grade.astype(str) == g].set_index("treatment")
        rows = {x["key"]: x for x in run({"comp": G[name], "bar_mm": 25, "temper_C": 600})["five"]["rows"]}
        for t, x in rows.items():
            worst = max(worst, abs(x["HV"] - ref.loc[t, "HV"]))
            if x.get("UTS"):
                worst = max(worst, abs(x["UTS"][0] - ref.loc[t, "UTS_MPa_pred"]))
    check(worst < 15, f"five treatments differ from results/s09_five_treatments.csv by up to {worst:.0f}")
    print(f"3a. five treatments vs results/s09_five_treatments.csv: largest difference {worst:.1f} (MPa or HV)")

    # 3b. bar-size sweep against stage 5
    ss = pd.read_csv(RES / "s05_section_size.csv")
    worst = 0.0
    for g in ss.grade.unique():
        name = g if g in G else f"{g}H"
        if name not in G:
            continue
        for _, row in ss[(ss.grade == g) & (ss.route == "AC")].iterrows():
            r = run({"comp": G[name], "bar_mm": row.section_mm})
            worst = max(worst, abs(r["routes"]["AC"]["UTS_MPa"][0] - row.UTS_MPa_pred))
    print(f"3b. air-cooled bars 10-150 mm vs results/s05_section_size.csv: largest difference {worst:.0f} MPa "
          "(the stage-5 compositions are the plain grades, the app's the list's)")

    # 3c. CCT curves against the poster's
    for tag, name in (("1045", "AISI 1045"), ("4140", "AISI 4140H")):
        ref = pd.read_csv(ROOT / f"reports/poster_v2/data/cct_{tag}.csv")
        mine = run({"comp": G[name], "bar_mm": 25})["cct"]["curves"]
        diffs = []
        for k, col in (("F", "ferrite"), ("P", "pearlite"), ("B", "bainite")):
            a = np.array(mine.get(k) or [[np.nan, np.nan]])
            b = ref[ref.curve == col].sort_values("t_s")[["t_s", "T_C"]].to_numpy()
            if len(b) and len(a) > 1:
                diffs.append(float(np.nanmax(np.abs(np.interp(np.log(b[:, 0]), np.log(a[:, 0]), a[:, 1]) - b[:, 1]))))
        d = max(diffs) if diffs else float("nan")
        check(d < 15, f"CCT {name}: start curves differ from the poster's by {d:.0f} °C")
        print(f"3c. CCT {name} vs the poster's curves: largest difference {d:.1f} °C")

    # 3d. measured handbook values
    rp = pd.read_csv(RES / "s01_route_pairs.csv").set_index("grade")
    for g, name in (("AISI 1040", "AISI 1040"), ("AISI 4140", "AISI 4140H"), ("AISI 1080", "AISI 1080")):
        r = run({"comp": G[name], "bar_mm": 25})["routes"]
        print(f"3d. {g}: handbook {rp.loc[g, 'UTS_MPa_FC']:.0f} / {rp.loc[g, 'UTS_MPa_AC']:.0f} MPa (annealed / "
              f"normalised), app {r['FC']['UTS_MPa'][0]:.0f} / {r['AC']['UTS_MPa'][0]:.0f}")

    # 4. odd inputs
    odd = {"no carbon": {"C": 0.0}, "1.2 % C": {"C": 1.2, "Mn": 0.4}, "2.5 % Si": {"C": 0.55, "Si": 2.5},
           "304 stainless": {"C": 0.05, "Mn": 1.5, "Si": 0.5, "Cr": 18.5, "Ni": 9.0},
           "D2 tool steel": {"C": 1.55, "Cr": 12.0, "Mo": 0.8, "V": 0.9}, "bad values": {"C": "abc", "Mn": -3, "Cr": 1e9},
           "empty": {}}
    for k, comp in odd.items():
        try:
            r = run({"comp": comp, "bar_mm": "x", "temper_C": None})
            print(f"4. {k:14s} -> {r['steel']['family']}, {'predicted' if r['in_scope'] else 'declined'}"
                  + (f", cautions: {len(r['change']['cautions'])}" if r["in_scope"] else ""))
        except Exception as e:
            check(False, f"odd input {k}: {type(e).__name__}: {e}")

    print("\nPROBLEMS:" if problems else "\nALL CHECKS PASSED")
    for p in problems:
        print("  -", p)
    return not problems


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
