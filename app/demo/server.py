"""Live demo of the project: one clean page that runs offline.

    make demo                         # or:  .venv/bin/python app/demo/server.py
                                      # the page opens at http://127.0.0.1:8050

Pick a steel, then move the bar-size and tempering sliders. The page follows the poster's story:
1 Classify (which family the steel belongs to), 2 Heat-treat (furnace against air cooling, with the
CCT diagram and both cooling paths), 3 Predict (microstructure and properties with 90 % intervals,
for all five treatments). Every number comes from the trained predictor in src/steelml, the same one
behind the report and the poster. Stop it with Ctrl+C.
"""

import argparse
import json
import math
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steelml import categories as cat  # noqa: E402
from steelml import config  # noqa: E402
from steelml.cct import cct_starts, full_comp  # noqa: E402
from steelml.physics import cooling, microstructure, thermo  # noqa: E402
from steelml.predictor import SteelPredictor  # noqa: E402

INPUTS = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Cu", "S", "P"]          # what the page can set
LIMITS = {"C": 2.0, "Mn": 3.0, "Si": 3.0, "Cr": 30.0, "Ni": 30.0, "Mo": 10.0, "V": 2.0, "Cu": 2.0, "S": 0.5, "P": 0.2}
LOCK = threading.Lock()       # one prediction at a time: the kinetics are not written to be thread-safe
SP = None
SCOPE = None


# ------------------------------------------------------------------ helpers
def f(v, nd=None):
    """numpy / pandas scalar -> JSON number (NaN -> None)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


def clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(v) else min(max(v, lo), hi)


def family(code):
    return cat.CATEGORY_LABELS.get(code, str(code))


def grades():
    out = []
    for _, r in SCOPE.iterrows():
        comp = {el: round(float(r[f"{el}_nom"]), 3) for el in INPUTS if pd.notna(r.get(f"{el}_nom"))}
        out.append({"name": r.grade, "family": family(r.category), "comp": comp})
    return {"grades": out, "default": "AISI 1045"}


def make_comp(raw):
    comp = {el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS}
    for el in INPUTS:
        if el in (raw or {}):
            comp[el] = clamp(raw[el], 0.0, LIMITS[el], comp[el])
    return comp


# ------------------------------------------------------------------ the three steps
def steel_json(r):
    return {"family": family(r.category), "confidence": f(r.category_confidence, 3),
            "unknown": bool(r.category_unknown), "rules": family(r.category_rules),
            "rules_agree": bool(r.category_rules_agree), "in_scope": r.category in cat.FERRITE_PEARLITE_SCOPE,
            "Ae1": f(r.k_Ae1, 0), "Ae3": f(r.k_Ae3, 0), "Ms": f(r.k_Ms_bulk, 0), "Bs": f(r.k_Bs, 0),
            "DI_mm": f(r.DI_mm, 0), "CE": f(r.CE_IIW, 2)}


def route_json(r):
    out = {"F": f(r.F_best, 3), "P": f(r.P_best, 3), "B": f(r.B_best, 3), "M": f(r.M_best, 3),
           "C": f(r.get("Fe3C_best", 0.0), 3), "T_aust": f(r.treat_T_C, 0), "rate700": f(r.cool_rate_700, 4),
           "spacing_nm": f(1000 * r.S0_um, 0) if r.P_best > 0.05 else None,
           "grain_um": f(r.d_alpha_um, 0) if r.F_best > 0.05 else None}
    for t in ("UTS_MPa", "YS_MPa", "El_pct", "HB"):
        out[t] = [f(r[f"{t}_pred"], 1), f(r[f"{t}_lo90"], 1), f(r[f"{t}_hi90"], 1)]
    return out


def explain(fc, ac, bar):
    d_uts = ac.UTS_MPa_pred - fc.UTS_MPa_pred
    hard_fc, hard_ac = fc.B_best + fc.M_best, ac.B_best + ac.M_best
    if hard_ac - hard_fc > 0.50:
        head = "A different steel"
        text = (f"A {bar:.0f} mm bar cools fast enough in still air to miss most of the ferrite and pearlite: "
                f"about {100 * hard_ac:.0f} % bainite + martensite forms instead. That is hardenability, and it is "
                "why the strength jumps; such a steel is usually tempered after normalising.")
    elif hard_ac - hard_fc > 0.10:
        head = "Partly bainitic"
        text = (f"A {bar:.0f} mm bar still forms mostly ferrite + pearlite in air, but about {100 * hard_ac:.0f} % "
                "bainite + martensite forms too. A thinner bar cools faster and would form more: this is the mass "
                "effect of a hardenable steel.")
    else:
        head = "Same phases, finer structure"
        parts = []
        if fc.P_best > 0.05:
            parts.append(f"the pearlite is finer ({1000 * fc.S0_um:.0f} → {1000 * ac.S0_um:.0f} nm)")
        if fc.F_best > 0.05:
            parts.append(f"the ferrite grains are smaller ({fc.d_alpha_um:.0f} → {ac.d_alpha_um:.0f} µm)")
        text = ("Both routes give ferrite + pearlite. Air cooling transforms the austenite at a larger "
                "undercooling, so " + " and ".join(parts) + ".") if parts else "Both routes give the same phases."
    cautions = []
    if not bool(fc.in_validity_window):
        cautions.append("outside the window of the critical-temperature equations (for example Si above 1 %)")
    if not bool(ac.microstructure_in_data):
        cautions.append("air cooling gives martensite, which no training steel had: the physics carries this estimate")
    if not bool(fc.in_domain):
        cautions.append("composition outside the training grades: an extrapolation, so the interval is wider")
    return {"head": head, "text": text, "d_uts": f(d_uts, 0), "d_ys": f(ac.YS_MPa_pred - fc.YS_MPa_pred, 0),
            "d_el": f(ac.El_pct_pred - fc.El_pct_pred, 1), "d_hb": f(ac.HB_pred - fc.HB_pred, 0), "cautions": cautions}


def thin(t, T, n=160):
    """Keep about n points of a cooling path, spaced evenly in log time between 1 s and 4e5 s."""
    t, T = np.asarray(t, float), np.asarray(T, float)
    ok = (t >= 0.5) & (t <= 5e5)
    t, T = t[ok], T[ok]
    if len(t) <= n:
        return [[f(a, 3), f(b, 1)] for a, b in zip(t, T)]
    grid = np.logspace(np.log10(max(t[0], 0.5)), np.log10(t[-1]), n)
    return [[f(a, 3), f(b, 1)] for a, b in zip(grid, np.interp(grid, t, T))]


def cct_json(comp, fc, ac, bar):
    fcomp = full_comp(comp)
    T0 = float(thermo.default_austenitising_T(comp, "AC"))
    df = cct_starts(fcomp, float(microstructure.austenite_astm(T0)), T0=T0)
    curves = {}
    for key, frac, name in (("T_ferrite_start", "f_ferrite", "F"), ("T_pearlite_start", "f_pearlite", "P"),
                            ("T_bainite_start", "f_bainite", "B"), ("T_martensite_start", "f_martensite", "Ms")):
        ok = df[key].notna() & (df[frac] >= 0.01)
        pts = sorted(((T0 - T) / rate, T) for T, rate in zip(df.loc[ok, key], df.loc[ok, "rate"]))
        curves[name] = [[f(a, 3), f(b, 1)] for a, b in pts if a > 0]
    last = np.fmin(df["T_pearlite_start"].fillna(np.inf), df["T_bainite_start"].fillna(np.inf))
    ok = df["T_finish"].notna() & (df["T_finish"] < last - 3.0)
    pts = sorted(((T0 - T) / rate, T) for T, rate in zip(df.loc[ok, "T_finish"], df.loc[ok, "rate"]))
    curves["finish"] = [[f(a, 3), f(b, 1)] for a, b in pts if a > 0]
    t_fc, T_fc = cooling.furnace_cooling_path(float(fc.treat_T_C), config.FC_RATE_700C)
    t_ac, T_ac = cooling.air_cooling_path(float(ac.treat_T_C), float(bar))
    return {"curves": curves, "paths": {"FC": thin(t_fc, T_fc), "AC": thin(t_ac, T_ac)}}


LABEL = {"anneal": "Annealed", "normalise": "Normalised", "harden": "Hardened",
         "quench_temper": "Quenched + tempered", "stress_relieve": "Stress relieved"}


def five_json(comp, bar, medium, temper):
    out = SP.compare_treatments(comp, section_mm=bar, medium=medium, temper_T=temper, sr_T=600.0)
    rows = []
    for _, r in out.iterrows():
        row = {"key": r.treatment, "label": LABEL[r.treatment], "medium": str(r.medium),
               "micro": str(r.microstructure), "HV": f(r.HV, 0), "HRC": f(r.get("HRC"), 0),
               "F": f(r.F, 3), "P": f(r.P, 3), "B": f(r.B, 3), "M": f(r.M, 3)}
        if r.treatment != "harden":
            row["UTS"] = [f(r.UTS_MPa_pred, 0), f(r.UTS_MPa_lo90, 0), f(r.UTS_MPa_hi90, 0)]
            row["YS"] = [f(r.YS_MPa_pred, 0), f(r.YS_MPa_lo90, 0), f(r.YS_MPa_hi90, 0)]
            row["El"] = [f(r.El_pct_pred, 0), f(r.El_pct_lo90, 0), f(r.El_pct_hi90, 0)]
        rows.append(row)
    notes = []
    h = out[out.treatment == "harden"].iloc[0]
    if h.M < 0.30:
        notes.append(f"A {bar:.0f} mm bar of this steel forms only {100 * h.M:.0f} % martensite when quenched, so "
                     "tempering changes it little: its hardenability is low.")
    q = out[out.treatment == "quench_temper"].iloc[0]
    if bool(q.get("TME_risk", False)):
        notes.append(f"Tempering at {temper:.0f} °C is inside the tempered-martensite embrittlement window "
                     "(250–400 °C).")
    if bool(q.get("TE_risk", False)):
        notes.append(f"Tempering at {temper:.0f} °C is inside the temper-embrittlement window (375–575 °C): "
                     "cool quickly after tempering.")
    sr = out[out.treatment == "stress_relieve"].iloc[0]
    if bool(sr.get("subcritical", True)):
        notes.append(f"Stress relief at 600 °C: residual stress at most {sr.residual_stress_bound_MPa:.0f} MPa; "
                     "structure unchanged. This step follows EN 1993-1-2 and is not tested against data.")
    flag = str(out.notes.iloc[0]) if pd.notna(out.notes.iloc[0]) else ""
    if flag.strip():
        notes.append(flag)
    return {"rows": rows, "notes": notes, "quench": str(out[out.treatment == "harden"].medium.iloc[0])}


def run(payload):
    comp = make_comp(payload.get("comp"))
    bar = clamp(payload.get("bar_mm"), 5, 300, 25.0)
    temper = clamp(payload.get("temper_C"), 100, 750, 600.0)
    medium = payload.get("medium") if payload.get("medium") in ("oil", "water", "brine") else None
    with LOCK:
        w = SP.what_if(comp, section_mm=bar)
        fc, ac = w.iloc[0], w.iloc[1]
        steel = steel_json(fc)
        if not steel["in_scope"]:
            return {"steel": steel, "in_scope": False,
                    "message": (f"This composition is classified as {steel['family'].lower()}. The models are built for "
                                "carbon and low-alloy steels that form ferrite and pearlite on slow cooling, so the tool "
                                "does not predict it: it says so instead of guessing.")}
        return {"steel": steel, "in_scope": True, "bar_mm": bar, "temper_C": temper,
                "routes": {"FC": route_json(fc), "AC": route_json(ac)}, "change": explain(fc, ac, bar),
                "cct": cct_json(comp, fc, ac, bar), "five": five_json(comp, bar, medium, temper)}


# ------------------------------------------------------------------ web server
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, allow_nan=False), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/grades":
            self._json(200, grades())
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, "not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            self._json(200, run(payload))
        except Exception as e:      # keep the demo alive and show the error on the page
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):
        pass


def main():
    global SP, SCOPE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    print("Loading the trained models ...", flush=True)
    SP = SteelPredictor.load()
    catalog = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv")
    SCOPE = catalog[catalog.in_fp_scope & catalog.standard.eq("AISI/SAE")].sort_values("grade")
    run({"comp": dict(zip(INPUTS, (0.45, 0.75, 0.25))), "bar_mm": 25})      # warm up, so the first click is instant
    for port in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        sys.exit(f"ports {args.port}-{args.port + 9} are all in use")
    url = f"http://127.0.0.1:{port}"
    print(f"Demo ready: {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
