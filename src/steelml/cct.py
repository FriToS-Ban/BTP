"""CCT diagrams drawn the way metallurgists draw them.

A model CCT diagram for constant-rate cooling (start curves of ferrite,
pearlite and bainite, the Ms line and the end of the diffusional
transformation), with the furnace-cooling and still-air cooling paths drawn
over it and the microstructure and hardness each path gives. Used by stage 5
(figures 9 and 9b) and by the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, viz
from .features import featurize
from .physics import cooling, kinetics, microstructure, thermo

PHASE_LETTER = {"ferrite": "F", "pearlite": "P", "bainite": "B", "martensite": "M"}

# Six steels that span the categories (nominal SAE compositions)
REPRESENTATIVE = {
    "AISI 1020 (low C)": dict(C=0.20, Mn=0.45, Si=0.22),
    "AISI 1045 (medium C)": dict(C=0.46, Mn=0.75, Si=0.22),
    "AISI 1080 (eutectoid)": dict(C=0.81, Mn=0.75, Si=0.22),
    "AISI 4140 (Cr-Mo)": dict(C=0.40, Mn=0.875, Si=0.25, Cr=0.975, Mo=0.20),
    "AISI 4340 (Ni-Cr-Mo)": dict(C=0.40, Mn=0.70, Si=0.25, Ni=1.80, Cr=0.80, Mo=0.25),
    "AISI 8620 (Ni-Cr-Mo, low C)": dict(C=0.20, Mn=0.80, Si=0.25, Ni=0.55, Cr=0.50, Mo=0.20),
}


def full_comp(comp):
    """Composition with the usual residuals filled in (as for catalogue grades)."""
    d = {el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS}
    d.update(comp)
    return d


def cct_starts(comp, G=7.5, rates=None, T0=None):
    """Constant-rate simulations from T0 (default Ae3 + 50 C)."""
    if rates is None:
        rates = np.logspace(-3, 2.5, 72)
    if T0 is None:
        T0 = float(thermo.ae3(comp)) + 50.0
    rows = []
    for r in rates:
        t, T = cooling.constant_rate_path(T0, r)
        d = kinetics.simulate(comp, G, t, T)
        d.update(rate=r, T0=T0)
        rows.append(d)
    return pd.DataFrame(rows)


def route_microstructures(comp, predictor=None, section_mm=config.AC_DEFAULT_DIAMETER_MM,
                          fc_rate=config.FC_RATE_700C, T_aust=None):
    """Reported microstructure and hardness after furnace cooling and after air
    cooling a bar, each from its usual austenitising temperature. With a
    predictor the hardness is the production model's HB and 90 % interval;
    without one it is the Maynier hardness (physics only)."""
    from .physics import properties as pp
    from .predictor import best_estimate_fractions
    rows = []
    for route in ("FC", "AC"):
        d = full_comp(comp)
        d.update(route=route, section_mm=section_mm, fc_rate=fc_rate)
        d["treat_T_C"] = float(thermo.default_austenitising_T(d, route)) if T_aust is None else float(T_aust)
        rows.append(d)
    X = pd.DataFrame(rows)
    if predictor is not None:
        F = predictor.predict(X)
        b = {k: F[k].to_numpy() for k in ("F_best", "P_best", "B_best", "M_best", "Fe3C_best")}
        hard = {"value": F.HB_pred.to_numpy(), "lo": F.HB_lo90.to_numpy(), "hi": F.HB_hi90.to_numpy(),
                "unit": "HB"}
    else:
        F = featurize(X)
        b = best_estimate_fractions(F)
        f_fp = b["F_best"] + b["P_best"] + b["Fe3C_best"]
        hv = pp.hv_mixture(X, F.cool_rate_700.to_numpy() * 3600.0, f_fp, b["B_best"], b["M_best"])
        hard = {"value": np.asarray(hv), "lo": np.full(2, np.nan), "hi": np.full(2, np.nan), "unit": "HV"}
    out = pd.DataFrame({"route": X.route, "T0": X.treat_T_C, "rate700": F.cool_rate_700.to_numpy(),
                        "F": b["F_best"], "P": b["P_best"] + b["Fe3C_best"], "B": b["B_best"],
                        "M": b["M_best"], "H": hard["value"], "H_lo": hard["lo"], "H_hi": hard["hi"],
                        "S0_um": F.S0_um.to_numpy(), "d_alpha_um": F.d_alpha_um.to_numpy()})
    out.attrs["unit"] = hard["unit"]
    return out.set_index("route")


def micro_text(r, unit):
    """'F 39% + P 61%, 180 HB' for one route row of route_microstructures."""
    parts = [f"{k} {100 * r[k]:.0f}%" for k in ("F", "P", "B", "M") if r[k] >= 0.015]
    return " + ".join(parts) + f", {r.H:.0f} {unit}"


def draw_cct(ax, name, comp, predictor=None, legend_labels=True, fontsize=9.0,
             section_mm=config.AC_DEFAULT_DIAMETER_MM, fc_rate=config.FC_RATE_700C, T_aust=None):
    """Draw one CCT diagram with both routes on ``ax``; returns the route table."""
    comp = full_comp(comp)
    T0 = float(thermo.default_austenitising_T(comp, "AC")) if T_aust is None else float(T_aust)
    G = float(microstructure.austenite_astm(T0))
    df = cct_starts(comp, G, T0=T0)
    t_of = lambda key: (T0 - df[key]) / df["rate"]  # noqa: E731
    label_pos = {}
    for key, frac, ph in (("T_ferrite_start", "f_ferrite", "ferrite"),
                          ("T_pearlite_start", "f_pearlite", "pearlite"),
                          ("T_bainite_start", "f_bainite", "bainite"),
                          ("T_martensite_start", "f_martensite", "martensite")):
        ok = df[key].notna() & (df[frac] >= 0.01)   # draw only where the constituent forms
        if ok.sum() < 2:
            continue
        t = t_of(key)[ok]
        ax.plot(t, df.loc[ok, key], "-", color=viz.PHASE_COLORS[ph], lw=2.4,
                label=(f"{ph} start" if ph != "martensite" else "Ms") if legend_labels else None)
        i = len(t) // 2 if ph != "martensite" else 0
        label_pos[ph] = (t.iloc[i], df.loc[ok, key].iloc[i])
    # end of the diffusional transformation, where it is distinct from the last start curve
    last_start = np.fmin(df["T_pearlite_start"].fillna(np.inf), df["T_bainite_start"].fillna(np.inf))
    fin = df["T_finish"].notna() & (df["T_finish"] < last_start - 3.0)
    if fin.sum() >= 2:
        ax.plot(t_of("T_finish")[fin], df.loc[fin, "T_finish"], "--", color=viz.INK2, lw=1.2,
                label="transformation finish (99 %)" if legend_labels else None)
    for ph, (tx, Ty) in label_pos.items():
        ax.text(tx, Ty - 32, PHASE_LETTER[ph], color=viz.INK, fontsize=fontsize + 2, fontweight="bold",
                ha="center", va="center")
    ax.text(1.5, T0 - 35, "A", color=viz.INK, fontsize=fontsize + 2, fontweight="bold")
    rm = route_microstructures(comp, predictor, section_mm=section_mm, fc_rate=fc_rate, T_aust=T_aust)
    unit = rm.attrs["unit"]
    styles = {"FC": (viz.INK, f"furnace cooling path ({fc_rate:.3g} K/s at 700 °C)"),
              "AC": (viz.MUTED, f"air cooling path, {section_mm:.0f} mm bar")}
    for route in ("FC", "AC"):
        T0r = float(rm.loc[route, "T0"])
        t, T = (cooling.furnace_cooling_path(T0r, fc_rate) if route == "FC"
                else cooling.air_cooling_path(T0r, section_mm))
        ok = t > 0
        color, lab = styles[route]
        ax.plot(t[ok], T[ok], color=color, lw=1.4, label=lab if legend_labels else None)
        j = np.argmin(np.abs(T[ok] - 160.0))
        ax.annotate(f"{route}\n{rm.loc[route, 'H']:.0f} {unit}", (t[ok][j], T[ok][j]), xytext=(6, 0),
                    textcoords="offset points", fontsize=fontsize - 0.5, color=viz.INK, va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc=viz.SURFACE, ec=color, lw=0.9))
    ae1, ae3 = float(thermo.ae1(comp)), float(thermo.ae3(comp))
    ax.axhline(ae1, color=viz.AXIS, lw=0.8)
    ax.axhline(ae3, color=viz.AXIS, lw=0.8)
    ax.text(2.5e5, ae3 + 5, "Ae3", color=viz.MUTED, fontsize=fontsize - 1, ha="right")
    ax.text(2.5e5, ae1 - 18, "Ae1", color=viz.MUTED, fontsize=fontsize - 1, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(1, 4e5)
    ax.set_ylim(100, T0 + 15)
    ax.set_title(f"{name}\nFC: {micro_text(rm.loc['FC'], unit)}\nAC: {micro_text(rm.loc['AC'], unit)}",
                 fontsize=fontsize + 0.5)
    return rm


def legend_below(fig, axes, ncol=4, y=-0.02):
    h, l = [], []
    for ax in np.ravel(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))
