"""Stage 1 - assemble the processed datasets and describe them.

Outputs
  data/processed/*.csv                      (see steelml.data.build)
  results/s01_dataset_summary.csv           one row per dataset
  results/s01_route_pairs.csv               annealed vs normalized pairs per grade
  figures/fig01_handbook_properties_vs_carbon.png
  figures/fig02_route_effect_by_grade.png
  figures/fig26_heat_treatment_practice.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import categories as cat  # noqa: E402
from steelml import config, viz  # noqa: E402
from steelml.data.build import build_all  # noqa: E402

plt = viz.plt


def summary(d):
    rows = [
        ("grade_catalog", len(d["catalog"]), d["catalog"].grade.nunique(), "composition ranges",
         "Summit Steel (SAE J403/J1268), Wikipedia, Zenodo 17704410", "classification"),
        ("properties_handbook", len(d["primary"]), d["primary"].grade.nunique(), "typical values",
         "ASM Handbook / Bethlehem Steel (via Engineers Edge)", "property models (training)"),
        ("properties_secondary", len(d["secondary"]), d["secondary"].grade.nunique(),
         "typical + specification minima", "Zenodo 17704410 (MatWeb-derived)", "inter-source check"),
        ("structural_external", int(d["external"].usable.sum()), int(d["external"].usable.sum()),
         "measured", "Zenodo 19000367 (Kingsley et al. 2026)", "external validation"),
        ("heats_steelbench", len(d["steelbench"]), d["steelbench"].grade_id.nunique(),
         "measured heat chemistry", "Zenodo 18530558 (SteelBench)", "classifier external test"),
    ]
    col = pd.read_csv(config.RAW_DIR / "collins2023_hardness.csv")
    rows.append(("collins2023", len(col), col.steel.nunique(), "measured",
                 "Zenodo 7770264 (Collins et al. 2023)", "transformation + cooling-rate validation"))
    return pd.DataFrame(rows, columns=["dataset", "rows", "steels_or_grades", "value_type", "source", "used_for"])


def fig_properties_vs_carbon(p):
    targets = [("UTS_MPa", "Tensile strength (MPa)"), ("YS_MPa", "Yield strength (MPa)"),
               ("El_pct", "Total elongation (%)"), ("HB", "Brinell hardness (HB)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True)
    plain = p.category.isin(["low_carbon", "medium_carbon", "high_carbon"])
    from steelml.physics import thermo
    base = {el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS} | {"C": 0.5, "Mn": 0.75, "Si": 0.22}
    ce = float(thermo.eutectoid_point(base)[0])
    for ax, (t, lab) in zip(axes.ravel(), targets):
        for route in ("FC", "AC"):
            col = viz.ROUTE_COLORS[route]
            s = p[(p.route == route) & plain].sort_values("C")
            ax.plot(s.C, s[t], "-o", color=col, ms=6, mec=viz.SURFACE, mew=1.2, zorder=3,
                    label=f"{viz.ROUTE_LABELS[route]}, plain carbon")
            a = p[(p.route == route) & ~plain]
            ax.plot(a.C, a[t], "o", mfc="none", mec=col, mew=1.4, ms=6, zorder=2,
                    label=f"{viz.ROUTE_LABELS[route]}, alloy / free-machining")
        ax.set_title(lab)
        ax.axvline(ce, color=viz.AXIS, lw=0.8, zorder=1)
        ax.text(ce + 0.005, ax.get_ylim()[1], f" eutectoid\n ({ce:.2f} % C at 0.75 % Mn)", va="top", ha="left",
                color=viz.MUTED, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Carbon (wt%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Handbook data: furnace cooling vs air cooling of 33 AISI grades", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return viz.savefig(fig, "fig01_handbook_properties_vs_carbon")


def route_pairs(p):
    fc = p[p.route == "FC"].set_index("grade")
    ac = p[p.route == "AC"].set_index("grade")
    g = fc.index.intersection(ac.index)
    out = pd.DataFrame({"category": fc.loc[g, "category"], "C": fc.loc[g, "C"]})
    for t in ("UTS_MPa", "YS_MPa", "El_pct", "HB"):
        out[f"{t}_FC"] = fc.loc[g, t]
        out[f"{t}_AC"] = ac.loc[g, t]
        out[f"{t}_delta"] = ac.loc[g, t] - fc.loc[g, t]
    return out.reset_index()


def fig_route_effect(pairs):
    order = list(cat.CATEGORIES)
    pairs = pairs.assign(o=pairs.category.map({c: i for i, c in enumerate(order)})).sort_values(["o", "C"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 8.2), sharey=True)
    y = np.arange(len(pairs))
    for ax, (t, lab) in zip(axes, [("UTS_MPa", "Tensile strength (MPa)"), ("El_pct", "Total elongation (%)")]):
        ax.hlines(y, pairs[f"{t}_FC"], pairs[f"{t}_AC"], color=viz.AXIS, lw=1.6, zorder=1)
        ax.plot(pairs[f"{t}_FC"], y, "o", color=viz.ROUTE_COLORS["FC"], ms=7, mec=viz.SURFACE, mew=1.2,
                label="Furnace cooled (annealed)", zorder=3)
        ax.plot(pairs[f"{t}_AC"], y, "o", color=viz.ROUTE_COLORS["AC"], ms=7, mec=viz.SURFACE, mew=1.2,
                label="Air cooled (normalized)", zorder=3)
        ax.set_title(lab)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{g.replace('AISI ', '')}  ({cat.CATEGORY_LABELS[c]})" for g, c in
                           zip(pairs.grade, pairs.category)], fontsize=8)
    axes[0].invert_yaxis()
    axes[1].legend(loc="upper left")
    fig.suptitle("What air cooling changes, grade by grade (measured, ASM/Bethlehem data)", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig02_route_effect_by_grade")


def fig_heat_treatment_practice(p):
    """The steel corner of the Fe-C diagram (Grange Ae3/Ae1 and the Acm fit,
    plain carbon with 0.75 Mn) with the temperatures at which the handbook
    grades were actually annealed and normalised, and the default
    austenitising rules the tool uses (derived from that practice)."""
    from steelml.physics import thermo
    C = np.linspace(0.0, 1.2, 241)
    base = {el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS}
    base.update(Mn=0.75, Si=0.22)
    comp = {k: np.full_like(C, v) for k, v in base.items()}
    comp["C"] = C
    ce, te = thermo.eutectoid_point(base | {"C": 0.5})
    ce, te = float(ce), float(te)
    a3, a1, acm = thermo.ae3(comp), np.full_like(C, float(thermo.ae1(base))), thermo.acm(comp)
    upper = thermo.upper_critical(comp)
    hypo = C <= ce
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.plot(C, upper, color=viz.INK, lw=1.6)
    fit = (~hypo) & (C <= 1.0)
    ax.plot(C[fit], acm[fit], "--", color=viz.MUTED, lw=1.1,
            label="Acm fit inside the kinetic model (Collins et al.), low above 0.9 % C")
    ax.plot(C, a1, color=viz.INK, lw=1.2)
    T_anneal = thermo.default_austenitising_T(comp)
    T_norm = thermo.default_austenitising_T(comp, np.full(C.shape, "AC"))
    ax.plot(C, T_anneal, "--", color=viz.ROUTE_COLORS["FC"], lw=1.4,
            label="tool default, full annealing: max(Ae3 + 50, Ae1 + 65) °C")
    ax.plot(C, T_norm, "--", color=viz.ROUTE_COLORS["AC"], lw=1.4,
            label="tool default, normalising: upper critical (Ae3 or Acm) + 110 °C, at least 870 °C")
    plain = p.category.isin(["low_carbon", "medium_carbon", "high_carbon"])
    for cond, route, mk in (("annealed", "FC", "o"), ("normalized", "AC", "s")):
        for sel, filled in ((plain, True), (~plain, False)):
            q = p[(p.condition == cond) & sel]
            ax.plot(q.C, q.treat_T_C, mk, ms=6.5, color=viz.ROUTE_COLORS[route],
                    mfc=viz.ROUTE_COLORS[route] if filled else viz.SURFACE, mec=viz.ROUTE_COLORS[route], mew=1.3,
                    label=f"handbook grades, {'annealed' if cond == 'annealed' else 'normalised'} "
                          f"({'plain carbon' if filled else 'alloy and free-machining'})")
    ax.text(0.06, 975, "γ (austenite)", fontsize=10, color=viz.INK2)
    ax.text(0.05, 742, "α + γ", fontsize=10, color=viz.INK2)
    ax.text(0.86, 728, "γ + Fe$_3$C", fontsize=9.5, color=viz.INK2)
    ax.text(0.30, 690, "α + pearlite", fontsize=10, color=viz.INK2)
    ax.text(0.86, 690, "pearlite + Fe$_3$C", fontsize=10, color=viz.INK2)
    ax.text(1.19, float(a1[-1]) + 4, "Ae1", fontsize=9, color=viz.MUTED, ha="right")
    ax.text(0.01, float(a3[0]) - 24, "Ae3", fontsize=9, color=viz.MUTED)
    ax.text(1.17, float(upper[-1]) - 22, "Acm", fontsize=9, color=viz.MUTED, ha="right")
    ax.set_xlim(0, 1.2)
    ax.set_ylim(660, 1000)
    ax.set_xlabel("Carbon (wt%)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    ax.set_title(f"Annealing and normalising temperatures in practice (boundaries for 0.75 % Mn; eutectoid "
                 f"{ce:.2f} % C at {te:.0f} °C)")
    fig.tight_layout()
    return viz.savefig(fig, "fig26_heat_treatment_practice")


def main():
    config.ensure_dirs()
    d = build_all()
    s = summary(d)
    s.to_csv(config.RESULTS_DIR / "s01_dataset_summary.csv", index=False)
    print(s.to_string(index=False))
    p = d["primary"]
    pairs = route_pairs(p)
    pairs.to_csv(config.RESULTS_DIR / "s01_route_pairs.csv", index=False)
    print("\nMean change on air cooling (normalized - annealed), 33 grades:")
    print(pairs[[c for c in pairs.columns if c.endswith("_delta")]].describe().round(1).to_string())
    fig_properties_vs_carbon(p)
    fig_route_effect(pairs)
    fig_heat_treatment_practice(p)
    print("figures written")


if __name__ == "__main__":
    main()
