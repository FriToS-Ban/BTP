"""Stage 5 - furnace cooling versus air cooling, category by category.

For every catalogue grade that decomposes to ferrite + pearlite on slow
cooling (plain-carbon, free-machining, C-Mn and low-alloy categories):
  * microstructure after furnace cooling (0.005 K/s) and after still-air
    cooling of 25 mm and 50 mm bars (transformation model);
  * properties with 90 % conformal intervals (production predictor);
  * a by-category summary of what air cooling changes.
Plus a plain-carbon sweep (the classic fine-versus-coarse pearlite chart,
regenerated with uncertainty) and a section-size study.

Outputs: results/s05_*.csv, figures fig15-fig18.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import categories as cat  # noqa: E402
from steelml import config, viz  # noqa: E402
from steelml.cct import REPRESENTATIVE, draw_cct, legend_below  # noqa: E402
from steelml.physics import thermo  # noqa: E402
from steelml.predictor import SteelPredictor  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt
PROPS = [("UTS_MPa", "Tensile strength (MPa)"), ("YS_MPa", "Yield strength (MPa)"),
         ("El_pct", "Total elongation (%)"), ("HB", "Brinell hardness (HB)")]


def regime(row, tol=0.05):
    """Short microstructure label from the best-estimate fractions."""
    parts = []
    for k, name in (("F_best", "F"), ("P_best", "P"), ("Fe3C_best", "Fe3C"),
                    ("B_best", "B"), ("M_best", "M")):
        if row[k] >= tol:
            parts.append(name)
    return "+".join(parts) if parts else "?"


def catalogue_routes(sp, catalog):
    scope = catalog[catalog.in_fp_scope].copy()
    base = scope[["grade", "category"] + [f"{el}_nom" for el in config.ELEMENTS]].rename(
        columns={f"{el}_nom": el for el in config.ELEMENTS})
    frames = []
    for route, D in (("FC", 25.0), ("AC", 25.0), ("AC", 50.0)):
        q = base.copy()
        q["route"] = route
        q["section_mm"] = D
        q["case"] = "FC" if route == "FC" else f"AC{int(D)}"
        frames.append(q)
    # air cooling from the *annealing* temperature: separates the effect of
    # the faster cooling from that of the hotter austenitisation (normalising
    # is done about 60 C hotter than annealing)
    q = base.copy()
    q["route"], q["section_mm"], q["case"] = "AC", 25.0, "AC25_from_TA"
    q["treat_T_C"] = np.asarray(thermo.default_austenitising_T(q), float)
    frames.append(q)
    X = pd.concat(frames, ignore_index=True)
    out = sp.predict(X)
    out["case"] = X["case"].to_numpy()
    out["regime"] = out.apply(regime, axis=1)
    out["eutectoid_class"] = thermo.eutectoid_class(X)
    return out


def category_summary(out):
    rows = []
    for c in cat.CATEGORIES:
        d = out[out.category_x == c] if "category_x" in out else out[out["category_true"] == c]
        if d.empty:
            continue
        fc, ac = d[d.case == "FC"], d[d.case == "AC25"]
        ac50 = d[d.case == "AC50"]
        acta = d[d.case == "AC25_from_TA"]
        fp = lambda s: float(np.mean(s.B_best + s.M_best < 0.05))  # noqa: E731
        rec = {"category": cat.CATEGORY_LABELS[c], "grades": len(fc),
               "FC_ferrite_pearlite_share": fp(fc), "AC25_ferrite_pearlite_share": fp(ac),
               "AC50_ferrite_pearlite_share": fp(ac50),
               "FC_pearlite_mean": fc.P_best.mean(), "AC25_pearlite_mean": ac.P_best.mean(),
               "AC25_pearlite_kinetic_mean": ac.k_f_pearlite.mean(),
               "AC25_bainite_martensite_mean": (ac.B_best + ac.M_best).mean(),
               "FC_regime_mode": fc.regime.mode().iat[0], "AC25_regime_mode": ac.regime.mode().iat[0],
               "FC_S0_um_median": fc.S0_um.median(), "AC25_S0_um_median": ac.S0_um.median()}
        rec.update({"AC25_pearlite_upper_mean": ac.P_upper.mean(), "AC50_pearlite_mean": ac50.P_best.mean(),
                    "AC50_pearlite_upper_mean": ac50.P_upper.mean(),
                    "AC50_bainite_martensite_mean": (ac50.B_best + ac50.M_best).mean(),
                    "AC50_S0_um_median": ac50.S0_um.median(),
                    "FC_d_alpha_um_median": fc.d_alpha_um.median(), "AC25_d_alpha_um_median": ac.d_alpha_um.median(),
                    "AC50_d_alpha_um_median": ac50.d_alpha_um.median(), "DI_mm_median": fc.DI_mm.median(),
                    "in_validity_share": float(np.mean(fc.in_validity_window))})
        for case, frame in (("FC", fc), ("AC25", ac), ("AC50", ac50)):
            fp_dom = (frame.B_best + frame.M_best) < 0.1
            rec[f"ITT_{case}_median_C"] = float(np.median(frame.ITT_best_C[fp_dom])) if fp_dom.any() else np.nan
        for t, _ in PROPS:
            rec[f"{t}_FC_median"] = fc[f"{t}_pred"].median()
            rec[f"{t}_AC25_median"] = ac[f"{t}_pred"].median()
            rec[f"{t}_AC50_median"] = ac50[f"{t}_pred"].median()
            rec[f"{t}_delta_median"] = float(np.median(ac[f"{t}_pred"].to_numpy() - fc[f"{t}_pred"].to_numpy()))
            rec[f"{t}_delta50_median"] = float(np.median(ac50[f"{t}_pred"].to_numpy() - fc[f"{t}_pred"].to_numpy()))
            # split of the 25 mm change: faster cooling at the annealing temperature,
            # then the hotter normalising austenitisation
            rec[f"{t}_delta_cooling_median"] = float(np.median(acta[f"{t}_pred"].to_numpy() - fc[f"{t}_pred"].to_numpy()))
            rec[f"{t}_delta_temperature_median"] = float(np.median(ac[f"{t}_pred"].to_numpy()
                                                                  - acta[f"{t}_pred"].to_numpy()))
        rec["n_FC_median"] = fc.n_est.median()
        rec["n_AC25_median"] = ac.n_est.median()
        rec["uniformEl_FC_median"] = fc.uniform_El_est_pct.median()
        rec["uniformEl_AC25_median"] = ac.uniform_El_est_pct.median()
        rec["in_domain_share"] = float(np.mean(fc.in_domain))
        rows.append(rec)
    return pd.DataFrame(rows)


def carbon_sweep(sp):
    Cs = np.round(np.arange(0.10, 1.001, 0.025), 3)
    rows = []
    for route in ("FC", "AC"):
        for C in Cs:
            comp = {el: config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS}
            comp.update(C=C, Mn=0.75, Si=0.25)
            rows.append({**comp, "route": route, "section_mm": 25.0})
    X = pd.DataFrame(rows)
    out = sp.predict(X)
    lr = thermo.lever_rule(X)
    out["f_pearlite_lever"] = lr["f_pearlite_eq"]
    return out


def composition_noise_floor(sp, catalog, n_heats=60, seed=config.RANDOM_SEED):
    """How much do the predictions move when a handbook grade's composition
    varies inside its own specification range? The training data give one
    value per grade at the nominal composition, so this spread is a floor
    under the error any model of nominal compositions can reach."""
    from steelml.data.catalog import sample_heats
    hb = pd.read_csv(config.PROCESSED_DIR / "properties_handbook.csv")
    grades = catalog[catalog.grade.isin(hb.catalog_grade.dropna().unique())]
    heats = sample_heats(grades, n_heats, np.random.default_rng(seed))
    rows = []
    for route, D in (("FC", 25.0), ("AC", config.HANDBOOK_AC_SECTION_MM)):
        X = heats[["grade"] + config.ELEMENTS].copy()
        X["route"], X["section_mm"] = route, D
        out = sp.predict(X.drop(columns=["grade"]))
        out["grade"] = heats.grade.to_numpy()
        for t, _ in PROPS:
            sd = out.groupby("grade")[f"{t}_pred"].std()
            rows.append({"route": route, "target": t, "grades": int(sd.size), "heats_per_grade": n_heats,
                         "median_sd": float(sd.median()), "rms_sd": float(np.sqrt(np.mean(sd ** 2))),
                         "max_sd": float(sd.max())})
    return pd.DataFrame(rows)


def section_study(sp):
    grades = {"AISI 1020": dict(C=0.205, Mn=0.45, Si=0.22), "AISI 1045": dict(C=0.465, Mn=0.75, Si=0.22),
              "AISI 4140": dict(C=0.405, Mn=0.875, Si=0.25, Cr=0.975, Mo=0.20),
              "AISI 4340": dict(C=0.405, Mn=0.725, Si=0.25, Ni=1.775, Cr=0.80, Mo=0.25)}
    rows = []
    for g, comp in grades.items():
        for D in [10, 15, 25, 35, 50, 75, 100, 150]:
            rows.append({**comp, "grade": g, "route": "AC", "section_mm": float(D)})
        rows.append({**comp, "grade": g, "route": "FC", "section_mm": 25.0})
    X = pd.DataFrame(rows)
    out = sp.predict(X)
    out["grade"] = X.grade.to_numpy()
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_carbon_sweep(cs, c_range):
    lo_c, hi_c = c_range
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 3)

    def shade(ax):
        ax.axvspan(cs.C.min(), lo_c, color=viz.GRID, alpha=0.6, lw=0)
        if hi_c < cs.C.max():
            ax.axvspan(hi_c, cs.C.max(), color=viz.GRID, alpha=0.6, lw=0)

    ax0 = fig.add_subplot(gs[0, 0])
    shade(ax0)
    for route in ("FC", "AC"):
        d = cs[cs.route == route]
        ax0.plot(d.C, 100 * d.P_best, "-", color=viz.ROUTE_COLORS[route], label=f"{viz.ROUTE_LABELS[route]}")
    d = cs[cs.route == "AC"]
    ax0.plot(d.C, 100 * d.k_f_pearlite, "-", color=viz.ROUTE_COLORS["AC"], lw=1, alpha=0.6,
             label="Air cooled, raw kinetic model (upper bound)")
    d = cs[cs.route == "FC"]
    ax0.plot(d.C, 100 * d.f_pearlite_lever, "-", color=viz.INK, lw=1, label="Lever rule (equilibrium)")
    ax0.set_title("Pearlite (%), best estimate")
    ax0.set_xlabel("Carbon (wt%)")
    ax0.legend(loc="lower right", fontsize=7.5)
    ax1 = fig.add_subplot(gs[1, 0])
    shade(ax1)
    for route in ("FC", "AC"):
        d = cs[(cs.route == route) & (cs.k_f_pearlite > 0.05)]
        ax1.plot(d.C, 1000 * d.S0_um, "-", color=viz.ROUTE_COLORS[route])
    ax1.set_title("Pearlite interlamellar spacing (nm)")
    ax1.set_xlabel("Carbon (wt%)")
    for i, (t, lab) in enumerate(PROPS):
        ax = fig.add_subplot(gs[i // 2, 1 + i % 2])
        shade(ax)
        for route in ("FC", "AC"):
            d = cs[cs.route == route]
            col = viz.ROUTE_COLORS[route]
            ax.fill_between(d.C, d[f"{t}_lo90"], d[f"{t}_hi90"], color=col, alpha=0.15, lw=0)
            ax.plot(d.C, d[f"{t}_pred"], "-", color=col, label=viz.ROUTE_LABELS[route])
        ce, _ = thermo.eutectoid_point({"C": 0.8, "Mn": 0.75, "Si": 0.25})
        ax.axvline(float(ce), color=viz.AXIS, lw=0.8)
        ax.set_title(lab)
        ax.set_xlabel("Carbon (wt%)")
    fig.suptitle("Plain-carbon steels (0.75 Mn, 0.25 Si): furnace vs air cooling with 90 % intervals "
                 "(grey: outside the carbon range of the training data)", x=0.01, ha="left", fontweight="bold",
                 fontsize=10.5)
    h, l = fig.axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return viz.savefig(fig, "fig16_carbon_sweep_fc_vs_ac")


def fig_section(ss):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    grades = list(dict.fromkeys(ss.grade))
    for ax, (t, lab) in zip(axes, [PROPS[0], PROPS[2], PROPS[3]]):
        for i, g in enumerate(grades):
            d = ss[(ss.grade == g) & (ss.route == "AC")].sort_values("section_mm")
            col = viz.SLOTS[[2, 4, 6, 7][i]]
            ax.fill_between(d.section_mm, d[f"{t}_lo90"], d[f"{t}_hi90"], color=col, alpha=0.12, lw=0)
            ax.plot(d.section_mm, d[f"{t}_pred"], "-o", color=col, ms=4, mec=viz.SURFACE, label=g)
            f = ss[(ss.grade == g) & (ss.route == "FC")]
            ax.axhline(float(f[f"{t}_pred"].iloc[0]), color=col, lw=1.2, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100, 150], ["10", "25", "50", "100", "150"])
        ax.set_xlabel("Diameter of the air-cooled bar (mm)")
        ax.set_title(lab)
    axes[1].text(0.02, 0.02, "dashed: the same steel furnace cooled", transform=axes[1].transAxes, fontsize=8,
                 color=viz.MUTED)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Section size: a thick air-cooled bar drifts towards the furnace-cooled state",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig17_section_size_effect")


def fig_category_effects(out):
    order = [c for c in cat.CATEGORIES if c in cat.FERRITE_PEARLITE_SCOPE]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, (t, lab) in zip(axes, [PROPS[0], PROPS[1], PROPS[2]]):
        for i, c in enumerate(order):
            fc = out[(out.category_true == c) & (out.case == "FC")][f"{t}_pred"].to_numpy()
            ac = out[(out.category_true == c) & (out.case == "AC25")][f"{t}_pred"].to_numpy()
            if len(fc) == 0:
                continue
            delta = ac - fc
            jitter = (np.random.default_rng(i).random(len(delta)) - 0.5) * 0.35
            ax.plot(delta, i + jitter, "o", ms=3.5, color=viz.ORANGE, alpha=0.55, mec="none")
            ax.plot([np.median(delta)], [i], "|", ms=16, mew=2.4, color=viz.INK)
        ax.axvline(0, color=viz.AXIS, lw=1)
        ax.set_title(f"Change on air cooling: {lab}")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), [cat.CATEGORY_LABELS[c] for c in order])
    axes[0].invert_yaxis()
    fig.suptitle("Predicted change from furnace cooling to air cooling (25 mm bar), every catalogue grade "
                 "(dots) and category median (bar)", x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return viz.savefig(fig, "fig18_category_route_effects")


def fig_regimes(out):
    order = [c for c in cat.CATEGORIES if c in cat.FERRITE_PEARLITE_SCOPE]
    cases = ["FC", "AC50", "AC25"]
    labels = {"FC": "Furnace cooled", "AC50": "Air cooled, 50 mm", "AC25": "Air cooled, 25 mm"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    phases = [("F_best", "ferrite"), ("P_best", "pearlite"), ("Fe3C_best", "cementite"),
              ("B_best", "bainite"), ("M_best", "martensite")]
    for ax, case in zip(axes, cases):
        left = np.zeros(len(order))
        for k, name in phases:
            vals = np.array([out[(out.category_true == c) & (out.case == case)][k].mean() for c in order])
            ax.barh(range(len(order)), vals, left=left, color=viz.PHASE_COLORS[name], height=0.66,
                    edgecolor=viz.SURFACE, linewidth=1.5,
                    label=name.capitalize() if name != "cementite" else "Proeutectoid Fe3C")
            left += vals
        ax.set_title(labels[case])
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Mean volume fraction")
    axes[0].set_yticks(range(len(order)), [cat.CATEGORY_LABELS[c] for c in order])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Microstructure by category after furnace cooling and after air cooling",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig19_category_microstructures")


def fig_cct_grid(sp):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.4), sharey=True)
    rows = []
    for ax, (name, comp) in zip(axes.ravel(), REPRESENTATIVE.items()):
        rm = draw_cct(ax, name, comp, predictor=sp)
        rows.append(rm.assign(steel=name).reset_index())
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature (°C)")
    legend_below(fig, axes, ncol=4)
    fig.suptitle("Model CCT diagrams (austenitised at the usual normalising temperature) with the furnace and "
                 "air cooling paths; titles give the reported microstructure and the predicted hardness",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    viz.savefig(fig, "fig09_cct_with_cooling_paths")
    return pd.concat(rows, ignore_index=True)


def fig_cct_pair(sp):
    """Two steels side by side at slide size: the same cooling paths give a
    ferrite-pearlite 1045 and a bainitic 4140."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True)
    for ax, name in zip(axes, ("AISI 1045 (medium C)", "AISI 4140 (Cr-Mo)")):
        draw_cct(ax, name, REPRESENTATIVE[name], predictor=sp, fontsize=10.5)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Temperature (°C)")
    legend_below(fig, axes, ncol=3, y=-0.04)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return viz.savefig(fig, "fig09b_cct_1045_vs_4140")


def main():
    config.ensure_dirs()
    sp = SteelPredictor.load()
    cct = fig_cct_grid(sp)
    cct.to_csv(config.RESULTS_DIR / "s05_cct_representative.csv", index=False)
    fig_cct_pair(sp)
    catalog = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv")
    out = catalogue_routes(sp, catalog)
    out = out.rename(columns={"category": "category_pred"})
    out["category_true"] = np.tile(catalog[catalog.in_fp_scope].category.to_numpy(), out["case"].nunique())
    keep = ["grade", "category_true", "category_pred", "case", "regime", "eutectoid_class", "cool_rate_700",
            "k_f_ferrite", "k_f_pearlite", "k_f_cementite_pro", "k_f_bainite", "k_f_martensite",
            "F_best", "P_best", "Fe3C_best", "B_best", "M_best", "S0_um", "d_alpha_um", "in_domain"] + [f"{t}_{s}" for t, _ in PROPS for s in ("pred", "lo90", "hi90")] + \
        ["n_est", "uniform_El_est_pct", "uniform_El_lo90", "uniform_El_hi90"]
    out[keep].to_csv(config.RESULTS_DIR / "s05_catalogue_fc_vs_ac.csv", index=False)
    summ = category_summary(out)
    summ.to_csv(config.RESULTS_DIR / "s05_category_summary.csv", index=False)
    print(summ[["category", "grades", "FC_regime_mode", "AC25_regime_mode", "FC_pearlite_mean",
                "AC25_pearlite_mean", "AC25_ferrite_pearlite_share", "UTS_MPa_delta_median",
                "YS_MPa_delta_median", "El_pct_delta_median"]].round(2).to_string(index=False))
    nf = composition_noise_floor(sp, catalog)
    nf.to_csv(config.RESULTS_DIR / "s05_composition_noise_floor.csv", index=False)
    print(nf.round(1).to_string(index=False))
    cs = carbon_sweep(sp)
    cs.to_csv(config.RESULTS_DIR / "s05_carbon_sweep.csv", index=False)
    ss = section_study(sp)
    ss.to_csv(config.RESULTS_DIR / "s05_section_size.csv", index=False)
    hb = pd.read_csv(config.PROCESSED_DIR / "properties_handbook.csv")
    plain = hb[hb.category.isin(["low_carbon", "medium_carbon", "high_carbon"])]
    fig_carbon_sweep(cs, (plain.C.min(), plain.C.max()))
    fig_section(ss)
    fig_category_effects(out)
    fig_regimes(out)
    print("figures written")


if __name__ == "__main__":
    main()
