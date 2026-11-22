"""Figures for the A1 poster, drawn at their final printed size.

Every value plotted comes from a file in results/ (the CCT curves come from
reports/poster/data/cct_1045.csv and cct_4140.csv, written by cct_data.py with the
project's own transformation model). Fonts are given in points at print size,
on the scale of the reference poster: tick labels 15 pt, axis labels 16 pt. Output: reports/poster/figures/*.svg
(vector, for the poster) and *.png (previews).

    python reports/poster/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
OUT = HERE / "figures"
DATA = HERE / "data"

# The repo's colour-vision-checked categorical order (src/steelml/viz.py), assigned by meaning
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e4e3dc", "#9d9c93", "#d6d5ce"
# one colour per treatment, used on every panel of the poster
C_FC, C_AC, C_HARD, C_QT, C_SR = BLUE, ORANGE, RED, VIOLET, MUTED
C_CAL, C_PUB = AQUA, INK2

TICK, LABEL, NOTE = 15, 16, 15      # pt at final size
LW = 2.6
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "Calibri", "font.size": TICK, "svg.fonttype": "path",
    "axes.labelsize": LABEL, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5, "xtick.major.pad": 3, "ytick.major.pad": 3,
    "legend.fontsize": NOTE, "legend.frameon": False, "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25, "legend.handletextpad": 0.4,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def new(w, h, l=23, b=14.5, r=3, t=2.5):
    """Figure of w x h mm with one axes; margins in mm."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([l / w, b / h, 1 - (l + r) / w, 1 - (b + t) / h])
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------------------------
ROUTE_PICK = [("AISI 1020", "1020 (low C)"), ("AISI 1040", "1040 (medium C)"), ("AISI 1060", "1060 (high C)"),
              ("AISI 1080", "1080 (high C)"), ("AISI 5140", "5140 (Cr)"), ("AISI 4140", "4140 (Cr-Mo)"),
              ("AISI 4340", "4340 (Ni-Cr-Mo)")]


def fig_routes(w, h):
    """Measured handbook UTS of the same grades annealed (furnace) and normalised (25 mm bar in air)."""
    rp = pd.read_csv(RES / "s01_route_pairs.csv").set_index("grade")
    fig, ax = new(w, h, l=41, b=15, r=3, t=2)
    n = len(ROUTE_PICK)
    for i, (g, _) in enumerate(ROUTE_PICK):
        y = n - 1 - i
        a, b = rp.loc[g, "UTS_MPa_FC"], rp.loc[g, "UTS_MPa_AC"]
        ax.plot([a, b], [y, y], color=BAND, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([a], [y], s=70, color=C_FC, zorder=3, lw=0)
        ax.scatter([b], [y], s=70, color=C_AC, zorder=3, lw=0)
        ax.text(b + 16, y, f"+{b - a:.0f} MPa", va="center", ha="left", fontsize=NOTE, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in ROUTE_PICK][::-1])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(340, 1420)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Tensile strength, handbook values (MPa)")
    ax.legend([Line2D([], [], ls="", marker="o", color=C_FC, ms=8),
               Line2D([], [], ls="", marker="o", color=C_AC, ms=8)],
              ["annealed (furnace cooled)", "normalised (air cooled, 25 mm bar)"], loc="upper right")
    save(fig, "f0_handbook_routes")
    d = [rp.loc[g, "UTS_MPa_AC"] - rp.loc[g, "UTS_MPa_FC"] for g, _ in ROUTE_PICK]
    return min(d), max(d)


def two_axes(w, h, l=23, b=14.5, r=3, t=7.5, mid=21):
    """Two side-by-side axes filling a w x h mm figure; margins in mm."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    aw = (w - l - r - mid) / 2
    axes = [fig.add_axes([(l + k * (aw + mid)) / w, b / h, aw / w, 1 - (b + t) / h]) for k in (0, 1)]
    return fig, axes


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=LABEL, fontweight="bold", color=INK, pad=4)


def fig_cooling(w, h):
    """Cooling rate at 700 C of the furnace, of bars in still air and of quenched bars (centre)."""
    cp = pd.read_csv(RES / "s03_cooling_paths.csv").set_index("path").rate_700C_K_s
    qr = pd.read_csv(RES / "s09_quench_rates.csv")
    q = lambda m: float(qr[(qr.D_mm == 25) & (qr.medium == m) & (qr.r_over_R == 0.0)].rate700_K_s.iloc[0])  # noqa: E731
    rows = [("Furnace", cp["furnace cooling"], C_FC),
            ("Still air, 100 mm", cp["still air, 100 mm bar"], C_AC),
            ("Still air, 50 mm", cp["still air, 50 mm bar"], C_AC),
            ("Still air, 25 mm", cp["still air, 25 mm bar"], C_AC),
            ("Still air, 10 mm", cp["still air, 10 mm bar"], C_AC),
            ("Oil, 25 mm centre", q("oil"), C_HARD),
            ("Water, 25 mm centre", q("water"), C_HARD)]
    fig, ax = new(w, h, l=51, b=14.5, r=3, t=2)
    n = len(rows)
    for i, (lab, v, col) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, left=1e-3, height=0.62, color=col)
        txt = f"{v:.3f}" if v < 0.01 else (f"{v:.2f}" if v < 1 else (f"{v:.1f}" if v < 10 else f"{v:.0f}"))
        ax.text(v * 1.25, y, txt, va="center", ha="left", fontsize=NOTE, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 600)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    ax.minorticks_off()
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Cooling rate at 700 °C (K/s)")
    save(fig, "f_methods_cooling_rates")
    return {r[0]: r[1] for r in rows}


def fig_carbon(w, h):
    """Furnace vs air cooling across carbon: (a) UTS, (b) elongation; model with 90 % band + handbook."""
    cs = pd.read_csv(RES / "s05_carbon_sweep.csv")
    rp = pd.read_csv(RES / "s01_route_pairs.csv")
    pc = rp[rp.category.isin(["low_carbon", "medium_carbon", "high_carbon"])]
    fig, (a1, a2) = two_axes(w, h)
    kw = dict(s=60, lw=1.8, zorder=4)
    for ax, t, lims, ticks, lab in ((a1, "UTS_MPa", (300, 1300), [400, 600, 800, 1000, 1200], "UTS (MPa)"),
                                    (a2, "El_pct", (5, 42), [10, 20, 30, 40], "Total elongation (%)")):
        for route, col in (("FC", C_FC), ("AC", C_AC)):
            d = cs[cs.route == route].sort_values("C")
            ax.fill_between(d.C, d[f"{t}_lo90"], d[f"{t}_hi90"], color=col, alpha=0.18, lw=0)
            ax.plot(d.C, d[f"{t}_pred"], color=col, lw=LW)
            ax.scatter(pc.C, pc[f"{t}_{route}"], facecolor="white", edgecolor=col, **kw)
        ax.set_xlim(0.08, 1.02)
        ax.set_ylim(*lims)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks(ticks)
        ax.set_xlabel("Carbon (wt %)")
        ax.set_ylabel(lab)
    panel_title(a1, "(a) Tensile strength")
    panel_title(a2, "(b) Ductility")
    a1.text(0.10, 1180, "Air cooled, 25 mm bar", color=C_AC, fontsize=NOTE, fontweight="bold", ha="left",
            va="center")
    a1.text(1.0, 420, "Furnace cooled", color=C_FC, fontsize=NOTE, fontweight="bold", ha="right", va="center")
    a2.text(0.10, 9.0, "circles: handbook values", color=INK2, fontsize=NOTE, ha="left", va="center")
    save(fig, "f1_furnace_vs_air_carbon")


def _cct_panel(ax, tag, show_ylabel):
    cc = pd.read_csv(DATA / f"cct_{tag}.csv")
    meta = json.loads((DATA / f"cct_{tag}.json").read_text())
    ax.grid(False)
    c = lambda k: cc[cc.curve == k].sort_values("t_s")  # noqa: E731
    for k in ("ferrite", "pearlite", "bainite", "martensite"):
        d = c(k)
        if len(d) >= 2:
            ax.plot(d.t_s, d.T_C, color=INK2, lw=2.0)
    d = c("finish")
    if len(d) >= 2:
        ax.plot(d.t_s, d.T_C, "--", color=MUTED, lw=1.6)
    for k, col in (("path_FC", C_FC), ("path_AC", C_AC)):
        d = c(k)
        ax.plot(d.t_s, d.T_C, color=col, lw=LW + 0.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1, 4e5)
    ax.set_ylim(150, 900)
    ax.set_xticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_yticks([200, 400, 600, 800])
    ax.minorticks_off()
    ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Temperature (°C)")
    lab = dict(fontsize=LABEL, fontweight="bold", color=INK, va="center")
    d = c("ferrite")
    if len(d) >= 2:
        ax.text(d.t_s.iloc[-1], d.T_C.iloc[-1] + 36, "F", ha="center", **lab)
    d = c("pearlite")
    if len(d) >= 2:
        ax.text(d.t_s.iloc[-1] * 1.35, d.T_C.iloc[-1] - 8, "P", ha="left", **lab)
    d = c("bainite")
    if len(d) >= 2:
        ax.text(d.t_s.iloc[-1] * 1.6, d.T_C.iloc[-1], "B", ha="left", **lab)
    d = c("martensite")
    if len(d) >= 2:
        ax.text(1.25, d.T_C.iloc[0] - 30, "Ms", ha="left", **lab)
    note = dict(fontsize=NOTE, fontweight="bold", va="center", linespacing=0.95)
    ax.text(1.25, 705, "Air,\n25 mm", color=C_AC, ha="left", **note)
    ax.text(8e4, 290, "Furnace", color=C_FC, ha="right", **note)
    return meta["routes"]


def fig_cct(w, h):
    """CCT of 1045 and 4140 (stage-9 compositions) with the same furnace and 25 mm still-air paths."""
    fig, (a1, a2) = two_axes(w, h, mid=14)
    r45 = _cct_panel(a1, "1045", True)
    r41 = _cct_panel(a2, "4140", False)
    pct = lambda v: f"{100 * v:.0f} %"  # noqa: E731
    same = all(abs(r45["FC"][k] - r45["AC"][k]) < 0.005 for k in "FPBM")
    panel_title(a1, "(a) AISI 1045: ferrite + pearlite" + (" on both paths" if same else ""))
    panel_title(a2, f"(b) AISI 4140: {pct(r41['AC']['B'])} bainite in air")
    save(fig, "f2_cct_4140")
    return {"1045": r45, "4140": r41}


def fig_parity(w, h):
    """Unseen-grade UTS: annealed / normalised (in-domain) and quenched + tempered, 90 % intervals."""
    o = pd.read_csv(RES / "s04_oof_predictions.csv")
    dom = pd.read_csv(RES / "s04_domain_flags.csv")[["grade", "condition", "in_domain"]]
    q90 = float(pd.read_csv(RES / "s04_conformal.csv").set_index("target").loc["UTS_MPa", "q90_normalised"])
    an = o.merge(dom, on=["grade", "condition"])
    an = an[an.route.isin(["FC", "AC"]) & an.in_domain]
    mu, sd = an["gated_ensemble|UTS_MPa|mu"], an["gated_ensemble|UTS_MPa|sd"]
    qt = pd.read_csv(RES / "s09_qt_oof.csv")
    b = pd.read_csv(RES / "s04_model_benchmark.csv").set_index(["target", "model"])
    rmse_an = b.loc[("UTS_MPa", "gated_ensemble"), "RMSE_inAD"]
    rmse_qt = pd.read_csv(RES / "s09_qt_benchmark.csv").set_index(["model", "target"]).loc[
        ("qt_ensemble", "UTS_MPa"), "RMSE"]
    fig, ax = new(w, h)
    ax.plot([300, 1950], [300, 1950], color=INK2, lw=1.3, zorder=1)
    ax.vlines(qt.UTS_MPa, qt.UTS_MPa_lo90, qt.UTS_MPa_hi90, color=C_QT, alpha=0.22, lw=0.9)
    ax.scatter(qt.UTS_MPa, qt.UTS_MPa_pred, s=16, color=C_QT, alpha=0.75, lw=0, zorder=3)
    for route, col in (("FC", C_FC), ("AC", C_AC)):
        m = (an.route == route).to_numpy()
        ax.vlines(an.UTS_MPa[m], (mu - q90 * sd)[m], (mu + q90 * sd)[m], color=col, alpha=0.45, lw=1.2)
        ax.scatter(an.UTS_MPa[m], mu[m], s=34, color=col, edgecolor="white", lw=0.7, zorder=4)
    ax.legend([Line2D([], [], ls="", marker="o", color=c, ms=7) for c in (C_FC, C_AC, C_QT)],
              ["Annealed", "Normalised", "Q + T"], loc="upper left", handletextpad=0.1, borderaxespad=0.1)
    ax.set_xlim(300, 1950)
    ax.set_ylim(300, 1950)
    ticks = [400, 800, 1200, 1600]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Measured UTS (MPa)")
    ax.set_ylabel("Predicted UTS (MPa)")
    save(fig, "f3_parity_unseen_grades")
    return rmse_an, rmse_qt


def fig_jominy(w, h, grade="AISI 4140H"):
    cv = pd.read_csv(RES / "s08_kinetic_calibration_cv.csv")
    met = pd.read_csv(RES / "s08_kinetic_calibration_cv_metrics.csv")
    d = cv[cv.grade == grade].sort_values("J")
    x = d.J * 25.4 / 16
    fig, ax = new(w, h)
    ax.fill_between(x, d.HRC_min, d.HRC_max, color=BAND, lw=0)
    ax.plot(x, d.HRC_published, "--", color=C_PUB, lw=LW - 0.3, dashes=(3, 1.5))
    ax.plot(x, d.HRC_calibrated_cv, color=C_CAL, lw=LW + 0.3)
    allr = met[met.distances == "all distances"].set_index("kinetics").RMSE_HRC
    ax.text(51, 53.5, "published", color=C_PUB, fontsize=NOTE, fontweight="bold", ha="right", va="bottom")
    ax.text(51, 33.5, "calibrated", color=C_CAL, fontsize=NOTE, fontweight="bold", ha="right", va="top")
    ax.text(30, 43.5, "SAE H-band", color=INK2, fontsize=NOTE, ha="center", va="center")
    ax.set_xlim(0, 52)
    ax.set_ylim(15, 65)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([20, 30, 40, 50, 60])
    ax.set_xlabel("Jominy distance (mm)")
    ax.set_ylabel("Hardness (HRC)")
    save(fig, "f4_jominy_4140H")
    return float(allr["HRC_published"]), float(allr["HRC_calibrated_cv"])


def fig_five(w, h, grade="4140"):
    """Five treatments of a 25 mm bar: UTS with its 90 % interval; structure written above each bar.
    The hardened state gets no tensile value (the project does not give one for untempered
    martensite), only its hardness."""
    ft = pd.read_csv(RES / "s09_five_treatments.csv")
    s = ft[ft.grade.astype(str) == grade].set_index("treatment")
    pct = lambda v: f"{100 * v:.0f} %"  # noqa: E731
    order = [("anneal", "Annealed", C_FC, lambda r: f"ferrite + {pct(r.P)} pearlite"),
             ("normalise", "Normalised", C_AC, lambda r: f"{pct(r.B)} bainite"),
             ("harden", "Oil-hardened", C_HARD, lambda r: f"{pct(r.M)} martensite"),
             ("quench_temper", "Tempered 600 °C", C_QT, lambda r: "tempered M + B"),
             ("stress_relieve", "Stress-relieved", C_SR, lambda r: "unchanged")]
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([1 / w, 1 / h, 1 - 3 / w, 1 - 2 / h])
    ax.axis("off")
    n = len(order)
    xmax = 1110.0
    for i, (key, lab, col, struct) in enumerate(order):
        r = s.loc[key]
        y = n - 1 - i
        ax.text(0, y + 0.27, f"{lab}: {struct(r)}", fontsize=NOTE, va="center", ha="left", color=INK)
        yb = y - 0.20
        if np.isfinite(r.UTS_MPa_pred):
            u, lo, hi = float(r.UTS_MPa_pred), float(r.UTS_MPa_lo90), float(r.UTS_MPa_hi90)
            ax.barh(yb, u, height=0.40, color=col)
            ax.plot([lo, hi], [yb, yb], color=INK, lw=1.6, solid_capstyle="butt")
            ax.plot([lo, lo], [yb - 0.11, yb + 0.11], color=INK, lw=1.6)
            ax.plot([hi, hi], [yb - 0.11, yb + 0.11], color=INK, lw=1.6)
            inside = u > 0.7 * xmax
            ax.text(lo - 12 if inside else hi + 18, yb, f"{u:.0f} MPa", fontsize=NOTE, fontweight="bold",
                    va="center", ha="right" if inside else "left", color="white" if inside else INK)
        else:
            ax.barh(yb, xmax * 0.995, height=0.40, color="#fbe3e3", edgecolor=col, lw=1.3, ls=(0, (4, 2)))
            ax.text(xmax * 0.5, yb, f"{float(r.HV):.0f} HV, too brittle to test", fontsize=NOTE,
                    fontweight="bold", va="center", ha="center", color=INK,
                    bbox=dict(fc="white", ec="none", pad=1.0))
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.55, n - 0.38)
    save(fig, "f5_five_treatments_4140")
    return {k: (float(s.loc[k].HV), float(s.loc[k].UTS_MPa_pred) if np.isfinite(s.loc[k].UTS_MPa_pred) else None)
            for k, *_ in order}


def fig_tempering(w, h, grade="4140"):
    tc = pd.read_csv(RES / "s09_tempering_curves.csv")
    qt = pd.read_csv(RES / "s09_qt_oof.csv")
    s = tc[tc.grade.astype(str) == grade].sort_values("temper_T_C")
    m = qt[(qt.grade.astype(str) == grade) & (qt.section_mm == 25) & (qt.quench_medium == s.quench_medium.iloc[0])]
    fig, ax = new(w, h)
    ax.axvspan(250, 400, color=RED, alpha=0.10, lw=0)
    ax.axvspan(375, 575, color=YELLOW, alpha=0.14, lw=0)
    ax.text(312, 1985, "TME", ha="center", va="top", fontsize=NOTE, color=INK2, fontweight="bold")
    ax.text(475, 1985, "TE", ha="center", va="top", fontsize=NOTE, color=INK2, fontweight="bold")
    ax.fill_between(s.temper_T_C, s.UTS_MPa_lo90, s.UTS_MPa_hi90, color=C_QT, alpha=0.18, lw=0)
    ax.plot(s.temper_T_C, s.UTS_MPa_pred, color=C_QT, lw=LW)
    ax.scatter(m.temper_T_C, m.UTS_MPa, s=60, facecolor="white", edgecolor=C_QT, lw=1.8, zorder=4)
    ax.set_xlim(140, 715)
    ax.set_ylim(600, 2000)
    ax.set_xticks([200, 300, 400, 500, 600, 700])
    ax.set_yticks([800, 1200, 1600, 2000])
    ax.set_xlabel("Tempering temperature (°C)")
    ax.set_ylabel("UTS (MPa)")
    save(fig, "f6_tempering_4140")


HALF = 133.25   # half of the 274.5 mm column, less an 8 mm gap
SIZES = {  # mm at print size; build_poster.py reads these so slots and figures always match
    "routes": (274.5, 90), "cooling": (HALF, 92),
    "carbon": (274.5, 92), "cct": (274.5, 100),
    "parity": (HALF, 118), "jominy": (HALF, 118),
    "five": (HALF, 108), "tempering": (HALF, 108),
}

if __name__ == "__main__":
    print("intro chart, smallest and largest gain:", fig_routes(*SIZES["routes"]))
    print("cooling rates (K/s):", fig_cooling(*SIZES["cooling"]))
    fig_carbon(*SIZES["carbon"])
    if (DATA / "cct_4140.csv").exists():
        print("CCT route fractions:", fig_cct(*SIZES["cct"]))
    else:
        print("cct_4140.csv missing: run cct_data.py with the project environment first")
    print("parity RMSE A/N, Q+T:", fig_parity(*SIZES["parity"]))
    print("Jominy RMSE published, calibrated:", fig_jominy(*SIZES["jominy"]))
    print("five treatments (HV, UTS):", fig_five(*SIZES["five"]))
    fig_tempering(*SIZES["tempering"])
