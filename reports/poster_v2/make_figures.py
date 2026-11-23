"""Figures for the A1 poster (version 2), drawn at their final printed size.

Every value plotted comes from a file in results/ (the CCT curves come from data/cct_1045.csv and
cct_4140.csv, written by cct_data.py with the project's own transformation model). Fonts are given in points
at print size, on the scale of the reference poster: tick labels 15 pt, axis labels 16 pt. Text is written as
outlines, so the figures look the same on any machine.

    python reports/poster_v2/make_figures.py        # writes figures/*.svg (vector) and *.png (300 dpi)

Figures 1, 2, 7, 8 and 12-14 follow the first poster (reports/poster/make_figures.py); 3-6, 9-11 and 15 are new.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
OUT = HERE / "figures"
DATA = HERE / "data"
sys.path.insert(0, str(HERE))
from poster_fonts import sans_fonts  # noqa: E402

_SANS = sans_fonts(HERE / ".cache")
for _p in _SANS["files"].values():
    font_manager.fontManager.addfont(str(_p))

# The repo's colour-vision-checked categorical order (src/steelml/viz.py), assigned by meaning
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e4e3dc", "#9d9c93", "#d6d5ce"
# one colour per treatment, used on every panel of the poster
C_FC, C_AC, C_HARD, C_QT, C_SR = BLUE, ORANGE, RED, VIOLET, MUTED
C_CAL, C_PUB = AQUA, INK2
# constituents: neutral ferrite and pearlite, so that bainite and martensite stand out
K_F, K_P, K_B, K_M = "#dfe3e8", "#8894a3", AQUA, YELLOW
OURS = "#0070C0"       # the poster's heading blue, for "this work"

TICK, LABEL, NOTE = 15, 16, 15      # pt at final size
LW = 2.6
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": _SANS["family"], "font.size": TICK, "svg.fonttype": "path",
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


def new(w, h, l=23, b=14.5, r=3, t=2.5):  # noqa: E741
    """Figure of w x h mm with one axes; margins in mm."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([l / w, b / h, 1 - (l + r) / w, 1 - (b + t) / h])
    return fig, ax


def square(w, h, l=23, b=14.5, t=3):  # noqa: E741
    """Figure of w x h mm with a square axes on the left; returns the x (mm) where side notes can start."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    s = h - b - t
    ax = fig.add_axes([l / w, b / h, s / w, s / h])
    return fig, ax, l + s + 5


def two_axes(w, h, l=23, b=14.5, r=3, t=7.5, mid=21):  # noqa: E741
    """Two side-by-side axes filling a w x h mm figure; margins in mm."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    aw = (w - l - r - mid) / 2
    axes = [fig.add_axes([(l + k * (aw + mid)) / w, b / h, aw / w, 1 - (b + t) / h]) for k in (0, 1)]
    return fig, axes


def side_notes(fig, w, h, x_mm, y_mm, rows, step=6.6):
    """Legend-like notes in the margin: rows of (marker or None, colour, text, bold). y_mm from the top."""
    for k, (mk, col, txt, bold) in enumerate(rows):
        y = 1 - (y_mm + k * step) / h
        if mk:
            fig.add_artist(Line2D([(x_mm + 1.6) / w], [y], marker=mk, ms=8.5, ls="", transform=fig.transFigure,
                                  markerfacecolor=col[0], markeredgecolor=col[1], markeredgewidth=1.0))
        fig.text((x_mm + (5.2 if mk else 0)) / w, y, txt, fontsize=NOTE, color=INK, va="center", ha="left",
                 fontweight="bold" if bold else "normal")


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontsize=LABEL, fontweight="bold", color=INK, pad=4)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ============================================================================= left column
ROUTE_PICK = [("AISI 1020", "1020 (low C)"), ("AISI 1040", "1040 (medium C)"), ("AISI 1060", "1060 (high C)"),
              ("AISI 1080", "1080 (high C)"), ("AISI 5140", "5140 (Cr)"), ("AISI 4140", "4140 (Cr-Mo)"),
              ("AISI 4340", "4340 (Ni-Cr-Mo)")]


def fig_routes(w, h):
    """Fig 1. Measured handbook UTS of the same grades annealed (furnace) and normalised (25 mm bar in air)."""
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
    save(fig, "f01_handbook_routes")
    d = [rp.loc[g, "UTS_MPa_AC"] - rp.loc[g, "UTS_MPa_FC"] for g, _ in ROUTE_PICK]
    return {"gain_min": min(d), "gain_max": max(d)}


def fig_cooling(w, h):
    """Fig 2. Cooling rate at 700 C of the furnace, of bars in still air and of quenched bars (centre)."""
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
    save(fig, "f02_cooling_rates")
    return {r[0]: r[1] for r in rows}


START_KEYS = [("Tf", "ferrite", "Fs", ("white", INK)), ("Tp", "pearlite", "Ps", (K_P, INK)),
              ("Tb", "bainite", "Bs", (K_B, INK)), ("Tm", "martensite", "Ms", (K_M, INK))]


def fig_starts(w, h):
    """Fig 3. Transformation-start temperatures: model against the dilatometry of Collins et al. (2023)."""
    kv = pd.read_csv(RES / "s03_kinetics_validation.csv")
    ks = pd.read_csv(RES / "s03_kinetics_summary.csv")
    ks = ks[ks.model == "this work"].set_index("start")
    fig, ax, xs = square(w, h)
    lo, hi = 180, 860
    ax.fill_between([lo, hi], [lo - 25, hi - 25], [lo + 25, hi + 25], color=BAND, alpha=0.5, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.3, zorder=1)
    steels = [("EN3B", "EN3B", "o"), ("EN8", "EN8", "s"), ("SA540", "SA-540", "^")]
    n = 0
    for st, _, mk in steels:
        d = kv[kv.steel == st]
        for key, _, _, (face, edge) in START_KEYS:
            m = d[f"{key}_measured"].notna() & d[f"{key}_model"].notna()
            n += int(m.sum())
            ax.scatter(d.loc[m, f"{key}_measured"], d.loc[m, f"{key}_model"], marker=mk, s=74, facecolor=face,
                       edgecolor=edge, lw=1.0, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [200, 400, 600, 800]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Measured start (°C)")
    ax.set_ylabel("Model start (°C)")
    rows = [(None, None, "MAE (°C)", True)]
    out = {"points": n}
    for key, name, short, col in START_KEYS:
        if name in ks.index:
            out[short] = float(ks.loc[name, "MAE_C"])
            rows.append(("o", col, f"{short}  {ks.loc[name, 'MAE_C']:.0f}", False))
    side_notes(fig, w, h, xs, 7, rows)
    side_notes(fig, w, h, xs, 7 + len(rows) * 6.6 + 4,
               [(mk, ("white", INK), lab, False) for _, lab, mk in steels])
    ax.text(0.97, 0.045, "band: ±25 °C", transform=ax.transAxes, fontsize=NOTE, color=INK2, va="bottom",
            ha="right")
    save(fig, "f03_transformation_starts")
    return out


def fig_atlas(w, h):
    """Fig 4. Hardness of 423 measured continuous-cooling points (CCT atlas) against the model."""
    a = pd.read_csv(RES / "s03_cct_atlas_predictions.csv").dropna(subset=["HV", "HV_pred"])
    sm = pd.read_csv(RES / "s03_cct_atlas_summary.csv").set_index("subset")
    fig, ax, xs = square(w, h)
    lo, hi = 80, 800
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.3, zorder=1)
    slow = (a.t500_s >= 20).to_numpy()
    ax.scatter(a.HV[slow], a.HV_pred[slow], s=26, color=C_AC, alpha=0.75, lw=0, zorder=3)
    ax.scatter(a.HV[~slow], a.HV_pred[~slow], s=26, color=C_HARD, alpha=0.75, lw=0, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [200, 400, 600, 800]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Measured hardness (HV)")
    ax.set_ylabel("Model hardness (HV)")
    side_notes(fig, w, h, xs, 7, [("o", (C_HARD, C_HARD), "quench-like", False),
                                 (None, None, "     (under 20 s", False),
                                 (None, None, "     to 500 °C)", False),
                                 ("o", (C_AC, C_AC), "bars in air", False),
                                 (None, None, "     (20 s or more)", False)])
    save(fig, "f04_cct_atlas_hardness")
    r = sm.loc["all points"]
    return {"n": int(len(a)), "steels": int(r.steels), "HV_RMSE": float(r.HV_RMSE),
            "MAE_FP": float(r.MAE_ferrite_pearlite), "MAE_B": float(r.MAE_bainite)}


def fig_parity(w, h):
    """Fig 5. Unseen-grade UTS: annealed / normalised (in-domain) and quenched + tempered, 90 % intervals."""
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
    fig, ax, xs = square(w, h)
    ax.plot([300, 1950], [300, 1950], color=INK2, lw=1.3, zorder=1)
    ax.vlines(qt.UTS_MPa, qt.UTS_MPa_lo90, qt.UTS_MPa_hi90, color=C_QT, alpha=0.22, lw=0.9)
    ax.scatter(qt.UTS_MPa, qt.UTS_MPa_pred, s=16, color=C_QT, alpha=0.75, lw=0, zorder=3)
    for route, col in (("FC", C_FC), ("AC", C_AC)):
        m = (an.route == route).to_numpy()
        ax.vlines(an.UTS_MPa[m], (mu - q90 * sd)[m], (mu + q90 * sd)[m], color=col, alpha=0.45, lw=1.2)
        ax.scatter(an.UTS_MPa[m], mu[m], s=34, color=col, edgecolor="white", lw=0.7, zorder=4)
    ax.set_xlim(300, 1950)
    ax.set_ylim(300, 1950)
    ticks = [400, 800, 1200, 1600]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Measured UTS (MPa)")
    ax.set_ylabel("Predicted UTS (MPa)")
    side_notes(fig, w, h, xs, 7, [("o", (C_FC, C_FC), "Annealed", False), ("o", (C_AC, C_AC), "Normalised", False),
                                 ("o", (C_QT, C_QT), "Quenched +", False), (None, None, "     tempered", False)])
    fig.text(xs / w, 1 - 40 / h, "bars: 90 %\nintervals", fontsize=NOTE, color=INK2, va="top", ha="left",
             linespacing=1.05)
    save(fig, "f05_parity_unseen_grades")
    return {"rmse_an": float(rmse_an), "rmse_qt": float(rmse_qt), "n_an": int(len(an)), "n_qt": int(len(qt))}


def fig_ladder(w, h):
    """Fig 6. UTS error on unseen grades for each kind of model, both families of treatment."""
    b = pd.read_csv(RES / "s04_model_benchmark.csv").set_index(["target", "model"]).RMSE_inAD
    q = pd.read_csv(RES / "s09_qt_benchmark.csv").set_index(["model", "target"]).RMSE
    groups = [("Annealed / normalised", [("Textbook laws", b[("UTS_MPa", "physics_lit")]),
                                         ("Gradient boosting", b[("UTS_MPa", "gbm_comp")]),
                                         ("Gaussian process", b[("UTS_MPa", "gp_comp")]),
                                         ("Physics + GP (ours)", b[("UTS_MPa", "gated_ensemble")])]),
              ("Quenched + tempered", [("Textbook laws", q[("qt_physics", "UTS_MPa")]),
                                       ("Gradient boosting", q[("qt_gbm", "UTS_MPa")]),
                                       ("Gaussian process", q[("qt_gp", "UTS_MPa")]),
                                       ("Physics + GP (ours)", q[("qt_ensemble", "UTS_MPa")])])]
    l, r = 47, 13  # noqa: E741
    fig, ax = new(w, h, l=l, b=14.5, r=r, t=1)
    ax.spines["left"].set_visible(False)
    y, yt, labs = 0.0, [], []
    for title, rows in groups:
        ax.text(-(l - 1) / (w - l - r), y, title, transform=ax.get_yaxis_transform(), fontsize=NOTE,
                fontweight="bold", color=INK, va="center", ha="left")
        y -= 1.0
        for lab, v in rows:
            ours = "ours" in lab
            ax.barh(y, v, height=0.66, color=OURS if ours else BAND)
            ax.text(v + 5, y, f"{v:.0f}", va="center", ha="left", fontsize=NOTE, color=INK,
                    fontweight="bold" if ours else "normal")
            yt.append(y)
            labs.append(lab)
            y -= 1.0
        y -= 0.35
    ax.set_yticks(yt)
    ax.set_yticklabels(labs)
    for tl in ax.get_yticklabels():
        if "ours" in tl.get_text():
            tl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 262)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_ylim(y + 0.75, 0.6)
    ax.set_xlabel("UTS error on unseen grades (MPa)")
    save(fig, "f06_model_ladder")
    return {t: {lab: float(v) for lab, v in rows} for t, rows in groups}


# ============================================================================= right column
def fig_carbon(w, h):
    """Fig 7. Furnace vs air cooling across carbon: (a) UTS, (b) elongation; model with 90 % band + handbook."""
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
    save(fig, "f07_furnace_vs_air_carbon")
    return {}


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
    """Fig 8. CCT of 1045 and 4140 (stage-9 compositions) with the same furnace and 25 mm still-air paths."""
    fig, (a1, a2) = two_axes(w, h, mid=14)
    r45 = _cct_panel(a1, "1045", True)
    r41 = _cct_panel(a2, "4140", False)
    pct = lambda v: f"{100 * v:.0f} %"  # noqa: E731
    same = all(abs(r45["FC"][k] - r45["AC"][k]) < 0.005 for k in "FPBM")
    panel_title(a1, "(a) AISI 1045: ferrite + pearlite" + (" on both paths" if same else ""))
    panel_title(a2, f"(b) AISI 4140: {pct(r41['AC']['B'])} bainite in air")
    save(fig, "f08_cct_1045_4140")
    return {"1045": r45, "4140": r41}


HARD_GROUPS = [("Plain carbon", ["low_carbon", "medium_carbon", "high_carbon"], MUTED),
               ("C-Mn, free-machining", ["carbon_manganese", "free_machining"], YELLOW),
               ("Cr, Cr-V", ["chromium"], MAGENTA),
               ("Cr-Mo, Ni-Cr-Mo", ["cr_mo", "ni_cr_mo"], AQUA)]


def fig_hardenability(w, h):
    """Fig 9. Every catalogue grade as a 25 mm air-cooled bar against its ideal critical diameter (ASTM A255
    factors): (a) the bainite + martensite it forms, (b) the tensile strength it gains over furnace cooling."""
    d = pd.read_csv(RES / "s08_ideal_diameter_catalogue.csv")
    d = d[d.category_true != "si_mn"]          # 2 % Si, outside the window of the critical-temperature equations
    fig, (a1, a2) = two_axes(w, h)
    for ax in (a1, a2):
        for x in (40, 100):
            ax.axvline(x, color=AXIS, lw=1.1, ls=(0, (4, 3)), zorder=1)
    for lab, cats, col in HARD_GROUPS:
        s = d[d.category_true.isin(cats)]
        kw = dict(s=42, color=col, edgecolor="white", lw=0.6, zorder=3, alpha=0.95)
        a1.scatter(s.DI_mm, 100 * s.hard_AC25, label=lab, **kw)
        a2.scatter(s.DI_mm, s.dUTS_AC25, **kw)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlim(0.6, 300)
        ax.set_xticks([1, 10, 40, 100])
        ax.set_xticklabels(["1", "10", "40", "100"])
        ax.minorticks_off()
        ax.set_xlabel("Ideal critical diameter (mm)")
    a1.set_ylim(-5, 105)
    a1.set_yticks([0, 25, 50, 75, 100])
    a1.set_ylabel("Share of bar (%)")
    a2.set_ylim(0, 100 * np.ceil((d.dUTS_AC25.max() + 30) / 100))
    a2.set_yticks([0, 200, 400, 600])
    a2.set_ylabel("UTS gain (MPa)")
    panel_title(a1, "(a) Bainite + martensite in a 25 mm air-cooled bar")
    panel_title(a2, "(b) Strength gained over furnace cooling")
    a1.legend(loc="upper left", handletextpad=0.1, borderaxespad=0.2, markerscale=1.25)
    save(fig, "f09_hardenability_catalogue")
    lo, hi = d[d.DI_mm < 40], d[d.DI_mm >= 100]
    return {"grades": int(len(d)), "below_40": [int((lo.hard_AC25 > 0.5).sum()), int(len(lo))],
            "above_100": [int((hi.hard_AC25 > 0.5).sum()), int(len(hi))],
            "DI_range": [float(d.DI_mm.min()), float(d.DI_mm.max())], "dUTS_max": float(d.dUTS_AC25.max())}


FAMILIES = [("Low-carbon steel", "Low carbon"), ("Medium-carbon steel", "Medium carbon"),
            ("High-carbon steel", "High carbon"), ("Free-machining steel", "Free-machining"),
            ("C-Mn / Mn steel", "C-Mn"), ("Cr / Cr-V steel", "Cr, Cr-V"), ("Mo / Cr-Mo steel", "Cr-Mo"),
            ("Ni-Cr-Mo / Ni steel", "Ni-Cr-Mo")]


def fig_family(w, h):
    """Fig 10. Per steel family (medians over the catalogue grades): UTS furnace -> air, and what the 25 mm
    air-cooled bar is made of."""
    c = pd.read_csv(RES / "s05_category_summary.csv").set_index("category")
    l, b, r, t, mid, wb = 37.5, 14.5, 12, 8.5, 4, 31  # noqa: E741
    wa = w - l - r - mid - wb
    fig = plt.figure(figsize=(w * MM, h * MM))
    a1 = fig.add_axes([l / w, b / h, wa / w, 1 - (b + t) / h])
    a2 = fig.add_axes([(l + wa + mid) / w, b / h, wb / w, 1 - (b + t) / h])
    n = len(FAMILIES)
    out = {}
    for i, (key, lab) in enumerate(FAMILIES):
        y = n - 1 - i
        s = c.loc[key]
        fc, ac = s.UTS_MPa_FC_median, s.UTS_MPa_AC25_median
        a1.plot([fc, ac], [y, y], color=BAND, lw=6, solid_capstyle="round", zorder=1)
        a1.scatter([fc], [y], s=70, color=C_FC, zorder=3, lw=0)
        a1.scatter([ac], [y], s=70, color=C_AC, zorder=3, lw=0)
        a1.text(ac + 22, y, f"+{s.UTS_MPa_delta_median:.0f}", va="center", ha="left", fontsize=NOTE, color=INK2)
        p, bm = 100 * s.AC25_pearlite_mean, 100 * s.AC25_bainite_martensite_mean
        f = 100 - p - bm
        a2.barh(y, f, height=0.62, color=K_F)
        a2.barh(y, p, left=f, height=0.62, color=K_P)
        a2.barh(y, bm, left=f + p, height=0.62, color=K_B)
        a2.text(103, y, f"{bm:.0f} %", va="center", ha="left", fontsize=NOTE,
                color=INK if bm >= 0.5 else MUTED, fontweight="bold" if bm >= 20 else "normal")
        out[lab] = {"FC": float(fc), "AC": float(ac), "gain": float(s.UTS_MPa_delta_median), "BM_pct": float(bm)}
    a1.set_yticks(range(n))
    a1.set_yticklabels([lab for _, lab in FAMILIES][::-1])
    a1.tick_params(axis="y", length=0)
    a1.grid(axis="y", visible=False)
    a1.spines["left"].set_visible(False)
    a1.set_xlim(340, 1030)
    a1.set_xticks([400, 600, 800])
    a1.set_ylim(-0.6, n - 0.4)
    a1.set_xlabel("Median UTS (MPa)")
    a2.set_yticks([])
    a2.grid(False)
    a2.spines["left"].set_visible(False)
    a2.set_xlim(0, 100)
    a2.set_xticks([0, 50, 100])
    a2.set_ylim(-0.6, n - 0.4)
    a2.set_xlabel("Constituents (%)")
    top = dict(transform=a1.transAxes, fontsize=NOTE, fontweight="bold", va="bottom")
    a1.text(0.0, 1.035, "furnace", color=C_FC, ha="left", **top)
    a1.text(0.42, 1.035, "air, 25 mm", color=C_AC, ha="left", **top)
    top2 = dict(transform=a2.transAxes, fontsize=NOTE, fontweight="bold", va="bottom")
    a2.text(0.0, 1.035, "F", color=MUTED, ha="left", **top2)
    a2.text(0.20, 1.035, "P", color=K_P, ha="left", **top2)
    a2.text(0.40, 1.035, "B + M", color=K_B, ha="left", **top2)
    save(fig, "f10_steel_families")
    return out


BAR_GRADES = ["4140", "1040"]


def fig_barsize(w, h):
    """Fig 11. Normalised bars of 13-100 mm (Birbilis et al. 2025): measured against the model, with the
    measured annealed strength of the same grade as the level a thick bar drifts towards."""
    b = pd.read_csv(RES / "s06_bar_sizes_predictions.csv")
    b["grade"] = b.grade.astype(str)
    met = pd.read_csv(RES / "s06_bar_sizes_metrics.csv")
    eff = met[met.check.str.startswith("bar-size effect")].iloc[0]
    fig, ax = new(w, h, l=23, b=14.5, r=19, t=9)
    for g in BAR_GRADES:
        d = b[(b.grade == g) & (b.route == "AC")].groupby("section_mm").mean(numeric_only=True).reset_index()
        d = d[d.section_mm.isin([13, 25, 50, 100])].sort_values("section_mm")
        ax.fill_between(d.section_mm, d.UTS_MPa_lo90, d.UTS_MPa_hi90, color=C_AC, alpha=0.16, lw=0)
        ax.plot(d.section_mm, d.UTS_MPa_pred, color=C_AC, lw=LW)
        ax.scatter(d.section_mm, d.UTS_MPa, s=62, facecolor="white", edgecolor=INK, lw=1.6, zorder=4)
        ax.text(108, d.UTS_MPa_pred.iloc[-1], g, fontsize=NOTE, fontweight="bold", color=INK, va="center",
                ha="left")
        fc = b[(b.grade == g) & (b.route == "FC")].UTS_MPa
        if len(fc):
            ax.plot([13, 100], [fc.mean(), fc.mean()], color=C_FC, lw=1.6, ls=(0, (4, 2)), zorder=2)
            up = fc.mean() > 600
            ax.text(13.2, fc.mean() + (14 if up else -16), f"{g} annealed", fontsize=NOTE, color=C_FC,
                    va="bottom" if up else "top", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(11.5, 104)
    ax.set_xticks([13, 25, 50, 100])
    ax.set_xticklabels(["13", "25", "50", "100"])
    ax.minorticks_off()
    ax.set_ylim(405, 1260)
    ax.set_yticks([600, 800, 1000, 1200])
    ax.set_xlabel("Bar diameter (mm)")
    ax.set_ylabel("UTS, normalised (MPa)")
    ax.legend([Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=INK, ms=8, mew=1.6),
               Line2D([], [], color=C_AC, lw=LW)],
              ["measured", "model, 90 % band"], loc="lower left", ncol=2, bbox_to_anchor=(-0.02, 0.985),
              columnspacing=1.2, borderaxespad=0)
    save(fig, "f11_bar_size")
    return {"n": int(eff.n), "sign": float(eff.sign_agreement), "pred": float(eff.mean_predicted),
            "meas": float(eff.mean_measured)}


def fig_jominy(w, h, grade="AISI 4140H"):
    """Fig 12. Jominy hardness of one H-steel against its SAE band: published vs recalibrated kinetics."""
    cv = pd.read_csv(RES / "s08_kinetic_calibration_cv.csv")
    met = pd.read_csv(RES / "s08_kinetic_calibration_cv_metrics.csv")
    d = cv[cv.grade == grade].sort_values("J")
    x = d.J * 25.4 / 16
    fig, ax = new(w, h)
    ax.fill_between(x, d.HRC_min, d.HRC_max, color=BAND, lw=0)
    ax.plot(x, d.HRC_published, "--", color=C_PUB, lw=LW - 0.3, dashes=(3, 1.5))
    ax.plot(x, d.HRC_calibrated_cv, color=C_CAL, lw=LW + 0.3)
    allr = met[met.distances == "all distances"].set_index("kinetics").RMSE_HRC
    ax.text(51, 63.5, "published constants", color=C_PUB, fontsize=NOTE, fontweight="bold", ha="right",
            va="top")
    ax.text(51, float(d.HRC_calibrated_cv.iloc[-1]) - 2.3, "recalibrated", color=C_CAL, fontsize=NOTE,
            fontweight="bold", ha="right", va="top")
    ax.text(30, 43.5, "SAE H-band", color=INK2, fontsize=NOTE, ha="center", va="center")
    ax.set_xlim(0, 52)
    ax.set_ylim(15, 65)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([20, 30, 40, 50, 60])
    ax.set_xlabel("Distance from quenched end (mm)")
    ax.set_ylabel("Hardness (HRC)")
    save(fig, "f12_jominy_4140H")
    return {"published": float(allr["HRC_published"]), "calibrated": float(allr["HRC_calibrated_cv"])}


def fig_five(w, h, grade="4140"):
    """Fig 13. Five treatments of a 25 mm bar: UTS with its 90 % interval; structure written above each bar.
    The hardened state gets no tensile value (the project does not give one for untempered martensite),
    only its hardness."""
    ft = pd.read_csv(RES / "s09_five_treatments.csv")
    s = ft[ft.grade.astype(str) == grade].set_index("treatment")
    pct = lambda v: f"{100 * v:.0f} %"  # noqa: E731
    order = [("anneal", "Annealed", C_FC, lambda r: f"ferrite + {pct(r.P)} pearlite"),
             ("normalise", "Normalised", C_AC, lambda r: f"{pct(r.B)} bainite"),
             ("harden", "Oil-hardened", C_HARD, lambda r: f"{pct(r.M)} martensite"),
             ("quench_temper", "Tempered 600 °C", C_QT, lambda r: "tempered martensite + bainite"),
             ("stress_relieve", "Stress-relieved 600 °C", C_SR, lambda r: "structure unchanged")]
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
            ax.text(xmax * 0.5, yb, f"{float(r.HV):.0f} HV, too brittle for a tensile test", fontsize=NOTE,
                    fontweight="bold", va="center", ha="center", color=INK,
                    bbox=dict(fc="white", ec="none", pad=1.0))
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.55, n - 0.38)
    save(fig, "f13_five_treatments_4140")
    return {k: (float(s.loc[k].HV), float(s.loc[k].UTS_MPa_pred) if np.isfinite(s.loc[k].UTS_MPa_pred) else None)
            for k, *_ in order}


def fig_tempering(w, h, grade="4140"):
    """Fig 14. Tempering curve of one grade (25 mm bar) with the measured bars and the embrittlement ranges."""
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
    save(fig, "f14_tempering_4140")
    return {"measured_T": [float(m.temper_T_C.min()), float(m.temper_T_C.max())]}


def fig_qtmass(w, h, grade="4140", medium="oil"):
    """Fig 15. Mass effect after quenching and tempering: UTS against bar diameter, the grade held out."""
    me = pd.read_csv(RES / "s09_mass_effect.csv")
    d = me[(me.grade.astype(str) == grade) & (me.quench_medium == medium)]
    full = [t for t, g in d.groupby("temper_T_C") if g.section_mm.nunique() >= 4]
    shades = ["#33248f", C_QT, "#9388e8"]
    fig, ax = new(w, h, l=23, b=14.5, r=19, t=3)
    out = {}
    for t, col in zip(sorted(full), shades):
        g = d[d.temper_T_C == t].groupby("section_mm").mean(numeric_only=True).reset_index().sort_values("section_mm")
        ax.fill_between(g.section_mm, g.UTS_MPa_lo90, g.UTS_MPa_hi90, color=col, alpha=0.13, lw=0)
        ax.plot(g.section_mm, g.UTS_MPa_pred, color=col, lw=LW)
        ax.scatter(g.section_mm, g.UTS_MPa, s=62, facecolor="white", edgecolor=col, lw=1.8, zorder=4)
        ax.text(108, g.UTS_MPa_pred.iloc[-1], f"{t:.0f} °C", fontsize=NOTE, fontweight="bold", color=col,
                va="center", ha="left")
        out[float(t)] = {"meas": g.UTS_MPa.round(0).tolist(), "pred": g.UTS_MPa_pred.round(0).tolist()}
    ax.set_xscale("log")
    ax.set_xlim(11.5, 104)
    ax.set_xticks([13, 25, 50, 100])
    ax.set_xticklabels(["13", "25", "50", "100"])
    ax.minorticks_off()
    lo = min(min(v["meas"] + v["pred"]) for v in out.values())
    hi = max(max(v["meas"] + v["pred"]) for v in out.values())
    ax.set_ylim(100 * np.floor((lo - 130) / 100), 100 * np.ceil((hi + 130) / 100))
    ax.set_xlabel("Bar diameter (mm)")
    ax.set_ylabel("UTS, tempered (MPa)")
    ax.text(0.97, 0.96, "tempered at", transform=ax.transAxes, fontsize=NOTE, color=INK2, ha="right", va="top")
    save(fig, "f15_qt_mass_effect")
    return out


COL = 274.5
HALF = 133.25      # half of the column, less an 8 mm gap
SIZES = {  # mm at print size; build_poster.py reads these so slots and figures always match
    "routes": (COL, 61), "cooling": (HALF, 80),
    "starts": (HALF, 76), "atlas": (HALF, 76), "parity": (HALF, 76), "ladder": (HALF, 76),
    "carbon": (COL, 76), "cct": (COL, 80), "harden": (COL, 70),
    "family": (HALF, 82), "barsize": (HALF, 82),
    "jominy": (HALF, 80), "five": (HALF, 80), "tempering": (HALF, 80), "qtmass": (HALF, 80),
}
FIGS = [("routes", fig_routes), ("cooling", fig_cooling), ("starts", fig_starts), ("atlas", fig_atlas),
        ("parity", fig_parity), ("ladder", fig_ladder), ("carbon", fig_carbon), ("cct", fig_cct),
        ("harden", fig_hardenability), ("family", fig_family), ("barsize", fig_barsize),
        ("jominy", fig_jominy), ("five", fig_five), ("tempering", fig_tempering), ("qtmass", fig_qtmass)]


def main(only=None):
    shown = {}
    for key, fn in FIGS:
        if only and key not in only:
            continue
        shown[key] = fn(*SIZES[key])
    (HERE / ".cache").mkdir(exist_ok=True)
    if not only:
        (HERE / ".cache" / "figure_numbers.json").write_text(json.dumps(shown, indent=1, default=str))
    return shown


if __name__ == "__main__":
    for k, v in main(set(sys.argv[1:]) or None).items():
        print(f"{k:10s}", v)
