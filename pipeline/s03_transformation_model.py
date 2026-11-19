"""Stage 3 - the transformation model: validation, calibration, CCT diagrams.

1. Validate the kinetic model (Li et al. 1998 kinetics with Hultgren start
   conditions) against measured dilatometry of three steels at nine cooling
   rates (Collins et al. 2023), and against the authors' own published
   Li-model numbers (an independent-implementation check).
2. Calibrate the effective cooling conditions behind the handbook data
   (normalising: equivalent bar diameter; annealing: furnace rate) by
   minimising the *physics-only* hardness error - no ML is involved, so the
   later ML evaluation is not contaminated.
3. Draw CCT diagrams with furnace- and air-cooling curves overlaid, and
   microstructure-versus-cooling-rate maps for representative steels.

Outputs: results/s03_*.csv, figures fig07, fig08, fig10, fig11 (the CCT
diagrams, fig09, are drawn in stage 5 with the production model's hardness).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import config, viz  # noqa: E402
from steelml.features import featurize  # noqa: E402
from steelml.cct import REPRESENTATIVE, cct_starts  # noqa: E402
from steelml.physics import cooling, kinetics  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt



# ---------------------------------------------------------------------------
# 1. validation against dilatometry
# ---------------------------------------------------------------------------

def validate_kinetics():
    comp = pd.read_csv(config.RAW_DIR / "collins2023_compositions.csv").set_index("steel")
    ts = pd.read_csv(config.RAW_DIR / "collins2023_Ts.csv")
    li = pd.read_csv(config.RAW_DIR / "collins2023_li_Ts.csv")
    rows = []
    for s, c in comp.iterrows():
        for r in sorted(ts[ts.steel == s].rate):
            t, T = cooling.constant_rate_path(c.T_aust_C, r)
            d = kinetics.simulate(c.to_dict(), c.astm_G, t, T)
            m = ts[(ts.steel == s) & (ts.rate == r)].iloc[0]
            lrow = li[(li.steel == s) & (li.rate == r)].iloc[0]
            rows.append({"steel": s, "rate_K_s": r,
                         "Tf_measured": m.Tf, "Tf_model": d["T_ferrite_start"], "Tf_li_published": lrow.f,
                         "Tp_measured": m.Tp, "Tp_model": d["T_pearlite_start"], "Tp_li_published": lrow.p,
                         "Tb_measured": m.Tb, "Tb_model": d["T_bainite_start"], "Tb_li_published": lrow.b,
                         "Tm_measured": m.Tm, "Tm_model": d["T_martensite_start"],
                         "f_ferrite": d["f_ferrite"], "f_pearlite": d["f_pearlite"],
                         "f_bainite": d["f_bainite"], "f_martensite": d["f_martensite"]})
    v = pd.DataFrame(rows)
    summ = []
    for ph in ("f", "p", "b", "m"):
        meas = v[f"T{ph}_measured"]
        for kind, col in (("this work", f"T{ph}_model"), ("Collins et al. Li model", f"T{ph}_li_published")):
            if col not in v:
                continue
            ok = meas.notna() & v[col].notna()
            e = (v[col] - meas)[ok]
            summ.append({"start": {"f": "ferrite", "p": "pearlite", "b": "bainite", "m": "martensite"}[ph],
                         "model": kind, "n": int(ok.sum()), "bias_C": e.mean(), "MAE_C": e.abs().mean()})
        if f"T{ph}_li_published" in v:
            ok = v[f"T{ph}_model"].notna() & v[f"T{ph}_li_published"].notna()
            e = (v[f"T{ph}_model"] - v[f"T{ph}_li_published"])[ok]
            summ.append({"start": {"f": "ferrite", "p": "pearlite", "b": "bainite", "m": "martensite"}[ph],
                         "model": "this work vs published Li", "n": int(ok.sum()), "bias_C": e.mean(),
                         "MAE_C": e.abs().mean()})
    return v, pd.DataFrame(summ)


def fig_validation(v):
    steels = ["EN3B", "EN8", "SA540"]
    titles = {"EN3B": "EN3B (0.18C)", "EN8": "EN8 (0.44C)", "SA540": "SA-540 (0.4C-1.8Ni-0.9Cr-0.3Mo)"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
    phases = [("f", "ferrite"), ("p", "pearlite"), ("b", "bainite"), ("m", "martensite")]
    for ax, s in zip(axes, steels):
        d = v[v.steel == s].sort_values("rate_K_s")
        for ph, name in phases:
            col = viz.PHASE_COLORS[name]
            if d[f"T{ph}_measured"].notna().any():
                ax.plot(d.rate_K_s, d[f"T{ph}_measured"], "o", color=col, ms=7, mec=viz.SURFACE, mew=1,
                        label=f"{name} start, measured")
            if d[f"T{ph}_model"].notna().any():
                ax.plot(d.rate_K_s, d[f"T{ph}_model"], "-", color=col, lw=1.8, label=f"{name} start, model")
        ax.set_xscale("log")
        ax.set_title(titles[s])
        ax.set_xlabel("Cooling rate (K/s)")
    axes[0].set_ylabel("Transformation start temperature (°C)")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Transformation model vs measured dilatometry (Collins et al. 2023)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig07_kinetics_validation")


# ---------------------------------------------------------------------------
# 2. calibration of the handbook cooling conditions
# ---------------------------------------------------------------------------

def calibrate():
    p = pd.read_csv(config.PROCESSED_DIR / "properties_handbook.csv")
    rows = []
    for D in [10, 15, 25, 35, 50, 65, 80, 100, 150]:
        q = p[p.condition == "normalized"].copy()
        q["section_mm"] = float(D)
        F = featurize(q)
        e = F.HB_phys - F.HB
        rows.append({"condition": "normalized", "parameter": "equivalent bar diameter (mm)", "value": D,
                     "rate_700_K_s": float(F.cool_rate_700.mean()), "HB_RMSE": float(np.sqrt((e ** 2).mean())),
                     "HB_MAE": float(e.abs().mean())})
    for r in [0.002, 0.003, 0.005, 0.0075, 0.01, 0.02, 0.03]:
        q = p[p.condition == "annealed"].copy()
        q["fc_rate"] = r
        F = featurize(q)
        e = F.HB_phys - F.HB
        rows.append({"condition": "annealed", "parameter": "furnace rate at 700 °C (K/s)", "value": r,
                     "rate_700_K_s": r, "HB_RMSE": float(np.sqrt((e ** 2).mean())), "HB_MAE": float(e.abs().mean())})
    return pd.DataFrame(rows)


def fig_calibration(cal):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    n = cal[cal.condition == "normalized"]
    axes[0].plot(n.value, n.HB_RMSE, "-o", color=viz.ROUTE_COLORS["AC"], mec=viz.SURFACE)
    axes[0].axvline(config.HANDBOOK_AC_SECTION_MM, color=viz.INK2, lw=1)
    axes[0].text(config.HANDBOOK_AC_SECTION_MM * 1.05, n.HB_RMSE.max() * 0.95,
                 f"documented: {config.HANDBOOK_AC_SECTION_MM:.0f} mm (1 in.) rounds", color=viz.INK2, fontsize=9)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Equivalent bar diameter in still air (mm)")
    axes[0].set_ylabel("Physics-only hardness RMSE (HB)")
    axes[0].set_title("Normalized rows (air cooled)")
    a = cal[cal.condition == "annealed"]
    axes[1].plot(a.value, a.HB_RMSE, "-o", color=viz.ROUTE_COLORS["FC"], mec=viz.SURFACE)
    axes[1].axvline(config.HANDBOOK_FC_RATE, color=viz.INK2, lw=1)
    axes[1].text(config.HANDBOOK_FC_RATE * 1.08, a.HB_RMSE.max() * 0.995, "assumed: 0.005 K/s (18 K/h)",
                 color=viz.INK2, fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Furnace cooling rate at 700 °C (K/s)")
    axes[1].set_title("Annealed rows (furnace cooled)")
    fig.suptitle("Calibrating the effective cooling conditions of the handbook data (physics only, no ML)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig08_cooling_calibration")


# ---------------------------------------------------------------------------
# 3. CCT diagrams and microstructure maps
# ---------------------------------------------------------------------------

def fig_cooling_paths():
    """What the two routes mean physically: temperature against time for the
    furnace cool, still-air cooling of bars of 10-150 mm and three Jominy
    positions, with the cooling rate of each at 700 C."""
    from steelml.physics import jominy
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    T0 = 880.0
    rows = []
    t, T = cooling.furnace_cooling_path(T0, config.FC_RATE_700C)
    ok = t > 0
    r = cooling.rate_at(t, T)
    ax.plot(t[ok], T[ok], color=viz.ROUTE_COLORS["FC"], lw=2.2, label=f"furnace cooling ({r:.3f} K/s at 700 °C)")
    rows.append({"path": "furnace cooling", "rate_700C_K_s": r, "t85_s": cooling.t85(t, T)})
    shades = ["#f3b08f", "#ee8a5e", "#eb6834", "#c4501f", "#8f3510"]
    for D, col in zip((10, 25, 50, 100, 150), shades):
        t, T = cooling.air_cooling_path(T0, D)
        ok = t > 0
        r = cooling.rate_at(t, T)
        ax.plot(t[ok], T[ok], color=col, lw=1.8, label=f"still air, {D} mm bar ({r:.2f} K/s)")
        rows.append({"path": f"still air, {D} mm bar", "rate_700C_K_s": r, "t85_s": cooling.t85(t, T)})
    jp = jominy.jominy_paths(T0, (4, 16, 32))
    for (t, T), J, ls in zip(jp, (4, 16, 32), (":", "--", "-.")):
        ok = t > 0
        r = cooling.rate_at(t, T)
        ax.plot(t[ok], T[ok], ls, color=viz.INK2, lw=1.2, label=f"Jominy J{J} ({J * 1.5875:.0f} mm, {r:.1f} K/s)")
        rows.append({"path": f"Jominy J{J}", "rate_700C_K_s": r, "t85_s": cooling.t85(t, T)})
    ax.axhline(700.0, color=viz.AXIS, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1, 3e5)
    ax.set_ylim(100, T0 + 10)
    ax.set_xlabel("Time from the start of cooling (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("The two routes as cooling curves (austenitised at 880 °C; heat balance of a round bar in still air)")
    fig.tight_layout()
    viz.savefig(fig, "fig27_cooling_paths")
    return pd.DataFrame(rows)


def fig_fraction_maps():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True, sharey=True)
    for ax, (name, comp) in zip(axes.ravel(), REPRESENTATIVE.items()):
        df = cct_starts(comp, 7.5, np.logspace(-3, 1.7, 60))
        stack = [("f_ferrite", "ferrite"), ("f_pearlite", "pearlite"), ("f_cementite_pro", "cementite"),
                 ("f_bainite", "bainite"), ("f_martensite", "martensite")]
        ys = [df[k].to_numpy() for k, _ in stack]
        ax.stackplot(df.rate, *ys, colors=[viz.PHASE_COLORS[n] for _, n in stack],
                     labels=[n.capitalize() if n != "cementite" else "Proeutectoid Fe3C" for _, n in stack],
                     edgecolor=viz.SURFACE, linewidth=0.6)
        for x, route, lab in ((config.FC_RATE_700C, "FC", "FC"), (0.5, "AC", "AC 50 mm"),
                              (1.05, "AC", "AC 25 mm")):
            ax.axvline(x, color=viz.INK, lw=0.9)
            ax.text(x * 1.08, 1.02, lab, fontsize=7.5, color=viz.INK2, rotation=90, va="bottom")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(name, pad=24)
        ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("Cooling rate at 700 °C (K/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Volume fraction")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Microstructure versus cooling rate from the kinetic model (raw fractions; the reported value "
                 "splits ferrite + pearlite by the lever rule, and this split is the top of the pearlite band)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return viz.savefig(fig, "fig10_microstructure_vs_cooling_rate")


# ---------------------------------------------------------------------------
# 4. measured microstructures from the literature
# ---------------------------------------------------------------------------

def _rate(v):
    """Cooling rate cell -> float; a range 'a-b' becomes its geometric mean."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if "-" in s[1:]:
        a, b = (float(x) for x in s.split("-", 1))
        return float(np.sqrt(a * b))
    return float(s)


def _literature_frame(require):
    from steelml.predictor import default_austenitising_T
    lit = pd.read_csv(config.RAW_DIR / "literature_compilation.csv")
    suspect = lit.get("quality_flag", pd.Series("", index=lit.index)).fillna("").str.startswith("suspect")
    lit = lit[lit[require].notna() & ~suspect].copy().reset_index(drop=True)
    rows, keep = [], []
    for i, r in lit.iterrows():
        c = str(r.condition)
        d = {el: (float(r[el]) if el in r and pd.notna(r[el]) else (0.25 if el == "Si" else
              config.RESIDUAL_TYPICAL.get(el, 0.0))) for el in config.ELEMENTS}
        if pd.notna(_rate(r.cooling_rate_C_per_s)):
            d["route"], d["cooling_rate"] = "CR", _rate(r.cooling_rate_C_per_s)
        elif c == "annealed_furnace_cooled":
            d["route"] = "FC"
        elif c in ("normalized_air_cooled", "hot_rolled_air_cooled"):
            d["route"] = "AC" if c.startswith("normalized") else "AR"
        else:
            continue
        d["section_mm"] = 25.0 if d["route"] != "AR" else 2.0 * float(r.section_size_mm or 12.5)
        d["fc_rate"] = config.FC_RATE_700C
        d["treat_T_C"] = float(r.austenitizing_T_C) if pd.notna(r.austenitizing_T_C) else np.nan
        rows.append(d)
        keep.append(i)
    X = pd.DataFrame(rows)
    X["treat_T_C"] = X.treat_T_C.fillna(pd.Series(np.asarray(default_austenitising_T(X, X.route), float),
                                                    index=X.index))
    return lit.loc[keep].reset_index(drop=True), X


def literature_fractions():
    from steelml.predictor import best_estimate_fractions
    lit, X = _literature_frame("pearlite_fraction_pct")
    F = featurize(X)
    best = best_estimate_fractions(F)
    out = lit[["source_id", "steel_designation", "condition", "C", "Mn", "pearlite_fraction_pct"]].copy()
    out["route"] = X.route.to_numpy()
    out["kinetic_pct"] = 100 * (F.k_f_pearlite + F.k_f_cementite_pro).to_numpy()
    out["lever_pct"] = 100 * F.f_pearlite_eq.to_numpy()
    out["best_estimate_pct"] = 100 * (best["P_best"] + best["Fe3C_best"])
    rows = []
    for name, sel in (("furnace cooled", out.route == "FC"), ("air cooled", out.route != "FC"), ("all", out.route.notna())):
        s = out[sel]
        for col in ("kinetic_pct", "lever_pct", "best_estimate_pct"):
            e = s[col] - s.pearlite_fraction_pct
            rows.append({"subset": name, "estimate": col.replace("_pct", ""), "n": len(s),
                         "MAE_pct_points": float(e.abs().mean()), "bias": float(e.mean())})
    return out, pd.DataFrame(rows)


def ferrite_factor_scan(factors=(1, 2, 3, 5, 8, 12)):
    comp = pd.read_csv(config.RAW_DIR / "collins2023_compositions.csv").set_index("steel")
    ts = pd.read_csv(config.RAW_DIR / "collins2023_Ts.csv")
    lit, X = _literature_frame("pearlite_fraction_pct")
    saved = config.FERRITE_RATE_FACTOR
    rows = []
    try:
        for lam in factors:
            config.FERRITE_RATE_FACTOR = float(lam)
            tf = []
            for s, c in comp.iterrows():
                for rr in sorted(ts[ts.steel == s].rate):
                    t, T = cooling.constant_rate_path(c.T_aust_C, rr)
                    d = kinetics.simulate(c.to_dict(), c.astm_G, t, T)
                    m = ts[(ts.steel == s) & (ts.rate == rr)].iloc[0]
                    if pd.notna(m.Tf) and np.isfinite(d["T_ferrite_start"]):
                        tf.append(d["T_ferrite_start"] - m.Tf)
            F = featurize(X)
            e = 100 * (F.k_f_pearlite + F.k_f_cementite_pro).to_numpy() - lit.pearlite_fraction_pct.to_numpy()
            air = (X.route != "FC").to_numpy()
            rows.append({"ferrite_rate_factor": lam, "ferrite_start_bias_C": float(np.mean(tf)),
                         "ferrite_start_MAE_C": float(np.mean(np.abs(tf))),
                         "pearlite_MAE_furnace": float(np.mean(np.abs(e[~air]))),
                         "pearlite_MAE_air": float(np.mean(np.abs(e[air])))})
    finally:
        config.FERRITE_RATE_FACTOR = saved
    return pd.DataFrame(rows)


CCT_ATLAS = config.RAW_DIR / "cct_rose_hougardy_rijeka.csv"


def cct_atlas_predictions():
    """Measured continuous-cooling results of 19 low-alloy steels (423 points:
    ferrite + pearlite, bainite and martensite fractions and Vickers hardness
    against the cooling time to 500 C; Rose & Hougardy atlas data as tabulated
    in the CC BY supplements of Smokvina Hanza et al. 2021 and Marohnic et al.
    2025). Each curve is taken as Newtonian cooling from the austenitising
    temperature that reaches 500 C at the stated t500."""
    from steelml.physics import microstructure as ms
    from steelml.physics import properties as pp
    d = pd.read_csv(CCT_ATLAS)
    comps = pd.DataFrame({el: d[el] if el in d else config.RESIDUAL_TYPICAL.get(el, 0.0) for el in config.ELEMENTS})
    paths = []
    for T0, t5 in zip(d.T_aust_C, d.t500_s):
        tau = t5 / np.log((T0 - 20.0) / (500.0 - 20.0))
        paths.append((cooling._time_on_grid(lambda T, tau=tau: (np.asarray(T, float) - 20.0) / tau, T0),
                      cooling.T_GRID.copy()))
    G = ms.austenite_astm(d.T_aust_C.to_numpy())
    rate = np.array([cooling.rate_at(t, T) for t, T in paths])
    k = kinetics.simulate_batch(comps, G, paths)
    fp = (k.f_ferrite + k.f_pearlite + k.f_cementite_pro).to_numpy()
    hv = pp.hv_mixture(comps, rate * 3600.0, fp, k.f_bainite, k.f_martensite)
    return d.assign(rate700=rate, FP_pred=fp, B_pred=k.f_bainite.to_numpy(), M_pred=k.f_martensite.to_numpy(),
                    HV_pred=np.asarray(hv))


def cct_atlas_summary(d):
    rows = []
    for name, sel in (("all points", d.t500_s > 0), ("t500 < 20 s (quench-like)", d.t500_s < 20),
                      ("t500 >= 20 s (air cooling of bars and slower)", d.t500_s >= 20)):
        s = d[sel]
        rows.append({"subset": name, "n": len(s), "steels": s.steel.nunique(),
                     "MAE_ferrite_pearlite": float(np.mean(np.abs(s.FP_pred - s.f_FP))),
                     "MAE_bainite": float(np.mean(np.abs(s.B_pred - s.f_B))),
                     "MAE_martensite": float(np.mean(np.abs(s.M_pred - s.f_M))),
                     "bias_ferrite_pearlite": float(np.mean(s.FP_pred - s.f_FP)),
                     "HV_RMSE": float(np.sqrt(np.mean((s.HV_pred - s.HV) ** 2))),
                     "HV_bias": float(np.mean(s.HV_pred - s.HV))})
    return pd.DataFrame(rows)


def fig_cct_atlas(d):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    lt = np.log10(d.t500_s)
    for ax, (pc, mc, lab) in zip(axes, (("FP_pred", "f_FP", "ferrite + pearlite"), ("B_pred", "f_B", "bainite"),
                                        ("M_pred", "f_M", "martensite"), ("HV_pred", "HV", "hardness (HV)"))):
        sc = ax.scatter(d[mc], d[pc], c=lt, cmap="viridis", s=14, alpha=0.85, edgecolors="none")
        lo, hi = (0, 1) if mc != "HV" else (100, 750)
        ax.plot([lo, hi], [lo, hi], color=viz.INK2, lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"measured {lab}")
        ax.set_ylabel(f"predicted {lab}")
        ax.set_title(lab.capitalize())
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("log10 of the cooling time to 500 °C (s)")
    fig.suptitle(f"Transformation model against {len(d)} measured CCT points of {d.steel.nunique()} low-alloy steels "
                 "(Rose & Hougardy atlas data, open supplements)", x=0.01, ha="left", fontweight="bold")
    return viz.savefig(fig, "fig28_cct_atlas_validation")


def grain_size_check():
    """Measured ferrite grain sizes against the two-branch Hodgson-Gibbs
    relation and the low-Ceq branch used here (nothing fitted)."""
    from steelml.physics import microstructure as ms
    lit, X = _literature_frame("ferrite_grain_size_um")
    F = featurize(X)
    meas = lit.ferrite_grain_size_um.to_numpy(float)
    preds = {"Hodgson-Gibbs, both branches": ms.ferrite_grain_size_um_hg_original(
                 X.C, X.Mn, F.cool_rate_700, F.d_gamma_um),
             "Hodgson-Gibbs low-Ceq branch, Ceq held at 0.35 (used)": F.d_alpha_um.to_numpy(float)}
    rows = []
    for name, p in preds.items():
        p = np.asarray(p, float)
        rows.append({"relation": name, "n": len(meas), "MAE_um": float(np.mean(np.abs(p - meas))),
                     "bias_um": float(np.mean(p - meas)),
                     "mean_error_factor": float(np.exp(np.mean(np.abs(np.log(p / meas))))),
                     "at_3um_floor": int(np.sum(p <= 3.0 + 1e-9))})
    return pd.DataFrame(rows)


def pearlite_coupling_ablation():
    """Is the excess pearlite of air-cooled steels caused by how the pearlite
    incubation is coupled to the ferrite reaction? Li et al. / Collins et al.
    accumulate the pearlite incubation from Ae1, so pearlite completes almost
    as soon as the Hultgren condition lets it start. The alternative starts
    the incubation only inside the Hultgren region (ferrite keeps growing
    meanwhile). Both are scored on the dilatometry start temperatures and on
    the measured pearlite fractions."""
    saved = getattr(config, "PEARLITE_INCUBATION", "ae1")
    rows = []
    try:
        for mode, label in (("ae1", "incubation from Ae1 (published)"),
                            ("hultgren", "incubation inside the Hultgren region")):
            config.PEARLITE_INCUBATION = mode
            _, summ = validate_kinetics()
            _, fs = literature_fractions()
            ps = summ[(summ.model == "this work") & (summ.start == "pearlite")].iloc[0]
            k = fs[fs.estimate == "kinetic"].set_index("subset")
            rows.append({"coupling": label, "pearlite_start_MAE_C": ps.MAE_C, "pearlite_start_bias_C": ps.bias_C,
                         "pearlite_MAE_furnace": k.loc["furnace cooled", "MAE_pct_points"],
                         "pearlite_MAE_air": k.loc["air cooled", "MAE_pct_points"]})
    finally:
        config.PEARLITE_INCUBATION = saved
    return pd.DataFrame(rows)


def spacing_calibration():
    """Fit S0 = A dT^-m on measured spacings; compare with constant-k forms."""
    lit, X = _literature_frame("pearlite_spacing_nm")
    F = featurize(X)
    dT = np.maximum(F.k_Ae1 - F.k_T_pearlite_mean, 5.0).to_numpy()
    S = lit.pearlite_spacing_nm.to_numpy() / 1000.0
    ok = np.isfinite(dT) & np.isfinite(S)
    x, y, src = np.log(dT[ok]), np.log(S[ok]), lit.source_id.to_numpy()[ok]
    m_neg, logA = np.polyfit(x, y, 1)
    A, m = float(np.exp(logA)), float(-m_neg)
    err_pow, err_k = [], []
    for s in np.unique(src):
        tr, te = src != s, src == s
        mm, aa = np.polyfit(x[tr], y[tr], 1)
        err_pow.extend(list(aa + mm * x[te] - y[te]))
        k = np.exp(np.mean(y[tr] + x[tr]))
        err_k.extend(list(np.log(k) - x[te] - y[te]))
    res = pd.DataFrame([{
        "n": int(ok.sum()), "sources": len(np.unique(src)), "A_um": A, "m": m,
        "dT_range_K": f"{np.exp(x.min()):.0f}-{np.exp(x.max()):.0f}",
        "loso_factor_power_law": float(np.exp(np.mean(np.abs(err_pow)))),
        "loso_factor_constant_k": float(np.exp(np.mean(np.abs(err_k)))),
        "factor_marder_bramfitt_8.02": float(np.exp(np.mean(np.abs(np.log(8.02 / dT[ok] / S[ok]))))),
    }])
    for name, fitted, cfg in (("A", A, config.SPACING_A_UM), ("m", m, config.SPACING_EXPONENT)):
        if abs(fitted - cfg) / cfg > 0.03:
            print(f"WARNING: fitted spacing {name} = {fitted:.3f} differs from config {cfg:.3f}")
    pairs = lit[["source_id", "steel_designation", "pearlite_spacing_nm"]].copy()
    pairs["predicted_nm"] = 1000 * config.SPACING_A_UM * dT ** (-config.SPACING_EXPONENT)
    pairs["isothermal_nm"] = 1000 * 8.02 / dT
    return res, pairs[ok]


def fig_literature_micro(fr, sp_pairs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    # literature rows with a stated cooling rate are forged bars and hot strip,
    # not dilatometry, so they get their own label here
    labels = {**viz.ROUTE_LABELS, "CR": "Forged bar / hot strip (stated rate)"}
    for route, mk in (("FC", "o"), ("AC", "s"), ("AR", "s"), ("CR", "^")):
        s = fr[fr.route == route]
        if s.empty:
            continue
        col = viz.ROUTE_COLORS.get(route, viz.VIOLET)
        ax.plot(s.pearlite_fraction_pct, s.best_estimate_pct, mk, color=col, ms=8, mec=viz.SURFACE,
                label=f"{labels.get(route, route)}: best estimate")
        ax.plot(s.pearlite_fraction_pct, s.kinetic_pct, mk, mfc="none", mec=col, ms=8, mew=1.2,
                label=f"{labels.get(route, route)}: raw kinetic model")
    ax.plot([0, 100], [0, 100], color=viz.AXIS, lw=1)
    ax.set_xlabel("Measured pearlite (%)")
    ax.set_ylabel("Predicted pearlite (%)")
    ax.set_title(f"Pearlite fraction, {fr.pearlite_fraction_pct.notna().sum()} literature samples")
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    ax.plot(sp_pairs.pearlite_spacing_nm, sp_pairs.predicted_nm, "o", color=viz.ORANGE, ms=8, mec=viz.SURFACE,
            label=rf"fitted $S_0 = {config.SPACING_A_UM}\,\Delta T^{{-{config.SPACING_EXPONENT}}}$ µm")
    ax.plot(sp_pairs.pearlite_spacing_nm, sp_pairs.isothermal_nm, "o",
            mfc="none", mec=viz.MUTED, ms=8, label=r"isothermal $S_0 = 8.02/\Delta T$ µm (Marder–Bramfitt)")
    ax.plot([20, 700], [20, 700], color=viz.AXIS, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured interlamellar spacing (nm)")
    ax.set_ylabel("Predicted (nm)")
    n_sp = int((sp_pairs.pearlite_spacing_nm.notna() & sp_pairs.predicted_nm.notna()).sum())
    ax.set_title(f"Pearlite spacing, {n_sp} literature measurements")
    ax.legend(fontsize=8)
    fig.suptitle("Microstructure predictions against measured literature data", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig11_microstructure_literature_validation")


def main():
    config.ensure_dirs()
    v, summ = validate_kinetics()
    v.to_csv(config.RESULTS_DIR / "s03_kinetics_validation.csv", index=False)
    summ.to_csv(config.RESULTS_DIR / "s03_kinetics_summary.csv", index=False)
    print(summ.round(1).to_string(index=False))
    fig_validation(v)
    cal = calibrate()
    cal.to_csv(config.RESULTS_DIR / "s03_calibration.csv", index=False)
    print(cal.round(3).to_string(index=False))
    fig_calibration(cal)
    cp = fig_cooling_paths()
    cp.to_csv(config.RESULTS_DIR / "s03_cooling_paths.csv", index=False)
    print(cp.round(3).to_string(index=False))
    fig_fraction_maps()
    fr, fr_summ = literature_fractions()
    fr.to_csv(config.RESULTS_DIR / "s03_literature_pearlite.csv", index=False)
    fr_summ.to_csv(config.RESULTS_DIR / "s03_literature_pearlite_summary.csv", index=False)
    print(fr_summ.round(2).to_string(index=False))
    scan = ferrite_factor_scan()
    scan.to_csv(config.RESULTS_DIR / "s03_ferrite_factor_scan.csv", index=False)
    print(scan.round(2).to_string(index=False))
    ca = cct_atlas_predictions()
    ca.to_csv(config.RESULTS_DIR / "s03_cct_atlas_predictions.csv", index=False)
    cas = cct_atlas_summary(ca)
    cas.to_csv(config.RESULTS_DIR / "s03_cct_atlas_summary.csv", index=False)
    print(cas.round(3).to_string(index=False))
    fig_cct_atlas(ca)
    gs = grain_size_check()
    gs.to_csv(config.RESULTS_DIR / "s03_grain_size_check.csv", index=False)
    print(gs.round(2).to_string(index=False))
    pc = pearlite_coupling_ablation()
    pc.to_csv(config.RESULTS_DIR / "s03_pearlite_coupling_ablation.csv", index=False)
    print(pc.round(2).to_string(index=False))
    spc, pairs = spacing_calibration()
    spc.to_csv(config.RESULTS_DIR / "s03_spacing_calibration.csv", index=False)
    print(spc.round(3).to_string(index=False))
    fig_literature_micro(fr, pairs)
    print("figures written")


if __name__ == "__main__":
    main()
