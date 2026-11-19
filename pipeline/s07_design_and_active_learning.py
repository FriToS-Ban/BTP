"""Stage 7 - proactive ML: inverse design, active learning, recalibration.

A. Grade-and-route recommender: for a property specification, rank every
   in-scope catalogue grade x {furnace cooling, air cooling of 10-100 mm
   bars} whose *lower 90 % bound* meets the specification inside the
   applicability domain, by weldability (IIW carbon equivalent).
B. The strength-ductility trade-off that standard grades can deliver: every
   in-scope AISI/SAE grade x route option scored on its lower 90 % bounds,
   and the options that no other option beats on both (the front). Only
   real grades are searched; an earlier version optimised free
   compositions, which is the composition-only inverse design the project
   set out to avoid.
C. Retrospective active learning on the handbook data: adding grades in
   order of model uncertainty vs at random - which learns faster?
D. Few-shot recalibration: how many measured members of a new product
   (hot-rolled structural sections, Stage 6) remove the domain-shift bias?
E. Laboratory plan: candidate experiments on commonly stocked steels
   (EN3B/EN8/EN9/EN19/EN24 ~ AISI 1018/1045/1055/4140/4340), ranked by
   expected information, with pre-registered predictions to test against.

Outputs: results/s07_*.csv, figures fig22-fig24.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import config, viz  # noqa: E402
from steelml.features import featurize  # noqa: E402
from steelml.models.evaluation import metrics  # noqa: E402
from steelml.models.property_models import PRODUCTION_MODEL, make_model  # noqa: E402
from steelml.predictor import SteelPredictor  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt
AC_SECTIONS = [10.0, 25.0, 50.0, 100.0]

SPECS = {
    "weldable_structural": {"label": "YS >= 300 MPa and El >= 28 %, best weldability",
                            "min": {"YS_MPa": 300, "El_pct": 28}},
    "shaft_no_quench": {"label": "UTS >= 750 MPa and El >= 18 % without quench-and-temper",
                        "min": {"UTS_MPa": 750, "El_pct": 18}},
    "high_strength_air": {"label": "UTS >= 950 MPa and El >= 12 %",
                          "min": {"UTS_MPa": 950, "El_pct": 12}},
}


# ---------------------------------------------------------------------------
# A. recommender
# ---------------------------------------------------------------------------

def candidate_table(sp, catalog):
    scope = catalog[catalog.in_fp_scope & catalog.standard.eq("AISI/SAE")]
    base = scope[["grade", "category"] + [f"{el}_nom" for el in config.ELEMENTS]].rename(
        columns={f"{el}_nom": el for el in config.ELEMENTS})
    frames = []
    q = base.copy()
    q["route"], q["section_mm"], q["option"] = "FC", 25.0, "furnace cooled"
    frames.append(q)
    for D in AC_SECTIONS:
        q = base.copy()
        q["route"], q["section_mm"], q["option"] = "AC", D, f"air cooled, {int(D)} mm"
        frames.append(q)
    X = pd.concat(frames, ignore_index=True)
    out = sp.predict(X.drop(columns=["grade", "category", "option"]))
    keep = ["k_f_ferrite", "k_f_pearlite", "k_f_bainite", "k_f_martensite", "CE_IIW", "in_domain",
            "microstructure_in_data", "in_validity_window"] + \
        [f"{t}_{s}" for t in ("UTS_MPa", "YS_MPa", "El_pct", "HB") for s in ("pred", "lo90", "hi90")]
    for k in keep:
        X[k] = out[k].to_numpy()
    return X


def recommend(cands, spec, top=8):
    # inside the composition domain, and a microstructure the training data contain
    ok = cands.in_domain & cands.microstructure_in_data & cands.in_validity_window
    for t, v in spec["min"].items():
        ok &= cands[f"{t}_lo90"] >= v
    r = cands[ok].sort_values(["CE_IIW", "UTS_MPa_pred"], ascending=[True, False])
    return r.drop_duplicates("grade").head(top)


# ---------------------------------------------------------------------------
# B. trade-off front of the standard grades
# ---------------------------------------------------------------------------

def grade_front(cands):
    """Options (grade x route) inside the domain with a microstructure the
    data contain, and for furnace and for air cooling the front of options
    that no other option beats on both lower 90 % bounds (UTS and El)."""
    ok = cands[cands.in_domain & cands.microstructure_in_data & cands.in_validity_window].copy()
    ok["family"] = np.where(ok.route == "FC", "FC", "AC")
    fronts = {}
    for fam, g in ok.groupby("family"):
        g = g.sort_values(["UTS_MPa_lo90", "El_pct_lo90"], ascending=[False, False])
        front, best = [], -np.inf
        for _, r in g.iterrows():
            if r.El_pct_lo90 > best:
                front.append(r)
                best = r.El_pct_lo90
        fronts[fam] = pd.DataFrame(front)
    return ok, fronts


# ---------------------------------------------------------------------------
# C. retrospective active learning
# ---------------------------------------------------------------------------

def retrospective_al(F, target="UTS_MPa", n_init=8, n_add=14, seeds=6):
    grades = F.grade.unique()
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        init = list(rng.choice(grades, n_init, replace=False))
        for strategy in ("uncertainty", "random"):
            chosen = list(init)
            for step in range(n_add + 1):
                tr = F.grade.isin(chosen).to_numpy()
                m = make_model("anchored_chain", target).fit(F[tr], F.loc[tr, target].to_numpy(float))
                pool = [g for g in grades if g not in chosen]
                te = F.grade.isin(pool).to_numpy() & (F.route != "AR").to_numpy()
                mu, sd = m.predict(F[te])
                rmse = float(np.sqrt(np.mean((mu - F.loc[te, target].to_numpy()) ** 2)))
                rows.append({"seed": seed, "strategy": strategy, "n_grades": len(chosen), "RMSE": rmse})
                if step == n_add or not pool:
                    break
                if strategy == "random":
                    nxt = rng.choice(pool)
                else:
                    Fp = F[F.grade.isin(pool)]
                    _, sdp = m.predict(Fp)
                    score = pd.Series(sdp / np.maximum(m.predict(Fp)[0], 1e-6), index=Fp.grade).groupby(level=0).mean()
                    nxt = score.idxmax()
                chosen.append(nxt)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D. few-shot recalibration on a new product (structural members)
# ---------------------------------------------------------------------------

def few_shot(F_hand, draws=8, ks=(0, 5, 10, 20, 40)):
    ext = pd.read_csv(config.PROCESSED_DIR / "structural_external.csv")
    ext = ext[ext.usable & ~ext.microalloyed].reset_index(drop=True)
    X = ext[config.ELEMENTS + ["section_mm", "YS_MPa", "UTS_MPa", "El_pct"]].copy()
    X["route"] = "AR"
    X["grade"] = "member_" + ext.entry.astype(str)
    X["HB"] = np.nan
    FX = featurize(X).reset_index(drop=True)
    rows = []
    for t in ("YS_MPa", "UTS_MPa"):
        for k in ks:
            for d in range(draws if k > 0 else 1):
                rng = np.random.default_rng(100 + d)
                idx = rng.permutation(len(FX))
                add, test = idx[:k], idx[k:]
                Ftr = pd.concat([F_hand, FX.iloc[add]], ignore_index=True)
                m = make_model(PRODUCTION_MODEL, t).fit(Ftr, Ftr[t].to_numpy(float))
                mu, _ = m.predict(FX.iloc[test])
                mm = metrics(FX.iloc[test][t], mu)
                rows.append({"target": t, "k_added": k, "draw": d, "RMSE": mm["RMSE"], "bias": mm["bias"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E. lab plan
# ---------------------------------------------------------------------------

LAB_STEELS = {
    "EN3B (~AISI 1018)": "AISI 1018", "EN8 (~AISI 1045)": "AISI 1045", "EN9 (~AISI 1055)": "AISI 1055",
    "EN19 (~AISI 4140)": "AISI 4140H", "EN24 (~AISI 4340)": "AISI 4340H", "EN18 (~AISI 5140)": "AISI 5140H",
    "EN353-type (~AISI 8620)": "AISI 8620H", "EN42 (~AISI 1080)": "AISI 1080",
}


def lab_plan(sp, catalog):
    rows = []
    for label, g in LAB_STEELS.items():
        r = catalog[catalog.grade == g].iloc[0]
        comp = {el: r[f"{el}_nom"] for el in config.ELEMENTS}
        for route, D, opt in (("FC", 25.0, "furnace cool (~18 K/h)"), ("AC", 25.0, "air cool, 25 mm bar"),
                              ("AC", 10.0, "air cool, 10 mm bar")):
            rows.append({**comp, "steel": label, "route": route, "section_mm": D, "option": opt})
    X = pd.DataFrame(rows)
    out = sp.predict(X.drop(columns=["steel", "option"]))
    plan = X[["steel", "option"]].copy()
    plan["austenitise_C"] = np.round(out.treat_T_C.to_numpy(), 0)
    for k, lab in (("F_best", "ferrite"), ("P_best", "pearlite"), ("B_best", "bainite"),
                   ("M_best", "martensite")):
        plan[f"{lab}_frac"] = np.round(out[k].to_numpy(), 2)
    plan["pearlite_spacing_nm"] = np.where(out.k_f_pearlite.to_numpy() >= 0.05,
                                           np.round(1000 * out.S0_um.to_numpy(), 0), np.nan)
    for t in ("YS_MPa", "UTS_MPa", "El_pct", "HB"):
        plan[f"{t}_pred"] = np.round(out[f"{t}_pred"].to_numpy(), 1)
        plan[f"{t}_90PI"] = [f"[{lo:.0f}, {hi:.0f}]" for lo, hi in zip(out[f"{t}_lo90"], out[f"{t}_hi90"])]
    plan["uniform_El_est_pct"] = np.round(out.uniform_El_est_pct.to_numpy(), 1)
    plan["uniform_El_90PI"] = [f"[{lo:.0f}, {hi:.0f}]" for lo, hi in zip(out.uniform_El_lo90, out.uniform_El_hi90)]
    plan["in_domain"] = out.in_domain.to_numpy()
    plan["microstructure_in_data"] = out.microstructure_in_data.to_numpy()
    # expected information: mean relative predictive spread across targets
    rel = np.mean([(out[f"{t}_hi90"] - out[f"{t}_lo90"]) / np.maximum(np.abs(out[f"{t}_pred"]), 1e-6)
                   for t in ("YS_MPa", "UTS_MPa", "El_pct", "HB")], axis=0)
    plan["relative_uncertainty"] = np.round(rel, 3)
    plan["priority_rank"] = plan.relative_uncertainty.rank(ascending=False, method="first").astype(int)
    return plan.sort_values("priority_rank")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_front(ok, fronts):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for fam in ("FC", "AC"):
        g = ok[ok.family == fam]
        ax.plot(g.UTS_MPa_lo90, g.El_pct_lo90, "o", ms=3.5, color=viz.ROUTE_COLORS[fam], alpha=0.25, mec="none",
                label=f"{viz.ROUTE_LABELS[fam]}: every standard grade" + (" (10-100 mm bars)" if fam == "AC" else ""))
        f = fronts[fam]
        ax.plot(f.UTS_MPa_lo90, f.El_pct_lo90, "-o", ms=5, color=viz.ROUTE_COLORS[fam], mec=viz.SURFACE,
                label=f"front, {viz.ROUTE_LABELS[fam].lower()}")
        for _, r in f.iterrows():
            lab = r.grade.replace("AISI ", "") + ("" if fam == "FC" else f" ({int(r.section_mm)} mm)")
            ax.annotate(lab, (r.UTS_MPa_lo90, r.El_pct_lo90), xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=viz.INK2)
    ax.set_xlabel("Tensile strength, lower 90 % bound (MPa)")
    ax.set_ylabel("Total elongation, lower 90 % bound (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Strength-ductility trade-off that standard AISI/SAE grades give (lower 90 % bounds)")
    return viz.savefig(fig, "fig22_grade_front_fc_vs_ac")


def fig_al(al):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for strategy, col in (("uncertainty", viz.BLUE), ("random", viz.NEUTRAL)):
        g = al[al.strategy == strategy].groupby("n_grades").RMSE
        m, s = g.mean(), g.std()
        ax.fill_between(m.index, m - s, m + s, color=col, alpha=0.18, lw=0)
        ax.plot(m.index, m, "-o", ms=4, color=col, mec=viz.SURFACE,
                label="most-uncertain grade next" if strategy == "uncertainty" else "random grade next")
    ax.set_xlabel("Grades in the training set")
    ax.set_ylabel("UTS RMSE on the remaining grades (MPa)")
    ax.legend()
    ax.set_title("Retrospective active learning on the handbook data (mean ± s.d. over 6 starts)")
    return viz.savefig(fig, "fig23_active_learning")


def fig_few_shot(fs):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, t, lab in zip(axes, ("YS_MPa", "UTS_MPa"), ("Yield strength", "Tensile strength")):
        g = fs[fs.target == t].groupby("k_added")
        for col_name, col, name in (("RMSE", viz.BLUE, "RMSE"), ("bias", viz.ORANGE, "bias")):
            m = g[col_name].mean()
            s = g[col_name].std().fillna(0)
            ax.fill_between(m.index, m - s, m + s, color=col, alpha=0.15, lw=0)
            ax.plot(m.index, m, "-o", color=col, ms=4, mec=viz.SURFACE, label=name)
        ax.axhline(0, color=viz.AXIS, lw=1)
        ax.set_xlabel("Measured members added to the training data")
        ax.set_ylabel("MPa")
        ax.set_title(f"{lab}: error on the remaining members")
    axes[0].legend()
    fig.suptitle("Few-shot recalibration to a new product (hot-rolled S275/S355 sections)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return viz.savefig(fig, "fig24_few_shot_recalibration")


def main():
    config.ensure_dirs()
    sp = SteelPredictor.load()
    catalog = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv")
    cands = candidate_table(sp, catalog)
    cands.to_csv(config.RESULTS_DIR / "s07_candidates.csv", index=False)
    recs = []
    for key, spec in SPECS.items():
        r = recommend(cands, spec)
        r = r.assign(spec=spec["label"])
        recs.append(r)
        print(f"\n{spec['label']}:")
        print(r[["grade", "option", "UTS_MPa_lo90", "YS_MPa_lo90", "El_pct_lo90", "CE_IIW"]].round(2).to_string(index=False))
    pd.concat(recs).to_csv(config.RESULTS_DIR / "s07_recommendations.csv", index=False)

    ok, fronts = grade_front(cands)
    for fam, f in fronts.items():
        f.to_csv(config.RESULTS_DIR / f"s07_grade_front_{fam}.csv", index=False)
        print(f"\nfront, {fam}:")
        print(f[["grade", "option", "UTS_MPa_lo90", "El_pct_lo90", "UTS_MPa_pred", "El_pct_pred"]].round(1)
              .to_string(index=False))
    fig_front(ok, fronts)

    p = pd.read_csv(config.PROCESSED_DIR / "properties_handbook.csv")
    F = featurize(p).reset_index(drop=True)
    al = retrospective_al(F)
    al.to_csv(config.RESULTS_DIR / "s07_active_learning.csv", index=False)
    fig_al(al)
    print(al.groupby(["strategy", "n_grades"]).RMSE.mean().unstack(0).round(1).to_string())

    fs = few_shot(F)
    fs.to_csv(config.RESULTS_DIR / "s07_few_shot.csv", index=False)
    fig_few_shot(fs)
    print(fs.groupby(["target", "k_added"])[["RMSE", "bias"]].mean().round(1).to_string())

    plan = lab_plan(sp, catalog)
    plan.to_csv(config.RESULTS_DIR / "s07_lab_plan_preregistered.csv", index=False)
    print(plan.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
