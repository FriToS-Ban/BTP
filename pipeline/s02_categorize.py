"""Stage 2 - differentiate steels into categories with machine learning.

1. Simulate heats inside the specification ranges of 547 catalogue grades.
2. Compare five classifiers with grade-grouped cross-validation (a grade's
   heats never appear in both training and test folds).
3. Fit the production classifier (+ novelty detector) and test it on real,
   measured heat analyses it has never seen: SteelBench heats (family
   level), hot-rolled structural steels (Kingsley et al.), the three
   dilatometry steels of Collins et al., and - as an open-set test - the
   312 maraging-type steels of the project's previous (legacy) dataset.
4. Compare with a metallurgist's decision list (the AISI/SAE composition
   limits written as rules, nothing fitted), on the same simulated and real
   heats, and check how often the two agree.
5. Explain the classifier with SHAP and compare the supervised categories
   with an unsupervised clustering of the catalogue.

Outputs: results/s02_*.csv, results/models/category_model.joblib, figures fig03-fig06.
"""

import io
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, normalized_mutual_info_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import categories as cat  # noqa: E402
from steelml import config, viz  # noqa: E402
from steelml.data.catalog import sample_heats  # noqa: E402
from steelml.models.classifier import CLASSIFIERS, FEATURES, CategoryModel, _log, make_classifier  # noqa: E402
from steelml.models.rules import RuleClassifier  # noqa: E402

warnings.filterwarnings("ignore")
plt = viz.plt
N_PER_GRADE = 40
COARSE = {"low_carbon": "plain_carbon", "medium_carbon": "plain_carbon", "high_carbon": "plain_carbon",
          "free_machining": "plain_carbon", "carbon_manganese": "plain_carbon", "chromium": "low_alloy",
          "cr_mo": "low_alloy", "ni_cr_mo": "low_alloy", "si_mn": "low_alloy",
          "stainless_austenitic": "stainless", "stainless_ferritic": "stainless",
          "stainless_martensitic": "stainless", "tool": "tool"}


def cross_validate(heats):
    """Grade-grouped 5-fold CV of every classifier; returns metrics and the
    out-of-fold predictions of each model."""
    gkf = GroupKFold(n_splits=5)
    X = heats[FEATURES].to_numpy(float)
    y = heats["category"].to_numpy()
    g = heats["grade"].to_numpy()
    rows, oof = [], {}
    for name in CLASSIFIERS:
        pred = np.empty(len(y), dtype=object)
        for tr, te in gkf.split(X, y, g):
            m = make_classifier(name).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        fam_true = pd.Series(y).map(cat.FAMILY_OF)
        fam_pred = pd.Series(pred).map(cat.FAMILY_OF)
        rows.append({"model": name, "cv_accuracy": accuracy_score(y, pred),
                     "cv_balanced_accuracy": balanced_accuracy_score(y, pred),
                     "cv_macro_F1": f1_score(y, pred, average="macro"),
                     "cv_family_accuracy": accuracy_score(fam_true, fam_pred)})
        oof[name] = pred
        print(rows[-1], flush=True)
    return pd.DataFrame(rows), oof


def rules_baseline(heats):
    """The AISI/SAE decision list on the simulated heats (it has nothing to
    train, so every heat is a test heat)."""
    y = heats["category"].to_numpy()
    pred = RuleClassifier().predict(heats).predicted.to_numpy()
    fam_true = pd.Series(y).map(cat.FAMILY_OF)
    fam_pred = pd.Series(pred).map(cat.FAMILY_OF)
    return {"model": "aisi_rules", "cv_accuracy": accuracy_score(y, pred),
            "cv_balanced_accuracy": balanced_accuracy_score(y, pred),
            "cv_macro_F1": f1_score(y, pred, average="macro"),
            "cv_family_accuracy": accuracy_score(fam_true, fam_pred)}, pred


def rules_agreement(model, sb):
    """How often the ML classifier and the decision list agree on real heats,
    and how accurate the family is when they agree and when they do not."""
    X = sb[FEATURES].copy()
    for el in FEATURES:
        X[el] = X[el].fillna(config.RESIDUAL_TYPICAL.get(el, 0.0))
    ml = model.predict(X).predicted.to_numpy()
    rl = RuleClassifier().predict(X).predicted.to_numpy()
    agree = ml == rl
    fam_ok = pd.Series(ml).map(COARSE).to_numpy() == sb["family_coarse"].to_numpy()
    rows = [{"subset": "all SteelBench heats", "n": len(sb), "share": 1.0,
             "family_accuracy_ML": float(fam_ok.mean())},
            {"subset": "ML and AISI rules agree on the category", "n": int(agree.sum()),
             "share": float(agree.mean()), "family_accuracy_ML": float(fam_ok[agree].mean())},
            {"subset": "ML and AISI rules disagree", "n": int((~agree).sum()),
             "share": float((~agree).mean()), "family_accuracy_ML": float(fam_ok[~agree].mean())}]
    return pd.DataFrame(rows)


def legacy_maraging():
    """The previous project's dataset (git tag legacy-composition-only-v1)."""
    try:
        txt = subprocess.run(["git", "show", "legacy-composition-only-v1:data/steels_wt_percent.csv"],
                             cwd=config.BASE_DIR, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    df = pd.read_csv(io.StringIO(txt))
    out = pd.DataFrame({el: df[el] if el in df.columns else 0.0 for el in FEATURES})
    out["source"] = "legacy_matbench_maraging"
    return out


def external_tests(model, catalog):
    res = []
    # SteelBench heats: family-level labels only
    sb = pd.read_csv(config.PROCESSED_DIR / "heats_steelbench.csv")
    X = sb[FEATURES].copy()
    for el in FEATURES:
        X[el] = X[el].fillna(config.RESIDUAL_TYPICAL.get(el, 0.0))
    p = model.predict(X)
    sb["pred"] = p.predicted.map(COARSE)
    sb["unknown"] = p.unknown.to_numpy()
    for tier, grp in sb.groupby("data_tier"):
        res.append({"test": f"SteelBench {tier}", "level": "family (carbon / low-alloy / stainless)",
                    "n": len(grp), "accuracy": float((grp.pred == grp.family_coarse).mean()),
                    "flagged_unknown": float(grp.unknown.mean())})
    res.append({"test": "SteelBench all", "level": "family (carbon / low-alloy / stainless)", "n": len(sb),
                "accuracy": float((sb.pred == sb.family_coarse).mean()), "flagged_unknown": float(sb.unknown.mean())})
    # Hot-rolled structural steels (measured heats): rule-based truth
    ext = pd.read_csv(config.PROCESSED_DIR / "structural_external.csv")
    ext = ext[ext.usable].copy()
    truth = np.where(ext.Mn >= 1.0, "carbon_manganese", cat.carbon_class(ext.C))
    pe = model.predict(ext[FEATURES])
    res.append({"test": "Structural S275/S355 members (measured)", "level": "category", "n": len(ext),
                "accuracy": float((pe.predicted.to_numpy() == truth).mean()),
                "flagged_unknown": float(pe.unknown.mean())})
    # Collins dilatometry steels
    col = pd.read_csv(config.RAW_DIR / "collins2023_compositions.csv")
    Xc = col.reindex(columns=FEATURES).fillna(0.0)
    pc = model.predict(Xc)
    truth_c = {"EN3B": "low_carbon", "EN8": "medium_carbon", "SA540": "ni_cr_mo"}
    ok = [pc.predicted.iloc[i] == truth_c[s] for i, s in enumerate(col.steel)]
    res.append({"test": "Collins et al. EN3B / EN8 / SA-540 (measured)", "level": "category", "n": 3,
                "accuracy": float(np.mean(ok)), "flagged_unknown": float(pc.unknown.mean()),
                "detail": "; ".join(f"{s}->{pc.predicted.iloc[i]}" for i, s in enumerate(col.steel))})
    # Open-set test: legacy maraging steels must be flagged unknown
    lg = legacy_maraging()
    if lg is not None:
        pl = model.predict(lg[FEATURES])
        res.append({"test": "Legacy dataset (18Ni-Co-Mo maraging type) - open-set",
                    "level": "should be flagged unknown", "n": len(lg),
                    "accuracy": float(pl.unknown.mean()), "flagged_unknown": float(pl.unknown.mean())})
    return pd.DataFrame(res), sb, lg


def fig_confusion(y, pred, name):
    labels = cat.CATEGORIES
    cm = confusion_matrix(y, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", [viz.SURFACE] + viz.SEQ_BLUE)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    short = [cat.CATEGORY_LABELS[c] for c in labels]
    ax.set_xticks(range(len(labels)), short, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), short, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if cm[i, j] > 0.55 else viz.INK2)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title(f"Category classifier on unseen grades (grade-grouped 5-fold CV, {name})")
    return viz.savefig(fig, "fig03_category_confusion")


def fig_shap(model, heats):
    import shap
    sub = heats.groupby("category", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 60), random_state=0))
    X = sub[FEATURES].to_numpy(float)
    expl = shap.TreeExplainer(getattr(model.clf, "model", model.clf))
    sv = expl.shap_values(X)
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[0] == len(model.clf.classes_):
        sv = np.moveaxis(sv, 0, -1)          # -> (n, features, classes)
    imp = np.abs(sv).mean(axis=0)            # (features, classes)
    classes = list(model.clf.classes_)
    df = pd.DataFrame(imp, index=FEATURES, columns=classes).reindex(columns=cat.CATEGORIES)
    df = df.loc[df.max(axis=1).sort_values(ascending=False).index[:12]]
    df.to_csv(config.RESULTS_DIR / "s02_shap_importance.csv")
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", [viz.SURFACE] + viz.SEQ_BLUE)
    norm = df.div(df.max(axis=0), axis=1)
    ax.imshow(norm.to_numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_yticks(range(len(df)), df.index)
    ax.set_xticks(range(len(df.columns)), [cat.CATEGORY_LABELS[c] for c in df.columns], rotation=50,
                  ha="right", fontsize=8)
    ax.set_title("Which elements decide each category (mean |SHAP|, scaled per category)")
    return viz.savefig(fig, "fig04_category_shap")


def fig_embedding(catalog):
    Z = _log(catalog[[f"{el}_nom" for el in FEATURES]].to_numpy(float))
    Zs = StandardScaler().fit_transform(Z)
    emb = PCA(n_components=2, random_state=0).fit_transform(Zs)
    fams = ["plain_carbon", "free_machining", "carbon_manganese", "low_alloy", "stainless", "tool"]
    names = {"plain_carbon": "Plain carbon", "free_machining": "Free-machining", "carbon_manganese": "C-Mn",
             "low_alloy": "Low-alloy", "stainless": "Stainless", "tool": "Tool"}
    km = KMeans(n_clusters=6, n_init=20, random_state=0).fit(Zs)
    ari = adjusted_rand_score(catalog.family, km.labels_)
    nmi = normalized_mutual_info_score(catalog.family, km.labels_)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, f in enumerate(fams):
        s = catalog.family == f
        ax.plot(emb[s, 0], emb[s, 1], "o", ms=5, color=viz.SLOTS[i], mec=viz.SURFACE, mew=0.8,
                alpha=0.9, label=f"{names[f]} ({int(s.sum())})")
    ax.set_xlabel("PC 1 (log composition)")
    ax.set_ylabel("PC 2 (log composition)")
    ax.legend(loc="best", ncol=2)
    ax.set_title(f"547 grades in composition space; k-means (k=6) vs families: ARI {ari:.2f}, NMI {nmi:.2f}")
    viz.savefig(fig, "fig05_catalog_embedding")
    return ari, nmi


def fig_novelty(model, heats, sb, lg):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 6, 61)
    tr = model.novelty.score(heats.sample(4000, random_state=0))
    ax.hist(np.clip(tr, 0, 6), bins=bins, color=viz.BLUE, alpha=0.85, label="Simulated training heats")
    Xs = sb[FEATURES].copy()
    for el in FEATURES:
        Xs[el] = Xs[el].fillna(config.RESIDUAL_TYPICAL.get(el, 0.0))
    ax.hist(np.clip(model.novelty.score(Xs), 0, 6), bins=bins, color=viz.ORANGE, alpha=0.75,
            label="Real heats (SteelBench)")
    if lg is not None:
        ax.hist(np.clip(model.novelty.score(lg[FEATURES]), 0, 6), bins=bins, color=viz.VIOLET, alpha=0.85,
                label="Legacy maraging steels")
    ax.axvline(1.0, color=viz.INK2, lw=1.2)
    ax.text(1.03, ax.get_ylim()[1] * 0.92, "unknown above 1", color=viz.INK2, fontsize=9)
    ax.set_xlabel("Novelty score (distance to nearest known grade / 99th percentile of training spread)")
    ax.set_ylabel("Heats")
    ax.legend(loc="upper right")
    ax.set_title("Open-set check: the classifier refuses compositions it has never seen")
    return viz.savefig(fig, "fig06_category_novelty")


def main():
    config.ensure_dirs()
    catalog = pd.read_csv(config.PROCESSED_DIR / "grade_catalog.csv")
    rng = np.random.default_rng(config.RANDOM_SEED)
    heats = sample_heats(catalog, N_PER_GRADE, rng)
    print(f"{len(heats)} simulated heats from {catalog.grade.nunique()} grades")
    cv, oof = cross_validate(heats)

    # External tests for every classifier. Selection rule (declared before
    # looking at the numbers): the production classifier is the one with the
    # highest family accuracy on the largest set of *real* heats (SteelBench);
    # simulated-heat CV is reported alongside.
    ext_all, models = [], {}
    for name in CLASSIFIERS:
        m = CategoryModel(name).fit(heats)
        models[name] = m
        e, _, _ = external_tests(m, catalog)
        e["model"] = name
        ext_all.append(e)
    ext_all = pd.concat(ext_all, ignore_index=True)
    ext_all.to_csv(config.RESULTS_DIR / "s02_external_tests_all_models.csv", index=False)
    sb_acc = ext_all[ext_all.test == "SteelBench all"].set_index("model").accuracy
    cv["steelbench_family_accuracy"] = cv.model.map(sb_acc)
    best = sb_acc.idxmax()
    cv["production"] = cv.model == best

    # Baseline: the AISI/SAE decision list (not a candidate for production,
    # because the selection rule above was declared for the ML models).
    rb, rules_pred = rules_baseline(heats)
    er, _, _ = external_tests(RuleClassifier(), catalog)
    er["model"] = "aisi_rules"
    ext_all = pd.concat([ext_all, er], ignore_index=True)
    ext_all.to_csv(config.RESULTS_DIR / "s02_external_tests_all_models.csv", index=False)
    rb["steelbench_family_accuracy"] = float(er.loc[er.test == "SteelBench all", "accuracy"].iloc[0])
    rb["production"] = False
    cv = pd.concat([cv, pd.DataFrame([rb])], ignore_index=True)
    cv.to_csv(config.RESULTS_DIR / "s02_classifier_cv.csv", index=False)
    print(cv.round(3).to_string(index=False))
    print("production classifier:", best)

    rep = pd.DataFrame({"category": cat.CATEGORIES})
    rep["grades"] = rep.category.map(catalog.category.value_counts()).fillna(0).astype(int)
    rep["cv_recall"] = [float(np.mean(oof[best][heats.category.to_numpy() == c] == c)) for c in cat.CATEGORIES]
    rep["aisi_rules_recall"] = [float(np.mean(rules_pred[heats.category.to_numpy() == c] == c))
                                for c in cat.CATEGORIES]
    rep.to_csv(config.RESULTS_DIR / "s02_per_category_recall.csv", index=False)
    print(rep.to_string(index=False))
    fig_confusion(heats.category.to_numpy(), oof[best], best)

    model = models[best]
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "category_model.joblib")
    ext, sb, lg = external_tests(model, catalog)
    ext.to_csv(config.RESULTS_DIR / "s02_external_tests.csv", index=False)
    print(ext.to_string(index=False))
    agr = rules_agreement(model, sb)
    agr.to_csv(config.RESULTS_DIR / "s02_rules_agreement.csv", index=False)
    print(agr.round(3).to_string(index=False))
    fig_shap(model, heats)
    ari, nmi = fig_embedding(catalog)
    pd.DataFrame([{"ARI": ari, "NMI": nmi}]).to_csv(config.RESULTS_DIR / "s02_unsupervised_agreement.csv",
                                                   index=False)
    print(f"unsupervised k-means vs families: ARI {ari:.3f}, NMI {nmi:.3f}")
    fig_novelty(model, heats, sb, lg)


if __name__ == "__main__":
    main()
