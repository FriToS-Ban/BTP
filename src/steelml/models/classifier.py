"""Steel-category classifier.

Input: a heat analysis (wt% of config.ELEMENTS). Output: probabilities over
the 13 categories of :mod:`steelml.categories`, and a novelty score that
flags compositions unlike any category seen in training (open-set
recognition), so that out-of-scope steels (maraging, Hadfield, high-speed
steels, ...) are reported as "unknown" instead of being forced into the
nearest class.

Training data are heats simulated within published specification ranges
(steelml.data.catalog.sample_heats); evaluation holds out whole grades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .. import config
from ..categories import CATEGORIES

FEATURES = config.ELEMENTS


def _log(X):
    return np.log10(np.asarray(X, float) + 1e-3)


def make_classifier(name, seed=config.RANDOM_SEED):
    if name == "logistic":
        return make_pipeline(FunctionTransformer(_log), StandardScaler(),
                             LogisticRegression(C=3.0, max_iter=3000))
    if name == "knn":
        return make_pipeline(FunctionTransformer(_log), StandardScaler(),
                             KNeighborsClassifier(n_neighbors=15, weights="distance"))
    if name == "mlp":
        return make_pipeline(FunctionTransformer(_log), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
                                           random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return _XGBWrapper(XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                                         subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                         n_jobs=4, eval_metric="mlogloss"))
    raise ValueError(name)


class _XGBWrapper:
    """Label-encoding wrapper so XGBoost exposes string classes like sklearn."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        idx = {c: i for i, c in enumerate(self.classes_)}
        self.model.fit(np.asarray(X, float), np.array([idx[v] for v in y]))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, float))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


CLASSIFIERS = ["logistic", "knn", "mlp", "random_forest", "xgboost"]


NOVELTY_FEATURES = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "W", "Co"]


class Novelty:
    """Open-set score: distance of a heat (log composition, standardised) to
    its nearest training *grade centroid*, relative to the spread of heats
    around their own centroid. Score > 1 means farther from every known
    grade than 99 % of the training heats are from theirs.

    Only the major alloying elements are used: real heat analyses often omit
    trace elements (Al, N, Ti, B, ...), and a missing trace element must not
    make a perfectly ordinary steel look novel."""

    @staticmethod
    def _t(X):
        # log(x + 0.05): resolves alloying levels (0.1-20 %) but compresses
        # residual/trace levels, whose differences carry no metallurgical meaning
        return np.log10(np.asarray(X, float) + 0.05)

    def fit(self, heats: pd.DataFrame):
        Z = self._t(heats[NOVELTY_FEATURES])
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        Zs = (Z - self.mu) / self.sd
        df = pd.DataFrame(Zs)
        df["grade"] = heats["grade"].to_numpy()
        self.centroids = df.groupby("grade").mean().to_numpy()
        d_own = np.linalg.norm(Zs - df.groupby("grade").transform("mean").to_numpy(), axis=1)
        self.ref = float(np.quantile(d_own, 0.99))
        return self

    def score(self, X: pd.DataFrame):
        Zs = (self._t(X[NOVELTY_FEATURES]) - self.mu) / self.sd
        d = np.sqrt(((Zs[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)).min(axis=1)
        return d / self.ref


class CategoryModel:
    """Production classifier (Random Forest, best grade-grouped CV score) +
    novelty detector."""

    def __init__(self, name="random_forest"):
        self.name = name

    def fit(self, heats: pd.DataFrame):
        self.clf = make_classifier(self.name).fit(heats[FEATURES].to_numpy(float), heats["category"].to_numpy())
        self.novelty = Novelty().fit(heats)
        return self

    def predict(self, X: pd.DataFrame, novelty_threshold=1.0):
        Xf = X.reindex(columns=FEATURES).fillna(0.0)
        P = self.clf.predict_proba(Xf.to_numpy(float))
        classes = np.asarray(self.clf.classes_)
        out = pd.DataFrame(P, columns=classes, index=X.index).reindex(columns=CATEGORIES, fill_value=0.0)
        out["predicted"] = out[CATEGORIES].idxmax(axis=1)
        out["confidence"] = out[CATEGORIES].max(axis=1)
        out["novelty"] = self.novelty.score(Xf)
        out["unknown"] = out.novelty > novelty_threshold
        return out
