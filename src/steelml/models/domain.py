"""Applicability domain (AD) of the property models.

A prediction is *in domain* when the steel (i) lies inside the training
range of every alloying element (with a small tolerance) and (ii) is not
farther from its nearest training steel than the training steels typically
are from one another (95th percentile of their own nearest-neighbour
distances), measured in standardised composition space.

The second test catches combinations that are individually in range but
jointly new (for example a steel that is both high-Ni and high-C).

Only C, Mn and Si are checked at the *low* end of their range: every grade
specifies them, so less of them than any training steel is a real
extrapolation. For Cr, Ni, Mo and V the low end of the training range is a
residual level that we had to assume for the plain carbon grades (their
specifications do not limit these elements), so a steel with less of them is
not new; checking it flagged most ordinary carbon steels (for example 62 % of
the measured structural members, mostly for Cr or Mo below 0.04).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AD_FEATURES = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V"]
LOWER_BOUND_CHECKED = ("C", "Mn", "Si")


class Domain:
    def __init__(self, features=AD_FEATURES, tol=0.05, quantile=0.95):
        self.features = list(features)
        self.tol = tol
        self.quantile = quantile

    def fit(self, X: pd.DataFrame):
        A = X[self.features].to_numpy(float)
        # one point per distinct composition (a grade's routes share it)
        A = np.unique(np.round(A, 6), axis=0)
        self.lo = A.min(axis=0)
        self.hi = A.max(axis=0)
        span = np.maximum(self.hi - self.lo, 1e-6)
        self.lo_t = self.lo - self.tol * span
        self.hi_t = self.hi + self.tol * span
        self.mu = A.mean(axis=0)
        self.sd = np.maximum(A.std(axis=0), 1e-3)
        Z = (A - self.mu) / self.sd
        self.Z = Z
        if len(Z) > 1:
            D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(D, np.inf)
            self.ref = float(np.quantile(D.min(axis=1), self.quantile))
        else:
            self.ref = np.inf
        return self

    def check(self, X: pd.DataFrame) -> pd.DataFrame:
        A = X[self.features].to_numpy(float)
        low_checked = np.array([f in LOWER_BOUND_CHECKED for f in self.features])
        below = (A < self.lo_t) & low_checked
        above = A > self.hi_t
        out_feats = [", ".join(f for f, b, a in zip(self.features, rb, ra) if b or a)
                     for rb, ra in zip(below, above)]
        Z = (A - self.mu) / self.sd
        nn = np.sqrt(((Z[:, None, :] - self.Z[None, :, :]) ** 2).sum(-1)).min(axis=1)
        res = pd.DataFrame({
            "in_range": ~(below | above).any(axis=1),
            "out_of_range_elements": out_feats,
            "nn_distance": nn,
            "nn_ratio": nn / self.ref,
        }, index=X.index)
        res["in_domain"] = res.in_range & (res.nn_ratio <= 1.0)
        return res


def logo_domain(F, groups, features=AD_FEATURES):
    """In-domain flag of every row when its whole group is held out."""
    groups = np.asarray(groups)
    flags = pd.DataFrame(index=F.index, columns=["in_range", "out_of_range_elements", "nn_ratio", "in_domain"])
    for g in np.unique(groups):
        te = groups == g
        dom = Domain(features).fit(F.loc[~te])
        chk = dom.check(F.loc[te])
        flags.loc[chk.index, ["in_range", "out_of_range_elements", "nn_ratio", "in_domain"]] = \
            chk[["in_range", "out_of_range_elements", "nn_ratio", "in_domain"]].to_numpy()
    flags["in_domain"] = flags.in_domain.astype(bool)
    flags["in_range"] = flags.in_range.astype(bool)
    return flags
