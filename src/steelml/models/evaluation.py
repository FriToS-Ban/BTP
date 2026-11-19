"""Evaluation helpers: grouped cross-validation, paired route differences and
conformal intervals.

Why grouped CV: each grade appears under two or three conditions. A grade's
annealed and normalised rows share the same composition, so random splits
would let the model see the "other half" of a grade during training. All
evaluation here holds out every row of a grade together (leave-one-grade-
out), which answers the question that matters: how well does the model
predict a steel it has never seen?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .property_models import make_model


def logo_oof(F, y, groups, model_name, target):
    """Leave-one-group-out out-of-fold predictions (mu, sd)."""
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    mu = np.full(len(y), np.nan)
    sd = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        m = make_model(model_name, target).fit(F.loc[tr], y[tr])
        p, s = m.predict(F.loc[te])
        mu[te], sd[te] = p, s
    return mu, sd


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    e = p - y
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(ok.sum()), "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "R2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "MAPE_pct": float(100 * np.mean(np.abs(e) / np.abs(y)))}


def paired_route_difference(df, y, p, a="FC", b="AC", key="grade"):
    """Compare predicted and measured property change from route a to route b."""
    d = pd.DataFrame({"key": df[key].to_numpy(), "route": df["route"].to_numpy(), "y": y, "p": p})
    wa = d[d.route == a].set_index("key")
    wb = d[d.route == b].set_index("key")
    common = wa.index.intersection(wb.index)
    dy = wb.loc[common, "y"] - wa.loc[common, "y"]
    dp = wb.loc[common, "p"] - wa.loc[common, "p"]
    out = pd.DataFrame({"delta_true": dy, "delta_pred": dp})
    m = metrics(dy, dp)
    m["sign_agreement"] = float(np.mean(np.sign(dy) == np.sign(dp)))
    return out, m


def conformal_coverage(resid_abs, groups, alpha=0.1):
    """Leave-one-group-out check of absolute-residual conformal intervals.

    For each group the half-width is the ceil((1-alpha)(n+1))/n empirical
    quantile of the absolute OOF residuals of all *other* groups; coverage is
    the fraction of held-out residuals within it. Returns (coverage, mean
    half-width, half-width on all data).
    """
    r = np.asarray(resid_abs, float)
    g = np.asarray(groups)
    covered, widths = [], []
    for grp in np.unique(g):
        te = g == grp
        other = r[~te]
        n = len(other)
        q = np.quantile(other, min(1.0, np.ceil((1 - alpha) * (n + 1)) / n), method="higher")
        covered.extend(list(r[te] <= q))
        widths.append(q)
    n = len(r)
    q_all = np.quantile(r, min(1.0, np.ceil((1 - alpha) * (n + 1)) / n), method="higher")
    return float(np.mean(covered)), float(np.mean(widths)), float(q_all)


def gp_calibration(y, mu, sd, z=1.645):
    ok = np.isfinite(sd) & (sd > 0)
    if not ok.any():
        return np.nan
    return float(np.mean(np.abs(np.asarray(y)[ok] - np.asarray(mu)[ok]) <= z * np.asarray(sd)[ok]))
