"""A metallurgist's decision list for the steel category.

This is the baseline the ML classifier has to beat. The AISI/SAE system
defines the categories through composition limits, so the natural question
is whether machine learning is needed at all. Here those limits are written
down as a textbook decision list, applied in the order a metallurgist would
apply them, with nothing fitted to data:

1. Ledeburitic tool steel: Cr >= 10.5 and C >= 1.3 (D-type cold-work steels).
2. Stainless steel: Cr >= 10.5, split as the AISI series are: austenitic
   if Ni >= 6 (300 series) or Mn >= 4 with Ni >= 3 (200 series, Mn instead
   of Ni); martensitic if C >= 0.15, or Cr < 14 with C >= 0.08, or Ni >= 2
   with Cr < 18 (precipitation-hardening and soft-martensitic grades);
   otherwise ferritic. (A first version used the Schaeffler nickel
   equivalent Ni + 30 C + 0.5 Mn, but that diagram describes weld metal, and
   its carbon term sent high-carbon martensitic grades such as 440C to the
   austenitic class.)
3. Outside every category ("unknown"): Ni >= 5 or Co >= 3 (maraging and
   9 % Ni steels), or more than 5 % of substitutional alloying in a steel
   that is not a tool steel (Hadfield, high-Mn and similar).
4. Tool steel: W >= 0.4, Cr >= 3, Mo >= 0.9, V >= 0.5 or Co >= 1.
5. Free-machining: S >= 0.07 (resulphurised 11xx/12xx).
6. Si-Mn spring steel: Si >= 1.0 (92xx).
7. Ni-bearing low-alloy steel: Ni >= 0.25 (31xx, 43xx, 46xx, 47xx, 48xx,
   81xx, 86xx, 87xx, 88xx, 93xx, 94xx).
8. Mo / Cr-Mo: Mo >= 0.08 (40xx, 41xx, 44xx).
9. Cr / Cr-V: Cr >= 0.30 (50xx, 51xx, 52xxx, 61xx).
10. C-Mn: Mn >= 1.0 (13xx, 15xx).
11. Plain carbon, split at 0.25 and 0.60 % C.

The thresholds are the specification limits of the AISI/SAE series (and the
usual residual levels of carbon steels, about 0.2 % for Ni and Cr and
0.06 % for Mo), not values tuned on the training heats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..categories import CATEGORIES, carbon_class

ALLOYING = ("Mn", "Si", "Cr", "Ni", "Mo", "V", "W", "Co", "Cu")


def rule_category(X: pd.DataFrame):
    """Category and an 'outside every category' flag for each heat."""
    g = {el: X[el].to_numpy(float) if el in X else np.zeros(len(X)) for el in
         ("C", "Mn", "Si", "S", "Cr", "Ni", "Mo", "V", "W", "Co", "Cu")}
    C, Mn, Si, S, Cr, Ni, Mo, V, W, Co = (g[k] for k in ("C", "Mn", "Si", "S", "Cr", "Ni", "Mo", "V", "W", "Co"))
    total_alloy = sum(np.nan_to_num(g[el]) for el in ALLOYING)

    out = np.array(carbon_class(C), dtype=object)
    unknown = np.zeros(len(X), dtype=bool)
    decided = np.zeros(len(X), dtype=bool)

    def assign(mask, label):
        nonlocal decided
        m = mask & ~decided
        out[m] = label
        decided |= m
        return m

    assign((Cr >= 10.5) & (C >= 1.3), "tool")
    stainless = (Cr >= 10.5) & ~decided
    assign(stainless & ((Ni >= 6.0) | ((Mn >= 4.0) & (Ni >= 3.0))), "stainless_austenitic")
    assign(stainless & ((C >= 0.15) | ((Cr < 14.0) & (C >= 0.08)) | ((Ni >= 2.0) & (Cr < 18.0))),
           "stainless_martensitic")
    assign(stainless, "stainless_ferritic")
    exotic = ~decided & ((Ni >= 5.0) | (Co >= 3.0))
    unknown |= exotic
    tool = ~decided & ~exotic & ((W >= 0.4) | (Cr >= 3.0) | (Mo >= 0.9) | (V >= 0.5) | (Co >= 1.0))
    assign(tool, "tool")
    unknown |= ~decided & (total_alloy > 5.0)
    assign(S >= 0.07, "free_machining")
    assign(Si >= 1.0, "si_mn")
    assign(Ni >= 0.25, "ni_cr_mo")
    assign(Mo >= 0.08, "cr_mo")
    assign(Cr >= 0.30, "chromium")
    assign(Mn >= 1.0, "carbon_manganese")
    return out, unknown


class RuleClassifier:
    """Same interface as CategoryModel, so it runs through the same tests."""

    name = "aisi_rules"

    def fit(self, heats=None):
        return self

    def predict(self, X: pd.DataFrame, novelty_threshold=None):
        label, unknown = rule_category(X)
        out = pd.DataFrame(0.0, index=X.index, columns=CATEGORIES)
        for c in CATEGORIES:
            out.loc[label == c, c] = 1.0
        out["predicted"] = label
        out["confidence"] = 1.0
        out["novelty"] = np.where(unknown, 2.0, 0.0)
        out["unknown"] = unknown
        return out
