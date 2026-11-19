"""Composition flags: steels whose properties the models cannot fully explain.

* microalloyed (V >= 0.03, Nb >= 0.01 or Ti >= 0.015 wt %): the training
  grades carry these only as residuals (V only in 6150). Precipitation
  strengthening is therefore added as a separate estimate
  (physics.precipitation) and the prediction is flagged.
* TMCP-like chemistry (C <= 0.12, Mn >= 1.0 and microalloyed): thermo-
  mechanically controlled plate and strip owe their strength to the rolling
  schedule, accelerated cooling and the precipitation it controls, none of
  which a composition and a heat treatment describe. The models apply to such
  a steel only *after* it is re-austenitised (normalised, annealed, hardened);
  its as-delivered properties are out of scope. Full TMCP modelling would need
  rolling-schedule data that no open dataset we found provides.
* boron (B >= 0.0005): 5-30 ppm of soluble boron raises hardenability
  strongly, while the Li kinetics and the Grossmann factors here contain no B
  term. Predicted hardenability is a lower bound (the B grades are kept out of
  the Jominy calibration for the same reason).
"""

from __future__ import annotations

import numpy as np

V_MICRO, NB_MICRO, TI_MICRO = 0.03, 0.01, 0.015
B_ACTIVE = 0.0005


def _g(comp, k):
    v = comp.get(k, 0.0) if hasattr(comp, "get") else comp[k]
    return np.nan_to_num(np.asarray(v, float))


def composition_flags(comp):
    V, Nb, Ti, B, C, Mn = (_g(comp, k) for k in ("V", "Nb", "Ti", "B", "C", "Mn"))
    micro = (V >= V_MICRO) | (Nb >= NB_MICRO) | (Ti >= TI_MICRO)
    tmcp = micro & (C <= 0.12) & (Mn >= 1.0)
    boron = B >= B_ACTIVE
    notes = []
    for m, t, b in zip(np.atleast_1d(micro), np.atleast_1d(tmcp), np.atleast_1d(boron)):
        n = []
        if t:
            n.append("TMCP/HSLA-type chemistry: valid only after re-austenitising; as-delivered "
                     "(controlled-rolled) properties are not modelled")
        elif m:
            n.append("microalloyed: precipitation strengthening added as an estimate (+/-50 %)")
        if b:
            n.append("boron steel: hardenability is a lower bound (no B term in the kinetics)")
        notes.append("; ".join(n))
    return {"microalloyed": micro, "tmcp_like": tmcp, "boron": boron, "composition_notes": np.array(notes)}
