"""Toughness: what the model estimates, what it flags, and what is out of scope.

* Ferrite-pearlite steels (annealed, normalised, stress-relieved): Pickering's
  (1978) impact transition temperature is in physics.properties.itt_pickering.
  It is reported as an estimate, because it was never trained on our data.
* Quenched-and-tempered steels: no Charpy estimate is given. Their toughness
  depends on the prior-austenite grain size, the impurity levels (P, Sn, Sb,
  As) and the cooling after tempering. Our data contain none of these, and no
  open dataset we found links them to measured Charpy values with the full
  heat treatment. We declare it out of scope and raise the two well-documented
  embrittlement warnings instead:
    - tempered martensite embrittlement (TME, "350 C embrittlement"):
      tempering at about 250-400 C (Krauss 2015);
    - temper embrittlement (TE): holding in, or slow cooling through, about
      375-575 C, for steels with Ni, Cr and Mn and impurity P. The usual
      composition index is Watanabe's J = (Mn + Si)(P + Sn) x 10^4 (wt %).
      Tin is not in our compositions, so a typical residual of 0.01 % is
      assumed. J above about 150 is the conventional sign of susceptibility
      (an indicator derived for Cr-Mo steels, used here only as a flag).
"""

from __future__ import annotations

import numpy as np

TME_RANGE_C = (250.0, 400.0)
TE_RANGE_C = (375.0, 575.0)
SN_RESIDUAL = 0.01
J_SUSCEPTIBLE = 150.0


def watanabe_j(comp, sn=SN_RESIDUAL):
    g = lambda k: np.nan_to_num(np.asarray(comp.get(k, 0.0) if hasattr(comp, "get") else comp[k], float))  # noqa: E731
    return (g("Mn") + g("Si")) * (g("P") + sn) * 1e4


def embrittlement_flags(comp, temper_T_C, f_martensite):
    """Flags for a tempered structure: TME, TE and Watanabe J."""
    T = np.asarray(temper_T_C, float)
    fm = np.asarray(f_martensite, float)
    tme = (T >= TME_RANGE_C[0]) & (T <= TME_RANGE_C[1]) & (fm >= 0.5)
    J = watanabe_j(comp)
    g = lambda k: np.nan_to_num(np.asarray(comp.get(k, 0.0) if hasattr(comp, "get") else comp[k], float))  # noqa: E731
    alloyed = (g("Ni") + g("Cr") + g("Mn")) >= 1.5
    te = (T >= TE_RANGE_C[0]) & (T <= TE_RANGE_C[1]) & alloyed & (J >= J_SUSCEPTIBLE)
    return {"TME_risk": tme, "TE_risk": te, "J_factor": J}
