"""Steel category taxonomy.

Two levels are used throughout the project.

``family`` (6 classes) - the broad metallurgical family
    plain_carbon, free_machining, carbon_manganese, low_alloy, stainless, tool

``category`` (13 classes) - the label the ML classifier predicts. It splits
plain-carbon steels by carbon content (the property-controlling variable for
ferrite-pearlite steels) and low-alloy steels by their alloying system (the
hardenability-controlling variable), following the AISI/SAE system:

    low_carbon        plain carbon, C <= 0.25 wt%
    medium_carbon     plain carbon, 0.25 < C <= 0.60
    high_carbon       plain carbon, C > 0.60
    free_machining    resulfurised / rephosphorised 11xx, 12xx
    carbon_manganese  15xx and 13xx (Mn 1.0-2.0 %), incl. C-Mn structural grades
    chromium          50xx, 51xx, 52xxx and 61xx (Cr, Cr-V)
    cr_mo             40xx, 41xx, 44xx (Mo, Cr-Mo)
    ni_cr_mo          31xx, 43xx, 46xx, 47xx, 48xx, 81xx, 86xx, 87xx, 88xx, 93xx,
                      94xx, 97xx, 98xx (Ni-bearing: Ni-Cr, Ni-Mo, Ni-Cr-Mo)
    si_mn             92xx (spring steels)
    stainless_austenitic, stainless_ferritic, stainless_martensitic
    tool              AISI tool steels (W, O, A, D, S, H, M, T, P, L)

The carbon thresholds follow the ASM convention (low < 0.25, medium
0.25-0.60, high > 0.60). Precipitation-hardening and duplex stainless steels
are rare in the sources and are merged with the nearest matrix class.
"""

from __future__ import annotations

import re

import numpy as np

CATEGORIES = [
    "low_carbon", "medium_carbon", "high_carbon", "free_machining", "carbon_manganese",
    "chromium", "cr_mo", "ni_cr_mo", "si_mn",
    "stainless_austenitic", "stainless_ferritic", "stainless_martensitic", "tool",
]

CATEGORY_LABELS = {
    "low_carbon": "Low-carbon steel", "medium_carbon": "Medium-carbon steel",
    "high_carbon": "High-carbon steel", "free_machining": "Free-machining steel",
    "carbon_manganese": "C-Mn / Mn steel", "chromium": "Cr / Cr-V steel",
    "cr_mo": "Mo / Cr-Mo steel", "ni_cr_mo": "Ni-Cr-Mo / Ni steel",
    "si_mn": "Si-Mn spring steel", "stainless_austenitic": "Austenitic stainless",
    "stainless_ferritic": "Ferritic stainless", "stainless_martensitic": "Martensitic stainless",
    "tool": "Tool steel",
}

FAMILY_OF = {
    "low_carbon": "plain_carbon", "medium_carbon": "plain_carbon", "high_carbon": "plain_carbon",
    "free_machining": "free_machining", "carbon_manganese": "carbon_manganese",
    "chromium": "low_alloy", "cr_mo": "low_alloy", "ni_cr_mo": "low_alloy", "si_mn": "low_alloy",
    "stainless_austenitic": "stainless", "stainless_ferritic": "stainless",
    "stainless_martensitic": "stainless", "tool": "tool",
}

# Categories whose steels decompose to ferrite + pearlite on slow cooling and
# are therefore inside the scope of the cooling-route models.
FERRITE_PEARLITE_SCOPE = {"low_carbon", "medium_carbon", "high_carbon", "free_machining",
                          "carbon_manganese", "chromium", "cr_mo", "ni_cr_mo", "si_mn"}


def carbon_class(c_nominal):
    c = np.asarray(c_nominal, dtype=float)
    return np.where(c <= 0.25, "low_carbon", np.where(c <= 0.60, "medium_carbon", "high_carbon"))


def aisi_series_category(grade: str, c_nominal: float):
    """Category from an AISI/SAE designation such as 'AISI 4140' or '15B41H'."""
    g = grade.upper().replace("AISI", "").replace("SAE", "").strip()
    m = re.match(r"(\d{2})B?L?(\d{2,3})H?$", g.replace(" ", ""))
    if not m:
        return None
    s = int(m.group(1))
    if s == 10:
        return str(carbon_class(c_nominal))
    if s in (11, 12):
        return "free_machining"
    if s in (13, 15):
        return "carbon_manganese"
    if s in (50, 51, 52, 61):
        return "chromium"
    if s in (40, 41, 44):
        return "cr_mo"
    if s in (23, 25, 31, 32, 33, 34, 43, 46, 47, 48, 81, 86, 87, 88, 93, 94, 97, 98):
        return "ni_cr_mo"
    if s == 92:
        return "si_mn"
    return None


def stainless_category(group: str, ni_nom: float, c_nom: float):
    g = (group or "").lower()
    if "austen" in g:
        return "stainless_austenitic"
    if "ferrit" in g:
        return "stainless_ferritic"
    if "martens" in g or "precip" in g:
        return "stainless_martensitic"
    # heat-resisting and unlabelled grades: austenitic if Ni-rich
    if ni_nom >= 6.0:
        return "stainless_austenitic"
    return "stainless_martensitic" if c_nom >= 0.15 else "stainless_ferritic"
