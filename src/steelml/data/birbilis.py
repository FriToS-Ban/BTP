"""Parser for the Birbilis, Ghorbani & Zhao (2025) steel property database.

Source: "A database of mechanical properties of steels", Mendeley Data,
doi:10.17632/jmwb9ddd43.1 (CC BY 4.0).  One sheet, one row per grade and
processing condition, with nominal composition and UTS / yield / elongation.

The 'Processing condition' column is free text (ASM handbook style, e.g.
'Oil Quenched 845C (1550F), 540C (1000F) Temper, 50 mm (2 In.) Round').
`parse` turns it into a route code and numeric heat-treatment fields.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import config

RAW_FILE = config.RAW_DIR / "birbilis2025_steel_database.xlsx"

ELEMENT_COLS = ["C", "Mn", "Si", "P", "S", "Cr", "Ni", "Mo", "V", "Cu",
                "Al", "Ti", "Nb", "N", "Co", "B", "Zr"]

ROUTES = ["AN", "SP", "NO", "NT", "HR", "CD", "Q", "QT", "CARB", "SR", "OTHER"]

CARBON_PREFIX = {"10", "11", "12", "15"}
ALLOY_PREFIX = {"13", "40", "41", "43", "44", "46", "47", "48", "50", "51",
                "52", "61", "81", "86", "87", "88", "92", "93", "94", "98"}

_GRADE_RE = re.compile(
    r"^(?:AISI|SAE)\s+(E?\d{2}[BL]?\d{2})(?:\s*(H))?(?=\s|$)", re.I)

_CARB_RE = re.compile(
    r"carburi[sz]|psueudocarb|pseudocarb|case depth|\bpot\b", re.I)
_CD_RE = re.compile(
    r"cold[\s-]*(drawn|finished|rolled)|as drawn|hot rolled cold drawn|"
    r"hot rolled and cold drawn", re.I)
_QUENCH_RE = re.compile(r"quench", re.I)
_TEMPER_RE = re.compile(r"\btemper(?:ed|ing)?\b", re.I)
_NORM_RE = re.compile(r"normali[sz]", re.I)
_SPH_RE = re.compile(r"spheroidi[sz]", re.I)
_ANNEAL_RE = re.compile(r"anneal", re.I)
_HR_RE = re.compile(r"hot[\s-]*rolled|as[\s-]*rolled|hot[\s-]*finished", re.I)
_SR_RE = re.compile(r"stress[\s-]*relie", re.I)

_MEDIUM_RE = re.compile(
    r"\b(oil|water|brine|polymer)[\s-]*quench|"
    r"quench(?:ed)?\s+in\s+(?:agitated\s+(?:in\s+)?)?(oil|water|brine|polymer)",
    re.I)

# a Celsius value, optionally a range, optionally followed by "(1550F)"
_TEMP_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*C\b"
    r"(?:\s*\([^()]*F\s*\))?", re.I)
_RATE_AFTER_RE = re.compile(r"^\s*(?:/|per\b)", re.I)

_MM_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*mm", re.I)
_MM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\b(?!\s*case)", re.I)
_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:h|hrs?|hours?)\b", re.I)


def load_raw(path=RAW_FILE) -> pd.DataFrame:
    return pd.read_excel(path)


def _clean_name(name: str) -> str:
    s = re.sub(r"[^\x00-\x7f]+", "", str(name))
    s = re.sub(r"\s*(Low Alloy |Alloy |Carbon |Stainless )?Steel$", "", s,
               flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _grade_info(name: str):
    """Return grade, base_grade, family for a material name."""
    m = _GRADE_RE.match(str(name).strip())
    low = str(name).lower()
    if m:
        core = m.group(1).upper()
        grade = core + ("H" if m.group(2) else "")
        base = core[1:] if core.startswith("E") else core
        digits = re.sub(r"[BL]", "", base)
        prefix = digits[:2]
        if prefix in CARBON_PREFIX:
            fam = "carbon"
        elif prefix in ALLOY_PREFIX:
            fam = "alloy"
        else:
            fam = "other"
        return grade, base, fam
    cleaned = _clean_name(name)
    if "cast" in low:
        fam = "cast"
    elif "stainless" in low:
        fam = "stainless"
    elif "tool" in low:
        fam = "tool"
    else:
        fam = "other"
    return cleaned, cleaned, fam


def _temperatures(text: str):
    """Classify every Celsius value in `text` by role."""
    out = {"aust": [], "norm": [], "temper": []}
    prev_end = 0
    for m in _TEMP_RE.finditer(text):
        a = float(m.group(1))
        b = m.group(2)
        val = (a + float(b)) / 2.0 if b else a
        before = text[prev_end:m.start()].lower()
        before = before[-40:]
        after = text[m.end():m.end() + 30].lower()
        prev_end = m.end()
        if _RATE_AFTER_RE.match(after):
            continue  # cooling rate, not a temperature level
        b_strip = before.rstrip(" :(,")
        if re.match(r"^[\s)\]]*temper", after) or \
                re.search(r"temper(?:ed|ing)?(?:\s+(?:at|to))?\W*$", before):
            out["temper"].append(val)
        elif re.search(r"carburi[sz]ed(?:\s+at)?\W*$|cooled\s+to\W*$|"
                       r"cooled\s+at\W*$", before):
            continue
        elif re.search(r"normali[sz]ed?[^0-9]*$", before) and \
                not re.match(r"^\W*(quench|reheat)", after):
            out["norm"].append(val)
        elif re.match(r"^\W*(quench|reheat)", after):
            out["aust"].append(val)
        elif re.search(r"(anneal\w*|austenitiz\w*|heated\s+to|reheated\s+to|"
                       r"quench(?:ed)?|solution[^0-9]*)"
                       r"(?:\s+(?:at|from))?\W*$", b_strip + " "):
            out["aust"].append(val)
    return out


def _route(t: str) -> str:
    if _CARB_RE.search(t):
        return "CARB"
    if _CD_RE.search(t):
        return "CD"
    if _QUENCH_RE.search(t):
        return "QT" if _TEMPER_RE.search(t) else "Q"
    if _TEMPER_RE.search(t):
        return "NT" if _NORM_RE.search(t) else "QT"
    if _NORM_RE.search(t):
        return "NO"
    if _SPH_RE.search(t):
        return "SP"
    if _ANNEAL_RE.search(t) and not re.search(r"\baged\b", t, re.I):
        return "AN"
    if _HR_RE.search(t):
        return "HR"
    if _SR_RE.search(t):
        return "SR"
    return "OTHER"


def _parse_processing(text) -> dict:
    text = "" if text is None or (isinstance(text, float) and np.isnan(text)) \
        else str(text)
    route = _route(text)
    nan = np.nan
    res = dict(route=route, quench_medium=nan, austenitize_T_C=nan,
               normalize_T_C=nan, temper_T_C=nan, temper_time_h=nan,
               section_mm=nan, section_lo_mm=nan, section_hi_mm=nan)

    temps = _temperatures(text)
    if temps["norm"]:
        res["normalize_T_C"] = temps["norm"][0]
    if temps["temper"]:
        res["temper_T_C"] = temps["temper"][-1]
    if route in ("AN", "SP") and temps["aust"]:
        res["austenitize_T_C"] = temps["aust"][0]
    elif temps["aust"]:
        res["austenitize_T_C"] = temps["aust"][-1]
    if route == "NO":
        if np.isnan(res["normalize_T_C"]) and temps["aust"]:
            res["normalize_T_C"] = temps["aust"][0]
        res["austenitize_T_C"] = res["normalize_T_C"]
    if route not in ("QT", "NT", "CARB"):
        res["temper_T_C"] = nan

    m = _MEDIUM_RE.search(text)
    if m and route in ("QT", "Q", "CARB"):
        res["quench_medium"] = (m.group(1) or m.group(2)).lower()

    # temper time: hours just after the word 'temper'
    for tm in _TEMPER_RE.finditer(text):
        win = text[tm.end():tm.end() + 25]
        win = re.split(r"carburi|[,;]", win, flags=re.I)[0]
        hm = _TIME_RE.search(win)
        if hm:
            res["temper_time_h"] = float(hm.group(1))
            break

    rng = _MM_RANGE_RE.search(text)
    if rng:
        res["section_lo_mm"] = float(rng.group(1))
        res["section_hi_mm"] = float(rng.group(2))
    else:
        ms = _MM_RE.findall(text)
        if ms:
            res["section_mm"] = float(ms[-1])
    return res


def parse(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        name = r["Name"]
        proc = r["Processing condition"]
        grade, base, fam = _grade_info(name)
        p = _parse_processing(proc)
        text = "" if pd.isna(proc) else str(proc)
        rec = dict(entry=r["Entry"], name=name, grade=grade,
                   base_grade=base, family=fam, processing=proc)
        rec.update(p)
        rec["double_quench"] = bool(re.search(r"double[\s-]*quench", text, re.I))
        rec["cold_worked"] = bool(_CD_RE.search(text))
        rec["carburizing"] = bool(_CARB_RE.search(text))
        rec["stress_relieved"] = bool(_SR_RE.search(text))
        for e in ELEMENT_COLS:
            v = pd.to_numeric(r.get(e, np.nan), errors="coerce")
            rec[e] = 0.0 if pd.isna(v) else float(v)
        for out, col in (("UTS_MPa", "(Ultimate) Tensile strength (MPa)"),
                         ("YS_MPa", "Yield strength (MPa)"),
                         ("El_pct", "Ductility (%)")):
            v = pd.to_numeric(r.get(col, np.nan), errors="coerce")
            rec[out] = float(v) if (pd.notna(v) and v > 0) else np.nan
        rec["cluster_label"] = r.get("Cluster label", np.nan)
        rows.append(rec)
    out = pd.DataFrame(rows)

    is_cast = (out["family"] == "cast") | \
        out["name"].astype(str).str.contains("cast", case=False)
    out["in_scope"] = (out["family"].isin(["carbon", "alloy"])
                       & (out["Cr"] < 3.5) & (out["Ni"] < 5.5)
                       & (out["Mo"] < 1.2) & ~is_cast)

    seen: dict = {}
    dup = []
    keycols = ["base_grade", "route", "quench_medium", "austenitize_T_C",
               "temper_T_C", "section_mm", "UTS_MPa", "YS_MPa", "El_pct"]
    for entry, *vals in out[["entry"] + keycols].itertuples(index=False):
        key = tuple(None if (isinstance(v, float) and np.isnan(v)) or v is None
                    else v for v in vals)
        if key in seen:
            dup.append(seen[key])
        else:
            seen[key] = entry
            dup.append(np.nan)
    out["dup_of"] = dup

    cols = ["entry", "name", "grade", "base_grade", "family", "in_scope",
            "processing", "route", "quench_medium", "austenitize_T_C",
            "normalize_T_C", "temper_T_C", "temper_time_h", "section_mm",
            "section_lo_mm", "section_hi_mm", "double_quench", "cold_worked",
            "carburizing", "stress_relieved"] + ELEMENT_COLS + \
           ["UTS_MPa", "YS_MPa", "El_pct", "cluster_label", "dup_of"]
    return out[cols]


def summary(parsed: pd.DataFrame) -> pd.DataFrame:
    """Row counts by route x in_scope."""
    tab = pd.crosstab(parsed["route"], parsed["in_scope"].astype(bool))
    tab = tab.rename(columns={False: "out_of_scope", True: "in_scope"})
    tab = tab.reindex(index=ROUTES, columns=["out_of_scope", "in_scope"],
                      fill_value=0)
    tab["total"] = tab.sum(axis=1)
    return tab


def unparsed(parsed: pd.DataFrame) -> pd.DataFrame:
    """In-scope rows whose processing text did not map to any route."""
    return parsed[parsed["in_scope"] & (parsed["route"] == "OTHER")]
