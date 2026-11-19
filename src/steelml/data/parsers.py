"""Parsers that turn each downloaded source into a tidy table.

Every correction applied to a source is listed in ``CORRECTIONS`` with the
reason, and is written to ``data/raw/source_corrections.csv`` by the fetch
step, so the curation is auditable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .. import config

# ---------------------------------------------------------------------------
# Documented corrections
# ---------------------------------------------------------------------------
CORRECTIONS = [
    # (source, grade, field, wrong, corrected, reason)
    ("summit_alloy_h", "1335H", "Si", ".15- 35", "0.15-0.35", "typographical: missing decimal point"),
    ("summit_alloy_h", "4037H", "C", ".39-.41", "0.34-0.41",
     "SAE J1268 4037H carbon is 0.34-0.41; .39-.41 is narrower than any H-band"),
    ("summit_alloy_h", "6150H", "Ni,Mo", ".15-.45 / .08-.15", "none",
     "row shift: 61xx are Cr-V steels without Ni or Mo (SAE family table); values belong to 81B45H"),
    ("summit_alloy_h", "81B45H", "Ni,Mo", "blank", "Ni 0.15-0.45, Mo 0.08-0.15",
     "row shift: 81xx family is Ni 0.30 / Cr 0.40 / Mo 0.12 (SAE family table)"),
    ("summit_alloy_h", "51B60H", "V", ".10-.15V", "none",
     "row shift: 51Bxx are Cr-B steels without V; the V range belongs to 6118H"),
    ("summit_alloy_h", "6118H", "V", ".15Min.V", "0.10-0.15", "row shift of the V annotations"),
    ("summit_alloy_h", "6150H", "V", "blank", "0.15-0.25 (0.15 min)", "row shift of the V annotations; 6150 is V 0.15 min"),
    ("summit_carbon", "1117", "Mn", "1.30/1.60", "1.00-1.30",
     "Mn ranges of 1117 and 1118 swapped relative to SAE J403 (cross-checked with Zenodo 17704410)"),
    ("summit_carbon", "1118", "Mn", "1.00/1.30", "1.30-1.60", "as above"),
    ("summit_carbon", "1212,1213,12L14", "C", "13 / 15", "0.13 max / 0.15 max", "missing decimal point"),
    ("summit_carbon", "15821H", "grade", "15821H", "15B21H",
     "typographical: the boron 'B' printed as '8'; C 0.17-0.24 / Mn 0.70-1.20 are the SAE 15B21H ranges"),
]

NUM = r"(\d*\.?\d+)"


def parse_range(text, single_is_max=False):
    """Parse a composition cell into (min, max).

    Handles '.06 max', '.25/.40 max', '0.10/0.16', '.27-.33', '16–18',
    '0.035 (max)', '.15Min.V' and plain numbers. Returns (nan, nan) for blanks.
    A plain number is a maximum if ``single_is_max`` else a nominal value.
    """
    if text is None:
        return (np.nan, np.nan)
    s = str(text).strip().replace("–", "-").replace("−", "-").replace(",", ".")
    s = s.replace("%", "").strip()
    if s in ("", "-", "--", "nan", "None"):
        return (np.nan, np.nan)
    low = s.lower()
    nums = [float(x) for x in re.findall(NUM, s)]
    if not nums:
        return (np.nan, np.nan)
    if "min" in low and "max" not in low:
        return (nums[0], np.nan)
    if len(nums) >= 2 and re.search(r"\d\s*[/\-]\s*\.?\d", s):
        return (min(nums[0], nums[1]), max(nums[0], nums[1]))
    if "max" in low or single_is_max:
        return (0.0, nums[0])
    return (nums[0], nums[0])


def _tables(html_path):
    soup = BeautifulSoup(Path(html_path).read_bytes(), "lxml")
    out = []
    for t in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])] for r in t.find_all("tr")]
        out.append(rows)
    return out


# ---------------------------------------------------------------------------
# ASM / Bethlehem property table (Engineers Edge reproduction)
# ---------------------------------------------------------------------------

_COND_RE = re.compile(r"(Hot Rolled|Normalized|Annealed)\s*(?:\((\d+)\s*F\))?")


def parse_asm_engineersedge(html_path, source_key):
    rows = []
    for table in _tables(html_path):
        for r in table:
            if len(r) < 6 or not re.fullmatch(r"\d{4}", r[0]):
                continue
            grade = r[0]
            conds = _COND_RE.findall(r[1])
            cols = [re.findall(r"[\d,]+\.?\d*", c) for c in r[2:6]]
            if not all(len(c) == len(conds) for c in cols):
                raise ValueError(f"column mismatch for {grade}: {r}")
            for i, (cond, tf) in enumerate(conds):
                uts_psi = float(cols[0][i].replace(",", ""))
                ys_psi = float(cols[1][i].replace(",", ""))
                rows.append({
                    "grade": f"AISI {grade}",
                    "condition": {"Hot Rolled": "hot_rolled", "Normalized": "normalized",
                                  "Annealed": "annealed"}[cond],
                    "treat_T_F": float(tf) if tf else np.nan,
                    "treat_T_C": round((float(tf) - 32) * 5 / 9, 0) if tf else np.nan,
                    "UTS_MPa": round(uts_psi * config.PSI_TO_MPA, 1),
                    "YS_MPa": round(ys_psi * config.PSI_TO_MPA, 1),
                    "El_pct": float(cols[2][i]),
                    "HB": float(cols[3][i]),
                    "source": source_key,
                    "value_type": "typical",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summit Steel composition ranges
# ---------------------------------------------------------------------------

def parse_summit_carbon(html_path):
    recs = []
    family_by_table = ["10xx", "15xx", "carbon_H", "carbon_boron_H", "11xx", "12xx"]
    for ti, table in enumerate(_tables(html_path)):
        header = [h.lower() for h in table[0]]
        fam = family_by_table[ti] if ti < len(family_by_table) else "other"
        for r in table[1:]:
            if not r or not r[0]:
                continue
            g = r[0].replace("*", "").replace("-H", "H").strip()
            rec = {"grade": f"AISI {g}", "series": fam}
            for col, val in zip(header[1:], r[1:]):
                el = col.split()[0].capitalize()
                txt = val
                if el == "C" and re.fullmatch(r"\d\d", txt):          # '13' -> 0.13 max
                    txt = "0." + txt
                if el == "C" and fam == "12xx":
                    lo, hi = parse_range(txt, single_is_max=True)
                elif el in ("P", "S") and "/" not in txt and "-" not in txt:
                    lo, hi = parse_range(txt, single_is_max=True)
                else:
                    lo, hi = parse_range(txt)
                if el == "S" and "pb" in val.lower():
                    rec["Pb_min"], rec["Pb_max"] = 0.15, 0.35
                rec[f"{el}_min"], rec[f"{el}_max"] = lo, hi
            recs.append(rec)
    df = pd.DataFrame(recs)
    # documented corrections (SAE J403)
    df.loc[df.grade == "AISI 1117", ["Mn_min", "Mn_max"]] = [1.00, 1.30]
    df.loc[df.grade == "AISI 1118", ["Mn_min", "Mn_max"]] = [1.30, 1.60]
    df.loc[df.grade == "AISI 15821H", "grade"] = "AISI 15B21H"
    return df


def parse_summit_alloy_h(html_path):
    recs = []
    for table in _tables(html_path)[:2]:
        for r in table[1:]:
            if not r or not r[0] or not re.match(r"\d", r[0]):
                continue
            g = r[0].replace("*", "").strip()
            rec = {"grade": f"AISI {g}", "series": "alloy_H"}
            for el, val in zip(["C", "Mn", "Si", "Ni", "Cr", "Mo"], r[1:7]):
                if "v" in val.lower():          # vanadium annotations sit in the Mo column
                    rec["V_note"] = val
                    val = ""
                rec[f"{el}_min"], rec[f"{el}_max"] = parse_range(val)
            recs.append(rec)
    df = pd.DataFrame(recs).drop(columns=["V_note"], errors="ignore")
    df = df[~df.grade.str.endswith("**")]  # 4340H** is a supplementary variant
    fix = {
        "AISI 1335H": {"Si_min": 0.15, "Si_max": 0.35},
        "AISI 4037H": {"C_min": 0.34, "C_max": 0.41},
        "AISI 6150H": {"Ni_min": np.nan, "Ni_max": np.nan, "Mo_min": np.nan, "Mo_max": np.nan,
                       "V_min": 0.15, "V_max": 0.25},
        "AISI 81B45H": {"Ni_min": 0.15, "Ni_max": 0.45, "Mo_min": 0.08, "Mo_max": 0.15},
        "AISI 6118H": {"V_min": 0.10, "V_max": 0.15},
    }
    for g, vals in fix.items():
        for k, v in vals.items():
            df.loc[df.grade == g, k] = v
    # SAE J1268 incidental residual limits (note 3 of the table)
    df["Cu_max_incidental"] = 0.35
    # boron steels: 0.0005-0.003 B
    isb = df.grade.str.contains("B", regex=False) & df.grade.str.match(r"AISI \d+B\d+")
    df.loc[isb, ["B_min", "B_max"]] = [0.0005, 0.003]
    return df


# ---------------------------------------------------------------------------
# Wikipedia tables
# ---------------------------------------------------------------------------

def _wiki_soup(json_path):
    html = json.loads(Path(json_path).read_text())["parse"]["text"]
    return BeautifulSoup(html, "lxml")


def parse_wiki_sae_families(json_path):
    soup = _wiki_soup(json_path)
    t = soup.find_all("table", class_="wikitable")[1]
    rows, group = [], None
    for r in t.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
        if len(cells) == 1:
            group = cells[0]
        elif len(cells) == 2 and cells[0] != "SAE designation":
            rows.append({"group": group, "designation": cells[0], "description": cells[1]})
    return pd.DataFrame(rows)


def parse_wiki_stainless(json_path):
    soup = _wiki_soup(json_path)
    t = soup.find_all("table", class_="wikitable")[2]
    recs, group = [], None
    header = None
    for r in t.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
        if cells[:2] == ["SAE", "UNS"]:
            header = cells
            continue
        if len(cells) == 1:
            group = cells[0]
            continue
        if header is None or len(cells) < len(header) - 1:
            continue
        rec = {"grade": f"AISI {cells[0]}", "uns": cells[1], "stainless_group": group}
        for col, val in zip(header[2:], cells[2:]):
            if col == "Other":
                rec["other"] = val
                for el, lo, hi in re.findall(r"(Mo|Ti|Nb|Cu|Al|Se|W|V|N|Co)\s*([\d.]+)\s*[–-]?\s*([\d.]*)", val):
                    rec[f"{el}_min"] = float(lo) if hi else 0.0
                    rec[f"{el}_max"] = float(hi) if hi else float(lo)
                continue
            rec[f"{col}_min"], rec[f"{col}_max"] = parse_range(val, single_is_max=True)
        recs.append(rec)
    return pd.DataFrame(recs)


_TOOL_TOKEN = re.compile(r"([\d.]+)\s*(?:[–-]\s*([\d.]+))?\s*%\s*([A-Z][a-z]?)\b")


def parse_wiki_tool(json_path):
    soup = _wiki_soup(json_path)
    recs = []
    for t in soup.find_all("table", class_="wikitable")[:3]:
        for r in t.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])]
            if len(cells) < 2 or cells[0] == "Grade":
                continue
            grade = re.sub(r"\s*\[.*?\]", "", cells[0]).strip()
            rec = {"grade": f"AISI {grade}", "tool_type": grade[0]}
            for lo, hi, el in _TOOL_TOKEN.findall(cells[1]):
                rec[f"{el}_min"] = float(lo)
                rec[f"{el}_max"] = float(hi) if hi else float(lo)
            recs.append(rec)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# Zenodo datasets
# ---------------------------------------------------------------------------

def load_zenodo_grades(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns={"Treatment_Condition": "treatment"})
    return df


COLLINS_STEELS = {"EN3B": {"T_aust": 900.0, "G": 5.6}, "EN8": {"T_aust": 900.0, "G": 11.4},
                  "SA540": {"T_aust": 870.0, "G": 6.7}}


def parse_collins(member_dir):
    """Collins et al. (2023) measured data -> tidy tables.

    Returns dict of DataFrames: compositions (one row per steel), Ts (measured
    transformation-start temperatures per cooling rate), hardness (mean and
    standard deviation of 16 microhardness indents per rate), li_Ts (the
    authors' own Li-model start temperatures, used only as a reference for
    our independent implementation) and fractions (their modelled constituent
    fractions).
    """
    d = Path(member_dir)
    comps, ts, hv, li, fr = [], [], [], [], []
    for s, meta in COLLINS_STEELS.items():
        chem = pd.read_excel(d / f"chem_{s}.xlsx", header=None).iloc[1:, :2]
        rec = {"steel": s, "T_aust_C": meta["T_aust"], "astm_G": meta["G"]}
        for el, val in zip(chem[0], chem[1]):
            v = str(val).strip()
            if v in ("nan", ""):
                continue
            rec[str(el).strip()] = float(v[1:]) / 2.0 if v.startswith("<") else float(v)
        comps.append(rec)
        t = pd.read_excel(d / f"Ts_{s}.xlsx", header=0).iloc[:, 1:]
        t.columns = ["rate", "tf", "Tf", "Stdf", "tp", "Tp", "Stdp", "tb", "Tb", "Stdb", "tm", "Tm", "Stdm"]
        t["steel"] = s
        ts.append(t)
        h = pd.read_excel(d / f"HV_{s}.xlsx", header=None)
        rates = h.iloc[1, 1:].astype(float).to_numpy()
        vals = h.iloc[2:, 1:].astype(float).to_numpy()
        for j, r in enumerate(rates):
            col = vals[:, j]
            col = col[np.isfinite(col)]
            hv.append({"steel": s, "rate": r, "HV_mean": col.mean(), "HV_std": col.std(ddof=1),
                       "HV_median": np.median(col), "n": len(col)})
        lt = pd.read_csv(d / f"li_Ts_{s}.csv").rename(columns={"Rate": "rate"})
        lt["steel"] = s
        li.append(lt)
        f = pd.read_csv(d / f"frac_{s}.csv").rename(columns={"Rate": "rate"})
        f["steel"] = s
        fr.append(f)
    return {"compositions": pd.DataFrame(comps), "Ts": pd.concat(ts, ignore_index=True),
            "hardness": pd.DataFrame(hv), "li_Ts": pd.concat(li, ignore_index=True),
            "fractions": pd.concat(fr, ignore_index=True)}


def load_cleveland(xlsx_path):
    """Measured structural-steel members (Kingsley et al. 2026).

    The sheet has a 6-row banner; values such as '<0,01' are below detection
    limit and are set to half the limit.
    """
    raw = pd.read_excel(xlsx_path, sheet_name=0, header=None)
    hdr_row = 5
    header = [str(h).replace("\n", " ").strip() for h in raw.iloc[hdr_row]]
    df = raw.iloc[hdr_row + 1:].copy()
    df.columns = header
    df = df[df.iloc[:, 0].notna()]

    def num(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().replace(",", ".")
        if s.startswith("<"):
            try:
                return float(s[1:]) / 2.0
            except ValueError:
                return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    out = pd.DataFrame({
        "entry": df.iloc[:, 0].astype(str),
        "spec": df.iloc[:, 4].astype(str),
        "member_type": df.iloc[:, 5].astype(str),
        "member_size": df.iloc[:, 6].astype(str),
        "thickness_mm": df.iloc[:, 10].map(num),
        "YS_MPa": df.iloc[:, 13].map(num),
        "yield_type": df.iloc[:, 14].astype(str),
        "UTS_MPa": df.iloc[:, 15].map(num),
        "gauge_length_mm": df.iloc[:, 16].map(num),
        "El_pct": df.iloc[:, 17].map(num),
        "El_converted_pct": df.iloc[:, 18].map(num),
        "RA_pct": df.iloc[:, 19].map(num),
        "charpy_T_C": df.iloc[:, 22].map(num),
        "charpy_avg_J": df.iloc[:, 26].map(num),
    })
    for el in ["C", "Si", "Mn", "P", "S", "N", "Cu", "Ni", "Cr", "Mo", "Ti", "Nb", "Al", "V"]:
        col = [c for c in df.columns if c == el]
        out[el] = df[col[0]].map(num).to_numpy() if col else np.nan
    out["CEV_reported"] = df["CEV"].map(num).to_numpy() if "CEV" in df.columns else np.nan
    return out.reset_index(drop=True)
