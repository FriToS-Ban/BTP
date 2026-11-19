"""Registry of every external data source used in the project.

Each entry records where the file comes from, its licence and how to cite
it. ``pipeline/s00_fetch_data.py`` downloads the files into
``data/external`` (not committed) and writes the parsed source tables to
``data/raw`` (committed, small), so the rest of the pipeline runs offline.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .. import config

USER_AGENT = "steelml-btp-research/2.0 (academic, one-time download)"


@dataclass
class Source:
    key: str
    url: str
    filename: str
    licence: str
    citation: str
    kind: str                 # what the source provides
    value_type: str = ""      # typical / specification minimum / measured / composition range
    notes: str = ""
    optional: bool = False
    extra: dict = field(default_factory=dict)


SOURCES = [
    Source(
        key="asm_carbon",
        url="https://www.engineersedge.com/materials/carbon-steel-properties.htm",
        filename="engineersedge_carbon_steel_properties.html",
        licence="Factual data reproduced from the ASM Handbook; page (c) Engineers Edge",
        citation=("ASM International, Metals Handbook Vol. 1, 10th ed. (1990), table 'Mechanical "
                  "properties of selected carbon and alloy steels in the hot-rolled, normalized and "
                  "annealed condition' (data of Bethlehem Steel Co.), as reproduced by Engineers Edge."),
        kind="tensile properties: hot rolled / normalized / annealed (carbon steels)",
        value_type="typical",
    ),
    Source(
        key="asm_alloy",
        url="https://www.engineersedge.com/materials/carbon-steel-properties1.htm",
        filename="engineersedge_alloy_steel_properties.html",
        licence="Factual data reproduced from the ASM Handbook; page (c) Engineers Edge",
        citation="As for asm_carbon (alloy-steel part of the same ASM table).",
        kind="tensile properties: normalized / annealed (alloy steels)",
        value_type="typical",
    ),
    Source(
        key="summit_carbon",
        url="https://summitsteel.com/steel-resources_summit-steel-corp/chemical-ranges/carbon-steel/",
        filename="summit_carbon_steel_ranges.html",
        licence="Factual specification data (SAE J403); page (c) Summit Steel",
        citation="SAE J403 chemical composition ranges of carbon steels, as tabulated by Summit Steel Corp.",
        kind="composition ranges: AISI/SAE 10xx, 11xx, 12xx, 15xx and carbon H-steels",
        value_type="composition range",
    ),
    Source(
        key="summit_alloy_h",
        url="https://www.summitsteel.com/steel-resources_summit-steel-corp/chemical-ranges/alloy-h-steel/",
        filename="summit_alloy_h_ranges.html",
        licence="Factual specification data (SAE J1268); page (c) Summit Steel",
        citation="SAE J1268 chemical composition ranges of alloy H-steels, as tabulated by Summit Steel Corp.",
        kind="composition ranges: AISI/SAE alloy H-steels",
        value_type="composition range",
    ),
    Source(
        key="wiki_sae",
        url=("https://en.wikipedia.org/w/api.php?action=parse&page=SAE_steel_grades&prop=text"
             "&format=json&formatversion=2"),
        filename="wikipedia_SAE_steel_grades.json",
        licence="CC BY-SA 4.0",
        citation="Wikipedia contributors, 'SAE steel grades', Wikipedia, The Free Encyclopedia.",
        kind="SAE family definitions; stainless steel composition table",
        value_type="composition range",
    ),
    Source(
        key="wiki_tool",
        url=("https://en.wikipedia.org/w/api.php?action=parse&page=Tool_steel&prop=text"
             "&format=json&formatversion=2"),
        filename="wikipedia_Tool_steel.json",
        licence="CC BY-SA 4.0",
        citation="Wikipedia contributors, 'Tool steel', Wikipedia, The Free Encyclopedia.",
        kind="tool steel compositions (O, A, D grades)",
        value_type="composition range",
    ),
    Source(
        key="zenodo_grades",
        url="https://zenodo.org/api/records/17704410/files/Carbon_stainless_new222.xlsx/content",
        filename="zenodo17704410_Carbon_stainless_grades.xlsx",
        licence="CC BY 4.0",
        citation=("Carbon and Stainless Steel Grades (2025), Zenodo, doi:10.5281/zenodo.17704410. "
                  "Compiled from MatWeb, MakeItFrom and steel-grades.com."),
        kind="composition ranges + properties + processing flags of carbon and stainless grades",
        value_type="mixed: typical (MatWeb/ASM) and specification-level values",
    ),
    Source(
        key="zenodo_cleveland",
        url=("https://zenodo.org/api/records/19000367/files/"
             "Cleveland%20test%20data_Kingsley%20et%20al%202026.xlsx/content"),
        filename="zenodo19000367_Cleveland_reused_members.xlsx",
        licence="CC BY 4.0",
        citation=("Kingsley et al. (2026) Experimental test results on reused steel members, Zenodo, "
                  "doi:10.5281/zenodo.19000367 (data courtesy of Cleveland Steel & Tubes)."),
        kind="measured heat chemistry + tensile + Charpy + hardness of 182 structural members",
        value_type="measured",
    ),
    Source(
        key="steelbench",
        url="https://zenodo.org/api/records/18530558/files/steelbench_core_open.csv/content",
        filename="steelbench_core_open.csv",
        licence="CC BY 4.0",
        citation=("Volkov, A. et al. (2026) SteelBench v1.0, Zenodo, doi:10.5281/zenodo.18530558."),
        kind="heat-level chemistry with steel-family labels (and mixed-provenance properties)",
        value_type="mixed: specification minima and measured",
    ),
    Source(
        key="collins2023",
        url="https://zenodo.org/api/records/7770264/files/Supplementary%20Material.zip/content",
        filename="collins2023_members",
        licence="CC BY 4.0",
        citation=("Collins, J. et al. (2023) Supplementary material for 'A Rapid, Open-Source CCT "
                  "Predictor for Low Alloy Steels...', Zenodo, doi:10.5281/zenodo.7770264; paper: "
                  "Metals 13 (2023) 1168."),
        kind="measured transformation-start temperatures and microhardness at 9 cooling rates, "
             "chemistry and prior-austenite grain size of EN3B, EN8 and SA-540 B24",
        value_type="measured",
        notes=("The 231 MB archive is not downloaded: only the 15 small members listed in "
               "COLLINS_MEMBERS are read from it with HTTP range requests."),
        extra={"zip_members": True},
    ),
    Source(
        key="birbilis2025",
        url=("https://data.mendeley.com/public-files/datasets/jmwb9ddd43/files/"
             "cc944ab8-d3c8-448c-ac1d-172e8cc7e11d/file_downloaded"),
        filename="birbilis2025_steel_database.xlsx",
        licence="CC BY 4.0",
        citation=("Birbilis, N., Ghorbani, M., Zhao, S. (2025) A database of mechanical properties of steels, "
                  "Mendeley Data, V1, doi:10.17632/jmwb9ddd43.1."),
        kind=("3234 steels with composition, a written processing condition (quench medium, austenitising "
              "and tempering temperature, bar size) and yield strength, tensile strength and ductility; "
              "used for the quenched-and-tempered and the bar-size data (stages 1, 6, 9, 10)"),
        value_type="typical (handbook-type data, measured on the stated bar sizes)",
        notes=("Compiled from handbook-type sources; the CC BY licence covers the compilation. Repeated "
               "H/non-H rows with identical values are removed (src/steelml/data/birbilis.py)."),
    ),
    Source(
        key="uhcsdb",
        url="https://materialsdata.nist.gov/pid/rest/local/cdcs/450Z3P0I08L7NGQ4",
        filename="uhcs_microstructures.sqlite",
        licence="CC BY 3.0 US",
        citation=("Hecht, M.D., DeCost, B.L., Francis, T., Holm, E.A., Picard, Y.N., Webler, B.A. "
                  "Ultrahigh Carbon Steel Micrographs, NIST Materials Data Repository (formerly "
                  "hdl:11256/940); DeCost et al. (2017) IMMI 6, 197-205."),
        kind="metadata (anneal temperature/time, cooling method, microconstituent) of 961 UHCS micrographs",
        value_type="measured",
        optional=True,
        extra={"images_url": "https://materialsdata.nist.gov/pid/rest/local/cdcs/micrographs.zip"},
    ),
]

_C = "Supplementary Material/"
COLLINS_MEMBERS = {
    "Ts_EN3B": _C + "1. Dilatometry Data/EN3B/4. Experimental CCT/EN3B_900_Measured Ts.xlsx",
    "Ts_EN8": _C + "1. Dilatometry Data/EN8/4. Experimental CCT/EN8_900_Measured Ts.xlsx",
    "Ts_SA540": _C + "1. Dilatometry Data/SA-540/4. Experimental CCT/SA-540_870_2_Measured Ts.xlsx",
    "HV_EN3B": _C + "3. Microhardness/Microhardness Data_EN3B_900.xlsx",
    "HV_EN8": _C + "3. Microhardness/Microhardness Data_EN8_900.xlsx",
    "HV_SA540": _C + "3. Microhardness/Microhardness Data_SA540_870_2.xlsx",
    "chem_EN3B": _C + "8. Alloy Chemical Analysis/Chemical Composition_EN3B.xlsx",
    "chem_EN8": _C + "8. Alloy Chemical Analysis/Chemical Composition_EN8.xlsx",
    "chem_SA540": _C + "8. Alloy Chemical Analysis/Chemical Composition_SA540.xlsx",
    "li_Ts_EN3B": _C + "5. CCT Modelling/EN3B/Li Model/Raw Data/En3B_G=5.6_Li Model_Ts.csv",
    "li_Ts_EN8": _C + "5. CCT Modelling/EN8/Li Model/Raw Data/En8_G=11.4_Li Model_Ts.csv",
    "li_Ts_SA540": _C + "5. CCT Modelling/SA-540/Li Model/Raw Data/SA540_G=6.7_Li Model_Ts.csv",
    "frac_EN3B": _C + "5. CCT Modelling/EN3B/Proposed Model/En3B_G=5.6_CCT fractions.csv",
    "frac_EN8": _C + "5. CCT Modelling/EN8/Proposed Model/En8_G=11.4_CCT fractions.csv",
    "frac_SA540": _C + "5. CCT Modelling/SA-540/Proposed Model/SA540_G=6.7_CCT fractions.csv",
}


def extract_remote_zip_members(url, members: dict, dest_dir: Path):
    """Read selected members of a remote zip via HTTP range requests.

    Only the central directory and the requested members are transferred.
    ``members`` maps a short key to the member path; files are written as
    ``dest_dir/<key><suffix>``.
    """
    import io
    import zipfile

    class _Remote(io.RawIOBase):
        def __init__(self, u):
            self.s = requests.Session()
            self.s.headers["User-Agent"] = USER_AGENT
            r = self.s.get(u, headers={"Range": "bytes=0-0"}, timeout=60)
            self.url = r.url
            self.size = int(r.headers["Content-Range"].split("/")[-1])
            self.pos = 0

        def seekable(self):
            return True

        def readable(self):
            return True

        def seek(self, off, whence=0):
            self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
            return self.pos

        def tell(self):
            return self.pos

        def readinto(self, b):
            n = min(len(b), self.size - self.pos)
            if n <= 0:
                return 0
            for attempt in range(5):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"},
                                   timeout=120)
                    data = r.content
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(3)
            b[:len(data)] = data
            self.pos += len(data)
            return len(data)

    dest_dir.mkdir(parents=True, exist_ok=True)
    todo = {k: m for k, m in members.items() if not (dest_dir / (k + Path(m).suffix)).exists()}
    if todo:
        zf = zipfile.ZipFile(io.BufferedReader(_Remote(url), buffer_size=1 << 20))
        for key, member in todo.items():
            with zf.open(member) as src:
                (dest_dir / (key + Path(member).suffix)).write_bytes(src.read())
    return dest_dir

SOURCE_BY_KEY = {s.key: s for s in SOURCES}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(source: Source, dest_dir: Path = config.EXTERNAL_DIR, force=False, timeout=120):
    """Download one source into ``dest_dir`` (resumable for large files)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / source.filename
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(4):
        try:
            with requests.get(source.url, headers=headers, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
                tmp.rename(path)
                return path
        except requests.RequestException as exc:
            if attempt == 3:
                raise RuntimeError(f"download failed for {source.key}: {exc}") from exc
            time.sleep(5 * (attempt + 1))
    return path
