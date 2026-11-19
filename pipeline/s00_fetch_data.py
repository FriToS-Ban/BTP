"""Stage 0 - fetch every external source and extract its tables.

    python pipeline/s00_fetch_data.py            # required sources
    python pipeline/s00_fetch_data.py --all      # also the 231 MB Collins et al. archive

Downloads go to data/external/ (git-ignored). Parsed source tables, the
licence/checksum manifest and the list of curation corrections go to
data/raw/ (committed), so later stages run without network access.
"""

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steelml import config  # noqa: E402
from steelml.data import parsers, sources  # noqa: E402


def main(fetch_all=False):
    config.ensure_dirs()
    manifest = []
    for src in sources.SOURCES:
        if src.optional and not fetch_all:
            path = config.EXTERNAL_DIR / src.filename
            if not path.exists():
                print(f"[skip optional] {src.key}")
                continue
        print(f"[fetch] {src.key}")
        if src.extra.get("zip_members"):
            folder = sources.extract_remote_zip_members(src.url, sources.COLLINS_MEMBERS,
                                                        config.EXTERNAL_DIR / "collins2023")
            files = sorted(folder.iterdir())
            nbytes = sum(f.stat().st_size for f in files)
            digest = sources.sha256(files[0]) if files else ""
        else:
            path = sources.download(src)
            nbytes, digest = path.stat().st_size, sources.sha256(path)
        manifest.append({
            "key": src.key, "url": src.url, "file": src.filename, "bytes": nbytes,
            "sha256": digest, "licence": src.licence, "value_type": src.value_type,
            "provides": src.kind, "citation": src.citation,
            "checked_on": dt.date.today().isoformat(),
        })
    pd.DataFrame(manifest).to_csv(config.RAW_DIR / "source_manifest.csv", index=False)

    ext = config.EXTERNAL_DIR
    raw = config.RAW_DIR

    asm = pd.concat([
        parsers.parse_asm_engineersedge(ext / "engineersedge_carbon_steel_properties.html", "asm_carbon"),
        parsers.parse_asm_engineersedge(ext / "engineersedge_alloy_steel_properties.html", "asm_alloy"),
    ], ignore_index=True)
    asm.to_csv(raw / "asm_bethlehem_properties.csv", index=False)
    print(f"ASM/Bethlehem property rows: {len(asm)} ({asm.grade.nunique()} grades)")

    sc = parsers.parse_summit_carbon(ext / "summit_carbon_steel_ranges.html")
    sc.to_csv(raw / "summit_carbon_ranges.csv", index=False)
    sa = parsers.parse_summit_alloy_h(ext / "summit_alloy_h_ranges.html")
    sa.to_csv(raw / "summit_alloy_h_ranges.csv", index=False)
    print(f"Summit carbon grades: {len(sc)}, alloy H grades: {len(sa)}")

    fam = parsers.parse_wiki_sae_families(ext / "wikipedia_SAE_steel_grades.json")
    fam.to_csv(raw / "wiki_sae_families.csv", index=False)
    ss = parsers.parse_wiki_stainless(ext / "wikipedia_SAE_steel_grades.json")
    ss.to_csv(raw / "wiki_stainless_ranges.csv", index=False)
    tool = parsers.parse_wiki_tool(ext / "wikipedia_Tool_steel.json")
    tool.to_csv(raw / "wiki_tool_ranges.csv", index=False)
    print(f"Wikipedia: {len(fam)} SAE families, {len(ss)} stainless, {len(tool)} tool grades")

    # CC BY 4.0 files are small and are kept verbatim for provenance
    for name in ("zenodo17704410_Carbon_stainless_grades.xlsx",
                 "zenodo19000367_Cleveland_reused_members.xlsx",
                 "steelbench_core_open.csv", "birbilis2025_steel_database.xlsx"):
        shutil.copy2(ext / name, raw / name)
    cl = parsers.load_cleveland(ext / "zenodo19000367_Cleveland_reused_members.xlsx")
    cl.to_csv(raw / "cleveland_members_parsed.csv", index=False)
    print(f"Cleveland members parsed: {len(cl)}")

    col = parsers.parse_collins(ext / "collins2023")
    for name, frame in col.items():
        frame.to_csv(raw / f"collins2023_{name}.csv", index=False)
    print("Collins et al. 2023: " + ", ".join(f"{k} {len(v)}" for k, v in col.items()))

    pd.DataFrame(parsers.CORRECTIONS, columns=["source", "grade", "field", "as_published", "corrected",
                                               "reason"]).to_csv(raw / "source_corrections.csv", index=False)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also fetch optional large sources")
    main(ap.parse_args().all)
