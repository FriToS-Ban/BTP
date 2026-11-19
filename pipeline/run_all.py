"""Run the whole pipeline in order and report timings.

    python pipeline/run_all.py              # stages 1-8 (uses data/raw as committed)
    python pipeline/run_all.py --fetch      # re-download and re-parse the sources first
    python pipeline/run_all.py --from 4     # resume from stage 4
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

STAGES = [
    (0, "s00_fetch_data", "fetch and parse sources"),
    (1, "s01_build_datasets", "assemble datasets"),
    (2, "s02_categorize", "steel categories (classifier)"),
    (3, "s03_transformation_model", "transformation model: validation, calibration, CCT"),
    (4, "s04_property_models", "property models: benchmark and production model"),
    (5, "s05_furnace_vs_air", "furnace vs air cooling by category"),
    (6, "s06_external_validation", "external validation"),
    (7, "s07_design_and_active_learning", "inverse design, active learning, lab plan"),
    (8, "s08_hardenability", "hardenability: ideal critical diameter and Jominy H-bands"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    args = ap.parse_args()
    t_all = time.time()
    for num, mod, desc in STAGES:
        if num == 0 and not args.fetch:
            continue
        if num < args.start and num != 0:
            continue
        print(f"\n=== Stage {num}: {desc} ===", flush=True)
        t0 = time.time()
        importlib.import_module(mod).main()
        print(f"--- stage {num} finished in {time.time() - t0:.0f} s", flush=True)
    print(f"\nall done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
