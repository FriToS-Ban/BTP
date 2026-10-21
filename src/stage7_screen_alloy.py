"""Stage 7. Score one composition with the saved screening model.

The model is the in-sample default random forest from Stage 5. A composition
outside the training box or outside the applicability domain is reported as
outside the domain. The printed prediction is then a model evaluation, not a
usable yield strength.

Example, from the repository root:
    python src/stage7_screen_alloy.py --example training-strongest
    python src/stage7_screen_alloy.py --example mild-steel
    python src/stage7_screen_alloy.py --wt Al=0.1,C=0.03,Ni=18,Co=8,Mo=5,Ti=0.4
"""

import argparse
import os

import numpy as np
import pandas as pd

from btp_common import ELEMENTS, NON_FE_ELEMENTS, RES_DIR, assign_family, carbon_equivalents, load_data
from stage5_inverse_design import load_surface


def blank_composition():
    return {el: 0.0 for el in ELEMENTS}


def composition_from_pairs(text):
    values = blank_composition()
    for piece in text.split(","):
        name, raw = piece.split("=")
        name = name.strip()
        if name not in values:
            raise SystemExit(f"Unknown element {name}. Expected one of {ELEMENTS}.")
        values[name] = float(raw)
    if values["Fe"] == 0.0:
        values["Fe"] = 100.0 - sum(values[el] for el in NON_FE_ELEMENTS)
    return values


def main():
    parser = argparse.ArgumentParser(description="Screen one steel composition.")
    parser.add_argument("--wt", default="", help="Comma-separated Element=wt% pairs. Iron is filled as the balance if omitted.")
    parser.add_argument("--example", choices=["training-strongest", "mild-steel"], default="")
    args = parser.parse_args()

    df_wt, y, _groups, _formulas = load_data()
    bundle_path = os.path.join(RES_DIR, "stage5_screening_bundle.joblib")
    if not os.path.exists(bundle_path):
        raise SystemExit("Missing results/stage5_screening_bundle.joblib. Run Stage 5 first.")
    surface = load_surface(bundle_path, df_wt, y)

    if args.example == "training-strongest":
        idx = int(np.argmax(y))
        values = df_wt.iloc[idx].to_dict()
        print(f"Example is training alloy {idx}, measured yield strength {y[idx]:.1f} MPa.")
        print("The score below is in-sample, because this alloy was used to fit the screening model.")
    elif args.example == "mild-steel":
        values = blank_composition()
        values["C"] = 0.15
        values["Mn"] = 0.50
        values["Fe"] = 99.35
        print("Example is a mild-steel composition, far below the dataset minimum of about 1006 MPa.")
    elif args.wt:
        values = composition_from_pairs(args.wt)
    else:
        raise SystemExit("Pass --example or --wt.")

    frame = pd.DataFrame([values])[ELEMENTS]
    solute_sum = float(frame[NON_FE_ELEMENTS].sum(axis=1).iloc[0] + frame["Fe"].iloc[0])
    if abs(solute_sum - 100.0) > 1e-4:
        raise SystemExit(f"Composition sums to {solute_sum:.4f}, not 100 wt%.")
    scored = surface.score_frame(frame).iloc[0]
    outside = not bool(scored["inside_domain"])
    print("---")
    for el in ELEMENTS:
        if frame.iloc[0][el] > 0:
            print(f"{el:2s}  {frame.iloc[0][el]:8.3f} wt%")
    print("---")
    if outside:
        print("OUTSIDE THE APPLICABILITY DOMAIN.")
        print("The numbers below are what the model returns. They are not a yield-strength estimate for this composition.")
    else:
        print(f"Domain label: {scored['domain_label']}.")
    print(f"Predicted yield strength: {scored['pred_mpa']:.1f} MPa")
    print(f"Tree disagreement:        {scored['tree_std_mpa']:.1f} MPa")
    print(f"Lower confidence bound:   {scored['lcb_mpa']:.1f} MPa")
    print(f"GP mean / GP std:         {scored['gp_mean_mpa']:.1f} / {scored['gp_std_mpa']:.1f} MPa")
    print(f"Distance to nearest training alloy (standardized non-Fe): {scored['nearest_dist_std']:.3f}")
    print(f"Criticality index:        {scored['criticality']:.2f}")
    print(f"Family rule:              {assign_family(frame)[0]}")
    ce, pcm = carbon_equivalents(frame)
    print(f"IIW carbon equivalent:    {ce[0]:.3f} (indicator only, usually outside its validity window here)")
    print(f"Pcm:                      {pcm[0]:.3f}")


if __name__ == "__main__":
    main()
