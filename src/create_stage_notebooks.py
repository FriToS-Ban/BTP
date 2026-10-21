"""Presentation notebooks for Stages 4–6.

They read the tables and figures the scripts already wrote. They do not fit
another model. Run the stage scripts first.
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(BASE_DIR, "notebooks")


def make_notebook(cells, filepath):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=2)
    print(f"Wrote {filepath}")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


nb4 = [
    md(
        "# Stage 4. What the trees are using\n\n"
        "This notebook reads the Stage 4 tables. The model is a composition-only predictor "
        "of yield strength. SHAP attributes the prediction. It does not prove a strengthening mechanism.\n\n"
        "The headline model is the default random forest on the 13 non-iron weight percents. "
        "A consistent or inconsistent label is kept only when that forest, the tuned forest, "
        "and tuned XGBoost agree.\n"
    ),
    code(
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "metrics = pd.read_csv('../results/stage4_nofe_cv_metrics.csv')\n"
        "paired = pd.read_csv('../results/stage4_nofe_paired_tests.csv')\n"
        "sanity = pd.read_csv('../results/stage4_metallurgical_sanity.csv')\n"
        "display(metrics)\n"
        "display(paired)\n"
        "display(sanity[['element', 'consensus', 'headline_rho', 'vote_default_rf_13', 'vote_tuned_rf_13', 'vote_tuned_xgb_13', 'reason']])\n"
    ),
    code(
        "for name in [\n"
        "    'fig14_shap_importance_bars.png',\n"
        "    'fig15_shap_beeswarm_default_rf.png',\n"
        "    'fig16_shap_dependence.png',\n"
        "    'fig17_shap_stability_folds.png',\n"
        "    'fig18_shap_family_heatmap.png',\n"
        "    'fig19_shap_interactions.png',\n"
        "    'fig20_partial_dependence_pairs.png',\n"
        "    'fig21_shap_vs_permutation.png',\n"
        "]:\n"
        "    display(Image(filename=f'../figures/{name}'))\n"
    ),
]

nb5 = [
    md(
        "# Stage 5. Local inverse design\n\n"
        "The screening model is fit on all 312 alloys. Scores in this notebook are in-sample. "
        "A feasible trial has a predicted yield of at least 1500 MPa, sits inside the applicability "
        "domain, and keeps iron inside the training range. The criticality axis is wt% times the "
        "EU 2023 supply-risk weights. It is not a price.\n"
    ),
    code(
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "audit = pd.read_csv('../results/stage5_search_audit.csv')\n"
        "candidates = pd.read_csv('../results/stage5_candidates.csv')\n"
        "scales = pd.read_csv('../results/stage5_gp_length_scale_summary.csv')\n"
        "display(audit.T)\n"
        "display(scales)\n"
        "cols = ['seed', 'rank_within_seed', 'pred_mpa', 'lcb_mpa', 'tree_std_mpa', 'criticality', 'domain_label', 'family']\n"
        "display(candidates[cols])\n"
        "display(Image(filename='../figures/fig22_inverse_search.png'))\n"
        "display(Image(filename='../figures/fig23_candidates_ti_ni.png'))\n"
    ),
]

nb6 = [
    md(
        "# Stage 6. A two-objective front\n\n"
        "The axes are the lower confidence bound and the criticality index. "
        "The IIW carbon equivalent is not an axis. The validity figure shows why: "
        "the published CET window does not cover these nickel and molybdenum levels.\n"
    ),
    code(
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "audit = pd.read_csv('../results/stage6_audit.csv')\n"
        "front = pd.read_csv('../results/stage6_pareto_front.csv')\n"
        "counts = pd.read_csv('../results/stage6_cet_validity_counts.csv')\n"
        "display(audit.T)\n"
        "display(counts)\n"
        "display(front)\n"
        "display(Image(filename='../figures/fig24_pareto_front.png'))\n"
        "display(Image(filename='../figures/fig25_carbon_equivalent_validity.png'))\n"
    ),
]

if __name__ == "__main__":
    os.makedirs(NB_DIR, exist_ok=True)
    make_notebook(nb4, os.path.join(NB_DIR, "04_shap_interpretation.ipynb"))
    make_notebook(nb5, os.path.join(NB_DIR, "05_inverse_design.ipynb"))
    make_notebook(nb6, os.path.join(NB_DIR, "06_pareto_front.ipynb"))
