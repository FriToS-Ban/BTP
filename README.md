# Machine-Learning Prediction of Mechanical Properties of Steels from Composition, with Inverse Alloy Design

**B.Tech Project (MTP-400A), Department of Metallurgical and Materials Engineering, Indian Institute of Technology Roorkee**

**Authors:** Aviral Singh, Shibasish Banerjee
**Supervisor:** Dr. Anish Karmakar

---

## Overview

Yield strength in structural steels depends on both chemical composition and thermo-mechanical processing. This project asks how much of that strength can be predicted from composition alone, what a model learns when it does so, and whether such a model can responsibly propose new compositions.

The pipeline has four parts:

1. **Forward prediction.** Predict yield strength (MPa) from 14 elemental weight percents, using linear baselines, Random Forest, XGBoost, a neural network, and a Gaussian process.
2. **Leakage-aware evaluation.** Group near-duplicate alloys into clusters and evaluate with 5×5 repeated grouped cross-validation, nested hyperparameter tuning, and Nadeau-Bengio corrected paired tests.
3. **Interpretation.** Attribute predictions with SHAP and compare the attributions with expectations from physical metallurgy, under a predeclared consensus rule.
4. **Inverse design.** Search locally around training alloys for compositions with a high lower confidence bound on strength, and map the trade-off against a criticality index built from EU 2023 supply-risk weights.

## Key Results

Values are the mean ± standard deviation across 25 outer folds of repeated grouped cross-validation.

| Model | RMSE (MPa) | MAE (MPa) | R² |
|:---|:---:|:---:|:---:|
| Default Random Forest | 158.3 ± 26.3 | 118.6 ± 22.4 | 0.684 ± 0.119 |
| Tuned Random Forest | 159.6 ± 28.5 | 119.7 ± 22.8 | 0.686 ± 0.104 |
| Tuned XGBoost | 160.9 ± 31.8 | 120.8 ± 25.3 | 0.676 ± 0.126 |
| Default XGBoost | 162.6 ± 30.6 | 122.9 ± 24.0 | 0.669 ± 0.124 |
| PyTorch MLP | 173.8 ± 39.8 | 123.1 ± 27.5 | 0.619 ± 0.168 |
| Gaussian Process (ARD Matérn) | 183.0 ± 43.8 | 124.1 ± 24.9 | 0.590 ± 0.149 |
| Ridge (13 non-Fe features) | 239.2 ± 45.1 | 173.0 ± 36.2 | 0.320 ± 0.161 |
| Dummy (mean) | 300.3 ± 49.9 | 232.0 ± 31.0 | −0.059 ± 0.055 |

- **Grouping changes the result.** Naive repeated cross-validation gives a Random Forest RMSE of 142.8 MPa and R² of 0.764, because near-duplicate alloys appear on both sides of the split. The grouped protocol gives 158.3 MPa and 0.684.
- **No model difference survives the corrected test.** Tuning did not beat the default Random Forest (corrected p = 0.837). The MLP and Gaussian process differences are significant on a plain paired test and not after the Nadeau-Bengio correction.
- **Attributions are stable, and mechanisms remain open.** Titanium has the largest mean absolute SHAP value (about 119 MPa on the default forest). Only aluminium and cobalt receive a "consistent" label under the consensus rule, because the models disagree on titanium and nickel. Rankings agree with permutation importance (Spearman 0.97).
- **The inverse search did not beat the training data.** Its best candidate has an in-sample lower confidence bound of 1665 MPa, against 2408 MPa for the best training alloy, and the two random seeds do not converge on one region. The Pareto front reaches 2212 MPa at low criticality, which is also below the best training alloys.
- **Carbon equivalent was audited and not optimized.** The published CET validity window contains 0 of the 312 alloys, so the formula is not a suitable objective for this dataset.

## Limitations

- **Composition only.** The dataset has no processing variables, so the model cannot separate chemistry from unrecorded heat treatment. Residual scatter of about 150 MPa includes that missing information.
- **High-strength domain.** Every alloy is above 1000 MPa. The model does not apply to ordinary mild steels.
- **Attribution is not causation.** SHAP explains the model. Statements about mechanisms are hypotheses and are worded as "consistent with".
- **Inverse-design scores are in-sample.** The screening model is fit on all 312 alloys, and tree disagreement is not a calibrated probability. Candidates are not recommendations to melt an alloy.
- **The applicability-domain link is exploratory.** The cluster-level near-versus-far gap in MAE is large (85.5 versus 151.6 MPa), but the distance metric was chosen after inspecting results and the continuous rank correlation is not significant (ρ = 0.086, p = 0.225).
- **No external validation yet.** All reported errors come from cross-validation on this dataset.

## Dataset

- **Source:** Matbench v0.1 `matbench_steels` (Dunn et al., 2020), derived from a Citrine Informatics dataset of steel mechanical properties.
- **Size:** 312 compositions.
- **Features:** Chemical formulas parsed with `pymatgen` into weight percents of 14 elements (Al, C, Co, Cr, Fe, Mn, Mo, N, Nb, Ni, Si, Ti, V, W). Iron is the balance, so most models use the 13 non-iron features.
- **Target:** Yield strength, 1005.9 to 2510.3 MPa.

## Repository Structure

```
BTP/
├── data/                  Processed composition and target tables (CSV)
├── figures/               Figures 1 to 25 (PNG)
├── notebooks/             Executed notebooks that present each stage
├── results/               Metrics, audits, candidates, and the screening model
├── src/
│   ├── btp_common.py                Shared data loading and helpers
│   ├── applicability_domain.py      Applicability-domain rule
│   ├── stage1_data_eda.py           Data parsing and exploratory analysis
│   ├── stage2_baselines.py          Baselines and grouped cross-validation
│   ├── stage2_addendum_audit.py     Cluster-level applicability-domain audit
│   ├── stage3_complete_audited.py   Nested tuning and audits (authoritative Stage 3)
│   ├── stage4_shap.py               SHAP interpretation
│   ├── stage5_inverse_design.py     Local inverse design with uncertainty
│   ├── stage6_pareto.py             Two-objective Pareto front
│   ├── stage7_screen_alloy.py       Command-line screening of one composition
│   ├── stage3_tuning.py             Superseded by stage3_complete_audited.py
│   ├── update_fig9.py               Redraws Figure 9 from the cluster-level audit
│   ├── create_notebooks.py          Generates notebooks 01 and 02
│   ├── create_stage_notebooks.py    Generates notebooks 04 to 06
│   └── audit_repo_terms.py          Checks wording in the repository
├── README.md
└── requirements.txt
```

## Setup

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

## Reproducing the Stages

Run each stage from the repository root.

```bash
python src/stage1_data_eda.py
python src/stage2_baselines.py
python src/stage3_complete_audited.py
python src/stage4_shap.py
python src/stage5_inverse_design.py
python src/stage6_pareto.py
```

Do not run `src/stage3_tuning.py` over the audited tables.

To screen a composition with the saved model:

```bash
python src/stage7_screen_alloy.py --example training-strongest
python src/stage7_screen_alloy.py --example mild-steel
python src/stage7_screen_alloy.py --wt Al=0.1,C=0.03,Ni=18,Co=8,Mo=5,Ti=0.4
```

The mild-steel example is reported as outside the applicability domain. The value printed for it is what the model returns, and it is not a yield-strength estimate.

A composition is also reported as outside the domain if any element falls outside its training range. Every training alloy lists small amounts of Al, Co, Cr, Mn, Mo, Ni, and Si, so leaving one of those out of `--wt` sets it to zero and triggers the flag, as in the last example above.

## Notebooks

| Notebook | Contents |
|:---|:---|
| [`01_data_and_eda`](notebooks/01_data_and_eda.ipynb) | Composition parsing, target distribution, correlations |
| [`02_baselines_and_evaluation`](notebooks/02_baselines_and_evaluation.ipynb) | Leakage gap, baselines, applicability-domain audit |
| [`03_tuning_and_residual_audit`](notebooks/03_tuning_and_residual_audit.ipynb) | Model comparison, corrected tests, sanity checks, outliers, alloy families |
| [`04_shap_interpretation`](notebooks/04_shap_interpretation.ipynb) | SHAP tables and figures |
| [`05_inverse_design`](notebooks/05_inverse_design.ipynb) | Inverse-design audit and candidates |
| [`06_pareto_front`](notebooks/06_pareto_front.ipynb) | Pareto front and carbon-equivalent validity |

The notebooks read saved results and do not fit models.

## Reading the Results

- **Two error summaries are used.** Tables report the mean across 25 outer folds. Figures 6 and 10 annotate each panel with the error of per-alloy predictions averaged across repeats, which is slightly lower (for example, 156.3 MPa against 158.3 MPa for the default Random Forest).
- **Audited tables replace earlier ones.** Use `stage3_paired_comparisons_nadeau_bengio.csv`, `stage3_family_errors_audited.csv`, and `stage3_worst_alloys_hedged.csv`. The files with the same stem and no suffix are earlier versions kept for traceability.
- **Library versions are not pinned.** Exact values can shift slightly between environments. A Stage 4 rerun of the 14-feature default forest gave 160.3 MPa against the archived 158.3 MPa, which is well inside the fold standard deviation of about 27 MPa.
- **Tested environment.** The notebooks and the screening script were last run on Python 3.13 with scikit-learn 1.7, pandas 2.3, and optuna 5.0.

## References

1. A. Dunn, Q. Wang, A. Ganose, D. Dopp, A. Jain. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. *npj Computational Materials* 6, 138 (2020).
2. C. Nadeau, Y. Bengio. Inference for the generalization error. *Machine Learning* 52, 239–281 (2003).
3. S. M. Lundberg et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* 2, 56–67 (2020).
4. K. Aas, M. Jullum, A. Løland. Explaining individual predictions when features are dependent. *Artificial Intelligence* 298, 103502 (2021).
5. S. Floreen. The physical metallurgy of maraging steels. *Metallurgical Reviews* 13, 115–128 (1968).
6. W. Sha, A. Cerezo, G. D. W. Smith. Phase chemistry and precipitation reactions in maraging steels: Part III. Model alloys. *Metallurgical Transactions A* 24, 1241–1249 (1993).
7. K. Deb, A. Pratap, S. Agarwal, T. Meyarivan. A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation* 6, 182–197 (2002).
8. European Commission. *Study on the Critical Raw Materials for the EU 2023*.
