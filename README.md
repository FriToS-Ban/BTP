# Machine-Learning Prediction of Mechanical Properties of Steels from Composition, with Inverse Alloy Design

**B.Tech Project (BTP)**  
**Department of Metallurgical and Materials Engineering, Indian Institute of Technology Roorkee (IIT Roorkee)**  
**Date:** 

---

## 1. Project Overview
Mechanical properties of structural steels—most notably **yield strength ($\sigma_y$)**—are critically governed by chemical composition and subsequent thermo-mechanical processing. In industrial alloy development, discovering novel compositions that meet stringent mechanical targets requires time-consuming trial-and-error melting, casting, and mechanical testing.

This project implements an end-to-end, reproducible Materials Informatics pipeline:
1. **Forward Property Prediction:** Predict yield strength (MPa) directly from elemental composition using tree-based machine learning models (Random Forest, XGBoost) and linear baselines.
2. **Model Interpretability via SHAP:** Deconstruct the model's non-linear decision surface using Shapley Additive exPlanations (`SHAP`) to evaluate elemental contributions against classic physical metallurgy mechanisms (solid solution strengthening, grain boundary pinning, carbide precipitation, austenite stabilization).
3. **Uncertainty Quantification (UQ):** Quantify epistemic prediction uncertainty using an ensemble of bootstrapped models.
4. **Inverse Alloy Design:** Employ Bayesian Optimization (`Optuna` with Expected Improvement / acquisition penalties) to navigate the composition simplex within empirical bounds, proposing novel candidate chemistries optimized for target yield strengths while penalizing high-uncertainty regions.

---

## 2. Dataset Information
- **Source:** Matbench benchmark suite (`matbench_steels`), originally curated by Agrawal et al. / Ward et al. (Materials Project / Citrine Informatics).
- **Sample Size:** 312 unique alloy compositions.
- **Features:** Elemental composition strings parsed into stoichiometric/atomic fractions via `pymatgen`.
- **Target:** Yield Strength ($\sigma_y$) in megapascals (MPa).
- **Crucial Domain Caveat:** The dataset contains **only chemical composition**. Processing variables (austenitizing temperature, cooling rate, quenching medium, tempering time/temperature, deformation strain) are not included. The forward model therefore learns the average effective response across standard commercial heat-treated conditions represented in the dataset.

---

## 3. Repository Structure
```
BTP/
│
├── data/                  # Raw and processed dataset files (CSV, Parquet)
├── notebooks/             # Exploratory analysis and presentation notebooks
├── src/                   # Python scripts for all stages
│   ├── stage1_data_eda.py
│   ├── stage2_baselines.py
│   ├── stage3_complete_audited.py  # Authoritative Stage 3 script
│   ├── stage4_shap.py
│   ├── stage5_inverse_design.py    # Includes the Gaussian-process uncertainty
│   ├── stage6_pareto.py
│   └── stage7_screen_alloy.py
├── WORK_PLAN.md           # Remaining defense work, in six blocks
├── figures/               # Generated high-resolution figures (.png, 300 DPI)
├── results/               # Metrics tables, candidate compositions, cross-validation logs
├── README.md              # Project documentation and reproduction guide
├── NOTES.md               # Student defense notes & metallurgical rationale per stage
├── LOG.md                 # Daily work tracking template
└── requirements.txt       # Frozen dependencies for exact reproducibility
```

---

## 4. Setup and Execution

### Step 1: Environment Setup
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Reproducing Project Stages
Each stage is self-contained and reproducible with fixed random seeds:
- **Stage 1 (Data & EDA):**
  ```bash
  python src/stage1_data_eda.py
  ```
- **Stage 2 (Baseline Modeling):**
  ```bash
  python src/stage2_baselines.py
  ```
- **Stage 3 (nested tuning and residual audit):**
  ```bash
  python src/stage3_complete_audited.py
  ```
  `src/stage3_tuning.py` is the earlier script. Do not rerun it over the audited tables.
- **Stage 4 (SHAP):**
  ```bash
  python src/stage4_shap.py
  ```
- **Stage 5 (inverse design, with uncertainty):**
  ```bash
  python src/stage5_inverse_design.py
  ```
- **Stage 6 (Pareto front):**
  ```bash
  python src/stage6_pareto.py
  ```
- **Stage 7 (screen one composition):**
  ```bash
  python src/stage7_screen_alloy.py --example mild-steel
  python src/stage7_screen_alloy.py --example training-strongest
  ```

---

## 5. Defense & Oral Examination Notes
Viva answers, the errata to the earlier prose, and the Stage 4–6 results are in [NOTES.md](NOTES.md). The six blocks of remaining defense work are in [WORK_PLAN.md](WORK_PLAN.md). The external test on a supervisor dataset is not in this repository yet.
