# PROJECT_STATE.md

## 1. Project Goal and Dataset
- **Objective:** Machine-learning prediction of yield strength from chemical composition in steels, interpretability via SHAP, and inverse alloy design with uncertainty quantification.
- **Dataset:** `matbench_steels` (312 steel alloys, chemical formula $\to$ yield strength in MPa).
- **Core Constraint:** Dataset records **elemental composition only**. Zero processing variables (austenitizing, cooling rates, tempering/aging times and temperatures) are recorded. Residual scatter ($\sim 150\text{ MPa}$) reflects omitted processing history and model error.

---

## 2. Stage Status & Workspace Artifacts
- **Stage 0 (Setup & Hygiene):** COMPLETE. Environment, git, directory layout (`src/`, `results/`, `figures/`, `notebooks/`), `NOTES.md`, `LOG.md`.
- **Stage 1 (EDA & Formulation):** COMPLETE. Composition parsing (wt%), summary statistics, sparsity, element correlations, formula validation.
  - Files: `src/stage1_data_eda.py`, `notebooks/01_data_and_eda.ipynb`, `figures/fig01_target_distribution.png` to `fig06_composition_heatmap.png`.
- **Stage 2 (Baselines & Evaluation):** COMPLETE. Connected-components clustering ($\tau = 0.50\text{ wt}\%$, 201 clusters), 5x5 Repeated Grouped CV ($S=25$ splits), Naive vs Grouped CV leakage gap, Dummy/Ridge/RF/XGBoost baselines, learning curve, applicability domain audit.
  - Files: `src/stage2_baselines.py`, `src/stage2_addendum_audit.py`, `src/applicability_domain.py`, `notebooks/02_baselines_and_evaluation.ipynb`, `results/stage2_*.csv`, `figures/fig07_*.png` to `fig09_*.png`.
- **Stage 3 (Tuning, Neural/GP Comparators, Outlier Audit):** COMPLETE (Audited).
  - Nested Optuna tuning per outer training fold (25 splits $\times$ 15 trials per model), fixed ARD Matérn GP, Leave-One-Cluster-Out (LOCO) extrapolation, Nadeau-Bengio corrected paired tests, hedged family and outlier audit, 20-permutation null target test.
  - Files: `src/stage3_complete_audited.py`, `src/audit_repo_terms.py`, `notebooks/03_tuning_and_residual_audit.ipynb`, `results/stage3_*.csv`, `figures/fig10_tuned_models_parity.png` to `fig13_family_error_comparison.png`.
- **Stage 4 (SHAP Interpretation):** NOT STARTED (Awaiting explicit user approval).

---

## 3. Key Methodological Decisions & Rationales
1. **Grouped CV via Connected-Components Clustering ($\tau = 0.50\text{ wt}\%$):** 312 alloys contain tight near-twin compositional variants. Naive random splitting leaks near-duplicates across train/test folds, inflating $R^2$ by $\sim 0.08$. Threshold $\tau = 0.50\text{ wt}\%$ isolates 201 distinct clusters.
2. **5x5 Repeated Grouped CV ($J=25$ Outer Splits):** Headline evaluation scheme. Mitigates high fold-to-fold variance caused by discrete compositional clustering.
3. **Naive CV Retained Only to Quantify Leakage Gap:** Shows the $15.5\text{ MPa}$ RMSE optimism of un-grouped evaluation.
4. **Compositional Closure & Dropping Fe (13 Non-Fe Features):** $\sum x_i = 100\text{ wt}\%$. Fe is redundant and collinear. Dropping Fe stabilizes Ridge regression ($\Delta\text{RMSE} = 6.65\text{ MPa}$ gain) and prevents ill-conditioned covariance matrices in Gaussian Processes.
5. **Nested Optuna Tuning Inside Each Outer Fold:** Prevents hyperparameter optimization leakage. Hyperparameters selected on inner 3-fold Grouped CV using training folds only.
6. **Nadeau-Bengio Corrected Resampled Paired Tests:** Resampled CV splits share $\sim 80\%$ training instances ($n_{\text{train}}=249, n_{\text{test}}=63$). Plain $t$-tests severely underestimate variance. Variance is inflated by $\frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}} \approx 0.293$ ($df=24$).
7. **Applicability Domain Status as EXPLORATORY:** Formulated post-hoc after raw metrics failed. Binned near vs far MAE difference shows separation ($\Delta = +66.1\text{ MPa}$), but continuous rank correlation is weak ($\rho = +0.086$, $p=0.225$). Must be confirmed on independent data.

---

## 4. Key Corrected Results Table (5x5 Repeated Grouped CV, 25 Splits)

| Model | Hyperparameter Strategy | Validation RMSE (MPa) | Validation MAE (MPa) | Validation $R^2$ | Paired $\Delta$ vs Tuned RF | Nadeau-Bengio Corrected 95% CI | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Default Random Forest** | Baseline ($N=100, d=10, s=4$) | **$158.30 \pm 26.33$** | **$118.58 \pm 22.35$** | **$0.684 \pm 0.119$** | $-1.27\text{ MPa}$ | $[-13.86, +11.33]\text{ MPa}$ | Not distinguishable |
| **Tuned Random Forest** | Nested Optuna (15 trials/split) | $159.57 \pm 28.49$ | $119.67 \pm 22.77$ | $0.686 \pm 0.104$ | Benchmark ($0.0$) | Benchmark | Benchmark |
| **Tuned XGBoost** | Nested Optuna (15 trials/split) | $160.86 \pm 31.78$ | $120.85 \pm 25.30$ | $0.676 \pm 0.126$ | $+1.29\text{ MPa}$ | $[-15.53, +18.10]\text{ MPa}$ | Not distinguishable |
| **Default XGBoost** | Baseline ($N=100, d=4, \eta=0.08$) | $162.59 \pm 30.62$ | $122.91 \pm 23.98$ | $0.669 \pm 0.124$ | $+3.02\text{ MPa}$ | $[-11.45, +17.49]\text{ MPa}$ | Not distinguishable |
| **PyTorch MLP** | 2-Layer (64-32, BatchNorm, Dropout) | $173.85 \pm 39.76$ | $123.11 \pm 27.46$ | $0.619 \pm 0.168$ | $+14.28\text{ MPa}$ | $[-22.39, +50.94]\text{ MPa}$ | Suggestive |
| **Gaussian Process** | ARD Matérn ($\nu=2.5$, 13 scales, non-Fe) | $183.04 \pm 43.77$ | $124.06 \pm 24.87$ | $0.590 \pm 0.149$ | $+23.47\text{ MPa}$ | $[-15.61, +62.56]\text{ MPa}$ | Suggestive |

### Core Empirical Conclusions:
- **No model difference survives the corrected test:** RF, XGBoost, MLP, and GP differences across folds are suggestive or indistinguishable.
- **Tuning gain is not distinguishable:** Default RF performs nominally as well as nested tuned RF ($\Delta = -1.27\text{ MPa}$, $p_{\text{corr}} = 0.837$).
- **Leakage gap is substantial:** Naive CV $R^2 \approx 0.764$ vs Grouped CV $R^2 \approx 0.684$ ($\Delta R^2 \approx 0.080$, $\Delta\text{RMSE} = 15.5\text{ MPa}$).

---

## 5. Known Limitations & Open Items
1. **Family-Error Number Shift Between Reports:**
   - *Explanation:* Initial report computed family errors from a single global-tuned model ($154.04\text{ MPa}$ benchmark). Corrected report computed family errors from out-of-fold predictions pooled from 25 nested-tuned models ($159.57\text{ MPa}$ benchmark). Out-of-fold predictions shifted slightly across folds.
2. **LOCO Criterion Benchmark Value:**
   - *Status:* Stale benchmark ($154.04\text{ MPa}$) set threshold at $1.5 \times 154.04 = 231.06\text{ MPa}$. Under the corrected nested tuned RF ($159.57\text{ MPa}$), threshold is $1.5 \times 159.57 = \mathbf{239.36\text{ MPa}}$. All 5 top clusters ($67.9, 65.7, 39.2, 51.5, 143.5\text{ MPa}$) comfortably pass under both thresholds.
3. **Pooled Applicability-Domain Number Discrepancy ($N=312$ vs $N=7800$):**
   - *Status:* Initial report ran single 5-fold CV ($N=312$ static distance calculations). Subsequent table pooled 25 outer folds ($N=7800$ split-dependent distance calculations where training neighbor set changes per repeat). If any implementation differences between scripts contributed, this remains partially unexplained and is noted as such.
4. **Tool Steels Sample Size:** $n=17$ is underpowered for statistical family conclusions.

---

## 6. Standing Rules (Verbatim)
1. Never fabricate results.
2. Never write dates or hours.
3. Hypothesis wording for mechanisms ("consistent with", never "proves").
4. Never state commercial grade names or processing history that is not in the data.
5. Investigate suspicious results in both directions.
6. Explain everything in NOTES.md for a viva.
7. Work stage by stage and stop for approval after each stage.
8. End-of-stage report format = what was done, key numbers, figures, caveats, unverifiable claims, next step.

---

## 7. Roadmap: Stages 4 to 7 and Extension Phase

### Stage 4: SHAP Interpretation (Awaiting Approval)
- **Models & Features:** Tuned RF and Tuned XGBoost. Train 13-feature (no-Fe) versions to verify accuracy is maintained and interpret effects as "replacing Fe".
- **Global Importance:** `shap.TreeExplainer` (beeswarm, mean |SHAP| bar chart, dependence plots for Ti, Ni, Co, Cr, Mo, C, Al, Si). Validate against Grouped CV Permutation Importance.
- **Stability:** Compute SHAP across 25 outer training folds; report rank correlation and spread of mean |SHAP|.
- **Family Confounding:** Dependence plots colored by family; within-family SHAP summaries (flag $n<30$).
- **Interactions:** TreeExplainer interaction values for top pairs (Ni-Ti, Co-Mo, Cr-Ni, C-Cr); compare with 2-way partial dependence.
- **Metallurgical Sanity Table:** Row per major element comparing literature metallurgical expectations with model SHAP; classify as "consistent", "inconsistent", or "unclear".
- **Limitations:** Record in `NOTES.md` that SHAP explains the model, not physical causality; correlation splits credit; closure constraints.
- **Outputs:** Results CSVs, publication figures, viva Q&A in `NOTES.md`, end-of-stage report, STOP for approval.

### Stage 5: Inverse Alloy Design via Bayesian Optimization
- Multi-objective BO (target yield strength $\ge 1500\text{ MPa}$, minimize raw material cost / critical elements).
- Uncertainty quantification via GP predictive standard deviation or tree ensemble variance.
- Applicability-domain constraint enforcing proposals within reliable training support.

### Stage 6: Multi-Objective & Pareto Optimization
- Pareto front exploration (strength vs alloy cost vs weldability indicators).
- Verification of candidate chemistries against metallurgical feasibility constraints.

### Stage 7: Synthesis, Dashboard & Viva Defense
- Synthesis report, interactive demonstration dashboard (Streamlit/script), final mock viva Q&A deck.

### Extension Phase: Supervisor-Paper Dataset
- Evaluate trained models and applicability-domain heuristic on independent extension dataset from supervisor's publication to test external generalization.
