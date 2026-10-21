# Work plan for the rest of the BTP

The scripts for Stages 4–7 have been run. The tables, figures, and viva answers are in `results/`, `figures/fig14`–`fig25`, and `NOTES.md`. The six blocks below are the reading, the report, and the defense. The extension is still blocked on the supervisor’s file.

Machine-learning prediction of yield strength from steel composition, with SHAP and inverse design.
Department of Metallurgical and Materials Engineering, IIT Roorkee.

This is the working plan for the unfinished part of the project. Stages 0–3 are already in the repository. Stages 4–7 are the scripts in `src/`. The extension needs the supervisor’s dataset and cannot be invented from another public file.

The standing rules still hold. Do not invent a number. Do not write dates or hours into `LOG.md`. Say “consistent with”, not “proves”. Do not name a commercial grade or a heat treatment that is not in the data. One suspicious result gets checked in both directions. After each stage, the notes need a viva answer. The end-of-stage report is: what was done, the numbers, the figures, the caveats, anything that cannot be verified, and the next step.

## What the computation is actually claiming

The dataset is Matbench `matbench_steels`: 312 compositions and a measured yield strength in MPa. Dunn et al., npj Computational Materials, 2020, took it from Citrine’s “Mechanical properties of some steels” (dataset 153092) and deduplicated it. The official task is composition only. There is no processing column.

The official leaderboard is a different test from ours. The task page, checked while this plan was written, listed TPOT-Mat at 79.9 MPa MAE and MODNet v0.1.12 at 87.8 MPa MAE, under Matbench’s nested cross-validation on a fixed row order. A plain random forest on composition (RF-Regex Steels) sits near 90.6 MPa MAE and 128 MPa RMSE on that protocol. Our headline number is harsher on purpose: grouped cross-validation at τ = 0.50 wt% gives a default random forest RMSE of 158.3 MPa and MAE of 118.6 MPa. The gap versus the leaderboard is mostly the split, not a broken script. Naive splitting on our own data drops the RMSE to 142.8 MPa. Do not put the leaderboard MAE next to our grouped RMSE without saying the protocols differ.

The alloys are all high-strength. The measured range starts at 1005.9 MPa. A mild steel is outside the model, and the screening script is supposed to say so.

## Literature the later stages are tied to

SHAP. Lundberg et al., Nature Machine Intelligence, 2020, is the tree explainer we use. Path-dependent TreeSHAP follows the training counts in the trees. Interventional TreeSHAP breaks feature dependence with a background sample (Janzing, Minorics and Blöbaum, 2020). Path-dependent values are not implementation-invariant (Chen, Covert, Lundberg and Lee, arXiv:2302.08434). Correlated inputs can move or flip an attribution (Aas, Jullum and Løland, 2021). So Stage 4 reports both explainers, the rank correlation between them, and the rank correlation with grouped-CV permutation importance. A story that only one of those supports does not go into the viva as a result.

Closure. The fourteen weight percents sum to 100. Iron is the balance. A SHAP value on all fourteen features mixes “more nickel” with “less iron”. The interpretable models drop iron, and the attribution is the effect of replacing iron. That is the usual warning for compositional data (Aitchison; the materials-science statement is in Gloor, Wu, Pawlowsky-Glahn and Egozcue, ACS Combinatorial Science, 2015). We do not switch the whole project to log-ratios this late. The 13-feature model is the check, and if its error moves by more than the corrected paired test can tolerate, the notes have to say the check failed.

Mechanisms used as expectations, not as labels we force onto the plot. Floreen, Metallurgical Reviews, 1968, and Floreen and Speich, 1964: titanium is the strongest hardener among the substitutional additions they compared, and cobalt amplifies molybdenum rather than hardening much by itself. Sha, Cerezo and Smith, Metallurgical Transactions A, 1993: titanium precipitates early as Ni3Ti, cobalt’s clearest role is lowering the solubility of molybdenum, and the molybdenum-rich phase in that work was Fe7Mo6. Chromium is handled as a confound. In this file a high chromium content is the stainless group, which sits lower in strength than the maraging group. A global negative chromium attribution is allowed to be that mix. It is not allowed to become “chromium weakens steel”.

Inverse design. Vela, Khatamsaz, Acemi, Karaman and Arróyave, Digital Discovery, 2026, review metallurgy-informed Bayesian optimization and argue for a feasibility filter before a wide search. Our feasibility filter is the applicability domain already coded in `src/applicability_domain.py`: inside the training min/max box, and within the 95th percentile of training nearest-neighbor distances in standardized non-iron space. The search does not wander the whole box. It starts at a training alloy and takes a short step, because a point outside that neighborhood is rejected anyway. The acquisition is a lower confidence bound: predicted yield minus one tree-standard-deviation. Tree disagreement is not a calibrated error bar. The Gaussian process from Stage 3 was a bit tight (62.8% of truths inside ±1σ, against 68.3% for a normal distribution), so its standard deviation is a second opinion, not a probability of passing a tensile test.

Criticality, not a made-up price. Weights are supply-risk numbers from the European Commission’s 2023 critical-raw-materials study. Cobalt 2.8, niobium 4.4, vanadium 2.3, tungsten 1.2, manganese 1.2, molybdenum 0.8, chromium 0.7, nickel 0.5, aluminium/bauxite 1.2, silicon metal 1.3, titanium metal 1.6. Iron, carbon, and nitrogen are not given a weight. Silicon metal is the nearest published row to alloying silicon, and titanium metal is used instead of the mineral row because the addition in the steel is metal. The index is wt% times supply risk. It is not euros, and it is not a quote.

Weldability. The IIW carbon equivalent is C + Mn/6 + (Cr+Mo+V)/5 + Ni/15. Pcm is the low-carbon companion formula. Both are indicators of hydrogen-cracking tendency in ferritic structural steels. The CET validity window restated with EN 1011-2 method B (Dillinger’s statement of it) stops around Ni 2.5 wt%, Mo 0.75, Cr 1.5, Ti 0.12. This dataset lives above those limits. Stage 6 therefore counts how many training alloys fall outside the window, and it does not optimize carbon equivalent. For these chemistries a high carbon equivalent is mostly a restatement of nickel and chromium, not a weldability measurement. Maraging steels are low-carbon, and the formula was not built for them.

Comparison point for inverse design, not a template to copy. Jung, Kang, Kim, Lee, Choi and Oh, Scientific Reports, 2021, did forward-and-inverse design on TMCP steels with NSGA-II, but their 5473 rows include processing. We do not have processing. Copying their claim would be wrong. Our candidates are chemistries scored by a composition-only model.

## What “done” means for each remaining stage

### Stage 4. SHAP

Script: `python src/stage4_shap.py`.

Headline model, chosen before looking at attributions: the default random forest on 13 non-iron features. Path-dependent TreeSHAP on the outer test rows. Tuned forest and tuned XGBoost are there to see whether the story depends on the model. Their hyperparameters come from `results/stage3_nested_hyperparameters_per_split.csv`. On the 13-feature fits, those hyperparameters were selected on 14 features, and the notes must say that.

A sanity label is allowed to be consistent or inconsistent only when all three 13-feature models agree, the mean absolute SHAP is at least the median element, and the Spearman correlation between weight percent and SHAP clears 0.20 in the way the rule states. The rules are in `verdict_for_element` and are checked on synthetic cases at the start of the script. Tool steels (n = 17) stay in the family heatmap and stay underpowered. Nitrogen and tungsten stay sparse.

The 14-feature default forest must reproduce the Stage 3 RMSE of 158.3 MPa. If it does not, the splits have drifted and the stage stops.

Figures fig14–fig21. Tables `results/stage4_*.csv`. Then stop and write the Stage 4 section of `NOTES.md` from those tables.

### Stage 5. Inverse design

Script: `python src/stage5_inverse_design.py`.

Four hundred Optuna trials at seed 42 and four hundred at seed 43. Feasible means predicted yield at least 1500 MPa, inside the domain, iron inside its training range. 1500 MPa is inside the measured range (1006 to 2510), so this is not a request for a stronger alloy than the data contain.

Report whether the top candidates are retrievals of a training neighborhood or actual in-domain variants. The cutoff is the median training nearest-neighbor distance, declared in the applicability-domain object. Report whether seed 43 lands near seed 42. Report whether any candidate beats the best in-sample training alloy on the lower confidence bound. Beating it would be surprising, and it would need the same suspicion as any other too-good result. The training scores are in-sample.

Also save the Gaussian-process length scales, per fold and for the full-data fit. The old sentence in `NOTES.md` that quotes numeric length scales and mentions copper is withdrawn. Copper is not in the data, and those numbers were never written to a file.

### Stage 6. Pareto

Script: `python src/stage6_pareto.py`.

NSGA-II, 600 trials, seed 42, same local moves. Axes are the lower confidence bound and the criticality index. Carbon equivalent is computed, plotted as a validity audit, and not optimized. If the front is mostly retrievals, say so. A front that only rediscovers the strongest training alloys is still a result.

### Stage 7. Something you can defend

`python src/stage7_screen_alloy.py --example training-strongest` and `--example mild-steel`. The mild-steel example must come back outside the domain. Update `README.md` so the commands match the files that exist. Add the Stage 4–6 viva answers to `NOTES.md` from the CSVs, and add a short errata for the family MAE/R² mismatch, the stale LOCO threshold of 231 MPa, and the copper length-scale sentence. The corrected LOCO threshold is 1.5 × 159.57 = 239.36 MPa, and the five clusters still pass either bar.

### Extension

Blocked until the supervisor’s file is in `data/`. Parse it with the same pymatgen weight-percent path. Do not refit. Score the frozen model and the frozen domain rule. Report error inside the domain and outside it separately. If the column is not yield strength in MPa, stop. Do not download a substitute dataset and call it the extension.

## Six blocks of work

These are the blocks to finish the project as a defense, not a list of hours. Computation for Stages 4–6 is the scripts above. The remaining work is to be able to say the numbers without reading them off the screen for the first time in the viva.

**Block 1. Own the data and the leakage result.**
Rerun `python src/stage1_data_eda.py` and `python src/stage2_baselines.py` once, so the figures are yours. Be able to answer why every yield strength is above 1000 MPa, why tungsten stayed, why the formula strings are atomic fractions while the model uses weight percent, and why naive R² 0.76 falls to grouped R² 0.68. Questions 1–7 in `NOTES.md` are the script for this block. The applicability-domain gap (near MAE 85.5, far MAE 151.6) is exploratory. The rank correlation is not significant (ρ = 0.086, p = 0.225). Say both.

**Block 2. Own the model comparison.**
Walk `results/stage3_paired_comparisons_nadeau_bengio.csv` and the family file `results/stage3_family_errors_audited.csv`. The prose MAE and R² in the Stage 3 section of `NOTES.md` do not match that CSV. The RMSE values do. Use the CSV. Be able to explain the Nadeau–Bengio factor 0.293 in one minute: the 25 test folds share training alloys, so an ordinary t-test is too sure of itself. Tuning did not beat the default forest. The neural net and the Gaussian process are only suggestively worse. Leave-one-cluster-out on the five largest clusters passed. The shuffled-target test failed in the way a clean pipeline should fail.

**Block 3. SHAP, slowly.**
Read the Stage 4 CSVs next to fig14–fig21. For each of Ti, Ni, Co, Mo, Cr, Al, write one sentence from `stage4_metallurgical_sanity.csv` and no sentence that the consensus column does not support. Check the additivity printout. Check that permutation importance and SHAP ranks move together on fig21. If they do not, the viva answer is the disagreement, not a smoothed ranking. Write the Stage 4 notes only after that reading.

**Block 4. Inverse design and the front.**
Read `stage5_search_audit.csv`, `stage5_candidates.csv`, and `stage6_pareto_front.csv` before describing a “designed alloy”. If the domain label says retrieval, call it a retrieval. Practice the sentence that 1500 MPa is inside the data, that the score is in-sample, and that no heat treatment was proposed. Practice the carbon-equivalent answer: the formula is real, and it is the wrong tool for this file, which is why it was not an objective. Fig25 is that answer.

**Block 5. Write the report from the notes.**
One chapter of data and leakage, one of models, one of interpretation, one of inverse design, one of what the model is not allowed to say. Every number comes from a CSV. The errata in the notes gets folded in, so the report does not repeat the copper sentence or the stale 231 MPa threshold. Update the README commands in the same pass.

**Block 6. Viva, and the extension if the file arrives.**
Close the laptop and answer questions 1–12 plus the new ones from Stages 4–6. Run the screening script on the strongest training alloy and on the mild steel while someone watches. If the supervisor file is here, do the extension as written above and add one short section. If it is not here, the defense says the external test is still open and the domain rule is still exploratory.

## Files to trust when two documents disagree

1. `src/stage3_complete_audited.py` and the Stage 3 CSVs, for anything already computed.
2. `src/stage4_shap.py`, `src/stage5_inverse_design.py`, `src/stage6_pareto.py`, and their CSVs, for the new work.
3. `NOTES.md` only after it has been checked against those CSVs.
4. `README.md` only after its commands exist on disk.
5. `src/stage3_tuning.py` is the earlier script. Do not rerun it over the audited tables.

## Sources

- Dunn, Wang, Ganose, Dopp and Jain, npj Computational Materials 6, 138 (2020). Matbench, including `matbench_steels`. https://doi.org/10.1038/s41524-020-00406-3
- Matbench steels leaderboard. https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_steels/
- Lundberg et al., Nature Machine Intelligence 2, 56–67 (2020). TreeSHAP. https://doi.org/10.1038/s42256-019-0138-9
- Janzing, Minorics and Blöbaum, AISTATS 2020. Interventional Shapley values.
- Aas, Jullum and Løland, Artificial Intelligence 298, 103502 (2021). Dependence and SHAP.
- Gloor et al., ACS Combinatorial Science 17, 130–136 (2015). Compositional data in materials science.
- Floreen, Metallurgical Reviews 13, 115–128 (1968). Physical metallurgy of maraging steels.
- Sha, Cerezo and Smith, Metallurgical Transactions A 24, 1241–1249 (1993). Precipitation in maraging steels.
- Vela et al., Digital Discovery 5, 3474 (2026). Metallurgy-informed Bayesian optimization. https://doi.org/10.1039/D5DD00370A
- European Commission, Study on the Critical Raw Materials for the EU 2023. Supply-risk weights.
- EN 1011-2 and the CET window as stated by Dillinger’s welding notes. Carbon equivalent as an indicator with a validity range.
- Jung et al., Scientific Reports 11, 11012 (2021). A processing-aware inverse-design comparison, not a method we can copy.
