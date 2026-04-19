# Triagegeist: Reducing Undertriage Through a ClinicalBERT + Stacking Ensemble with BERT-SHAP Escalation

## Clinical Problem Statement

Undertriage — assigning a patient to a lower acuity class than warranted — is the most dangerous failure mode in emergency triage. In ESI-based systems, 3–12% of patients are systematically under-assigned, particularly when compensatory physiology masks true severity: near-drowning, sudden vision loss, or acute presentations with deceptively normal arrival vitals. Human inter-rater agreement on ESI assignment yields κ = 0.40–0.69 across multi-site studies, meaning a substantial proportion of triage decisions vary by clinician. This submission targets that specific gap: not general acuity prediction accuracy, but the directed reduction of undertriage without introducing excess overtriage.

## Data

Three structured files were joined on `patient_id`: triage vitals and demographics (`train.csv`), chief complaint free text (`chief_complaints.csv`), and medical history (`patient_history.csv`). The column `chief_complaint_system` — a body-system taxonomy algorithmically derived from the raw text — was excluded because it would be unavailable at real triage time and would constitute methodological leakage. Pain scores of −1 were treated as a clinical sentinel value (unassessed pain) and converted to a binary `pain_not_recorded` flag before imputation.

## Methodology

The pipeline proceeds in four strictly ordered stages to prevent data leakage:

1. **Train/validation split** (80/20, stratified) before any statistics are computed from labels or feature distributions.

2. **Imputation:** group-median imputation stratified by age group and shift, computed exclusively on training data and applied to the validation set.

3. **Preprocessing:** a single `ColumnTransformer` applies `OrdinalEncoder` to categoricals (enabling LightGBM's native categorical split handler), median + `StandardScaler` to numericals, and `ClinicalBERTEmbedder` to free-text chief complaints. ClinicalBERT (`medicalai/ClinicalBERT`, trained on 87 K+ MIMIC-III clinical notes) produces 768-dimensional `[CLS]` embeddings per patient, disk-cached by MD5 key.

4. **Ensemble:** three LightGBM classifiers (a baseline and two Optuna-tuned variants selected from 25 trials of 5-fold CV) are stacked via 5-fold out-of-fold probabilities into a logistic regression meta-learner. Temperature Scaling ($T_{\text{opt}}$, fitted on the held-out validation set) reduces calibration error. A final BERT-SHAP escalation layer computes per-patient SHAP attributions for the 768 BERT dimensions within the stacking model; if the summed BERT attribution toward a higher-acuity class exceeds `best_T_bert` (selected via OOF sweep on training data — no validation-set contact), the prediction is overridden upward.

## Results

All metrics are on the held-out 20% validation set:

| Stage | QWK | Undertriage cases | Overtriage cases |
|---|---|---|---|
| Single LightGBM + ClinicalBERT | 0.9995 | 11 | 4 |
| Stacking meta-learner | 0.9996 | 11 | 4 |
| + Temperature Scaling | 0.9996 | 11 | 4 |
| + BERT-SHAP escalation | 0.9998 | 4 | 4 |

The escalation layer applied 7 corrections, reducing undertriage by 63.6% while holding overtriage flat. Dominant predictive features were `news2_score`, `gcs_total`, `shock_index`, `pain_score`, and `mental_status_triage` — consistent with established early warning score literature. ClinicalBERT contributed most for mechanism-of-injury cases where structured vitals were misleading.

## Limitations and Reproducibility

Two limitations require honest disclosure. First, $T_{\text{opt}}$ (Temperature Scaling) is fitted on the held-out validation set — standard practice with a single degree of freedom (negligible overfitting risk). The BERT-SHAP escalation threshold (`best_T_bert`) is selected via OOF sweep on the training set only, with no validation-set contact; post-escalation QWK on the held-out validation set is therefore a genuine out-of-sample estimate. On a fully independent external test set, recalibration may still be advisable. Second, LightGBM's multi-threaded histogram builder introduces ±0.0002 QWK and ±1 undertriage/overtriage variance across runs even with `SEED = 42`; add `force_row_wise=True`, `deterministic=True` to reproduce exactly. Prospective validation on multi-site real-world data is necessary before any clinical deployment.

## Novelty and Impact

The BERT-SHAP escalation layer is a principled safety mechanism: rather than adding an independent post-hoc classifier, it interrogates the stacking model's own internal reasoning about the chief complaint and escalates only when the model's BERT evidence was overruled by structured features. This architecture provides a direct interpretability pathway for every escalation decision — a prerequisite for clinical adoption — and a template for building asymmetric safety nets into any ordinal risk classifier.
