# Credit Card Fraud Detection

Case study submission. Builds and evaluates a model that classifies credit card transactions as
fraudulent or benign on a 594,643 row transaction log (BankSim-style synthetic data, 1.21% fraud).

## Result

Tuned XGBoost, evaluated on a held-out temporal window (days 6 to 7, 133,231 transactions):

| Metric | Value |
|---|---|
| PR-AUC (primary) | 0.9352 |
| ROC-AUC | 0.9989 |
| Fraud recall at operating threshold 0.71 | 96.3% |
| Alert precision | 60.4% |
| Transactions flagged for review | 1.72% |
| Fraud value prevented | 99.5% of exposure in window |

## Repository contents

```
fraud_detection.ipynb            main notebook, all outputs and figures rendered
outputs/metrics.json             final metrics, best hyperparameters, business figures
outputs/model_comparison.csv     side by side model scores
outputs/fraud_model.joblib       trained model, feature list, aggregates, threshold
outputs/figures/                 all nine charts as PNG
requirements.txt
```

## Reproducing

```bash
pip install -r requirements.txt
# place fraud.csv in the repository root
jupyter notebook fraud_detection.ipynb
```

To run in Google Colab, upload `fraud.csv` and the notebook, then run the install cell at the top
before executing the rest.

A full run takes about 12 minutes on a single CPU core, most of it in the hyperparameter search.
Lower `N_SEARCH_ITER` in the configuration cell for a faster pass. No GPU is required.

Gradient boosting is not bit-for-bit deterministic across thread counts, so the cost-optimal
threshold can land a few hundredths either side of 0.71. The net value curve is flat across 0.4 to
0.8, so this does not change any conclusion.

## Method notes

**Metric.** The dataset is 1.21% fraud, so accuracy is meaningless: a model predicting "never fraud"
scores 98.8% and catches nothing. PR-AUC is the primary metric, with recall at a cost-selected
operating threshold as the decision metric. ROC-AUC is reported but is optimistically biased under
this level of imbalance; it separated the five candidate models across a range of 0.004 while PR-AUC
separated them across 0.06.

**Temporal split.** `step` is a clock (1 step = 1 hour). Training covers days 0 to 5, testing covers
days 6 to 7. A random split lets the model learn from a customer's future transactions; the notebook
measures that optimism at +0.011 PR-AUC (0.9339 random vs 0.9226 temporal) and reports the honest
number throughout. Hyperparameter search validates on the last day of the training window via
`PredefinedSplit` rather than shuffled k-fold, for the same reason.

**Dropped columns.** `zipcodeOri` and `zipMerchant` each hold a single constant value across all
594k rows and carry no information. A raw day index was also rejected as a feature, since days 6 and
7 never appear in training.

**Leakage control.** Customer and merchant behavioural aggregates (mean amount, spend volatility,
transaction count, and the current amount relative to those baselines) are computed from the
training window only, then mapped onto the test set, with a global-mean fallback for unseen entities.

**Imbalance.** Four strategies were compared on identical architecture: no correction,
`scale_pos_weight`, SMOTE, and SMOTE plus majority undersampling. Cost-sensitive weighting matched
SMOTE (0.9206 vs 0.9205 PR-AUC) on a third of the training rows, so the final model uses class
weighting and no synthetic oversampling.

**Threshold.** Chosen by maximising net value: fraud amount recovered on caught cases, minus a
per-alert review cost. Review cost is currently an assumption (5 currency units) and is the one
input the business needs to supply.

**Interpretation.** Gain importance and SHAP disagree on the ranking. Gain puts the transportation
category flag first because one early split cheaply removes 85% of the rows; SHAP, which measures
contribution to individual predictions, puts merchant and customer spending baselines first. SHAP is
the more faithful account of how the model actually decides.

## Known limitations

Synthetic data, 7.5 days of history, no geography (single zip code throughout), no velocity features,
and no modelling of the chargeback delay that governs when labels actually become available in
production. Merchant identity also carries a large share of the model's weight, which degrades
against merchants with no history. The final section of the notebook covers each of these and their
implications for deployment.
