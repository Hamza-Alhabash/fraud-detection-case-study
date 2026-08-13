# Credit Card Fraud Detection

Case study submission. Builds and evaluates a model that classifies credit card transactions as
fraudulent or benign on a 594,643 row transaction log (BankSim-style synthetic data, 1.21% fraud).

## Result

Tuned XGBoost, evaluated on a held-out temporal window (days 6 to 7, 133,231 transactions):

| Metric | Value |
|---|---|
| PR-AUC (primary) | 0.9353 |
| ROC-AUC | 0.9989 |
| Fraud recall at operating threshold 0.75 | 95.8% |
| Alert precision | 62.8% |
| Transactions flagged for review | 1.65% |
| Fraud value prevented | 99.4% of exposure in window |

## Repository contents

```
fraud_detection.ipynb            main notebook, code only
fraud_detection_executed.ipynb   same notebook with all outputs and figures rendered
outputs/metrics.json             final metrics, best hyperparameters, business figures
outputs/model_comparison.csv     side by side model scores
outputs/fraud_model.joblib       trained model, feature list, aggregates, threshold
outputs/figures/                 all charts as PNG, ready for the deck
requirements.txt
```

## Reproducing

```bash
pip install -r requirements.txt
# place fraud.csv in the repository root
jupyter notebook fraud_detection.ipynb
```

Full run takes about 12 minutes on a single CPU core, most of it in the hyperparameter search.
Set `N_SEARCH_ITER` lower in the config cell for a faster pass.

## Method notes

**Metric.** The dataset is 1.21% fraud, so accuracy is meaningless: a model predicting "never fraud"
scores 98.8% and catches nothing. PR-AUC is the primary metric, with recall at a cost-selected
operating threshold as the decision metric. ROC-AUC is reported but is optimistically biased under
this level of imbalance, and it separated the candidate models far less than PR-AUC did.

**Temporal split.** `step` is a clock (1 step = 1 hour). Training covers days 0 to 5, testing covers
days 6 to 7. A random split lets the model learn from a customer's future transactions; the notebook
measures that optimism at +0.011 PR-AUC and reports the honest number throughout. Hyperparameter
search validates on the last day of the training window via `PredefinedSplit` rather than shuffled
k-fold, for the same reason.

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

## Known limitations

Synthetic data, 7.5 days of history, no geography (single zip code throughout), no velocity features,
and no modelling of the chargeback delay that governs when labels actually become available in
production. Section 11 of the notebook covers these and their implications for deployment.
