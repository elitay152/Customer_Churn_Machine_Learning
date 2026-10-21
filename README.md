# Predicting customer churn

Compare logistic regression, random forest, and gradient boosting on the
[Kaggle bank churn dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling).

## Setup and data

Use Python 3.11. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In the prepared cloud environment, activate
`/workspace/.venvs/customer-churn` instead of creating another environment.

Download `Churn_Modelling.csv` from Kaggle. Its public API is also available:

```bash
curl --fail --location 'https://www.kaggle.com/api/v1/datasets/download/shubh0799/churn-modelling' --output /tmp/churn.zip
python -m zipfile -e /tmp/churn.zip .
```

Do not overwrite existing local data when extracting. Network access needs
`www.kaggle.com` and `storage.googleapis.com`. The download worked without
credentials during setup. Follow the provider's terms and never commit customer
CSVs or credentials. The notebook accepts either `Churn_Modelling.csv` or `Churn_Modeling.csv`.

An optional cleaned CSV can be generated with:

```bash
python prepare_data.py Churn_Modelling.csv
```

This preserves equivalent existing output and refuses to overwrite differing
output. Model pipelines read raw data and fit preprocessing inside training folds.
The cleaned CSV is an optional data export.

## Run

```bash
python -m unittest discover -s tests -v
jupyter lab
```

Open **[Customer_Churn_Analysis.ipynb](Customer_Churn_Analysis.ipynb)** and run all cells.
The analysis uses `churn_workflow.py`, a deterministic,
stratified 70/30 split, and five-fold training-only cross-validation. To execute
the consolidated notebook without replacing its saved outputs:

```bash
mkdir -p artifacts
jupyter nbconvert --to notebook --execute Customer_Churn_Analysis.ipynb --output-dir artifacts --ExecutePreprocessor.timeout=600
```

Searches use at most two workers. Imputation, scaling, and categorical encoding
are learned inside each fold. Identifiers and surnames are excluded; all other
predictors are retained. Unknown categories are accepted. Logistic regression has
an intercept and tuned regularization; tree models capture nonlinear effects.

Models are selected by mean cross-validated **average precision**, focusing on
ranking the minority churn class. CV selection scores are not independent final
estimates. The notebook reports one final test evaluation per model with probability-based
ROC AUC, average precision, log loss, Brier score, precision, recall, F1, accuracy,
balanced accuracy, and a training-prior baseline. The classification threshold is
fixed at 0.5 in advance. Weighted probabilities may need calibration before use
as risk estimates.

## Interpretation and limits

Recall measures how many churners are found; precision measures how many flagged
customers churn. Neither means overall accuracy. Compare the precision-recall
curve with churn prevalence, not accuracy alone.

Results are exploratory and require validation on an independent cohort. Before deployment,
define intervention cost and benefit, choose a threshold using training-only
out-of-fold predictions or a separate validation set, and assess the final policy
on fresh data. Prefer a forward-in-time split if customer timestamps become
available. Predicting churn does not establish that an intervention prevents it.

The regularized logistic baseline retains the available predictors. Compare
splines through training-only CV if richer logistic relationships are needed.
The small parameter grids are starting points. The three final test results are
descriptive comparisons; choosing a production winner after inspecting them
requires another independent evaluation.

## Outreach limited to the highest-risk customers

For a limited outreach budget, rank customers by score and contact the top group
rather than assuming that a score of 0.5 is the right cutoff. Compare budgets
without using the final test rows:

```bash
python threshold_analysis.py
```

This uses the **7,000 training rows only** and performs three outer
stratified folds. Within each outer training fold, a fresh five-fold search selects
parameters and fits preprocessing. Each customer's score therefore comes from a
model whose preprocessing and tuning excluded that customer. This is more costly
than evaluating a single already-selected model, but avoids using a customer's
label to help select the model that scores them.

The command writes aggregate results under `artifacts/thresholds/`:

- `capacity-comparison.csv`: contact budgets from 1% to 50%, precision, recall,
  lift versus random outreach, and counts per 1,000 customers.
- `outreach-capacity.png`: precision and churner coverage versus contact budget.
- `threshold-comparison.csv` and `threshold-tradeoffs.png`: corresponding
  score-threshold tradeoffs for context.

The console highlights illustrative 5%, 10%, and 20% budgets. No budget or final
model is selected automatically. Precision is the observed churn rate among
contacted customers; recall is the share of all churners reached. Neither is a
retention success rate. Lift compares the contacted group's churn rate to the
overall churn prevalence, not to the effectiveness of an intervention.

Capacity is exactly `floor(contact_fraction * population)`. Tied scores use stable
input order, independent of outcome labels. The minimum selected score is
diagnostic: using it as a threshold could exceed the budget when scores tie.
Use rank plus a stable, outcome-independent tie rule operationally. Scores from
separate outer models are pooled for these exploratory curves; recalibrate and
recheck the chosen policy with the final fitted model on fresh data. A fixed
score cutoff can drift with the population, while ranking enforces capacity.

Choosing a budget after inspecting these curves is policy development, not an
independent estimate of the chosen policy's future performance. The held-out test
set is excluded from this analysis. Before deployment, specify outreach
capacity and evaluate the frozen policy on an independent cohort. To estimate
financial value, also measure outreach cost and the causal retention benefit.

On the current 7,000-row training partition, nested-CV gradient boosting produced
these illustrative budget tradeoffs (rounded):

| Contact budget | Precision among contacted customers | Share of churners identified |
| --- | --- | --- |
| Top 5% | 93.7% | 23.0% |
| Top 10% | 82.1% | 40.3% |
| Top 20% | 62.0% | 60.9% |

At the 10% budget, random forest precision was 77.4% and logistic regression
precision was 53.6%. These are development comparisons, not a final policy
recommendation or measured retention benefit. Each outer model was trained on
two-thirds of the training partition, so this is not an evaluation of the final
7,000-row fitted model.

### Selected demonstration scenario

For this public-data portfolio project, assume a fictional bank can contact
**10% of its customers per campaign**, ranking them with gradient boosting.
For a population of 10,000 customers, that means 1,000 contacts. This is an
illustrative capacity assumption chosen after exploring the tradeoffs, not an
observed business requirement or an optimized financial decision. It provides
a middle scenario between the narrower 5% and broader 20% campaigns.

In the actual nested-CV training analysis, this policy selected **700 of 7,000
customers**: 575 observed churners and 125 nonchurners. It identified 40.3% of
the 1,426 churners, with 82.1% precision and approximately 4.0 times the churn
rate of random outreach. These are descriptive development results. Identifying
575 churners does not mean retaining 575 customers.

Select the top `floor(0.10 * population)` scores, using stable input order to
break ties. Do not substitute a fixed 0.5 score cutoff. Outreach cost, treatment
success, and revenue are left unspecified: the dataset does not measure them,
so this demonstration makes no monetary savings or return-on-investment claim.

![Project illustration](finance-4858797_1280.jpg)
