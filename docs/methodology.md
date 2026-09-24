# Methodology

This framework encodes a predictive feasibility discipline. It estimates whether a pre-registered full model adds discrimination beyond a pre-registered baseline, under a cutoff that keeps later information out of the predictors, and it refuses to treat that comparison as informative when the design is too wide for the number of events.

It is not a clinical tool, a causal model, or a deployment scorer.

## Cutoff

Each predictor carries an availability date, or is explicitly time-invariant. A column may enter the design matrix only when it is confirmed and its information is available on or before the cutoff. The outcome date must be strictly after the cutoff. Unknown timing fails closed. The engine runs this check; skipping ahead to modeling is not a supported call.

## Confirmation

Predictor blocks and individual predictors each carry `confirmed: bool`. An unconfirmed block or variable is a hypothesis, not a specification, and validation rejects it. The framework does not discover blocks by scanning a file.

## Events per variable, before modeling

EPV is the number of outcome events in the repeated cross-validation training folds, divided by the number of columns in the full model's design matrix after imputation and one-hot encoding. The default, locked basis is the **minimum** training-fold event count (`epv_basis="min_fold_train_events"`). The mean fold count is still recorded for inspection. The alternative locked basis `mean_fold_train_events` uses the mean instead. The floor is a field on the locked config. The diagnostic fits the preprocessor only.

One-hot encoding uses `OneHotEncoder(handle_unknown="ignore")` with **no category drop**. Every observed training level becomes one design column, including the first level. A single categorical predictor with five observed levels therefore contributes five design columns, not four.

If EPV is below the floor, the run stops and returns `UNDERPOWERED`. No coefficient, holdout score, or contrast p-value is produced for that specification. Changing the encoding (for example collapsing a high-cardinality block with a registered composite) is an amendment: lock again, run again, and chain the new result onto the old one with `supersedes`. The first result stays in the record as superseded.

## Comparison and inference

The locked config names a baseline block set and a full block set. The baseline must be a strict subset of the full set. The estimator is L2-penalized logistic regression with a fixed `C` (default 1.0). A stratified holdout is confirmation only. Repeated stratified cross-validation on the training portion supplies the paired fold differences.

The primary contrast is full minus baseline on ROC-AUC, PR-AUC, and Brier score. It is not multiplicity-adjusted. Leave-one-block-out contrasts on the full model form a second family. Within that family, and separately per metric, p-values are Holm-adjusted.

The variance correction is the Nadeau–Bengio corrected resampled t-test: the variance of the mean difference is inflated by `(1/n + n_test/n_train)` times the unbiased sample variance of the fold differences. The engine calls whichever inference method the config names. The built-in method is this test plus Holm.

## Decision gate

The gate reads floors from the same locked config: usable N, event count, prevalence, baseline ROC-AUC, and incremental ROC-AUC and PR-AUC lifts. `RED` means the sample or the readiness checks failed. `YELLOW` means a model was fit but the baseline or the incremental lift missed its floor, or the incremental contrast could not be scored. `GREEN` means those checks passed. `UNDERPOWERED` is reserved for an EPV failure and is returned without a model fit. None of these colors is a clinical decision rule.

## Worked example, in summary

A Medical Expenditure Panel Survey (MEPS) panel study asked whether a social-needs block added discrimination for a later-year acute-care outcome beyond demographic, health, access, socioeconomic, and prior-utilization blocks. The cutoff was the end of the year before the outcome year. The social-needs block was first entered as a large set of categorical items. After one-hot encoding, that specification was far below a pre-registered EPV floor of 10 (on the order of 1.5 events per design column). An amendment replaced the raw items with a handful of domain scores. EPV improved and was still below 10, including for the model that omitted the social-needs block entirely. That was a limit on events in the cohort, not a finding that the block had no signal. An earlier yellow gate on the raw specification was superseded: it reflected overparameterization, and the incremental question stayed unanswered. This repository does not contain that dataset, those column names, or those fitted scores.
