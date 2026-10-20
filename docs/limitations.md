# Limitations

The framework enforces a study design. It does not make a weak design scientific, and it does not decide that a block of predictors is useful.

**EPV is a screen, not a proof of power.** Clearing a floor means the locked width is not obviously too large for the observed events. It does not guarantee a detectable incremental effect, and missing the floor does not prove the extra block is useless. `UNDERPOWERED` means the comparison was not accepted.

**The gate is triage.** `GREEN`, `YELLOW`, and `RED` apply thresholds stored on the config. They are not a validated clinical rule, a utility analysis, or a license to deploy a score.

**One estimator.** The built-in fit is L2-penalized logistic regression with a fixed penalty. There is no search over that penalty after the lock. A different estimator would be a different pre-registration, not a switch inside `run`.

**Internal splits only.** The holdout and the repeated cross-validation are resamples of the same analytic table. They are not an external population or a later time period.

**Unweighted rows.** The engine does not apply survey weights or a cluster design. If the rows are a complex sample, the metrics are not population estimates.

**One imputation pass inside each fold.** Numeric columns use a training-fold median, categorical columns a training-fold mode, then scaling and one-hot encoding. That is not multiple imputation.

**Composites are only as good as the locked rule.** Sum-score and mean-score coerce values to numbers. They do not know which codes mean "endorsed." A count of endorsed items has to be registered by the study, with the endorsement rule written down before the lock.

**No cross-study store.** Results are objects and JSON files. Nothing here compares studies, draws a dashboard, or pools datasets.

**Linkage is structural.** The engine can require a unique key and can describe partial overlap. It cannot decide that two source files are the right files.

**Confirmation is a flag, not a literature review.** `confirmed=True` records that the author is willing to lock the variable. The framework does not check a codebook.

**The pre-registration lock is self-attested.** `lock_preregistration()` can only be called before `run()` reaches the modeling stage, and a changed config must be re-locked with `--amend`, which moves the prior lock into `locks/history/` under its original timestamp rather than overwriting it. That preserves a record of every lock the author made. It does not prove that the first lock was written before the author inspected the data: nothing stops someone from looking at results, editing the config, and amending the lock to match what they saw. The engine enforces order within one run; it has no way to verify that a human did not peek between runs. The amendment history turns a silent rewrite into a visible, timestamped one, which is weaker than true pre-registration but stronger than an unaudited config file. Treating the first committed lock, with its commit timestamp as external witness, as the actual pre-registration is the intended use.

This repository ships the framework and a synthetic toy study. It does not fit a model to any real cohort.
