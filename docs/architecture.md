# Architecture

The framework calls user code. A study is a subclass of `FeasibilityStudy` with two required hooks: `define_config` and `load_data`. The engine method `run` is the only path that moves a study through ingest, linkage, the leakage guard, feature construction, the events-per-variable (EPV) gate, modeling, inference, and the decision gate.

`lock_preregistration` is deliberately not inside `run`. It writes a dated JSON lock containing a SHA-256 hash of the canonical config. `run` refuses to start when that file is missing, and it refuses to start when `define_config` no longer hashes to the locked payload. Amending a floor, a block, or a composite strategy is a new conscious lock, not a silent edit.

## Stage order

Stages are an enum. A table lists the only legal next stage from each current stage:

| From | Allowed next |
|---|---|
| init | locked |
| locked | ingested |
| ingested | linked |
| linked | leakage_checked |
| leakage_checked | features_built |
| features_built | epv_checked |
| epv_checked | modeled, or underpowered |
| modeled | inferred |
| inferred | decided |
| underpowered, decided | (none) |

`_enter_modeling` asks for the `modeled` stage. From `init` or `locked` that transition is not in the table, so the call raises `StageOrderError`. There is no public method that fits a model. `fit_comparison` also requires an `EpvClearance` object. That object can be constructed only by `issue_clearance`, and only when the dimensionality report says the pre-registered floor was met. A failing report cannot be turned into a clearance.

If the floor is missed, `run` moves to `underpowered` and returns. The modeling function is not called.

## What the engine checks itself

- Every returned frame is a data frame, the configured sources match, and the identifier column exists.
- Linkage rejects a missing key or a duplicate identifier. Partial overlap is reported and kept.
- The leakage guard rejects the outcome, the identifier, any column with no specification, any unconfirmed predictor, unknown timing, and any availability date after the cutoff. Composite columns are accepted only by re-checking their source predictors.
- EPV is mean repeated-CV training events divided by the full model's design-matrix width. The preprocessor is fit on the outer training portion. The classifier is not fit for this count.
- Inference is `self.inference_method.compare(...)`. The method name comes from the locked config and is resolved through the registry.

## Registries

Readers, composite strategies, and inference methods are dictionaries populated by decorators at import time. CSV and parquet readers, sum-score and mean-score composites, and the Nadeau–Bengio plus Holm method ship in the package. Registration is in-process. This version does not load plugins from other installed distributions.

## Results

`FeasibilityResult` is a typed object. The gate is one of `GREEN`, `YELLOW`, `RED`, or `UNDERPOWERED`. `result_b.supersedes(result_a)` sets `result_a.superseded_by` to `result_b.result_id`. A later run after an amended lock can point at the run it replaces.
