"""Baseline versus full-model fit. Requires an EPV clearance token."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feasibility_framework.config import StudyConfig
from feasibility_framework.dimensionality import EpvClearance
from feasibility_framework.features import PreparedFeatures
from feasibility_framework.registries import ContrastSpec


class ModelingError(RuntimeError):
    """Raised when a locked model cannot be fit without inventing a result."""


@dataclass
class ModelComparison:
    holdout: dict[str, float | int | None]
    folds: pd.DataFrame
    contrasts: list[ContrastSpec]


def holdout_indices(y: pd.Series, *, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(y))
    try:
        train_idx, test_idx = train_test_split(
            positions,
            test_size=test_size,
            random_state=seed,
            stratify=np.asarray(y),
        )
    except ValueError as exc:
        raise ModelingError(
            "Stratified holdout could not be built from this outcome. "
            "Refusing to fall back to an unstratified split."
        ) from exc
    return np.sort(train_idx), np.sort(test_idx)


def _preprocessor(columns: list[str], treatment: dict[str, str]) -> ColumnTransformer:
    numeric = [name for name in columns if treatment[name] == "numeric"]
    categorical = [name for name in columns if treatment[name] == "categorical"]
    unknown = [name for name in columns if name not in treatment]
    if unknown:
        raise ModelingError(f"Columns have no treatment and fail closed: {unknown}.")
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ModelingError("No columns were available to preprocess.")
    return ColumnTransformer(transformers)


def design_column_count(
    x_train: pd.DataFrame,
    columns: list[str],
    treatment: dict[str, str],
) -> int:
    if not columns:
        return 0
    preprocessor = _preprocessor(columns, treatment)
    transformed = preprocessor.fit_transform(x_train.loc[:, columns])
    return int(transformed.shape[1])


def _pipeline(config: StudyConfig, columns: list[str], treatment: dict[str, str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(columns, treatment)),
            (
                "clf",
                LogisticRegression(
                    C=config.estimator_c,
                    max_iter=2000,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _scores(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if np.unique(y_true).size < 2:
        raise ModelingError("Evaluation fold has a single outcome class.")
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def _fit_predict(
    config: StudyConfig,
    x: pd.DataFrame,
    y: pd.Series,
    columns: list[str],
    treatment: dict[str, str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    y_train = np.asarray(y.iloc[train_idx])
    if np.unique(y_train).size < 2:
        raise ModelingError("A training fold has a single outcome class.")
    pipe = _pipeline(config, columns, treatment)
    pipe.fit(x.iloc[train_idx].loc[:, columns], y_train)
    return pipe.predict_proba(x.iloc[test_idx].loc[:, columns])[:, 1]


def _block_columns(
    config: StudyConfig,
    prepared: PreparedFeatures,
    block_names: tuple[str, ...],
) -> list[str]:
    wanted = set(block_names)
    columns: list[str] = []
    for name in prepared.x.columns:
        sources = prepared.derived_from.get(name)
        if sources:
            owners = {config.predictor_index()[source][0].name for source in sources}
            if owners <= wanted and owners:
                columns.append(name)
            continue
        owner = config.predictor_index()[name][0].name
        if owner in wanted:
            columns.append(name)
    return columns


def fit_comparison(
    clearance: EpvClearance,
    config: StudyConfig,
    prepared: PreparedFeatures,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> ModelComparison:
    if not isinstance(clearance, EpvClearance):
        raise ModelingError(
            "fit_comparison requires EPV clearance from the dimensionality gate."
        )
    del clearance
    treatment = prepared.column_treatment
    x = prepared.x
    y = prepared.y
    baseline_cols = _block_columns(config, prepared, config.baseline_blocks)
    full_cols = _block_columns(config, prepared, config.full_blocks)
    if not baseline_cols or not full_cols:
        raise ModelingError("Baseline or full model resolved to zero columns.")
    if set(baseline_cols) == set(full_cols):
        raise ModelingError("Baseline and full designs are identical after feature build.")

    y_test = np.asarray(y.iloc[test_idx])
    base_prob = _fit_predict(config, x, y, baseline_cols, treatment, train_idx, test_idx)
    full_prob = _fit_predict(config, x, y, full_cols, treatment, train_idx, test_idx)
    base_holdout = _scores(y_test, base_prob)
    full_holdout = _scores(y_test, full_prob)
    holdout = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "baseline_roc_auc": base_holdout["roc_auc"],
        "baseline_pr_auc": base_holdout["pr_auc"],
        "baseline_brier": base_holdout["brier"],
        "full_roc_auc": full_holdout["roc_auc"],
        "full_pr_auc": full_holdout["pr_auc"],
        "full_brier": full_holdout["brier"],
        "roc_lift": full_holdout["roc_auc"] - base_holdout["roc_auc"],
        "pr_lift": full_holdout["pr_auc"] - base_holdout["pr_auc"],
    }

    splitter = RepeatedStratifiedKFold(
        n_splits=config.cv_splits,
        n_repeats=config.cv_repeats,
        random_state=config.cv_seed,
    )
    x_train = x.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    ablation_blocks = list(config.full_blocks)
    fold_rows: list[dict] = []
    y_train_values = np.asarray(y_train)
    for repeat_id, (inner_train, inner_val) in enumerate(
        splitter.split(np.zeros((len(y_train), 1)), y_train_values),
        start=1,
    ):
        split_id = ((repeat_id - 1) % config.cv_splits) + 1
        repeat_number = ((repeat_id - 1) // config.cv_splits) + 1
        row: dict[str, float | int] = {
            "repeat": repeat_number,
            "split": split_id,
            "n_cv_train": int(len(inner_train)),
            "n_cv_val": int(len(inner_val)),
        }
        probabilities = {
            "baseline": _fit_predict(
                config, x_train, y_train, baseline_cols, treatment, inner_train, inner_val
            ),
            "full": _fit_predict(
                config, x_train, y_train, full_cols, treatment, inner_train, inner_val
            ),
        }
        y_val = y_train_values[inner_val]
        scored = {name: _scores(y_val, prob) for name, prob in probabilities.items()}
        for metric in ("roc_auc", "pr_auc", "brier"):
            row[f"delta_{metric}_full_minus_baseline"] = (
                scored["full"][metric] - scored["baseline"][metric]
            )
        for block_name in ablation_blocks:
            reduced_names = tuple(
                name for name in config.full_blocks if name != block_name
            )
            reduced_cols = _block_columns(config, prepared, reduced_names)
            if not reduced_cols or set(reduced_cols) == set(full_cols):
                raise ModelingError(
                    f"Ablation of block {block_name!r} did not change the design."
                )
            reduced_prob = _fit_predict(
                config, x_train, y_train, reduced_cols, treatment, inner_train, inner_val
            )
            reduced_scores = _scores(y_val, reduced_prob)
            safe = block_name.replace(" ", "_")
            for metric in ("roc_auc", "pr_auc", "brier"):
                row[f"delta_{metric}_without_{safe}_minus_full"] = (
                    reduced_scores[metric] - scored["full"][metric]
                )
        fold_rows.append(row)

    contrasts = [
        ContrastSpec(
            contrast_id="A1",
            family="A",
            label="Full − baseline",
            column=f"delta_{metric}_full_minus_baseline",
            metric=metric,
            adjust=False,
        )
        for metric in ("roc_auc", "pr_auc", "brier")
    ]
    for index, block_name in enumerate(ablation_blocks, start=1):
        safe = block_name.replace(" ", "_")
        for metric in ("roc_auc", "pr_auc", "brier"):
            contrasts.append(
                ContrastSpec(
                    contrast_id=f"B{index}",
                    family="B",
                    label=f"Without {block_name} − full",
                    column=f"delta_{metric}_without_{safe}_minus_full",
                    metric=metric,
                    adjust=True,
                )
            )
    return ModelComparison(holdout=holdout, folds=pd.DataFrame(fold_rows), contrasts=contrasts)
