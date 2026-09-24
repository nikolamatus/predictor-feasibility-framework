"""Feature audit and design-matrix construction from a locked config."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from feasibility_framework.config import StudyConfig
from feasibility_framework.leakage_guard import LeakageError, assert_no_leakage
from feasibility_framework.registries import get_composite


class FeatureError(RuntimeError):
    """Raised when the analytic table cannot support the locked predictors."""


@dataclass
class PreparedFeatures:
    frame: pd.DataFrame
    x: pd.DataFrame
    y: pd.Series
    audit: pd.DataFrame
    derived_from: dict[str, list[str]]
    collapsed_blocks: tuple[str, ...]
    column_treatment: dict[str, str]


def _mask_invalid(series: pd.Series, invalid_values: tuple[float, ...]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if invalid_values:
        numeric = numeric.mask(numeric.isin(invalid_values))
    return numeric


def prepare_features(config: StudyConfig, frame: pd.DataFrame) -> PreparedFeatures:
    if config.outcome not in frame.columns:
        raise FeatureError(f"Outcome column {config.outcome!r} is missing.")
    if config.id_key not in frame.columns:
        raise FeatureError(f"Identifier {config.id_key!r} is missing.")

    y_raw = _mask_invalid(frame[config.outcome], config.invalid_values)
    if y_raw.isna().any():
        raise FeatureError(
            "Outcome has missing or non-numeric values after invalid-code handling. "
            "Refusing to drop rows silently."
        )
    y = (y_raw >= config.outcome_positive_if_at_least).astype(int)
    y.name = config.outcome
    if y.nunique() < 2:
        raise FeatureError("Outcome has a single class. Refusing to continue.")

    index = config.predictor_index()
    audit_rows = []
    pieces: list[pd.Series] = []
    derived_from: dict[str, list[str]] = {}
    collapsed: list[str] = []
    treatment: dict[str, str] = {}
    selected = set(config.full_blocks)

    for block in config.blocks:
        if block.name not in selected:
            continue
        source_names = [predictor.name for predictor in block.predictors]
        missing = [name for name in source_names if name not in frame.columns]
        if missing:
            raise FeatureError(
                f"Block {block.name!r} is missing columns {missing}. "
                "Blocks are configured explicitly and are not inferred."
            )
        for predictor in block.predictors:
            present = True
            values = frame[predictor.name]
            if predictor.treatment == "numeric":
                usable_values = _mask_invalid(values, config.invalid_values)
                all_missing = bool(usable_values.isna().all())
            else:
                all_missing = bool(values.isna().all())
            audit_rows.append(
                {
                    "feature": predictor.name,
                    "block": block.name,
                    "treatment": predictor.treatment,
                    "availability": (
                        predictor.availability.isoformat()
                        if predictor.availability is not None
                        else None
                    ),
                    "time_invariant": predictor.time_invariant,
                    "confirmed": predictor.confirmed and block.confirmed,
                    "present": present,
                    "usable": not all_missing,
                    "reason": (
                        "all values missing"
                        if all_missing
                        else predictor.description or "configured predictor"
                    ),
                }
            )
            if all_missing:
                raise FeatureError(
                    f"Predictor {predictor.name!r} is entirely missing. Failing closed."
                )

        cat_predictors = [p for p in block.predictors if p.treatment == "categorical"]
        max_levels = 0
        for predictor in cat_predictors:
            max_levels = max(max_levels, int(frame[predictor.name].nunique(dropna=True)))
        exceed = (
            block.max_categorical_levels is not None
            and max_levels > block.max_categorical_levels
        )
        if exceed:
            if not block.composite_strategy:
                raise FeatureError(
                    f"Block {block.name!r} has {max_levels} categorical levels, "
                    f"above max_categorical_levels={block.max_categorical_levels}, "
                    "and no composite_strategy is set on the locked config."
                )
            strategy = get_composite(block.composite_strategy)
            score = strategy(frame, source_names)
            column = f"{block.name}__score"
            score.name = column
            pieces.append(score)
            derived_from[column] = list(source_names)
            treatment[column] = "numeric"
            collapsed.append(block.name)
            continue

        for predictor in block.predictors:
            if predictor.treatment == "numeric":
                column = _mask_invalid(frame[predictor.name], config.invalid_values)
            else:
                column = frame[predictor.name].copy()
            column.name = predictor.name
            pieces.append(column)
            treatment[predictor.name] = predictor.treatment

    if not pieces:
        raise FeatureError("No predictor columns were built.")
    x = pd.concat(pieces, axis=1)
    x.index = frame.index
    assert_no_leakage(config, list(x.columns), derived_from=derived_from)
    unresolved = [name for name in x.columns if name not in index and name not in derived_from]
    if unresolved:
        raise LeakageError(f"Predictor matrix contains unresolved columns: {unresolved}.")

    usable = sum(1 for row in audit_rows if row["usable"])
    if usable / len(audit_rows) < config.min_present_fraction:
        raise FeatureError(
            f"Only {usable}/{len(audit_rows)} configured predictors are usable, "
            f"below min_present_fraction={config.min_present_fraction}."
        )

    return PreparedFeatures(
        frame=frame,
        x=x,
        y=y,
        audit=pd.DataFrame(audit_rows),
        derived_from=derived_from,
        collapsed_blocks=tuple(collapsed),
        column_treatment=treatment,
    )
