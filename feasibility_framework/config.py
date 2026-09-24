"""Study configuration: predictor blocks, floors, and fail-closed validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any


class ConfigError(ValueError):
    """Raised when a study configuration violates a pre-registration rule."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        detail = "\n- ".join(self.problems)
        super().__init__(f"StudyConfig is invalid:\n- {detail}")


@dataclass(frozen=True)
class Predictor:
    """One column that may enter the design matrix only if it is confirmed and pre-cutoff."""

    name: str
    treatment: str
    confirmed: bool
    availability: date | None = None
    time_invariant: bool = False
    description: str = ""


@dataclass(frozen=True)
class PredictorBlock:
    """A named group of predictors. Unconfirmed blocks never enter a model."""

    name: str
    predictors: tuple[Predictor, ...]
    confirmed: bool
    max_categorical_levels: int | None = None
    composite_strategy: str | None = None


@dataclass(frozen=True)
class StudyConfig:
    """Locked specification for one feasibility study.

    Floors are part of the configuration so they are hashed into the
    pre-registration lock. They are not inferred from data.
    """

    study_id: str
    outcome: str
    outcome_date: date
    cutoff: date
    id_key: str
    blocks: tuple[PredictorBlock, ...]
    baseline_blocks: tuple[str, ...]
    full_blocks: tuple[str, ...]
    sources: tuple[str, ...]
    primary_source: str
    epv_floor: float
    min_usable_n: int
    min_outcome_events: int
    min_outcome_prevalence: float
    min_baseline_roc_auc: float
    min_incremental_roc_lift: float
    min_incremental_pr_lift: float
    min_present_fraction: float = 1.0
    outcome_positive_if_at_least: float = 1.0
    invalid_values: tuple[float, ...] = ()
    holdout_fraction: float = 0.25
    random_seed: int = 42
    cv_splits: int = 5
    cv_repeats: int = 5
    cv_seed: int = 2021
    estimator_c: float = 1.0
    inference_method: str = "nadeau_bengio_holm"
    epv_basis: str = "min_fold_train_events"

    def block_map(self) -> dict[str, PredictorBlock]:
        return {block.name: block for block in self.blocks}

    def predictor_index(self) -> dict[str, tuple[PredictorBlock, Predictor]]:
        index: dict[str, tuple[PredictorBlock, Predictor]] = {}
        for block in self.blocks:
            for predictor in block.predictors:
                index[predictor.name] = (block, predictor)
        return index

    def columns_for(self, block_names: tuple[str, ...]) -> list[str]:
        mapping = self.block_map()
        columns: list[str] = []
        for name in block_names:
            block = mapping[name]
            columns.extend(predictor.name for predictor in block.predictors)
        return columns

    def validate(self) -> None:
        problems: list[str] = []
        if not self.study_id.strip():
            problems.append("study_id is empty.")
        if not self.outcome.strip():
            problems.append("outcome name is empty.")
        if not self.id_key.strip():
            problems.append("id_key is empty.")
        _require_date(self.cutoff, "cutoff", problems)
        _require_date(self.outcome_date, "outcome_date", problems)
        if type(self.cutoff) is date and type(self.outcome_date) is date:
            if self.outcome_date <= self.cutoff:
                problems.append(
                    f"outcome_date {self.outcome_date.isoformat()} must be after "
                    f"cutoff {self.cutoff.isoformat()}."
                )
        if self.outcome == self.id_key:
            problems.append("outcome and id_key must be different columns.")

        if not self.blocks:
            problems.append("at least one predictor block is required.")
        if not self.sources:
            problems.append("at least one data source is required.")
        if self.primary_source not in self.sources:
            problems.append(
                f"primary_source {self.primary_source!r} is not listed in sources."
            )
        if len(set(self.sources)) != len(self.sources):
            problems.append("source names must be unique.")

        seen_blocks: set[str] = set()
        seen_predictors: set[str] = set()
        for block in self.blocks:
            if not block.name.strip():
                problems.append("a predictor block has an empty name.")
                continue
            if block.name in seen_blocks:
                problems.append(f"duplicate block name {block.name!r}.")
            seen_blocks.add(block.name)
            if type(block.confirmed) is not bool:
                problems.append(f"block {block.name!r} confirmed must be a bool.")
            elif not block.confirmed:
                problems.append(
                    f"block {block.name!r} is unconfirmed and fails closed."
                )
            if not block.predictors:
                problems.append(f"block {block.name!r} has no predictors.")
            if block.max_categorical_levels is not None and block.max_categorical_levels < 2:
                problems.append(
                    f"block {block.name!r} max_categorical_levels must be >= 2 when set."
                )
            if block.composite_strategy is not None and not str(block.composite_strategy).strip():
                problems.append(f"block {block.name!r} composite_strategy is blank.")
            for predictor in block.predictors:
                _validate_predictor(self, block, predictor, seen_predictors, problems)

        block_names = {block.name for block in self.blocks}
        for label, selected in (
            ("baseline_blocks", self.baseline_blocks),
            ("full_blocks", self.full_blocks),
        ):
            if not selected:
                problems.append(f"{label} is empty.")
            missing = [name for name in selected if name not in block_names]
            if missing:
                problems.append(f"{label} names unknown blocks: {missing}.")
            if len(set(selected)) != len(selected):
                problems.append(f"{label} contains duplicate names.")

        if (
            self.baseline_blocks
            and self.full_blocks
            and set(self.baseline_blocks) == set(self.full_blocks)
        ):
            problems.append(
                "baseline_blocks and full_blocks are the same set, so there is "
                "no incremental contrast to pre-register."
            )
        if self.baseline_blocks and self.full_blocks:
            extra = [name for name in self.baseline_blocks if name not in set(self.full_blocks)]
            if extra:
                problems.append(
                    f"baseline_blocks must be a subset of full_blocks; extra: {extra}."
                )

        _positive(self.epv_floor, "epv_floor", problems, strict=True)
        _non_negative_int(self.min_usable_n, "min_usable_n", problems, minimum=1)
        _non_negative_int(self.min_outcome_events, "min_outcome_events", problems, minimum=1)
        if not isinstance(self.min_outcome_prevalence, (int, float)) or isinstance(
            self.min_outcome_prevalence, bool
        ):
            problems.append("min_outcome_prevalence must be a number in (0, 1].")
        elif not 0 < float(self.min_outcome_prevalence) <= 1:
            problems.append("min_outcome_prevalence must be in (0, 1].")
        for field_name in (
            "min_baseline_roc_auc",
            "min_incremental_roc_lift",
            "min_incremental_pr_lift",
            "min_present_fraction",
            "outcome_positive_if_at_least",
            "holdout_fraction",
            "estimator_c",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{field_name} must be numeric.")
        if isinstance(self.holdout_fraction, (int, float)) and not isinstance(
            self.holdout_fraction, bool
        ):
            if not 0 < float(self.holdout_fraction) < 0.5:
                problems.append("holdout_fraction must be in (0, 0.5).")
        if isinstance(self.min_present_fraction, (int, float)) and not isinstance(
            self.min_present_fraction, bool
        ):
            if not 0 < float(self.min_present_fraction) <= 1:
                problems.append("min_present_fraction must be in (0, 1].")
        if isinstance(self.min_baseline_roc_auc, (int, float)) and not isinstance(
            self.min_baseline_roc_auc, bool
        ):
            if not 0.5 <= float(self.min_baseline_roc_auc) < 1:
                problems.append("min_baseline_roc_auc must be in [0.5, 1).")
        if isinstance(self.estimator_c, (int, float)) and not isinstance(self.estimator_c, bool):
            if float(self.estimator_c) <= 0:
                problems.append("estimator_c must be positive.")
        _non_negative_int(self.cv_splits, "cv_splits", problems, minimum=2)
        _non_negative_int(self.cv_repeats, "cv_repeats", problems, minimum=1)
        _non_negative_int(self.random_seed, "random_seed", problems, minimum=0)
        _non_negative_int(self.cv_seed, "cv_seed", problems, minimum=0)
        if not str(self.inference_method).strip():
            problems.append("inference_method is empty.")
        allowed_epv_bases = {"min_fold_train_events", "mean_fold_train_events"}
        if self.epv_basis not in allowed_epv_bases:
            problems.append(
                f"epv_basis {self.epv_basis!r} is not one of {sorted(allowed_epv_bases)}."
            )
        if problems:
            raise ConfigError(problems)


def _validate_predictor(
    config: StudyConfig,
    block: PredictorBlock,
    predictor: Predictor,
    seen: set[str],
    problems: list[str],
) -> None:
    if not predictor.name.strip():
        problems.append(f"block {block.name!r} has a predictor with an empty name.")
        return
    if predictor.name in seen:
        problems.append(f"duplicate predictor name {predictor.name!r}.")
    seen.add(predictor.name)
    if predictor.name in {config.outcome, config.id_key}:
        problems.append(
            f"predictor {predictor.name!r} collides with the outcome or id key."
        )
    if predictor.treatment not in {"numeric", "categorical"}:
        problems.append(
            f"predictor {predictor.name!r} treatment must be 'numeric' or 'categorical'."
        )
    if type(predictor.confirmed) is not bool:
        problems.append(f"predictor {predictor.name!r} confirmed must be a bool.")
    elif not predictor.confirmed:
        problems.append(
            f"predictor {predictor.name!r} is unconfirmed and fails closed."
        )
    if type(predictor.time_invariant) is not bool:
        problems.append(f"predictor {predictor.name!r} time_invariant must be a bool.")
        return
    if predictor.availability is not None and type(predictor.availability) is not date:
        problems.append(
            f"predictor {predictor.name!r} availability must be a datetime.date or None."
        )
        return
    if predictor.time_invariant:
        if (
            predictor.availability is not None
            and type(config.cutoff) is date
            and predictor.availability > config.cutoff
        ):
            problems.append(
                f"predictor {predictor.name!r} is marked time-invariant but its "
                "availability date is after the cutoff."
            )
        return
    if predictor.availability is None:
        problems.append(
            f"predictor {predictor.name!r} has unknown temporal availability "
            "and fails closed."
        )
        return
    if type(config.cutoff) is date and predictor.availability > config.cutoff:
        problems.append(
            f"predictor {predictor.name!r} availability "
            f"{predictor.availability.isoformat()} is after cutoff "
            f"{config.cutoff.isoformat()}."
        )


def _require_date(value: object, field_name: str, problems: list[str]) -> None:
    if type(value) is not date:
        problems.append(f"{field_name} must be a datetime.date.")


def _positive(value: object, field_name: str, problems: list[str], *, strict: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        problems.append(f"{field_name} must be a positive number.")
        return
    if strict and float(value) <= 0:
        problems.append(f"{field_name} must be > 0.")


def _non_negative_int(value: object, field_name: str, problems: list[str], *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        problems.append(f"{field_name} must be an int >= {minimum}.")
        return
    if value < minimum:
        problems.append(f"{field_name} must be >= {minimum}.")


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        raise TypeError("Config dates must be datetime.date, not datetime.")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_config(config: StudyConfig) -> dict[str, Any]:
    """JSON-ready payload. Key order is fixed later by the hasher."""
    config.validate()
    return _jsonable(asdict(config))


def config_hash(config: StudyConfig) -> str:
    payload = canonical_config(config)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
