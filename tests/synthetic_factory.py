"""Synthetic configuration fixtures. Invented numbers only — not study results."""

from __future__ import annotations

from datetime import date

from feasibility_framework.config import Predictor, PredictorBlock, StudyConfig


def predictor(
    name: str,
    *,
    treatment: str = "numeric",
    confirmed: bool = True,
    availability: date | None = date(2020, 12, 31),
    time_invariant: bool = False,
) -> Predictor:
    return Predictor(
        name=name,
        treatment=treatment,
        confirmed=confirmed,
        availability=availability,
        time_invariant=time_invariant,
        description="synthetic predictor",
    )


def valid_config(**overrides) -> StudyConfig:
    blocks = (
        PredictorBlock(
            name="history",
            confirmed=True,
            predictors=(predictor("prior_count"),),
        ),
        PredictorBlock(
            name="extra",
            confirmed=True,
            predictors=(
                predictor("signal"),
                predictor(
                    "group_code",
                    treatment="categorical",
                    availability=None,
                    time_invariant=True,
                ),
            ),
        ),
    )
    raw = dict(
        study_id="synthetic-study",
        outcome="event",
        outcome_date=date(2022, 1, 1),
        cutoff=date(2021, 12, 31),
        id_key="person_id",
        blocks=blocks,
        baseline_blocks=("history",),
        full_blocks=("history", "extra"),
        sources=("cohort",),
        primary_source="cohort",
        epv_floor=10.0,
        min_usable_n=200,
        min_outcome_events=40,
        min_outcome_prevalence=0.05,
        min_baseline_roc_auc=0.55,
        min_incremental_roc_lift=0.02,
        min_incremental_pr_lift=0.01,
    )
    raw.update(overrides)
    return StudyConfig(**raw)
