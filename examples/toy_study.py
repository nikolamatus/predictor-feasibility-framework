"""Synthetic toy study. Every row is invented. Nothing here is a real cohort."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from feasibility_framework import FeasibilityStudy, Predictor, PredictorBlock, StudyConfig


class ToyStudy(FeasibilityStudy):
    """Minimal end-to-end study used by the CLI demo and the tests."""

    def define_config(self) -> StudyConfig:
        history = PredictorBlock(
            name="history",
            confirmed=True,
            predictors=(
                Predictor(
                    name="prior_count",
                    treatment="numeric",
                    confirmed=True,
                    availability=date(2021, 12, 31),
                    description="synthetic prior count",
                ),
            ),
        )
        extra = PredictorBlock(
            name="extra",
            confirmed=True,
            predictors=(
                Predictor(
                    name="signal",
                    treatment="numeric",
                    confirmed=True,
                    availability=date(2021, 6, 30),
                    description="synthetic extra signal",
                ),
                Predictor(
                    name="group_code",
                    treatment="categorical",
                    confirmed=True,
                    time_invariant=True,
                    description="synthetic group",
                ),
            ),
        )
        return StudyConfig(
            study_id="toy-synthetic",
            outcome="event",
            outcome_date=date(2022, 1, 1),
            cutoff=date(2021, 12, 31),
            id_key="person_id",
            blocks=(history, extra),
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

    def load_data(self, path_map: dict[str, Path]) -> dict[str, pd.DataFrame]:
        del path_map
        return {"cohort": synthetic_cohort()}


def synthetic_cohort(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """Invented rows with a fixed seed. Not observations from any dataset."""
    rng = np.random.default_rng(seed)
    prior = rng.normal(size=n)
    signal = rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], size=n)
    logits = -0.2 + 0.2 * prior + 1.7 * signal
    probability = 1.0 / (1.0 + np.exp(-logits))
    event = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "person_id": np.arange(n),
            "prior_count": prior,
            "signal": signal,
            "group_code": group,
            "event": event,
        }
    )


if __name__ == "__main__":
    print("This module is a FeasibilityStudy subclass. Run it with: feasibility run examples/toy_study.py")
