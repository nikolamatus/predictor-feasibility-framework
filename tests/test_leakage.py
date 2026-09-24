"""Leakage guard edge cases. Synthetic metadata only."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from feasibility_framework.config import ConfigError, Predictor, PredictorBlock
from feasibility_framework.leakage_guard import LeakageError, assert_no_leakage
from tests.synthetic_factory import predictor, valid_config


def test_confirmed_predictor_without_availability_fails_closed():
    cfg = valid_config(
        blocks=(
            PredictorBlock(
                name="history",
                confirmed=True,
                predictors=(predictor("prior_count", availability=None),),
            ),
            valid_config().blocks[1],
        )
    )
    with pytest.raises(ConfigError, match="unknown temporal availability"):
        cfg.validate()


def test_availability_exactly_on_cutoff_is_allowed():
    # Cutoff comparison is inclusive: availability == cutoff may enter X.
    cutoff = date(2021, 12, 31)
    cfg = valid_config(
        cutoff=cutoff,
        blocks=(
            PredictorBlock(
                name="history",
                confirmed=True,
                predictors=(predictor("prior_count", availability=cutoff),),
            ),
            valid_config().blocks[1],
        ),
    )
    cfg.validate()
    assert_no_leakage(cfg, ["prior_count", "signal", "group_code"])


def test_null_availability_fails_in_leakage_guard():
    # Build a config that skips validate() so the guard itself is the fail-closed path.
    history = Predictor(
        name="prior_count",
        treatment="numeric",
        confirmed=True,
        availability=None,
        time_invariant=False,
        description="synthetic",
    )
    cfg = valid_config(
        blocks=(
            PredictorBlock(name="history", confirmed=True, predictors=(history,)),
            valid_config().blocks[1],
        )
    )
    with pytest.raises(LeakageError, match="unknown temporal availability"):
        assert_no_leakage(cfg, ["prior_count"])


def test_unparseable_availability_fails_config_validation():
    bad = Predictor(
        name="prior_count",
        treatment="numeric",
        confirmed=True,
        availability="2020-01-01",  # type: ignore[arg-type]
        description="synthetic",
    )
    cfg = valid_config(
        blocks=(
            PredictorBlock(name="history", confirmed=True, predictors=(bad,)),
            valid_config().blocks[1],
        )
    )
    with pytest.raises(ConfigError, match="availability must be a datetime.date"):
        cfg.validate()


def test_datetime_instead_of_date_fails_closed():
    bad = Predictor(
        name="prior_count",
        treatment="numeric",
        confirmed=True,
        availability=datetime(2020, 1, 1, 12, 0),  # type: ignore[arg-type]
        description="synthetic",
    )
    cfg = valid_config(
        blocks=(
            PredictorBlock(name="history", confirmed=True, predictors=(bad,)),
            valid_config().blocks[1],
        )
    )
    with pytest.raises(ConfigError, match="availability must be a datetime.date"):
        cfg.validate()
