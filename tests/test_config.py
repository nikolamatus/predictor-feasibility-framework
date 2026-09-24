"""Config validation against synthetic specifications."""

from datetime import date

import pytest

from feasibility_framework.config import ConfigError, Predictor, PredictorBlock, config_hash
from tests.synthetic_factory import predictor, valid_config


def test_valid_config_hashes_stably():
    cfg = valid_config()
    cfg.validate()
    assert config_hash(cfg) == config_hash(valid_config())


def test_unconfirmed_block_fails_closed():
    history, extra = valid_config().blocks
    cfg = valid_config(
        blocks=(
            history,
            PredictorBlock(name=extra.name, predictors=extra.predictors, confirmed=False),
        )
    )
    with pytest.raises(ConfigError, match="unconfirmed"):
        cfg.validate()


def test_unconfirmed_predictor_fails_closed():
    cfg = valid_config(
        blocks=(
            PredictorBlock(
                name="history",
                confirmed=True,
                predictors=(predictor("prior_count", confirmed=False),),
            ),
            valid_config().blocks[1],
        )
    )
    with pytest.raises(ConfigError, match="unconfirmed"):
        cfg.validate()


def test_post_cutoff_predictor_is_rejected():
    cfg = valid_config(
        blocks=(
            PredictorBlock(
                name="history",
                confirmed=True,
                predictors=(predictor("prior_count", availability=date(2022, 6, 1)),),
            ),
            valid_config().blocks[1],
        )
    )
    with pytest.raises(ConfigError, match="after cutoff"):
        cfg.validate()


def test_unknown_availability_fails_closed():
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


def test_outcome_on_or_before_cutoff_is_rejected():
    cfg = valid_config(outcome_date=date(2021, 12, 31))
    with pytest.raises(ConfigError, match="outcome_date"):
        cfg.validate()


def test_baseline_must_be_a_strict_subset():
    cfg = valid_config(baseline_blocks=("history", "extra"), full_blocks=("history", "extra"))
    with pytest.raises(ConfigError, match="incremental contrast"):
        cfg.validate()


def test_unknown_block_reference_is_rejected():
    cfg = valid_config(baseline_blocks=("missing",))
    with pytest.raises(ConfigError, match="unknown blocks"):
        cfg.validate()


def test_non_positive_epv_floor_is_rejected():
    cfg = valid_config(epv_floor=0)
    with pytest.raises(ConfigError, match="epv_floor"):
        cfg.validate()


def test_datetime_is_not_accepted_as_a_date():
    from datetime import datetime

    cfg = valid_config(cutoff=datetime(2021, 12, 31, 23, 59))
    with pytest.raises(ConfigError, match="cutoff"):
        cfg.validate()


def test_confirmed_must_be_a_real_bool():
    cfg = valid_config(
        blocks=(
            PredictorBlock(
                name="history",
                confirmed=1,  # type: ignore[arg-type]
                predictors=(Predictor(name="prior_count", treatment="numeric", confirmed=True, availability=date(2020, 1, 1)),),
            ),
            valid_config().blocks[1],
        )
    )
    with pytest.raises(ConfigError, match="confirmed must be a bool"):
        cfg.validate()
