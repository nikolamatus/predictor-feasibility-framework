"""Direct invariants that the mutation check relies on. Synthetic only."""

from __future__ import annotations

from datetime import date

import pytest

from feasibility_framework import FeasibilityStudy, LockMismatchError
from feasibility_framework.dimensionality import resolve_epv
from feasibility_framework.leakage_guard import LeakageError, assert_no_leakage
from tests.synthetic_factory import predictor, valid_config


def test_epv_clears_floor_uses_comparison_not_constant_true():
    # Guards against mutating clears_floor to always True.
    epv, _, _, clears = resolve_epv(
        [1, 1, 1],
        n_design_columns=10,
        epv_floor=10.0,
        epv_basis="min_fold_train_events",
    )
    assert epv < 10.0
    assert clears is False


def test_lock_hash_mismatch_still_raises(tmp_path):
    # Guards against mutating the lock-hash comparison to always match.
    class Study(FeasibilityStudy):
        def define_config(self):
            return valid_config()

        def load_data(self, path_map):
            raise AssertionError("unused")

    Study(lock_dir=tmp_path).lock_preregistration()

    class Edited(Study):
        def define_config(self):
            return valid_config(min_usable_n=201)

    with pytest.raises(LockMismatchError):
        Edited(lock_dir=tmp_path).run()


def test_cutoff_rejects_only_dates_after_cutoff():
    # Guards against mutating availability > cutoff into >= cutoff.
    cutoff = date(2021, 12, 31)
    on_cutoff = valid_config(
        cutoff=cutoff,
        blocks=(
            valid_config().blocks[0].__class__(
                name="history",
                confirmed=True,
                predictors=(predictor("prior_count", availability=cutoff),),
            ),
            valid_config().blocks[1],
        ),
    )
    assert_no_leakage(on_cutoff, ["prior_count", "signal", "group_code"])
    after = valid_config(
        cutoff=cutoff,
        blocks=(
            valid_config().blocks[0].__class__(
                name="history",
                confirmed=True,
                predictors=(predictor("prior_count", availability=date(2022, 1, 1)),),
            ),
            valid_config().blocks[1],
        ),
    )
    # Config validation rejects post-cutoff predictors before leakage runs.
    from feasibility_framework.config import ConfigError

    with pytest.raises(ConfigError, match="after cutoff"):
        after.validate()
    # Leakage guard itself still fails closed if such a name reached it.
    with pytest.raises(LeakageError, match="after"):
        assert_no_leakage(after, ["prior_count"])
