"""EPV basis, floor edge cases, and design-column counting. Synthetic only."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from feasibility_framework import FeasibilityStudy, GateVerdict
from feasibility_framework.config import ConfigError, Predictor, PredictorBlock, config_hash
from feasibility_framework.dimensionality import resolve_epv
from feasibility_framework.modeling import design_column_count
from tests.synthetic_factory import valid_config


def test_mean_clears_but_minimum_fold_does_not():
    # Hand-built fold event counts: mean = 41, min = 5, columns = 2.
    # mean EPV = 20.5 clears a floor of 10; min EPV = 2.5 does not.
    fold_events = [50, 50, 50, 50, 5]
    epv, min_events, mean_events, clears = resolve_epv(
        fold_events,
        n_design_columns=2,
        epv_floor=10.0,
        epv_basis="min_fold_train_events",
    )
    assert min_events == 5
    assert mean_events == 41
    assert epv == pytest.approx(2.5)
    assert clears is False
    mean_epv, _, _, mean_clears = resolve_epv(
        fold_events,
        n_design_columns=2,
        epv_floor=10.0,
        epv_basis="mean_fold_train_events",
    )
    assert mean_epv == pytest.approx(20.5)
    assert mean_clears is True


def test_min_basis_underpowered_blocks_modeling(tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("modeling ran after an EPV failure on the minimum fold")

    monkeypatch.setattr(
        "feasibility_framework.modeling.fit_comparison",
        fail_if_called,
    )
    monkeypatch.setattr(
        "feasibility_framework.dimensionality.fold_train_event_counts",
        lambda *args, **kwargs: ([50, 50, 50, 50, 5], [80, 80, 80, 80, 80]),
    )

    class Study(FeasibilityStudy):
        def define_config(self):
            return valid_config(
                epv_floor=10.0,
                epv_basis="min_fold_train_events",
                min_usable_n=20,
                min_outcome_events=5,
            )

        def load_data(self, path_map):
            del path_map
            rng = np.random.default_rng(0)
            n = 120
            return {
                "cohort": pd.DataFrame(
                    {
                        "person_id": np.arange(n),
                        "prior_count": rng.normal(size=n),
                        "signal": rng.normal(size=n),
                        "group_code": rng.choice(["a", "b", "c"], size=n),
                        "event": rng.integers(0, 2, size=n),
                    }
                )
            }

    study = Study(lock_dir=tmp_path)
    study.lock_preregistration()
    result = study.run()
    assert result.gate is GateVerdict.UNDERPOWERED
    assert result.model_metrics is None
    assert result.dimensionality is not None
    assert result.dimensionality.epv_basis == "min_fold_train_events"
    assert result.dimensionality.min_cv_train_events == 5
    assert result.dimensionality.mean_cv_train_events == 41
    assert result.dimensionality.clears_floor is False


def test_minimum_exactly_at_floor_clears_and_one_below_does_not():
    # columns = 2, floor = 10 → need 20 events in the minimum fold.
    at_floor, _, _, clears = resolve_epv(
        [20, 30, 40],
        n_design_columns=2,
        epv_floor=10.0,
        epv_basis="min_fold_train_events",
    )
    assert at_floor == pytest.approx(10.0)
    assert clears is True
    below, _, _, fails = resolve_epv(
        [19, 30, 40],
        n_design_columns=2,
        epv_floor=10.0,
        epv_basis="min_fold_train_events",
    )
    assert below == pytest.approx(9.5)
    assert fails is False


def test_onehot_keeps_all_five_levels_as_design_columns():
    # OneHotEncoder is configured without drop=, so a 5-level categorical
    # contributes exactly 5 design columns (first level is not dropped).
    frame = pd.DataFrame(
        {
            "cat": ["a", "b", "c", "d", "e"] * 4,
            "num": np.arange(20, dtype=float),
        }
    )
    n = design_column_count(
        frame,
        ["cat"],
        {"cat": "categorical"},
    )
    assert n == 5


def test_epv_basis_changes_config_hash_and_unknown_is_rejected():
    base = valid_config(epv_basis="min_fold_train_events")
    other = valid_config(epv_basis="mean_fold_train_events")
    assert config_hash(base) != config_hash(other)
    with pytest.raises(ConfigError, match="epv_basis"):
        valid_config(epv_basis="median_fold_train_events").validate()
