"""Leakage guard and the EPV gate, on synthetic rows only."""

import pandas as pd
import pytest

from feasibility_framework import FeasibilityStudy, GateVerdict
from feasibility_framework.leakage_guard import LeakageError, assert_no_leakage
from feasibility_framework.modeling import ModelingError, fit_comparison
from tests.synthetic_factory import valid_config


def _frame(n: int = 240, seed: int = 1) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "person_id": np.arange(n),
            "prior_count": rng.normal(size=n),
            "signal": rng.normal(size=n),
            "group_code": rng.choice(["a", "b", "c"], size=n),
            "event": rng.integers(0, 2, size=n),
        }
    )


class _PowerStudy(FeasibilityStudy):
    def __init__(self, lock_dir, epv_floor: float):
        super().__init__(lock_dir=lock_dir)
        self._floor = epv_floor

    def define_config(self):
        return valid_config(epv_floor=self._floor, min_usable_n=50, min_outcome_events=10)

    def load_data(self, path_map):
        return {"cohort": _frame()}


def test_outcome_column_cannot_enter_x():
    cfg = valid_config()
    with pytest.raises(LeakageError, match="outcome"):
        assert_no_leakage(cfg, ["event"])


def test_unknown_column_fails_closed():
    cfg = valid_config()
    with pytest.raises(LeakageError, match="no predictor specification"):
        assert_no_leakage(cfg, ["later_count"])


def test_epv_failure_does_not_call_modeling(tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("modeling ran after an EPV failure")

    monkeypatch.setattr(
        "feasibility_framework.modeling.fit_comparison",
        fail_if_called,
    )
    study = _PowerStudy(tmp_path, epv_floor=1000)
    study.lock_preregistration()
    result = study.run()
    assert result.gate is GateVerdict.UNDERPOWERED
    assert result.model_metrics is None
    assert result.inference is None
    assert result.dimensionality is not None
    assert result.dimensionality.clears_floor is False
    assert study.stage.value == "underpowered"


def test_fit_comparison_without_clearance_raises():
    with pytest.raises(ModelingError, match="clearance"):
        fit_comparison(None, valid_config(), None, None, None)  # type: ignore[arg-type]


def test_powered_run_reaches_inference(tmp_path):
    study = _PowerStudy(tmp_path, epv_floor=1)
    study.lock_preregistration()
    result = study.run()
    assert result.gate is not GateVerdict.UNDERPOWERED
    assert result.model_metrics is not None
    assert result.inference is not None
    assert result.inference["method"] == "nadeau_bengio_holm"
    assert any(row["contrast_id"] == "A1" for row in result.inference["rows"])
    assert study.stage.value == "decided"
