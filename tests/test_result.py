"""Supersession chaining on synthetic result objects. No fitted metrics."""

import pytest

from feasibility_framework.result import (
    DimensionalityReport,
    FeasibilityResult,
    GateVerdict,
    ResultError,
)


def _result(study_id: str = "synthetic-study", gate: GateVerdict = GateVerdict.RED) -> FeasibilityResult:
    return FeasibilityResult.create(
        study_id=study_id,
        config_hash="abc",
        gate=gate,
        dimensionality=DimensionalityReport(
            n_design_columns=4,
            n_baseline_columns=1,
            mean_cv_train_n=80,
            mean_cv_train_events=20,
            epv=5.0,
            epv_floor=10.0,
            clears_floor=False,
        ),
        stop_reason="synthetic fixture",
        synthetic=True,
    )


def test_later_result_supersedes_earlier():
    first = _result(gate=GateVerdict.UNDERPOWERED)
    second = _result(gate=GateVerdict.YELLOW)
    assert first.superseded_by is None
    second.supersedes(first)
    assert first.superseded_by == second.result_id
    assert second.superseded_by is None


def test_cannot_supersede_self():
    result = _result()
    with pytest.raises(ResultError, match="itself"):
        result.supersedes(result)


def test_cannot_supersede_a_different_study():
    first = _result(study_id="study-a")
    second = _result(study_id="study-b")
    with pytest.raises(ResultError, match="same study"):
        second.supersedes(first)


def test_json_roundtrip_keeps_supersession(tmp_path):
    first = _result(gate=GateVerdict.UNDERPOWERED)
    second = _result(gate=GateVerdict.GREEN)
    second.supersedes(first)
    path = tmp_path / "result.json"
    first.to_json(path)
    loaded = FeasibilityResult.from_json(path)
    assert loaded.superseded_by == second.result_id
    assert loaded.gate is GateVerdict.UNDERPOWERED
    assert loaded.dimensionality is not None
    assert loaded.dimensionality.clears_floor is False
    assert loaded.synthetic is True


def test_chain_of_three():
    first = _result()
    second = _result()
    third = _result()
    second.supersedes(first)
    third.supersedes(second)
    assert first.superseded_by == second.result_id
    assert second.superseded_by == third.result_id
