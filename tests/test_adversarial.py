"""Adversarial checks for clearance, stage order, and lock hash. Synthetic only."""

from __future__ import annotations

import pytest

from feasibility_framework import (
    FeasibilityStudy,
    LockMismatchError,
    Stage,
    StageOrderError,
)
from feasibility_framework.dimensionality import ClearanceError, EpvClearance
from feasibility_framework.modeling import ModelingError, fit_comparison
from feasibility_framework.result import DimensionalityReport
from tests.synthetic_factory import valid_config


def _passing_report() -> DimensionalityReport:
    return DimensionalityReport(
        n_design_columns=2,
        n_baseline_columns=1,
        mean_cv_train_n=80,
        mean_cv_train_events=40,
        min_cv_train_events=40,
        epv=20.0,
        epv_floor=10.0,
        epv_basis="min_fold_train_events",
        clears_floor=True,
    )


class _Study(FeasibilityStudy):
    def define_config(self):
        return valid_config()

    def load_data(self, path_map):
        raise AssertionError("load_data must not run in these adversarial checks")


def test_direct_epv_clearance_construction_is_rejected():
    report = _passing_report()
    with pytest.raises(ClearanceError, match="dimensionality gate"):
        EpvClearance(report, object())
    with pytest.raises(ModelingError, match="clearance"):
        fit_comparison(None, valid_config(), None, None, None)  # type: ignore[arg-type]


def test_private_stage_methods_refuse_unlocked_or_unprepared_study(tmp_path):
    study = _Study(lock_dir=tmp_path)
    assert study.stage is Stage.INIT
    with pytest.raises(StageOrderError):
        study._enter_modeling()
    for target in (
        Stage.INGESTED,
        Stage.LINKED,
        Stage.LEAKAGE_CHECKED,
        Stage.FEATURES_BUILT,
        Stage.EPV_CHECKED,
        Stage.MODELED,
        Stage.INFERRED,
        Stage.DECIDED,
        Stage.UNDERPOWERED,
    ):
        with pytest.raises(StageOrderError):
            study._begin(target)

    study.lock_preregistration()
    assert study.stage is Stage.LOCKED
    with pytest.raises(StageOrderError):
        study._enter_modeling()
    for target in (
        Stage.LINKED,
        Stage.LEAKAGE_CHECKED,
        Stage.FEATURES_BUILT,
        Stage.EPV_CHECKED,
        Stage.MODELED,
        Stage.INFERRED,
        Stage.DECIDED,
        Stage.UNDERPOWERED,
    ):
        with pytest.raises(StageOrderError):
            study._begin(target)


def test_run_rejects_config_edited_after_lock(tmp_path):
    study = _Study(lock_dir=tmp_path)
    study.lock_preregistration()

    class Edited(_Study):
        def define_config(self):
            return valid_config(epv_floor=11.0)

    edited = Edited(lock_dir=tmp_path)
    with pytest.raises(LockMismatchError, match="does not match"):
        edited.run()
