"""Stage order cannot be skipped."""

import pytest

from feasibility_framework import FeasibilityStudy, NotLockedError, StageOrderError
from tests.synthetic_factory import valid_config


class _Study(FeasibilityStudy):
    def define_config(self):
        return valid_config()

    def load_data(self, path_map):
        raise AssertionError("load_data must not run before the lock is accepted")


def test_modeling_before_lock_raises():
    study = _Study()
    with pytest.raises(StageOrderError, match="modeled"):
        study._enter_modeling()


def test_run_before_lock_raises(tmp_path):
    study = _Study(lock_dir=tmp_path)
    with pytest.raises(NotLockedError):
        study.run()


def test_modeling_after_lock_but_before_epv_raises(tmp_path):
    study = _Study(lock_dir=tmp_path)
    study.lock_preregistration()
    assert study.stage.value == "locked"
    with pytest.raises(StageOrderError, match="modeled"):
        study._enter_modeling()


def test_amended_config_does_not_match_the_old_lock(tmp_path):
    study = _Study(lock_dir=tmp_path)
    study.lock_preregistration()

    class Amended(_Study):
        def define_config(self):
            return valid_config(epv_floor=12.0)

    amended = Amended(lock_dir=tmp_path)
    from feasibility_framework import LockMismatchError

    with pytest.raises(LockMismatchError):
        amended.run()
