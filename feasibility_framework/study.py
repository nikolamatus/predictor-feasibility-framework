"""FeasibilityStudy: the engine owns stage order. Users implement two hooks."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from feasibility_framework.config import StudyConfig, config_hash
from feasibility_framework.result import FeasibilityResult, GateVerdict

if TYPE_CHECKING:
    import pandas as pd


class Stage(Enum):
    INIT = "init"
    LOCKED = "locked"
    INGESTED = "ingested"
    LINKED = "linked"
    LEAKAGE_CHECKED = "leakage_checked"
    FEATURES_BUILT = "features_built"
    EPV_CHECKED = "epv_checked"
    MODELED = "modeled"
    INFERRED = "inferred"
    DECIDED = "decided"
    UNDERPOWERED = "underpowered"


_TRANSITIONS: dict[Stage, frozenset[Stage]] = {
    Stage.INIT: frozenset({Stage.LOCKED}),
    Stage.LOCKED: frozenset({Stage.INGESTED}),
    Stage.INGESTED: frozenset({Stage.LINKED}),
    Stage.LINKED: frozenset({Stage.LEAKAGE_CHECKED}),
    Stage.LEAKAGE_CHECKED: frozenset({Stage.FEATURES_BUILT}),
    Stage.FEATURES_BUILT: frozenset({Stage.EPV_CHECKED}),
    Stage.EPV_CHECKED: frozenset({Stage.MODELED, Stage.UNDERPOWERED}),
    Stage.MODELED: frozenset({Stage.INFERRED}),
    Stage.INFERRED: frozenset({Stage.DECIDED}),
    Stage.UNDERPOWERED: frozenset(),
    Stage.DECIDED: frozenset(),
}


class StageOrderError(RuntimeError):
    """Raised when a stage is entered from a state that does not allow it."""


class NotLockedError(RuntimeError):
    """Raised when run() is called before a pre-registration lock exists."""


class LockMismatchError(RuntimeError):
    """Raised when the current config hash does not match the lock file."""


class LockRefusedError(RuntimeError):
    """Raised when a lock already records a different config and amend was not requested."""


class FeasibilityStudy(ABC):
    """Base class for one pre-registered feasibility study.

    Subclasses implement ``define_config`` and ``load_data``. They call
    ``lock_preregistration`` as a separate, explicit step, then ``run``.
    There is no public method that fits a model before the EPV gate.
    """

    def __init__(
        self,
        lock_dir: str | Path | None = None,
        *,
        study_path: str | Path | None = None,
    ) -> None:
        self.study_path = Path(study_path).resolve() if study_path is not None else None
        if lock_dir is not None:
            self.lock_dir = Path(lock_dir)
        elif self.study_path is not None:
            self.lock_dir = self.study_path.parent / "locks"
        else:
            self.lock_dir = Path("locks")
        self._stage = Stage.INIT
        self.lock_action = ""

    @property
    def stage(self) -> Stage:
        return self._stage

    @abstractmethod
    def define_config(self) -> StudyConfig:
        """Return the study specification. Must be deterministic across calls."""

    @abstractmethod
    def load_data(self, path_map: dict[str, Path]) -> dict[str, pd.DataFrame]:
        """Read each named source. The engine checks keys and the identifier."""

    @property
    def inference_method(self):
        """Resolved from the config name so run() does not hardcode a test."""
        from feasibility_framework.registries import get_inference_method

        return get_inference_method(self.define_config().inference_method)

    def lock_preregistration(self, *, amend: bool = False) -> Path:
        """Write a dated, hashed lock. This is intentionally not part of run().

        An existing lock with the same hash is left untouched. A different
        hash is refused unless ``amend`` is true, in which case the old file
        is moved under ``locks/history/`` and a new lock is written.
        """
        if self._stage not in {Stage.INIT, Stage.LOCKED}:
            raise StageOrderError(
                "Pre-registration can only be locked before a run has started."
            )
        config = self.define_config()
        config.validate()
        path = self._lock_path(config)
        current = config_hash(config)
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("config_hash") == current and previous.get("study_id") == config.study_id:
                self.lock_action = "unchanged"
                if self._stage is Stage.INIT:
                    self._begin(Stage.LOCKED)
                return path
            if not amend:
                raise LockRefusedError(
                    f"Lock at {path} records a different config hash. "
                    f"Re-locking requires: feasibility lock {self._study_file()} --amend"
                )
            self._archive_lock(path, previous)
            self.lock_action = "amended"
        else:
            self.lock_action = "created"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "study_id": config.study_id,
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": current,
            "config": _lock_body(config),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self._stage is Stage.INIT:
            self._begin(Stage.LOCKED)
        return path

    def run(self, path_map: dict[str, Path] | None = None) -> FeasibilityResult:
        """Execute the fixed stage order and return one FeasibilityResult."""
        from feasibility_framework.dimensionality import (
            ClearanceError,
            assess_dimensionality,
            issue_clearance,
        )
        from feasibility_framework.features import prepare_features
        from feasibility_framework.gate import decide_gate
        from feasibility_framework.leakage_guard import assert_no_leakage
        from feasibility_framework.linkage import link_sources
        from feasibility_framework.modeling import fit_comparison, holdout_indices

        config = self.define_config()
        config.validate()
        self._accept_lock(config)
        frames = self._ingest(config, path_map or {})
        self._begin(Stage.INGESTED)
        linked, linkage = link_sources(
            frames,
            id_key=config.id_key,
            primary=config.primary_source,
        )
        self._begin(Stage.LINKED)
        assert_no_leakage(config, config.columns_for(config.full_blocks))
        self._begin(Stage.LEAKAGE_CHECKED)
        prepared = prepare_features(config, linked)
        self._begin(Stage.FEATURES_BUILT)
        train_idx, test_idx = holdout_indices(
            prepared.y,
            test_size=config.holdout_fraction,
            seed=config.random_seed,
        )
        dimensionality = assess_dimensionality(config, prepared, train_idx)
        self._begin(Stage.EPV_CHECKED)
        lock_meta = self._lock_metadata(config)
        thresholds = _gate_thresholds(config)
        if not dimensionality.clears_floor:
            self._begin(Stage.UNDERPOWERED)
            return FeasibilityResult.create(
                study_id=config.study_id,
                config_hash=config_hash(config),
                gate=GateVerdict.UNDERPOWERED,
                linkage=linkage,
                dimensionality=dimensionality,
                gate_components={
                    "epv": {
                        "status": "FAIL",
                        "reasons": [
                            f"EPV {dimensionality.epv:.4f} is below the "
                            f"pre-registered floor {config.epv_floor:.4f}. "
                            "Modeling was not run."
                        ],
                    }
                },
                gate_thresholds=thresholds,
                lock_path=lock_meta["lock_path"],
                lock_config_hash=lock_meta["lock_config_hash"],
                lock_locked_at=lock_meta["lock_locked_at"],
                stop_reason=(
                    f"EPV {dimensionality.epv:.4f} < floor {config.epv_floor:.4f}. "
                    "Stopped before modeling."
                ),
            )
        try:
            clearance = issue_clearance(dimensionality)
        except ClearanceError as exc:
            raise StageOrderError(str(exc)) from exc
        self._enter_modeling()
        comparison = fit_comparison(clearance, config, prepared, train_idx, test_idx)
        self._begin(Stage.INFERRED)
        inference = self.inference_method.compare(comparison.folds, comparison.contrasts)
        n_events = int(prepared.y.sum())
        prevalence = float(prepared.y.mean())
        present_fraction = float(prepared.audit["usable"].mean())
        blocks_present = set(prepared.audit.loc[prepared.audit["usable"], "block"])
        verdict, components = decide_gate(
            config,
            n_rows=int(len(prepared.y)),
            n_events=n_events,
            prevalence=prevalence,
            blocks_present=blocks_present,
            present_fraction=present_fraction,
            linkage=linkage,
            dimensionality=dimensionality,
            holdout=comparison.holdout,
            leakage_ok=True,
        )
        self._begin(Stage.DECIDED)
        return FeasibilityResult.create(
            study_id=config.study_id,
            config_hash=config_hash(config),
            gate=verdict,
            linkage=linkage,
            dimensionality=dimensionality,
            gate_components=components,
            model_metrics=comparison.holdout,
            inference={"method": inference.method, "rows": inference.rows},
            gate_thresholds=thresholds,
            lock_path=lock_meta["lock_path"],
            lock_config_hash=lock_meta["lock_config_hash"],
            lock_locked_at=lock_meta["lock_locked_at"],
        )

    def _enter_modeling(self) -> None:
        """Move to the modeling stage. Refuses unless the EPV gate just passed."""
        self._begin(Stage.MODELED)

    def _begin(self, target: Stage) -> None:
        allowed = _TRANSITIONS[self._stage]
        if target not in allowed:
            allowed_names = ", ".join(sorted(stage.value for stage in allowed)) or "none"
            raise StageOrderError(
                f"Cannot enter {target.value} from {self._stage.value}. "
                f"Allowed next: {allowed_names}."
            )
        self._stage = target

    def _accept_lock(self, config: StudyConfig) -> None:
        path = self._lock_path(config)
        if not path.is_file():
            raise NotLockedError(
                f"No lock file at {path}. Run: feasibility lock {self._study_file()}"
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        current = config_hash(config)
        if record.get("config_hash") != current or record.get("study_id") != config.study_id:
            raise LockMismatchError(
                "Locked configuration does not match define_config(). "
                "Call lock_preregistration() again to amend it explicitly."
            )
        if self._stage is Stage.INIT:
            self._begin(Stage.LOCKED)
        elif self._stage is not Stage.LOCKED:
            raise StageOrderError(
                f"run() must start from a locked study, not {self._stage.value}."
            )

    def _ingest(self, config: StudyConfig, path_map: dict[str, Path]) -> dict[str, pd.DataFrame]:
        import pandas as pd

        from feasibility_framework.modeling import ModelingError

        unknown = [key for key in path_map if key not in config.sources]
        if unknown:
            raise ModelingError(
                f"path_map contains sources that are not in the config: {unknown}."
            )
        frames = self.load_data(path_map)
        if set(frames) != set(config.sources):
            raise ModelingError(
                f"load_data returned {sorted(frames)}, expected {sorted(config.sources)}."
            )
        for name, frame in frames.items():
            if not isinstance(frame, pd.DataFrame):
                raise ModelingError(f"Source {name!r} did not return a DataFrame.")
            if config.id_key not in frame.columns:
                raise ModelingError(f"Source {name!r} is missing id_key {config.id_key!r}.")
        if config.outcome not in frames[config.primary_source].columns:
            raise ModelingError(
                f"Primary source is missing outcome column {config.outcome!r}."
            )
        return frames

    def _study_file(self) -> str:
        if self.study_path is None:
            return "<study.py>"
        return str(self.study_path)

    def _archive_lock(self, path: Path, previous: dict) -> Path:
        history = self.lock_dir / "history"
        history.mkdir(parents=True, exist_ok=True)
        locked_at = str(previous.get("locked_at") or "unknown-time")
        safe_time = locked_at.replace(":", "-")
        dest = history / f"{safe_time}_{path.name}"
        if dest.exists():
            raise LockRefusedError(
                f"Refusing to overwrite archived lock {dest}."
            )
        path.replace(dest)
        return dest

    def _lock_metadata(self, config: StudyConfig) -> dict[str, str]:
        path = self._lock_path(config)
        record = json.loads(path.read_text(encoding="utf-8"))
        if self.study_path is not None:
            try:
                relative = str(path.resolve().relative_to(self.study_path.parent.resolve()))
            except ValueError:
                relative = path.name
        else:
            relative = str(path)
        return {
            "lock_path": relative.replace("\\", "/"),
            "lock_config_hash": str(record["config_hash"]),
            "lock_locked_at": str(record["locked_at"]),
        }

    def _lock_path(self, config: StudyConfig) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in config.study_id)
        return self.lock_dir / f"{safe}.lock.json"


def _gate_thresholds(config: StudyConfig) -> dict:
    return {
        "epv_floor": config.epv_floor,
        "epv_basis": config.epv_basis,
        "min_usable_n": config.min_usable_n,
        "min_outcome_events": config.min_outcome_events,
        "min_outcome_prevalence": config.min_outcome_prevalence,
        "min_baseline_roc_auc": config.min_baseline_roc_auc,
        "min_incremental_roc_lift": config.min_incremental_roc_lift,
        "min_incremental_pr_lift": config.min_incremental_pr_lift,
        "min_present_fraction": config.min_present_fraction,
    }


def _lock_body(config: StudyConfig) -> dict:
    from feasibility_framework.config import canonical_config

    return canonical_config(config)
