"""Typed feasibility results and explicit supersession."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class GateVerdict(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    UNDERPOWERED = "UNDERPOWERED"


class ResultError(ValueError):
    """Raised when a result object is chained or loaded incorrectly."""


@dataclass
class LinkageReport:
    id_key: str
    n_primary: int
    n_duplicate_ids: int
    sources: dict[str, dict[str, float | int | None]]


@dataclass
class DimensionalityReport:
    n_design_columns: int
    n_baseline_columns: int
    mean_cv_train_n: float
    mean_cv_train_events: float
    min_cv_train_events: float
    epv: float
    epv_floor: float
    epv_basis: str
    clears_floor: bool
    collapsed_blocks: tuple[str, ...] = ()


@dataclass
class FeasibilityResult:
    """One engine run. Supersession is a field, not a note in a write-up."""

    result_id: str
    study_id: str
    config_hash: str
    created_at: str
    gate: GateVerdict
    linkage: LinkageReport | None = None
    dimensionality: DimensionalityReport | None = None
    gate_components: dict[str, Any] | None = None
    model_metrics: dict[str, Any] | None = None
    inference: dict[str, Any] | None = None
    stop_reason: str | None = None
    superseded_by: str | None = None
    synthetic: bool = False

    @classmethod
    def create(
        cls,
        *,
        study_id: str,
        config_hash: str,
        gate: GateVerdict,
        linkage: LinkageReport | None = None,
        dimensionality: DimensionalityReport | None = None,
        gate_components: dict[str, Any] | None = None,
        model_metrics: dict[str, Any] | None = None,
        inference: dict[str, Any] | None = None,
        stop_reason: str | None = None,
        synthetic: bool = False,
    ) -> FeasibilityResult:
        return cls(
            result_id=uuid.uuid4().hex,
            study_id=study_id,
            config_hash=config_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            gate=gate,
            linkage=linkage,
            dimensionality=dimensionality,
            gate_components=gate_components,
            model_metrics=model_metrics,
            inference=inference,
            stop_reason=stop_reason,
            synthetic=synthetic,
        )

    def supersedes(self, earlier: FeasibilityResult) -> None:
        """Record that this result replaces an earlier one from the same study."""
        if earlier is self or earlier.result_id == self.result_id:
            raise ResultError("A result cannot supersede itself.")
        if earlier.study_id != self.study_id:
            raise ResultError(
                "Supersession is only defined between results of the same study."
            )
        if earlier.superseded_by not in {None, self.result_id}:
            raise ResultError(
                f"Result {earlier.result_id} is already superseded by "
                f"{earlier.superseded_by}."
            )
        earlier.superseded_by = self.result_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate"] = self.gate.value
        return payload

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeasibilityResult:
        data = dict(payload)
        try:
            gate = GateVerdict(data["gate"])
        except (KeyError, ValueError) as exc:
            raise ResultError(f"Unreadable gate verdict: {data.get('gate')!r}.") from exc
        linkage = data.get("linkage")
        dimensionality = data.get("dimensionality")
        return cls(
            result_id=str(data["result_id"]),
            study_id=str(data["study_id"]),
            config_hash=str(data["config_hash"]),
            created_at=str(data["created_at"]),
            gate=gate,
            linkage=LinkageReport(**linkage) if linkage else None,
            dimensionality=(
                DimensionalityReport(
                    **{
                        **dimensionality,
                        "collapsed_blocks": tuple(dimensionality.get("collapsed_blocks") or ()),
                    }
                )
                if dimensionality
                else None
            ),
            gate_components=data.get("gate_components"),
            model_metrics=data.get("model_metrics"),
            inference=data.get("inference"),
            stop_reason=data.get("stop_reason"),
            superseded_by=data.get("superseded_by"),
            synthetic=bool(data.get("synthetic", False)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> FeasibilityResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ResultError("Result file must contain a JSON object.")
        return cls.from_dict(payload)
