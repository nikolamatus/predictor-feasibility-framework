"""Cutoff enforcement. Unknown timing and post-cutoff columns raise."""

from __future__ import annotations

from feasibility_framework.config import StudyConfig


class LeakageError(RuntimeError):
    """Raised when a column is not allowed in the predictor matrix."""


def assert_no_leakage(
    config: StudyConfig,
    columns: list[str],
    *,
    derived_from: dict[str, list[str]] | None = None,
) -> None:
    """Refuse any column that is not a confirmed pre-cutoff predictor.

    ``derived_from`` maps an engine-built composite column to the source
    predictors it was collapsed from. Those sources are checked; the
    composite name itself is not accepted unless every source is allowed.
    """
    derived_from = derived_from or {}
    index = config.predictor_index()
    for name in columns:
        if name in derived_from:
            sources = derived_from[name]
            if not sources:
                raise LeakageError(f"Derived column {name} has no source predictors.")
            assert_no_leakage(config, sources)
            continue
        if name == config.outcome:
            raise LeakageError(f"{name} is the outcome and cannot enter X.")
        if name == config.id_key:
            raise LeakageError(f"{name} is the identifier and cannot enter X.")
        found = index.get(name)
        if found is None:
            raise LeakageError(
                f"{name} has no predictor specification. Failing closed."
            )
        block, predictor = found
        if not block.confirmed or not predictor.confirmed:
            raise LeakageError(f"{name} is unconfirmed and cannot enter X.")
        if predictor.time_invariant:
            continue
        if predictor.availability is None:
            raise LeakageError(
                f"{name} has unknown temporal availability and cannot enter X."
            )
        if predictor.availability > config.cutoff:
            raise LeakageError(
                f"{name} availability {predictor.availability.isoformat()} is after "
                f"cutoff {config.cutoff.isoformat()}."
            )
