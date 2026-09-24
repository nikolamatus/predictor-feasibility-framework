"""Composite-strategy registry. Built-ins: sum-score and mean-score."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

Composite = Callable[[pd.DataFrame, list[str]], pd.Series]

_STRATEGIES: dict[str, Composite] = {}


class RegistryError(KeyError):
    """Raised when a composite strategy name is missing or already taken."""


def register_composite(name: str) -> Callable[[Composite], Composite]:
    key = name.strip().lower()
    if not key:
        raise RegistryError("Composite strategy name is empty.")

    def decorator(fn: Composite) -> Composite:
        existing = _STRATEGIES.get(key)
        if existing is not None and existing is not fn:
            raise RegistryError(f"Composite strategy {key!r} is already registered.")
        _STRATEGIES[key] = fn
        return fn

    return decorator


def get_composite(name: str) -> Composite:
    key = name.strip().lower()
    try:
        return _STRATEGIES[key]
    except KeyError as exc:
        known = ", ".join(sorted(_STRATEGIES)) or "(none)"
        raise RegistryError(
            f"No composite strategy registered as {key!r}. Known: {known}."
        ) from exc


def registered_composites() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def unregister_composite(name: str) -> None:
    _STRATEGIES.pop(name.strip().lower(), None)


def _numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        raise RegistryError("Composite strategy received no columns.")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise RegistryError(f"Composite columns are missing from the frame: {missing}.")
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return numeric


@register_composite("sum-score")
def sum_score(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    numeric = _numeric_frame(frame, columns)
    score = numeric.sum(axis=1, min_count=1)
    score.name = "sum_score"
    return score


@register_composite("mean-score")
def mean_score(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    numeric = _numeric_frame(frame, columns)
    score = numeric.mean(axis=1)
    score.name = "mean_score"
    return score
