"""Inference-method registry. Built-in: Nadeau-Bengio corrected t-test with Holm."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

_METHODS: dict[str, type] = {}


class RegistryError(KeyError):
    """Raised when an inference method name is missing or already taken."""


class InferenceInputError(ValueError):
    """Raised when fold-level contrasts cannot support the requested test."""


@dataclass(frozen=True)
class ContrastSpec:
    contrast_id: str
    family: str
    label: str
    column: str
    metric: str
    adjust: bool


@dataclass(frozen=True)
class NBResult:
    n: int
    mean: float
    sample_variance: float
    test_train_ratio: float
    corrected_variance: float
    corrected_se: float
    t_statistic: float
    df: int
    p_value: float


@dataclass
class InferenceResult:
    method: str
    rows: list[dict]


def register_inference(name: str):
    key = name.strip().lower()
    if not key:
        raise RegistryError("Inference method name is empty.")

    def decorator(cls: type) -> type:
        existing = _METHODS.get(key)
        if existing is not None and existing is not cls:
            raise RegistryError(f"Inference method {key!r} is already registered.")
        _METHODS[key] = cls
        return cls

    return decorator


def get_inference_method(name: str):
    key = name.strip().lower()
    try:
        cls = _METHODS[key]
    except KeyError as exc:
        known = ", ".join(sorted(_METHODS)) or "(none)"
        raise RegistryError(
            f"No inference method registered as {key!r}. Known: {known}."
        ) from exc
    return cls()


def registered_inference_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))


def unregister_inference(name: str) -> None:
    _METHODS.pop(name.strip().lower(), None)


def nadeau_bengio_ttest(deltas: np.ndarray, test_train_ratio: float) -> NBResult:
    """Nadeau & Bengio (2003) corrected resampled t-test.

    Var = (1/n + n_test/n_train) * s^2, with s^2 the unbiased variance of
    the paired fold differences. Two-sided p from Student's t on n - 1 df.
    """
    import numpy as np
    from scipy import stats

    d = np.asarray(deltas, dtype=float)
    if d.size < 2:
        raise InferenceInputError("Need at least two paired differences.")
    if not np.isfinite(d).all():
        raise InferenceInputError("Fold differences must be finite.")
    if test_train_ratio <= 0:
        raise InferenceInputError("test/train ratio must be positive.")
    n = int(d.size)
    mean = float(d.mean())
    s2 = float(d.var(ddof=1))
    var_nb = (1.0 / n + test_train_ratio) * s2
    se = float(np.sqrt(var_nb))
    t_stat = mean / se if se > 0 else float("inf")
    df = n - 1
    p = float(2.0 * stats.t.sf(abs(t_stat), df))
    return NBResult(
        n=n,
        mean=mean,
        sample_variance=s2,
        test_train_ratio=float(test_train_ratio),
        corrected_variance=float(var_nb),
        corrected_se=se,
        t_statistic=float(t_stat),
        df=df,
        p_value=p,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    import numpy as np

    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return []
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise InferenceInputError("Holm adjustment requires finite p-values in [0, 1].")
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * float(p[idx]))
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


def _test_train_ratio(folds: pd.DataFrame) -> float:
    if "n_cv_val" not in folds.columns or "n_cv_train" not in folds.columns:
        raise InferenceInputError("Fold table is missing n_cv_val or n_cv_train.")
    train = folds["n_cv_train"].astype(float)
    if (train <= 0).any():
        raise InferenceInputError("n_cv_train must be positive.")
    return float((folds["n_cv_val"].astype(float) / train).mean())


@register_inference("nadeau_bengio_holm")
class NadeauBengioHolm:
    """Default inference method. Holm is applied only inside families that request it."""

    name = "nadeau_bengio_holm"

    def compare(self, folds: pd.DataFrame, contrasts: Sequence[ContrastSpec]) -> InferenceResult:
        if folds.empty:
            raise InferenceInputError("Fold table is empty.")
        if not contrasts:
            raise InferenceInputError("No contrasts were supplied.")
        ratio = _test_train_ratio(folds)
        pending: list[dict] = []
        for spec in contrasts:
            if spec.column not in folds.columns:
                raise InferenceInputError(f"Fold table is missing contrast column {spec.column}.")
            nb = nadeau_bengio_ttest(folds[spec.column].to_numpy(dtype=float), ratio)
            pending.append(
                {
                    "contrast_id": spec.contrast_id,
                    "family": spec.family,
                    "contrast": spec.label,
                    "metric": spec.metric,
                    "column": spec.column,
                    "mean_delta": nb.mean,
                    "corrected_se": nb.corrected_se,
                    "t_statistic": nb.t_statistic,
                    "df": nb.df,
                    "raw_p_value": nb.p_value,
                    "holm_adjusted_p_value": None,
                    "adjust": spec.adjust,
                    "test_train_ratio": nb.test_train_ratio,
                    "n_folds": nb.n,
                }
            )
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in pending:
            if row["adjust"]:
                groups.setdefault((row["family"], row["metric"]), []).append(row)
        for rows in groups.values():
            adjusted = holm_adjust([row["raw_p_value"] for row in rows])
            for row, p_adj in zip(rows, adjusted):
                row["holm_adjusted_p_value"] = p_adj
        for row in pending:
            row.pop("adjust")
        return InferenceResult(method=self.name, rows=pending)
