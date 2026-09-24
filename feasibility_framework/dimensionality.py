"""Events-per-variable diagnostic. Counts design columns; does not fit a classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feasibility_framework.config import StudyConfig
from feasibility_framework.result import DimensionalityReport

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from feasibility_framework.features import PreparedFeatures


class DimensionalityError(RuntimeError):
    """Raised when the EPV diagnostic cannot be computed as specified."""


class ClearanceError(RuntimeError):
    """Raised when modeling is requested without an engine-issued EPV clearance."""


_CLEARANCE_SECRET = object()


class EpvClearance:
    """Token issued only when the pre-registered EPV floor is met.

    Constructing this with a failing report, or without the engine secret,
    raises. Modeling functions require an instance of this class.
    """

    def __init__(self, report: DimensionalityReport, secret: object) -> None:
        if secret is not _CLEARANCE_SECRET:
            raise ClearanceError(
                "EPV clearance can only be issued by the dimensionality gate."
            )
        if not report.clears_floor:
            raise ClearanceError(
                f"EPV {report.epv:.4f} is below the pre-registered floor "
                f"{report.epv_floor:.4f}. Modeling clearance refused."
            )
        self.report = report


def issue_clearance(report: DimensionalityReport) -> EpvClearance:
    return EpvClearance(report, _CLEARANCE_SECRET)


def _mean_cv_counts(
    y_train: pd.Series,
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> tuple[float, float]:
    import numpy as np
    from sklearn.model_selection import RepeatedStratifiedKFold

    if y_train.nunique() < 2:
        raise DimensionalityError("Training outcome has a single class.")
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    y_values = np.asarray(y_train)
    events: list[int] = []
    sizes: list[int] = []
    x_dummy = np.zeros((len(y_values), 1))
    for train_idx, _val_idx in splitter.split(x_dummy, y_values):
        sizes.append(int(len(train_idx)))
        events.append(int(y_values[train_idx].sum()))
    return float(np.mean(events)), float(np.mean(sizes))


def assess_dimensionality(
    config: StudyConfig,
    prepared: PreparedFeatures,
    train_idx: np.ndarray,
) -> DimensionalityReport:
    """EPV = mean repeated-CV training events / full-model design columns.

    The preprocessor is fit on the outer training portion only. No classifier
    is fit here.
    """
    from feasibility_framework.modeling import design_column_count

    if len(train_idx) < config.cv_splits:
        raise DimensionalityError(
            "Training portion is smaller than cv_splits. Refusing to invent an EPV."
        )

    y_train = prepared.y.iloc[train_idx]
    x_train = prepared.x.iloc[train_idx]
    full_columns = list(prepared.x.columns)
    baseline_names = set(config.columns_for(config.baseline_blocks))
    baseline_columns = [
        name
        for name in full_columns
        if name in baseline_names or _collapsed_baseline(config, prepared, name)
    ]
    n_full = design_column_count(x_train, full_columns, prepared.column_treatment)
    n_baseline = design_column_count(
        x_train, baseline_columns, prepared.column_treatment
    )
    if n_full <= 0:
        raise DimensionalityError("Full model has zero design columns.")
    mean_events, mean_n = _mean_cv_counts(
        y_train,
        n_splits=config.cv_splits,
        n_repeats=config.cv_repeats,
        seed=config.cv_seed,
    )
    epv = mean_events / n_full
    return DimensionalityReport(
        n_design_columns=n_full,
        n_baseline_columns=n_baseline,
        mean_cv_train_n=mean_n,
        mean_cv_train_events=mean_events,
        epv=float(epv),
        epv_floor=float(config.epv_floor),
        clears_floor=bool(epv >= config.epv_floor),
        collapsed_blocks=prepared.collapsed_blocks,
    )


def _collapsed_baseline(config: StudyConfig, prepared: PreparedFeatures, column: str) -> bool:
    """A collapsed baseline block contributes its score column, not raw names."""
    sources = prepared.derived_from.get(column)
    if not sources:
        return False
    baseline = set(config.baseline_blocks)
    owners = {config.predictor_index()[name][0].name for name in sources}
    return owners <= baseline and bool(owners & baseline)
