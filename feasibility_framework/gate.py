"""Decision gate. Floors are read from the locked study config."""

from __future__ import annotations

from typing import Any

from feasibility_framework.config import StudyConfig
from feasibility_framework.result import DimensionalityReport, GateVerdict, LinkageReport


def decide_gate(
    config: StudyConfig,
    *,
    n_rows: int,
    n_events: int,
    prevalence: float,
    blocks_present: set[str],
    present_fraction: float,
    linkage: LinkageReport,
    dimensionality: DimensionalityReport,
    holdout: dict[str, Any] | None,
    leakage_ok: bool,
) -> tuple[GateVerdict, dict[str, Any]]:
    """Return UNDERPOWERED before any performance rule if EPV missed the floor."""
    if not dimensionality.clears_floor:
        return GateVerdict.UNDERPOWERED, {
            "epv": {
                "status": "FAIL",
                "reasons": [
                    f"EPV {dimensionality.epv:.4f} is below the pre-registered "
                    f"floor {config.epv_floor:.4f}. Modeling was not accepted."
                ],
            }
        }

    data_reasons: list[str] = []
    if n_rows < config.min_usable_n:
        data_reasons.append(
            f"Analytic N ({n_rows}) is below min_usable_n={config.min_usable_n}."
        )
    if n_events < config.min_outcome_events:
        data_reasons.append(
            f"Events ({n_events}) are below min_outcome_events={config.min_outcome_events}."
        )
    if prevalence < config.min_outcome_prevalence:
        data_reasons.append(
            f"Prevalence ({prevalence:.4f}) is below "
            f"min_outcome_prevalence={config.min_outcome_prevalence}."
        )
    required = set(config.full_blocks)
    missing_blocks = sorted(required - blocks_present)
    if missing_blocks:
        data_reasons.append(f"Required blocks missing from X: {missing_blocks}.")
    if present_fraction < config.min_present_fraction:
        data_reasons.append(
            f"Usable predictor fraction {present_fraction:.3f} is below "
            f"min_present_fraction={config.min_present_fraction}."
        )
    data = _component(
        "FAIL" if data_reasons else "PASS",
        data_reasons or ["Sample size, events, prevalence, and blocks clear the floors."],
    )

    baseline_reasons: list[str] = []
    baseline_roc = None if holdout is None else holdout.get("baseline_roc_auc")
    if baseline_roc is None or float(baseline_roc) < config.min_baseline_roc_auc:
        baseline_reasons.append(
            f"Baseline ROC-AUC ({baseline_roc}) is below "
            f"min_baseline_roc_auc={config.min_baseline_roc_auc}."
        )
    baseline = _component(
        "FAIL" if baseline_reasons else "PASS",
        baseline_reasons
        or [f"Baseline ROC-AUC ({float(baseline_roc):.3f}) clears the floor."],
    )

    incr_reasons: list[str] = []
    roc_lift = None if holdout is None else holdout.get("roc_lift")
    pr_lift = None if holdout is None else holdout.get("pr_lift")
    if roc_lift is None or pr_lift is None:
        incr_reasons.append("Incremental value cannot be assessed because holdout metrics are missing.")
        incremental = _component("INSUFFICIENT", incr_reasons)
    else:
        if float(roc_lift) < config.min_incremental_roc_lift:
            incr_reasons.append(
                f"ROC-AUC lift ({float(roc_lift):.4f}) is below "
                f"min_incremental_roc_lift={config.min_incremental_roc_lift}."
            )
        if float(pr_lift) < config.min_incremental_pr_lift:
            incr_reasons.append(
                f"PR-AUC lift ({float(pr_lift):.4f}) is below "
                f"min_incremental_pr_lift={config.min_incremental_pr_lift}."
            )
        incremental = _component(
            "FAIL" if incr_reasons else "PASS",
            incr_reasons
            or [
                f"Full model improves ROC-AUC by {float(roc_lift):.4f} and "
                f"PR-AUC by {float(pr_lift):.4f} over the baseline."
            ],
        )

    ready_reasons: list[str] = []
    if not leakage_ok:
        ready_reasons.append("Leakage guard did not pass.")
    if linkage.n_duplicate_ids != 0:
        ready_reasons.append("Identifier duplicates were recorded.")
    if holdout is None:
        ready_reasons.append("Holdout metrics were not produced.")
    readiness = _component(
        "FAIL" if ready_reasons else "PASS",
        ready_reasons or ["Readiness checks passed."],
    )

    components = {
        "data_adequacy": data,
        "baseline_signal": baseline,
        "incremental_value": incremental,
        "readiness": readiness,
    }
    if data["status"] == "FAIL" or readiness["status"] == "FAIL":
        verdict = GateVerdict.RED
    elif incremental["status"] == "INSUFFICIENT" or baseline["status"] == "FAIL" or incremental["status"] == "FAIL":
        verdict = GateVerdict.YELLOW
    else:
        verdict = GateVerdict.GREEN
    return verdict, components


def _component(status: str, reasons: list[str]) -> dict[str, Any]:
    return {"status": status, "reasons": reasons}
