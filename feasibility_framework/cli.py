"""Command line: validate, lock, run, or print a saved result."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

from feasibility_framework.config import ConfigError
from feasibility_framework.result import FeasibilityResult, ResultError
from feasibility_framework.study import (
    FeasibilityStudy,
    LockMismatchError,
    LockRefusedError,
    NotLockedError,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feasibility")
    sub = parser.add_subparsers(dest="command", required=True)

    validate_parser = sub.add_parser("validate", help="Check config only. Does not read data.")
    validate_parser.add_argument("study", type=Path)

    lock_parser = sub.add_parser("lock", help="Write the pre-registration lock next to the study file.")
    lock_parser.add_argument("study", type=Path)
    lock_parser.add_argument(
        "--amend",
        action="store_true",
        help="Archive the existing lock and write a new one after a config change.",
    )
    lock_parser.add_argument("--lock-dir", type=Path, default=None)

    run_parser = sub.add_parser("run", help="Run a locked FeasibilityStudy subclass.")
    run_parser.add_argument("study", type=Path)
    run_parser.add_argument("--lock-dir", type=Path, default=None)
    run_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Result JSON path. Default: <study dir>/results/<stem>.<hash8>.result.json",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing result file at the destination path.",
    )

    report_parser = sub.add_parser("report", help="Print a saved FeasibilityResult.")
    report_parser.add_argument("result", type=Path)

    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.study)
    if args.command == "lock":
        return _lock(args.study, args.lock_dir, amend=args.amend)
    if args.command == "run":
        return _run(args.study, args.lock_dir, args.output, force=args.force)
    if args.command == "report":
        return _report(args.result)
    parser.error(f"Unknown command {args.command}")
    return 2


def load_study_class(path: Path) -> type[FeasibilityStudy]:
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    module = types.ModuleType(f"user_study_{digest}")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    found = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, FeasibilityStudy)
        and obj is not FeasibilityStudy
    ]
    if len(found) != 1:
        raise ImportError(
            f"{path} must define exactly one FeasibilityStudy subclass; found {len(found)}."
        )
    return found[0]


def _validate(path: Path) -> int:
    try:
        study_cls = load_study_class(path)
        config = study_cls().define_config()
        config.validate()
    except (ConfigError, ImportError, FileNotFoundError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"config ok: {config.study_id}")
    return 0


def _open_study(path: Path, lock_dir: Path | None) -> FeasibilityStudy:
    resolved = path.resolve()
    study_cls = load_study_class(resolved)
    if lock_dir is None:
        return study_cls(study_path=resolved)
    return study_cls(lock_dir=lock_dir, study_path=resolved)


def _lock(path: Path, lock_dir: Path | None, *, amend: bool) -> int:
    try:
        study = _open_study(path, lock_dir)
        written = study.lock_preregistration(amend=amend)
    except (LockRefusedError, ConfigError, ImportError, FileNotFoundError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    record = json.loads(written.read_text(encoding="utf-8"))
    if study.lock_action == "unchanged":
        print("already locked, hash unchanged")
    print(f"path: {written}")
    print(f"hash: {record['config_hash']}")
    print(f"locked at: {record['locked_at']}")
    return 0


def _default_result_path(study_path: Path, config_hash_value: str) -> Path:
    resolved = study_path.resolve()
    return (
        resolved.parent
        / "results"
        / f"{resolved.stem}.{config_hash_value[:8]}.result.json"
    )


class ResultExistsError(RuntimeError):
    """Raised when a result path already exists and --force was not passed."""


def _run(
    path: Path,
    lock_dir: Path | None,
    output: Path | None,
    *,
    force: bool = False,
) -> int:
    try:
        study = _open_study(path, lock_dir)
        result = study.run()
    except (NotLockedError, LockMismatchError, ConfigError, ImportError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    destination = (
        output if output is not None else _default_result_path(path, result.config_hash)
    )
    if destination.exists() and not force:
        print(
            f"Result file already exists at {destination}. Pass --force to overwrite.",
            file=sys.stderr,
        )
        return 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_json(destination)
    print(f"gate: {result.gate.value}")
    print(f"wrote {destination}")
    return 0


def _report(path: Path) -> int:
    try:
        result = FeasibilityResult.from_json(path)
    except (OSError, ResultError, KeyError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"study: {result.study_id}")
    print(f"result: {result.result_id}")
    print(f"gate: {result.gate.value}")
    print(f"config_hash: {result.config_hash}")
    if result.lock_path:
        print(f"lock_path: {result.lock_path}")
    if result.lock_config_hash:
        print(f"lock_config_hash: {result.lock_config_hash}")
    if result.lock_locked_at:
        print(f"lock_locked_at: {result.lock_locked_at}")
    if result.superseded_by:
        print(f"superseded_by: {result.superseded_by}")
    if result.stop_reason:
        print(f"stop_reason: {result.stop_reason}")
    if result.dimensionality is not None:
        dim = result.dimensionality
        print(
            f"epv: {dim.epv:.4f} (floor {dim.epv_floor:.4f}, "
            f"basis={dim.epv_basis}, "
            f"min_events={dim.min_cv_train_events:.4f}, "
            f"mean_events={dim.mean_cv_train_events:.4f}, "
            f"columns {dim.n_design_columns}, clears={dim.clears_floor})"
        )
    if result.gate_thresholds:
        print(f"gate_thresholds: {json.dumps(result.gate_thresholds, sort_keys=True)}")
    if result.linkage is not None:
        print(
            f"linkage: n_primary={result.linkage.n_primary} "
            f"sources={list(result.linkage.sources)}"
        )
    if result.model_metrics:
        metrics = result.model_metrics
        print(
            "holdout: "
            f"baseline_roc={metrics.get('baseline_roc_auc')} "
            f"full_roc={metrics.get('full_roc_auc')} "
            f"roc_lift={metrics.get('roc_lift')}"
        )
    if result.inference:
        print(f"inference_method: {result.inference.get('method')}")
        for row in result.inference.get("rows") or []:
            print(
                "inference: "
                f"{row.get('contrast_id')} {row.get('metric')} "
                f"raw_p={row.get('raw_p_value')} "
                f"holm_p={row.get('holm_adjusted_p_value')}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
