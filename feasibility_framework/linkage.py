"""Join named source frames on a study's identifier. Fail closed on bad keys."""

from __future__ import annotations

import pandas as pd

from feasibility_framework.result import LinkageReport


class LinkageError(RuntimeError):
    """Raised when identifiers are missing, duplicated, or otherwise unusable."""


def _assert_key(frame: pd.DataFrame, key: str, label: str) -> None:
    if key not in frame.columns:
        raise LinkageError(f"{label}: linkage key {key!r} is missing.")
    if frame[key].isna().any():
        raise LinkageError(f"{label}: linkage key {key!r} has missing values.")
    n_dup = int(frame[key].duplicated().sum())
    if n_dup:
        raise LinkageError(
            f"{label}: {n_dup} duplicate {key!r} values; refusing to join."
        )


def link_sources(
    frames: dict[str, pd.DataFrame],
    *,
    id_key: str,
    primary: str,
) -> tuple[pd.DataFrame, LinkageReport]:
    """Left-join auxiliary sources onto the primary frame and report overlap.

    Partial overlap is reported and retained as missing values on the
    primary rows. Missing keys and duplicate identifiers raise.
    """
    if primary not in frames:
        raise LinkageError(f"Primary source {primary!r} was not loaded.")
    primary_frame = frames[primary]
    if not isinstance(primary_frame, pd.DataFrame):
        raise LinkageError(f"Primary source {primary!r} is not a DataFrame.")
    _assert_key(primary_frame, id_key, primary)
    merged = primary_frame.copy()
    source_reports: dict[str, dict[str, float | int | None]] = {}
    for name, frame in frames.items():
        if name == primary:
            continue
        if not isinstance(frame, pd.DataFrame):
            raise LinkageError(f"Source {name!r} is not a DataFrame.")
        _assert_key(frame, id_key, name)
        left_ids = set(merged[id_key].astype(str))
        right_ids = set(frame[id_key].astype(str))
        matched = left_ids & right_ids
        source_reports[name] = {
            "n_rows": int(len(frame)),
            "n_matched": int(len(matched)),
            "n_primary_only": int(len(left_ids - right_ids)),
            "n_source_only": int(len(right_ids - left_ids)),
            "match_rate_of_primary": (
                float(len(matched) / len(left_ids)) if left_ids else None
            ),
        }
        extra = [col for col in frame.columns if col != id_key]
        collision = [col for col in extra if col in merged.columns]
        if collision:
            raise LinkageError(
                f"Column collision between {primary!r} and {name!r}: {collision}."
            )
        merged = merged.merge(
            frame[[id_key, *extra]],
            on=id_key,
            how="left",
            validate="one_to_one",
        )
    report = LinkageReport(
        id_key=id_key,
        n_primary=int(len(primary_frame)),
        n_duplicate_ids=0,
        sources=source_reports,
    )
    return merged, report
