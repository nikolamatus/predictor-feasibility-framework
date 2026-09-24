"""Linkage on invented multi-file tables."""

import pandas as pd
import pytest

from feasibility_framework.linkage import LinkageError, link_sources


def test_clean_join():
    primary = pd.DataFrame({"person_id": [1, 2], "event": [0, 1]})
    extra = pd.DataFrame({"person_id": [1, 2], "flag": [5, 6]})
    merged, report = link_sources(
        {"cohort": primary, "aux": extra},
        id_key="person_id",
        primary="cohort",
    )
    assert len(merged) == 2
    assert list(merged["flag"]) == [5, 6]
    assert report.sources["aux"]["n_matched"] == 2
    assert report.sources["aux"]["n_primary_only"] == 0


def test_missing_key_raises():
    primary = pd.DataFrame({"event": [0, 1]})
    with pytest.raises(LinkageError, match="missing"):
        link_sources({"cohort": primary}, id_key="person_id", primary="cohort")


def test_duplicate_id_raises():
    primary = pd.DataFrame({"person_id": [1, 1], "event": [0, 1]})
    with pytest.raises(LinkageError, match="duplicate"):
        link_sources({"cohort": primary}, id_key="person_id", primary="cohort")


def test_partial_overlap_is_reported_and_kept():
    primary = pd.DataFrame({"person_id": [1, 2, 3], "event": [0, 1, 0]})
    extra = pd.DataFrame({"person_id": [2, 3, 9], "flag": [4, 5, 6]})
    merged, report = link_sources(
        {"cohort": primary, "aux": extra},
        id_key="person_id",
        primary="cohort",
    )
    assert len(merged) == 3
    assert merged.loc[merged["person_id"] == 1, "flag"].isna().all()
    assert report.sources["aux"]["n_matched"] == 2
    assert report.sources["aux"]["n_primary_only"] == 1
    assert report.sources["aux"]["n_source_only"] == 1
