"""Plugin registration for readers, composites, and inference methods."""

from pathlib import Path

import pandas as pd
import pytest

from feasibility_framework.registries import (
    ContrastSpec,
    get_composite,
    get_inference_method,
    get_reader,
    register_composite,
    register_inference,
    register_reader,
    unregister_composite,
    unregister_inference,
    unregister_reader,
)
from feasibility_framework.registries.composites import RegistryError as CompositeRegistryError
from feasibility_framework.registries.inference import InferenceResult
from feasibility_framework.registries.readers import RegistryError as ReaderRegistryError


def test_builtin_csv_and_parquet_roundtrip(tmp_path: Path):
    frame = pd.DataFrame({"person_id": [1, 2], "value": [0.1, 0.2]})
    csv_path = tmp_path / "synthetic.csv"
    parquet_path = tmp_path / "synthetic.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    loaded_csv = get_reader("csv")(csv_path)
    loaded_parquet = get_reader("parquet")(parquet_path)
    assert list(loaded_csv["person_id"]) == [1, 2]
    assert list(loaded_parquet["value"]) == [0.1, 0.2]


def test_custom_reader_can_be_registered(tmp_path: Path):
    @register_reader("toyfmt")
    def read_toy(path: Path) -> pd.DataFrame:
        text = path.read_text(encoding="utf-8")
        return pd.DataFrame({"person_id": [int(text)]})

    path = tmp_path / "one.toyfmt"
    path.write_text("7", encoding="utf-8")
    try:
        loaded = get_reader("toyfmt")(path)
        assert loaded.loc[0, "person_id"] == 7
    finally:
        unregister_reader("toyfmt")
    with pytest.raises(ReaderRegistryError):
        get_reader("toyfmt")


def test_sum_and_mean_scores_and_a_custom_strategy():
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert list(get_composite("sum-score")(frame, ["a", "b"])) == [4, 6]
    assert list(get_composite("mean-score")(frame, ["a", "b"])) == [2, 3]

    @register_composite("count-of-endorsed")
    def count_of_endorsed(frame, columns):
        return (frame.loc[:, columns] > 0).sum(axis=1)

    try:
        scores = get_composite("count-of-endorsed")(frame, ["a", "b"])
        assert list(scores) == [2, 2]
    finally:
        unregister_composite("count-of-endorsed")


def test_duplicate_builtin_registration_fails():
    with pytest.raises(CompositeRegistryError):
        register_composite("sum-score")(lambda frame, columns: frame.iloc[:, 0])


def test_custom_inference_method_is_what_compare_calls():
    @register_inference("marker")
    class Marker:
        name = "marker"

        def compare(self, folds, contrasts):
            return InferenceResult(
                method=self.name,
                rows=[{"contrast_id": contrasts[0].contrast_id, "seen_rows": len(folds)}],
            )

    folds = pd.DataFrame({"n_cv_train": [8, 8], "n_cv_val": [2, 2], "delta": [0.1, 0.2]})
    contrasts = [
        ContrastSpec("A1", "A", "Full − baseline", "delta", "roc_auc", False)
    ]
    try:
        method = get_inference_method("marker")
        result = method.compare(folds, contrasts)
        assert result.method == "marker"
        assert result.rows[0]["seen_rows"] == 2
    finally:
        unregister_inference("marker")
