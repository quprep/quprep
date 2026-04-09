"""Unit tests for OpenMLIngester (openml library fully mocked)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quprep.ingest.openml_ingester import OpenMLIngester

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_oml_dataset(X: pd.DataFrame, y=None, name="test_ds", version=1, target="label"):
    """Return a mock OpenMLDataset whose get_data() returns (X, y, ...)."""
    mock_ds = MagicMock()
    mock_ds.name = name
    mock_ds.version = version
    mock_ds.default_target_attribute = target

    categorical_indicator = [False] * len(X.columns)
    attribute_names = list(X.columns)

    mock_ds.get_data.return_value = (X, y, categorical_indicator, attribute_names)
    return mock_ds


@contextmanager
def _mock_openml(mock_ds, datasets_df: pd.DataFrame | None = None):
    """Patch sys.modules['openml'] so get_dataset returns mock_ds."""
    fake_mod = MagicMock()
    fake_mod.datasets.get_dataset.return_value = mock_ds

    if datasets_df is not None:
        fake_mod.datasets.list_datasets.return_value = datasets_df

    _sentinel = object()
    old = sys.modules.get("openml", _sentinel)
    sys.modules["openml"] = fake_mod
    try:
        yield fake_mod
    finally:
        if old is _sentinel:
            sys.modules.pop("openml", None)
        else:
            sys.modules["openml"] = old


# ---------------------------------------------------------------------------
# ImportError
# ---------------------------------------------------------------------------

def test_import_error_when_openml_missing():
    old = sys.modules.get("openml")
    sys.modules["openml"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError, match="pip install quprep\\[openml\\]"):
            OpenMLIngester().load(61)
    finally:
        if old is None:
            sys.modules.pop("openml", None)
        else:
            sys.modules["openml"] = old


# ---------------------------------------------------------------------------
# Basic load by integer ID
# ---------------------------------------------------------------------------

def test_load_by_id_shape():
    X = pd.DataFrame({"sepal_length": [5.1, 4.9], "sepal_width": [3.5, 3.0]})
    mock_ds = _make_oml_dataset(X)

    with _mock_openml(mock_ds):
        ds = OpenMLIngester().load(61)

    assert ds.data.shape == (2, 2)
    assert list(ds.feature_names) == ["sepal_length", "sepal_width"]
    assert ds.metadata["source"] == "openml:61"
    assert ds.metadata["dataset_id"] == 61


def test_load_extracts_y_as_labels():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = pd.Series([0, 1], name="class")
    mock_ds = _make_oml_dataset(X, y=y)

    with _mock_openml(mock_ds):
        ds = OpenMLIngester(target_column="class").load(61)

    assert ds.labels is not None
    assert ds.labels.tolist() == [0, 1]


def test_load_no_labels_when_y_is_none():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    mock_ds = _make_oml_dataset(X, y=None)
    mock_ds.default_target_attribute = None

    with _mock_openml(mock_ds):
        ds = OpenMLIngester().load(554)

    assert ds.labels is None


def test_load_uses_default_target_when_none_specified():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([0, 1])
    mock_ds = _make_oml_dataset(X, y=y, target="class")

    with _mock_openml(mock_ds):
        OpenMLIngester().load(61)  # no target_column set

    # get_data should be called with the default target
    call_kwargs = mock_ds.get_data.call_args[1]
    assert call_kwargs["target"] == "class"


def test_metadata_fields():
    X = pd.DataFrame({"x": [1.0]})
    mock_ds = _make_oml_dataset(X, name="iris", version=3)

    with _mock_openml(mock_ds):
        ds = OpenMLIngester().load(61)

    assert ds.metadata["dataset_name"] == "iris"
    assert ds.metadata["version"] == 3
    assert "original_columns" in ds.metadata
    assert "n_dropped_categorical" in ds.metadata


# ---------------------------------------------------------------------------
# Load by name
# ---------------------------------------------------------------------------

def test_load_by_name_resolves_id():
    X = pd.DataFrame({"a": [1.0]})
    mock_ds = _make_oml_dataset(X, name="iris")

    # Fake list_datasets result
    datasets_df = pd.DataFrame({
        "name": ["iris", "iris"],
        "version": [1, 2],
        "did": [61, 969],
    })

    with _mock_openml(mock_ds, datasets_df=datasets_df) as fake_mod:
        OpenMLIngester().load("iris")

    # Should pick highest version (did=969)
    fake_mod.datasets.get_dataset.assert_called_once()
    call_id = fake_mod.datasets.get_dataset.call_args[0][0]
    assert call_id == 969


def test_load_by_name_specific_version():
    X = pd.DataFrame({"a": [1.0]})
    mock_ds = _make_oml_dataset(X)

    datasets_df = pd.DataFrame({
        "name": ["iris", "iris"],
        "version": [1, 2],
        "did": [61, 969],
    })

    with _mock_openml(mock_ds, datasets_df=datasets_df) as fake_mod:
        OpenMLIngester(version=1).load("iris")

    call_id = fake_mod.datasets.get_dataset.call_args[0][0]
    assert call_id == 61


def test_load_by_name_not_found_raises():
    X = pd.DataFrame({"a": [1.0]})
    mock_ds = _make_oml_dataset(X)

    datasets_df = pd.DataFrame({"name": ["other"], "version": [1], "did": [999]})

    with _mock_openml(mock_ds, datasets_df=datasets_df):
        with pytest.raises(ValueError, match="No OpenML dataset found with name"):
            OpenMLIngester().load("nonexistent_dataset")


def test_load_by_name_missing_version_raises():
    X = pd.DataFrame({"a": [1.0]})
    mock_ds = _make_oml_dataset(X)

    datasets_df = pd.DataFrame({"name": ["iris"], "version": [1], "did": [61]})

    with _mock_openml(mock_ds, datasets_df=datasets_df):
        with pytest.raises(ValueError, match="has no version 99"):
            OpenMLIngester(version=99).load("iris")


# ---------------------------------------------------------------------------
# Numeric/categorical handling
# ---------------------------------------------------------------------------

def test_no_numeric_columns_raises():
    X = pd.DataFrame({"text": ["a", "b"]})
    mock_ds = _make_oml_dataset(X, y=None)
    mock_ds.default_target_attribute = None

    with _mock_openml(mock_ds):
        with pytest.raises(ValueError, match="No numeric columns"):
            OpenMLIngester().load(61)


def test_categorical_stored_when_numeric_only_false():
    X = pd.DataFrame({"num": [1.0, 2.0], "cat": pd.Categorical(["a", "b"])})
    mock_ds = _make_oml_dataset(X, y=None)
    mock_ds.default_target_attribute = None

    with _mock_openml(mock_ds):
        ds = OpenMLIngester(numeric_only=False).load(61)

    assert "cat" in ds.categorical_data


def test_categorical_dropped_when_numeric_only_true():
    X = pd.DataFrame({"num": [1.0, 2.0], "cat": pd.Categorical(["a", "b"])})
    mock_ds = _make_oml_dataset(X, y=None)
    mock_ds.default_target_attribute = None

    with _mock_openml(mock_ds):
        ds = OpenMLIngester(numeric_only=True).load(61)

    assert ds.categorical_data == {}
