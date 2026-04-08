"""Tests for the Dataset core class."""

import numpy as np

from quprep.core.dataset import Dataset


def _make(n=10, f=4, **kwargs):
    return Dataset(data=np.ones((n, f), dtype=np.float64), **kwargs)


def test_basic_properties():
    ds = _make(n=10, f=4)
    assert ds.n_samples == 10
    assert ds.n_features == 4
    assert ds.n_categorical == 0


def test_defaults_are_empty():
    ds = _make()
    assert ds.feature_names == []
    assert ds.feature_types == []
    assert ds.categorical_data == {}
    assert ds.metadata == {}
    assert ds.labels is None


def test_with_all_fields():
    labels = np.zeros(5)
    ds = Dataset(
        data=np.ones((5, 2), dtype=np.float64),
        feature_names=["a", "b"],
        feature_types=["continuous", "binary"],
        categorical_data={"cat": ["x", "y", "z", "x", "y"]},
        metadata={"source": "test"},
        labels=labels,
    )
    assert ds.n_categorical == 1
    assert ds.feature_names == ["a", "b"]
    assert ds.metadata["source"] == "test"
    assert ds.labels is labels


def test_copy_is_independent():
    ds = Dataset(
        data=np.array([[1.0, 2.0]]),
        feature_names=["x", "y"],
        feature_types=["continuous", "continuous"],
        categorical_data={"c": ["a"]},
        metadata={"k": "v"},
        labels=np.array([1.0]),
    )
    copy = ds.copy()
    copy.data[0, 0] = 99.0
    copy.feature_names.append("z")
    copy.metadata["k"] = "changed"
    assert ds.data[0, 0] == 1.0
    assert len(ds.feature_names) == 2
    assert ds.metadata["k"] == "v"


def test_copy_no_labels():
    ds = _make()
    copy = ds.copy()
    assert copy.labels is None


def test_repr():
    ds = _make(n=5, f=3)
    r = repr(ds)
    assert "n_samples=5" in r
    assert "n_features=3" in r


def test_repr_with_categorical():
    ds = Dataset(
        data=np.ones((4, 2), dtype=np.float64),
        categorical_data={"col": ["a", "b", "c", "d"]},
    )
    assert "categorical=1" in repr(ds)
