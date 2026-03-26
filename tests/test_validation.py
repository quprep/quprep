"""Tests for input validation, schema enforcement, and warnings."""

import warnings

import numpy as np
import pytest

from quprep.core.dataset import Dataset
from quprep.validation import (
    DataSchema,
    FeatureSpec,
    QuPrepWarning,
    SchemaViolationError,
    validate_dataset,
)


def _ds(data):
    return Dataset(data=data.astype(np.float64))


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------

def test_validate_ok():
    validate_dataset(_ds(np.ones((10, 3))))


def test_validate_not_2d():
    with pytest.raises(ValueError, match="2-D"):
        validate_dataset(_ds(np.ones(5)))


def test_validate_no_samples():
    with pytest.raises(ValueError, match="no samples"):
        validate_dataset(_ds(np.ones((0, 3))))


def test_validate_no_features():
    with pytest.raises(ValueError, match="no features"):
        validate_dataset(_ds(np.ones((5, 0))))


def test_validate_wrong_dtype():
    ds = Dataset(data=np.ones((5, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="dtype"):
        validate_dataset(ds)


def test_validate_nan_warning():
    data = np.ones((5, 3))
    data[0, 1] = np.nan
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_dataset(_ds(data))
    assert len(w) == 1
    assert issubclass(w[0].category, QuPrepWarning)
    assert "NaN" in str(w[0].message)


# ---------------------------------------------------------------------------
# DataSchema
# ---------------------------------------------------------------------------

def test_schema_ok():
    ds = Dataset(
        data=np.array([[1.0, 0.5], [2.0, 0.8]]),
        feature_names=["age", "score"],
    )
    schema = DataSchema([
        FeatureSpec("age", dtype="continuous", min_value=0.0),
        FeatureSpec("score", dtype="continuous", min_value=0.0, max_value=1.0),
    ])
    schema.validate(ds)  # should not raise


def test_schema_feature_count_mismatch():
    ds = Dataset(data=np.ones((5, 2)))
    schema = DataSchema([FeatureSpec("a", dtype="continuous")])
    with pytest.raises(SchemaViolationError, match="Feature count"):
        schema.validate(ds)


def test_schema_min_violation():
    ds = Dataset(
        data=np.array([[-1.0], [2.0]]),
        feature_names=["x"],
    )
    schema = DataSchema([FeatureSpec("x", dtype="continuous", min_value=0.0)])
    with pytest.raises(SchemaViolationError, match="min value"):
        schema.validate(ds)


def test_schema_max_violation():
    ds = Dataset(
        data=np.array([[5.0], [2.0]]),
        feature_names=["x"],
    )
    schema = DataSchema([FeatureSpec("x", dtype="continuous", max_value=3.0)])
    with pytest.raises(SchemaViolationError, match="max value"):
        schema.validate(ds)


def test_schema_nullable_ok():
    data = np.array([[1.0], [np.nan]])
    ds = Dataset(data=data, feature_names=["x"])
    schema = DataSchema([FeatureSpec("x", dtype="continuous", nullable=True)])
    schema.validate(ds)  # should not raise


def test_schema_nullable_fail():
    data = np.array([[1.0], [np.nan]])
    ds = Dataset(data=data, feature_names=["x"])
    schema = DataSchema([FeatureSpec("x", dtype="continuous", nullable=False)])
    with pytest.raises(SchemaViolationError, match="NaN"):
        schema.validate(ds)


def test_schema_binary_violation():
    ds = Dataset(data=np.array([[0.0], [2.0]]), feature_names=["b"])
    schema = DataSchema([FeatureSpec("b", dtype="binary")])
    with pytest.raises(SchemaViolationError, match="binary"):
        schema.validate(ds)


def test_schema_infer_roundtrip():
    ds = Dataset(
        data=np.array([[1.0, 0.0], [2.0, 1.0]]),
        feature_names=["a", "b"],
        feature_types=["continuous", "binary"],
    )
    schema = DataSchema.infer(ds)
    assert len(schema.features) == 2
    assert schema.features[0].name == "a"
    schema.validate(ds)  # inferred schema should always validate its source


def test_schema_collects_all_violations():
    ds = Dataset(
        data=np.array([[-1.0, 5.0]]),
        feature_names=["x", "y"],
    )
    schema = DataSchema([
        FeatureSpec("x", dtype="continuous", min_value=0.0),
        FeatureSpec("y", dtype="continuous", max_value=3.0),
    ])
    with pytest.raises(SchemaViolationError) as exc_info:
        schema.validate(ds)
    # Both violations should appear in the same error message
    msg = str(exc_info.value)
    assert "x" in msg and "y" in msg
