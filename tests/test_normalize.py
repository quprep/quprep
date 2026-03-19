"""Tests for normalization — Scaler and auto_normalizer."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.core.dataset import Dataset
from quprep.normalize.scalers import ENCODING_NORMALIZER_MAP, Scaler, auto_normalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(data, feature_names=None):
    data = np.asarray(data, dtype=float)
    n = data.shape[1] if data.ndim == 2 else 1
    return Dataset(
        data=data,
        feature_names=feature_names or [f"x{i}" for i in range(n)],
        feature_types=["continuous"] * n,
    )


def simple_ds():
    return make_dataset([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])


# ---------------------------------------------------------------------------
# Scaler — validation
# ---------------------------------------------------------------------------

class TestScalerValidation:

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            Scaler(strategy="quantum_magic")

    def test_valid_strategies_accepted(self):
        for s in ("l2", "minmax", "minmax_pi", "minmax_pm_pi", "zscore", "binary", "pm_one"):
            Scaler(strategy=s)  # should not raise

    def test_returns_dataset(self):
        result = Scaler().fit_transform(simple_ds())
        assert isinstance(result, Dataset)

    def test_original_dataset_not_mutated(self):
        ds = simple_ds()
        original = ds.data.copy()
        Scaler(strategy="minmax").fit_transform(ds)
        np.testing.assert_array_equal(ds.data, original)

    def test_metadata_preserved(self):
        ds = simple_ds()
        ds.metadata["source"] = "test"
        result = Scaler().fit_transform(ds)
        assert result.metadata["source"] == "test"

    def test_feature_names_preserved(self):
        ds = make_dataset([[1.0, 2.0], [3.0, 4.0]], feature_names=["a", "b"])
        result = Scaler().fit_transform(ds)
        assert result.feature_names == ["a", "b"]

    def test_categorical_data_preserved(self):
        ds = simple_ds()
        ds.categorical_data["label"] = ["x", "y", "z"]
        result = Scaler().fit_transform(ds)
        assert result.categorical_data["label"] == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# minmax — [0, 1]
# ---------------------------------------------------------------------------

class TestMinMax:

    def test_range_zero_to_one(self):
        result = Scaler(strategy="minmax").fit_transform(simple_ds())
        assert result.data.min() >= 0.0
        assert result.data.max() <= 1.0

    def test_min_becomes_zero(self):
        result = Scaler(strategy="minmax").fit_transform(simple_ds())
        np.testing.assert_allclose(result.data.min(axis=0), [0.0, 0.0])

    def test_max_becomes_one(self):
        result = Scaler(strategy="minmax").fit_transform(simple_ds())
        np.testing.assert_allclose(result.data.max(axis=0), [1.0, 1.0])

    def test_constant_column_maps_to_zero(self):
        ds = make_dataset([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        result = Scaler(strategy="minmax").fit_transform(ds)
        np.testing.assert_allclose(result.data[:, 0], [0.0, 0.0, 0.0])

    def test_single_sample(self):
        ds = make_dataset([[3.0, 7.0]])
        result = Scaler(strategy="minmax").fit_transform(ds)
        # single row → min == max → constant → maps to 0
        np.testing.assert_allclose(result.data, [[0.0, 0.0]])


# ---------------------------------------------------------------------------
# minmax_pi — [0, π]
# ---------------------------------------------------------------------------

class TestMinMaxPi:

    def test_range_zero_to_pi(self):
        result = Scaler(strategy="minmax_pi").fit_transform(simple_ds())
        assert result.data.min() >= 0.0 - 1e-10
        assert result.data.max() <= np.pi + 1e-10

    def test_min_becomes_zero(self):
        result = Scaler(strategy="minmax_pi").fit_transform(simple_ds())
        np.testing.assert_allclose(result.data.min(axis=0), [0.0, 0.0])

    def test_max_becomes_pi(self):
        result = Scaler(strategy="minmax_pi").fit_transform(simple_ds())
        np.testing.assert_allclose(result.data.max(axis=0), [np.pi, np.pi])


# ---------------------------------------------------------------------------
# minmax_pm_pi — [−π, π]
# ---------------------------------------------------------------------------

class TestMinMaxPmPi:

    def test_range_minus_pi_to_pi(self):
        result = Scaler(strategy="minmax_pm_pi").fit_transform(simple_ds())
        assert result.data.min() >= -np.pi - 1e-10
        assert result.data.max() <= np.pi + 1e-10

    def test_min_becomes_minus_pi(self):
        result = Scaler(strategy="minmax_pm_pi").fit_transform(simple_ds())
        np.testing.assert_allclose(result.data.min(axis=0), [-np.pi, -np.pi])

    def test_max_becomes_pi(self):
        result = Scaler(strategy="minmax_pm_pi").fit_transform(simple_ds())
        np.testing.assert_allclose(result.data.max(axis=0), [np.pi, np.pi])


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------

class TestZScore:

    def test_mean_near_zero(self):
        result = Scaler(strategy="zscore").fit_transform(simple_ds())
        np.testing.assert_allclose(result.data.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_std_near_one(self):
        result = Scaler(strategy="zscore").fit_transform(simple_ds())
        np.testing.assert_allclose(result.data.std(axis=0), [1.0, 1.0], atol=1e-10)

    def test_constant_column_becomes_zero(self):
        ds = make_dataset([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
        result = Scaler(strategy="zscore").fit_transform(ds)
        np.testing.assert_allclose(result.data[:, 0], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# l2
# ---------------------------------------------------------------------------

class TestL2:

    def test_each_row_unit_norm(self):
        data = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
        ds = make_dataset(data)
        result = Scaler(strategy="l2").fit_transform(ds)
        norms = np.linalg.norm(result.data, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-10)

    def test_known_values(self):
        # [3, 4] → norm 5 → [0.6, 0.8]
        ds = make_dataset([[3.0, 4.0]])
        result = Scaler(strategy="l2").fit_transform(ds)
        np.testing.assert_allclose(result.data[0], [0.6, 0.8], atol=1e-10)

    def test_zero_row_unchanged(self):
        ds = make_dataset([[0.0, 0.0], [1.0, 0.0]])
        result = Scaler(strategy="l2").fit_transform(ds)
        np.testing.assert_allclose(result.data[0], [0.0, 0.0])

    def test_operates_row_wise_not_column_wise(self):
        # two rows with same column values but different row norms
        ds = make_dataset([[1.0, 0.0], [3.0, 4.0]])
        result = Scaler(strategy="l2").fit_transform(ds)
        norms = np.linalg.norm(result.data, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# binary
# ---------------------------------------------------------------------------

class TestBinary:

    def test_output_is_zero_or_one(self):
        ds = make_dataset([[0.3, 0.7], [0.5, 0.9], [0.1, 0.4]])
        result = Scaler(strategy="binary").fit_transform(ds)
        assert set(result.data.flatten().tolist()).issubset({0.0, 1.0})

    def test_default_threshold_0_5(self):
        ds = make_dataset([[0.4, 0.6]])
        result = Scaler(strategy="binary").fit_transform(ds)
        np.testing.assert_array_equal(result.data[0], [0.0, 1.0])

    def test_at_threshold_maps_to_one(self):
        ds = make_dataset([[0.5, 0.5]])
        result = Scaler(strategy="binary").fit_transform(ds)
        np.testing.assert_array_equal(result.data[0], [1.0, 1.0])

    def test_custom_threshold(self):
        ds = make_dataset([[0.3, 0.8]])
        result = Scaler(strategy="binary", threshold=0.5).fit_transform(ds)
        np.testing.assert_array_equal(result.data[0], [0.0, 1.0])


# ---------------------------------------------------------------------------
# pm_one
# ---------------------------------------------------------------------------

class TestPmOne:

    def test_output_is_minus_one_or_one(self):
        ds = make_dataset([[0.3, 0.7], [0.5, 0.1]])
        result = Scaler(strategy="pm_one").fit_transform(ds)
        assert set(result.data.flatten().tolist()).issubset({-1.0, 1.0})

    def test_below_threshold_maps_to_minus_one(self):
        ds = make_dataset([[0.4, 0.6]])
        result = Scaler(strategy="pm_one").fit_transform(ds)
        np.testing.assert_array_equal(result.data[0], [-1.0, 1.0])

    def test_at_threshold_maps_to_one(self):
        ds = make_dataset([[0.5]])
        result = Scaler(strategy="pm_one").fit_transform(ds)
        np.testing.assert_array_equal(result.data[0], [1.0])


# ---------------------------------------------------------------------------
# auto_normalizer
# ---------------------------------------------------------------------------

class TestAutoNormalizer:

    def test_unknown_encoding_raises(self):
        with pytest.raises(ValueError, match="Unknown encoding"):
            auto_normalizer("not_an_encoding")

    @pytest.mark.parametrize("encoding,expected_strategy", [
        ("amplitude",   "l2"),
        ("angle_ry",    "minmax_pi"),
        ("angle_rx",    "minmax_pm_pi"),
        ("angle_rz",    "minmax_pm_pi"),
        ("basis",       "binary"),
        ("iqp",         "minmax_pm_pi"),
        ("qubo",        "binary"),
        ("ising",       "pm_one"),
    ])
    def test_correct_strategy_returned(self, encoding, expected_strategy):
        scaler = auto_normalizer(encoding)
        assert isinstance(scaler, Scaler)
        assert scaler.strategy == expected_strategy

    def test_all_encodings_covered(self):
        for encoding in ENCODING_NORMALIZER_MAP:
            scaler = auto_normalizer(encoding)
            assert isinstance(scaler, Scaler)

    def test_returned_scaler_is_functional(self):
        ds = simple_ds()
        scaler = auto_normalizer("angle_ry")
        result = scaler.fit_transform(ds)
        assert result.data.min() >= 0.0 - 1e-10
        assert result.data.max() <= np.pi + 1e-10
