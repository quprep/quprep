"""Tests for the clean stage — Imputer, OutlierHandler, CategoricalEncoder, FeatureSelector."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.clean.categorical import CategoricalEncoder
from quprep.clean.imputer import Imputer
from quprep.clean.outlier import OutlierHandler
from quprep.clean.selector import FeatureSelector
from quprep.core.dataset import Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(data, feature_names=None, feature_types=None, categorical_data=None):
    n_features = data.shape[1]
    return Dataset(
        data=data.astype(float),
        feature_names=feature_names or [f"x{i}" for i in range(n_features)],
        feature_types=feature_types or ["continuous"] * n_features,
        categorical_data=categorical_data or {},
    )


def make_nan_dataset():
    data = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
        [4.0, np.nan],
        [5.0, 6.0],
    ])
    return make_dataset(data)


# ---------------------------------------------------------------------------
# Imputer
# ---------------------------------------------------------------------------

class TestImputer:

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            Imputer(strategy="magic")

    def test_invalid_drop_threshold_raises(self):
        with pytest.raises(ValueError):
            Imputer(drop_threshold=1.5)

    def test_no_missing_unchanged(self):
        ds = make_dataset(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = Imputer(strategy="mean").fit_transform(ds)
        np.testing.assert_array_equal(result.data, ds.data)

    def test_mean_imputation(self):
        ds = make_nan_dataset()
        result = Imputer(strategy="mean").fit_transform(ds)
        assert not np.isnan(result.data).any()
        # col 0: mean of [1, 4, 5] = 10/3 ≈ 3.333
        np.testing.assert_allclose(result.data[1, 0], 10 / 3)
        # col 1: mean of [2, 3, 6] = 11/3 ≈ 3.667
        np.testing.assert_allclose(result.data[2, 1], 11 / 3)

    def test_median_imputation(self):
        ds = make_nan_dataset()
        result = Imputer(strategy="median").fit_transform(ds)
        assert not np.isnan(result.data).any()
        # col 0: median of [1, 4, 5] = 4
        np.testing.assert_allclose(result.data[1, 0], 4.0)

    def test_mode_imputation(self):
        data = np.array([[1.0, 2.0], [1.0, np.nan], [2.0, 2.0]])
        ds = make_dataset(data)
        result = Imputer(strategy="mode").fit_transform(ds)
        assert not np.isnan(result.data).any()
        # col 1: mode of [2, 2] = 2
        assert result.data[1, 1] == 2.0

    def test_knn_imputation(self):
        ds = make_nan_dataset()
        result = Imputer(strategy="knn").fit_transform(ds)
        assert not np.isnan(result.data).any()
        assert result.n_samples == ds.n_samples

    def test_mice_imputation(self):
        ds = make_nan_dataset()
        result = Imputer(strategy="mice").fit_transform(ds)
        assert not np.isnan(result.data).any()

    def test_drop_rows(self):
        ds = make_nan_dataset()
        result = Imputer(strategy="drop").fit_transform(ds)
        assert result.n_samples == 2  # rows 0 and 3 have no NaN
        assert not np.isnan(result.data).any()

    def test_drop_high_missing_column(self):
        data = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan],
            [4.0, 1.0],
        ])
        ds = make_dataset(data)
        # col 1 is 75% missing > default threshold 50% → dropped
        result = Imputer(strategy="mean", drop_threshold=0.5).fit_transform(ds)
        assert result.n_features == 1

    def test_drop_threshold_zero_drops_any_missing(self):
        ds = make_nan_dataset()
        result = Imputer(strategy="mean", drop_threshold=0.0).fit_transform(ds)
        # both columns have missing values → both dropped
        assert result.n_features == 0

    def test_categorical_data_preserved(self):
        ds = make_nan_dataset()
        ds.categorical_data["label"] = ["a", "b", "c", "d"]
        result = Imputer(strategy="mean").fit_transform(ds)
        assert "label" in result.categorical_data

    def test_feature_names_updated_after_column_drop(self):
        data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        ds = make_dataset(data, feature_names=["keep", "drop"])
        result = Imputer(strategy="mean", drop_threshold=0.5).fit_transform(ds)
        assert result.feature_names == ["keep"]

    def test_returns_dataset(self):
        result = Imputer().fit_transform(make_nan_dataset())
        assert isinstance(result, Dataset)


# ---------------------------------------------------------------------------
# OutlierHandler
# ---------------------------------------------------------------------------

class TestOutlierHandler:

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            OutlierHandler(method="magic")

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            OutlierHandler(action="magic")

    def test_iqr_clip_no_outliers_unchanged(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ds = make_dataset(data)
        result = OutlierHandler(method="iqr", action="clip").fit_transform(ds)
        np.testing.assert_array_equal(result.data, data)

    def test_iqr_clip_clamps_outlier(self):
        data = np.array([[1.0], [2.0], [3.0], [100.0]])
        ds = make_dataset(data)
        result = OutlierHandler(method="iqr", action="clip").fit_transform(ds)
        assert result.data.max() < 100.0
        assert result.n_samples == 4  # clip keeps all rows

    def test_iqr_remove_drops_outlier_row(self):
        data = np.array([[1.0], [2.0], [3.0], [100.0]])
        ds = make_dataset(data)
        result = OutlierHandler(method="iqr", action="remove").fit_transform(ds)
        assert result.n_samples == 3
        assert result.data.max() < 100.0

    def test_zscore_clip(self):
        # need enough normal samples so the outlier doesn't dominate the std
        normal = np.zeros((30, 1))
        outlier = np.array([[50.0]])
        data = np.vstack([normal, outlier])
        ds = make_dataset(data)
        result = OutlierHandler(method="zscore", action="clip", threshold=3.0).fit_transform(ds)
        assert result.data.max() < 50.0
        assert result.n_samples == 31  # clip keeps all rows

    def test_zscore_remove(self):
        normal = np.zeros((30, 1))
        outlier = np.array([[50.0]])
        data = np.vstack([normal, outlier])
        ds = make_dataset(data)
        result = OutlierHandler(method="zscore", action="remove", threshold=3.0).fit_transform(ds)
        assert result.n_samples == 30

    def test_isolation_forest_clip(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, (50, 2))
        outliers = np.array([[50.0, 50.0], [-50.0, -50.0]])
        data = np.vstack([normal, outliers])
        ds = make_dataset(data)
        result = OutlierHandler(method="isolation_forest", action="clip").fit_transform(ds)
        assert result.n_samples == data.shape[0]
        assert result.data.max() < 50.0

    def test_isolation_forest_remove(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, (50, 2))
        outliers = np.array([[50.0, 50.0], [-50.0, -50.0]])
        data = np.vstack([normal, outliers])
        ds = make_dataset(data)
        result = OutlierHandler(method="isolation_forest", action="remove").fit_transform(ds)
        assert result.n_samples < data.shape[0]

    def test_categorical_data_synced_on_remove(self):
        data = np.array([[1.0], [2.0], [3.0], [100.0]])
        ds = make_dataset(data)
        ds.categorical_data["label"] = ["a", "b", "c", "outlier"]
        result = OutlierHandler(method="iqr", action="remove").fit_transform(ds)
        assert len(result.categorical_data["label"]) == result.n_samples

    def test_returns_dataset(self):
        ds = make_dataset(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert isinstance(OutlierHandler().fit_transform(ds), Dataset)

    def test_custom_threshold(self):
        data = np.array([[1.0], [2.0], [3.0], [4.0], [10.0]])
        ds = make_dataset(data)
        # very tight threshold — more values clipped
        result_tight = OutlierHandler(method="iqr", action="clip", threshold=0.1).fit_transform(ds)
        result_loose = OutlierHandler(method="iqr", action="clip", threshold=3.0).fit_transform(ds)
        assert result_tight.data.max() <= result_loose.data.max()


# ---------------------------------------------------------------------------
# CategoricalEncoder
# ---------------------------------------------------------------------------

class TestCategoricalEncoder:

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            CategoricalEncoder(strategy="magic")

    def test_invalid_handle_unknown_raises(self):
        with pytest.raises(ValueError):
            CategoricalEncoder(handle_unknown="magic")

    def test_no_categorical_data_unchanged(self):
        ds = make_dataset(np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = CategoricalEncoder().fit_transform(ds)
        np.testing.assert_array_equal(result.data, ds.data)

    def test_onehot_expands_columns(self):
        ds = make_dataset(np.zeros((3, 1)))
        ds.categorical_data["colour"] = ["red", "blue", "red"]
        result = CategoricalEncoder(strategy="onehot").fit_transform(ds)
        assert result.n_features == 3  # 1 numeric + 2 onehot
        assert result.categorical_data == {}

    def test_onehot_values_binary(self):
        ds = make_dataset(np.zeros((4, 0), dtype=float))
        ds.categorical_data["cat"] = ["a", "b", "a", "b"]
        result = CategoricalEncoder(strategy="onehot").fit_transform(ds)
        # all values should be 0 or 1
        assert set(result.data.flatten().tolist()).issubset({0.0, 1.0})

    def test_label_encoding_compact(self):
        ds = make_dataset(np.zeros((3, 0), dtype=float))
        ds.categorical_data["size"] = ["small", "medium", "large"]
        result = CategoricalEncoder(strategy="label").fit_transform(ds)
        assert result.n_features == 1
        assert result.categorical_data == {}

    def test_label_encoding_integers(self):
        ds = make_dataset(np.zeros((3, 0), dtype=float))
        ds.categorical_data["cat"] = ["a", "b", "c"]
        result = CategoricalEncoder(strategy="label").fit_transform(ds)
        vals = result.data[:, 0]
        assert set(vals).issubset({0.0, 1.0, 2.0})

    def test_ordinal_same_as_label_basic(self):
        ds = make_dataset(np.zeros((3, 0), dtype=float))
        ds.categorical_data["cat"] = ["x", "y", "z"]
        result = CategoricalEncoder(strategy="ordinal").fit_transform(ds)
        assert result.n_features == 1

    def test_multiple_categorical_columns(self):
        ds = make_dataset(np.zeros((3, 1)))
        ds.categorical_data["a"] = ["x", "y", "x"]
        ds.categorical_data["b"] = ["p", "q", "p"]
        result = CategoricalEncoder(strategy="label").fit_transform(ds)
        assert result.n_features == 3  # 1 numeric + 2 label cols
        assert result.categorical_data == {}

    def test_feature_types_updated(self):
        ds = make_dataset(np.zeros((3, 0), dtype=float))
        ds.categorical_data["cat"] = ["a", "b", "a"]
        result = CategoricalEncoder(strategy="onehot").fit_transform(ds)
        assert all(t == "binary" for t in result.feature_types)

    def test_returns_dataset(self):
        ds = make_dataset(np.zeros((2, 1)))
        ds.categorical_data["c"] = ["a", "b"]
        assert isinstance(CategoricalEncoder().fit_transform(ds), Dataset)


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------

class TestFeatureSelector:

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            FeatureSelector(method="magic")

    def test_correlation_removes_duplicate(self):
        # two perfectly correlated features
        x = np.arange(10, dtype=float).reshape(-1, 1)
        data = np.hstack([x, x * 2])
        ds = make_dataset(data)
        result = FeatureSelector(method="correlation", threshold=0.95).fit_transform(ds)
        assert result.n_features == 1

    def test_correlation_keeps_uncorrelated(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        data = np.column_stack([a, b])
        ds = make_dataset(data)
        result = FeatureSelector(method="correlation", threshold=0.95).fit_transform(ds)
        assert result.n_features == 2

    def test_variance_drops_constant(self):
        data = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        ds = make_dataset(data)
        result = FeatureSelector(method="variance", threshold=0.1).fit_transform(ds)
        assert result.n_features == 1
        assert result.feature_names == ["x1"]

    def test_variance_keeps_all_when_threshold_zero(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        ds = make_dataset(data)
        result = FeatureSelector(method="variance", threshold=0.0).fit_transform(ds)
        assert result.n_features == 2

    def test_mutual_info_requires_labels(self):
        ds = make_dataset(np.random.rand(10, 3))
        with pytest.raises(ValueError, match="labels"):
            FeatureSelector(method="mutual_info").fit_transform(ds)

    def test_mutual_info_selects_features(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, 20)
        signal = labels.astype(float) + rng.normal(0, 0.1, 20)
        noise = rng.standard_normal(20)
        data = np.column_stack([signal, noise])
        ds = make_dataset(data)
        result = FeatureSelector(
            method="mutual_info", threshold=0.1
        ).fit_transform(ds, labels=labels)
        # signal feature should be selected, noise might not be
        assert result.n_features >= 1

    def test_max_features_cap(self):
        data = np.random.rand(10, 5)
        ds = make_dataset(data)
        result = FeatureSelector(method="variance", threshold=0.0, max_features=2).fit_transform(ds)
        assert result.n_features == 2

    def test_feature_names_updated(self):
        x = np.arange(10, dtype=float).reshape(-1, 1)
        data = np.hstack([x, x])
        ds = make_dataset(data, feature_names=["a", "b"])
        result = FeatureSelector(method="correlation", threshold=0.95).fit_transform(ds)
        assert result.feature_names == ["a"]

    def test_categorical_data_preserved(self):
        data = np.random.rand(5, 2)
        ds = make_dataset(data)
        ds.categorical_data["label"] = ["x"] * 5
        result = FeatureSelector(method="variance", threshold=0.0).fit_transform(ds)
        assert "label" in result.categorical_data

    def test_returns_dataset(self):
        ds = make_dataset(np.random.rand(5, 2))
        assert isinstance(FeatureSelector().fit_transform(ds), Dataset)
