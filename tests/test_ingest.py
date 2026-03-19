"""Tests for data ingestion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quprep.core.dataset import Dataset
from quprep.ingest.csv_ingester import CSVIngester
from quprep.ingest.numpy_ingester import NumpyIngester
from quprep.ingest.profiler import DatasetProfile, profile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n")
    return p


@pytest.fixture
def mixed_csv(tmp_path):
    p = tmp_path / "mixed.csv"
    p.write_text("name,age,score\nalice,30,0.9\nbob,25,0.7\ncarol,35,0.8\n")
    return p


@pytest.fixture
def tsv_file(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("x\ty\n1.0\t2.0\n3.0\t4.0\n")
    return p


@pytest.fixture
def csv_with_missing(tmp_path):
    p = tmp_path / "missing.csv"
    p.write_text("a,b\n1.0,2.0\n,3.0\n4.0,\n")
    return p


@pytest.fixture
def binary_csv(tmp_path):
    p = tmp_path / "binary.csv"
    p.write_text("label,score\n0,0.1\n1,0.9\n0,0.2\n1,0.8\n")
    return p


# ---------------------------------------------------------------------------
# CSVIngester
# ---------------------------------------------------------------------------

class TestCSVIngester:

    def test_returns_dataset(self, simple_csv):
        ds = CSVIngester().load(simple_csv)
        assert isinstance(ds, Dataset)

    def test_shape(self, simple_csv):
        ds = CSVIngester().load(simple_csv)
        assert ds.n_samples == 3
        assert ds.n_features == 3

    def test_feature_names(self, simple_csv):
        ds = CSVIngester().load(simple_csv)
        assert ds.feature_names == ["a", "b", "c"]

    def test_data_values(self, simple_csv):
        ds = CSVIngester().load(simple_csv)
        expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        np.testing.assert_array_equal(ds.data, expected)

    def test_data_is_float64(self, simple_csv):
        ds = CSVIngester().load(simple_csv)
        assert ds.data.dtype == np.float64

    def test_tsv_auto_detected(self, tsv_file):
        ds = CSVIngester().load(tsv_file)
        assert ds.n_samples == 2
        assert ds.n_features == 2

    def test_tsv_explicit_delimiter(self, tsv_file):
        ds = CSVIngester(delimiter="\t").load(tsv_file)
        assert ds.n_samples == 2

    def test_mixed_csv_drops_non_numeric(self, mixed_csv):
        ds = CSVIngester().load(mixed_csv)
        # 'name' is categorical — dropped from numeric data matrix
        assert ds.n_features == 2  # age, score

    def test_mixed_csv_feature_names_preserved(self, mixed_csv):
        ds = CSVIngester().load(mixed_csv)
        # numeric columns go into feature_names, categoricals into categorical_data
        assert "age" in ds.feature_names
        assert "name" in ds.categorical_data

    def test_feature_types_detected(self, mixed_csv):
        ds = CSVIngester().load(mixed_csv)
        # 'name' is categorical — lives in categorical_data, not feature_types
        assert "name" in ds.categorical_data

    def test_missing_values_preserved_as_nan(self, csv_with_missing):
        ds = CSVIngester().load(csv_with_missing)
        assert np.isnan(ds.data).any()

    def test_binary_column_detected(self, binary_csv):
        ds = CSVIngester().load(binary_csv)
        assert "binary" in ds.feature_types

    def test_source_in_metadata(self, simple_csv):
        ds = CSVIngester().load(simple_csv)
        assert "source" in ds.metadata

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            CSVIngester().load("/nonexistent/path/data.csv")

    def test_string_path_accepted(self, simple_csv):
        ds = CSVIngester().load(str(simple_csv))
        assert ds.n_samples == 3

    def test_repr(self, simple_csv):
        ds = CSVIngester().load(simple_csv)
        assert "Dataset" in repr(ds)
        assert "3" in repr(ds)


# ---------------------------------------------------------------------------
# NumpyIngester
# ---------------------------------------------------------------------------

class TestNumpyIngester:

    def test_ndarray_2d(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ds = NumpyIngester().load(arr)
        assert ds.n_samples == 3
        assert ds.n_features == 2

    def test_ndarray_1d_reshaped(self):
        arr = np.array([1.0, 2.0, 3.0])
        ds = NumpyIngester().load(arr)
        assert ds.n_samples == 3
        assert ds.n_features == 1

    def test_data_is_float64(self):
        arr = np.array([[1, 2], [3, 4]])
        ds = NumpyIngester().load(arr)
        assert ds.data.dtype == np.float64

    def test_auto_feature_names(self):
        arr = np.ones((4, 3))
        ds = NumpyIngester().load(arr)
        assert ds.feature_names == ["x0", "x1", "x2"]

    def test_all_features_continuous(self):
        arr = np.random.rand(5, 4)
        ds = NumpyIngester().load(arr)
        assert all(t == "continuous" for t in ds.feature_types)

    def test_values_preserved(self):
        arr = np.array([[1.5, 2.5], [3.5, 4.5]])
        ds = NumpyIngester().load(arr)
        np.testing.assert_array_equal(ds.data, arr)

    def test_dataframe_input(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        ds = NumpyIngester().load(df)
        assert ds.n_samples == 3
        assert ds.n_features == 2
        assert ds.feature_names == ["x", "y"]

    def test_dataframe_drops_non_numeric(self):
        df = pd.DataFrame({"name": ["a", "b"], "val": [1.0, 2.0]})
        ds = NumpyIngester().load(df)
        assert ds.n_features == 1

    def test_list_of_lists_accepted(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        ds = NumpyIngester().load(data)
        assert ds.n_samples == 2
        assert ds.n_features == 2

    def test_3d_array_raises(self):
        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="2-D"):
            NumpyIngester().load(arr)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            NumpyIngester().load("not an array")


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class TestProfiler:

    @pytest.fixture
    def simple_dataset(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        return Dataset(
            data=arr, feature_names=["a", "b"], feature_types=["continuous", "continuous"]
        )

    @pytest.fixture
    def dataset_with_nan(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        return Dataset(
            data=arr, feature_names=["a", "b"], feature_types=["continuous", "continuous"]
        )

    def test_returns_profile(self, simple_dataset):
        p = profile(simple_dataset)
        assert isinstance(p, DatasetProfile)

    def test_n_samples(self, simple_dataset):
        p = profile(simple_dataset)
        assert p.n_samples == 3

    def test_n_features(self, simple_dataset):
        p = profile(simple_dataset)
        assert p.n_features == 2

    def test_feature_names(self, simple_dataset):
        p = profile(simple_dataset)
        assert p.feature_names == ["a", "b"]

    def test_means(self, simple_dataset):
        p = profile(simple_dataset)
        np.testing.assert_allclose(p.means, [3.0, 4.0])

    def test_stds(self, simple_dataset):
        p = profile(simple_dataset)
        np.testing.assert_allclose(p.stds, np.std([[1, 3, 5], [2, 4, 6]], axis=1))

    def test_mins(self, simple_dataset):
        p = profile(simple_dataset)
        np.testing.assert_array_equal(p.mins, [1.0, 2.0])

    def test_maxs(self, simple_dataset):
        p = profile(simple_dataset)
        np.testing.assert_array_equal(p.maxs, [5.0, 6.0])

    def test_missing_counts_no_nan(self, simple_dataset):
        p = profile(simple_dataset)
        np.testing.assert_array_equal(p.missing_counts, [0, 0])

    def test_missing_counts_with_nan(self, dataset_with_nan):
        p = profile(dataset_with_nan)
        np.testing.assert_array_equal(p.missing_counts, [1, 1])

    def test_means_ignore_nan(self, dataset_with_nan):
        p = profile(dataset_with_nan)
        np.testing.assert_allclose(p.means[0], 2.0)   # mean of [1, 3]
        np.testing.assert_allclose(p.means[1], 5.0)   # mean of [4, 6]

    def test_str_output(self, simple_dataset):
        p = profile(simple_dataset)
        s = str(p)
        assert "samples" in s
        assert "features" in s

    def test_feature_types_in_profile(self, simple_dataset):
        p = profile(simple_dataset)
        assert p.feature_types == ["continuous", "continuous"]
