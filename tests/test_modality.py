"""Tests for sparse data, multi-label, and time series support (v0.7.0)."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(tmp_path, content: str, name: str = "data.csv"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).strip())
    return p


# ===========================================================================
# Sparse data support
# ===========================================================================

class TestSparseIngestion:
    def test_csr_matrix_accepted(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        from quprep.ingest.numpy_ingester import NumpyIngester
        X = scipy_sparse.csr_matrix(np.eye(4))
        dataset = NumpyIngester().load(X)
        assert dataset.data.shape == (4, 4)
        assert isinstance(dataset.data, np.ndarray)

    def test_csc_matrix_accepted(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        from quprep.ingest.numpy_ingester import NumpyIngester
        X = scipy_sparse.csc_matrix(np.eye(3))
        dataset = NumpyIngester().load(X)
        assert dataset.data.shape == (3, 3)

    def test_coo_matrix_accepted(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        from quprep.ingest.numpy_ingester import NumpyIngester
        X = scipy_sparse.coo_matrix(np.diag([1.0, 2.0, 3.0]))
        dataset = NumpyIngester().load(X)
        assert dataset.data.shape == (3, 3)

    def test_sparse_values_preserved(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        from quprep.ingest.numpy_ingester import NumpyIngester
        dense = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]])
        X = scipy_sparse.csr_matrix(dense)
        dataset = NumpyIngester().load(X)
        np.testing.assert_array_equal(dataset.data, dense)

    def test_sparse_with_y_labels(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        from quprep.ingest.numpy_ingester import NumpyIngester
        X = scipy_sparse.csr_matrix(np.eye(5))
        y = np.array([0, 1, 0, 1, 0])
        dataset = NumpyIngester().load(X, y=y)
        assert dataset.labels is not None
        np.testing.assert_array_equal(dataset.labels, y)

    def test_sparse_pipeline_roundtrip(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder
        X = scipy_sparse.csr_matrix(np.random.default_rng(0).random((10, 4)))
        result = Pipeline(encoder=AngleEncoder()).fit_transform(X)
        assert len(result.encoded) == 10


# ===========================================================================
# Multi-label support
# ===========================================================================

class TestMultiLabelNumpyIngester:
    def test_single_target_y(self):
        from quprep.ingest.numpy_ingester import NumpyIngester
        X = np.random.default_rng(0).random((20, 4))
        y = np.array([0, 1] * 10)
        dataset = NumpyIngester().load(X, y=y)
        assert dataset.labels is not None
        assert dataset.labels.shape == (20,)

    def test_multi_label_y(self):
        from quprep.ingest.numpy_ingester import NumpyIngester
        X = np.random.default_rng(0).random((15, 3))
        y = np.random.randint(0, 2, size=(15, 4))
        dataset = NumpyIngester().load(X, y=y)
        assert dataset.labels.shape == (15, 4)

    def test_no_labels_by_default(self):
        from quprep.ingest.numpy_ingester import NumpyIngester
        dataset = NumpyIngester().load(np.eye(3))
        assert dataset.labels is None


class TestMultiLabelCSVIngester:
    def test_single_target_column(self, tmp_path):
        from quprep.ingest.csv_ingester import CSVIngester
        p = _make_csv(tmp_path, """
            a,b,c,label
            1,2,3,0
            4,5,6,1
            7,8,9,0
        """)
        dataset = CSVIngester(target_columns="label").load(p)
        assert dataset.data.shape == (3, 3)
        assert dataset.labels is not None
        assert dataset.labels.shape == (3,)
        np.testing.assert_array_equal(dataset.labels, [0, 1, 0])

    def test_multi_label_columns(self, tmp_path):
        from quprep.ingest.csv_ingester import CSVIngester
        p = _make_csv(tmp_path, """
            a,b,y1,y2
            1,2,0,1
            3,4,1,0
            5,6,1,1
        """)
        dataset = CSVIngester(target_columns=["y1", "y2"]).load(p)
        assert dataset.data.shape == (3, 2)
        assert dataset.labels.shape == (3, 2)

    def test_target_columns_excluded_from_features(self, tmp_path):
        from quprep.ingest.csv_ingester import CSVIngester
        p = _make_csv(tmp_path, """
            feat1,feat2,target
            0.1,0.2,1
            0.3,0.4,0
        """)
        dataset = CSVIngester(target_columns="target").load(p)
        assert "target" not in dataset.feature_names

    def test_no_target_columns_default(self, tmp_path):
        from quprep.ingest.csv_ingester import CSVIngester
        p = _make_csv(tmp_path, """
            a,b,c
            1,2,3
            4,5,6
        """)
        dataset = CSVIngester().load(p)
        assert dataset.labels is None
        assert dataset.data.shape == (2, 3)


class TestDatasetLabels:
    def test_labels_preserved_in_copy(self):
        from quprep.core.dataset import Dataset
        y = np.array([0, 1, 2])
        ds = Dataset(data=np.eye(3), labels=y)
        ds2 = ds.copy()
        assert ds2.labels is not None
        np.testing.assert_array_equal(ds2.labels, y)
        ds2.labels[0] = 99
        assert ds.labels[0] == 0  # original unmodified

    def test_labels_none_by_default(self):
        from quprep.core.dataset import Dataset
        ds = Dataset(data=np.eye(3))
        assert ds.labels is None


class TestMultiLabelFeatureSelector:
    def test_mutual_info_single_label(self):
        from quprep.clean.selector import FeatureSelector
        from quprep.core.dataset import Dataset
        rng = np.random.default_rng(0)
        X = rng.random((50, 6))
        y = (X[:, 0] > 0.5).astype(int)  # only feat 0 is informative
        ds = Dataset(data=X, feature_names=[f"f{i}" for i in range(6)])
        sel = FeatureSelector(method="mutual_info", threshold=0.0, max_features=3)
        result = sel.fit_transform(ds, labels=y)
        assert result.n_features == 3

    def test_mutual_info_multi_label(self):
        from quprep.clean.selector import FeatureSelector
        from quprep.core.dataset import Dataset
        rng = np.random.default_rng(1)
        X = rng.random((60, 8))
        y = rng.integers(0, 2, size=(60, 3))  # 3 label columns
        ds = Dataset(data=X)
        sel = FeatureSelector(method="mutual_info", threshold=0.0, max_features=4)
        result = sel.fit_transform(ds, labels=y)
        assert result.n_features == 4


class TestPipelineLabels:
    def test_y_kwarg_stored_in_dataset(self):
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder
        X = np.random.default_rng(0).random((10, 3))
        y = np.array([0, 1] * 5)
        result = Pipeline(encoder=AngleEncoder()).fit_transform(X, y=y)
        assert result.dataset.labels is not None
        np.testing.assert_array_equal(result.dataset.labels, y)

    def test_labels_from_csv_ingester(self, tmp_path):
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder
        from quprep.ingest.csv_ingester import CSVIngester
        p = _make_csv(tmp_path, """
            a,b,c,label
            0.1,0.2,0.3,0
            0.4,0.5,0.6,1
            0.7,0.8,0.9,0
        """)
        pipeline = Pipeline(
            ingester=CSVIngester(target_columns="label"),
            encoder=AngleEncoder(),
        )
        result = pipeline.fit_transform(p)
        assert result.dataset.labels is not None
        assert result.dataset.labels.shape == (3,)

    def test_feature_selector_uses_dataset_labels(self):
        from quprep.clean.selector import FeatureSelector
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder
        from quprep.ingest.numpy_ingester import NumpyIngester
        rng = np.random.default_rng(2)
        X = rng.random((50, 6))
        y = (X[:, 0] > 0.5).astype(int)
        dataset = NumpyIngester().load(X, y=y)
        pipeline = Pipeline(
            cleaner=FeatureSelector(method="mutual_info", threshold=0.0, max_features=3),
            encoder=AngleEncoder(),
        )
        result = pipeline.fit_transform(dataset)
        assert result.dataset.n_features == 3

    def test_labels_survive_feature_selector(self):
        from quprep.clean.selector import FeatureSelector
        from quprep.core.dataset import Dataset
        rng = np.random.default_rng(3)
        X = rng.random((20, 5))
        y = np.arange(20)
        ds = Dataset(data=X, labels=y)
        sel = FeatureSelector(method="variance", threshold=0.0, max_features=3)
        out = sel.fit_transform(ds)
        assert out.labels is not None
        np.testing.assert_array_equal(out.labels, y)

    def test_labels_survive_imputer_drop(self):
        from quprep.clean.imputer import Imputer
        from quprep.core.dataset import Dataset
        data = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],  # dropped
            [4.0, 5.0],
            [6.0, np.nan],  # dropped
            [7.0, 8.0],
        ])
        y = np.array([10, 20, 30, 40, 50])
        ds = Dataset(data=data, labels=y)
        imputer = Imputer(strategy="drop")
        out = imputer.fit_transform(ds)
        assert out.labels is not None
        np.testing.assert_array_equal(out.labels, [10, 30, 50])

    def test_labels_survive_outlier_remove(self):
        from quprep.clean.outlier import OutlierHandler
        from quprep.core.dataset import Dataset
        rng = np.random.default_rng(0)
        data = rng.standard_normal((20, 2))
        data[5] = [100.0, 100.0]  # outlier row
        y = np.arange(20)
        ds = Dataset(data=data, labels=y)
        handler = OutlierHandler(method="zscore", action="remove", threshold=2.0)
        out = handler.fit_transform(ds)
        assert out.labels is not None
        assert len(out.labels) == out.data.shape[0]
        assert 5 not in out.labels  # outlier row label removed


# ===========================================================================
# Time series support
# ===========================================================================

class TestTimeSeriesIngester:
    def test_basic_load(self, tmp_path):
        from quprep.ingest.timeseries_ingester import TimeSeriesIngester
        p = _make_csv(tmp_path, """
            date,temp,humidity
            2024-01-01,20.1,55.0
            2024-01-02,21.3,60.0
            2024-01-03,19.8,58.0
        """)
        ds = TimeSeriesIngester(time_column="date").load(p)
        assert ds.data.shape == (3, 2)
        assert ds.feature_names == ["temp", "humidity"]
        assert "time_index" in ds.metadata
        assert len(ds.metadata["time_index"]) == 3

    def test_no_time_column(self, tmp_path):
        from quprep.ingest.timeseries_ingester import TimeSeriesIngester
        p = _make_csv(tmp_path, """
            a,b
            1.0,2.0
            3.0,4.0
            5.0,6.0
        """)
        ds = TimeSeriesIngester().load(p)
        assert ds.data.shape == (3, 2)
        assert ds.metadata["time_index"] == [0, 1, 2]

    def test_modality_metadata(self, tmp_path):
        from quprep.ingest.timeseries_ingester import TimeSeriesIngester
        p = _make_csv(tmp_path, "a,b\n1,2\n3,4\n")
        ds = TimeSeriesIngester().load(p)
        assert ds.metadata["modality"] == "time_series"

    def test_target_columns(self, tmp_path):
        from quprep.ingest.timeseries_ingester import TimeSeriesIngester
        p = _make_csv(tmp_path, """
            t,feat,label
            0,1.0,0
            1,2.0,1
            2,3.0,0
        """)
        ds = TimeSeriesIngester(time_column="t", target_columns="label").load(p)
        assert ds.data.shape == (3, 1)
        assert ds.labels is not None
        np.testing.assert_array_equal(ds.labels, [0, 1, 0])

    def test_file_not_found(self):
        from quprep.ingest.timeseries_ingester import TimeSeriesIngester
        with pytest.raises(FileNotFoundError):
            TimeSeriesIngester().load("/nonexistent/path.csv")


class TestWindowTransformer:
    def _make_dataset(self, n_timesteps=20, n_features=3):
        from quprep.core.dataset import Dataset
        rng = np.random.default_rng(0)
        data = rng.random((n_timesteps, n_features))
        return Dataset(
            data=data,
            feature_names=[f"f{i}" for i in range(n_features)],
            metadata={"time_index": list(range(n_timesteps)), "modality": "time_series"},
        )

    def test_output_shape(self):
        from quprep.preprocess.window import WindowTransformer
        ds = self._make_dataset(20, 3)
        wt = WindowTransformer(window_size=5, step=1)
        out = wt.fit_transform(ds)
        # n_windows = (20 - 5) // 1 + 1 = 16
        assert out.data.shape == (16, 15)  # 16 windows × (5×3) features

    def test_non_overlapping_windows(self):
        from quprep.preprocess.window import WindowTransformer
        ds = self._make_dataset(20, 2)
        wt = WindowTransformer(window_size=4, step=4)
        out = wt.fit_transform(ds)
        # n_windows = (20 - 4) // 4 + 1 = 5
        assert out.data.shape == (5, 8)  # 5 × (4×2)

    def test_feature_names_lag_format(self):
        from quprep.preprocess.window import WindowTransformer
        ds = self._make_dataset(10, 2)
        wt = WindowTransformer(window_size=3, step=1)
        out = wt.fit_transform(ds)
        assert "f0_lag2" in out.feature_names
        assert "f0_lag0" in out.feature_names

    def test_window_too_large_raises(self):
        from quprep.preprocess.window import WindowTransformer
        ds = self._make_dataset(5, 2)
        wt = WindowTransformer(window_size=10)
        with pytest.raises(ValueError, match="window_size"):
            wt.fit_transform(ds)

    def test_window_time_index_preserved(self):
        from quprep.preprocess.window import WindowTransformer
        ds = self._make_dataset(10, 2)
        wt = WindowTransformer(window_size=3, step=1)
        out = wt.fit_transform(ds)
        assert out.metadata["window_time_index"] is not None
        assert len(out.metadata["window_time_index"]) == 8

    def test_labels_aligned_to_last_timestep(self):
        from quprep.core.dataset import Dataset
        from quprep.preprocess.window import WindowTransformer
        data = np.arange(10, dtype=float).reshape(-1, 1)
        labels = np.arange(10)
        ds = Dataset(data=data, labels=labels,
                     metadata={"time_index": list(range(10))})
        wt = WindowTransformer(window_size=3, step=1)
        out = wt.fit_transform(ds)
        # 8 windows; each label = index of last timestep in window
        np.testing.assert_array_equal(out.labels, [2, 3, 4, 5, 6, 7, 8, 9])

    def test_no_labels_propagates_none(self):
        from quprep.preprocess.window import WindowTransformer
        ds = self._make_dataset(10, 2)
        out = WindowTransformer(window_size=3).fit_transform(ds)
        assert out.labels is None

    def test_modality_metadata_updated(self):
        from quprep.preprocess.window import WindowTransformer
        ds = self._make_dataset(10, 2)
        out = WindowTransformer(window_size=3).fit_transform(ds)
        assert out.metadata["modality"] == "time_series_windowed"
        assert out.metadata["window_size"] == 3
        assert out.metadata["original_n_timesteps"] == 10

    def test_invalid_window_size_raises(self):
        from quprep.preprocess.window import WindowTransformer
        with pytest.raises(ValueError):
            WindowTransformer(window_size=0)

    def test_invalid_step_raises(self):
        from quprep.preprocess.window import WindowTransformer
        with pytest.raises(ValueError):
            WindowTransformer(step=0)


class TestTimeSeriesPipelineIntegration:
    def test_timeseries_pipeline(self, tmp_path):
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder
        from quprep.ingest.timeseries_ingester import TimeSeriesIngester
        from quprep.preprocess.window import WindowTransformer

        p = _make_csv(tmp_path, "\n".join(
            ["t,a,b"] + [f"{i},{i*0.1:.2f},{i*0.2:.2f}" for i in range(30)]
        ))

        pipeline = Pipeline(
            ingester=TimeSeriesIngester(time_column="t"),
            preprocessor=WindowTransformer(window_size=5, step=1),
            encoder=AngleEncoder(),
        )
        result = pipeline.fit_transform(p)
        # n_windows = (30 - 5) // 1 + 1 = 26
        assert len(result.encoded) == 26
        assert result.encoded[0].metadata["n_qubits"] == 10  # 5 × 2 features

    def test_preprocessor_in_pipeline_summary(self, tmp_path):
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder
        from quprep.preprocess.window import WindowTransformer

        pipeline = Pipeline(
            preprocessor=WindowTransformer(window_size=4),
            encoder=AngleEncoder(),
        )
        summary = pipeline.summary()
        assert "preprocessor" in summary

    def test_preprocessor_in_get_params(self):
        from quprep.core.pipeline import Pipeline
        from quprep.preprocess.window import WindowTransformer

        wt = WindowTransformer(window_size=8)
        pipeline = Pipeline(preprocessor=wt)
        params = pipeline.get_params()
        assert params["preprocessor"] is wt

    def test_preprocessor_in_set_params(self):
        from quprep.core.pipeline import Pipeline
        from quprep.preprocess.window import WindowTransformer

        pipeline = Pipeline()
        wt = WindowTransformer(window_size=8)
        pipeline.set_params(preprocessor=wt)
        assert pipeline.preprocessor is wt
