"""Tests for streaming ingestion and Pipeline.stream()."""


import numpy as np
import pandas as pd
import pytest

import quprep as qd
from quprep.ingest.csv_ingester import CSVIngester
from quprep.ingest.numpy_ingester import NumpyIngester


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.uniform(0, 1, (250, 4)),
        columns=["a", "b", "c", "d"],
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def array_250():
    return np.random.default_rng(0).uniform(0, 1, (250, 4))


class TestCSVIngesterStream:
    def test_chunk_count(self, csv_path):
        chunks = list(CSVIngester().stream(csv_path, chunksize=100))
        assert len(chunks) == 3  # 100 + 100 + 50

    def test_chunk_sizes(self, csv_path):
        chunks = list(CSVIngester().stream(csv_path, chunksize=100))
        assert chunks[0].n_samples == 100
        assert chunks[1].n_samples == 100
        assert chunks[2].n_samples == 50

    def test_total_rows(self, csv_path):
        total = sum(c.n_samples for c in CSVIngester().stream(csv_path, chunksize=80))
        assert total == 250

    def test_feature_names_consistent(self, csv_path):
        chunks = list(CSVIngester().stream(csv_path, chunksize=100))
        for c in chunks:
            assert c.feature_names == ["a", "b", "c", "d"]

    def test_chunk_metadata(self, csv_path):
        chunks = list(CSVIngester().stream(csv_path, chunksize=100))
        for i, c in enumerate(chunks):
            assert c.metadata["chunk"] == i

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            list(CSVIngester().stream("/no/such/file.csv"))

    def test_label_extraction(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x0": rng.uniform(0, 1, 50),
                "x1": rng.uniform(0, 1, 50),
                "label": np.random.default_rng(1).integers(0, 2, 50),
            }
        )
        path = tmp_path / "labelled.csv"
        df.to_csv(path, index=False)
        chunks = list(CSVIngester(target_columns="label").stream(path, chunksize=25))
        for c in chunks:
            assert c.labels is not None
            assert c.labels.shape == (25,)


class TestNumpyIngesterStream:
    def test_chunk_count(self, array_250):
        chunks = list(NumpyIngester().stream(array_250, chunksize=100))
        assert len(chunks) == 3

    def test_total_rows(self, array_250):
        total = sum(c.n_samples for c in NumpyIngester().stream(array_250, chunksize=70))
        assert total == 250

    def test_data_values_correct(self, array_250):
        chunks = list(NumpyIngester().stream(array_250, chunksize=100))
        reconstructed = np.vstack([c.data for c in chunks])
        np.testing.assert_array_equal(reconstructed, array_250)

    def test_labels_propagated(self, array_250):
        y = np.arange(250)
        chunks = list(NumpyIngester().stream(array_250, y=y, chunksize=100))
        all_labels = np.concatenate([c.labels for c in chunks])
        np.testing.assert_array_equal(all_labels, y)

    def test_no_labels(self, array_250):
        chunks = list(NumpyIngester().stream(array_250, chunksize=100))
        for c in chunks:
            assert c.labels is None


class TestPipelineStream:
    def test_yields_results(self, array_250):
        pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), exporter=qd.QASMExporter())
        pipeline.fit(array_250[:50])
        results = list(pipeline.stream(array_250, chunksize=100))
        assert len(results) == 3

    def test_circuit_count_per_chunk(self, array_250):
        pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), exporter=qd.QASMExporter())
        pipeline.fit(array_250[:50])
        results = list(pipeline.stream(array_250, chunksize=100))
        assert len(results[0].circuits) == 100
        assert len(results[2].circuits) == 50

    def test_total_circuits(self, array_250):
        pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), exporter=qd.QASMExporter())
        pipeline.fit(array_250[:50])
        total = sum(len(r.circuits) for r in pipeline.stream(array_250, chunksize=80))
        assert total == 250

    def test_requires_fitted_pipeline(self, array_250):
        pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), exporter=qd.QASMExporter())
        with pytest.raises(RuntimeError, match="fitted"):
            list(pipeline.stream(array_250))

    def test_stream_csv(self, csv_path):
        pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), exporter=qd.QASMExporter())
        # Fit on the full file first
        pipeline.fit_transform(str(csv_path))
        results = list(pipeline.stream(str(csv_path), chunksize=100))
        assert len(results) == 3

    def test_invalid_source_type(self, array_250):
        pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), exporter=qd.QASMExporter())
        pipeline.fit(array_250[:50])
        with pytest.raises(TypeError):
            list(pipeline.stream([1, 2, 3]))
