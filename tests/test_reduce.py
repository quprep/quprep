"""Tests for dimensionality reduction stage."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.core.dataset import Dataset
from quprep.reduce.hardware_aware import HardwareAwareReducer
from quprep.reduce.lda import LDAReducer
from quprep.reduce.pca import PCAReducer
from quprep.reduce.spectral import SpectralReducer, TSNEReducer, UMAPReducer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n_samples=50, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_samples, n_features))
    return Dataset(
        data=data,
        feature_names=[f"x{i}" for i in range(n_features)],
        feature_types=["continuous"] * n_features,
        metadata={"source": "test"},
        categorical_data={},
    )


def _make_labels(n_samples=50, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n_samples)


# ---------------------------------------------------------------------------
# PCAReducer
# ---------------------------------------------------------------------------

class TestPCAReducer:
    def test_reduces_feature_count(self):
        ds = _make_dataset(n_features=8)
        result = PCAReducer(n_components=4).fit_transform(ds)
        assert result.data.shape == (50, 4)

    def test_feature_names_are_pc(self):
        result = PCAReducer(n_components=3).fit_transform(_make_dataset())
        assert result.feature_names == ["pc0", "pc1", "pc2"]

    def test_feature_types_all_continuous(self):
        result = PCAReducer(n_components=3).fit_transform(_make_dataset())
        assert all(t == "continuous" for t in result.feature_types)

    def test_variance_fraction(self):
        ds = _make_dataset(n_features=8)
        result = PCAReducer(n_components=0.95).fit_transform(ds)
        assert result.data.shape[1] <= 8
        assert result.data.shape[1] >= 1

    def test_n_components_capped_at_n_features(self):
        ds = _make_dataset(n_features=4)
        result = PCAReducer(n_components=10).fit_transform(ds)
        assert result.data.shape[1] <= 4

    def test_metadata_contains_reducer_key(self):
        result = PCAReducer(n_components=2).fit_transform(_make_dataset())
        assert result.metadata["reducer"] == "pca"
        assert "explained_variance_ratio" in result.metadata

    def test_explained_variance_ratio_property(self):
        reducer = PCAReducer(n_components=3)
        reducer.fit_transform(_make_dataset())
        evr = reducer.explained_variance_ratio_
        assert len(evr) == 3
        assert abs(sum(evr) - 1.0) < 0.01 or sum(evr) <= 1.0

    def test_explained_variance_property_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            _ = PCAReducer().explained_variance_ratio_

    def test_preserves_categorical_data(self):
        ds = _make_dataset()
        ds.categorical_data["label"] = ["a"] * 50
        result = PCAReducer(n_components=2).fit_transform(ds)
        assert result.categorical_data == ds.categorical_data

    def test_preserves_metadata(self):
        ds = _make_dataset()
        ds.metadata["custom"] = "value"
        result = PCAReducer(n_components=2).fit_transform(ds)
        assert result.metadata["custom"] == "value"

    def test_output_dtype_float64(self):
        result = PCAReducer(n_components=2).fit_transform(_make_dataset())
        assert result.data.dtype == np.float64

    def test_pipeline_integration(self):
        from quprep import Pipeline
        from quprep.encode.angle import AngleEncoder
        from quprep.export.qasm_export import QASMExporter

        ds = _make_dataset(n_features=8)
        pipeline = Pipeline(
            reducer=PCAReducer(n_components=4),
            encoder=AngleEncoder(),
            exporter=QASMExporter(),
        )
        result = pipeline.fit_transform(ds)
        assert len(result.circuits) == 50
        assert result.encoded[0].metadata["n_qubits"] == 4


# ---------------------------------------------------------------------------
# LDAReducer
# ---------------------------------------------------------------------------

class TestLDAReducer:
    def test_reduces_feature_count(self):
        ds = _make_dataset(n_features=8)
        labels = _make_labels(n_samples=50, n_classes=3)
        result = LDAReducer(n_components=2).fit_transform(ds, labels=labels)
        assert result.data.shape[1] <= 2

    def test_feature_names_are_ld(self):
        ds = _make_dataset()
        labels = _make_labels()
        result = LDAReducer(n_components=2).fit_transform(ds, labels=labels)
        assert result.feature_names[0] == "ld0"

    def test_metadata_reducer_key(self):
        ds = _make_dataset()
        labels = _make_labels()
        result = LDAReducer(n_components=2).fit_transform(ds, labels=labels)
        assert result.metadata["reducer"] == "lda"

    def test_labels_at_init(self):
        ds = _make_dataset()
        labels = _make_labels()
        reducer = LDAReducer(n_components=2, labels=labels)
        result = reducer.fit_transform(ds)
        assert result.data.shape[1] <= 2

    def test_labels_kwarg_overrides_init(self):
        ds = _make_dataset()
        labels = _make_labels(n_classes=3)
        reducer = LDAReducer(n_components=2, labels=np.zeros(50, dtype=int))
        result = reducer.fit_transform(ds, labels=labels)
        assert result.metadata["reducer"] == "lda"

    def test_missing_labels_raises(self):
        with pytest.raises(ValueError, match="labels"):
            LDAReducer(n_components=2).fit_transform(_make_dataset())

    def test_n_components_capped_at_n_classes_minus_1(self):
        ds = _make_dataset(n_samples=60, n_features=8)
        labels = _make_labels(n_samples=60, n_classes=3)
        result = LDAReducer(n_components=10).fit_transform(ds, labels=labels)
        assert result.data.shape[1] <= 2  # max = 3-1 = 2

    def test_preserves_categorical_data(self):
        ds = _make_dataset()
        ds.categorical_data["label"] = ["a"] * 50
        labels = _make_labels()
        result = LDAReducer(n_components=2).fit_transform(ds, labels=labels)
        assert result.categorical_data == ds.categorical_data

    def test_output_dtype_float64(self):
        ds = _make_dataset()
        result = LDAReducer(n_components=2).fit_transform(ds, labels=_make_labels())
        assert result.data.dtype == np.float64


# ---------------------------------------------------------------------------
# SpectralReducer
# ---------------------------------------------------------------------------

class TestSpectralReducer:
    def test_reduces_feature_count(self):
        ds = _make_dataset(n_features=16)
        result = SpectralReducer(n_components=4).fit_transform(ds)
        assert result.data.shape == (50, 4)

    def test_feature_names_are_freq(self):
        result = SpectralReducer(n_components=3).fit_transform(_make_dataset())
        assert result.feature_names == ["freq0", "freq1", "freq2"]

    def test_n_components_capped_at_fft_bins(self):
        ds = _make_dataset(n_features=4)
        # rfft of 4 features → 3 bins max
        result = SpectralReducer(n_components=100).fit_transform(ds)
        assert result.data.shape[1] <= 3

    def test_metadata_reducer_key(self):
        result = SpectralReducer(n_components=2).fit_transform(_make_dataset())
        assert result.metadata["reducer"] == "spectral"

    def test_output_all_non_negative(self):
        # magnitudes are always >= 0
        result = SpectralReducer(n_components=4).fit_transform(_make_dataset())
        assert np.all(result.data >= 0)

    def test_output_dtype_float64(self):
        result = SpectralReducer(n_components=2).fit_transform(_make_dataset())
        assert result.data.dtype == np.float64

    def test_preserves_categorical_data(self):
        ds = _make_dataset()
        ds.categorical_data["cat"] = ["x"] * 50
        result = SpectralReducer(n_components=2).fit_transform(ds)
        assert result.categorical_data == ds.categorical_data

    def test_pipeline_integration(self):
        from quprep import Pipeline
        from quprep.encode.angle import AngleEncoder
        from quprep.export.qasm_export import QASMExporter

        ds = _make_dataset(n_features=16)
        result = Pipeline(
            reducer=SpectralReducer(n_components=4),
            encoder=AngleEncoder(),
            exporter=QASMExporter(),
        ).fit_transform(ds)
        assert len(result.circuits) == 50
        assert result.encoded[0].metadata["n_qubits"] == 4


# ---------------------------------------------------------------------------
# TSNEReducer
# ---------------------------------------------------------------------------

class TestTSNEReducer:
    def test_reduces_to_2d(self):
        ds = _make_dataset(n_features=8)
        result = TSNEReducer(n_components=2).fit_transform(ds)
        assert result.data.shape == (50, 2)

    def test_feature_names_are_tsne(self):
        result = TSNEReducer(n_components=2).fit_transform(_make_dataset())
        assert result.feature_names == ["tsne0", "tsne1"]

    def test_metadata_reducer_key(self):
        result = TSNEReducer(n_components=2).fit_transform(_make_dataset())
        assert result.metadata["reducer"] == "tsne"

    def test_output_dtype_float64(self):
        result = TSNEReducer(n_components=2).fit_transform(_make_dataset())
        assert result.data.dtype == np.float64


# ---------------------------------------------------------------------------
# UMAPReducer
# ---------------------------------------------------------------------------

class TestUMAPReducer:
    def test_missing_dep_raises_import_error(self):
        import sys
        umap_available = "umap" in sys.modules or _umap_installed()
        if umap_available:
            pytest.skip("umap-learn is installed — skipping missing-dep test")
        ds = _make_dataset()
        with pytest.raises(ImportError, match="umap-learn"):
            UMAPReducer().fit_transform(ds)

    def test_reduces_if_umap_available(self):
        if not _umap_installed():
            pytest.skip("umap-learn not installed")
        ds = _make_dataset()
        result = UMAPReducer(n_components=2).fit_transform(ds)
        assert result.data.shape == (50, 2)


def _umap_installed() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# HardwareAwareReducer
# ---------------------------------------------------------------------------

class TestHardwareAwareReducer:
    def test_reduces_when_over_budget(self):
        ds = _make_dataset(n_features=16)
        result = HardwareAwareReducer(backend=4, encoding="angle").fit_transform(ds)
        assert result.data.shape[1] == 4

    def test_passthrough_when_within_budget(self):
        ds = _make_dataset(n_features=4)
        result = HardwareAwareReducer(backend=8, encoding="angle").fit_transform(ds)
        assert result.data.shape[1] == 4  # unchanged

    def test_named_backend(self):
        ds = _make_dataset(n_features=200)
        result = HardwareAwareReducer(backend="ionq_forte", encoding="angle").fit_transform(ds)
        assert result.data.shape[1] == 36  # ionq_forte has 36 qubits

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            HardwareAwareReducer(backend="fake_backend")._qubit_budget()

    def test_metadata_has_backend_and_budget(self):
        ds = _make_dataset(n_features=16)
        result = HardwareAwareReducer(backend=4).fit_transform(ds)
        assert result.metadata["backend"] == 4
        assert result.metadata["qubit_budget"] == 4

    def test_amplitude_encoding_larger_budget(self):
        ds = _make_dataset(n_features=100)
        # amplitude: max_features = min(2^4, 512) = 16
        result = HardwareAwareReducer(backend=4, encoding="amplitude").fit_transform(ds)
        assert result.data.shape[1] == 16

    def test_pipeline_integration(self):
        from quprep import Pipeline
        from quprep.encode.angle import AngleEncoder
        from quprep.export.qasm_export import QASMExporter

        ds = _make_dataset(n_features=20)
        result = Pipeline(
            reducer=HardwareAwareReducer(backend=4, encoding="angle"),
            encoder=AngleEncoder(),
            exporter=QASMExporter(),
        ).fit_transform(ds)
        assert result.encoded[0].metadata["n_qubits"] == 4
