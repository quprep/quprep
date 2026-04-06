"""Tests for ImageIngester (v0.7.0)."""

from __future__ import annotations

import pytest


def _make_image(tmp_path, name="img.png", size=(16, 16), mode="L", value=128):
    """Save a solid-colour PIL image and return the path."""
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new(mode, (size[1], size[0]), color=value)
    p = tmp_path / name
    img.save(str(p))
    return p


# ===========================================================================
# Single image loading
# ===========================================================================

class TestImageIngesterSingleFile:
    def test_grayscale_shape(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        p = _make_image(tmp_path, size=(8, 8))
        ds = ImageIngester(size=(8, 8), grayscale=True).load(p)
        assert ds.data.shape == (1, 64)

    def test_rgb_shape(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        p = _make_image(tmp_path, mode="RGB", value=(100, 150, 200))
        ds = ImageIngester(size=(4, 4), grayscale=False).load(p)
        assert ds.data.shape == (1, 4 * 4 * 3)

    def test_normalize_true(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        p = _make_image(tmp_path, value=255)
        ds = ImageIngester(size=(4, 4), normalize=True).load(p)
        assert ds.data.max() <= 1.0
        assert ds.data.min() >= 0.0

    def test_normalize_false(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        p = _make_image(tmp_path, value=200)
        ds = ImageIngester(size=(4, 4), normalize=False).load(p)
        assert ds.data.max() > 1.0

    def test_feature_names(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        p = _make_image(tmp_path, size=(4, 4))
        ds = ImageIngester(size=(4, 4)).load(p)
        assert ds.feature_names[0] == "px_0"
        assert len(ds.feature_names) == 16

    def test_modality_metadata(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        p = _make_image(tmp_path)
        ds = ImageIngester(size=(8, 8)).load(p)
        assert ds.metadata["modality"] == "image"
        assert ds.metadata["channels"] == 1
        assert ds.metadata["size"] == (8, 8)

    def test_file_not_found(self):
        from quprep.ingest.image_ingester import ImageIngester
        with pytest.raises(FileNotFoundError):
            ImageIngester().load("/nonexistent/img.png")

    def test_no_labels_single_file(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        p = _make_image(tmp_path)
        ds = ImageIngester(size=(4, 4)).load(p)
        assert ds.labels is None


# ===========================================================================
# Directory loading — flat
# ===========================================================================

class TestImageIngesterFlatDirectory:
    def test_flat_directory_shape(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        for i in range(4):
            _make_image(tmp_path, name=f"img{i}.png", size=(6, 6))
        ds = ImageIngester(size=(6, 6)).load(tmp_path)
        assert ds.data.shape == (4, 36)

    def test_flat_directory_no_labels(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        for i in range(3):
            _make_image(tmp_path, name=f"img{i}.png")
        ds = ImageIngester(size=(4, 4)).load(tmp_path)
        assert ds.labels is None

    def test_empty_directory_raises(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        with pytest.raises(ValueError, match="No supported image files"):
            ImageIngester().load(tmp_path)

    def test_n_images_metadata(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        for i in range(5):
            _make_image(tmp_path, name=f"img{i}.jpg")
        ds = ImageIngester(size=(4, 4)).load(tmp_path)
        assert ds.metadata["n_images"] == 5


# ===========================================================================
# Directory loading — subdirectory labels (ImageFolder convention)
# ===========================================================================

class TestImageIngesterSubdirLabels:
    def _make_class_dir(self, tmp_path, class_names, n_per_class=3, size=(8, 8)):
        for cls in class_names:
            d = tmp_path / cls
            d.mkdir()
            for i in range(n_per_class):
                _make_image(d, name=f"img{i}.png", size=size)

    def test_labels_from_subdirs(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        self._make_class_dir(tmp_path, ["cat", "dog"], n_per_class=3)
        ds = ImageIngester(size=(8, 8)).load(tmp_path)
        assert ds.labels is not None
        assert set(ds.labels) == {"cat", "dog"}
        assert len(ds.labels) == 6

    def test_label_count_matches_data(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        self._make_class_dir(tmp_path, ["a", "b", "c"], n_per_class=4)
        ds = ImageIngester(size=(4, 4)).load(tmp_path)
        assert ds.data.shape[0] == len(ds.labels) == 12

    def test_three_classes(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        self._make_class_dir(tmp_path, ["alpha", "beta", "gamma"], n_per_class=2)
        ds = ImageIngester(size=(4, 4)).load(tmp_path)
        assert set(ds.labels) == {"alpha", "beta", "gamma"}

    def test_values_in_range_normalized(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        self._make_class_dir(tmp_path, ["x", "y"], n_per_class=2)
        ds = ImageIngester(size=(4, 4), normalize=True).load(tmp_path)
        assert ds.data.min() >= 0.0
        assert ds.data.max() <= 1.0


# ===========================================================================
# Resize and mismatched size
# ===========================================================================

class TestImageIngesterResize:
    def test_resize_applied(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        _make_image(tmp_path, size=(64, 64))
        ds = ImageIngester(size=(8, 8)).load(tmp_path / "img.png")
        assert ds.data.shape == (1, 64)

    def test_no_size_mismatched_raises(self, tmp_path):
        from quprep.ingest.image_ingester import ImageIngester
        _make_image(tmp_path, name="small.png", size=(4, 4))
        _make_image(tmp_path, name="large.png", size=(8, 8))
        with pytest.raises(ValueError, match="different sizes"):
            ImageIngester(size=None).load(tmp_path)


# ===========================================================================
# Pipeline integration
# ===========================================================================

class TestImageIngesterPipeline:
    def test_pipeline_roundtrip(self, tmp_path):
        import quprep as qd
        for i in range(5):
            _make_image(tmp_path, name=f"img{i}.png", size=(4, 4))
        pipeline = qd.Pipeline(
            ingester=qd.ImageIngester(size=(4, 4)),
            encoder=qd.AngleEncoder(),
        )
        result = pipeline.fit_transform(tmp_path)
        assert len(result.encoded) == 5
        assert result.encoded[0].metadata["n_qubits"] == 16

    def test_pipeline_with_reducer(self, tmp_path):
        import quprep as qd
        for cls in ["a", "b"]:
            d = tmp_path / cls
            d.mkdir()
            for i in range(4):
                _make_image(d, name=f"img{i}.png", size=(8, 8))
        pipeline = qd.Pipeline(
            ingester=qd.ImageIngester(size=(8, 8)),
            reducer=qd.PCAReducer(n_components=4),
            encoder=qd.AngleEncoder(),
        )
        result = pipeline.fit_transform(tmp_path)
        assert result.dataset.n_features == 4
        assert result.dataset.labels is not None
