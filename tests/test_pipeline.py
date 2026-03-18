"""Tests for the Pipeline orchestrator."""

import pytest


class TestPipeline:
    def test_import(self):
        from quprep import Pipeline
        assert Pipeline is not None

    def test_fit_transform_not_implemented(self):
        from quprep import Pipeline
        p = Pipeline()
        with pytest.raises(NotImplementedError):
            p.fit_transform(None)

    def test_version_accessible(self):
        import quprep
        assert quprep.__version__ == "0.1.0"
