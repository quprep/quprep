"""Tests for DriftDetector and DriftReport."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.core.dataset import Dataset
from quprep.core.drift import DriftDetector, DriftReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ds(data, names=None):
    n, d = data.shape
    return Dataset(
        data=data.astype(float),
        feature_names=names or [f"f{i}" for i in range(d)],
        feature_types=["continuous"] * d,
    )


def _train(n=50, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return _ds(rng.normal(0.0, 1.0, size=(n, d)))


def _similar(train_ds, n=20, seed=1):
    """New data from the same distribution."""
    rng = np.random.default_rng(seed)
    d = train_ds.n_features
    return _ds(rng.normal(0.0, 1.0, size=(n, d)), names=train_ds.feature_names)


def _drifted(train_ds, mean_shift=10.0, n=20, seed=2):
    """New data with large mean shift."""
    rng = np.random.default_rng(seed)
    d = train_ds.n_features
    data = rng.normal(mean_shift, 1.0, size=(n, d))
    return _ds(data, names=train_ds.feature_names)


# ---------------------------------------------------------------------------
# DriftDetector — basic fit / check
# ---------------------------------------------------------------------------

class TestDriftDetectorFitCheck:
    def test_fit_returns_self(self):
        det = DriftDetector()
        train = _train()
        assert det.fit(train) is det

    def test_fitted_flag(self):
        det = DriftDetector()
        assert det._fitted is False
        det.fit(_train())
        assert det._fitted is True

    def test_check_before_fit_raises(self):
        det = DriftDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            det.check(_train())

    def test_check_wrong_feature_count_raises(self):
        det = DriftDetector()
        train = _train(d=4)
        det.fit(train)
        wrong = _ds(np.ones((5, 3)))
        with pytest.raises(ValueError, match="Feature count mismatch"):
            det.check(wrong)

    def test_returns_drift_report(self):
        det = DriftDetector()
        train = _train()
        det.fit(train)
        report = det.check(_similar(train))
        assert isinstance(report, DriftReport)


# ---------------------------------------------------------------------------
# No-drift case
# ---------------------------------------------------------------------------

class TestNoDrift:
    def test_similar_data_no_drift(self):
        det = DriftDetector(warn=False)
        train = _train()
        det.fit(train)
        report = det.check(_similar(train))
        assert report.overall_drift is False

    def test_no_drift_empty_drifted_list(self):
        det = DriftDetector(warn=False)
        train = _train()
        det.fit(train)
        report = det.check(_similar(train))
        assert report.drifted_features == []

    def test_no_drift_zero_count(self):
        det = DriftDetector(warn=False)
        train = _train()
        det.fit(train)
        report = det.check(_similar(train))
        assert report.n_features_drifted == 0

    def test_same_data_no_drift(self):
        det = DriftDetector(warn=False)
        train = _train()
        det.fit(train)
        report = det.check(train)
        assert report.overall_drift is False


# ---------------------------------------------------------------------------
# Drift detected
# ---------------------------------------------------------------------------

class TestDriftDetected:
    def test_large_mean_shift_triggers_drift(self):
        det = DriftDetector(mean_threshold=3.0, warn=False)
        train = _train()
        det.fit(train)
        report = det.check(_drifted(train, mean_shift=20.0))
        assert report.overall_drift is True

    def test_drifted_features_non_empty(self):
        det = DriftDetector(mean_threshold=3.0, warn=False)
        train = _train(d=4)
        det.fit(train)
        report = det.check(_drifted(train, mean_shift=20.0))
        assert len(report.drifted_features) > 0

    def test_n_features_drifted_positive(self):
        det = DriftDetector(mean_threshold=3.0, warn=False)
        train = _train()
        det.fit(train)
        report = det.check(_drifted(train, mean_shift=20.0))
        assert report.n_features_drifted > 0

    def test_std_ratio_drift(self):
        """Data with 10× larger std should trigger std_ratio check."""
        det = DriftDetector(std_threshold=2.0, warn=False)
        train = _train(n=100)
        det.fit(train)
        rng = np.random.default_rng(99)
        big_std = _ds(rng.normal(0.0, 10.0, size=(50, train.n_features)),
                      names=train.feature_names)
        report = det.check(big_std)
        assert report.overall_drift is True

    def test_feature_stats_keys_present(self):
        det = DriftDetector(warn=False)
        train = _train(d=3)
        det.fit(train)
        report = det.check(_similar(train))
        for name in ["f0", "f1", "f2"]:
            assert name in report.feature_stats
            stats = report.feature_stats[name]
            for key in ["train_mean", "new_mean", "train_std", "new_std",
                        "mean_shift_sigmas", "std_ratio"]:
                assert key in stats


# ---------------------------------------------------------------------------
# Warning behaviour
# ---------------------------------------------------------------------------

class TestWarningBehaviour:
    def test_warn_true_issues_warning(self):
        det = DriftDetector(mean_threshold=3.0, warn=True)
        train = _train()
        det.fit(train)
        with pytest.warns(match="drift"):
            det.check(_drifted(train, mean_shift=20.0))

    def test_warn_false_no_warning(self, recwarn):
        det = DriftDetector(mean_threshold=3.0, warn=False)
        train = _train()
        det.fit(train)
        det.check(_drifted(train, mean_shift=20.0))
        drift_warns = [w for w in recwarn.list if "drift" in str(w.message).lower()]
        assert drift_warns == []

    def test_no_drift_no_warning(self, recwarn):
        det = DriftDetector(warn=True)
        train = _train()
        det.fit(train)
        det.check(_similar(train))
        drift_warns = [w for w in recwarn.list if "drift" in str(w.message).lower()]
        assert drift_warns == []


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------

class TestCustomThresholds:
    def test_low_mean_threshold_more_sensitive(self):
        det_sensitive = DriftDetector(mean_threshold=0.1, warn=False)
        det_default = DriftDetector(mean_threshold=3.0, warn=False)
        train = _train()
        det_sensitive.fit(train)
        det_default.fit(train)
        # Small shift that default wouldn't catch
        rng = np.random.default_rng(5)
        slight = _ds(rng.normal(0.5, 1.0, size=(20, 4)), names=train.feature_names)
        assert det_sensitive.check(slight).overall_drift is True
        assert det_default.check(slight).overall_drift is False

    def test_high_threshold_less_sensitive(self):
        det = DriftDetector(mean_threshold=100.0, warn=False)
        train = _train()
        det.fit(train)
        report = det.check(_drifted(train, mean_shift=5.0))
        assert report.overall_drift is False


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

class TestFeatureNames:
    def test_feature_names_used_in_report(self):
        det = DriftDetector(mean_threshold=3.0, warn=False)
        data = np.ones((20, 3)) * 0.5
        train = Dataset(data=data, feature_names=["age", "score", "rate"],
                        feature_types=["continuous"] * 3)
        det.fit(train)
        drifted_data = np.ones((10, 3)) * 50.0
        new = Dataset(data=drifted_data, feature_names=["age", "score", "rate"],
                      feature_types=["continuous"] * 3)
        report = det.check(new)
        for name in report.drifted_features:
            assert name in ["age", "score", "rate"]

    def test_auto_feature_names_when_absent(self):
        det = DriftDetector(warn=False)
        train = Dataset(data=np.ones((10, 2)), feature_names=[], feature_types=[])
        det.fit(train)
        assert det._feature_names == ["feature[0]", "feature[1]"]


# ---------------------------------------------------------------------------
# NaN tolerance
# ---------------------------------------------------------------------------

class TestNaNTolerance:
    def test_nan_in_training_data_does_not_crash(self):
        det = DriftDetector(warn=False)
        data = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]] * 10, dtype=float)
        train = _ds(data.reshape(-1, 2))
        det.fit(train)
        report = det.check(_similar(train))
        assert isinstance(report, DriftReport)

    def test_nan_in_new_data_does_not_crash(self):
        det = DriftDetector(warn=False)
        train = _train()
        det.fit(train)
        data = np.ones((10, 4))
        data[0, 0] = np.nan
        new = _ds(data, names=train.feature_names)
        report = det.check(new)
        assert isinstance(report, DriftReport)


# ---------------------------------------------------------------------------
# Constant feature edge case
# ---------------------------------------------------------------------------

class TestConstantFeature:
    def test_constant_train_feature_no_crash(self):
        det = DriftDetector(warn=False)
        data = np.column_stack([np.ones(30), np.random.default_rng(0).normal(size=30)])
        train = _ds(data)
        det.fit(train)
        new = _ds(np.column_stack([np.ones(10), np.ones(10)]))
        # Should not raise even though train_std=0 for first feature
        report = det.check(new)
        assert isinstance(report, DriftReport)


# ---------------------------------------------------------------------------
# String representations
# ---------------------------------------------------------------------------

class TestStringRepr:
    def test_str_no_drift(self):
        det = DriftDetector(warn=False)
        train = _train()
        det.fit(train)
        report = det.check(_similar(train))
        assert "no drift" in str(report)

    def test_str_with_drift(self):
        det = DriftDetector(mean_threshold=3.0, warn=False)
        train = _train()
        det.fit(train)
        report = det.check(_drifted(train, mean_shift=20.0))
        text = str(report)
        assert "drifted" in text
        assert "σ shift" in text

    def test_repr_compact(self):
        det = DriftDetector(warn=False)
        train = _train()
        det.fit(train)
        report = det.check(_similar(train))
        r = repr(report)
        assert "DriftReport" in r
        assert "overall_drift" in r


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_pipeline_with_drift_detector(self):
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder

        rng = np.random.default_rng(0)
        X_train = rng.normal(0.0, 1.0, size=(50, 4))
        X_test = rng.normal(0.0, 1.0, size=(10, 4))

        det = DriftDetector(warn=False)
        pipeline = Pipeline(encoder=AngleEncoder(), drift_detector=det)
        pipeline.fit(X_train)
        result = pipeline.transform(X_test)

        assert result.drift_report is not None
        assert isinstance(result.drift_report, DriftReport)

    def test_pipeline_no_drift_detector_report_is_none(self):
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder

        rng = np.random.default_rng(0)
        X_train = rng.normal(0.0, 1.0, size=(50, 4))
        X_test = rng.normal(0.0, 1.0, size=(10, 4))

        pipeline = Pipeline(encoder=AngleEncoder())
        pipeline.fit(X_train)
        result = pipeline.transform(X_test)
        assert result.drift_report is None

    def test_pipeline_drift_warning_on_shifted_data(self):
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder

        rng = np.random.default_rng(0)
        X_train = rng.normal(0.0, 1.0, size=(50, 4))
        X_drifted = rng.normal(20.0, 1.0, size=(10, 4))

        det = DriftDetector(mean_threshold=3.0, warn=True)
        pipeline = Pipeline(encoder=AngleEncoder(), drift_detector=det)
        pipeline.fit(X_train)
        with pytest.warns(match="drift"):
            result = pipeline.transform(X_drifted)
        assert result.drift_report.overall_drift is True

    def test_drift_detector_in_get_params(self):
        from quprep.core.pipeline import Pipeline

        det = DriftDetector()
        pipeline = Pipeline(drift_detector=det)
        assert pipeline.get_params()["drift_detector"] is det

    def test_pipeline_serialization_preserves_drift_detector(self, tmp_path):
        from quprep.core.pipeline import Pipeline
        from quprep.encode.angle import AngleEncoder

        rng = np.random.default_rng(0)
        X_train = rng.normal(0.0, 1.0, size=(50, 4))
        X_test = rng.normal(0.0, 1.0, size=(10, 4))

        det = DriftDetector(warn=False)
        pipeline = Pipeline(encoder=AngleEncoder(), drift_detector=det)
        pipeline.fit(X_train)
        pipeline.save(tmp_path / "pipeline.pkl")
        loaded = Pipeline.load(tmp_path / "pipeline.pkl")
        result = loaded.transform(X_test)
        assert result.drift_report is not None


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_top_level_import():
    import quprep as qd
    assert hasattr(qd, "DriftDetector")
    assert hasattr(qd, "DriftReport")
    det = qd.DriftDetector(warn=False)
    train = _train()
    det.fit(train)
    report = det.check(_similar(train))
    assert isinstance(report, qd.DriftReport)
