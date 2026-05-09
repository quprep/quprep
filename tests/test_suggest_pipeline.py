"""Tests for suggest_pipeline and preprocessing_report."""

from __future__ import annotations

import numpy as np
import pytest

import quprep as qd
from quprep.core.dataset import Dataset


def _ds(data, labels=None, feature_names=None):
    n, d = data.shape
    return Dataset(
        data=data.astype(np.float64),
        feature_names=feature_names or [f"f{i}" for i in range(d)],
        feature_types=["continuous"] * d,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# suggest_pipeline
# ---------------------------------------------------------------------------

class TestSuggestPipeline:
    def test_returns_suggestion(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (50, 6)))
        suggestion = qd.suggest_pipeline(ds)
        assert isinstance(suggestion, qd.PipelineSuggestion)
        assert isinstance(suggestion.encoder, str)
        assert isinstance(suggestion.normalizer, str)

    def test_imputer_suggested_when_missing(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (30, 4))
        data[0, 0] = np.nan
        ds = _ds(data)
        suggestion = qd.suggest_pipeline(ds)
        assert suggestion.imputer is not None
        assert suggestion.imputer in ("mean", "median")

    def test_no_imputer_when_clean(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (30, 4)))
        suggestion = qd.suggest_pipeline(ds)
        assert suggestion.imputer is None

    def test_reducer_suggested_when_over_budget(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (50, 20)))
        suggestion = qd.suggest_pipeline(ds, qubits=6)
        assert suggestion.reducer == "pca"
        assert suggestion.reducer_n_components == 6

    def test_no_reducer_within_budget(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (50, 4)))
        suggestion = qd.suggest_pipeline(ds, qubits=8)
        assert suggestion.reducer is None

    def test_build_returns_pipeline(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (30, 4)))
        suggestion = qd.suggest_pipeline(ds)
        pipeline = suggestion.build()
        assert isinstance(pipeline, qd.Pipeline)

    def test_invalid_task_raises(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (10, 4)))
        with pytest.raises(ValueError, match="Unknown task"):
            qd.suggest_pipeline(ds, task="invalid")

    def test_str_repr(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (20, 4)))
        s = qd.suggest_pipeline(ds)
        assert "PipelineSuggestion" in str(s)
        assert "PipelineSuggestion" in repr(s)

    def test_build_runs_fit_transform(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, np.pi, (30, 4))
        ds = _ds(data)
        suggestion = qd.suggest_pipeline(ds, qubits=4)
        pipeline = suggestion.build()
        result = pipeline.fit_transform(ds)
        assert result.encoded is not None or result.dataset is not None


# ---------------------------------------------------------------------------
# preprocessing_report
# ---------------------------------------------------------------------------

class TestPreprocessingReport:
    def test_clean_dataset_no_issues(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (30, 4)))
        report = qd.preprocessing_report(ds)
        assert isinstance(report, qd.PreprocessingReport)
        assert report.n_issues == 0

    def test_nan_detected(self):
        data = np.ones((10, 3))
        data[0, 1] = np.nan
        ds = _ds(data)
        report = qd.preprocessing_report(ds)
        assert report.n_issues >= 1
        assert any("imputation" in r for r in report.recommendations)

    def test_outliers_detected(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (30, 3))
        data[0, 0] = 1000.0  # extreme outlier → large range/IQR ratio
        ds = _ds(data)
        report = qd.preprocessing_report(ds)
        assert any("OutlierHandler" in r for r in report.recommendations)

    def test_qubit_budget_flag(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (30, 15)))
        report = qd.preprocessing_report(ds, qubit_budget=8)
        assert any("PCAReducer" in r for r in report.recommendations)

    def test_many_features_without_budget(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (30, 25)))
        report = qd.preprocessing_report(ds)
        assert any("PCAReducer" in r or "HardwareAwareReducer" in r for r in report.recommendations)

    def test_class_imbalance_flagged(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (40, 4))
        # 35 class 0, 5 class 1 → ratio 7:1
        labels = np.array([0] * 35 + [1] * 5)
        ds = _ds(data, labels=labels)
        report = qd.preprocessing_report(ds)
        assert any("imbalance" in r for r in report.recommendations)

    def test_encoder_compatibility_included(self):
        data = np.ones((10, 3))
        data[0, 1] = np.nan
        ds = _ds(data)
        enc = qd.AngleEncoder()
        report = qd.preprocessing_report(ds, encoder=enc)
        assert any("NaN" in r or "imputation" in r for r in report.recommendations)

    def test_str_repr(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (10, 3)))
        report = qd.preprocessing_report(ds)
        assert "PreprocessingReport" in str(report)
        assert "PreprocessingReport" in repr(report)
