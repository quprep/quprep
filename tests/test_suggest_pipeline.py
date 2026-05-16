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

    def test_outlier_suggested_when_extreme_values(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (40, 4))
        data[0, 0] = 1000.0  # extreme outlier → IQR ratio > 10
        ds = _ds(data)
        suggestion = qd.suggest_pipeline(ds)
        assert suggestion.outlier_handler == "iqr"

    def test_lda_reducer_for_classification(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (60, 4))
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        ds = _ds(data, labels=labels)
        # 4 features, 3 classes, qubit budget 8 → no PCA, LDA suggested
        suggestion = qd.suggest_pipeline(ds, task="classification", qubits=8)
        assert suggestion.reducer == "lda"
        assert suggestion.reducer_n_components == 2

    def test_build_with_imputer_and_outlier_handler(self):
        from quprep.core.recommender import PipelineSuggestion
        s = PipelineSuggestion(
            encoder="angle", normalizer="minmax_pi",
            imputer="mean", outlier_handler="iqr",
            reducer=None, reducer_n_components=None, reason="test",
        )
        pipeline = s.build()
        assert isinstance(pipeline, qd.Pipeline)

    def test_build_with_lda_reducer(self):
        from quprep.core.recommender import PipelineSuggestion
        s = PipelineSuggestion(
            encoder="angle", normalizer="minmax_pi",
            imputer=None, outlier_handler=None,
            reducer="lda", reducer_n_components=2, reason="test",
        )
        pipeline = s.build()
        assert isinstance(pipeline, qd.Pipeline)

    def test_build_with_pca_reducer(self):
        from quprep.core.recommender import PipelineSuggestion
        s = PipelineSuggestion(
            encoder="angle", normalizer="minmax_pi",
            imputer=None, outlier_handler=None,
            reducer="pca", reducer_n_components=4, reason="test",
        )
        pipeline = s.build()
        assert isinstance(pipeline, qd.Pipeline)


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
        assert str(report.n_issues) in str(report)

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

    def test_encoder_warning_included(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(5, 10, (10, 3))  # out of [0,π] for AngleEncoder ry
        ds = _ds(data)
        enc = qd.AngleEncoder(rotation="ry")
        report = qd.preprocessing_report(ds, encoder=enc)
        assert any("encoder warning" in r for r in report.recommendations)

    def test_str_repr(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (10, 3)))
        report = qd.preprocessing_report(ds)
        assert "PreprocessingReport" in str(report)
        assert "PreprocessingReport" in repr(report)


# ---------------------------------------------------------------------------
# H-2 regression: suggest_pipeline normalizer for pauli_feature_map / qaoa_problem
# ---------------------------------------------------------------------------

class TestSuggestPipelineNormalizer:
    def _force_encoder(self, ds, method: str):
        import unittest.mock as mock

        from quprep.core.recommender import EncodingRecommendation, suggest_pipeline
        fake_rec = EncodingRecommendation(
            method=method, score=100.0, qubits=ds.n_features,
            depth="O(d)", nisq_safe=True, reason="forced for test",
        )
        with mock.patch("quprep.core.recommender.recommend", return_value=fake_rec):
            return suggest_pipeline(ds)

    def test_pauli_feature_map_gets_pm_pi_normalizer(self):
        # H-2 regression: pauli_feature_map needs [-π,π] → minmax_pm_pi, not minmax_pi
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (20, 3)))
        suggestion = self._force_encoder(ds, "pauli_feature_map")
        assert suggestion.normalizer == "minmax_pm_pi", (
            f"Expected minmax_pm_pi, got {suggestion.normalizer!r}"
        )

    def test_qaoa_problem_gets_pm_pi_normalizer(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, 1, (20, 3)))
        suggestion = self._force_encoder(ds, "qaoa_problem")
        assert suggestion.normalizer == "minmax_pm_pi", (
            f"Expected minmax_pm_pi, got {suggestion.normalizer!r}"
        )
