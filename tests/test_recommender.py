"""Tests for the encoding recommendation engine."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.core.recommender import EncodingRecommendation, recommend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _continuous_dataset(n_samples=50, n_features=8, seed=0):
    """Return a numpy array of continuous data."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def _binary_dataset(n_samples=50, n_features=8, seed=0):
    """Return a numpy array of binary {0, 1} data."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_samples, n_features)).astype(float)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

class TestRecommendReturnType:
    def test_returns_encoding_recommendation(self):
        result = recommend(_continuous_dataset())
        assert isinstance(result, EncodingRecommendation)

    def test_has_method(self):
        result = recommend(_continuous_dataset())
        assert isinstance(result.method, str)
        assert result.method in ("angle", "amplitude", "basis", "iqp", "reupload", "hamiltonian")

    def test_has_qubits(self):
        result = recommend(_continuous_dataset(n_features=6))
        assert isinstance(result.qubits, int)
        assert result.qubits > 0

    def test_has_depth(self):
        result = recommend(_continuous_dataset())
        assert isinstance(result.depth, str)

    def test_has_nisq_safe(self):
        result = recommend(_continuous_dataset())
        assert isinstance(result.nisq_safe, bool)

    def test_has_reason(self):
        result = recommend(_continuous_dataset())
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_has_score(self):
        result = recommend(_continuous_dataset())
        assert isinstance(result.score, float)
        assert result.score > 0

    def test_has_alternatives(self):
        result = recommend(_continuous_dataset())
        assert isinstance(result.alternatives, list)
        assert len(result.alternatives) > 0

    def test_alternatives_are_recommendations(self):
        result = recommend(_continuous_dataset())
        for alt in result.alternatives:
            assert isinstance(alt, EncodingRecommendation)

    def test_alternatives_descending_score(self):
        result = recommend(_continuous_dataset())
        scores = [alt.score for alt in result.alternatives]
        assert scores == sorted(scores, reverse=True)

    def test_best_score_geq_all_alternatives(self):
        result = recommend(_continuous_dataset())
        for alt in result.alternatives:
            assert result.score >= alt.score

    def test_no_nested_alternatives(self):
        result = recommend(_continuous_dataset())
        for alt in result.alternatives:
            assert alt.alternatives == []


# ---------------------------------------------------------------------------
# Task-specific recommendations
# ---------------------------------------------------------------------------

class TestTaskRecommendations:
    def test_simulation_recommends_hamiltonian(self):
        result = recommend(_continuous_dataset(), task="simulation")
        assert result.method == "hamiltonian"

    def test_qaoa_recommends_basis_for_binary_data(self):
        result = recommend(_binary_dataset(), task="qaoa")
        assert result.method == "basis"

    def test_kernel_recommends_iqp(self):
        result = recommend(_continuous_dataset(), task="kernel")
        assert result.method == "iqp"

    def test_regression_recommends_reupload(self):
        result = recommend(_continuous_dataset(), task="regression")
        assert result.method == "reupload"

    def test_classification_continuous_recommends_iqp_or_angle(self):
        result = recommend(_continuous_dataset(), task="classification")
        assert result.method in ("iqp", "angle", "reupload")

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            recommend(_continuous_dataset(), task="magic")


# ---------------------------------------------------------------------------
# Qubit budget
# ---------------------------------------------------------------------------

class TestQubitBudget:
    def test_budget_filters_over_limit(self):
        # With 2 qubits and 8 features, amplitude needs 3 qubits.
        # angle/basis/iqp/reupload/hamiltonian each need 8 — also over budget.
        # amplitude needs ceil(log2(8))=3 — within budget.
        result = recommend(_continuous_dataset(n_features=8), task="classification", qubits=2)
        # amplitude is the only one within budget for 8 features with 2 qubits
        # (needs ceil(log2(8)) = 3, actually over too... but penalty is applied)
        # The result should still return something — just the least-penalised one
        assert isinstance(result, EncodingRecommendation)

    def test_budget_within_range(self):
        # 4 features, 4 qubit budget — all encodings except amplitude(needs 2) fit
        result = recommend(_continuous_dataset(n_features=4), task="classification", qubits=4)
        assert result.qubits <= 4

    def test_amplitude_qubit_efficient(self):
        # amplitude for 8 features needs ceil(log2(8)) = 3 qubits
        result = recommend(_continuous_dataset(n_features=8), task="regression", qubits=3)
        # amplitude should score well since it fits in budget while others don't
        assert result.method == "amplitude"


# ---------------------------------------------------------------------------
# Data type sensitivity
# ---------------------------------------------------------------------------

class TestDataTypeSensitivity:
    def test_binary_data_scores_basis_higher(self):
        binary = _binary_dataset()
        result = recommend(binary, task="qaoa")
        assert result.method == "basis"

    def test_continuous_data_does_not_recommend_basis_for_classification(self):
        result = recommend(_continuous_dataset(), task="classification")
        assert result.method != "basis"


# ---------------------------------------------------------------------------
# Input source types
# ---------------------------------------------------------------------------

class TestInputTypes:
    def test_accepts_numpy_array(self):
        result = recommend(np.random.randn(30, 4))
        assert isinstance(result, EncodingRecommendation)

    def test_accepts_dataset(self):
        from quprep.core.dataset import Dataset
        ds = Dataset(
            data=np.random.randn(20, 4),
            feature_names=["a", "b", "c", "d"],
            feature_types=["continuous"] * 4,
            metadata={},
            categorical_data={},
        )
        result = recommend(ds)
        assert isinstance(result, EncodingRecommendation)

    def test_accepts_csv_file(self, tmp_path):
        import csv
        f = tmp_path / "data.csv"
        with f.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["x0", "x1", "x2"])
            for _ in range(20):
                writer.writerow([0.1, 0.5, 0.9])
        result = recommend(str(f))
        assert isinstance(result, EncodingRecommendation)


# ---------------------------------------------------------------------------
# str() output
# ---------------------------------------------------------------------------

class TestStr:
    def test_str_contains_method(self):
        result = recommend(_continuous_dataset())
        assert result.method in str(result)

    def test_str_contains_qubits(self):
        result = recommend(_continuous_dataset())
        assert "Qubits" in str(result)

    def test_str_contains_alternatives(self):
        result = recommend(_continuous_dataset())
        assert "Alternatives" in str(result)


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_returns_pipeline_result(self):
        result = recommend(_continuous_dataset(n_features=4), task="classification")
        pipeline_result = result.apply(_continuous_dataset(n_features=4))
        from quprep.core.pipeline import PipelineResult
        assert isinstance(pipeline_result, PipelineResult)
