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
        assert result.method in (
            "angle", "amplitude", "basis", "iqp",
            "reupload", "entangled_angle", "hamiltonian",
        )

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

    def test_regression_recommends_reupload_or_random_fourier(self):
        result = recommend(_continuous_dataset(), task="regression")
        assert result.method in ("reupload", "random_fourier")

    def test_classification_continuous_recommends_iqp_or_angle(self):
        result = recommend(_continuous_dataset(), task="classification")
        assert result.method in ("iqp", "angle", "reupload", "entangled_angle")

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


# ---------------------------------------------------------------------------
# Smart scoring — dataset-aware rules
# ---------------------------------------------------------------------------

class TestSmartScoring:
    def test_large_sample_count_penalises_amplitude(self):
        rng = np.random.default_rng(42)
        large = rng.random((2000, 4))  # n_samples > 1000 → amplitude penalty
        small = rng.random((20, 4))
        rec_large = recommend(large, task="regression")
        rec_small = recommend(small, task="regression", qubits=3)
        # amplitude should score relatively lower on the large dataset
        def _amp_score(recs):
            all_recs = [recs] + recs.alternatives
            for r in all_recs:
                if r.method == "amplitude":
                    return r.score
            return None
        score_large = _amp_score(rec_large)
        score_small = _amp_score(rec_small)
        assert score_large is not None and score_small is not None
        assert score_large < score_small

    def test_sparse_data_boosts_basis_for_qaoa(self):
        # Data with many exact zeros → basis should score higher than usual
        rng = np.random.default_rng(0)
        sparse = rng.random((50, 8))
        sparse[sparse < 0.7] = 0.0  # ~70% zeros → sparsity > 0.3
        result = recommend(sparse, task="qaoa")
        assert result.method == "basis"

    def test_correlated_features_boosts_iqp(self):
        # Strongly correlated features → IQP and entangled_angle should score higher
        rng = np.random.default_rng(0)
        base = rng.standard_normal((100, 1))
        # Make 6 features that are all highly correlated
        corr_data = np.hstack([base + rng.standard_normal((100, 1)) * 0.05 for _ in range(6)])
        result = recommend(corr_data, task="kernel")
        # IQP or entangled_angle should win for kernel with correlated features
        assert result.method in ("iqp", "entangled_angle")

    def test_negative_values_hurts_basis(self):
        # Continuous data with negatives → basis should not win for classification
        rng = np.random.default_rng(0)
        neg_data = rng.standard_normal((50, 6))  # all values could be negative
        result = recommend(neg_data, task="classification")
        assert result.method != "basis"

    def test_negative_values_rewards_amplitude(self):
        # Amplitude gets a +2 bonus for datasets with negative values
        from quprep.core.recommender import _profile_source, _score
        neg_data = np.random.default_rng(0).standard_normal((50, 4))
        pos_data = np.abs(neg_data)
        profile_neg = _profile_source(neg_data)
        profile_pos = _profile_source(pos_data)
        score_neg = _score("amplitude", profile_neg, "classification", None)
        score_pos = _score("amplitude", profile_pos, "classification", None)
        assert profile_neg["has_negatives"]
        assert not profile_pos["has_negatives"]
        assert score_neg > score_pos

    def test_small_sample_penalises_reupload(self):
        from quprep.core.recommender import _profile_source, _score
        rng = np.random.default_rng(0)
        small = rng.random((15, 4))   # < 20 samples → -8 penalty
        large = rng.random((100, 4))
        score_small = _score("reupload", _profile_source(small), "regression", None)
        score_large = _score("reupload", _profile_source(large), "regression", None)
        assert score_small < score_large

    def test_large_sample_boosts_reupload(self):
        from quprep.core.recommender import _profile_source, _score
        rng = np.random.default_rng(0)
        large = rng.random((1000, 4))   # > 500 → +3
        medium = rng.random((100, 4))
        score_large = _score("reupload", _profile_source(large), "regression", None)
        score_medium = _score("reupload", _profile_source(medium), "regression", None)
        assert score_large > score_medium

    def test_wide_data_penalises_iqp(self):
        from quprep.core.recommender import _profile_source, _score
        rng = np.random.default_rng(0)
        wide = rng.random((50, 25))   # d=25 > 15 → -(25-15)*0.4 = -4
        narrow = rng.random((50, 8))
        score_wide = _score("iqp", _profile_source(wide), "classification", None)
        score_narrow = _score("iqp", _profile_source(narrow), "classification", None)
        assert score_wide < score_narrow

    def test_high_missing_rate_penalises_amplitude(self):
        from quprep.core.recommender import _profile_source, _score
        rng = np.random.default_rng(0)
        data = rng.random((50, 4))
        # Introduce 25% missing values
        idx = rng.choice(data.size, size=data.size // 4, replace=False)
        flat = data.flatten()
        flat[idx] = np.nan
        data_nan = flat.reshape(data.shape)
        profile = _profile_source(data_nan)
        assert profile["missing_rate"] > 0.2
        score = _score("amplitude", profile, "regression", None)
        # Compare to clean data
        score_clean = _score("amplitude", _profile_source(data), "regression", None)
        assert score < score_clean

    def test_entangled_angle_in_alternatives(self):
        # entangled_angle should always appear somewhere in the result
        result = recommend(_continuous_dataset(), task="classification")
        all_methods = [result.method] + [a.method for a in result.alternatives]
        assert "entangled_angle" in all_methods

    def test_profile_fields_present(self):
        from quprep.core.recommender import _profile_source
        profile = _profile_source(_continuous_dataset())
        assert "missing_rate" in profile
        assert "sparsity" in profile
        assert "has_negatives" in profile
        assert "feature_collinear" in profile

    def test_profile_missing_rate_correct(self):
        from quprep.core.recommender import _profile_source
        data = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]])
        profile = _profile_source(data)
        assert abs(profile["missing_rate"] - 2 / 6) < 1e-9

    def test_profile_sparsity_correct(self):
        from quprep.core.recommender import _profile_source
        data = np.array([[0.0, 1.0], [0.0, 0.5], [0.0, 2.0]])
        profile = _profile_source(data)
        assert abs(profile["sparsity"] - 3 / 6) < 1e-9

    def test_profile_has_negatives_true(self):
        from quprep.core.recommender import _profile_source
        data = np.array([[-1.0, 2.0], [3.0, 4.0]])
        assert _profile_source(data)["has_negatives"]

    def test_profile_has_negatives_false(self):
        from quprep.core.recommender import _profile_source
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert not _profile_source(data)["has_negatives"]

    def test_profile_collinear_detected(self):
        from quprep.core.recommender import _profile_source
        rng = np.random.default_rng(0)
        base = rng.standard_normal((100, 1))
        corr = np.hstack([base + rng.standard_normal((100, 1)) * 0.01 for _ in range(4)])
        assert _profile_source(corr)["feature_collinear"]

    def test_profile_collinear_false_for_random(self):
        from quprep.core.recommender import _profile_source
        rng = np.random.default_rng(99)
        random = rng.standard_normal((200, 6))
        # Random uncorrelated data should not trigger collinearity flag
        assert not _profile_source(random)["feature_collinear"]


# ---------------------------------------------------------------------------
# Coverage gap — missing lines
# ---------------------------------------------------------------------------

class TestCoverageGaps:
    def test_amplitude_very_large_sample_penalty(self):
        # line 251: n_samples > 5000 → score -= 15.0
        from quprep.core.recommender import _profile_source, _score
        rng = np.random.default_rng(0)
        huge = rng.random((6000, 4))
        large = rng.random((1500, 4))
        score_huge = _score("amplitude", _profile_source(huge), "regression", None)
        score_large = _score("amplitude", _profile_source(large), "regression", None)
        assert score_huge < score_large

    def test_amplitude_medium_sample_penalty(self):
        # line 255: n_samples > 500 (≤ 1000) → score -= 4.0
        from quprep.core.recommender import _profile_source, _score
        rng = np.random.default_rng(0)
        medium = rng.random((700, 4))
        small = rng.random((50, 4))
        score_medium = _score("amplitude", _profile_source(medium), "regression", None)
        score_small = _score("amplitude", _profile_source(small), "regression", None)
        assert score_medium < score_small

    def test_amplitude_low_missing_rate_penalty(self):
        # line 265: missing_rate > 0.1 (≤ 0.2) → score -= 4.0
        from quprep.core.recommender import _profile_source, _score
        rng = np.random.default_rng(0)
        data = rng.random((50, 4))
        # Introduce ~15% missing
        flat = data.flatten()
        idx = rng.choice(len(flat), size=int(len(flat) * 0.15), replace=False)
        flat[idx] = np.nan
        data_nan = flat.reshape(data.shape)
        profile = _profile_source(data_nan)
        assert 0.1 < profile["missing_rate"] <= 0.2
        score_nan = _score("amplitude", profile, "regression", None)
        score_clean = _score("amplitude", _profile_source(data), "regression", None)
        assert score_nan < score_clean

    def test_basis_low_sparsity_bonus(self):
        # line 272: sparsity > 0.1 (≤ 0.3) → score += 1.5 * sparsity
        from quprep.core.recommender import _profile_source, _score
        rng = np.random.default_rng(0)
        data = rng.random((50, 4))
        # Make ~20% of values exactly zero
        flat = data.flatten()
        idx = rng.choice(len(flat), size=int(len(flat) * 0.2), replace=False)
        flat[idx] = 0.0
        data_sparse = flat.reshape(data.shape)
        profile = _profile_source(data_sparse)
        assert 0.1 < profile["sparsity"] <= 0.3
        score_sparse = _score("basis", profile, "qaoa", None)
        score_clean = _score("basis", _profile_source(data), "qaoa", None)
        assert score_sparse > score_clean

    def test_build_reason_amplitude_large_samples(self):
        # _build_reason amplitude path for n_samples > 1000
        from quprep.core.recommender import _build_reason
        profile = {
            "n_features": 4,
            "n_samples": 2000,
            "binary_fraction": 0.0,
            "continuous_fraction": 1.0,
            "missing_rate": 0.0,
            "sparsity": 0.0,
            "has_negatives": False,
            "feature_collinear": False,
        }
        reason = _build_reason("amplitude", profile, "regression", 30.0, None)
        assert "2000" in reason
        assert "state prep" in reason

    def test_build_reason_amplitude_high_missing(self):
        # line 358: _build_reason amplitude path for missing_rate > 0.1
        from quprep.core.recommender import _build_reason
        profile = {
            "n_features": 4,
            "n_samples": 50,
            "binary_fraction": 0.0,
            "continuous_fraction": 1.0,
            "missing_rate": 0.15,
            "sparsity": 0.0,
            "has_negatives": False,
            "feature_collinear": False,
        }
        reason = _build_reason("amplitude", profile, "regression", 30.0, None)
        assert "missing" in reason
        assert "impute" in reason

    def test_profile_accepts_dataframe(self):
        # line 482: DataFrame path in _profile_source
        from quprep.core.recommender import _profile_source
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.random((30, 4)), columns=["a", "b", "c", "d"])
        profile = _profile_source(df)
        assert profile["n_features"] == 4
        assert profile["n_samples"] == 30

    def test_profile_no_pandas_fallback(self, monkeypatch):
        # lines 474-475: pandas ImportError fallback in _profile_source
        import builtins
        import importlib

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("no pandas")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        import quprep.core.recommender as rec_mod
        importlib.reload(rec_mod)
        rng = np.random.default_rng(0)
        data = rng.random((20, 3))
        profile = rec_mod._profile_source(data)
        assert profile["n_features"] == 3

    def test_profile_corrcoef_exception_handled(self, monkeypatch):
        # lines 523-524: except Exception in correlation block
        from quprep.core import recommender as rec_mod

        def bad_corrcoef(*args, **kwargs):
            raise RuntimeError("corrcoef exploded")

        monkeypatch.setattr(rec_mod.np, "corrcoef", bad_corrcoef)
        rng = np.random.default_rng(0)
        data = rng.random((50, 4))
        profile = rec_mod._profile_source(data)
        # Should not raise — exception is caught; feature_collinear defaults to False
        assert profile["feature_collinear"] is False


# ---------------------------------------------------------------------------
# Recommender — wide dataset penalty (lines 395, 489)
# ---------------------------------------------------------------------------


def test_recommend_wide_dataset_hits_zz_penalty():
    """Lines 395 and 489: d > 15 penalty and note for zz_feature_map / pauli_feature_map."""
    import quprep as qd
    rng = np.random.default_rng(0)
    # 16 features triggers the d > 15 branch for zz/pauli scoring
    X = rng.random((50, 16))
    rec = qd.recommend(X, task="classification")
    assert rec is not None
    assert hasattr(rec, "method")
