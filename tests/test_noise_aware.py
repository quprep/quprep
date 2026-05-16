"""Tests for NoiseProfile and NoiseAwarePreprocessor."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.core.dataset import Dataset
from quprep.preprocess.noise_aware import NoiseAwarePreprocessor, NoiseProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(variances: list[float], n_samples: int = 60, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    d = len(variances)
    data = rng.standard_normal((n_samples, d))
    for i, v in enumerate(variances):
        data[:, i] = data[:, i] * np.sqrt(v)
    return Dataset(
        data=data,
        feature_names=[f"x{i}" for i in range(d)],
        feature_types=["continuous"] * d,
    )


def _linear_profile(n_qubits: int = 5, base_error: float = 0.01) -> NoiseProfile:
    """Linear chain with error rates 0.01, 0.02, ..., 0.01*n_qubits (qubit 0 is best)."""
    coupling = [(i, i + 1) for i in range(n_qubits - 1)]
    error_rates = [base_error * (i + 1) for i in range(n_qubits)]
    return NoiseProfile(qubit_error_rates=error_rates, coupling_map=coupling)


# ---------------------------------------------------------------------------
# NoiseProfile
# ---------------------------------------------------------------------------

class TestNoiseProfile:
    def test_n_qubits(self):
        assert _linear_profile(4).n_qubits == 4

    def test_t1_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="t1"):
            NoiseProfile(
                qubit_error_rates=[0.01, 0.02],
                coupling_map=[(0, 1)],
                t1=[100.0],
            )

    def test_t2_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="t2"):
            NoiseProfile(
                qubit_error_rates=[0.01, 0.02],
                coupling_map=[(0, 1)],
                t2=[80.0],
            )

    def test_coupling_map_out_of_range_raises(self):
        with pytest.raises(ValueError, match="qubit 5"):
            NoiseProfile(
                qubit_error_rates=[0.01, 0.02],
                coupling_map=[(0, 5)],
            )

    def test_qubit_score_no_coherence(self):
        profile = _linear_profile(3)
        assert profile.qubit_score(0) == pytest.approx(0.01)
        assert profile.qubit_score(2) == pytest.approx(0.03)

    def test_qubit_score_with_t1_t2(self):
        profile = NoiseProfile(
            qubit_error_rates=[0.01],
            coupling_map=[],
            t1=[100.0],
            t2=[80.0],
        )
        expected = 0.01 + 1.0 / 100.0 + 1.0 / 80.0
        assert profile.qubit_score(0) == pytest.approx(expected)

    def test_qubit_score_ignores_zero_coherence(self):
        profile = NoiseProfile(
            qubit_error_rates=[0.02],
            coupling_map=[],
            t1=[0.0],
            t2=[0.0],
        )
        assert profile.qubit_score(0) == pytest.approx(0.02)

    def test_empty_coupling_map_valid(self):
        profile = NoiseProfile(qubit_error_rates=[0.01, 0.02], coupling_map=[])
        assert profile.n_qubits == 2


# ---------------------------------------------------------------------------
# NoiseAwarePreprocessor — construction and validation
# ---------------------------------------------------------------------------

class TestNoiseAwarePreprocessorInit:
    def test_invalid_deadzone_negative(self):
        with pytest.raises(ValueError, match="angle_deadzone"):
            NoiseAwarePreprocessor(_linear_profile(), angle_deadzone=-0.1)

    def test_invalid_deadzone_too_large(self):
        with pytest.raises(ValueError, match="angle_deadzone"):
            NoiseAwarePreprocessor(_linear_profile(), angle_deadzone=0.5)

    def test_zero_deadzone_valid(self):
        prep = NoiseAwarePreprocessor(_linear_profile(), angle_deadzone=0.0)
        assert prep.angle_deadzone == 0.0

    def test_just_under_half_deadzone_valid(self):
        prep = NoiseAwarePreprocessor(_linear_profile(), angle_deadzone=0.499)
        assert prep.angle_deadzone == pytest.approx(0.499)


# ---------------------------------------------------------------------------
# NoiseAwarePreprocessor — fit behaviour
# ---------------------------------------------------------------------------

class TestNoiseAwarePreprocessorFit:
    def test_too_many_features_raises(self):
        ds = _make_dataset([1.0] * 8)
        with pytest.raises(ValueError, match="more features"):
            NoiseAwarePreprocessor(_linear_profile(4)).fit(ds)

    def test_permutation_covers_all_feature_indices(self):
        ds = _make_dataset([1.0, 3.0, 2.0, 0.5])
        prep = NoiseAwarePreprocessor(_linear_profile(6)).fit(ds)
        assert sorted(prep.permutation_) == list(range(4))

    def test_qubit_assignment_has_correct_length(self):
        ds = _make_dataset([1.0, 2.0, 3.0])
        prep = NoiseAwarePreprocessor(_linear_profile(5)).fit(ds)
        assert len(prep.qubit_assignment_) == 3

    def test_qubit_assignment_all_valid_qubit_indices(self):
        ds = _make_dataset([1.0, 2.0, 3.0])
        prep = NoiseAwarePreprocessor(_linear_profile(5)).fit(ds)
        for q in prep.qubit_assignment_:
            assert 0 <= q < 5

    def test_high_variance_feature_gets_low_error_qubit(self):
        # Feature 1 clearly has much higher variance than feature 0.
        ds = _make_dataset([0.05, 9.0], n_samples=200)
        profile = NoiseProfile(
            qubit_error_rates=[0.001, 0.050],  # qubit 0 is clearly best
            coupling_map=[(0, 1)],
        )
        prep = NoiseAwarePreprocessor(profile, encoding="angle").fit(ds)
        assert prep.qubit_assignment_[1] == 0  # high-var → low-error qubit

    def test_swap_estimate_attributes_set_after_fit(self):
        ds = _make_dataset([1.0, 2.0])
        prep = NoiseAwarePreprocessor(_linear_profile(4)).fit(ds)
        assert prep.estimated_swaps_before_ is not None
        assert prep.estimated_swaps_after_ is not None

    def test_topology_never_increases_swap_count(self):
        profile = NoiseProfile(
            qubit_error_rates=[0.05, 0.04, 0.01, 0.03, 0.02],
            coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        ds = _make_dataset([1.0, 2.0])
        prep = NoiseAwarePreprocessor(profile, encoding="entangled_angle").fit(ds)
        assert prep.estimated_swaps_after_ <= prep.estimated_swaps_before_

    def test_adjacent_best_qubits_zero_swap_cost(self):
        # Qubits 0 and 1 are the best; they are adjacent in the coupling map.
        profile = NoiseProfile(
            qubit_error_rates=[0.01, 0.02, 0.05],
            coupling_map=[(0, 1), (1, 2)],
        )
        ds = _make_dataset([1.0, 2.0])
        prep = NoiseAwarePreprocessor(profile, encoding="entangled_angle").fit(ds)
        assert prep.estimated_swaps_after_ == 0

    def test_no_coupling_map_swap_zero(self):
        profile = NoiseProfile(qubit_error_rates=[0.01, 0.02, 0.03], coupling_map=[])
        ds = _make_dataset([1.0, 2.0])
        prep = NoiseAwarePreprocessor(profile, encoding="angle").fit(ds)
        assert prep.estimated_swaps_after_ == 0

    def test_single_qubit_encoding_no_topology_reorder(self):
        profile = NoiseProfile(
            qubit_error_rates=[0.05, 0.01, 0.03],
            coupling_map=[(0, 1), (1, 2)],
        )
        ds = _make_dataset([1.0, 2.0])
        prep = NoiseAwarePreprocessor(profile, encoding="angle").fit(ds)
        # angle encoding is not entangled, so topology path is not applied.
        # Both estimated SWAP counts should be identical (0).
        assert prep.estimated_swaps_after_ == prep.estimated_swaps_before_

    def test_single_feature(self):
        ds = _make_dataset([1.0])
        prep = NoiseAwarePreprocessor(_linear_profile(3)).fit(ds)
        assert list(prep.permutation_) == [0]
        assert len(prep.qubit_assignment_) == 1

    def test_entangled_encoding_no_coupling_map_zero_after_swaps(self):
        # entangled encoding + empty coupling map → lines 281-282
        profile = NoiseProfile(qubit_error_rates=[0.01, 0.02, 0.03], coupling_map=[])
        ds = _make_dataset([1.0, 2.0])
        prep = NoiseAwarePreprocessor(profile, encoding="entangled_angle").fit(ds)
        assert prep.estimated_swaps_after_ == 0

    def test_single_feature_entangled_encoding_zero_swaps(self):
        # single feature with entangled encoding → _count_adjacent_swaps returns 0 early (line 437)
        profile = _linear_profile(3)
        ds = _make_dataset([2.0])
        prep = NoiseAwarePreprocessor(profile, encoding="entangled_angle").fit(ds)
        assert prep.estimated_swaps_after_ == 0


# ---------------------------------------------------------------------------
# NoiseAwarePreprocessor — transform behaviour
# ---------------------------------------------------------------------------

class TestNoiseAwarePreprocessorTransform:
    def test_not_fitted_raises(self):
        from sklearn.exceptions import NotFittedError
        ds = _make_dataset([1.0, 2.0])
        with pytest.raises(NotFittedError):
            NoiseAwarePreprocessor(_linear_profile()).transform(ds)

    def test_wrong_feature_count_raises(self):
        ds2 = _make_dataset([1.0, 2.0])
        ds3 = _make_dataset([1.0, 2.0, 3.0])
        prep = NoiseAwarePreprocessor(_linear_profile()).fit(ds2)
        with pytest.raises(ValueError, match="features"):
            prep.transform(ds3)

    def test_output_shape_preserved(self):
        ds = _make_dataset([1.0, 2.0, 3.0])
        result = NoiseAwarePreprocessor(_linear_profile()).fit_transform(ds)
        assert result.data.shape == ds.data.shape

    def test_feature_names_reordered_consistently(self):
        ds = _make_dataset([1.0, 2.0, 3.0])
        prep = NoiseAwarePreprocessor(_linear_profile())
        result = prep.fit_transform(ds)
        # Every original name must appear exactly once.
        assert sorted(result.feature_names) == sorted(ds.feature_names)
        # Output column i must carry the name of permutation_[i].
        for i, orig_idx in enumerate(prep.permutation_):
            assert result.feature_names[i] == ds.feature_names[int(orig_idx)]

    def test_feature_types_reordered_consistently(self):
        ds = _make_dataset([1.0, 2.0, 3.0])
        prep = NoiseAwarePreprocessor(_linear_profile())
        result = prep.fit_transform(ds)
        for i, orig_idx in enumerate(prep.permutation_):
            assert result.feature_types[i] == ds.feature_types[int(orig_idx)]

    def test_labels_preserved(self):
        ds = _make_dataset([1.0, 2.0])
        ds.labels = np.array([0, 1] * 30)
        result = NoiseAwarePreprocessor(_linear_profile()).fit_transform(ds)
        np.testing.assert_array_equal(result.labels, ds.labels)

    def test_categorical_data_preserved(self):
        ds = _make_dataset([1.0, 2.0])
        ds.categorical_data = {"color": ["red", "blue"] * 30}
        result = NoiseAwarePreprocessor(_linear_profile()).fit_transform(ds)
        assert result.categorical_data == ds.categorical_data

    def test_metadata_noise_aware_flag(self):
        ds = _make_dataset([1.0, 2.0])
        result = NoiseAwarePreprocessor(_linear_profile()).fit_transform(ds)
        assert result.metadata["noise_aware"] is True

    def test_metadata_contains_qubit_assignment(self):
        ds = _make_dataset([1.0, 2.0])
        result = NoiseAwarePreprocessor(_linear_profile()).fit_transform(ds)
        assert "qubit_assignment" in result.metadata

    def test_metadata_contains_swap_estimates(self):
        ds = _make_dataset([1.0, 2.0])
        result = NoiseAwarePreprocessor(_linear_profile()).fit_transform(ds)
        assert "estimated_swaps_before" in result.metadata
        assert "estimated_swaps_after" in result.metadata

    def test_fit_transform_equals_fit_then_transform(self):
        ds = _make_dataset([1.0, 2.0, 3.0])
        profile = _linear_profile()
        r1 = NoiseAwarePreprocessor(profile).fit_transform(ds)
        prep = NoiseAwarePreprocessor(profile)
        prep.fit(ds)
        r2 = prep.transform(ds)
        np.testing.assert_array_equal(r1.data, r2.data)

    def test_column_values_are_original_columns_permuted(self):
        ds = _make_dataset([1.0, 4.0, 0.5])
        prep = NoiseAwarePreprocessor(_linear_profile())
        result = prep.fit_transform(ds)
        for i, orig_idx in enumerate(prep.permutation_):
            np.testing.assert_array_almost_equal(
                result.data[:, i], ds.data[:, int(orig_idx)]
            )

    def test_single_feature_passthrough(self):
        ds = _make_dataset([2.0])
        result = NoiseAwarePreprocessor(_linear_profile(3)).fit_transform(ds)
        np.testing.assert_array_equal(result.data, ds.data)


# ---------------------------------------------------------------------------
# Angle dead-zone remapping
# ---------------------------------------------------------------------------

class TestAngleDeadzone:
    def _pi_dataset(self, values: list[list[float]]) -> Dataset:
        data = np.array(values, dtype=float)
        n_feat = data.shape[1]
        return Dataset(
            data=data,
            feature_names=[f"f{i}" for i in range(n_feat)],
            feature_types=["continuous"] * n_feat,
        )

    def test_values_within_deadzone_bounds(self):
        # All extreme values (0 and π) should be pushed inside [0.1π, 0.9π].
        ds = self._pi_dataset([[0.0, np.pi, np.pi / 2]])
        profile = _linear_profile(5)
        result = NoiseAwarePreprocessor(
            profile, encoding="angle", angle_deadzone=0.1
        ).fit_transform(ds)
        lo, hi = 0.1 * np.pi, 0.9 * np.pi
        assert result.data.min() >= lo - 1e-10
        assert result.data.max() <= hi + 1e-10

    def test_midpoint_preserved_under_remapping(self):
        ds = self._pi_dataset([[np.pi / 2, np.pi / 2]])
        profile = _linear_profile(3)
        result = NoiseAwarePreprocessor(
            profile, encoding="angle", angle_deadzone=0.1
        ).fit_transform(ds)
        np.testing.assert_allclose(result.data, np.pi / 2, atol=1e-10)

    def test_deadzone_not_applied_for_basis_encoding(self):
        # 'basis' is not in _ANGLE_ENCODINGS so remapping must be skipped.
        ds = self._pi_dataset([[0.0, np.pi]])
        profile = _linear_profile(3)
        result = NoiseAwarePreprocessor(
            profile, encoding="basis", angle_deadzone=0.2
        ).fit_transform(ds)
        # Values should still span the full range {0, π}.
        assert 0.0 in result.data or np.pi in result.data

    def test_zero_deadzone_leaves_values_unchanged(self):
        ds = _make_dataset([1.0, 2.0, 3.0])
        profile = _linear_profile()
        prep = NoiseAwarePreprocessor(profile, encoding="angle", angle_deadzone=0.0)
        result = prep.fit_transform(ds)
        for i, orig_idx in enumerate(prep.permutation_):
            np.testing.assert_array_almost_equal(
                result.data[:, i], ds.data[:, int(orig_idx)]
            )

    def test_remapping_is_monotone(self):
        # After remapping, relative ordering of values within a column should
        # be preserved (linear map is strictly monotone).
        rng = np.random.default_rng(7)
        vals = np.sort(rng.uniform(0, np.pi, 20))
        ds = Dataset(data=vals.reshape(-1, 1), feature_names=["v"])
        profile = _linear_profile(2)
        result = NoiseAwarePreprocessor(
            profile, encoding="angle", angle_deadzone=0.05
        ).fit_transform(ds)
        remapped = result.data[:, 0]
        assert np.all(np.diff(remapped) >= 0)

    def test_pm_pi_encoding_deadzone_scales_symmetrically(self):
        # iqp is in _PM_PI_ENCODINGS → line 349: X = X * (1.0 - 2.0 * deadzone)
        rng = np.random.default_rng(0)
        data = rng.uniform(-np.pi, np.pi, (20, 2))
        ds = Dataset(data=data, feature_names=["a", "b"])
        profile = _linear_profile(3)
        deadzone = 0.1
        prep = NoiseAwarePreprocessor(profile, encoding="iqp", angle_deadzone=deadzone)
        result = prep.fit_transform(ds)
        scale = 1.0 - 2.0 * deadzone
        for i, orig_idx in enumerate(prep.permutation_):
            expected = data[:, int(orig_idx)] * scale
            np.testing.assert_allclose(result.data[:, i], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Integration: top-level import
# ---------------------------------------------------------------------------

def test_importable_from_quprep():
    import quprep as qd
    assert hasattr(qd, "NoiseProfile")
    assert hasattr(qd, "NoiseAwarePreprocessor")


def test_importable_from_preprocess_module():
    from quprep.preprocess import NoiseAwarePreprocessor, NoiseProfile
    assert NoiseProfile is not None
    assert NoiseAwarePreprocessor is not None


# ---------------------------------------------------------------------------
# H-4 regression: zz_feature_map deadzone must use [0, 2π] formula
# ---------------------------------------------------------------------------

class TestZZFeatureMapDeadzone:
    def test_zz_feature_map_in_two_pi_encodings(self):
        # zz_feature_map must not be in _PI_ENCODINGS (was the bug)
        assert "zz_feature_map" not in NoiseAwarePreprocessor._PI_ENCODINGS
        assert "zz_feature_map" in NoiseAwarePreprocessor._TWO_PI_ENCODINGS

    def test_zz_deadzone_output_stays_within_two_pi_range(self):
        # After deadzone remapping, all values must stay in [deadzone*2π, (1-deadzone)*2π]
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 2 * np.pi, (30, 3))
        ds = Dataset(data=data, feature_names=["a", "b", "c"])
        profile = _linear_profile(3)
        deadzone = 0.1
        prep = NoiseAwarePreprocessor(profile, encoding="zz_feature_map", angle_deadzone=deadzone)
        result = prep.fit_transform(ds)
        lo = deadzone * 2.0 * np.pi
        hi = (1.0 - deadzone) * 2.0 * np.pi
        assert result.data.min() >= lo - 1e-10
        assert result.data.max() <= hi + 1e-10

    def test_zz_deadzone_does_not_extrapolate(self):
        # With the old [0,π] formula, a value of 2π would map to 2*(hi-lo)+lo > hi
        # With the correct [0,2π] formula it maps to hi exactly.
        data = np.array([[2 * np.pi, np.pi]])
        ds = Dataset(data=data, feature_names=["a", "b"])
        profile = _linear_profile(2)
        deadzone = 0.05
        prep = NoiseAwarePreprocessor(profile, encoding="zz_feature_map", angle_deadzone=deadzone)
        result = prep.fit_transform(ds)
        hi = (1.0 - deadzone) * 2.0 * np.pi
        assert result.data.max() <= hi + 1e-10
