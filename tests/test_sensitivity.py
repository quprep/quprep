"""Tests for encoding_sensitivity."""

from __future__ import annotations

import numpy as np

import quprep as qd
from quprep.core.dataset import Dataset


def _ds(data, feature_names=None):
    n, d = data.shape
    return Dataset(
        data=data.astype(np.float64),
        feature_names=feature_names or [f"f{i}" for i in range(d)],
        feature_types=["continuous"] * d,
    )


class TestEncodingSensitivity:
    def test_returns_one_score_per_feature(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, np.pi, (30, 4)))
        enc = qd.AngleEncoder(rotation="ry")
        result = qd.encoding_sensitivity(enc, ds, n_samples=10, seed=0)
        assert len(result.scores) == 4
        assert len(result.feature_names) == 4

    def test_scores_non_negative(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, np.pi, (20, 3)))
        enc = qd.AngleEncoder(rotation="ry")
        result = qd.encoding_sensitivity(enc, ds, n_samples=10, seed=0)
        assert (result.scores >= 0).all()

    def test_uses_feature_names(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, np.pi, (20, 3)), feature_names=["a", "b", "c"])
        enc = qd.AngleEncoder()
        result = qd.encoding_sensitivity(enc, ds, n_samples=5)
        assert result.feature_names == ["a", "b", "c"]

    def test_most_sensitive(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, np.pi, (20, 4)))
        enc = qd.AngleEncoder()
        result = qd.encoding_sensitivity(enc, ds, n_samples=5)
        top = result.most_sensitive(n=2)
        assert len(top) == 2
        assert all(isinstance(name, str) and isinstance(score, float) for name, score in top)

    def test_unsupported_encoder_returns_zeros(self):
        # AmplitudeEncoder with > 12 qubits — statevector_from_encoded returns None
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (10, 16))
        n, d = data.shape
        ds = Dataset(
            data=data.astype(float),
            feature_names=[f"f{i}" for i in range(d)],
            feature_types=["continuous"] * d,
        )
        enc = qd.AmplitudeEncoder()
        result = qd.encoding_sensitivity(enc, ds, n_samples=5)
        assert (result.scores == 0).all()

    def test_str_repr(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, np.pi, (10, 3)))
        enc = qd.AngleEncoder()
        result = qd.encoding_sensitivity(enc, ds, n_samples=5)
        assert "SensitivityResult" in str(result)
        assert "SensitivityResult" in repr(result)

    def test_epsilon_stored(self):
        rng = np.random.default_rng(0)
        ds = _ds(rng.uniform(0, np.pi, (10, 3)))
        enc = qd.AngleEncoder()
        result = qd.encoding_sensitivity(enc, ds, epsilon=0.05, n_samples=5)
        assert result.epsilon == 0.05
