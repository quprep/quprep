"""Tests for quprep.metrics — expressibility, entanglement, kernel alignment."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.core.dataset import Dataset
from quprep.metrics import (
    EncoderMetrics,
    entanglement_capability,
    expressibility,
    kernel_alignment,
    score_encoding,
)
from quprep.metrics._simulate import MAX_QUBITS, Statevector, statevector_from_encoded

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ds(n: int = 40, d: int = 3, seed: int = 0, with_labels: bool = False) -> Dataset:
    rng = np.random.default_rng(seed)
    data = rng.uniform(0, np.pi, (n, d))
    labels = (rng.integers(0, 2, n) * 2 - 1).astype(float) if with_labels else None
    return Dataset(
        data=data,
        feature_names=[f"f{i}" for i in range(d)],
        feature_types=["continuous"] * d,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Statevector simulator
# ---------------------------------------------------------------------------

class TestStatevector:
    def test_initial_state_is_zero_ket(self):
        sv = Statevector(2)
        assert sv.state[0] == pytest.approx(1.0)
        assert np.allclose(sv.state[1:], 0.0)

    def test_x_gate_flips_qubit(self):
        sv = Statevector(1)
        sv.x(0)
        assert sv.state[1] == pytest.approx(1.0)
        assert sv.state[0] == pytest.approx(0.0)

    def test_hadamard_creates_superposition(self):
        sv = Statevector(1)
        sv.h(0)
        assert abs(sv.state[0]) == pytest.approx(1 / np.sqrt(2), abs=1e-10)
        assert abs(sv.state[1]) == pytest.approx(1 / np.sqrt(2), abs=1e-10)

    def test_cnot_entangles_bell_state(self):
        sv = Statevector(2)
        sv.h(0)
        sv.cnot(0, 1)
        # Bell state: (|00⟩ + |11⟩) / √2
        assert abs(sv.state[0]) == pytest.approx(1 / np.sqrt(2), abs=1e-10)
        assert abs(sv.state[3]) == pytest.approx(1 / np.sqrt(2), abs=1e-10)
        assert abs(sv.state[1]) < 1e-10
        assert abs(sv.state[2]) < 1e-10

    def test_ry_rotation(self):
        sv = Statevector(1)
        sv.ry(np.pi, 0)
        # Ry(π)|0⟩ = |1⟩
        assert abs(sv.state[1]) == pytest.approx(1.0, abs=1e-10)

    def test_statevector_is_normalised(self):
        sv = Statevector(3)
        sv.h(0)
        sv.h(1)
        sv.h(2)
        sv.cnot(0, 1)
        sv.cnot(1, 2)
        assert np.linalg.norm(sv.state) == pytest.approx(1.0, abs=1e-10)

    def test_ising_zz_phase(self):
        # IsingZZ(π)|00⟩ should give phase e^(−iπ/2) = −i
        sv = Statevector(2)
        sv.ising_zz(np.pi, 0, 1)
        assert sv.state[0] == pytest.approx(-1j, abs=1e-10)


# ---------------------------------------------------------------------------
# statevector_from_encoded
# ---------------------------------------------------------------------------

class TestStatevectorFromEncoded:
    def test_angle_encoder_product_state(self):
        from quprep.encode.angle import AngleEncoder
        enc = AngleEncoder()
        row = np.array([np.pi / 2, 0.0, np.pi])
        result = enc.encode(row)
        sv = statevector_from_encoded(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)
        assert len(sv) == 2 ** 3

    def test_amplitude_encoder_statevector(self):
        from quprep.encode.amplitude import AmplitudeEncoder
        enc = AmplitudeEncoder()
        row = np.array([1.0, 0.0, 0.0, 0.0])
        result = enc.encode(row)
        sv = statevector_from_encoded(result)
        assert sv is not None
        assert sv[0] == pytest.approx(1.0, abs=1e-10)

    def test_iqp_encoder_returns_statevector(self):
        from quprep.encode.iqp import IQPEncoder
        enc = IQPEncoder()
        row = np.array([0.5, -0.3, 0.8])
        result = enc.encode(row)
        sv = statevector_from_encoded(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_entangled_angle_returns_statevector(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        enc = EntangledAngleEncoder()
        row = np.array([1.0, 0.5, 1.5])
        result = enc.encode(row)
        sv = statevector_from_encoded(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_angle_ry_statevector_matches_formula(self):
        from quprep.encode.angle import AngleEncoder
        enc = AngleEncoder(rotation="ry")
        theta = np.pi / 3
        result = enc.encode(np.array([theta]))
        sv = statevector_from_encoded(result)
        # Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        expected = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_basis_encoder_returns_statevector(self):
        from quprep.encode.basis import BasisEncoder
        enc = BasisEncoder()
        result = enc.encode(np.array([1.0, 0.0]))
        sv = statevector_from_encoded(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_tensor_product_encoder(self):
        from quprep.encode.tensor_product import TensorProductEncoder
        enc = TensorProductEncoder()
        row = np.array([0.5, 1.0])
        result = enc.encode(row)
        sv = statevector_from_encoded(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_pauli_feature_map_encoder_returns_statevector(self):
        from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
        enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1)
        row = np.array([0.5, 1.0])
        result = enc.encode(row)
        sv = statevector_from_encoded(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_pauli_feature_map_encoder_xx_terms(self):
        from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
        paulis = ["XX", "YY", "XZ", "ZX", "XY", "YX", "YZ", "ZY"]
        enc = PauliFeatureMapEncoder(paulis=paulis, reps=1)
        row = np.array([0.3, 0.7])
        result = enc.encode(row)
        sv = statevector_from_encoded(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_qaoa_problem_encoder_returns_statevector(self):
        from quprep.encode.qaoa_problem import QAOAProblemEncoder
        enc = QAOAProblemEncoder(p=1)
        row = np.array([0.5, 1.0])
        result = enc.encode(row)
        sv = statevector_from_encoded(result)
        assert sv is not None
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-10)

    def test_unsupported_encoding_returns_none(self):
        from quprep.encode.base import EncodedResult
        enc_result = EncodedResult(
            parameters=np.array([]),
            metadata={"encoding": "unsupported_xyz", "n_qubits": 1},
        )
        sv = statevector_from_encoded(enc_result)
        assert sv is None


# ---------------------------------------------------------------------------
# Expressibility
# ---------------------------------------------------------------------------

class TestExpressibility:
    def test_returns_float_for_angle_encoder(self):
        from quprep.encode.angle import AngleEncoder
        ds = _ds(d=2)
        result = expressibility(AngleEncoder(), ds, n_samples=50, seed=0)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_returns_float_for_iqp_encoder(self):
        from quprep.encode.iqp import IQPEncoder
        ds = _ds(d=2)
        result = expressibility(IQPEncoder(), ds, n_samples=50, seed=0)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_returns_none_for_too_many_qubits(self):
        from quprep.encode.angle import AngleEncoder
        # Create a dataset with more features than MAX_QUBITS
        ds = _ds(d=MAX_QUBITS + 2)
        result = expressibility(AngleEncoder(), ds, n_samples=10, seed=0)
        assert result is None

    def test_entangled_encoder_finite_expressibility(self):
        # Entangled encodings should produce a finite, non-negative KL divergence.
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        ds = _ds(d=2)
        result = expressibility(EntangledAngleEncoder(), ds, n_samples=50, seed=0)
        assert result is not None
        assert np.isfinite(result)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Entanglement capability
# ---------------------------------------------------------------------------

class TestEntanglementCapability:
    def test_angle_encoder_zero_entanglement(self):
        from quprep.encode.angle import AngleEncoder
        ds = _ds(d=3)
        ent = entanglement_capability(AngleEncoder(), ds, n_samples=30, seed=0)
        assert ent is not None
        assert ent == pytest.approx(0.0, abs=1e-6)

    def test_iqp_encoder_positive_entanglement(self):
        from quprep.encode.iqp import IQPEncoder
        ds = _ds(d=3)
        ent = entanglement_capability(IQPEncoder(), ds, n_samples=30, seed=0)
        assert ent is not None
        assert ent > 0.0

    def test_entangled_angle_positive_entanglement(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        ds = _ds(d=3)
        ent = entanglement_capability(EntangledAngleEncoder(), ds, n_samples=30, seed=0)
        assert ent is not None
        assert ent > 0.0

    def test_returns_value_in_range(self):
        from quprep.encode.iqp import IQPEncoder
        ds = _ds(d=2)
        ent = entanglement_capability(IQPEncoder(), ds, n_samples=20, seed=0)
        if ent is not None:
            assert 0.0 <= ent <= 1.0 + 1e-6

    def test_returns_none_for_too_large(self):
        from quprep.encode.angle import AngleEncoder
        ds = _ds(d=MAX_QUBITS + 2)
        ent = entanglement_capability(AngleEncoder(), ds, n_samples=5, seed=0)
        assert ent is None


# ---------------------------------------------------------------------------
# Kernel alignment
# ---------------------------------------------------------------------------

class TestKernelAlignment:
    def test_returns_none_without_labels(self):
        from quprep.encode.angle import AngleEncoder
        ds = _ds(d=2, with_labels=False)
        result = kernel_alignment(AngleEncoder(), ds, seed=0)
        assert result is None

    def test_returns_float_with_labels(self):
        from quprep.encode.angle import AngleEncoder
        ds = _ds(d=2, with_labels=True)
        result = kernel_alignment(AngleEncoder(), ds, seed=0)
        assert isinstance(result, float)

    def test_alignment_in_valid_range(self):
        from quprep.encode.iqp import IQPEncoder
        ds = _ds(d=2, with_labels=True)
        result = kernel_alignment(IQPEncoder(), ds, seed=0)
        if result is not None:
            assert -1.0 - 1e-6 <= result <= 1.0 + 1e-6

    def test_perfect_alignment_separable_data(self):
        # Linearly separable: class +1 all at x>π/2, class -1 all at x<π/2.
        n = 20
        X = np.zeros((n, 1))
        X[:n // 2, 0] = np.linspace(0.1, 0.4, n // 2)   # low
        X[n // 2:, 0] = np.linspace(0.6 * np.pi, np.pi, n // 2)  # high
        y = np.array([-1.0] * (n // 2) + [1.0] * (n // 2))
        ds = Dataset(data=X, feature_names=["x"], labels=y)
        from quprep.encode.angle import AngleEncoder
        result = kernel_alignment(AngleEncoder(), ds, seed=0)
        # For this trivially separable data the alignment should be positive
        if result is not None:
            assert result > 0.0

    def test_returns_none_when_too_few_samples(self):
        # n < 4 branch
        from quprep.encode.angle import AngleEncoder
        rng = np.random.default_rng(0)
        data = rng.uniform(0, np.pi, (3, 2))
        labels = np.array([-1.0, 1.0, -1.0])
        ds = Dataset(data=data, feature_names=["a", "b"], labels=labels)
        result = kernel_alignment(AngleEncoder(), ds, seed=0)
        assert result is None

    def test_subsamples_when_n_exceeds_max(self):
        # n > max_samples triggers subsampling
        from quprep.encode.angle import AngleEncoder
        rng = np.random.default_rng(0)
        data = rng.uniform(0, np.pi, (20, 2))
        labels = (rng.integers(0, 2, 20) * 2 - 1).astype(float)
        ds = Dataset(data=data, feature_names=["a", "b"], labels=labels)
        result = kernel_alignment(AngleEncoder(), ds, max_samples=5, seed=0)
        # result may be float or None, but subsampling code path is exercised
        assert result is None or isinstance(result, float)

    def test_handles_2d_labels(self):
        # 2D label array → y = y[:, 0] branch
        from quprep.encode.angle import AngleEncoder
        rng = np.random.default_rng(0)
        data = rng.uniform(0, np.pi, (20, 2))
        labels_2d = (rng.integers(0, 2, (20, 1)) * 2 - 1).astype(float)
        ds = Dataset(data=data, feature_names=["a", "b"], labels=labels_2d)
        result = kernel_alignment(AngleEncoder(), ds, seed=0)
        assert result is None or isinstance(result, float)

    def test_returns_none_when_sv_is_none(self):
        # encoder.encode() succeeds but statevector_from_encoded returns None
        from quprep.encode.base import BaseEncoder, EncodedResult

        class _UnsupportedEncoder(BaseEncoder):
            encoding = "unsupported_xyz"

            def encode(self, x):
                return EncodedResult(
                    parameters=np.zeros(len(x)),
                    metadata={"encoding": "unsupported_xyz", "n_qubits": len(x)},
                )

            @property
            def n_qubits(self):
                return None

            @property
            def depth(self):
                return None

        ds = _ds(d=2, with_labels=True)
        result = kernel_alignment(_UnsupportedEncoder(), ds, seed=0)
        assert result is None


# ---------------------------------------------------------------------------
# score_encoding + EncoderMetrics
# ---------------------------------------------------------------------------

class TestScoreEncoding:
    def test_returns_encoder_metrics(self):
        from quprep.encode.angle import AngleEncoder
        ds = _ds(d=2, with_labels=True)
        m = score_encoding(AngleEncoder(), ds, n_samples=30, seed=0)
        assert isinstance(m, EncoderMetrics)

    def test_fields_populated(self):
        from quprep.encode.angle import AngleEncoder
        ds = _ds(d=2)
        m = score_encoding(AngleEncoder(), ds, n_samples=20, seed=0)
        assert m.encoding is not None
        assert m.n_qubits > 0
        assert m.expressibility is not None
        assert m.entanglement_capability is not None

    def test_kernel_alignment_none_without_labels(self):
        from quprep.encode.iqp import IQPEncoder
        ds = _ds(d=2, with_labels=False)
        m = score_encoding(IQPEncoder(), ds, n_samples=20, seed=0)
        assert m.kernel_alignment is None

    def test_str_representation(self):
        from quprep.encode.angle import AngleEncoder
        ds = _ds(d=2)
        m = score_encoding(AngleEncoder(), ds, n_samples=20, seed=0)
        s = str(m)
        assert "expressibility" in s
        assert "entanglement_capability" in s

    def test_score_encoding_auto_fits_encoder_without_w(self):
        # score_encoding has an elif branch for encoders that have fit() but no _W
        # attribute at all (lines 182-186 in kernel.py). Exercise it with a
        # custom encoder that wraps AngleEncoder but doesn't set _W or _fitted.
        from quprep.encode.angle import AngleEncoder
        from quprep.encode.base import BaseEncoder

        _inner = AngleEncoder()

        class _FitOnlyEncoder(BaseEncoder):
            encoding = "angle"

            def fit(self, dataset):
                pass  # no _W or _fitted set — stays absent

            def encode(self, x):
                return _inner.encode(x)

            @property
            def n_qubits(self):
                return None

            @property
            def depth(self):
                return None

        enc = _FitOnlyEncoder()
        assert not hasattr(enc, "_W")  # confirms elif branch will fire
        ds = _ds(d=2, n=20)
        m = score_encoding(enc, ds, n_samples=20, seed=0)
        assert isinstance(m, EncoderMetrics)


# ---------------------------------------------------------------------------
# recommend() use_metrics integration
# ---------------------------------------------------------------------------

class TestRecommendWithMetrics:
    def test_use_metrics_returns_recommendation(self):
        from quprep.core.recommender import recommend
        ds = _ds(d=3, with_labels=True)
        rec = recommend(ds, task="classification", use_metrics=True)
        assert rec.method in {"angle", "amplitude", "basis", "iqp", "reupload",
                              "entangled_angle", "hamiltonian", "zz_feature_map",
                              "pauli_feature_map", "random_fourier", "tensor_product",
                              "qaoa_problem"}

    def test_use_metrics_does_not_crash_without_labels(self):
        from quprep.core.recommender import recommend
        ds = _ds(d=3, with_labels=False)
        rec = recommend(ds, task="regression", use_metrics=True)
        assert rec.score is not None

    def test_use_metrics_skipped_for_large_feature_count(self):
        # With many features, metrics should be silently skipped
        from quprep.core.recommender import recommend
        ds = _ds(d=MAX_QUBITS + 2)
        # Should not raise
        rec = recommend(ds, task="classification", use_metrics=True)
        assert rec.method is not None


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_importable_from_quprep():
    import quprep as qd
    assert hasattr(qd, "expressibility")
    assert hasattr(qd, "entanglement_capability")
    assert hasattr(qd, "kernel_alignment")
    assert hasattr(qd, "EncoderMetrics")
    assert hasattr(qd, "score_encoding")
