"""Tests for quantum encoders.

Property-based tests (via hypothesis) are required for all encoders.
Key invariants:
  - AmplitudeEncoder output must satisfy ‖amplitudes‖₂ = 1.
  - AngleEncoder output must have values in the correct rotation range.
  - All encoders must be deterministic (same input → same output).
"""

import hypothesis.extra.numpy as npst
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.base import EncodedResult
from quprep.encode.basis import BasisEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy of x (safe for near-zero vectors)."""
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        x = x.copy()
        x[0] = 1.0
        norm = 1.0
    return x / norm


def _unit_vectors(min_dim=1, max_dim=16):
    """Hypothesis strategy: random L2-normalized float vectors."""
    return (
        npst.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=min_dim, max_value=max_dim),
            elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        )
        .filter(lambda x: np.linalg.norm(x) > 1e-10)
        .map(_l2_normalize)
    )


# ---------------------------------------------------------------------------
# AngleEncoder
# ---------------------------------------------------------------------------

class TestAngleEncoder:
    def test_invalid_rotation_raises(self):
        with pytest.raises(ValueError, match="rotation"):
            AngleEncoder(rotation="rq")

    @pytest.mark.parametrize("rotation", ["ry", "rx", "rz"])
    def test_valid_rotations_accepted(self, rotation):
        enc = AngleEncoder(rotation=rotation)
        assert enc.rotation == rotation

    def test_returns_encoded_result(self):
        enc = AngleEncoder()
        x = np.array([0.1, 0.5, 1.2])
        result = enc.encode(x)
        assert isinstance(result, EncodedResult)

    def test_output_shape(self):
        enc = AngleEncoder()
        x = np.array([0.0, 0.5, 1.0, 1.5])
        result = enc.encode(x)
        assert result.parameters.shape == (4,)

    def test_parameters_match_input(self):
        enc = AngleEncoder()
        x = np.array([0.1, 0.5, 1.2])
        result = enc.encode(x)
        np.testing.assert_array_equal(result.parameters, x)

    def test_metadata_n_qubits(self):
        enc = AngleEncoder()
        x = np.array([0.1, 0.5, 1.2])
        result = enc.encode(x)
        assert result.metadata["n_qubits"] == 3
        assert result.metadata["depth"] == 1
        assert result.metadata["encoding"] == "angle"
        assert result.metadata["rotation"] == "ry"

    def test_deterministic(self):
        enc = AngleEncoder()
        x = np.array([0.3, 0.7, 1.1])
        r1 = enc.encode(x)
        r2 = enc.encode(x)
        np.testing.assert_array_equal(r1.parameters, r2.parameters)

    def test_2d_input_raises(self):
        enc = AngleEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_empty_input_raises(self):
        enc = AngleEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.array([]))

    def test_list_input_accepted(self):
        enc = AngleEncoder()
        result = enc.encode([0.1, 0.5, 1.2])
        assert result.parameters.shape == (3,)

    @given(
        x=npst.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=20),
            elements=st.floats(0.0, np.pi, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=100)
    def test_property_output_shape_matches_input(self, x):
        enc = AngleEncoder(rotation="ry")
        result = enc.encode(x)
        assert result.parameters.shape == x.shape

    @given(
        x=npst.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=20),
            elements=st.floats(0.0, np.pi, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50)
    def test_property_deterministic(self, x):
        enc = AngleEncoder()
        r1 = enc.encode(x)
        r2 = enc.encode(x)
        np.testing.assert_array_equal(r1.parameters, r2.parameters)


# ---------------------------------------------------------------------------
# AmplitudeEncoder
# ---------------------------------------------------------------------------

class TestAmplitudeEncoder:
    def test_unit_norm_invariant(self):
        """Core invariant: output must always be unit norm."""
        enc = AmplitudeEncoder()
        x = _l2_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        result = enc.encode(x)
        assert np.isclose(np.linalg.norm(result.parameters), 1.0, atol=1e-10)

    def test_non_unit_norm_input_raises(self):
        enc = AmplitudeEncoder()
        x = np.array([1.0, 2.0, 3.0, 4.0])  # not normalized
        with pytest.raises(ValueError, match="L2-normalized"):
            enc.encode(x)

    def test_power_of_two_dim_no_padding(self):
        enc = AmplitudeEncoder()
        x = _l2_normalize(np.array([1.0, 2.0, 3.0, 4.0]))  # d=4 = 2²
        result = enc.encode(x)
        assert result.metadata["padded"] is False
        assert result.metadata["n_qubits"] == 2
        assert len(result.parameters) == 4

    def test_non_power_of_two_pads(self):
        enc = AmplitudeEncoder(pad=True)
        x = _l2_normalize(np.array([1.0, 2.0, 3.0]))  # d=3 → padded to 4
        result = enc.encode(x)
        assert result.metadata["padded"] is True
        assert len(result.parameters) == 4
        assert np.isclose(np.linalg.norm(result.parameters), 1.0, atol=1e-10)

    def test_non_power_of_two_no_pad_raises(self):
        enc = AmplitudeEncoder(pad=False)
        x = _l2_normalize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="power of two"):
            enc.encode(x)

    def test_single_element(self):
        enc = AmplitudeEncoder()
        x = np.array([1.0])  # d=1 = 2⁰, already normalized
        result = enc.encode(x)
        assert result.metadata["n_qubits"] == 0
        assert np.isclose(np.linalg.norm(result.parameters), 1.0, atol=1e-10)

    def test_metadata_fields(self):
        enc = AmplitudeEncoder()
        x = _l2_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
        result = enc.encode(x)
        assert result.metadata["encoding"] == "amplitude"
        assert "n_qubits" in result.metadata
        assert "original_dim" in result.metadata

    def test_2d_input_raises(self):
        enc = AmplitudeEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_empty_input_raises(self):
        enc = AmplitudeEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.array([]))

    def test_deterministic(self):
        enc = AmplitudeEncoder()
        x = _l2_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        r1 = enc.encode(x)
        r2 = enc.encode(x)
        np.testing.assert_array_equal(r1.parameters, r2.parameters)

    @given(x=_unit_vectors(min_dim=1, max_dim=8))
    @settings(max_examples=100)
    def test_property_output_always_unit_norm(self, x):
        """Core property: ‖output‖₂ = 1 for any valid input."""
        enc = AmplitudeEncoder(pad=True)
        result = enc.encode(x)
        assert np.isclose(np.linalg.norm(result.parameters), 1.0, atol=1e-10)

    @given(x=_unit_vectors(min_dim=1, max_dim=8))
    @settings(max_examples=50)
    def test_property_output_length_is_power_of_two(self, x):
        enc = AmplitudeEncoder(pad=True)
        result = enc.encode(x)
        n = len(result.parameters)
        assert n & (n - 1) == 0  # power of two iff n & (n-1) == 0

    @given(x=_unit_vectors(min_dim=1, max_dim=8))
    @settings(max_examples=50)
    def test_property_deterministic(self, x):
        enc = AmplitudeEncoder(pad=True)
        r1 = enc.encode(x)
        r2 = enc.encode(x)
        np.testing.assert_array_equal(r1.parameters, r2.parameters)


# ---------------------------------------------------------------------------
# BasisEncoder
# ---------------------------------------------------------------------------

class TestBasisEncoder:
    def test_binary_input_passthrough(self):
        enc = BasisEncoder()
        x = np.array([0.0, 1.0, 0.0, 1.0])
        result = enc.encode(x)
        np.testing.assert_array_equal(result.parameters, [0.0, 1.0, 0.0, 1.0])

    def test_continuous_input_binarized(self):
        enc = BasisEncoder(threshold=0.5)
        x = np.array([0.2, 0.6, 0.5, 0.9])
        result = enc.encode(x)
        # 0.5 >= 0.5 is True → 1
        np.testing.assert_array_equal(result.parameters, [0.0, 1.0, 1.0, 1.0])

    def test_custom_threshold(self):
        enc = BasisEncoder(threshold=0.0)
        x = np.array([-1.0, 0.0, 1.0])
        result = enc.encode(x)
        np.testing.assert_array_equal(result.parameters, [0.0, 1.0, 1.0])

    def test_output_is_binary(self):
        enc = BasisEncoder()
        x = np.random.default_rng(0).uniform(-5, 5, size=20)
        result = enc.encode(x)
        unique = np.unique(result.parameters)
        assert set(unique).issubset({0.0, 1.0})

    def test_output_shape(self):
        enc = BasisEncoder()
        x = np.array([0.1, 0.9, 0.4, 0.7, 0.3])
        result = enc.encode(x)
        assert result.parameters.shape == (5,)

    def test_metadata_fields(self):
        enc = BasisEncoder(threshold=0.3)
        x = np.array([0.1, 0.5])
        result = enc.encode(x)
        assert result.metadata["encoding"] == "basis"
        assert result.metadata["threshold"] == 0.3
        assert result.metadata["n_qubits"] == 2
        assert result.metadata["depth"] == 1

    def test_returns_encoded_result(self):
        enc = BasisEncoder()
        result = enc.encode(np.array([0.2, 0.8]))
        assert isinstance(result, EncodedResult)

    def test_2d_input_raises(self):
        enc = BasisEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.array([[0.1, 0.9]]))

    def test_empty_input_raises(self):
        enc = BasisEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.array([]))

    def test_deterministic(self):
        enc = BasisEncoder()
        x = np.array([0.3, 0.7, 0.1, 0.9])
        r1 = enc.encode(x)
        r2 = enc.encode(x)
        np.testing.assert_array_equal(r1.parameters, r2.parameters)

    def test_all_zeros(self):
        enc = BasisEncoder(threshold=0.5)
        x = np.array([0.0, 0.1, 0.2, 0.4])
        result = enc.encode(x)
        np.testing.assert_array_equal(result.parameters, [0.0, 0.0, 0.0, 0.0])

    def test_all_ones(self):
        enc = BasisEncoder(threshold=0.5)
        x = np.array([0.5, 0.7, 0.9, 1.0])
        result = enc.encode(x)
        np.testing.assert_array_equal(result.parameters, [1.0, 1.0, 1.0, 1.0])

    @given(
        x=npst.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=32),
            elements=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
        ),
        threshold=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_property_output_is_always_binary(self, x, threshold):
        enc = BasisEncoder(threshold=threshold)
        result = enc.encode(x)
        assert set(np.unique(result.parameters)).issubset({0.0, 1.0})

    @given(
        x=npst.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=32),
            elements=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50)
    def test_property_output_shape_matches_input(self, x):
        enc = BasisEncoder()
        result = enc.encode(x)
        assert result.parameters.shape == x.shape


# ---------------------------------------------------------------------------
# Phase 2 stubs — verify they raise NotImplementedError cleanly
# ---------------------------------------------------------------------------

class TestPhase2Encoders:
    def test_iqp_encodes(self):
        from quprep.encode.iqp import IQPEncoder
        result = IQPEncoder().encode(np.array([0.1, 0.2, 0.3]))
        assert result.metadata["encoding"] == "iqp"
        assert result.metadata["n_qubits"] == 3
        # parameters = 3 features + 3 pairs
        assert len(result.parameters) == 3 + 3

    def test_iqp_invalid_reps(self):
        from quprep.encode.iqp import IQPEncoder
        with pytest.raises(ValueError):
            IQPEncoder(reps=0)

    def test_iqp_empty_input_raises(self):
        from quprep.encode.iqp import IQPEncoder
        with pytest.raises(ValueError):
            IQPEncoder().encode(np.array([]))

    def test_reupload_encodes(self):
        from quprep.encode.reupload import ReUploadEncoder
        result = ReUploadEncoder(layers=3).encode(np.array([0.1, 0.2]))
        assert result.metadata["encoding"] == "reupload"
        assert result.metadata["layers"] == 3
        assert len(result.parameters) == 2

    def test_reupload_invalid_rotation(self):
        from quprep.encode.reupload import ReUploadEncoder
        with pytest.raises(ValueError):
            ReUploadEncoder(rotation="rw")

    def test_reupload_invalid_layers(self):
        from quprep.encode.reupload import ReUploadEncoder
        with pytest.raises(ValueError):
            ReUploadEncoder(layers=0)

    def test_hamiltonian_encodes(self):
        from quprep.encode.hamiltonian import HamiltonianEncoder
        result = HamiltonianEncoder(evolution_time=1.0, trotter_steps=4).encode(
            np.array([0.5, 1.0])
        )
        assert result.metadata["encoding"] == "hamiltonian"
        assert result.metadata["trotter_steps"] == 4
        assert len(result.parameters) == 2

    def test_hamiltonian_invalid_trotter(self):
        from quprep.encode.hamiltonian import HamiltonianEncoder
        with pytest.raises(ValueError):
            HamiltonianEncoder(trotter_steps=0)

    def test_hamiltonian_invalid_time(self):
        from quprep.encode.hamiltonian import HamiltonianEncoder
        with pytest.raises(ValueError):
            HamiltonianEncoder(evolution_time=-1.0)
