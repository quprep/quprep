"""Tests for quantum encoders.

Property-based tests (via hypothesis) are required for all encoders.
Key invariants:
  - AmplitudeEncoder output must satisfy ‖amplitudes‖₂ = 1.
  - AngleEncoder output must have values in the correct rotation range.
  - All encoders must be deterministic (same input → same output).
"""

import math

import hypothesis.extra.numpy as npst
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.base import EncodedResult
from quprep.encode.basis import BasisEncoder
from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
from quprep.encode.qaoa_problem import QAOAProblemEncoder
from quprep.encode.random_fourier import RandomFourierEncoder
from quprep.encode.tensor_product import TensorProductEncoder
from quprep.encode.zz_feature_map import ZZFeatureMapEncoder
from quprep.validation import QuPrepWarning

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

    def test_non_power_of_two_emits_warning(self):
        enc = AmplitudeEncoder(pad=True)
        x = _l2_normalize(np.array([1.0, 2.0, 3.0]))  # d=3 → padded to 4
        with pytest.warns(QuPrepWarning, match="zero-padded"):
            enc.encode(x)

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


# ---------------------------------------------------------------------------
# EntangledAngleEncoder
# ---------------------------------------------------------------------------

class TestEntangledAngleEncoder:
    def test_encodes_basic(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        x = np.array([0.1, 0.5, 0.9])
        result = EntangledAngleEncoder().encode(x)
        assert result.metadata["encoding"] == "entangled_angle"
        assert result.metadata["n_qubits"] == 3
        np.testing.assert_array_equal(result.parameters, x)

    def test_parameters_are_copy(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        x = np.array([0.1, 0.5, 0.9])
        result = EntangledAngleEncoder().encode(x)
        x[0] = 999.0
        assert result.parameters[0] != 999.0

    def test_linear_cnot_pairs(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        result = EntangledAngleEncoder(entanglement="linear").encode(np.ones(4))
        assert result.metadata["cnot_pairs"] == [(0, 1), (1, 2), (2, 3)]

    def test_circular_cnot_pairs(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        result = EntangledAngleEncoder(entanglement="circular").encode(np.ones(4))
        assert result.metadata["cnot_pairs"] == [(0, 1), (1, 2), (2, 3), (3, 0)]

    def test_full_cnot_pairs(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        result = EntangledAngleEncoder(entanglement="full").encode(np.ones(3))
        assert result.metadata["cnot_pairs"] == [(0, 1), (0, 2), (1, 2)]

    def test_single_qubit_no_cnot_pairs(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        result = EntangledAngleEncoder().encode(np.array([0.5]))
        assert result.metadata["cnot_pairs"] == []

    def test_layers_stored_in_metadata(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        result = EntangledAngleEncoder(layers=3).encode(np.ones(4))
        assert result.metadata["layers"] == 3

    def test_rotation_stored_in_metadata(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        result = EntangledAngleEncoder(rotation="rx").encode(np.ones(3))
        assert result.metadata["rotation"] == "rx"

    def test_invalid_rotation_raises(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        with pytest.raises(ValueError, match="rotation"):
            EntangledAngleEncoder(rotation="rw")

    def test_invalid_layers_raises(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        with pytest.raises(ValueError, match="layers"):
            EntangledAngleEncoder(layers=0)

    def test_invalid_entanglement_raises(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        with pytest.raises(ValueError, match="entanglement"):
            EntangledAngleEncoder(entanglement="star")

    def test_empty_input_raises(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        with pytest.raises(ValueError):
            EntangledAngleEncoder().encode(np.array([]))

    def test_2d_input_raises(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        with pytest.raises(ValueError):
            EntangledAngleEncoder().encode(np.ones((3, 3)))

    def test_deterministic(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        x = np.array([0.1, 0.5, 0.9])
        enc = EntangledAngleEncoder()
        assert np.array_equal(enc.encode(x).parameters, enc.encode(x).parameters)

    def test_qasm_export(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        from quprep.export.qasm_export import QASMExporter
        result = EntangledAngleEncoder(layers=2, entanglement="linear").encode(np.ones(3))
        qasm = QASMExporter().export(result)
        assert "OPENQASM 3.0;" in qasm
        assert qasm.count("cx") == 2 * 2  # 2 pairs × 2 layers
        assert qasm.count("ry(") == 3 * 2  # 3 qubits × 2 layers

    def test_pipeline_integration(self):
        from quprep.core.dataset import Dataset
        from quprep.core.pipeline import Pipeline
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        from quprep.export.qasm_export import QASMExporter
        data = np.random.default_rng(0).standard_normal((10, 4))
        ds = Dataset(data=data, feature_names=["a","b","c","d"],
                     feature_types=["continuous"]*4, metadata={}, categorical_data={})
        result = Pipeline(
            encoder=EntangledAngleEncoder(), exporter=QASMExporter()
        ).fit_transform(ds)
        assert len(result.circuits) == 10

    def test_prepare_one_liner(self):
        import quprep
        data = np.random.default_rng(0).standard_normal((5, 3))
        result = quprep.prepare(data, encoding="entangled_angle")
        assert len(result.circuits) == 5

# ---------------------------------------------------------------------------
# ZZFeatureMapEncoder
# ---------------------------------------------------------------------------

class TestZZFeatureMapEncoder:
    def test_invalid_reps_raises(self):
        with pytest.raises(ValueError, match="reps"):
            ZZFeatureMapEncoder(reps=0)

    def test_output_shape(self):
        enc = ZZFeatureMapEncoder(reps=1)
        x = np.linspace(0, 2 * math.pi, 4)
        result = enc.encode(x)
        assert result.parameters.shape == (4,)

    def test_metadata_fields(self):
        enc = ZZFeatureMapEncoder(reps=2)
        x = np.array([0.5, 1.0, 1.5])
        result = enc.encode(x)
        assert result.metadata["encoding"] == "zz_feature_map"
        assert result.metadata["n_qubits"] == 3
        assert result.metadata["reps"] == 2
        assert len(result.metadata["single_angles"]) == 3
        assert len(result.metadata["pair_angles"]) == 3   # C(3,2) = 3
        assert len(result.metadata["pairs"]) == 3

    def test_single_angles_formula(self):
        enc = ZZFeatureMapEncoder(reps=1)
        x = np.array([0.0, math.pi])
        result = enc.encode(x)
        expected = [2.0 * (math.pi - 0.0), 2.0 * (math.pi - math.pi)]
        np.testing.assert_allclose(result.metadata["single_angles"], expected)

    def test_pair_angles_formula(self):
        enc = ZZFeatureMapEncoder(reps=1)
        x = np.array([1.0, 2.0])
        result = enc.encode(x)
        expected_pair = 2.0 * (math.pi - 1.0) * (math.pi - 2.0)
        np.testing.assert_allclose(result.metadata["pair_angles"][0], expected_pair)

    def test_empty_input_raises(self):
        enc = ZZFeatureMapEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.array([]))

    def test_2d_input_raises(self):
        enc = ZZFeatureMapEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.ones((2, 3)))

    def test_deterministic(self):
        enc = ZZFeatureMapEncoder(reps=3)
        x = np.random.default_rng(0).random(5)
        r1 = enc.encode(x)
        r2 = enc.encode(x)
        np.testing.assert_array_equal(r1.parameters, r2.parameters)

    def test_depth_scales_with_reps(self):
        x = np.ones(4)
        d1 = ZZFeatureMapEncoder(reps=1).encode(x).metadata["depth"]
        d2 = ZZFeatureMapEncoder(reps=2).encode(x).metadata["depth"]
        assert d2 == 2 * d1


# ---------------------------------------------------------------------------
# PauliFeatureMapEncoder
# ---------------------------------------------------------------------------

class TestPauliFeatureMapEncoder:
    def test_default_paulis(self):
        enc = PauliFeatureMapEncoder()
        assert enc.paulis == ["Z", "ZZ"]

    def test_invalid_pauli_raises(self):
        with pytest.raises(ValueError, match="Unknown Pauli"):
            PauliFeatureMapEncoder(paulis=["Z", "ZZZZ"])

    def test_invalid_reps_raises(self):
        with pytest.raises(ValueError, match="reps"):
            PauliFeatureMapEncoder(reps=0)

    def test_metadata_zz(self):
        enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=2)
        x = np.array([0.5, 1.0, 1.5])
        result = enc.encode(x)
        assert result.metadata["encoding"] == "pauli_feature_map"
        assert result.metadata["n_qubits"] == 3
        assert "Z" in result.metadata["single_terms"]
        assert "ZZ" in result.metadata["pair_terms"]

    def test_single_only_paulis(self):
        enc = PauliFeatureMapEncoder(paulis=["X", "Y"], reps=1)
        x = np.array([0.3, 0.7])
        result = enc.encode(x)
        assert "X" in result.metadata["single_terms"]
        assert "Y" in result.metadata["single_terms"]
        assert result.metadata["pair_terms"] == {}

    def test_pair_angles_formula(self):
        enc = PauliFeatureMapEncoder(paulis=["ZZ"], reps=1)
        x = np.array([1.0, 2.0])
        result = enc.encode(x)
        entries = result.metadata["pair_terms"]["ZZ"]
        i, j, angle = entries[0]
        np.testing.assert_allclose(angle, 2.0 * 1.0 * 2.0)

    def test_depth_quadratic_with_pairs(self):
        enc_pair = PauliFeatureMapEncoder(paulis=["ZZ"], reps=1)
        enc_single = PauliFeatureMapEncoder(paulis=["Z"], reps=1)
        assert "d²" in enc_pair.depth
        assert "d²" not in enc_single.depth

    def test_deterministic(self):
        enc = PauliFeatureMapEncoder()
        x = np.random.default_rng(7).random(4)
        r1 = enc.encode(x)
        r2 = enc.encode(x)
        np.testing.assert_array_equal(r1.parameters, r2.parameters)


# ---------------------------------------------------------------------------
# RandomFourierEncoder
# ---------------------------------------------------------------------------

class TestRandomFourierEncoder:
    def test_invalid_n_components_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            RandomFourierEncoder(n_components=0)

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            RandomFourierEncoder(gamma=-1.0)

    def test_encode_before_fit_raises(self):
        enc = RandomFourierEncoder()
        with pytest.raises(RuntimeError, match="fitted"):
            enc.encode(np.array([1.0, 2.0]))

    def test_output_shape_equals_n_components(self):
        enc = RandomFourierEncoder(n_components=12, random_state=0)
        X = np.random.default_rng(0).random((20, 5))
        enc.fit(X)
        result = enc.encode(X[0])
        assert result.parameters.shape == (12,)

    def test_n_qubits_property(self):
        enc = RandomFourierEncoder(n_components=6)
        assert enc.n_qubits == 6

    def test_depth_property(self):
        enc = RandomFourierEncoder()
        assert enc.depth == 1

    def test_output_range(self):
        enc = RandomFourierEncoder(n_components=16, random_state=42)
        X = np.random.default_rng(0).random((50, 4))
        enc.fit(X)
        for row in X[:10]:
            result = enc.encode(row)
            assert np.all(result.parameters >= -1e-9)
            assert np.all(result.parameters <= math.pi + 1e-9)

    def test_reproducible_with_random_state(self):
        X = np.random.default_rng(0).random((20, 4))
        enc1 = RandomFourierEncoder(n_components=8, random_state=99)
        enc2 = RandomFourierEncoder(n_components=8, random_state=99)
        enc1.fit(X)
        enc2.fit(X)
        r1 = enc1.encode(X[0])
        r2 = enc2.encode(X[0])
        np.testing.assert_array_equal(r1.parameters, r2.parameters)

    def test_different_seeds_different_output(self):
        X = np.random.default_rng(0).random((20, 4))
        enc1 = RandomFourierEncoder(n_components=8, random_state=1)
        enc2 = RandomFourierEncoder(n_components=8, random_state=2)
        enc1.fit(X)
        enc2.fit(X)
        r1 = enc1.encode(X[0])
        r2 = enc2.encode(X[0])
        assert not np.allclose(r1.parameters, r2.parameters)

    def test_metadata_fields(self):
        enc = RandomFourierEncoder(n_components=4, gamma=0.5, random_state=0)
        enc.fit(np.ones((5, 3)))
        result = enc.encode(np.ones(3))
        assert result.metadata["encoding"] == "random_fourier"
        assert result.metadata["n_qubits"] == 4
        assert result.metadata["gamma"] == 0.5

    def test_fit_with_1d_input(self):
        enc = RandomFourierEncoder(n_components=4, random_state=0)
        enc.fit(np.ones(3))  # should not raise
        result = enc.encode(np.ones(3))
        assert result.parameters.shape == (4,)


# ---------------------------------------------------------------------------
# TensorProductEncoder
# ---------------------------------------------------------------------------

class TestTensorProductEncoder:
    def test_even_input(self):
        enc = TensorProductEncoder()
        x = np.array([0.5, 1.0, 1.5, 2.0])
        result = enc.encode(x)
        assert result.metadata["n_qubits"] == 2
        assert len(result.parameters) == 4

    def test_odd_input_padded(self):
        enc = TensorProductEncoder()
        x = np.array([0.5, 1.0, 1.5])  # d=3 → n_qubits=2, pad rz of last qubit=0
        result = enc.encode(x)
        assert result.metadata["n_qubits"] == 2
        assert len(result.parameters) == 4

    def test_single_feature_one_qubit(self):
        enc = TensorProductEncoder()
        x = np.array([0.7])
        result = enc.encode(x)
        assert result.metadata["n_qubits"] == 1

    def test_ry_rz_separation(self):
        enc = TensorProductEncoder()
        x = np.array([0.1, 0.2, 0.3, 0.4])
        result = enc.encode(x)
        ry = result.metadata["ry_angles"]
        rz = result.metadata["rz_angles"]
        np.testing.assert_allclose(ry, [0.1, 0.3])
        np.testing.assert_allclose(rz, [0.2, 0.4])

    def test_parameters_interleaved(self):
        enc = TensorProductEncoder()
        x = np.array([0.1, 0.2, 0.3, 0.4])
        result = enc.encode(x)
        # [ry_0, rz_0, ry_1, rz_1]
        np.testing.assert_allclose(result.parameters, [0.1, 0.2, 0.3, 0.4])

    def test_depth_is_2(self):
        enc = TensorProductEncoder()
        assert enc.depth == 2

    def test_metadata_encoding_name(self):
        enc = TensorProductEncoder()
        result = enc.encode(np.array([1.0, 2.0]))
        assert result.metadata["encoding"] == "tensor_product"

    def test_empty_input_raises(self):
        enc = TensorProductEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.array([]))

    def test_2d_input_raises(self):
        enc = TensorProductEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.ones((2, 3)))

    def test_deterministic(self):
        enc = TensorProductEncoder()
        x = np.random.default_rng(0).random(6)
        r1 = enc.encode(x)
        r2 = enc.encode(x)
        np.testing.assert_array_equal(r1.parameters, r2.parameters)

    def test_odd_last_rz_is_zero(self):
        enc = TensorProductEncoder()
        x = np.array([0.5, 1.0, 1.5])  # d=3 → last rz should be 0
        result = enc.encode(x)
        np.testing.assert_allclose(result.metadata["rz_angles"][1], 0.0)


# ---------------------------------------------------------------------------
# Encoder properties — n_qubits and depth (coverage)
# ---------------------------------------------------------------------------

class TestEncoderProperties:
    """Ensure n_qubits and depth properties are exercised for every encoder."""

    def test_angle_properties(self):
        enc = AngleEncoder()
        assert enc.n_qubits is None
        assert enc.depth == 1

    def test_basis_properties(self):
        enc = BasisEncoder()
        assert enc.n_qubits is None
        assert enc.depth == 1

    def test_amplitude_properties(self):
        from quprep.encode.amplitude import AmplitudeEncoder
        enc = AmplitudeEncoder()
        assert enc.n_qubits is None

    def test_iqp_properties(self):
        from quprep.encode.iqp import IQPEncoder
        enc = IQPEncoder(reps=2)
        assert enc.n_qubits is None
        assert enc.depth == "O(d² · reps)"

    def test_reupload_properties(self):
        from quprep.encode.reupload import ReUploadEncoder
        enc = ReUploadEncoder()
        assert enc.n_qubits is None

    def test_hamiltonian_properties(self):
        from quprep.encode.hamiltonian import HamiltonianEncoder
        enc = HamiltonianEncoder()
        assert enc.n_qubits is None

    def test_entangled_angle_properties(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        enc = EntangledAngleEncoder()
        assert enc.n_qubits is None

    def test_zz_feature_map_properties(self):
        enc = ZZFeatureMapEncoder()
        assert enc.n_qubits is None
        assert enc.depth == "O(d² · reps)"

    def test_pauli_feature_map_properties(self):
        enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"])
        assert enc.n_qubits is None

    def test_pauli_single_only_depth(self):
        enc = PauliFeatureMapEncoder(paulis=["Z"])
        assert "d²" not in enc.depth

    def test_random_fourier_properties(self):
        enc = RandomFourierEncoder(n_components=8)
        assert enc.n_qubits == 8
        assert enc.depth == 1

    def test_tensor_product_properties(self):
        enc = TensorProductEncoder()
        assert enc.n_qubits is None
        assert enc.depth == 2

    def test_qaoa_problem_properties(self):
        enc = QAOAProblemEncoder()
        assert enc.n_qubits is None
        assert enc.depth == "O(p)"

    def test_qaoa_problem_full_connectivity_depth(self):
        enc = QAOAProblemEncoder(connectivity="full")
        assert "d" in enc.depth


# ---------------------------------------------------------------------------
# QAOAProblemEncoder
# ---------------------------------------------------------------------------

class TestQAOAProblemEncoder:
    def test_output_shape_linear(self):
        enc = QAOAProblemEncoder(connectivity="linear")
        x = np.array([0.1, 0.2, 0.3, 0.4])
        result = enc.encode(x)
        # parameters = local(d) + pairs(d-1) = 4 + 3 = 7
        assert len(result.parameters) == 7

    def test_output_shape_full(self):
        enc = QAOAProblemEncoder(connectivity="full")
        x = np.array([0.1, 0.2, 0.3, 0.4])
        result = enc.encode(x)
        # parameters = local(4) + pairs(6) = 10
        assert len(result.parameters) == 10

    def test_local_angles_scale_with_gamma(self):
        x = np.array([1.0, 2.0, 3.0])
        gamma = 0.5
        enc = QAOAProblemEncoder(gamma=gamma)
        result = enc.encode(x)
        expected_local = gamma * x
        np.testing.assert_allclose(result.parameters[:3], expected_local)

    def test_coupling_angles_are_products(self):
        x = np.array([1.0, 2.0, 3.0])
        gamma = 1.0
        enc = QAOAProblemEncoder(gamma=gamma, connectivity="linear")
        result = enc.encode(x)
        # coupling angles: γ*x[0]*x[1], γ*x[1]*x[2]
        np.testing.assert_allclose(result.parameters[3], 1.0 * 1.0 * 2.0)
        np.testing.assert_allclose(result.parameters[4], 1.0 * 2.0 * 3.0)

    def test_metadata_keys(self):
        enc = QAOAProblemEncoder()
        result = enc.encode(np.array([0.5, 1.0]))
        for key in ("encoding", "n_qubits", "p", "gamma", "beta", "connectivity", "depth"):
            assert key in result.metadata

    def test_metadata_encoding_name(self):
        enc = QAOAProblemEncoder()
        result = enc.encode(np.array([0.5, 1.0]))
        assert result.metadata["encoding"] == "qaoa_problem"

    def test_metadata_n_qubits(self):
        enc = QAOAProblemEncoder()
        x = np.linspace(0, 1, 6)
        result = enc.encode(x)
        assert result.metadata["n_qubits"] == 6

    def test_single_feature(self):
        enc = QAOAProblemEncoder()
        result = enc.encode(np.array([0.7]))
        assert result.metadata["n_qubits"] == 1
        # linear: no pairs for d=1
        assert result.metadata["n_pairs"] == 0
        assert len(result.parameters) == 1

    def test_invalid_connectivity_raises(self):
        with pytest.raises(ValueError, match="connectivity"):
            QAOAProblemEncoder(connectivity="ring")

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError, match="p"):
            QAOAProblemEncoder(p=0)

    def test_empty_input_raises(self):
        enc = QAOAProblemEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.array([]))

    def test_2d_input_raises(self):
        enc = QAOAProblemEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.ones((2, 3)))

    def test_deterministic(self):
        enc = QAOAProblemEncoder(p=2)
        x = np.random.default_rng(7).uniform(-np.pi, np.pi, 5)
        r1 = enc.encode(x)
        r2 = enc.encode(x)
        np.testing.assert_array_equal(r1.parameters, r2.parameters)

    def test_multi_layer(self):
        enc = QAOAProblemEncoder(p=3)
        x = np.array([0.1, 0.2, 0.3])
        result = enc.encode(x)
        assert result.metadata["p"] == 3
        # depth = 1 + 5*p = 16
        assert result.metadata["depth"] == 1 + 5 * 3

    def test_result_is_encoded_result(self):
        enc = QAOAProblemEncoder()
        result = enc.encode(np.array([0.5, 1.0, 1.5]))
        assert isinstance(result, EncodedResult)

    @given(
        npst.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=10),
            elements=st.floats(-np.pi, np.pi, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50)
    def test_property_output_shape_linear(self, x):
        enc = QAOAProblemEncoder(connectivity="linear")
        result = enc.encode(x)
        d = len(x)
        n_pairs = max(d - 1, 0)
        assert len(result.parameters) == d + n_pairs

    @given(
        npst.arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=8),
            elements=st.floats(-np.pi, np.pi, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50)
    def test_property_output_shape_full(self, x):
        enc = QAOAProblemEncoder(connectivity="full")
        result = enc.encode(x)
        d = len(x)
        n_pairs = d * (d - 1) // 2
        assert len(result.parameters) == d + n_pairs
