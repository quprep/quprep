"""Tests for circuit cost estimation."""


from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.basis import BasisEncoder
from quprep.encode.entangled_angle import EntangledAngleEncoder
from quprep.encode.hamiltonian import HamiltonianEncoder
from quprep.encode.iqp import IQPEncoder
from quprep.encode.reupload import ReUploadEncoder
from quprep.validation import CostEstimate, estimate_cost


def _cost(encoder, n):
    return estimate_cost(encoder, n)


# ---------------------------------------------------------------------------
# AngleEncoder
# ---------------------------------------------------------------------------

def test_angle_cost():
    c = _cost(AngleEncoder(), 4)
    assert c.encoding == "angle"
    assert c.n_qubits == 4
    assert c.circuit_depth == 1
    assert c.two_qubit_gates == 0
    assert c.nisq_safe is True
    assert c.warning is None


# ---------------------------------------------------------------------------
# BasisEncoder
# ---------------------------------------------------------------------------

def test_basis_cost():
    c = _cost(BasisEncoder(), 8)
    assert c.n_qubits == 8
    assert c.circuit_depth == 1
    assert c.two_qubit_gates == 0
    assert c.nisq_safe is True


# ---------------------------------------------------------------------------
# AmplitudeEncoder
# ---------------------------------------------------------------------------

def test_amplitude_cost_small():
    # 4 features → 2 qubits → depth 4 → nisq_safe
    c = _cost(AmplitudeEncoder(), 4)
    assert c.n_qubits == 2
    assert c.circuit_depth == 4
    assert c.nisq_safe is True


def test_amplitude_cost_large_warns():
    # 256 features → 8 qubits → depth 256 → not nisq_safe
    c = _cost(AmplitudeEncoder(), 256)
    assert c.nisq_safe is False
    assert c.warning is not None
    assert "AmplitudeEncoder" in c.warning


# ---------------------------------------------------------------------------
# IQPEncoder
# ---------------------------------------------------------------------------

def test_iqp_cost_small():
    c = _cost(IQPEncoder(reps=1), 4)
    assert c.n_qubits == 4
    assert c.two_qubit_gates == 4 * 3 // 2 * 1  # 6


def test_iqp_cost_large_warns():
    c = _cost(IQPEncoder(reps=3), 20)
    assert c.nisq_safe is False
    assert c.warning is not None


# ---------------------------------------------------------------------------
# ReUploadEncoder
# ---------------------------------------------------------------------------

def test_reupload_cost():
    c = _cost(ReUploadEncoder(layers=3), 5)
    assert c.n_qubits == 5
    assert c.circuit_depth == 15
    assert c.two_qubit_gates == 0


# ---------------------------------------------------------------------------
# EntangledAngleEncoder
# ---------------------------------------------------------------------------

def test_entangled_linear():
    c = _cost(EntangledAngleEncoder(layers=1, entanglement="linear"), 5)
    assert c.two_qubit_gates == 4  # d-1 per layer


def test_entangled_circular():
    c = _cost(EntangledAngleEncoder(layers=2, entanglement="circular"), 5)
    assert c.two_qubit_gates == 10  # d per layer * layers


def test_entangled_full():
    c = _cost(EntangledAngleEncoder(layers=1, entanglement="full"), 4)
    assert c.two_qubit_gates == 6  # 4*3//2


# ---------------------------------------------------------------------------
# HamiltonianEncoder
# ---------------------------------------------------------------------------

def test_hamiltonian_cost():
    c = _cost(HamiltonianEncoder(trotter_steps=4), 4)
    assert c.n_qubits == 4
    assert c.circuit_depth == 16  # d * steps


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_cost_estimate():
    c = _cost(AngleEncoder(), 3)
    assert isinstance(c, CostEstimate)
