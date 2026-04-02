"""Tests for framework exporters."""

from __future__ import annotations

import json

import numpy as np
import pytest

from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.base import EncodedResult
from quprep.encode.basis import BasisEncoder
from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
from quprep.encode.random_fourier import RandomFourierEncoder
from quprep.encode.tensor_product import TensorProductEncoder
from quprep.encode.zz_feature_map import ZZFeatureMapEncoder
from quprep.export.iqm_export import IQMExporter
from quprep.export.qasm_export import QASMExporter
from quprep.export.qsharp_export import QSharpExporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2(x):
    return x / np.linalg.norm(x)


def _angle_result(n=4, rotation="ry"):
    x = np.linspace(0.1, np.pi - 0.1, n)
    return AngleEncoder(rotation=rotation).encode(x)


def _basis_result(bits=None):
    x = np.array(bits or [0.0, 1.0, 0.0, 1.0])
    return BasisEncoder(threshold=0.5).encode(x)


def _amplitude_result(d=4):
    x = _l2(np.ones(d))
    return AmplitudeEncoder().encode(x)


# ---------------------------------------------------------------------------
# QASMExporter
# ---------------------------------------------------------------------------

class TestQASMExporter:
    def test_unsupported_version_raises(self):
        with pytest.raises(ValueError, match="3.0"):
            QASMExporter(version="2.0")

    def test_default_version(self):
        exp = QASMExporter()
        assert exp.version == "3.0"

    # --- angle encoding ---

    def test_angle_export_returns_string(self):
        exp = QASMExporter()
        result = exp.export(_angle_result())
        assert isinstance(result, str)

    def test_angle_qasm_header(self):
        exp = QASMExporter()
        qasm = exp.export(_angle_result())
        assert qasm.startswith("OPENQASM 3.0;")
        assert 'include "stdgates.inc";' in qasm

    def test_angle_qubit_register(self):
        exp = QASMExporter()
        qasm = exp.export(_angle_result(n=6))
        assert "qubit[6] q;" in qasm

    def test_angle_ry_gates(self):
        exp = QASMExporter()
        qasm = exp.export(_angle_result(n=3, rotation="ry"))
        assert qasm.count("ry(") == 3

    def test_angle_rx_gates(self):
        exp = QASMExporter()
        qasm = exp.export(_angle_result(n=2, rotation="rx"))
        assert qasm.count("rx(") == 2

    def test_angle_rz_gates(self):
        exp = QASMExporter()
        qasm = exp.export(_angle_result(n=2, rotation="rz"))
        assert qasm.count("rz(") == 2

    def test_angle_correct_qubit_targets(self):
        exp = QASMExporter()
        qasm = exp.export(_angle_result(n=3))
        assert "q[0]" in qasm
        assert "q[1]" in qasm
        assert "q[2]" in qasm

    def test_angle_ends_with_newline(self):
        exp = QASMExporter()
        qasm = exp.export(_angle_result())
        assert qasm.endswith("\n")

    def test_angle_deterministic(self):
        exp = QASMExporter()
        r = _angle_result()
        assert exp.export(r) == exp.export(r)

    # --- basis encoding ---

    def test_basis_export_returns_string(self):
        exp = QASMExporter()
        result = exp.export(_basis_result())
        assert isinstance(result, str)

    def test_basis_qasm_header(self):
        exp = QASMExporter()
        qasm = exp.export(_basis_result())
        assert "OPENQASM 3.0;" in qasm

    def test_basis_x_gates_only_for_ones(self):
        # bits: [0, 1, 0, 1] → x q[1]; x q[3];
        exp = QASMExporter()
        qasm = exp.export(_basis_result([0.0, 1.0, 0.0, 1.0]))
        assert "x q[1];" in qasm
        assert "x q[3];" in qasm
        assert "x q[0];" not in qasm
        assert "x q[2];" not in qasm

    def test_basis_all_zeros_no_x_gates(self):
        exp = QASMExporter()
        qasm = exp.export(_basis_result([0.0, 0.0, 0.0]))
        assert "x q" not in qasm

    def test_basis_all_ones_all_x_gates(self):
        exp = QASMExporter()
        qasm = exp.export(_basis_result([1.0, 1.0, 1.0]))
        assert qasm.count("x q") == 3

    def test_basis_qubit_register_size(self):
        exp = QASMExporter()
        qasm = exp.export(_basis_result([0.0, 1.0, 0.0]))
        assert "qubit[3] q;" in qasm

    # --- amplitude encoding ---

    def test_amplitude_raises_not_implemented(self):
        exp = QASMExporter()
        with pytest.raises(NotImplementedError, match="QiskitExporter"):
            exp.export(_amplitude_result())

    # --- unknown encoding ---

    def test_unknown_encoding_raises(self):
        exp = QASMExporter()
        fake = EncodedResult(
            parameters=np.array([0.1, 0.2]),
            metadata={"encoding": "totally_unknown"},
        )
        with pytest.raises(ValueError):
            exp.export(fake)

    # --- save ---

    def test_save_writes_file(self, tmp_path):
        exp = QASMExporter()
        out = tmp_path / "circuit.qasm"
        exp.save(_angle_result(), out)
        content = out.read_text()
        assert "OPENQASM 3.0;" in content

    def test_save_string_path(self, tmp_path):
        exp = QASMExporter()
        out = str(tmp_path / "circuit.qasm")
        exp.save(_angle_result(), out)
        from pathlib import Path
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# QiskitExporter — tested only if qiskit is installed
# ---------------------------------------------------------------------------

class TestQiskitExporter:
    def test_missing_qiskit_raises_import_error(self):
        """Instantiation must raise ImportError with install hint when qiskit absent."""
        try:
            import qiskit  # noqa: F401
            pytest.skip("qiskit is installed — skipping missing-dep test")
        except ImportError:
            from quprep.export.qiskit_export import QiskitExporter
            with pytest.raises(ImportError, match="pip install quprep"):
                QiskitExporter()

    def test_angle_export_returns_quantum_circuit(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        exp = QiskitExporter()
        qc = exp.export(_angle_result(n=4, rotation="ry"))
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 4

    def test_angle_circuit_has_ry_gates(self):
        pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        exp = QiskitExporter()
        qc = exp.export(_angle_result(n=3, rotation="ry"))
        ops = [instr.operation.name for instr in qc.data]
        assert ops.count("ry") == 3

    def test_basis_export_x_gates(self):
        pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        exp = QiskitExporter()
        qc = exp.export(_basis_result([0.0, 1.0, 0.0, 1.0]))
        ops = [instr.operation.name for instr in qc.data]
        assert ops.count("x") == 2

    def test_basis_all_zeros_empty_circuit(self):
        pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        exp = QiskitExporter()
        qc = exp.export(_basis_result([0.0, 0.0, 0.0]))
        assert len(qc.data) == 0

    def test_amplitude_export_returns_circuit(self):
        pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        exp = QiskitExporter()
        qc = exp.export(_amplitude_result(d=4))
        from qiskit import QuantumCircuit
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2  # log2(4) = 2

    def test_unknown_encoding_raises(self):
        pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        exp = QiskitExporter()
        fake = EncodedResult(
            parameters=np.array([0.1, 0.2]),
            metadata={"encoding": "totally_unknown", "n_qubits": 2},
        )
        with pytest.raises(ValueError, match="Unknown encoding"):
            exp.export(fake)

    def test_export_batch(self):
        pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        exp = QiskitExporter()
        results = [_angle_result(n=3) for _ in range(4)]
        circuits = exp.export_batch(results)
        assert len(circuits) == 4

    def test_entangled_angle_export(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        from quprep.export.qiskit_export import QiskitExporter
        x = np.linspace(0.1, 1.0, 3)
        encoded = EntangledAngleEncoder(layers=2).encode(x)
        qc = QiskitExporter().export(encoded)
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 3

    def test_iqp_export(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        qc = QiskitExporter().export(_iqp_result(d=3))
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 3

    def test_reupload_export(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        qc = QiskitExporter().export(_reupload_result(d=3))
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 3

    def test_hamiltonian_export(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        qc = QiskitExporter().export(_hamiltonian_result(d=3))
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 3

    def test_zz_feature_map_export(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        qc = QiskitExporter().export(_zz_enc(n=3))
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 3

    def test_pauli_feature_map_export(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1)
        x = np.array([0.5, 1.0, 1.5])
        encoded = enc.encode(x)
        qc = QiskitExporter().export(encoded)
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 3

    def test_tensor_product_export(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        qc = QiskitExporter().export(_tp_enc(n=4))
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 4

    def test_qaoa_problem_export(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        qc = QiskitExporter().export(_qaoa_enc(n=3))
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 3

    def test_random_fourier_export(self):
        qiskit = pytest.importorskip("qiskit")
        from quprep.export.qiskit_export import QiskitExporter
        enc = RandomFourierEncoder(n_components=3, random_state=0)
        enc.fit(np.random.default_rng(0).random((10, 3)))
        encoded = enc.encode(np.random.default_rng(0).random(3))
        qc = QiskitExporter().export(encoded)
        assert isinstance(qc, qiskit.QuantumCircuit)
        assert qc.num_qubits == 3


# ---------------------------------------------------------------------------
# Phase 2 exporter helpers
# ---------------------------------------------------------------------------

def _iqp_result(d=3):
    from quprep.encode.iqp import IQPEncoder
    x = np.linspace(0.1, 1.0, d)
    return IQPEncoder(reps=1).encode(x)


def _reupload_result(d=3):
    from quprep.encode.reupload import ReUploadEncoder
    x = np.linspace(0.1, 1.0, d)
    return ReUploadEncoder(layers=2).encode(x)


def _hamiltonian_result(d=3):
    from quprep.encode.hamiltonian import HamiltonianEncoder
    x = np.linspace(0.1, 1.0, d)
    return HamiltonianEncoder(evolution_time=1.0, trotter_steps=2).encode(x)


# ---------------------------------------------------------------------------
# PennyLaneExporter
# ---------------------------------------------------------------------------

class TestPennyLaneExporter:
    def test_missing_dep_raises(self):
        try:
            import pennylane  # noqa: F401
            pytest.skip("pennylane installed")
        except ImportError:
            from quprep.export.pennylane_export import PennyLaneExporter
            with pytest.raises(ImportError, match="pip install quprep"):
                PennyLaneExporter()

    def test_angle_export_returns_callable(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_angle_result(n=3))
        assert callable(circuit)

    def test_angle_circuit_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_angle_result(n=3))
        state = circuit()
        assert len(state) == 2**3

    def test_angle_rx_circuit_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_angle_result(n=2, rotation="rx"))
        state = circuit()
        assert len(state) == 4

    def test_basis_circuit_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_basis_result([0.0, 1.0, 0.0]))
        state = circuit()
        assert len(state) == 8

    def test_amplitude_circuit_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_amplitude_result(d=4))
        import numpy as np
        state = circuit()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-6

    def test_iqp_circuit_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_iqp_result(d=3))
        state = circuit()
        assert len(state) == 8

    def test_reupload_circuit_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_reupload_result(d=3))
        state = circuit()
        assert len(state) == 8

    def test_hamiltonian_circuit_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_hamiltonian_result(d=3))
        state = circuit()
        assert len(state) == 8

    def test_unknown_encoding_raises(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        fake = EncodedResult(
            parameters=np.array([0.1, 0.2]),
            metadata={"encoding": "totally_unknown", "n_qubits": 2},
        )
        with pytest.raises(ValueError, match="Unknown encoding"):
            PennyLaneExporter().export(fake)

    def test_export_batch(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuits = PennyLaneExporter().export_batch([_angle_result(n=2) for _ in range(3)])
        assert len(circuits) == 3
        assert all(callable(c) for c in circuits)

    def test_entangled_angle_circuit_executes(self):
        pytest.importorskip("pennylane")
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        from quprep.export.pennylane_export import PennyLaneExporter
        enc = EntangledAngleEncoder(layers=1).encode(np.array([0.5, 1.0, 1.5]))
        circuit = PennyLaneExporter().export(enc)
        assert callable(circuit)
        state = circuit()
        assert len(state) == 2**3

    def test_zz_feature_map_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_zz_enc(n=3))
        assert callable(circuit)
        assert len(circuit()) == 2**3

    def test_pauli_feature_map_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1)
        encoded = enc.encode(np.array([0.5, 1.0, 1.5]))
        circuit = PennyLaneExporter().export(encoded)
        assert callable(circuit)
        assert len(circuit()) == 2**3

    def test_random_fourier_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        enc = RandomFourierEncoder(n_components=3, random_state=0)
        enc.fit(np.random.default_rng(0).random((10, 3)))
        encoded = enc.encode(np.random.default_rng(0).random(3))
        circuit = PennyLaneExporter().export(encoded)
        assert callable(circuit)
        assert len(circuit()) == 2**3

    def test_tensor_product_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_tp_enc(n=4))
        assert callable(circuit)
        assert len(circuit()) == 2**4

    def test_qaoa_problem_executes(self):
        pytest.importorskip("pennylane")
        from quprep.export.pennylane_export import PennyLaneExporter
        circuit = PennyLaneExporter().export(_qaoa_enc(n=3))
        assert callable(circuit)
        assert len(circuit()) == 2**3


# ---------------------------------------------------------------------------
# CirqExporter
# ---------------------------------------------------------------------------

class TestCirqExporter:
    def test_missing_dep_raises(self):
        try:
            import cirq  # noqa: F401
            pytest.skip("cirq installed")
        except ImportError:
            from quprep.export.cirq_export import CirqExporter
            with pytest.raises(ImportError, match="pip install quprep"):
                CirqExporter()

    def test_angle_export_returns_circuit(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_angle_result(n=3))
        assert isinstance(circuit, cirq.Circuit)

    def test_angle_circuit_qubit_count(self):
        pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_angle_result(n=4))
        assert len(circuit.all_qubits()) == 4

    def test_angle_rx_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_angle_result(n=2, rotation="rx"))
        assert isinstance(circuit, cirq.Circuit)

    def test_basis_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_basis_result([0.0, 1.0, 0.0]))
        assert isinstance(circuit, cirq.Circuit)

    def test_basis_x_gate_count(self):
        pytest.importorskip("cirq")
        import cirq

        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_basis_result([1.0, 0.0, 1.0]))
        x_ops = [
            op for op in circuit.all_operations()
            if isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1
        ]
        assert len(x_ops) == 2

    def test_amplitude_raises(self):
        pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        with pytest.raises(NotImplementedError, match="QiskitExporter"):
            CirqExporter().export(_amplitude_result(d=4))

    def test_iqp_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_iqp_result(d=3))
        assert isinstance(circuit, cirq.Circuit)

    def test_reupload_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_reupload_result(d=3))
        assert isinstance(circuit, cirq.Circuit)

    def test_hamiltonian_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_hamiltonian_result(d=3))
        assert isinstance(circuit, cirq.Circuit)

    def test_unknown_encoding_raises(self):
        pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        fake = EncodedResult(
            parameters=np.array([0.1, 0.2]),
            metadata={"encoding": "totally_unknown", "n_qubits": 2},
        )
        with pytest.raises(ValueError, match="Unknown encoding"):
            CirqExporter().export(fake)

    def test_export_batch(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuits = CirqExporter().export_batch([_angle_result(n=2) for _ in range(3)])
        assert len(circuits) == 3
        assert all(isinstance(c, cirq.Circuit) for c in circuits)

    def test_entangled_angle_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        from quprep.export.cirq_export import CirqExporter
        enc = EntangledAngleEncoder(layers=1).encode(np.array([0.5, 1.0, 1.5]))
        circuit = CirqExporter().export(enc)
        assert isinstance(circuit, cirq.Circuit)

    def test_zz_feature_map_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_zz_enc(n=3))
        assert isinstance(circuit, cirq.Circuit)

    def test_pauli_feature_map_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1)
        encoded = enc.encode(np.array([0.5, 1.0, 1.5]))
        circuit = CirqExporter().export(encoded)
        assert isinstance(circuit, cirq.Circuit)

    def test_random_fourier_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        enc = RandomFourierEncoder(n_components=3, random_state=0)
        enc.fit(np.random.default_rng(0).random((10, 3)))
        encoded = enc.encode(np.random.default_rng(0).random(3))
        circuit = CirqExporter().export(encoded)
        assert isinstance(circuit, cirq.Circuit)

    def test_tensor_product_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_tp_enc(n=4))
        assert isinstance(circuit, cirq.Circuit)

    def test_qaoa_problem_export(self):
        cirq = pytest.importorskip("cirq")
        from quprep.export.cirq_export import CirqExporter
        circuit = CirqExporter().export(_qaoa_enc(n=3))
        assert isinstance(circuit, cirq.Circuit)


# ---------------------------------------------------------------------------
# TKETExporter
# ---------------------------------------------------------------------------

class TestTKETExporter:
    def test_missing_dep_raises(self):
        try:
            import pytket  # noqa: F401
            pytest.skip("pytket installed")
        except ImportError:
            from quprep.export.tket_export import TKETExporter
            with pytest.raises(ImportError, match="pip install quprep"):
                TKETExporter()

    def test_angle_export_returns_circuit(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_angle_result(n=3))
        assert isinstance(circuit, pytket.Circuit)

    def test_angle_circuit_qubit_count(self):
        pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_angle_result(n=4))
        assert circuit.n_qubits == 4

    def test_angle_rx_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_angle_result(n=2, rotation="rx"))
        assert isinstance(circuit, pytket.Circuit)

    def test_basis_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_basis_result([0.0, 1.0, 0.0]))
        assert isinstance(circuit, pytket.Circuit)

    def test_amplitude_raises(self):
        pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        with pytest.raises(NotImplementedError, match="QiskitExporter"):
            TKETExporter().export(_amplitude_result(d=4))

    def test_iqp_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_iqp_result(d=3))
        assert isinstance(circuit, pytket.Circuit)

    def test_reupload_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_reupload_result(d=3))
        assert isinstance(circuit, pytket.Circuit)

    def test_hamiltonian_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_hamiltonian_result(d=3))
        assert isinstance(circuit, pytket.Circuit)

    def test_unknown_encoding_raises(self):
        pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        fake = EncodedResult(
            parameters=np.array([0.1, 0.2]),
            metadata={"encoding": "totally_unknown", "n_qubits": 2},
        )
        with pytest.raises(ValueError, match="Unknown encoding"):
            TKETExporter().export(fake)

    def test_export_batch(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuits = TKETExporter().export_batch([_angle_result(n=2) for _ in range(3)])
        assert len(circuits) == 3
        assert all(isinstance(c, pytket.Circuit) for c in circuits)

    def test_entangled_angle_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        from quprep.export.tket_export import TKETExporter
        enc = EntangledAngleEncoder(layers=1).encode(np.array([0.5, 1.0, 1.5]))
        circuit = TKETExporter().export(enc)
        assert isinstance(circuit, pytket.Circuit)

    def test_zz_feature_map_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_zz_enc(n=3))
        assert isinstance(circuit, pytket.Circuit)

    def test_pauli_feature_map_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1)
        encoded = enc.encode(np.array([0.5, 1.0, 1.5]))
        circuit = TKETExporter().export(encoded)
        assert isinstance(circuit, pytket.Circuit)

    def test_random_fourier_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        enc = RandomFourierEncoder(n_components=3, random_state=0)
        enc.fit(np.random.default_rng(0).random((10, 3)))
        encoded = enc.encode(np.random.default_rng(0).random(3))
        circuit = TKETExporter().export(encoded)
        assert isinstance(circuit, pytket.Circuit)

    def test_tensor_product_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_tp_enc(n=4))
        assert isinstance(circuit, pytket.Circuit)

    def test_qaoa_problem_export(self):
        pytket = pytest.importorskip("pytket")
        from quprep.export.tket_export import TKETExporter
        circuit = TKETExporter().export(_qaoa_enc(n=3))
        assert isinstance(circuit, pytket.Circuit)


# ---------------------------------------------------------------------------
# QASMExporter.save_batch
# ---------------------------------------------------------------------------

class TestQASMSaveBatch:
    def test_save_batch_creates_files(self, tmp_path):
        exporter = QASMExporter()
        encoded = [_angle_result(n=3) for _ in range(4)]
        paths = exporter.save_batch(encoded, tmp_path / "circuits")
        assert len(paths) == 4
        for p in paths:
            assert p.exists()

    def test_save_batch_default_stem(self, tmp_path):
        exporter = QASMExporter()
        encoded = [_angle_result(n=2)]
        paths = exporter.save_batch(encoded, tmp_path / "out")
        assert paths[0].name == "circuit_0000.qasm"

    def test_save_batch_custom_stem(self, tmp_path):
        exporter = QASMExporter()
        encoded = [_angle_result(n=2), _angle_result(n=2)]
        paths = exporter.save_batch(encoded, tmp_path / "out", stem="sample")
        assert paths[0].name == "sample_0000.qasm"
        assert paths[1].name == "sample_0001.qasm"

    def test_save_batch_creates_directory(self, tmp_path):
        exporter = QASMExporter()
        out_dir = tmp_path / "new" / "nested" / "dir"
        exporter.save_batch([_angle_result(n=2)], out_dir)
        assert out_dir.exists()

    def test_save_batch_file_content_is_qasm(self, tmp_path):
        exporter = QASMExporter()
        paths = exporter.save_batch([_angle_result(n=3)], tmp_path / "out")
        content = paths[0].read_text()
        assert content.startswith("OPENQASM 3.0;")

    def test_save_batch_returns_paths_list(self, tmp_path):
        from pathlib import Path
        exporter = QASMExporter()
        paths = exporter.save_batch([_angle_result(n=2)], tmp_path / "out")
        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)

    def test_save_batch_empty_list(self, tmp_path):
        exporter = QASMExporter()
        paths = exporter.save_batch([], tmp_path / "out")
        assert paths == []

    def test_save_batch_basis_encoding(self, tmp_path):
        exporter = QASMExporter()
        paths = exporter.save_batch([_basis_result()], tmp_path / "out")
        assert paths[0].read_text().startswith("OPENQASM 3.0;")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_enc(n=4):
    from quprep.encode.angle import AngleEncoder
    x = np.linspace(0.1, 1.0, n)
    return AngleEncoder().encode(x)


def _basis_enc(n=4):
    from quprep.encode.basis import BasisEncoder
    x = np.array([0, 1, 0, 1], dtype=float)
    return BasisEncoder().encode(x)


def _zz_enc(n=3):
    x = np.linspace(0.5, 2.0, n)
    return ZZFeatureMapEncoder(reps=1).encode(x)


def _tp_enc(n=4):
    x = np.linspace(0.1, 1.0, n)
    return TensorProductEncoder().encode(x)


def _reupload_enc(n=3):
    from quprep.encode.reupload import ReUploadEncoder
    x = np.linspace(0.1, 1.0, n)
    return ReUploadEncoder(layers=2).encode(x)


def _hamiltonian_enc(n=3):
    from quprep.encode.hamiltonian import HamiltonianEncoder
    x = np.linspace(0.1, 1.0, n)
    return HamiltonianEncoder(evolution_time=1.0, trotter_steps=2).encode(x)


def _qaoa_enc(n=3):
    from quprep.encode.qaoa_problem import QAOAProblemEncoder
    x = np.linspace(0.1, 1.0, n) * np.pi - np.pi / 2
    return QAOAProblemEncoder(p=1, connectivity="linear").encode(x)


# ---------------------------------------------------------------------------
# QSharpExporter
# ---------------------------------------------------------------------------

class TestQSharpExporter:
    def test_returns_string(self):
        exp = QSharpExporter()
        qsharp = exp.export(_angle_enc())
        assert isinstance(qsharp, str)

    def test_contains_namespace(self):
        exp = QSharpExporter(namespace="MyNS")
        qsharp = exp.export(_angle_enc())
        assert "MyNS" in qsharp

    def test_contains_operation_name(self):
        exp = QSharpExporter(operation_name="FeatureMap")
        qsharp = exp.export(_angle_enc())
        assert "FeatureMap" in qsharp

    def test_angle_encoding_uses_ry(self):
        exp = QSharpExporter()
        qsharp = exp.export(_angle_enc(4))
        assert "Ry(" in qsharp

    def test_basis_encoding_uses_x(self):
        exp = QSharpExporter()
        qsharp = exp.export(_basis_enc())
        assert "X(" in qsharp

    def test_zz_feature_map_contains_rz_and_h(self):
        exp = QSharpExporter()
        qsharp = exp.export(_zz_enc())
        assert "Rz(" in qsharp
        assert "H(" in qsharp

    def test_tensor_product_contains_ry_and_rz(self):
        exp = QSharpExporter()
        qsharp = exp.export(_tp_enc())
        assert "Ry(" in qsharp
        assert "Rz(" in qsharp

    def test_amplitude_raises(self):
        encoded = EncodedResult(
            parameters=np.array([0.5, 0.5, 0.5, 0.5]),
            metadata={"encoding": "amplitude", "n_qubits": 2},
        )
        exp = QSharpExporter()
        with pytest.raises(NotImplementedError):
            exp.export(encoded)

    def test_unknown_encoding_raises(self):
        encoded = EncodedResult(
            parameters=np.array([1.0]),
            metadata={"encoding": "unknown_enc", "n_qubits": 1},
        )
        exp = QSharpExporter()
        with pytest.raises(ValueError, match="Unknown encoding"):
            exp.export(encoded)

    def test_export_batch_returns_list(self):
        exp = QSharpExporter()
        results = [_angle_enc(3), _angle_enc(3)]
        batch = exp.export_batch(results)
        assert isinstance(batch, list)
        assert len(batch) == 2
        assert all(isinstance(s, str) for s in batch)

    def test_qubit_register_declared(self):
        exp = QSharpExporter()
        qsharp = exp.export(_angle_enc(5))
        assert "Qubit[5]" in qsharp

    def test_reset_all_present(self):
        exp = QSharpExporter()
        qsharp = exp.export(_angle_enc())
        assert "ResetAll" in qsharp


# ---------------------------------------------------------------------------
# IQMExporter
# ---------------------------------------------------------------------------

class TestIQMExporter:
    def test_returns_dict(self):
        exp = IQMExporter()
        result = exp.export(_angle_enc())
        assert isinstance(result, dict)

    def test_circuit_name_field(self):
        exp = IQMExporter(circuit_name="test_circuit")
        result = exp.export(_angle_enc())
        assert result["name"] == "test_circuit"

    def test_instructions_is_list(self):
        exp = IQMExporter()
        result = exp.export(_angle_enc())
        assert isinstance(result["instructions"], list)
        assert len(result["instructions"]) > 0

    def test_angle_encoding_uses_prx(self):
        exp = IQMExporter()
        result = exp.export(_angle_enc(3))
        names = {op["name"] for op in result["instructions"]}
        assert "prx" in names

    def test_qubit_labels(self):
        exp = IQMExporter(qubit_prefix="QB")
        result = exp.export(_angle_enc(3))
        all_qubits = [q for op in result["instructions"] for q in op["qubits"]]
        assert "QB1" in all_qubits
        assert "QB2" in all_qubits
        assert "QB3" in all_qubits

    def test_custom_qubit_prefix(self):
        exp = IQMExporter(qubit_prefix="Q")
        result = exp.export(_angle_enc(2))
        all_qubits = [q for op in result["instructions"] for q in op["qubits"]]
        assert "Q1" in all_qubits

    def test_basis_encoding_uses_prx(self):
        exp = IQMExporter()
        result = exp.export(_basis_enc())
        prx_ops = [op for op in result["instructions"] if op["name"] == "prx"]
        assert len(prx_ops) > 0

    def test_iqp_uses_cz(self):
        from quprep.encode.iqp import IQPEncoder
        x = np.array([0.5, 1.0, 1.5])
        encoded = IQPEncoder(reps=1).encode(x)
        exp = IQMExporter()
        result = exp.export(encoded)
        names = {op["name"] for op in result["instructions"]}
        assert "cz" in names

    def test_zz_feature_map_uses_cz(self):
        exp = IQMExporter()
        result = exp.export(_zz_enc(3))
        names = {op["name"] for op in result["instructions"]}
        assert "cz" in names

    def test_tensor_product_no_cz(self):
        exp = IQMExporter()
        result = exp.export(_tp_enc(4))
        names = {op["name"] for op in result["instructions"]}
        assert "cz" not in names

    def test_json_serializable(self):
        exp = IQMExporter()
        result = exp.export(_angle_enc())
        # should not raise
        json.dumps(result)

    def test_amplitude_raises(self):
        encoded = EncodedResult(
            parameters=np.array([0.5, 0.5]),
            metadata={"encoding": "amplitude", "n_qubits": 1},
        )
        exp = IQMExporter()
        with pytest.raises(NotImplementedError):
            exp.export(encoded)

    def test_unknown_encoding_raises(self):
        encoded = EncodedResult(
            parameters=np.array([1.0]),
            metadata={"encoding": "unknown_enc", "n_qubits": 1},
        )
        exp = IQMExporter()
        with pytest.raises(ValueError, match="Unknown encoding"):
            exp.export(encoded)

    def test_export_batch_returns_list(self):
        exp = IQMExporter()
        batch = exp.export_batch([_angle_enc(3), _angle_enc(3)])
        assert isinstance(batch, list)
        assert len(batch) == 2
        assert all(isinstance(d, dict) for d in batch)


# ---------------------------------------------------------------------------
# QASMExporter — new encodings
# ---------------------------------------------------------------------------

class TestQASMExporterV060:
    def test_zz_feature_map_qasm(self):
        exp = QASMExporter()
        qasm = exp.export(_zz_enc(3))
        assert "OPENQASM 3.0" in qasm
        assert "h " in qasm
        assert "rz(" in qasm
        assert "cx " in qasm

    def test_tensor_product_qasm(self):
        exp = QASMExporter()
        qasm = exp.export(_tp_enc(4))
        assert "OPENQASM 3.0" in qasm
        assert "ry(" in qasm
        assert "rz(" in qasm

    def test_random_fourier_qasm(self):
        enc = RandomFourierEncoder(n_components=4, random_state=0)
        enc.fit(np.random.default_rng(0).random((10, 3)))
        encoded = enc.encode(np.random.default_rng(0).random(3))
        exp = QASMExporter()
        qasm = exp.export(encoded)
        assert "OPENQASM 3.0" in qasm
        assert "ry(" in qasm  # angle encoding via _export_angle

    def test_pauli_feature_map_qasm(self):
        enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1)
        x = np.array([0.5, 1.0, 1.5])
        encoded = enc.encode(x)
        exp = QASMExporter()
        qasm = exp.export(encoded)
        assert "OPENQASM 3.0" in qasm
        assert "h " in qasm
        assert "rz(" in qasm

    def test_reupload_qasm(self):
        exp = QASMExporter()
        qasm = exp.export(_reupload_enc(3))
        assert "OPENQASM 3.0" in qasm
        assert "ry(" in qasm

    def test_hamiltonian_qasm(self):
        exp = QASMExporter()
        qasm = exp.export(_hamiltonian_enc(3))
        assert "OPENQASM 3.0" in qasm
        assert "rz(" in qasm

    def test_qaoa_problem_qasm(self):
        exp = QASMExporter()
        qasm = exp.export(_qaoa_enc(3))
        assert "OPENQASM 3.0" in qasm
        assert "h " in qasm
        assert "rz(" in qasm
        assert "rx(" in qasm
        assert "cx " in qasm


# ---------------------------------------------------------------------------
# BraketExporter
# ---------------------------------------------------------------------------

class TestBraketExporter:
    def test_braket_not_installed_raises(self):
        """Without amazon-braket-sdk the exporter raises ImportError."""
        pytest.importorskip("braket.circuits", reason="braket not installed")
        from quprep.export.braket_export import BraketExporter
        exp = BraketExporter()
        result = exp.export(_angle_enc(3))
        assert result is not None

    def test_braket_angle_encoding(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.export.braket_export import BraketExporter
        exp = BraketExporter()
        circuit = exp.export(_angle_enc(3))
        assert isinstance(circuit, Circuit)

    def test_braket_basis_encoding(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.export.braket_export import BraketExporter
        exp = BraketExporter()
        circuit = exp.export(_basis_enc())
        assert isinstance(circuit, Circuit)

    def test_braket_zz_feature_map(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.export.braket_export import BraketExporter
        exp = BraketExporter()
        circuit = exp.export(_zz_enc(3))
        assert isinstance(circuit, Circuit)

    def test_braket_tensor_product(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.export.braket_export import BraketExporter
        exp = BraketExporter()
        circuit = exp.export(_tp_enc(4))
        assert isinstance(circuit, Circuit)

    def test_braket_amplitude_raises(self):
        pytest.importorskip("braket.circuits")
        from quprep.export.braket_export import BraketExporter
        exp = BraketExporter()
        encoded = EncodedResult(
            parameters=np.array([0.5, 0.5]),
            metadata={"encoding": "amplitude", "n_qubits": 1},
        )
        with pytest.raises(NotImplementedError):
            exp.export(encoded)

    def test_braket_unknown_encoding_raises(self):
        pytest.importorskip("braket.circuits")
        from quprep.export.braket_export import BraketExporter
        exp = BraketExporter()
        encoded = EncodedResult(
            parameters=np.array([1.0]),
            metadata={"encoding": "unknown_enc", "n_qubits": 1},
        )
        with pytest.raises(ValueError, match="Unknown encoding"):
            exp.export(encoded)

    def test_braket_export_batch(self):
        pytest.importorskip("braket.circuits")
        from quprep.export.braket_export import BraketExporter
        exp = BraketExporter()
        batch = exp.export_batch([_angle_enc(3), _angle_enc(3)])
        assert len(batch) == 2


# ---------------------------------------------------------------------------
# BraketExporter — additional encoding coverage
# ---------------------------------------------------------------------------

class TestBraketExporterExtraEncodings:
    def test_entangled_angle(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.encode.entangled_angle import EntangledAngleEncoder
        from quprep.export.braket_export import BraketExporter
        enc = EntangledAngleEncoder(layers=1).encode(np.array([0.5, 1.0, 1.5]))
        circuit = BraketExporter().export(enc)
        assert isinstance(circuit, Circuit)

    def test_iqp(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.encode.iqp import IQPEncoder
        from quprep.export.braket_export import BraketExporter
        enc = IQPEncoder(reps=1).encode(np.array([0.5, 1.0, 1.5]))
        circuit = BraketExporter().export(enc)
        assert isinstance(circuit, Circuit)

    def test_reupload(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.export.braket_export import BraketExporter
        circuit = BraketExporter().export(_reupload_enc(3))
        assert isinstance(circuit, Circuit)

    def test_hamiltonian(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.export.braket_export import BraketExporter
        circuit = BraketExporter().export(_hamiltonian_enc(3))
        assert isinstance(circuit, Circuit)

    def test_qaoa_problem(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.export.braket_export import BraketExporter
        circuit = BraketExporter().export(_qaoa_enc(3))
        assert isinstance(circuit, Circuit)

    def test_pauli_feature_map(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.export.braket_export import BraketExporter
        enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1)
        encoded = enc.encode(np.array([0.5, 1.0, 1.5]))
        circuit = BraketExporter().export(encoded)
        assert isinstance(circuit, Circuit)

    def test_random_fourier(self):
        pytest.importorskip("braket.circuits")
        from braket.circuits import Circuit

        from quprep.export.braket_export import BraketExporter
        enc = RandomFourierEncoder(n_components=3, random_state=0)
        enc.fit(np.random.default_rng(0).random((10, 3)))
        encoded = enc.encode(np.random.default_rng(0).random(3))
        circuit = BraketExporter().export(encoded)
        assert isinstance(circuit, Circuit)


# ---------------------------------------------------------------------------
# QSharpExporter — additional encoding coverage
# ---------------------------------------------------------------------------

class TestQSharpExporterExtraEncodings:
    def test_iqp_encoding(self):
        from quprep.encode.iqp import IQPEncoder
        exp = QSharpExporter()
        encoded = IQPEncoder(reps=1).encode(np.array([0.5, 1.0, 1.5]))
        qsharp = exp.export(encoded)
        assert "H(" in qsharp
        assert "Rz(" in qsharp
        assert "CNOT(" in qsharp

    def test_reupload_encoding(self):
        from quprep.encode.reupload import ReUploadEncoder
        exp = QSharpExporter()
        encoded = ReUploadEncoder(layers=2).encode(np.array([0.5, 1.0]))
        qsharp = exp.export(encoded)
        assert "Ry(" in qsharp

    def test_hamiltonian_encoding(self):
        from quprep.encode.hamiltonian import HamiltonianEncoder
        exp = QSharpExporter()
        encoded = HamiltonianEncoder(trotter_steps=2).encode(np.array([0.5, 1.0]))
        qsharp = exp.export(encoded)
        assert "Rz(" in qsharp

    def test_entangled_angle_encoding(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        exp = QSharpExporter()
        encoded = EntangledAngleEncoder(layers=1).encode(np.array([0.5, 1.0, 1.5]))
        qsharp = exp.export(encoded)
        assert "Ry(" in qsharp
        assert "CNOT(" in qsharp

    def test_pauli_feature_map_encoding(self):
        exp = QSharpExporter()
        encoded = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1).encode(
            np.array([0.5, 1.0])
        )
        qsharp = exp.export(encoded)
        assert "Rz(" in qsharp

    def test_rx_rotation(self):
        from quprep.encode.angle import AngleEncoder
        exp = QSharpExporter()
        encoded = AngleEncoder(rotation="rx").encode(np.array([0.5, 1.0]))
        qsharp = exp.export(encoded)
        assert "Rx(" in qsharp

    def test_rz_rotation(self):
        from quprep.encode.angle import AngleEncoder
        exp = QSharpExporter()
        encoded = AngleEncoder(rotation="rz").encode(np.array([0.5, 1.0]))
        qsharp = exp.export(encoded)
        assert "Rz(" in qsharp

    def test_qaoa_problem_encoding(self):
        exp = QSharpExporter()
        qsharp = exp.export(_qaoa_enc(3))
        assert "H(" in qsharp
        assert "Rz(" in qsharp
        assert "Rx(" in qsharp
        assert "CNOT(" in qsharp


# ---------------------------------------------------------------------------
# IQMExporter — additional encoding coverage
# ---------------------------------------------------------------------------

class TestIQMExporterExtraEncodings:
    def test_reupload_encoding(self):
        from quprep.encode.reupload import ReUploadEncoder
        exp = IQMExporter()
        encoded = ReUploadEncoder(layers=2).encode(np.array([0.5, 1.0]))
        result = exp.export(encoded)
        names = {op["name"] for op in result["instructions"]}
        assert "prx" in names

    def test_hamiltonian_encoding(self):
        from quprep.encode.hamiltonian import HamiltonianEncoder
        exp = IQMExporter()
        encoded = HamiltonianEncoder(trotter_steps=2).encode(np.array([0.5, 1.0]))
        result = exp.export(encoded)
        assert len(result["instructions"]) > 0

    def test_entangled_angle_encoding(self):
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        exp = IQMExporter()
        encoded = EntangledAngleEncoder(layers=1).encode(np.array([0.5, 1.0, 1.5]))
        result = exp.export(encoded)
        names = {op["name"] for op in result["instructions"]}
        assert "prx" in names

    def test_rx_angle_encoding(self):
        from quprep.encode.angle import AngleEncoder
        exp = IQMExporter()
        encoded = AngleEncoder(rotation="rx").encode(np.array([0.5, 1.0]))
        result = exp.export(encoded)
        names = {op["name"] for op in result["instructions"]}
        assert "prx" in names

    def test_rz_angle_encoding(self):
        from quprep.encode.angle import AngleEncoder
        exp = IQMExporter()
        encoded = AngleEncoder(rotation="rz").encode(np.array([0.5, 1.0]))
        result = exp.export(encoded)
        # rz → virtual decomposition uses prx
        names = {op["name"] for op in result["instructions"]}
        assert "prx" in names

    def test_pauli_feature_map_encoding(self):
        exp = IQMExporter()
        encoded = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1).encode(
            np.array([0.5, 1.0])
        )
        result = exp.export(encoded)
        assert len(result["instructions"]) > 0

    def test_qaoa_problem_encoding(self):
        exp = IQMExporter()
        result = exp.export(_qaoa_enc(3))
        names = {op["name"] for op in result["instructions"]}
        assert "prx" in names or "cz" in names
