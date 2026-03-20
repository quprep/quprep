"""Tests for framework exporters."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.basis import BasisEncoder
from quprep.export.qasm_export import QASMExporter

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
        from quprep.encode.base import EncodedResult
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
        from quprep.encode.base import EncodedResult
        from quprep.export.qiskit_export import QiskitExporter
        exp = QiskitExporter()
        fake = EncodedResult(
            parameters=np.array([0.1, 0.2]),
            metadata={"encoding": "iqp", "n_qubits": 2},
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
        from quprep.encode.base import EncodedResult
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
        from quprep.encode.base import EncodedResult
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
        from quprep.encode.base import EncodedResult
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
