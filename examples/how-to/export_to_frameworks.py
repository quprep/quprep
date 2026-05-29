"""
How to Export to Quantum Frameworks
=====================================
QuPrep encodes your data into circuits that can be exported to 8 frameworks:
OpenQASM 3.0, Qiskit, PennyLane, Cirq, TKET, Amazon Braket, Q#, and IQM.

Frameworks without their optional dependency installed are skipped automatically.

Install extras as needed:
    pip install quprep[qiskit]
    pip install quprep[pennylane]
    pip install quprep[cirq]
    pip install quprep[tket]
    pip install quprep[braket]
    pip install quprep[qsharp]
    pip install quprep[iqm]

    uv run python examples/how-to/export_to_frameworks.py
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning

rng = np.random.default_rng(0)
X = rng.uniform(0, np.pi, (4, 3))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = qd.Pipeline(
        normalizer=qd.Scaler(strategy="minmax_pi"),
        encoder=qd.AngleEncoder(),
    ).fit_transform(qd.NumpyIngester().load(X))

encoded = result.encoded
n_qubits = encoded[0].metadata["n_qubits"]
print(f"quprep {qd.__version__} | {len(encoded)} circuits, {n_qubits} qubits\n")


# ── OpenQASM 3.0 (no extras required) ────────────────────────────────────────

print("── OpenQASM 3.0 ─────────────────────────────────────────────────────────")
qasm = qd.QASMExporter().export(encoded[0])
print(qasm)


# ── Qiskit ────────────────────────────────────────────────────────────────────

print("── Qiskit  (pip install quprep[qiskit]) ─────────────────────────────────")
try:
    from quprep.export.qiskit_export import QiskitExporter
    circuit = QiskitExporter().export(encoded[0])
    print(f"   QuantumCircuit: {circuit}")
    print(f"   n_qubits={circuit.num_qubits}  depth={circuit.depth()}")
except ImportError:
    print("   skipped — run: pip install quprep[qiskit]")
print()


# ── PennyLane ─────────────────────────────────────────────────────────────────

print("── PennyLane  (pip install quprep[pennylane]) ───────────────────────────")
try:
    from quprep.export.pennylane_export import PennyLaneExporter
    tape = PennyLaneExporter().export(encoded[0])
    print(f"   type : {type(tape).__name__}")
except ImportError:
    print("   skipped — run: pip install quprep[pennylane]")
print()


# ── Cirq ──────────────────────────────────────────────────────────────────────

print("── Cirq  (pip install quprep[cirq]) ─────────────────────────────────────")
try:
    from quprep.export.cirq_export import CirqExporter
    cirq_circuit = CirqExporter().export(encoded[0])
    print(f"   Circuit type : {type(cirq_circuit).__name__}")
    print(cirq_circuit)
except ImportError:
    print("   skipped — run: pip install quprep[cirq]")
print()


# ── TKET ──────────────────────────────────────────────────────────────────────

print("── TKET  (pip install quprep[tket]) ─────────────────────────────────────")
try:
    from quprep.export.tket_export import TKETExporter
    tket_circuit = TKETExporter().export(encoded[0])
    print(f"   Circuit type : {type(tket_circuit).__name__}")
    print(f"   n_qubits     : {tket_circuit.n_qubits}")
except ImportError:
    print("   skipped — run: pip install quprep[tket]")
print()


# ── Amazon Braket ─────────────────────────────────────────────────────────────

print("── Amazon Braket  (pip install quprep[braket]) ──────────────────────────")
try:
    from quprep.export.braket_export import BraketExporter
    braket_circuit = BraketExporter().export(encoded[0])
    print(f"   Circuit type : {type(braket_circuit).__name__}")
except ImportError:
    print("   skipped — run: pip install quprep[braket]")
print()


# ── Quick export via qd.prepare() ────────────────────────────────────────────
#
# qd.prepare() is a one-shot convenience that ingests, preprocesses, encodes,
# and exports in a single call. Pass framework= to select the target.

print("── qd.prepare() one-shot export ─────────────────────────────────────────")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    r = qd.prepare(X, encoding="angle", framework="qasm")
print(f"   prepare(framework='qasm') → {type(r.circuit).__name__}")
print(r.circuit)
