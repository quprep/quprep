"""
04 — Framework Export
======================
Export circuits to Qiskit, PennyLane, Cirq, TKET, and OpenQASM 3.0.
Frameworks without the optional dep installed are skipped automatically.

Install extras as needed:
    pip install quprep[qiskit]
    pip install quprep[pennylane]
    pip install quprep[cirq]
    pip install quprep[tket]

    uv run python examples/04_export.py
"""

import numpy as np

import quprep as qd

rng = np.random.default_rng(42)
X = rng.uniform(0, 1, size=(5, 3))

enc = qd.AngleEncoder(rotation="ry").encode(np.array([0.3, 1.1, 0.7]) * np.pi)

# ── OpenQASM 3.0 — no dependencies ───────────────────────────────────────────

print("=" * 55)
print("OpenQASM 3.0  (no extra dependencies)")
print("=" * 55)

exp = qd.QASMExporter()
print(exp.export(enc))

# Save a batch of circuits to individual files
result = qd.prepare(X, encoding="angle")
paths = exp.save_batch(result.encoded, "/tmp/quprep_batch/", stem="circuit")
print(f"Saved {len(paths)} circuits → {paths[0].parent}/")
print(f"  first : {paths[0].name}")
print(f"  last  : {paths[-1].name}")
print()

# ── Qiskit ───────────────────────────────────────────────────────────────────

print("=" * 55)
print("Qiskit  (pip install quprep[qiskit])")
print("=" * 55)

try:
    from quprep.export.qiskit_export import QiskitExporter

    qc = QiskitExporter().export(enc)
    print(qc.draw(output="text"))
    print(f"  num_qubits : {qc.num_qubits}")
    print(f"  depth      : {qc.depth()}")
except ImportError as e:
    print(f"  skipped — {e}")

print()

# ── PennyLane ─────────────────────────────────────────────────────────────────

print("=" * 55)
print("PennyLane  (pip install quprep[pennylane])")
print("=" * 55)

try:
    from quprep.export.pennylane_export import PennyLaneExporter

    qnode = PennyLaneExporter().export(enc)
    state = qnode()
    print(f"  QNode      : {qnode}")
    print(f"  State dim  : {len(state)}")
except ImportError as e:
    print(f"  skipped — {e}")

print()

# ── Cirq ─────────────────────────────────────────────────────────────────────

print("=" * 55)
print("Cirq  (pip install quprep[cirq])")
print("=" * 55)

try:
    from quprep.export.cirq_export import CirqExporter

    circuit = CirqExporter().export(enc)
    print(circuit)
    print(f"  qubits : {sorted(circuit.all_qubits())}")
except ImportError as e:
    print(f"  skipped — {e}")

print()

# ── TKET ─────────────────────────────────────────────────────────────────────

print("=" * 55)
print("TKET  (pip install quprep[tket])")
print("=" * 55)

try:
    from quprep.export.tket_export import TKETExporter

    circuit = TKETExporter().export(enc)
    print(f"  n_qubits : {circuit.n_qubits}")
    print(f"  n_gates  : {circuit.n_gates}")
    print(f"  depth    : {circuit.depth()}")
except ImportError as e:
    print(f"  skipped — {e}")

print()

# ── Amplitude encoding via Qiskit (StatePreparation) ─────────────────────────

print("=" * 55)
print("Amplitude → Qiskit  (StatePreparation gate)")
print("=" * 55)

try:
    from quprep.encode.amplitude import AmplitudeEncoder
    from quprep.export.qiskit_export import QiskitExporter

    unit_vec = np.array([0.5, 0.5, 0.5, 0.5])
    enc_amp = AmplitudeEncoder().encode(unit_vec)
    qc_amp = QiskitExporter().export(enc_amp)
    print(qc_amp.draw(output="text"))
except ImportError as e:
    print(f"  skipped — {e}")
