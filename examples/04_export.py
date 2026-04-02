"""
04 — Framework Export
======================
Export circuits to all 8 supported frameworks:
OpenQASM 3.0, Qiskit, PennyLane, Cirq, TKET, Amazon Braket, Q#, IQM.

Frameworks without the optional dep installed are skipped automatically.

Install extras as needed:
    pip install quprep[qiskit]
    pip install quprep[pennylane]
    pip install quprep[cirq]
    pip install quprep[tket]
    pip install quprep[braket]
    pip install quprep[qsharp]   # Azure Quantum submission
    pip install quprep[iqm]      # IQM hardware submission

    uv run python examples/04_export.py
"""

import json

import numpy as np

import quprep as qd
from quprep.encode.zz_feature_map import ZZFeatureMapEncoder
from quprep.export.iqm_export import IQMExporter
from quprep.export.qsharp_export import QSharpExporter

rng = np.random.default_rng(42)
X = rng.uniform(0, 1, size=(5, 3))

enc = qd.AngleEncoder(rotation="ry").encode(np.array([0.3, 1.1, 0.7]) * np.pi)
enc_zz = ZZFeatureMapEncoder(reps=1).encode(np.array([0.5, 1.2, 0.8]))

# ── 1. OpenQASM 3.0 — no dependencies ────────────────────────────────────────

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

# ── 2. Qiskit ─────────────────────────────────────────────────────────────────

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

# ── 3. PennyLane ──────────────────────────────────────────────────────────────

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

# ── 4. Cirq ───────────────────────────────────────────────────────────────────

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

# ── 5. TKET ───────────────────────────────────────────────────────────────────

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

# ── 6. Amazon Braket ──────────────────────────────────────────────────────────

print("=" * 55)
print("Amazon Braket  (pip install quprep[braket])")
print("=" * 55)

try:
    from braket.circuits import Circuit  # noqa: F401

    from quprep.export.braket_export import BraketExporter

    exp_bk = BraketExporter()

    circuit = exp_bk.export(enc)
    print("Angle encoding:")
    print(circuit)
    print()

    circuit_zz = exp_bk.export(enc_zz)
    print("ZZ feature map:")
    print(circuit_zz)
    print()

    result2 = qd.prepare(X[:2], encoding="angle", framework="braket")
    print(f"prepare() → braket: {type(result2.circuits[0])}")

except ImportError as e:
    print(f"  skipped — {e}")

print()

# ── 7. Q# (Microsoft Azure Quantum) ──────────────────────────────────────────
#
#   No deps needed for string generation.
#   pip install quprep[qsharp] only for actual Azure Quantum submission.

print("=" * 55)
print("Q#  (no extra deps for string generation)")
print("=" * 55)

exp_qs = QSharpExporter(namespace="QuPrepDemo", operation_name="EncodeFeatures")

qsharp_src = exp_qs.export(enc)
print("Angle encoding → Q# source:")
print(qsharp_src)

qsharp_zz = exp_qs.export(enc_zz)
print("ZZ feature map (first 6 lines):")
for line in qsharp_zz.splitlines()[:6]:
    print(line)
print("  ...")
print()

result_qs = qd.prepare(X[:2], encoding="angle", framework="qsharp")
print(f"prepare() → qsharp: str = {isinstance(result_qs.circuits[0], str)}")
print()

# ── 8. IQM Native Format ──────────────────────────────────────────────────────
#
#   No deps needed for dict generation.
#   pip install quprep[iqm] for iqm_client.Circuit.from_dict() + hardware.
#   Native gate set: PRX(angle_t, phase_t) and CZ.

print("=" * 55)
print("IQM  (no extra deps for dict generation)")
print("=" * 55)

exp_iqm = IQMExporter(circuit_name="feature_map", qubit_prefix="QB")

circuit_iqm = exp_iqm.export(enc)
print("Angle encoding (IQM dict):")
print(json.dumps(circuit_iqm, indent=2))
print()

circuit_iqm_zz = exp_iqm.export(enc_zz)
gate_names = [op["name"] for op in circuit_iqm_zz["instructions"]]
print(f"ZZ feature map: {len(circuit_iqm_zz['instructions'])} instructions")
print(f"  gate types : {sorted(set(gate_names))}")
print(f"  CZ count   : {gate_names.count('cz')}")
print()

result_iqm = qd.prepare(X[:2], encoding="angle", framework="iqm")
print(f"prepare() → iqm: dict = {isinstance(result_iqm.circuits[0], dict)}")
print()

# ── 9. Amplitude encoding via Qiskit (StatePreparation) ──────────────────────

print("=" * 55)
print("Amplitude → Qiskit  (StatePreparation gate)")
print("=" * 55)

try:
    from quprep.export.qiskit_export import QiskitExporter

    unit_vec = np.array([0.5, 0.5, 0.5, 0.5])
    enc_amp = qd.AmplitudeEncoder().encode(unit_vec)
    qc_amp = QiskitExporter().export(enc_amp)
    print(qc_amp.draw(output="text"))
except ImportError as e:
    print(f"  skipped — {e}")
