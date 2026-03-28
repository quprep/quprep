"""
03 — Encoders Compared
======================
All 7 encoders side by side: Angle, Amplitude, Basis,
EntangledAngle, IQP, ReUpload, Hamiltonian.

    uv run python examples/03_encoders.py
"""

import numpy as np

import quprep as qd

exporter = qd.QASMExporter()

feature_vector = np.array([0.1, 0.9, 0.4, 0.6])
x3 = np.array([0.5, 1.2, 0.75])

# ── 1. Angle encoding ────────────────────────────────────────────────────────
#
#   Each feature maps to a rotation angle on one qubit.
#   Pipeline auto-normalizes to [0, π] for Ry, [−π, π] for Rx/Rz.

print("=" * 55)
print("AngleEncoder (Ry)")
print("=" * 55)

angle_input = feature_vector * np.pi
enc_angle = qd.AngleEncoder(rotation="ry").encode(angle_input)
print(f"Input      : {angle_input.round(4)}")
print(f"Metadata   : {enc_angle.metadata}")
print(exporter.export(enc_angle))

# ── 2. Amplitude encoding ─────────────────────────────────────────────────────
#
#   |ψ(x)⟩ = Σ xᵢ|i⟩  where ‖x‖₂ = 1. Uses log₂(d) qubits.

print("=" * 55)
print("AmplitudeEncoder")
print("=" * 55)

unit_vec = feature_vector / np.linalg.norm(feature_vector)
enc_amp = qd.AmplitudeEncoder().encode(unit_vec)
print(f"Input      : {unit_vec.round(4)}  (‖x‖ = {np.linalg.norm(unit_vec):.4f})")
print(f"n_qubits   : {enc_amp.metadata['n_qubits']}  (log₂(4) = 2)")
print(f"padded     : {enc_amp.metadata['padded']}")
print("(QASM not supported for amplitude — use QiskitExporter)")
print()

# ── 3. Basis encoding ─────────────────────────────────────────────────────────
#
#   Binarizes features: value ≥ threshold → X gate (|1⟩), else |0⟩.

print("=" * 55)
print("BasisEncoder (threshold=0.5)")
print("=" * 55)

enc_basis = qd.BasisEncoder(threshold=0.5).encode(feature_vector)
print(f"Input      : {feature_vector}")
print(f"Parameters : {enc_basis.parameters}  (binary)")
print(exporter.export(enc_basis))

# ── 4. Entangled angle encoding ───────────────────────────────────────────────
#
#   Rotation layer + CNOT entangling layer, repeated `layers` times.
#   Topologies: linear, circular, full.

print("=" * 55)
print("EntangledAngleEncoder  (circular, 2 layers)")
print("=" * 55)

enc_ent = qd.EntangledAngleEncoder(rotation="ry", layers=2, entanglement="circular").encode(
    x3 * np.pi
)
print(f"n_qubits   : {enc_ent.metadata['n_qubits']}")
print(f"cnot_pairs : {enc_ent.metadata['cnot_pairs']}")
print(f"depth      : {enc_ent.metadata['depth']}")
print(exporter.export(enc_ent))

# ── 5. IQP encoding ───────────────────────────────────────────────────────────
#
#   Havlíček et al. 2019. Hadamard + Rz + ZZ interactions.
#   Best for kernel methods.

print("=" * 55)
print("IQPEncoder  (reps=1)")
print("=" * 55)

enc_iqp = qd.IQPEncoder(reps=1).encode(x3 * np.pi)
print(f"n_qubits   : {enc_iqp.metadata['n_qubits']}")
print(f"n_pairs    : {enc_iqp.metadata['n_pairs']}")
print(f"depth      : {enc_iqp.metadata['depth']}")
print(exporter.export(enc_iqp))

# ── 6. Data re-uploading ──────────────────────────────────────────────────────
#
#   Pérez-Salinas et al. 2020. Same rotation layer repeated L times.
#   Universal approximator with enough layers.

print("=" * 55)
print("ReUploadEncoder  (layers=3)")
print("=" * 55)

enc_ru = qd.ReUploadEncoder(layers=3, rotation="ry").encode(x3 * np.pi)
print(f"n_qubits   : {enc_ru.metadata['n_qubits']}")
print(f"layers     : {enc_ru.metadata['layers']}")
print(f"depth      : {enc_ru.metadata['depth']}")
print(exporter.export(enc_ru))

# ── 7. Hamiltonian encoding ───────────────────────────────────────────────────
#
#   Trotterized Z Hamiltonian: e^{-iH(x)T}. Best for physics simulation / VQE.

print("=" * 55)
print("HamiltonianEncoder  (trotter_steps=4)")
print("=" * 55)

enc_ham = qd.HamiltonianEncoder(evolution_time=1.0, trotter_steps=4).encode(x3)
print(f"n_qubits       : {enc_ham.metadata['n_qubits']}")
print(f"trotter_steps  : {enc_ham.metadata['trotter_steps']}")
print(f"depth          : {enc_ham.metadata['depth']}")
print(exporter.export(enc_ham))

# ── 8. prepare() one-liner ────────────────────────────────────────────────────

print("=" * 55)
print("prepare() one-liner")
print("=" * 55)

data = np.random.default_rng(0).uniform(0, 1, size=(10, 3))
for enc_name in ("angle", "basis", "iqp", "reupload", "hamiltonian"):
    result = qd.prepare(data, encoding=enc_name, framework="qasm")
    meta = result.encoded[0].metadata
    print(f"  {enc_name:20s}  qubits={meta['n_qubits']}  depth={meta['depth']}")
