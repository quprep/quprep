"""
03 — Encoders Compared
======================
All 12 encoders side by side: Angle, Amplitude, Basis,
EntangledAngle, IQP, ReUpload, Hamiltonian, ZZFeatureMap,
PauliFeatureMap, RandomFourier, TensorProduct, QAOAProblem.

    uv run python examples/03_encoders.py
"""

import numpy as np

import quprep as qd
from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
from quprep.encode.qaoa_problem import QAOAProblemEncoder
from quprep.encode.random_fourier import RandomFourierEncoder
from quprep.encode.tensor_product import TensorProductEncoder
from quprep.encode.zz_feature_map import ZZFeatureMapEncoder

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

# ── 8. ZZ Feature Map ────────────────────────────────────────────────────────
#
#   Havlíček 2019 Qiskit convention. H + Rz(2(π−xᵢ)) + pairwise ZZ.

print("=" * 55)
print("ZZFeatureMapEncoder  (reps=1)")
print("=" * 55)

enc_zz = ZZFeatureMapEncoder(reps=1).encode(x3 * np.pi)
print(f"n_qubits   : {enc_zz.metadata['n_qubits']}")
print(f"depth      : {enc_zz.metadata['depth']}")
print(exporter.export(enc_zz))

# ── 9. Pauli Feature Map ──────────────────────────────────────────────────────
#
#   Generalised Pauli strings. ["Z", "ZZ"] is equivalent to ZZFeatureMap.

print("=" * 55)
print("PauliFeatureMapEncoder  (paulis=['Z','ZZ'], reps=1)")
print("=" * 55)

enc_pf = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1).encode(x3 * np.pi)
print(f"n_qubits   : {enc_pf.metadata['n_qubits']}")
print(f"depth      : {enc_pf.metadata['depth']}")
print(exporter.export(enc_pf))

# ── 10. Random Fourier ────────────────────────────────────────────────────────
#
#   RBF kernel approximation. Requires fit() before encode().

print("=" * 55)
print("RandomFourierEncoder  (n_components=6)")
print("=" * 55)

data_batch = np.random.default_rng(0).uniform(0, 1, size=(20, 3))
enc_rf_obj = RandomFourierEncoder(n_components=6, random_state=0)
enc_rf_obj.fit(data_batch)
enc_rf = enc_rf_obj.encode(x3)
print(f"n_qubits   : {enc_rf.metadata['n_qubits']}  (= n_components)")
print(f"depth      : {enc_rf.metadata['depth']}")
print(exporter.export(enc_rf))

# ── 11. Tensor Product ────────────────────────────────────────────────────────
#
#   Ry + Rz per qubit — full Bloch sphere, 2 features per qubit.

print("=" * 55)
print("TensorProductEncoder")
print("=" * 55)

enc_tp = TensorProductEncoder().encode(x3 * np.pi)
print(f"n_qubits   : {enc_tp.metadata['n_qubits']}  (ceil(3/2) = 2)")
print(f"depth      : {enc_tp.metadata['depth']}")
print(exporter.export(enc_tp))

# ── 12. QAOA Problem ──────────────────────────────────────────────────────────
#
#   QAOA-inspired feature map. Features as cost Hamiltonian parameters.
#   H + RZ(2γxᵢ) + CNOT-RZ(2γxᵢxⱼ)-CNOT + RX(2β). Linear connectivity.

print("=" * 55)
print("QAOAProblemEncoder  (p=1, linear)")
print("=" * 55)

enc_qaoa = QAOAProblemEncoder(p=1, connectivity="linear").encode(x3 * np.pi - np.pi / 2)
print(f"n_qubits       : {enc_qaoa.metadata['n_qubits']}")
print(f"n_pairs        : {enc_qaoa.metadata['n_pairs']}")
print(f"depth          : {enc_qaoa.metadata['depth']}")
print(exporter.export(enc_qaoa))

# ── 13. prepare() one-liner — all encoders ────────────────────────────────────

print("=" * 55)
print("prepare() one-liner — all encoders")
print("=" * 55)

data = np.random.default_rng(0).uniform(0, 1, size=(10, 3))
for enc_name in (
    "angle", "basis", "iqp", "reupload", "hamiltonian",
    "zz_feature_map", "tensor_product", "qaoa_problem",
):
    result = qd.prepare(data, encoding=enc_name, framework="qasm")
    meta = result.encoded[0].metadata
    print(f"  {enc_name:20s}  qubits={meta['n_qubits']}  depth={meta['depth']}")
