"""
03 — Encoders Compared
======================
AngleEncoder, AmplitudeEncoder, and BasisEncoder side by side.

    uv run python examples/03_encoders.py
"""

import numpy as np

from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.basis import BasisEncoder
from quprep.export.qasm_export import QASMExporter

exporter = QASMExporter()

# ── Shared input ─────────────────────────────────────────────────────────────

feature_vector = np.array([0.1, 0.9, 0.4, 0.6])

# ── 1. Angle encoding ────────────────────────────────────────────────────────
#
#   Each feature maps to a rotation angle on one qubit.
#   Encoder expects values in [0, π] for Ry, [-π, π] for Rx/Rz.
#   The Pipeline applies the correct normalization automatically.
#   Here we scale manually for illustration.

print("=" * 50)
print("AngleEncoder (Ry)")
print("=" * 50)

angle_input = feature_vector * np.pi          # scale to [0, π]
enc_angle = AngleEncoder(rotation="ry").encode(angle_input)
print(f"Input      : {angle_input.round(4)}")
print(f"Parameters : {enc_angle.parameters.round(4)}")
print(f"Metadata   : {enc_angle.metadata}")
print(exporter.export(enc_angle))

# ── 2. Amplitude encoding ────────────────────────────────────────────────────
#
#   The state vector |ψ⟩ = Σ xᵢ |i⟩ where ‖x‖₂ = 1.
#   Encodes 2ⁿ features into n qubits.

print("=" * 50)
print("AmplitudeEncoder")
print("=" * 50)

unit_vec = feature_vector / np.linalg.norm(feature_vector)   # must be unit norm
enc_amp = AmplitudeEncoder().encode(unit_vec)
print(f"Input      : {unit_vec.round(4)}  (‖x‖ = {np.linalg.norm(unit_vec):.4f})")
print(f"Parameters : {enc_amp.parameters.round(4)}")
print(f"Metadata   : {enc_amp.metadata}")
print("(QASM export not supported for amplitude — use QiskitExporter)")
print()

# ── 3. Basis encoding ────────────────────────────────────────────────────────
#
#   Each feature is binarized: 1 → X gate (flip qubit), 0 → leave |0⟩.

print("=" * 50)
print("BasisEncoder (threshold=0.5)")
print("=" * 50)

enc_basis = BasisEncoder(threshold=0.5).encode(feature_vector)
print(f"Input      : {feature_vector}")
print(f"Parameters : {enc_basis.parameters}  (binary)")
print(f"Metadata   : {enc_basis.metadata}")
print(exporter.export(enc_basis))

# ── 4. Padding in AmplitudeEncoder ───────────────────────────────────────────

print("=" * 50)
print("AmplitudeEncoder — auto-padding to power of two")
print("=" * 50)

vec5 = np.array([0.4, 0.3, 0.5, 0.6, 0.4])          # 5 features → pad to 8
vec5 = vec5 / np.linalg.norm(vec5)
enc5 = AmplitudeEncoder(pad=True).encode(vec5)
print(f"Input length  : 5 → padded to {len(enc5.parameters)}")
print(f"n_qubits      : {enc5.metadata['n_qubits']}")
print(f"‖padded vec‖  : {np.linalg.norm(enc5.parameters):.6f}")
