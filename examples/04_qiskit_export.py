"""
04 — Qiskit Export
==================
Export circuits as Qiskit QuantumCircuit objects.

Requires:
    pip install quprep[qiskit]

    uv run python examples/04_qiskit_export.py
"""

import sys

import numpy as np
import pandas as pd

# ── Check optional dependency ─────────────────────────────────────────────────

try:
    import qiskit  # noqa: F401
except ImportError:
    print("Qiskit is not installed.")
    print("Install it with:  pip install quprep[qiskit]")
    sys.exit(0)

from quprep import Pipeline
from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.basis import BasisEncoder
from quprep.export.qiskit_export import QiskitExporter

exporter = QiskitExporter()

# ── 1. Angle encoding → QuantumCircuit ───────────────────────────────────────

print("=" * 50)
print("Angle (Ry) → Qiskit QuantumCircuit")
print("=" * 50)

vec = np.array([0.3, 1.1, 0.7, 2.0])
enc = AngleEncoder(rotation="ry").encode(vec)
qc = exporter.export(enc)
print(qc)
print(qc.draw(output="text"))

# ── 2. Amplitude encoding → QuantumCircuit ───────────────────────────────────

print("=" * 50)
print("Amplitude → Qiskit QuantumCircuit")
print("=" * 50)

unit_vec = np.array([0.5, 0.5, 0.5, 0.5])
enc_amp = AmplitudeEncoder().encode(unit_vec)
qc_amp = exporter.export(enc_amp)
print(qc_amp.draw(output="text"))

# ── 3. Basis encoding → QuantumCircuit ───────────────────────────────────────

print("=" * 50)
print("Basis → Qiskit QuantumCircuit")
print("=" * 50)

bits = np.array([1.0, 0.0, 1.0, 1.0])
enc_basis = BasisEncoder().encode(bits)
qc_basis = exporter.export(enc_basis)
print(qc_basis.draw(output="text"))

# ── 4. Full pipeline → Qiskit ────────────────────────────────────────────────

print("=" * 50)
print("Pipeline → Qiskit (angle encoding)")
print("=" * 50)

df = pd.DataFrame(
    {
        "x1": [0.1, 0.5, 0.9],
        "x2": [0.4, 0.2, 0.8],
        "x3": [0.7, 0.6, 0.3],
    }
)

pipeline = Pipeline(
    encoder=AngleEncoder(rotation="rx"),
    exporter=QiskitExporter(),
)
result = pipeline.fit_transform(df)

for i, qc in enumerate(result.circuits):
    print(f"\nSample {i}:")
    print(qc.draw(output="text"))
