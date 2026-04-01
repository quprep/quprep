"""
12 — New Exporters (v0.6.0)
============================
Amazon Braket, Q# (Microsoft Azure Quantum), IQM native format.

Install extras:
    pip install quprep[braket]   # Amazon Braket
    pip install quprep[qsharp]   # Azure Quantum (string generation is free)
    pip install quprep[iqm]      # IQM hardware submission (dict generation is free)

    uv run python examples/12_exporters.py
"""

import json

import numpy as np

import quprep as qd
from quprep.encode.zz_feature_map import ZZFeatureMapEncoder
from quprep.export.iqm_export import IQMExporter
from quprep.export.qsharp_export import QSharpExporter

rng = np.random.default_rng(42)
X = rng.uniform(0, 1, size=(5, 3)) * np.pi

enc_angle = qd.AngleEncoder(rotation="ry").encode(np.array([0.3, 1.1, 0.7]))
enc_zz    = ZZFeatureMapEncoder(reps=1).encode(np.array([0.5, 1.2, 0.8]))

# ── 1. Amazon Braket ─────────────────────────────────────────────────────────

print("=" * 55)
print("BraketExporter  (pip install quprep[braket])")
print("=" * 55)

try:
    from braket.circuits import Circuit  # noqa: F401

    from quprep.export.braket_export import BraketExporter

    exp = BraketExporter()

    # Angle encoding
    circuit = exp.export(enc_angle)
    print("Angle encoding:")
    print(circuit)
    print()

    # ZZ feature map
    circuit_zz = exp.export(enc_zz)
    print("ZZ feature map:")
    print(circuit_zz)
    print()

    # Batch export
    result = qd.prepare(X, encoding="angle")
    batch = exp.export_batch(result.encoded)
    print(f"Batch: {len(batch)} circuits, first circuit depth: {batch[0].depth}")
    print()

    # Via prepare()
    result2 = qd.prepare(X[:2], encoding="angle", framework="braket")
    print(f"prepare() → braket: {type(result2.circuits[0])}")

except ImportError as e:
    print(f"  skipped — {e}")

print()

# ── 2. Q# (Microsoft Azure Quantum) ─────────────────────────────────────────
#
#   No deps needed for string generation.
#   pip install quprep[qsharp] only for actual Azure Quantum submission.

print("=" * 55)
print("QSharpExporter  (no extra deps for string generation)")
print("=" * 55)

exp_qs = QSharpExporter(namespace="QuPrepDemo", operation_name="EncodeFeatures")

# Angle encoding → Q# source
qsharp_angle = exp_qs.export(enc_angle)
print("Angle encoding → Q# source:")
print(qsharp_angle)

# ZZ feature map → Q# source
qsharp_zz = exp_qs.export(enc_zz)
print("ZZ feature map → Q# source (first 6 lines):")
for line in qsharp_zz.splitlines()[:6]:
    print(line)
print("  ...")
print()

# Custom namespace / operation name
exp_qs2 = QSharpExporter(namespace="MyOrg.Circuits", operation_name="KernelMap")
qsharp_custom = exp_qs2.export(enc_angle)
print("Custom namespace:")
print(qsharp_custom.splitlines()[0])   # namespace line
print()

# Batch
batch_qs = exp_qs.export_batch([enc_angle, enc_zz])
print(f"Batch: {len(batch_qs)} Q# source strings")
print()

# Via prepare()
result_qs = qd.prepare(X[:2], encoding="angle", framework="qsharp")
print(f"prepare() → qsharp: first circuit is str = {isinstance(result_qs.circuits[0], str)}")
print()

# ── 3. IQM Native Format ─────────────────────────────────────────────────────
#
#   No deps needed for dict generation.
#   pip install quprep[iqm] for iqm_client.Circuit.from_dict() and hardware submission.
#
#   Native gate set: PRX(angle_t, phase_t) and CZ.
#   Ry = PRX(θ/2π, 0.25), Rx = PRX(θ/2π, 0.0), Rz = H·Rx·H (virtual).

print("=" * 55)
print("IQMExporter  (no extra deps for dict generation)")
print("=" * 55)

exp_iqm = IQMExporter(circuit_name="feature_map", qubit_prefix="QB")

# Angle encoding
circuit_iqm = exp_iqm.export(enc_angle)
print("Angle encoding (IQM dict):")
print(json.dumps(circuit_iqm, indent=2))
print()

# ZZ feature map — shows CZ gates from pairwise interactions
circuit_iqm_zz = exp_iqm.export(enc_zz)
gate_names = [op["name"] for op in circuit_iqm_zz["instructions"]]
print(f"ZZ feature map: {len(circuit_iqm_zz['instructions'])} instructions")
print(f"  gate types used: {sorted(set(gate_names))}")
print(f"  CZ count: {gate_names.count('cz')}")
print()

# Custom qubit prefix (e.g. IQM Garnet uses QB1..QB20)
exp_iqm2 = IQMExporter(circuit_name="garnet_circuit", qubit_prefix="QB")
c2 = exp_iqm2.export(enc_angle)
qubit_labels = sorted({q for op in c2["instructions"] for q in op["qubits"]})
print(f"Qubit labels: {qubit_labels}")
print()

# JSON-serializable — ready to write to disk
json_str = json.dumps(circuit_iqm, indent=2)
print(f"JSON bytes: {len(json_str)}")
print()

# Batch
batch_iqm = exp_iqm.export_batch([enc_angle, enc_zz])
print(f"Batch: {len(batch_iqm)} circuit dicts")
print()

# Via prepare()
result_iqm = qd.prepare(X[:2], encoding="angle", framework="iqm")
print(f"prepare() → iqm: first circuit is dict = {isinstance(result_iqm.circuits[0], dict)}")
