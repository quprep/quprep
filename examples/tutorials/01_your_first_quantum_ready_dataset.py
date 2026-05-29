"""
Your First Quantum-Ready Dataset
=================================
You have a table of data. You want to run it through a variational quantum
classifier. But quantum computers don't understand raw numbers — they work
with quantum states. This tutorial shows you the full path from a raw dataset
to encoded quantum circuits, and explains *why* each step matters.

    uv run python examples/tutorials/01_your_first_quantum_ready_dataset.py

No optional dependencies required.
"""

from sklearn.datasets import load_iris

import quprep as qd

print(f"quprep {qd.__version__}\n")


# ── 1. Load a real dataset ────────────────────────────────────────────────────
#
# We use the Iris dataset — 150 samples, 4 features (sepal/petal length and
# width), 3 flower species. It ships inside scikit-learn, which QuPrep
# already depends on, so no extra install is needed.

iris = load_iris()
X_raw = iris.data[:12]   # 12 samples to keep output readable
y = iris.target[:12]

print("── 1. Raw data ──────────────────────────────────────────────────────────")
print(f"   Shape : {X_raw.shape}  (samples × features)")
print(f"   Range : [{X_raw.min():.2f}, {X_raw.max():.2f}]")
print(f"   Labels: {y}\n")


# ── 2. Why preprocessing matters for quantum encoding ─────────────────────────
#
# Angle encoding maps each feature value to a rotation angle on a qubit.
# AngleEncoder (the most common encoder) uses Ry gates, which expect angles
# in [0, π] ≈ [0, 3.14]. Raw Iris values go up to 7.9 — far outside that
# range. Feeding raw data in would saturate the encoding, compressing most
# of the variation into the last few degrees of rotation.
#
# QuPrep's Pipeline handles this automatically. Here's what it does:
#   Imputer  → fills any missing values (NaN)
#   Scaler   → rescales each feature to [0, π] for angle encoding
#   AngleEncoder → maps each feature to an Ry gate rotation

dataset = qd.NumpyIngester().load(X_raw, y=y)

pipeline = qd.Pipeline(
    cleaner=qd.Imputer(strategy="mean"),
    normalizer=qd.Scaler(strategy="minmax_pi"),   # → [0, π]
    encoder=qd.AngleEncoder(),
)

print("── 2. Pipeline ──────────────────────────────────────────────────────────")
print("   Imputer(mean) → Scaler(minmax_pi) → AngleEncoder")
print()


# ── 3. Fit and transform ──────────────────────────────────────────────────────
#
# fit_transform() learns the scaling parameters from this data and applies
# the full pipeline in one shot. The result is a PipelineResult that holds
# the encoded circuits alongside metadata about every stage.

result = pipeline.fit_transform(dataset)

print("── 3. After preprocessing ───────────────────────────────────────────────")
print(f"   Scaled range : [{result.dataset.data.min():.3f}, {result.dataset.data.max():.3f}]")
print(f"   Encoded      : {len(result.encoded)} circuits")
print()


# ── 4. Check compatibility before trusting the output ─────────────────────────
#
# check_compatibility() verifies that every value in your encoded dataset
# actually falls within the valid range for the encoder you chose. It catches
# problems that would silently produce wrong circuits.

report = qd.check_compatibility(qd.AngleEncoder(), result.dataset)

print("── 4. Compatibility check ───────────────────────────────────────────────")
print(f"   Compatible : {report.is_compatible}")
print(f"   Warnings   : {len(report.warnings)}")
print(f"   Errors     : {len(report.errors)}")
print()


# ── 5. Look at a circuit ──────────────────────────────────────────────────────
#
# Each sample becomes a quantum circuit. The ASCII diagram shows one qubit
# per feature. Each qubit gets an Ry rotation whose angle is the scaled
# feature value. This is the circuit you would hand to PennyLane, Qiskit,
# or any other quantum framework.

print("── 5. First encoded circuit (ASCII) ─────────────────────────────────────")
print(qd.draw_ascii(result.encoded[0]))
print()


# ── 6. Export to OpenQASM 3.0 ─────────────────────────────────────────────────
#
# Most quantum frameworks accept OpenQASM 3.0. QuPrep's exporter converts
# the encoded result to a QASM string you can paste directly into Qiskit,
# PennyLane, or any QASM-compatible simulator.

exporter = qd.QASMExporter()
qasm = exporter.export(result.encoded[0])

print("── 6. OpenQASM 3.0 output (first sample) ────────────────────────────────")
print(qasm)


# ── Next steps ────────────────────────────────────────────────────────────────
print("── Next steps ───────────────────────────────────────────────────────────")
print("   → tutorials/02  : handle messy real-world data (NaN, outliers, imbalance)")
print("   → tutorials/03  : connect to Qiskit end-to-end with auto pipeline")
print("   → how-to/choose_an_encoder : pick the right encoder for your task")
print("   → how-to/export_to_frameworks : export to PennyLane, Cirq, TKET, ...")
