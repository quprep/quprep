"""
End-to-End with a Quantum Framework
=====================================
You have a dataset and you're not sure which encoder to use, how many qubits
you need, or how to structure the preprocessing pipeline. QuPrep can answer
all three questions automatically — and then produce circuits that go straight
into Qiskit, PennyLane, or any other framework.

This tutorial shows the intelligent path: let QuPrep audit your data, suggest
a pipeline, build it, verify the output, and export ready-to-use QASM circuits.

    uv run python examples/tutorials/03_end_to_end_with_a_framework.py

No optional dependencies required.
"""

import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer

import quprep as qd
from quprep import QuPrepWarning

print(f"quprep {qd.__version__}\n")


# ── 1. Load a dataset ─────────────────────────────────────────────────────────
#
# Breast cancer classification: 569 samples, 30 features, binary target.
# We use 60 samples and 6 features to keep the output readable, but the
# same workflow scales to the full dataset.

bc = load_breast_cancer()
X_raw = bc.data[:60, :6]
y     = bc.target[:60]

dataset = qd.NumpyIngester().load(X_raw, y=y)

print("── 1. Dataset ───────────────────────────────────────────────────────────")
print(f"   Samples  : {dataset.data.shape[0]}")
print(f"   Features : {dataset.data.shape[1]}")
print(f"   Classes  : {np.bincount(y.astype(int))}")
print(f"   Range    : [{dataset.data.min():.2f}, {dataset.data.max():.2f}]")
print()


# ── 2. Audit the dataset before doing anything ────────────────────────────────
#
# preprocessing_report() runs a structured audit of your dataset and encoder
# combination. It checks for NaN, outliers, qubit budget overruns, class
# imbalance (>3:1 ratio), and encoder compatibility — giving you a clear
# picture of what needs fixing before you build a pipeline.

print("── 2. Preprocessing report ──────────────────────────────────────────────")
report = qd.preprocessing_report(
    dataset,
    encoder=qd.AngleEncoder(),
    qubit_budget=6,
)
print(f"   Issues          : {report.n_issues}")
if report.recommendations:
    for rec in report.recommendations:
        print(f"   Recommendation  : {rec}")
else:
    print("   Recommendations : none — data looks healthy for this encoder")
print()


# ── 3. Let QuPrep suggest a pipeline ─────────────────────────────────────────
#
# suggest_pipeline() analyses your dataset and returns a PipelineSuggestion:
# a recommended combination of imputer, outlier handler, reducer, normalizer,
# and encoder. It uses simple heuristics (skewness proxy, IQR detection, PCA
# when features > qubit budget) to make reasonable choices.
#
# This is not magic — it's a starting point. You can always override any
# component. But for getting started quickly, it removes the guesswork.

suggestion = qd.suggest_pipeline(dataset, task="classification", qubits=6)

print("── 3. Pipeline suggestion ───────────────────────────────────────────────")
print(f"   Encoder          : {suggestion.encoder}")
print(f"   Normalizer       : {suggestion.normalizer}")
print(f"   Imputer          : {suggestion.imputer or 'none needed'}")
print(f"   Outlier handler  : {suggestion.outlier_handler or 'none needed'}")
print(f"   Reasoning        : {suggestion.reason}")
print()


# ── 4. Build and run the suggested pipeline ───────────────────────────────────
#
# PipelineSuggestion.build() constructs a ready-to-use Pipeline from the
# suggestion. You can also build it manually if you want to customise
# individual components.

pipeline = suggestion.build()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = pipeline.fit_transform(dataset)

encoder_instance = pipeline.encoder  # keep a reference for verify_encoding

print("── 4. Pipeline output ───────────────────────────────────────────────────")
print(f"   Circuits    : {len(result.encoded)}")
print(f"   Qubits      : {result.encoded[0].metadata.get('n_qubits')}")
print(f"   Depth       : {result.encoded[0].metadata.get('depth')}")
print()


# ── 5. Verify the encoding ────────────────────────────────────────────────────
#
# verify_encoding() checks post-encoding invariants: for angle encoders, that
# all values are within the valid rotation range; for amplitude encoders, that
# state vectors have unit norm. It's a final sanity check before you hand
# circuits to a simulator or real hardware.

verify_report = qd.verify_encoding(result.encoded, encoder_instance)

print("── 5. Encoding verification ─────────────────────────────────────────────")
print(f"   Passed  : {verify_report.passed}")
for check in verify_report.checks:
    status = "✓" if check["passed"] else "✗"
    print(f"   {status} {check['name']} : {check['detail']}")
print()


# ── 6. Export to OpenQASM 3.0 ─────────────────────────────────────────────────
#
# OpenQASM 3.0 is accepted by Qiskit, PennyLane, Cirq, and most simulators.
# For native framework objects (Qiskit QuantumCircuit, PennyLane tapes),
# see how-to/export_to_frameworks.

exporter = qd.QASMExporter()
qasm = exporter.export(result.encoded[0])

print("── 6. OpenQASM 3.0 — first circuit ─────────────────────────────────────")
print(qasm)


# ── 7. ASCII circuit diagram ──────────────────────────────────────────────────

print("── 7. Circuit diagram ───────────────────────────────────────────────────")
print(qd.draw_ascii(result.encoded[0]))


# ── Next steps ────────────────────────────────────────────────────────────────
print("── Next steps ───────────────────────────────────────────────────────────")
print("   → how-to/choose_an_encoder    : compare all 15 encoders for your task")
print("   → how-to/export_to_frameworks : export to PennyLane, Cirq, TKET, ...")
print("   → how-to/validate_before_encoding : deep-dive into check_compatibility")
print("   → how-to/assess_encoding_quality  : expressibility, barren plateau risk")
