"""
How to Assess Encoding Quality
================================
Two complementary tools for evaluating an encoding before training:
  - encoding_sensitivity() + score_encoding() : how much variation each feature
    contributes and overall expressibility / kernel alignment
  - detect_barren_plateau() : analytical estimate of gradient variance risk

    uv run python examples/how-to/assess_encoding_quality.py
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning
from quprep.core.dataset import Dataset

rng = np.random.default_rng(42)
X = rng.uniform(0, np.pi, (60, 4))
dataset = Dataset(data=X)

print(f"quprep {qd.__version__}\n")


# ── 1. Encoding sensitivity ───────────────────────────────────────────────────
#
# encoding_sensitivity() perturbs each feature by a small epsilon and measures
# the infidelity 1 − |⟨ψ|ψ′⟩|² between the original and perturbed circuits.
# High infidelity → that feature strongly influences the circuit. Low infidelity
# → the feature is nearly invisible to the encoding.

print("── 1. encoding_sensitivity() ────────────────────────────────────────────")
enc = qd.AngleEncoder()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    sensitivity = qd.encoding_sensitivity(enc, dataset, epsilon=0.01, n_samples=20, seed=42)

print(f"   Features  : {len(sensitivity.feature_names)}")
print(f"   Epsilon   : {sensitivity.epsilon}")
print(f"   Samples   : {sensitivity.n_samples}")
print()
print(f"   {'Feature':<10} {'Infidelity':>12}")
print("   " + "─" * 24)
for i, score in enumerate(sensitivity.scores):
    print(f"   x{i:<9} {score:>12.6f}")

top = sensitivity.most_sensitive(2)
print(f"\n   Most sensitive : {top}")
print()


# ── 2. Encoding quality metrics ───────────────────────────────────────────────
#
# score_encoding() computes three simulation-based metrics:
#   expressibility  : how uniformly the encoder covers the Hilbert space
#   entanglement    : average entanglement capability across samples
#   kernel_alignment: how well the quantum kernel matches the linear kernel
# Circuits > 12 qubits return None (simulation is too expensive).

print("── 2. score_encoding() ──────────────────────────────────────────────────")
encoders_to_score = [
    ("angle",         qd.AngleEncoder()),
    ("dense_angle",   qd.DenseAngleEncoder()),
    ("entangled",     qd.EntangledAngleEncoder()),
]

def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"

for name, encoder in encoders_to_score:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuPrepWarning)
        scores = qd.score_encoding(encoder, dataset, n_samples=20, seed=42)
    expr = _fmt(scores.expressibility)
    ent  = _fmt(scores.entanglement_capability)
    ka   = _fmt(scores.kernel_alignment)
    print(f"   {name:<15} expr={expr}  entanglement={ent}  kernel={ka}")
print()


# ── 3. Barren plateau detection ───────────────────────────────────────────────
#
# detect_barren_plateau() uses analytical bounds (McClean 2018, Cerezo 2021)
# to estimate gradient variance without any circuit simulation. It returns a
# risk level: "none", "mild", "high", or "severe".

print("── 3. detect_barren_plateau() ───────────────────────────────────────────")

ds_large = Dataset(data=rng.uniform(0, np.pi, (30, 8)))
encoders_bp = [
    ("angle (shallow)",       qd.AngleEncoder(),               dataset),
    ("IQP (deep, large)",     qd.IQPEncoder(reps=3),           ds_large),
    ("entangled (moderate)",  qd.EntangledAngleEncoder(layers=2), dataset),
]

for name, encoder, ds in encoders_bp:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuPrepWarning)
        bp = qd.detect_barren_plateau(encoder, ds, cost_type="local")
    print(f"   {name:<30} qubits={bp.n_qubits}  depth={bp.circuit_depth}"
          f"  risk={bp.risk_level}")
print()
print("   Risk levels: none < mild < high < severe")
print("   Mitigation strategies for high/severe risk:")
bp_severe = qd.detect_barren_plateau(
    qd.IQPEncoder(reps=3), ds_large, cost_type="global"
)
for m in bp_severe.mitigations:
    print(f"   ↳ {m}")
