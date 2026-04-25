"""
14 — Noise-Aware Preprocessing
================================
Assign high-variance features to the least-noisy qubits, minimise SWAP
overhead given a real hardware topology, and remap angles away from the
0/π poles where gate errors peak.

    uv run python examples/14_noise_aware.py
"""

import numpy as np

from quprep.core.dataset import Dataset
from quprep.preprocess.noise_aware import NoiseAwarePreprocessor, NoiseProfile

# ── Shared data ───────────────────────────────────────────────────────────────

rng = np.random.default_rng(0)
# 4 features with very different variances — the preprocessor should map the
# high-variance ones onto the best qubits.
X = np.column_stack(
    [
        rng.normal(0, 3.0, 200),   # feature 0: high variance
        rng.normal(0, 0.1, 200),   # feature 1: low variance
        rng.normal(0, 1.5, 200),   # feature 2: medium variance
        rng.normal(0, 0.5, 200),   # feature 3: low-medium variance
    ]
)
ds = Dataset(data=X, feature_names=["f0", "f1", "f2", "f3"])

# ── 1. Define a hardware noise profile ───────────────────────────────────────

print("=" * 55)
print("Noise profile — 5-qubit linear chain")
print("=" * 55)

# Simulates a small superconducting chip: qubit 2 is notably noisier.
profile = NoiseProfile(
    qubit_error_rates=[0.001, 0.002, 0.010, 0.001, 0.002],
    coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4)],
    t1=[150.0, 120.0,  60.0, 160.0, 140.0],   # µs
    t2=[ 80.0,  70.0,  30.0,  85.0,  75.0],   # µs
)

print(f"Qubits            : {profile.n_qubits}")
print("Qubit quality scores (lower = better):")
for q in range(profile.n_qubits):
    print(f"  qubit {q}: {profile.qubit_score(q):.5f}")
print()

# ── 2. Fit and transform ───────────────────────────────────────────────────────

print("=" * 55)
print("Noise-aware preprocessing  (encoding='angle_ry')")
print("=" * 55)

preprocessor = NoiseAwarePreprocessor(
    noise_profile=profile,
    encoding="angle_ry",
    angle_deadzone=0.05,
)

ds_out = preprocessor.fit_transform(ds)

print("Feature → qubit assignment (best qubits get high-variance features):")
for feat, qubit in preprocessor.feature_to_qubit_.items():
    print(f"  {feat:4s}  →  qubit {qubit}")
print()
print(f"Output data shape : {ds_out.data.shape}")
print(f"Value range       : [{ds_out.data.min():.4f}, {ds_out.data.max():.4f}]")
print()

# ── 3. Deadzone demo ──────────────────────────────────────────────────────────

print("=" * 55)
print("Angle deadzone effect  (5% on each side of [0, π])")
print("=" * 55)

# With deadzone=0.05 on a [0,π] encoder, output is clamped to [0.05π, 0.95π].
lo = 0.05 * np.pi
hi = 0.95 * np.pi
print(f"Expected angle range : [{lo:.4f}, {hi:.4f}]  rad")
actual_min = ds_out.data.min()
actual_max = ds_out.data.max()
print(f"Actual  angle range  : [{actual_min:.4f}, {actual_max:.4f}]  rad")
print()

# ── 4. IQP (±π) encoding ─────────────────────────────────────────────────────

print("=" * 55)
print("Same profile, IQP encoding  (range [-π, π])")
print("=" * 55)

preprocessor_iqp = NoiseAwarePreprocessor(
    noise_profile=profile,
    encoding="iqp",
    angle_deadzone=0.05,
)
ds_iqp = preprocessor_iqp.fit_transform(ds)

lo_iqp = -np.pi * (1.0 - 0.05 * 2)
hi_iqp =  np.pi * (1.0 - 0.05 * 2)
print(f"Expected angle range : [{lo_iqp:.4f}, {hi_iqp:.4f}]  rad")
print(f"Actual  angle range  : [{ds_iqp.data.min():.4f}, {ds_iqp.data.max():.4f}]  rad")
