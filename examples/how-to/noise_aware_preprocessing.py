"""
How to Use Noise-Aware Preprocessing
======================================
On NISQ hardware, qubits have different error rates and connectivity
constraints. NoiseAwarePreprocessor assigns high-variance features to
the most reliable qubits and remaps angles away from gate-error hotspots.

    uv run python examples/how-to/noise_aware_preprocessing.py
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning
from quprep.core.dataset import Dataset
from quprep.preprocess.noise_aware import NoiseAwarePreprocessor, NoiseProfile

rng = np.random.default_rng(0)
print(f"quprep {qd.__version__}\n")

# Dataset with features of very different variances
X = np.column_stack([
    rng.normal(0.5, 2.0, 40),   # feature 0: high variance
    rng.normal(0.5, 0.1, 40),   # feature 1: low variance
    rng.normal(0.5, 1.5, 40),   # feature 2: medium variance
    rng.normal(0.5, 0.05, 40),  # feature 3: very low variance
])
ds = Dataset(data=X)


# ── 1. Define a hardware noise profile ────────────────────────────────────────
#
# NoiseProfile describes the hardware: qubit error rates, connectivity (which
# qubits are coupled), and the readout error for each qubit.
# The preprocessor uses this to decide which feature gets which qubit.

noise = NoiseProfile(
    qubit_error_rates=[0.001, 0.003, 0.002, 0.008],  # qubit 3 is noisiest
    coupling_map=[(0, 1), (1, 2), (2, 3)],            # linear chain
)

print("── 1. Noise profile ─────────────────────────────────────────────────────")
print(f"   Qubit error rates : {noise.qubit_error_rates}")
print(f"   Connectivity      : {noise.coupling_map}")
print()


# ── 2. Fit the preprocessor ───────────────────────────────────────────────────
#
# The preprocessor learns three things from the dataset + noise profile:
#   qubit_assignment_  : which feature maps to which qubit
#   angle_deadzone     : applies to angle encoders — avoids 0/π pole regions
#                        where gate errors are highest

nap = NoiseAwarePreprocessor(noise_profile=noise)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    ds_out = nap.fit_transform(ds)

print("── 2. Qubit assignment ───────────────────────────────────────────────────")
print(f"   Feature variances  : {np.var(X, axis=0).round(3)}")
print(f"   Qubit error rates  : {noise.qubit_error_rates}")
print(f"   Qubit assignment   : {nap.qubit_assignment_}")
print("   (high-variance features → low-error qubits)")
print()


# ── 3. SWAP overhead estimation ───────────────────────────────────────────────
#
# swap_overhead_ is the number of SWAP gates needed to implement the qubit
# assignment on the given connectivity graph. Fewer SWAPs = fewer errors.

print("── 3. SWAP overhead ──────────────────────────────────────────────────────")
print(f"   SWAP gates before : {nap.estimated_swaps_before_}")
print(f"   SWAP gates after  : {nap.estimated_swaps_after_}")
print()


# ── 4. Angle dead-zone ────────────────────────────────────────────────────────
#
# Angles near 0 or π (for Ry) or near ±π (for ZZ) have the highest gate
# error sensitivity. angle_deadzone_ remaps values away from these poles.

print("── 4. Angle dead-zone ────────────────────────────────────────────────────")
print(f"   Dead-zone width : {nap.angle_deadzone}")
print(f"   Output range    : [{ds_out.data.min():.3f}, {ds_out.data.max():.3f}]")
print()


# ── 5. Encode the noise-aware dataset ────────────────────────────────────────

print("── 5. Encode ─────────────────────────────────────────────────────────────")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = qd.Pipeline(
        normalizer=qd.Scaler(strategy="minmax_pi"),
        encoder=qd.AngleEncoder(),
    ).fit_transform(ds_out)

print(f"   Circuits  : {len(result.encoded)}")
print(f"   Qubits    : {result.encoded[0].metadata['n_qubits']}")
print(qd.draw_ascii(result.encoded[0]))
