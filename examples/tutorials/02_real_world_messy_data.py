"""
Real-World Messy Data
======================
Real datasets are never clean. They have missing values, outliers, and
skewed class distributions. Each of these problems affects quantum encoding
differently than it affects classical ML — and in worse ways, because quantum
circuits have much tighter constraints on input ranges and distributions.

This tutorial walks through a deliberately messy dataset and shows how to
handle each problem with QuPrep, with explanations of *why* each step matters
specifically for quantum encoding.

    uv run python examples/tutorials/02_real_world_messy_data.py

No optional dependencies required.
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning

rng = np.random.default_rng(42)
print(f"quprep {qd.__version__}\n")


# ── 1. The messy dataset ──────────────────────────────────────────────────────
#
# This dataset has three realistic problems:
#   - Missing values (NaN) in two columns
#   - An extreme outlier in column 'a' (value 150.0 in a [1–4] range)
#   - A heavily imbalanced target: 90% class 0, 10% class 1
#
# All three problems would silently corrupt a quantum encoding if left untreated.

n = 40
X = np.column_stack([
    rng.uniform(1.0, 4.0, n),
    rng.uniform(0.0, 1.0, n),
    rng.uniform(0.5, 2.5, n),
    rng.uniform(0.2, 0.8, n),
])

X[3, 0]  = np.nan    # missing value
X[7, 2]  = np.nan    # missing value
X[12, 0] = 150.0     # extreme outlier
y = np.array([0] * 36 + [1] * 4)  # 90/10 class imbalance

ds = qd.NumpyIngester().load(X, y=y)

print("── 1. Raw data ──────────────────────────────────────────────────────────")
print(f"   Shape      : {ds.data.shape}")
print(f"   NaN count  : {np.isnan(ds.data).sum()}")
print(f"   Max value  : {np.nanmax(ds.data):.1f}  (outlier in feature 0)")
print(f"   Class dist : {np.bincount(y)}  (90/10 imbalance)")
print()


# ── 2. Step 1 — Fix missing values ───────────────────────────────────────────
#
# NaN in a feature array propagates through all arithmetic. Encoders call
# numpy operations (arcsin, angle scaling) that produce NaN angles — circuits
# that are mathematically undefined. The Imputer replaces each NaN with the
# column mean, computed from the non-missing rows.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    ds_clean = qd.Imputer(strategy="mean").fit_transform(ds)

print("── 2. After Imputer ─────────────────────────────────────────────────────")
print(f"   NaN count  : {np.isnan(ds_clean.data).sum()}  (was 2)")
print(f"   Max value  : {ds_clean.data.max():.1f}  (outlier still present)")
print()


# ── 3. Step 2 — Remove outliers ───────────────────────────────────────────────
#
# MinMax scaling maps [min, max] → [0, π]. One outlier at 150 makes the
# effective range [1, 150], so all 39 normal samples get compressed into
# the first 2% of the rotation range — the quantum model sees almost no
# variation between samples.
#
# OutlierHandler(method="iqr", action="clip") computes the 1.5×IQR fence
# and clips values to the fence boundary, preserving all rows.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    ds_no_outliers = qd.OutlierHandler(method="iqr", action="clip").fit_transform(ds_clean)

print("── 3. After OutlierHandler ──────────────────────────────────────────────")
print(f"   Max value  : {ds_no_outliers.data.max():.2f}  (was 150.0, now clipped)")
print(f"   Rows kept  : {ds_no_outliers.data.shape[0]}  (clip preserves all rows)")
print()


# ── 4. Step 3 — Balance classes ───────────────────────────────────────────────
#
# A 90/10 split means a VQC trains on 36 majority-class examples before
# seeing a minority sample. This leads to degenerate gradients and circuits
# that collapse to predicting the majority class.
#
# SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic
# minority samples by interpolating between real minority examples in feature
# space. It increases the minority class to match the majority without
# simply duplicating rows.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    ds_balanced = qd.ImbalanceHandler(strategy="smote").fit_transform(ds_no_outliers)

print("── 4. After ImbalanceHandler(smote) ─────────────────────────────────────")
labels = ds_balanced.labels.astype(int)
print(f"   Class dist : {np.bincount(labels)}  (was [36, 4])")
print(f"   Total rows : {ds_balanced.data.shape[0]}  (was 40)")
print()


# ── 5. Step 4 — Scale and encode ─────────────────────────────────────────────
#
# With the data clean and balanced, the final step is scaling to [0, π] and
# encoding. The Pipeline here handles just normalisation and encoding — the
# cleaning steps above ran standalone.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = qd.Pipeline(
        normalizer=qd.Scaler(strategy="minmax_pi"),
        encoder=qd.AngleEncoder(),
    ).fit_transform(ds_balanced)

print("── 5. Encoded ───────────────────────────────────────────────────────────")
print(f"   Scaled range : [{result.dataset.data.min():.3f}, {result.dataset.data.max():.3f}]")
print(f"   Circuits     : {len(result.encoded)}")
print()


# ── 6. Compatibility check ────────────────────────────────────────────────────

report = qd.check_compatibility(qd.AngleEncoder(), result.dataset)

print("── 6. Compatibility check ───────────────────────────────────────────────")
print(f"   Compatible : {report.is_compatible}")
print(f"   Warnings   : {len(report.warnings)}")
print(f"   Errors     : {len(report.errors)}")
print()
print(qd.draw_ascii(result.encoded[0]))


# ── Next steps ────────────────────────────────────────────────────────────────
print("── Next steps ───────────────────────────────────────────────────────────")
print("   → tutorials/03         : auto pipeline + Qiskit export end-to-end")
print("   → how-to/fix_class_imbalance  : oversample, undersample, ADASYN")
print("   → how-to/validate_before_encoding : preprocessing_report, DataSchema")
