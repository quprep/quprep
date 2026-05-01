"""
16 — Class Imbalance Handling
================================
Balance skewed class distributions before quantum encoding using
oversampling, undersampling, and SMOTE.

    uv run python examples/16_imbalance.py
"""

from collections import Counter

import numpy as np

import quprep as qd
from quprep.core.dataset import Dataset

# ── Shared data ───────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
# 90/10 split — severely imbalanced
X = rng.uniform(0, 1, (110, 4))
y = np.array([0] * 100 + [1] * 10)
ds = Dataset(data=X, feature_names=["a", "b", "c", "d"], labels=y)

print(f"Original class distribution: {dict(Counter(y))}")
print()

# ── 1. Random oversampling ────────────────────────────────────────────────────

print("=" * 50)
print("Strategy: oversample (duplicate minority samples)")
print("=" * 50)

ds_over = qd.ImbalanceHandler(strategy="oversample").fit_transform(ds)
print(f"Before: {dict(Counter(y))}")
print(f"After : {dict(Counter(ds_over.labels))}")
print(f"Shape : {ds_over.data.shape}")
print()

# ── 2. Random undersampling ───────────────────────────────────────────────────

print("=" * 50)
print("Strategy: undersample (remove majority samples)")
print("=" * 50)

ds_under = qd.ImbalanceHandler(strategy="undersample").fit_transform(ds)
print(f"Before: {dict(Counter(y))}")
print(f"After : {dict(Counter(ds_under.labels))}")
print(f"Shape : {ds_under.data.shape}")
print()

# ── 3. SMOTE (synthetic samples) ──────────────────────────────────────────────

print("=" * 50)
print("Strategy: SMOTE (synthetic interpolated samples)")
print("=" * 50)

ds_smote = qd.ImbalanceHandler(strategy="smote", k_neighbors=5).fit_transform(ds)
print(f"Before: {dict(Counter(y))}")
print(f"After : {dict(Counter(ds_smote.labels))}")
print(f"Shape : {ds_smote.data.shape}")

# Verify synthetic samples stay within the feature range
print(f"Feature range (original) : [{X.min():.4f}, {X.max():.4f}]")
print(f"Feature range (SMOTE out): [{ds_smote.data.min():.4f}, {ds_smote.data.max():.4f}]")
print()

# ── 4. Multi-class imbalance ──────────────────────────────────────────────────

print("=" * 50)
print("Multi-class oversampling  (3 classes)")
print("=" * 50)

X3 = rng.uniform(0, 1, (75, 4))
y3 = np.array([0] * 50 + [1] * 20 + [2] * 5)
ds3 = Dataset(data=X3, labels=y3)

ds3_bal = qd.ImbalanceHandler(strategy="oversample").fit_transform(ds3)
print(f"Before: {dict(Counter(y3))}")
print(f"After : {dict(Counter(ds3_bal.labels))}")
print()

# ── 5. Integrate into a pipeline ─────────────────────────────────────────────

print("=" * 50)
print("Encode balanced dataset with AngleEncoder")
print("=" * 50)

pipeline = qd.Pipeline(
    encoder=qd.AngleEncoder(),
    exporter=qd.QASMExporter(),
)
result = pipeline.fit_transform(ds_smote)
print(f"Circuits produced : {len(result.circuits)}")
print(f"Qubits per circuit: {result.encoded[0].metadata['n_qubits']}")
