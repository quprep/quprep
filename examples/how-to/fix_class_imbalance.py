"""
How to Fix Class Imbalance
===========================
Imbalanced class distributions cause variational quantum classifiers to
collapse toward predicting the majority class. QuPrep's ImbalanceHandler
provides four resampling strategies: oversample, undersample, SMOTE, and
ADASYN (requires pip install quprep[imbalanced]).

    uv run python examples/how-to/fix_class_imbalance.py
"""

import warnings
from collections import Counter

import numpy as np

import quprep as qd
from quprep import QuPrepWarning

rng = np.random.default_rng(42)

# 80% class 0, 20% class 1
X = rng.uniform(0, 1, (100, 4))
y = np.array([0] * 80 + [1] * 20)
ds = qd.NumpyIngester().load(X, y=y)

print(f"quprep {qd.__version__} | original: {Counter(y.tolist())}\n")


# ── 1. Random oversampling ────────────────────────────────────────────────────
#
# Duplicates minority samples at random until classes are balanced.
# Simple and fast; may increase overfitting on small datasets.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    ds_over = qd.ImbalanceHandler(strategy="oversample").fit_transform(ds)

print("── 1. oversample ────────────────────────────────────────────────────────")
print(f"   {Counter(ds_over.labels.astype(int).tolist())}  (rows: {ds_over.data.shape[0]})")
print()


# ── 2. Random undersampling ───────────────────────────────────────────────────
#
# Removes majority samples at random until classes are balanced.
# Reduces dataset size — only suitable when you have enough majority samples.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    ds_under = qd.ImbalanceHandler(strategy="undersample").fit_transform(ds)

print("── 2. undersample ───────────────────────────────────────────────────────")
print(f"   {Counter(ds_under.labels.astype(int).tolist())}  (rows: {ds_under.data.shape[0]})")
print()


# ── 3. SMOTE ─────────────────────────────────────────────────────────────────
#
# Generates synthetic minority samples by interpolating between real minority
# examples in feature space. Better than simple duplication because it adds
# variation. Requires k_neighbors minority samples (default k=5).

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    ds_smote = qd.ImbalanceHandler(strategy="smote", k_neighbors=5).fit_transform(ds)

print("── 3. smote ─────────────────────────────────────────────────────────────")
print(f"   {Counter(ds_smote.labels.astype(int).tolist())}  (rows: {ds_smote.data.shape[0]})")
print()


# ── 4. ADASYN (optional) ──────────────────────────────────────────────────────
#
# Adaptive density-based resampling. Generates more synthetic samples in
# regions where the minority class is harder to learn.
# Requires: pip install quprep[imbalanced]

print("── 4. adasyn  (pip install quprep[imbalanced]) ──────────────────────────")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuPrepWarning)
        ds_adasyn = qd.ImbalanceHandler(strategy="adasyn").fit_transform(ds)
    print(f"   {Counter(ds_adasyn.labels.astype(int).tolist())}  (rows: {ds_adasyn.data.shape[0]})")
except ImportError:
    print("   skipped — run: pip install quprep[imbalanced]")
print()


# ── 5. Multiclass imbalance ───────────────────────────────────────────────────

print("── 5. Multiclass (3 classes, imbalanced) ────────────────────────────────")
X3 = rng.uniform(0, 1, (90, 4))
y3 = np.array([0] * 60 + [1] * 20 + [2] * 10)
ds3 = qd.NumpyIngester().load(X3, y=y3)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    ds3_bal = qd.ImbalanceHandler(strategy="oversample").fit_transform(ds3)

print(f"   Before : {Counter(y3.tolist())}")
print(f"   After  : {Counter(ds3_bal.labels.astype(int).tolist())}")
