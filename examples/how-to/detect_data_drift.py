"""
How to Detect Data Drift
=========================
DriftDetector monitors whether incoming data has shifted significantly from
the distribution your pipeline was trained on. Critical for production quantum
ML systems where circuit parameters are fixed after training.

    uv run python examples/how-to/detect_data_drift.py
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning

rng = np.random.default_rng(42)
print(f"quprep {qd.__version__}\n")

# Reference distribution: training data
X_train = rng.normal(loc=0.5, scale=0.1, size=(100, 4))
ds_train = qd.NumpyIngester().load(X_train)


# ── 1. Fit the detector on training data ─────────────────────────────────────

detector = qd.DriftDetector()
detector.fit(ds_train)

print("── 1. Fit on training data ───────────────────────────────────────────────")
print(f"   Reference mean  : {X_train.mean(axis=0).round(3)}")
print(f"   Reference std   : {X_train.std(axis=0).round(3)}")
print()


# ── 2. No drift (same distribution) ──────────────────────────────────────────

X_same = rng.normal(loc=0.5, scale=0.1, size=(50, 4))
ds_same = qd.NumpyIngester().load(X_same)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    report_no_drift = detector.check(ds_same)

print("── 2. No drift ──────────────────────────────────────────────────────────")
print(f"   Drift detected      : {report_no_drift.overall_drift}")
print(f"   Features drifted    : {report_no_drift.n_features_drifted}")
print()


# ── 3. Clear drift (shifted distribution) ────────────────────────────────────

X_drift = rng.normal(loc=2.0, scale=0.3, size=(50, 4))  # mean shifted from 0.5 → 2.0
ds_drift = qd.NumpyIngester().load(X_drift)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    report_drift = detector.check(ds_drift)

print("── 3. Clear drift ───────────────────────────────────────────────────────")
print(f"   Drift detected      : {report_drift.overall_drift}")
print(f"   Features drifted    : {report_drift.n_features_drifted}")
if report_drift.drifted_features:
    print(f"   Drifted features    : {report_drift.drifted_features}")
print()


# ── 4. Drift detector inside a Pipeline ──────────────────────────────────────

print("── 4. Drift detector in Pipeline ────────────────────────────────────────")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    pipeline = qd.Pipeline(
        normalizer=qd.Scaler(strategy="minmax_pi"),
        encoder=qd.AngleEncoder(),
        drift_detector=qd.DriftDetector(),
    )
    result_train = pipeline.fit_transform(ds_train)
    result_new   = pipeline.transform(ds_drift)

train_drift = result_train.drift_report.overall_drift if result_train.drift_report else "n/a (fit)"
print(f"   Training drift  : {train_drift}")
print(f"   New data drift  : {result_new.drift_report.overall_drift}")
print(f"   Features drifted: {result_new.drift_report.n_features_drifted}")
