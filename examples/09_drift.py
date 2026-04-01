"""
09 — Data Drift Detection
==========================
Demonstrates DriftDetector: fit on training data, check new data,
inspect DriftReport, use standalone (no pipeline), and with pipeline.

    uv run python examples/09_drift.py
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning
from quprep.core.dataset import Dataset

rng = np.random.default_rng(42)

# ── 1. No drift ───────────────────────────────────────────────────────────────

print("=" * 55)
print("1 — No drift (same distribution)")
print("=" * 55)

X_train = rng.normal(loc=0.0, scale=1.0, size=(200, 4))
X_same  = rng.normal(loc=0.0, scale=1.0, size=(50,  4))

det = qd.DriftDetector(warn=False)

pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), drift_detector=det)
pipeline.fit(X_train)
result = pipeline.transform(X_same)

print(result.drift_report)
print(f"  overall_drift       : {result.drift_report.overall_drift}")
print(f"  n_features_drifted  : {result.drift_report.n_features_drifted}")

# ── 2. Drift detected ─────────────────────────────────────────────────────────

print()
print("=" * 55)
print("2 — Drift detected (mean shifted by 10σ)")
print("=" * 55)

X_shifted = rng.normal(loc=10.0, scale=1.0, size=(50, 4))  # huge mean shift

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    det2 = qd.DriftDetector(warn=True)
    p2 = qd.Pipeline(encoder=qd.AngleEncoder(), drift_detector=det2)
    p2.fit(X_train)
    result2 = p2.transform(X_shifted)

print(result2.drift_report)
print(f"  overall_drift       : {result2.drift_report.overall_drift}")
print(f"  n_features_drifted  : {result2.drift_report.n_features_drifted}")
print(f"  drifted_features    : {result2.drift_report.drifted_features}")
if caught:
    print(f"  warning issued      : {caught[0].message}")

# ── 3. Per-feature stats ──────────────────────────────────────────────────────

print()
print("=" * 55)
print("3 — Per-feature stats")
print("=" * 55)

first = result2.drift_report.drifted_features[0]
stats = result2.drift_report.feature_stats[first]
print(f"  train_mean         : {stats['train_mean']:.4f}")
print(f"  new_mean           : {stats['new_mean']:.4f}")
print(f"  mean_shift_sigmas  : {stats['mean_shift_sigmas']:.2f}σ")
print(f"  std_ratio          : {stats['std_ratio']:.2f}")

# ── 4. Custom thresholds ──────────────────────────────────────────────────────

print()
print("=" * 55)
print("4 — Custom thresholds (tight: 1σ mean, 1.5× std)")
print("=" * 55)

X_slight = rng.normal(loc=1.5, scale=1.0, size=(50, 4))  # 1.5σ shift — subtle

det3 = qd.DriftDetector(mean_threshold=1.0, std_threshold=1.5, warn=False)
p3 = qd.Pipeline(encoder=qd.AngleEncoder(), drift_detector=det3)
p3.fit(X_train)
result3 = p3.transform(X_slight)

print(f"  drift detected (tight)  : {result3.drift_report.overall_drift}")
print(f"  drifted_features        : {result3.drift_report.drifted_features}")

# ── 5. Standalone (no Pipeline) ───────────────────────────────────────────────

print()
print("=" * 55)
print("5 — Standalone DriftDetector (no Pipeline)")
print("=" * 55)

ds_train = Dataset(
    data=X_train,
    feature_names=["age", "income", "score", "rate"],
)
ds_new = Dataset(
    data=X_shifted,
    feature_names=["age", "income", "score", "rate"],
)

det4 = qd.DriftDetector(warn=False)
det4.fit(ds_train)
report = det4.check(ds_new)

print(report)
print(f"  drifted features: {report.drifted_features}")

# ── 6. Suppressing warnings ───────────────────────────────────────────────────

print()
print("=" * 55)
print("6 — Suppress warning, check report programmatically")
print("=" * 55)

det5 = qd.DriftDetector(warn=True)
p5 = qd.Pipeline(encoder=qd.AngleEncoder(), drift_detector=det5)
p5.fit(X_train)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result5 = p5.transform(X_shifted)

if result5.drift_report.overall_drift:
    print(f"  Drift found in: {result5.drift_report.drifted_features}")
else:
    print("  No drift.")

# ── 7. Serialization preserves drift state ────────────────────────────────────

print()
print("=" * 55)
print("7 — Drift state preserved through save() / load()")
print("=" * 55)

p5.save("/tmp/quprep_drift_pipeline.pkl")
loaded = qd.Pipeline.load("/tmp/quprep_drift_pipeline.pkl")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result6 = loaded.transform(X_shifted)

print(f"  loaded overall_drift        : {result6.drift_report.overall_drift}")
print(f"  loaded n_features_drifted   : {result6.drift_report.n_features_drifted}")
print("  (same result as before serialization)")

print("\nDone.")
