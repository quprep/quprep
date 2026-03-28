"""
08 — Validation, Schema & Cost Estimation
==========================================
Shows the v0.4.0 validation layer:
  - import quprep as qd (short alias)
  - DataSchema: define, validate, infer, save, load
  - Cost estimation
  - PipelineResult.cost and .audit_log
  - Pipeline.summary() and PipelineResult.summary()
  - quprep validate CLI equivalent (programmatic)

    uv run python examples/08_validation.py
"""

import json
import warnings

import numpy as np

import quprep as qd
from quprep.core.dataset import Dataset
from quprep.validation import QuPrepWarning, validate_dataset

# ── shared data ────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
X = rng.uniform(0.0, 1.0, size=(20, 6)).astype(np.float64)

print("=" * 60)
print("  08 — Validation, Schema & Cost Estimation")
print("=" * 60)

# ── 1. import quprep as qd ────────────────────────────────────────────────────

print("\n[1] import quprep as qd")
print(f"  version       : {qd.__version__}")
print(f"  qd.AngleEncoder : {qd.AngleEncoder}")
print(f"  qd.Imputer      : {qd.Imputer}")
print(f"  qd.DataSchema   : {qd.DataSchema}")

# ── 2. Basic validation warning ───────────────────────────────────────────────

print("\n[2] validate_dataset — NaN warning")
X_with_nan = X.copy()
X_with_nan[0, 0] = np.nan
X_with_nan[2, 0] = np.nan

ds_nan = Dataset(data=X_with_nan, feature_names=[f"f{i}" for i in range(6)])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    validate_dataset(ds_nan, context="example")

for w in caught:
    if issubclass(w.category, QuPrepWarning):
        print(f"  Warning: {w.message}")

# ── 3. Schema: define and validate ───────────────────────────────────────────

print("\n[3] DataSchema — define and validate")
schema = qd.DataSchema([
    qd.FeatureSpec(f"f{i}", dtype="continuous", min_value=0.0, max_value=1.0)
    for i in range(6)
])

ds_clean = Dataset(data=X, feature_names=[f"f{i}" for i in range(6)])
schema.validate(ds_clean)
print("  Clean dataset   : OK — no violations")

# Introduce a violation
X_bad = X.copy()
X_bad[0, 2] = 1.5  # exceeds max_value=1.0
ds_bad = Dataset(data=X_bad, feature_names=[f"f{i}" for i in range(6)])

try:
    schema.validate(ds_bad)
except qd.SchemaViolationError as e:
    print(f"  Violated dataset: {str(e).splitlines()[0]}")

# ── 4. Schema: infer from data ────────────────────────────────────────────────

print("\n[4] DataSchema.infer()")
inferred = qd.DataSchema.infer(ds_clean)
print(f"  Features inferred: {len(inferred.features)}")
print(f"  f0 range: [{inferred.features[0].min_value:.3f}, {inferred.features[0].max_value:.3f}]")
inferred.validate(ds_clean)
print("  Inferred schema validates source: OK")

# ── 5. Schema: save and load (JSON) ───────────────────────────────────────────

print("\n[5] DataSchema serialisation")
schema_path = "/tmp/quprep_schema.json"
with open(schema_path, "w") as f:
    f.write(inferred.to_json())
print(f"  Saved to {schema_path}")

with open(schema_path) as f:
    restored = qd.DataSchema.from_json(f.read())
restored.validate(ds_clean)
print("  Loaded + validated: OK")
print(f"  First entry: {json.loads(inferred.to_json())[0]}")

# ── 6. Cost estimation ────────────────────────────────────────────────────────

print("\n[6] Cost estimation")
for encoder_cls, label in [
    (qd.AngleEncoder,         "AngleEncoder      "),
    (qd.AmplitudeEncoder,     "AmplitudeEncoder  "),
    (qd.IQPEncoder,           "IQPEncoder        "),
    (qd.EntangledAngleEncoder,"EntangledAngle    "),
]:
    cost = qd.estimate_cost(encoder_cls(), n_features=6)
    print(
        f"  {label}: {cost.n_qubits} qubits | "
        f"depth {cost.circuit_depth:3d} | "
        f"gates {cost.gate_count:3d} | "
        f"2q {cost.two_qubit_gates:3d} | "
        f"NISQ {'✓' if cost.nisq_safe else '✗'}"
    )

# ── 7. Pipeline with schema — result.cost and result.audit_log ───────────────

print("\n[7] Pipeline with schema → result.cost + result.audit_log")
pipeline = qd.Pipeline(
    cleaner=qd.Imputer(),
    reducer=qd.PCAReducer(n_components=3),
    encoder=qd.AngleEncoder(),
    schema=inferred,
)
result = pipeline.fit_transform(ds_clean)

print(f"  result.cost.encoding  : {result.cost.encoding}")
print(f"  result.cost.n_qubits  : {result.cost.n_qubits}")
print(f"  result.cost.nisq_safe : {result.cost.nisq_safe}")

print(f"\n  audit_log ({len(result.audit_log)} stages):")
for entry in result.audit_log:
    print(
        f"    {entry['stage']:<12}"
        f"  {entry['n_samples_in']:>4} × {entry['n_features_in']:<3}"
        f"  →  {entry['n_samples_out']:>4} × {entry['n_features_out']}"
    )

# ── 8. Pipeline.summary() ─────────────────────────────────────────────────────

print("\n[8] Pipeline.summary()")
print(pipeline.summary())

# ── 9. PipelineResult.summary() ───────────────────────────────────────────────

print("\n[9] PipelineResult.summary()")
result.summary()
print(result.summary())

# ── 10. fit / transform split ─────────────────────────────────────────────────

print("\n[10] fit() + transform() (sklearn-style)")
p2 = qd.Pipeline(encoder=qd.AngleEncoder())
p2.fit(X)
r_train = p2.transform(X)
r_test  = p2.transform(X[:5])
print(f"  train result: {r_train}")
print(f"  test  result: {r_test}")

print("\nDone.")
