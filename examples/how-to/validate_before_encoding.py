"""
How to Validate Before Encoding
=================================
Three complementary tools for validating your data before encoding:
  - preprocessing_report()  : structured dataset audit (NaN, outliers, budget, imbalance)
  - check_compatibility()   : encoder-specific range and type checks
  - DataSchema              : define, validate, infer, and save schemas

    uv run python examples/how-to/validate_before_encoding.py
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning

rng = np.random.default_rng(0)
print(f"quprep {qd.__version__}\n")


# ── 1. preprocessing_report() ─────────────────────────────────────────────────
#
# Runs a structured audit of your dataset before any preprocessing step.
# Checks: NaN, outliers (IQR), qubit budget overrun, class imbalance (>3:1),
# and encoder compatibility. Returns a PreprocessingReport with n_issues and
# a list of human-readable recommendations.

X = rng.uniform(0, 10, (60, 6))   # values outside [0, π] — angle encoder will flag this
X[0, 2] = np.nan
y = np.array([0] * 54 + [1] * 6)  # 9:1 imbalance

ds = qd.NumpyIngester().load(X, y=y)

print("── 1. preprocessing_report() ────────────────────────────────────────────")
report = qd.preprocessing_report(ds, encoder=qd.AngleEncoder(), qubit_budget=4)
print(f"   Issues : {report.n_issues}")
for rec in report.recommendations:
    print(f"   ↳ {rec}")
print()


# ── 2. check_compatibility() ──────────────────────────────────────────────────
#
# Checks that preprocessed data is within the valid range for the chosen
# encoder. Run this after preprocessing, before handing circuits to a
# simulator or hardware. Returns a CompatibilityReport.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    ds_clean = qd.Pipeline(
        cleaner=qd.Imputer(strategy="mean"),
        normalizer=qd.Scaler(strategy="minmax_pi"),
    ).fit_transform(ds).dataset

compat = qd.check_compatibility(qd.AngleEncoder(), ds_clean)

print("── 2. check_compatibility() ─────────────────────────────────────────────")
print(f"   Compatible : {compat.is_compatible}")
print(f"   Warnings   : {len(compat.warnings)}")
print(f"   Errors     : {len(compat.errors)}")
print()


# ── 3. verify_encoding() ──────────────────────────────────────────────────────
#
# Post-encoding invariant check. For angle encoders: verifies all angles are
# in the valid range. For amplitude encoders: checks unit norm (|‖ψ‖−1| < 1e-6).

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = qd.Pipeline(
        cleaner=qd.Imputer(strategy="mean"),
        normalizer=qd.Scaler(strategy="minmax_pi"),
        encoder=qd.AngleEncoder(),
    ).fit_transform(ds)

verify = qd.verify_encoding(result.encoded, qd.AngleEncoder())

print("── 3. verify_encoding() ─────────────────────────────────────────────────")
print(f"   Passed : {verify.passed}")
for check in verify.checks:
    status = "✓" if check["passed"] else "✗"
    print(f"   {status} {check['name']}: {check['detail']}")
print()


# ── 4. DataSchema ─────────────────────────────────────────────────────────────
#
# Define a schema to enforce data types, value ranges, and required columns.
# Infer it from existing data, or define it manually. Save and reload as JSON.

print("── 4. DataSchema — infer ────────────────────────────────────────────────")
schema = qd.DataSchema.infer(ds_clean)
print(f"   Columns     : {len(schema.features)}")
print(f"   Schema JSON (first feature): {schema.to_dict()[0]}")
print()

schema_path = "/tmp/quprep_schema.json"
with open(schema_path, "w") as f:
    f.write(schema.to_json())

with open(schema_path) as f:
    schema2 = qd.DataSchema.from_json(f.read())

print("── 4. DataSchema — validate ─────────────────────────────────────────────")
try:
    schema2.validate(ds_clean)
    print("   Valid  : True")
    print("   Errors : 0")
except Exception as e:
    print(f"   Invalid: {e}")
