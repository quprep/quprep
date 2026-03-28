# Validation, Schema & Cost Estimation

QuPrep v0.4.0 adds a full validation layer so problems are caught early — before encoding, before circuits are built, before any framework is involved.

---

## Quick example

```python
import quprep as qd

schema = qd.DataSchema([
    qd.FeatureSpec("age",    dtype="continuous", min_value=0,   max_value=120),
    qd.FeatureSpec("income", dtype="continuous", min_value=0),
    qd.FeatureSpec("flag",   dtype="binary"),
])

pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), schema=schema)
result = pipeline.fit_transform(df)

print(result.cost.nisq_safe)   # True / False
result.summary()               # audit table + cost breakdown
```

---

## Input validation

`validate_dataset()` runs automatically at pipeline entry and checks:

- Dataset is 2-D and non-empty
- Data dtype is float (not int, object, etc.)
- NaN coverage: warns if any column has missing values (`QuPrepWarning`)
- Qubit mismatch: warns if `n_features > n_qubits` for the chosen encoder

```python
import warnings
from quprep.validation import validate_dataset, QuPrepWarning

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    validate_dataset(dataset)

for warning in w:
    if issubclass(warning.category, QuPrepWarning):
        print(warning.message)
```

To suppress QuPrep warnings selectively:

```python
import warnings
from quprep.validation import QuPrepWarning
warnings.filterwarnings("ignore", category=QuPrepWarning)
```

---

## Schema enforcement

A `DataSchema` defines what the pipeline expects at entry. Violations are collected and reported together so you get the full picture in one error.

### Define a schema

```python
import quprep as qd

schema = qd.DataSchema([
    qd.FeatureSpec("age",     dtype="continuous", min_value=0, max_value=120),
    qd.FeatureSpec("income",  dtype="continuous", min_value=0),
    qd.FeatureSpec("is_employed", dtype="binary"),
    qd.FeatureSpec("score",   dtype="continuous", nullable=True),
])
```

`FeatureSpec` parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | str | required | Expected column name |
| `dtype` | str | required | `'continuous'`, `'discrete'`, or `'binary'` |
| `min_value` | float | None | Minimum allowed value (inclusive) |
| `max_value` | float | None | Maximum allowed value (inclusive) |
| `nullable` | bool | False | Whether NaN is permitted |

### Attach to a pipeline

```python
pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), schema=schema)
# raises SchemaViolationError at fit() / fit_transform() if violated
```

### Validate standalone

```python
try:
    schema.validate(dataset)
except qd.SchemaViolationError as e:
    print(e)
# DataSchema validation failed with 2 violation(s):
#   - Feature 'age': min value -5.0 < allowed minimum 0
#   - Feature 'is_employed': expected binary {0, 1}, got values [-1.0, 0.0, 1.0]
```

### Infer from data

```python
schema = qd.DataSchema.infer(train_dataset)
# auto-detects names, types, and min/max from the training set
```

### Save and load (JSON)

```python
# Save
with open("schema.json", "w") as f:
    f.write(schema.to_json())

# Load
with open("schema.json") as f:
    schema = qd.DataSchema.from_json(f.read())
```

The JSON format is a plain list of dicts — easy to edit by hand:

```json
[
  {"name": "age",    "dtype": "continuous", "min_value": 0, "max_value": 120},
  {"name": "income", "dtype": "continuous", "min_value": 0},
  {"name": "flag",   "dtype": "binary"}
]
```

---

## CLI: `quprep validate`

Inspect any CSV file without writing any Python:

```bash
# Structural report — shape, NaN counts, value ranges
quprep validate dataset.csv

# Infer schema and save to JSON
quprep validate dataset.csv --infer-schema schema.json

# Print inferred schema to stdout
quprep validate dataset.csv --infer-schema -

# Validate against a saved schema
quprep validate new_data.csv --schema schema.json
```

Typical output:

```
Dataset : dataset.csv
Shape   : 150 samples × 4 features
Columns : sepal_length, sepal_width, petal_length, petal_width
NaN     : none
Ranges  :
          'sepal_length': [4.3, 7.9]
          'sepal_width':  [2.0, 4.4]
          'petal_length': [1.0, 6.9]
          'petal_width':  [0.1, 2.5]

Schema  : checking ...
Schema  : OK — no violations
```

Exits 0 on success, 1 on violation (safe to use in CI).

---

## Cost estimation

Know your circuit complexity before encoding:

```python
cost = qd.estimate_cost(qd.AngleEncoder(), n_features=8)

print(cost.encoding)       # "angle"
print(cost.n_qubits)       # 8
print(cost.gate_count)     # 8
print(cost.circuit_depth)  # 1
print(cost.two_qubit_gates) # 0
print(cost.nisq_safe)      # True   (depth < 200 and CNOTs < 50)
print(cost.warning)        # None   (or a warning string if unsafe)
```

NISQ thresholds: `circuit_depth < 200` and `two_qubit_gates < 50`. Both must hold for `nisq_safe=True`.

### Cost on PipelineResult

`PipelineResult.cost` is populated automatically — computed after all reduction stages are applied so the qubit count reflects the actual dimensionality, not the raw input:

```python
result = pipeline.fit_transform(df)
print(result.cost.nisq_safe)
print(result.cost.gate_count)
```

---

## Preprocessing audit log

`PipelineResult.audit_log` records what happened to the data at each stage:

```python
result = qd.Pipeline(
    cleaner=qd.Imputer(),
    reducer=qd.PCAReducer(n_components=4),
    encoder=qd.AngleEncoder(),
).fit_transform(df)

for entry in result.audit_log:
    print(entry)
# {'stage': 'cleaner',    'n_samples_in': 150, 'n_features_in': 10, 'n_samples_out': 148, 'n_features_out': 10}
# {'stage': 'reducer',    'n_samples_in': 148, 'n_features_in': 10, 'n_samples_out': 148, 'n_features_out':  4}
# {'stage': 'normalizer', 'n_samples_in': 148, 'n_features_in':  4, 'n_samples_out': 148, 'n_features_out':  4}
```

`audit_log` is `None` when no preprocessing stages ran.

---

## Summary output

Both `Pipeline` and `PipelineResult` have a `.summary()` method useful in notebooks and scripts:

```python
pipeline.fit(df)
print(pipeline.summary())

result = pipeline.transform(df)
result.summary()
```

`str(pipeline)` also calls `.summary()`.

---

## sklearn compatibility

Every stage and the Pipeline itself now support the full `fit` / `transform` split:

```python
# Fit once on training data
pipeline.fit(X_train)

# Transform any number of test sets
X_val_result   = pipeline.transform(X_val)
X_test_result  = pipeline.transform(X_test)
```

`get_params()` and `set_params()` are also implemented for hyperparameter search:

```python
from sklearn.model_selection import GridSearchCV

pipeline.set_params(encoder=qd.BasisEncoder())
params = pipeline.get_params()
```
