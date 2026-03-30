# Validation & Cost

---

## DataSchema

::: quprep.validation.schema.DataSchema
    options:
      show_source: false

---

## FeatureSpec

::: quprep.validation.schema.FeatureSpec
    options:
      show_source: false

---

## SchemaViolationError

::: quprep.validation.schema.SchemaViolationError
    options:
      show_source: false

---

## validate_dataset

::: quprep.validation.validate_dataset
    options:
      show_source: false

---

## warn_qubit_mismatch

::: quprep.validation.warn_qubit_mismatch
    options:
      show_source: false

---

## CostEstimate

::: quprep.validation.cost.CostEstimate
    options:
      show_source: false

---

## estimate_cost

::: quprep.validation.cost.estimate_cost
    options:
      show_source: false

---

## Examples

### Define and validate a schema

```python
import quprep as qd

schema = qd.DataSchema([
    qd.FeatureSpec("age",    dtype="continuous", min_value=0, max_value=120),
    qd.FeatureSpec("income", dtype="continuous", min_value=0),
    qd.FeatureSpec("label",  dtype="discrete"),
])

try:
    schema.validate(dataset)
except qd.SchemaViolationError as e:
    print(e)
```

### Infer schema from data and save

```python
import quprep as qd

schema = qd.DataSchema.infer(dataset)
schema.to_json()                          # JSON string
schema.to_json()                          # save to file
restored = qd.DataSchema.from_json(s)    # reload
```

### Cost estimation

```python
import quprep as qd

cost = qd.estimate_cost(qd.IQPEncoder(), n_features=8)
print(cost.n_qubits)       # 8
print(cost.circuit_depth)  # depends on reps
print(cost.nisq_safe)      # True / False
print(cost.warning)        # str | None
```
