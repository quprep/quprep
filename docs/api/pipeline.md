# Pipeline

The `Pipeline` class chains all preprocessing stages. Each stage is optional — use only the stages you need.

---

## Pipeline

::: quprep.core.pipeline.Pipeline
    options:
      show_source: true

---

## PipelineResult

::: quprep.core.pipeline.PipelineResult
    options:
      show_source: false

---

## Examples

### Minimal — encode only

```python
import quprep as qd

pipeline = qd.Pipeline(encoder=qd.AngleEncoder())
result = pipeline.fit_transform(data)

result.encoded       # list[EncodedResult]
result.encoded[0].parameters   # rotation angles for first sample
result.encoded[0].metadata     # {"n_qubits": 4, "depth": 1, ...}
```

### Full — clean + encode + export

```python
import quprep as qd

pipeline = qd.Pipeline(
    cleaner=qd.Imputer(strategy="knn"),
    encoder=qd.AngleEncoder(rotation="ry"),
    exporter=qd.QASMExporter(),
)
result = pipeline.fit_transform("data.csv")
result.circuits[0]   # QASM string for first sample
```

### With schema validation

```python
import quprep as qd

schema = qd.DataSchema([
    qd.FeatureSpec("age",    dtype="continuous", min_value=0, max_value=120),
    qd.FeatureSpec("income", dtype="continuous", min_value=0),
])
pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), schema=schema)
result = pipeline.fit_transform("data.csv")

print(result.cost.nisq_safe)   # True / False
result.summary()               # audit table + cost breakdown
```

### sklearn-style fit / transform split

```python
import quprep as qd

pipeline = qd.Pipeline(
    reducer=qd.PCAReducer(n_components=4),
    encoder=qd.AngleEncoder(),
)
pipeline.fit(X_train)
r_train = pipeline.transform(X_train)
r_test  = pipeline.transform(X_test)
```

### Explicit normalizer

```python
import quprep as qd

pipeline = qd.Pipeline(
    encoder=qd.AngleEncoder(),
    normalizer=qd.Scaler("zscore"),  # override auto-selection
)
```
