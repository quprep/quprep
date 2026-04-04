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

### Saving and loading a fitted pipeline

```python
import quprep as qd

pipeline = qd.Pipeline(
    reducer=qd.PCAReducer(n_components=4),
    encoder=qd.AngleEncoder(),
)
pipeline.fit(X_train)
pipeline.save("pipeline.pkl")

# Later — in a different process or deployment
loaded = qd.Pipeline.load("pipeline.pkl")
result = loaded.transform(X_new)
```

The parent directory is created automatically. All fitted state (reducer, normalizer, encoder) is preserved.

### With drift detection

```python
import quprep as qd

det = qd.DriftDetector(mean_threshold=3.0, std_threshold=2.0)

pipeline = qd.Pipeline(
    encoder=qd.AngleEncoder(),
    drift_detector=det,
)
pipeline.fit(X_train)
result = pipeline.transform(X_test)

print(result.drift_report.overall_drift)      # True / False
print(result.drift_report.drifted_features)   # list of feature names
```

Drift is checked automatically on every `transform()` call. A `QuPrepWarning` is issued when drift is detected. The drift detector state is preserved through `save()`/`load()`.

### Time series pipeline (v0.7.0)

```python
import quprep as qd

pipeline = qd.Pipeline(
    ingester=qd.TimeSeriesIngester(time_column="date"),
    preprocessor=qd.WindowTransformer(window_size=8, step=1),
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform("sensor_data.csv")

print(len(result.encoded))                        # n_windows
print(result.encoded[0].metadata["n_qubits"])     # window_size × n_features
```

The `preprocessor` stage runs after ingestion and before cleaning/reduction. It is designed for shape-changing transforms like `WindowTransformer`.

### Sparse data (v0.7.0)

```python
import scipy.sparse as sp
import quprep as qd

sparse_matrix = sp.csr_matrix(X)
result = qd.Pipeline(encoder=qd.AngleEncoder()).fit_transform(sparse_matrix)
```

scipy.sparse matrices are accepted anywhere a NumPy array is expected. They are converted to dense at ingestion.

### Labels and multi-label (v0.7.0)

```python
import quprep as qd

# Attach labels at fit_transform time
result = qd.Pipeline(encoder=qd.AngleEncoder()).fit_transform(X, y=y)
print(result.dataset.labels)   # preserved through all stages

# Or embed labels in the Dataset via CSVIngester
from quprep.ingest.csv_ingester import CSVIngester

pipeline = qd.Pipeline(
    ingester=CSVIngester(target_columns="label"),
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform("data.csv")
print(result.dataset.labels.shape)   # (n_samples,)
```

For `FeatureSelector(method="mutual_info")`, labels in `dataset.labels` are used automatically — no separate `labels=` argument needed.
