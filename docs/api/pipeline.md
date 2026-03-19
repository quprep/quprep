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
from quprep import Pipeline
from quprep.encode.angle import AngleEncoder

pipeline = Pipeline(encoder=AngleEncoder())
result = pipeline.fit_transform(data)

result.encoded       # list[EncodedResult]
result.encoded[0].parameters   # rotation angles for first sample
result.encoded[0].metadata     # {"n_qubits": 4, "depth": 1, ...}
```

### Full — clean + encode + export

```python
from quprep import Pipeline
from quprep.clean.imputer import Imputer
from quprep.clean.outlier import OutlierHandler
from quprep.encode.angle import AngleEncoder
from quprep.export.qasm_export import QASMExporter

pipeline = Pipeline(
    cleaner=Imputer(strategy="knn"),
    encoder=AngleEncoder(rotation="ry"),
    exporter=QASMExporter(),
)
result = pipeline.fit_transform("data.csv")
result.circuits[0]   # QASM string for first sample
```

### Explicit normalizer

```python
from quprep.normalize.scalers import Scaler

pipeline = Pipeline(
    encoder=AngleEncoder(),
    normalizer=Scaler("zscore"),  # override auto-selection
)
```

### Explicit ingester

```python
from quprep.ingest.csv_ingester import CSVIngester

pipeline = Pipeline(
    ingester=CSVIngester(delimiter=","),
    encoder=AngleEncoder(),
)
```
