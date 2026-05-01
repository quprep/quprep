# Streaming / Chunked Ingestion

When a dataset is too large to fit in RAM, QuPrep's streaming API lets you process it row-chunk by row-chunk without loading the full file at once.

Three components support streaming:

| Component | Method | Description |
|---|---|---|
| `CSVIngester` | `.stream(path, chunksize)` | Yields `Dataset` chunks from a CSV |
| `NumpyIngester` | `.stream(X, chunksize)` | Yields `Dataset` slices from a NumPy array |
| `Pipeline` | `.stream(source, chunksize)` | Applies a **fitted** pipeline to chunks; yields `PipelineResult` objects |

---

## CSVIngester.stream()

```python
from quprep.ingest.csv_ingester import CSVIngester

for chunk in CSVIngester().stream("big_data.csv", chunksize=1000):
    print(chunk.n_samples, chunk.feature_names)
```

Each chunk is a full `Dataset` object — feature names, types, labels (if `target_columns` is set), and a `metadata["chunk"]` index are all populated.

```python
# With label extraction
ingester = CSVIngester(target_columns="label")
for chunk in ingester.stream("labelled.csv", chunksize=500):
    print(chunk.labels.shape)
```

---

## NumpyIngester.stream()

```python
import numpy as np
from quprep.ingest.numpy_ingester import NumpyIngester

X = np.load("embeddings.npy")   # (1_000_000, 512)
y = np.load("labels.npy")

for chunk in NumpyIngester().stream(X, y=y, chunksize=2000):
    # chunk.data: shape (2000, 512)  |  chunk.labels: shape (2000,)
    ...
```

The array is sliced in place — no copies are made.

---

## Pipeline.stream()

The pipeline must be **fitted first** (via `fit()` or `fit_transform()`). Normaliser statistics and all other fitted parameters are reused for every chunk; only `transform` is called per chunk.

```python
import quprep as qd
import numpy as np

# Fit on a representative sample
pipeline = qd.Pipeline(
    cleaner=qd.Imputer(strategy="mean"),
    encoder=qd.IQPEncoder(),
    exporter=qd.QASMExporter(),
)
pipeline.fit(X[:5000])

# Stream the full dataset
all_circuits = []
for result in pipeline.stream("big_data.csv", chunksize=1000):
    all_circuits.extend(result.circuits)
```

`Pipeline.stream()` accepts both a file path (CSV) and a NumPy array:

```python
for result in pipeline.stream(X_full, chunksize=1000):
    process(result.circuits)
```

---

## Typical workflow for large datasets

```python
import quprep as qd

# 1. Profile a sample to choose an encoding
sample = qd.CSVIngester().load("data_sample.csv")
rec = qd.recommend(sample, task="classification")
print(rec.method)

# 2. Fit the pipeline on the sample
pipeline = qd.Pipeline(
    encoder=getattr(qd, f"{rec.method.title().replace('_','')}Encoder")(),
    exporter=qd.QASMExporter(),
)
pipeline.fit(sample)

# 3. Stream the full dataset in chunks
for result in pipeline.stream("big_data.csv", chunksize=500):
    save_circuits(result.circuits)   # your storage / queue logic here
```

---

## Notes

- `Pipeline.stream()` raises `RuntimeError` if the pipeline has not been fitted.
- Only CSV file paths and NumPy arrays are accepted as `source`. For other ingester types (HuggingFace, OpenML, etc.) load each chunk manually and call `pipeline.transform(chunk)`.
- Normaliser min/max values are from the **fit sample**, not the full streamed data. For production workloads, fit on a representative sample that covers the expected data range.
