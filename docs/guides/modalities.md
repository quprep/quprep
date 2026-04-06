# Data Modalities

QuPrep v0.7.0 extends beyond tabular data with native support for time series, sparse matrices, multi-label datasets, and images. All existing encoders and exporters work unchanged — the new components handle the ingestion and reshaping steps.

---

## Time series

### Ingesting a time series CSV

`TimeSeriesIngester` reads a CSV where rows are timesteps and columns are features. A datetime column is extracted into `metadata["time_index"]` rather than treated as a feature.

```python
import quprep as qd

ingester = qd.TimeSeriesIngester(time_column="date")
dataset = ingester.load("sensor_data.csv")

print(dataset.data.shape)                     # (n_timesteps, n_features)
print(dataset.metadata["time_index"][:3])     # [Timestamp('2024-01-01'), ...]
print(dataset.metadata["modality"])           # "time_series"
```

If the file has no datetime column, omit `time_column` and QuPrep stores an integer index instead.

```python
ingester = qd.TimeSeriesIngester()   # no time column
```

To extract one or more label columns at ingestion, use `target_columns`:

```python
ingester = qd.TimeSeriesIngester(time_column="date", target_columns="label")
dataset = ingester.load("data.csv")
print(dataset.labels.shape)    # (n_timesteps,)
```

---

### Sliding-window transformation

`WindowTransformer` converts a `(n_timesteps, n_features)` Dataset into a `(n_windows, window_size × n_features)` Dataset. Each window becomes one training sample.

```python
wt = qd.WindowTransformer(window_size=8, step=1)
windowed = wt.fit_transform(dataset)

# n_windows = (n_timesteps - window_size) // step + 1
# Features named: {feat}_lag{k} where k=0 is the most recent timestep
print(windowed.feature_names[:4])   # ['temp_lag7', 'temp_lag6', ..., 'temp_lag0']
print(windowed.metadata["modality"])         # "time_series_windowed"
print(windowed.metadata["window_size"])      # 8
print(windowed.metadata["original_n_timesteps"])
```

**Non-overlapping windows** — set `step=window_size`:

```python
wt = qd.WindowTransformer(window_size=8, step=8)  # no overlap
```

**Labels** are aligned to the **last timestep** of each window:

```python
# If dataset has labels y[0..T-1], windowed.labels[i] = y[i + window_size - 1]
```

---

### Full time series pipeline

Chain ingestion → windowing → encoding in a single `Pipeline`:

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
print(result.summary())
```

The `preprocessor` slot in `Pipeline` runs immediately after ingestion and before cleaning/reduction.

---

## Sparse data

scipy.sparse matrices (CSR, CSC, COO, and all other formats) are accepted anywhere QuPrep expects a NumPy array. They are converted to dense at ingestion — no changes to your encoding or export workflow.

```python
import scipy.sparse as sp
import quprep as qd

sparse_matrix = sp.csr_matrix(X)   # could be TF-IDF, adjacency matrix, etc.

result = qd.Pipeline(encoder=qd.AngleEncoder()).fit_transform(sparse_matrix)
```

This works via both the auto-detection path in `Pipeline._ingest()` and directly through `NumpyIngester`:

```python
from quprep.ingest.numpy_ingester import NumpyIngester

dataset = NumpyIngester().load(sparse_matrix)
print(isinstance(dataset.data, np.ndarray))   # True — always dense after ingestion
```

---

## Multi-label datasets

### NumpyIngester — y= parameter

Pass a 1-D or 2-D array as `y` to attach labels at ingestion time:

```python
from quprep.ingest.numpy_ingester import NumpyIngester
import numpy as np

# Single-label
dataset = NumpyIngester().load(X, y=np.array([0, 1, 0, 1, ...]))
print(dataset.labels.shape)   # (n_samples,)

# Multi-label (2-D)
y_multi = np.random.randint(0, 2, size=(n_samples, 3))
dataset = NumpyIngester().load(X, y=y_multi)
print(dataset.labels.shape)   # (n_samples, 3)
```

### CSVIngester — target_columns

Name one or more columns to treat as labels:

```python
from quprep.ingest.csv_ingester import CSVIngester

# Single label column
dataset = CSVIngester(target_columns="label").load("data.csv")
print(dataset.labels.shape)   # (n_samples,)

# Multiple label columns
dataset = CSVIngester(target_columns=["y1", "y2", "y3"]).load("data.csv")
print(dataset.labels.shape)   # (n_samples, 3)

# Label columns are excluded from dataset.data
print("label" not in dataset.feature_names)   # True
```

### Pipeline — y= parameter

Attach labels at `fit_transform()` time when they are not embedded in the data source:

```python
import quprep as qd

result = qd.Pipeline(encoder=qd.AngleEncoder()).fit_transform(X, y=y)
print(result.dataset.labels)   # preserved through all pipeline stages
```

### Labels and FeatureSelector

When labels are present in the dataset, `FeatureSelector(method="mutual_info")` uses them automatically — no need to pass them separately:

```python
import quprep as qd

pipeline = qd.Pipeline(
    cleaner=qd.FeatureSelector(method="mutual_info", threshold=0.0, max_features=4),
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform(X, y=y)
print(result.dataset.n_features)   # 4 — selected by MI with y
```

For 2-D multi-label `y`, mutual information is averaged across all label columns.

### Labels are preserved through all transforms

Labels survive every pipeline stage — normalization, imputation, feature selection, dimensionality reduction. For row-filtering stages (`Imputer(strategy="drop")`, `OutlierHandler(action="remove")`), the same row mask is applied to labels:

```python
from quprep.clean.imputer import Imputer
from quprep.core.dataset import Dataset
import numpy as np

data = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
y    = np.array([10, 20, 30])
ds   = Dataset(data=data, labels=y)

out = Imputer(strategy="drop").fit_transform(ds)
print(out.data.shape)    # (2, 2) — NaN row dropped
print(out.labels)        # [10, 30] — matching labels kept
```

---

## Image data

Requires `pip install quprep[image]` (Pillow).

### Single image file

```python
import quprep as qd

ingester = qd.ImageIngester(size=(28, 28), grayscale=True)
dataset = ingester.load("photo.png")

print(dataset.data.shape)              # (1, 784)  — 28×28 pixels flattened
print(dataset.metadata["modality"])    # "image"
print(dataset.metadata["channels"])    # 1
```

### Directory — flat (no labels)

```python
ingester = qd.ImageIngester(size=(28, 28))
dataset = ingester.load("images/")    # all .png/.jpg/.jpeg/... at top level

print(dataset.data.shape)   # (n_images, 784)
print(dataset.labels)       # None
```

### Directory — class subfolders (ImageFolder convention)

```
images/
  cat/
    img1.jpg
    img2.jpg
  dog/
    img1.jpg
```

```python
ingester = qd.ImageIngester(size=(32, 32))
dataset = ingester.load("images/")

print(dataset.data.shape)    # (3, 1024)
print(dataset.labels)        # ['cat', 'cat', 'dog']
```

### Full image pipeline

```python
import quprep as qd

pipeline = qd.Pipeline(
    ingester=qd.ImageIngester(size=(28, 28)),
    reducer=qd.PCAReducer(n_components=8),   # 784 → 8 features
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform("images/")

print(len(result.encoded))                        # n_images
print(result.encoded[0].metadata["n_qubits"])     # 8
print(result.dataset.labels)                      # class labels preserved
```

**RGB images** — set `grayscale=False` to keep all 3 channels:

```python
ingester = qd.ImageIngester(size=(16, 16), grayscale=False)
dataset = ingester.load("images/")
print(dataset.data.shape)   # (n, 16 × 16 × 3) = (n, 768)
```

**Without normalization** — set `normalize=False` to keep raw [0, 255] pixel values:

```python
ingester = qd.ImageIngester(size=(28, 28), normalize=False)
```
