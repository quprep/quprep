# Data Modalities

QuPrep v0.7.0 extends beyond tabular data with native support for time series, sparse matrices, multi-label datasets, images, text, and graphs. All existing encoders and exporters work unchanged — the new components handle the ingestion and reshaping steps.

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

---

## Text data

Two embedding methods are available:

| Method | Deps | Output size | Best for |
|---|---|---|---|
| `tfidf` | none (sklearn) | up to `max_features` | Keyword-based tasks, no download needed |
| `sentence_transformers` | `quprep[text]` | 384–768d dense | Semantic similarity, general NLP |

### From a list of strings

```python
import quprep as qd

texts = [
    "quantum computing is powerful",
    "machine learning needs data",
    "hybrid algorithms combine both",
]

ingester = qd.TextIngester(method="tfidf", max_features=32)
dataset = ingester.load(texts)

print(dataset.data.shape)            # (3, 32)
print(dataset.metadata["modality"])  # "text"
print(dataset.metadata["method"])    # "tfidf"
```

### From a .txt file (one sentence per line)

```python
ingester = qd.TextIngester(method="tfidf", max_features=64)
dataset = ingester.load("corpus.txt")
```

Blank lines are skipped automatically.

### From a CSV

Use `text_column` to name the column containing text, and `target_column` to extract labels:

```python
ingester = qd.TextIngester(
    method="tfidf",
    text_column="review",
    target_column="sentiment",
    max_features=128,
)
dataset = ingester.load("reviews.csv")

print(dataset.data.shape)     # (n_rows, 128)
print(dataset.labels)         # sentiment column values
```

Multi-label targets work the same way as other ingesters:

```python
ingester = qd.TextIngester(
    text_column="text",
    target_column=["y1", "y2"],
)
dataset = ingester.load("data.csv")
print(dataset.labels.shape)   # (n_rows, 2)
```

### Semantic embeddings (sentence-transformers)

Requires `pip install quprep[text]`.

```python
ingester = qd.TextIngester(
    method="sentence_transformers",
    model="all-MiniLM-L6-v2",   # 384-d, fast — default
)
dataset = ingester.load(texts)
print(dataset.data.shape)   # (3, 384) — directly encode, no PCA needed
```

### Full text pipeline

```python
import quprep as qd

pipeline = qd.Pipeline(
    ingester=qd.TextIngester(method="tfidf", max_features=64),
    reducer=qd.PCAReducer(n_components=8),   # 64 → 8 features
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform("corpus.txt")

print(len(result.encoded))                       # n_sentences
print(result.encoded[0].metadata["n_qubits"])    # 8
```

With sentence-transformers the PCA step is often unnecessary since the output is already compact:

```python
pipeline = qd.Pipeline(
    ingester=qd.TextIngester(method="sentence_transformers"),
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform(texts)
# 384 qubits — add a PCAReducer if your qubit budget is smaller
```

---

## Graph data

Two paths are provided, matching different use cases:

| Path | Class | Output | Best for |
|---|---|---|---|
| **Lossy** | `GraphIngester` | Feature vector (Laplacian + degree) | Graph classification with standard encoders |
| **Lossless** | `GraphStateEncoder` | Graph state circuit $\|G\rangle$ | Structure-preserving quantum graph algorithms |

### Lossy path — GraphIngester

Extracts a fixed-size feature vector from each graph. Works with any existing encoder.

```python
import numpy as np
import quprep as qd

# Triangle graph adjacency matrix
adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)

ingester = qd.GraphIngester()   # features="all" (Laplacian eigenvalues + degrees)
dataset = ingester.load(adj)

print(dataset.data.shape)             # (1, 6)  — 3 eigenvalues + 3 degrees
print(dataset.metadata["modality"])   # "graph"
```

**Feature options:**

```python
qd.GraphIngester(features="laplacian_eigenvalues")  # Laplacian spectrum only
qd.GraphIngester(features="degree")                 # degree sequence only
qd.GraphIngester(features="all")                    # both (default)
```

**Batch of graphs** — use `n_features` to pad/truncate to a common size:

```python
graphs = [adj_3node, adj_5node, adj_7node]
dataset = qd.GraphIngester(n_features=8).load(graphs)
print(dataset.data.shape)   # (3, 8) — all padded/truncated to 8 features
```

**networkx graphs:**

```python
import networkx as nx

G = nx.karate_club_graph()
dataset = qd.GraphIngester(n_features=16).load(G)
```

**Full lossy pipeline:**

```python
import quprep as qd

pipeline = qd.Pipeline(
    ingester=qd.GraphIngester(n_features=8),
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform([adj1, adj2, adj3])
print(len(result.encoded))                       # 3 graphs
print(result.encoded[0].metadata["n_qubits"])    # 8
```

### Lossless path — GraphStateEncoder

Produces a true graph state $|G\rangle = \prod_{(i,j)\in E} CZ_{ij}\, H^{\otimes n}|0\rangle^n$. Every edge becomes a CZ entangling gate — the full graph structure is preserved in the circuit.

```python
import numpy as np
import quprep as qd

adj = np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], dtype=float)

encoder = qd.GraphStateEncoder()
result = encoder.encode_graph(adj)

print(result.metadata["n_qubits"])   # 4
print(result.metadata["edges"])      # [(0,1),(0,2),(1,2),(2,3)]
print(result.metadata["n_edges"])    # 4
```

**Export to QASM:**

```python
from quprep.export.qasm_export import QASMExporter

qasm = QASMExporter().export(result)
print(qasm)
# OPENQASM 3.0;
# include "stdgates.inc";
# qubit[4] q;
# h q[0]; h q[1]; h q[2]; h q[3];
# cz q[0], q[1];
# cz q[0], q[2];
# ...
```

**Batch encoding:**

```python
results = encoder.encode_batch_graphs([adj1, adj2, adj3])
circuits = QASMExporter().export_batch(results)
```
