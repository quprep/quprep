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

---

## Kaggle datasets

Requires `pip install quprep[kaggle]` and a Kaggle API token.

### Authentication

Kaggle requires an account and an API token for all downloads (including public datasets).

1. Create an account at [kaggle.com](https://www.kaggle.com)
2. Go to **Settings → API → Create New Token** — this downloads `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows)

Alternatively, set environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### Datasets

Kaggle datasets use the `owner/dataset-name` format, visible in the dataset URL:
`https://www.kaggle.com/datasets/owner/dataset-name`

```python
import quprep as qd

# Load all files — picks the first CSV found
ds = qd.KaggleIngester(target_columns="Survived").load("heptapod/titanic")

print(ds.data.shape)           # (n_samples, n_features)
print(ds.labels.shape)         # (n_samples,)
print(ds.metadata["source"])   # "kaggle:dataset:heptapod/titanic"
print(ds.metadata["file"])     # name of the CSV loaded
```

Load a specific file from a multi-file dataset:

```python
ds = qd.KaggleIngester(
    file_name="train.csv",
    target_columns="label",
).load("owner/dataset-name")
```

### Competition data

Competition data uses just the competition slug (visible in the URL):
`https://www.kaggle.com/competitions/titanic`

```python
# Download all competition files, load first CSV
ds = qd.KaggleIngester(target_columns="Survived").load_competition("titanic")

# Or load a specific file
ds = qd.KaggleIngester(file_name="train.csv").load_competition("titanic")

print(ds.metadata["source"])       # "kaggle:competition:titanic"
print(ds.metadata["competition"])  # "titanic"
```

### Full pipeline

```python
import quprep as qd

pipeline = qd.Pipeline(
    cleaner=qd.Imputer(strategy="median"),
    reducer=qd.PCAReducer(n_components=8),
    encoder=qd.AngleEncoder(),
)

ds = qd.KaggleIngester(
    file_name="train.csv",
    target_columns="SalePrice",
).load("competitions/house-prices-advanced-regression-techniques")

result = pipeline.fit_transform(ds)
print(len(result.encoded))
```

### Categorical columns

By default non-numeric columns are dropped. Set `numeric_only=False` to keep them in `Dataset.categorical_data` for encoding with `CategoricalEncoder`:

```python
ds = qd.KaggleIngester(
    target_columns="Survived",
    numeric_only=False,
).load("heptapod/titanic")

print(ds.categorical_data.keys())   # e.g. {'Sex', 'Embarked', 'Name', ...}
```

---

## OpenML datasets

Requires `pip install quprep[openml]`. No account is needed for public datasets — OpenML is fully open.

OpenML datasets are identified by an integer ID or by name. You can browse datasets at [openml.org/search?type=data](https://www.openml.org/search?type=data).

### Load by dataset ID

```python
import quprep as qd

# Iris — dataset ID 61
ds = qd.OpenMLIngester(target_column="class").load(61)

print(ds.data.shape)              # (150, 4)
print(ds.labels.shape)            # (150,)
print(ds.metadata["source"])      # "openml:61"
print(ds.metadata["dataset_name"])  # "iris"
print(ds.metadata["version"])     # e.g. 1
```

### Load by dataset name

```python
# Latest version is used automatically
ds = qd.OpenMLIngester(target_column="class").load("iris")

# Specific version
ds = qd.OpenMLIngester(target_column="class", version=1).load("iris")
```

### No target (unsupervised)

```python
# MNIST_784 — no label extraction
ds = qd.OpenMLIngester().load(554)

print(ds.data.shape)   # (70000, 784)
print(ds.labels)       # None
```

When `target_column` is not set, OpenML's default target attribute is used if the dataset defines one. Pass `target_column=None` explicitly and set `default_target_attribute` to empty on the dataset if you want no labels at all — or just ignore `ds.labels`.

### Full pipeline

```python
import quprep as qd

pipeline = qd.Pipeline(
    cleaner=qd.Imputer(strategy="median"),
    reducer=qd.PCAReducer(n_components=8),
    encoder=qd.AngleEncoder(),
)

ds = qd.OpenMLIngester(target_column="class").load("credit-g")
result = pipeline.fit_transform(ds)

print(len(result.encoded))
print(result.encoded[0].metadata["n_qubits"])   # 8
```

### Categorical columns

OpenML datasets often have categorical features. By default they are dropped. Set `numeric_only=False` to keep them in `Dataset.categorical_data`:

```python
ds = qd.OpenMLIngester(
    target_column="class",
    numeric_only=False,
).load("credit-g")

print(ds.categorical_data.keys())   # categorical feature names
```

Then use `CategoricalEncoder` to encode them before encoding:

```python
pipeline = qd.Pipeline(
    cleaner=qd.CategoricalEncoder(method="onehot"),
    encoder=qd.AngleEncoder(),
)
```

---

## HuggingFace datasets

Requires `pip install quprep[huggingface]` (the `datasets` library from HuggingFace).

### Connecting to HuggingFace

`HuggingFaceIngester` is a direct wrapper around HuggingFace's `load_dataset`. The mapping is straightforward:

```python
# Standard HuggingFace usage:
from datasets import load_dataset

dataset    = load_dataset("username/my_dataset")
train_data = load_dataset("username/my_dataset", split="train")
valid_data = load_dataset("username/my_dataset", split="validation")
test_data  = load_dataset("username/my_dataset", split="test")

# QuPrep equivalent — same dataset, same splits:
import quprep as qd

ds_train = qd.HuggingFaceIngester(split="train").load("username/my_dataset")
ds_valid = qd.HuggingFaceIngester(split="validation").load("username/my_dataset")
ds_test  = qd.HuggingFaceIngester(split="test").load("username/my_dataset")
```

QuPrep adds automatic modality detection, label extraction, and direct integration with encoders and pipelines on top of the raw `load_dataset` call.

**Public datasets require no account.** Anyone can load public datasets with just `pip install quprep[huggingface]` — no HuggingFace account or token needed.

**Gated datasets require a token.** Datasets where you've accepted terms on the HuggingFace website require authentication:

```python
# Option 1 — login once in your terminal, token stored in ~/.huggingface/token:
#   huggingface-cli login
# Then pass token=True in code:
ds = qd.HuggingFaceIngester(split="train", token=True).load("meta-llama/Llama-3-8B")

# Option 2 — pass the token string directly:
ds = qd.HuggingFaceIngester(split="train", token="hf_abc123").load("owner/gated-dataset")
```

### Modality auto-detection

| HuggingFace feature type | Detected modality | Notes |
|---|---|---|
| `Image` column | `image` | Requires `quprep[image]` (Pillow) |
| String-only, no numeric | `text` | TF-IDF by default; no extra deps |
| Numeric / mixed | `tabular` | String columns dropped unless `numeric_only=False` |
| Audio / Video only | Error | `NotImplementedError` with clear message |

### Tabular dataset

```python
import quprep as qd

ds = qd.HuggingFaceIngester(
    split="train",
    target_columns="label",
).load("imodels/credit-card")

print(ds.data.shape)              # (n_samples, n_features)
print(ds.labels.shape)            # (n_samples,)
print(ds.metadata["source"])      # "huggingface:imodels/credit-card"
print(ds.metadata["modality"])    # "tabular"
```

Full pipeline:

```python
pipeline = qd.Pipeline(
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform(ds)
print(len(result.encoded))
```

### Image dataset (auto-detected)

```python
ds = qd.HuggingFaceIngester(
    split="train",
    target_columns="label",
    image_size=(28, 28),    # resize to 28×28 before flattening
    grayscale=True,
).load("ylecun/mnist")

print(ds.data.shape)     # (60000, 784)
print(ds.labels.shape)   # (60000,)
```

Override the image column explicitly when needed:

```python
ds = qd.HuggingFaceIngester(
    modality="image",
    image_column="img",
    image_size=(32, 32),
    grayscale=False,     # keep RGB → 32×32×3 = 3072 pixels
).load("cifar10")
```

### Text dataset (auto-detected)

```python
ds = qd.HuggingFaceIngester(
    split="train",
    target_columns="label",
    text_method="tfidf",   # default — no extra deps
    max_features=64,
).load("imdb")

print(ds.data.shape)   # (25000, 64)
```

Semantic embeddings with sentence-transformers (requires `quprep[text]`):

```python
ds = qd.HuggingFaceIngester(
    split="train",
    text_method="sentence_transformers",
    text_model="all-MiniLM-L6-v2",   # 384-d, fast
).load("imdb")

print(ds.data.shape)   # (25000, 384)
```

Override the text column when the dataset has multiple string columns:

```python
ds = qd.HuggingFaceIngester(
    modality="text",
    text_column="premise",
    target_columns="label",
).load("snli")
```

### Graph dataset (explicit)

Graph datasets require `modality="graph"` — auto-detection does not apply here since graph structure is stored in varying column formats.

```python
ds = qd.HuggingFaceIngester(
    modality="graph",
    split="train",
    target_columns="y",
    edge_index_column="edge_index",  # COO format: shape [2, E]
    n_graph_features=8,              # pad/truncate Laplacian+degree features
).load("graphs-datasets/ogbg-molhiv")

print(ds.data.shape)     # (n_graphs, 8)
print(ds.labels.shape)   # (n_graphs,)
```

Internally, graph datasets are routed through `GraphIngester` — all the same feature options apply (`features="laplacian_eigenvalues"`, `features="all"`, etc.).

### Dataset configs / subsets

Some datasets have multiple configurations (languages, domains, etc.). Pass `config_name` — this maps to the `name` argument in `load_dataset`:

```python
# Standard HF:
load_dataset("amazon_reviews_multi", name="en", split="train")

# QuPrep:
ds = qd.HuggingFaceIngester(split="train").load("amazon_reviews_multi", config_name="en")
ds = qd.HuggingFaceIngester(split="train").load("amazon_reviews_multi", config_name="de")
```

### Unsupported modalities

If a dataset contains only audio or video columns (with no numeric fallback), QuPrep raises `NotImplementedError` with a clear message:

```python
# Raises NotImplementedError: Dataset 'foo/bar' contains audio data (column(s): ['audio']).
# QuPrep currently supports: ['tabular', 'image', 'text', 'graph'].
# Pass modality='tabular' to ignore unsupported columns and process any remaining numeric features.
ds = qd.HuggingFaceIngester().load("foo/audio-only-dataset")
```

If a dataset has mixed audio + numeric columns, auto-detection falls through to `tabular` and processes the numeric columns, ignoring audio.
