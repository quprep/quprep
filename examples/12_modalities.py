"""
12 — Data Modalities
====================
Time series, sparse data, and multi-label support.

QuPrep v0.7.0 extends beyond tabular data with three new capabilities:
  - Time series: ingest timestamped CSV, apply sliding-window, encode
  - Sparse data: scipy.sparse matrices accepted anywhere a NumPy array is
  - Multi-label: CSVIngester / NumpyIngester both support multiple target columns

    uv run python examples/12_modalities.py
"""

import os
import tempfile

import numpy as np

import quprep as qd
from quprep.ingest.csv_ingester import CSVIngester
from quprep.ingest.numpy_ingester import NumpyIngester

rng = np.random.default_rng(42)

# ── 1. Time series — ingestion ────────────────────────────────────────────────
#
#   TimeSeriesIngester reads a timestamped CSV and stores the time column in
#   dataset.metadata["time_index"] rather than treating it as a feature.

print("=" * 55)
print("Time series — ingestion")
print("=" * 55)

_ts_csv = "\n".join(
    ["date,temp,humidity,pressure"]
    + [f"2024-01-{i+1:02d},{18+i*0.3:.1f},{55+i*0.5:.1f},{1013+i*0.2:.1f}"
       for i in range(30)]
)

with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
    f.write(_ts_csv)
    ts_path = f.name

ingester = qd.TimeSeriesIngester(time_column="date")
ts_dataset = ingester.load(ts_path)

print(f"Shape        : {ts_dataset.data.shape}")       # (30, 3)
print(f"Features     : {ts_dataset.feature_names}")
print(f"Modality     : {ts_dataset.metadata['modality']}")
print(f"Time index[0]: {ts_dataset.metadata['time_index'][0]}")
print()

# ── 2. Time series — sliding window ───────────────────────────────────────────
#
#   WindowTransformer converts (n_timesteps, n_features) into
#   (n_windows, window_size × n_features) with lag-named features.

print("=" * 55)
print("Time series — sliding window")
print("=" * 55)

wt = qd.WindowTransformer(window_size=5, step=1)
windowed = wt.fit_transform(ts_dataset)

print(f"Input  shape : {ts_dataset.data.shape}")   # (30, 3)
print(f"Output shape : {windowed.data.shape}")      # (26, 15)
print(f"Feature names (first 6): {windowed.feature_names[:6]}")
print(f"Modality     : {windowed.metadata['modality']}")
print(f"window_size  : {windowed.metadata['window_size']}")
print()

# ── 3. Time series — full pipeline ────────────────────────────────────────────
#
#   Chain ingestion → windowing → encoding in a single Pipeline.

print("=" * 55)
print("Time series — full pipeline")
print("=" * 55)

pipeline = qd.Pipeline(
    ingester=qd.TimeSeriesIngester(time_column="date"),
    preprocessor=qd.WindowTransformer(window_size=5, step=1),
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform(ts_path)

print(f"Windows encoded : {len(result.encoded)}")               # 26
print(f"Qubits per window: {result.encoded[0].metadata['n_qubits']}")  # 15
print(f"Audit log:\n{result.summary()}")
print()

# ── 4. Sparse data ────────────────────────────────────────────────────────────
#
#   scipy.sparse matrices are converted to dense at ingestion.
#   All encoders work without modification.

print("=" * 55)
print("Sparse data")
print("=" * 55)

try:
    import scipy.sparse as sp

    dense = rng.random((20, 6))
    dense[dense < 0.7] = 0.0          # ~70% zeros → sparse
    sparse_mat = sp.csr_matrix(dense)
    print(f"scipy.sparse type : {type(sparse_mat).__name__}")
    print(f"Sparsity          : {1 - sparse_mat.nnz / sparse_mat.size:.0%}")

    result_sparse = qd.Pipeline(encoder=qd.AngleEncoder()).fit_transform(sparse_mat)
    print(f"Encoded samples   : {len(result_sparse.encoded)}")
    print(f"Qubits            : {result_sparse.encoded[0].metadata['n_qubits']}")
except ImportError:
    print("scipy not installed — skipping sparse example")
print()

# ── 5. Multi-label — NumpyIngester ────────────────────────────────────────────
#
#   Pass a 2-D label array via y= to attach multi-label targets.

print("=" * 55)
print("Multi-label — NumpyIngester")
print("=" * 55)

X = rng.random((40, 4))
y_multi = rng.integers(0, 2, size=(40, 3))   # 3 binary label columns

dataset = NumpyIngester().load(X, y=y_multi)
print(f"Data shape   : {dataset.data.shape}")
print(f"Labels shape : {dataset.labels.shape}")  # (40, 3)
print()

# ── 6. Multi-label — CSVIngester ──────────────────────────────────────────────
#
#   target_columns names one or more columns to treat as labels.

print("=" * 55)
print("Multi-label — CSVIngester")
print("=" * 55)

_ml_csv = (
    "a,b,c,y1,y2\n"
    + "\n".join(
        f"{rng.random():.3f},{rng.random():.3f},{rng.random():.3f},"
        f"{int(rng.random() > 0.5)},{int(rng.random() > 0.5)}"
        for _ in range(20)
    )
)

with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
    f.write(_ml_csv)
    ml_path = f.name

ds = CSVIngester(target_columns=["y1", "y2"]).load(ml_path)
print(f"Features     : {ds.feature_names}")    # ['a', 'b', 'c']
print(f"Labels shape : {ds.labels.shape}")     # (20, 2)
print()

# ── 7. Labels survive the pipeline ────────────────────────────────────────────
#
#   Labels are propagated through every transform stage (normalization,
#   imputation, feature selection, reduction, etc.).

print("=" * 55)
print("Labels through pipeline")
print("=" * 55)

pipeline_y = qd.Pipeline(
    ingester=CSVIngester(target_columns="y1"),
    encoder=qd.AngleEncoder(),
)
result_y = pipeline_y.fit_transform(ml_path)

print(f"result.dataset.labels : {result_y.dataset.labels[:5]}")  # first 5 labels
print(f"Labels preserved      : {result_y.dataset.labels is not None}")
print()

# ── 8. Labels through feature selection ───────────────────────────────────────
#
#   Pipeline.fit() uses dataset.labels automatically for mutual_info selection.

print("=" * 55)
print("Feature selection with embedded labels")
print("=" * 55)

X_cls = rng.random((60, 8))
y_cls = (X_cls[:, 0] > 0.5).astype(int)   # only feature 0 is informative

pipeline_sel = qd.Pipeline(
    cleaner=qd.FeatureSelector(method="mutual_info", threshold=0.0, max_features=4),
    encoder=qd.AngleEncoder(),
)
result_sel = pipeline_sel.fit_transform(X_cls, y=y_cls)

print("Input features  : 8")
print(f"Selected features: {result_sel.dataset.n_features}")   # 4
print(f"Labels preserved : {result_sel.dataset.labels is not None}")

# cleanup
os.unlink(ts_path)
os.unlink(ml_path)
