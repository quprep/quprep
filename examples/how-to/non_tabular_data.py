"""
How to Work with Non-Tabular Data
===================================
QuPrep handles four data modalities beyond standard 2D arrays:
time series (sliding window), sparse matrices, multi-label targets,
and image/text feature representations.

    uv run python examples/how-to/non_tabular_data.py
"""

import tempfile
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

import quprep as qd
from quprep import QuPrepWarning

print(f"quprep {qd.__version__}\n")


# ── 1. Time series — sliding window ───────────────────────────────────────────
#
# QuPrep works with time series by converting overlapping windows into rows.
# Use pandas to read the raw CSV, then build sliding windows manually before
# handing the 2D array to NumpyIngester.

print("── 1. Time series (sliding window) ──────────────────────────────────────")

rng_ts = np.random.default_rng(7)
raw_ts = rng_ts.uniform(0, 1, (20, 2))  # 20 time steps, 2 sensors

window_size = 4
windows = np.array(
    [raw_ts[i : i + window_size].flatten() for i in range(len(raw_ts) - window_size + 1)]
)
ds = qd.NumpyIngester().load(windows)
print(f"   Raw time steps   : {len(raw_ts)}")
print(f"   Window size      : {window_size}")
print(f"   Dataset shape    : {ds.data.shape}  (17 windows × 8 features)")
print()


# ── 2. Sparse matrices ────────────────────────────────────────────────────────
#
# NumpyIngester accepts scipy.sparse matrices anywhere a dense array is
# expected. Useful for text TF-IDF features or graph adjacency representations.

print("── 2. Sparse matrices ───────────────────────────────────────────────────")
sparse_X = sp.random(30, 10, density=0.2, format="csr", random_state=42)
ds_sparse = qd.NumpyIngester().load(sparse_X)
n_elements = sparse_X.shape[0] * sparse_X.shape[1]
print(f"   Sparse shape    : {sparse_X.shape}  density={sparse_X.nnz / n_elements:.2f}")
print(f"   Dataset shape   : {ds_sparse.data.shape}")
print()


# ── 3. Multi-label targets ────────────────────────────────────────────────────
#
# NumpyIngester supports multi-label targets: pass a 2D label array where each
# column is one label. See section 4 for the CSV equivalent via pandas.

print("── 3. Multi-label ───────────────────────────────────────────────────────")
rng = np.random.default_rng(0)
X_ml = rng.uniform(0, 1, (20, 4))
y_ml = np.column_stack([
    (X_ml[:, 0] > 0.5).astype(int),
    (X_ml[:, 1] > 0.5).astype(int),
])

ds_ml = qd.NumpyIngester().load(X_ml, y=y_ml)
print(f"   Feature shape  : {ds_ml.data.shape}")
print(f"   Label shape    : {ds_ml.labels.shape}  (2 label columns)")
print()


# ── 4. Multi-label from CSV ───────────────────────────────────────────────────
#
# CSVIngester loads all columns as features. To separate label columns, load
# with pandas first and split explicitly before passing to NumpyIngester.

print("── 4. Multi-label from CSV ──────────────────────────────────────────────")
ml_csv = "f0,f1,f2,tag_A,tag_B\n"
for i in range(10):
    ml_csv += f"{rng.uniform():.3f},{rng.uniform():.3f},{rng.uniform():.3f},"
    ml_csv += f"{int(rng.uniform()>0.5)},{int(rng.uniform()>0.5)}\n"

with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
    f.write(ml_csv)
    ml_path = f.name

ml_df = pd.read_csv(ml_path)
X_ml_csv = ml_df[["f0", "f1", "f2"]].values
y_ml_csv = ml_df[["tag_A", "tag_B"]].values

ds_ml_csv = qd.NumpyIngester().load(X_ml_csv, y=y_ml_csv)
print(f"   Feature shape  : {ds_ml_csv.data.shape}")
print(f"   Label shape    : {ds_ml_csv.labels.shape}")
print()


# ── 5. Encode time series data ────────────────────────────────────────────────

print("── 5. Encode time series ────────────────────────────────────────────────")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = qd.Pipeline(
        normalizer=qd.Scaler(strategy="minmax_pi"),
        encoder=qd.AngleEncoder(),
    ).fit_transform(ds)

print(f"   Circuits  : {len(result.encoded)}")
print(f"   Qubits    : {result.encoded[0].metadata['n_qubits']}")
print("   (8 features → 8 qubits per window)")
