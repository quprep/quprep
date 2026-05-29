"""
How to Load External Data
==========================
QuPrep can load data from HuggingFace, OpenML, Kaggle, and streaming CSV/
NumPy sources in addition to local files. This guide covers all four.

Optional dependencies per source:
    pip install quprep[huggingface]   # HuggingFaceIngester
    pip install quprep[openml]        # OpenMLIngester
    pip install quprep[kaggle]        # KaggleIngester  (also needs ~/.kaggle/kaggle.json)

    uv run python examples/how-to/load_external_data.py
"""

import tempfile
import warnings

import numpy as np
import pandas as pd

import quprep as qd
from quprep import QuPrepWarning

print(f"quprep {qd.__version__}\n")


# ── 1. CSV file ───────────────────────────────────────────────────────────────

print("── 1. CSVIngester ───────────────────────────────────────────────────────")
df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.1, 0.2, 0.3], "c": [0.5, 0.6, 0.7]})
with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
    df.to_csv(f, index=False)
    csv_path = f.name

dataset = qd.CSVIngester().load(csv_path)
print(f"   Shape  : {dataset.data.shape}")
print()


# ── 2. NumPy array ────────────────────────────────────────────────────────────

print("── 2. NumpyIngester ─────────────────────────────────────────────────────")
rng = np.random.default_rng(0)
X = rng.uniform(0, 1, (50, 4))
y = (X[:, 0] > 0.5).astype(int)
dataset = qd.NumpyIngester().load(X, y=y)
print(f"   Shape  : {dataset.data.shape}")
print(f"   Labels : {np.bincount(y.astype(int))}")
print()


# ── 3. Streaming CSV (chunked ingestion) ──────────────────────────────────────
#
# For datasets larger than RAM, CSVIngester.stream() yields Dataset chunks.
# Each chunk carries metadata: chunk index, chunk size, and source path.
# Pipeline.stream() applies a fitted pipeline to a stream lazily.

print("── 3. CSVIngester.stream() ──────────────────────────────────────────────")
big_df = pd.DataFrame({"x0": rng.uniform(0, 1, 200), "x1": rng.uniform(0, 1, 200)})
with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
    big_df.to_csv(f, index=False)
    big_csv = f.name

chunks = list(qd.CSVIngester().stream(big_csv, chunksize=50))
print(f"   Total chunks : {len(chunks)}")
print(f"   Chunk shapes : {[c.data.shape for c in chunks]}")
print()


# ── 4. Streaming through a fitted pipeline ────────────────────────────────────

print("── 4. Pipeline.stream() ─────────────────────────────────────────────────")
first_chunk = chunks[0]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    fitted_pipeline = qd.Pipeline(
        normalizer=qd.Scaler(strategy="minmax_pi"),
        encoder=qd.AngleEncoder(),
    ).fit(first_chunk)

encoded_chunks = list(fitted_pipeline.stream(big_csv, chunksize=50))
print(f"   Encoded chunks : {len(encoded_chunks)}")
print(f"   Circuits/chunk : {[len(c.encoded) for c in encoded_chunks]}")
print()


# ── 5. HuggingFace (optional) ─────────────────────────────────────────────────

print("── 5. HuggingFaceIngester  (pip install quprep[huggingface]) ────────────")
try:
    from quprep.ingest.huggingface_ingester import HuggingFaceIngester
    ds = HuggingFaceIngester().load("scikit-learn/iris")
    print(f"   Shape  : {ds.data.shape}")
    print(f"   Labels : {np.unique(ds.labels)}")
except ImportError:
    print("   skipped — run: pip install quprep[huggingface]")
except Exception as e:
    print(f"   skipped: {e}")
print()


# ── 6. OpenML (optional) ──────────────────────────────────────────────────────

print("── 6. OpenMLIngester  (pip install quprep[openml]) ──────────────────────")
try:
    from quprep.ingest.openml_ingester import OpenMLIngester
    ds = OpenMLIngester().load(61)  # Iris dataset ID on OpenML
    print(f"   Shape  : {ds.data.shape}")
    print(f"   Labels : {np.unique(ds.labels)}")
except ImportError:
    print("   skipped — run: pip install quprep[openml]")
except Exception as e:
    print(f"   skipped: {e}")
