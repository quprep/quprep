"""
18 — Streaming / Chunked Ingestion
=====================================
Process datasets larger than RAM without loading them fully into memory.
CSVIngester and NumpyIngester both support streaming; Pipeline.stream()
applies a fitted pipeline to a source in chunks.

    uv run python examples/18_streaming.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import quprep as qd
from quprep.ingest.csv_ingester import CSVIngester
from quprep.ingest.numpy_ingester import NumpyIngester

# ── 1. CSVIngester.stream() ───────────────────────────────────────────────────

print("=" * 55)
print("CSVIngester.stream()  — 500-row CSV in 100-row chunks")
print("=" * 55)

# Write a temporary CSV
rng = np.random.default_rng(0)
df = pd.DataFrame(rng.uniform(0, 1, (500, 6)), columns=[f"f{i}" for i in range(6)])

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
    csv_path = Path(f.name)
    df.to_csv(csv_path, index=False)

total_rows = 0
for i, chunk in enumerate(CSVIngester().stream(csv_path, chunksize=100)):
    total_rows += chunk.n_samples
    print(f"  chunk {i}: {chunk.n_samples} rows × {chunk.n_features} features")

print(f"Total rows processed: {total_rows}")
print()

# ── 2. NumpyIngester.stream() ─────────────────────────────────────────────────

print("=" * 55)
print("NumpyIngester.stream()  — 1000-row array in 250-row chunks")
print("=" * 55)

X = rng.uniform(0, 1, (1000, 4))
y = rng.integers(0, 2, 1000)

chunks = list(NumpyIngester().stream(X, y=y, chunksize=250))
print(f"Number of chunks  : {len(chunks)}")
print(f"Rows per chunk    : {[c.n_samples for c in chunks]}")

# Verify full data is preserved across chunks
reconstructed = np.vstack([c.data for c in chunks])
assert np.allclose(reconstructed, X), "Data mismatch!"
print("Data integrity    : OK (all rows match original)")
print()

# ── 3. Pipeline.stream() — fit once, encode in chunks ─────────────────────────

print("=" * 55)
print("Pipeline.stream()  — fit on sample, stream full dataset")
print("=" * 55)

# Fit on a representative sample, then stream the full array
pipeline = qd.Pipeline(
    cleaner=qd.Imputer(strategy="mean"),
    encoder=qd.IQPEncoder(),
    exporter=qd.QASMExporter(),
)

sample = X[:100]
pipeline.fit(sample)

total_circuits = 0
for i, result in enumerate(pipeline.stream(X, chunksize=250)):
    total_circuits += len(result.circuits)
    print(f"  chunk {i}: {len(result.circuits)} circuits")

print(f"Total circuits    : {total_circuits}  (expected {len(X)})")
print()

# ── 4. Streaming a CSV through a pipeline ────────────────────────────────────

print("=" * 55)
print("Pipeline.stream() with CSV source")
print("=" * 55)

pipeline2 = qd.Pipeline(encoder=qd.AngleEncoder(), exporter=qd.QASMExporter())
pipeline2.fit_transform(str(csv_path))   # fit on the full CSV first

all_circuits = []
for result in pipeline2.stream(str(csv_path), chunksize=150):
    all_circuits.extend(result.circuits)

print(f"Total circuits from streaming: {len(all_circuits)}")
print(f"Expected                     : {len(df)}")

csv_path.unlink()  # clean up temp file
