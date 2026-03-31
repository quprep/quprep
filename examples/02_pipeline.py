"""
02 — Full Pipeline
==================
Chain cleaning → normalization → encoding → export using the Pipeline API.

    uv run python examples/02_pipeline.py
"""

import numpy as np
import pandas as pd

import quprep as qd

# ── 1. Messy dataset with missing values and outliers ────────────────────────

rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "a": [1.2, None, 3.4, 2.1, 999.0, 2.8, None, 1.9],   # outlier + NaN
        "b": [0.5, 0.6, None, 0.4, 0.7, None, 0.3, 0.8],     # NaN
        "c": rng.uniform(1, 5, 8).tolist(),
        "d": rng.uniform(0, 1, 8).tolist(),
    }
)

print("Raw data:")
print(df.to_string())
print()

# ── 2. Build the pipeline ────────────────────────────────────────────────────

pipeline = qd.Pipeline(
    cleaner=qd.Imputer(strategy="mean"),     # fill NaNs with column mean
    encoder=qd.AngleEncoder(rotation="ry"),  # Ry rotation gates
    exporter=qd.QASMExporter(),              # OpenQASM 3.0 output
)

result = pipeline.fit_transform(df)

# ── 3. Inspect results ───────────────────────────────────────────────────────

print(f"Encoded {len(result.encoded)} samples")
print(f"Qubits per circuit : {result.encoded[0].metadata['n_qubits']}")
print(f"Circuit depth      : {result.encoded[0].metadata['depth']}")
print()
print("First QASM circuit:")
print(result.circuit)

# ── 4. Save all circuits to disk (save_batch) ────────────────────────────────

exporter = qd.QASMExporter()
paths = exporter.save_batch(result.encoded, "/tmp/quprep_circuits/", stem="circuit")
print(f"Saved {len(paths)} circuits → {paths[0].parent}/")

# ── 5. Save and reload a fitted pipeline ─────────────────────────────────────

pipeline.save("/tmp/quprep_pipeline.pkl")
loaded = qd.Pipeline.load("/tmp/quprep_pipeline.pkl")

X_new = df.fillna(df.mean(numeric_only=True)).to_numpy()[:3]
result2 = loaded.transform(X_new)
print(f"Loaded pipeline → {len(result2.encoded)} samples encoded")
