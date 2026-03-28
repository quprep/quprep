"""
01 — Quickstart
===============
The fastest way to go from a CSV file to quantum circuits.

    uv run python examples/01_quickstart.py
"""

import numpy as np
import pandas as pd

import quprep as qd

# ── 1. Create a small dataset ────────────────────────────────────────────────

data = pd.DataFrame(
    {
        "age": [25, 32, 45, 28, 38],
        "income": [40_000, 55_000, 90_000, 48_000, 72_000],
        "score": [0.72, 0.85, 0.91, 0.60, 0.78],
        "label": [0, 1, 1, 0, 1],
    }
)
data.to_csv("/tmp/customers.csv", index=False)

# ── 2. One call: CSV → QASM circuits ─────────────────────────────────────────

result = qd.prepare("/tmp/customers.csv", encoding="angle", framework="qasm")

print(f"Samples encoded : {len(result.circuits)}")
print(f"First circuit   :\n{result.circuit}")

# ── 3. Encode a raw numpy array ───────────────────────────────────────────────

X = np.random.default_rng(42).uniform(0, 1, size=(3, 4))
result2 = qd.prepare(X, encoding="basis", framework="qasm")

print("Basis circuits:")
for i, c in enumerate(result2.circuits):
    print(f"  Sample {i}:\n{c}")
