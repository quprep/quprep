"""
05 — Encoding Recommendation
=============================
Use qd.recommend() to automatically select the best encoding
for your dataset and task.

    uv run python examples/05_recommend.py
"""

import numpy as np

import quprep as qd

# ── Shared data ───────────────────────────────────────────────────────────────

# Simulated dataset: 100 samples, 6 features, mixed continuous
rng = np.random.default_rng(42)
X = rng.uniform(0, 1, size=(100, 6))

# ── 1. Recommend for a classification task ────────────────────────────────────

print("=" * 55)
print("Task: classification  |  qubit budget: 8")
print("=" * 55)

rec = qd.recommend(X, task="classification", qubits=8)
print(rec)

# ── 2. Recommend for a kernel method ─────────────────────────────────────────

print("=" * 55)
print("Task: kernel  |  qubit budget: 6")
print("=" * 55)

rec_kernel = qd.recommend(X, task="kernel", qubits=6)
print(rec_kernel)

# ── 3. Recommend for QAOA (binary optimization) ───────────────────────────────

print("=" * 55)
print("Task: qaoa  |  qubit budget: 6  |  binary data")
print("=" * 55)

X_binary = (X > 0.5).astype(float)
rec_qaoa = qd.recommend(X_binary, task="qaoa", qubits=6)
print(rec_qaoa)

# ── 4. Apply the recommendation directly ─────────────────────────────────────

print("=" * 55)
print("Applying recommendation to data")
print("=" * 55)

rec_apply = qd.recommend(X, task="classification", qubits=8)
result = rec_apply.apply(X)

print(f"Encoding used : {result.encoded[0].metadata['encoding']}")
print(f"Circuits      : {len(result.circuits)}")
print(f"Qubits        : {result.encoded[0].metadata['n_qubits']}")
print()

# ── 5. Alternatives ───────────────────────────────────────────────────────────

print("=" * 55)
print("Ranked alternatives")
print("=" * 55)

rec2 = qd.recommend(X, task="classification", qubits=8)
for i, alt in enumerate(rec2.alternatives):
    print(f"  #{i + 1}: {alt.method:20s}  score={alt.score:.1f}")
