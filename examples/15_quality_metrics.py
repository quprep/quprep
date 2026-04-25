"""
15 — Encoding Quality Metrics
================================
Measure the *actual* quality of a quantum encoding with simulation-based
metrics: expressibility, entanglement capability, and kernel alignment.
All three are computed with a lightweight numpy statevector backend — no
quantum framework required.  Circuits with more than 12 qubits return None.

    uv run python examples/15_quality_metrics.py
"""

import numpy as np

import quprep as qd
from quprep.core.dataset import Dataset

# ── Shared data ───────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
X = rng.uniform(0, np.pi, (120, 4))
y = (X[:, 0] > np.pi / 2).astype(float) * 2 - 1   # ±1 labels
ds = Dataset(data=X, feature_names=["a", "b", "c", "d"], labels=y)

# ── 1. Score a single encoding ────────────────────────────────────────────────

print("=" * 55)
print("score_encoding — IQP on 4-feature dataset")
print("=" * 55)

m = qd.score_encoding(qd.IQPEncoder(), ds, n_samples=200)
print(m)
print()

# ── 2. Score multiple encodings ───────────────────────────────────────────────

print("=" * 55)
print("Comparing encoders")
print("=" * 55)

encoders = {
    "angle"         : qd.AngleEncoder(),
    "iqp"           : qd.IQPEncoder(),
    "entangled_angle": qd.EntangledAngleEncoder(),
    "reupload"      : qd.ReUploadEncoder(),
}

results = {name: qd.score_encoding(enc, ds, n_samples=150) for name, enc in encoders.items()}

print(f"{'Encoding':<20} {'Expressibility':>17} {'Entanglement':>14} {'Kernel align':>14}")
print("-" * 68)
for name, r in results.items():
    exp = f"{r.expressibility:.4f}" if r.expressibility is not None else "  N/A"
    ent = f"{r.entanglement_capability:.4f}" if r.entanglement_capability is not None else "  N/A"
    ka  = f"{r.kernel_alignment:.4f}"  if r.kernel_alignment is not None else "  N/A"
    print(f"{name:<20} {exp:>17} {ent:>14} {ka:>14}")
print()

# ── 3. Individual metric functions ────────────────────────────────────────────

print("=" * 55)
print("Individual functions")
print("=" * 55)

# Expressibility: KL divergence from Haar (lower = more expressive)
exp = qd.expressibility(qd.AngleEncoder(), ds, n_samples=400)
print(f"AngleEncoder  expressibility         : {exp:.4f}  (lower = more expressive)")

# Entanglement capability: avg. Meyer-Wallach measure [0, 1]
ent = qd.entanglement_capability(qd.IQPEncoder(), ds, n_samples=200)
print(f"IQPEncoder    entanglement_capability: {ent:.4f}  (higher = more entangled)")

# Kernel alignment: requires labelled data
ka = qd.kernel_alignment(qd.IQPEncoder(), ds, max_samples=100)
print(f"IQPEncoder    kernel_alignment       : {ka:.4f}  (higher = better class sep.)")
print()

# ── 4. Metric-augmented recommendation ───────────────────────────────────────

print("=" * 55)
print("recommend(use_metrics=True) — data-driven re-ranking")
print("=" * 55)

rec_plain   = qd.recommend(ds, task="classification")
rec_metrics = qd.recommend(ds, task="classification", use_metrics=True)

print(f"Without metrics  →  {rec_plain.method}")
print(f"With    metrics  →  {rec_metrics.method}")
print()
print("Top-3 alternatives (with metrics):")
for alt in rec_metrics.alternatives[:3]:
    print(f"  {alt.method:<20}  score={alt.score:.1f}")
