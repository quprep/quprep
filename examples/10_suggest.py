"""
10 — Qubit Suggestion
======================
Demonstrates suggest_qubits: task-aware qubit budget recommendation
with encoding hints, NISQ ceiling, and pipeline integration.

    uv run python examples/10_suggest.py
"""

import numpy as np

import quprep as qd

rng = np.random.default_rng(0)

# ── 1. Basic suggestion ───────────────────────────────────────────────────────

print("=" * 55)
print("1 — Basic suggestion (8 features, classification)")
print("=" * 55)

X_small = rng.uniform(0, 1, size=(100, 8))
s = qd.suggest_qubits(X_small)
print(s)

# ── 2. Task-aware hints ───────────────────────────────────────────────────────

print()
print("=" * 55)
print("2 — Task-aware encoding hints")
print("=" * 55)

tasks = ["classification", "regression", "kernel", "qaoa", "simulation"]
for task in tasks:
    s = qd.suggest_qubits(X_small, task=task)
    print(f"  {task:<16} → n_qubits={s.n_qubits}, hint={s.encoding_hint}")

# ── 3. NISQ ceiling — wide dataset ───────────────────────────────────────────

print()
print("=" * 55)
print("3 — Wide dataset (50 features) → NISQ ceiling")
print("=" * 55)

X_wide = rng.uniform(0, 1, size=(100, 50))
s_wide = qd.suggest_qubits(X_wide)
print(f"  n_features : {s_wide.n_features}")
print(f"  n_qubits   : {s_wide.n_qubits}  (capped at NISQ ceiling)")
print(f"  nisq_safe  : {s_wide.nisq_safe}")
print(f"  warning    : {s_wide.warning}")

# ── 4. Override ceiling ───────────────────────────────────────────────────────

print()
print("=" * 55)
print("4 — Override NISQ ceiling (fault-tolerant / simulator)")
print("=" * 55)

s_ft = qd.suggest_qubits(X_wide, max_qubits=50)
print(f"  n_qubits   : {s_ft.n_qubits}")
print(f"  nisq_safe  : {s_ft.nisq_safe}")
print(f"  warning    : {s_ft.warning}")

# ── 5. Apply suggestion to a pipeline ────────────────────────────────────────

print()
print("=" * 55)
print("5 — Use suggestion to configure a pipeline")
print("=" * 55)

s = qd.suggest_qubits(X_wide, task="classification")
print(f"  Suggested n_qubits  : {s.n_qubits}")
print(f"  Suggested encoding  : {s.encoding_hint}")

pipeline = qd.Pipeline(
    reducer=qd.PCAReducer(n_components=s.n_qubits),
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform(X_wide)
print(f"  Encoded {len(result.encoded)} samples at {s.n_qubits} qubits each")

# ── 6. repr vs str ────────────────────────────────────────────────────────────

print()
print("=" * 55)
print("6 — repr vs str")
print("=" * 55)

s = qd.suggest_qubits(X_small, task="kernel")
print(f"  repr : {repr(s)}")
print(f"  str  :\n{s}")

print("\nDone.")
